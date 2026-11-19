"""D4RT decoder for the published Gen4P inference checkpoint."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .embeddings import FourierEmbedding, TimestepEmbedding, PatchEmbeddingFast
from utils.utils import soft_clamp_max

CONFIDENCE_MAX_VAL = 200


def piecewise_tanh_clamp(
    x,
    boundary=10.0,
    limit=2.0,
):
    """
    Clamps the input tensor x using a piecewise function that return x within the `boundary` and
    applies tanh outside a specified boundary with a `limit` discount.
    """
    out_pos = boundary + limit * torch.tanh((x - boundary) / limit)
    out_neg = -boundary + limit * torch.tanh((x + boundary) / limit)

    return torch.where(
        x > boundary,
        out_pos,
        torch.where(x < -boundary, out_neg, x),  # keep the same within the boundary
    )


class CrossAttention(nn.Module):
    """Efficient cross-attention using PyTorch's scaled_dot_product_attention.

    Automatically uses FlashAttention or memory-efficient attention when available.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        qk_norm: tuple[nn.Module, nn.Module],
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, C) query tokens
            key_value: (B, N_kv, C) encoder features for both keys and values
            qk_norm: normalization layers for query and key

        Returns:
            out: (B, N_q, C)
        """
        B, N_q, C = query.shape

        N_kv = key_value.shape[1]

        # Project queries, keys, values
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim)

        norm_q, norm_k = qk_norm
        q = norm_q(q)
        k = norm_k(k)

        # Transpose: (B, num_heads, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch's efficient attention (FlashAttention when available)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=1.0)

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)

        return x


class MLP(nn.Module):
    """MLP block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DecoderBlock(nn.Module):
    """Independent point-query cross-attention and feed-forward block."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        mlp_size=None,
    ):
        super().__init__()
        self.cross_attn = CrossAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.qknorm_q = nn.RMSNorm(dim // num_heads, 1e-5)
        self.qknorm_k = nn.RMSNorm(dim // num_heads, 1e-5)
        hidden_dim = int(mlp_size) if mlp_size is not None else int(dim * mlp_ratio)
        hidden_dim = hidden_dim or dim * 4
        self.mlp = MLP(dim, hidden_dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        query,
        encoder_features,
    ):
        query = query + self.cross_attn(self.norm1(query), self.norm_kv(encoder_features), (self.qknorm_q, self.qknorm_k))
        query = query + self.mlp(self.norm2(query))
        return query


class D4RTDecoder(nn.Module):
    """Decode RGB point queries for inference, tracking, and depth estimation."""

    def __init__(
        self,
        base_num_frames: int,
        embed_dim: int = 1536,
        depth: int = 8,
        num_heads: int = 24,
        mlp_ratio: float = 4.0,
        mlp_size: Optional[int] = None,
        max_timesteps: int = 128,
        patch_size: int = 9,
        num_fourier_freqs: int = 10,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if mlp_ratio <= 0 or (mlp_size is not None and mlp_size <= 0):
            raise ValueError("Decoder MLP size must be positive.")

        self.fourier_embed = FourierEmbedding(embed_dim=embed_dim, num_frequencies=num_fourier_freqs)
        self.timestep_embed = TimestepEmbedding(max_timesteps=max_timesteps, embed_dim=embed_dim, base_num_frames=base_num_frames)
        self.patch_embed = PatchEmbeddingFast(patch_size, embed_dim)
        self.input_query_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.blocks = nn.ModuleList([DecoderBlock(embed_dim, num_heads, mlp_ratio, mlp_size) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head_3d = nn.Linear(embed_dim, 3)
        self.head_vis = nn.Linear(embed_dim, 1)
        self.head_conf = nn.Linear(embed_dim, 1)

    def build_query(
        self,
        frames: torch.Tensor,
        coords: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
    ) -> torch.Tensor:
        """Build query embeddings from components.

        Args:
            frames: (B, T, C, H, W) video frames for patch extraction
            coords: (B, N, 2) normalized (u, v) coordinates
            t_src: (B, N) source timestep indices
            t_tgt: (B, N) target timestep indices
            t_cam: (B, N) camera reference timestep indices

        Returns:
            query: (B, N, embed_dim)
        """
        coord_emb = self.fourier_embed(coords)
        src_emb, tgt_emb, cam_emb = self.timestep_embed(t_src, t_tgt, t_cam, active_num_frames=int(frames.shape[1]))
        patch_emb = self.patch_embed(frames, coords, t_src)
        query = coord_emb + src_emb + tgt_emb + cam_emb + patch_emb
        return self.input_query_mlp(query)

    def forward(
        self,
        encoder_features: torch.Tensor,
        frames: torch.Tensor,
        coords: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            encoder_features: (B, N_enc, embed_dim) Global Scene Representation
            frames: (B, T, C, H, W) video frames
            coords: (B, N_q, 2) normalized query coordinates
            t_src: (B, N_q) source timesteps
            t_tgt: (B, N_q) target timesteps
            t_cam: (B, N_q) camera reference timesteps

        Returns:
            Dictionary with predictions:
                - pos_3d: (B, N_q, 3) 3D positions
                - visibility: (B, N_q, 1) visibility logits
                - confidence: (B, N_q, 1) confidence scores
        """
        query = self.build_query(frames, coords, t_src, t_tgt, t_cam)
        encoder_features = piecewise_tanh_clamp(encoder_features, boundary=10.0, limit=2.0)
        for block in self.blocks:
            query = block(query, encoder_features)
        query = self.norm(query)
        pos_3d = self.head_3d(query)
        visibility = self.head_vis(query)
        confidence = soft_clamp_max(1.0 + torch.exp(self.head_conf(query)), max_val=CONFIDENCE_MAX_VAL, beta=10.0)
        return {
            "pos_3d": pos_3d,
            "visibility": visibility,
            "confidence": confidence,
        }
