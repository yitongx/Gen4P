"""Embedding modules for D4RT queries."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEmbedding(nn.Module):
    """Fourier feature embedding for continuous 2D coordinates.

    Maps (u, v) coordinates to higher-dimensional space using sinusoidal functions.
    """

    def __init__(
        self,
        embed_dim: int,
        num_frequencies: int = 10,
    ):
        super().__init__()
        freqs = 2.0 ** torch.linspace(0, num_frequencies, num_frequencies + 1)
        self.register_buffer('freqs', freqs)

        fourier_dim = 2 + 2 * 2 * (num_frequencies + 1)
        self.proj = nn.Linear(fourier_dim, embed_dim)

    def forward(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 2) normalized coordinates in [0, 1]

        Returns:
            embeddings: (B, N, embed_dim)
        """
        B, N, _ = coords.shape

        coords_freq = coords.unsqueeze(2) * self.freqs.view(-1, 1) * math.pi
        fourier_features = torch.cat([torch.sin(coords_freq), torch.cos(coords_freq)], dim=-1)
        fourier_features = fourier_features.view(B, N, -1) 
        fourier_features = torch.cat([coords, fourier_features], dim=-1)  # (B, N, 2 + (num_freqs+1)*4)

        return self.proj(fourier_features)


class TimestepEmbedding(nn.Module):
    """Learnable discrete timestep embeddings.

    Provides separate embeddings for source, target, and camera timesteps.
    """

    def __init__(
        self,
        max_timesteps: int,
        embed_dim: int,
        base_num_frames: int,
    ):
        super().__init__()
        self.max_timesteps = max_timesteps
        self.base_num_frames = int(base_num_frames)
        if self.base_num_frames <= 0 or self.base_num_frames > self.max_timesteps:
            raise ValueError(f'base_num_frames must be in [1, {self.max_timesteps}], got {self.base_num_frames}.')

        # Separate embeddings for each timestep type
        self.src_embedding = nn.Embedding(max_timesteps, embed_dim)
        self.tgt_embedding = nn.Embedding(max_timesteps, embed_dim)
        self.cam_embedding = nn.Embedding(max_timesteps, embed_dim)

    def forward(
        self,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
        active_num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            t_src: (B, N) source timestep indices
            t_tgt: (B, N) target timestep indices
            t_cam: (B, N) camera reference timestep indices

        Returns:
            Tuple of embeddings, each (B, N, embed_dim)
        """
        active_num_frames = int(active_num_frames)
        if active_num_frames <= 0 or active_num_frames > self.max_timesteps:
            raise ValueError(f'active_num_frames must be in [1, {self.max_timesteps}], got {active_num_frames}.')

        max_index = max(int(t_src.max().item()) if t_src.numel() > 0 else 0, int(t_tgt.max().item()) if t_tgt.numel() > 0 else 0, int(t_cam.max().item()) if t_cam.numel() > 0 else 0)
        if max_index >= active_num_frames:
            raise ValueError(f'Timestep index {max_index} exceeds active_num_frames={active_num_frames}.')

        # Resize from the learned base prefix for every active clip length.
        src_weight = self.linearResizeTembed(self.src_embedding.weight, active_num_frames)
        tgt_weight = self.linearResizeTembed(self.tgt_embedding.weight, active_num_frames)
        cam_weight = self.linearResizeTembed(self.cam_embedding.weight, active_num_frames)

        src_emb = F.embedding(t_src, src_weight)
        tgt_emb = F.embedding(t_tgt, tgt_weight)
        cam_emb = F.embedding(t_cam, cam_weight)

        return src_emb, tgt_emb, cam_emb

    def linearResizeTembed(
        self,
        weight: torch.Tensor,
        active_num_frames: int,
    ) -> torch.Tensor:
        """
        Interpolate top-base_num_frames learnable embeddings to active_num_frames using linear
        interpolation.
        - weight: [T=128, D]
        """
        if active_num_frames == self.base_num_frames:
            return weight[:active_num_frames]

        source = weight[: self.base_num_frames].transpose(0, 1).unsqueeze(0).to(dtype=torch.float32)
        resized = F.interpolate(source, size=active_num_frames, mode='linear', align_corners=True).squeeze(0).transpose(0, 1)
        return resized.to(dtype=weight.dtype, device=weight.device)


class PatchEmbeddingFast(nn.Module):
    """Faster vectorized patch embedding using grid_sample."""

    def __init__(
        self,
        patch_size: int = 9,
        embed_dim: int = 1536,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError(f'patch_size must be positive, got {self.patch_size}.')

        patch_dim = 3 * self.patch_size * self.patch_size
        self.mlp = nn.Sequential(nn.Linear(patch_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

        offsets = self.buildOffsets(self.patch_size)
        self.register_buffer('offsets', offsets)

    def buildOffsets(
        self,
        patch_size: int,
    ) -> torch.Tensor:
        base = torch.arange(patch_size, dtype=torch.get_default_dtype()) - float(patch_size // 2)
        return torch.stack(torch.meshgrid(base, base, indexing='xy'), dim=-1)

    def forward(
        self,
        frames: torch.Tensor,
        coords: torch.Tensor,
        t_src: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) video frames
            coords: (B, N, 2) normalized coordinates in [0, 1]
            t_src: (B, N) source frame indices

        Returns:
            embeddings: (B, N, embed_dim)
        """
        B, T, _, H, W = frames.shape
        N = coords.shape[1]
        ps = self.patch_size

        frames_3d = frames.permute(0, 2, 1, 3, 4)

        # chunk-wise processing for N dim, in case of dense query OOM
        chunk_size = 2048
        all_embeddings = []

        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            current_chunk_size = end_i - i

            chunk_coords = coords[:, i:end_i, :]
            chunk_t_src = t_src[:, i:end_i]

            # `coords` use pixel-area semantics: u = x / W, v = y / H.
            coords_pixel = chunk_coords.clone()
            coords_pixel[..., 0] = coords_pixel[..., 0] * W
            coords_pixel[..., 1] = coords_pixel[..., 1] * H
            offsets = self.offsets.view(1, 1, ps, ps, 2).to(coords_pixel.dtype)
            grid_xy = coords_pixel.view(B, current_chunk_size, 1, 1, 2) + offsets
            grid_xy[..., 0] = 2.0 * grid_xy[..., 0] / W - 1.0
            grid_xy[..., 1] = 2.0 * grid_xy[..., 1] / H - 1.0

            # also consider the thickness of z-dim, similar to pixel-area
            # exact frame centers are at (t + 0.5) / T.
            t_norm = (2.0 * chunk_t_src.float() + 1.0) / max(1, T) - 1.0
            grid_z = t_norm.view(B, current_chunk_size, 1, 1, 1).expand(-1, -1, ps, ps, 1).to(coords_pixel.dtype)
            grid_3d = torch.cat([grid_xy, grid_z], dim=-1)  # (B, N, ps, ps, 3)

            patches = F.grid_sample(
                frames_3d, 
                grid_3d,
                mode='bilinear',
                padding_mode='border',
                align_corners=False,
            )  # (B, C, N, ps, ps)

            patches_flat = patches.permute(0, 2, 3, 4, 1).reshape(B, current_chunk_size, -1)

            chunk_emb = self.mlp(patches_flat)
            all_embeddings.append(chunk_emb)

        return torch.cat(all_embeddings, dim=1)  # (B, N, embed_dim)
