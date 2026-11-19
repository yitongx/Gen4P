"""Gen4P video encoding and queries used by the inference scripts."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .encoder import WanEncoder
from .decoder import D4RTDecoder


class D4RT(nn.Module):
    """Encode RGB video and decode geometry, depth and source-world tracks."""

    def __init__(
        self,
        temporal_size: int = 48,
        decoder_depth: int = 8,
        decoder_num_heads: int = 24,
        decoder_mlp_ratio: float = 4.0,
        decoder_mlp_size: Optional[int] = 4096,
        max_timesteps: int = 128,
        query_patch_size: int = 9,
        num_fourier_freqs: int = 10,
        wan_feature_model_id: str = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
        wan_feature_t: int = 999,
        wan_feature_output_layer: int = 29,
        wan_feature_set_vae_f32: bool = False,
        wan_torch_dtype: str = 'bfloat16',
        wan_device: str = 'cuda',
        wan_train_mode: str = 'lora',
        wan_lora_rank: int = 512,
        wan_lora_alpha: Optional[int] = None,
        wan_lora_target_modules: str = 'q,k,v,o,ffn.0,ffn.2',
        wan_lora_adapter_name: str = 'default',
    ):
        super().__init__()
        self.encoder = WanEncoder(
            model_id=wan_feature_model_id,
            t=wan_feature_t,
            output_layer_index=wan_feature_output_layer,
            set_vae_f32=wan_feature_set_vae_f32,
            torch_dtype=wan_torch_dtype,
            device=wan_device,
            wan_train_mode=wan_train_mode,
            wan_lora_rank=wan_lora_rank,
            wan_lora_alpha=wan_lora_alpha,
            wan_lora_target_modules=wan_lora_target_modules,
            wan_lora_adapter_name=wan_lora_adapter_name,
        )
        self.decoder = D4RTDecoder(
            embed_dim=self.encoder.embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=decoder_mlp_ratio,
            mlp_size=decoder_mlp_size,
            max_timesteps=max_timesteps,
            base_num_frames=temporal_size,
            patch_size=query_patch_size,
            num_fourier_freqs=num_fourier_freqs,
        )
        self.temporal_size = temporal_size

    def encode(
        self,
        video: torch.Tensor,
        enc_img_size: tuple[int, int],
    ) -> torch.Tensor:
        """Resize (B, T, 3, H, W) RGB input and encode it once."""
        B, T, C, H, W = video.shape
        assert C == 3, f"Expected video shape (B, T, 3, H, W), got {video.shape}"
        enc_h, enc_w = enc_img_size
        if H != enc_h or W != enc_w:
            assert video.dtype != torch.uint8 and video.max() <= 1.0, "Encoder resize requires normalized floating-point RGB."
            video = F.interpolate(video.view(B * T, C, H, W), size=(enc_h, enc_w), mode='bilinear', align_corners=False).view(B, T, C, enc_h, enc_w)
        return self.encoder(video)

    def decode(
        self,
        encoder_features: torch.Tensor,
        frames: torch.Tensor,
        coords: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode normalized RGB queries into position, visibility and confidence."""
        return self.decoder(encoder_features, frames, coords, t_src, t_tgt, t_cam)

    @torch.no_grad()
    def predict_depth(
        self,
        video: torch.Tensor,
        enc_img_size: tuple[int, int],
        output_resolution: tuple[int, int],
        encode_video: torch.Tensor,
        decoder_dtype: Optional[torch.dtype] = None,
        target_frame_indices: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return depth and confidence maps for all or selected clip frames."""
        B, T, C, _, _ = video.shape
        assert encode_video.shape[:3] == (B, T, C), f"Expected encode_video shape (B, T, C, H, W), got {encode_video.shape}"
        encoder_features = self.encode(encode_video, enc_img_size)

        # Create query grid for all pixels, at pixel center
        device = video.device
        out_H, out_W = output_resolution
        u = (torch.arange(out_W, device=device, dtype=torch.float32) + 0.5) / float(out_W)
        v = (torch.arange(out_H, device=device, dtype=torch.float32) + 0.5) / float(out_H)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')
        coords = torch.stack([grid_u, grid_v], dim=-1).reshape(-1, 2)  # (H*W, 2)
        if decoder_dtype is not None:
            encoder_features = encoder_features.to(decoder_dtype)
            video = video.to(decoder_dtype)
            coords = coords.to(decoder_dtype)

        depths = []
        confidences = []
        N = out_H * out_W
        target_frames = list(range(T)) if target_frame_indices is None else [int(t) for t in target_frame_indices]

        for t in target_frames:
            if t < 0 or t >= T:
                raise ValueError(f'target_frame_indices contains out-of-range frame {t} for clip length {T}.')
            if t % 10 == 0:
                print(f"Predicting depth for frame {t}/{T - 1}...")
            depth_flat = torch.empty((B, N), device=device, dtype=video.dtype)
            confidence_flat = torch.empty((B, N), device=device, dtype=video.dtype)
            rand_total_points = torch.randperm(N, device=device)

            # infer depth in chunks to reduce peak VRAM usage.
            for point_start in range(0, N, 2048):
                point_end = min(point_start + 2048, N)
                point_count = point_end - point_start
                point_indices = rand_total_points[point_start:point_end]

                # For depth: t_src = t_tgt = t_cam = t
                t_indices = torch.full((B, point_count), t, device=device, dtype=torch.long)
                coords_batch = coords[point_indices].unsqueeze(0).expand(B, -1, -1)

                chunk_outputs = self.decode(
                    encoder_features=encoder_features,
                    frames=video,
                    coords=coords_batch,
                    t_src=t_indices,
                    t_tgt=t_indices,
                    t_cam=t_indices,
                )

                depth_flat[:, point_indices] = chunk_outputs['pos_3d'][:, :, 2]
                confidence_flat[:, point_indices] = chunk_outputs['confidence'].squeeze(-1)

            depth = depth_flat.reshape(B, out_H, out_W)
            confidence = confidence_flat.reshape(B, out_H, out_W)
            depths.append(depth)
            confidences.append(confidence)

        return (
            torch.stack(depths, dim=1),  # (B, T_out, H, W)
            torch.stack(confidences, dim=1),
        )  # (B, T_out, H, W)

    @torch.no_grad()
    def predict_point_tracks(
        self,
        video: torch.Tensor,
        query_points: torch.Tensor,
        t_src: torch.Tensor,
        enc_img_size: tuple[int, int],
        encode_video: torch.Tensor,
        decoder_dtype: Optional[torch.dtype] = None,
        max_query_points_per_batch: int = 2048,
    ) -> dict[str, torch.Tensor]:
        """Predict source-world 3D tracks and visibility for each clip frame."""
        N = query_points.shape[1]
        device = video.device
        B, T, C, _, _ = video.shape
        assert encode_video.shape[:3] == (B, T, C), f"Expected encode_video shape (B, T, C, H, W), got {encode_video.shape}"
        encoder_features = self.encode(encode_video, enc_img_size)
        if decoder_dtype is not None:
            encoder_features = encoder_features.to(decoder_dtype)
            video = video.to(decoder_dtype)
            query_points = query_points.to(decoder_dtype)

        # Preserve the query batching RNG stream used by the existing SRCW benchmark.
        for _ in range(T):
            torch.randperm(N, device=device)

        print("[Point Track] Start source-world tracking: t_tgt=t, t_cam=t_src")
        tracks_3d = torch.empty((B, N, T, 3), device=device, dtype=video.dtype)
        visibility = torch.empty((B, N, T), device=device, dtype=video.dtype)
        for t in range(T):
            if t % 10 == 0:
                print(f"Predicting source-world point tracking for frame {t}/{T - 1}...")

            rand_total_points = torch.randperm(N, device=device)
            for point_start in range(0, N, max_query_points_per_batch):
                point_end = min(point_start + max_query_points_per_batch, N)
                point_count = point_end - point_start
                point_indices = rand_total_points[point_start:point_end]

                # source-world: t_src=t_cam=fixed, t_tgt=t
                query_points_chunk = query_points[:, point_indices]
                t_src_chunk = t_src[:, point_indices]
                t_tgt_chunk = torch.full((B, point_count), t, device=device, dtype=torch.long)

                outputs = self.decode(
                    encoder_features=encoder_features,
                    frames=video,
                    coords=query_points_chunk,
                    t_src=t_src_chunk,
                    t_tgt=t_tgt_chunk,
                    t_cam=t_src_chunk,
                )

                tracks_3d[:, point_indices, t] = outputs['pos_3d']
                visibility[:, point_indices, t] = torch.sigmoid(outputs['visibility']).squeeze(-1)

        return {
            'tracks_3d_srcw': tracks_3d,
            'visibility_srcw': visibility,
        }
