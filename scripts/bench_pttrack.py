#!/usr/bin/env python3
"""
Evaluate Gen4P sparse point tracking on WorldTrack.

Track first-frame query points in source-world coordinates and report APD_3D/EPE
for all, static, and dynamic points.

Rerun recordings visualize aligned predictions and ground truths.
"""

import argparse
import imageio
import json
import multiprocessing as mp
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from utils.inference import str2bool, setRandomness, get_visible_cuda_devices, load_model
from utils.inference import validate_image_aspect_ratios


PRED_VISIBLITY_THRES = float(os.getenv("PRED_VISIBLITY_THRES", "0.5"))
STATIC_WORLD_MOTION_MAX = 5e-3  # set thres for accumulated traj, see paper appendix E1
ALIGNMENT_MODE_CHOICES = ("global_median", "global_sim3")
WORLDTRACK_FIXED_THRESHOLDS = {
    1: 0.1,
    2: 0.3,
    4: 0.5,
    8: 1.0,
}
APD_3D_UNIT = "percent"
SCENE_PLACEHOLDER_ROOT_DIR_NAME = ".bench_pttrack_placeholder"
SCENE_PLACEHOLDER_PROCESSING_DIR_NAME = "processing"
SCENE_PLACEHOLDER_DONE_DIR_NAME = "done"
WORKER_ARGS = None
WORKER_DEVICE = None
WORKER_MODEL = None


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark sparse point tracking on WorldTrack')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dvd_checkpoint', type=str, default=None, help='Override ckpt/dvd_1.1.safetensors')
    parser.add_argument('--data_root', type=str, default=None, help='WorldTrack root')
    parser.add_argument('--split', type=str, default=None, help='WorldTrack subset name, e.g. ds_mini')
    parser.add_argument('--split_configs', nargs='*', default=None, help=('Sequential persistent-worker configs formatted as split,proc_h,proc_w,img_h,img_w,output_dir'))
    parser.add_argument('--target_scene_list', nargs='*', default=None, help='Scene stem list or `*` for all scenes')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of scene workers')
    parser.add_argument('--output_dir', type=str, default='output_eval_pttrack_worldtrack', help='Output directory')
    parser.add_argument('--infer_total_num_frames', type=int, default=64, help=('Frames to evaluate within one model window; -1 uses all remaining frames from start_frame if they fit'))
    parser.add_argument('--infer_chunk_num_frames', type=int, default=None, help='Model temporal window used for one-shot clip inference')
    parser.add_argument('--train_num_frames', type=int, default=2, help='Optional override for checkpoint training num_frames used by time-embedding resize')
    parser.add_argument('--start_frame', type=int, default=0, help='Starting frame inside the source sequence')
    parser.add_argument('--frame_stride', type=int, default=1, help='Sample stride inside the source sequence')
    parser.add_argument('--img_size', type=int, default=224, help='Encoder square size when --img_h/--img_w are unset')
    parser.add_argument('--img_h', type=int, default=None, help='Optional encoder input height')
    parser.add_argument('--img_w', type=int, default=None, help='Optional encoder input width')
    parser.add_argument('--proc_h', type=int, default=None, help='Optional benchmark video height before encoder resize')
    parser.add_argument('--proc_w', type=int, default=None, help='Optional benchmark video width before encoder resize')
    parser.add_argument('--center_crop', type=str2bool, default=False, help=('Center-crop raw frames to proc aspect ratio before resize; false keeps full-frame resize'))
    parser.add_argument('--raw_rgb', type=str2bool, default=True, help='Keep benchmark video at proc size and let the model resize internally')
    parser.add_argument(
        '--alignment_mode',
        type=str,
        default='global_median',
        choices=list(ALIGNMENT_MODE_CHOICES),
        help=('Metric/output alignment mode: global_median uses median-scale alignment; global_sim3 uses OpenD4RT-style global RANSAC Sim3 alignment'),
    )
    parser.add_argument('--amp_infer', type=str2bool, default=False, help='Enable AMP autocast during inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic kernels for reproducibility (may reduce speed)')
    parser.add_argument('--max_query_points_per_batch', type=int, default=2048, help='Decoder query chunk size')
    parser.add_argument('--only_condition_t_but_no_noise', type=str2bool, default=True, help='Must be true: keep the timestep condition and disable latent noise')
    parser.add_argument('--pre_vae_encode', type=str2bool, default=None, help='Override pre_vae_encode from checkpoint args')
    args = parser.parse_args()
    has_img_h = args.img_h is not None
    has_img_w = args.img_w is not None
    if has_img_h != has_img_w:
        parser.error('--img_h and --img_w must be provided together.')
    if has_img_h and has_img_w:
        if args.img_h <= 0 or args.img_w <= 0:
            parser.error('--img_h and --img_w must be positive.')
        args.img_h = int(args.img_h)
        args.img_w = int(args.img_w)
    else:
        args.img_h = int(args.img_size)
        args.img_w = int(args.img_size)
    has_proc_h = args.proc_h is not None
    has_proc_w = args.proc_w is not None
    if has_proc_h != has_proc_w:
        parser.error('--proc_h and --proc_w must be provided together.')
    if has_proc_h and has_proc_w:
        if args.proc_h <= 0 or args.proc_w <= 0:
            parser.error('--proc_h and --proc_w must be positive.')
        args.proc_h = int(args.proc_h)
        args.proc_w = int(args.proc_w)
        if not args.split_configs:
            try:
                validate_image_aspect_ratios(args.img_h, args.img_w, args.proc_h, args.proc_w)
            except ValueError as error:
                parser.error(str(error))
    if args.frame_stride <= 0:
        parser.error('--frame_stride must be positive.')
    if args.start_frame < 0:
        parser.error('--start_frame must be >= 0.')
    if args.infer_total_num_frames == 0 or args.infer_total_num_frames < -1:
        parser.error('--infer_total_num_frames must be positive or -1.')
    if args.num_workers is not None and args.num_workers <= 0:
        parser.error('--num_workers must be positive.')
    if args.max_query_points_per_batch <= 0:
        parser.error('--max_query_points_per_batch must be positive.')
    if args.infer_chunk_num_frames is not None and args.infer_chunk_num_frames <= 0:
        parser.error('--infer_chunk_num_frames must be positive.')
    if args.train_num_frames is not None and args.train_num_frames <= 0:
        parser.error('--train_num_frames must be positive when provided.')
    has_single_split_inputs = bool(args.data_root is not None and args.split is not None and args.target_scene_list)
    has_split_configs = bool(args.data_root is not None and args.split_configs and args.target_scene_list)
    if has_single_split_inputs == has_split_configs:
        parser.error('Provide either --data_root/--split/--target_scene_list or --data_root/--split_configs/--target_scene_list.')
    args.parser_defaults = {k: parser.get_default(k) for k in vars(args).keys()}
    return args


def get_model_clip_frames(
    model,
    fallback: Optional[int],
) -> int:
    temporal_size = getattr(model, "temporal_size", None)
    if temporal_size is not None:
        return int(temporal_size)
    if fallback is not None:
        return int(fallback)
    raise ValueError("Cannot infer model clip length; set --infer_chunk_num_frames explicitly.")


def decode_jpeg_rgb(
    frame_bytes: bytes,
) -> np.ndarray:
    arr = np.frombuffer(frame_bytes, np.uint8)
    image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        raise RuntimeError("Failed to decode JPEG frame bytes.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def project_points_to_video_frame(
    camera_pov_points3d: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(camera_pov_points3d, dtype=np.float64)
    intr = np.asarray(camera_intrinsics, dtype=np.float64).reshape(-1)
    fx, fy, cx, cy = intr[:4]
    z = pts[..., 2]
    safe_z = np.where(np.abs(z) > 1e-8, z, np.nan)
    u = (pts[..., 0] / safe_z) * fx + cx
    v = (pts[..., 1] / safe_z) * fy + cy
    return np.stack([u, v], axis=-1)


def resolve_sequence_num_frames(
    npz_path: Path,
    requested_num_frames: int,
    start_frame: int,
    frame_stride: int,
) -> int:
    with np.load(npz_path, allow_pickle=True) as pack:
        total_frames = min(int(np.asarray(pack["images_jpeg_bytes"]).shape[0]), int(np.asarray(pack["tracks_XYZ"]).shape[0]), int(np.asarray(pack["visibility"]).shape[0]))
    if start_frame >= total_frames:
        raise ValueError(f"start_frame {start_frame} exceeds available frame count {total_frames} for {npz_path}")
    available = len(range(start_frame, total_frames, frame_stride))
    if requested_num_frames > 0:
        return min(int(requested_num_frames), available)
    return available


def build_metric_payload_for_frame(
    frame_idx_local: int,
    frame_idx_global: int,
    metrics: Dict[str, object],
) -> Dict[str, object]:
    """
    Convert one-frame metric outputs into a JSON-friendly payload with frame ids.
    """
    payload = {
        "frame_idx_local": int(frame_idx_local),
        "frame_idx_global": int(frame_idx_global),
    }
    for key, value in metrics.items():
        if key == "pred_tracks_aligned_tn3":
            continue
        payload[key] = value
    return payload


def load_worldtrack_sequence(
    npz_path: Path,
    start_frame: int,
    num_frames: int,
    frame_stride: int,
) -> dict:
    pack = np.load(npz_path, allow_pickle=True)
    images_jpeg_bytes = np.asarray(pack["images_jpeg_bytes"])
    tracks_xyz_cam = np.asarray(pack["tracks_XYZ"], dtype=np.float64)
    intrinsics = np.asarray(pack["fx_fy_cx_cy"], dtype=np.float64)
    visibility = np.asarray(pack["visibility"], dtype=bool)
    extrinsics_w2c_raw = pack["extrinsics_w2c"] if "extrinsics_w2c" in pack.files else None

    total_frames = min(int(images_jpeg_bytes.shape[0]), int(tracks_xyz_cam.shape[0]), int(visibility.shape[0]))
    sampled_indices = [start_frame + i * frame_stride for i in range(num_frames)]
    if sampled_indices[-1] >= total_frames:
        raise ValueError(f"Requested sampled frame {sampled_indices[-1]} exceeds available frame count {total_frames} for {npz_path}")

    selected = np.asarray(sampled_indices, dtype=np.int64)
    images_jpeg_bytes = images_jpeg_bytes[selected]
    tracks_xyz_cam = tracks_xyz_cam[selected]
    visibility = visibility[selected]
    video_rgb = np.stack([decode_jpeg_rgb(frame_bytes) for frame_bytes in images_jpeg_bytes], axis=0)  # uin8
    tracks_uv = project_points_to_video_frame(tracks_xyz_cam, intrinsics)

    if extrinsics_w2c_raw is not None:
        extrinsics_w2c = np.asarray(extrinsics_w2c_raw, dtype=np.float64)[selected]
        first_inv = np.linalg.inv(extrinsics_w2c[0])
        extrinsics_w2c = np.asarray([extr @ first_inv for extr in extrinsics_w2c], dtype=np.float64)
        extrinsics_c2w = np.linalg.inv(extrinsics_w2c)
        tracks_xyz_ref0 = np.empty_like(tracks_xyz_cam, dtype=np.float64)
        for frame_idx in range(int(extrinsics_c2w.shape[0])):
            rot = extrinsics_c2w[frame_idx, :3, :3]
            trans = extrinsics_c2w[frame_idx, :3, 3]
            tracks_xyz_ref0[frame_idx] = (rot @ tracks_xyz_cam[frame_idx].T).T + trans
    else:
        extrinsics_w2c = np.tile(np.eye(4, dtype=np.float64), (tracks_xyz_cam.shape[0], 1, 1))
        tracks_xyz_ref0 = tracks_xyz_cam.copy()

    return {
        "video_rgb": video_rgb,
        "tracks_xyz_cam": tracks_xyz_cam,
        "tracks_xyz_ref0": tracks_xyz_ref0,
        "tracks_uv": tracks_uv,
        "visibility": visibility,
        "intrinsics": intrinsics,
        "extrinsics_w2c": extrinsics_w2c,
        "sampled_indices_global": selected,
        "video_name": npz_path.stem,
        "sequence_path": str(npz_path),
    }


def resize_video_frames(
    video_rgb: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    resized = [cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for frame in video_rgb]
    return np.stack(resized, axis=0)


def compute_crop_and_proc_shapes(
    height: int,
    width: int,
    target_h: Optional[int],
    target_w: Optional[int],
    center_crop: bool = True,
) -> Tuple[int, int, int, int]:
    if target_h is None or target_w is None:
        return int(height), int(width), int(height), int(width)

    proc_h = int(target_h)
    proc_w = int(target_w)
    if proc_h <= 0 or proc_w <= 0:
        raise ValueError(f"Invalid proc shape: {(target_h, target_w)}")
    if not bool(center_crop):
        return int(height), int(width), proc_h, proc_w

    ratio_gcd = int(np.gcd(proc_h, proc_w))
    unit_h = proc_h // ratio_gcd
    unit_w = proc_w // ratio_gcd
    resize_scale = min(int(height) // unit_h, int(width) // unit_w)
    if resize_scale <= 0:
        raise ValueError(f"Source frame too small for requested proc aspect ratio. source={(height, width)}, proc={(proc_h, proc_w)}, unit={(unit_h, unit_w)}")

    crop_h = unit_h * resize_scale
    crop_w = unit_w * resize_scale
    return int(crop_h), int(crop_w), proc_h, proc_w


def transform_query_uv_to_model(
    query_uv: np.ndarray,
    meta: dict,
) -> np.ndarray:
    scaled = np.asarray(query_uv, dtype=np.float32).copy()
    crop_h, crop_w = int(meta["crop_shape"][0]), int(meta["crop_shape"][1])
    crop_top, crop_left = int(meta["crop_offsets"][0]), int(meta["crop_offsets"][1])
    proc_h, proc_w = int(meta["proc_shape"][0]), int(meta["proc_shape"][1])
    model_h, model_w = int(meta["model_shape"][0]), int(meta["model_shape"][1])

    scaled[:, 0] = (scaled[:, 0] - float(crop_left)) * float(proc_w) / max(float(crop_w), 1.0)
    scaled[:, 1] = (scaled[:, 1] - float(crop_top)) * float(proc_h) / max(float(crop_h), 1.0)
    if model_w != proc_w or model_h != proc_h:
        scaled[:, 0] = scaled[:, 0] * float(model_w) / max(float(proc_w), 1.0)
        scaled[:, 1] = scaled[:, 1] * float(model_h) / max(float(proc_h), 1.0)
    return scaled


def prepare_video_and_queries(
    video_rgb: np.ndarray,
    query_uv: np.ndarray,
    enc_img_h: int,
    enc_img_w: int,
    proc_h: Optional[int],
    proc_w: Optional[int],
    raw_rgb: bool,
    center_crop: bool,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
    orig_h = int(video_rgb.shape[1])
    orig_w = int(video_rgb.shape[2])
    crop_h, crop_w, proc_h_final, proc_w_final = compute_crop_and_proc_shapes(orig_h, orig_w, proc_h, proc_w, center_crop=center_crop)
    validate_image_aspect_ratios(enc_img_h, enc_img_w, proc_h_final, proc_w_final)
    crop_top = (orig_h - crop_h) // 2
    crop_left = (orig_w - crop_w) // 2
    video_crop_rgb = video_rgb[:, crop_top : crop_top + crop_h, crop_left : crop_left + crop_w, :]
    if proc_h_final != crop_h or proc_w_final != crop_w:
        video_proc_rgb = resize_video_frames(video_crop_rgb, proc_h_final, proc_w_final)
    else:
        video_proc_rgb = video_crop_rgb
    if raw_rgb or not bool(center_crop):
        video_model_rgb = video_proc_rgb
    elif enc_img_h != crop_h or enc_img_w != crop_w:
        video_model_rgb = resize_video_frames(video_crop_rgb, enc_img_h, enc_img_w)
    else:
        video_model_rgb = video_crop_rgb
    video_encode_rgb = resize_video_frames(video_crop_rgb, enc_img_h, enc_img_w)
    video_tchw = torch.from_numpy(video_model_rgb.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
    encode_video_tchw = torch.from_numpy(video_encode_rgb.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()

    meta = {
        'orig_shape': (orig_h, orig_w),
        'center_crop': bool(center_crop),
        'crop_shape': (crop_h, crop_w),
        'proc_shape': (proc_h_final, proc_w_final),
        'crop_offsets': (crop_top, crop_left),
        'model_shape': (int(video_model_rgb.shape[1]), int(video_model_rgb.shape[2])),
        'encode_shape': (int(video_encode_rgb.shape[1]), int(video_encode_rgb.shape[2])),
    }
    query_uv_model = transform_query_uv_to_model(query_uv, meta)
    return (video_tchw, encode_video_tchw, query_uv_model, meta)


def build_sparse_queries_and_gt(
    sample: dict,
    enc_img_h: int,
    enc_img_w: int,
    proc_h: Optional[int],
    proc_w: Optional[int],
    raw_rgb: bool,
    center_crop: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray], dict]:
    video_rgb = sample['video_rgb']
    tracks_uv_orig_tn2 = np.asarray(sample['tracks_uv'], dtype=np.float64)
    video_tchw, encode_video_tchw, query_uv_model, video_meta = prepare_video_and_queries(
        video_rgb=video_rgb, query_uv=np.asarray(tracks_uv_orig_tn2[0], dtype=np.float64), enc_img_h=enc_img_h, enc_img_w=enc_img_w, proc_h=proc_h, proc_w=proc_w, raw_rgb=raw_rgb, center_crop=center_crop
    )
    model_h = int(video_meta['model_shape'][0])
    model_w = int(video_meta['model_shape'][1])
    tracks_uv_model_tn2 = transform_query_uv_to_model(tracks_uv_orig_tn2.reshape(-1, 2), video_meta).reshape(tracks_uv_orig_tn2.shape[0], tracks_uv_orig_tn2.shape[1], 2)
    inplane_tn = (
        np.isfinite(tracks_uv_model_tn2).all(axis=-1)
        & (tracks_uv_model_tn2[..., 0] >= 0.0)
        & (tracks_uv_model_tn2[..., 0] < float(model_w))
        & (tracks_uv_model_tn2[..., 1] >= 0.0)
        & (tracks_uv_model_tn2[..., 1] < float(model_h))
    )
    gt_visible_all_tn = np.asarray(sample['visibility'], dtype=bool)
    gt_visible_all_tn = gt_visible_all_tn & inplane_tn
    visible_mask = np.asarray(gt_visible_all_tn[0], dtype=bool)
    if not np.any(visible_mask):
        raise RuntimeError('Frame-0 has no in-plane visible query points.')
    query_uv_orig = np.asarray(tracks_uv_orig_tn2[0, visible_mask], dtype=np.float64)
    query_uv_model = np.asarray(tracks_uv_model_tn2[0, visible_mask], dtype=np.float32)
    finite_mask = np.isfinite(query_uv_orig).all(axis=-1)
    finite_mask = finite_mask & np.isfinite(query_uv_model).all(axis=-1)
    tracks_xyz_src_n3 = np.asarray(sample['tracks_xyz_cam'][0, visible_mask], dtype=np.float64)
    depth0 = np.asarray(tracks_xyz_src_n3[:, 2], dtype=np.float64)
    finite_mask = finite_mask & np.isfinite(tracks_xyz_src_n3).all(axis=-1) & np.isfinite(depth0) & (np.abs(depth0) > 1e-08)
    if not np.any(finite_mask):
        raise RuntimeError('No finite frame-0 in-plane visible query points remain after depth filtering.')
    query_uv_orig = query_uv_orig[finite_mask]
    query_uv_model = query_uv_model[finite_mask]
    tracks_uv_model_tn2 = np.asarray(tracks_uv_model_tn2[:, visible_mask], dtype=np.float64)[:, finite_mask].astype(np.float32)
    gt_tracks_srcw_tn3 = np.asarray(sample['tracks_xyz_ref0'][:, visible_mask], dtype=np.float64)[:, finite_mask].astype(np.float32)
    gt_visible_tn = np.asarray(gt_visible_all_tn[:, visible_mask], dtype=bool)[:, finite_mask]
    gt_occluded_tn = ~gt_visible_tn
    query_points_norm = query_uv_model.astype(np.float32)
    query_points_norm[:, 0] /= float(max(model_w, 1))
    query_points_norm[:, 1] /= float(max(model_h, 1))
    query_points_norm = np.clip(query_points_norm, 0.0, 1.0)
    t_src = np.zeros((query_points_norm.shape[0],), dtype=np.int64)
    labels = {
        'gt_tracks_srcw_tn3': gt_tracks_srcw_tn3,
        'gt_visible_tn': gt_visible_tn,
        'gt_occluded_tn': gt_occluded_tn,
        'num_query_points': int(query_points_norm.shape[0]),
    }
    return (
        video_tchw,
        encode_video_tchw,
        labels,
        {
            'query_points_norm': query_points_norm,
            'query_xy_model': query_uv_model.astype(np.float32),
            'tracks_uv_model_tn2': tracks_uv_model_tn2,
            'gt_visible_tn': gt_visible_tn,
            't_src': t_src,
            'video_meta': video_meta,
        },
    )


@torch.no_grad()
def run_point_track_clip_inference(
    model,
    video_clip_tchw: torch.Tensor,
    encoder_clip_tchw: torch.Tensor,
    query_points_norm: np.ndarray,
    t_src: np.ndarray,
    enc_img_size: Tuple[int, int],
    amp_infer: bool,
    max_query_points_per_batch: int,
) -> Dict[str, np.ndarray]:
    video_b = video_clip_tchw.unsqueeze(0)
    encoder_b = encoder_clip_tchw.unsqueeze(0)

    query_points_t = torch.from_numpy(query_points_norm).to(device=video_clip_tchw.device, dtype=torch.float32).unsqueeze(0)
    t_src_t = torch.from_numpy(t_src).to(device=video_clip_tchw.device, dtype=torch.long).unsqueeze(0)
    amp_enabled = bool(amp_infer and video_clip_tchw.device.type == 'cuda')
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
        outputs = model.predict_point_tracks(
            video=video_b,
            query_points=query_points_t,
            t_src=t_src_t,
            encode_video=encoder_b,
            enc_img_size=enc_img_size,
            decoder_dtype=None if amp_enabled else torch.get_default_dtype(),
            max_query_points_per_batch=int(max_query_points_per_batch),
        )
    result = {}
    for key in ['tracks_3d_srcw', 'visibility_srcw']:
        result[key] = outputs[key].squeeze(0).detach().cpu().to(torch.float32).numpy()
    return result


def infer_worldtrack_sparse_tracks(
    model,
    video_tchw: torch.Tensor,
    encoder_video_tchw: torch.Tensor,
    query_points_norm: np.ndarray,
    t_src: np.ndarray,
    clip_frames: int,
    enc_img_size: Tuple[int, int],
    amp_infer: bool,
    max_query_points_per_batch: int,
) -> Dict[str, np.ndarray]:
    num_frames = int(video_tchw.shape[0])
    if not 0 < num_frames <= int(clip_frames):
        raise ValueError(f'Single-clip tracking requires 1 to {clip_frames} frames, got {num_frames}.')
    if int(encoder_video_tchw.shape[0]) != num_frames:
        raise ValueError(f'encoder_video_tchw frame count {int(encoder_video_tchw.shape[0])} does not match video_tchw {num_frames}.')
    model_device = next(model.parameters()).device
    outputs = run_point_track_clip_inference(
        model=model,
        video_clip_tchw=video_tchw.to(model_device),
        encoder_clip_tchw=encoder_video_tchw.to(model_device),
        query_points_norm=query_points_norm,
        t_src=t_src,
        enc_img_size=enc_img_size,
        amp_infer=amp_infer,
        max_query_points_per_batch=max_query_points_per_batch,
    )
    return {
        'tracks_srcw_tn3': outputs['tracks_3d_srcw'].transpose(1, 0, 2).copy(),
        'visibility_srcw_tn': outputs['visibility_srcw'].T.copy(),
    }


def create_prediction_dirs(
    scene_output_dir: Path,
) -> Dict[str, Path]:
    pred_root = scene_output_dir / 'predictions'
    pred_dirs = {'srcw': pred_root / 'dataset_points-mode_srcw'}
    for path in pred_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return pred_dirs


def save_predictions_per_frame(
    out_dir: Path,
    tracks_3d_tn3: np.ndarray,
    visibility_tn: np.ndarray,
) -> None:
    t_count = int(tracks_3d_tn3.shape[0])
    for t in range(t_count):
        np.save(out_dir / f"tracks_3d_{t:05d}.npy", tracks_3d_tn3[t].astype(np.float32))
        np.save(out_dir / f"visibility_{t:05d}.npy", visibility_tn[t].astype(np.float32))


def try_load_predictions_per_frame(
    out_dir: Path,
    num_frames: int,
) -> Optional[Dict[str, np.ndarray]]:
    tracks_3d_frames = []
    visibility_frames = []
    n_points = None
    for t in range(num_frames):
        path_3d = out_dir / f"tracks_3d_{t:05d}.npy"
        path_vis = out_dir / f"visibility_{t:05d}.npy"
        if (not path_3d.exists()) or (not path_vis.exists()):
            return None
        try:
            arr_3d = np.load(path_3d)
            arr_vis = np.load(path_vis)
        except Exception:
            return None
        if arr_3d.ndim != 2 or arr_3d.shape[1] != 3:
            return None
        if arr_vis.ndim != 1:
            return None
        if arr_3d.shape[0] != arr_vis.shape[0]:
            return None
        if n_points is None:
            n_points = int(arr_3d.shape[0])
        elif int(arr_3d.shape[0]) != n_points:
            return None
        tracks_3d_frames.append(arr_3d.astype(np.float32))
        visibility_frames.append(arr_vis.astype(np.float32))
    return {
        "tracks_3d_tn3": np.stack(tracks_3d_frames, axis=0),
        "visibility_tn": np.stack(visibility_frames, axis=0),
    }


def try_load_cached_scene_predictions(
    pred_dirs: Dict[str, Path],
    num_frames: int,
    expected_num_queries: int,
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    srcw = try_load_predictions_per_frame(pred_dirs["srcw"], num_frames)
    if srcw is None:
        return None
    if int(srcw["tracks_3d_tn3"].shape[1]) != expected_num_queries:
        return None
    return {"srcw": srcw}


def sample_point_colors_from_image(
    source_rgb: np.ndarray,
    query_xy: np.ndarray,
) -> np.ndarray:
    if query_xy.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    h, w = source_rgb.shape[:2]
    rgb = source_rgb.astype(np.float32)

    x = np.clip(query_xy[:, 0].astype(np.float32), 0.0, float(max(w - 1, 0)))
    y = np.clip(query_xy[:, 1].astype(np.float32), 0.0, float(max(h - 1, 0)))

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, max(w - 1, 0))
    y1 = np.clip(y0 + 1, 0, max(h - 1, 0))

    wx = (x - x0.astype(np.float32))[:, None]
    wy = (y - y0.astype(np.float32))[:, None]

    c00 = rgb[y0, x0]
    c10 = rgb[y0, x1]
    c01 = rgb[y1, x0]
    c11 = rgb[y1, x1]

    c0 = c00 * (1.0 - wx) + c10 * wx
    c1 = c01 * (1.0 - wx) + c11 * wx
    c = c0 * (1.0 - wy) + c1 * wy
    return np.clip(c, 0.0, 255.0).astype(np.uint8)


def desaturate_by_visibility(
    colors: np.ndarray,
    visible_mask: np.ndarray,
) -> np.ndarray:
    gray = np.mean(colors.astype(np.float32), axis=1, keepdims=True)
    gray_rgb = np.repeat(gray, 3, axis=1)
    out = colors.astype(np.float32).copy()
    invis = ~visible_mask
    out[invis] = 0.2 * out[invis] + 0.8 * gray_rgb[invis]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def input_artifacts_complete(
    input_dir: Path,
    num_frames: int,
    target_fps: float = 10.0,
) -> bool:
    if not input_dir.is_dir():
        return False
    for frame_idx in range(int(num_frames)):
        if not (input_dir / f"frame_{frame_idx:05d}.png").is_file():
            return False
    video_path = input_dir / "input.mp4"
    if not video_path.is_file():
        return False
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False
    video_frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
    video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if video_frame_count != int(num_frames) or abs(video_fps - float(target_fps)) > 0.5:
        return False
    with open(video_path, "rb") as f:
        header = f.read()
    return b"avc1" in header and b"mp4v" not in header


def export_input_visualizations(
    input_dir: Path,
    video_rgb_tnhwc: np.ndarray,
    tracks_uv_tn2: np.ndarray,
    visibility_tn: np.ndarray,
    fps: float = 10.0,
) -> None:
    input_dir.mkdir(exist_ok=True)
    video_path = input_dir / "input.mp4"
    video_writer = imageio.get_writer(str(video_path), fps=float(fps), codec="libx264", macro_block_size=None)
    try:
        for frame_idx in range(int(video_rgb_tnhwc.shape[0])):
            frame_rgb = np.asarray(video_rgb_tnhwc[frame_idx], dtype=np.uint8).copy()
            for point_idx in range(int(tracks_uv_tn2.shape[1])):
                xy = np.asarray(tracks_uv_tn2[frame_idx, point_idx], dtype=np.float32)
                if not np.isfinite(xy).all():
                    continue
                x = int(np.round(float(xy[0])))
                y = int(np.round(float(xy[1])))
                if x < 0 or x >= int(frame_rgb.shape[1]) or y < 0 or y >= int(frame_rgb.shape[0]):
                    continue
                thickness = -1 if bool(visibility_tn[frame_idx, point_idx]) else 2
                cv2.circle(frame_rgb, (x, y), 3, (255, 0, 0), thickness, lineType=cv2.LINE_AA)
                cv2.circle(frame_rgb, (x, y), 4, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            Image.fromarray(frame_rgb).save(input_dir / f"frame_{frame_idx:05d}.png")
            video_writer.append_data(frame_rgb)
    finally:
        video_writer.close()


def save_mode_srcw_rrd(
    output_path: Path,
    tracks_tn3: np.ndarray,
    visibility_tn: np.ndarray,
    query_xy: np.ndarray,
    source_rgb: np.ndarray,
    stream_name: str,
    gt_tracks_tn3: Optional[np.ndarray] = None,
) -> None:
    """
    Save point tracks into rerun using the OpenCV RDF convention.
    """
    try:
        import rerun as rr
    except Exception as e:
        print(f"[rerun-sdk] Skip {output_path}, rerun import failed: {e}")
        return

    base_colors = sample_point_colors_from_image(source_rgb=source_rgb, query_xy=query_xy)
    gt_tracks_rrd_tn3 = None
    if gt_tracks_tn3 is not None:
        if gt_tracks_tn3.shape != tracks_tn3.shape:
            raise ValueError(f"gt_tracks_tn3 shape mismatch: {gt_tracks_tn3.shape}, expected {tracks_tn3.shape}")
        gt_tracks_rrd_tn3 = gt_tracks_tn3.astype(np.float32)

    ref_points_n3 = gt_tracks_rrd_tn3[0] if gt_tracks_rrd_tn3 is not None else tracks_tn3[0]
    ref_finite_n = np.isfinite(ref_points_n3).all(axis=-1)
    ref_points_n3 = ref_points_n3[ref_finite_n]
    if ref_points_n3.shape[0] > 0:
        ref_min = ref_points_n3.min(axis=0)
        ref_max = ref_points_n3.max(axis=0)
        ref_bbox_diag = float(np.linalg.norm(ref_max - ref_min))
        ref_positive_z = ref_points_n3[:, 2][ref_points_n3[:, 2] > 1e-6]
        if ref_positive_z.shape[0] > 0:
            ref_forward_dist = float(np.median(ref_positive_z))
        else:
            ref_forward_dist = max(ref_bbox_diag * 0.5, 0.1)
    else:
        ref_bbox_diag = 1.0
        ref_forward_dist = 1.0

    axis_len = max(ref_bbox_diag * 0.2, ref_forward_dist * 0.25, 0.1)
    point_radius = max(ref_bbox_diag * 0.003, 0.01)
    line_radius = max(point_radius * 0.5, 0.005)
    eye_z = -0.25 * axis_len
    look_z = max(ref_forward_dist, axis_len)

    rr.init(stream_name, spawn=False)
    rr.save(str(output_path))
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    rr.log("world/xyz", rr.Arrows3D(vectors=[[axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, axis_len]], colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]], radii=[line_radius, line_radius, line_radius]), static=True)
    try:
        import rerun.blueprint as rrb

        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(
                name="Main View",
                origin="/",
                contents=["/world/**", f"/{stream_name}/points", f"/{stream_name}/gt_points", f"/{stream_name}/pred_gt_links"],
                eye_controls=rrb.EyeControls3D(position=[0.0, 0.0, eye_z], look_target=[0.0, 0.0, look_z], eye_up=[0.0, -1.0, 0.0]),
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint, make_active=True, make_default=True)
    except Exception as e:
        print(f"[rerun-sdk] Blueprint setup skipped for {output_path.name}: {e}")

    for t in range(tracks_tn3.shape[0]):
        rr.set_time("frame_idx", sequence=t)
        visible_t = visibility_tn[t] > PRED_VISIBLITY_THRES
        colors_t = desaturate_by_visibility(base_colors, visible_t)
        rr.log(f"{stream_name}/points", rr.Points3D(tracks_tn3[t], colors=colors_t, radii=point_radius))
        if gt_tracks_rrd_tn3 is not None:
            rr.log(f"{stream_name}/gt_points", rr.Points3D(gt_tracks_rrd_tn3[t], colors=base_colors, radii=point_radius))
            pred_gt_segments = np.stack([tracks_tn3[t], gt_tracks_rrd_tn3[t]], axis=1)
            pred_gt_colors = np.tile(np.array([[0, 0, 255]], dtype=np.uint8), (pred_gt_segments.shape[0], 1))
            rr.log(f"{stream_name}/pred_gt_links", rr.LineStrips3D(pred_gt_segments, colors=pred_gt_colors, radii=line_radius))
        if t > 0:
            prev_visible_t = visibility_tn[t - 1] > PRED_VISIBLITY_THRES
            seg_vis = visible_t & prev_visible_t
            if np.any(seg_vis):
                segments = np.stack([tracks_tn3[t - 1, seg_vis], tracks_tn3[t, seg_vis]], axis=1)
                trace_colors = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (segments.shape[0], 1))
                rr.log(f"{stream_name}/trace", rr.LineStrips3D(segments, colors=trace_colors, radii=line_radius))


def get_scene_placeholder_path(
    output_root_dir: Path,
    scene_name: str,
    placeholder_kind: str,
) -> Path:
    return output_root_dir / SCENE_PLACEHOLDER_ROOT_DIR_NAME / placeholder_kind / scene_name


def try_create_scene_placeholder(
    output_root_dir: Path,
    scene_name: str,
    placeholder_kind: str,
) -> Tuple[bool, Path]:
    placeholder_path = get_scene_placeholder_path(output_root_dir, scene_name, placeholder_kind)
    placeholder_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        placeholder_path.mkdir()
        return True, placeholder_path
    except FileExistsError:
        return False, placeholder_path


def remove_scene_placeholder(
    placeholder_path: Path,
) -> None:
    try:
        placeholder_path.rmdir()
    except OSError:
        pass


def write_json_atomic(
    json_path: Path,
    payload: Dict[str, object],
    compact_scalar_list_keys: Optional[List[str]] = None,
) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if compact_scalar_list_keys is not None:
        for key in compact_scalar_list_keys:
            if key not in payload:
                continue
            value = payload[key]
            if not isinstance(value, list):
                continue
            if not all(isinstance(item, (int, float, bool)) for item in value):
                continue
            compact = json.dumps(value, ensure_ascii=False)
            pattern = rf'"{re.escape(key)}": \[[\s\S]*?\]'
            text = re.sub(pattern, f'"{key}": {compact}', text, count=1)
    tmp_path = json_path.with_name(f"{json_path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, json_path)


def compute_global_scale_factor(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
) -> float:
    gt_flat = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)
    pred_flat = np.asarray(pred_points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(gt_flat).all(axis=-1) & np.isfinite(pred_flat).all(axis=-1)
    if not np.any(finite):
        return 1.0
    gt_norm = np.linalg.norm(gt_flat[finite], axis=-1)
    pred_norm = np.linalg.norm(pred_flat[finite], axis=-1)
    eps = 1e-12
    if gt_norm.size <= 0 or pred_norm.size <= 0:
        return 1.0
    gt_norm = np.maximum(gt_norm, eps)
    pred_norm = np.maximum(pred_norm, eps)
    return float(np.median(gt_norm) / max(float(np.median(pred_norm)), eps))


def estimate_sim3_closed_form(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    centroid_src = src.mean(axis=0, keepdims=True)
    centroid_dst = dst.mean(axis=0, keepdims=True)
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    h = src_centered.T @ dst_centered
    u, s, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] = -vt[-1, :]
        rot = vt.T @ u.T
    var_src = float((src_centered**2).sum())
    scale = float(np.sum(s) / max(var_src, 1e-12))
    trans = centroid_dst[0] - scale * (rot @ centroid_src[0])
    return scale, rot, trans


def estimate_sim3_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    iterations: int = 1000,
    inlier_threshold: float = 0.05,
) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 points for Sim3 estimation.")
    best_count = -1
    best_model = None
    best_mask = None
    rng = np.random.default_rng(0)
    for _ in range(int(iterations)):
        subset_idx = rng.choice(src.shape[0], size=3, replace=False)
        try:
            scale, rot, trans = estimate_sim3_closed_form(src[subset_idx], dst[subset_idx])
        except np.linalg.LinAlgError:
            continue
        transformed = scale * (rot @ src.T).T + trans
        dists = np.linalg.norm(transformed - dst, axis=1)
        mask = dists < float(inlier_threshold)
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_model = (scale, rot, trans)
            best_mask = mask
    if best_model is None:
        return estimate_sim3_closed_form(src, dst)
    if best_mask is not None and int(best_mask.sum()) >= 3:
        return estimate_sim3_closed_form(src[best_mask], dst[best_mask])
    return best_model


def finite_correspondence_pairs(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    src_flat = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst_flat = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(src_flat).all(axis=-1) & np.isfinite(dst_flat).all(axis=-1)
    return src_flat[finite], dst_flat[finite]


def apply_alignment_transform(
    points_tn3: np.ndarray,
    scale: Optional[float],
    rot: Optional[np.ndarray],
    trans: Optional[np.ndarray],
) -> np.ndarray:
    points = np.asarray(points_tn3, dtype=np.float64)
    if scale is None or rot is None or trans is None:
        return np.full_like(points, np.nan, dtype=np.float64)
    points_flat = points.reshape(-1, 3)
    finite = np.isfinite(points_flat).all(axis=-1)
    aligned_flat = np.full_like(points_flat, np.nan, dtype=np.float64)
    aligned_flat[finite] = (float(scale) * (np.asarray(rot, dtype=np.float64) @ points_flat[finite].T)).T + np.asarray(trans, dtype=np.float64)
    return aligned_flat.reshape(points.shape)


def compute_track_alignment(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    alignment_mode: str,
) -> Dict[str, object]:
    gt = np.asarray(gt_points, dtype=np.float64)
    pred = np.asarray(pred_points, dtype=np.float64)
    if alignment_mode == "global_median":
        scale = compute_global_scale_factor(gt, pred)
        rot = np.eye(3, dtype=np.float64)
        trans = np.zeros((3,), dtype=np.float64)
        pred_aligned = pred * float(scale)
    elif alignment_mode == "global_sim3":
        src_fit, dst_fit = finite_correspondence_pairs(pred, gt)
        if src_fit.shape[0] < 3:
            scale = None
            rot = None
            trans = None
            pred_aligned = np.full_like(pred, np.nan, dtype=np.float64)
        else:
            if src_fit.shape[0] > 16384:
                rng = np.random.default_rng(0)
                pick = rng.choice(src_fit.shape[0], size=16384, replace=False)
                src_sample = src_fit[pick]
                dst_sample = dst_fit[pick]
            else:
                src_sample = src_fit
                dst_sample = dst_fit
            scale, rot, trans = estimate_sim3_ransac(src_sample, dst_sample)
            pred_aligned = apply_alignment_transform(pred, scale, rot, trans)
    else:
        raise ValueError(f"Unsupported alignment_mode: {alignment_mode}")
    return {
        "alignment_mode": str(alignment_mode),
        "alignment_scale": None if scale is None else float(scale),
        "alignment_rot": None if rot is None else np.asarray(rot, dtype=np.float64),
        "alignment_trans": None if trans is None else np.asarray(trans, dtype=np.float64),
        "pred_tracks_aligned_tn3": np.asarray(pred_aligned, dtype=np.float32),
    }


def threshold_pred_visibility(
    pred_visibility_tn: np.ndarray,
    pred_tracks_tn3: np.ndarray,
) -> np.ndarray:
    pred_visible = np.asarray(pred_visibility_tn, dtype=np.float32) > float(PRED_VISIBLITY_THRES)
    finite_track_mask = np.isfinite(pred_tracks_tn3).all(axis=-1)
    return pred_visible & finite_track_mask


def compute_apd_metrics(
    gt_tracks_tn3: np.ndarray,
    pred_tracks_tn3: np.ndarray,
    alignment_mode: str,
) -> Dict[str, object]:
    alignment = compute_track_alignment(gt_tracks_tn3, pred_tracks_tn3, alignment_mode=alignment_mode)
    pred_aligned_tn3 = np.asarray(alignment["pred_tracks_aligned_tn3"], dtype=np.float64)
    gt = np.asarray(gt_tracks_tn3, dtype=np.float64)
    dists = np.linalg.norm(pred_aligned_tn3 - gt, axis=-1)
    finite = np.isfinite(dists)
    total_points = int(np.count_nonzero(finite))
    fractions_by_thresh = {}
    for thresh_key, fixed_threshold in WORLDTRACK_FIXED_THRESHOLDS.items():
        within = finite & (dists <= float(fixed_threshold))
        fractions_by_thresh[str(thresh_key)] = float(100.0 * np.count_nonzero(within) / max(total_points, 1))
    apd_3d = float(np.mean(list(fractions_by_thresh.values()))) if fractions_by_thresh else float("nan")
    epe = float(np.mean(dists[finite])) if np.any(finite) else float("inf")
    return {
        "alignment_mode": str(alignment_mode),
        "alignment_scale": alignment["alignment_scale"],
        "APD_3D": apd_3d,
        "EPE": epe,
        "correct_rate_by_thresh": fractions_by_thresh,
        "num_finite_points": total_points,
        "pred_tracks_aligned_tn3": pred_aligned_tn3.astype(np.float32),
        "alignment_rot": alignment["alignment_rot"],
        "alignment_trans": alignment["alignment_trans"],
    }


def build_empty_srcw_metrics(
    num_query_points: int,
) -> Dict[str, object]:
    return {
        "APD_3D": None,
        "EPE": None,
        "alignment_scale": None,
        "correct_rate_by_thresh": {str(k): None for k in WORLDTRACK_FIXED_THRESHOLDS},
        "num_finite_points": 0,
        "num_query_points": int(num_query_points),
    }


def build_subset_srcw_metrics(
    query_mask: np.ndarray,
    gt_tracks_srcw_tn3: np.ndarray,
    pred_tracks_srcw_tn3: np.ndarray,
    alignment_mode: str,
) -> Dict[str, object]:
    query_mask = np.asarray(query_mask, dtype=bool)
    num_query_points = int(np.count_nonzero(query_mask))
    if num_query_points <= 0:
        return build_empty_srcw_metrics(num_query_points=0)

    srcw_apd = compute_apd_metrics(gt_tracks_srcw_tn3[:, query_mask], pred_tracks_srcw_tn3[:, query_mask], alignment_mode=alignment_mode)
    return {
        "alignment_mode": str(alignment_mode),
        "APD_3D": srcw_apd["APD_3D"],
        "EPE": srcw_apd["EPE"],
        "alignment_scale": srcw_apd["alignment_scale"],
        "correct_rate_by_thresh": srcw_apd["correct_rate_by_thresh"],
        "num_finite_points": srcw_apd["num_finite_points"],
        "num_query_points": num_query_points,
    }


def compute_motion_query_groups(
    gt_tracks_srcw_tn3: np.ndarray,
) -> Dict[str, object]:
    metric_num_frames = int(gt_tracks_srcw_tn3.shape[0])
    motion_frame_indices = list(range(metric_num_frames))
    adjacent_motion_tn = np.linalg.norm(np.diff(np.asarray(gt_tracks_srcw_tn3, dtype=np.float64), axis=0), axis=-1)
    motion_total_n = np.sum(adjacent_motion_tn, axis=0)
    motion_valid_n = np.isfinite(adjacent_motion_tn).all(axis=0)
    static_mask_n = motion_valid_n & (motion_total_n < float(STATIC_WORLD_MOTION_MAX))
    dynamic_mask_n = motion_valid_n & (motion_total_n >= float(STATIC_WORLD_MOTION_MAX))
    return {
        "motion_frame_idx_local": motion_frame_indices,
        "motion_valid_n": motion_valid_n,
        "static_mask_n": static_mask_n,
        "dynamic_mask_n": dynamic_mask_n,
        "motion_split_query_counts": {
            "all": int(gt_tracks_srcw_tn3.shape[1]),
            "motion_valid": int(np.count_nonzero(motion_valid_n)),
            "static": int(np.count_nonzero(static_mask_n)),
            "dynamic": int(np.count_nonzero(dynamic_mask_n)),
        },
    }


def build_scene_group_metrics(
    query_group: str,
    scene_name: str,
    sequence_path: str,
    labels: Dict[str, np.ndarray],
    pred_tracks_srcw_tn3: np.ndarray,
    pred_visibility_srcw_tn: np.ndarray,
    sampled_indices_global: np.ndarray,
    video_meta: dict,
    metric_num_frames: int,
    alignment_mode: str,
) -> Dict[str, object]:
    gt_tracks_srcw_tn3 = labels['gt_tracks_srcw_tn3'][:metric_num_frames]
    pred_tracks_srcw_tn3 = pred_tracks_srcw_tn3[:metric_num_frames]
    pred_visibility_srcw_tn = pred_visibility_srcw_tn[:metric_num_frames]
    sampled_indices_global = sampled_indices_global[:metric_num_frames]
    motion_group_info = compute_motion_query_groups(gt_tracks_srcw_tn3)
    if query_group == 'static':
        query_mask = np.asarray(motion_group_info['static_mask_n'], dtype=bool)
    elif query_group == 'dynamic':
        query_mask = np.asarray(motion_group_info['dynamic_mask_n'], dtype=bool)
    else:
        raise ValueError(f'Unsupported query_group: {query_group}')
    pred_visible_srcw_tn = threshold_pred_visibility(pred_visibility_srcw_tn, pred_tracks_srcw_tn3)
    mode_srcw = build_subset_srcw_metrics(query_mask=query_mask, gt_tracks_srcw_tn3=gt_tracks_srcw_tn3, pred_tracks_srcw_tn3=pred_tracks_srcw_tn3, alignment_mode=alignment_mode)
    motion_split_query_counts = dict(motion_group_info['motion_split_query_counts'])
    motion_split_query_counts['selected'] = int(np.count_nonzero(query_mask))
    visible_fraction = None
    if np.count_nonzero(query_mask) > 0:
        visible_fraction = float(np.mean(pred_visible_srcw_tn[:, query_mask].astype(np.float32)))
    return {
        'scene': scene_name,
        'sequence_path': sequence_path,
        'query_group': str(query_group),
        'metric_units': {
            'APD_3D': APD_3D_UNIT,
            'correct_rate_by_thresh': 'percent',
            'EPE': 'meter',
        },
        'alignment_mode': str(alignment_mode),
        'alignment_scope': 'within_query_group',
        'infer_total_num_frames': int(metric_num_frames),
        'num_query_points': int(np.count_nonzero(query_mask)),
        'source_frame_local': 0,
        'metrics_mode_srcw': mode_srcw,
        'pred_visibility_thres': float(PRED_VISIBLITY_THRES),
        'static_world_motion_max_m': float(STATIC_WORLD_MOTION_MAX),
        'motion_split_frames_local': list(motion_group_info['motion_frame_idx_local']),
        'motion_split_frames_global': [int(sampled_indices_global[idx]) for idx in motion_group_info['motion_frame_idx_local']],
        'motion_split_query_counts': motion_split_query_counts,
        'fixed_metric_thresholds_m': {str(k): float(v) for k, v in WORLDTRACK_FIXED_THRESHOLDS.items()},
        'video_meta': {
            'orig_shape': [int(video_meta['orig_shape'][0]), int(video_meta['orig_shape'][1])],
            'center_crop': bool(video_meta['center_crop']),
            'crop_shape': [int(video_meta['crop_shape'][0]), int(video_meta['crop_shape'][1])],
            'crop_offsets': [int(video_meta['crop_offsets'][0]), int(video_meta['crop_offsets'][1])],
            'proc_shape': [int(video_meta['proc_shape'][0]), int(video_meta['proc_shape'][1])],
            'model_shape': [int(video_meta['model_shape'][0]), int(video_meta['model_shape'][1])],
            'encode_shape': [int(video_meta['encode_shape'][0]), int(video_meta['encode_shape'][1])],
        },
        'srcw_visibility_pred_stats': {'visible_fraction': visible_fraction},
        'sampled_indices_global': [int(v) for v in sampled_indices_global.tolist()],
    }


def build_scene_metrics(
    scene_name: str,
    sequence_path: str,
    labels: Dict[str, np.ndarray],
    pred_tracks_srcw_tn3: np.ndarray,
    pred_visibility_srcw_tn: np.ndarray,
    clip_frames: int,
    num_unique_clips: int,
    sampled_indices_global: np.ndarray,
    video_meta: dict,
    metric_num_frames: int,
    alignment_mode: str,
) -> Dict[str, object]:
    gt_tracks_srcw_tn3 = labels['gt_tracks_srcw_tn3'][:metric_num_frames]
    pred_tracks_srcw_tn3 = pred_tracks_srcw_tn3[:metric_num_frames]
    pred_visibility_srcw_tn = pred_visibility_srcw_tn[:metric_num_frames]
    sampled_indices_global = sampled_indices_global[:metric_num_frames]
    pred_visible_srcw_tn = threshold_pred_visibility(pred_visibility_srcw_tn, pred_tracks_srcw_tn3)
    motion_group_info = compute_motion_query_groups(gt_tracks_srcw_tn3)
    mode_srcw = build_subset_srcw_metrics(query_mask=np.ones((gt_tracks_srcw_tn3.shape[1],), dtype=bool), gt_tracks_srcw_tn3=gt_tracks_srcw_tn3, pred_tracks_srcw_tn3=pred_tracks_srcw_tn3, alignment_mode=alignment_mode)
    metrics_per_frame_mode_srcw = []
    for frame_idx_local in range(metric_num_frames):
        frame_idx_global = int(sampled_indices_global[frame_idx_local])
        srcw_apd_frame = compute_apd_metrics(gt_tracks_srcw_tn3[frame_idx_local : frame_idx_local + 1], pred_tracks_srcw_tn3[frame_idx_local : frame_idx_local + 1], alignment_mode=alignment_mode)
        metrics_per_frame_mode_srcw.append(
            build_metric_payload_for_frame(
                frame_idx_local=frame_idx_local,
                frame_idx_global=frame_idx_global,
                metrics={
                    'alignment_mode': str(alignment_mode),
                    'APD_3D': srcw_apd_frame['APD_3D'],
                    'EPE': srcw_apd_frame['EPE'],
                    'alignment_scale': srcw_apd_frame['alignment_scale'],
                    'correct_rate_by_thresh': srcw_apd_frame['correct_rate_by_thresh'],
                    'num_finite_points': srcw_apd_frame['num_finite_points'],
                },
            )
        )
    return {
        'scene': scene_name,
        'sequence_path': sequence_path,
        'metric_units': {
            'APD_3D': APD_3D_UNIT,
            'correct_rate_by_thresh': 'percent',
            'EPE': 'meter',
        },
        'infer_total_num_frames': int(metric_num_frames),
        'num_query_points': int(labels['num_query_points']),
        'source_frame_local': 0,
        'alignment_mode': str(alignment_mode),
        'metrics_mode_srcw': mode_srcw,
        'clip_frames': int(clip_frames),
        'num_unique_clips': int(num_unique_clips),
        'pred_visibility_thres': float(PRED_VISIBLITY_THRES),
        'static_world_motion_max_m': float(STATIC_WORLD_MOTION_MAX),
        'motion_split_frames_local': list(motion_group_info['motion_frame_idx_local']),
        'motion_split_frames_global': [int(sampled_indices_global[idx]) for idx in motion_group_info['motion_frame_idx_local']],
        'motion_split_query_counts': dict(motion_group_info['motion_split_query_counts']),
        'fixed_metric_thresholds_m': {str(k): float(v) for k, v in WORLDTRACK_FIXED_THRESHOLDS.items()},
        'video_meta': {
            'orig_shape': [int(video_meta['orig_shape'][0]), int(video_meta['orig_shape'][1])],
            'center_crop': bool(video_meta['center_crop']),
            'crop_shape': [int(video_meta['crop_shape'][0]), int(video_meta['crop_shape'][1])],
            'crop_offsets': [int(video_meta['crop_offsets'][0]), int(video_meta['crop_offsets'][1])],
            'proc_shape': [int(video_meta['proc_shape'][0]), int(video_meta['proc_shape'][1])],
            'model_shape': [int(video_meta['model_shape'][0]), int(video_meta['model_shape'][1])],
            'encode_shape': [int(video_meta['encode_shape'][0]), int(video_meta['encode_shape'][1])],
        },
        'srcw_visibility_pred_stats': {'visible_fraction': float(np.mean(pred_visible_srcw_tn.astype(np.float32)))},
        'sampled_indices_global': [int(v) for v in sampled_indices_global.tolist()],
        'metrics_per_frame_mode_srcw': metrics_per_frame_mode_srcw,
    }


def resolve_multi_scene_names(
    data_root: Path,
    split_name: str,
    target_scene_list: List[str],
) -> List[str]:
    split_dir = data_root / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"WorldTrack split directory not found: {split_dir}")
    if len(target_scene_list) == 1 and str(target_scene_list[0]).strip() == "*":
        return sorted(path.stem for path in split_dir.glob("*.npz"))
    resolved = []
    for raw_name in target_scene_list:
        scene_name = str(raw_name).strip()
        if scene_name.endswith(".npz"):
            scene_stem = Path(scene_name).stem
        else:
            scene_stem = scene_name
        seq_path = split_dir / f"{scene_stem}.npz"
        if not seq_path.exists():
            raise FileNotFoundError(f"WorldTrack sequence not found: {seq_path}")
        resolved.append(scene_stem)
    return resolved


def resolve_sequence_path(
    data_root: Path,
    split_name: str,
    scene_name: str,
) -> Path:
    seq_path = data_root / split_name / f"{scene_name}.npz"
    if not seq_path.exists():
        raise FileNotFoundError(f"WorldTrack sequence not found: {seq_path}")
    return seq_path


def mean_scalar_from_scene_modes(
    scene_entries: List[Dict[str, object]],
    mode_key: str,
) -> Dict[str, object]:
    scalars = {}
    dict_scalars = {}
    for item in scene_entries:
        metrics = item.get(mode_key)
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, dict):
                bucket = dict_scalars.setdefault(key, {})
                for sub_key, sub_value in value.items():
                    if sub_value is None:
                        continue
                    bucket.setdefault(sub_key, []).append(float(sub_value))
            elif isinstance(value, (int, float)):
                scalars.setdefault(key, []).append(float(value))
    out = {}
    for key, values in scalars.items():
        out[key] = float(np.mean(values)) if values else None
    for key, sub_map in dict_scalars.items():
        out[key] = {sub_key: float(np.mean(sub_values)) for sub_key, sub_values in sub_map.items()}
    return out


def write_group_benchmark_summary(
    benchmark_root_dir: Path,
    split_name: str,
    seed: int,
    start_frame: int,
    infer_total_num_frames: int,
    infer_chunk_num_frames: int,
    alignment_mode: str,
    query_group: str,
) -> None:
    scene_entries = []
    metrics_file_name = f"pttrack_eval_metrics-{query_group}.json"
    for scene_dir in sorted(benchmark_root_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        metrics_path = scene_dir / metrics_file_name
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            scene_metrics = json.load(f)
        if scene_metrics.get("metric_units", {}).get("APD_3D") != APD_3D_UNIT:
            continue
        scene_entries.append(
            {
                "scene": scene_dir.name,
                "metrics_path": str(Path(scene_dir.name) / metrics_file_name),
                "num_query_points": int(scene_metrics.get("num_query_points", 0)),
                "metrics_mode_srcw": scene_metrics.get("metrics_mode_srcw"),
            }
        )

    summary = {
        "split": split_name,
        "metric_units": {
            "APD_3D": APD_3D_UNIT,
            "correct_rate_by_thresh": "percent",
            "EPE": "meter",
        },
        "seed": int(seed),
        "start_frame": int(start_frame),
        "infer_total_num_frames": int(infer_total_num_frames),
        "infer_chunk_num_frames": int(infer_chunk_num_frames),
        "alignment_mode": str(alignment_mode),
        "query_group": str(query_group),
        "alignment_scope": "within_query_group",
        "num_scenes": int(len(scene_entries)),
        "metrics_mode_srcw_mean": mean_scalar_from_scene_modes(scene_entries, "metrics_mode_srcw"),
        "scenes": scene_entries,
    }
    write_json_atomic(benchmark_root_dir / f"pttrack_bench_metrics-{query_group}.json", summary)


def write_benchmark_summary(
    benchmark_root_dir: Path,
    split_name: str,
    seed: int,
    start_frame: int,
    infer_total_num_frames: int,
    infer_chunk_num_frames: int,
    alignment_mode: str,
) -> None:
    scene_entries = []
    for scene_dir in sorted(benchmark_root_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        metrics_path = scene_dir / "pttrack_eval_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            scene_metrics = json.load(f)
        if scene_metrics.get("metric_units", {}).get("APD_3D") != APD_3D_UNIT:
            continue
        scene_entries.append(
            {
                "scene": scene_dir.name,
                "metrics_path": str(Path(scene_dir.name) / "pttrack_eval_metrics.json"),
                "num_query_points": int(scene_metrics.get("num_query_points", 0)),
                "metrics_mode_srcw": scene_metrics.get("metrics_mode_srcw"),
            }
        )

    summary = {
        "split": split_name,
        "metric_units": {
            "APD_3D": APD_3D_UNIT,
            "correct_rate_by_thresh": "percent",
            "EPE": "meter",
        },
        "seed": int(seed),
        "start_frame": int(start_frame),
        "infer_total_num_frames": int(infer_total_num_frames),
        "infer_chunk_num_frames": int(infer_chunk_num_frames),
        "alignment_mode": str(alignment_mode),
        "num_scenes": int(len(scene_entries)),
        "metrics_mode_srcw_mean": mean_scalar_from_scene_modes(scene_entries, "metrics_mode_srcw"),
        "scenes": scene_entries,
    }
    write_json_atomic(benchmark_root_dir / "pttrack_bench_metrics-ALL.json", summary)
    write_group_benchmark_summary(
        benchmark_root_dir=benchmark_root_dir,
        split_name=split_name,
        seed=seed,
        start_frame=start_frame,
        infer_total_num_frames=infer_total_num_frames,
        infer_chunk_num_frames=infer_chunk_num_frames,
        alignment_mode=alignment_mode,
        query_group="static",
    )
    write_group_benchmark_summary(
        benchmark_root_dir=benchmark_root_dir,
        split_name=split_name,
        seed=seed,
        start_frame=start_frame,
        infer_total_num_frames=infer_total_num_frames,
        infer_chunk_num_frames=infer_chunk_num_frames,
        alignment_mode=alignment_mode,
        query_group="dynamic",
    )


def bindWorkerDevice(
    num_workers: int,
) -> int:
    visible_cuda_devices = get_visible_cuda_devices()
    worker_identity = mp.current_process()._identity
    worker_idx = int(worker_identity[0]) - 1 if worker_identity else 0
    if visible_cuda_devices:
        assigned_device_count = min(len(visible_cuda_devices), num_workers)
        local_cuda_idx = worker_idx % assigned_device_count
        device = torch.device(f"cuda:{local_cuda_idx}")
        torch.cuda.set_device(device)
        global WORKER_DEVICE
        WORKER_DEVICE = device
    else:
        WORKER_DEVICE = torch.device("cpu")
    return worker_idx


def initSceneWorker(
    worker_args,
    num_workers: int,
) -> None:
    global WORKER_ARGS
    global WORKER_MODEL
    WORKER_ARGS = worker_args
    bindWorkerDevice(num_workers)
    setRandomness(seed=worker_args.seed, deterministic=worker_args.deterministic)
    WORKER_MODEL = load_model(checkpoint_path=worker_args.checkpoint, device=WORKER_DEVICE, cli_args=worker_args)


def run_scene_benchmark(
    scene_name: str,
) -> Dict[str, object]:
    assert WORKER_ARGS is not None
    assert WORKER_MODEL is not None
    output_root_dir = Path(WORKER_ARGS.output_dir)
    scene_output_dir = output_root_dir / scene_name
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    processing_placeholder_path = None
    sequence_path = resolve_sequence_path(Path(WORKER_ARGS.data_root), WORKER_ARGS.split, scene_name)
    available_num_frames = resolve_sequence_num_frames(npz_path=sequence_path, requested_num_frames=-1, start_frame=int(WORKER_ARGS.start_frame), frame_stride=int(WORKER_ARGS.frame_stride))
    metric_num_frames = resolve_sequence_num_frames(npz_path=sequence_path, requested_num_frames=int(WORKER_ARGS.infer_total_num_frames), start_frame=int(WORKER_ARGS.start_frame), frame_stride=int(WORKER_ARGS.frame_stride))
    infer_chunk_num_frames = int(get_model_clip_frames(WORKER_MODEL, WORKER_ARGS.infer_chunk_num_frames))
    if metric_num_frames > infer_chunk_num_frames:
        raise ValueError(f'Scene {scene_name} requests {metric_num_frames} evaluation frames, but single-clip tracking supports at most {infer_chunk_num_frames}. Reduce --infer_total_num_frames to fit the model window.')
    inference_num_frames = infer_chunk_num_frames
    if int(available_num_frames) < int(inference_num_frames):
        raise ValueError(
            f'Scene {scene_name} has only {available_num_frames} available frames from '
            f'start_frame={int(WORKER_ARGS.start_frame)}, but inference requires '
            f'{inference_num_frames} frames (infer_total_num_frames={metric_num_frames}, '
            f'infer_chunk_num_frames={infer_chunk_num_frames}).'
        )
    sample = load_worldtrack_sequence(npz_path=sequence_path, start_frame=int(WORKER_ARGS.start_frame), num_frames=int(inference_num_frames), frame_stride=int(WORKER_ARGS.frame_stride))
    try:
        video_tchw, encode_video_tchw, labels, query_payload = build_sparse_queries_and_gt(
            sample=sample, enc_img_h=int(WORKER_ARGS.img_h), enc_img_w=int(WORKER_ARGS.img_w), proc_h=WORKER_ARGS.proc_h, proc_w=WORKER_ARGS.proc_w, raw_rgb=bool(WORKER_ARGS.raw_rgb), center_crop=bool(WORKER_ARGS.center_crop)
        )
    except RuntimeError as e:
        err_msg = str(e)
        if 'Frame-0 has no in-plane visible query points.' not in err_msg:
            raise
        skip_note_path = scene_output_dir / 'skip_no_frame0_points.txt'
        with open(skip_note_path, 'w', encoding='utf-8') as f:
            f.write(
                f'scene={scene_name}\n'
                f'reason={err_msg}\n'
                f'start_frame={int(WORKER_ARGS.start_frame)}\n'
                f'frame_stride={int(WORKER_ARGS.frame_stride)}\n'
                f'infer_total_num_frames={int(metric_num_frames)}\n'
                f'infer_chunk_num_frames={int(infer_chunk_num_frames)}\n'
                f'sequence_path={sequence_path}\n'
            )
        print(f'[Skip] {scene_name}: {err_msg}')
        return {'scene': scene_name, 'status': 'skip_no_frame0_points', 'reason': err_msg}
    pred_dirs = create_prediction_dirs(scene_output_dir)
    input_dir = scene_output_dir / 'input'
    video_np_uint8 = np.clip(video_tchw.permute(0, 2, 3, 1).cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8)
    if not input_artifacts_complete(input_dir, int(video_np_uint8.shape[0]), target_fps=10.0):
        export_input_visualizations(input_dir=input_dir, video_rgb_tnhwc=video_np_uint8, tracks_uv_tn2=np.asarray(query_payload['tracks_uv_model_tn2'], dtype=np.float32), visibility_tn=np.asarray(query_payload['gt_visible_tn'], dtype=bool), fps=10.0)
    cached = try_load_cached_scene_predictions(pred_dirs=pred_dirs, num_frames=int(inference_num_frames), expected_num_queries=int(labels['num_query_points']))
    try:
        if cached is None:
            acquired, processing_placeholder_path = try_create_scene_placeholder(output_root_dir, scene_name, SCENE_PLACEHOLDER_PROCESSING_DIR_NAME)
            if not acquired:
                return {'scene': scene_name, 'status': 'skip_processing'}
            outputs = infer_worldtrack_sparse_tracks(
                model=WORKER_MODEL,
                video_tchw=video_tchw,
                encoder_video_tchw=encode_video_tchw,
                query_points_norm=query_payload['query_points_norm'],
                t_src=query_payload['t_src'],
                clip_frames=inference_num_frames,
                enc_img_size=(int(WORKER_ARGS.img_h), int(WORKER_ARGS.img_w)),
                amp_infer=bool(WORKER_ARGS.amp_infer),
                max_query_points_per_batch=int(WORKER_ARGS.max_query_points_per_batch),
            )
            save_predictions_per_frame(pred_dirs['srcw'], outputs['tracks_srcw_tn3'], outputs['visibility_srcw_tn'])
            cached = {
                'srcw': {
                    'tracks_3d_tn3': outputs['tracks_srcw_tn3'],
                    'visibility_tn': outputs['visibility_srcw_tn'],
                }
            }
        metrics = build_scene_metrics(
            scene_name=scene_name,
            sequence_path=str(sequence_path),
            labels=labels,
            pred_tracks_srcw_tn3=cached['srcw']['tracks_3d_tn3'],
            pred_visibility_srcw_tn=cached['srcw']['visibility_tn'],
            clip_frames=inference_num_frames,
            num_unique_clips=1,
            sampled_indices_global=sample['sampled_indices_global'],
            video_meta=query_payload['video_meta'],
            metric_num_frames=int(metric_num_frames),
            alignment_mode=str(WORKER_ARGS.alignment_mode),
        )
        group_metrics_static = build_scene_group_metrics(
            query_group='static',
            scene_name=scene_name,
            sequence_path=str(sequence_path),
            labels=labels,
            pred_tracks_srcw_tn3=cached['srcw']['tracks_3d_tn3'],
            pred_visibility_srcw_tn=cached['srcw']['visibility_tn'],
            sampled_indices_global=sample['sampled_indices_global'],
            video_meta=query_payload['video_meta'],
            metric_num_frames=int(metric_num_frames),
            alignment_mode=str(WORKER_ARGS.alignment_mode),
        )
        group_metrics_dynamic = build_scene_group_metrics(
            query_group='dynamic',
            scene_name=scene_name,
            sequence_path=str(sequence_path),
            labels=labels,
            pred_tracks_srcw_tn3=cached['srcw']['tracks_3d_tn3'],
            pred_visibility_srcw_tn=cached['srcw']['visibility_tn'],
            sampled_indices_global=sample['sampled_indices_global'],
            video_meta=query_payload['video_meta'],
            metric_num_frames=int(metric_num_frames),
            alignment_mode=str(WORKER_ARGS.alignment_mode),
        )
        metric_gt_tracks_srcw_tn3 = labels['gt_tracks_srcw_tn3'][:metric_num_frames]
        motion_group_info = compute_motion_query_groups(metric_gt_tracks_srcw_tn3)
        group_query_masks = {
            'static': np.asarray(motion_group_info['static_mask_n'], dtype=bool),
            'dynamic': np.asarray(motion_group_info['dynamic_mask_n'], dtype=bool),
        }
        group_metrics_by_name = {'static': group_metrics_static, 'dynamic': group_metrics_dynamic}
        sparse_align_srcw = compute_track_alignment(labels['gt_tracks_srcw_tn3'], cached['srcw']['tracks_3d_tn3'], alignment_mode=str(WORKER_ARGS.alignment_mode))
        save_mode_srcw_rrd(
            output_path=scene_output_dir / 'dataset_points-mode_srcw.rrd',
            tracks_tn3=np.asarray(sparse_align_srcw['pred_tracks_aligned_tn3'], dtype=np.float32),
            visibility_tn=cached['srcw']['visibility_tn'],
            query_xy=query_payload['query_xy_model'],
            source_rgb=video_np_uint8[0],
            stream_name='dataset_points_mode_srcw',
            gt_tracks_tn3=labels['gt_tracks_srcw_tn3'],
        )
        for query_group, query_mask in group_query_masks.items():
            write_json_atomic(scene_output_dir / f'pttrack_eval_metrics-{query_group}.json', group_metrics_by_name[query_group], compact_scalar_list_keys=['sampled_indices_global'])
            group_tracks_srcw_tn3 = cached['srcw']['tracks_3d_tn3'][:, query_mask]
            if np.count_nonzero(query_mask) > 0:
                group_align_srcw = compute_track_alignment(metric_gt_tracks_srcw_tn3[:, query_mask], cached['srcw']['tracks_3d_tn3'][:metric_num_frames, query_mask], alignment_mode=str(WORKER_ARGS.alignment_mode))
                group_tracks_srcw_aligned_tn3 = apply_alignment_transform(group_tracks_srcw_tn3, group_align_srcw['alignment_scale'], group_align_srcw['alignment_rot'], group_align_srcw['alignment_trans']).astype(np.float32)
            else:
                group_tracks_srcw_aligned_tn3 = np.asarray(group_tracks_srcw_tn3, dtype=np.float32)
            save_mode_srcw_rrd(
                output_path=scene_output_dir / f'dataset_points-mode_srcw-{query_group}.rrd',
                tracks_tn3=group_tracks_srcw_aligned_tn3,
                visibility_tn=cached['srcw']['visibility_tn'][:, query_mask],
                query_xy=query_payload['query_xy_model'][query_mask],
                source_rgb=video_np_uint8[0],
                stream_name=f'dataset_points_mode_srcw_{query_group}',
                gt_tracks_tn3=labels['gt_tracks_srcw_tn3'][:, query_mask],
            )
        write_json_atomic(scene_output_dir / 'pttrack_eval_metrics.json', metrics, compact_scalar_list_keys=['sampled_indices_global'])
        metrics_path = scene_output_dir / 'pttrack_eval_metrics.json'
        print('\n--- Point Track Eval Summary ---')
        print(f'Scene                        : {scene_name}')
        print(f"Query points                 : {metrics['num_query_points']}")
        print(f"Alignment mode              : {metrics['alignment_mode']}")
        print(f"Motion split counts          : {metrics['motion_split_query_counts']}")
        for label, srcw_metrics in (
            ('Mode-SRCW', metrics['metrics_mode_srcw']),
            ('Mode-SRCW Static', group_metrics_static['metrics_mode_srcw']),
            ('Mode-SRCW Dynamic', group_metrics_dynamic['metrics_mode_srcw']),
        ):
            apd_3d = srcw_metrics['APD_3D']
            epe = srcw_metrics['EPE']
            apd_text = f'{apd_3d:.3f}' if apd_3d is not None else 'None'
            epe_text = f'{epe:.4f}' if epe is not None else 'None'
            print(f'{label + " APD_3D (%)":<30}: {apd_text}')
            print(f'{label + " EPE":<30}: {epe_text}')
        print(f'Metrics file                 : {metrics_path}')
        print('--------------------------------\n')
        try_create_scene_placeholder(output_root_dir, scene_name, SCENE_PLACEHOLDER_DONE_DIR_NAME)
        return {'scene': scene_name, 'status': 'done', 'metrics_path': str(metrics_path)}
    finally:
        if processing_placeholder_path is not None:
            remove_scene_placeholder(processing_placeholder_path)


def run_scene_benchmark_with_config(
    task,
) -> Dict[str, object]:
    """
    Switch one persistent worker to the current split configuration before processing a scene.
    """
    global WORKER_ARGS
    config_args, scene_name = task
    WORKER_ARGS = config_args
    return run_scene_benchmark(scene_name)


def run_multi_scene(
    args,
    initialize_worker: bool = True,
    worker_pool=None,
) -> None:
    global WORKER_ARGS
    data_root = Path(args.data_root).resolve()
    output_root_dir = Path(args.output_dir).resolve()
    output_root_dir.mkdir(parents=True, exist_ok=True)
    selected_scene_names = resolve_multi_scene_names(data_root, args.split, list(args.target_scene_list))
    random.Random(args.seed).shuffle(selected_scene_names)

    visible_cuda_devices = get_visible_cuda_devices()
    if args.num_workers is not None:
        num_workers = int(args.num_workers)
    elif visible_cuda_devices:
        num_workers = len(visible_cuda_devices)
    else:
        num_workers = 1
    if visible_cuda_devices:
        num_workers = min(num_workers, len(visible_cuda_devices))
    num_workers = max(1, num_workers)

    if worker_pool is not None:
        tasks = [(args, scene_name) for scene_name in selected_scene_names]
        list(worker_pool.imap_unordered(run_scene_benchmark_with_config, tasks))
    elif num_workers <= 1:
        if initialize_worker:
            initSceneWorker(args, num_workers)
        else:
            WORKER_ARGS = args
        for scene_name in selected_scene_names:
            run_scene_benchmark(scene_name)
    else:
        pool_context = mp.get_context("spawn")
        with pool_context.Pool(processes=num_workers, initializer=initSceneWorker, initargs=(args, num_workers)) as pool:
            list(pool.imap_unordered(run_scene_benchmark, selected_scene_names))

    infer_chunk_num_frames = int(args.infer_chunk_num_frames) if args.infer_chunk_num_frames is not None else 0
    write_benchmark_summary(
        benchmark_root_dir=output_root_dir,
        split_name=args.split,
        seed=args.seed,
        start_frame=args.start_frame,
        infer_total_num_frames=args.infer_total_num_frames,
        infer_chunk_num_frames=infer_chunk_num_frames,
        alignment_mode=str(args.alignment_mode),
    )


def run_split_configs(
    args,
) -> None:
    """
    Run splits sequentially through persistent GPU workers that each load the model once.
    """
    configs = []
    for raw_config in args.split_configs:
        fields = str(raw_config).split(",", 5)
        if len(fields) != 6:
            raise ValueError(f"Invalid --split_configs entry {raw_config!r}; expected split,proc_h,proc_w,img_h,img_w,output_dir")
        split_name, proc_h, proc_w, img_h, img_w, output_dir = fields
        config_args = argparse.Namespace(**vars(args))
        config_args.split = split_name
        config_args.proc_h = int(proc_h)
        config_args.proc_w = int(proc_w)
        config_args.img_h = int(img_h)
        config_args.img_w = int(img_w)
        config_args.output_dir = output_dir
        if min(config_args.proc_h, config_args.proc_w, config_args.img_h, config_args.img_w) <= 0:
            raise ValueError(f"All proc/img dimensions must be positive in --split_configs entry {raw_config!r}")
        validate_image_aspect_ratios(config_args.img_h, config_args.img_w, config_args.proc_h, config_args.proc_w)
        configs.append(config_args)

    visible_cuda_devices = get_visible_cuda_devices()
    if args.num_workers is not None:
        num_workers = int(args.num_workers)
    elif visible_cuda_devices:
        num_workers = len(visible_cuda_devices)
    else:
        num_workers = 1
    if visible_cuda_devices:
        num_workers = min(num_workers, len(visible_cuda_devices))
    num_workers = max(1, num_workers)

    def run_configs(
        worker_pool=None,
    ) -> None:
        for config_args in configs:
            setRandomness(seed=config_args.seed, deterministic=config_args.deterministic)
            print(f"[Split Config] split={config_args.split} workers={num_workers} proc={config_args.proc_h}x{config_args.proc_w} img={config_args.img_h}x{config_args.img_w} output={Path(config_args.output_dir).resolve()}")
            run_multi_scene(config_args, initialize_worker=False, worker_pool=worker_pool)
            if worker_pool is None and WORKER_DEVICE is not None and WORKER_DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    if num_workers <= 1:
        initSceneWorker(configs[0], num_workers)
        run_configs()
    else:
        pool_context = mp.get_context("spawn")
        with pool_context.Pool(processes=num_workers, initializer=initSceneWorker, initargs=(configs[0], num_workers)) as pool:
            run_configs(worker_pool=pool)


def main():
    args = parse_args()
    if args.split_configs:
        run_split_configs(args)
    else:
        run_multi_scene(args)


if __name__ == "__main__":
    main()
