#!/usr/bin/env python3
"""
Dense D4RT inference and 4D tracking visualization with viser.
"""

import argparse
import colorsys
import copy
import gc
import hashlib
import multiprocessing as mp
import os
import re
import signal
import sys
import threading
import time
from contextlib import closing
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_ROOT = REPO_ROOT / "inputs"
OUTPUT_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_CAMERA_CENTER_RDF = (0.0, 0.0, -0.3)
DEFAULT_CAMERA_LOOK_AT_RDF = (0.0, 0.0, 1.0)
DEFAULT_CAMERA_UP_RDF = (0.0, -1.0, 0.0)
DEFAULT_CAMERA_FOV_DEG = 45.0
DEFAULT_CAMERA_VIEWPORT_FILL = 0.95  # Central viewport width/height fraction; use 0 < value <= 1.

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from utils.inference import str2bool, setRandomness, load_model, prepareDecodeContext, get_visible_cuda_devices
from utils.inference import validate_image_aspect_ratios


def parse_args():
    """
    Parse dense inference and viser options.
    """
    parser = argparse.ArgumentParser(description="Run dense D4RT inference and visualize the result with viser")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a D4RT checkpoint")
    parser.add_argument("--dvd_checkpoint", type=str, default=None, help="Override ckpt/dvd_1.1.safetensors")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input_path", type=str, help="Video file, image folder, or WorldTrack NPZ; relative paths resolve from the current directory, repository root, or inputs/")
    inputs.add_argument("--input_paths", nargs="+", help="Batch inputs: keep live servers when inputs fit on the worker GPUs; otherwise export offline")
    parser.add_argument("--num_workers", type=int, default=None, help="Batch workers; defaults to one per visible GPU")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_ROOT), help="Directory for saved .viser recordings")
    parser.add_argument("--infer_chunk_num_frames", type=int, default=None, help="Total frames encoded in one model call; defaults to checkpoint num_frames")
    parser.add_argument("--infer_total_num_frames", type=int, default=None, help=("Prefix frames decoded and visualized from the encoded chunk; defaults to infer_chunk_num_frames"))
    parser.add_argument("--infer_visualize_frame_list", type=int, nargs="+", default=None, help=("Strictly increasing 0-based indices below infer_total_num_frames; defaults to the full prefix"))
    parser.add_argument("--auto_adapt_frame_count", type=str2bool, default=False, help="On short input, use the largest available 4N+1 frame prefix instead of failing")
    parser.add_argument("--dense_grid_stride", type=int, default=4, help="Pixel stride of the dense query grid")
    parser.add_argument("--track_stride", type=int, default=10, help="Draw one srcw trajectory for every N dense queries")
    parser.add_argument("--track_length", type=int, default=5, help="Number of recent visualization-frame segments drawn per srcw trajectory")
    parser.add_argument("--track_line_width", type=float, default=2.0, help="Newest srcw trajectory line width; older segments decay linearly")
    parser.add_argument("--distance_scale", type=float, default=1.0, help="Target frame-0 mean point distance to world origin used to scale all geometry")
    parser.add_argument("--point_size", type=float, default=None, help="Initial viser point size; defaults to a value derived from distance_scale")
    parser.add_argument("--show_rdf", type=str2bool, default=False, help="Show the origin RDF axes (+X right, +Y down, +Z forward)")
    parser.add_argument("--confidence_percentile", type=float, default=0.0, help="Hide queries below this frame-0 confidence percentile in viser; 0 disables filtering")
    parser.add_argument("--foreground_mask_root", type=str, default=None, help="Optional root containing <scene>/00000.png query-frame masks for source-0 foreground track selection in Viser")
    parser.add_argument("--img_h", type=int, default=None, help="Encoder/VAE input height; defaults to checkpoint config")
    parser.add_argument("--img_w", type=int, default=None, help="Encoder/VAE input width; defaults to checkpoint config")
    parser.add_argument("--proc_h", type=int, default=None, help="Decoder patch-frame height; defaults to checkpoint config")
    parser.add_argument("--proc_w", type=int, default=None, help="Decoder patch-frame width; defaults to checkpoint config")
    parser.add_argument("--center_crop", type=str2bool, default=True, help="Max-center-crop each source frame to the proc aspect ratio before resize")
    parser.add_argument("--train_num_frames", type=int, default=2, help="Optional training frame-count override for time-embedding resize")
    parser.add_argument("--max_query_points_per_batch", type=int, default=2048, help="Decoder query batch size")
    parser.add_argument("--pre_vae_encode", type=str2bool, default=None, help="Raw input inference requires false; defaults to false")
    parser.add_argument("--only_condition_t_but_no_noise", type=str2bool, default=True, help="Must be true: keep the timestep condition and disable latent noise")
    parser.add_argument("--wan_torch_dtype", type=str, default=None, help="Optional Wan inference dtype override")
    parser.add_argument("--amp_infer", type=str2bool, default=False, help="Use bfloat16 CUDA autocast for inference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081, help="Preferred Viser port; batch workers request consecutive ports and skip occupied ones")
    parser.add_argument("--fps", type=float, default=15.0, help="Live playback and .viser recording FPS")
    parser.add_argument("--serve", type=str2bool, default=True, help="Keep a live viser server running after saving the recording")
    args = parser.parse_args()
    args.parser_defaults = {key: parser.get_default(key) for key in vars(args).keys()}

    if args.num_workers is not None and args.num_workers <= 0:
        parser.error("--num_workers must be positive.")
    if (args.img_h is None) != (args.img_w is None):
        parser.error("--img_h and --img_w must be provided together.")
    if (args.proc_h is None) != (args.proc_w is None):
        parser.error("--proc_h and --proc_w must be provided together.")
    if args.img_h is not None and args.proc_h is not None:
        try:
            validate_image_aspect_ratios(args.img_h, args.img_w, args.proc_h, args.proc_w)
        except ValueError as error:
            parser.error(str(error))
    if args.infer_chunk_num_frames is not None and args.infer_chunk_num_frames <= 0:
        parser.error("--infer_chunk_num_frames must be positive.")
    if args.infer_total_num_frames is not None and args.infer_total_num_frames <= 0:
        parser.error("--infer_total_num_frames must be positive.")
    if args.dense_grid_stride <= 0 or args.track_stride <= 0 or args.track_length <= 0:
        parser.error("--dense_grid_stride, --track_stride, and --track_length must be positive.")
    if args.track_line_width <= 0.0:
        parser.error("--track_line_width must be positive.")
    if args.distance_scale <= 0.0 or args.max_query_points_per_batch <= 0 or args.fps <= 0.0:
        parser.error("--distance_scale, --max_query_points_per_batch, and --fps must be positive.")
    if args.point_size is not None and args.point_size <= 0.0:
        parser.error("--point_size must be positive when provided.")
    if not 0.0 <= args.confidence_percentile <= 100.0:
        parser.error("--confidence_percentile must lie in [0, 100].")
    if args.pre_vae_encode is True:
        parser.error("--pre_vae_encode=True is incompatible with raw video/image inputs.")
    args.pre_vae_encode = False
    return args


def read_checkpoint_args(
    checkpoint_path: Path,
) -> dict:
    """
    Read checkpoint args without materializing tensor storage when mmap is supported.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_args = checkpoint.get("args", {})
    checkpoint_args = dict(raw_args) if isinstance(raw_args, dict) else dict(vars(raw_args))
    del checkpoint
    return checkpoint_args


def resolve_runtime_args(
    args,
    checkpoint_args: dict,
):
    """
    Resolve optional runtime sizes from checkpoint config while preserving explicit CLI overrides.
    """
    if args.infer_chunk_num_frames is None:
        checkpoint_num_frames = checkpoint_args.get("num_frames", None)
        if checkpoint_num_frames is None:
            raise ValueError("Checkpoint has no num_frames; provide --infer_chunk_num_frames.")
        args.infer_chunk_num_frames = int(checkpoint_num_frames)

    if args.img_h is None:
        checkpoint_img_h = checkpoint_args.get("img_h", None)
        checkpoint_img_w = checkpoint_args.get("img_w", None)
        if checkpoint_img_h is not None and checkpoint_img_w is not None:
            args.img_h, args.img_w = int(checkpoint_img_h), int(checkpoint_img_w)
        elif checkpoint_args.get("img_size", None) is not None:
            args.img_h = args.img_w = int(checkpoint_args["img_size"])
        else:
            raise ValueError("Checkpoint has no img_h/img_w or img_size; provide --img_h and --img_w.")

    if args.proc_h is None:
        checkpoint_proc_h = checkpoint_args.get("proc_h", None)
        checkpoint_proc_w = checkpoint_args.get("proc_w", None)
        if checkpoint_proc_h is None or checkpoint_proc_w is None:
            raise ValueError("Checkpoint has no proc_h/proc_w; provide --proc_h and --proc_w.")
        args.proc_h, args.proc_w = int(checkpoint_proc_h), int(checkpoint_proc_w)

    if min(args.infer_chunk_num_frames, args.img_h, args.img_w, args.proc_h, args.proc_w) <= 0:
        raise ValueError("Resolved frame count and image sizes must be positive.")
    validate_image_aspect_ratios(args.img_h, args.img_w, args.proc_h, args.proc_w)
    if args.infer_total_num_frames is None:
        args.infer_total_num_frames = int(args.infer_chunk_num_frames)
    if args.infer_total_num_frames > args.infer_chunk_num_frames:
        raise ValueError("--infer_total_num_frames must not exceed --infer_chunk_num_frames.")

    if args.infer_visualize_frame_list is None:
        args.infer_visualize_frame_list = list(range(args.infer_total_num_frames))
    else:
        frame_list = [int(frame_idx) for frame_idx in args.infer_visualize_frame_list]
        if frame_list != sorted(set(frame_list)):
            raise ValueError("--infer_visualize_frame_list must contain unique, strictly increasing indices.")
        if frame_list[0] < 0 or frame_list[-1] >= args.infer_total_num_frames:
            raise ValueError("--infer_visualize_frame_list indices must lie below --infer_total_num_frames.")
        args.infer_visualize_frame_list = frame_list
    return args


def resolve_input_path(
    input_path: str,
) -> Path:
    """
    Resolve an input from the current directory, repository root, or inputs/.
    """
    path = Path(input_path).expanduser()
    if not path.is_absolute() and not path.exists():
        repo_path = REPO_ROOT / path
        path = repo_path if repo_path.exists() else INPUT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def natural_sort_key(
    path: Path,
):
    """
    Return a numeric-aware key for image-frame filenames.
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def convert_frame_to_rgb(
    frame: np.ndarray,
    source: Path,
) -> np.ndarray:
    """
    Convert one OpenCV image to uint8 RGB.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"Expected a 3- or 4-channel image from {source}, got {None if frame is None else frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 input from {source}, got {frame.dtype}")
    conversion = cv2.COLOR_BGRA2RGB if frame.shape[2] == 4 else cv2.COLOR_BGR2RGB
    return cv2.cvtColor(frame, conversion)


def load_input_frames(
    input_path: Path,
    num_frames: int,
    allow_short: bool = False,
) -> list[np.ndarray]:
    """
    Load up to num_frames from a WorldTrack NPZ, image folder, or video and optionally allow short
    input.
    """
    frames = []
    if input_path.suffix.lower() == ".npz":
        with np.load(input_path, allow_pickle=True) as pack:
            if "images_jpeg_bytes" not in pack.files:
                raise ValueError(f"WorldTrack NPZ '{input_path}' does not contain images_jpeg_bytes.")
            images_jpeg_bytes = np.asarray(pack["images_jpeg_bytes"])
            if images_jpeg_bytes.ndim != 1:
                raise ValueError(f"WorldTrack NPZ '{input_path}' has invalid images_jpeg_bytes shape {images_jpeg_bytes.shape}.")
            if images_jpeg_bytes.shape[0] < num_frames and not allow_short:
                raise ValueError(f"WorldTrack NPZ '{input_path}' has {images_jpeg_bytes.shape[0]} frames, but infer_chunk_num_frames={num_frames}.")
            for frame_idx, frame_bytes in enumerate(images_jpeg_bytes[:num_frames]):
                frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"Could not decode frame {frame_idx} from WorldTrack NPZ: {input_path}")
                frames.append(convert_frame_to_rgb(frame, input_path))
        if len({frame.shape for frame in frames}) != 1:
            raise ValueError(f"WorldTrack NPZ '{input_path}' contains inconsistent frame shapes.")
    elif input_path.is_dir():
        image_paths = sorted([path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}], key=natural_sort_key)
        if len(image_paths) < num_frames and not allow_short:
            raise ValueError(f"Input folder '{input_path}' has {len(image_paths)} frames, but infer_chunk_num_frames={num_frames}.")
        for image_path in image_paths[:num_frames]:
            frames.append(convert_frame_to_rgb(cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED), image_path))
    else:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        while len(frames) < num_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(convert_frame_to_rgb(frame, input_path))
        capture.release()
        if len(frames) < num_frames and not allow_short:
            raise ValueError(f"Input video '{input_path}' has only {len(frames)} readable frames, but infer_chunk_num_frames={num_frames}.")
    return frames


def prepare_frames(
    frames,
    proc_h: int,
    proc_w: int,
    img_h: int,
    img_w: int,
    center_crop: bool,
):
    """
    Build proc-sized decoder patch frames and img-sized encoder/VAE frames from each frame's own
    source shape.
    """
    patch_frames = []
    encoder_frames = []
    target_ratio = float(proc_w) / float(proc_h)
    for frame in frames:
        orig_h, orig_w = frame.shape[:2]
        if center_crop:
            source_ratio = float(orig_w) / float(orig_h)
            if source_ratio > target_ratio:
                crop_h, crop_w = orig_h, max(1, int(round(orig_h * target_ratio)))
            else:
                crop_h, crop_w = max(1, int(round(orig_w / target_ratio))), orig_w
            crop_top = (orig_h - crop_h) // 2
            crop_left = (orig_w - crop_w) // 2
            frame = frame[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w]
        patch_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        encoder_frame = cv2.resize(patch_frame, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        patch_frames.append(patch_frame)
        encoder_frames.append(encoder_frame)
    patch_frames = np.stack(patch_frames, axis=0)
    encoder_frames = np.stack(encoder_frames, axis=0)
    video_tchw = torch.from_numpy(patch_frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
    encoder_tchw = torch.from_numpy(encoder_frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()

    return patch_frames, video_tchw, encoder_tchw


def build_dense_queries(
    height: int,
    width: int,
    stride: int,
    source_rgb: np.ndarray,
):
    """
    Build a full-image pixel-center grid and take each query's exact source-pixel RGB color.
    """
    x_indices = np.arange(0, width, stride, dtype=np.int64)
    y_indices = np.arange(0, height, stride, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(x_indices, y_indices, indexing="xy")
    query_points = np.stack([(grid_x.astype(np.float32) + 0.5) / float(width), (grid_y.astype(np.float32) + 0.5) / float(height)], axis=-1).reshape(-1, 2)
    point_colors = source_rgb[grid_y.reshape(-1), grid_x.reshape(-1)].astype(np.uint8)
    return query_points, point_colors


def load_foreground_mask(
    mask_root: str,
    scene_name: str,
    proc_h: int,
    proc_w: int,
    center_crop: bool,
) -> np.ndarray:
    """
    Load the source-frame foreground mask and align it with the processed RGB frames.
    """
    mask_path = Path(mask_root).expanduser().resolve() / scene_name / "00000.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read query-frame mask: {mask_path}")
    foreground = np.any(mask[..., :3] != 0, axis=-1) if mask.ndim == 3 else mask != 0
    if center_crop:
        orig_h, orig_w = foreground.shape
        target_ratio = float(proc_w) / float(proc_h)
        source_ratio = float(orig_w) / float(orig_h)
        if source_ratio > target_ratio:
            crop_h, crop_w = orig_h, max(1, int(round(orig_h * target_ratio)))
        else:
            crop_h, crop_w = max(1, int(round(orig_w / target_ratio))), orig_w
        crop_top = (orig_h - crop_h) // 2
        crop_left = (orig_w - crop_w) // 2
        foreground = foreground[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w]
    return cv2.resize(foreground.astype(np.uint8), (proc_w, proc_h), interpolation=cv2.INTER_NEAREST).astype(bool)


@torch.no_grad()
def infer_dense_tracks(
    model,
    video_tchw: torch.Tensor,
    encoder_tchw: torch.Tensor,
    query_points: np.ndarray,
    frame_list,
    args,
    scene_name: str,
):
    """
    Encode the complete chunk once and decode srcw tracks and query-frame confidence with t_cam=0.
    """
    device = next(model.parameters()).device
    video_tchw = video_tchw.to(device)
    encoder_tchw = encoder_tchw.to(device)
    decoder_dtype = None if bool(args.amp_infer and device.type == "cuda") else torch.get_default_dtype()
    video_b, encoder_features = prepareDecodeContext(model=model, video_clip_tchw=video_tchw, encoder_clip_tchw=encoder_tchw, enc_img_size=(args.img_h, args.img_w), amp_infer=args.amp_infer, decoder_dtype=decoder_dtype)
    query_points_t = torch.from_numpy(query_points).to(device=device, dtype=torch.float32).unsqueeze(0)
    if decoder_dtype is not None:
        query_points_t = query_points_t.to(decoder_dtype)
    num_queries = int(query_points.shape[0])
    tracks = np.empty((len(frame_list), num_queries, 3), dtype=np.float32)
    src0_confidence = np.empty(num_queries, dtype=np.float32)
    amp_enabled = bool(args.amp_infer and device.type == "cuda")

    decode_frame_list = list(dict.fromkeys([0] + list(frame_list)))
    output_frame_indices = {frame_idx: output_idx for output_idx, frame_idx in enumerate(frame_list)}
    for decode_frame_idx, target_frame_idx in enumerate(decode_frame_list):
        for point_start in range(0, num_queries, args.max_query_points_per_batch):
            point_end = min(point_start + args.max_query_points_per_batch, num_queries)
            point_count = point_end - point_start
            coords_batch = query_points_t[:, point_start:point_end]
            t_src = torch.zeros((1, point_count), device=device, dtype=torch.long)
            t_tgt = torch.full((1, point_count), target_frame_idx, device=device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                outputs = model.decode(encoder_features, video_b, coords_batch, t_src, t_tgt, t_src)
            position = outputs["pos_3d"].squeeze(0)
            if target_frame_idx in output_frame_indices:
                output_frame_idx = output_frame_indices[target_frame_idx]
                tracks[output_frame_idx, point_start:point_end] = position.detach().cpu().to(torch.float32).numpy()
            if target_frame_idx == 0:
                confidence = outputs["confidence"].squeeze(0)
                confidence = confidence.reshape(-1)
                src0_confidence[point_start:point_end] = confidence.detach().cpu().to(torch.float32).numpy()
        print(f"[dense] [{decode_frame_idx + 1}/{len(decode_frame_list)}] scene={scene_name} target_frame={target_frame_idx}", flush=True)
    return tracks, src0_confidence


def normalize_tracks(
    tracks: np.ndarray,
    distance_scale: float,
):
    """
    Select finite srcw display points and scale all frames from frame 0's mean point distance to
    world origin without translating geometry.
    """
    valid = np.isfinite(tracks).all(axis=-1)
    valid_points = tracks[0, valid[0]]
    if valid_points.shape[0] == 0:
        raise RuntimeError("No finite frame-0 display points are available for mean-distance normalization.")
    mean_distance = float(np.mean(np.linalg.norm(valid_points, axis=-1)))
    if not np.isfinite(mean_distance) or mean_distance <= 1e-8:
        raise RuntimeError(f"Invalid visible-point mean distance: {mean_distance}")
    scale_factor = float(distance_scale) / (mean_distance + 1e-6)
    return tracks * scale_factor, valid, mean_distance, scale_factor


def build_track_colors(
    num_queries: int,
    foreground_grid: np.ndarray | None = None,
) -> np.ndarray:
    """
    Assign stable rainbow colors by query index or map source-0 foreground v from red to violet.
    """
    hues = [(query_idx * 0.61803398875) % 1.0 for query_idx in range(num_queries)]
    if foreground_grid is not None and np.any(foreground_grid):
        # Uniform query-grid rows are affine in v, so stride and normalization cancel.
        query_v = np.repeat(np.arange(foreground_grid.shape[0]), foreground_grid.shape[1])
        foreground_v = query_v[foreground_grid.reshape(-1)]
        v_min, v_max = foreground_v.min(), foreground_v.max()
        hues = np.clip((query_v - v_min) / max(v_max - v_min, 1), 0.0, 1.0) * 0.75
    colors = []
    for hue in hues:
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append([int(round(channel * 255.0)) for channel in rgb])
    return np.asarray(colors, dtype=np.uint8)


def build_trail_layers(
    timeline_idx: int,
    tracks: np.ndarray,
    valid: np.ndarray,
    track_indices: np.ndarray,
    track_colors: np.ndarray,
    track_length: int,
):
    """
    Build recent srcw trajectory layers with fixed pure-RGB colors and linear line-width decay.
    """
    layers = []
    for age in range(track_length):
        end_idx = timeline_idx - age
        start_idx = end_idx - 1
        if start_idx < 0:
            continue
        selected_valid = valid[start_idx, track_indices] & valid[end_idx, track_indices]
        selected_indices = track_indices[selected_valid]
        if selected_indices.shape[0] == 0:
            continue
        segments = np.stack([tracks[start_idx, selected_indices], tracks[end_idx, selected_indices]], axis=1).astype(np.float32)
        decay = float(track_length - age) / float(track_length)
        colors = track_colors[selected_indices]
        colors = np.repeat(colors[:, None, :], 2, axis=1)
        layers.append((age, segments, colors, decay))
    return layers


def compute_fitted_fov(
    points: np.ndarray,
    position,
    look_at,
    up,
    aspect: float,
) -> float | None:
    """
    Fit point projections inside the configured central viewport box by changing only the vertical
    FOV.
    """
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(look_at, dtype=np.float64) - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    world_to_camera_rotation = np.stack([right, down, forward])
    camera_points = np.einsum("ij,nj->ni", world_to_camera_rotation, points - position)
    front_points = camera_points[:, 2] > 0.0
    if not np.all(front_points):
        print(f"[viser] Warning: {int((~front_points).sum())} points are behind/on the camera plane and cannot be framed by changing FOV alone; camera pose is unchanged.")
    if not np.any(front_points):
        return None
    camera_points = camera_points[front_points]
    projected_extent = np.max(np.abs(camera_points[:, :2] / camera_points[:, 2:3]), axis=0)
    half_fov_tangent = max(float(projected_extent[0]) / aspect, float(projected_extent[1])) / DEFAULT_CAMERA_VIEWPORT_FILL
    fov_deg = float(np.rad2deg(2.0 * np.arctan(half_fov_tangent)))
    if fov_deg > 179.0:
        print(f"[viser] Warning: fitted FOV {fov_deg:g} exceeds the 179-degree limit; complete {DEFAULT_CAMERA_VIEWPORT_FILL * 100:g}-percent framing is not possible with this camera pose.")
    return float(np.deg2rad(np.clip(fov_deg, 1.0, 179.0)))


def make_output_path(
    input_path: Path,
    output_root=OUTPUT_ROOT,
) -> Path:
    """
    Build a second-timestamped .viser path without overwriting an existing recording.
    """
    output_root = Path(output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    input_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", input_path.stem if input_path.is_file() else input_path.name)
    output_path = output_root / f"{datetime.now().strftime('%m_%d-%H_%M_%S')}-{input_name}-srcw.viser"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing viser recording: {output_path}")
    return output_path


def save_and_serve_viser(
    tracks: np.ndarray,
    valid: np.ndarray,
    point_colors: np.ndarray,
    patch_frames: np.ndarray,
    frame_list,
    output_path: Path,
    args,
    src0_foreground_mask: np.ndarray | None = None,
    src0_confidence: np.ndarray | None = None,
):
    """
    Save a fixed-playback .viser recording and optionally continue with an interactive live server.
    """
    import viser
    from viser import _messages

    initial_src0_fg_only_tracks = src0_foreground_mask is not None
    default_show_rdf = bool(getattr(args, "show_rdf", True))
    default_confidence_percentile = float(getattr(args, "confidence_percentile", 0.0))
    confidence_suffix = f"-confp{default_confidence_percentile:g}" if src0_confidence is not None else ""
    initial_output_path = output_path.with_name(f"{output_path.stem}-src0_fg_only_tracks{int(initial_src0_fg_only_tracks)}{confidence_suffix}-rdf{int(default_show_rdf)}.viser")
    if initial_output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing viser recording: {initial_output_path}")
    displayed_points = tracks[valid]
    display_center = (displayed_points.min(axis=0) + displayed_points.max(axis=0)) * 0.5
    display_radius = float(np.linalg.norm(displayed_points - display_center, axis=-1).max())
    default_camera_center = np.asarray(DEFAULT_CAMERA_CENTER_RDF, dtype=np.float64)
    default_camera_look_at = np.asarray(DEFAULT_CAMERA_LOOK_AT_RDF, dtype=np.float64)
    default_camera_forward = default_camera_look_at - default_camera_center
    default_camera_forward = default_camera_forward / np.linalg.norm(default_camera_forward)
    server = viser.ViserServer(host=args.host, port=args.port)
    actual_port = server.get_port()
    if actual_port != args.port:
        print(f"[viser] {output_path.stem}: requested port {args.port} is unavailable; using {actual_port}.", flush=True)
    server.scene.set_up_direction("-y")
    server.scene.set_background_image(np.full((2, 2, 3), 255, dtype=np.uint8), format="png")
    server.initial_camera.position = tuple(default_camera_center)
    server.initial_camera.look_at = tuple(default_camera_look_at)
    server.initial_camera.up = DEFAULT_CAMERA_UP_RDF
    server.initial_camera.fov = np.deg2rad(DEFAULT_CAMERA_FOV_DEG)

    rdf_frame = server.scene.add_frame("/rdf", axes_length=max(args.distance_scale * 0.25, 0.1), axes_radius=max(args.distance_scale * 0.005, 0.002), visible=default_show_rdf)
    server.gui.configure_theme(control_layout="collapsible")
    default_point_size = args.point_size if args.point_size is not None else max(args.distance_scale * 0.02, 0.01)
    with server.gui.add_folder("Timeline", expand_by_default=True):
        frame_slider = server.gui.add_slider("frame", min=0, max=len(frame_list) - 1, step=1, initial_value=0)
        frame_text = server.gui.add_text("encoded_frame", str(frame_list[0]))
        play_box = server.gui.add_checkbox("play", initial_value=False)
        fps_slider = server.gui.add_slider("fps", min=1.0, max=30.0, step=1.0, initial_value=float(np.clip(args.fps, 1.0, 30.0)))
    with server.gui.add_folder("Display", expand_by_default=True):
        show_rdf = server.gui.add_checkbox("show_rdf", initial_value=default_show_rdf)
        point_size = server.gui.add_slider("point_size", min=default_point_size * 0.1, max=default_point_size * 5.0, step=default_point_size * 0.1, initial_value=default_point_size)
        confidence_percentile = server.gui.add_number("confidence_percentile", initial_value=default_confidence_percentile, min=0.0, max=100.0, step=1.0, disabled=src0_confidence is None)
        show_tracks = server.gui.add_checkbox("show_tracks", initial_value=True)
        src0_fg_only_tracks = server.gui.add_checkbox("src0_fg_only_tracks", initial_value=initial_src0_fg_only_tracks, disabled=src0_foreground_mask is None)
        track_length = server.gui.add_number("track_length", initial_value=args.track_length, min=1, step=1)
        track_stride = server.gui.add_number("track_stride", initial_value=args.track_stride, min=1, step=1)
        track_line_width = server.gui.add_number("track_line_width", initial_value=args.track_line_width, min=0.1, step=0.1)
    with server.gui.add_folder("Camera", expand_by_default=True):
        camera_position = server.gui.add_vector3("position", initial_value=default_camera_center, step=0.01)
        camera_look_at = server.gui.add_vector3("look_at", initial_value=default_camera_look_at, step=0.01)
        camera_up = server.gui.add_vector3("up", initial_value=DEFAULT_CAMERA_UP_RDF, step=0.01)
        camera_fov_deg = server.gui.add_number("fov_deg", initial_value=DEFAULT_CAMERA_FOV_DEG, min=1.0, max=179.0, step=0.01)
        lock_camera = server.gui.add_checkbox("lock_camera", initial_value=False)
        read_current_camera_button = server.gui.add_button("read_current_camera")
        apply_camera_button = server.gui.add_button("apply_camera")
        frame_all_points_button = server.gui.add_button("frame_all_points")
    with server.gui.add_folder("Input", expand_by_default=True):
        frame_image = server.gui.add_image(patch_frames[frame_list[0]], label="rgb_frame")
    with server.gui.add_folder("Export", expand_by_default=True):
        save_visualization_button = server.gui.add_button("save_visualization", icon=viser.Icon.DEVICE_FLOPPY)
        save_screenshot_button = server.gui.add_button("save_screenshot", icon=viser.Icon.CAMERA)
        save_mp4_button = server.gui.add_button("save_mp4", icon=viser.Icon.VIDEO)
    export_buttons = (save_visualization_button, save_screenshot_button, save_mp4_button)

    render_handles = []
    render_lock = threading.Lock()
    track_colors = build_track_colors(tracks.shape[1])
    src0_foreground_track_colors = track_colors
    src0_foreground_points = None
    if src0_foreground_mask is not None:
        foreground_grid = src0_foreground_mask[:: args.dense_grid_stride, :: args.dense_grid_stride].astype(bool)
        src0_foreground_points = foreground_grid.reshape(-1)
        src0_foreground_track_colors = build_track_colors(tracks.shape[1], foreground_grid)
    confidence_points = np.ones(tracks.shape[1], dtype=bool)
    cached_confidence_percentile = None

    def apply_online_camera(
        client,
    ):
        """
        Apply the shared Camera GUI values to one connected client.
        """
        with client.atomic():
            client.camera.position = tuple(float(value) for value in camera_position.value)
            client.camera.look_at = tuple(float(value) for value in camera_look_at.value)
            client.camera.up_direction = tuple(float(value) for value in camera_up.value)
            client.camera.fov = np.deg2rad(float(camera_fov_deg.value))
        client.flush()

    def fit_all_points(
        client,
    ):
        """
        Update the Camera GUI pose to fit all displayed inference points while preserving its
        viewing direction.
        """
        forward = np.asarray(camera_look_at.value, dtype=np.float64) - np.asarray(camera_position.value, dtype=np.float64)
        if np.linalg.norm(forward) < 1e-8:
            forward = default_camera_forward
        forward = forward / np.linalg.norm(forward)
        vertical_half_fov = np.deg2rad(float(camera_fov_deg.value)) * 0.5
        horizontal_half_fov = np.arctan(np.tan(vertical_half_fov) * float(client.camera.aspect))
        distance = display_radius * 1.1 / np.sin(min(vertical_half_fov, horizontal_half_fov))
        camera_position.value = tuple((display_center - forward * distance).tolist())
        camera_look_at.value = tuple(display_center.tolist())
        apply_online_camera(client)

    @read_current_camera_button.on_click
    def read_current_camera(
        event,
    ):
        """
        Copy the triggering client's current camera into the editable Camera GUI values.
        """
        assert event.client is not None, "read_current_camera must be clicked from a connected Viser client."
        with server.atomic():
            camera_position.value = tuple(float(value) for value in event.client.camera.position)
            camera_look_at.value = tuple(float(value) for value in event.client.camera.look_at)
            camera_up.value = tuple(float(value) for value in event.client.camera.up_direction)
            camera_fov_deg.value = float(np.rad2deg(event.client.camera.fov))

    @apply_camera_button.on_click
    def apply_camera(
        event,
    ):
        """
        Apply the editable Camera GUI pose to the triggering client.
        """
        assert event.client is not None, "apply_camera must be clicked from a connected Viser client."
        apply_online_camera(event.client)

    @frame_all_points_button.on_click
    def frame_all_points(
        event,
    ):
        """
        Move the triggering camera backward along its viewing direction until all displayed
        inference points fit in its view.
        """
        assert event.client is not None, "frame_all_points must be clicked from a connected Viser client."
        fit_all_points(event.client)

    def render(
        timeline_idx: int,
        rendered_point_size: float,
        render_tracks: bool,
        rendered_track_length: int,
        rendered_track_stride: int,
        rendered_track_line_width: float,
        render_src0_fg_only_tracks: bool,
        rendered_confidence_percentile: float,
        rendered_show_rdf: bool,
    ):
        """
        Replace live scene geometry for one visualization-timeline frame.
        """
        nonlocal render_handles, confidence_points, cached_confidence_percentile
        with render_lock, server.atomic():
            rdf_frame.visible = rendered_show_rdf
            if cached_confidence_percentile != rendered_confidence_percentile:
                confidence_points = np.ones(tracks.shape[1], dtype=bool)
                if src0_confidence is not None and rendered_confidence_percentile > 0.0:
                    finite_confidence = src0_confidence[np.isfinite(src0_confidence)]
                    confidence_threshold = float(np.percentile(finite_confidence, rendered_confidence_percentile)) if finite_confidence.size > 0 else np.inf
                    confidence_points = np.isfinite(src0_confidence) & (src0_confidence >= confidence_threshold)
                cached_confidence_percentile = rendered_confidence_percentile
            for handle in render_handles:
                handle.remove()
            render_handles = []
            point_mask = valid[timeline_idx] & confidence_points
            display_points = tracks[timeline_idx]
            if np.any(point_mask):
                render_handles.append(
                    server.scene.add_point_cloud("/dense/points", points=display_points[point_mask].astype(np.float32), colors=point_colors[point_mask], point_size=float(rendered_point_size), point_shape="circle", precision="float32")
                )
            if render_tracks:
                track_indices = np.arange(0, tracks.shape[1], rendered_track_stride, dtype=np.int64)
                rendered_track_indices = track_indices[confidence_points[track_indices]]
                if render_src0_fg_only_tracks:
                    rendered_track_indices = rendered_track_indices[src0_foreground_points[rendered_track_indices]]
                rendered_track_colors = src0_foreground_track_colors if render_src0_fg_only_tracks else track_colors
                for age, segments, colors, line_width_decay in build_trail_layers(timeline_idx, tracks, valid, rendered_track_indices, rendered_track_colors, rendered_track_length):
                    render_handles.append(server.scene.add_line_segments(f"/dense/tracks/age_{age}", points=segments, colors=colors, line_width=rendered_track_line_width * line_width_decay))
            frame_text.value = str(frame_list[timeline_idx])
            frame_image.image = patch_frames[frame_list[timeline_idx]]

    def render_live(
        _event=None,
        timeline_idx: int | None = None,
    ):
        """
        Render the current GUI settings at the selected frame or an explicit recording frame.
        """
        if timeline_idx is None:
            timeline_idx = int(np.clip(frame_slider.value, 0, len(frame_list) - 1))
            if timeline_idx != frame_slider.value:
                frame_slider.value = timeline_idx
                return
        render(
            timeline_idx,
            float(point_size.value),
            bool(show_tracks.value),
            int(track_length.value),
            int(track_stride.value),
            float(track_line_width.value),
            bool(src0_fg_only_tracks.value),
            float(confidence_percentile.value),
            bool(show_rdf.value),
        )

    render_live()
    displayed_points = tracks[valid & confidence_points[None]]
    fitted_fov = compute_fitted_fov(displayed_points, default_camera_center, default_camera_look_at, DEFAULT_CAMERA_UP_RDF, patch_frames.shape[2] / patch_frames.shape[1])
    if fitted_fov is not None:
        camera_fov_deg.value = float(np.rad2deg(fitted_fov))
        server.initial_camera.fov = fitted_fov

    @server.on_client_connect
    def set_live_rdf_camera(
        client: viser.ClientHandle,
    ):
        """
        Preserve the Camera GUI pose and initialize its FOV from the connecting client's actual
        viewport aspect ratio.
        """
        fitted_fov = compute_fitted_fov(displayed_points, camera_position.value, camera_look_at.value, camera_up.value, float(client.camera.aspect))
        if fitted_fov is not None:
            camera_fov_deg.value = float(np.rad2deg(fitted_fov))
        apply_online_camera(client)

        @client.camera.on_update
        def enforce_locked_camera(
            _camera,
        ):
            if lock_camera.value:
                apply_online_camera(client)

    serializer = server.get_scene_serializer()
    for timeline_idx in range(len(frame_list)):
        render_live(timeline_idx=timeline_idx)
        serializer.insert_sleep(1.0 / float(fps_slider.value))
    initial_output_path.write_bytes(serializer.serialize())
    print(f"[viser] Saved recording: {initial_output_path}")

    if not args.serve:
        server.stop()
        return

    frame_slider.on_update(render_live)
    show_rdf.on_update(render_live)
    point_size.on_update(render_live)
    confidence_percentile.on_update(render_live)
    show_tracks.on_update(render_live)
    src0_fg_only_tracks.on_update(render_live)
    track_length.on_update(render_live)
    track_stride.on_update(render_live)
    track_line_width.on_update(render_live)
    render_live()

    def get_export_settings(
        client,
    ):
        """
        Snapshot the current display settings and triggering client's camera for export.
        """
        settings = {
            "show_rdf": bool(show_rdf.value),
            "show_tracks": bool(show_tracks.value),
            "src0_fg_only_tracks": bool(src0_fg_only_tracks.value),
            "confidence_percentile": float(confidence_percentile.value),
            "point_size": float(point_size.value),
            "track_length": int(track_length.value),
            "track_stride": int(track_stride.value),
            "track_line_width": float(track_line_width.value),
            "fps": float(fps_slider.value),
            "camera_position": np.asarray(client.camera.position, dtype=np.float64),
            "camera_look_at": np.asarray(client.camera.look_at, dtype=np.float64),
            "camera_up": np.asarray(client.camera.up_direction, dtype=np.float64),
            "camera_fov": float(client.camera.fov),
        }
        camera_values = np.concatenate(
            [
                settings["camera_position"],
                settings["camera_look_at"],
                settings["camera_up"],
                [settings["camera_fov"]],
            ]
        )
        camera_hash = hashlib.sha256(camera_values.tobytes()).hexdigest()[:12]
        suffix = (
            f"tracks{int(settings['show_tracks'])}-"
            f"src0_fg_only_tracks{int(settings['src0_fg_only_tracks'])}-"
            f"ps{settings['point_size']:g}-"
            f"len{settings['track_length']}-stride{settings['track_stride']}-"
            f"width{settings['track_line_width']:g}-fps{settings['fps']:g}-"
            f"cam{camera_hash}-fov{np.rad2deg(settings['camera_fov']):g}"
        )
        if src0_confidence is not None:
            suffix = f"{suffix}-confp{settings['confidence_percentile']:g}"
        suffix = f"{suffix}-rdf{int(settings['show_rdf'])}"
        settings["output_base"] = output_path.with_name(f"{output_path.stem}-{suffix}")
        return settings

    def render_export_frame(
        timeline_idx: int,
        settings: dict,
    ):
        """
        Render one timeline frame using a frozen export-settings snapshot.
        """
        render(
            timeline_idx,
            settings["point_size"],
            settings["show_tracks"],
            settings["track_length"],
            settings["track_stride"],
            settings["track_line_width"],
            settings["src0_fg_only_tracks"],
            settings["confidence_percentile"],
            settings["show_rdf"],
        )

    @save_screenshot_button.on_click
    def save_screenshot(
        event,
    ):
        """
        Save the current scene view as a PNG named by timestamp, scene, and encoded frame number.
        """
        assert event.client is not None, "save_screenshot must be clicked from a connected Viser client."
        if any(button.disabled for button in export_buttons):
            return
        was_playing = bool(play_box.value)
        play_box.value = False
        for button in export_buttons:
            button.disabled = True
        try:
            render_live()
            screenshot_stem = output_path.stem.removesuffix("-srcw")
            screenshot_path = output_path.with_name(f"{screenshot_stem}-frame{int(frame_list[int(frame_slider.value)]):03d}.png")
            server.flush()
            image = event.client.get_render(height=event.client.camera.image_height, width=event.client.camera.image_width, transport_format="png")
            Image.fromarray(image).save(screenshot_path)
            print(f"[viser] Saved screenshot: {screenshot_path}")
        except Exception as error:
            print(f"[viser] Failed to save screenshot: {error}")
        finally:
            play_box.value = was_playing
            for button in export_buttons:
                button.disabled = False

    @save_mp4_button.on_click
    def save_mp4(
        event,
    ):
        """
        Save every timeline frame as an H.264 MP4 using the current camera, display settings, and FPS.
        """
        assert event.client is not None, "save_mp4 must be clicked from a connected Viser client."
        if any(button.disabled for button in export_buttons):
            return
        settings = get_export_settings(event.client)
        output_target = settings["output_base"].with_name(f"{settings['output_base'].name}.mp4")
        if output_target.exists():
            print(f"[viser] MP4 already exists: {output_target}")
            return
        current_frame = int(frame_slider.value)
        render_height = int(event.client.camera.image_height)
        render_width = int(event.client.camera.image_width)
        was_playing = bool(play_box.value)
        play_box.value = False
        for button in export_buttons:
            button.disabled = True
        try:
            with closing(imageio.get_writer(str(output_target), fps=settings["fps"], codec="libx264", macro_block_size=None)) as writer:
                for timeline_idx in range(len(frame_list)):
                    render_export_frame(timeline_idx, settings)
                    with event.client.atomic():
                        event.client.camera.position = tuple(settings["camera_position"])
                        event.client.camera.look_at = tuple(settings["camera_look_at"])
                        event.client.camera.up_direction = tuple(settings["camera_up"])
                        event.client.camera.fov = settings["camera_fov"]
                    server.flush()
                    event.client.flush()
                    image = event.client.get_render(height=render_height, width=render_width, transport_format="png")
                    pad_bottom, pad_right = image.shape[0] % 2, image.shape[1] % 2
                    if pad_bottom or pad_right:
                        image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    writer.append_data(image)
            print(f"[viser] Saved MP4: {output_target}")
        except Exception as error:
            output_target.unlink(missing_ok=True)
            print(f"[viser] Failed to save MP4: {error}")
        finally:
            frame_slider.value = current_frame
            render_live()
            play_box.value = was_playing
            for button in export_buttons:
                button.disabled = False

    @save_visualization_button.on_click
    def save_visualization(
        event,
    ):
        """
        Serialize all frames with the current online visualization parameters and triggering
        client's camera.
        """
        assert event.client is not None, "save_visualization must be clicked from a connected Viser client."
        if any(button.disabled for button in export_buttons):
            return
        settings = get_export_settings(event.client)
        saved_output_path = settings["output_base"].with_name(f"{settings['output_base'].name}.viser")
        if saved_output_path.exists():
            print(f"[viser] Recording already exists: {saved_output_path}")
            return

        current_frame = int(frame_slider.value)
        was_playing = bool(play_box.value)
        play_box.value = False
        for button in export_buttons:
            button.disabled = True
        try:
            saved_serializer = server.get_scene_serializer()
            camera_message_types = {"SetCameraPositionMessage", "SetCameraLookAtMessage", "SetCameraUpDirectionMessage", "SetCameraFovMessage"}
            saved_serializer._messages = [(time, message) for time, message in saved_serializer._messages if message["type"] not in camera_message_types]
            saved_serializer._insert_message(_messages.SetCameraPositionMessage(tuple(settings["camera_position"]), initial=True))
            saved_serializer._insert_message(_messages.SetCameraLookAtMessage(tuple(settings["camera_look_at"]), initial=True))
            saved_serializer._insert_message(_messages.SetCameraUpDirectionMessage(tuple(settings["camera_up"]), initial=True))
            saved_serializer._insert_message(_messages.SetCameraFovMessage(settings["camera_fov"], initial=True))
            for timeline_idx in range(len(frame_list)):
                render_export_frame(timeline_idx, settings)
                saved_serializer.insert_sleep(1.0 / settings["fps"])
            saved_output_path.write_bytes(saved_serializer.serialize())
            print(f"[viser] Saved online visualization: {saved_output_path}")
        except Exception as error:
            print(f"[viser] Failed to save online visualization: {error}")
        finally:
            frame_slider.value = current_frame
            render_live()
            play_box.value = was_playing
            for button in export_buttons:
                button.disabled = False

    def playback_loop():
        """
        Advance the live timeline while playback is enabled.
        """
        while True:
            time.sleep(1.0 / max(float(fps_slider.value), 1.0))
            if play_box.value:
                frame_slider.value = (int(frame_slider.value) + 1) % len(frame_list)

    threading.Thread(target=playback_loop, daemon=True).start()
    print(f"[viser] Live server ({output_path.stem}): http://{args.host}:{actual_port}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


def run_input(
    args,
):
    """
    Run checkpoint-driven dense inference, normalize geometry, and create the viser outputs.
    """
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_args = read_checkpoint_args(checkpoint_path)
    resolve_runtime_args(args, checkpoint_args)
    input_path = resolve_input_path(args.input_path)
    requested_chunk_num_frames = args.infer_chunk_num_frames
    frames = load_input_frames(input_path, requested_chunk_num_frames, allow_short=args.auto_adapt_frame_count)
    if len(frames) < requested_chunk_num_frames:
        available_num_frames = len(frames)
        adapted_num_frames = 1 + 4 * ((available_num_frames - 1) // 4)
        if adapted_num_frames <= 0:
            raise ValueError(f"Input '{input_path}' has no readable frames.")
        frames = frames[:adapted_num_frames]
        args.infer_chunk_num_frames = adapted_num_frames
        args.infer_total_num_frames = min(args.infer_total_num_frames, adapted_num_frames)
        args.infer_visualize_frame_list = [frame_idx for frame_idx in args.infer_visualize_frame_list if frame_idx < args.infer_total_num_frames]
        if not args.infer_visualize_frame_list:
            raise ValueError("No requested visualization frames remain after adapting to the available input length.")
        print(f"[WARNING] Input has {available_num_frames} readable frames below requested chunk {requested_chunk_num_frames}; using largest 4N+1 prefix {adapted_num_frames} with total={args.infer_total_num_frames}.")
    patch_frames, video_tchw, encoder_tchw = prepare_frames(frames, proc_h=args.proc_h, proc_w=args.proc_w, img_h=args.img_h, img_w=args.img_w, center_crop=args.center_crop)
    query_points, point_colors = build_dense_queries(args.proc_h, args.proc_w, args.dense_grid_stride, patch_frames[0])
    print(
        f"[dense] input={input_path} | chunk={args.infer_chunk_num_frames} | "
        f"total={args.infer_total_num_frames} | visualize={args.infer_visualize_frame_list} | "
        f"proc={args.proc_h}x{args.proc_w} | img={args.img_h}x{args.img_w} | "
        f"queries={query_points.shape[0]}"
    )

    setRandomness(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(checkpoint_path), device, args)
    infer_frame_list = args.infer_visualize_frame_list if args.infer_visualize_frame_list[0] == 0 else [0] + args.infer_visualize_frame_list
    tracks, src0_confidence = infer_dense_tracks(model, video_tchw, encoder_tchw, query_points, infer_frame_list, args, input_path.stem if input_path.is_file() else input_path.name)
    tracks, valid, mean_distance, scale_factor = normalize_tracks(tracks, args.distance_scale)
    if args.infer_visualize_frame_list[0] != 0:
        tracks = tracks[1:]
        valid = valid[1:]
    print(f"[dense] display frame0_mean_distance={mean_distance:.6f} | display scale_factor={scale_factor:.6f}")
    scene_name = input_path.stem if input_path.is_file() else input_path.name
    src0_foreground_mask = None
    if args.foreground_mask_root:
        src0_foreground_mask = load_foreground_mask(args.foreground_mask_root, scene_name, args.proc_h, args.proc_w, args.center_crop)
    output_path = make_output_path(input_path, args.output_dir)
    save_and_serve_viser(tracks, valid, point_colors, patch_frames, args.infer_visualize_frame_list, output_path, args, src0_foreground_mask, src0_confidence=src0_confidence)


def run_input_worker(
    args,
    input_paths: list[str],
    gpu: str,
):
    """Run the assigned inputs sequentially on one GPU, with fresh per-input settings."""
    os.setpgrp()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for input_path in input_paths:
        input_args = copy.deepcopy(args)
        input_args.input_path = input_path
        if input_args.foreground_mask_root:
            scene_name = Path(input_path).stem if Path(input_path).is_file() else Path(input_path).name
            mask_path = Path(input_args.foreground_mask_root).expanduser() / scene_name / "00000.png"
            if not mask_path.is_file():
                input_args.foreground_mask_root = None
        print(f"[viser] GPU={gpu or 'CPU'} INPUT={input_path} SERVE={input_args.serve} PORT={input_args.port}", flush=True)
        run_input(input_args)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_inputs(
    args,
):
    """Serve inputs that fit on the worker GPUs; otherwise export recordings offline."""
    gpus = get_visible_cuda_devices()
    num_workers = min(args.num_workers or len(gpus) or 1, len(gpus) or 1, len(args.input_paths))
    input_paths = [str(resolve_input_path(path)) for path in args.input_paths]
    serve = args.serve and len(input_paths) <= min(len(gpus), num_workers)
    if serve:
        print(f"[viser] {len(input_paths)} inputs <= {num_workers} GPUs: requesting ports {args.port}-{args.port + num_workers - 1}; occupied ports are skipped. Saving exports keeps servers running; Ctrl+C stops all servers.", flush=True)
    else:
        print(f"[viser] {len(input_paths)} inputs, {min(len(gpus), num_workers)} worker GPUs: offline batch export of .viser recordings.", flush=True)

    def stop_workers(
        signum,
        frame,
    ):
        raise SystemExit(128 + signum)

    previous_handlers = {sig: signal.signal(sig, stop_workers) for sig in (signal.SIGINT, signal.SIGTERM)}
    original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    processes = []
    try:
        context = mp.get_context("spawn")
        for worker_idx in range(num_workers):
            worker_args = copy.deepcopy(args)
            worker_args.port += worker_idx
            worker_args.serve = serve
            process = context.Process(target=run_input_worker, args=(worker_args, input_paths[worker_idx::num_workers], gpus[worker_idx] if gpus else ""))
            # Set visibility before the spawned worker imports CUDA libraries.
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus[worker_idx] if gpus else ""
            process.start()
            processes.append(process)
        pending = {process.sentinel: process for process in processes}
        while pending:
            for sentinel in wait(pending):
                process = pending.pop(sentinel)
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f"Viser worker {process.pid} exited with code {process.exitcode}.")
    finally:
        if original_cuda_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
        for process in processes:
            try:
                # Include renderer/FFmpeg children in each worker's process group.
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main():
    args = parse_args()
    if args.input_paths:
        run_inputs(args)
    else:
        run_input(args)


if __name__ == "__main__":
    main()
