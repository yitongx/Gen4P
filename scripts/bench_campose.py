#!/usr/bin/env python3
"""
Run Gen4P camera-pose inference and evaluation on Sintel, TUM-dynamics, and ScanNet v2.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.inference import load_model, prepareDecodeContext, str2bool, validate_image_aspect_ratios


SINTEL_TAG_FLOAT = 202021.25
SINTEL_EVAL_SCENES = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2", "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]
TUM_EVAL_SCENES = [
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_sitting_rpy",
    "rgbd_dataset_freiburg3_sitting_static",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_walking_halfsphere",
    "rgbd_dataset_freiburg3_walking_rpy",
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_walking_xyz",
]
SCANNET_EVAL_SCENES = [f"scene{scene_idx:04d}_00" for scene_idx in range(707, 807)]
SCANNET_INVALID_GT_FRAMES = {
    "scene0711_00": [201, 204, 207],
    "scene0737_00": list(range(171, 208, 3)),
    "scene0750_00": [213, 222],
    "scene0761_00": list(range(105, 130, 3)),
    "scene0791_00": [252, 255, 258],
    "scene0796_00": [123, 177],
}
PLACEHOLDER_DIR_NAME = ".bench_campose_placeholder"
WORKER_ARGS = None
WORKER_DEVICE = None
WORKER_MODEL = None


def parse_args():
    parser = argparse.ArgumentParser(description='Gen4P camera-pose benchmark')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dvd_checkpoint', type=str, default=None, help='Override ckpt/dvd_1.1.safetensors')
    parser.add_argument('--dataset_configs', nargs='+', required=True, help='Entries: dataset_id,data_root,proc_h,proc_w,img_h,img_w,output_dir')
    parser.add_argument('--target_scene_list', nargs='*', default=['*'], help='Scene stem list or `*` for all scenes')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of scene workers')
    parser.add_argument('--infer_total_num_frames', type=int, default=-1, help='Frames to evaluate; -1 uses all remaining frames from start_frame in the prepared sequence')
    parser.add_argument('--infer_chunk_num_frames', type=int, default=49, help='Model temporal window used for each camera-pose chunk; defaults to the checkpoint window')
    parser.add_argument('--train_num_frames', type=int, default=2, help='Optional override for checkpoint training num_frames used by time-embedding resize')
    parser.add_argument('--start_frame', type=int, default=0, help='Starting frame inside the source sequence')
    parser.add_argument('--center_crop', type=str2bool, default=False, help=('Center-crop raw frames to proc aspect ratio before resize; false keeps full-frame resize'))
    parser.add_argument('--amp_infer', type=str2bool, default=False, help='Enable AMP autocast during inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic kernels for reproducibility (may reduce speed)')
    parser.add_argument('--max_query_points_per_batch', type=int, default=2048, help='Decoder query chunk size')
    parser.add_argument('--infer_chunks_overlap_size', type=int, default=34)
    parser.add_argument('--adjacent_chunk_length', type=int, default=3)
    parser.add_argument('--adjacent_sliding_stride', type=int, default=1)
    parser.add_argument('--chunk_align_loss', choices=['soft_l1', 'linear'], default='soft_l1')
    parser.add_argument('--align_lambda_r', type=float, default=1.0)
    parser.add_argument('--align_lambda_t', type=float, default=1.0)
    parser.add_argument('--extrinsic_grid_interval', type=int, default=8)
    parser.add_argument('--extrinsic_conf_threshold', type=float, default=0.5, help='Top-confidence fraction retained independently by each camera-geometry fit.')
    parser.add_argument('--save_inputs', type=str2bool, default=True)
    parser.add_argument('--fps', type=float, default=15.0)

    args = parser.parse_args()
    if args.infer_total_num_frames == 0 or args.infer_total_num_frames < -1:
        parser.error("--infer_total_num_frames must be positive or -1.")
    if args.start_frame < 0:
        parser.error('--start_frame must be >= 0.')
    if args.infer_chunk_num_frames is not None and (args.infer_chunk_num_frames <= 1 or (args.infer_chunk_num_frames - 1) % 4):
        parser.error("--infer_chunk_num_frames must be at least 5 and satisfy 4N+1.")
    if args.infer_chunks_overlap_size <= 0 or (args.infer_chunk_num_frames is not None and args.infer_chunks_overlap_size >= args.infer_chunk_num_frames):
        parser.error("--infer_chunks_overlap_size must be positive and smaller than --infer_chunk_num_frames.")
    if args.extrinsic_grid_interval <= 0:
        parser.error('--extrinsic_grid_interval must be positive.')
    if args.max_query_points_per_batch <= 0:
        parser.error('--max_query_points_per_batch must be positive.')
    if not 0 < args.extrinsic_conf_threshold <= 1:
        parser.error("--extrinsic_conf_threshold must be in (0, 1] and denotes the retained top-confidence fraction.")
    if args.adjacent_chunk_length < 2 or args.adjacent_sliding_stride <= 0 or args.adjacent_sliding_stride >= args.adjacent_chunk_length:
        parser.error("Adjacent camera windows require length >= 2 and 1 <= stride < length.")
    if args.align_lambda_r < 0 or args.align_lambda_t < 0 or args.align_lambda_r + args.align_lambda_t <= 0:
        parser.error("Chunk alignment weights must be non-negative and not both zero.")
    if args.num_workers is not None and args.num_workers <= 0:
        parser.error("--num_workers must be positive.")
    if args.train_num_frames is not None and args.train_num_frames <= 0:
        parser.error('--train_num_frames must be positive when provided.')
    if args.infer_total_num_frames == 1:
        parser.error("Camera-pose evaluation requires at least two frames.")
    if not np.isfinite([args.align_lambda_r, args.align_lambda_t]).all():
        parser.error("Chunk alignment weights must be finite.")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive.")
    args.parser_defaults = {k: parser.get_default(k) for k in vars(args).keys()}
    return args


def set_randomness(
    seed: int,
    deterministic: bool,
) -> None:
    """
    Seed camera inference without enabling cuDNN autotuning, matching the original benchmark.
    """
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def parse_dataset_configs(
    raw_configs: List[str],
) -> List[dict]:
    """
    Parse launcher dataset entries and reject ambiguous comma-containing layouts.
    """
    configs = []
    for raw_config in raw_configs:
        fields = raw_config.split(",", 6)
        if len(fields) != 7:
            raise ValueError(f"Invalid dataset config {raw_config!r}; expected dataset_id,data_root,proc_h,proc_w,img_h,img_w,output_dir.")
        dataset_id, data_root, proc_h, proc_w, img_h, img_w, output_dir = fields
        dataset_id = dataset_id.upper()
        if dataset_id not in {"SINTEL", "TUM", "SCANNET"}:
            raise ValueError(f"Unsupported camera-pose dataset: {dataset_id}.")
        config = {
            "dataset_id": dataset_id,
            "data_root": str(Path(data_root).expanduser()),
            "proc_h": int(proc_h),
            "proc_w": int(proc_w),
            "img_h": int(img_h),
            "img_w": int(img_w),
            "output_dir": str(Path(output_dir).expanduser()),
        }
        validate_image_aspect_ratios(config["img_h"], config["img_w"], config["proc_h"], config["proc_w"])
        configs.append(config)
    return configs


def select_scene_names(
    dataset_id: str,
    data_root: Path,
    target_scene_list: Optional[List[str]],
) -> List[str]:
    """
    Resolve the official MonST3R scene list and verify that every requested scene exists.
    """
    if dataset_id == "SINTEL":
        final_root = data_root / "training" / "final" if (data_root / "training" / "final").is_dir() else data_root
        official = SINTEL_EVAL_SCENES
        exists = {name for name in official if (final_root / name).is_dir()}
    elif dataset_id == "TUM":
        official = TUM_EVAL_SCENES
        exists = set()
        for name in official:
            if (data_root / f"{name}_0.npz").is_file() or (data_root / f"{name}.npz").is_file() or (data_root / name / "rgb_90").is_dir():
                exists.add(name)
    else:
        official = SCANNET_EVAL_SCENES
        exists = {name for name in official if (data_root / name / "color_90").is_dir() or (data_root / name / "color").is_dir()}

    use_all = not target_scene_list or "*" in target_scene_list
    selected = official if use_all else [name for name in official if name in set(target_scene_list)]
    if dataset_id == "SCANNET":
        skipped = [name for name in selected if name in SCANNET_INVALID_GT_FRAMES]
        selected = [name for name in selected if name not in SCANNET_INVALID_GT_FRAMES]
        if skipped:
            print(f"[Skip] SCANNET scenes with known invalid GT cameras: {skipped}")
    missing = [name for name in selected if name not in exists]
    if missing:
        raise FileNotFoundError(f"{dataset_id} is missing {len(missing)} requested scenes under {data_root}: {missing[:10]}.")
    if not selected:
        raise ValueError(f"No {dataset_id} scenes selected.")
    return selected


def read_sintel_cam(
    path: Path,
) -> np.ndarray:
    """
    Read one Sintel left-multiply world-to-camera extrinsic.
    """
    with path.open("rb") as file:
        tag = np.fromfile(file, dtype=np.float32, count=1)
        if tag.size != 1 or float(tag[0]) != SINTEL_TAG_FLOAT:
            raise ValueError(f"Invalid Sintel camera tag in {path}.")
        file.seek(9 * np.dtype(np.float64).itemsize, 1)  # Skip the 3x3 intrinsic matrix.
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3] = np.fromfile(file, dtype=np.float64, count=12).reshape(3, 4)
    return w2c


def decode_jpeg_rgb(
    frame_bytes,
) -> np.ndarray:
    """
    Decode one fixed-width WorldTrack NPZ JPEG payload as RGB.
    """
    encoded = np.frombuffer(np.asarray(frame_bytes).tobytes(), dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode WorldTrack JPEG bytes.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_rgb_paths(
    paths: List[Path],
) -> np.ndarray:
    """
    Load an ordered image list as RGB uint8 frames.
    """
    frames = []
    for path in paths:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {path}.")
        frames.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def load_sintel_scene(
    data_root: Path,
    scene_name: str,
) -> dict:
    """
    Load all frames and camera metadata for one of MonST3R's 14 Sintel scenes.
    """
    training_root = data_root / "training" if (data_root / "training").is_dir() else data_root.parent
    final_root = training_root / "final"
    if data_root.name == "final":
        final_root = data_root
        training_root = data_root.parent
    rgb_paths = sorted((final_root / scene_name).glob("*.png"))
    cam_by_stem = {path.stem: path for path in (training_root / "camdata_left" / scene_name).glob("*.cam")}
    if not rgb_paths:
        raise RuntimeError(f"No Sintel RGB frames found for {scene_name}.")
    w2cs = []
    for rgb_path in rgb_paths:
        if rgb_path.stem not in cam_by_stem:
            raise RuntimeError(f"Missing Sintel camera for {rgb_path}.")
        w2cs.append(read_sintel_cam(cam_by_stem[rgb_path.stem]))
    c2w = np.linalg.inv(np.stack(w2cs, axis=0))
    c2w[:, :3, :3] = Rotation.from_matrix(c2w[:, :3, :3]).as_matrix()
    return {
        "scene_name": scene_name,
        "rgb": read_rgb_paths(rgb_paths),
        "frame_ids": [path.stem for path in rgb_paths],
        "c2w": c2w,
    }


def load_tum_scene(
    data_root: Path,
    scene_name: str,
) -> dict:
    """
    Load either a WorldTrack NPZ pack or MonST3R's prepared raw TUM directory.
    """
    npz_candidates = [data_root / f"{scene_name}_0.npz", data_root / f"{scene_name}.npz"]
    npz_path = next((path for path in npz_candidates if path.is_file()), None)
    if npz_path is not None:
        with np.load(npz_path, allow_pickle=True) as pack:
            jpeg_bytes = np.asarray(pack["images_jpeg_bytes"])
            rgb = np.stack([decode_jpeg_rgb(item) for item in jpeg_bytes], axis=0)
            w2c = np.asarray(pack["extrinsics_w2c"], dtype=np.float64)
        return {
            "scene_name": scene_name,
            "rgb": rgb,
            "frame_ids": [f"prepared_{idx:04d}" for idx in range(rgb.shape[0])],
            "c2w": np.linalg.inv(w2c),
        }

    scene_root = data_root / scene_name
    rgb_paths = sorted((scene_root / "rgb_90").glob("*.png"))
    gt_rows = np.loadtxt(scene_root / "groundtruth_90.txt", dtype=np.float64)
    gt_rows = np.atleast_2d(gt_rows)
    if len(rgb_paths) != gt_rows.shape[0]:
        raise RuntimeError(f"TUM RGB/pose count mismatch for {scene_name}: {len(rgb_paths)} vs {gt_rows.shape[0]}.")
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], gt_rows.shape[0], axis=0)
    c2w[:, :3, :3] = Rotation.from_quat(gt_rows[:, 4:8]).as_matrix()
    c2w[:, :3, 3] = gt_rows[:, 1:4]
    return {
        "scene_name": scene_name,
        "rgb": read_rgb_paths(rgb_paths),
        "frame_ids": [path.stem for path in rgb_paths],
        "c2w": c2w,
    }


def load_scannet_scene(
    data_root: Path,
    scene_name: str,
) -> dict:
    """
    Load prepared ScanNet v2 data or sample every third frame from the first 270 raw frames.
    """
    scene_root = data_root / scene_name
    prepared_pose_path = scene_root / "pose_90.txt"
    if prepared_pose_path.is_file() and (scene_root / "color_90").is_dir():
        rgb_paths = sorted((scene_root / "color_90").glob("*.jpg"))
        pose_rows = np.atleast_2d(np.loadtxt(prepared_pose_path, dtype=np.float64))
        c2w = pose_rows.reshape(-1, 4, 4) if pose_rows.ndim == 2 and pose_rows.shape[1] == 16 else np.empty((0, 4, 4), dtype=np.float64)
    else:
        color_by_id = {int(path.stem): path for path in (scene_root / "color").glob("*.jpg") if path.stem.isdigit()}
        frame_indices = sorted(color_by_id)[:270:3]
        rgb_paths = [color_by_id[frame_idx] for frame_idx in frame_indices]
        pose_paths = [scene_root / "pose" / f"{frame_idx}.txt" for frame_idx in frame_indices]
        missing_paths = [path for path in pose_paths if not path.is_file()]
        if len(frame_indices) != 90 or missing_paths:
            raise RuntimeError(f"Raw ScanNet data is incomplete for {scene_name}: selected_rgb={len(frame_indices)} missing_pose={len(missing_paths)}.")
        c2w = np.stack([np.loadtxt(path, dtype=np.float64) for path in pose_paths], axis=0)
    if len(rgb_paths) != 90 or c2w.shape != (90, 4, 4):
        raise RuntimeError(f"ScanNet data format mismatch for {scene_name}: RGB={len(rgb_paths)} C2W={c2w.shape}.")
    return {
        "scene_name": scene_name,
        "rgb": read_rgb_paths(rgb_paths),
        "frame_ids": [path.stem for path in rgb_paths],
        "c2w": c2w,
    }


def load_scene(
    config: dict,
    scene_name: str,
    requested_frames: int,
    start_frame: int,
) -> dict:
    """
    Load and select a scene interval after dataset-specific MonST3R sampling has been applied.
    """
    data_root = Path(config["data_root"])
    if config["dataset_id"] == "SINTEL":
        scene = load_sintel_scene(data_root, scene_name)
    elif config["dataset_id"] == "TUM":
        scene = load_tum_scene(data_root, scene_name)
    else:
        scene = load_scannet_scene(data_root, scene_name)
    source_count = int(scene["rgb"].shape[0])
    if config["dataset_id"] in {"TUM", "SCANNET"} and source_count != 90:
        raise RuntimeError(f"{config['dataset_id']} requires 90 prepared frames, got {source_count} for {scene_name}.")
    if start_frame >= source_count:
        raise RuntimeError(f"Start frame {start_frame} is outside the {source_count}-frame sequence for {scene_name}.")
    end_frame = source_count if requested_frames == -1 else min(source_count, start_frame + requested_frames)
    eval_count = end_frame - start_frame
    if eval_count < 2:
        raise ValueError(f"Camera-pose evaluation requires at least two frames for {scene_name}.")
    for key in ("rgb", "frame_ids", "c2w"):
        scene[key] = scene[key][start_frame:end_frame]
    scene["source_num_frames"] = source_count
    scene["num_frames"] = eval_count
    return scene


def preprocess_scene_images(
    scene: dict,
    config: dict,
    center_crop: bool,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Resize one shared full-frame or center-cropped RGB source independently to decoder and encoder resolutions.
    """
    source_h, source_w = scene["rgb"].shape[1:3]
    crop_h, crop_w = source_h, source_w
    if center_crop:
        ratio_gcd = int(np.gcd(config["proc_h"], config["proc_w"]))
        unit_h, unit_w = config["proc_h"] // ratio_gcd, config["proc_w"] // ratio_gcd
        crop_scale = min(source_h // unit_h, source_w // unit_w)
        crop_h, crop_w = unit_h * crop_scale, unit_w * crop_scale
    crop_top, crop_left = (source_h - crop_h) // 2, (source_w - crop_w) // 2
    source_rgb = scene["rgb"][:, crop_top : crop_top + crop_h, crop_left : crop_left + crop_w]
    scene["crop_box"] = (crop_top, crop_left, crop_h, crop_w)
    proc_frames = []
    encode_frames = []
    for frame in source_rgb:
        proc_frames.append(cv2.resize(frame, (config["proc_w"], config["proc_h"]), interpolation=cv2.INTER_LINEAR))
        encode_frames.append(cv2.resize(frame, (config["img_w"], config["img_h"]), interpolation=cv2.INTER_LINEAR))
    proc_rgb = np.stack(proc_frames, axis=0).astype(np.uint8)
    encode_rgb = np.stack(encode_frames, axis=0).astype(np.uint8)
    video = torch.from_numpy(proc_rgb).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    encode_video = torch.from_numpy(encode_rgb).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    return video, encode_video, proc_rgb


def build_sliding_pose_ranges(
    total_frames: int,
    chunk_frames: int,
    overlap_frames: int,
) -> List[Tuple[int, int]]:
    """
    Build full-size sliding windows and shift the final window backward to end exactly at the sequence tail.
    """
    if total_frames <= chunk_frames:
        return [(0, total_frames)]
    stride = chunk_frames - overlap_frames
    starts = [0]
    while starts[-1] + chunk_frames < total_frames:
        next_start = starts[-1] + stride
        if next_start + chunk_frames > total_frames:
            next_start = total_frames - chunk_frames
        if next_start <= starts[-1]:
            break
        starts.append(next_start)
    return [(start, start + chunk_frames) for start in starts]


def build_extrinsic_grid(
    proc_h: int,
    proc_w: int,
    interval: int,
) -> np.ndarray:
    """
    Build a normalized pixel-center grid over the full processed frame.
    """
    xs = np.arange(0, proc_w, interval, dtype=np.float32) + 0.5
    ys = np.arange(0, proc_h, interval, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([grid_x / proc_w, grid_y / proc_h], axis=-1).reshape(-1, 2).astype(np.float32)


def estimate_rigid_transform(
    source: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve target = R @ source + t with a left-multiply Kabsch fit.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] = -vt[-1]
        rotation = vt.T @ u.T
    translation = target_mean - np.einsum("ij,j->i", rotation, source_mean)
    return rotation, translation


def initialize_camera_poses(
    num_frames: int,
    edges: List[Tuple[int, int, np.ndarray]],
) -> np.ndarray:
    """
    Initialize frame-0-referenced C2W poses by traversing relative camera edges.
    """
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], num_frames, axis=0)
    known = {0}
    while len(known) < num_frames:
        previous_count = len(known)
        for source_idx, target_idx, measurement in edges:
            if source_idx in known and target_idx not in known:
                poses[target_idx] = poses[source_idx] @ measurement
                known.add(target_idx)
            elif target_idx in known and source_idx not in known:
                poses[source_idx] = poses[target_idx] @ np.linalg.inv(measurement)
                known.add(source_idx)
        if len(known) == previous_count:
            raise RuntimeError(f"Camera pose graph is disconnected; initialized {len(known)}/{num_frames} frames.")
    return poses


def select_top_confidence(
    valid: np.ndarray,
    confidence: np.ndarray,
    keep_ratio: float,
) -> np.ndarray:
    """
    Select the configured top-confidence fraction from finite candidate points.
    """
    confidence = np.asarray(confidence)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(confidence)
    candidate_indices = np.flatnonzero(valid)
    if candidate_indices.size == 0:
        return valid
    keep_count = max(1, int(np.ceil(candidate_indices.size * keep_ratio)))
    if keep_count == candidate_indices.size:
        return valid
    selected = np.zeros_like(valid)
    top_local_indices = np.argpartition(confidence[candidate_indices], candidate_indices.size - keep_count)[-keep_count:]
    selected[candidate_indices[top_local_indices]] = True
    return selected


def run_pose_inference(
    model,
    video_b,
    encoder_features,
    config: dict,
    args,
    real_count: int,
    chunk_name: str,
) -> np.ndarray:
    """
    Recover one chunk-local C2W trajectory from adjacent query windows and a camera pose graph.
    """
    device = video_b.device
    query_points = build_extrinsic_grid(config["proc_h"], config["proc_w"], args.extrinsic_grid_interval)
    query_tensor = torch.from_numpy(query_points).to(device=device, dtype=video_b.dtype).unsqueeze(0)
    num_queries = int(query_tensor.shape[1])
    amp_enabled = bool(args.amp_infer and device.type == "cuda")

    def decode_grid(
        t_src: int,
        t_cam: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode the fixed spatial grid for one D4RT time tuple.
        """
        decoded_points = []
        decoded_scores = []
        for point_start in range(0, num_queries, args.max_query_points_per_batch):
            point_end = min(point_start + args.max_query_points_per_batch, num_queries)
            point_count = point_end - point_start
            time_source = torch.full((1, point_count), t_src, device=device, dtype=torch.long)
            time_camera = torch.full((1, point_count), t_cam, device=device, dtype=torch.long)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                outputs = model.decode(encoder_features, video_b, query_tensor[:, point_start:point_end], time_source, time_source, time_camera)
            points = outputs["pos_3d"].squeeze(0)
            scores = outputs["confidence"].squeeze(0).reshape(-1)
            decoded_points.append(points.detach().cpu().to(torch.float32).numpy())
            decoded_scores.append(scores.detach().cpu().to(torch.float32).numpy())
        return np.concatenate(decoded_points, axis=0), np.concatenate(decoded_scores, axis=0)

    def estimate_corresponding_edge(
        points_current: np.ndarray,
        confidence_current: np.ndarray,
        points_source: np.ndarray,
        confidence_source: np.ndarray,
        edge_name: str,
    ) -> np.ndarray:
        """
        Fit camera-current to camera-source from predictions with shared query identities.
        """
        scores_current = np.minimum(confidence_current, confidence_source)
        valid = np.isfinite(points_current).all(axis=-1)
        valid = valid & np.isfinite(points_source).all(axis=-1)
        valid = select_top_confidence(valid, scores_current, args.extrinsic_conf_threshold)
        if int(valid.sum()) < 6:
            raise RuntimeError(f"{chunk_name} {edge_name} has only {int(valid.sum())} valid pose correspondences.")
        rotation, translation = estimate_rigid_transform(points_current[valid], points_source[valid])
        measurement = np.eye(4, dtype=np.float64)
        measurement[:3, :3] = rotation
        measurement[:3, 3] = translation
        return measurement

    pose_edges = []
    ranges = build_sliding_pose_ranges(real_count, args.adjacent_chunk_length, args.adjacent_chunk_length - args.adjacent_sliding_stride)
    for start, end in tqdm(ranges, desc=f"{chunk_name} adjacent camera windows", leave=False):
        window_points = []
        window_confidence = []
        for local_t in range(start, end):
            points_current, confidence_current = decode_grid(start, local_t)
            window_points.append(points_current)
            window_confidence.append(confidence_current)
        for source_offset in range(end - start):
            for target_offset in range(source_offset + 1, end - start):
                source_idx = start + source_offset
                target_idx = start + target_offset
                measurement = estimate_corresponding_edge(window_points[target_offset], window_confidence[target_offset], window_points[source_offset], window_confidence[source_offset], f"edge {source_idx}->{target_idx} from query frame {start}")
                pose_edges.append((source_idx, target_idx, measurement))
    pred_c2w = initialize_camera_poses(real_count, pose_edges)
    pred_c2w = optimize_camera_pose_graph(pred_c2w, pose_edges, args, chunk_name).astype(np.float32)
    return pred_c2w


def project_rotation(
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Project a noisy 3x3 matrix onto SO(3).
    """
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    output = u @ vt
    if np.linalg.det(output) < 0:
        u[:, -1] = -u[:, -1]
        output = u @ vt
    return output


def identity_similarity() -> dict:
    return {"scale": 1.0, "rotation": np.eye(3, dtype=np.float64), "translation": np.zeros(3, dtype=np.float64)}


def estimate_chunk_similarity(
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
) -> dict:
    """
    Estimate one Sim(3) between overlapping chunk-local C2W trajectories using rotations and camera centers.
    """
    relative_rotations = np.einsum("tij,tjk->tik", target_c2w[:, :3, :3], np.transpose(source_c2w[:, :3, :3], (0, 2, 1)))
    rotation = project_rotation(relative_rotations.sum(axis=0))
    source_centers = source_c2w[:, :3, 3]
    target_centers = target_c2w[:, :3, 3]
    source_mean = source_centers.mean(axis=0)
    target_mean = target_centers.mean(axis=0)
    source_rotated = np.einsum("ij,tj->ti", rotation, source_centers - source_mean)
    target_centered = target_centers - target_mean
    denominator = float(np.sum(source_rotated * source_rotated))
    scale = 1.0 if denominator <= 1e-12 else float(np.sum(target_centered * source_rotated) / denominator)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    translation = target_mean - scale * np.einsum("ij,j->i", rotation, source_mean)
    return {"scale": scale, "rotation": rotation, "translation": translation}


def compose_similarities(
    left: dict,
    right: dict,
) -> dict:
    """
    Compose x_mid=right(x) followed by x_out=left(x_mid).
    """
    return {
        "scale": float(left["scale"] * right["scale"]),
        "rotation": project_rotation(left["rotation"] @ right["rotation"]),
        "translation": left["translation"] + left["scale"] * np.einsum("ij,j->i", left["rotation"], right["translation"]),
    }


def transform_c2w_similarity(
    c2w: np.ndarray,
    similarity: dict,
) -> np.ndarray:
    """
    Apply a world-coordinate Sim(3) to a C2W trajectory without scaling camera rotations.
    """
    output = np.repeat(np.eye(4, dtype=np.float64)[None], c2w.shape[0], axis=0)
    output[:, :3, :3] = np.einsum("ij,tjk->tik", similarity["rotation"], c2w[:, :3, :3])
    output[:, :3, 3] = similarity["scale"] * np.einsum("ij,tj->ti", similarity["rotation"], c2w[:, :3, 3]) + similarity["translation"]
    return output


def matrix_to_rotation_vector(
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Convert one projected rotation matrix into an axis-angle vector.
    """
    rotation_vector, _ = cv2.Rodrigues(project_rotation(rotation))
    return rotation_vector.reshape(3)


def rotation_vector_to_matrix(
    rotation_vector: np.ndarray,
) -> np.ndarray:
    """
    Convert one axis-angle vector into a rotation matrix.
    """
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64).reshape(3, 1))
    return rotation


def optimize_camera_pose_graph(
    initial_c2w: np.ndarray,
    edges: List[Tuple[int, int, np.ndarray]],
    args,
    chunk_name: str,
) -> np.ndarray:
    """
    Refine frame-0-referenced camera poses using all adjacent-window edges.
    """
    num_poses = len(initial_c2w)
    initial_values = []
    for pose in initial_c2w[1:]:
        initial_values.extend(matrix_to_rotation_vector(pose[:3, :3]).tolist())
        initial_values.extend(pose[:3, 3].tolist())
    initial_values = np.asarray(initial_values, dtype=np.float64)

    def unpack(
        values: np.ndarray,
    ) -> np.ndarray:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], num_poses, axis=0)
        for frame_idx in range(1, num_poses):
            offset = 6 * (frame_idx - 1)
            poses[frame_idx, :3, :3] = rotation_vector_to_matrix(values[offset : offset + 3])
            poses[frame_idx, :3, 3] = values[offset + 3 : offset + 6]
        return poses

    rotation_weight = float(np.sqrt(args.align_lambda_r))
    translation_weight = float(np.sqrt(args.align_lambda_t))

    def residual(
        values: np.ndarray,
    ) -> np.ndarray:
        poses = unpack(values)
        terms = []
        for source_idx, target_idx, measurement in edges:
            predicted = np.linalg.inv(poses[source_idx]) @ poses[target_idx]
            error = np.linalg.inv(measurement) @ predicted
            terms.extend((rotation_weight * matrix_to_rotation_vector(error[:3, :3])).tolist())
            terms.extend((translation_weight * error[:3, 3]).tolist())
        return np.asarray(terms, dtype=np.float64)

    jacobian_sparsity = lil_matrix((6 * len(edges), 6 * (num_poses - 1)), dtype=np.int8)
    for edge_idx, (source_idx, target_idx, _) in enumerate(edges):
        row = 6 * edge_idx
        if source_idx > 0:
            column = 6 * (source_idx - 1)
            jacobian_sparsity[row : row + 6, column : column + 6] = 1
        if target_idx > 0:
            column = 6 * (target_idx - 1)
            jacobian_sparsity[row : row + 6, column : column + 6] = 1
    result = least_squares(residual, initial_values, jac_sparsity=jacobian_sparsity.tocsr(), max_nfev=200, loss=args.chunk_align_loss)
    print(f"[Camera Graph] {chunk_name} edges={len(edges)} loss={args.chunk_align_loss} cost={float(result.cost):.6f} nfev={int(result.nfev)}")
    return unpack(result.x)


def fit_global_chunk_similarities(
    records: List[dict],
    initial: List[dict],
    args,
) -> List[dict]:
    """
    Jointly refine every non-reference chunk Sim(3) from overlapping camera poses.
    """
    if len(records) <= 1:
        return initial
    frame_maps = [{int(global_idx): local_idx for local_idx, global_idx in enumerate(record["global_indices"])} for record in records]
    pairs = []
    for left_idx in range(len(records)):
        for right_idx in range(left_idx + 1, len(records)):
            common = sorted(set(frame_maps[left_idx]).intersection(frame_maps[right_idx]))
            if common:
                pairs.append((left_idx, right_idx, common))
    if not pairs:
        return initial

    def pack(
        similarities: List[dict],
    ) -> np.ndarray:
        values = []
        for similarity in similarities[1:]:
            values.extend(matrix_to_rotation_vector(similarity["rotation"]).tolist())
            values.extend(np.asarray(similarity["translation"]).tolist())
            values.append(float(np.log(max(similarity["scale"], 1e-8))))
        return np.asarray(values, dtype=np.float64)

    def unpack(
        values: np.ndarray,
    ) -> List[dict]:
        similarities = [identity_similarity()]
        cursor = 0
        for _ in range(1, len(records)):
            rotation = rotation_vector_to_matrix(values[cursor : cursor + 3])
            cursor = cursor + 3
            translation = values[cursor : cursor + 3]
            cursor = cursor + 3
            scale = float(np.exp(values[cursor]))
            cursor = cursor + 1
            similarities.append({"scale": scale, "rotation": rotation, "translation": translation.copy()})
        return similarities

    rotation_weight = float(np.sqrt(args.align_lambda_r))
    translation_weight = float(np.sqrt(args.align_lambda_t))

    def residual(
        values: np.ndarray,
    ) -> np.ndarray:
        similarities = unpack(values)
        transformed = [transform_c2w_similarity(record["pose_c2w"], similarities[idx]) for idx, record in enumerate(records)]
        terms = []
        for left_idx, right_idx, common in pairs:
            for global_idx in common:
                left_pose = transformed[left_idx][frame_maps[left_idx][global_idx]]
                right_pose = transformed[right_idx][frame_maps[right_idx][global_idx]]
                rotation_error = project_rotation(left_pose[:3, :3].T @ right_pose[:3, :3])
                terms.extend((rotation_weight * matrix_to_rotation_vector(rotation_error)).tolist())
                terms.extend((translation_weight * (left_pose[:3, 3] - right_pose[:3, 3])).tolist())
        return np.asarray(terms, dtype=np.float64)

    result = least_squares(residual, pack(initial), max_nfev=200, loss=args.chunk_align_loss)
    print(f"[Chunk Align] global mode=cameras loss={args.chunk_align_loss} cost={float(result.cost):.6f} nfev={int(result.nfev)}")
    return unpack(result.x)


def merge_chunk_poses(
    records: List[dict],
    total_frames: int,
    args,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Initialize adjacent chunk Sim(3), globally refine them, and average every overlapping C2W prediction.
    """
    similarities = [identity_similarity()]
    for record_idx in range(1, len(records)):
        previous = records[record_idx - 1]
        current = records[record_idx]
        previous_map = {int(global_idx): local_idx for local_idx, global_idx in enumerate(previous["global_indices"])}
        current_map = {int(global_idx): local_idx for local_idx, global_idx in enumerate(current["global_indices"])}
        common = sorted(set(previous_map).intersection(current_map))
        if not common:
            raise RuntimeError(f"Chunks {record_idx - 1} and {record_idx} have no overlapping poses.")
        source = current["pose_c2w"][[current_map[idx] for idx in common]]
        target = previous["pose_c2w"][[previous_map[idx] for idx in common]]
        relative = estimate_chunk_similarity(source, target)
        similarities.append(compose_similarities(similarities[-1], relative))
        print(f"[Chunk Align] {record_idx}->{record_idx + 1} mode=cameras overlap={len(common)} scale={relative['scale']:.6f}")
    similarities = fit_global_chunk_similarities(records, similarities, args)

    rotation_sum = np.zeros((total_frames, 3, 3), dtype=np.float64)
    center_sum = np.zeros((total_frames, 3), dtype=np.float64)
    counts = np.zeros((total_frames,), dtype=np.float64)
    for record, similarity in zip(records, similarities):
        transformed = transform_c2w_similarity(record["pose_c2w"], similarity)
        for global_idx, pose in zip(record["global_indices"], transformed):
            rotation_sum[int(global_idx)] = rotation_sum[int(global_idx)] + pose[:3, :3]
            center_sum[int(global_idx)] = center_sum[int(global_idx)] + pose[:3, 3]
            counts[int(global_idx)] = counts[int(global_idx)] + 1.0
    if np.any(counts <= 0):
        raise RuntimeError("Chunk inference did not cover every evaluation frame.")
    merged = np.repeat(np.eye(4, dtype=np.float64)[None], total_frames, axis=0)
    for frame_idx in range(total_frames):
        merged[frame_idx, :3, :3] = project_rotation(rotation_sum[frame_idx])
        merged[frame_idx, :3, 3] = center_sum[frame_idx] / counts[frame_idx]
    return merged, similarities


def infer_scene_poses(
    model,
    video: torch.Tensor,
    encode_video: torch.Tensor,
    config: dict,
    args,
) -> Tuple[np.ndarray, List[dict], List[dict], List[Tuple[int, int]]]:
    """
    Run every temporal chunk and merge their locally normalized camera trajectories.
    """
    total_frames = int(video.shape[0])
    chunk_frames = args.infer_chunk_num_frames if args.infer_chunk_num_frames is not None else int(model.temporal_size)
    if chunk_frames <= 1 or (chunk_frames - 1) % 4:
        raise ValueError(f'Resolved model window {chunk_frames} must be at least 5 and satisfy 4N+1; set --infer_chunk_num_frames explicitly.')
    if not 0 < args.infer_chunks_overlap_size < chunk_frames:
        raise ValueError(f'--infer_chunks_overlap_size must be positive and smaller than the resolved model window {chunk_frames}.')
    ranges = build_sliding_pose_ranges(total_frames, chunk_frames, args.infer_chunks_overlap_size)
    print(f"[Chunk Plan] total={total_frames} chunk={chunk_frames} overlap={args.infer_chunks_overlap_size} ranges={ranges}")
    records = []
    for chunk_idx, (start, end) in enumerate(ranges):
        real_indices = np.arange(start, end, dtype=np.int64)
        encode_count = len(real_indices) + ((1 - len(real_indices)) % 4)
        if encode_count > chunk_frames:
            raise RuntimeError(f"Chunk [{start},{end}) needs padded length {encode_count}.")
        padded_indices = np.concatenate([real_indices, np.full(encode_count - len(real_indices), int(real_indices[-1]), dtype=np.int64)])
        index_tensor = torch.from_numpy(padded_indices).to(device=video.device, dtype=torch.long)

        decoder_dtype = None if args.amp_infer and video.device.type == "cuda" else torch.get_default_dtype()
        with torch.inference_mode():
            video_b, features = prepareDecodeContext(model, video[index_tensor], encode_video[index_tensor], (config["img_h"], config["img_w"]), args.amp_infer, decoder_dtype)
        pose_c2w = run_pose_inference(model, video_b, features, config, args, len(real_indices), f"chunk_{chunk_idx:03d}")
        records.append(
            {
                "name": f"chunk_{chunk_idx:03d}",
                "global_indices": real_indices,
                "pose_c2w": pose_c2w,
            }
        )
    merged, similarities = merge_chunk_poses(records, total_frames, args)
    return merged, records, similarities, ranges


def save_input_artifacts(
    path: Path,
    proc_rgb: np.ndarray,
    scene: dict,
    ranges: List[Tuple[int, int]],
    fps: float,
) -> None:
    """
    Save the exact processed frames and temporal plan used by the model.
    """
    path.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path / "input.mp4"), fps=float(fps), codec="libx264", macro_block_size=None)
    try:
        for frame_idx, frame in enumerate(proc_rgb):
            Image.fromarray(frame).save(path / f"frame_{frame_idx:05d}.png")
            writer.append_data(frame)
    finally:
        writer.close()
    write_json(
        path / "input_meta.json",
        {
            "frame_ids": scene["frame_ids"],
            "source_num_frames": scene["source_num_frames"],
            "num_eval_frames": scene["num_frames"],
            "crop_box": list(scene["crop_box"]),
            "chunk_ranges": [[start, end] for start, end in ranges],
        },
    )


def write_json(
    path: Path,
    payload: dict,
) -> None:
    """
    Write one human-readable benchmark JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def align_pose_trajectory_sim3(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    valid_gt_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Align one predicted C2W trajectory to its valid ground-truth cameras with evo's Sim(3) implementation.
    """
    from evo.core.trajectory import PoseTrajectory3D

    pred_c2w = np.asarray(pred_c2w, dtype=np.float64).copy()
    gt_c2w = np.asarray(gt_c2w, dtype=np.float64).copy()
    if pred_c2w.shape != gt_c2w.shape or pred_c2w.ndim != 3 or pred_c2w.shape[1:] != (4, 4):
        raise ValueError(f'Expected matching [T,4,4] C2W trajectories, got pred={pred_c2w.shape}, gt={gt_c2w.shape}.')
    valid_gt_mask = valid_camera_pose_mask(gt_c2w) if valid_gt_mask is None else np.asarray(valid_gt_mask, dtype=bool)
    if valid_gt_mask.shape != (pred_c2w.shape[0],):
        raise ValueError(f'Expected valid_gt_mask shape {(pred_c2w.shape[0],)}, got {valid_gt_mask.shape}.')
    if np.count_nonzero(valid_gt_mask) < 2:
        raise ValueError('Sim(3) trajectory alignment requires at least two poses.')
    if not np.isfinite(pred_c2w[valid_gt_mask]).all() or not np.isfinite(gt_c2w[valid_gt_mask]).all():
        raise ValueError('Valid pose correspondences must contain only finite values.')
    pred_c2w[:, :3, :3] = Rotation.from_matrix(pred_c2w[:, :3, :3]).as_matrix()
    gt_valid = gt_c2w[valid_gt_mask]
    gt_valid[:, :3, :3] = Rotation.from_matrix(gt_valid[:, :3, :3]).as_matrix()

    timestamps = np.flatnonzero(valid_gt_mask).astype(np.float64)
    pred_traj = PoseTrajectory3D(poses_se3=pred_c2w[valid_gt_mask].copy(), timestamps=timestamps)
    gt_traj = PoseTrajectory3D(poses_se3=gt_valid, timestamps=timestamps)
    align_rot, align_trans, align_scale = pred_traj.align(gt_traj, correct_scale=True)
    aligned_c2w = pred_c2w.copy()
    aligned_c2w[:, :3, :3] = np.einsum('ij,tjk->tik', align_rot, pred_c2w[:, :3, :3])
    aligned_c2w[:, :3, 3] = align_scale * np.einsum('ij,tj->ti', align_rot, pred_c2w[:, :3, 3]) + align_trans
    return aligned_c2w, align_rot, align_trans, float(align_scale)


def valid_camera_pose_mask(
    c2w: np.ndarray,
) -> np.ndarray:
    """
    Mark finite, non-degenerate homogeneous C2W camera matrices while allowing small rotation-rounding errors.
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f'Expected [T,4,4] C2W cameras, got {c2w.shape}.')
    finite = np.isfinite(c2w).all(axis=(1, 2))
    homogeneous = np.zeros(c2w.shape[0], dtype=bool)
    determinant = np.full(c2w.shape[0], np.nan, dtype=np.float64)
    homogeneous[finite] = np.all(np.isclose(c2w[finite, 3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5), axis=1)
    determinant[finite] = np.linalg.det(c2w[finite, :3, :3])
    return finite & homogeneous & np.isfinite(determinant) & (determinant > 1e-8)


def compute_monst3r_pose_metrics(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    valid_gt_mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Compute MonST3R camera-pose metrics using only valid GT cameras and original-adjacency RPE pairs.
    """
    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe
    from evo.core import lie_algebra as lie
    from evo.core.metrics import PoseRelation, RPE, Unit
    from evo.core.trajectory import PoseTrajectory3D

    pred_c2w = np.asarray(pred_c2w, dtype=np.float64).copy()
    gt_c2w = np.asarray(gt_c2w, dtype=np.float64).copy()
    if pred_c2w.shape != gt_c2w.shape or pred_c2w.ndim != 3 or pred_c2w.shape[1:] != (4, 4):
        raise ValueError(f'Expected matching [T,4,4] C2W trajectories, got pred={pred_c2w.shape}, gt={gt_c2w.shape}.')
    valid_gt_mask = valid_camera_pose_mask(gt_c2w) if valid_gt_mask is None else np.asarray(valid_gt_mask, dtype=bool)
    if valid_gt_mask.shape != (pred_c2w.shape[0],):
        raise ValueError(f'Expected valid_gt_mask shape {(pred_c2w.shape[0],)}, got {valid_gt_mask.shape}.')
    valid_indices = np.flatnonzero(valid_gt_mask)
    if valid_indices.size < 2:
        raise ValueError('MonST3R pose metrics require at least two poses.')
    if not np.any(np.diff(valid_indices) == 1):
        raise ValueError('MonST3R RPE requires at least one pair of adjacent valid GT poses.')
    pred_c2w = pred_c2w[valid_gt_mask]
    gt_c2w = gt_c2w[valid_gt_mask]
    if not np.isfinite(pred_c2w).all() or not np.isfinite(gt_c2w).all():
        raise ValueError('Valid pose correspondences must contain only finite values.')
    pred_c2w[:, :3, :3] = Rotation.from_matrix(pred_c2w[:, :3, :3]).as_matrix()
    gt_c2w[:, :3, :3] = Rotation.from_matrix(gt_c2w[:, :3, :3]).as_matrix()

    timestamps = valid_indices.astype(np.float64)
    has_rpe_gap = not np.all(np.diff(valid_indices) == 1)
    pred_traj = PoseTrajectory3D(poses_se3=pred_c2w.copy(), timestamps=timestamps)
    gt_traj = PoseTrajectory3D(poses_se3=gt_c2w.copy(), timestamps=timestamps)
    metric_kwargs = {
        'traj_ref': gt_traj,
        'traj_est': pred_traj,
        'est_name': 'traj',
        'align': True,
        'correct_scale': True,
    }
    ate_result = main_ape.ape(pose_relation=PoseRelation.translation_part, **metric_kwargs)
    if has_rpe_gap:
        adjacent_pair_indices = np.flatnonzero(np.diff(valid_indices) == 1)
        relative_errors = [RPE.rpe_base(gt_traj.poses_se3[idx], gt_traj.poses_se3[idx + 1], pred_traj.poses_se3[idx], pred_traj.poses_se3[idx + 1]) for idx in adjacent_pair_indices]
        rpe_trans = float(np.sqrt(np.mean([np.linalg.norm(error[:3, 3]) ** 2 for error in relative_errors])))
        rpe_rot = float(np.sqrt(np.mean([abs(lie.so3_log_angle(error[:3, :3], True)) ** 2 for error in relative_errors])))
    else:
        rpe_rot_result = main_rpe.rpe(pose_relation=PoseRelation.rotation_angle_deg, delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True, **metric_kwargs)
        rpe_trans_result = main_rpe.rpe(pose_relation=PoseRelation.translation_part, delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True, **metric_kwargs)
        rpe_trans = float(rpe_trans_result.stats['rmse'])
        rpe_rot = float(rpe_rot_result.stats['rmse'])
    return {
        'ate': float(ate_result.stats['rmse']),
        'rpe_trans': rpe_trans,
        'rpe_rot': rpe_rot,
    }


def cache_signature(
    config: dict,
    args,
) -> dict:
    """
    Describe every inference setting that changes cached pose predictions or metrics.
    """
    return {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dvd_checkpoint": str(Path(args.dvd_checkpoint).resolve()) if args.dvd_checkpoint else None,
        "only_condition_t_but_no_noise": True,
        "max_query_points_per_batch": args.max_query_points_per_batch,
        "amp_infer": args.amp_infer,
        "deterministic": args.deterministic,
        "save_inputs": args.save_inputs,
        "fps": args.fps if args.save_inputs else None,
        "dataset_id": config["dataset_id"],
        "data_root": str(Path(config["data_root"]).resolve()),
        "proc_shape": [config["proc_h"], config["proc_w"]],
        "encoder_shape": [config["img_h"], config["img_w"]],
        "center_crop": bool(args.center_crop),
        "infer_start_frame": args.start_frame,
        "infer_total_num_frames": args.infer_total_num_frames,
        "infer_chunk_num_frames": args.infer_chunk_num_frames,
        "infer_chunks_overlap_size": args.infer_chunks_overlap_size,
        # Keep fixed settings so existing adjacent/global-camera caches remain reusable.
        "global_chunk_align": True,
        "chunk_align_mode": "cameras",
        "align_with_adjacent_cam_graph": True,
        "camera_mode": "adjacent",
        "adjacent_chunk_length": args.adjacent_chunk_length,
        "adjacent_sliding_stride": args.adjacent_sliding_stride,
        "adjacent_chunk_query_center_frame": False,
        "adjacent_graph_edges": "complete",
        "adjacent_align_chunks_with_scale": False,
        "chunk_align_loss": args.chunk_align_loss,
        "align_lambda_r": args.align_lambda_r,
        "align_lambda_t": args.align_lambda_t,
        "extrinsic_grid_interval": args.extrinsic_grid_interval,
        "extrinsic_conf_threshold": args.extrinsic_conf_threshold,
        "train_num_frames": args.train_num_frames,
        "time_embed_resize_mode": "linear",
        "seed": args.seed,
    }


def try_load_complete_scene(
    scene_dir: Path,
    signature: dict,
) -> Optional[dict]:
    """
    Reuse a scene only when pose arrays, metrics, and current settings are complete.
    """
    metrics_path = scene_dir / "campose_eval_metrics.json"
    prediction_path = scene_dir / "camera_poses.npz"
    if not metrics_path.is_file() or not prediction_path.is_file():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        cached_signature = payload.get("cache_signature")
        # Ignore retired settings while requiring every current setting to match.
        if not isinstance(cached_signature, dict) or any(key not in cached_signature or cached_signature[key] != value for key, value in signature.items()):
            return None
        payload["cache_signature"] = signature
        return payload
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def acquire_scene(
    output_dir: Path,
    scene_name: str,
) -> Optional[Path]:
    """
    Atomically claim one scene across persistent workers.
    """
    processing = output_dir / PLACEHOLDER_DIR_NAME / "processing" / scene_name
    processing.parent.mkdir(parents=True, exist_ok=True)
    try:
        processing.mkdir()
    except FileExistsError:
        return None
    return processing


def process_scene(
    task: Tuple[dict, str],
) -> dict:
    """
    Run or reuse one complete camera-pose scene benchmark.
    """
    config, scene_name = task
    args = WORKER_ARGS
    output_dir = Path(config["output_dir"])
    scene_dir = output_dir / scene_name
    signature = cache_signature(config, args)
    cached = try_load_complete_scene(scene_dir, signature)
    if cached is not None:
        print(f"[Cache] {config['dataset_id']}/{scene_name}")
        return {"success": True, "scene_name": scene_name, "payload": cached}

    processing = acquire_scene(output_dir, scene_name)
    if processing is None:
        cached = try_load_complete_scene(scene_dir, signature)
        if cached is not None:
            return {"success": True, "scene_name": scene_name, "payload": cached}
        return {"success": False, "scene_name": scene_name, "error": "scene is already processing without a compatible cache"}

    try:
        set_randomness(args.seed, args.deterministic)
        scene = load_scene(config, scene_name, args.infer_total_num_frames, args.start_frame)
        video, encode_video, proc_rgb = preprocess_scene_images(scene, config, args.center_crop)
        video = video.to(WORKER_DEVICE)
        encode_video = encode_video.to(WORKER_DEVICE)
        pred_c2w, records, similarities, ranges = infer_scene_poses(WORKER_MODEL, video, encode_video, config, args)
        gt_c2w = np.asarray(scene["c2w"], dtype=np.float64)
        valid_gt_mask = valid_camera_pose_mask(gt_c2w)
        valid_gt_frame_ids = [scene["frame_ids"][idx] for idx in np.flatnonzero(valid_gt_mask)]
        invalid_gt_frame_ids = [scene["frame_ids"][idx] for idx in np.flatnonzero(~valid_gt_mask)]
        num_rpe_pairs = int(np.count_nonzero(valid_gt_mask[:-1] & valid_gt_mask[1:]))
        metrics = compute_monst3r_pose_metrics(pred_c2w, gt_c2w, valid_gt_mask)
        aligned_c2w, align_rotation, align_translation, align_scale = align_pose_trajectory_sim3(pred_c2w, gt_c2w, valid_gt_mask)
        scene_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            scene_dir / "camera_poses.npz",
            pred_c2w=pred_c2w.astype(np.float32),
            pred_c2w_aligned=aligned_c2w.astype(np.float32),
            gt_c2w=gt_c2w.astype(np.float32),
            valid_gt_mask=valid_gt_mask,
            frame_ids=np.asarray(scene["frame_ids"], dtype=str),
            chunk_ranges=np.asarray(ranges, dtype=np.int64),
        )
        if args.save_inputs:
            save_input_artifacts(scene_dir / "input", proc_rgb, scene, ranges, args.fps)
        payload = {
            "dataset_id": config["dataset_id"],
            "scene_name": scene_name,
            "source_num_frames": scene["source_num_frames"],
            "num_eval_frames": scene["num_frames"],
            "num_valid_gt_frames": len(valid_gt_frame_ids),
            "num_invalid_gt_frames": len(invalid_gt_frame_ids),
            "num_rpe_pairs": num_rpe_pairs,
            "invalid_gt_frame_ids": invalid_gt_frame_ids,
            "metrics": {
                "Absolute Translation Error (ATE)": metrics["ate"],
                "Relative Translation Error (RPE trans)": metrics["rpe_trans"],
                "Relative Rotation Error (RPE rot)": metrics["rpe_rot"],
            },
            "metric_units": {"ATE": "m", "RPE trans": "m", "RPE rot": "degree"},
            "aggregation": "evo RMSE per scene after valid-GT Sim(3); RPE uses originally adjacent valid GT pairs; root summary is the unweighted scene mean",
            "chunk_ranges": [[start, end] for start, end in ranges],
            "chunk_alignment": [
                {
                    "chunk": records[idx]["name"],
                    "mode": "cameras",
                    "scale": float(similarity["scale"]),
                    "rotation": np.asarray(similarity["rotation"]).tolist(),
                    "translation": np.asarray(similarity["translation"]).tolist(),
                }
                for idx, similarity in enumerate(similarities)
            ],
            "evaluation_alignment": {
                "scale": float(align_scale),
                "rotation": np.asarray(align_rotation).tolist(),
                "translation": np.asarray(align_translation).tolist(),
            },
            "cache_signature": signature,
            "frame_ids": scene["frame_ids"],
        }
        write_json(scene_dir / "campose_eval_metrics.json", payload)
        processing.rmdir()
        print(f"[Metrics] {config['dataset_id']}/{scene_name} ATE={metrics['ate']:.6f} RPE_trans={metrics['rpe_trans']:.6f} RPE_rot={metrics['rpe_rot']:.6f}")
        return {"success": True, "scene_name": scene_name, "payload": payload}
    except Exception as error:
        traceback.print_exc()
        if processing.exists():
            processing.rmdir()
        return {"success": False, "scene_name": scene_name, "error": str(error)}


def init_worker(
    args,
    num_workers: int,
) -> None:
    """
    Bind one persistent process to one visible GPU and load the checkpoint exactly once.
    """
    global WORKER_ARGS, WORKER_DEVICE, WORKER_MODEL
    WORKER_ARGS = args
    identity = mp.current_process()._identity
    worker_idx = int(identity[0] - 1) if identity else 0
    if torch.cuda.is_available():
        local_device_idx = worker_idx % max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(local_device_idx)
        WORKER_DEVICE = torch.device(f"cuda:{local_device_idx}")
    else:
        WORKER_DEVICE = torch.device("cpu")
    set_randomness(args.seed + worker_idx, args.deterministic)
    print(f"[Worker {worker_idx + 1}/{num_workers}] device={WORKER_DEVICE} loading checkpoint once")
    WORKER_MODEL = load_model(args.checkpoint, WORKER_DEVICE, args, per_clip_time=True)


def write_root_summary(
    config: dict,
    results: List[dict],
    args,
) -> None:
    """
    Write MonST3R-style unweighted means and explicitly retain failed scene diagnostics.
    """
    successful = [result for result in results if result["success"]]
    failures = [{"scene_name": result["scene_name"], "error": result["error"]} for result in results if not result["success"]]
    requested = SCANNET_EVAL_SCENES if config["dataset_id"] == "SCANNET" and (not args.target_scene_list or "*" in args.target_scene_list) else (args.target_scene_list or [])
    skipped = {name: {"reason": "known invalid GT cameras", "invalid_source_frame_ids": SCANNET_INVALID_GT_FRAMES[name]} for name in requested if config["dataset_id"] == "SCANNET" and name in SCANNET_INVALID_GT_FRAMES}
    metric_names = ["Absolute Translation Error (ATE)", "Relative Translation Error (RPE trans)", "Relative Rotation Error (RPE rot)"]
    means = {}
    for metric_name in metric_names:
        values = [float(result["payload"]["metrics"][metric_name]) for result in successful]
        means[metric_name] = float(np.mean(values)) if values else None
    payload = {
        "dataset_id": config["dataset_id"],
        "num_requested_scenes": len(results) + len(skipped),
        "num_skipped_scenes": len(skipped),
        "num_successful_scenes": len(successful),
        "num_failed_scenes": len(failures),
        "scene_mean_metrics": means,
        "metric_units": {"ATE": "m", "RPE trans": "m", "RPE rot": "degree"},
        "scene_metrics": {result["scene_name"]: result["payload"]["metrics"] for result in successful},
        "failed_scenes": failures,
        "skipped_scenes": skipped,
        "cache_signature": cache_signature(config, args),
    }
    write_json(Path(config["output_dir"]) / "campose_bench_metrics-ALL.json", payload)


def main() -> None:
    args = parse_args()
    configs = parse_dataset_configs(args.dataset_configs)
    scene_lists = {config["dataset_id"]: select_scene_names(config["dataset_id"], Path(config["data_root"]), args.target_scene_list) for config in configs}
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_workers = args.num_workers if args.num_workers is not None else max(visible_gpus, 1)
    if num_workers <= 0:
        raise ValueError("--num_workers must be positive.")
    if visible_gpus > 0 and num_workers > visible_gpus:
        raise ValueError(f"num_workers={num_workers} exceeds visible CUDA devices={visible_gpus}.")

    if num_workers == 1:
        init_worker(args, 1)
        for config in configs:
            results = [process_scene((config, scene_name)) for scene_name in scene_lists[config["dataset_id"]]]
            write_root_summary(config, results, args)
    else:
        context = mp.get_context("spawn")
        with context.Pool(num_workers, initializer=init_worker, initargs=(args, num_workers)) as pool:
            for config in configs:
                scene_names = scene_lists[config["dataset_id"]]
                print(f"[Dataset] {config['dataset_id']} scenes={len(scene_names)} output={config['output_dir']}")
                results = list(pool.imap_unordered(process_scene, [(config, scene_name) for scene_name in scene_names]))
                results.sort(key=lambda item: item["scene_name"])
                write_root_summary(config, results, args)

    for config in configs:
        print(f"[Output] {config['dataset_id']}: {Path(config['output_dir']).resolve()}", flush=True)


if __name__ == "__main__":
    main()
