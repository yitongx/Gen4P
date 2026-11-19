#!/usr/bin/env python3
"""
Evaluate Gen4P video depth on Bonn, Sintel, and KITTI.

Align and merge chunk predictions, then report AbsRel and delta1 under scale-only
and scale-shift alignment. Export depth, confidence, and error visualizations.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import imageio
import matplotlib
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.visualization import visualize_depth

from utils.inference import str2bool, setRandomness, get_visible_cuda_devices
from utils.inference import validate_image_aspect_ratios
from utils.inference import CUDA_VISIBLE_DEVICES_RAW
from utils.inference import load_model as _load_model


def load_model(
    checkpoint_path,
    device,
    cli_args=None,
):
    return _load_model(checkpoint_path, device, cli_args, per_clip_time=True)


FIXED_ERROR_VMIN = 0.0
FIXED_ERROR_VMAX = 15.0
SINTEL_TAG_FLOAT = 202021.25
BONN_DEPTH_SCALE = 5000.0
KITTI_DEPTH_SCALE = 256.0
DEPTH_EPSILON = 1e-6
DELTA1_UNIT = 'percent'
BONN_EVAL_SCENES = ['balloon2', 'crowd2', 'crowd3', 'person_tracking2', 'synchronous']
SCENE_PLACEHOLDER_ROOT_DIR_NAME = '.bench_depth_placeholder'
SCENE_PLACEHOLDER_PROCESSING_DIR_NAME = 'processing'
SCENE_PLACEHOLDER_DONE_DIR_NAME = 'done'
WORKER_ARGS = None
WORKER_DEVICE = None
WORKER_MODEL = None
WORKER_DATASET_TYPE = None
WORKER_DATASET_ID = None


def parse_args():
    parser = argparse.ArgumentParser(description='D4RT benchmark depth inference + evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dvd_checkpoint', type=str, default=None, help='Override ckpt/dvd_1.1.safetensors')
    parser.add_argument('--video', type=str, default=None, help='Path to input video file for single-scene mode')
    parser.add_argument('--target_folder', type=str, default=None, help='Folder containing target depth files for single-scene mode')
    parser.add_argument('--data_root', type=str, default=None, help='Dataset root for multi-scene benchmark mode')
    parser.add_argument('--split', type=str, default=None, help='Dataset split for multi-scene benchmark mode')
    parser.add_argument('--dataset_configs', nargs='*', default=None, help=('Sequential persistent-worker configs formatted as data_root,split,proc_h,proc_w,img_h,img_w,target_max,output_dir'))
    parser.add_argument('--target_scene_list', nargs='*', default=None, help='Scene list for multi-scene benchmark mode')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of scene workers for multi-scene benchmark mode')
    parser.add_argument('--output_dir', type=str, default='output_eval_depth', help='Output directory')
    parser.add_argument('--infer_total_num_frames', type=int, default=-1, help=('Number of frames to process; accepts -1 for the dataset-defined full scene or 110 for a fixed-length interval'))
    parser.add_argument('--infer_chunk_num_frames', type=int, default=None, help='Maximum clip length processed by the model at once')
    parser.add_argument('--infer_chunks_overlap_size', type=int, default=None, help=('Depth benchmark overlap frames used for adjacent chunk alignment; nominal stride is infer_chunk_num_frames - infer_chunks_overlap_size'))
    parser.add_argument('--train_num_frames', type=int, default=2, help='Optional override for checkpoint training num_frames used by time-embedding resize')
    parser.add_argument('--align_method', type=str, default='l1+adam', choices=['l1+adam', 'l2+lsq'], help='Scale-shift alignment method for chunk merging and final depth metrics')
    parser.add_argument('--align_iters', type=int, default=1000, help='Maximum Adam iterations for chunk merging and final scale-shift alignment')
    parser.add_argument('--align_lr', type=float, default=1e-4, help='Adam learning rate for chunk merging and final scale-shift alignment')
    parser.add_argument('--align_mask', type=str, default='gt+pred', choices=['gt', 'gt+pred'], help=('Mask used for final depth alignment and metrics: GT-valid only (MonST3R) or GT/prediction-valid intersection'))
    parser.add_argument('--sequential_chunk_align', type=str, default='forward', choices=['forward', 'backward'], help='Sequential chunk merge direction when --global_chunk_align is false')
    parser.add_argument('--global_chunk_align', type=str2bool, default=False, help=('Optimize one scale-shift per chunk over all decoded overlaps instead of sequential chunk alignment'))
    parser.add_argument('--max_stride', type=int, default=1, help=('Run extra single-pass temporal strides from 2 through max_stride and merge their decoded frames with the chunk predictions'))
    parser.add_argument('--start_frame', type=int, default=0, help='Starting frame (global index in source video)')
    parser.add_argument('--frame_stride', type=int, default=1, help='Sample stride')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size for encoder (square side length when --img_h/--img_w are not set)')
    parser.add_argument('--img_h', type=int, default=None, help='Optional encoder input height override (must be used with --img_w)')
    parser.add_argument('--img_w', type=int, default=None, help='Optional encoder input width override (must be used with --img_h)')
    parser.add_argument('--proc_h', type=int, default=None, help='Optional processed/model-output height (must be used with --proc_w)')
    parser.add_argument('--proc_w', type=int, default=None, help='Optional processed/model-output width (must be used with --proc_h)')
    parser.add_argument('--center_crop', type=str2bool, default=True, help=('Center-crop raw frames to proc aspect ratio before resize; false keeps full-frame resize.'))
    parser.add_argument('--raw_rgb', type=str2bool, default=True, help='Use raw RGB images without square resize')
    parser.add_argument('--fps', type=int, default=12, help='Output video FPS')
    parser.add_argument('--target_max', type=float, default=65504.0, help='Valid-mask upper bound for target depth')
    parser.add_argument('--vis_conf_percentile', type=str2bool, default=True, help=('Use per-frame percentile normalization for confidence visualization instead of absolute [0,1].'))
    parser.add_argument('--conf_vis_low', type=float, default=0.0, help=('Lower bound of displayed confidence range; confidence is visualized over [conf_vis_low, 1.0].'))
    parser.add_argument('--amp_infer', type=str2bool, default=False, help='Enable AMP autocast (bf16 on CUDA) during depth inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible inference (Python/NumPy/Torch)')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic kernels for reproducibility (may reduce speed)')
    parser.add_argument('--use_wan_feature_encoder', type=str2bool, default=None, help='Override use_wan_feature_encoder from checkpoint args (true/false)')
    parser.add_argument('--wan_feature_model_id', type=str, default=None, help='Override wan_feature_model_id from checkpoint args')
    parser.add_argument('--wan_torch_dtype', type=str, default=None, help='Override Wan torch dtype, set both VAE and DiT.')
    parser.add_argument('--only_condition_t_but_no_noise', type=str2bool, default=True, help='Must be true: keep the timestep condition and disable latent noise')
    parser.add_argument('--pre_vae_encode', type=str2bool, default=None, help=("Override pre_vae_encode flag used to build model structure (true/false). Should ALWAYS set false in the cli_args since we don't have pre-encoded latent for inference."))
    parser.add_argument('--wan_device', type=str, default=None, help='Override Wan runtime device')
    parser.add_argument('--wan_train_mode', type=str, default=None, choices=['lora', 'frozen'], help='Wan adapter mode used to restore checkpoint modules')
    parser.add_argument('--wan_lora_rank', type=int, default=None, help='Override Wan LoRA rank used to build model structure')
    parser.add_argument('--wan_lora_alpha', type=int, default=None, help='Override Wan LoRA alpha used to build model structure')
    parser.add_argument('--wan_lora_target_modules', type=str, default=None, help='Override Wan LoRA target modules used to build model structure')
    parser.add_argument('--wan_lora_adapter_name', type=str, default=None, help='Override Wan LoRA adapter name used to build model structure')
    args = parser.parse_args()
    has_img_h = args.img_h is not None
    has_img_w = args.img_w is not None
    if has_img_h != has_img_w:
        parser.error('--img_h and --img_w must be provided together.')
    if has_img_h and has_img_w:
        assert args.img_h > 0 and args.img_w > 0, '--img_h and --img_w must be positive integers.'
        args.img_h = int(args.img_h)
        args.img_w = int(args.img_w)
    else:
        args.img_h = int(args.img_size)
        args.img_w = int(args.img_size)
    has_proc_h = args.proc_h is not None
    has_proc_w = args.proc_w is not None
    if has_proc_h != has_proc_w:
        parser.error('--proc_h and --proc_w must be provided together.')
    if not args.dataset_configs and (not (has_proc_h and has_proc_w)):
        parser.error('--proc_h and --proc_w must be provided together and are required for this benchmark.')
    if has_proc_h and has_proc_w:
        assert args.proc_h > 0 and args.proc_w > 0, '--proc_h and --proc_w must be positive integers.'
        args.proc_h = int(args.proc_h)
        args.proc_w = int(args.proc_w)
    if has_proc_h and has_proc_w and not args.dataset_configs:
        try:
            validate_image_aspect_ratios(args.img_h, args.img_w, args.proc_h, args.proc_w)
        except ValueError as error:
            parser.error(str(error))
    args.parser_defaults = {k: parser.get_default(k) for k in vars(args).keys()}
    has_single_scene_inputs = bool(args.video is not None and args.target_folder is not None)
    has_multi_scene_inputs = bool(args.data_root is not None and args.split is not None and args.target_scene_list)
    has_dataset_configs = bool(args.dataset_configs)
    if sum([has_single_scene_inputs, has_multi_scene_inputs, has_dataset_configs]) != 1:
        parser.error('Provide exactly one of --video/--target_folder, --data_root/--split/--target_scene_list, or --dataset_configs.')
    if args.num_workers is not None and args.num_workers <= 0:
        parser.error('--num_workers must be positive when provided.')
    if args.infer_total_num_frames not in (-1, 110):
        parser.error('--infer_total_num_frames must be -1 for dataset-defined full-scene evaluation or 110 for a fixed-length interval.')
    if args.infer_chunk_num_frames is not None and args.infer_chunk_num_frames <= 0:
        parser.error('--infer_chunk_num_frames must be positive when provided.')
    if args.infer_chunk_num_frames is not None and (int(args.infer_chunk_num_frames) - 1) % 4 != 0:
        parser.error('--infer_chunk_num_frames must satisfy 4N+1.')
    if args.infer_chunks_overlap_size is not None and args.infer_chunks_overlap_size <= 0:
        parser.error('--infer_chunks_overlap_size must be positive when provided.')
    if args.max_stride <= 0:
        parser.error('--max_stride must be positive.')
    if args.align_iters <= 0:
        parser.error('--align_iters must be positive.')
    if not np.isfinite(args.align_lr) or args.align_lr <= 0:
        parser.error('--align_lr must be finite and positive.')
    if args.train_num_frames is not None and args.train_num_frames <= 0:
        parser.error('--train_num_frames must be positive when provided.')
    return args


def detect_dataset_type(
    video_path: str,
    target_folder: Union[str, Path],
) -> str:
    target_folder_str = str(target_folder).lower()
    video_path_str = str(video_path).lower()
    path_text = f'{target_folder_str} {video_path_str}'

    if 'pointodyssey' in path_text:
        return 'po'
    if 'dynamic_replica' in path_text:
        return 'dr'
    if 'sintel' in path_text:
        return 'sintel'
    if 'bonn' in path_text:
        return 'bonn'
    if 'kitti' in path_text:
        return 'kitti'

    target_folder_path = Path(target_folder)
    if any(target_folder_path.glob('*.geometric.png')):
        return 'dr'
    if any(target_folder_path.glob('*.dpt')):
        return 'sintel'
    if target_folder_path.name == 'depth_110' and any(target_folder_path.glob('*.png')):
        return 'bonn'
    if (target_folder_path / 'depth_00000.png').exists():
        return 'po'

    raise ValueError(f'Could not infer dataset type from paths. video={video_path}, target_folder={target_folder}.')


def loadFloat16DepthPng(
    depth_path: Path,
) -> np.ndarray:
    depth_pil = Image.open(depth_path)
    depth = np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16).astype(np.float32).reshape((depth_pil.size[1], depth_pil.size[0]))
    depth_pil.close()
    depth[~np.isfinite(depth)] = 0.0
    return depth


def loadSintelDepthDpt(
    depth_path: Path,
) -> np.ndarray:
    """
    Read one MPI Sintel `.dpt` z-depth file into a float32 HxW array.

    Sintel stores depth with the same binary header convention as its `.flo` files: one float tag
    followed by int32 width/height and then row-major float32 depth values. Non-finite values are
    treated as invalid depth and set to zero so the shared benchmark valid-mask logic can drop them
    later.
    """
    with open(depth_path, 'rb') as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        if check != SINTEL_TAG_FLOAT:
            raise ValueError(f'Wrong Sintel depth tag in {depth_path}: {check}')
        width = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        height = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        if width <= 0 or height <= 0 or width * height <= 1:
            raise ValueError(f'Invalid Sintel depth size in {depth_path}: {(height, width)}')
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def loadBonnDepthPng(
    depth_path: Path,
) -> np.ndarray:
    """
    Read one Bonn RGB-D depth PNG into metric depth in meters.

    Bonn stores depth as uint16 with scale 5000. Raw zero pixels are invalid and remain zero so the
    shared `target > 0` valid-mask logic drops them.
    """
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth_raw is None:
        raise ValueError(f'Could not read Bonn depth PNG: {depth_path}')
    depth = depth_raw.astype(np.float32) / float(BONN_DEPTH_SCALE)
    depth[depth_raw == 0] = 0.0
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def loadKittiDepthPng(
    depth_path: Path,
) -> np.ndarray:
    """
    Read one KITTI projected ground-truth depth PNG into metric depth in meters.

    KITTI stores depth as uint16 with scale 256. Raw zero pixels are invalid and remain zero so the
    shared `target > 0` valid-mask logic drops them.
    """
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth_raw is None:
        raise ValueError(f'Could not read KITTI depth PNG: {depth_path}')
    depth = depth_raw.astype(np.float32) / float(KITTI_DEPTH_SCALE)
    depth[depth_raw == 0] = 0.0
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


# config args keys -> model kwargs keys


def resolve_scene_num_frames(
    video_path: str,
    requested_num_frames: int,
    start_frame: int,
    frame_stride: int,
    image_paths: Optional[List[Path]] = None,
) -> int:
    """
    Resolve the actual number of frames to evaluate for one scene.
    """
    if start_frame < 0:
        raise ValueError(f'start_frame must be >= 0, got {start_frame}')
    if frame_stride <= 0:
        raise ValueError(f'frame_stride must be > 0, got {frame_stride}')
    if requested_num_frames == 0 or requested_num_frames < -1:
        raise ValueError(f'num_frames must be positive or -1 for full-scene evaluation, got {requested_num_frames}')

    video_path = Path(video_path)
    if image_paths is not None:
        total_source_frames = len(image_paths)
    elif video_path.is_dir():
        source_image_paths = sorted(video_path.glob('*.png'))
        if not source_image_paths:
            raise ValueError(f'Could not find PNG frames under directory: {video_path}')
        total_source_frames = len(source_image_paths)
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f'Could not open video: {video_path}')
        total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_source_frames <= 0:
            raise ValueError(f'Could not infer total frame count from video for num_frames=-1. video={video_path}')

    if start_frame >= total_source_frames:
        raise ValueError(f'start_frame {start_frame} exceeds available frame count {total_source_frames} for {video_path}')

    if requested_num_frames > 0:
        return min(int(requested_num_frames), len(range(start_frame, total_source_frames, frame_stride)))

    return len(range(start_frame, total_source_frames, frame_stride))


def load_video(
    video_path: str,
    num_frames: int,
    enc_img_h: int,
    enc_img_w: int,
    proc_h: Optional[int],
    proc_w: Optional[int],
    raw_rgb: bool,
    center_crop: bool = True,
    start_frame: int = 0,
    frame_stride: int = 1,
    image_paths: Optional[List[Path]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """
    Load one benchmark scene video and return processed frames plus metadata.
    """
    video_path = Path(video_path)

    if num_frames <= 0:
        raise ValueError(f'num_frames must be > 0, got {num_frames}')
    if start_frame < 0:
        raise ValueError(f'start_frame must be >= 0, got {start_frame}')
    if frame_stride <= 0:
        raise ValueError(f'frame_stride must be > 0, got {frame_stride}')
    frames = []
    orig_h, orig_w = None, None

    if video_path.is_dir():
        source_image_paths = sorted(video_path.glob('*.png')) if image_paths is None else image_paths
        if not source_image_paths:
            raise ValueError(f'Could not find PNG frames under directory: {video_path}')

        sampled_indices = [start_frame + i * frame_stride for i in range(num_frames)]
        if sampled_indices[-1] >= len(source_image_paths):
            raise ValueError(f'Requested sampled frame {sampled_indices[-1]} exceeds available frame count {len(source_image_paths)} for {video_path}.')

        for frame_idx in sampled_indices:
            frame = cv2.imread(str(source_image_paths[frame_idx]), cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise ValueError(f'Could not read frame: {source_image_paths[frame_idx]}')
            if frame.ndim != 3:
                raise ValueError(f'Expected HxWxC image, got shape {frame.shape} from {source_image_paths[frame_idx]}')
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f'Expected 3 or 4 image channels, got shape {frame.shape} from {image_paths[frame_idx]}')
            if orig_h is None or orig_w is None:
                orig_h, orig_w = frame.shape[:2]
            frames.append(frame)
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f'Could not open video: {video_path}')

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            sampled_indices = [start_frame + i * frame_stride for i in range(num_frames)]
            if sampled_indices[-1] >= total_frames:
                cap.release()
                raise ValueError(f'Requested sampled frame {sampled_indices[-1]} exceeds available frame count {total_frames} for {video_path}.')
        else:
            sampled_indices = [start_frame + i * frame_stride for i in range(num_frames)]

        for frame_idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if orig_h is None or orig_w is None:
                orig_h, orig_w = frame.shape[:2]
            frames.append(frame)
        cap.release()

    if not frames:
        raise ValueError(f'Could not load frames from {video_path}')

    video_np = np.stack(frames, axis=0)  # uint8

    def compute_crop_and_proc_shapes(
        height: int,
        width: int,
        target_h: Optional[int],
        target_w: Optional[int],
        center_crop: bool,
    ) -> Tuple[int, int, int, int]:
        if target_h is None or target_w is None:
            raise ValueError('proc_h and proc_w are required.')

        proc_h_out = int(target_h)
        proc_w_out = int(target_w)
        if proc_h_out <= 0 or proc_w_out <= 0:
            raise ValueError(f'Invalid proc shape: {(proc_h_out, proc_w_out)}')
        if not bool(center_crop):
            return int(height), int(width), proc_h_out, proc_w_out

        ratio_gcd = int(np.gcd(proc_h_out, proc_w_out))
        unit_h = proc_h_out // ratio_gcd
        unit_w = proc_w_out // ratio_gcd
        resize_scale = min(height // unit_h, width // unit_w)
        if resize_scale <= 0:
            raise ValueError(f'Source frame too small for requested proc aspect ratio. source={(height, width)}, proc={(proc_h_out, proc_w_out)}, unit={(unit_h, unit_w)}')

        crop_h = unit_h * resize_scale
        crop_w = unit_w * resize_scale
        return crop_h, crop_w, proc_h_out, proc_w_out

    crop_h, crop_w, proc_h_out, proc_w_out = compute_crop_and_proc_shapes(orig_h, orig_w, proc_h, proc_w, center_crop=center_crop)
    crop_top = (orig_h - crop_h) // 2
    crop_left = (orig_w - crop_w) // 2
    video_crop_np = video_np[:, crop_top : crop_top + crop_h, crop_left : crop_left + crop_w, :]
    if proc_h_out != crop_h or proc_w_out != crop_w:
        resized_video_np = np.empty((video_np.shape[0], proc_h_out, proc_w_out, 3), dtype=video_np.dtype)
        for i in range(video_np.shape[0]):
            resized_video_np[i] = cv2.resize(video_crop_np[i], (proc_w_out, proc_h_out), interpolation=cv2.INTER_LINEAR)
        video_proc_np = resized_video_np
    else:
        video_proc_np = video_crop_np
    scale_x = float(proc_w_out) / max(float(crop_w), 1.0)
    scale_y = float(proc_h_out) / max(float(crop_h), 1.0)
    if bool(center_crop) and abs(scale_x - scale_y) > 1e-6:
        raise RuntimeError(f'Non-uniform resize scale detected: sx={scale_x}, sy={scale_y}')

    # convert to float32 [0, 1]
    if raw_rgb or (not bool(center_crop)):
        video_np_out = video_proc_np
    else:
        if enc_img_h != crop_h or enc_img_w != crop_w:
            video_np_out = np.empty((video_np.shape[0], enc_img_h, enc_img_w, 3), dtype=video_np.dtype)
            for i in range(video_np.shape[0]):
                video_np_out[i] = cv2.resize(video_crop_np[i], (enc_img_w, enc_img_h), interpolation=cv2.INTER_LINEAR)
        else:
            video_np_out = video_crop_np

    # Build encoder frames directly from the same raw crop instead of upsampling proc frames.
    encode_video_np = np.empty((video_np.shape[0], enc_img_h, enc_img_w, 3), dtype=video_np.dtype)
    for i in range(video_np.shape[0]):
        encode_video_np[i] = cv2.resize(video_crop_np[i], (enc_img_w, enc_img_h), interpolation=cv2.INTER_LINEAR)
    video = torch.from_numpy(video_np_out.astype(np.float32) / 255.0).to(torch.float32)
    encode_video = torch.from_numpy(encode_video_np.astype(np.float32) / 255.0).to(torch.float32)

    meta = {
        'orig_shape': (orig_h, orig_w),
        'center_crop': bool(center_crop),
        'crop_shape': (crop_h, crop_w),
        'proc_shape': (proc_h_out, proc_w_out),
        'output_shape': (int(video_np_out.shape[1]), int(video_np_out.shape[2])),
        'encode_shape': (int(encode_video_np.shape[1]), int(encode_video_np.shape[2])),
        'crop_offsets': (crop_top, crop_left),
        'sampled_indices': sampled_indices,
    }
    return video, encode_video, meta


@torch.no_grad()
def run_depth_inference(
    model,
    video_tchw: torch.Tensor,
    encode_video_tchw: torch.Tensor,
    enc_img_size: Tuple[int, int],
    output_resolution: Tuple[int, int],
    amp_infer: bool = False,
    target_frame_indices: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run one clip of depth inference and return `(depth, confidence)` in `(T,H,W)`.

    When `target_frame_indices` is provided, only those local target frames are decoded and returned
    in that order. This reduces dense decoder work for sliding-window overlap chunks while keeping
    the full 4N+1 RGB clip available to the encoder.
    """
    video_b = video_tchw.unsqueeze(0)

    amp_enabled = bool(amp_infer and video_tchw.device.type == 'cuda')
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
        encode_video_b = encode_video_tchw.unsqueeze(0)
        depth, confidence = model.predict_depth(video_b, enc_img_size, encode_video=encode_video_b, output_resolution=output_resolution, decoder_dtype=None if amp_enabled else torch.get_default_dtype(), target_frame_indices=target_frame_indices)
    return depth.squeeze(0).cpu(), confidence.squeeze(0).cpu()


def inferDepthChunkRecord(
    model,
    video_tchw: torch.Tensor,
    encode_video_tchw: torch.Tensor,
    enc_img_size: Tuple[int, int],
    output_resolution: Tuple[int, int],
    sampled_global_indices: np.ndarray,
    target_global_indices: np.ndarray,
    infer_clip_num_frames: int,
    record_name: str,
    record_stride: int,
    amp_infer: bool = False,
) -> Dict[str, object]:
    """
    Infer one encoded RGB clip and return only the requested decoded global frames.

    The encoder still receives the full sampled clip, padded only to the next valid `4N+1` length
    when needed. The dense depth decoder receives `target_frame_indices`, so repeated overlap frames
    outside the alignment set are not decoded.
    """
    sampled_global_indices = np.asarray(sampled_global_indices, dtype=np.int64)
    target_global_indices = np.asarray(target_global_indices, dtype=np.int64)
    if sampled_global_indices.size == 0 or target_global_indices.size == 0:
        raise ValueError(f'{record_name} has empty sampled or target frame indices.')
    model_device = next(model.parameters()).device
    clip_video_tchw = video_tchw[torch.as_tensor(sampled_global_indices, dtype=torch.long)].to(model_device)
    clip_encode_video_tchw = encode_video_tchw[torch.as_tensor(sampled_global_indices, dtype=torch.long)].to(model_device)
    keep_clip_frames = int(sampled_global_indices.size)
    if keep_clip_frames > int(infer_clip_num_frames):
        raise RuntimeError(f'{record_name} sampled {keep_clip_frames} frames, exceeding max chunk {infer_clip_num_frames}.')
    if keep_clip_frames < int(infer_clip_num_frames):
        encode_clip_frames = (keep_clip_frames + 2) // 4 * 4 + 1
        if encode_clip_frames > int(infer_clip_num_frames):
            raise RuntimeError(f'Internal sliding-window error: clip length {keep_clip_frames} requires 4N+1 encode length {encode_clip_frames}, which exceeds max chunk {infer_clip_num_frames}.')
        pad_count = int(encode_clip_frames) - keep_clip_frames
        if pad_count > 3:
            raise RuntimeError(f'Internal padding error: expected at most 3 padded frames, got {pad_count}.')
        print(f'[Sliding Window] {record_name} stride={record_stride} real_frames={keep_clip_frames} max_chunk={infer_clip_num_frames} encode_frames={encode_clip_frames} padded_by_repeating_last_frame={pad_count}')
        if pad_count > 0:
            pad_frames = clip_video_tchw[-1:].repeat(pad_count, 1, 1, 1)
            clip_video_tchw = torch.cat([clip_video_tchw, pad_frames], dim=0)
            pad_encode_frames = clip_encode_video_tchw[-1:].repeat(pad_count, 1, 1, 1)
            clip_encode_video_tchw = torch.cat([clip_encode_video_tchw, pad_encode_frames], dim=0)
    local_index_by_global = {int(global_idx): local_idx for local_idx, global_idx in enumerate(sampled_global_indices.tolist())}
    target_frame_indices = [int(local_index_by_global[int(global_idx)]) for global_idx in target_global_indices.tolist()]
    print(f'[Sliding Decode] {record_name} stride={record_stride} sampled_global={sampled_global_indices.tolist()} decode_global={target_global_indices.tolist()}')
    clip_depth, clip_confidence = run_depth_inference(
        model=model, video_tchw=clip_video_tchw, encode_video_tchw=clip_encode_video_tchw, enc_img_size=enc_img_size, output_resolution=output_resolution, amp_infer=amp_infer, target_frame_indices=target_frame_indices
    )
    return {
        'name': record_name,
        'stride': int(record_stride),
        'global_indices': target_global_indices,
        'depth': clip_depth.to(torch.float32).cpu().numpy().astype(np.float32),
        'confidence': clip_confidence.to(torch.float32).cpu().numpy().astype(np.float32),
    }


def buildDenseDepthChunkTargets(
    total_frames: int,
    clip_ranges: List[Tuple[int, int]],
    overlap_align_frames: int,
    sequential_chunk_align: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build sampled and decoded global frame indices for the dense stride-1 sliding chunks.

    Forward mode matches the original benchmark behavior. Backward mode simulates coverage from the
    tail chunk first, so the last chunk is decoded fully and earlier chunks decode their new prefix
    plus the first requested overlap frames against the already-covered tail.
    """
    total_frames = int(total_frames)
    decoded_count = np.zeros((total_frames,), dtype=np.float32)
    targets_by_clip: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None for _ in clip_ranges]
    clip_order = list(range(len(clip_ranges))) if sequential_chunk_align == 'forward' else list(range(len(clip_ranges) - 1, -1, -1))
    for order_idx, clip_idx in enumerate(clip_order):
        clip_start, clip_end = clip_ranges[clip_idx]
        sampled_global_indices = np.arange(clip_start, clip_end, dtype=np.int64)
        if order_idx == 0:
            target_global_indices = sampled_global_indices.copy()
        else:
            existing_count = decoded_count[sampled_global_indices]
            overlap_global_indices = sampled_global_indices[existing_count > 0]
            if overlap_global_indices.size > 0:
                if sequential_chunk_align == 'forward':
                    overlap_global_indices = overlap_global_indices[-min(int(overlap_align_frames), overlap_global_indices.size) :]
                else:
                    overlap_global_indices = overlap_global_indices[: min(int(overlap_align_frames), overlap_global_indices.size)]
            new_global_indices = sampled_global_indices[existing_count <= 0]
            target_global_indices = np.unique(np.concatenate([overlap_global_indices, new_global_indices], axis=0))
        decoded_count[target_global_indices] = decoded_count[target_global_indices] + 1.0
        targets_by_clip[clip_idx] = (sampled_global_indices, target_global_indices)
    return [item for item in targets_by_clip if item is not None]


def collectDepthChunkRecords(
    model,
    video_tchw: torch.Tensor,
    encode_video_tchw: torch.Tensor,
    enc_img_size: Tuple[int, int],
    output_resolution: Tuple[int, int],
    dense_chunk_targets: List[Tuple[np.ndarray, np.ndarray]],
    infer_clip_num_frames: int,
    max_stride: int,
    amp_infer: bool = False,
    progress_bar: Optional[tqdm] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Run all dense stride-1 chunks and optional one-shot temporal stride chunks.

    `max_stride=1` only runs the dense sliding chunks. Larger values add one extra record for every
    stride in `[2, max_stride]`, using at most `infer_clip_num_frames` sampled frames and dropping
    sampled indices outside the scene.
    """
    total_frames = int(video_tchw.shape[0])
    if int(encode_video_tchw.shape[0]) != total_frames:
        raise ValueError(f'encode_video_tchw frame count {int(encode_video_tchw.shape[0])} does not match video_tchw {total_frames}.')
    coverage_seen = np.zeros((total_frames,), dtype=bool)
    dense_records: List[Dict[str, object]] = []
    stride_records: List[Dict[str, object]] = []
    for clip_idx, (sampled_global_indices, target_global_indices) in enumerate(dense_chunk_targets):
        if progress_bar is not None:
            progress_bar.set_postfix_str(f'sliding_chunk={clip_idx + 1}/{len(dense_chunk_targets)} dense infer')
        record = inferDepthChunkRecord(
            model=model,
            video_tchw=video_tchw,
            encode_video_tchw=encode_video_tchw,
            enc_img_size=enc_img_size,
            output_resolution=output_resolution,
            sampled_global_indices=sampled_global_indices,
            target_global_indices=target_global_indices,
            infer_clip_num_frames=infer_clip_num_frames,
            record_name=f'dense_chunk={clip_idx + 1}/{len(dense_chunk_targets)}',
            record_stride=1,
            amp_infer=amp_infer,
        )
        dense_records.append(record)
        if progress_bar is not None:
            new_frames = np.asarray(record['global_indices'], dtype=np.int64)[~coverage_seen[np.asarray(record['global_indices'], dtype=np.int64)]]
            coverage_seen[new_frames] = True
            progress_bar.update(int(new_frames.size))
    for stride in range(2, int(max_stride) + 1):
        sampled_global_indices = np.arange(0, int(infer_clip_num_frames) * stride, stride, dtype=np.int64)
        sampled_global_indices = sampled_global_indices[sampled_global_indices < total_frames]
        if sampled_global_indices.size == 0:
            continue
        if progress_bar is not None:
            progress_bar.set_postfix_str(f'extra_stride={stride} infer')
        record = inferDepthChunkRecord(
            model=model,
            video_tchw=video_tchw,
            encode_video_tchw=encode_video_tchw,
            enc_img_size=enc_img_size,
            output_resolution=output_resolution,
            sampled_global_indices=sampled_global_indices,
            target_global_indices=sampled_global_indices,
            infer_clip_num_frames=infer_clip_num_frames,
            record_name=f'extra_stride={stride}',
            record_stride=stride,
            amp_infer=amp_infer,
        )
        stride_records.append(record)
    return (dense_records, stride_records)


def mergeSequentialDepthChunkRecords(
    records: List[Dict[str, object]],
    total_frames: int,
    output_resolution: Tuple[int, int],
    align_method: str,
    align_iters: int,
    align_lr: float,
    target_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge chunk records in caller-provided order by fitting each new record to already-merged
    overlap frames.
    """
    out_h, out_w = int(output_resolution[0]), int(output_resolution[1])
    depth_sum = np.zeros((int(total_frames), out_h, out_w), dtype=np.float32)
    confidence_sum = np.zeros((int(total_frames), out_h, out_w), dtype=np.float32)
    depth_count = np.zeros((int(total_frames),), dtype=np.float32)
    confidence_count = np.zeros((int(total_frames),), dtype=np.float32)
    for record_idx, record in enumerate(records):
        global_indices = np.asarray(record['global_indices'], dtype=np.int64)
        clip_depth_np = np.asarray(record['depth'], dtype=np.float32)
        clip_confidence_np = np.asarray(record['confidence'], dtype=np.float32)
        if record_idx == 0:
            scale_chunk, shift_chunk = 1.0, 0.0
            overlap_frame_mask = np.zeros((global_indices.size,), dtype=bool)
        else:
            decoded_existing_count = depth_count[global_indices]
            overlap_frame_mask = decoded_existing_count > 0
            if np.any(overlap_frame_mask):
                existing_depth = depth_sum[global_indices[overlap_frame_mask]] / decoded_existing_count[overlap_frame_mask, None, None]
                clip_depth_overlap = clip_depth_np[overlap_frame_mask]
                align_mask = np.isfinite(existing_depth) & np.isfinite(clip_depth_overlap) & (existing_depth > 0) & (clip_depth_overlap > 0)
                scale_chunk, shift_chunk = fitDepthScaleShift(pred_values=clip_depth_overlap[align_mask], target_values=existing_depth[align_mask], align_method=align_method, align_iters=align_iters, align_lr=align_lr)
            else:
                scale_chunk, shift_chunk = 1.0, 0.0
        print(f'[Sliding Align] record={record_idx + 1}/{len(records)} {record["name"]} stride={record["stride"]} overlap_global={global_indices[overlap_frame_mask].tolist()} scale={scale_chunk:.6f} shift={shift_chunk:.6f}')
        clip_depth_np = applyDepthAlignment(clip_depth_np, scale_chunk, shift_chunk, target_max)
        depth_sum[global_indices] = depth_sum[global_indices] + clip_depth_np
        confidence_sum[global_indices] = confidence_sum[global_indices] + clip_confidence_np
        depth_count[global_indices] = depth_count[global_indices] + 1.0
        confidence_count[global_indices] = confidence_count[global_indices] + 1.0
    if np.any(depth_count <= 0):
        raise RuntimeError('Sliding-window depth inference did not cover every output frame.')
    depth_out = depth_sum / np.maximum(depth_count[:, None, None], 1.0)
    confidence_out = confidence_sum / np.maximum(confidence_count[:, None, None], 1.0)
    return torch.from_numpy(depth_out.astype(np.float32)), torch.from_numpy(confidence_out.astype(np.float32))


def fitGlobalDepthChunkScales(
    records: List[Dict[str, object]],
    align_method: str,
    align_iters: int,
    align_lr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jointly optimize one scale and shift for every chunk record, fixing the first record to
    identity.

    The objective uses shared decoded frames between every record pair. To keep memory bounded for
    high-resolution dense depth, each pair contributes a deterministic subset of valid overlap
    pixels.
    """
    num_records = len(records)
    if num_records <= 1:
        return np.ones((num_records,), dtype=np.float32), np.zeros((num_records,), dtype=np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_pair_samples = 500000
    pair_samples = []
    for left_idx in range(num_records):
        left_indices = np.asarray(records[left_idx]['global_indices'], dtype=np.int64)
        left_pos = {int(global_idx): idx for idx, global_idx in enumerate(left_indices.tolist())}
        for right_idx in range(left_idx + 1, num_records):
            right_indices = np.asarray(records[right_idx]['global_indices'], dtype=np.int64)
            common_indices = np.intersect1d(left_indices, right_indices)
            if common_indices.size == 0:
                continue
            left_depth = np.asarray(records[left_idx]['depth'], dtype=np.float32)[[left_pos[int(global_idx)] for global_idx in common_indices.tolist()]]
            right_pos = {int(global_idx): idx for idx, global_idx in enumerate(right_indices.tolist())}
            right_depth = np.asarray(records[right_idx]['depth'], dtype=np.float32)[[right_pos[int(global_idx)] for global_idx in common_indices.tolist()]]
            valid = np.isfinite(left_depth) & np.isfinite(right_depth) & (left_depth > 0) & (right_depth > 0)
            if not np.any(valid):
                continue
            left_values = left_depth[valid].reshape(-1)
            right_values = right_depth[valid].reshape(-1)
            if left_values.size > max_pair_samples:
                sample_indices = np.linspace(0, left_values.size - 1, num=max_pair_samples, dtype=np.int64)
                left_values = left_values[sample_indices]
                right_values = right_values[sample_indices]
            pair_samples.append((left_idx, right_idx, left_values.astype(np.float32), right_values.astype(np.float32)))
    if not pair_samples:
        return np.ones((num_records,), dtype=np.float32), np.zeros((num_records,), dtype=np.float32)

    if align_method == 'l2+lsq':
        num_vars = 2 * (num_records - 1)
        normal_matrix = np.zeros((num_vars, num_vars), dtype=np.float64)
        normal_rhs = np.zeros((num_vars,), dtype=np.float64)
        for left_idx, right_idx, left_values, right_values in pair_samples:
            row_count = int(left_values.size)
            design = np.zeros((row_count, num_vars), dtype=np.float64)
            fixed_value = np.zeros((row_count,), dtype=np.float64)
            if left_idx == 0:
                fixed_value = fixed_value + left_values.astype(np.float64)
            else:
                left_offset = 2 * (left_idx - 1)
                design[:, left_offset] = left_values.astype(np.float64)
                design[:, left_offset + 1] = 1.0
            if right_idx == 0:
                fixed_value = fixed_value - right_values.astype(np.float64)
            else:
                right_offset = 2 * (right_idx - 1)
                design[:, right_offset] = design[:, right_offset] - right_values.astype(np.float64)
                design[:, right_offset + 1] = design[:, right_offset + 1] - 1.0
            rhs = -fixed_value
            normal_matrix = normal_matrix + design.T @ design
            normal_rhs = normal_rhs + design.T @ rhs
        sol, _, _, _ = np.linalg.lstsq(normal_matrix, normal_rhs, rcond=None)
        scales = np.ones((num_records,), dtype=np.float32)
        shifts = np.zeros((num_records,), dtype=np.float32)
        for record_idx in range(1, num_records):
            sol_offset = 2 * (record_idx - 1)
            scales[record_idx] = float(sol[sol_offset])
            shifts[record_idx] = float(sol[sol_offset + 1])
        return scales, shifts

    pair_samples = [(left_idx, right_idx, torch.as_tensor(left_values, dtype=torch.float32, device=device), torch.as_tensor(right_values, dtype=torch.float32, device=device)) for left_idx, right_idx, left_values, right_values in pair_samples]
    scale_params = torch.ones((num_records - 1,), dtype=torch.float32, device=device, requires_grad=True)
    shift_params = torch.zeros((num_records - 1,), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([scale_params, shift_params], lr=align_lr)
    prev_loss = None
    for _ in range(align_iters):
        optimizer.zero_grad()
        losses = []
        for left_idx, right_idx, left_values, right_values in pair_samples:
            left_scale = torch.ones((), dtype=torch.float32, device=device) if left_idx == 0 else scale_params[left_idx - 1]
            left_shift = torch.zeros((), dtype=torch.float32, device=device) if left_idx == 0 else shift_params[left_idx - 1]
            right_scale = torch.ones((), dtype=torch.float32, device=device) if right_idx == 0 else scale_params[right_idx - 1]
            right_shift = torch.zeros((), dtype=torch.float32, device=device) if right_idx == 0 else shift_params[right_idx - 1]
            residual = (left_scale * left_values + left_shift) - (right_scale * right_values + right_shift)
            losses.append(torch.mean(torch.abs(residual)))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if prev_loss is not None and abs(prev_loss - loss_value) < 1e-7:
            break
        prev_loss = loss_value
    scales = np.ones((num_records,), dtype=np.float32)
    shifts = np.zeros((num_records,), dtype=np.float32)
    scales[1:] = scale_params.detach().cpu().numpy().astype(np.float32)
    shifts[1:] = shift_params.detach().cpu().numpy().astype(np.float32)
    return scales, shifts


def mergeGlobalDepthChunkRecords(
    records: List[Dict[str, object]],
    total_frames: int,
    output_resolution: Tuple[int, int],
    align_method: str,
    align_iters: int,
    align_lr: float,
    target_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge chunk records after one global scale-shift optimization over all shared decoded frames.
    """
    out_h, out_w = int(output_resolution[0]), int(output_resolution[1])
    depth_sum = np.zeros((int(total_frames), out_h, out_w), dtype=np.float32)
    confidence_sum = np.zeros((int(total_frames), out_h, out_w), dtype=np.float32)
    depth_count = np.zeros((int(total_frames),), dtype=np.float32)
    confidence_count = np.zeros((int(total_frames),), dtype=np.float32)
    scales, shifts = fitGlobalDepthChunkScales(records, align_method=align_method, align_iters=align_iters, align_lr=align_lr)
    for record_idx, record in enumerate(records):
        global_indices = np.asarray(record['global_indices'], dtype=np.int64)
        clip_depth_np = applyDepthAlignment(np.asarray(record['depth'], dtype=np.float32), float(scales[record_idx]), float(shifts[record_idx]), target_max)
        clip_confidence_np = np.asarray(record['confidence'], dtype=np.float32)
        print(f'[Global Align] record={record_idx + 1}/{len(records)} {record["name"]} stride={record["stride"]} scale={float(scales[record_idx]):.6f} shift={float(shifts[record_idx]):.6f}')
        depth_sum[global_indices] = depth_sum[global_indices] + clip_depth_np
        confidence_sum[global_indices] = confidence_sum[global_indices] + clip_confidence_np
        depth_count[global_indices] = depth_count[global_indices] + 1.0
        confidence_count[global_indices] = confidence_count[global_indices] + 1.0
    if np.any(depth_count <= 0):
        raise RuntimeError('Global sliding-window depth inference did not cover every output frame.')
    depth_out = depth_sum / np.maximum(depth_count[:, None, None], 1.0)
    confidence_out = confidence_sum / np.maximum(confidence_count[:, None, None], 1.0)
    return torch.from_numpy(depth_out.astype(np.float32)), torch.from_numpy(confidence_out.astype(np.float32))


def run_depth_sliding_window_inference(
    model,
    video_tchw: torch.Tensor,
    encode_video_tchw: torch.Tensor,
    enc_img_size: Tuple[int, int],
    output_resolution: Tuple[int, int],
    clip_ranges: List[Tuple[int, int]],
    infer_clip_num_frames: int,
    overlap_align_frames: int,
    align_method: str,
    align_iters: int,
    align_lr: float,
    sequential_chunk_align: str,
    global_chunk_align: bool,
    max_stride: int,
    target_max: float,
    amp_infer: bool = False,
    progress_bar: Optional[tqdm] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run sliding-window depth inference, optional extra temporal stride passes, and the selected
    chunk merge strategy.
    """
    dense_target_direction = 'forward' if bool(global_chunk_align) else str(sequential_chunk_align)
    dense_chunk_targets = buildDenseDepthChunkTargets(total_frames=int(video_tchw.shape[0]), clip_ranges=clip_ranges, overlap_align_frames=int(overlap_align_frames), sequential_chunk_align=dense_target_direction)
    dense_records, stride_records = collectDepthChunkRecords(
        model=model,
        video_tchw=video_tchw,
        encode_video_tchw=encode_video_tchw,
        enc_img_size=enc_img_size,
        output_resolution=output_resolution,
        dense_chunk_targets=dense_chunk_targets,
        infer_clip_num_frames=infer_clip_num_frames,
        max_stride=max_stride,
        amp_infer=amp_infer,
        progress_bar=progress_bar,
    )
    if bool(global_chunk_align):
        return mergeGlobalDepthChunkRecords(
            records=dense_records + stride_records,
            total_frames=int(video_tchw.shape[0]),
            output_resolution=output_resolution,
            align_method=align_method,
            align_iters=align_iters,
            align_lr=align_lr,
            target_max=target_max,
        )
    if sequential_chunk_align == 'forward':
        merge_records = dense_records + stride_records
    else:
        merge_records = list(reversed(stride_records)) + list(reversed(dense_records))
    return mergeSequentialDepthChunkRecords(
        records=merge_records,
        total_frames=int(video_tchw.shape[0]),
        output_resolution=output_resolution,
        align_method=align_method,
        align_iters=align_iters,
        align_lr=align_lr,
        target_max=target_max,
    )


def try_load_cached_depth(
    depth_dir: Path,
    num_frames: int,
) -> Optional[torch.Tensor]:
    """
    Try to load one scene's cached depth npy files.
    """
    depth_arrays = []
    for i in range(num_frames):
        npy_path = depth_dir / f'depth_{i:05d}.npy'
        if not npy_path.exists():
            return None
        try:
            arr = np.load(npy_path)
        except Exception:
            return None
        if arr.ndim != 2:
            return None
        depth_arrays.append(arr.astype(np.float32))

    if len(depth_arrays) != num_frames:
        return None

    return torch.from_numpy(np.stack(depth_arrays, axis=0))


def try_load_cached_confidence(
    confidence_dir: Path,
    num_frames: int,
) -> Optional[torch.Tensor]:
    """
    Try to load one scene's cached confidence npy files.
    """
    confidence_arrays = []
    for i in range(num_frames):
        npy_path = confidence_dir / f'confidence_{i:05d}.npy'
        if not npy_path.exists():
            return None
        try:
            arr = np.load(npy_path)
        except Exception:
            return None
        if arr.ndim != 2:
            return None
        confidence_arrays.append(arr.astype(np.float32))

    if len(confidence_arrays) != num_frames:
        return None

    return torch.from_numpy(np.stack(confidence_arrays, axis=0))


def build_clip_ranges(
    total_frames: int,
    model_temp_size: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Build benchmark clip ranges and the kept frame span for each clip.
    """
    clip_ranges = []
    if total_frames <= model_temp_size:
        clip_ranges.append((0, total_frames, 0, total_frames))
        return clip_ranges

    full_frames = (total_frames // model_temp_size) * model_temp_size
    for clip_start in range(0, full_frames, model_temp_size):
        clip_end = clip_start + model_temp_size
        clip_ranges.append((clip_start, clip_end, clip_start, clip_end))

    if full_frames < total_frames:
        clip_start = total_frames - model_temp_size
        clip_end = total_frames
        keep_start = full_frames
        clip_ranges.append((clip_start, clip_end, keep_start, clip_end))

    return clip_ranges


def build_sliding_depth_clip_ranges(
    total_frames: int,
    clip_frames: int,
    requested_stride: int,
) -> List[Tuple[int, int]]:
    """
    Build `[start, end)` frame windows for overlapping depth inference.

    `requested_stride` is derived as `clip_frames - requested_overlap_frames` by the caller. It is
    used while it keeps the next full clip inside the scene. If the next regular step would overrun
    the scene, the final window is shifted backward so `end == total_frames`; this can make the
    actual last stride smaller and the actual overlap larger. Short scenes return one shorter window
    and are padded only inside inference.
    """
    total_frames = int(total_frames)
    clip_frames = int(clip_frames)
    requested_stride = int(requested_stride)
    if total_frames <= clip_frames:
        return [(0, total_frames)]

    starts = [0]
    while starts[-1] + clip_frames < total_frames:
        raw_next_start = starts[-1] + requested_stride
        next_start = raw_next_start
        if next_start + clip_frames > total_frames:
            next_start = total_frames - clip_frames
        if next_start <= starts[-1]:
            break
        starts.append(next_start)

    return [(int(start), int(start + clip_frames)) for start in starts]


def print_depth_clip_plan(
    scene_name: str,
    clip_ranges: List[Tuple[int, int]],
    requested_stride: Optional[int],
    requested_overlap_frames: Optional[int],
) -> None:
    """
    Print the exact sliding-window schedule before running model inference.

    This makes tail adjustments visible in job logs, including requested overlap, nominal stride,
    actual stride, and actual overlap span for every non-first window. A final window whose stride
    is smaller than the requested stride is marked with `adjusted_tail_stride`; alignment still uses
    at most `requested_overlap_frames` frames from the tail of the actual overlap.
    """
    if requested_stride is None:
        return
    print(f'[Sliding Window] scene={scene_name} requested_overlap={requested_overlap_frames} requested_stride={requested_stride} windows={len(clip_ranges)}')
    for idx, (clip_start, clip_end) in enumerate(clip_ranges):
        if idx == 0:
            print(f'[Sliding Window] window {idx + 1}/{len(clip_ranges)} infer[{clip_start}:{clip_end})')
            continue
        prev_start, prev_end = clip_ranges[idx - 1]
        actual_stride = clip_start - prev_start
        overlap_start = max(prev_start, clip_start)
        overlap_end = min(prev_end, clip_end)
        actual_overlap = max(0, overlap_end - overlap_start)
        adjusted = ' adjusted_tail_stride' if actual_stride != int(requested_stride) else ''
        print(
            f'[Sliding Window] window {idx + 1}/{len(clip_ranges)} infer[{clip_start}:{clip_end}) '
            f'actual_stride={actual_stride} actual_overlap={actual_overlap} '
            f'overlap[{overlap_start}:{overlap_end}) align_tail_overlap={requested_overlap_frames}'
            f'{adjusted}'
        )


def write_benchmark_summary(
    benchmark_root_dir: Path,
    dataset_id: str,
    split_name: str,
    seed: int,
    start_frame: int,
    infer_total_num_frames: int,
    infer_chunk_num_frames: int,
    args,
) -> None:
    """
    Rebuild the root-level benchmark summary from all finished scene metrics.
    """
    scene_entries = []
    for scene_dir in sorted(benchmark_root_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        metrics_path = scene_dir / 'depth_eval_metrics.json'
        if not metrics_path.exists():
            continue
        with open(metrics_path, 'r', encoding='utf-8') as f:
            scene_metrics = json.load(f)
        if scene_metrics.get('metric_units', {}).get('delta1') != DELTA1_UNIT:
            continue
        if scene_metrics.get('align_mask') != args.align_mask:
            continue
        scene_entries.append(
            {
                'scene': scene_dir.name,
                'num_compared_frames': int(scene_metrics.get('num_compared_frames', 0)),
                'valid_pixels': int(scene_metrics.get('valid_pixels', 0)),
                'target_valid_pixels': int(scene_metrics.get('target_valid_pixels', scene_metrics.get('valid_pixels', 0))),
                'prediction_valid_pixels': int(scene_metrics.get('prediction_valid_pixels', 0)),
                'valid_pixel_ratio': float(scene_metrics.get('valid_pixel_ratio', 0.0)),
                'absrel_s_mean': float(scene_metrics.get('absrel_s_mean', 0.0)),
                'delta1_s_mean': float(scene_metrics.get('delta1_s_mean', 0.0)),
                'absrel_ss_mean': float(scene_metrics.get('absrel_ss_mean', 0.0)),
                'delta1_ss_mean': float(scene_metrics.get('delta1_ss_mean', 0.0)),
                'metrics_path': str(Path(scene_dir.name) / 'depth_eval_metrics.json'),
            }
        )

    weights = np.asarray([max(int(item['valid_pixels']), 0) for item in scene_entries], dtype=np.float64)
    has_weights = bool(weights.size > 0 and np.sum(weights) > 0)

    def average_scene_metric(
        metric_name: str,
    ) -> float:
        values = np.asarray([float(item[metric_name]) for item in scene_entries], dtype=np.float64)
        if not values.size:
            return 0.0
        if has_weights:
            return float(np.average(values, weights=weights))
        return float(np.mean(values))

    summary = {
        'align_mask': str(args.align_mask),
        'metric_mask': str(args.align_mask),
        'prediction_post_clip': 'max_depth_only' if dataset_id == 'SINTEL' else 'none' if dataset_id in ('BONN', 'KITTI') else 'positive_and_max_depth',
        'metric_units': {'delta1': DELTA1_UNIT},
        'dataset_id': dataset_id,
        'absrel_s_mean': average_scene_metric('absrel_s_mean'),
        'delta1_s_mean': average_scene_metric('delta1_s_mean'),
        'absrel_ss_mean': average_scene_metric('absrel_ss_mean'),
        'delta1_ss_mean': average_scene_metric('delta1_ss_mean'),
        'valid_pixels': int(np.sum(weights)) if weights.size else 0,
        'target_valid_pixels': int(sum(item['target_valid_pixels'] for item in scene_entries)),
        'prediction_valid_pixels': int(sum(item['prediction_valid_pixels'] for item in scene_entries)),
        'split': split_name,
        'seed': int(seed),
        'start_frame': int(start_frame),
        'infer_total_num_frames': int(infer_total_num_frames),
        'num_scenes': len(scene_entries),
        'img_h': int(args.img_h),
        'img_w': int(args.img_w),
        'proc_h': int(args.proc_h),
        'proc_w': int(args.proc_w),
        'center_crop': bool(args.center_crop),
        'train_num_frames': None if args.train_num_frames is None else int(args.train_num_frames),
        'infer_chunk_num_frames': int(infer_chunk_num_frames),
        'time_embed_resize_mode': 'linear',
        'align_method': str(args.align_method),
        'align_iters': int(args.align_iters),
        'align_lr': float(args.align_lr),
        'global_chunk_align': bool(args.global_chunk_align),
        'max_stride': int(args.max_stride),
        'scenes': scene_entries,
    }
    summary['valid_pixel_ratio'] = float(summary['valid_pixels'] / summary['target_valid_pixels']) if summary['target_valid_pixels'] > 0 else 0.0
    if not bool(args.global_chunk_align):
        summary['sequential_chunk_align'] = str(args.sequential_chunk_align)

    write_json_atomic(benchmark_root_dir / 'depth_bench_metrics-ALL.json', summary)


def write_json_atomic(
    json_path: Path,
    payload: Dict[str, object],
) -> None:
    """
    Atomically write one JSON file so readers never observe a partial file.
    """
    tmp_path = json_path.with_name(f'{json_path.name}.tmp.{os.getpid()}')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, json_path)


def get_scene_placeholder_path(
    output_root_dir: Path,
    scene_name: str,
    placeholder_kind: str,
) -> Path:
    """
    Return the scene placeholder directory path under the benchmark root.
    """
    return output_root_dir / SCENE_PLACEHOLDER_ROOT_DIR_NAME / placeholder_kind / scene_name


def try_create_scene_placeholder(
    output_root_dir: Path,
    scene_name: str,
    placeholder_kind: str,
) -> Tuple[bool, Path]:
    """
    Try to create one scene placeholder directory.
    """
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
    """
    Remove one scene placeholder directory if it still exists.
    """
    try:
        placeholder_path.rmdir()
    except OSError:
        pass


def scene_prediction_complete(
    scene_output_dir: Path,
    num_frames: int,
) -> bool:
    """
    Check whether one scene already has a complete raw prediction cache.
    """
    depth_dir = scene_output_dir / 'depth'
    confidence_dir = scene_output_dir / 'confidence'
    for i in range(num_frames):
        depth_path = depth_dir / f'depth_{i:05d}.npy'
        confidence_path = confidence_dir / f'confidence_{i:05d}.npy'
        if not depth_path.exists() or not confidence_path.exists():
            return False
        try:
            depth_arr = np.load(depth_path)
            confidence_arr = np.load(confidence_path)
        except Exception:
            return False
        if depth_arr.ndim != 2 or confidence_arr.ndim != 2:
            return False
    return True


def resolve_scene_inputs(
    data_root: str,
    split_name: str,
    scene_name: str,
) -> Tuple[str, str, str]:
    """
    Resolve `(video_path, target_folder, dataset_type)` for one scene.
    """
    if 'sintel' in str(data_root).lower():
        data_root_path = Path(data_root)
        if data_root_path.name == 'final':
            return (str(data_root_path / scene_name), str(data_root_path.parent / 'depth' / scene_name), 'sintel')
        scene_root = data_root_path / 'training'
        return (str(scene_root / 'final' / scene_name), str(scene_root / 'depth' / scene_name), 'sintel')
    if 'bonn' in str(data_root).lower():
        scene_dir_name = scene_name if str(scene_name).startswith('rgbd_bonn_') else f'rgbd_bonn_{scene_name}'
        scene_root = Path(data_root) / scene_dir_name
        return str(scene_root / 'rgb_110'), str(scene_root / 'depth_110'), 'bonn'
    if 'kitti' in str(data_root).lower():
        scene_root = Path(data_root) / split_name / scene_name
        video_path = Path(data_root) / scene_name[:10] / scene_name / 'image_02' / 'data'
        return str(video_path), str(scene_root / 'proj_depth' / 'groundtruth' / 'image_02'), 'kitti'
    scene_root = Path(data_root) / split_name / scene_name
    if 'dynamic_replica' in str(data_root).lower():
        return str(scene_root / 'images'), str(scene_root / 'depths'), 'dr'
    return str(Path(data_root) / split_name / f'{scene_name}.mp4'), str(scene_root / 'depths'), 'po'


def colorize_depth_map(
    depth_array: np.ndarray,
) -> np.ndarray:
    depth_float = depth_array.astype(np.float32)
    valid_mask = np.isfinite(depth_float) & (depth_float > 0)
    if np.any(valid_mask):
        min_depth = np.min(depth_float[valid_mask])
        max_depth = np.max(depth_float[valid_mask])
        if max_depth > min_depth:
            depth_norm = (depth_float - min_depth) / (max_depth - min_depth)
        else:
            depth_norm = np.zeros_like(depth_float)
    else:
        depth_norm = np.zeros_like(depth_float)
    depth_norm = np.clip(depth_norm, 0, 1)
    colored = cm.magma(depth_norm)[..., :3]
    colored[~valid_mask] = 0
    return (colored * 255).astype(np.uint8)


def colorize_error_map(
    error_array: np.ndarray,
    valid_mask: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    if not np.any(valid_mask):
        return np.zeros((*error_array.shape, 3), dtype=np.uint8)
    if vmax > vmin:
        error_norm = (error_array - vmin) / (vmax - vmin)
    else:
        error_norm = np.zeros_like(error_array)
    error_norm = np.clip(error_norm, 0, 1)
    colored = cm.jet(error_norm)[..., :3]
    colored[~valid_mask] = 0
    return (colored * 255).astype(np.uint8)


def colorize_confidence_map(
    confidence_array: np.ndarray,
    use_percentile: bool = False,
    vis_low: float = 0.0,
) -> Tuple[np.ndarray, float, float]:
    """Blue-low to green-high confidence colormap visualization."""
    conf = confidence_array.astype(np.float32)
    valid_mask = np.isfinite(conf)

    p_low = 0.0
    p_high = 1.0
    if use_percentile:
        conf_vis = np.zeros_like(conf, dtype=np.float32)
        valid_values = conf[valid_mask]
        if valid_values.size > 0:
            p_low = float(np.percentile(valid_values, 1.0))
            p_high = float(np.percentile(valid_values, 99.0))
            if np.isfinite(p_low) and np.isfinite(p_high) and p_high > p_low:
                conf_vis = (conf - p_low) / (p_high - p_low)
            elif np.isfinite(p_low):
                conf_vis = conf - p_low
    else:
        # Absolute [0, 1] visualization:
        conf_vis = conf.copy()

    # Old absolute [0, 1] visualization kept here for reference:
    # conf_vis = conf.copy()
    # conf_vis = np.clip(conf_vis, 0.0, 1.0)
    vis_low = float(np.clip(vis_low, 0.0, 1.0))
    conf_vis = (conf_vis - vis_low) / max(1.0 - vis_low, 1e-6)
    conf_vis = np.clip(conf_vis, 0.0, 1.0)
    colored = cm.viridis(conf_vis)[..., :3]  # blue -> green
    colored[~valid_mask] = 0
    return (colored * 255).astype(np.uint8), p_low, p_high


def add_right_vertical_legend(
    image: np.ndarray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    label: str,
    top_text: Optional[str] = None,
    bot_text: Optional[str] = None,
) -> np.ndarray:
    """Overlay a right-side vertical color legend on a visualization image."""
    out = image.copy()
    h, w = out.shape[:2]

    bar_w = min(18, int(w * 0.06))
    pad = max(4, int(w * 0.01))
    bar_x0 = max(0, w - bar_w - pad)
    bar_x1 = min(w, w - pad)
    bar_y0 = max(8, int(h * 0.08))
    bar_y1 = min(h - 8, int(h * 0.92))
    bar_h = max(1, bar_y1 - bar_y0)

    # translucent dark background to improve readability
    overlay = out.copy()
    cv2.rectangle(overlay, (bar_x0 - 18, bar_y0 - 14), (w - 1, bar_y1 + 14), (0, 0, 0), thickness=-1)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    gradient = np.linspace(1.0, 0.0, bar_h, dtype=np.float32).reshape(bar_h, 1)
    gradient = np.repeat(gradient, max(1, bar_x1 - bar_x0), axis=1)
    colored_bar = (matplotlib.colormaps.get_cmap(cmap_name)(gradient)[..., :3] * 255).astype(np.uint8)
    out[bar_y0:bar_y1, bar_x0:bar_x1] = colored_bar

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1
    text_color = (255, 255, 255)

    # min/max ticks
    top_text_val = f'{vmax:.3f}' if top_text is None else top_text
    bot_text_val = f'{vmin:.3f}' if bot_text is None else bot_text
    cv2.putText(out, top_text_val, (bar_x0 - 16, max(10, bar_y0 - 4)), font, font_scale, text_color, thickness, cv2.LINE_AA)
    cv2.putText(out, bot_text_val, (bar_x0 - 16, min(h - 4, bar_y1 + 12)), font, font_scale, text_color, thickness, cv2.LINE_AA)

    label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    label_x = max(2, w - label_size[0] - 2)
    label_y = max(10, bar_y0 - 18)
    cv2.putText(out, label, (label_x, label_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return out


def draw_metrics_text_on_error_map(
    err_vis: np.ndarray,
    global_frame: int,
    valid_count: int,
    absrel_s: float,
    absrel_ss: float,
) -> np.ndarray:
    """Draw compact metric text directly onto depth error map."""
    overlay = err_vis.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    line1 = f'Frame {global_frame} | Valid: {valid_count}'
    line2 = f'AbsRel(S): {absrel_s:.4f} | AbsRel(SS): {absrel_ss:.4f}'

    for text, y in ((line1, 20), (line2, 40)):
        cv2.putText(overlay, text, (10, y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(overlay, text, (10, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return overlay


def colorize_validity_difference_map(
    pred_valid_mask: np.ndarray,
    target_valid_mask: np.ndarray,
) -> np.ndarray:
    """Create RGB map comparing prediction/target validity.

    Color convention:
    - red   : pred invalid only
    - blue  : target invalid only
    - black : both invalid
    - white : both valid
    """
    if pred_valid_mask.shape != target_valid_mask.shape:
        raise ValueError(f"Mask shape mismatch: pred={pred_valid_mask.shape}, target={target_valid_mask.shape}")

    h, w = pred_valid_mask.shape
    out = np.full((h, w, 3), 0, dtype=np.uint8)  # default black: both invalid
    both_valid = pred_valid_mask & target_valid_mask
    pred_invalid_only = (~pred_valid_mask) & target_valid_mask
    target_invalid_only = pred_valid_mask & (~target_valid_mask)

    out[both_valid] = np.array([255, 255, 255], dtype=np.uint8)  # white
    out[pred_invalid_only] = np.array([255, 0, 0], dtype=np.uint8)  # red
    out[target_invalid_only] = np.array([0, 0, 255], dtype=np.uint8)  # blue
    return out


def padVideoFrameToEvenShape(
    frame: np.ndarray,
) -> np.ndarray:
    """
    Pad only MP4 visualization frames to the even dimensions required by libx264 yuv420p.
    """
    pad_bottom = frame.shape[0] % 2
    pad_right = frame.shape[1] % 2
    if pad_bottom == 0 and pad_right == 0:
        return frame
    return cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)


def process_target_raw(
    target_img: np.ndarray,
    orig_shape: Tuple[int, int],
    crop_shape: Tuple[int, int],
    crop_offsets: Tuple[int, int],
) -> np.ndarray:
    """
    Keep GT on its native grid while selecting the RGB crop field of view.
    """
    oh, ow = orig_shape
    ch, cw = crop_shape
    y0, x0 = crop_offsets
    if target_img.ndim != 2:
        raise ValueError(f'Expected 2D target depth, got shape {target_img.shape}')
    if (y0, x0, ch, cw) == (0, 0, oh, ow):
        return target_img.astype(np.float32)

    target_h, target_w = target_img.shape
    target_y0 = int(round(y0 * target_h / oh))
    target_y1 = int(round((y0 + ch) * target_h / oh))
    target_x0 = int(round(x0 * target_w / ow))
    target_x1 = int(round((x0 + cw) * target_w / ow))
    target_y0 = min(max(target_y0, 0), target_h - 1)
    target_y1 = min(max(target_y1, target_y0 + 1), target_h)
    target_x0 = min(max(target_x0, 0), target_w - 1)
    target_x1 = min(max(target_x1, target_x0 + 1), target_w)
    return target_img[target_y0:target_y1, target_x0:target_x1].astype(np.float32)


def resizePredictionToTargetShape(
    pred_raw: np.ndarray,
    confidence_raw: np.ndarray,
    target_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resize prediction data to the native GT grid, then determine prediction validity.
    """
    if pred_raw.ndim != 2 or confidence_raw.shape != pred_raw.shape:
        raise ValueError(f'Expected matching 2D depth/confidence, got {pred_raw.shape} and {confidence_raw.shape}')
    pred_resized = np.asarray(pred_raw, dtype=np.float32)
    confidence_safe = np.nan_to_num(confidence_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if pred_raw.shape != target_shape:
        target_h, target_w = target_shape
        pred_resized = cv2.resize(pred_resized, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        confidence_safe = cv2.resize(confidence_safe, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    prediction_valid_mask = np.isfinite(pred_resized) & (pred_resized > DEPTH_EPSILON)
    confidence_safe[prediction_valid_mask == 0] = 0.0
    return pred_resized, confidence_safe, prediction_valid_mask


def load_target_image_for_frame(
    global_idx: int,
    target_folder: Path,
    dataset_type: str,
    target_depth_paths: Optional[List[Path]],
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Load one GT depth frame for per-frame evaluation / visualization.
    """
    if dataset_type == 'po':
        target_path = target_folder / f'depth_{global_idx:05d}.png'
        if not target_path.exists():
            return None, None
        target_img = cv2.imread(str(target_path), cv2.IMREAD_ANYDEPTH)
        if target_img is None:
            return None, None
        return str(target_path), target_img

    if dataset_type == 'dr':
        if target_depth_paths is None or global_idx >= len(target_depth_paths):
            return None, None
        target_path = target_depth_paths[global_idx]
        return str(target_path), loadFloat16DepthPng(target_path)

    if dataset_type == 'sintel':
        if target_depth_paths is None or global_idx >= len(target_depth_paths):
            return None, None
        target_path = target_depth_paths[global_idx]
        return str(target_path), loadSintelDepthDpt(target_path)

    if dataset_type == 'bonn':
        if target_depth_paths is None or global_idx >= len(target_depth_paths):
            return None, None
        target_path = target_depth_paths[global_idx]
        return str(target_path), loadBonnDepthPng(target_path)

    if dataset_type == 'kitti':
        if target_depth_paths is None or global_idx >= len(target_depth_paths):
            return None, None
        target_path = target_depth_paths[global_idx]
        return str(target_path), loadKittiDepthPng(target_path)

    raise ValueError(f'Unsupported dataset_type: {dataset_type}')


def list_target_depth_paths(
    target_folder: Path,
    dataset_type: str,
) -> Optional[List[Path]]:
    """
    Return sorted per-frame GT depth files for datasets that need path-list indexing.

    PointOdyssey uses deterministic `depth_%05d.png` names and can resolve each path directly from
    the frame index, so it returns `None`. Dynamic Replica, Sintel, Bonn, and KITTI use sorted
    native filenames and pass the resulting list into `load_target_image_for_frame()`.
    """
    if dataset_type == 'dr':
        paths = sorted(target_folder.glob('*.geometric.png'))
        if not paths:
            raise RuntimeError(f'No Dynamic Replica depth files found under: {target_folder}')
        return paths
    if dataset_type == 'sintel':
        paths = sorted(target_folder.glob('*.dpt'))
        if not paths:
            raise RuntimeError(f'No Sintel depth files found under: {target_folder}')
        return paths
    if dataset_type == 'bonn':
        paths = sorted(target_folder.glob('*.png'))
        if not paths:
            raise RuntimeError(f'No Bonn depth files found under: {target_folder}')
        return paths
    if dataset_type == 'kitti':
        paths = sorted(target_folder.glob('*.png'))
        if not paths:
            raise RuntimeError(f'No KITTI depth files found under: {target_folder}')
        return paths
    return None


def listKittiPairedFramePaths(
    video_path: Path,
    target_folder: Path,
) -> Tuple[List[Path], List[Path]]:
    """
    Pair KITTI raw RGB and projected GT depth frames by their exact filenames.
    """
    target_depth_paths = list_target_depth_paths(target_folder, 'kitti')
    image_paths = [video_path / target_path.name for target_path in target_depth_paths]
    missing_image_paths = [image_path for image_path in image_paths if not image_path.is_file()]
    if missing_image_paths:
        raise RuntimeError(f'Missing KITTI RGB frame for GT: {missing_image_paths[0]}')
    return image_paths, target_depth_paths


def fitDepthScale(
    pred_values: np.ndarray,
    target_values: np.ndarray,
    filter_prediction_valid: bool = True,
) -> float:
    """
    Fit a scale-only alignment from prediction depth to target depth using median depth ratio.

    This is the benchmark's `S` alignment path. Prediction validity is filtered for `gt+pred`
    and left untouched for `gt`.
    """
    pred_flat = np.asarray(pred_values, dtype=np.float64).reshape(-1)
    target_flat = np.asarray(target_values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(target_flat) & (target_flat > 0)
    if filter_prediction_valid:
        valid = valid & np.isfinite(pred_flat) & (pred_flat > DEPTH_EPSILON)
    if not np.any(valid):
        return 1.0
    pred_median = float(np.median(pred_flat[valid]))
    if abs(pred_median) <= 1e-12:
        return 1.0
    return float(np.median(target_flat[valid]) / pred_median)


def fitDepthScaleShift(
    pred_values: np.ndarray,
    target_values: np.ndarray,
    align_method: str,
    align_iters: int,
    align_lr: float,
    filter_prediction_valid: bool = True,
) -> Tuple[float, float]:
    """
    Fit a scale-shift alignment `target ~= scale * pred + shift` for chunk merge or final `SS`
    metrics.

    `l2+lsq` uses a closed-form least-squares solve, matching the older local benchmark behavior.
    `l1+adam` initializes scale from the median depth ratio and optimizes an L1 residual with Adam.
    Prediction validity is filtered only when `filter_prediction_valid` is enabled.
    """
    pred_flat = np.asarray(pred_values, dtype=np.float32).reshape(-1)
    target_flat = np.asarray(target_values, dtype=np.float32).reshape(-1)
    valid = np.isfinite(target_flat) & (target_flat > 0)
    if filter_prediction_valid:
        valid = valid & np.isfinite(pred_flat) & (pred_flat > DEPTH_EPSILON)
    if not np.any(valid):
        return 1.0, 0.0
    pred_valid = pred_flat[valid]
    target_valid = target_flat[valid]

    if align_method == 'l2+lsq':
        a = np.stack([pred_valid.astype(np.float64), np.ones_like(pred_valid, dtype=np.float64)], axis=1)
        sol, _, _, _ = np.linalg.lstsq(a, target_valid.astype(np.float64), rcond=None)
        return float(sol[0]), float(sol[1])

    if align_method != 'l1+adam':
        raise ValueError(f'Unsupported align_method: {align_method}')

    pred_median = float(np.median(pred_valid))
    target_median = float(np.median(target_valid))
    s_init = target_median / pred_median if abs(pred_median) > 1e-12 else 1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_t = torch.as_tensor(pred_valid, device=device, dtype=torch.float32)
    target_t = torch.as_tensor(target_valid, device=device, dtype=torch.float32)
    scale = torch.tensor([float(s_init)], requires_grad=True, device=device, dtype=torch.float32)
    shift = torch.tensor([0.0], requires_grad=True, device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam([scale, shift], lr=align_lr)
    prev_loss = None
    for _ in range(align_iters):
        optimizer.zero_grad()
        loss = torch.sum(torch.abs(scale * pred_t + shift - target_t))
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if prev_loss is not None and abs(prev_loss - loss_value) < 1e-6:
            break
        prev_loss = loss_value
    return float(scale.detach().cpu()[0]), float(shift.detach().cpu()[0])


def applyDepthAlignment(
    depth: np.ndarray,
    scale: float,
    shift: float,
    target_max: float,
) -> np.ndarray:
    """
    Apply a fitted scale-shift transform to a depth tensor and sanitize it for metric use.

    The returned depth is finite float32, strictly positive, and optionally clipped by `target_max`,
    which keeps downstream AbsRel and delta computations from seeing invalid values.
    """
    aligned = np.asarray(depth, dtype=np.float32) * float(scale) + float(shift)
    aligned = np.nan_to_num(aligned, nan=DEPTH_EPSILON, posinf=target_max, neginf=DEPTH_EPSILON)
    if np.isfinite(target_max):
        aligned = np.clip(aligned, a_min=DEPTH_EPSILON, a_max=float(target_max))
    else:
        aligned = np.clip(aligned, a_min=DEPTH_EPSILON, a_max=None)
    return aligned.astype(np.float32)


def applyDepthMetricAlignment(
    depth: np.ndarray,
    scale: float,
    shift: float,
    dataset_type: str,
    target_max: float,
) -> np.ndarray:
    """
    Apply final metric alignment and dataset-specific depth clipping.
    """
    aligned = np.asarray(depth, dtype=np.float32) * float(scale) + float(shift)
    if dataset_type == 'sintel':
        aligned = np.minimum(aligned, float(target_max))
        return aligned.astype(np.float32)
    if dataset_type in ('bonn', 'kitti'):
        return aligned.astype(np.float32)
    return applyDepthAlignment(depth, scale, shift, target_max)


def computeDepthMetrics(
    pred_aligned: np.ndarray,
    target: np.ndarray,
    metric_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Compute depth metrics on already-aligned prediction and target tensors.

    The function expects `pred_aligned` to have already gone through either the scene-global `S` or
    `SS` alignment. It reports AbsRel, SqRel, RMSE, log RMSE, and MonST3R/KITTI-style threshold
    accuracies `delta1/2/3`, where `delta1` is `delta < 1.25`.
    """
    pred_metric = np.asarray(pred_aligned, dtype=np.float64)[metric_mask]
    target_metric = np.asarray(target, dtype=np.float64)[metric_mask]
    valid_pixels = int(pred_metric.size)
    if valid_pixels == 0:
        return {
            'abs_rel': 0.0,
            'sq_rel': 0.0,
            'rmse': 0.0,
            'log_rmse': 0.0,
            'delta1': 0.0,
            'delta2': 0.0,
            'delta3': 0.0,
            'valid_pixels': 0,
        }
    eps = 1e-5
    diff = pred_metric - target_metric
    pred_metric_positive = np.clip(pred_metric, a_min=eps, a_max=None)
    target_metric_positive = np.clip(target_metric, a_min=eps, a_max=None)
    ratio = np.maximum(pred_metric_positive / target_metric_positive, target_metric_positive / pred_metric_positive)
    return {
        'abs_rel': float(np.mean(np.abs(diff) / target_metric)),
        'sq_rel': float(np.mean((diff**2) / target_metric)),
        'rmse': float(np.sqrt(np.mean(diff**2))),
        'log_rmse': float(np.sqrt(np.mean((np.log(pred_metric_positive) - np.log(target_metric_positive)) ** 2))),
        'delta1': float(100.0 * np.mean(ratio < 1.25)),
        'delta2': float(np.mean(ratio < 1.25**2)),
        'delta3': float(np.mean(ratio < 1.25**3)),
        'valid_pixels': valid_pixels,
    }


def compute_absrel_s_ss(
    pred: np.ndarray,
    target: np.ndarray,
    metric_mask: np.ndarray,
    align_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Compute AbsRel-S, AbsRel-SS and SS aligned absolute error map."""
    if align_mask is None:
        align_mask = metric_mask

    p_metric = pred[metric_mask]
    t_metric = target[metric_mask]
    if p_metric.size == 0:
        return 0.0, 0.0, np.zeros_like(pred, dtype=np.float32)

    p_align = pred[align_mask]
    t_align = target[align_mask]

    s_s = 1.0
    s_ss = 1.0
    t_ss = 0.0
    if p_align.size > 0:
        p_align = p_align.astype(np.float64)
        t_align = t_align.astype(np.float64)

        denom = np.sum(p_align * p_align)
        s_s = float(np.sum(p_align * t_align) / denom) if denom > 1e-12 else 1.0

        a = np.stack([p_align, np.ones_like(p_align)], axis=1)
        sol, _, _, _ = np.linalg.lstsq(a, t_align, rcond=None)
        s_ss, t_ss = float(sol[0]), float(sol[1])

    p_metric = p_metric.astype(np.float64)
    t_metric = t_metric.astype(np.float64)
    eps = DEPTH_EPSILON
    p_s = np.clip(p_metric * s_s, a_min=eps, a_max=None)
    p_ss = np.clip(p_metric * s_ss + t_ss, a_min=eps, a_max=None)

    absrel_s = float(np.mean(np.abs(p_s - t_metric) / np.maximum(t_metric, eps)))
    absrel_ss = float(np.mean(np.abs(p_ss - t_metric) / np.maximum(t_metric, eps)))

    err_map = np.zeros_like(pred, dtype=np.float32)
    err_map[metric_mask] = np.abs(p_ss - t_metric)
    return absrel_s, absrel_ss, err_map


def build_frame_eval_record(
    pred_raw: np.ndarray,
    confidence_raw: np.ndarray,
    global_idx: int,
    target_folder: Path,
    dataset_type: str,
    target_depth_paths: Optional[List[Path]],
    orig_shape: Tuple[int, int],
    crop_shape: Tuple[int, int],
    crop_offsets: Tuple[int, int],
    target_max: float,
    align_mask_mode: str,
) -> Optional[Dict[str, object]]:
    """
    Build one frame's evaluation record from prediction raw tensors and GT depth.
    """
    target_path, target_img = load_target_image_for_frame(global_idx=global_idx, target_folder=target_folder, dataset_type=dataset_type, target_depth_paths=target_depth_paths)
    if target_path is None or target_img is None:
        return None

    target_raw = process_target_raw(target_img, orig_shape, crop_shape, crop_offsets)
    pred_raw, confidence_raw, prediction_valid_mask = resizePredictionToTargetShape(pred_raw, confidence_raw, target_raw.shape)
    target_valid_mask = np.isfinite(target_raw) & (target_raw > 0) & (target_raw < target_max)
    metric_mask = target_valid_mask if align_mask_mode == 'gt' else target_valid_mask & prediction_valid_mask
    align_mask = metric_mask
    pred_safe = np.nan_to_num(pred_raw, nan=DEPTH_EPSILON, posinf=target_max, neginf=DEPTH_EPSILON)
    pred_safe = np.clip(pred_safe, a_min=DEPTH_EPSILON, a_max=None)
    abs_s, abs_ss, err_map = compute_absrel_s_ss(pred=pred_safe, target=target_raw, metric_mask=metric_mask, align_mask=align_mask)

    return {
        'global_frame': int(global_idx),
        'target_path': target_path,
        'absrel_s': abs_s,
        'absrel_ss': abs_ss,
        'valid_count': int(metric_mask.sum()),
        'target_valid_count': int(target_valid_mask.sum()),
        'prediction_valid_count': int(prediction_valid_mask.sum()),
        'pred_raw': pred_raw,
        'confidence_raw': confidence_raw,
        'target_raw': target_raw,
        'valid_mask': metric_mask,
        'err_map': err_map,
    }


def evaluate_and_visualize(
    depth_pred: torch.Tensor,
    confidence_pred: torch.Tensor,
    rgb_video: np.ndarray,
    target_folder: Path,
    dataset_type: str,
    target_depth_paths: Optional[List[Path]],
    sampled_indices: List[int],
    orig_shape: Tuple[int, int],
    crop_shape: Tuple[int, int],
    crop_offsets: Tuple[int, int],
    output_dir: Path,
    fps: int,
    target_max: float,
    vis_conf_percentile: bool,
    conf_vis_low: float,
    align_method: str,
    align_iters: int,
    align_lr: float,
    align_mask_mode: str,
) -> Dict[str, object]:
    """
    Evaluate scene-level depth against GT and export fixed-scale visualizations.
    """
    pred_np = depth_pred.to(torch.float32).numpy().astype(np.float32)
    confidence_np = confidence_pred.to(torch.float32).numpy().astype(np.float32)
    t_count, _, _ = pred_np.shape
    if confidence_np.shape != pred_np.shape:
        raise ValueError(f'Confidence shape mismatch: {confidence_np.shape} vs depth {pred_np.shape}')

    records = []
    target_frames = []
    pred_frames = []
    for i in range(t_count):
        global_idx = sampled_indices[i]
        target_path, target_img = load_target_image_for_frame(global_idx=global_idx, target_folder=target_folder, dataset_type=dataset_type, target_depth_paths=target_depth_paths)
        if target_path is None or target_img is None:
            continue
        target_raw = process_target_raw(target_img, orig_shape, crop_shape, crop_offsets)
        pred_raw, confidence_raw, prediction_valid_mask = resizePredictionToTargetShape(pred_np[i], confidence_np[i], target_raw.shape)
        target_valid_mask = np.isfinite(target_raw) & (target_raw > 0) & (target_raw < target_max)
        metric_mask = target_valid_mask if align_mask_mode == 'gt' else target_valid_mask & prediction_valid_mask
        rgb = rgb_video[i]
        if rgb.shape[:2] != target_raw.shape:
            rgb = cv2.resize(rgb, (target_raw.shape[1], target_raw.shape[0]), interpolation=cv2.INTER_LINEAR)
        records.append(
            {
                'rel_frame': i,
                'global_frame': int(global_idx),
                'target_path': target_path,
                'target_raw': target_raw,
                'valid_mask': metric_mask,
                'valid_count': int(metric_mask.sum()),
                'target_valid_count': int(target_valid_mask.sum()),
                'prediction_valid_count': int(prediction_valid_mask.sum()),
                'confidence_raw': confidence_raw,
                'pred_raw': pred_raw,
                'rgb': rgb,
            }
        )
        target_frames.append(target_raw)
        pred_frames.append(pred_raw)

    if not records:
        raise RuntimeError('No valid paired frames found between prediction and target depth.')

    target_eval = np.stack(target_frames, axis=0).astype(np.float32)
    pred_eval_raw = np.stack(pred_frames, axis=0).astype(np.float32)
    target_valid_mask_eval = np.isfinite(target_eval) & (target_eval > 0) & (target_eval < target_max)
    prediction_valid_mask_eval = np.isfinite(pred_eval_raw) & (pred_eval_raw > DEPTH_EPSILON)
    metric_mask_eval = target_valid_mask_eval if align_mask_mode == 'gt' else target_valid_mask_eval & prediction_valid_mask_eval
    align_mask_eval = metric_mask_eval

    filter_prediction_valid = align_mask_mode == 'gt+pred'
    scale_s = fitDepthScale(pred_eval_raw[align_mask_eval], target_eval[align_mask_eval], filter_prediction_valid=filter_prediction_valid)
    scale_ss, shift_ss = fitDepthScaleShift(
        pred_values=pred_eval_raw[align_mask_eval], target_values=target_eval[align_mask_eval], align_method=align_method, align_iters=align_iters, align_lr=align_lr, filter_prediction_valid=filter_prediction_valid
    )
    pred_aligned_s = applyDepthMetricAlignment(pred_eval_raw, scale_s, 0.0, dataset_type, target_max)
    pred_aligned_ss = applyDepthMetricAlignment(pred_eval_raw, scale_ss, shift_ss, dataset_type, target_max)
    scene_metrics_s = computeDepthMetrics(pred_aligned_s, target_eval, metric_mask_eval)
    scene_metrics_ss = computeDepthMetrics(pred_aligned_ss, target_eval, metric_mask_eval)

    for record_idx, frame_record in enumerate(records):
        valid_mask = frame_record['valid_mask']
        frame_metrics_s = computeDepthMetrics(pred_aligned_s[record_idx], target_eval[record_idx], valid_mask)
        frame_metrics_ss = computeDepthMetrics(pred_aligned_ss[record_idx], target_eval[record_idx], valid_mask)
        err_map = np.zeros_like(target_eval[record_idx], dtype=np.float32)
        err_map[valid_mask] = np.abs(pred_aligned_ss[record_idx][valid_mask] - target_eval[record_idx][valid_mask])
        frame_record['pred_aligned_s'] = pred_aligned_s[record_idx]
        frame_record['pred_aligned_ss'] = pred_aligned_ss[record_idx]
        frame_record['err_map'] = err_map
        frame_record['absrel_s'] = frame_metrics_s['abs_rel']
        frame_record['absrel_ss'] = frame_metrics_ss['abs_rel']
        frame_record['delta1_s'] = frame_metrics_s['delta1']
        frame_record['delta1_ss'] = frame_metrics_ss['delta1']

    fixed_vis_dir = output_dir / f'eval_vis-{FIXED_ERROR_VMIN:g}_{FIXED_ERROR_VMAX:g}'
    fixed_vis_dir.mkdir(parents=True, exist_ok=True)
    fixed_writer = imageio.get_writer(str(fixed_vis_dir / 'fixed-depth_eval_comparison.mp4'), fps=fps, codec='libx264', macro_block_size=None)

    for item in records:
        rel_i = item['rel_frame']
        rgb = item['rgb']
        pred_raw = item['pred_raw']
        pred_aligned_ss_frame = item['pred_aligned_ss']
        confidence_raw = item['confidence_raw']
        target_raw = item['target_raw']
        valid_mask = item['valid_mask']
        err_map = item['err_map']
        pred_valid_mask = np.isfinite(pred_raw) & (pred_raw > DEPTH_EPSILON)
        target_valid_mask = np.isfinite(target_raw) & (target_raw > 0) & (target_raw < target_max)

        pred_vis = colorize_depth_map(pred_aligned_ss_frame)
        target_vis = colorize_depth_map(target_raw)
        err_vis_fixed = colorize_error_map(err_map, valid_mask, FIXED_ERROR_VMIN, FIXED_ERROR_VMAX)
        err_vis_fixed = add_right_vertical_legend(image=err_vis_fixed, cmap_name='jet', vmin=FIXED_ERROR_VMIN, vmax=FIXED_ERROR_VMAX, label='err')
        err_vis_fixed = draw_metrics_text_on_error_map(err_vis=err_vis_fixed, global_frame=item['global_frame'], valid_count=item['valid_count'], absrel_s=item['absrel_s'], absrel_ss=item['absrel_ss'])
        validity_diff_vis = colorize_validity_difference_map(pred_valid_mask, target_valid_mask)
        confidence_vis, _, _ = colorize_confidence_map(confidence_raw, use_percentile=vis_conf_percentile, vis_low=conf_vis_low)
        confidence_vis = add_right_vertical_legend(image=confidence_vis, cmap_name='viridis', vmin=conf_vis_low, vmax=1.0, label='conf')

        top = np.hstack([rgb, pred_vis, target_vis])
        # Row 2, column 2 compares depth validity before alignment on the GT grid:
        # prediction: finite and > DEPTH_EPSILON; GT: finite and 0 < depth < target_max.
        # White = both valid; black = both invalid; red = only prediction invalid;
        # blue = only GT invalid.
        fixed_bottom = np.hstack([err_vis_fixed, validity_diff_vis, confidence_vis])
        fixed_frame_vis = np.vstack([top, fixed_bottom])
        fixed_writer.append_data(padVideoFrameToEvenShape(fixed_frame_vis))

        Image.fromarray(err_vis_fixed).save(fixed_vis_dir / f'fixed-error_vis_{rel_i:05d}.png')

    fixed_writer.close()

    summary = {
        'align_mask': align_mask_mode,
        'metric_mask': align_mask_mode,
        'metric_units': {'delta1': DELTA1_UNIT},
        'num_compared_frames': len(records),
        'valid_pixels': int(scene_metrics_ss['valid_pixels']),
        'target_valid_pixels': int(target_valid_mask_eval.sum()),
        'prediction_valid_pixels': int(prediction_valid_mask_eval.sum()),
        'valid_pixel_ratio': (float(metric_mask_eval.sum() / target_valid_mask_eval.sum()) if np.any(target_valid_mask_eval) else 0.0),
        'align_method': align_method,
        'align_iters': int(align_iters),
        'align_lr': float(align_lr),
        'prediction_post_clip': 'max_depth_only' if dataset_type == 'sintel' else 'none' if dataset_type in ('bonn', 'kitti') else 'positive_and_max_depth',
        'absrel_s_mean': float(scene_metrics_s['abs_rel']),
        'delta1_s_mean': float(scene_metrics_s['delta1']),
        'scale_s': float(scale_s),
        'absrel_ss_mean': float(scene_metrics_ss['abs_rel']),
        'delta1_ss_mean': float(scene_metrics_ss['delta1']),
        'scale_ss': float(scale_ss),
        'shift_ss': float(shift_ss),
        # 'sqrel_ss_mean': float(scene_metrics_ss['sq_rel']),
        # 'rmse_ss_mean': float(scene_metrics_ss['rmse']),
        # 'log_rmse_ss_mean': float(scene_metrics_ss['log_rmse']),
        # 'delta2_ss_mean': float(scene_metrics_ss['delta2']),
        # 'delta3_ss_mean': float(scene_metrics_ss['delta3']),
        'error_fixed_vmin': FIXED_ERROR_VMIN,
        'error_fixed_vmax': FIXED_ERROR_VMAX,
        'frames': [
            {
                'rel_frame': int(item['rel_frame']),
                'global_frame': int(item['global_frame']),
                'target_path': item['target_path'],
                'absrel_s': float(item['absrel_s']),
                'delta1_s': float(item['delta1_s']),
                'absrel_ss': float(item['absrel_ss']),
                'delta1_ss': float(item['delta1_ss']),
                'valid_count': int(item['valid_count']),
                'target_valid_count': int(item['target_valid_count']),
                'prediction_valid_count': int(item['prediction_valid_count']),
            }
            for item in records
        ],
    }

    write_json_atomic(output_dir / 'depth_eval_metrics.json', summary)

    print('\n--- Depth Eval Summary ---')
    print(f"Compared frames: {summary['num_compared_frames']}")
    print(f"Valid pixels   : {summary['valid_pixels']}")
    print(f"Align method   : {summary['align_method']} | mask={summary['align_mask']} | S={summary['scale_s']:.6f} | SS=({summary['scale_ss']:.6f}, {summary['shift_ss']:.6f})")
    print(f"AbsRel(S)↓ mean : {summary['absrel_s_mean']:.4f}")
    print(f"Delta1(S)↑ mean (%): {summary['delta1_s_mean']:.2f}")
    print(f"AbsRel(SS)↓ mean: {summary['absrel_ss_mean']:.4f}")
    print(f"Delta1(SS)↑ mean (%): {summary['delta1_ss_mean']:.2f}")
    print(f"Error scale     : fixed [{FIXED_ERROR_VMIN:g}, {FIXED_ERROR_VMAX:g}]")
    print(f"Metrics file    : {output_dir / 'depth_eval_metrics.json'}")
    print(f"Fixed-Scale Vis : {fixed_vis_dir / 'fixed-depth_eval_comparison.mp4'}")
    print('--------------------------\n')

    return summary


def bindWorkerDevice(
    num_workers: int,
) -> int:
    """
    Bind the current worker process to one visible CUDA device.
    """
    global WORKER_DEVICE
    visible_cuda_devices = get_visible_cuda_devices()
    worker_identity = mp.current_process()._identity
    worker_idx = int(worker_identity[0]) - 1 if worker_identity else 0

    if torch.cuda.is_available() and visible_cuda_devices:
        assigned_device_count = min(len(visible_cuda_devices), num_workers)
        local_cuda_idx = worker_idx % assigned_device_count
        WORKER_DEVICE = torch.device(f'cuda:{local_cuda_idx}')
        print(f"[GPU Assign] worker_idx={worker_idx} -> cuda:{local_cuda_idx} (num_workers={num_workers}, CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES_RAW or 'ALL'})")
    else:
        WORKER_DEVICE = torch.device('cpu')
        print(f'[GPU Assign] worker_idx={worker_idx} -> cpu')

    return worker_idx


def initSceneWorker(
    worker_args,
    dataset_type: str,
    dataset_id: str,
    num_workers: int,
) -> None:
    """
    Initialize one benchmark worker process.
    """
    global WORKER_ARGS, WORKER_DATASET_TYPE, WORKER_DATASET_ID, WORKER_MODEL
    WORKER_ARGS = worker_args
    WORKER_DATASET_TYPE = dataset_type
    WORKER_DATASET_ID = dataset_id
    WORKER_MODEL = None
    bindWorkerDevice(num_workers)
    if WORKER_DEVICE is not None and WORKER_DEVICE.type == 'cuda':
        torch.cuda.set_device(WORKER_DEVICE)
    setRandomness(seed=worker_args.seed, deterministic=worker_args.deterministic)


def getWorkerModel():
    """
    Load the checkpoint once per worker and reuse it across scenes.
    """
    global WORKER_MODEL
    if WORKER_MODEL is None:
        print(f'Loading model once for worker from: {WORKER_ARGS.checkpoint}')
        WORKER_MODEL = load_model(WORKER_ARGS.checkpoint, WORKER_DEVICE, cli_args=WORKER_ARGS)
    return WORKER_MODEL


def processScene(
    scene_name: str,
    video_path: str,
    target_folder: str,
    split_name: str,
    dataset_type: str,
    dataset_id: str,
    output_root_dir: Path,
    args,
    device: torch.device,
    tqdm_position: int = 0,
) -> Dict[str, object]:
    """
    Process one benchmark scene on one worker device.
    """
    paired_image_paths = None
    target_depth_paths = None
    if dataset_type == 'kitti':
        paired_image_paths, target_depth_paths = listKittiPairedFramePaths(Path(video_path), Path(target_folder))
    else:
        target_depth_paths = list_target_depth_paths(Path(target_folder), dataset_type)
    scene_num_frames = resolve_scene_num_frames(video_path=video_path, requested_num_frames=args.infer_total_num_frames, start_frame=args.start_frame, frame_stride=args.frame_stride, image_paths=paired_image_paths)
    infer_chunk_num_frames = scene_num_frames if args.infer_chunk_num_frames is None else int(args.infer_chunk_num_frames)
    if infer_chunk_num_frames <= 0:
        raise ValueError(f'infer_chunk_num_frames must be > 0, got {infer_chunk_num_frames}')
    infer_chunks_overlap_size = None if args.infer_chunks_overlap_size is None else int(args.infer_chunks_overlap_size)
    if infer_chunks_overlap_size is not None and infer_chunks_overlap_size >= infer_chunk_num_frames:
        raise ValueError(f'infer_chunks_overlap_size={infer_chunks_overlap_size} must be smaller than infer_chunk_num_frames={infer_chunk_num_frames}.')
    if scene_num_frames < infer_chunk_num_frames and infer_chunks_overlap_size is None:
        raise ValueError(f'scene {scene_name} only has {scene_num_frames} usable frames, which is smaller than infer_chunk_num_frames={infer_chunk_num_frames}')
    print(f'[Dataset] type={dataset_type} | video={video_path} | target_folder={target_folder}')
    print(f'[Benchmark] scene={scene_name} | requested_infer_total_num_frames={args.infer_total_num_frames} | eval_num_frames={scene_num_frames} | infer_chunk_num_frames={infer_chunk_num_frames}')
    output_dir = output_root_dir / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_done_placeholder_path = get_scene_placeholder_path(output_root_dir, scene_name, SCENE_PLACEHOLDER_DONE_DIR_NAME)
    scene_processing_placeholder_path = None
    scene_cache_complete = scene_prediction_complete(output_dir, scene_num_frames)
    if scene_done_placeholder_path.is_dir() and (not scene_cache_complete):
        print(f'[Placeholder] stale done placeholder detected for {scene_name}, removing it.')
        remove_scene_placeholder(scene_done_placeholder_path)
    scene_progress_bar = tqdm(total=scene_num_frames, desc=f'Scene {scene_name}', unit='frame', dynamic_ncols=True, position=tqdm_position)
    try:
        video_thwc, encode_video_thwc, meta = load_video(
            video_path=video_path,
            num_frames=scene_num_frames,
            enc_img_h=args.img_h,
            enc_img_w=args.img_w,
            proc_h=args.proc_h,
            proc_w=args.proc_w,
            raw_rgb=args.raw_rgb,
            center_crop=bool(args.center_crop),
            start_frame=args.start_frame,
            frame_stride=args.frame_stride,
            image_paths=paired_image_paths,
        )
        input_dir = output_dir / 'input'
        input_dir.mkdir(exist_ok=True)
        video_np_uint8 = np.clip(video_thwc.numpy() * 255.0, 0, 255).astype(np.uint8)
        video_tchw = video_thwc.permute(0, 3, 1, 2)
        encode_video_tchw = encode_video_thwc.permute(0, 3, 1, 2)
        assert video_tchw.shape[1] == 3, f'Expected (T,3,H,W), got {video_tchw.shape}'
        assert encode_video_tchw.shape[:2] == video_tchw.shape[:2], f'Expected encode video to match video T/C, got {encode_video_tchw.shape} vs {video_tchw.shape}'
        depth_dir = output_dir / 'depth'
        depth_dir.mkdir(exist_ok=True)
        confidence_dir = output_dir / 'confidence'
        confidence_dir.mkdir(exist_ok=True)
        fixed_vis_dir = output_dir / f'eval_vis-{FIXED_ERROR_VMIN:g}_{FIXED_ERROR_VMAX:g}'
        fixed_vis_dir.mkdir(exist_ok=True)
        cached_depth = None
        cached_confidence = None
        if scene_cache_complete:
            cached_depth = try_load_cached_depth(depth_dir, num_frames=scene_num_frames)
            cached_confidence = try_load_cached_confidence(confidence_dir, num_frames=scene_num_frames)
        if scene_cache_complete and cached_depth is not None and (cached_confidence is not None):
            depth = cached_depth
            confidence = cached_confidence
            scene_progress_bar.set_postfix_str('cached predictions | rebuild metrics')
            scene_progress_bar.update(scene_num_frames)
            print(f'[Cache] Found complete cached raw predictions in {output_dir}, skip inference and rebuild postprocess outputs.')
        else:
            if scene_cache_complete:
                print(f'[Cache] Found complete raw scene cache for {scene_name}, but cached npy reload failed; rerunning inference.')
                if scene_done_placeholder_path.is_dir():
                    remove_scene_placeholder(scene_done_placeholder_path)
            else:
                print(f'[Cache] Scene raw prediction cache incomplete for {scene_name}; resuming missing chunks.')
            acquired_processing_placeholder, scene_processing_placeholder_path = try_create_scene_placeholder(output_root_dir, scene_name, SCENE_PLACEHOLDER_PROCESSING_DIR_NAME)
            if not acquired_processing_placeholder:
                print(f'[Placeholder] scene processing placeholder exists, skip duplicate run: {scene_name}')
                return {
                    'scene': scene_name,
                    'status': 'skip_processing',
                    'dataset_id': dataset_id,
                    'split': split_name,
                }
            print('Running benchmark depth inference...')
            enc_img_size = (args.img_h, args.img_w)
            target_shape = (video_tchw.shape[2], video_tchw.shape[3])
            if target_depth_paths is None:
                target_depth_paths = list_target_depth_paths(Path(target_folder), dataset_type)
            if infer_chunks_overlap_size is not None:
                requested_stride = int(infer_chunk_num_frames) - int(infer_chunks_overlap_size)
                sliding_clip_ranges = build_sliding_depth_clip_ranges(total_frames=video_tchw.shape[0], clip_frames=infer_chunk_num_frames, requested_stride=requested_stride)
                print_depth_clip_plan(scene_name=scene_name, clip_ranges=sliding_clip_ranges, requested_stride=requested_stride, requested_overlap_frames=infer_chunks_overlap_size)
                model = getWorkerModel()
                depth, confidence = run_depth_sliding_window_inference(
                    model=model,
                    video_tchw=video_tchw,
                    encode_video_tchw=encode_video_tchw,
                    enc_img_size=enc_img_size,
                    output_resolution=target_shape,
                    clip_ranges=sliding_clip_ranges,
                    infer_clip_num_frames=infer_chunk_num_frames,
                    overlap_align_frames=infer_chunks_overlap_size,
                    align_method=args.align_method,
                    align_iters=args.align_iters,
                    align_lr=args.align_lr,
                    sequential_chunk_align=args.sequential_chunk_align,
                    global_chunk_align=bool(args.global_chunk_align),
                    max_stride=int(args.max_stride),
                    target_max=args.target_max,
                    amp_infer=args.amp_infer,
                    progress_bar=scene_progress_bar,
                )
                for global_idx in range(scene_num_frames):
                    depth_raw = depth[global_idx].to(torch.float32).cpu().numpy()
                    confidence_raw = confidence[global_idx].to(torch.float32).cpu().numpy().astype(np.float32)
                    depth_vis = visualize_depth(depth[global_idx].to(torch.float32))
                    Image.fromarray(video_np_uint8[global_idx]).save(input_dir / f'frame_{global_idx:05d}.png')
                    Image.fromarray(depth_vis).save(depth_dir / f'depth_{global_idx:05d}.png')
                    np.save(depth_dir / f'depth_{global_idx:05d}.npy', depth_raw)
                    np.save(confidence_dir / f'confidence_{global_idx:05d}.npy', confidence_raw)
                clip_ranges = []
            else:
                clip_ranges = build_clip_ranges(video_tchw.shape[0], infer_chunk_num_frames)
            for clip_idx, (clip_start, clip_end, keep_start, keep_end) in enumerate(clip_ranges):
                chunk_cache_complete = True
                for global_idx in range(keep_start, keep_end):
                    depth_npy_path = depth_dir / f'depth_{global_idx:05d}.npy'
                    confidence_npy_path = confidence_dir / f'confidence_{global_idx:05d}.npy'
                    if not depth_npy_path.exists() or not confidence_npy_path.exists():
                        chunk_cache_complete = False
                        break
                    try:
                        depth_arr = np.load(depth_npy_path)
                        confidence_arr = np.load(confidence_npy_path)
                    except Exception:
                        chunk_cache_complete = False
                        break
                    if depth_arr.ndim != 2 or confidence_arr.ndim != 2:
                        chunk_cache_complete = False
                        break
                if chunk_cache_complete:
                    scene_progress_bar.set_postfix_str(f'chunk={clip_idx + 1}/{len(clip_ranges)} infer[{clip_start}:{clip_end}) keep[{keep_start}:{keep_end}) cached')
                    scene_progress_bar.update(keep_end - keep_start)
                    continue
                scene_progress_bar.set_postfix_str(f'chunk={clip_idx + 1}/{len(clip_ranges)} infer[{clip_start}:{clip_end}) keep[{keep_start}:{keep_end}) infer')
                model = getWorkerModel()
                clip_video_tchw = video_tchw[clip_start:clip_end].to(device)
                clip_encode_video_tchw = encode_video_tchw[clip_start:clip_end].to(device)
                clip_depth, clip_confidence = run_depth_inference(model=model, video_tchw=clip_video_tchw, encode_video_tchw=clip_encode_video_tchw, enc_img_size=enc_img_size, output_resolution=target_shape, amp_infer=args.amp_infer)
                keep_local_start = keep_start - clip_start
                keep_local_end = keep_end - clip_start
                for local_idx in range(keep_local_start, keep_local_end):
                    global_idx = clip_start + local_idx
                    depth_raw = clip_depth[local_idx].to(torch.float32).cpu().numpy()
                    depth_vis = visualize_depth(clip_depth[local_idx].to(torch.float32))
                    confidence_raw = clip_confidence[local_idx].to(torch.float32).cpu().numpy().astype(np.float32)
                    Image.fromarray(video_np_uint8[global_idx]).save(input_dir / f'frame_{global_idx:05d}.png')
                    Image.fromarray(depth_vis).save(depth_dir / f'depth_{global_idx:05d}.png')
                    np.save(depth_dir / f'depth_{global_idx:05d}.npy', depth_raw)
                    np.save(confidence_dir / f'confidence_{global_idx:05d}.npy', confidence_raw)
                    frame_record = build_frame_eval_record(
                        pred_raw=depth_raw,
                        confidence_raw=confidence_raw,
                        global_idx=meta['sampled_indices'][global_idx],
                        target_folder=Path(target_folder),
                        dataset_type=dataset_type,
                        target_depth_paths=target_depth_paths,
                        orig_shape=meta['orig_shape'],
                        crop_shape=meta['crop_shape'],
                        crop_offsets=meta['crop_offsets'],
                        target_max=args.target_max,
                        align_mask_mode=args.align_mask,
                    )
                    if frame_record is None:
                        continue
                    err_vis_fixed = colorize_error_map(frame_record['err_map'], frame_record['valid_mask'], FIXED_ERROR_VMIN, FIXED_ERROR_VMAX)
                    err_vis_fixed = add_right_vertical_legend(image=err_vis_fixed, cmap_name='jet', vmin=FIXED_ERROR_VMIN, vmax=FIXED_ERROR_VMAX, label='err')
                    err_vis_fixed = draw_metrics_text_on_error_map(err_vis=err_vis_fixed, global_frame=frame_record['global_frame'], valid_count=frame_record['valid_count'], absrel_s=frame_record['absrel_s'], absrel_ss=frame_record['absrel_ss'])
                    Image.fromarray(err_vis_fixed).save(fixed_vis_dir / f'fixed-error_vis_{global_idx:05d}.png')
                scene_progress_bar.update(keep_end - keep_start)
            depth = try_load_cached_depth(depth_dir, num_frames=scene_num_frames)
            confidence = try_load_cached_confidence(confidence_dir, num_frames=scene_num_frames)
            if depth is None:
                raise RuntimeError('Benchmark depth raw cache is still incomplete after chunk inference.')
            if confidence is None:
                raise RuntimeError('Benchmark confidence raw cache is still incomplete after chunk inference.')
        scene_progress_bar.set_postfix_str('rebuilding scene outputs from raw cache')
        print(f'Rebuilding postprocess outputs from raw predictions under: {output_dir}')
        for i in range(video_np_uint8.shape[0]):
            input_png_path = input_dir / f'frame_{i:05d}.png'
            if not input_png_path.exists():
                Image.fromarray(video_np_uint8[i]).save(input_png_path)
        writer = imageio.get_writer(str(output_dir / 'depth_video.mp4'), fps=args.fps, codec='libx264', macro_block_size=None)
        for i in range(depth.shape[0]):
            depth_raw = depth[i].to(torch.float32).numpy()
            depth_vis = visualize_depth(depth[i].to(torch.float32))
            rgb = video_np_uint8[i]
            if rgb.shape[:2] != depth_vis.shape[:2]:
                rgb = cv2.resize(rgb, (depth_vis.shape[1], depth_vis.shape[0]), interpolation=cv2.INTER_LINEAR)
            writer.append_data(padVideoFrameToEvenShape(np.concatenate([rgb, depth_vis], axis=1)))
            depth_png_path = depth_dir / f'depth_{i:05d}.png'
            if not depth_png_path.exists():
                Image.fromarray(depth_vis).save(depth_png_path)
            if not (depth_dir / f'depth_{i:05d}.npy').exists():
                np.save(depth_dir / f'depth_{i:05d}.npy', depth_raw)
        writer.close()
        print(f'Depth artifacts ready: {depth_dir}')
        scene_summary = evaluate_and_visualize(
            depth_pred=depth,
            confidence_pred=confidence,
            rgb_video=video_np_uint8,
            target_folder=Path(target_folder),
            dataset_type=dataset_type,
            target_depth_paths=target_depth_paths,
            sampled_indices=meta['sampled_indices'],
            orig_shape=meta['orig_shape'],
            crop_shape=meta['crop_shape'],
            crop_offsets=meta['crop_offsets'],
            output_dir=output_dir,
            fps=args.fps,
            target_max=args.target_max,
            vis_conf_percentile=args.vis_conf_percentile,
            conf_vis_low=args.conf_vis_low,
            align_method=args.align_method,
            align_iters=args.align_iters,
            align_lr=args.align_lr,
            align_mask_mode=args.align_mask,
        )
        try_create_scene_placeholder(output_root_dir, scene_name, SCENE_PLACEHOLDER_DONE_DIR_NAME)
        print(f"[Scene Summary] scene={scene_name} | absrel_s_mean={scene_summary['absrel_s_mean']:.4f} | absrel_ss_mean={scene_summary['absrel_ss_mean']:.4f}\n")
        return {
            'scene': scene_name,
            'status': 'done',
            'dataset_id': dataset_id,
            'split': split_name,
            'summary': scene_summary,
        }
    finally:
        scene_progress_bar.close()
        if scene_processing_placeholder_path is not None:
            remove_scene_placeholder(scene_processing_placeholder_path)


def processSceneWorker(
    scene_name: str,
) -> Dict[str, object]:
    """
    Worker entry that resolves paths and processes one scene.
    """
    assert WORKER_ARGS is not None, 'Worker args must be initialized before processing scenes.'
    video_path, target_folder, dataset_type = resolve_scene_inputs(WORKER_ARGS.data_root, WORKER_ARGS.split, scene_name)
    worker_identity = mp.current_process()._identity
    worker_position = int(worker_identity[0]) - 1 if worker_identity else 0
    return processScene(
        scene_name=scene_name,
        video_path=video_path,
        target_folder=target_folder,
        split_name=WORKER_ARGS.split,
        dataset_type=dataset_type,
        dataset_id=WORKER_DATASET_ID,
        output_root_dir=Path(WORKER_ARGS.output_dir),
        args=WORKER_ARGS,
        device=WORKER_DEVICE,
        tqdm_position=worker_position,
    )


def processSceneWorkerWithConfig(
    task,
) -> Dict[str, object]:
    """
    Switch one persistent worker to the current dataset configuration before processing a scene.
    """
    global WORKER_ARGS, WORKER_DATASET_TYPE, WORKER_DATASET_ID
    config_args, dataset_type, dataset_id, scene_name = task
    WORKER_ARGS = config_args
    WORKER_DATASET_TYPE = dataset_type
    WORKER_DATASET_ID = dataset_id
    return processSceneWorker(scene_name)


def runSingleSceneBenchmark(
    args,
) -> None:
    """
    Run benchmark logic for one explicitly provided scene.
    """
    global WORKER_ARGS, WORKER_DEVICE, WORKER_MODEL, WORKER_DATASET_TYPE, WORKER_DATASET_ID
    setRandomness(seed=args.seed, deterministic=args.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_type = detect_dataset_type(args.video, args.target_folder)
    dataset_id = dataset_type.upper()
    WORKER_ARGS = args
    WORKER_DEVICE = device
    WORKER_MODEL = None
    WORKER_DATASET_TYPE = dataset_type
    WORKER_DATASET_ID = dataset_id
    target_folder_path = Path(args.target_folder)
    scene_name = target_folder_path.parent.name
    split_name = target_folder_path.parent.parent.name
    output_root_dir = Path(args.output_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)
    processScene(scene_name=scene_name, video_path=args.video, target_folder=args.target_folder, split_name=split_name, dataset_type=dataset_type, dataset_id=dataset_id, output_root_dir=output_root_dir, args=args, device=device, tqdm_position=0)
    write_benchmark_summary(
        benchmark_root_dir=output_root_dir,
        dataset_id=dataset_id,
        split_name=split_name,
        seed=args.seed,
        start_frame=args.start_frame,
        infer_total_num_frames=args.infer_total_num_frames,
        infer_chunk_num_frames=(args.infer_chunk_num_frames if args.infer_chunk_num_frames is not None else args.infer_total_num_frames),
        args=args,
    )


def resolveMultiSceneNames(
    data_root: Path,
    split_name: str,
    target_scene_list: list[str],
) -> list[str]:
    """
    Resolve explicit scene names or one quoted '*' into the full split scene list.
    """
    if 'sintel' in str(data_root).lower() and str(split_name) == '':
        split_dir = data_root if data_root.name == 'final' else data_root / 'training' / 'final'
        if not split_dir.is_dir():
            raise FileNotFoundError(f'Sintel final-pass scene directory not found: {split_dir}')
        if len(target_scene_list) == 1 and str(target_scene_list[0]).strip() == '*':
            return sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        return [str(scene_name).strip() for scene_name in target_scene_list]

    if 'bonn' in str(data_root).lower() and str(split_name) == '':
        if len(target_scene_list) == 1 and str(target_scene_list[0]).strip() == '*':
            return list(BONN_EVAL_SCENES)
        return [str(scene_name).strip() for scene_name in target_scene_list]

    split_dir = data_root / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f'Multi-scene split directory not found: {split_dir}')
    if len(target_scene_list) == 1 and str(target_scene_list[0]).strip() == '*':
        return sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    return [str(scene_name).strip() for scene_name in target_scene_list]


def runMultiSceneBenchmark(
    args,
    initialize_worker: bool = True,
    worker_pool=None,
) -> None:
    """
    Run benchmark logic for one whole scene list using one worker pool.
    """
    output_root_dir = Path(args.output_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)
    data_root_lower = str(args.data_root).lower()
    if 'dynamic_replica' in data_root_lower:
        dataset_type = 'dr'
    elif 'sintel' in data_root_lower:
        dataset_type = 'sintel'
    elif 'bonn' in data_root_lower:
        dataset_type = 'bonn'
    elif 'kitti' in data_root_lower:
        dataset_type = 'kitti'
    else:
        dataset_type = 'po'
    dataset_id = dataset_type.upper()
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

    selected_scene_names = resolveMultiSceneNames(data_root=Path(args.data_root), split_name=args.split, target_scene_list=list(args.target_scene_list))
    random.Random(args.seed).shuffle(selected_scene_names)
    print(f"[Benchmark Pool] scenes={len(selected_scene_names)} | num_workers={num_workers} | CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES_RAW or 'ALL'}")

    progress_bar = tqdm(total=len(selected_scene_names), desc=f'Exec Split: {args.split}', dynamic_ncols=True)
    if worker_pool is not None:
        tasks = [(args, dataset_type, dataset_id, scene_name) for scene_name in selected_scene_names]
        result_iterator = worker_pool.imap_unordered(processSceneWorkerWithConfig, tasks)
        for _ in result_iterator:
            write_benchmark_summary(
                benchmark_root_dir=output_root_dir,
                dataset_id=dataset_id,
                split_name=args.split,
                seed=args.seed,
                start_frame=args.start_frame,
                infer_total_num_frames=args.infer_total_num_frames,
                infer_chunk_num_frames=(args.infer_chunk_num_frames if args.infer_chunk_num_frames is not None else args.infer_total_num_frames),
                args=args,
            )
            progress_bar.update(1)
    elif num_workers <= 1:
        if initialize_worker:
            initSceneWorker(args, dataset_type, dataset_id, num_workers)
        else:
            global WORKER_ARGS, WORKER_DATASET_TYPE, WORKER_DATASET_ID
            WORKER_ARGS = args
            WORKER_DATASET_TYPE = dataset_type
            WORKER_DATASET_ID = dataset_id
        for scene_name in selected_scene_names:
            processSceneWorker(scene_name)
            write_benchmark_summary(
                benchmark_root_dir=output_root_dir,
                dataset_id=dataset_id,
                split_name=args.split,
                seed=args.seed,
                start_frame=args.start_frame,
                infer_total_num_frames=args.infer_total_num_frames,
                infer_chunk_num_frames=(args.infer_chunk_num_frames if args.infer_chunk_num_frames is not None else args.infer_total_num_frames),
                args=args,
            )
            progress_bar.update(1)
    else:
        pool_context = mp.get_context('spawn')
        with pool_context.Pool(processes=num_workers, initializer=initSceneWorker, initargs=(args, dataset_type, dataset_id, num_workers)) as pool:
            for _ in pool.imap_unordered(processSceneWorker, selected_scene_names):
                write_benchmark_summary(
                    benchmark_root_dir=output_root_dir,
                    dataset_id=dataset_id,
                    split_name=args.split,
                    seed=args.seed,
                    start_frame=args.start_frame,
                    infer_total_num_frames=args.infer_total_num_frames,
                    infer_chunk_num_frames=(args.infer_chunk_num_frames if args.infer_chunk_num_frames is not None else args.infer_total_num_frames),
                    args=args,
                )
                progress_bar.update(1)
    progress_bar.close()


def runDatasetConfigs(
    args,
) -> None:
    """
    Run dataset configurations sequentially through persistent GPU scene workers.
    """
    configs = []
    for raw_config in args.dataset_configs:
        fields = str(raw_config).split(',', 7)
        if len(fields) != 8:
            raise ValueError(f'Invalid --dataset_configs entry {raw_config!r}; expected data_root,split,proc_h,proc_w,img_h,img_w,target_max,output_dir')
        data_root, split_name, proc_h, proc_w, img_h, img_w, target_max, output_dir = fields
        config_args = argparse.Namespace(**vars(args))
        config_args.data_root = data_root
        config_args.split = split_name
        config_args.proc_h = int(proc_h)
        config_args.proc_w = int(proc_w)
        config_args.img_h = int(img_h)
        config_args.img_w = int(img_w)
        config_args.target_max = float(target_max)
        config_args.output_dir = output_dir
        config_args.target_scene_list = ['*']
        if min(config_args.proc_h, config_args.proc_w, config_args.img_h, config_args.img_w) <= 0:
            raise ValueError(f'All proc/img dimensions must be positive in --dataset_configs entry {raw_config!r}')
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

    def runConfigs(
        worker_pool=None,
    ) -> None:
        for config_args in configs:
            print(
                f'[Dataset Config] data_root={config_args.data_root} split={config_args.split!r} '
                f'workers={num_workers} '
                f'proc={config_args.proc_h}x{config_args.proc_w} img={config_args.img_h}'
                f'x{config_args.img_w} '
                f'output={Path(config_args.output_dir).resolve()}'
            )
            runMultiSceneBenchmark(config_args, initialize_worker=False, worker_pool=worker_pool)

    first_root = str(configs[0].data_root).lower()
    first_dataset_type = 'sintel' if 'sintel' in first_root else 'bonn' if 'bonn' in first_root else 'kitti' if 'kitti' in first_root else 'dr' if 'dynamic_replica' in first_root else 'po'
    first_dataset_id = first_dataset_type.upper()
    if num_workers <= 1:
        initSceneWorker(configs[0], first_dataset_type, first_dataset_id, num_workers)
        runConfigs()
    else:
        pool_context = mp.get_context('spawn')
        with pool_context.Pool(processes=num_workers, initializer=initSceneWorker, initargs=(configs[0], first_dataset_type, first_dataset_id, num_workers)) as pool:
            runConfigs(worker_pool=pool)


def main():
    args = parse_args()
    if args.video is not None and args.target_folder is not None:
        runSingleSceneBenchmark(args)
        return
    if args.dataset_configs:
        runDatasetConfigs(args)
        return
    runMultiSceneBenchmark(args)


if __name__ == '__main__':
    main()
