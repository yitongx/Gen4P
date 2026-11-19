#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-inputs/DAVIS}
CHECKPOINT=${CHECKPOINT:-ckpt/gen4p.pth}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/dense_viser}
BASE_PORT=${BASE_PORT:-8081}
DENSE_GRID_STRIDE=${DENSE_GRID_STRIDE:-1}
# NOTE: aspect ratio difference should < 3%
IMG_H=${IMG_H:-480}
IMG_W=${IMG_W:-832}
PROC_H=${PROC_H:-360}
PROC_W=${PROC_W:-640}

INFER_TOTAL_NUM_FRAMES=${INFER_TOTAL_NUM_FRAMES:-49}
INFER_CHUNK_NUM_FRAMES=${INFER_CHUNK_NUM_FRAMES:-49}

DAVIS_MASK_ROOT=${DAVIS_MASK_ROOT-$DATA_ROOT/Annotations/Full-Resolution}

# Videos <= GPUs: one live server per video/GPU, with ports starting at BASE_PORT.
# Videos > GPUs: export .viser recordings offline, processing videos sequentially per GPU.
# Saving screenshots or MP4s keeps live servers running; Ctrl+C stops all servers.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
IFS=',' read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"

input_paths=(
    "$DATA_ROOT/bmx-bumps.mp4"
    "$DATA_ROOT/flamingo.mp4"
    "$DATA_ROOT/kite-surf.mp4"
    "$DATA_ROOT/tennis.mp4"
)

python scripts/dense_viser.py \
    --checkpoint "$CHECKPOINT" \
    --input_paths "${input_paths[@]}" \
    --output_dir "$OUTPUT_ROOT" \
    --num_workers "${#gpus[@]}" \
    --infer_total_num_frames "$INFER_TOTAL_NUM_FRAMES" \
    --infer_chunk_num_frames "$INFER_CHUNK_NUM_FRAMES" \
    --auto_adapt_frame_count True \
    --train_num_frames 2 \
    --img_h "$IMG_H" \
    --img_w "$IMG_W" \
    --proc_h "$PROC_H" \
    --proc_w "$PROC_W" \
    --center_crop False \
    --dense_grid_stride $DENSE_GRID_STRIDE \
    --confidence_percentile 2.0 \
    --track_stride 50 \
    --track_length 20 \
    --track_line_width 5 \
    --distance_scale 1.0 \
    --point_size 0.0015 \
    --pre_vae_encode False \
    --foreground_mask_root "$DAVIS_MASK_ROOT" \
    --port "$BASE_PORT"
