#!/usr/bin/env bash
set -euo pipefail

TUM_ROOT=${TUM_ROOT:-/dataset/yitongx/camera_bench/tum}
SINTEL_ROOT=${SINTEL_ROOT:-/dataset/yitongx/camera_bench/sintel}
SCANNET_ROOT=${SCANNET_ROOT:-/dataset/yitongx/camera_bench/scannetv2}

CHECKPOINT=${CHECKPOINT:-ckpt/gen4p.pth}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/campose}
INFER_TOTAL_NUM_FRAMES=-1
INFER_CHUNK_NUM_FRAMES=49
INFER_CHUNKS_OVERLAP_SIZE=34

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
IFS=',' read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"

dataset_configs=(
    "TUM,$TUM_ROOT,480,640,480,640,$OUTPUT_ROOT/TUM"
    "SINTEL,$SINTEL_ROOT,436,1024,352,832,$OUTPUT_ROOT/SINTEL"
    "SCANNET,$SCANNET_ROOT,968,1296,480,640,$OUTPUT_ROOT/SCANNET"
)

python scripts/bench_campose.py \
    --checkpoint "$CHECKPOINT" \
    --dataset_configs "${dataset_configs[@]}" \
    --target_scene_list '*' \
    --num_workers "${#gpus[@]}" \
    --infer_total_num_frames "$INFER_TOTAL_NUM_FRAMES" \
    --infer_chunk_num_frames "$INFER_CHUNK_NUM_FRAMES" \
    --infer_chunks_overlap_size "$INFER_CHUNKS_OVERLAP_SIZE" \
    --train_num_frames 2 \
    --center_crop False \
    --adjacent_chunk_length 3 \
    --extrinsic_grid_interval 8 \
    --extrinsic_conf_threshold 0.5 \
    --deterministic \
    --seed 42
