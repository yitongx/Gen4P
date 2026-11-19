#!/usr/bin/env bash
set -euo pipefail

BONN_ROOT=${BONN_ROOT:-/dataset/yitongx/video_depth_bench/Bonn/rgbd_bonn_dataset}
SINTEL_ROOT=${SINTEL_ROOT:-/dataset/yitongx/video_depth_bench/Sintel/training/final}
KITTI_ROOT=${KITTI_ROOT:-/dataset/yitongx/video_depth_bench/kitti}
CHECKPOINT=${CHECKPOINT:-ckpt/gen4p.pth}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/videodepth}

INFER_TOTAL_NUM_FRAMES=110
INFER_CHUNK_NUM_FRAMES=49
INFER_CHUNKS_OVERLAP_SIZE=24

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
IFS=',' read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"

dataset_configs=(
    "$BONN_ROOT,,384,512,480,640,70.0,$OUTPUT_ROOT/BONN"
    "$SINTEL_ROOT,,218,512,352,832,70.0,$OUTPUT_ROOT/SINTEL"
    "$KITTI_ROOT,val,154,512,256,832,inf,$OUTPUT_ROOT/KITTIDepth"
)

python scripts/bench_depth.py \
    --checkpoint "$CHECKPOINT" \
    --dataset_configs "${dataset_configs[@]}" \
    --num_workers "${#gpus[@]}" \
    --infer_total_num_frames "$INFER_TOTAL_NUM_FRAMES" \
    --infer_chunk_num_frames "$INFER_CHUNK_NUM_FRAMES" \
    --infer_chunks_overlap_size "$INFER_CHUNKS_OVERLAP_SIZE" \
    --train_num_frames 2 \
    --align_method l1+adam \
    --align_iters 1000 \
    --align_lr 1e-4 \
    --global_chunk_align True \
    --center_crop False \
    --raw_rgb True \
    --pre_vae_encode False \
    --deterministic \
    --seed 42
