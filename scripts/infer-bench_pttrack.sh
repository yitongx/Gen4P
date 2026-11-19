#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/dataset/yitongx/WorldTrack}
CHECKPOINT=${CHECKPOINT:-ckpt/gen4p.pth}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/worldtrack}

INFER_TOTAL_NUM_FRAMES=64
INFER_CHUNK_NUM_FRAMES=65

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
IFS=',' read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"

split_configs=(
    "ds_mini,288,512,480,832,$OUTPUT_ROOT/ds_mini"
    "po_mini,288,512,480,832,$OUTPUT_ROOT/po_mini"
    "pstudio_mini,288,512,480,832,$OUTPUT_ROOT/pstudio_mini"
    "adt_mini,512,512,480,480,$OUTPUT_ROOT/adt_mini"
)

python scripts/bench_pttrack.py \
    --checkpoint "$CHECKPOINT" \
    --data_root "$DATA_ROOT" \
    --split_configs "${split_configs[@]}" \
    --target_scene_list '*' \
    --num_workers "${#gpus[@]}" \
    --infer_total_num_frames "$INFER_TOTAL_NUM_FRAMES" \
    --infer_chunk_num_frames "$INFER_CHUNK_NUM_FRAMES" \
    --train_num_frames 2 \
    --center_crop False \
    --raw_rgb True \
    --alignment_mode global_median \
    --pre_vae_encode False \
    --deterministic \
    --seed 42
