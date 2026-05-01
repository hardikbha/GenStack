#!/bin/bash
# Phase 2c: Threshold tuning on v1_finetune — quick inference only
# Runs on gpu01 — any free GPU
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate Qwen2.5
cd /home/sachin.chaudhary/xgendet

FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 20000) print $1}' | head -1)

echo "============================================"
echo "Phase 2c: Threshold Tuning (v1_finetune)"
echo "Using GPU: $FREE_GPU"
echo "Started: $(date)"
echo "============================================"

mkdir -p logs checkpoints/v5_thresholded

CUDA_VISIBLE_DEVICES=$FREE_GPU python training/threshold_tuning.py \
    --checkpoint checkpoints/hydrafake_finetune/best_model.pth \
    --output_dir checkpoints/v5_thresholded \
    --batch_size 64 \
    --num_workers 4 \
    2>&1 | tee logs/v5_thresholded.log

echo ""
echo "============================================"
echo "Phase 2c Done: $(date)"
echo "Results: checkpoints/v5_thresholded/test_results.json"
echo "============================================"
