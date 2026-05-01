#!/bin/bash
# Phase 1: Ensemble eval (3 models + TTA + weight tuning) on gpu01
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate Qwen2.5
cd /home/sachin.chaudhary/xgendet

FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 15000) print $1}' | head -1)

echo "============================================"
echo "Phase 1: Ensemble Eval + TTA"
echo "Using GPU: $FREE_GPUS"
echo "Started: $(date)"
echo "============================================"

mkdir -p logs checkpoints/ensemble_v1

CUDA_VISIBLE_DEVICES=$FREE_GPUS python training/ensemble_eval.py \
    --checkpoints \
        checkpoints/hydrafake_scratch/best_model.pth \
        checkpoints/hydrafake_finetune/best_model.pth \
        checkpoints/v3_resaug/best_model.pth \
    --output_dir checkpoints/ensemble_v1 \
    --batch_size 64 \
    --num_workers 4 \
    --tta \
    --tune_weights \
    2>&1 | tee logs/ensemble_v1.log

echo ""
echo "============================================"
echo "Phase 1 Done: $(date)"
echo "Results: checkpoints/ensemble_v1/test_results.json"
echo "============================================"
