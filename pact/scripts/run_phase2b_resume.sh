#!/bin/bash
# Phase 2b: Simple resume v1_finetune — higher LR, longer training, back to basics
# Runs on gpu02 — all free GPUs
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate Qwen2.5
cd /home/sachin.chaudhary/xgendet

FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 20000) print $1}' | tr '\n' ',' | sed 's/,$//')

echo "============================================"
echo "Phase 2b: Resume v1_finetune (Higher LR)"
echo "Free GPUs: $FREE_GPUS"
echo "Started: $(date)"
echo "============================================"

mkdir -p logs checkpoints/v5_resume_hl

CUDA_VISIBLE_DEVICES=$FREE_GPUS python training/train_hydrafake_v3.py \
    --model base \
    --pretrained checkpoints/hydrafake_finetune/best_model.pth \
    --aug v1 \
    --lr 5e-5 \
    --lr_ln 1e-7 \
    --batch_size 32 \
    --epochs 30 \
    --patience 10 \
    --label_smoothing 0.0 \
    --w_family 0.5 \
    --w_proto_div 0.3 \
    --w_proto_compact 0.2 \
    --w_heatmap 0.3 \
    --w_attr 0.1 \
    --w_calib 0.2 \
    --test_interval 10 \
    --exp_name v5_resume_hl \
    --output_dir checkpoints \
    --num_workers 4 \
    2>&1 | tee logs/v5_resume_hl.log

echo ""
echo "============================================"
echo "Phase 2b Done: $(date)"
echo "Results: checkpoints/v5_resume_hl/test_results.json"
echo "============================================"
