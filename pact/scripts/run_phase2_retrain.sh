#!/bin/bash
# Phase 2: Targeted retrain — resume v1_finetune with family weighting + resolution aug
# Runs on gpu02 GPUs 2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate Qwen2.5
cd /home/sachin.chaudhary/xgendet

FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 15000) print $1}' | tr '\n' ',' | sed 's/,$//')

echo "============================================"
echo "Phase 2: Retrain v4_weighted"
echo "Free GPUs: $FREE_GPUS"
echo "Started: $(date)"
echo "============================================"

mkdir -p logs

CUDA_VISIBLE_DEVICES=$FREE_GPUS python training/train_hydrafake_v3.py \
    --model base \
    --pretrained checkpoints/hydrafake_finetune/best_model.pth \
    --aug v2 \
    --resolution_aug_prob 0.6 \
    --family_weights "0:1.0,1:1.5,2:0.8,3:2.0" \
    --lr 1e-5 \
    --lr_ln 1e-7 \
    --batch_size 32 \
    --epochs 20 \
    --patience 7 \
    --label_smoothing 0.1 \
    --w_family 0.5 \
    --w_proto_div 0.3 \
    --w_proto_compact 0.2 \
    --w_heatmap 0.3 \
    --w_attr 0.1 \
    --w_calib 0.2 \
    --exp_name v4_weighted \
    --output_dir checkpoints \
    --num_workers 4 \
    2>&1 | tee logs/v4_weighted.log

echo ""
echo "============================================"
echo "Phase 2 Done: $(date)"
echo "Results: checkpoints/v4_weighted/test_results.json"
echo "============================================"
