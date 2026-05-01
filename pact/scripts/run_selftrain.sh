#!/bin/bash
# Pipeline: pseudo-label generation → self-training (runs sequentially on gpu02)
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate Qwen2.5
cd /home/sachin.chaudhary/xgendet
mkdir -p logs checkpoints/v6_selftrain

FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 20000) print $1}' | tr '\n' ',' | sed 's/,$//')

echo "============================================"
echo "Self-Train Pipeline (gpu02)"
echo "Free GPUs: $FREE_GPUS"
echo "Started: $(date)"
echo "============================================"

# Step 1: Generate pseudo-labels using 1 GPU (fast, inference only)
echo ""
echo "--- Step 1: Building pseudo-labels ---"
FIRST_GPU=$(echo $FREE_GPUS | cut -d',' -f1)
CUDA_VISIBLE_DEVICES=$FIRST_GPU python training/build_pseudo_labels.py \
    --checkpoint checkpoints/v5_resume_hl/best_model.pth \
    --splits cf cd \
    --fake_thresh 0.85 \
    --real_thresh 0.15 \
    --batch_size 128 \
    --output_dir checkpoints/v6_selftrain \
    2>&1 | tee logs/v6_pseudolabels.log

if [ ! -f checkpoints/v6_selftrain/pseudo_labels.json ]; then
    echo "ERROR: pseudo_labels.json not created. Aborting."
    exit 1
fi
echo "Pseudo-labels done. Proceeding to self-training..."

# Step 2: Self-train on all free GPUs
echo ""
echo "--- Step 2: Self-training ---"
CUDA_VISIBLE_DEVICES=$FREE_GPUS python training/self_train.py \
    --pretrained checkpoints/v5_resume_hl/best_model.pth \
    --pseudo_json checkpoints/v6_selftrain/pseudo_labels.json \
    --lr 5e-6 \
    --batch_size 32 \
    --epochs 8 \
    --patience 4 \
    --pseudo_weight 2.0 \
    --label_smoothing 0.05 \
    --test_interval 4 \
    --exp_name v6_selftrain \
    --output_dir checkpoints \
    2>&1 | tee logs/v6_selftrain.log

echo ""
echo "============================================"
echo "Self-Train Done: $(date)"
echo "Results: checkpoints/v6_selftrain/test_results.json"
echo "============================================"
