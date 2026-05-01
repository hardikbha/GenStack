#!/bin/bash
# ForenSight Full Pipeline: Cache → Build SFT Data → Train → Eval
# Runs sequentially on available GPUs

export PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate Qwen2.5
cd /home/sachin.chaudhary/xgendet
mkdir -p logs checkpoints

echo "============================================"
echo "ForenSight Pipeline"
echo "Start: $(date)"
echo "Host: $(hostname)"
echo "GPUs available: $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
echo "============================================"

# Step 1: Cache XGenDet features on 36K training images
echo ""
echo ">>> STEP 1: Caching XGenDet features..."
export CUDA_VISIBLE_DEVICES=0
python training/cache_xgendet_features.py
if [ $? -ne 0 ]; then
    echo "STEP 1 FAILED"
    exit 1
fi

# Step 2: Build ForenSight SFT data
echo ""
echo ">>> STEP 2: Building ForenSight SFT data..."
python training/build_forensight_sft_data.py
if [ $? -ne 0 ]; then
    echo "STEP 2 FAILED"
    exit 1
fi

# Step 3: Train Qwen3-VL-8B with LoRA (uses all available GPUs via device_map=auto)
echo ""
echo ">>> STEP 3: Training ForenSight (Qwen3-VL-8B + LoRA)..."
unset CUDA_VISIBLE_DEVICES
python training/train_forensight.py \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --lora_r 64 \
    --lora_alpha 128 \
    --epochs 3 \
    --lr 5e-5 \
    --batch_size 1 \
    --grad_accum 8 \
    --max_len 1024 \
    --output_dir checkpoints/forensight_sft
# Train includes auto eval at the end

echo ""
echo "============================================"
echo "ForenSight Pipeline Complete: $(date)"
echo "============================================"
