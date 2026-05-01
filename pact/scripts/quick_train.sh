#!/bin/bash
# Quick training script - uses a single generator (ProGAN) for fast iteration
# Use this for debugging before full training

source ~/.bashrc
conda activate D3

cd /home/sachin.chaudhary/xgendet
mkdir -p logs checkpoints

echo "============================================"
echo "XGenDet Quick Train (ProGAN only, 3 epochs)"
echo "============================================"

python training/train_stage1.py \
    --config configs/train_stage1.yaml \
    --batch_size 32 \
    --epochs 3 \
    --max_samples 500 \
    --exp_name xgendet_quick_test \
    --log_freq 10

echo ""
echo "Quick training completed!"
echo "Check results in: ./checkpoints/xgendet_quick_test/"
