#!/bin/bash
#PBS -N XGenDet_FullData
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=47:30:00
#PBS -o /home/sachin.chaudhary/xgendet/logs/train_full.out
#PBS -j oe

# Activate environment
source ~/.bashrc
conda activate D3

cd /home/sachin.chaudhary/xgendet
mkdir -p logs

# Full dataset training — all 3.1M images, 3 epochs
# ~1.5M real + 1.5M fake = ~48K batches/epoch at batch_size=64
# Estimate: ~8-10 hours per epoch on single GPU
python training/train_stage1.py \
    --config configs/train_stage1.yaml \
    --epochs 3 \
    --batch_size 64 \
    --num_workers 4 \
    --save_freq 1 \
    --exp_name xgendet_fulldata \
    --patience 2

echo "Full data training completed!"
