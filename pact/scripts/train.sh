#!/bin/bash
#PBS -N XGenDet_Stage1
#PBS -q gpu
#PBS -l select=1:ncpus=2:ngpus=1
#PBS -l walltime=11:30:00
#PBS -o /home/sachin.chaudhary/xgendet/logs/train_stage1.out
#PBS -j oe

# Activate environment
source ~/.bashrc
conda activate D3

cd /home/sachin.chaudhary/xgendet
mkdir -p logs

# Stage 1 Training — start with 50K samples/class for faster iteration
python training/train_stage1.py \
    --config configs/train_stage1.yaml \
    --max_samples 50000 \
    --epochs 5

echo ""
echo "To train with CLI overrides instead:"
echo "  python training/train_stage1.py --config configs/train_stage1.yaml --batch_size 32 --epochs 5"

echo "Training completed!"
