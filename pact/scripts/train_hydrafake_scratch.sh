#!/bin/bash
#PBS -N HF_scratch
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:host=gpu01
#PBS -l walltime=23:30:00
#PBS -o /home/sachin.chaudhary/xgendet/logs/hydrafake_scratch.out
#PBS -j oe

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4
source ~/.bashrc
conda activate D3

cd /home/sachin.chaudhary/xgendet
mkdir -p logs checkpoints

echo "============================================"
echo "XGenDet on HydraFake — SCRATCH (Resume from Epoch 2)"
echo "Start: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null
echo "============================================"

python training/train_hydrafake.py \
    --exp_name hydrafake_scratch \
    --resume ./checkpoints/hydrafake_scratch/epoch_2.pth \
    --lr 2e-4 \
    --lr_ln 2e-6 \
    --batch_size 64 \
    --epochs 15 \
    --warmup_steps 300 \
    --patience 5 \
    --num_workers 4 \
    --jpeg_prob 0.5 \
    --blur_prob 0.5 \
    --log_freq 25 \
    --save_freq 1

echo "Completed: $(date)"
