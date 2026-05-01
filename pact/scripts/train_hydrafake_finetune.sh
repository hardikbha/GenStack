#!/bin/bash
#PBS -N HF_finetune
#PBS -q workq
#PBS -l select=1:ncpus=8:ngpus=4:mem=64gb
#PBS -l walltime=23:30:00
#PBS -o /home/sachin.chaudhary/xgendet/logs/hydrafake_finetune.out
#PBS -j oe

source ~/.bashrc
conda activate D3

cd /home/sachin.chaudhary/xgendet
mkdir -p logs checkpoints

echo "============================================"
echo "XGenDet on HydraFake — FINE-TUNE (4-GPU DataParallel)"
echo "Start: $(date)"
echo "Host: $(hostname)"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================"

python training/train_hydrafake.py \
    --exp_name hydrafake_finetune \
    --pretrained ./checkpoints/xgendet_fulldata/best_model.pth \
    --lr 5e-5 \
    --lr_ln 5e-7 \
    --batch_size 64 \
    --epochs 10 \
    --warmup_steps 200 \
    --patience 4 \
    --num_workers 8 \
    --jpeg_prob 0.5 \
    --blur_prob 0.5 \
    --log_freq 25 \
    --save_freq 1

echo "Completed: $(date)"
