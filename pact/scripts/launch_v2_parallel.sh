#!/bin/bash
# Launch 3 v2 training experiments in parallel across GPU01 and GPU02
# Each runs training → auto test eval → saves results

export PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate D3
cd /home/sachin.chaudhary/xgendet
mkdir -p logs checkpoints

echo "============================================"
echo "Launching 3 parallel v2 experiments"
echo "$(date)"
echo "============================================"

# Exp 1: Scratch + OHEM + LabelSmooth (GPU01, GPU#4)
ssh gpu01 "
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4
source ~/.bashrc
conda activate D3
cd /home/sachin.chaudhary/xgendet
nohup python training/train_hydrafake_v2.py \
    --exp_name v2_scratch_ohem \
    --seed 42 \
    --lr 2e-4 --lr_ln 2e-6 \
    --batch_size 64 --epochs 20 --patience 5 \
    --ohem --ohem_hard_ratio 0.5 --ohem_warmup 3 \
    --label_smoothing 0.1 \
    --num_workers 4 --log_freq 25 \
    > logs/v2_scratch_ohem.log 2>&1 &
echo 'Exp1 PID:' \$!
" 2>&1 &

sleep 2

# Exp 2: Finetune from GenImage + OHEM + LabelSmooth (GPU02, all GPUs via DataParallel)
ssh gpu02 "
export PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate D3
cd /home/sachin.chaudhary/xgendet
nohup python training/train_hydrafake_v2.py \
    --exp_name v2_finetune_ohem \
    --pretrained ./checkpoints/xgendet_fulldata/best_model.pth \
    --seed 42 \
    --lr 5e-5 --lr_ln 5e-7 \
    --batch_size 64 --epochs 15 --patience 4 \
    --ohem --ohem_hard_ratio 0.5 --ohem_warmup 2 \
    --label_smoothing 0.1 \
    --num_workers 8 --log_freq 15 \
    > logs/v2_finetune_ohem.log 2>&1 &
echo 'Exp2 PID:' \$!
" 2>&1 &

sleep 2

# Exp 3: Scratch seed=1337 + OHEM (GPU01, GPU#1 — currently ~24GB used, has 57GB free)
ssh gpu01 "
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
source ~/.bashrc
conda activate D3
cd /home/sachin.chaudhary/xgendet
nohup python training/train_hydrafake_v2.py \
    --exp_name v2_scratch_seed1337 \
    --seed 1337 \
    --lr 2e-4 --lr_ln 2e-6 \
    --batch_size 64 --epochs 20 --patience 5 \
    --ohem --ohem_hard_ratio 0.5 --ohem_warmup 3 \
    --label_smoothing 0.1 \
    --num_workers 4 --log_freq 25 \
    > logs/v2_scratch_seed1337.log 2>&1 &
echo 'Exp3 PID:' \$!
" 2>&1 &

wait
echo "All 3 experiments launched. $(date)"
echo ""
echo "Monitor with:"
echo "  tail -5 logs/v2_scratch_ohem.log"
echo "  tail -5 logs/v2_finetune_ohem.log"
echo "  tail -5 logs/v2_scratch_seed1337.log"
