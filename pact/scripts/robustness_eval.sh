#!/bin/bash
#PBS -N XGenDet_Robustness
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=02:00:00
#PBS -o /home/sachin.chaudhary/xgendet/logs/robustness_eval.out
#PBS -j oe

# Activate environment
source ~/.bashrc
conda activate D3

cd /home/sachin.chaudhary/xgendet
mkdir -p logs

echo "======================================================"
echo "XGenDet Robustness Evaluation"
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "======================================================"

python scripts/robustness_eval.py \
    --checkpoint /home/sachin.chaudhary/xgendet/checkpoints/xgendet_fulldata/best_model.pth \
    --device cuda

echo ""
echo "End time: $(date)"
echo "Robustness evaluation completed!"
