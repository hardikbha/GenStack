#!/bin/bash
#PBS -N D3_Baseline_Eval
#PBS -q gpu
#PBS -l select=1:ncpus=2:ngpus=1
#PBS -l walltime=02:00:00
#PBS -o /home/sachin.chaudhary/xgendet/logs/d3_baseline_eval.out
#PBS -j oe

# D3 Baseline Evaluation on OOD Generators
# Runs the D3 (CVPR 2025) model on all 13 OOD generators to establish
# baseline numbers for comparison with XGenDet.

source ~/.bashrc
conda activate D3

cd /home/sachin.chaudhary/xgendet
mkdir -p logs

echo "============================================"
echo "D3 Baseline Evaluation"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Run full evaluation on all OOD generators (up to 500 images per class)
python scripts/d3_baseline_eval.py \
    --full-eval \
    --max-per-generator 500 \
    --device auto

echo ""
echo "============================================"
echo "D3 Baseline Evaluation Completed: $(date)"
echo "============================================"
