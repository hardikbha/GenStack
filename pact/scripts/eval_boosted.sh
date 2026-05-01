#!/bin/bash
#PBS -N HF_boosted_eval
#PBS -q workq
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:host=gpu01
#PBS -l walltime=12:00:00
#PBS -o /home/sachin.chaudhary/xgendet/logs/boosted_eval.out
#PBS -j oe

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
source ~/.bashrc
conda activate D3

cd /home/sachin.chaudhary/xgendet

echo "============================================"
echo "Boosted Eval: TTA + Ensemble + OptThreshold"
echo "Start: $(date)"
echo "Host: $(hostname)"
echo "============================================"

# Ensemble both models: scratch (best) + finetune (best)
python training/eval_hydrafake_boosted.py \
    --checkpoints \
        ./checkpoints/hydrafake_scratch/best_model.pth \
        ./checkpoints/hydrafake_finetune/best_model.pth \
    --output ./checkpoints/boosted_results.json

echo ""
echo "============================================"
echo "Also running: Scratch-only + TTA (no ensemble)"
echo "============================================"

python training/eval_hydrafake_boosted.py \
    --checkpoints \
        ./checkpoints/hydrafake_scratch/best_model.pth \
    --output ./checkpoints/scratch_tta_results.json

echo "Completed: $(date)"
