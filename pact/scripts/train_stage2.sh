#!/bin/bash
#PBS -N XGenDet_Stage2
#PBS -q workq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -o /home/sachin.chaudhary/xgendet/logs/train_stage2_${PBS_JOBID}.out
#PBS -j oe

source ~/.bashrc
conda activate D3

cd /home/sachin.chaudhary/xgendet
mkdir -p logs

# Stage 2: MLLM Fine-tuning
python training/train_stage2.py \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --lora_r 16 \
    --lora_alpha 32 \
    --annotation_file ./data/annotations/train.jsonl \
    --image_root /home/sachin.chaudhary/GTA \
    --heatmap_root ./data/heatmaps \
    --lr 2e-5 \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 8 \
    --output_dir ./checkpoints/stage2 \
    --exp_name xgendet_stage2 \
    --seed 42

echo "Stage 2 training completed!"
