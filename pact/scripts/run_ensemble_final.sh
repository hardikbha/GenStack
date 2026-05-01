#!/bin/bash
# Final ensemble: v1_scratch + v5_resume_hl (runs on gpu01 in parallel with self-training)
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate Qwen2.5
cd /home/sachin.chaudhary/xgendet
mkdir -p logs checkpoints/ensemble_scratch_v5

FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 20000) print $1}' | tr '\n' ',' | sed 's/,$//')
FIRST_GPU=$(echo $FREE_GPUS | cut -d',' -f1)

echo "============================================"
echo "Final Ensemble: scratch + v5_resume_hl"
echo "Using GPU: $FIRST_GPU"
echo "Started: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=$FIRST_GPU python training/ensemble_eval.py \
    --checkpoints \
        checkpoints/hydrafake_scratch/best_model.pth \
        checkpoints/v5_resume_hl/best_model.pth \
    --output_dir checkpoints/ensemble_scratch_v5 \
    --batch_size 128 \
    --num_workers 4 \
    --tune_weights \
    2>&1 | tee logs/ensemble_scratch_v5.log

echo ""
echo "============================================"
echo "Ensemble Done: $(date)"
echo "Results: checkpoints/ensemble_scratch_v5/test_results.json"
echo "============================================"
