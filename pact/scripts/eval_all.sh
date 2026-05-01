#!/bin/bash
#PBS -N XGenDet_Eval
#PBS -q workq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -o /home/sachin.chaudhary/xgendet/logs/eval_${PBS_JOBID}.out
#PBS -j oe

source ~/.bashrc
conda activate D3

cd /home/sachin.chaudhary/xgendet
mkdir -p logs

CHECKPOINT=${1:-"./checkpoints/xgendet_stage1/best_model.pth"}
OUTPUT_DIR=${2:-"./results"}

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "XGenDet Full Evaluation"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# Main evaluation on OOD generators
echo ""
echo "--- Evaluating on OOD generators ---"
python evaluation/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data_root /home/sachin.chaudhary/GTA \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

# Robustness: JPEG compression
echo ""
echo "--- Robustness: JPEG compression ---"
for quality in 30 50 70 90; do
    echo "  JPEG quality=$quality"
    python evaluation/evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --data_root /home/sachin.chaudhary/GTA \
        --output_dir "$OUTPUT_DIR/robust_jpeg_${quality}" \
        --jpeg_quality $quality \
        --device cuda 2>/dev/null
done

# Robustness: Gaussian blur
echo ""
echo "--- Robustness: Gaussian blur ---"
for sigma in 0.5 1.0 1.5 2.0; do
    echo "  Blur sigma=$sigma"
    python evaluation/evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --data_root /home/sachin.chaudhary/GTA \
        --output_dir "$OUTPUT_DIR/robust_blur_${sigma}" \
        --blur_sigma $sigma \
        --device cuda 2>/dev/null
done

echo ""
echo "============================================"
echo "Evaluation complete. Results in: $OUTPUT_DIR"
echo "============================================"
