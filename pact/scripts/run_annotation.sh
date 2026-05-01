#!/bin/bash
#PBS -N XGenDet_Annotate
#PBS -q workq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -o /home/sachin.chaudhary/xgendet/logs/annotate_${PBS_JOBID}.out
#PBS -j oe

source ~/.bashrc
conda activate D3

cd /home/sachin.chaudhary/xgendet
mkdir -p logs

STAGE1_CKPT=${1:-"./checkpoints/xgendet_stage1/best_model.pth"}
IMAGE_LIST=${2:-"./data/annotation_image_list.txt"}
OUTPUT=${3:-"./data/annotations/train.jsonl"}

mkdir -p ./data/annotations
mkdir -p ./data/heatmaps

echo "Running annotation pipeline..."
echo "Stage 1 checkpoint: $STAGE1_CKPT"
echo "Image list: $IMAGE_LIST"
echo "Output: $OUTPUT"

python data/annotation_pipeline.py \
    --image_list "$IMAGE_LIST" \
    --stage1_checkpoint "$STAGE1_CKPT" \
    --output_file "$OUTPUT" \
    --heatmap_dir ./data/heatmaps \
    --num_threads 4 \
    --gpt_model gpt-4o \
    --device cuda

echo "Annotation pipeline complete!"
