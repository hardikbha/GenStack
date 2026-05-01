#!/bin/bash
# ForenSight Sharded Evaluation — 1 shard per GPU
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate Qwen2.5
cd /home/sachin.chaudhary/xgendet

TRAINED_MODEL="/home/sachin.chaudhary/xgendet/checkpoints/forensight_qwen3"
EVAL_DIR="checkpoints/forensight_qwen3/eval_shards"
mkdir -p $EVAL_DIR logs

# Auto-detect free GPUs
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 15000) print $1}' | head -8)
NUM_GPUS=$(echo $FREE_GPUS | wc -w)

echo "============================================"
echo "ForenSight Sharded Evaluation"
echo "Model: $TRAINED_MODEL"
echo "Free GPUs: $FREE_GPUS ($NUM_GPUS total)"
echo "Started: $(date)"
echo "============================================"

# Launch one shard per GPU
SHARD=0
for GPU in $FREE_GPUS; do
    echo "Launching shard $SHARD on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU python training/eval_forensight_shard.py \
        --model_name "$TRAINED_MODEL" \
        --adapter_path "NONE" \
        --shard_id $SHARD \
        --num_shards $NUM_GPUS \
        --output_dir $EVAL_DIR \
        > logs/forensight_eval_shard_${SHARD}.log 2>&1 &
    SHARD=$((SHARD + 1))
done

echo "Waiting for all $NUM_GPUS shards..."
wait
echo "All shards done: $(date)"

echo ""
echo ">>> Merging results..."
python training/merge_eval_shards.py $EVAL_DIR checkpoints/forensight_qwen3/test_results.json

echo ""
echo "============================================"
echo "Evaluation Complete: $(date)"
echo "Results: checkpoints/forensight_qwen3/test_results.json"
echo "============================================"
