#!/bin/bash
# ForenSight Pipeline v2: Train on GPU02 → Sharded eval on all free GPUs
# Steps 1&2 already done. Starts from Step 3 (training).

export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
source ~/.bashrc
conda activate Qwen2.5
cd /home/sachin.chaudhary/xgendet
mkdir -p logs checkpoints/forensight_sft/eval_shards

MODEL_PATH="/home/sachin.chaudhary/models/Qwen3-VL-8B-Instruct"

echo "============================================"
echo "ForenSight Pipeline v2"
echo "Start: $(date)"
echo "Host: $(hostname)"
echo "Model: $MODEL_PATH"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo "============================================"

# ─── STEP 3: Train Qwen3-VL-8B + LoRA ───
echo ""
echo ">>> STEP 3: Training ForenSight..."
python training/train_forensight.py \
    --model_name "$MODEL_PATH" \
    --lora_r 64 --lora_alpha 128 \
    --epochs 3 --lr 5e-5 \
    --batch_size 1 --grad_accum 8 \
    --max_len 1024 \
    --output_dir checkpoints/forensight_sft

if [ $? -ne 0 ]; then
    echo "STEP 3 FAILED at $(date)"
    exit 1
fi
echo "Training complete at $(date)"

# ─── STEP 4: Sharded evaluation (1 shard per GPU) ───
echo ""
echo ">>> STEP 4: Sharded evaluation on test set..."
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Launching $NUM_GPUS evaluation shards in parallel..."

for SHARD_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  Launching shard $SHARD_ID on GPU $SHARD_ID"
    CUDA_VISIBLE_DEVICES=$SHARD_ID python training/eval_forensight_shard.py \
        --model_name "$MODEL_PATH" \
        --adapter_path checkpoints/forensight_sft/final_adapter \
        --shard_id $SHARD_ID \
        --num_shards $NUM_GPUS \
        --output_dir checkpoints/forensight_sft/eval_shards \
        > logs/forensight_eval_shard_${SHARD_ID}.log 2>&1 &
done

echo "Waiting for all shards to complete..."
wait
echo "All shards done at $(date)"

# ─── STEP 5: Merge results ───
echo ""
echo ">>> STEP 5: Merging shard results..."
python training/merge_eval_shards.py \
    checkpoints/forensight_sft/eval_shards \
    checkpoints/forensight_sft/test_results.json

echo ""
echo "============================================"
echo "ForenSight Pipeline Complete: $(date)"
echo "Results: checkpoints/forensight_sft/test_results.json"
echo "============================================"
