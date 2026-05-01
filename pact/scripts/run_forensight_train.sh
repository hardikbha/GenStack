#!/bin/bash
# ForenSight: Train Qwen3-VL-8B on HydraFake with XGenDet evidence
# Uses the same proven training framework as FACT experiments
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate Qwen2.5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_IB_TIMEOUT=50
export NCCL_TIMEOUT=3600000
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1

# Auto-detect free GPUs
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 10000) print $1}' | tr '\n' ',')
FREE_GPUS=${FREE_GPUS%,}
N_GPUS=$(echo $FREE_GPUS | tr ',' '\n' | wc -l)

echo "============================================"
echo "ForenSight: Qwen3-VL-8B + XGenDet Evidence"
echo "Using $N_GPUS GPUs: $FREE_GPUS"
echo "Dataset: 36,750 HydraFake samples with XGenDet evidence"
echo "Started: $(date)"
echo "============================================"

export CUDA_VISIBLE_DEVICES=$FREE_GPUS

~/.conda/envs/Qwen2.5/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=$N_GPUS \
  --master_addr=127.0.0.1 --master_port=29581 \
  /home/sachin.chaudhary/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/train_qwen.py \
  --deepspeed /home/sachin.chaudhary/Qwen2.5-VL/qwen-vl-finetune/scripts/zero3.json \
  --model_name_or_path /home/sachin.chaudhary/models/Qwen3-VL-8B-Instruct \
  --dataset_use forensight_hydrafake \
  --data_flatten True --tune_mm_vision False --tune_mm_mlp True --tune_mm_llm True --bf16 \
  --output_dir /home/sachin.chaudhary/xgendet/checkpoints/forensight_qwen3 \
  --num_train_epochs 3 --per_device_train_batch_size 1 --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 --max_pixels 200704 --min_pixels 784 \
  --eval_strategy no --save_strategy steps --save_steps 500 --save_total_limit 3 \
  --learning_rate 1e-5 --weight_decay 0.05 --warmup_ratio 0.1 --max_grad_norm 1 \
  --lr_scheduler_type cosine --logging_steps 10 --model_max_length 2048 \
  --gradient_checkpointing True --dataloader_num_workers 2 \
  --run_name forensight_hydrafake --report_to none

echo "Training finished: $(date)"
