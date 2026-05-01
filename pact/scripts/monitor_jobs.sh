#!/bin/bash
# XGenBench Job Monitor
# Checks status of all submitted PBS jobs and reports results
# Run: bash monitor_jobs.sh

echo "============================================"
echo "XGenBench Job Monitor — $(date)"
echo "============================================"

echo ""
echo "--- PBS Job Status ---"
qstat -u sachin.chaudhary 2>/dev/null | grep -E "Job ID|------|sachin"

echo ""
echo "--- Full-Data Training (Job 15159) ---"
CKPT_DIR="/home/sachin.chaudhary/xgendet/checkpoints/xgendet_fulldata"
if [ -d "$CKPT_DIR" ]; then
    echo "Checkpoints found:"
    ls -lh "$CKPT_DIR"/*.pth 2>/dev/null
    # Try to read epoch from latest checkpoint
    python3 -c "
import torch, sys
try:
    ckpt = torch.load('${CKPT_DIR}/best_model.pth', map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', 'N/A')
    step = ckpt.get('global_step', 'N/A')
    val = ckpt.get('val_metrics', {})
    ap = val.get('ap', 'N/A')
    acc = val.get('accuracy', 'N/A')
    print(f'  Best model: epoch={epoch}, step={step}, AP={ap}, Acc={acc}')
except Exception as e:
    print(f'  Could not read checkpoint: {e}')
" 2>/dev/null
else
    echo "No checkpoints yet"
fi

echo ""
echo "--- Robustness Eval Results ---"
ROBUST_FILE="/home/sachin.chaudhary/xgendet/eval_outputs/robustness_results.json"
if [ -f "$ROBUST_FILE" ]; then
    echo "Results available!"
    python3 -c "
import json
with open('${ROBUST_FILE}') as f:
    data = json.load(f)
for k, v in data.items():
    if isinstance(v, dict):
        ap = v.get('ap', v.get('AP', 'N/A'))
        acc = v.get('accuracy', v.get('acc', 'N/A'))
        print(f'  {k}: AP={ap}, Acc={acc}')
    else:
        print(f'  {k}: {v}')
" 2>/dev/null
else
    echo "Not yet available (job in queue/running)"
fi

echo ""
echo "--- D3 Baseline Results ---"
D3_FILE="/home/sachin.chaudhary/xgendet/eval_outputs/d3_baseline_results.json"
if [ -f "$D3_FILE" ]; then
    echo "Results available!"
    cat "$D3_FILE" | python3 -m json.tool 2>/dev/null | head -30
else
    echo "Not yet available (job in queue/running)"
fi

echo ""
echo "--- Training Log Tail ---"
LOG_FILE="/home/sachin.chaudhary/xgendet/logs/train_full.out"
if [ -f "$LOG_FILE" ]; then
    tail -5 "$LOG_FILE"
else
    echo "No training log file found"
fi

echo ""
echo "============================================"
echo "Next check: run 'bash /home/sachin.chaudhary/xgendet/scripts/monitor_jobs.sh'"
echo "============================================"
