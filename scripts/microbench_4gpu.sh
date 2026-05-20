#!/usr/bin/env bash
# 4-GPU 50-step micro-bench. Uses the same launcher as run_100b.sh but
# overrides total_steps + ckpt_dir + log paths to a throwaway directory.
#
# Usage:
#   bash scripts/microbench_4gpu.sh [extra train flags...]
set -uo pipefail
cd "$(dirname "$0")/.."

NPROC=${NPROC:-4}
N_STEPS=${N_STEPS:-50}
BS=${BS:-8}
SEQ=${SEQ:-2048}
TAG=${TAG:-bench-$(date +%Y%m%d-%H%M%S)}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NSOCKS_PERTHREAD=4
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT_DIR=notes/bench/$TAG/ckpts
LOG_DIR=notes/bench/$TAG
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# Make sure the 100B stop flag isn't tripping the trainer
rm -f .stop_200m_qwen3

echo "[microbench] TAG=$TAG NPROC=$NPROC steps=$N_STEPS bs=$BS seq=$SEQ" | tee "$LOG_DIR/launch.log"

# Build a no-ckpt step list so save_ckpt never fires inside this short window.
python3.10 -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="$NPROC" \
  train/train_200m.py \
  --batch_size "$BS" --seq_len "$SEQ" \
  --total_steps "$N_STEPS" --warmup_steps 5 --decay_steps "$N_STEPS" \
  --peak_lr 6e-4 --min_lr 6e-5 \
  --ema_start 1000000 \
  --num_workers 2 --bucket_cap_mb 100 \
  --ckpt_dir "$CKPT_DIR" --train_log "$LOG_DIR/train.log" \
  --eval_log "$LOG_DIR/eval.log" --eval_every 0 \
  --ckpt_steps "$((N_STEPS + 100))" \
  --wall_cap_s 900 --nan_cap 100 \
  --progress_every "$((N_STEPS + 100))" \
  ${COMPILE_FLAG:---no_compile_model} \
  "$@" 2>&1 | tee "$LOG_DIR/stdout.log" \
  || { echo "[microbench] launch failed; see $LOG_DIR/stdout.log"; exit 1; }

# Summarize the train.log
python3.10 - <<EOF
import json, sys, os
log = "$LOG_DIR/train.log"
rows = [json.loads(l) for l in open(log)]
if not rows:
    print("[microbench] no rows in train.log"); sys.exit(1)
# Skip first 5 warmup-ish steps for stable mean
stable = rows[5:] if len(rows) > 10 else rows
toks = [r["tok_per_s"] for r in stable]
mean_t = sum(toks) / len(toks)
n = len(stable)
loss0 = rows[0]["lm_loss"]; lossN = rows[-1]["lm_loss"]
cv0 = rows[0]["router_cv"]; cvN = rows[-1]["router_cv"]
print(f"[microbench] TAG=$TAG n_steps={n} mean_tok_s={mean_t:.0f} min={min(toks):.0f} max={max(toks):.0f}")
print(f"[microbench]            loss {loss0:.3f}->{lossN:.3f}  cv {cv0:.3f}->{cvN:.3f}")
EOF
