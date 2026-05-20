#!/usr/bin/env bash
# 2-GPU micro-bench on GPUs 2,3 (variant A). Same launcher as run_100b.sh
# but overrides total_steps + ckpt_dir + log paths to a throwaway dir.
# Default 300 steps for the variant-A smoke gates (NaN=0, CV<=1.0,
# entropy>=3.0, throughput>=20k, lm_loss<=6.5 at step 300).
#
# Usage:
#   bash scripts/microbench_2gpu.sh [extra train flags...]
#   N_STEPS=50 bash scripts/microbench_2gpu.sh        # quick sanity
#   TAG=foo bash scripts/microbench_2gpu.sh           # custom tag
set -uo pipefail
cd "$(dirname "$0")/.."

NPROC=${NPROC:-2}
N_STEPS=${N_STEPS:-300}
BS=${BS:-8}
SEQ=${SEQ:-2048}
PEAK_LR=${PEAK_LR:-4e-4}
TAG=${TAG:-bench-$(date +%Y%m%d-%H%M%S)}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NSOCKS_PERTHREAD=4
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT_DIR=notes/bench/$TAG/ckpts
LOG_DIR=notes/bench/$TAG
mkdir -p "$CKPT_DIR" "$LOG_DIR"

rm -f .stop_200m_qwen3

echo "[microbench-2gpu] TAG=$TAG NPROC=$NPROC GPUs=$CUDA_VISIBLE_DEVICES steps=$N_STEPS bs=$BS seq=$SEQ peak_lr=$PEAK_LR" \
  | tee "$LOG_DIR/launch.log"

python3.10 -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="$NPROC" \
  train/train_200m.py \
  --batch_size "$BS" --seq_len "$SEQ" \
  --total_steps "$N_STEPS" --warmup_steps 50 --decay_steps "$N_STEPS" \
  --peak_lr "$PEAK_LR" --min_lr "$(python3.10 -c "print(${PEAK_LR}/10)")" \
  --ema_start 1000000 \
  --num_workers 2 --bucket_cap_mb 100 \
  --ckpt_dir "$CKPT_DIR" --train_log "$LOG_DIR/train.log" \
  --eval_log "$LOG_DIR/eval.log" --eval_every 0 \
  --ckpt_steps "$((N_STEPS + 100))" \
  --wall_cap_s 1800 --nan_cap 100 \
  --progress_every "$((N_STEPS + 100))" \
  ${COMPILE_FLAG:---no_compile_model} \
  "$@" 2>&1 | tee "$LOG_DIR/stdout.log" \
  || { echo "[microbench-2gpu] launch failed; see $LOG_DIR/stdout.log"; exit 1; }

# Summarize train.log
python3.10 - <<EOF
import json, sys, os
log = "$LOG_DIR/train.log"
rows = [json.loads(l) for l in open(log)]
if not rows:
    print("[microbench-2gpu] no rows in train.log"); sys.exit(1)
stable = rows[5:] if len(rows) > 10 else rows
toks = [r.get("tok_per_s", 0) for r in stable]
mean_t = sum(toks) / max(len(toks), 1)
n = len(stable)
loss0 = rows[0]["lm_loss"]; lossN = rows[-1]["lm_loss"]
cv0 = rows[0].get("router_cv", float("nan")); cvN = rows[-1].get("router_cv", float("nan"))
ent0 = rows[0].get("router_entropy_bits", float("nan")); entN = rows[-1].get("router_entropy_bits", float("nan"))
print(f"[microbench-2gpu] TAG=$TAG n_steps={n} mean_tok_s={mean_t:.0f} min={min(toks):.0f} max={max(toks):.0f}")
print(f"[microbench-2gpu]   loss {loss0:.3f}->{lossN:.3f}  cv {cv0:.3f}->{cvN:.3f}  entropy {ent0:.3f}->{entN:.3f}")
EOF
