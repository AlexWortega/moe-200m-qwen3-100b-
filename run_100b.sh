#!/usr/bin/env bash
# 100B-token full pretrain — fresh from-scratch (no warm-start, no
# prior-model dependency).
#
# Idempotent entrypoint. Three states it auto-handles:
#   (a) ckpts_100b/step_*.pt exists  → resume (full state load)
#   (b) ckpts_100b/ empty            → fresh start
#   (c) .stop_200m_qwen3 file        → exit cleanly without launching
#
# Usage:
#   bash run_100b.sh                       # auto-detect
#   STEPS=100 bash run_100b.sh             # smoke
#   FORCE_FRESH=1 bash run_100b.sh         # ignore existing ckpts
#   CP_SIZE=3 bash run_100b.sh             # 3-GPU failover
#
# Env knobs (with full-run defaults):
#   STEPS           override --total_steps  (default 1525879)
#   WARMUP          override --warmup_steps (default 2000)
#   DECAY           override --decay_steps  (default 76294)
#   EMA_START       override --ema_start    (default 1449585)
#   EVAL_EVERY      eval stride             (default 2000)
#   CKPT_EVERY      ckpt stride             (default 5000)
#   PROGRESS_EVERY  progress notify stride  (default 20000)
#   HF_PUSH_EVERY   HF-push every N tokens  (default 10e9)
#   NPROC           num GPUs                (default 2; GPUs 2,3)
#   PEAK_LR         override LR             (default 4e-4)
#   WALL_BUDGET_S   global wall budget seconds (default 2592000 = 30 days)
#   CKPT_DIR        ckpt dir                (default ckpts_100b)
#   FORCE_FRESH     if =1, ignore ckpts_100b/ — start fresh from scratch
set -uo pipefail
cd "$(dirname "$0")"

STOP_FILE=${STOP_FILE:-.stop_200m_qwen3}
if [ -f "$STOP_FILE" ]; then
  echo "[run_100b] $STOP_FILE present — exiting"
  exit 0
fi

# --- GPU set ---
if [ -n "${CP_SIZE:-}" ]; then
  NPROC=${NPROC:-$CP_SIZE}
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
NPROC=${NPROC:-2}

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NSOCKS_PERTHREAD=4
export OMP_NUM_THREADS=8
export TRANSFORMERS_VERBOSITY=error
# Memory headroom hint. 2× V100 has more headroom per GPU than 4× since
# DDP buckets are smaller relative to model+activation memory, but the
# alloc config is still cheap insurance against fragmentation.
# PyTorch 2.x is no-op on platforms that don't support it, with a warning.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset TORCH_LOGS

STEPS=${STEPS:-1525879}
WARMUP=${WARMUP:-2000}
DECAY=${DECAY:-76294}
EMA_START=${EMA_START:-1449585}
EVAL_EVERY=${EVAL_EVERY:-2000}
EVAL_BATCHES=${EVAL_BATCHES:-32}
CKPT_EVERY=${CKPT_EVERY:-5000}
PROGRESS_EVERY=${PROGRESS_EVERY:-20000}
HF_PUSH_EVERY=${HF_PUSH_EVERY:-10000000000}  # 10 B tokens
PEAK_LR=${PEAK_LR:-4e-4}
MIN_LR=${MIN_LR:-4e-5}
START_STEP=${START_STEP:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
SEQ_LEN=${SEQ_LEN:-2048}

# Wall budget bookkeeping
WALL_BUDGET_S=${WALL_BUDGET_S:-2592000}  # 30 days
CKPT_DIR=${CKPT_DIR:-ckpts_100b}
TRAIN_LOG=${TRAIN_LOG:-train_100b.log}
EVAL_LOG=${EVAL_LOG:-eval_100b.log}
START_FILE=${START_FILE:-.run_100b_start_t.txt}
SEED=${SEED:-0}
NAN_CAP=${NAN_CAP:-3000}
FORCE_FRESH=${FORCE_FRESH:-0}
CKPT_KEEP=${CKPT_KEEP:-3}

mkdir -p "$CKPT_DIR"

if [ "$FORCE_FRESH" = "1" ]; then
  rm -f "$START_FILE"
  echo "[run_100b] FORCE_FRESH=1 — anchor cleared; $CKPT_DIR NOT touched"
fi

if [ ! -f "$START_FILE" ]; then
  date +%s > "$START_FILE"
fi
RUN_START=$(cat "$START_FILE")
NOW=$(date +%s)
ELAPSED=$((NOW - RUN_START))
REMAINING=$((WALL_BUDGET_S - ELAPSED))
if [ "$REMAINING" -le 120 ]; then
  echo "[run_100b] wall budget exhausted ($ELAPSED / $WALL_BUDGET_S s) — exiting"
  bash ~/.claude/skills/ml-intern/scripts/notify.sh error \
    "200m-qwen3 100B wall budget exhausted at elapsed=${ELAPSED}s" || true
  exit 0
fi
WALL_CAP_S=$((REMAINING - 300))
echo "[run_100b] wall budget: elapsed=${ELAPSED}s remaining=${REMAINING}s per-launch_cap=${WALL_CAP_S}s"

# Auto-detect latest ckpt for resume
RESUME=""
if [ "$FORCE_FRESH" != "1" ]; then
  LATEST_STEP=$(ls -1 "$CKPT_DIR"/step_*.pt 2>/dev/null \
                | sed 's:.*/step_::; s:\.pt$::' \
                | sort -n | tail -1)
  if [ -n "$LATEST_STEP" ]; then
    RESUME="$CKPT_DIR/step_$LATEST_STEP.pt"
    echo "[run_100b] resuming from $RESUME (step=$LATEST_STEP)"
  fi
fi

# Auto-derive ckpt steps. Variant A: early-frequent rollback points
# (500/1k/2k/3k/4k) on top of the regular CKPT_EVERY cadence so we can
# recover from a fast NaN strike without losing >500 steps.
CKPT_STEPS=$(python3.10 -c "
ce=int('$CKPT_EVERY'); st=int('$STEPS')
early=[500, 1000, 2000, 3000, 4000]
late=[ce*i for i in range(1, st//ce + 2) if ce*i < st]
xs=sorted(set(early + late))
print(','.join(str(x) for x in xs))")

extra_args=()
[ -n "$RESUME" ] && extra_args+=(--resume "$RESUME")
# RESET_BIAS=1 zeroes all MoE router biases right after the resume
# state-dict load. Use after the 2026-05-19 router-bias-controller fix
# when the loaded ckpt was trained under the old (broken) controller and
# has degenerate bias state (range outside [-5, +5]). The supervisor
# only needs this on the FIRST restart; subsequent restarts resume from
# a ckpt that was already saved with the new controller, so RESET_BIAS
# should be unset. Set RESET_BIAS_ONCE_FILE for a self-clearing pattern.
RESET_BIAS=${RESET_BIAS:-0}
if [ "$RESET_BIAS" = "1" ]; then
  extra_args+=(--reset_router_bias_on_resume)
  echo "[run_100b] RESET_BIAS=1 → --reset_router_bias_on_resume"
fi
# Additive Gaussian noise on router sel_logits during training only.
# Use ROUTER_NOISE=1.0 for the first restart after the 2026-05-19
# router-bias-controller fix to revive dead experts; drop back to 0
# (default) once routing balance has been achieved.
ROUTER_NOISE=${ROUTER_NOISE:-0}

echo "[run_100b] NPROC=$NPROC GPUs=$CUDA_VISIBLE_DEVICES"
echo "[run_100b] WSD total=$STEPS warmup=$WARMUP decay=$DECAY ema_start=$EMA_START"
echo "[run_100b] ckpt_every=$CKPT_EVERY eval_every=$EVAL_EVERY"
echo "[run_100b] hf_push_every=$HF_PUSH_EVERY tokens"
echo "[run_100b] batch=$BATCH_SIZE seq=$SEQ_LEN seed=$SEED"

exec python3.10 -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="$NPROC" \
  train/train_200m.py \
  --batch_size "$BATCH_SIZE" --seq_len "$SEQ_LEN" \
  --total_steps "$STEPS" --warmup_steps "$WARMUP" --decay_steps "$DECAY" \
  --peak_lr "$PEAK_LR" --min_lr "$MIN_LR" \
  --ema_start "$EMA_START" --ema_decay 0.9999 \
  --num_workers 4 --bucket_cap_mb 100 --seed "$SEED" \
  --ckpt_dir "$CKPT_DIR" --train_log "$TRAIN_LOG" --eval_log "$EVAL_LOG" \
  --eval_every "$EVAL_EVERY" --eval_batches "$EVAL_BATCHES" \
  --ckpt_steps "$CKPT_STEPS" --wall_cap_s "$WALL_CAP_S" \
  --nan_cap "$NAN_CAP" --progress_every "$PROGRESS_EVERY" \
  --ckpt_keep_last "$CKPT_KEEP" \
  --train_started_at_warmup \
  \
  --hf_push_every_tokens "$HF_PUSH_EVERY" \
  --router_noise_std "$ROUTER_NOISE" \
  "${extra_args[@]}"
