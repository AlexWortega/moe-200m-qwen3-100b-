#!/usr/bin/env bash
# 1B-token shakedown for the 200M-qwen3 model — 15 000 steps.
# Validates: loss decreases, router stays balanced, no OOM, no DDP asserts,
# throughput is in the projected band.
#
# Auto-resumes from `ckpts_shakedown/step_*.pt` if any exist.
#
# Usage:
#   bash run_shakedown.sh                     # foreground
#   nohup bash run_shakedown.sh > shakedown.stdout 2>&1 &
#
# Env knobs (with shakedown defaults):
#   STEPS           override --total_steps (default 15000)
#   WARMUP          override --warmup_steps (default 2000)
#   DECAY           override --decay_steps (default 750)
#   NPROC           num GPUs (default 4)
#   CKPT_EVERY      ckpt stride (default 5000)
#   EVAL_EVERY      eval stride (default 1000)
#   PEAK_LR         override LR (default 6e-4)
#   SEED            base seed (default 0)
set -uo pipefail
cd "$(dirname "$0")"

NPROC=${NPROC:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NSOCKS_PERTHREAD=4
export OMP_NUM_THREADS=8
export TRANSFORMERS_VERBOSITY=error
unset TORCH_LOGS

STEPS=${STEPS:-15000}
WARMUP=${WARMUP:-2000}
DECAY=${DECAY:-750}
EMA_START=${EMA_START:-15001}    # off for shakedown
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_BATCHES=${EVAL_BATCHES:-32}
CKPT_EVERY=${CKPT_EVERY:-5000}
PROGRESS_EVERY=${PROGRESS_EVERY:-1000}
PEAK_LR=${PEAK_LR:-6e-4}
MIN_LR=${MIN_LR:-6e-5}
SEED=${SEED:-0}
NAN_CAP=${NAN_CAP:-100}
BATCH_SIZE=${BATCH_SIZE:-8}
SEQ_LEN=${SEQ_LEN:-2048}

CKPT_DIR=${CKPT_DIR:-ckpts_shakedown}
TRAIN_LOG=${TRAIN_LOG:-train_shakedown.log}
EVAL_LOG=${EVAL_LOG:-eval_shakedown.log}

mkdir -p "$CKPT_DIR"

# Auto-resume from latest step ckpt
RESUME=""
LATEST_STEP=$(ls -1 "$CKPT_DIR"/step_*.pt 2>/dev/null \
              | sed 's:.*/step_::; s:\.pt$::' \
              | sort -n | tail -1)
if [ -n "$LATEST_STEP" ]; then
  RESUME="$CKPT_DIR/step_$LATEST_STEP.pt"
  echo "[shakedown] resuming from $RESUME (step=$LATEST_STEP)"
fi

# Auto-derive ckpt step list
CKPT_STEPS=$(python3.10 -c "
ce=int('$CKPT_EVERY'); st=int('$STEPS')
xs=[ce*i for i in range(1, st//ce + 2) if ce*i < st]
print(','.join(str(x) for x in xs))")

extra_args=()
[ -n "$RESUME" ] && extra_args+=(--resume "$RESUME")

echo "[shakedown] NPROC=$NPROC GPUs=$CUDA_VISIBLE_DEVICES"
echo "[shakedown] STEPS=$STEPS warmup=$WARMUP decay=$DECAY peak_lr=$PEAK_LR"
echo "[shakedown] ckpt_every=$CKPT_EVERY eval_every=$EVAL_EVERY"
echo "[shakedown] batch=$BATCH_SIZE seq=$SEQ_LEN seed=$SEED"

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
  --ckpt_steps "$CKPT_STEPS" --wall_cap_s 86400 \
  --nan_cap "$NAN_CAP" --progress_every "$PROGRESS_EVERY" \
  --ckpt_keep_last 2 \
  --train_started_at_warmup \
  --no_compile_model \
  --hf_push_every_tokens 0 \
  "${extra_args[@]}"
