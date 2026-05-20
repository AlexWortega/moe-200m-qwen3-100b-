#!/usr/bin/env bash
# Simple supervisor for run_100b.sh — restarts on non-zero exit up to MAX_RESTART times,
# unless .stop_200m_qwen3 flag is set.
#
# Usage: nohup bash supervise_100b.sh > supervise.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")"

STOP_FILE=.stop_200m_qwen3
MAX_RESTART=${MAX_RESTART:-15}
RESTART_DELAY=${RESTART_DELAY:-30}
SUP_LOG=supervise.log

attempt=0
while true; do
  attempt=$((attempt + 1))
  if [ -f "$STOP_FILE" ]; then
    echo "[$(date -Is)] [supervise] $STOP_FILE present — exit cleanly" | tee -a "$SUP_LOG"
    break
  fi
  if [ "$attempt" -gt "$MAX_RESTART" ]; then
    echo "[$(date -Is)] [supervise] reached MAX_RESTART=$MAX_RESTART — bailing" | tee -a "$SUP_LOG"
    bash ~/.claude/skills/ml-intern/scripts/notify.sh error \
      "200m-qwen3 supervisor bailed after $MAX_RESTART restarts" || true
    break
  fi
  echo "[$(date -Is)] [supervise] launch attempt $attempt" | tee -a "$SUP_LOG"
  STDOUT_FILE="train_100b.stdout.sup${attempt}"
  bash run_100b.sh > "$STDOUT_FILE" 2>&1
  rc=$?
  echo "[$(date -Is)] [supervise] launch attempt $attempt exited rc=$rc" | tee -a "$SUP_LOG"
  if [ "$rc" -eq 0 ]; then
    echo "[$(date -Is)] [supervise] rc=0 — assuming completion or stop_file; exit" | tee -a "$SUP_LOG"
    break
  fi
  echo "[$(date -Is)] [supervise] sleeping ${RESTART_DELAY}s before restart" | tee -a "$SUP_LOG"
  sleep "$RESTART_DELAY"
done
