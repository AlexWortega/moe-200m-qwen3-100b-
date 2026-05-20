#!/usr/bin/env bash
# Wrapper for push_milestone.py that the trainer spawns in the background.
# Arg 1: ckpt path (absolute or relative to run dir)
# Arg 2: tokens_seen (int)
set -uo pipefail
cd "$(dirname "$0")/.."
exec python3.10 scripts/push_milestone.py "$@"
