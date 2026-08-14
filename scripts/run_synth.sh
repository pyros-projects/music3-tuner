#!/usr/bin/env bash
# Synthesis pass launcher: cached code sequences → wavs (the audio side of
# the Phase-0 encoder-distillation pairs). Detached via nohup, resumable
# (existing wavs are skipped). Needs the FULL=1 model set (FM transformer,
# vocoder, scheduler) — re-run runpod_setup.sh with FULL=1 if missing.
#
#   bash run_synth.sh [codes_dir]     (default: cache/codes_templates)
#
# Knobs:
#   M3_STEPS   flow-matching Euler steps per chunk   (default 30)
#   M3_LIMIT   tracks, 0 = all                       (default 0)
#   M3_SEED    FM noise seed                         (default 0)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
export MINIMAX_M3_DIR="${MINIMAX_M3_DIR:-$WORKSPACE/models/MiniMaxM3}"
export PATH="$HOME/.local/bin:$PATH"

CODES_DIR="${1:-$REPO_DIR/cache/codes_templates}"
OUT_DIR="$REPO_DIR/cache/wavs"
M3_STEPS="${M3_STEPS:-30}"
M3_LIMIT="${M3_LIMIT:-0}"
M3_SEED="${M3_SEED:-0}"

[[ -d "$CODES_DIR" ]] || { echo "codes dir $CODES_DIR missing" >&2; exit 1; }
[[ -d "$MINIMAX_M3_DIR/transformer" ]] || {
    echo "FM transformer missing — re-run runpod_setup.sh with FULL=1" >&2
    exit 1
}

log="$WORKSPACE/synth_$(date +%Y%m%d_%H%M%S).log"
nohup bash -c "cd '$REPO_DIR' && uv run music3-synth '$CODES_DIR' \
    --out '$OUT_DIR' --steps '$M3_STEPS' --limit '$M3_LIMIT' --seed '$M3_SEED'" \
    >"$log" 2>&1 &
echo "synth pass running (PID $!)"
echo "  log:  tail -f $log"
echo "  out:  $OUT_DIR"
