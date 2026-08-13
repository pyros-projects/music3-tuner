#!/usr/bin/env bash
# Corpus engine launcher: generate model-labeled code caches from the 1000
# official caption templates. Detached via nohup, resumable (existing
# <template>_s<seed>.safetensors are skipped), multi-round via seed offsets.
#
#   bash run_corpus.sh
#
# Knobs:
#   M3_SECONDS  clip length per track            (default 30)
#   M3_LIMIT    templates per round, 0 = all     (default 0)
#   M3_ROUNDS   rounds with different seeds      (default 1)
#   M3_SEED     base seed                        (default 0)
#   M3_QUANT    auto|nf4|bf16                    (default auto: bf16 on >=30GB VRAM)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
export MINIMAX_M3_DIR="${MINIMAX_M3_DIR:-$WORKSPACE/models/MiniMaxM3}"
export PATH="$HOME/.local/bin:$PATH"
TPL_DIR="$REPO_DIR/cache/m3-github/skills/music-caption-rewriter/templates"
OUT_DIR="$REPO_DIR/cache/codes_templates"

M3_SECONDS="${M3_SECONDS:-30}"
M3_LIMIT="${M3_LIMIT:-0}"
M3_ROUNDS="${M3_ROUNDS:-1}"
M3_SEED="${M3_SEED:-0}"
M3_QUANT="${M3_QUANT:-auto}"

[[ -d "$TPL_DIR" ]] || { echo "templates missing — run runpod_setup.sh first" >&2; exit 1; }

quant_flag=""
if [[ "$M3_QUANT" == "bf16" ]]; then
    quant_flag="--no-quant"
elif [[ "$M3_QUANT" == "auto" ]]; then
    vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
    (( vram_mib >= 30000 )) && quant_flag="--no-quant"
fi

log="$WORKSPACE/corpus_$(date +%Y%m%d_%H%M%S).log"
runner="$WORKSPACE/.corpus_runner.sh"
cat >"$runner" <<INNER
set -uo pipefail
cd "$REPO_DIR"
for round in \$(seq 0 $((M3_ROUNDS - 1))); do
    seed=\$((M3_SEED_BASE + round * 100000))
    echo "=== round \$round (seed \$seed, ${M3_SECONDS}s, limit ${M3_LIMIT}, ${quant_flag:-nf4}) ==="
    uv run music3-generate-codes \
        --templates "$TPL_DIR" \
        --shuffle --seconds "$M3_SECONDS" --limit "$M3_LIMIT" --seed "\$seed" \
        --out "$OUT_DIR" $quant_flag
done
echo "=== corpus run complete: \$(ls "$OUT_DIR" | wc -l) caches ==="
INNER
chmod +x "$runner"

M3_SEED_BASE="$M3_SEED" nohup bash "$runner" >"$log" 2>&1 &
echo "corpus engine running (PID $!)"
echo "  log:  tail -f $log"
echo "  out:  $OUT_DIR"
