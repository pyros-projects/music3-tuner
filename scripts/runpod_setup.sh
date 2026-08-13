#!/usr/bin/env bash
# RunPod bootstrap for music3-tuner (MiniMax-Music3 LoRA trainer / corpus engine).
#
# Target: a naked Ubuntu-ish GPU pod (root shell) with the persistent volume
# mounted at /workspace. Installs uv + the project venv (torch cu128 — covers
# Ada/Hopper/Blackwell incl. sm_120), downloads the Music3 components the
# corpus engine needs, fetches the 1000 official caption templates, verifies
# CUDA + tokenizer, and prints the go-command.
#
#   bash <(curl -sL https://raw.githubusercontent.com/pyros-projects/music3-tuner/main/scripts/runpod_setup.sh)
#
# Then:  bash /workspace/music3-tuner/scripts/run_corpus.sh
#
# Env knobs: WORKSPACE (/workspace), REPO_BRANCH (main), PY_VER (3.12),
#            FULL=1 additionally pulls the FM transformer + vocoder stack
#            (codes→audio leg, ~16GB extra).
# Re-running is safe: every step is idempotent and downloads resume.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/pyros-projects/music3-tuner.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
PY_VER="${PY_VER:-3.12}"
HUB_SPEC="huggingface_hub[cli,hf_transfer]==0.34.3"

REPO_DIR="$WORKSPACE/music3-tuner"
MODELS_DIR="$WORKSPACE/models/MiniMaxM3"

# Fat caches go on the volume — naked pods often have tiny container disks.
export UV_CACHE_DIR="$WORKSPACE/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE/.uv-python"
export HF_HOME="$WORKSPACE/.hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export DEBIAN_FRONTEND=noninteractive

banner() { printf '\n=== %s ===\n' "$*"; }

banner "GPU / disk sanity"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || {
    echo "nvidia-smi failed — is this a GPU pod?" >&2
    exit 1
}
free_gb="$(df -BG --output=avail "$WORKSPACE" | tail -1 | tr -dc '0-9')"
echo "Free on $WORKSPACE: ${free_gb}GB"
need_gb=40; [[ "${FULL:-0}" == "1" ]] && need_gb=60
if (( free_gb < need_gb )); then
    echo "WARNING: <${need_gb}GB free — models + venv + corpus need roughly that much." >&2
fi

banner "System packages"
apt-get update -qq
apt-get install -y -qq git curl ca-certificates procps

banner "uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

banner "Model download (background) — corpus set ~19GB$([[ "${FULL:-0}" == "1" ]] && echo ', FULL adds ~16GB')"
download_models() {
    mkdir -p "$MODELS_DIR"
    # Corpus engine: 8B global LM + depth decoder + tokenizer (+ DAV for later).
    uvx --from "$HUB_SPEC" hf download MiniMaxAI/MiniMax-Music3 \
        --include "language_model/*" "rvq_depth_decoder/*" "tokenizer/*" \
                  "condition_encoder/*" "dav.pth" "config.json" "modular_model_index.json" \
        --local-dir "$MODELS_DIR"
    if [[ "${FULL:-0}" == "1" ]]; then
        # codes→audio leg (FM transformer + vocoder + scheduler + raw FM ckpt)
        uvx --from "$HUB_SPEC" hf download MiniMaxAI/MiniMax-Music3 \
            --include "transformer/*" "vocoder/*" "scheduler/*" "flowmatching_vae.pth" \
            --local-dir "$MODELS_DIR"
    fi
}
download_models >"$WORKSPACE/hf_download.log" 2>&1 &
DL_PID=$!
echo "Downloading in background (PID $DL_PID) — log: $WORKSPACE/hf_download.log"

banner "music3-tuner: clone + venv (torch cu128)"
if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" fetch origin "$REPO_BRANCH"
    git -C "$REPO_DIR" checkout "$REPO_BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$REPO_BRANCH"
fi
(cd "$REPO_DIR" && uv sync --python "$PY_VER")

banner "Official caption templates (sparse clone, 1000 files)"
TPL_DIR="$REPO_DIR/cache/m3-github"
if [[ ! -d "$TPL_DIR/.git" ]]; then
    mkdir -p "$REPO_DIR/cache"
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/MiniMax-AI/MiniMax-Music3 "$TPL_DIR"
    git -C "$TPL_DIR" sparse-checkout set skills/music-caption-rewriter/templates
fi
echo "templates: $(ls "$TPL_DIR/skills/music-caption-rewriter/templates" | wc -l)"

banner "Point the loaders at the volume"
grep -q MINIMAX_M3_DIR ~/.bashrc 2>/dev/null || \
    echo "export MINIMAX_M3_DIR=$MODELS_DIR" >>~/.bashrc
export MINIMAX_M3_DIR="$MODELS_DIR"

banner "Waiting for model downloads"
if ! wait "$DL_PID"; then
    echo "Model download failed — tail of $WORKSPACE/hf_download.log:" >&2
    tail -n 20 "$WORKSPACE/hf_download.log" >&2
    exit 1
fi

banner "Verify"
(cd "$REPO_DIR" && uv run python - <<'PY'
import torch

from music3_tuner.loading import load_depth_components, load_tokenizer

assert torch.cuda.is_available(), "CUDA not available"
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | {torch.cuda.get_device_name(0)}")
load_tokenizer()
print("tokenizer: special ids validated")
decoder, embedding = load_depth_components()
print(f"depth decoder: {sum(p.numel() for p in decoder.parameters()) / 1e6:.0f}M params OK")
PY
)

banner "DONE — corpus engine ready"
cat <<EOF
Start the corpus run (defaults: all 1000 templates x 30s, auto bf16 on >=30GB GPUs):

    bash $REPO_DIR/scripts/run_corpus.sh

Knobs (env): M3_SECONDS=30 M3_LIMIT=0 M3_ROUNDS=1 M3_SEED=0 M3_QUANT=auto
Output:      $REPO_DIR/cache/codes_templates/   (on the volume)
Monitor:     tail -f $WORKSPACE/corpus_*.log
Fetch home:  runpodctl / rsync the codes_templates dir — it's just safetensors.
EOF
