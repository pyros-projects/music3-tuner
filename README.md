# music3-tuner

LoRA trainer for [MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)'s
Hybrid-LM — the 8B global LM (codebook-0 / musical structure) and the 0.6B RVQ
depth decoder (codebooks 1–7 / acoustic detail). FM transformer, Flow-VAE and
vocoder stay frozen.

## The one hard fact

**MiniMax did not release the audio→codes tokenizer.** The 57.4 GB repo
contains no RVQ quantizer: `dav.pth` is a continuous Flow-VAE (no codebooks),
the synthesis path consumes LM *hidden states* (never codes), and codes exist
publicly only as AR **outputs**. Ground-truth codes for arbitrary training
audio therefore require a distilled encoder — the model itself labels
unlimited (audio, codes) pairs via generation, and an encoder is trained on
those pairs. That inversion is Phase 0 of this repo; everything downstream is
already in place and testable with model-generated codes.

## Layout

| module | what |
|---|---|
| `prompt.py` | template + special tokens + CFG uncond masking |
| `dav.py` | Flow-VAE **encoder+decoder** port of `dav.pth` (44.1 kHz stereo, hop 512) |
| `ar.py` | global LM wrapper + depth decoder, teacher-forced losses for both stages |
| `loading.py` | tokenizer/8B(NF4)/depth-decoder loaders against `~/models/MiniMaxM3` |
| `dataset.py` | cached code sequences + ace-style caption sidecar parser |
| `cache_audio.py` | wav dir → DAV latents (+ `--roundtrip` SNR verification) |
| `generate_codes.py` | AR sampling (CFG 1.5 / top-k 50) → dataset-format code caches |
| `train.py` | QLoRA (NF4 + peft) on the 8B, 23 GiB VRAM watchdog, caption dropout |

## Usage

```bash
uv sync

# 1. verify the DAV port against real audio (writes *_roundtrip.wav)
uv run music3-cache-audio ~/music/ace/audio/wav_neon --seconds 20 --roundtrip

# 2. model-labeled code caches from caption sidecars (needs 8B + rvq_depth_decoder)
uv run music3-generate-codes ~/music/ace/audio/wav_neon --seconds 10 --limit 3

# 2b. corpus generation from the 1000 official structured-caption templates
#     (in-distribution captions = the Phase-0 data engine; fetch once via:)
#     cd cache && git clone --depth 1 --filter=blob:none --sparse \
#       https://github.com/MiniMax-AI/MiniMax-Music3 m3-github && \
#       cd m3-github && git sparse-checkout set skills/music-caption-rewriter/templates
uv run music3-generate-codes --templates cache/m3-github/skills/music-caption-rewriter/templates \
    --shuffle --limit 50 --seconds 120 --out cache/codes_templates

# 3. QLoRA smoke on those caches
uv run music3-train --data cache/codes --steps 50 --max-frames 250

# tests (tiny-model CPU tests + weight-gated smokes)
uv run pytest
```

## RunPod

One-liner on a naked GPU pod (volume at `/workspace`):

```bash
bash <(curl -sL https://raw.githubusercontent.com/pyros-projects/music3-tuner/main/scripts/runpod_setup.sh)
bash /workspace/music3-tuner/scripts/run_corpus.sh   # M3_SECONDS/M3_LIMIT/M3_ROUNDS/M3_SEED knobs
```

Downloads the corpus model set (~19 GB; `FULL=1` adds the FM+vocoder stack),
fetches the 1000 official caption templates, verifies CUDA+tokenizer, then the
corpus engine runs detached, resumable, with auto bf16 on ≥30 GB GPUs.

## Phases

- **Phase 0 — encoder distillation** (the actual research): generate
  (audio, codes) pairs at scale with `generate_codes.py` + the full synthesis
  pipeline, train an audio→codes encoder (DAV latents in, 8 codebook heads
  out). Until it exists, training data is model-generated only.
- **Phase 1 — global-LM LoRA** (`train.py`, works today on generated codes):
  concept/style LoRAs on the 8B. Caption dropout keeps the CFG uncond stream
  calibrated (the H3 guidance-preservation lesson, AR flavor).
- **Phase 2 — depth-decoder finetune** (`ar.py:depth_loss`, wired): timbre-
  level detail, needs Phase-0 codes for real audio.

## Facts (verified against ComfyUI reference + checkpoints)

- 25 audio frames/s; ≤9000 frames; prompt ≤5000 tokens; positions ≤10240
- frame embedding = `(embed(c0+151675) + Σ extra(c_k)) · 8^-0.5` — training
  the global LM needs **all 8 codebooks**, not just c0
- c0 vocab 16384 at offset 151675; acoustic vocab 1024 ×7; stop `<|audio_end|>`
- inference CFG 1.5, uncond = interior→`<|audio_cfg|>`; top-k 50
- DAV: 44.1 kHz stereo, per-channel 64-dim latent @ hop 512, stereo folded to 128
- license: no EU exclusion (unlike MiniMax-H3)
