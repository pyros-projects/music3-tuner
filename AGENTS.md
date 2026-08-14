# AGENTS.md — agent onboarding for music3-tuner

Everything an agent needs to work here without re-deriving it from the web.
Read this before touching code. `CLAUDE.md` imports this file.

## What this repo is

A complete training + generation toolchain for **MiniMax-Music3**
(HF `MiniMaxAI/MiniMax-Music3`, released 2026-08-13), built day-0/1. It
covers what the official release does not: **training**. Music3's Hybrid-LM
= 8B global LM (Qwen3, predicts semantic codebook-0 per audio frame) + 617M
RVQ depth decoder (codebooks 1–7) + 2.4B flow-matching transformer + Flow-VAE
vocoder (both frozen here).

## The one fact that shapes everything

**MiniMax did not release the audio→codes tokenizer.** The full 57.4 GB HF
repo is decode-only: `dav.pth` is a continuous Flow-VAE (no quantizer),
synthesis consumes LM *hidden states* (never codes), diffusers/ComfyUI/
sglang-omni are all inference-only. Codes exist publicly only as AR outputs.
Therefore this repo **distills the missing encoder** (Phase 0): the model
labels unlimited (audio, codes) pairs via generation; `CodesEncoder` learns
the inverse mapping. Proven feasible (see Results).

## Checkpoint contract (verified against ComfyUI + diffusers references)

- 25 audio frames/s; ≤ 9000 frames; prompt ≤ 5000 tokens; positions ≤ 10240
- codebook-0 (semantic): vocab 16384 at token offset **151675**; stop token
  `<|audio_end|>` = 151670; codebooks 1–7 (acoustic): vocab 1024 each
- frame input embedding = `(embed(c0+151675) + Σ_k extra_embed(c_k)) · 8^-0.5`
  → **training the global LM needs all 8 codebooks**, not just c0
- prompt template (whitespace-exact):
  `<|im_start|><|caption_start|>{caption}<|caption_end|><|lyrics_start|>[start]\n{lyrics}<|lyrics_end|><|im_end|><|audio_start|>`
- AR inference CFG 1.5 / top-k 50; uncond = ids[1:-2] → `<|audio_cfg|>` (151654)
- synthesis conditioning per frame = concat of **8×4096 hiddens**: the LM
  state that *predicted* the frame (cond stream) + the 7 depth-decoder
  outputs (positions 1..7)
- condition encoder: softmax-mix of the 8 slices → Conv1d(4096→2048,k3) →
  nearest-resample ×3.4453125 onto the latent grid
- FM: 200-frame windows, 100-frame hop, CFG 1.7 vs **zero** conditioning,
  30 Euler steps, sigmas linspace(1→1/30) with `invert_sigmas` (transformer
  time: 0=noise→1=data), 172-latent overlap blend, crop 86/258 stitch
- DAV Flow-VAE: 44.1 kHz stereo, per-channel 64-dim latent @ hop 512, stereo
  folded → 128 channels; old-style `weight_g/weight_v` keys (torch's
  parametrization compat hook remaps them)
- license: no EU exclusion (unlike MiniMax-H3)

## Module map

| module | role |
|---|---|
| `prompt.py` | template, token constants, CFG uncond masking |
| `dav.py` | Flow-VAE encoder+decoder port of `dav.pth` (encode is chunked — see gotchas) |
| `ar.py` | `Music3AR`: HF Qwen3 wrapper + depth decoder, teacher-forced losses (`global_loss`, `depth_loss`, `depth_hiddens`) |
| `loading.py` | tokenizer / 8B(NF4) / depth-decoder loaders; `MINIMAX_M3_DIR` env (default `~/models/MiniMaxM3`) |
| `synth.py` | codes→audio: teacher-forced hidden recovery + chunked FM + vocoder (diffusers classes, git-pinned) |
| `generate_codes.py` | AR sampler → dataset-format code caches; `--templates` mode uses the 1000 official captions |
| `gen.py` | `music3-gen -p ... -s ...` one-command prompt→wav |
| `encoder.py` | `CodesEncoder` (Phase-0 distilled tokenizer) + windowed `music3-encode` |
| `pairs.py` / `train_encoder.py` | (latent, codes) pair prep + encoder training (rich UI, JSONL log) |
| `dataset.py` / `train.py` | code-sequence dataset + QLoRA loop for the 8B (LoRA on attn/mlp, VRAM watchdog) |
| `scripts/runpod_setup.sh` | pod bootstrap (`FULL=1` adds the FM stack); `run_corpus.sh` / `run_synth.sh` = resumable nohup launchers |

## Current state & results (2026-08-14)

- Corpus: 1000 model-generated (audio, codes) pairs (30 s each, ≈8 h),
  from the official caption templates; cache format: safetensors
  `{codes [T,8] int32}` + metadata `{caption, lyrics}`
- Encoder v1 (178M, c0-conditioned acoustic heads): **val c0 top-1 ~27%,
  top-5 ~57%, acoustic ~2.7%** — this is the 1000-pair data ceiling;
  longer/bigger training overfits (peak ~step 2000, schedule is 4000 steps)
- AR-loss ladder (teacher-forced, the quality currency): model-own codes
  **1.8–2.1**, encoder-v1 codes on real audio **5.7**, random **9.7**;
  self-prediction ≈ 2 is the sanity anchor — if a refactor moves it to ~9.7,
  label alignment broke
- Reconstruction envelope-correlation: 0.93 in-domain, 0.75–0.79 on real
  out-of-domain tracks (baseline −0.49)
- QLoRA on the 8B works (43.6M trainable, ~1 s/step @ 201 frames, 10.2 GB)

## Environment

- **uv only** (`uv run`, `uv add`, `uv sync`) — never raw python/pip
- torch cu128, transformers 5.x, diffusers **git-pinned** to the Music3
  merge commit (see `pyproject.toml`) — don't bump casually
- Local box: RTX 4090 24 GB under **WSL2** — WSL2 does not OOM, it spills
  into shared memory and grinds; `train.py:arm_vram_watchdog` hard-exits at
  total−1 GiB (override `M3_VRAM_LIMIT_GIB`)
- Models: full HF checkout at `~/models/MiniMaxM3` (override `MINIMAX_M3_DIR`)
- Tests: `uv run pytest` — CPU tiny-model tests + `weights`-marked smokes
  that auto-skip without the local checkout

## Gotchas (each cost real debugging time)

1. **peft wrapping**: after `get_peft_model`, `model.lm.model` is the
   CausalLM, not the backbone — always `lm.get_decoder()`
2. **DAV encode VRAM**: full-resolution convs over a 4-min track spike to
   24 GB — `MusicDav.encode` chunks (~30 s HOP-aligned, shift-invariant,
   chunked==whole verified in tests); don't bypass it
3. **Encoder inference must be windowed** at the training crop length
   (sinusoidal positions don't extrapolate) — `encode_wav_to_codes` handles
   overlap-centered windows
4. **torchaudio ≥2.9 needs torchcodec for I/O** — this repo uses soundfile
5. **transformers v5 `rope_parameters`**: `loading._fix_rope` patches
   rope_theta for older versions; theta is 1e6, silent default is 10000
6. **Left-padding only** for batched prompts in `global_loss` (frames must
   start contiguously); position_ids are derived from the mask
7. WSL2 "GPU shared memory full" in Task Manager is usually the Linux page
   cache after reading model shards — `drop_caches` or `.wslconfig`
   `autoMemoryReclaim=gradual`, not a process leak

## Open threads

- Encoder ceiling is data-bound: next jump = bigger corpus (pod runs with
  `M3_SECONDS=120`, multi-seed) — infra is ready and resumable
- Phase-2 depth-decoder finetune is wired (`ar.py:depth_loss`) but untrained
- LoRA-on-real-audio waits on encoder quality (target: AR loss ≤ ~4)
- Caption dropout (`--uncond-p`) keeps the AR's CFG uncond stream calibrated
  during finetunes — same guidance-preservation principle as image/video
  distilled models; untested at scale here
