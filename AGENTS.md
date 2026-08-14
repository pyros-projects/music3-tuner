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
- the first sampled 8-code frame is a **non-emitted primer**: feed it back to
  advance past `<|audio_start|>`, then emit up to the requested frame count.
  V2 caches store it separately as `primer_codes [1,8]`; `codes [T,8]`
  contains emitted audio frames only. Legacy caches without a primer remain
  readable under their original direct-frame alignment.
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

- Live legacy corpus: 1157 model-generated (audio, codes) pairs, 12.21 h / 1.10M
  frames, from 1000 official caption templates plus 157 seed variants. Future
  V2 caches add `primer_codes` and explicit `termination` metadata; do not
  silently reinterpret or overwrite the legacy WAV/pair artifacts.
- Encoder v1 (178M, c0-conditioned acoustic heads), recomputed over complete
  overlapping windows on the historically clean 46-track holdout: **c0
  top-1 29.26%, top-5 61.88%, acoustic 3.07%**. Offset 0 decisively wins a
  −4…+4 alignment sweep. Longer/bigger runs still overfit near step 2000–2500;
  the next controlled retrain must use the corrected grouped validation.
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
3. **Encoder inference must be windowed** at the saved training crop length
   (sinusoidal positions don't extrapolate) — `encode_wav_to_codes` averages
   c0 first, then recomputes overlapping acoustic predictions using that one
   finalized c0 sequence
4. **torchaudio ≥2.9 needs torchcodec for I/O** — this repo uses soundfile
5. **transformers v5 `rope_parameters`**: `loading._fix_rope` patches
   rope_theta for older versions; theta is 1e6, silent default is 10000
6. **Left-padding only** for batched prompts in `global_loss` (frames must
   start contiguously); position_ids are derived from the mask
7. WSL2 "GPU shared memory full" in Task Manager is usually the Linux page
   cache after reading model shards — `drop_caches` or `.wslconfig`
   `autoMemoryReclaim=gradual`, not a process leak

## Decode/selection extras (post-review upgrades)

- `music3-encode --refine N --refine-lambda 0.5`: AR-prior re-decoding —
  fuses the encoder's frame-local log-probs with the frozen 8B's sequential
  prior (and the depth decoder's per-book prior) via iterated conditional
  modes. **Measured Goodhart trade-off (2026-08-14): λ=0.5×2 cuts real-audio
  AR loss 5.5→4.3 but drops envelope correlation everywhere (~0.02–0.04) —
  on out-of-domain audio the flatter encoder posteriors let the prior
  dominate, and the listening verdict preferred the unrefined decode. Keep
  refine OFF for reconstruction judging; AR loss is diagnosis, not a decode
  objective. Refined codes remain an open experiment as LoRA training data.**
- windowed inference now **averages c0 across half-overlapping windows first**,
  then runs a coherent second acoustic pass conditioned on the finalized c0
- `music3-train-encoder --scheduled-max 0.5` (default on): scheduled
  sampling for the acoustic c0 conditioning — fixes the teacher/inference
  exposure bias; per-book acoustic accuracies are logged and printed
- `music3-train-encoder --ar-diagnostic` logs teacher-forced AR loss on a
  deterministic, family-diverse panel; it is not used for checkpoint selection
  because lower prior loss can Goodhart reconstruction. `--ar-select` remains
  a deprecated alias. Full-track c0 fidelity selects, acoustic fidelity breaks ties.

## Open threads

- Next step is a short same-corpus retrain with corrected grouped/full-track
  validation and independent worker RNG; do not expand the corpus until that
  establishes the remaining software ceiling
- Phase-2 depth-decoder finetune is wired (`ar.py:depth_loss`) but untrained
- LoRA-on-real-audio waits on encoder quality (target: AR loss ≤ ~4)
- Caption dropout (`--uncond-p`) keeps the AR's CFG uncond stream calibrated
  during finetunes — same guidance-preservation principle as image/video
  distilled models; untested at scale here
