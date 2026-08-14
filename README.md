# music3-tuner

Training + generation toolchain for [MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)'s
Hybrid-LM — the part the official release doesn't cover. LoRA-train the 8B
global LM, generate tracks end-to-end, and **distill the audio→codes
tokenizer MiniMax didn't release**.

```bash
uv sync
uv run music3-gen --prompt "dark synthwave, driving bass, 120 bpm" --seconds 30
uv run music3-gen -p "..." -l "[Verse]\nneon lights ahead\n[Chorus]\nwe run tonight" -s 60 --seed 7
```

Agents/contributors: read [AGENTS.md](AGENTS.md) — checkpoint contract,
module map, gotchas, current results.

## Why a distilled encoder

The 57.4 GB official release is decode-only: `dav.pth` is a continuous
Flow-VAE (no quantizer), synthesis consumes LM hidden states, and codes exist
publicly only as AR *outputs*. Training on real audio needs audio→codes —
so the model labels its own (audio, codes) pairs via generation, and a small
encoder learns the inverse. Feasibility is proven: **full-track held-out c0
top-1 29.26% (chance 0.006%)** on the current self-labeled corpus, and real never-seen tracks
survive the full code bottleneck (encode → teacher-force → FM → vocoder)
with envelope correlation 0.75–0.93 (unrelated baseline −0.49).

## Pipeline

```
                    1000 official caption templates
                                ↓
prompt ──► 8B AR + depth decoder ──► primer [1,8] + emitted codes [T,8]
              (music3-generate-codes)                        ↓
                                                teacher-forced hiddens [T,8×4096]
                                              condition fusion → chunked FM → vocoder
                                                     (music3-synth / music3-gen)
                                                             ↓
real wav ──► DAV encoder ──► CodesEncoder ──► codes    44.1 kHz stereo wav
             (music3-encode: the distilled tokenizer)
codes + captions ──► QLoRA on the 8B (music3-train)
```

## CLIs

| command | what |
|---|---|
| `music3-gen` | prompt → wav, one command |
| `music3-generate-codes` | AR sampler → V2 code caches with separate non-emitted primer and explicit termination metadata |
| `music3-synth` | code caches → wavs (the audio side of training pairs; resumable) |
| `music3-encode` | real audio → code caches via the distilled encoder |
| `music3-prepare-pairs` / `music3-train-encoder` | build (latent, codes) pairs / train the encoder (rich UI, JSONL log) |
| `music3-train` | QLoRA on the 8B over code caches (NF4 + peft, VRAM watchdog) |
| `music3-cache-audio` | DAV latents + roundtrip verification |

## RunPod

```bash
FULL=1 bash <(curl -sL https://raw.githubusercontent.com/pyros-projects/music3-tuner/main/scripts/runpod_setup.sh)
bash /workspace/music3-tuner/scripts/run_corpus.sh   # codes corpus (M3_SECONDS/LIMIT/ROUNDS/SEED)
bash /workspace/music3-tuner/scripts/run_synth.sh    # audio side of the pairs
```

Corpus set ~19 GB, `FULL=1` adds the FM stack (~16 GB). Both launchers run
detached and resume across pod restarts. Any ≥16 GB GPU works (NF4);
≥30 GB auto-switches to bf16. Measured on an A40: ~10 s per 30 s track
(synthesis), near-realtime AR sampling.

## Status

- [x] DAV Flow-VAE port (encoder+decoder), roundtrip-verified
- [x] AR + depth decoder port, teacher-forced losses, self-prediction sanity ≈ 2.0 CE
- [x] codes→audio synthesis (mirrors the diffusers modular pipeline, merged upstream as #14456)
- [x] corpus engine + 1157-pair legacy corpus (12.21 h)
- [x] Phase 0 encoder v1: full-track held-out c0 29.26% / top-5 61.88%
- [x] QLoRA training loop for the 8B
- [ ] corrected same-corpus retrain before deciding whether encoder v2 needs more data
- [ ] depth-decoder finetune (loss wired, untrained)
- [ ] first real-audio LoRA (waits on encoder AR-loss ≤ ~4; today: 5.7)

License note: MiniMax-Music3's license has no EU exclusion (unlike MiniMax-H3).
