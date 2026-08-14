# Codie review: Music3 encoder pipeline corrections

Date: 2026-08-14
Scope: software and evaluation corrections completed before the next encoder retrain
Status: implementation, review, and the corrected 4000-step retrain are complete

## Executive summary

The previous pipeline produced a working audio-to-codes encoder, but several
bookkeeping and alignment bugs made its validation optimistic or internally
inconsistent. The main model architecture was not replaced. The changes repair
the data contract, evaluation path, inference aggregation, reproducibility, and
QLoRA checkpoint contract around it.

The most important corrections are:

1. Generated Music3 codes now follow the reference primer contract. The first
   sampled frame is a non-emitted warm-up frame, not audio frame zero.
2. Hard-capped generations are no longer mislabeled as natural EOS examples.
3. Encoder validation is grouped by template family and evaluates complete
   tracks through the same overlapping-window path used at inference.
4. Acoustic predictions from overlapping windows are conditioned on one final,
   globally aggregated codebook-0 sequence.
5. AR loss is diagnostic only. It no longer selects encoder checkpoints.
6. QLoRA trains only state that the existing adapter checkpoint actually saves.

No existing corpus, WAV, pair cache, or checkpoint was rewritten. Legacy caches
remain readable. Future caches use the corrected V2 format.

## Completed retrain

Pyro ran:

```bash
uv run music3-train-encoder \
  --pairs cache/pairs \
  --out out/encoder-v2 \
  --steps 4000
```

This intentionally omits `--ar-diagnostic`. The 8B AR model is therefore not
loaded, and AR loss is not logged. It is an optional expensive diagnostic, not
a checkpoint-selection objective.

The `run_start` record in `out/encoder-v2/train_log.jsonl` reports:

- run ID: `1786740273923415310`
- base Git HEAD: `c4c81d327bfea4bf0eeede344db1b6746f8b29ba`
- pairs: 1101 train / 56 validation
- corpus fingerprint: `4802fb5d97409e42`
- crop/window: 512 frames
- batch: 16
- seed: 42
- augmentation: enabled
- scheduled-sampling ceiling: 0.5
- checkpoint selection: c0 top-1, with acoustic top-1 as tie-breaker

Important provenance caveat: the implementation changes were uncommitted when
the run started. `git_head` identifies the base commit, while this document and
the working-tree diff identify the actual training code.

The complete log is available with:

```bash
rg '"val"' out/encoder-v2/train_log.jsonl
```

Validation history:

| step | c0 top-1 | c0 top-5 | acoustic top-1 | saved |
|---:|---:|---:|---:|:---:|
| 500 | 3.88% | 11.91% | 0.53% | yes |
| 1000 | 17.81% | 42.60% | 1.35% | yes |
| 1500 | 24.59% | 54.58% | 2.01% | yes |
| 2000 | 26.72% | 57.98% | 2.49% | yes |
| 2500 | 27.23% | 58.99% | 2.76% | yes |
| 3000 | 27.39% | 59.03% | 2.86% | yes |
| 3500 | 27.33% | 58.86% | 2.90% | no |
| 4000 | 27.32% | 58.79% | 2.91% | no |

The saved `out/encoder-v2/encoder.safetensors` is the EMA checkpoint from step
3000. Later acoustic accuracy rose slightly, but c0 top-1 is the primary
selection metric and had already peaked.

These figures use the new 56-track grouped full-track validation set and must
not be compared directly to the old intro-only/leaky validation log.

## Root causes and fixes

### 1. Reference AR primer alignment

#### Problem

The pinned Diffusers implementation and live ComfyUI Music3 implementation both
treat the first sampled 8-code frame as a warm-up: it is fed back to the AR
model, but it is not emitted as an audio frame. The local generator previously
appended that frame to `codes`. Local synthesis was self-consistent with this
legacy behavior, but it did not match the released inference contract.

#### Implementation

- `generate_codes.generate` now returns a frozen `GenerationResult` containing:
  - `codes [T, 8]`: emitted frames only
  - `primer_codes [1, 8]`: the non-emitted warm-up frame
  - `ended`: true only if `<|audio_end|>` was actually sampled
- Generation performs at most one primer step plus `max_frames` emitted steps.
- Frame counts are validated against the reference 1..9000 range.
- V2 safetensors keep the backward-readable `codes` tensor and add
  `primer_codes`; metadata includes `cache_version`, `termination`,
  `max_frames`, and `seed`.
- `gen.py`, `synth.py`, `pairs.py`, `dataset.py`, `train.py`, and `ar.py` pass
  the optional primer through their existing paths.
- `collect_frame_hiddens` uses the primer hidden as the predictor of emitted
  frame zero for V2 data, while retaining the exact legacy slice when no primer
  exists.

#### Compatibility decision

Legacy caches do not contain a recoverable primer. They retain their previous
alignment rather than being guessed or silently shifted. Existing synthesized
WAV pairs were also generated under that legacy alignment, so they were not
partially migrated.

### 2. False EOS supervision

#### Problem

The old cache format did not record why generation stopped. `CodesDataset`
assumed every uncropped sequence naturally reached `<|audio_end|>`. In the live
corpus, 984 files end at the old 751-frame cap and 48 end at 3000 frames. Those
hard limits were therefore being taught to the global LM as semantic EOS.

#### Implementation

- V2 metadata explicitly records `termination=audio_end` or
  `termination=max_frames`.
- EOS is supervised only for an explicit natural end that remains in the
  selected training slice.
- Legacy caches have unknown termination and conservatively receive no EOS
  target. This avoids inventing labels.
- The teacher-forced global loss supports official-primer and legacy rows in
  one vectorized batch, with correct masked target positions.

### 3. QLoRA trained state that was never saved

#### Problem

PEFT wrapped only `model.lm`, but the optimizer previously included every
trainable parent parameter. `audio_extra_embedding` lives outside the LM and
received gradients: about 29.4M additional parameters. Checkpointing saved only
`model.lm.save_pretrained`, so that learned table was discarded and generation
loaded a different model than training had optimized.

#### Implementation

`_trainable_adapter_parameters` freezes `audio_extra_embedding` and returns only
trainable parameters under `model.lm`. This matches the existing PEFT save/load
contract without adding another artifact format.

### 4. Validation family leakage

#### Problem

The old split hashed the complete filename. Multi-seed siblings such as
`template_s1` and `template_s2` could land on opposite sides. The live cache has
1157 files from 1000 template families. Eighteen of the old 64 validation
tracks shared a family with training data.

#### Implementation

`pair_family` strips a terminal `_s<seed>` suffix before stable hashing. All
seed variants now remain in one partition. The active run sees 1101 train and
56 validation files with zero family crossing.

The optional AR diagnostic panel also chooses stable, seeded representatives
from distinct families rather than the alphabetically first four paths.

### 5. Intro-only validation

#### Problem

The old validation dataset always selected `start=0` and cropped to 512 frames.
Only 32,768 of 63,463 historical validation frames were measured (51.6%), while
deployment averaged overlapping windows over complete tracks.

#### Implementation

- Validation examples now return the complete track.
- Validation uses batch size one and calls the shared `windowed_logprobs` path.
- Metrics therefore cover all frames and include the same overlap behavior as
  deployed encoding.
- JSONL validation records include the evaluated frame count.

### 6. AR loss Goodhart and checkpoint selection

#### Problem

The old `--ar-select` path selected checkpoints solely by frozen-AR likelihood.
Earlier experiments showed the failure mode directly: refinement improved AR
loss while envelope correlation and listening quality became worse. The old AR
panel was also alphabetically biased and included leaked data.

#### Implementation

- The option is now named `--ar-diagnostic`; `--ar-select` remains a deprecated
  compatibility alias.
- AR loss is reported only.
- Checkpoints rank lexicographically by full-track `(c0_top1,
  acoustic_top1)`.
- V2 primer data is passed into the diagnostic loss, so that optional metric is
  not shifted by one position.

### 7. Correlated worker randomness

#### Problem

`PairsDataset` stored a private `random.Random(seed)`. Persistent DataLoader
workers copied the same RNG state and could replay correlated crop,
stereo-swap, noise, and mask decisions.

#### Implementation

The dataset now uses Python's module RNG, which PyTorch independently seeds in
each worker. A seeded `torch.Generator` controls deterministic DataLoader
shuffling and worker seeds.

### 8. Training window was not checkpointed

#### Problem

`--crop` was configurable, but encoder config files did not record it.
Inference hardcoded 512, which is unsafe because sinusoidal positions should
not be extrapolated beyond the training crop.

#### Implementation

- `CodesEncoder(window=...)` persists the value in `config.json`.
- Inference APIs default to the saved window.
- Legacy configs emit a `RuntimeWarning` and fall back to 512.
- The existing legacy encoder was trained with 512, so it is compatible with
  that fallback.

### 9. Incoherent overlapping-window acoustic predictions

#### Problem

Old window inference predicted each window's acoustic books conditioned on
that window's local c0 argmax, then independently averaged c0 and acoustic
log-probabilities. Where windows disagreed, the final acoustic result mixed
posteriors conditioned on c0 values different from the final aggregated c0.

#### Implementation

`windowed_logprobs` now uses two passes:

1. Average c0 log-probabilities across all overlapping windows and finalize one
   c0 sequence.
2. Re-run the windows with slices of that final c0 as `c0_teacher`, then average
   acoustic log-probabilities.

This costs roughly 2x encoder inference. It is a coherence fix, not a claimed
accuracy win. On the historical clean holdout, acoustic top-1 changed from
3.0990% to 3.0750% (-0.024 percentage points), while 3.56% to 4.80% of book
decisions changed depending on the acoustic codebook.

### 10. Stereo roundtrip correlation bug

#### Problem

The roundtrip metric flattened stereo tensors before trimming unequal time
lengths. If the estimate was slightly longer, channel two became time-shifted
relative to the reference. Reported correlations around 0.25..0.55 were false.

#### Implementation

`cache_audio.correlation` crops both tensors on their final/time dimension
before flattening. Corrected examples measure about 0.846..0.9997. SNR was not
affected.

### 11. Path and provenance fixes

- `pairs.py` now defaults to `cache/codes_templates`, matching the corpus and
  synth launch scripts.
- Every encoder run appends a `run_start` record with run ID, arguments, base
  Git SHA, split sizes, corpus fingerprint, and optional AR panel.
- Every validation record says whether it replaced the saved checkpoint and
  names the selection metric.

## Corrected baseline of the existing encoder

The old `out/encoder` checkpoint was re-evaluated without training it.

### Historical validation set, all 64 tracks / 63,463 frames

- c0 top-1: 28.875%
- c0 top-5: 60.858%
- acoustic top-1: 2.966%

### Historically clean family holdout, 46 tracks / 34,546 frames

These are the old validation families that never appeared in the old training
partition:

- c0 top-1: 29.260%
- c0 top-5: 61.877%
- acoustic top-1: 3.075%
- acoustic books 1..7: 5.506%, 5.347%, 3.413%, 2.391%, 1.948%, 1.560%,
  1.361%

An alignment sweep over offsets -4..+4 frames was decisive: offset zero gave
29.26% c0 top-1, while offsets +/-1 fell to roughly 14.6..14.8%. There is no
evidence for trimming or shifting the current pairs before retraining.

The old peak numbers in the appended historical log (roughly 29.7% c0 top-1,
62.1% top-5, and 3.19% acoustic) were based on a leaky, intro-only validation
path and are provisional rather than directly comparable targets.

## Old versus new learned weights

The corrected pipeline is substantially more trustworthy, but the new weights
are not demonstrated to be better than the old weights.

Changing the split means only two tracks (1502 frames) were genuinely held out
from both training runs. On that very small common set:

| model | c0 top-1 | c0 top-5 | acoustic top-1 |
|---|---:|---:|---:|
| old `out/encoder` | 26.17% | 60.85% | 2.56% |
| new `out/encoder-v2` | 26.43% | 60.45% | 2.43% |

That is effectively a tie: slightly higher new c0 top-1, slightly lower new
top-5 and acoustic accuracy, with far too few independent tracks for a quality
claim. Each model scores extremely highly on the other model's validation set
because most of those files belonged to its own training partition. This
confirms substantial corpus memorization and prevents a larger fair retrospective
A/B test.

Operational conclusion: retain v2 as the corrected, reproducible baseline, but
do not call it a quality upgrade over v1 without a blind reconstruction/listening
comparison on real out-of-domain audio.

## Verification completed before retraining

```bash
uv run pytest -q
# 42 passed

uv run pytest -q -m weights
# 2 passed, 40 deselected

uv run python -m compileall -q src tests
git diff --check
```

CLI help smoke tests also passed for generation, encoder training, QLoRA
training, encoding, and synthesis entry points.

Focused regression coverage includes:

- exact primer/natural-end generation sequence
- hard-cap generation without false EOS
- 1..9000 frame validation
- legacy and V2 dataset behavior
- mixed official/legacy global-loss target alignment
- capped primer sequence without EOS
- primer-aligned synthesis hidden selection
- QLoRA optimizer parameter scope
- PEFT optimizer-step save/reload roundtrip
- family-safe split and distinct AR panel
- independent worker crop RNG
- full-track validation path
- checkpoint selection independent of AR loss
- saved-window and legacy fallback behavior
- two-pass c0 conditioning and numerical overlap averaging
- stereo time-axis correlation trimming

The weight-marked checks load the local tokenizer and DAV weights. They do not
qualify a complete 8B NF4 training run, full AR generation, FM synthesis, or
vocoder output. Those expensive workflows were intentionally not run as part
of the pre-retrain correction pass.

## Files changed

| file | purpose |
|---|---|
| `src/music3_tuner/generate_codes.py` | official primer, explicit termination, V2 cache metadata, frame cap |
| `src/music3_tuner/dataset.py` | legacy/V2 loading, primer batching, conservative EOS |
| `src/music3_tuner/ar.py` | primer-aware and mixed-row teacher-forced targets |
| `src/music3_tuner/synth.py` | official predictor-hidden alignment |
| `src/music3_tuner/gen.py` | consume `GenerationResult` |
| `src/music3_tuner/pairs.py` | primer propagation and corrected default path |
| `src/music3_tuner/train.py` | persisted-only QLoRA optimization and primer forwarding |
| `src/music3_tuner/encoder.py` | saved window and coherent two-pass inference |
| `src/music3_tuner/train_encoder.py` | family split, full-track validation, RNG, selection, provenance |
| `src/music3_tuner/cache_audio.py` | correct stereo correlation trimming |
| `README.md`, `AGENTS.md` | corrected contracts, results, and operational guidance |
| `tests/test_*.py` | focused regression coverage listed above |

## Deliberate residual boundaries

These were reviewed and are not blockers for the current encoder retrain:

1. **Long V2 QLoRA caches use only their prefix.** The primer is valid only at
   the start of a track. `CodesDataset` therefore does not attach it to a
   random mid-track crop. Add loss-masked burn-in before exposing later random
   windows from primer-bearing tracks.
2. **Mixed EOS state is scalar at QLoRA collation.** `train.py` currently fixes
   batch size to one, so this is dormant. Introduce a per-row EOS mask before
   supporting QLoRA batch sizes above one.
3. **Unknown nondefault legacy crop cannot be reconstructed.** Legacy encoder
   configs warn and assume 512. Manually annotate a known non-512 historical
   checkpoint before inference.

## Post-run checklist for Claude

1. The log contains its final validation record at step 4000. Remember that the
   trainer saved the best checkpoint at step 3000, not the final step.
2. Reinspect validation records with:

   ```bash
   rg '"val"' out/encoder-v2/train_log.jsonl
   ```

3. Select by c0 top-1, with acoustic top-1 only as a tie-breaker. Do not select
   by training loss or infer reconstruction quality from AR loss.
4. Keep `out/encoder` as the historical baseline and `out/encoder-v2` as the
   corrected-run artifact until listening/reconstruction checks are complete.
5. Review and commit only the intended source, documentation, and test files.
   The untracked files under `in/` belong to Pyro and were not touched.
6. Do not migrate legacy caches or resynthesize WAVs implicitly. A V2 corpus
   generation is a separate, explicit experiment.
