"""codes → audio: teacher-forced hidden collection + chunked flow matching.

The synthesis contract (mirrors diffusers' MiniMax Music 3 modular pipeline,
merged as huggingface/diffusers#14456):

1. Per frame the conditioning is 8 concatenated 4096-dim hiddens: the global
   LM state that *predicted* the frame's semantic code, then the 7 depth-
   decoder outputs (positions 1..7). Generation collects them on the fly; we
   recover the identical tensors by teacher-forcing cached codes.
2. The condition encoder softmax-mixes the 8 slices, projects to 2048 and
   nearest-resamples 25 Hz frames onto the ~86.13 Hz Flow-VAE latent grid.
3. The 2.4B FM transformer denoises 200-frame windows (100-frame hop) with
   CFG 1.7 against zero conditioning, 30 Euler steps (sigmas 1 → 1/30,
   scheduler has invert_sigmas so transformer time runs 0=noise → 1=data),
   blending 172 overlap latents toward the previous window every step.
4. The vocoder decodes each window; overlapping spans are cropped
   (86 latents left / 258 right) so the windows tile the full song.

This module is the audio side of the Phase-0 encoder-distillation pairs and
the preview path for LoRA judging.
"""

from __future__ import annotations

import numpy as np
import torch

from .ar import Music3AR

CHUNK_FRAMES = 200
CHUNK_HOP = 100
OVERLAP_LATENT = 172
CROP_LEFT_LATENT = 86
CROP_RIGHT_LATENT = 344 - 86
FM_CFG_SCALE = 1.7
FM_STEPS = 30
LATENT_HOP = 512


def chunk_starts(num_frames: int) -> list[int]:
    if num_frames <= CHUNK_FRAMES:
        return [0]
    return list(range(0, num_frames - CHUNK_HOP, CHUNK_HOP))


def crop_bounds(chunk_index: int, num_chunks: int, latent_length: int) -> tuple[int, int]:
    """Waveform-sample crop for a decoded window so kept spans tile the song."""
    left = 0 if chunk_index == 0 else CROP_LEFT_LATENT * LATENT_HOP
    right = 0 if chunk_index == num_chunks - 1 else CROP_RIGHT_LATENT * LATENT_HOP
    return left, latent_length * LATENT_HOP - right


@torch.no_grad()
def collect_frame_hiddens(
    model: Music3AR,
    prompt_ids: torch.Tensor,
    codes: torch.Tensor,
    depth_chunk: int = 512,
) -> torch.Tensor:
    """Teacher-force [prompt, frames] and rebuild the per-frame conditioning.

    prompt_ids [1, P] (unpadded, conditional prompt), codes [1, T, 8] →
    frame_hiddens [1, T, 8 * hidden]. Slice 0 is the LM state at the position
    that predicted frame t (last prompt token for t=0, frame t-1 after), the
    rest are the depth-decoder hiddens for codebooks 1..7.
    """
    prompt_len = prompt_ids.shape[1]
    frames = codes.shape[1]
    embeds = torch.cat([model._embed_tokens(prompt_ids), model.embed_frames(codes)], dim=1)
    hidden = model.lm.get_decoder()(inputs_embeds=embeds, use_cache=False).last_hidden_state
    lm_hidden = hidden[:, prompt_len - 1 : prompt_len + frames - 1]  # [1, T, H]

    depth_parts = []
    flat_hidden, flat_codes = lm_hidden.squeeze(0), codes.squeeze(0)
    for start in range(0, frames, depth_chunk):
        chunk_h = flat_hidden[start : start + depth_chunk]
        chunk_c = flat_codes[start : start + depth_chunk]
        depth_parts.append(model.depth_hiddens(chunk_h, chunk_c))
    depth = torch.cat(depth_parts, dim=0)  # [T, 7, H]

    return torch.cat([lm_hidden, depth.reshape(1, frames, -1)], dim=-1)


@torch.no_grad()
def synthesize(
    frame_hiddens: torch.Tensor,
    condition_encoder,
    transformer,
    scheduler,
    vocoder,
    generator: torch.Generator | None = None,
    num_steps: int = FM_STEPS,
    cfg_scale: float = FM_CFG_SCALE,
    progress: bool = True,
) -> torch.Tensor:
    """frame_hiddens [1, T, 8H] → stereo waveform [1, 2, samples] @ 44.1 kHz."""
    from tqdm import tqdm

    device = transformer.device
    dtype = transformer.dtype
    starts = chunk_starts(frame_hiddens.shape[1])
    previous_latent = previous_condition = None
    latent_chunks: list[torch.Tensor] = []

    bar = tqdm(total=len(starts) * num_steps, desc="FM denoise", disable=not progress)
    for start in starts:
        window = frame_hiddens[:, start : start + CHUNK_FRAMES].to(device)
        condition = condition_encoder(window.to(condition_encoder.dtype)).to(dtype)

        overlap = 0
        if previous_latent is not None:
            overlap = min(previous_latent.shape[-1], condition.shape[1])
            condition[:, :overlap] = previous_condition[:, :overlap]

        latents = torch.randn(
            (1, transformer.config.in_channels, condition.shape[1]),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        noise_prompt = latents[..., :overlap].clone() if overlap else None

        scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / num_steps, num_steps), device=device)
        both_condition = torch.cat([condition, torch.zeros_like(condition)])
        for t in scheduler.timesteps:
            if overlap:
                time_value = t.to(dtype)
                latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * time_value) * noise_prompt + (
                    time_value * previous_latent[..., :overlap]
                )
            velocity = transformer(
                hidden_states=latents.repeat(2, 1, 1),
                timestep=t.expand(2).to(dtype),
                encoder_hidden_states=both_condition,
                return_dict=False,
            )[0]
            velocity = velocity[1:2] + cfg_scale * (velocity[0:1] - velocity[1:2])
            latents = scheduler.step(velocity, t, latents, return_dict=False)[0].to(dtype)
            bar.update()

        if overlap:
            latents[..., :overlap] = previous_latent[..., :overlap]
        carry_start = max(0, latents.shape[-1] - 2 * OVERLAP_LATENT)
        carry_end = max(carry_start, latents.shape[-1] - OVERLAP_LATENT)
        previous_latent = latents[..., carry_start:carry_end]
        previous_condition = condition[:, carry_start:carry_end]
        latent_chunks.append(latents)
    bar.close()

    parts = []
    for index, chunk in enumerate(latent_chunks):
        waveform = vocoder(chunk.to(vocoder.dtype))
        left, right = crop_bounds(index, len(latent_chunks), chunk.shape[-1])
        parts.append(waveform[..., left:right])
    return torch.cat(parts, dim=-1).float().clamp(-1.0, 1.0)


def load_synthesis_components(root=None, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
    """condition encoder + FM transformer + scheduler + vocoder from the
    diffusers-format subfolders of the local MiniMax-Music3 checkout."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models import MiniMaxMusic3Transformer1DModel, MiniMaxMusic3Vocoder
    from diffusers.models.condition_embedders import MiniMaxMusic3ConditionEncoder

    from .loading import models_dir

    root = root or models_dir()
    condition_encoder = MiniMaxMusic3ConditionEncoder.from_pretrained(
        root / "condition_encoder", torch_dtype=dtype
    ).to(device)
    transformer = MiniMaxMusic3Transformer1DModel.from_pretrained(
        root / "transformer", torch_dtype=dtype
    ).to(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler")
    vocoder = MiniMaxMusic3Vocoder.from_pretrained(root / "vocoder", torch_dtype=dtype).to(device)
    return condition_encoder, transformer, scheduler, vocoder


def main() -> None:
    import argparse
    from pathlib import Path

    import soundfile
    from safetensors import safe_open

    from .loading import load_music3_ar, load_tokenizer
    from .prompt import encode_prompt

    parser = argparse.ArgumentParser(description="Synthesize wavs from cached code sequences.")
    parser.add_argument("codes_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("cache/wavs"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=FM_STEPS)
    parser.add_argument("--cfg", type=float, default=FM_CFG_SCALE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-quant", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    model = load_music3_ar(quantize=not args.no_quant, device=args.device, with_depth=True)
    components = load_synthesis_components(device=args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.codes_dir.glob("*.safetensors"))
    if args.limit:
        paths = paths[: args.limit]
    done = 0
    for path in paths:
        target = args.out / f"{path.stem}.wav"
        if target.exists():  # resumable synth pass
            continue
        with safe_open(path, framework="pt") as f:
            codes = f.get_tensor("codes").long()
            meta = f.metadata() or {}
        prompt_ids = torch.tensor(
            [encode_prompt(tokenizer, meta.get("caption", ""), meta.get("lyrics", ""))],
            device=args.device,
        )
        hiddens = collect_frame_hiddens(model, prompt_ids, codes.unsqueeze(0).to(args.device))
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        waveform = synthesize(
            hiddens, *components, generator=generator, num_steps=args.steps, cfg_scale=args.cfg
        )
        soundfile.write(str(target), waveform.squeeze(0).cpu().numpy().T, 44100)
        done += 1
        print(f"{target.name}: {waveform.shape[-1] / 44100:.1f}s")
    print(f"synthesized {done} wavs → {args.out}")


if __name__ == "__main__":
    main()
