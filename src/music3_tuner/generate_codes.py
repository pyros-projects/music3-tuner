"""Sample the AR (global LM + depth decoder) to produce (codes, prompt) pairs.

Two jobs:
1. Trainer smoke data — cached sequences in the exact dataset format.
2. Encoder-distillation data — model-labeled (audio, codes) pairs are the
   only way to learn the unreleased audio→codes tokenizer.

Mirrors the reference sampler: CFG 1.5 (uncond = <|audio_cfg|>-masked
prompt), top-k 50, c0 vocab-masked to audio slots + <|audio_end|>.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dataset import compose_caption, parse_sidecar
from .loading import load_music3_ar, load_tokenizer
from .prompt import AUDIO_CODE_OFFSET, C0_VOCAB_SIZE, MAX_AUDIO_FRAMES, encode_prompt, uncond_ids

CFG_SCALE = 1.5
TOP_K = 50
CACHE_VERSION = "2"


@dataclass(frozen=True)
class GenerationResult:
    codes: torch.Tensor  # emitted audio frames [T, 8]
    primer_codes: torch.Tensor  # one non-emitted warm-up frame [1, 8]
    ended: bool  # True only when the model sampled <|audio_end|>


def sample_topk(logits: torch.Tensor, top_k: int, generator: torch.Generator) -> torch.Tensor:
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    logits = logits.masked_fill(logits < threshold, -float("inf"))
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(model, prompt: list[int], max_frames: int, seed: int, device: str) -> GenerationResult:
    from transformers import DynamicCache

    if not 1 <= max_frames <= MAX_AUDIO_FRAMES:
        raise ValueError(f"max_frames must be between 1 and {MAX_AUDIO_FRAMES}, got {max_frames}")
    cfg = model.cfg
    generator = torch.Generator(device=device).manual_seed(seed)
    conditioned = torch.tensor(prompt, device=device)
    ids = torch.stack([conditioned, torch.tensor(uncond_ids(prompt), device=device)])
    embeds = model._embed_tokens(ids)

    vocab_size = model.lm.get_output_embeddings().weight.shape[0]
    vocab_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
    vocab_mask[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + C0_VOCAB_SIZE] = False
    vocab_mask[cfg.audio_end_token] = False

    cache = DynamicCache()
    frames: list[torch.Tensor] = []
    primer_codes = None
    ended = False
    # Reference contract: the first sampled frame is fed back as a primer but
    # is not emitted. Generation therefore takes at most max_frames + 1 steps.
    for _ in tqdm(range(max_frames + 1), desc="AR sampling"):
        output = model.lm.get_decoder()(inputs_embeds=embeds, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        hidden = output.last_hidden_state[:, -1]

        logits = model.lm.get_output_embeddings()(hidden).float()
        logits = logits.masked_fill(vocab_mask, -float("inf"))
        guided = logits[1:2] + (logits[0:1] - logits[1:2]) * CFG_SCALE
        threshold = torch.topk(logits[0:1], TOP_K, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(logits[0:1] < threshold, -float("inf"))
        token = sample_topk(guided, TOP_K, generator)
        if int(token.item()) == cfg.audio_end_token:
            ended = True
            break
        c0 = (token - AUDIO_CODE_OFFSET).repeat(2)

        # depth decoder: codebooks 1..7, CFG per step
        decoder = model.depth_decoder
        c0_embed = model._embed_tokens(c0 + AUDIO_CODE_OFFSET)
        sequence = [decoder.projection(hidden).unsqueeze(1), decoder.projection(c0_embed).unsqueeze(1)]
        codes = [c0]
        for book in range(1, cfg.num_codebooks):
            out = decoder(torch.cat(sequence, dim=1))[:, -1]
            book_logits = decoder.audio_heads[book - 1](out).float()
            guided_book = book_logits[1:2] + (book_logits[0:1] - book_logits[1:2]) * CFG_SCALE
            code = sample_topk(guided_book, TOP_K, generator).repeat(2)
            codes.append(code)
            if book < cfg.num_codebooks - 1:
                embedding = model.audio_extra_embedding(code + (book - 1) * cfg.audio_vocab_size)
                sequence.append(decoder.projection(embedding.to(hidden.dtype)).unsqueeze(1))

        frame = torch.stack(codes, dim=1)  # [2, 8]
        if primer_codes is None:
            primer_codes = frame[0].cpu().unsqueeze(0)
        else:
            frames.append(frame[0].cpu())
            if len(frames) >= max_frames:
                break
        embeds = model.embed_frames(frame.unsqueeze(1))

    if not frames:
        raise RuntimeError("zero frames generated")
    assert primer_codes is not None
    return GenerationResult(torch.stack(frames), primer_codes, ended)


def generation_metadata(
    result: GenerationResult, *, caption: str, lyrics: str, max_frames: int, seed: int
) -> dict[str, str]:
    return {
        "caption": caption,
        "lyrics": lyrics,
        "cache_version": CACHE_VERSION,
        "termination": "audio_end" if result.ended else "max_frames",
        "max_frames": str(max_frames),
        "seed": str(seed),
    }


# Generic section-tagged lyric sets for template-driven corpus generation
# (official structured captions carry no lyrics). Rotated per track; a share
# of tracks uses the official [Instrumental] section tag instead.
LYRIC_POOL = [
    "[Verse]\nCity lights are calling out my name tonight\n"
    "Every shadow knows the road I take\n"
    "[Chorus]\nWe rise, we fall, we run again\n"
    "Nothing here can hold us down\n"
    "[Outro]\nNothing here can hold us down",
    "[Verse]\nMorning breaks across the silent water\n"
    "Footsteps fading on the empty shore\n"
    "[Chorus]\nCarry me home where the wild wind blows\n"
    "Carry me home tonight\n"
    "[Bridge]\nAnd if the sky should fall, I'll still be standing here",
    "[Verse]\nOne more mile on this broken highway\n"
    "Dust and echoes in the rear-view glass\n"
    "[Pre-Chorus]\nHold on, hold on\n"
    "[Chorus]\nWe are the fire that never sleeps\n"
    "Burning brighter through the night",
    "[Verse]\nWhisper soft, the night is young and endless\n"
    "Every heartbeat writes another line\n"
    "[Chorus]\nStay with me until the morning finds us\n"
    "Stay with me tonight\n"
    "[Solo]\n[Outro]\nStay with me tonight",
]


def iter_prompts(args) -> list[tuple[str, str, str]]:
    """Yield (name, caption, lyrics) from sidecars or official templates."""
    if args.templates:
        templates = sorted(args.templates.glob("*.txt"))
        if args.shuffle:
            import random

            random.Random(args.seed).shuffle(templates)
        if args.limit:
            templates = templates[: args.limit]
        jobs = []
        for index, path in enumerate(templates):
            if args.instrumental_ratio and (index % max(1, round(1 / args.instrumental_ratio)) == 0):
                lyrics = "[Instrumental]"
            else:
                lyrics = LYRIC_POOL[index % len(LYRIC_POOL)]
            jobs.append((path.stem, path.read_text().strip(), lyrics))
        return jobs
    sidecars = sorted(args.sidecar_dir.glob("*.txt"))
    if args.limit:
        sidecars = sidecars[: args.limit]
    jobs = []
    for sidecar in sidecars:
        fields = parse_sidecar(sidecar)
        jobs.append((sidecar.stem, compose_caption(fields), fields.get("lyrics", "")))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar_dir", type=Path, nargs="?", help="dir with ace-style .txt caption files")
    parser.add_argument("--templates", type=Path, default=None, help="dir of official structured-caption templates (captions verbatim, lyrics from a generic pool)")
    parser.add_argument("--instrumental-ratio", type=float, default=0.25, help="share of template tracks generated with [Instrumental] lyrics")
    parser.add_argument("--shuffle", action="store_true", help="shuffle template order (seeded)")
    parser.add_argument("--out", type=Path, default=Path("cache/codes"))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-quant", action="store_true")
    args = parser.parse_args()
    if not args.templates and not args.sidecar_dir:
        parser.error("give a sidecar dir or --templates")

    from safetensors.torch import save_file

    tokenizer = load_tokenizer()
    model = load_music3_ar(quantize=not args.no_quant, device=args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    max_frames = int(args.seconds * 25)
    for index, (name, caption, lyrics) in enumerate(iter_prompts(args)):
        seed = args.seed + index
        target = args.out / f"{name}_s{seed}.safetensors"
        if target.exists():  # idempotent corpus runs: resume after interruption
            continue
        prompt = encode_prompt(tokenizer, caption, lyrics)
        result = generate(model, prompt, max_frames, seed, args.device)
        save_file(
            {
                "codes": result.codes.to(torch.int32),
                "primer_codes": result.primer_codes.to(torch.int32),
            },
            str(target),
            metadata=generation_metadata(
                result, caption=caption, lyrics=lyrics, max_frames=max_frames, seed=seed
            ),
        )
        print(
            f"{target.stem}: {result.codes.shape[0]} frames "
            f"({result.codes.shape[0] / 25:.1f}s, "
            f"{'natural end' if result.ended else 'capped'})"
        )


if __name__ == "__main__":
    main()
