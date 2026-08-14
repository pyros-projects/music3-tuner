"""Dataset over cached code sequences.

Cache format: one .safetensors per track with tensor "codes" [T, 8] (int32)
and string metadata {"caption": ..., "lyrics": ...}. V2 generated caches
also carry the non-emitted "primer_codes" [1, 8] warm-up frame and explicit
termination metadata. Legacy caches remain readable but their unknown ending
is never supervised as a natural audio end.

Caption sidecar parser for Pyro's ace-style .txt files is here too
(caption:/genre:/bpm:/key:/signature:/lyrics: fields).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .prompt import MAX_AUDIO_FRAMES, SPECIAL_TOKEN_IDS, encode_prompt, uncond_ids

SIDECAR_FIELDS = ("caption", "genre", "bpm", "key", "signature", "is_instrumental")


def parse_sidecar(path: Path) -> dict[str, str]:
    """Parse `field: value` lines; everything after `lyrics:` is the lyrics."""
    fields: dict[str, str] = {}
    lyrics_lines: list[str] = []
    in_lyrics = False
    for line in path.read_text().splitlines():
        if in_lyrics:
            lyrics_lines.append(line)
            continue
        if line.strip().lower().startswith("lyrics:"):
            in_lyrics = True
            rest = line.split(":", 1)[1].strip()
            if rest:
                lyrics_lines.append(rest)
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip().lower() in SIDECAR_FIELDS:
                fields[key.strip().lower()] = value.strip()
    fields["lyrics"] = "\n".join(lyrics_lines).strip()
    return fields


def compose_caption(fields: dict[str, str]) -> str:
    """Fold the structured sidecar fields into a Music3 caption."""
    parts = [fields.get("caption", "").strip()]
    meta = [
        f"{name}: {fields[name]}"
        for name in ("genre", "bpm", "key", "signature")
        if fields.get(name)
    ]
    if meta:
        parts.append(". ".join(meta) + ".")
    return "\n".join(p for p in parts if p)


@dataclass
class CodesBatch:
    prompt_ids: torch.Tensor  # [B, P] left-padded
    prompt_mask: torch.Tensor  # [B, P] bool
    codes: torch.Tensor  # [B, T, 8]
    primer_codes: torch.Tensor  # [B, 1, 8], zero-filled where mask is False
    primer_mask: torch.Tensor  # [B] bool
    supervise_audio_end: bool


class CodesDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        tokenizer,
        max_frames: int = MAX_AUDIO_FRAMES,
        uncond_p: float = 0.0,
        seed: int = 0,
    ):
        self.paths = sorted(Path(cache_dir).glob("*.safetensors"))
        if not self.paths:
            raise FileNotFoundError(f"no cached code sequences under {cache_dir}")
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.uncond_p = uncond_p
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        from safetensors import safe_open

        with safe_open(self.paths[index], framework="pt") as f:
            codes = f.get_tensor("codes").long()
            primer_codes = f.get_tensor("primer_codes").long() if "primer_codes" in f.keys() else None
            meta = f.metadata() or {}
        if primer_codes is not None and primer_codes.shape != (1, codes.shape[-1]):
            raise ValueError(
                f"{self.paths[index]} has primer_codes {tuple(primer_codes.shape)}; "
                f"expected (1, {codes.shape[-1]})"
            )

        ids = encode_prompt(self.tokenizer, meta.get("caption", ""), meta.get("lyrics", ""))
        if self.uncond_p > 0 and self.rng.random() < self.uncond_p:
            ids = uncond_ids(ids)

        frames = codes.shape[0]
        start = 0
        if frames > self.max_frames:
            # The start-of-track primer is only valid for a prefix crop. Until
            # loss-masked burn-in exists, do not attach it to a mid-song crop.
            if primer_codes is None:
                start = self.rng.randrange(frames - self.max_frames + 1)
            codes = codes[start : start + self.max_frames]
        if start:
            primer_codes = None
        reaches_end = meta.get("termination") == "audio_end" and start + len(codes) == frames
        return {
            "prompt_ids": ids,
            "codes": codes,
            "primer_codes": primer_codes,
            "reaches_end": reaches_end,
        }


def collate_codes(items: list[dict], pad_id: int = SPECIAL_TOKEN_IDS["<|im_end|>"]) -> CodesBatch:
    """Left-pad prompts; code lengths must match across the batch (use
    batch_size=1 with gradient accumulation for ragged tracks)."""
    frames = {item["codes"].shape[0] for item in items}
    assert len(frames) == 1, f"ragged code lengths {frames}; use batch_size=1"
    max_prompt = max(len(item["prompt_ids"]) for item in items)
    prompt_ids = torch.full((len(items), max_prompt), pad_id, dtype=torch.long)
    prompt_mask = torch.zeros((len(items), max_prompt), dtype=torch.bool)
    primer_codes = torch.zeros((len(items), 1, items[0]["codes"].shape[-1]), dtype=torch.long)
    primer_mask = torch.zeros(len(items), dtype=torch.bool)
    for row, item in enumerate(items):
        ids = torch.tensor(item["prompt_ids"], dtype=torch.long)
        prompt_ids[row, max_prompt - len(ids) :] = ids
        prompt_mask[row, max_prompt - len(ids) :] = True
        primer = item.get("primer_codes")
        if primer is not None:
            primer_codes[row] = primer
            primer_mask[row] = True
    return CodesBatch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        codes=torch.stack([item["codes"] for item in items]),
        primer_codes=primer_codes,
        primer_mask=primer_mask,
        supervise_audio_end=all(item["reaches_end"] for item in items),
    )
