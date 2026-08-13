"""Prompt template and token constants for MiniMax-Music3's global LM.

The template and special-token IDs are model interface facts (same in
sglang-omni and ComfyUI):

    <|im_start|><|caption_start|>{caption}<|caption_end|>
    <|lyrics_start|>[start]\n{lyrics}<|lyrics_end|><|im_end|><|audio_start|>{audio codes}

The unconditional (CFG) stream replaces everything between the first token
and the trailing ``<|im_end|><|audio_start|>`` with ``<|audio_cfg|>``.
"""

from __future__ import annotations

import re

SPECIAL_TOKEN_IDS = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|audio_cfg|>": 151654,
    "<|audio_start|>": 151669,
    "<|audio_end|>": 151670,
    "<|caption_start|>": 151671,
    "<|caption_end|>": 151672,
    "<|lyrics_start|>": 151673,
    "<|lyrics_end|>": 151674,
}

AUDIO_CODE_OFFSET = 151675
C0_VOCAB_SIZE = 16384  # semantic codebook (codebook 0), predicted by the 8B global LM
AUDIO_VOCAB_SIZE = 1024  # acoustic codebooks 1..7, predicted by the depth decoder
NUM_CODEBOOKS = 8
AUDIO_FRAMES_PER_SECOND = 25
MAX_PROMPT_TOKENS = 5000
MAX_AUDIO_FRAMES = 9000

_LYRIC_TAG_RE = re.compile(r"\s*(\[[^\]]+\])\s*")


def normalize_lyrics(lyrics: str) -> str:
    """Lowercase section tags, one segment per line, ensure the [start] head."""
    parts = _LYRIC_TAG_RE.split(lyrics)
    lines = [part.lower() if part.startswith("[") else part.strip() for part in parts if part.strip()]
    return "[start]\n" + "\n".join(lines)


def clean_caption(caption: str) -> str:
    """Collapse blank runs; captions are expected to be plain prose."""
    return re.sub(r"\n{2,}", "\n", caption.strip())


def build_prompt(caption: str, lyrics: str) -> str:
    return (
        "<|im_start|><|caption_start|>"
        f"{clean_caption(caption)}"
        "<|caption_end|><|lyrics_start|>"
        f"{normalize_lyrics(lyrics)}"
        "<|lyrics_end|><|im_end|><|audio_start|>"
    )


def encode_prompt(tokenizer, caption: str, lyrics: str) -> list[int]:
    ids = tokenizer.encode(build_prompt(caption, lyrics), add_special_tokens=False)
    if len(ids) > MAX_PROMPT_TOKENS:
        raise ValueError(f"prompt has {len(ids)} tokens; maximum is {MAX_PROMPT_TOKENS}")
    return ids


def uncond_ids(ids: list[int]) -> list[int]:
    """CFG stream: keep first token and trailing <|im_end|><|audio_start|>."""
    cfg = SPECIAL_TOKEN_IDS["<|audio_cfg|>"]
    return [ids[0]] + [cfg] * (len(ids) - 3) + ids[-2:]


def validate_tokenizer(tokenizer) -> None:
    for token, expected in SPECIAL_TOKEN_IDS.items():
        got = tokenizer.convert_tokens_to_ids(token)
        if got != expected:
            raise ValueError(f"tokenizer mismatch for {token}: expected {expected}, got {got}")
