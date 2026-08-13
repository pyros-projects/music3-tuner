from pathlib import Path

import pytest

from music3_tuner.prompt import (
    SPECIAL_TOKEN_IDS,
    build_prompt,
    normalize_lyrics,
    uncond_ids,
)


def test_build_prompt_template():
    prompt = build_prompt("dark synthwave", "[Verse] neon lights ahead")
    assert prompt.startswith("<|im_start|><|caption_start|>dark synthwave<|caption_end|>")
    assert "<|lyrics_start|>[start]\n[verse]\nneon lights ahead<|lyrics_end|>" in prompt
    assert prompt.endswith("<|im_end|><|audio_start|>")


def test_normalize_lyrics_lowercases_tags_and_adds_start():
    assert normalize_lyrics("[Chorus] Fire") == "[start]\n[chorus]\nFire"


def test_uncond_ids_keeps_frame():
    ids = [1, 10, 11, 12, 2, 3]
    out = uncond_ids(ids)
    cfg = SPECIAL_TOKEN_IDS["<|audio_cfg|>"]
    assert out == [1, cfg, cfg, cfg, 2, 3]
    assert len(out) == len(ids)


@pytest.mark.weights
def test_tokenizer_special_ids():
    from music3_tuner.loading import load_tokenizer, models_dir

    if not models_dir().exists():
        pytest.skip("MiniMaxM3 checkout missing")
    tokenizer = load_tokenizer()  # load_tokenizer validates all special ids
    ids = tokenizer.encode(build_prompt("test", "la la"), add_special_tokens=False)
    assert ids[0] == SPECIAL_TOKEN_IDS["<|im_start|>"]
    assert ids[-1] == SPECIAL_TOKEN_IDS["<|audio_start|>"]
    assert ids[-2] == SPECIAL_TOKEN_IDS["<|im_end|>"]
