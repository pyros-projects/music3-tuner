from types import SimpleNamespace

import pytest
import torch

import music3_tuner.generate_codes as generate_codes
from music3_tuner.prompt import AUDIO_CODE_OFFSET, C0_VOCAB_SIZE, SPECIAL_TOKEN_IDS


class OutputHead:
    def __init__(self):
        self.weight = torch.empty(AUDIO_CODE_OFFSET + C0_VOCAB_SIZE, 1)

    def __call__(self, hidden):
        return torch.zeros(*hidden.shape[:-1], self.weight.shape[0], device=hidden.device)


class GlobalDecoder:
    def __init__(self):
        self.inputs = []

    def __call__(self, inputs_embeds, past_key_values, use_cache):
        self.inputs.append(inputs_embeds)
        return SimpleNamespace(last_hidden_state=inputs_embeds, past_key_values=past_key_values)


class LanguageModel:
    def __init__(self):
        self.head = OutputHead()
        self.decoder = GlobalDecoder()

    def get_output_embeddings(self):
        return self.head

    def get_decoder(self):
        return self.decoder


class AudioHead:
    def __call__(self, hidden):
        return torch.zeros(hidden.shape[0], 1024, device=hidden.device)


class DepthDecoder:
    def __init__(self):
        self.projection = torch.nn.Identity()
        self.audio_heads = [AudioHead() for _ in range(7)]

    def __call__(self, sequence):
        return sequence


class Model:
    def __init__(self):
        self.cfg = SimpleNamespace(
            audio_end_token=SPECIAL_TOKEN_IDS["<|audio_end|>"],
            num_codebooks=8,
            audio_vocab_size=1024,
        )
        self.lm = LanguageModel()
        self.depth_decoder = DepthDecoder()

    def _embed_tokens(self, ids):
        return ids.float().unsqueeze(-1)

    def audio_extra_embedding(self, ids):
        return ids.float().unsqueeze(-1)

    def embed_frames(self, frames):
        return frames.float().sum(dim=-1, keepdim=True)


def frame_samples(c0, acoustic_start):
    return [AUDIO_CODE_OFFSET + c0, *range(acoustic_start, acoustic_start + 7)]


def install_samples(monkeypatch, samples):
    values = iter(samples)

    def sample(logits, top_k, generator):
        return torch.tensor([next(values)], device=logits.device)

    monkeypatch.setattr(generate_codes, "sample_topk", sample)
    monkeypatch.setattr(generate_codes, "tqdm", lambda iterable, **kwargs: iterable)


def test_generate_separates_official_primer_and_natural_end(monkeypatch):
    install_samples(
        monkeypatch,
        frame_samples(10, 1)
        + frame_samples(20, 11)
        + [SPECIAL_TOKEN_IDS["<|audio_end|>"]],
    )
    model = Model()

    result = generate_codes.generate(model, [1, 2, 3], max_frames=4, seed=7, device="cpu")

    assert result.primer_codes.tolist() == [[10, 1, 2, 3, 4, 5, 6, 7]]
    assert result.codes.tolist() == [[20, 11, 12, 13, 14, 15, 16, 17]]
    assert result.ended is True
    assert [inputs.shape[1] for inputs in model.lm.decoder.inputs] == [3, 1, 1]
    assert generate_codes.generation_metadata(
        result, caption="caption", lyrics="lyrics", max_frames=4, seed=7
    ) == {
        "caption": "caption",
        "lyrics": "lyrics",
        "cache_version": "2",
        "termination": "audio_end",
        "max_frames": "4",
        "seed": "7",
    }


def test_generate_marks_hard_cap_without_extra_decode(monkeypatch):
    install_samples(
        monkeypatch,
        frame_samples(10, 1) + frame_samples(20, 11) + frame_samples(30, 21),
    )
    model = Model()

    result = generate_codes.generate(model, [1, 2, 3], max_frames=2, seed=0, device="cpu")

    assert result.primer_codes[:, 0].tolist() == [10]
    assert result.codes[:, 0].tolist() == [20, 30]
    assert result.ended is False
    assert [inputs.shape[1] for inputs in model.lm.decoder.inputs] == [3, 1, 1]
    assert generate_codes.generation_metadata(
        result, caption="", lyrics="", max_frames=2, seed=0
    )["termination"] == "max_frames"


def test_generate_rejects_frame_counts_outside_reference_contract():
    with pytest.raises(ValueError, match="between 1 and 9000"):
        generate_codes.generate(Model(), [1, 2, 3], max_frames=9001, seed=0, device="cpu")
