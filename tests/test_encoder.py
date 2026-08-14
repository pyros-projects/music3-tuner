import json

import pytest
import torch
import torch.nn.functional as F

from music3_tuner.encoder import (
    FEATURE_RATE,
    CodesEncoder,
    latent_to_features,
    load_encoder,
    save_encoder,
    windowed_logprobs,
)


def tiny(**kwargs) -> CodesEncoder:
    torch.manual_seed(0)
    return CodesEncoder(
        latent_dim=16, d_model=32, num_layers=2, num_heads=4, ff_dim=64,
        c0_vocab=64, audio_vocab=8, num_codebooks=4,
        **kwargs,
    )


def test_forward_shapes():
    model = tiny()
    features = torch.randn(2, 16, FEATURE_RATE * 10)
    c0_logits, acoustic_logits = model(features)
    assert c0_logits.shape == (2, 10, 64)
    assert acoustic_logits.shape == (2, 10, 3, 8)


def test_gradients_flow():
    model = tiny()
    features = torch.randn(1, 16, FEATURE_RATE * 6)
    c0_logits, acoustic_logits = model(features)
    (c0_logits.mean() + acoustic_logits.mean()).backward()
    assert model.stem[0].weight.grad is not None
    assert model.c0_head.weight.grad is not None


def test_latent_to_features_length():
    latent = torch.randn(16, 87)  # ~86.13 Hz worth of one second
    features = latent_to_features(latent, 25)
    assert features.shape == (16, 25 * FEATURE_RATE)


class ConditioningSpy:
    config = {"c0_vocab": 2, "audio_vocab": 2, "num_codebooks": 2, "window": 2}

    def __init__(self):
        self.unconditioned_calls = 0
        self.teachers = []

    def __call__(self, features, c0_teacher=None):
        frames = features.shape[-1] // FEATURE_RATE
        if c0_teacher is None:
            patterns = (
                torch.tensor([[4.0, 0.0], [1.0, 0.0]]),
                torch.tensor([[0.0, 4.0], [0.0, 4.0]]),
            )
            c0_logits = patterns[self.unconditioned_calls][:frames].unsqueeze(0)
            self.unconditioned_calls += 1
            acoustic_ids = c0_logits.argmax(-1)
        else:
            self.teachers.append(c0_teacher.squeeze(0).tolist())
            c0_logits = torch.zeros(1, frames, 2)
            acoustic_ids = c0_teacher
        acoustic_logits = F.one_hot(acoustic_ids, 2).float().mul(10).unsqueeze(2)
        return c0_logits, acoustic_logits


def test_windowed_acoustics_use_final_averaged_c0():
    encoder = ConditioningSpy()
    features = torch.zeros(1, FEATURE_RATE * 3)

    c0_logprobs, acoustic_logprobs = windowed_logprobs(encoder, features, 3)

    assert c0_logprobs.argmax(-1).tolist() == [0, 1, 1]
    assert encoder.teachers == [[0, 1], [1, 1]]
    assert acoustic_logprobs.argmax(-1).squeeze(-1).tolist() == [0, 1, 1]


class AcousticAveragingSpy:
    config = {"c0_vocab": 2, "audio_vocab": 2, "num_codebooks": 2, "window": 2}

    def __init__(self):
        self.conditioned_calls = 0

    def __call__(self, features, c0_teacher=None):
        frames = features.shape[-1] // FEATURE_RATE
        c0_logits = torch.zeros(1, frames, 2)
        if c0_teacher is None:
            acoustic = torch.zeros(1, frames, 1, 2)
        else:
            patterns = (
                torch.tensor([[[[2.0, 0.0]], [[4.0, 0.0]]]]),
                torch.tensor([[[[0.0, 2.0]], [[0.0, 4.0]]]]),
            )
            acoustic = patterns[self.conditioned_calls][:, :frames]
            self.conditioned_calls += 1
        return c0_logits, acoustic


def test_windowed_acoustic_logprobs_are_numerically_averaged():
    encoder = AcousticAveragingSpy()
    features = torch.zeros(1, FEATURE_RATE * 3)

    _, acoustic_logprobs = windowed_logprobs(encoder, features, 3)

    expected_overlap = (
        F.log_softmax(torch.tensor([4.0, 0.0]), dim=-1)
        + F.log_softmax(torch.tensor([0.0, 2.0]), dim=-1)
    ) / 2
    assert torch.allclose(acoustic_logprobs[1, 0], expected_overlap)


def test_saved_window_roundtrip_and_legacy_default(tmp_path):
    save_encoder(tiny(window=7), tmp_path)
    assert load_encoder(tmp_path).config["window"] == 7

    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("window")
    config_path.write_text(json.dumps(config))
    with pytest.warns(RuntimeWarning, match="assuming 512"):
        assert load_encoder(tmp_path).config["window"] == 512
