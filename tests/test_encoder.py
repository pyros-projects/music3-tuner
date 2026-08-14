import torch

from music3_tuner.encoder import FEATURE_RATE, CodesEncoder, latent_to_features


def tiny() -> CodesEncoder:
    torch.manual_seed(0)
    return CodesEncoder(
        latent_dim=16, d_model=32, num_layers=2, num_heads=4, ff_dim=64,
        c0_vocab=64, audio_vocab=8, num_codebooks=4,
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
