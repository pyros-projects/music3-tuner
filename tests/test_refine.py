import torch

from music3_tuner.ar import Music3AR, Music3ArConfig, RVQDepthDecoder
from music3_tuner.encoder import FEATURE_RATE, CodesEncoder, logprobs_to_codes, windowed_logprobs
from music3_tuner.refine import refine_codes


def tiny_ar() -> Music3AR:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Music3ArConfig(
        hidden_size=64,
        audio_code_offset=300,
        c0_vocab_size=32,
        audio_vocab_size=8,
        num_codebooks=4,
        audio_end_token=200,
        depth_num_layers=2,
        depth_num_heads=4,
        depth_intermediate_size=96,
    )
    lm = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
        )
    )
    return Music3AR(lm, cfg, depth_decoder=RVQDepthDecoder(cfg))


def test_refine_runs_and_stays_in_range():
    model = tiny_ar()
    frames = 6
    prompt_ids = torch.randint(0, 200, (1, 5))
    c0_logprobs = torch.log_softmax(torch.randn(frames, 32), dim=-1)
    acoustic_logprobs = torch.log_softmax(torch.randn(frames, 3, 8), dim=-1)
    codes = refine_codes(model, prompt_ids, c0_logprobs, acoustic_logprobs, lam=0.5, iterations=2)
    assert codes.shape == (frames, 4)
    assert 0 <= codes[:, 0].min() and codes[:, 0].max() < 32
    assert 0 <= codes[:, 1:].min() and codes[:, 1:].max() < 8


def test_refine_lambda_zero_keeps_encoder_argmax():
    model = tiny_ar()
    frames = 5
    prompt_ids = torch.randint(0, 200, (1, 4))
    c0_logprobs = torch.log_softmax(torch.randn(frames, 32), dim=-1)
    acoustic_logprobs = torch.log_softmax(torch.randn(frames, 3, 8), dim=-1)
    refined = refine_codes(model, prompt_ids, c0_logprobs, acoustic_logprobs, lam=0.0, iterations=1)
    plain = logprobs_to_codes(c0_logprobs, acoustic_logprobs)
    assert torch.equal(refined, plain)


def test_windowed_logprobs_averages_overlaps():
    torch.manual_seed(0)
    encoder = CodesEncoder(
        latent_dim=16, d_model=32, num_layers=2, num_heads=4, ff_dim=64,
        c0_vocab=64, audio_vocab=8, num_codebooks=4, dropout=0.0,
    ).eval()
    frames = 30
    features = torch.randn(16, frames * FEATURE_RATE)
    c0_lp, ac_lp = windowed_logprobs(encoder, features, frames, window=16)
    assert c0_lp.shape == (frames, 64)
    assert ac_lp.shape == (frames, 3, 8)
    assert torch.isfinite(c0_lp).all() and torch.isfinite(ac_lp).all()
    # short input: single window == direct forward
    short = torch.randn(16, 8 * FEATURE_RATE)
    c0_short, _ = windowed_logprobs(encoder, short, 8, window=16)
    direct, _ = encoder(short.unsqueeze(0))
    assert torch.allclose(c0_short, torch.log_softmax(direct.squeeze(0).float(), -1), atol=1e-5)


def test_scheduled_sampling_forward_runs():
    torch.manual_seed(0)
    encoder = CodesEncoder(
        latent_dim=16, d_model=32, num_layers=2, num_heads=4, ff_dim=64,
        c0_vocab=64, audio_vocab=8, num_codebooks=4,
    )
    features = torch.randn(2, 16, 10 * FEATURE_RATE)
    teacher = torch.randint(0, 64, (2, 10))
    c0_logits, acoustic_logits = encoder(features, c0_teacher=teacher, scheduled_p=1.0)
    assert c0_logits.shape == (2, 10, 64)
    (c0_logits.mean() + acoustic_logits.mean()).backward()