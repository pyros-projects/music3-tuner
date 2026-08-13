import pytest
import torch

from music3_tuner.ar import Music3AR, Music3ArConfig, RVQDepthDecoder


@pytest.fixture(scope="module")
def tiny():
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
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=512,
        )
    )
    return Music3AR(lm, cfg, depth_decoder=RVQDepthDecoder(cfg))


def test_global_loss_runs_and_backprops(tiny):
    prompt_ids = torch.randint(0, 200, (2, 7))
    codes = torch.randint(0, 8, (2, 5, 4))
    codes[..., 0] = torch.randint(0, 32, (2, 5))
    loss = tiny.global_loss(prompt_ids, codes)
    assert torch.isfinite(loss)
    loss.backward()
    assert tiny.audio_extra_embedding.weight.grad is not None
    assert tiny.lm.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_global_loss_label_alignment(tiny):
    """A prompt the model has memorized nothing about must yield ~uniform CE
    over the supervised targets — and adding audio_end supervision changes
    the supervised-position count from T to T+1."""
    prompt_ids = torch.randint(0, 200, (1, 5))
    codes = torch.randint(0, 8, (1, 3, 4))
    codes[..., 0] = torch.randint(0, 32, (1, 3))
    with torch.no_grad():
        loss_with_end = tiny.global_loss(prompt_ids, codes, supervise_audio_end=True)
        loss_without = tiny.global_loss(prompt_ids, codes, supervise_audio_end=False)
    assert torch.isfinite(loss_with_end) and torch.isfinite(loss_without)
    assert loss_with_end.item() != pytest.approx(loss_without.item())


def test_left_padding_matches_unpadded(tiny):
    """Left-padded prompt must produce the same loss as the unpadded one."""
    torch.manual_seed(1)
    prompt = torch.randint(0, 200, (1, 6))
    codes = torch.randint(0, 8, (1, 4, 4))
    codes[..., 0] = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        plain = tiny.global_loss(prompt, codes)
        padded = torch.cat([torch.full((1, 3), 7, dtype=torch.long), prompt], dim=1)
        mask = torch.cat([torch.zeros(1, 3, dtype=torch.bool), torch.ones(1, 6, dtype=torch.bool)], dim=1)
        masked = tiny.global_loss(padded, codes, prompt_mask=mask)
    assert plain.item() == pytest.approx(masked.item(), abs=2e-3)


def test_depth_loss_runs(tiny):
    hidden = torch.randn(6, 64)
    codes = torch.randint(0, 8, (6, 4))
    codes[:, 0] = torch.randint(0, 32, (6,))
    loss = tiny.depth_loss(hidden, codes)
    assert torch.isfinite(loss)


def test_embed_frames_shape_and_scale(tiny):
    codes = torch.zeros(1, 3, 4, dtype=torch.long)
    frames = tiny.embed_frames(codes)
    assert frames.shape == (1, 3, 64)
    # scale = num_codebooks ** -0.5
    manual = (
        tiny.lm.get_input_embeddings()(torch.tensor([300]))
        + tiny.audio_extra_embedding(torch.tensor([0]))
        + tiny.audio_extra_embedding(torch.tensor([8]))
        + tiny.audio_extra_embedding(torch.tensor([16]))
    ) * 0.5
    assert torch.allclose(frames[0, 0], manual[0], atol=1e-5)
