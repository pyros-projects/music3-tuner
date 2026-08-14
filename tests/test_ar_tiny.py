import pytest
import torch
import torch.nn.functional as F

from music3_tuner.ar import Music3AR, Music3ArConfig, RVQDepthDecoder
from music3_tuner.synth import collect_frame_hiddens
from music3_tuner.train import _trainable_adapter_parameters


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


def test_global_loss_with_primer_supervises_complete_sequence(tiny):
    prompt_ids = torch.randint(0, 200, (1, 5))
    primer = torch.randint(0, 8, (1, 1, 4))
    primer[..., 0] = torch.randint(0, 32, (1, 1))
    codes = torch.randint(0, 8, (1, 3, 4))
    codes[..., 0] = torch.randint(0, 32, (1, 3))

    with torch.no_grad():
        actual = tiny.global_loss(prompt_ids, codes, primer_codes=primer)
        sequence = torch.cat([primer, codes], dim=1)
        embeds = torch.cat([tiny._embed_tokens(prompt_ids), tiny.embed_frames(sequence)], dim=1)
        hidden = tiny.lm.get_decoder()(inputs_embeds=embeds).last_hidden_state
        predictors = hidden[:, prompt_ids.shape[1] - 1 :]
        targets = torch.cat(
            [
                sequence[..., 0] + tiny.cfg.audio_code_offset,
                torch.full((1, 1), tiny.cfg.audio_end_token),
            ],
            dim=1,
        )
        expected = F.cross_entropy(
            tiny.lm.get_output_embeddings()(predictors).float().flatten(0, 1),
            targets.flatten(),
        )

    assert actual.item() == pytest.approx(expected.item(), abs=1e-6)


def test_global_loss_capped_primer_sequence_omits_audio_end(tiny):
    prompt_ids = torch.randint(0, 200, (1, 5))
    primer = torch.randint(0, 8, (1, 1, 4))
    primer[..., 0] = torch.randint(0, 32, (1, 1))
    codes = torch.randint(0, 8, (1, 3, 4))
    codes[..., 0] = torch.randint(0, 32, (1, 3))

    with torch.no_grad():
        actual = tiny.global_loss(
            prompt_ids,
            codes,
            primer_codes=primer,
            supervise_audio_end=False,
        )
        sequence = torch.cat([primer, codes], dim=1)
        embeds = torch.cat([tiny._embed_tokens(prompt_ids), tiny.embed_frames(sequence)], dim=1)
        hidden = tiny.lm.get_decoder()(inputs_embeds=embeds).last_hidden_state
        predictors = hidden[
            :, prompt_ids.shape[1] - 1 : prompt_ids.shape[1] + sequence.shape[1] - 1
        ]
        expected = F.cross_entropy(
            tiny.lm.get_output_embeddings()(predictors).float().flatten(0, 1),
            (sequence[..., 0] + tiny.cfg.audio_code_offset).flatten(),
        )

    assert actual.item() == pytest.approx(expected.item(), abs=1e-6)


def test_global_loss_mixes_primer_and_legacy_rows(tiny):
    prompt_ids = torch.randint(0, 200, (2, 5))
    codes = torch.randint(0, 8, (2, 3, 4))
    codes[..., 0] = torch.randint(0, 32, (2, 3))
    primers = torch.randint(0, 8, (2, 1, 4))
    primers[..., 0] = torch.randint(0, 32, (2, 1))

    with torch.no_grad():
        mixed = tiny.global_loss(
            prompt_ids,
            codes,
            primer_codes=primers,
            primer_mask=torch.tensor([False, True]),
        )
        legacy = tiny.global_loss(prompt_ids[:1], codes[:1])
        official = tiny.global_loss(prompt_ids[1:], codes[1:], primer_codes=primers[1:])

    # Legacy contributes T codes + EOS; official contributes primer + T codes + EOS.
    expected = (legacy * 4 + official * 5) / 9
    assert mixed.item() == pytest.approx(expected.item(), abs=1e-6)


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


def test_collect_frame_hiddens_uses_official_primer_alignment(tiny):
    prompt_ids = torch.randint(0, 200, (1, 5))
    primer = torch.randint(0, 8, (1, 1, 4))
    primer[..., 0] = torch.randint(0, 32, (1, 1))
    codes = torch.randint(0, 8, (1, 3, 4))
    codes[..., 0] = torch.randint(0, 32, (1, 3))

    official = collect_frame_hiddens(tiny, prompt_ids, codes, primer_codes=primer)
    legacy = collect_frame_hiddens(tiny, prompt_ids, codes)
    with torch.no_grad():
        sequence = torch.cat([primer, codes], dim=1)
        embeds = torch.cat([tiny._embed_tokens(prompt_ids), tiny.embed_frames(sequence)], dim=1)
        hidden = tiny.lm.get_decoder()(inputs_embeds=embeds).last_hidden_state
        expected_global = hidden[:, prompt_ids.shape[1] : prompt_ids.shape[1] + codes.shape[1]]
        legacy_embeds = torch.cat(
            [tiny._embed_tokens(prompt_ids), tiny.embed_frames(codes)], dim=1
        )
        legacy_hidden = tiny.lm.get_decoder()(inputs_embeds=legacy_embeds).last_hidden_state
        expected_legacy = legacy_hidden[
            :, prompt_ids.shape[1] - 1 : prompt_ids.shape[1] + codes.shape[1] - 1
        ]

    assert official.shape == legacy.shape == (1, 3, 4 * tiny.cfg.hidden_size)
    assert torch.allclose(official[..., : tiny.cfg.hidden_size], expected_global, atol=1e-6)
    assert torch.allclose(legacy[..., : tiny.cfg.hidden_size], expected_legacy, atol=1e-6)
    assert not torch.allclose(
        official[:, 0, : tiny.cfg.hidden_size], legacy[:, 0, : tiny.cfg.hidden_size]
    )


def test_qlora_optimizer_only_gets_persisted_adapter_parameters(tiny):
    from copy import deepcopy

    from peft import LoraConfig, get_peft_model

    model = deepcopy(tiny)
    model.lm = get_peft_model(
        model.lm,
        LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"),
    )
    optimizer = torch.optim.SGD(_trainable_adapter_parameters(model), lr=0.1)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    lora_parameters = [
        parameter for name, parameter in model.lm.named_parameters() if "lora_" in name
    ]

    assert not model.audio_extra_embedding.weight.requires_grad
    assert id(model.audio_extra_embedding.weight) not in optimized
    assert lora_parameters
    assert all(parameter.requires_grad and id(parameter) in optimized for parameter in lora_parameters)


def test_qlora_optimizer_state_roundtrips_through_peft(tmp_path, tiny):
    from copy import deepcopy

    from peft import LoraConfig, PeftModel, get_peft_model

    trained = deepcopy(tiny)
    trained.lm = get_peft_model(
        trained.lm,
        LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"),
    )
    optimizer = torch.optim.SGD(_trainable_adapter_parameters(trained), lr=0.1)
    prompt_ids = torch.randint(0, 200, (1, 5))
    codes = torch.randint(0, 8, (1, 3, 4))
    codes[..., 0] = torch.randint(0, 32, (1, 3))
    trained.global_loss(prompt_ids, codes).backward()
    optimizer.step()
    trained.lm.save_pretrained(tmp_path)

    reloaded = deepcopy(tiny)
    reloaded.lm = PeftModel.from_pretrained(reloaded.lm, tmp_path)
    trained_adapter = {
        name: parameter.detach()
        for name, parameter in trained.lm.named_parameters()
        if "lora_" in name
    }
    reloaded_adapter = {
        name: parameter.detach()
        for name, parameter in reloaded.lm.named_parameters()
        if "lora_" in name
    }

    assert trained_adapter.keys() == reloaded_adapter.keys()
    assert all(
        torch.equal(trained_adapter[name], reloaded_adapter[name])
        for name in trained_adapter
    )
