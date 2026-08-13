"""Weight/tokenizer loading against a local MiniMax-Music3 checkout.

Default layout: ~/models/MiniMaxM3 (override with MINIMAX_M3_DIR), containing
the HF repo download: language_model/ (8B Qwen3), rvq_depth_decoder/,
tokenizer/ (or qwen_7B/qwen3-8B-tokenizer-music), dav.pth, ...
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn

from .ar import Music3ArConfig, Music3AR, RVQDepthDecoder
from .prompt import validate_tokenizer


def models_dir() -> Path:
    return Path(os.environ.get("MINIMAX_M3_DIR", "~/models/MiniMaxM3")).expanduser()


def load_tokenizer(root: Path | None = None):
    from transformers import AutoTokenizer

    root = root or models_dir()
    for candidate in (root / "tokenizer", root / "qwen_7B" / "qwen3-8B-tokenizer-music"):
        if (candidate / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            validate_tokenizer(tokenizer)
            return tokenizer
    raise FileNotFoundError(f"no tokenizer found under {root}")


def _fix_rope(config) -> None:
    # language_model/config.json was written by transformers 5.x
    # (rope_parameters); older transformers fall back to rope_theta=10000,
    # which silently breaks RoPE. Patch explicitly.
    rope = getattr(config, "rope_parameters", None)
    if isinstance(rope, dict) and "rope_theta" in rope:
        config.rope_theta = rope["rope_theta"]


def load_global_lm(
    root: Path | None = None,
    quantize: bool = True,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
):
    from transformers import AutoConfig, Qwen3ForCausalLM

    root = root or models_dir()
    path = root / "language_model"
    config = AutoConfig.from_pretrained(path)
    _fix_rope(config)

    kwargs: dict = {"config": config, "torch_dtype": dtype}
    if quantize:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": device}
    model = Qwen3ForCausalLM.from_pretrained(path, **kwargs)
    if not quantize:
        model = model.to(device)
    return model


def _load_safetensors_dir(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {path} (download still running?)")
    for shard in shards:
        state.update(load_file(shard))
    return state


def load_depth_components(
    root: Path | None = None,
    cfg: Music3ArConfig | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[RVQDepthDecoder, nn.Embedding]:
    """Load the depth decoder and the codebooks-1..7 input embedding table
    from rvq_depth_decoder/. Key names are matched by suffix so diffusers
    prefix conventions don't matter."""
    cfg = cfg or Music3ArConfig()
    root = root or models_dir()
    state = _load_safetensors_dir(root / "rvq_depth_decoder")

    extra_rows = cfg.audio_vocab_size * (cfg.num_codebooks - 1)
    embedding = nn.Embedding(extra_rows, cfg.hidden_size)
    embed_key = next(
        (k for k, v in state.items() if v.ndim == 2 and v.shape == (extra_rows, cfg.hidden_size)),
        None,
    )
    if embed_key is None:
        raise KeyError(
            f"no [{extra_rows}, {cfg.hidden_size}] embedding in rvq_depth_decoder; "
            f"keys: {sorted(state)[:10]}"
        )
    embedding.weight.data.copy_(state.pop(embed_key))

    decoder = RVQDepthDecoder(cfg)
    wanted = dict(decoder.state_dict())
    remapped: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []
    renames = (
        (".attn.to_q.", ".self_attn.q_proj."),
        (".attn.to_k.", ".self_attn.k_proj."),
        (".attn.to_v.", ".self_attn.v_proj."),
        (".attn.to_out.", ".self_attn.o_proj."),
        (".gate_proj.", ".mlp.gate_proj."),
        (".up_proj.", ".mlp.up_proj."),
        (".down_proj.", ".mlp.down_proj."),
        ("position_embeddings", "pos_embedding"),
    )
    for key, value in state.items():
        renamed = key
        for old, new in renames:
            renamed = renamed.replace(old, new)
        if renamed in wanted:
            remapped[renamed] = value
            continue
        hits = [w for w in wanted if renamed.endswith(w) or w.endswith(renamed)]
        if len(hits) == 1:
            remapped[hits[0]] = value
        else:
            unmatched.append(key)
    missing = [w for w in wanted if w not in remapped]
    if missing:
        raise KeyError(
            f"depth decoder keys unmatched — missing {missing[:6]}, "
            f"checkpoint leftovers {unmatched[:6]}"
        )
    decoder.load_state_dict(remapped)
    return (
        decoder.to(device=device, dtype=dtype).eval(),
        embedding.to(device=device, dtype=dtype),
    )


def load_music3_ar(
    root: Path | None = None,
    quantize: bool = True,
    device: str = "cuda:0",
    with_depth: bool = True,
    allow_random_extras: bool = False,
) -> Music3AR:
    root = root or models_dir()
    cfg = Music3ArConfig()
    lm = load_global_lm(root, quantize=quantize, device=device)
    depth, extra = None, None
    try:
        depth, extra = load_depth_components(root, cfg, device=device)
    except FileNotFoundError:
        if not allow_random_extras:
            raise
        print(
            "WARNING: rvq_depth_decoder weights not available — using RANDOM "
            "audio_extra_embedding. Only useful for plumbing smoke tests."
        )
    model = Music3AR(lm, cfg, depth_decoder=depth if with_depth else None, audio_extra_embedding=extra)
    if extra is None:
        model.audio_extra_embedding.to(device=device, dtype=torch.bfloat16)
    return model


def load_generation_defaults(root: Path | None = None) -> dict:
    root = root or models_dir()
    path = root / "language_model" / "generation_config.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}
