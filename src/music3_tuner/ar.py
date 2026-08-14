"""MiniMax-Music3 autoregressive Hybrid-LM, training-oriented port.

Architecture (verified against the ComfyUI reference implementation and the
released checkpoints):

- Global LM: a plain Qwen3ForCausalLM (36L/4096, vocab 200k). Audio frames
  live at 25 fps; codebook-0 (semantic, 16384 entries) occupies token ids
  [AUDIO_CODE_OFFSET, AUDIO_CODE_OFFSET + 16384).
- Frame input embedding = (embed(c0 + offset) + Σ_k extra_embed(c_k)) * 8^-0.5
  over all 8 codebooks — training the global LM therefore needs *all* 8
  codebooks of the target audio, not just codebook 0.
- Depth decoder (0.6B "local LM"): per frame, predicts codebooks 1..7 from
  [proj(hidden), proj(embed(c0)), proj(extra(c1)), ..., proj(extra(c6))]
  with a 4-layer causal transformer and 7 classification heads (vocab 1024).
- Inference CFG: scale 1.5 against an unconditional prompt whose interior is
  replaced by <|audio_cfg|> — see prompt.uncond_ids().
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .prompt import (
    AUDIO_CODE_OFFSET,
    AUDIO_VOCAB_SIZE,
    C0_VOCAB_SIZE,
    NUM_CODEBOOKS,
    SPECIAL_TOKEN_IDS,
)


@dataclass
class Music3ArConfig:
    hidden_size: int = 4096
    audio_code_offset: int = AUDIO_CODE_OFFSET
    c0_vocab_size: int = C0_VOCAB_SIZE
    audio_vocab_size: int = AUDIO_VOCAB_SIZE
    num_codebooks: int = NUM_CODEBOOKS
    audio_end_token: int = SPECIAL_TOKEN_IDS["<|audio_end|>"]
    # depth decoder (rvq_depth_decoder/config.json)
    depth_num_layers: int = 4
    depth_num_heads: int = 16
    depth_intermediate_size: int = 6144
    depth_max_positions: int = 16


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


class DepthAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        batch, length, _ = x.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(out)


class DepthMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DepthBlock(nn.Module):
    def __init__(self, cfg: Music3ArConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size)
        self.self_attn = DepthAttention(cfg.hidden_size, cfg.depth_num_heads)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size)
        self.mlp = DepthMLP(cfg.hidden_size, cfg.depth_intermediate_size)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class RVQDepthDecoder(nn.Module):
    """The 0.6B local LM: codebooks 1..7 from the global hidden + c0."""

    def __init__(self, cfg: Music3ArConfig):
        super().__init__()
        self.projection = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(cfg.depth_max_positions, cfg.hidden_size)
        self.audio_heads = nn.ModuleList(
            [
                nn.Linear(cfg.hidden_size, cfg.audio_vocab_size, bias=False)
                for _ in range(cfg.num_codebooks - 1)
            ]
        )
        self.layers = nn.ModuleList([DepthBlock(cfg) for _ in range(cfg.depth_num_layers)])
        self.norm = RMSNorm(cfg.hidden_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        x = sequence + self.pos_embedding(positions).to(sequence.dtype).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def chunked_cross_entropy(
    hidden: torch.Tensor,
    head: nn.Module,
    targets: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """CE over a 200k vocab without materializing full-sequence logits."""
    losses = []
    for start in range(0, hidden.shape[0], chunk_size):
        logits = head(hidden[start : start + chunk_size]).float()
        losses.append(F.cross_entropy(logits, targets[start : start + chunk_size], reduction="sum"))
    return torch.stack(losses).sum() / targets.numel()


class Music3AR(nn.Module):
    """Wraps a HF Qwen3ForCausalLM with the audio-frame embedding scheme and
    teacher-forced losses for both LM stages."""

    def __init__(
        self,
        language_model,
        cfg: Music3ArConfig | None = None,
        depth_decoder: RVQDepthDecoder | None = None,
        audio_extra_embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.cfg = cfg or Music3ArConfig()
        self.lm = language_model
        self.depth_decoder = depth_decoder
        self.audio_extra_embedding = audio_extra_embedding or nn.Embedding(
            self.cfg.audio_vocab_size * (self.cfg.num_codebooks - 1), self.cfg.hidden_size
        )
        self.embedding_scale = self.cfg.num_codebooks**-0.5

    # --- embedding -------------------------------------------------------
    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(ids)

    def embed_frames(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B, T, 8] → frame embeddings [B, T, H]."""
        cfg = self.cfg
        c0 = self._embed_tokens(codes[..., 0] + cfg.audio_code_offset)
        offsets = torch.arange(cfg.num_codebooks - 1, device=codes.device) * cfg.audio_vocab_size
        extra = self.audio_extra_embedding(codes[..., 1:] + offsets).sum(dim=-2)
        return (c0 + extra.to(c0.dtype)) * self.embedding_scale

    # --- losses ----------------------------------------------------------
    def global_loss(
        self,
        prompt_ids: torch.Tensor,
        codes: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
        supervise_audio_end: bool = True,
    ) -> torch.Tensor:
        """Teacher-forced CE for the global LM (codebook-0 prediction).

        prompt_ids [B, P] must be LEFT-padded (prompt_mask False on pads) so
        the last prompt token sits at position P-1 for every item and frames
        start contiguously at P. The hidden at the last prompt position
        predicts frame 0's c0; the hidden at frame t predicts frame t+1's c0;
        the hidden at the last frame predicts <|audio_end|>.
        """
        cfg = self.cfg
        batch, prompt_len = prompt_ids.shape
        frames = codes.shape[1]
        prompt_embeds = self._embed_tokens(prompt_ids)
        frame_embeds = self.embed_frames(codes)
        embeds = torch.cat([prompt_embeds, frame_embeds], dim=1)

        if prompt_mask is None:
            prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
        attention_mask = torch.cat(
            [prompt_mask, torch.ones(batch, frames, dtype=torch.bool, device=codes.device)], dim=1
        )
        # left-padding shifts real positions; keep RoPE indices contiguous
        position_ids = (attention_mask.long().cumsum(dim=1) - 1).clamp(min=0)

        outputs = self.lm.get_decoder()(
            inputs_embeds=embeds, attention_mask=attention_mask, position_ids=position_ids
        )
        hidden = outputs.last_hidden_state

        # shifted[b, i] = token the hidden at position i must predict (next token)
        shifted = torch.full(embeds.shape[:2], -100, dtype=torch.long, device=codes.device)
        shifted[:, prompt_len - 1 : prompt_len + frames - 1] = codes[..., 0] + cfg.audio_code_offset
        if supervise_audio_end:
            shifted[:, prompt_len + frames - 1] = cfg.audio_end_token

        keep = shifted != -100
        return chunked_cross_entropy(
            hidden[keep], self.lm.get_output_embeddings(), shifted[keep]
        )

    def _depth_forward(self, hidden: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced depth-decoder pass for N frames.

        hidden [N, H] global-LM hiddens, codes [N, 8]. Sequence per frame:
        [proj(h), proj(embed_c0), proj(extra(c1..c6))]; returns decoder
        outputs [N, 8, H] — causal position j (1..7) predicts codebook j.
        """
        cfg = self.cfg
        assert self.depth_decoder is not None, "depth decoder not loaded"
        decoder = self.depth_decoder
        c0_embed = self._embed_tokens(codes[:, 0] + cfg.audio_code_offset)
        offsets = torch.arange(cfg.num_codebooks - 2, device=codes.device) * cfg.audio_vocab_size
        extra = self.audio_extra_embedding(codes[:, 1 : cfg.num_codebooks - 1] + offsets)
        sequence = torch.cat(
            [hidden.unsqueeze(1), c0_embed.unsqueeze(1), extra.to(hidden.dtype)], dim=1
        )
        return decoder(decoder.projection(sequence))

    def depth_hiddens(self, hidden: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Per-frame depth hiddens [N, 7, H] (positions 1..7) — the residual
        slices of the synthesis conditioning."""
        return self._depth_forward(hidden, codes)[:, 1:]

    def depth_loss(self, hidden: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced CE for the depth decoder (codebooks 1..7)."""
        cfg = self.cfg
        out = self._depth_forward(hidden, codes)
        losses = []
        for book in range(1, cfg.num_codebooks):
            logits = self.depth_decoder.audio_heads[book - 1](out[:, book]).float()
            losses.append(F.cross_entropy(logits, codes[:, book]))
        return torch.stack(losses).mean()
