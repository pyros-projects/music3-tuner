"""AR-prior re-decoding: fuse frame-local encoder posteriors with the frozen
Hybrid-LM's sequential prior (iterated conditional modes).

The encoder predicts p(code_t | audio) per frame with no sequence model; the
8B defines p(c0_t | c0_<t, caption) and the depth decoder p(c_k | hidden,
c_<k). Per iteration: teacher-force the current code estimate, fuse
log-probs (encoder + λ · prior), re-argmax, repeat. λ is a fidelity dial —
too much prior decodes "plausible" instead of "faithful", so verify against
envelope correlation when tuning.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .ar import Music3AR


@torch.no_grad()
def refine_codes(
    model: Music3AR,
    prompt_ids: torch.Tensor,
    c0_logprobs: torch.Tensor,
    acoustic_logprobs: torch.Tensor,
    lam: float = 0.5,
    iterations: int = 2,
    depth_chunk: int = 512,
) -> torch.Tensor:
    """prompt_ids [1, P], encoder log-probs c0 [T, 16384] / acoustic
    [T, 7, 1024] → refined codes [T, 8]."""
    cfg = model.cfg
    device = c0_logprobs.device
    frames = c0_logprobs.shape[0]
    prompt_len = prompt_ids.shape[1]
    codes = torch.cat(
        [c0_logprobs.argmax(-1).unsqueeze(-1), acoustic_logprobs.argmax(-1)], dim=-1
    )

    # restrict the LM head to the semantic-code slice of the vocab
    head_weight = model.lm.get_output_embeddings().weight
    c0_head_weight = head_weight[cfg.audio_code_offset : cfg.audio_code_offset + cfg.c0_vocab_size]

    for _ in range(iterations):
        # --- c0: global-LM prior, teacher-forced on the current estimate ---
        embeds = torch.cat(
            [model._embed_tokens(prompt_ids), model.embed_frames(codes.unsqueeze(0))], dim=1
        )
        hidden = model.lm.get_decoder()(inputs_embeds=embeds, use_cache=False).last_hidden_state
        lm_hidden = hidden[0, prompt_len - 1 : prompt_len + frames - 1]  # predicts frame t
        ar_logits = F.linear(lm_hidden, c0_head_weight).float()
        fused = c0_logprobs + lam * F.log_softmax(ar_logits, dim=-1)
        codes = codes.clone()
        codes[:, 0] = fused.argmax(-1)

        # --- acoustic: depth-decoder prior with the updated c0 ---
        for start in range(0, frames, depth_chunk):
            chunk_hidden = lm_hidden[start : start + depth_chunk]
            chunk_codes = codes[start : start + depth_chunk]
            depth_out = model._depth_forward(chunk_hidden, chunk_codes)  # [N, 8, H]
            for book in range(1, cfg.num_codebooks):
                prior = model.depth_decoder.audio_heads[book - 1](depth_out[:, book]).float()
                fused_book = acoustic_logprobs[start : start + depth_chunk, book - 1] + (
                    lam * F.log_softmax(prior, dim=-1)
                )
                codes[start : start + depth_chunk, book] = fused_book.argmax(-1)
    return codes.to(device)
