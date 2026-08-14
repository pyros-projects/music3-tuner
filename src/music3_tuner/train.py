"""QLoRA training loop for the Music3 global LM (codebook-0 prediction).

NF4-quantized 8B + LoRA on attention/MLP projections, teacher-forced CE over
cached code sequences. Batch size is 1 with gradient accumulation (tracks are
ragged); a VRAM watchdog kills the run before WSL2 starts spilling into
shared memory.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CodesDataset, collate_codes
from .loading import load_music3_ar, load_tokenizer


def _trainable_adapter_parameters(model) -> list[torch.nn.Parameter]:
    """Return only state persisted by ``model.lm.save_pretrained``."""
    model.audio_extra_embedding.requires_grad_(False)
    return [parameter for parameter in model.lm.parameters() if parameter.requires_grad]


def arm_vram_watchdog(device: str, limit_gib: float | None = None) -> None:
    """WSL2 doesn't OOM — it spills into shared memory and grinds the box.
    Hard-exit before that happens. Default limit: total - 1 GiB (23 GiB on
    the 24 GB 4090, scales up on bigger pods); override with M3_VRAM_LIMIT_GIB."""
    index = torch.device(device).index or 0
    if limit_gib is None:
        env = os.environ.get("M3_VRAM_LIMIT_GIB")
        total_gib = torch.cuda.mem_get_info(index)[1] / 1024**3
        limit_gib = float(env) if env else total_gib - 1.0

    def watch() -> None:
        while True:
            free, total = torch.cuda.mem_get_info(index)
            used_gib = (total - free) / 1024**3
            if used_gib > limit_gib:
                print(f"\nVRAM watchdog: {used_gib:.1f} GiB > {limit_gib} GiB — aborting")
                os._exit(3)
            time.sleep(2)

    threading.Thread(target=watch, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dir of cached code sequences")
    parser.add_argument("--out", type=Path, default=Path("out/lora"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1500,
        help="maximum emitted frames (legacy caches sample random windows; primer caches keep a correct prefix)",
    )
    parser.add_argument("--uncond-p", type=float, default=0.1, help="caption dropout → keeps the CFG uncond stream calibrated")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--allow-random-extras", action="store_true", help="plumbing smoke without rvq_depth_decoder weights")
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    arm_vram_watchdog(args.device)

    tokenizer = load_tokenizer()
    model = load_music3_ar(
        quantize=not args.no_quant,
        device=args.device,
        with_depth=False,
        allow_random_extras=args.allow_random_extras,
    )

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model.lm = get_peft_model(model.lm, lora)
    model.lm.print_trainable_parameters()
    model.lm.gradient_checkpointing_enable()
    model.lm.enable_input_require_grads()
    trainable = _trainable_adapter_parameters(model)

    dataset = CodesDataset(
        args.data, tokenizer, max_frames=args.max_frames, uncond_p=args.uncond_p, seed=args.seed
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_codes)

    import bitsandbytes as bnb

    optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 20)
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model.train()
    step, accumulated = 0, 0
    progress = tqdm(total=args.steps, desc="train")
    running = 0.0
    while step < args.steps:
        for batch in loader:
            loss = model.global_loss(
                batch.prompt_ids.to(args.device),
                batch.codes.to(args.device),
                batch.prompt_mask.to(args.device),
                supervise_audio_end=batch.supervise_audio_end,
                primer_codes=batch.primer_codes.to(args.device),
                primer_mask=batch.primer_mask.to(args.device),
            )
            (loss / args.accum).backward()
            running += loss.item() / args.accum
            accumulated += 1
            if accumulated == args.accum:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                accumulated = 0
                free, total = torch.cuda.mem_get_info(torch.device(args.device).index or 0)
                progress.update(1)
                progress.set_postfix(
                    loss=f"{running:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.1e}",
                    vram=f"{(total - free) / 1024**3:.1f}G",
                )
                running = 0.0
                if step % args.save_every == 0 or step == args.steps:
                    target = args.out / f"step-{step}"
                    model.lm.save_pretrained(str(target))
                    if step >= args.steps:
                        break
    progress.close()
    print(f"done — adapters under {args.out}")


if __name__ == "__main__":
    main()
