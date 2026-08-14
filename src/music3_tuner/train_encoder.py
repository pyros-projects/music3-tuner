"""Train the Phase-0 audio→codes encoder on (DAV-latent, codes) pairs.

v1 recipe (all software-side, same corpus): RVQ-factorized acoustic heads
(c0-conditioned, teacher-forced), 512-frame crops, EMA weights, label
smoothing, latent-noise + stereo-swap augmentation. Rich live UI + JSONL
metrics log (out/train_log.jsonl) for later plotting.

Validation is honest inference (no c0 teacher): per-codebook top-1/top-5
against chance (c0: 1/16384, acoustic: 1/1024).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.utils.data import DataLoader, Dataset

from .encoder import FEATURE_RATE, CodesEncoder, latent_to_features, save_encoder
from .train import arm_vram_watchdog

console = Console()


class PairsDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        crop_frames: int,
        seed: int = 0,
        train: bool = True,
        augment: bool = True,
    ):
        self.paths = paths
        self.crop = crop_frames
        self.train = train
        self.augment = augment and train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        from safetensors import safe_open

        with safe_open(self.paths[index], framework="pt") as f:
            latent = f.get_tensor("latent").float()
            codes = f.get_tensor("codes").long()
        if self.augment:
            half = latent.shape[0] // 2
            if self.rng.random() < 0.5:  # stereo swap: latent is [L(64), R(64)]
                latent = torch.cat([latent[half:], latent[:half]])
            if self.rng.random() < 0.5:  # domain-gap robustness (synthetic → real audio)
                latent = latent + torch.randn_like(latent) * latent.std() * self.rng.uniform(0.01, 0.08)
        frames = codes.shape[0]
        features = latent_to_features(latent, frames)  # [128, 4*T]
        crop = min(self.crop, frames)
        start = self.rng.randrange(frames - crop + 1) if (self.train and frames > crop) else 0
        return {
            "features": features[:, start * FEATURE_RATE : (start + crop) * FEATURE_RATE],
            "codes": codes[start : start + crop],
        }


def collate(items: list[dict]) -> dict:
    crop = min(item["codes"].shape[0] for item in items)
    return {
        "features": torch.stack([item["features"][:, : crop * FEATURE_RATE] for item in items]),
        "codes": torch.stack([item["codes"][:crop] for item in items]),
    }


def split_pairs(pairs_dir: Path, val_fraction: float) -> tuple[list[Path], list[Path]]:
    """Stable track-level split by stem hash — a track is never in both sets."""
    train, val = [], []
    for path in sorted(pairs_dir.glob("*.safetensors")):
        digest = int(hashlib.blake2b(path.stem.encode(), digest_size=4).hexdigest(), 16)
        (val if digest % 1000 < val_fraction * 1000 else train).append(path)
    return train, val


class Ema:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        decay = min(self.decay, (1 + step) / (10 + step))
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(decay).add_(value.float(), alpha=1 - decay)
            else:
                self.shadow[key].copy_(value)

    def copy_to(self, model: torch.nn.Module) -> dict:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(backup[k].dtype) for k, v in self.shadow.items()})
        return backup


@torch.no_grad()
def evaluate(model: CodesEncoder, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    c0_top1 = c0_top5 = acoustic_top1 = total = 0
    for batch in loader:
        features = batch["features"].to(device)
        codes = batch["codes"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            c0_logits, acoustic_logits = model(features)  # honest: no c0 teacher
        n = codes.shape[0] * codes.shape[1]
        total += n
        c0_top1 += (c0_logits.argmax(-1) == codes[..., 0]).sum().item()
        top5 = c0_logits.topk(5, dim=-1).indices
        c0_top5 += (top5 == codes[..., 0].unsqueeze(-1)).any(-1).sum().item()
        acoustic_top1 += (acoustic_logits.argmax(-1) == codes[..., 1:]).float().mean(-1).sum().item()
    model.train()
    return {
        "c0_top1": c0_top1 / total,
        "c0_top5": c0_top5 / total,
        "acoustic_top1": acoustic_top1 / total,
    }


def vram_gib(device: str) -> float:
    if not device.startswith("cuda"):
        return 0.0
    free, total = torch.cuda.mem_get_info(torch.device(device).index or 0)
    return (total - free) / 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("cache/pairs"))
    parser.add_argument("--out", type=Path, default=Path("out/encoder"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop", type=int, default=512, help="crop length in code frames (512 ≈ 20.5s)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        arm_vram_watchdog(args.device)

    train_paths, val_paths = split_pairs(args.pairs, args.val_frac)
    if not train_paths or not val_paths:
        raise SystemExit(f"bad split: {len(train_paths)} train / {len(val_paths)} val under {args.pairs}")

    train_loader = DataLoader(
        PairsDataset(train_paths, args.crop, args.seed, train=True, augment=not args.no_aug),
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        PairsDataset(val_paths, args.crop, train=False),
        batch_size=args.batch,
        collate_fn=collate,
        num_workers=2,
    )

    model = CodesEncoder(d_model=args.d_model, num_layers=args.layers).to(args.device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    ema = Ema(model, args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress_frac = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress_frac)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"
    log_file = log_path.open("a")

    def log(record: dict) -> None:
        log_file.write(json.dumps({"ts": round(time.time(), 1), **record}) + "\n")
        log_file.flush()

    console.print(
        Panel.fit(
            f"[bold]music3-tuner encoder v1[/bold]\n"
            f"pairs [cyan]{len(train_paths)}[/cyan] train / [cyan]{len(val_paths)}[/cyan] val   "
            f"model [cyan]{params_m:.0f}M[/cyan] (d{args.d_model}/{args.layers}L)\n"
            f"crop [cyan]{args.crop}[/cyan] frames   batch [cyan]{args.batch}[/cyan]   "
            f"steps [cyan]{args.steps}[/cyan]   EMA [cyan]{args.ema_decay}[/cyan]   "
            f"aug [cyan]{'off' if args.no_aug else 'on'}[/cyan]\n"
            f"log [dim]{log_path}[/dim]",
            title="training",
            border_style="magenta",
        )
    )

    val_history: list[tuple[int, dict[str, float]]] = []
    best_top1 = -1.0
    step = 0
    columns = [
        TextColumn("[bold magenta]encoder"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("c0 [yellow]{task.fields[c0]}[/yellow] ac [yellow]{task.fields[ac]}[/yellow] "
                   "lr {task.fields[lr]} vram {task.fields[vram]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("train", total=args.steps, c0="-", ac="-", lr="-", vram="-")
        while step < args.steps:
            for batch in train_loader:
                if step >= args.steps:
                    break
                features = batch["features"].to(args.device, non_blocking=True)
                codes = batch["codes"].to(args.device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                    c0_logits, acoustic_logits = model(features, c0_teacher=codes[..., 0])
                    c0_loss = F.cross_entropy(
                        c0_logits.flatten(0, 1).float(),
                        codes[..., 0].flatten(),
                        label_smoothing=args.label_smoothing,
                    )
                    acoustic_loss = F.cross_entropy(
                        acoustic_logits.flatten(0, 2).float(),
                        codes[..., 1:].flatten(),
                        label_smoothing=args.label_smoothing,
                    )
                    loss = c0_loss + acoustic_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model, step)
                step += 1

                progress.update(
                    task,
                    advance=1,
                    c0=f"{c0_loss.item():.2f}",
                    ac=f"{acoustic_loss.item():.2f}",
                    lr=f"{scheduler.get_last_lr()[0]:.1e}",
                    vram=f"{vram_gib(args.device):.1f}G",
                )
                if step % 50 == 0:
                    log({"step": step, "c0_loss": round(c0_loss.item(), 4),
                         "ac_loss": round(acoustic_loss.item(), 4),
                         "lr": scheduler.get_last_lr()[0], "vram_gib": round(vram_gib(args.device), 2)})

                if step % args.val_every == 0 or step == args.steps:
                    backup = ema.copy_to(model)
                    metrics = evaluate(model, val_loader, args.device)
                    model.load_state_dict(backup)
                    val_history.append((step, metrics))
                    log({"step": step, "val": {k: round(v, 5) for k, v in metrics.items()}})
                    marker = ""
                    if metrics["c0_top1"] > best_top1:
                        best_top1 = metrics["c0_top1"]
                        backup = ema.copy_to(model)
                        save_encoder(model, args.out)
                        model.load_state_dict(backup)
                        marker = " [bold green]← best, saved[/bold green]"
                    console.print(
                        f"  [bold]step {step:>6}[/bold]  "
                        f"c0 top1 [bold cyan]{metrics['c0_top1']:>6.1%}[/bold cyan]  "
                        f"top5 [cyan]{metrics['c0_top5']:>6.1%}[/cyan]  "
                        f"acoustic [cyan]{metrics['acoustic_top1']:>6.1%}[/cyan]{marker}"
                    )

    table = Table(title="validation history (EMA weights, no c0 teacher)", border_style="dim")
    table.add_column("step", justify="right")
    table.add_column("c0 top-1", justify="right")
    table.add_column("c0 top-5", justify="right")
    table.add_column("acoustic top-1", justify="right")
    for at_step, metrics in val_history:
        table.add_row(
            str(at_step),
            f"{metrics['c0_top1']:.1%}",
            f"{metrics['c0_top5']:.1%}",
            f"{metrics['acoustic_top1']:.1%}",
        )
    console.print(table)
    console.print(
        Panel.fit(
            f"best val c0 top-1: [bold green]{best_top1:.1%}[/bold green]  "
            f"(chance {1 / 16384:.3%})\nencoder saved to [cyan]{args.out}[/cyan]",
            border_style="green",
        )
    )
    log_file.close()


if __name__ == "__main__":
    main()
