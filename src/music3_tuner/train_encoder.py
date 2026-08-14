"""Train the Phase-0 audio→codes encoder on (DAV-latent, codes) pairs.

v1 recipe (all software-side, same corpus): RVQ-factorized acoustic heads
(c0-conditioned, teacher-forced), 512-frame crops, EMA weights, label
smoothing, latent-noise + stereo-swap augmentation. Rich live UI + JSONL
metrics log (out/train_log.jsonl) for later plotting.

Validation runs complete tracks through the same overlapping-window inference
path (no c0 teacher): per-codebook top-1/top-5 against chance (c0: 1/16384,
acoustic: 1/1024).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
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

from .encoder import FEATURE_RATE, CodesEncoder, latent_to_features, save_encoder, windowed_logprobs
from .train import arm_vram_watchdog

console = Console()


class PairsDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        crop_frames: int,
        train: bool = True,
        augment: bool = True,
    ):
        self.paths = paths
        self.crop = crop_frames
        self.train = train
        self.augment = augment and train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        from safetensors import safe_open

        with safe_open(self.paths[index], framework="pt") as f:
            latent = f.get_tensor("latent").float()
            codes = f.get_tensor("codes").long()
        if self.augment:
            half = latent.shape[0] // 2
            if random.random() < 0.5:  # DataLoader seeds Python's RNG independently per worker
                latent = torch.cat([latent[half:], latent[:half]])
            if random.random() < 0.5:  # domain-gap robustness (synthetic → real audio)
                latent = (
                    latent
                    + torch.randn_like(latent) * latent.std() * random.uniform(0.01, 0.08)
                )
        frames = codes.shape[0]
        features = latent_to_features(latent, frames)  # [128, 4*T]
        crop = min(self.crop, frames) if self.train else frames
        start = random.randrange(frames - crop + 1) if (self.train and frames > crop) else 0
        features = features[:, start * FEATURE_RATE : (start + crop) * FEATURE_RATE]
        if self.augment:
            # SpecAugment-style masking — the anti-memorization lever: force
            # the model to infer codes from context instead of lookup.
            features = features.clone()
            for _ in range(2):  # time masks (up to ~1.6s each)
                width = random.randrange(1, 40) * FEATURE_RATE
                begin = random.randrange(max(1, features.shape[1] - width))
                features[:, begin : begin + width] = 0.0
            for _ in range(2):  # latent-channel masks
                width = random.randrange(1, 16)
                begin = random.randrange(max(1, features.shape[0] - width))
                features[begin : begin + width] = 0.0
        return {
            "features": features,
            "codes": codes[start : start + crop],
        }


def collate(items: list[dict]) -> dict:
    crop = min(item["codes"].shape[0] for item in items)
    return {
        "features": torch.stack([item["features"][:, : crop * FEATURE_RATE] for item in items]),
        "codes": torch.stack([item["codes"][:crop] for item in items]),
    }


def pair_family(path: Path) -> str:
    """Corpus identity shared by seed variants such as ``track_s1`` and ``track_s2``."""
    return re.sub(r"_s\d+$", "", path.stem)


def split_pairs(pairs_dir: Path, val_fraction: float) -> tuple[list[Path], list[Path]]:
    """Stable template-family split — seed variants never cross the boundary."""
    train, val = [], []
    for path in sorted(pairs_dir.glob("*.safetensors")):
        digest = int(hashlib.blake2b(pair_family(path).encode(), digest_size=4).hexdigest(), 16)
        (val if digest % 1000 < val_fraction * 1000 else train).append(path)
    return train, val


def select_ar_panel(paths: list[Path], count: int, seed: int) -> list[Path]:
    """Stable seeded sample of distinct validation families for AR diagnostics."""
    if count <= 0:
        raise ValueError("ar_tracks must be positive")
    representatives: dict[str, Path] = {}
    for path in sorted(paths):
        representatives.setdefault(pair_family(path), path)
    return sorted(
        representatives.values(),
        key=lambda path: hashlib.blake2b(
            f"{seed}:{pair_family(path)}".encode(), digest_size=8
        ).digest(),
    )[:count]


def selection_score(metrics: dict) -> tuple[float, float]:
    """Rank checkpoints by full-track label fidelity, with an acoustic tie-breaker."""
    return metrics["c0_top1"], metrics["acoustic_top1"]


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
def evaluate(model: CodesEncoder, loader: DataLoader, device: str, window: int) -> dict:
    """Evaluate complete tracks through the same overlapping-window path as inference."""
    model.eval()
    num_books = model.config["num_codebooks"] - 1
    c0_top1 = c0_top5 = total = 0
    book_top1 = torch.zeros(num_books)
    for batch in loader:
        if batch["features"].shape[0] != 1:
            raise ValueError("full-track validation requires batch_size=1")
        features = batch["features"][0].to(device)
        codes = batch["codes"][0].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            c0_logits, acoustic_logits = windowed_logprobs(
                model, features, codes.shape[0], window
            )
        n = codes.shape[0]
        total += n
        c0_top1 += (c0_logits.argmax(-1) == codes[:, 0]).sum().item()
        top5 = c0_logits.topk(5, dim=-1).indices
        c0_top5 += (top5 == codes[:, 0].unsqueeze(-1)).any(-1).sum().item()
        book_top1 += (acoustic_logits.argmax(-1) == codes[:, 1:]).float().sum(dim=0).cpu()
    model.train()
    books = (book_top1 / total).tolist()
    return {
        "frames": total,
        "c0_top1": c0_top1 / total,
        "c0_top5": c0_top5 / total,
        "acoustic_top1": sum(books) / len(books),
        "books": books,
    }


def vram_gib(device: str) -> float:
    if not device.startswith("cuda"):
        return 0.0
    free, total = torch.cuda.mem_get_info(torch.device(device).index or 0)
    return (total - free) / 1024**3


def paths_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.blake2b(digest_size=8)
    for path in sorted(paths):
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("cache/pairs"))
    parser.add_argument("--out", type=Path, default=Path("out/encoder"))
    parser.add_argument("--steps", type=int, default=4000, help="schedule sized to the ~1000-pair corpus: val peaks ~2k, cosine decays into it")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop", type=int, default=512, help="crop length in code frames (512 ≈ 20.5s)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--scheduled-max", type=float, default=0.5, help="scheduled-sampling ceiling for the acoustic c0 conditioning (ramps over the first half)")
    parser.add_argument(
        "--ar-diagnostic",
        "--ar-select",
        dest="ar_select",
        action="store_true",
        help=(
            "report diagnostic AR loss on a fixed val panel (loads the 8B, ~5.5GB "
            "extra VRAM); --ar-select is a deprecated alias"
        ),
    )
    parser.add_argument(
        "--ar-tracks", type=int, default=4, help="tracks in the fixed AR diagnostic panel"
    )
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device.startswith("cuda"):
        arm_vram_watchdog(args.device)

    train_paths, val_paths = split_pairs(args.pairs, args.val_frac)
    if not train_paths or not val_paths:
        raise SystemExit(f"bad split: {len(train_paths)} train / {len(val_paths)} val under {args.pairs}")

    data_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PairsDataset(train_paths, args.crop, train=True, augment=not args.no_aug),
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        drop_last=True,
        persistent_workers=True,
        generator=data_generator,
    )
    val_loader = DataLoader(
        PairsDataset(val_paths, args.crop, train=False),
        batch_size=1,
        collate_fn=collate,
        num_workers=2,
    )

    model = CodesEncoder(
        d_model=args.d_model, num_layers=args.layers, window=args.crop
    ).to(args.device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    ema = Ema(model, args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    ar_model = ar_tokenizer = None
    ar_paths: list[Path] = []
    if args.ar_select:
        from .loading import load_music3_ar, load_tokenizer

        ar_tokenizer = load_tokenizer()
        ar_model = load_music3_ar(quantize=True, device=args.device, with_depth=False)
        ar_paths = select_ar_panel(val_paths, args.ar_tracks, args.seed)

    @torch.no_grad()
    def ar_loss_of_current(encoder: CodesEncoder) -> float:
        """Teacher-forced AR loss of the encoder's predicted codes — the
        downstream quality currency (call with EMA weights loaded)."""
        from safetensors import safe_open

        from .encoder import logprobs_to_codes, windowed_logprobs
        from .prompt import encode_prompt

        encoder.eval()
        losses = []
        for path in ar_paths:
            with safe_open(path, framework="pt") as f:
                latent = f.get_tensor("latent").float()
                truth = f.get_tensor("codes")
                primer_codes = (
                    f.get_tensor("primer_codes").long()
                    if "primer_codes" in f.keys()
                    else None
                )
                meta = f.metadata() or {}
            frames = truth.shape[0]
            feats = latent_to_features(latent, frames).to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                c0_lp, ac_lp = windowed_logprobs(encoder, feats, frames, args.crop)
            predicted = logprobs_to_codes(c0_lp, ac_lp)
            ids = torch.tensor(
                [encode_prompt(ar_tokenizer, meta.get("caption", ""), meta.get("lyrics", ""))],
                device=args.device,
            )
            loss = ar_model.global_loss(
                ids,
                predicted.unsqueeze(0).to(args.device),
                supervise_audio_end=False,
                primer_codes=(
                    primer_codes.unsqueeze(0).to(args.device)
                    if primer_codes is not None
                    else None
                ),
            )
            losses.append(loss.item())
        encoder.train()
        return sum(losses) / len(losses)

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress_frac = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress_frac)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"
    log_file = log_path.open("a")
    run_id = str(time.time_ns())

    def log(record: dict) -> None:
        log_file.write(
            json.dumps({"ts": round(time.time(), 1), "run_id": run_id, **record}) + "\n"
        )
        log_file.flush()

    log(
        {
            "event": "run_start",
            "git_head": git_head(),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "pairs": {
                "train": len(train_paths),
                "val": len(val_paths),
                "fingerprint": paths_fingerprint(train_paths + val_paths),
            },
            "ar_panel": [path.name for path in ar_paths],
        }
    )

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

    val_history: list[tuple[int, dict]] = []
    best_fidelity = (-1.0, -1.0)
    best_ar = float("inf")
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
                # exposure-bias fix: ramp scheduled sampling over the first half
                scheduled_p = args.scheduled_max * min(1.0, step / max(1, args.steps * 0.5))
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                    c0_logits, acoustic_logits = model(
                        features, c0_teacher=codes[..., 0], scheduled_p=scheduled_p
                    )
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
                    metrics = evaluate(model, val_loader, args.device, args.crop)
                    if args.ar_select:
                        metrics["ar_loss"] = ar_loss_of_current(model)
                        best_ar = min(best_ar, metrics["ar_loss"])
                    score = selection_score(metrics)
                    is_best = score > best_fidelity
                    if is_best:
                        best_fidelity = score
                        save_encoder(model, args.out)
                    model.load_state_dict(backup)
                    val_history.append((step, metrics))
                    log(
                        {
                            "step": step,
                            "val": metrics,
                            "saved": is_best,
                            "selection_metric": "c0_top1_then_acoustic_top1",
                        }
                    )
                    books = "/".join(f"{b:.1%}" for b in metrics["books"])
                    ar_part = (
                        f"  ar [bold yellow]{metrics['ar_loss']:.2f}[/bold yellow]"
                        if args.ar_select
                        else ""
                    )
                    marker = " [bold green]← best, saved[/bold green]" if is_best else ""
                    console.print(
                        f"  [bold]step {step:>6}[/bold]  "
                        f"c0 top1 [bold cyan]{metrics['c0_top1']:>6.1%}[/bold cyan]  "
                        f"top5 [cyan]{metrics['c0_top5']:>6.1%}[/cyan]  "
                        f"acoustic [cyan]{metrics['acoustic_top1']:>6.1%}[/cyan] "
                        f"[dim]({books})[/dim]{ar_part}{marker}"
                    )

    table = Table(
        title="full-track validation (EMA weights, overlapping windows)", border_style="dim"
    )
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
    summary = (
        "selected full-track fidelity: "
        f"c0 [bold green]{best_fidelity[0]:.1%}[/bold green] / "
        f"acoustic [bold green]{best_fidelity[1]:.1%}[/bold green]  "
        f"(c0 chance {1 / 16384:.3%})"
    )
    if args.ar_select:
        summary += (
            f"\nbest diagnostic AR loss: [bold yellow]{best_ar:.3f}[/bold yellow]  "
            "(not used for selection; model-own ≈ 2, random ≈ 9.7)"
        )
    console.print(Panel.fit(summary + f"\nencoder saved to [cyan]{args.out}[/cyan]", border_style="green"))
    log_file.close()


if __name__ == "__main__":
    main()
