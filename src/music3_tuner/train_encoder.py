"""Train the Phase-0 audio→codes encoder on (DAV-latent, codes) pairs.

Random 256-frame crops, bf16 autocast, AdamW with warmup+cosine. Validation
reports per-codebook top-1/top-5 against chance (c0: 1/16384, acoustic:
1/1024) — the feasibility metric of the whole distillation bet.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .encoder import FEATURE_RATE, CodesEncoder, latent_to_features, save_encoder
from .train import arm_vram_watchdog


class PairsDataset(Dataset):
    def __init__(self, paths: list[Path], crop_frames: int, seed: int = 0, train: bool = True):
        self.paths = paths
        self.crop = crop_frames
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        from safetensors import safe_open

        with safe_open(self.paths[index], framework="pt") as f:
            latent = f.get_tensor("latent").float()
            codes = f.get_tensor("codes").long()
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


@torch.no_grad()
def evaluate(model: CodesEncoder, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    c0_top1 = c0_top5 = acoustic_top1 = total = 0
    for batch in loader:
        features = batch["features"].to(device)
        codes = batch["codes"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            c0_logits, acoustic_logits = model(features)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("cache/pairs"))
    parser.add_argument("--out", type=Path, default=Path("out/encoder"))
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop", type=int, default=256, help="crop length in code frames (256 ≈ 10.2s)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        arm_vram_watchdog(args.device)

    train_paths, val_paths = split_pairs(args.pairs, args.val_frac)
    if not train_paths or not val_paths:
        raise SystemExit(f"bad split: {len(train_paths)} train / {len(val_paths)} val under {args.pairs}")
    print(f"pairs: {len(train_paths)} train / {len(val_paths)} val")

    train_loader = DataLoader(
        PairsDataset(train_paths, args.crop, args.seed, train=True),
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
    print(f"encoder params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        import math

        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_top1 = -1.0
    step = 0
    progress = tqdm(total=args.steps, desc="encoder")
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            features = batch["features"].to(args.device, non_blocking=True)
            codes = batch["codes"].to(args.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                c0_logits, acoustic_logits = model(features)
                c0_loss = F.cross_entropy(c0_logits.flatten(0, 1).float(), codes[..., 0].flatten())
                acoustic_loss = F.cross_entropy(
                    acoustic_logits.flatten(0, 2).float(), codes[..., 1:].flatten()
                )
                loss = c0_loss + acoustic_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)
            progress.set_postfix(c0=f"{c0_loss.item():.2f}", ac=f"{acoustic_loss.item():.2f}")

            if step % args.val_every == 0 or step == args.steps:
                metrics = evaluate(model, val_loader, args.device)
                tqdm.write(
                    f"step {step}: c0 top1 {metrics['c0_top1']:.1%} top5 {metrics['c0_top5']:.1%} "
                    f"acoustic top1 {metrics['acoustic_top1']:.1%} "
                    f"(chance: {1 / 16384:.3%} / {5 / 16384:.3%} / {1 / 1024:.2%})"
                )
                if metrics["c0_top1"] > best_top1:
                    best_top1 = metrics["c0_top1"]
                    save_encoder(model, args.out)
    progress.close()
    print(f"best val c0 top-1: {best_top1:.1%} — encoder saved to {args.out}")


if __name__ == "__main__":
    main()
