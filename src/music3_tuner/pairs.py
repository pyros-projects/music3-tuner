"""Build (DAV-latent, codes) training pairs for the Phase-0 encoder.

Matches wavs (synthesized by music3-synth) with their code caches by stem,
encodes each wav through the DAV Flow-VAE encoder and stores the pair plus
the prompt metadata (kept for later reconstruction experiments).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .cache_audio import load_wav_44k_stereo
from .dav import load_dav
from .loading import models_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavs", type=Path, default=Path("cache/wavs"))
    parser.add_argument("--codes", type=Path, default=Path("cache/codes_templates"))
    parser.add_argument("--out", type=Path, default=Path("cache/pairs"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from safetensors import safe_open
    from safetensors.torch import save_file

    dav = load_dav(str(models_dir() / "dav.pth"), device=args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    stems = sorted(
        {p.stem for p in args.wavs.glob("*.wav")} & {p.stem for p in args.codes.glob("*.safetensors")}
    )
    done = 0
    for stem in stems:
        target = args.out / f"{stem}.safetensors"
        if target.exists():
            continue
        with safe_open(args.codes / f"{stem}.safetensors", framework="pt") as f:
            codes = f.get_tensor("codes")
            primer_codes = f.get_tensor("primer_codes") if "primer_codes" in f.keys() else None
            meta = f.metadata() or {}
        waveform = load_wav_44k_stereo(args.wavs / f"{stem}.wav").to(args.device)
        latent = dav.encode(waveform).squeeze(0)  # [128, T_lat]
        tensors = {"latent": latent.to(torch.float16).cpu().contiguous(), "codes": codes}
        if primer_codes is not None:
            tensors["primer_codes"] = primer_codes
        save_file(
            tensors,
            str(target),
            metadata=meta,
        )
        done += 1
        if done % 50 == 0:
            print(f"{done}/{len(stems)} pairs")
    print(f"prepared {done} new pairs ({len(stems)} total) → {args.out}")


if __name__ == "__main__":
    main()
