"""Cache DAV Flow-VAE latents for a directory of wavs (+ roundtrip check).

This is the continuous-latent leg: latents feed the future audio→codes
encoder (distillation) and FM-side work. `--roundtrip` decodes the latent
back to audio and reports SNR/correlation — the end-to-end verification that
our encoder port matches the released decoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile
import torch
import torchaudio

from .dav import HOP, SAMPLE_RATE, load_dav
from .loading import models_dir


def load_wav_44k_stereo(path: Path) -> torch.Tensor:
    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # [C, S]
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    waveform = waveform[:2]
    if rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, rate, SAMPLE_RATE)
    return waveform.unsqueeze(0)  # [1, 2, S]


def snr_db(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    length = min(reference.shape[-1], estimate.shape[-1])
    reference, estimate = reference[..., :length], estimate[..., :length]
    noise = reference - estimate
    return float(10 * torch.log10(reference.pow(2).mean() / noise.pow(2).mean().clamp_min(1e-12)))


def correlation(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    length = min(reference.shape[-1], estimate.shape[-1])
    reference, estimate = reference[..., :length], estimate[..., :length]
    return float(torch.corrcoef(torch.stack([reference.flatten(), estimate.flatten()]))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("cache/latents"))
    parser.add_argument("--dav", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seconds", type=float, default=0.0, help="crop input (0 = full track)")
    parser.add_argument("--roundtrip", action="store_true", help="decode back and report SNR")
    args = parser.parse_args()

    from safetensors.torch import save_file

    dav = load_dav(str(args.dav or models_dir() / "dav.pth"), device=args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for wav_path in sorted(args.audio_dir.glob("*.wav")):
        waveform = load_wav_44k_stereo(wav_path).to(args.device)
        if args.seconds:
            waveform = waveform[..., : int(args.seconds * SAMPLE_RATE)]
        latent = dav.encode(waveform)
        item = {"latent": latent.squeeze(0).to(torch.float16).cpu().contiguous()}
        meta = {"source": str(wav_path), "sample_rate": str(SAMPLE_RATE), "hop": str(HOP)}
        save_file(item, str(args.out / f"{wav_path.stem}.safetensors"), metadata=meta)
        line = f"{wav_path.name}: {waveform.shape[-1]} samples -> latent {tuple(latent.shape)}"
        if args.roundtrip:
            decoded = dav.decode(latent)
            score = snr_db(waveform.cpu(), decoded.cpu())
            corr = correlation(waveform.cpu(), decoded.cpu())
            soundfile.write(
                str(args.out / f"{wav_path.stem}_roundtrip.wav"),
                decoded.squeeze(0).clamp(-1, 1).cpu().float().numpy().T,
                SAMPLE_RATE,
            )
            line += f"  SNR {score:.2f} dB  corr {corr:.3f}"
            report[wav_path.name] = {"snr_db": score, "corr": corr}
        print(line)
    if report:
        (args.out / "roundtrip_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
