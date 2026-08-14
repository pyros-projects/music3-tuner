"""Phase-0 audio→codes encoder — the tokenizer MiniMax didn't release.

Distilled from model-generated (audio, codes) pairs: DAV Flow-VAE latents in
(interpolated to 4x the 25 Hz code rate), a conv stem downsamples to the code
grid, a bidirectional transformer refines, 1 + 7 classification heads emit
the semantic (16384) and acoustic (1024 each) codebooks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .prompt import AUDIO_FRAMES_PER_SECOND

FEATURE_RATE = 4  # latent features per code frame fed to the stem


class CodesEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        d_model: int = 768,
        num_layers: int = 8,
        num_heads: int = 12,
        ff_dim: int = 3072,
        c0_vocab: int = 16384,
        audio_vocab: int = 1024,
        num_codebooks: int = 8,
    ):
        super().__init__()
        self.config = {
            "latent_dim": latent_dim,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "c0_vocab": c0_vocab,
            "audio_vocab": audio_vocab,
            "num_codebooks": num_codebooks,
        }
        self.stem = nn.Sequential(
            nn.Conv1d(latent_dim, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
        )
        layer = nn.TransformerEncoderLayer(
            d_model, num_heads, ff_dim, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.c0_head = nn.Linear(d_model, c0_vocab)
        self.acoustic_head = nn.Linear(d_model, (num_codebooks - 1) * audio_vocab)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """features [B, latent_dim, FEATURE_RATE*T] → (c0 [B,T,16384], acoustic [B,T,7,1024])."""
        x = self.stem(features).transpose(1, 2)  # [B, T, d]
        position = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
        div = torch.exp(
            torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / x.shape[-1])
        )
        pos = torch.zeros(x.shape[1], x.shape[-1], device=x.device, dtype=torch.float32)
        pos[:, 0::2] = torch.sin(position[:, None] * div)
        pos[:, 1::2] = torch.cos(position[:, None] * div)
        x = x + pos.to(x.dtype)
        x = self.norm(self.encoder(x))
        batch, frames, _ = x.shape
        c0_logits = self.c0_head(x)
        cfg = self.config
        acoustic = self.acoustic_head(x).view(
            batch, frames, cfg["num_codebooks"] - 1, cfg["audio_vocab"]
        )
        return c0_logits, acoustic


def latent_to_features(latent: torch.Tensor, num_frames: int) -> torch.Tensor:
    """[latent_dim, T_lat] (86.13 Hz) → [latent_dim, FEATURE_RATE*num_frames]."""
    return F.interpolate(
        latent.unsqueeze(0).float(), size=num_frames * FEATURE_RATE, mode="linear", align_corners=False
    ).squeeze(0)


def save_encoder(model: CodesEncoder, out_dir: Path) -> None:
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(out_dir / "encoder.safetensors"))
    (out_dir / "config.json").write_text(json.dumps(model.config, indent=2))


def load_encoder(out_dir: Path, device: str = "cpu") -> CodesEncoder:
    from safetensors.torch import load_file

    config = json.loads((Path(out_dir) / "config.json").read_text())
    model = CodesEncoder(**config)
    model.load_state_dict(load_file(str(Path(out_dir) / "encoder.safetensors")))
    return model.to(device).eval()


@torch.no_grad()
def encode_wav_to_codes(
    wav_path: Path, encoder: CodesEncoder, dav, device: str
) -> torch.Tensor:
    """wav → DAV latent → predicted codes [T, 8] (argmax)."""
    from .cache_audio import load_wav_44k_stereo
    from .dav import SAMPLE_RATE

    waveform = load_wav_44k_stereo(wav_path).to(device)
    latent = dav.encode(waveform).squeeze(0)
    num_frames = int(round(waveform.shape[-1] / SAMPLE_RATE * AUDIO_FRAMES_PER_SECOND))
    features = latent_to_features(latent, num_frames).unsqueeze(0).to(device)
    c0_logits, acoustic_logits = encoder(features)
    codes = torch.cat(
        [c0_logits.argmax(-1).unsqueeze(-1), acoustic_logits.argmax(-1)], dim=-1
    )
    return codes.squeeze(0)  # [T, 8]


def main() -> None:
    """music3-encode: real audio → code caches (dataset format, ready for
    music3-synth reconstruction or music3-train)."""
    import argparse

    from safetensors.torch import save_file

    from .dataset import compose_caption, parse_sidecar
    from .dav import load_dav
    from .loading import models_dir

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("audio_dir", type=Path, help="dir of wavs (sidecar .txt captions optional)")
    parser.add_argument("--model", type=Path, default=Path("out/encoder"))
    parser.add_argument("--out", type=Path, default=Path("cache/codes_real"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    encoder = load_encoder(args.model, device=args.device)
    dav = load_dav(str(models_dir() / "dav.pth"), device=args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    wavs = sorted(args.audio_dir.glob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    for wav_path in wavs:
        codes = encode_wav_to_codes(wav_path, encoder, dav, args.device)
        sidecar = wav_path.with_suffix(".txt")
        caption, lyrics = "", ""
        if sidecar.exists():
            fields = parse_sidecar(sidecar)
            caption, lyrics = compose_caption(fields), fields.get("lyrics", "")
        save_file(
            {"codes": codes.to(torch.int32).cpu()},
            str(args.out / f"{wav_path.stem}.safetensors"),
            metadata={"caption": caption, "lyrics": lyrics},
        )
        print(f"{wav_path.stem}: {codes.shape[0]} frames ({codes.shape[0] / 25:.1f}s)")


if __name__ == "__main__":
    main()
