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
        d_model: int = 1024,
        num_layers: int = 10,
        num_heads: int = 16,
        ff_dim: int = 4096,
        c0_vocab: int = 16384,
        audio_vocab: int = 1024,
        num_codebooks: int = 8,
        dropout: float = 0.2,
        window: int = 512,
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
            "dropout": dropout,
            "window": window,
        }
        self.stem = nn.Sequential(
            nn.Conv1d(latent_dim, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
        )
        layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.c0_head = nn.Linear(d_model, c0_vocab)
        # RVQ factorization: codebooks 1..7 are residuals *given* the semantic
        # code — condition the acoustic heads on c0 (teacher-forced in
        # training, predicted at inference).
        self.c0_embed = nn.Embedding(c0_vocab, d_model)
        self.acoustic_norm = nn.LayerNorm(d_model)
        self.acoustic_head = nn.Linear(d_model, (num_codebooks - 1) * audio_vocab)

    def forward(
        self,
        features: torch.Tensor,
        c0_teacher: torch.Tensor | None = None,
        scheduled_p: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """features [B, latent_dim, FEATURE_RATE*T] → (c0 [B,T,16384], acoustic [B,T,7,1024]).

        c0_teacher [B, T]: ground-truth semantic codes for the acoustic
        conditioning (training); defaults to the model's own argmax.
        scheduled_p: per-position probability of replacing the teacher c0
        with the model's own prediction — scheduled sampling against the
        train/inference exposure bias of the acoustic conditioning."""
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
        if c0_teacher is None:
            c0_ids = c0_logits.argmax(-1)
        elif scheduled_p > 0:
            predicted = c0_logits.detach().argmax(-1)
            use_own = torch.rand(c0_teacher.shape, device=c0_teacher.device) < scheduled_p
            c0_ids = torch.where(use_own, predicted, c0_teacher)
        else:
            c0_ids = c0_teacher
        y = self.acoustic_norm(x + self.c0_embed(c0_ids).to(x.dtype))
        acoustic = self.acoustic_head(y).view(
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
    if "window" not in config:
        import warnings

        warnings.warn(
            "legacy encoder config has no training window; assuming 512",
            RuntimeWarning,
            stacklevel=2,
        )
    model = CodesEncoder(**config)
    model.load_state_dict(load_file(str(Path(out_dir) / "encoder.safetensors")))
    return model.to(device).eval()


@torch.no_grad()
def windowed_logprobs(
    encoder: CodesEncoder, features: torch.Tensor, num_frames: int, window: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coherent averaged log-probs over half-overlapping windows.

    Windows are the training crop length (positions don't extrapolate);
    c0 is averaged and finalized first, then acoustic logits are recomputed
    with that same final c0 conditioning in every overlapping window.
    Returns (c0 [T, c0_vocab], acoustic [T, books-1, audio_vocab])."""
    cfg = encoder.config
    device = features.device
    window = cfg.get("window", 512) if window is None else window
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    c0_sum = torch.zeros(num_frames, cfg["c0_vocab"], device=device)
    counts = torch.zeros(num_frames, device=device)

    hop = max(1, window // 2)
    starts = list(range(0, max(num_frames - window, 0) + 1, hop))
    if not starts or starts[-1] + window < num_frames:
        starts.append(max(0, num_frames - window))
    starts = list(dict.fromkeys(starts))
    for start in starts:
        end = min(start + window, num_frames)
        chunk = features[:, start * FEATURE_RATE : end * FEATURE_RATE].unsqueeze(0)
        c0_logits, _ = encoder(chunk)
        c0_sum[start:end] += F.log_softmax(c0_logits.squeeze(0).float(), dim=-1)
        counts[start:end] += 1
    c0_logprobs = c0_sum / counts[:, None]
    c0_ids = c0_logprobs.argmax(-1)

    acoustic_sum = torch.zeros(
        num_frames, cfg["num_codebooks"] - 1, cfg["audio_vocab"], device=device
    )
    for start in starts:
        end = min(start + window, num_frames)
        chunk = features[:, start * FEATURE_RATE : end * FEATURE_RATE].unsqueeze(0)
        _, acoustic_logits = encoder(chunk, c0_teacher=c0_ids[start:end].unsqueeze(0))
        acoustic_sum[start:end] += F.log_softmax(acoustic_logits.squeeze(0).float(), dim=-1)
    return c0_logprobs, acoustic_sum / counts[:, None, None]


@torch.no_grad()
def encode_wav_to_logprobs(
    wav_path: Path, encoder: CodesEncoder, dav, device: str, window: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """wav → DAV latent → windowed encoder log-probs (c0, acoustic)."""
    from .cache_audio import load_wav_44k_stereo
    from .dav import SAMPLE_RATE

    waveform = load_wav_44k_stereo(wav_path).to(device)
    latent = dav.encode(waveform).squeeze(0)
    num_frames = int(round(waveform.shape[-1] / SAMPLE_RATE * AUDIO_FRAMES_PER_SECOND))
    features = latent_to_features(latent, num_frames).to(device)
    return windowed_logprobs(encoder, features, num_frames, window)


def logprobs_to_codes(c0_logprobs: torch.Tensor, acoustic_logprobs: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [c0_logprobs.argmax(-1).unsqueeze(-1), acoustic_logprobs.argmax(-1)], dim=-1
    )


@torch.no_grad()
def encode_wav_to_codes(
    wav_path: Path, encoder: CodesEncoder, dav, device: str, window: int | None = None
) -> torch.Tensor:
    """wav → predicted codes [T, 8] (argmax of the averaged window log-probs)."""
    c0_logprobs, acoustic_logprobs = encode_wav_to_logprobs(wav_path, encoder, dav, device, window)
    return logprobs_to_codes(c0_logprobs, acoustic_logprobs).cpu()


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
    parser.add_argument("--refine", type=int, default=0, help="AR-prior re-decoding iterations (loads the 8B + depth decoder)")
    parser.add_argument("--refine-lambda", type=float, default=0.5, help="prior weight — fidelity dial, verify recos when changing")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    encoder = load_encoder(args.model, device=args.device)
    dav = load_dav(str(models_dir() / "dav.pth"), device=args.device)
    ar_model = tokenizer = None
    if args.refine:
        from .loading import load_music3_ar, load_tokenizer

        tokenizer = load_tokenizer()
        ar_model = load_music3_ar(quantize=True, device=args.device, with_depth=True)
    args.out.mkdir(parents=True, exist_ok=True)

    wavs = sorted(args.audio_dir.glob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    for wav_path in wavs:
        sidecar = wav_path.with_suffix(".txt")
        caption, lyrics = "", ""
        if sidecar.exists():
            fields = parse_sidecar(sidecar)
            caption, lyrics = compose_caption(fields), fields.get("lyrics", "")
        c0_logprobs, acoustic_logprobs = encode_wav_to_logprobs(
            wav_path, encoder, dav, args.device
        )
        if args.refine:
            from .prompt import encode_prompt
            from .refine import refine_codes

            prompt_ids = torch.tensor(
                [encode_prompt(tokenizer, caption, lyrics)], device=args.device
            )
            codes = refine_codes(
                ar_model, prompt_ids, c0_logprobs, acoustic_logprobs,
                lam=args.refine_lambda, iterations=args.refine,
            )
        else:
            codes = logprobs_to_codes(c0_logprobs, acoustic_logprobs)
        save_file(
            {"codes": codes.to(torch.int32).cpu()},
            str(args.out / f"{wav_path.stem}.safetensors"),
            metadata={"caption": caption, "lyrics": lyrics},
        )
        print(f"{wav_path.stem}: {codes.shape[0]} frames ({codes.shape[0] / 25:.1f}s)"
              + (f" [refined x{args.refine}, λ={args.refine_lambda}]" if args.refine else ""))


if __name__ == "__main__":
    main()
