"""MiniMax-Music3 DAV Flow-VAE (dav.pth) — encoder AND decoder.

DAC-style convolutional codec at 44.1 kHz stereo. Each stereo channel is
folded into the batch and processed as mono; the per-channel latent is 64
dims at hop 512 (~86.13 latent frames/s), so a stereo latent is [B, 128, T].

Checkpoint layout (548 keys): encoder(119) / mean_proj / logs_proj /
dec_in_proj / decoder(119) / flow(304). The flow (a VITS-style residual
coupling stack) refines the prior and is NOT needed for encode→decode
round-trips; we skip loading it here.

NOTE: this is a *continuous* VAE — it contains no RVQ quantizer. The
discrete 8-codebook representation the LMs use is produced by an internal
MiniMax tokenizer that was not released.
"""

from __future__ import annotations

import math

import torch
from torch import nn

SAMPLE_RATE = 44100
HOP = 512  # 2*4*8*8 (encoder strides)
LATENT_DIM_PER_CHANNEL = 64
ENCODER_STRIDES = (2, 4, 8, 8)
DECODER_STRIDES = (8, 8, 4, 2)


def _wn_conv(*args, **kwargs) -> nn.Module:
    return nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


def _wn_conv_transpose(*args, **kwargs) -> nn.Module:
    return nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    flat = x.reshape(shape[0], shape[1], -1)
    flat = flat + (alpha + 1e-9).reciprocal() * torch.sin(alpha * flat).pow(2)
    return flat.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(dim),
            _wn_conv(dim, dim, kernel_size=7, dilation=dilation, padding=3 * dilation),
            Snake1d(dim),
            _wn_conv(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        residual = self.block(x)
        if residual.shape[-1] != x.shape[-1]:
            pad = (x.shape[-1] - residual.shape[-1]) // 2
            x = x[..., pad : x.shape[-1] - pad]
        return x + residual


class EncoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(input_dim, 1),
            ResidualUnit(input_dim, 3),
            ResidualUnit(input_dim, 9),
            Snake1d(input_dim),
            _wn_conv(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """1 ch → 1024 ch at hop 512. Key layout matches dav.pth (`encoder.block.N`)."""

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = [_wn_conv(1, 64, kernel_size=7, padding=3)]
        dim = 64
        for stride in ENCODER_STRIDES:
            layers.append(EncoderBlock(dim, dim * 2, stride))
            dim *= 2
        layers.extend([Snake1d(dim), _wn_conv(dim, dim, kernel_size=3, padding=1)])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            _wn_conv_transpose(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, 1),
            ResidualUnit(output_dim, 3),
            ResidualUnit(output_dim, 9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    """1024 ch latent-projection → mono waveform. Key layout: `decoder.model.N`."""

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = [_wn_conv(1024, 1536, kernel_size=7, padding=3)]
        channels = 1536
        output_dim = channels
        for index, stride in enumerate(DECODER_STRIDES):
            input_dim = channels // (2**index)
            output_dim = channels // (2 ** (index + 1))
            layers.append(DecoderBlock(input_dim, output_dim, stride))
        layers.extend(
            [
                Snake1d(output_dim),
                _wn_conv(output_dim, 1, kernel_size=7, padding=3),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MusicDav(nn.Module):
    """Encoder + VAE posterior + decoder from dav.pth (flow keys ignored)."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.mean_proj = nn.Conv1d(1024, LATENT_DIM_PER_CHANNEL, kernel_size=1)
        self.logs_proj = nn.Conv1d(1024, LATENT_DIM_PER_CHANNEL, kernel_size=1)
        self.dec_in_proj = nn.Conv1d(LATENT_DIM_PER_CHANNEL, 1024, kernel_size=1)
        self.decoder = Decoder()

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
        """[B, 2, samples] @ 44.1 kHz → latent [B, 128, T] (stereo folded)."""
        batch, channels, samples = waveform.shape
        assert channels == 2, "MusicDav expects stereo input; duplicate mono first"
        folded = waveform.reshape(batch * 2, 1, samples)
        hidden = self.encoder(folded)
        mean = self.mean_proj(hidden)
        if sample_posterior:
            logs = self.logs_proj(hidden)
            mean = mean + torch.randn_like(mean) * torch.exp(logs)
        return mean.reshape(batch, 2 * LATENT_DIM_PER_CHANNEL, -1)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """[B, 128, T] → waveform [B, 2, samples]."""
        batch, _, frames = latent.shape
        folded = latent.reshape(batch * 2, LATENT_DIM_PER_CHANNEL, frames)
        waveform = self.decoder(self.dec_in_proj(folded))
        return waveform.reshape(batch, 2, -1)


def load_dav(path: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> MusicDav:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = MusicDav()
    # Old-style weight_norm keys (weight_g/weight_v) are remapped by torch's
    # parametrization compat hook; flow.* keys belong to the unused prior flow.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"dav.pth missing keys for MusicDav: {missing[:8]}")
    leftovers = [k for k in unexpected if not k.startswith("flow.")]
    if leftovers:
        raise RuntimeError(f"unexpected non-flow keys: {leftovers[:8]}")
    return model.to(device=device, dtype=dtype).eval()
