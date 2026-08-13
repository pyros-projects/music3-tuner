from pathlib import Path

import pytest
import torch

WAV_DIR = Path("/home/pyro/music/ace/audio/wav_neon")


@pytest.mark.weights
def test_dav_loads_and_roundtrips():
    from music3_tuner.cache_audio import load_wav_44k_stereo, snr_db
    from music3_tuner.dav import HOP, load_dav
    from music3_tuner.loading import models_dir

    dav_path = models_dir() / "dav.pth"
    if not dav_path.exists():
        pytest.skip("dav.pth missing")
    wavs = sorted(WAV_DIR.glob("*.wav"))
    if not wavs:
        pytest.skip("no smoke wavs")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dav = load_dav(str(dav_path), device=device)
    waveform = load_wav_44k_stereo(wavs[0])[..., : 44100 * 4].to(device)

    latent = dav.encode(waveform)
    assert latent.shape[0] == 1 and latent.shape[1] == 128
    expected_frames = waveform.shape[-1] // HOP
    assert abs(latent.shape[2] - expected_frames) <= 2

    decoded = dav.decode(latent)
    assert decoded.shape[1] == 2
    score = snr_db(waveform.cpu(), decoded.cpu())
    # A correct port reconstructs audibly; a broken key-mapping yields noise
    # (negative SNR). The bar is deliberately conservative.
    assert score > 0.0, f"roundtrip SNR {score:.2f} dB — encoder/decoder mismatch?"
