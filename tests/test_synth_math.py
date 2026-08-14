from music3_tuner.synth import (
    CHUNK_FRAMES,
    CHUNK_HOP,
    LATENT_HOP,
    chunk_starts,
    crop_bounds,
)

LATENTS_PER_FRAME = 44100 / 24000 * 960 / 512  # 3.4453125


def window_latents(start: int, num_frames: int) -> int:
    frames = min(CHUNK_FRAMES, num_frames - start)
    return max(1, int(frames * LATENTS_PER_FRAME))


def test_chunk_starts_short_track():
    assert chunk_starts(200) == [0]
    assert chunk_starts(50) == [0]


def test_chunk_starts_hop_covers_all_frames():
    starts = chunk_starts(751)
    assert starts[0] == 0
    assert all(b - a == CHUNK_HOP for a, b in zip(starts, starts[1:]))
    assert starts[-1] + CHUNK_FRAMES >= 751


def test_crops_tile_the_song():
    """Stitched waveform length must match the full-track latent timeline:
    last window's start offset plus its own latent length (within int()
    rounding slack per window boundary)."""
    for num_frames in (200, 751, 1000, 3000):
        starts = chunk_starts(num_frames)
        kept_samples = 0
        for index, start in enumerate(starts):
            latents = window_latents(start, num_frames)
            left, right = crop_bounds(index, len(starts), latents)
            assert 0 <= left <= right
            kept_samples += right - left
        expected_latents = int(starts[-1] * LATENTS_PER_FRAME) + window_latents(starts[-1], num_frames)
        slack = LATENT_HOP * len(starts)  # int() rounding per window boundary
        assert abs(kept_samples - expected_latents * LATENT_HOP) <= slack
