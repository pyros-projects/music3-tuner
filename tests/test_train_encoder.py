from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from music3_tuner.encoder import FEATURE_RATE
from music3_tuner.train_encoder import (
    PairsDataset,
    collate,
    evaluate,
    pair_family,
    select_ar_panel,
    selection_score,
    split_pairs,
)


def save_pair(path: Path, frames: int = 12) -> None:
    save_file(
        {
            "latent": torch.zeros(4, 17),
            "codes": torch.arange(frames).unsqueeze(1).expand(-1, 8).int(),
        },
        path,
    )


def test_split_pairs_keeps_seed_variants_in_one_partition(tmp_path):
    paths = [
        tmp_path / f"template_{family:02}_s{seed}.safetensors"
        for family in range(12)
        for seed in (1, 2)
    ]
    for path in paths:
        path.touch()

    train, val = split_pairs(tmp_path, 0.5)
    partitions = {path: "train" for path in train} | {path: "val" for path in val}

    assert train and val
    for family in range(12):
        assert partitions[tmp_path / f"template_{family:02}_s1.safetensors"] == partitions[
            tmp_path / f"template_{family:02}_s2.safetensors"
        ]


def test_ar_panel_is_seeded_and_contains_distinct_families():
    paths = [
        Path(f"template_{family:02}_s{seed}.safetensors")
        for family in range(12)
        for seed in (1, 2)
    ]

    panel = select_ar_panel(paths, count=4, seed=42)
    alphabetical = sorted({pair_family(path): path for path in reversed(paths)}.values())[:4]

    assert panel == select_ar_panel(paths, count=4, seed=42)
    assert panel != select_ar_panel(paths, count=4, seed=43)
    assert panel != alphabetical
    assert len({pair_family(path) for path in panel}) == len(panel) == 4


def test_validation_dataset_returns_the_complete_track(tmp_path):
    path = tmp_path / "track_s1.safetensors"
    save_pair(path, frames=12)

    item = PairsDataset([path], crop_frames=5, train=False, augment=False)[0]

    assert item["codes"].shape == (12, 8)
    assert item["features"].shape == (4, 12 * FEATURE_RATE)


def test_workers_do_not_replay_the_same_crop_rng(tmp_path):
    paths = [tmp_path / f"track_{index}_s1.safetensors" for index in range(2)]
    for path in paths:
        save_pair(path, frames=64)
    loader = DataLoader(
        PairsDataset(paths, crop_frames=8, train=True, augment=False),
        batch_size=1,
        collate_fn=collate,
        num_workers=2,
        generator=torch.Generator().manual_seed(0),
    )

    batches = list(loader)
    starts = [batch["codes"][0, 0, 0].item() for batch in batches]

    assert starts[0] != starts[1]


class TinyMetricModel(torch.nn.Module):
    config = {"num_codebooks": 3}


def test_evaluate_scores_full_tracks_through_windowed_path(monkeypatch):
    calls = []

    def fake_windowed(model, features, num_frames, window):
        calls.append((features.shape, num_frames, window))
        c0 = torch.zeros(num_frames, 8)
        acoustic = torch.zeros(num_frames, 2, 4)
        c0[:, 0] = 1
        acoustic[:, :, 0] = 1
        return c0, acoustic

    monkeypatch.setattr("music3_tuner.train_encoder.windowed_logprobs", fake_windowed)
    batch = {
        "features": torch.zeros(1, 4, 7 * FEATURE_RATE),
        "codes": torch.zeros(1, 7, 3, dtype=torch.long),
    }

    metrics = evaluate(TinyMetricModel(), [batch], device="cpu", window=3)

    assert calls == [(torch.Size([4, 7 * FEATURE_RATE]), 7, 3)]
    assert metrics == {
        "frames": 7,
        "c0_top1": 1.0,
        "c0_top5": 1.0,
        "acoustic_top1": 1.0,
        "books": [1.0, 1.0],
    }


def test_checkpoint_selection_prefers_c0_then_acoustic():
    assert selection_score(
        {"c0_top1": 0.3, "acoustic_top1": 0.01, "ar_loss": 9.0}
    ) > selection_score(
        {"c0_top1": 0.29, "acoustic_top1": 0.99, "ar_loss": 1.0}
    )
    assert selection_score(
        {"c0_top1": 0.3, "acoustic_top1": 0.02, "ar_loss": 9.0}
    ) > selection_score(
        {"c0_top1": 0.3, "acoustic_top1": 0.01, "ar_loss": 1.0}
    )
