import torch
from safetensors.torch import save_file

from music3_tuner.dataset import CodesDataset, collate_codes, compose_caption, parse_sidecar
from music3_tuner.prompt import SPECIAL_TOKEN_IDS


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


def test_parse_sidecar(tmp_path):
    sidecar = tmp_path / "track.txt"
    sidecar.write_text(
        "caption: Driving synthwave banger.\n"
        "genre: electronic, synthwave\n"
        "bpm: 120\n"
        "key: A minor\n"
        "signature: 4/4\n"
        "is_instrumental: \n"
        "lyrics:\n"
        "[Verse]\n"
        "neon lights ahead\n"
    )
    fields = parse_sidecar(sidecar)
    assert fields["caption"] == "Driving synthwave banger."
    assert fields["bpm"] == "120"
    assert fields["lyrics"] == "[Verse]\nneon lights ahead"
    caption = compose_caption(fields)
    assert "Driving synthwave banger." in caption
    assert "bpm: 120" in caption


def test_collate_left_pads():
    items = [
        {
            "prompt_ids": [1, 2, 3],
            "codes": torch.zeros(4, 8, dtype=torch.long),
            "primer_codes": torch.arange(8).reshape(1, 8),
            "reaches_end": True,
        },
        {"prompt_ids": [1, 2, 3, 4, 5], "codes": torch.zeros(4, 8, dtype=torch.long), "reaches_end": False},
    ]
    batch = collate_codes(items)
    assert batch.prompt_ids.shape == (2, 5)
    assert batch.prompt_ids[0, 0] == SPECIAL_TOKEN_IDS["<|im_end|>"]
    assert batch.prompt_mask[0].tolist() == [False, False, True, True, True]
    assert batch.prompt_ids[0, 2:].tolist() == [1, 2, 3]
    assert batch.codes.shape == (2, 4, 8)
    assert batch.primer_codes.shape == (2, 1, 8)
    assert batch.primer_codes[0, 0].tolist() == list(range(8))
    assert batch.primer_codes[1].count_nonzero() == 0
    assert batch.primer_mask.tolist() == [True, False]
    assert batch.supervise_audio_end is False


def test_dataset_uses_explicit_termination_and_prefix_crops_primer_cache(tmp_path):
    codes = torch.arange(6 * 8, dtype=torch.int32).reshape(6, 8)
    primer = torch.arange(8, dtype=torch.int32).reshape(1, 8)

    generated = tmp_path / "generated"
    generated.mkdir()
    save_file(
        {"codes": codes, "primer_codes": primer},
        generated / "track.safetensors",
        metadata={"caption": "test", "lyrics": "[Instrumental]", "termination": "audio_end"},
    )
    item = CodesDataset(generated, Tokenizer(), max_frames=4, seed=0)[0]
    assert torch.equal(item["codes"], codes[:4].long())
    assert torch.equal(item["primer_codes"], primer.long())
    assert item["reaches_end"] is False

    complete = tmp_path / "complete"
    complete.mkdir()
    save_file(
        {"codes": codes[:4], "primer_codes": primer},
        complete / "track.safetensors",
        metadata={"termination": "audio_end"},
    )
    assert CodesDataset(complete, Tokenizer(), max_frames=4)[0]["reaches_end"] is True

    capped = tmp_path / "capped"
    capped.mkdir()
    save_file(
        {"codes": codes[:4], "primer_codes": primer},
        capped / "track.safetensors",
        metadata={"termination": "max_frames"},
    )
    assert CodesDataset(capped, Tokenizer(), max_frames=4)[0]["reaches_end"] is False

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    save_file({"codes": codes[:4]}, legacy / "track.safetensors")
    legacy_item = CodesDataset(legacy, Tokenizer(), max_frames=4)[0]
    assert legacy_item["primer_codes"] is None
    assert legacy_item["reaches_end"] is False
