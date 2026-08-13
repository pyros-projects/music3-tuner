import torch

from music3_tuner.dataset import collate_codes, compose_caption, parse_sidecar
from music3_tuner.prompt import SPECIAL_TOKEN_IDS


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
        {"prompt_ids": [1, 2, 3], "codes": torch.zeros(4, 8, dtype=torch.long), "reaches_end": True},
        {"prompt_ids": [1, 2, 3, 4, 5], "codes": torch.zeros(4, 8, dtype=torch.long), "reaches_end": False},
    ]
    batch = collate_codes(items)
    assert batch.prompt_ids.shape == (2, 5)
    assert batch.prompt_ids[0, 0] == SPECIAL_TOKEN_IDS["<|im_end|>"]
    assert batch.prompt_mask[0].tolist() == [False, False, True, True, True]
    assert batch.prompt_ids[0, 2:].tolist() == [1, 2, 3]
    assert batch.codes.shape == (2, 4, 8)
    assert batch.supervise_audio_end is False
