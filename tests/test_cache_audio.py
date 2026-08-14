import torch

from music3_tuner.cache_audio import correlation


def test_correlation_crops_stereo_on_time_axis_before_flattening():
    reference = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0]]])
    estimate = reference[..., :3]

    assert correlation(reference, estimate) == 1.0
