import numpy as np
import pytest

from cosmos_framework.inference.future_fidelity_metrics import rgb_frame_metrics, tensor_metrics, to_rgb01


def test_identical_and_scaled():
    x = np.arange(1, 13).reshape(3, 4)
    same = tensor_metrics(x, x)
    assert same["mse"] == 0 and same["relative_l2"] == 0
    assert same["cosine"] == pytest.approx(1)
    scaled = tensor_metrics(x, x / 2)
    assert scaled["relative_l2"] == pytest.approx(0.5)
    assert scaled["cosine"] == pytest.approx(1)


def test_rgb_ssim_psnr_and_shared_conversion():
    x = np.random.default_rng(0).uniform(0, 1, (16, 18, 3)).astype(np.float32)
    same = rgb_frame_metrics(x, x)
    assert same["ssim"] == 1 and same["psnr_db"] is None and same["psnr_infinite"]
    changed = rgb_frame_metrics(x, x * 0.5)
    assert changed["ssim"] < 1 and np.isfinite(changed["psnr_db"])
    # Even if a decoded tensor happens to be entirely positive, do NOT switch
    # to a different per-sample/per-strategy range conversion.
    positive = np.full((1, 3, 2, 12, 12), 0.5)
    assert np.all(to_rgb01(positive) == 0.75)
    assert to_rgb01(positive).shape == (2, 12, 12, 3)


def test_invalid_and_zero_cosine():
    with pytest.raises(ValueError, match="NaN/Inf"):
        tensor_metrics([1, 2], [1, np.nan])
    with pytest.raises(ValueError, match="Mismatched"):
        tensor_metrics([1, 2], [1])
    assert tensor_metrics([0, 0], [1, 2])["cosine"] is None
