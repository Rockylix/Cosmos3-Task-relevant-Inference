"""Offline paired-output metrics. RGB inputs are float [0,1], not PNG bytes."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def tensor_metrics(reference, candidate):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or not a.size:
        raise ValueError("Mismatched or empty paired tensors")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("NaN/Inf in paired tensors")
    a, b = a.ravel(), b.ravel()
    delta = b - a
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {
        "mse": float(np.mean(delta**2)),
        "cosine": float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else None,
        "relative_l2": float(np.linalg.norm(delta) / max(na, 1e-12)),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }


def rgb_frame_metrics(reference, candidate):
    a, b = np.asarray(reference, dtype=np.float32), np.asarray(candidate, dtype=np.float32)
    if a.ndim != 3 or a.shape[-1] != 3 or min(a.shape[:2]) < 11:
        raise ValueError("Expected RGB [height,width,3] with spatial extent >=11")
    metrics = tensor_metrics(a, b)
    if min(a.min(), b.min()) < 0 or max(a.max(), b.max()) > 1:
        raise ValueError("RGB must already use a shared [0,1] conversion")
    metrics["psnr_db"] = float(-10 * np.log10(metrics["mse"])) if metrics["mse"] > 0 else None
    metrics["psnr_infinite"] = metrics["mse"] == 0
    metrics["ssim"] = float(
        structural_similarity(
            a,
            b,
            data_range=1.0,
            channel_axis=-1,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            win_size=11,
        )
    )
    return metrics


def to_rgb01(decoded):
    """Wan VAE uses [-1,1]; apply one fixed mapping to ALL strategies."""
    value = np.asarray(decoded, dtype=np.float32)
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4 or value.shape[0] != 3 or not np.isfinite(value).all():
        raise ValueError("Expected finite decoded [1,3,T,H,W] or [3,T,H,W]")
    return np.ascontiguousarray(np.clip((value + 1.0) / 2.0, 0, 1).transpose(1, 2, 3, 0))
