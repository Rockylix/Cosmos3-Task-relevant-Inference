"""Explicit single-sample latent axes for experiment-only metrics."""


def future_vision(latent):
    """Return L1..L8 as [C,8,H,W], never slice a channel as time."""
    if latent.ndim == 5:
        if latent.shape[0] != 1:
            raise ValueError("Only one vision sample is supported")
        latent = latent[0]
    if latent.ndim != 4 or latent.shape[1] != 9:
        raise ValueError(f"Expected [C,9,H,W] or [1,C,9,H,W], got {latent.shape}")
    return latent[:, 1:]
