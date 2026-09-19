"""RGB observation patches aligned to the native VAE/GEN grid, without inference."""

import math

import torch
import torch.nn.functional as F


def condition_rgb(data_batch, key="video"):
    """Read the actual post-transform, pre-normalization condition frame only."""
    value = data_batch[key]
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Observation Dynamic supports one sample and one vision item")
        value = value[0]
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4 or value.shape[0] != 3 or value.dtype != torch.uint8:
        raise ValueError("Expected original uint8 [3,T,H,W] transformed video")
    return value[:, 0].detach().clone()


def observation_patches(rgb, *, latent_hw, latent_patch_size, spatial_factor, image_size):
    """Match native top-left latent unpadding and bottom/right patch padding.

    A p-by-p latent patch maps to a nominal (p*spatial_factor)-square RGB
    bin. This is coordinate alignment, not an exact convolutional receptive field.
    RGB beyond the latent floor-crop and any artificial padding is excluded.
    """
    if rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.dtype != torch.uint8:
        raise ValueError("Expected uint8 RGB [3,H,W]")
    h, w = map(int, latent_hw)
    p, sf = int(latent_patch_size), int(spatial_factor)
    size = torch.as_tensor(image_size).flatten().tolist()
    if len(size) != 4 or any(float(x) != int(x) for x in size) or min(h, w, p, sf) <= 0:
        raise ValueError("Invalid image_size/latent geometry")
    th, tw, oh, ow = map(int, size)
    if (th, tw) != tuple(rgb.shape[-2:]) or oh > th or ow > tw or min(oh, ow) <= 0:
        raise ValueError("image_size does not describe the transformed RGB")
    if (max(oh // sf, 1), max(ow // sf, 1)) != (h, w):
        raise ValueError("Observation/latent crop mismatch; refusing guessed grid resize")
    gh, gw = math.ceil(h / p), math.ceil(w / p)
    side = p * sf
    vh, vw = min(oh, h * sf), min(ow, w * sf)
    image = rgb[:, :vh, :vw].float()
    padded = F.pad(image, (0, gw * side - vw, 0, gh * side - vh))
    patches = padded.reshape(3, gh, side, gw, side).permute(1, 3, 0, 2, 4).reshape(gh * gw, -1)
    meta = dict(
        transformed_hw=[th, tw],
        original_hw=[oh, ow],
        latent_hw=[h, w],
        grid_hw=[gh, gw],
        nominal_rgb_patch_side=side,
        compared_rgb_hw=[vh, vw],
        image_size=size,
        mapping="top_left_native_latent_floor_crop_then_zero_pad_patch_bins",
        similarity_input="RGB_0_255_no_centering_no_added_resize",
    )
    return patches, meta
