"""Request-local, post-CFG flow-velocity reuse for ASI's excluded future positions.

No attention V caching, no second integration, no scheduler/history modification.
The native UniPC sampler consumes the mixed velocity exactly once per step.
"""

from __future__ import annotations

import math

import torch


def full_velocity_keep_mask(execution_mask, vision_shape, grid_thw, patch_size, flat_size):
    """Expand exact patch footprints, crop bottom/right padding; keep L0/actions."""
    shape = tuple(map(int, vision_shape))
    if len(shape) == 5 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) != 4:
        raise ValueError(f"Expected one [C,T,H,W] vision latent, got {vision_shape}")
    channels, frames, height, width = shape
    grid_t, grid_h, grid_w = map(int, grid_thw)
    if (
        frames != 9
        or grid_t != frames
        or patch_size < 1
        or (grid_h, grid_w) != (math.ceil(height / patch_size), math.ceil(width / patch_size))
        or tuple(execution_mask.shape) != (8, grid_h * grid_w)
        or execution_mask.dtype != torch.bool
    ):
        raise ValueError("Execution mask and native patch layout do not agree")
    vision_size = math.prod(shape)
    if flat_size <= vision_size:
        raise ValueError("This policy experiment requires a [vision | action] velocity")
    spatial = execution_mask.reshape(8, grid_h, grid_w)
    spatial = spatial.repeat_interleave(patch_size, -2).repeat_interleave(patch_size, -1)
    spatial = spatial[:, :height, :width]
    video_keep = torch.ones(shape, dtype=torch.bool, device=execution_mask.device)
    video_keep[:, 1:] = spatial.unsqueeze(0)
    keep = torch.ones(flat_size, dtype=torch.bool, device=execution_mask.device)
    keep[:vision_size] = video_keep.reshape(-1)
    return keep


class RemovedTokenVelocityCache:
    """Cache only excluded coordinates of the first fully guided velocity."""

    def __init__(self, keep_mask, *, audit=True):
        if keep_mask.ndim != 1 or keep_mask.dtype != torch.bool:
            raise ValueError("Expected a flat boolean keep mask")
        self.keep = keep_mask.clone()
        self.removed_indices = torch.nonzero(~self.keep, as_tuple=False).flatten()
        self.audit = audit
        self.cache = None
        self.next_step = 0

    def apply(self, step, velocity):
        if step != self.next_step or not 0 <= step < 4:
            raise ValueError("Cache requires ordered steps 0,1,2,3 in one request")
        if velocity.ndim != 1 or velocity.shape != self.keep.shape or velocity.device != self.keep.device:
            raise ValueError("Velocity layout/device changed")
        if self.audit and not torch.isfinite(velocity).all():
            raise FloatingPointError("Nonfinite current velocity")
        if step == 0:
            self.cache = velocity.index_select(0, self.removed_indices).detach()
            result = velocity
        else:
            if self.cache is None or self.cache.dtype != velocity.dtype:
                raise RuntimeError("Step0 cache missing or dtype changed")
            result = velocity.clone()
            result.index_copy_(0, self.removed_indices, self.cache)
        self.next_step += 1
        return result


class CacheSamplerAdapter:
    """Wrap velocity_fn at UniPC's public boundary, AFTER native CFG/masking."""

    def __init__(
        self, native_sampler, controller, layout, patch_size, enabled, *, step0_source="dense_cfg", audit=True
    ):
        if step0_source not in ("dense_cfg", "asi_cfg"):
            raise ValueError("Unknown Step0 guided velocity source")
        self.native_sampler = native_sampler
        self.controller = controller
        self.layout = layout
        self.patch_size = patch_size
        self.enabled = enabled
        self.step0_source = step0_source
        self.audit = audit
        self.cache = None
        self.initial_velocity = None
        self.records = []
        self.called = False
        self.evaluations = 0

    def __call__(self, velocity_fn, noise, **kwargs):
        if self.called or kwargs.get("num_steps") != 4:
            raise RuntimeError("Adapter is single-request, four-step only")
        if not isinstance(noise, list) or len(noise) != 1:
            raise ValueError("This experiment supports exactly one captured request")
        if self.enabled:
            require_dense = self.step0_source == "dense_cfg"
            if self.controller is None or self.controller.dense_step0 != require_dense:
                if require_dense:
                    raise ValueError("Velocity cache requires dense conditional AND unconditional step0")
                raise ValueError("ASI cache requires one dense conditional and one sparse unconditional Step0")
        self.called = True

        def mixed_velocity(latent, timestep):
            current = velocity_fn(latent, timestep)
            step = self.evaluations
            if len(current) != 1 or step >= 4:
                raise RuntimeError("Unexpected sampler evaluations")
            value = current[0]
            if self.audit and not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite native guided velocity")
            if step == 0:
                if self.audit:
                    self.initial_velocity = value.detach().cpu().clone()
                if self.enabled:
                    expected = (2, 0) if self.step0_source == "dense_cfg" else (1, 1)
                    if (self.controller._dense_stacks, self.controller._sparse_stacks) != expected:
                        raise RuntimeError(f"Step0 velocity has incorrect dense/sparse counts; expected {expected}")
                    keep = full_velocity_keep_mask(
                        self.controller.plan["execution_mask"],
                        self.layout["vision_shape"],
                        self.layout["grid_thw"],
                        self.patch_size,
                        value.numel(),
                    )
                    self.cache = RemovedTokenVelocityCache(keep, audit=self.audit)
            result = self.cache.apply(step, value) if self.enabled else value
            if self.enabled and self.audit:
                kept_exact = torch.equal(result[self.cache.keep], value[self.cache.keep])
                cached_exact = torch.equal(result[~self.cache.keep], self.cache.cache)
                if not kept_exact or not cached_exact:
                    raise AssertionError("Cache changed kept tokens/actions or failed exact replacement")
                removed = int((~self.cache.keep).sum())
            else:
                kept_exact, cached_exact, removed = True, None, 0
            if self.audit:
                self.records.append(
                    {
                        "step": step,
                        "timestep": float(timestep.reshape(-1)[0]),
                        "all_finite": bool(torch.isfinite(result).all()),
                        "reused_velocity_scalars": removed if step > 0 else 0,
                        "kept_and_action_unchanged": kept_exact,
                        "removed_equal_step0_cache": cached_exact,
                    }
                )
            self.evaluations += 1
            return [result]

        result = self.native_sampler(mixed_velocity, noise, **kwargs)
        if self.evaluations != 4:
            raise RuntimeError("UniPC did not finish all four steps")
        return result
