from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.inference.asi_velocity_cache import (
    CacheSamplerAdapter,
    RemovedTokenVelocityCache,
    full_velocity_keep_mask,
)


def test_native_grid_patch_footprints_and_padding():
    mask = torch.zeros(8, 340, dtype=torch.bool)
    mask[0, 0] = True
    mask[1, -1] = True
    shape = (1, 48, 9, 33, 40)
    n = 48 * 9 * 33 * 40
    keep = full_velocity_keep_mask(mask, shape, (9, 17, 20), 2, n + 264)
    spatial = keep[:n].reshape(48, 9, 33, 40)
    assert spatial[:, 0].all() and keep[n:].all()
    assert spatial[:, 1, :2, :2].all()
    assert spatial[:, 1].sum() == 48 * 4
    assert spatial[:, 2, -1, -2:].all()
    assert spatial[:, 2].sum() == 48 * 2  # final padded row is cropped, not shifted
    assert not spatial[:, 3:].any()


def test_replace_not_add_and_no_inplace_mutation():
    keep = torch.tensor([True, False, True, False, True])
    cache = RemovedTokenVelocityCache(keep)
    v0 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert torch.equal(cache.apply(0, v0), v0)
    v0.fill_(99)  # cached storage must be independent
    for step in range(1, 4):
        current = torch.full((5,), float(step * 10))
        result = cache.apply(step, current)
        assert torch.equal(result[~keep], torch.tensor([2.0, 4.0]))
        assert torch.equal(result[keep], current[keep])
        assert torch.equal(current, torch.full((5,), float(step * 10)))


def test_order_finite_and_request_isolation():
    keep = torch.tensor([True, False])
    one, two = RemovedTokenVelocityCache(keep), RemovedTokenVelocityCache(keep)
    with pytest.raises(ValueError, match="ordered"):
        one.apply(1, torch.ones(2))
    with pytest.raises(FloatingPointError):
        one.apply(0, torch.tensor([1.0, float("nan")]))
    one.apply(0, torch.tensor([1.0, 2.0]))
    two.apply(0, torch.tensor([1.0, 9.0]))
    assert one.apply(1, torch.ones(2))[1] == 2
    assert two.apply(1, torch.ones(2))[1] == 9


def test_wrong_geometry_rejected():
    with pytest.raises(ValueError):
        full_velocity_keep_mask(torch.ones(8, 340, dtype=torch.bool), (48, 9, 33, 40), (9, 20, 17), 2, 570504)


def test_adapter_passes_mixed_velocity_to_native_unipc():
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler

    native = UniPCSampler(tensor_kwargs={"device": "cpu", "dtype": torch.float32})
    execution = torch.zeros(8, 4, dtype=torch.bool)
    execution[:, 0] = True
    controller = SimpleNamespace(
        dense_step0=True, _dense_stacks=2, _sparse_stacks=0, plan={"execution_mask": execution}
    )
    layout = {"vision_shape": (1, 9, 3, 4), "grid_thw": (9, 2, 2)}
    size = 1 * 9 * 3 * 4 + 2
    keep = full_velocity_keep_mask(execution, layout["vision_shape"], layout["grid_thw"], 2, size)

    def velocity(latent, timestep):
        return [torch.full_like(latent[0], float(timestep.item()) / 1000)]

    adapter = CacheSamplerAdapter(native, controller, layout, 2, True)
    actual = adapter(velocity, [torch.zeros(size)], num_steps=4, shift=5.0, seed=[7])
    initial = []

    def reference_velocity(latent, timestep):
        value = velocity(latent, timestep)[0]
        if not initial:
            initial.append(value.clone())
        return [torch.where(keep, value, initial[0])]

    expected = native(reference_velocity, [torch.zeros(size)], num_steps=4, shift=5.0, seed=[7])
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert len(adapter.records) == 4
    assert all(row["kept_and_action_unchanged"] for row in adapter.records)
    assert all(row["removed_equal_step0_cache"] for row in adapter.records)
    with pytest.raises(RuntimeError, match="single-request"):
        adapter(velocity, [torch.zeros(size)], num_steps=4, shift=5.0, seed=[7])


def test_cache_requires_both_cfg_branches_dense():
    adapter = CacheSamplerAdapter(None, SimpleNamespace(dense_step0=False), {}, 2, True)
    with pytest.raises(ValueError, match="conditional AND unconditional"):
        adapter(None, [torch.zeros(5)], num_steps=4)


@pytest.mark.parametrize("audit", [True, False])
def test_single_dense_cache_matches_explicit_unipc_reference(audit):
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler

    native = UniPCSampler(tensor_kwargs={"device": "cpu", "dtype": torch.float32})
    execution = torch.zeros(8, 4, dtype=torch.bool)
    execution[:, 1] = True
    controller = SimpleNamespace(
        dense_step0=False, _dense_stacks=1, _sparse_stacks=1, plan={"execution_mask": execution}
    )
    layout = {"vision_shape": (1, 9, 3, 4), "grid_thw": (9, 2, 2)}
    size = 110
    keep = full_velocity_keep_mask(execution, layout["vision_shape"], layout["grid_thw"], 2, size)

    def velocity(latent, timestep):
        return [latent[0] * 0.1 + timestep.to(latent[0]).reshape(-1)[0] / 1000]

    initial = []

    def reference(latent, timestep):
        value = velocity(latent, timestep)[0]
        if not initial:
            initial.append(value.clone())
        return [torch.where(keep, value, initial[0])]

    adapter = CacheSamplerAdapter(native, controller, layout, 2, True, step0_source="asi_cfg", audit=audit)
    actual = adapter(velocity, [torch.zeros(size)], num_steps=4, shift=5.0, seed=[7])
    expected = native(reference, [torch.zeros(size)], num_steps=4, shift=5.0, seed=[7])
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert adapter.evaluations == 4 and adapter.cache.next_step == 4
    assert len(adapter.records) == (4 if audit else 0)
    assert (adapter.initial_velocity is None) == (not audit)


def test_asi_source_rejects_extra_dense_forward():
    adapter = CacheSamplerAdapter(None, SimpleNamespace(dense_step0=True), {}, 2, True, step0_source="asi_cfg")
    with pytest.raises(ValueError, match="one dense conditional"):
        adapter(None, [torch.zeros(5)], num_steps=4)


def test_source_verifies_actual_step0_counts():
    controller = SimpleNamespace(dense_step0=False, _dense_stacks=2, _sparse_stacks=0)
    adapter = CacheSamplerAdapter(
        lambda velocity, noise, **kw: velocity(noise, torch.tensor(999)),
        controller,
        {},
        2,
        True,
        step0_source="asi_cfg",
        audit=False,
    )
    with pytest.raises(RuntimeError, match="incorrect dense/sparse"):
        adapter(lambda x, t: x, [torch.ones(5)], num_steps=4)
