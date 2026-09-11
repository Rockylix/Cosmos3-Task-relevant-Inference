from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.inference.task_core_layers import TaskCoreLayerCache
from cosmos_framework.scripts import robolab_version1 as policy


def records(seed=73):
    rng = torch.Generator().manual_seed(seed)
    return [{"block": b, "profiles": torch.rand(8, 340, generator=rng) + 0.01} for b in range(28)]


def test_top7_selected_once_but_weights_and_masks_are_current(monkeypatch):
    selections = []
    original = torch.topk

    def track(value, k, *args, **kwargs):
        if tuple(value.shape) == (28,) and k == 7:
            selections.append(True)
        return original(value, k, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", track)
    cache = TaskCoreLayerCache()
    cache.begin_task("banana")
    first = policy.build_core_stable_plan(torch=torch, profile_records=records(), core_block_count=7)
    cache.commit(first["core_blocks"])
    second = policy.build_core_stable_plan(
        torch=torch, profile_records=records(91), core_block_count=7, fixed_core_blocks=cache.blocks
    )
    cache.commit(second["core_blocks"])
    assert len(selections) == 1
    assert first["core_blocks"] == second["core_blocks"]
    assert not torch.equal(first["core_weights"], second["core_weights"])
    assert not torch.equal(first["execution_mask"], second["execution_mask"])
    assert second["execution_mask"].sum(1).tolist() == [184] * 8
    assert second["core_masks"].sum(1).tolist() == [80] * 8
    assert second["stable_mask"].sum() == 104
    assert not (second["stable_mask"] & second["core_masks"].any(0)).any()
    assert cache.completed_chunks == 2


def test_stable_still_uses_unselected_layers_and_preserves_original_scores():
    recs = records()
    old = policy.build_core_stable_plan(torch=torch, profile_records=recs)
    new = policy.build_core_stable_plan(torch=torch, profile_records=recs, core_block_count=7)
    for key in (
        "block_quality",
        "block_mass",
        "block_entropy",
        "stable_mean",
        "stable_std",
        "stable_cv",
        "stable_scores",
    ):
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    extra = next(b for b in range(28) if b not in new["core_blocks"])
    position = (~new["core_masks"].any(0)).nonzero().flatten()[0]
    recs[extra]["profiles"] = torch.zeros(8, 340)
    recs[extra]["profiles"][:, position] = 0.9
    updated = policy.build_core_stable_plan(
        torch=torch, profile_records=recs, core_block_count=7, fixed_core_blocks=new["core_blocks"]
    )
    torch.testing.assert_close(updated["core_scores"], new["core_scores"], rtol=0, atol=0)
    assert not torch.equal(updated["stable_scores"], new["stable_scores"])
    assert updated["stable_mask"][position]


def test_task_switch_and_explicit_reset_do_not_reuse_old_ids():
    cache = TaskCoreLayerCache()
    cache.begin_task("A")
    cache.commit(range(7))
    cache.begin_task("A")
    assert cache.blocks == tuple(range(7)) and cache.completed_chunks == 1
    cache.begin_task("B")
    assert cache.blocks is None and cache.completed_chunks == 0
    cache.commit(range(7, 14))
    cache.begin_task("A")
    assert cache.blocks is None and cache.completed_chunks == 0
    cache.commit(range(14, 21))
    cache.begin_task("A", reset=True)
    assert cache.blocks is None and cache.completed_chunks == 0


def test_failed_generation_does_not_commit_layer_cache():
    cache = TaskCoreLayerCache()
    cache.begin_task("A")
    policy.build_core_stable_plan(torch=torch, profile_records=records(), core_block_count=7)
    # Generation failed after building a temporary plan: commit must not have happened.
    assert cache.blocks is None and cache.completed_chunks == 0
    cache.commit(range(7))
    with pytest.raises(RuntimeError, match="changed inside"):
        cache.commit(range(1, 8))
    assert cache.blocks == tuple(range(7)) and cache.completed_chunks == 1


@pytest.mark.parametrize("ids", [range(6), [0] * 7, [0, 1, 2, 3, 4, 5, 28]])
def test_invalid_fixed_layers_rejected(ids):
    with pytest.raises(ValueError, match="unique valid"):
        policy.build_core_stable_plan(torch=torch, profile_records=records(), core_block_count=7, fixed_core_blocks=ids)


def test_all_28_profiles_are_required_even_when_core_ids_are_fixed():
    with pytest.raises(RuntimeError, match="every block"):
        policy.build_core_stable_plan(
            torch=torch, profile_records=records()[:7], core_block_count=7, fixed_core_blocks=range(7)
        )


def test_controller_passes_fixed_ids_and_reports_seven_layers():
    layers = [SimpleNamespace() for _ in range(28)]
    net = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))
    selected = (21, 15, 19, 20, 23, 18, 22)
    controller = policy.Version1Controller(
        torch=torch, net=net, guidance=3, num_steps=4, core_block_count=7, fixed_core_blocks=selected
    )
    controller._profile_records = records()
    controller._original_pack = {}
    controller.stack_active = True
    controller._current = {"step": 0, "branch": "conditional"}
    assert controller.end_stack({"stub": True}) == {"stub": True}
    assert controller.plan["core_blocks"] == list(selected)
    controller._completed_stacks, controller._dense_stacks, controller._sparse_stacks = 8, 1, 7
    controller._sparse_block_calls = 196
    summary = controller.finish()
    assert summary["core_block_count"] == 7 and summary["core_layers_reused"]
    assert summary["profiled_block_count"] == 28
    assert summary["stable_profile_blocks"] == list(range(28))
