from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.inference.edge_core_stable_layout import _action_query_layout
from cosmos_framework.scripts import robolab_version1 as policy


def _records():
    rng = torch.Generator().manual_seed(73)
    return [{"block": b, "profiles": torch.rand((8, 340), generator=rng) + 0.01} for b in range(28)]


def _layout(spatial=340):
    count = 9 * spatial + 33
    packed = SimpleNamespace(
        attn_modes=["causal", "full"],
        split_lens=[7, count],
        vision=SimpleNamespace(sequence_indexes=torch.arange(7, 7 + 9 * spatial), token_shapes=[(9, 1, spatial)]),
        action=SimpleNamespace(
            sequence_indexes=torch.arange(7 + 9 * spatial, 7 + count),
            token_shapes=[(33,)],
            condition_mask=[torch.tensor([True] + [False] * 32)],
        ),
    )
    return _action_query_layout(torch, packed)


def _reference_attention(*, query, key, value, scale, return_lse, **kwargs):
    assert return_lse
    repeat = query.shape[2] // key.shape[2]
    key = key.repeat_interleave(repeat, dim=2)
    value = value.repeat_interleave(repeat, dim=2)
    logits = torch.einsum("bqhd,bkhd->bhqk", query, key) * scale
    return torch.einsum("bhqk,bkhd->bqhd", logits.softmax(-1), value), logits.logsumexp(-1).permute(0, 2, 1).unsqueeze(
        -1
    )


def _reference_profile(q, ka, kg, layout, weights):
    actions = torch.tensor([r["gen_position"] for r in layout["action_queries"] if r["query_role"] == "predicted"])
    keys = torch.cat([ka, kg]).repeat_interleave(q.shape[1] // kg.shape[1], dim=1)
    probs = torch.einsum("qhd,khd->hqk", q[actions].float() * 0.25, keys.float()).softmax(-1).mean(0)
    return torch.stack(
        [
            (
                probs[4 * i : 4 * i + 4][:, len(ka) + torch.tensor(layout["latent_positions"][f"L{i + 1}"])]
                * torch.tensor(weights)[:, None]
            ).sum(0)
            for i in range(8)
        ]
    )


def test_fixed_budget_and_disjoint_masks():
    plan = policy.build_core_stable_plan(torch=torch, profile_records=_records())
    assert plan["core_masks"].sum(1).tolist() == [80] * 8
    assert plan["stable_mask"].sum().item() == 104
    assert not (plan["core_masks"] & plan["stable_mask"]).any()
    assert plan["execution_mask"].sum(1).tolist() == [184] * 8
    assert policy.STRATEGY_VERSION == "edge-core80-stable104-action-weighted"


def test_action_weighted_lse_matches_independent_softmax(monkeypatch):
    monkeypatch.setattr(policy, "attention", _reference_attention)
    layout = _layout(3)
    rng = torch.Generator().manual_seed(7)
    q = torch.randn(layout["num_gen_tokens"], 4, 8, generator=rng)
    ka = torch.randn(5, 2, 8, generator=rng)
    kg = torch.randn(layout["num_gen_tokens"], 2, 8, generator=rng)
    actual = policy.action_aligned_future_profiles_with_lse(
        torch=torch,
        q_gen=q,
        k_ar=ka,
        k_gen=kg,
        v_ar=torch.zeros_like(ka),
        v_gen=torch.zeros_like(kg),
        scaling=0.25,
        token_layout=layout,
    )
    expected = _reference_profile(q, ka, kg, layout, [1 / 6, 1 / 3, 1 / 3, 1 / 6])
    uniform = _reference_profile(q, ka, kg, layout, [0.25] * 4)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(actual, uniform)


def test_fixed_policy_rejects_ablation_arguments():
    with pytest.raises(TypeError):
        policy.build_core_stable_plan(torch=torch, profile_records=_records(), core_token_budget=64)
    with pytest.raises(ValueError, match="CFG 3"):
        policy.Version1Controller(torch=torch, net=None, guidance=1.0, num_steps=4)


def test_layout_includes_condition_frame_and_all_actions():
    layout = _layout()
    plan = policy.build_core_stable_plan(torch=torch, profile_records=_records())
    selected = policy._selected_original_positions(torch, layout, plan["execution_mask"], "cpu")
    assert selected.numel() == 340 + 8 * 184 + 33
    assert torch.isin(torch.tensor(layout["latent_positions"]["L0"]), selected).all()
    assert torch.isin(torch.tensor(layout["action_positions"]), selected).all()
    assert len([q for q in layout["action_queries"] if q["query_role"] == "predicted"]) == 32


def test_lse_failure_propagates_without_alternative_backend(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("test attention kernel error")

    monkeypatch.setattr(policy, "attention", fail)
    layout = _layout(2)
    q = torch.zeros(layout["num_gen_tokens"], 2, 1)
    k = torch.zeros(layout["num_gen_tokens"], 1, 1)
    with pytest.raises(RuntimeError, match="test attention kernel error"):
        policy.action_aligned_future_profiles_with_lse(
            torch=torch, q_gen=q, k_ar=k[:3], k_gen=k, v_ar=k[:3], v_gen=k, scaling=1.0, token_layout=layout
        )


def test_each_stack_updates_l0_and_restores_only_current_input():
    class Layer:
        def __call__(self, pack, *args, **kwargs):
            return policy.from_und_gen_splits(policy.get_und_seq(pack), policy.get_gen_seq(pack) + 1, pack), {}, None

    layers = [Layer() for _ in range(28)]
    net = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))
    controller = policy.Version1Controller(torch=torch, net=net, guidance=3.0, num_steps=4)
    controller._layout = _layout()
    controller.plan = policy.build_core_stable_plan(torch=torch, profile_records=_records())
    selected = policy._selected_original_positions(torch, controller._layout, controller.plan["execution_mask"], "cpu")
    for step in range(1, 4):
        for branch in ["conditional", "unconditional"]:
            value = 100 * step + (50 if branch == "unconditional" else 0)
            full = torch.full((3093, 2), float(value))
            pack = policy._make_sequence_pack(und_seq=torch.zeros(7, 2), gen_seq=full)
            controller._current = {"step": step, "branch": branch}
            controller.begin_stack(hidden_states=pack, position_embeddings=(pack, pack), natten_metadata_list=None)
            result = pack
            for block, layer in enumerate(layers):
                result, _, _ = controller.run_layer(
                    block=block,
                    decoder_layer=layer,
                    hidden_states=result,
                    attention_mask=None,
                    memory_value=None,
                    gen_only=False,
                )
                assert torch.equal(policy.get_gen_seq(result)[:340], torch.full((340, 2), float(value + block + 1)))
            result = controller.end_stack(result)
            expected = full.clone()
            expected[selected] += 28
            assert torch.equal(policy.get_gen_seq(result), expected)
            assert controller._side_buffer is None


def test_opt_in_dense_step0_unconditional_does_not_profile_again():
    calls = []

    class Layer:
        def __call__(self, pack, *args, **kwargs):
            calls.append(len(policy.get_gen_seq(pack)))
            return policy.from_und_gen_splits(policy.get_und_seq(pack), policy.get_gen_seq(pack) + 1, pack), {}, None

    layers = [Layer() for _ in range(28)]
    net = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))
    controller = policy.Version1Controller(torch=torch, net=net, guidance=3.0, num_steps=4, dense_step0=True)
    controller._layout = _layout()
    controller._profile_records = _records()
    controller.plan = policy.build_core_stable_plan(torch=torch, profile_records=controller._profile_records)
    original_plan = controller.plan
    # Conditional step0/profile already completed; now exercise the 7 remaining stacks.
    controller._completed_stacks = controller._dense_stacks = 1
    for step, branch in [(0, "unconditional")] + [
        (s, b) for s in range(1, 4) for b in ("conditional", "unconditional")
    ]:
        controller._current = {"step": step, "branch": branch}
        pack = policy._make_sequence_pack(und_seq=torch.zeros(7, 2), gen_seq=torch.zeros(3093, 2))
        controller.begin_stack(hidden_states=pack, position_embeddings=(pack, pack), natten_metadata_list=None)
        if step == 0:
            assert controller._side_buffer is None
        result = pack
        for block, layer in enumerate(layers):
            result, _, _ = controller.run_layer(
                block=block,
                decoder_layer=layer,
                hidden_states=result,
                attention_mask=None,
                memory_value=None,
                gen_only=False,
            )
        result = controller.end_stack(result)
        if step == 0:
            assert torch.equal(policy.get_gen_seq(result), torch.full((3093, 2), 28.0))
    assert controller.plan is original_plan and len(controller._profile_records) == 28
    assert calls == [3093] * 28 + [1845] * (6 * 28)
    summary = controller.finish()
    assert (summary["dense_stack_count"], summary["sparse_stack_count"]) == (2, 6)
