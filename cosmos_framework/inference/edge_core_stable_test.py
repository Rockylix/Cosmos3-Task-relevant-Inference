from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.inference.edge_core_stable_layout import _action_query_layout
from cosmos_framework.inference import edge_core_stable as policy


def _records():
    rng = torch.Generator().manual_seed(73)
    return [
        {"block": b, "profiles": torch.rand((8, 340), generator=rng) + 0.01}
        for b in range(28)
    ]


def _layout(spatial=340):
    count = 9 * spatial + 33
    packed = SimpleNamespace(
        attn_modes=["causal", "full"],
        split_lens=[7, count],
        vision=SimpleNamespace(
            sequence_indexes=torch.arange(7, 7 + 9 * spatial),
            token_shapes=[(9, 1, spatial)],
        ),
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
    return torch.einsum("bhqk,bkhd->bqhd", logits.softmax(-1), value), logits.logsumexp(
        -1
    ).permute(0, 2, 1).unsqueeze(-1)


def _reference_profile(q, ka, kg, layout, weights):
    actions = torch.tensor(
        [
            r["gen_position"]
            for r in layout["action_queries"]
            if r["query_role"] == "predicted"
        ]
    )
    keys = torch.cat([ka, kg]).repeat_interleave(q.shape[1] // kg.shape[1], dim=1)
    probs = (
        torch.einsum("qhd,khd->hqk", q[actions].float() * 0.25, keys.float())
        .softmax(-1)
        .mean(0)
    )
    return torch.stack(
        [
            (
                probs[4 * i : 4 * i + 4][
                    :, len(ka) + torch.tensor(layout["latent_positions"][f"L{i + 1}"])
                ]
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
        policy.build_core_stable_plan(
            torch=torch, profile_records=_records(), core_token_budget=64
        )
    with pytest.raises(ValueError, match="CFG 3"):
        policy.Version1Controller(torch=torch, net=None, guidance=1.0, num_steps=4)


def test_layout_includes_condition_frame_and_all_actions():
    layout = _layout()
    plan = policy.build_core_stable_plan(torch=torch, profile_records=_records())
    selected = policy._selected_original_positions(
        torch, layout, plan["execution_mask"], "cpu"
    )
    assert selected.numel() == 340 + 8 * 184 + 33
    assert torch.isin(torch.tensor(layout["latent_positions"]["L0"]), selected).all()
    assert torch.isin(torch.tensor(layout["action_positions"]), selected).all()
    assert (
        len([q for q in layout["action_queries"] if q["query_role"] == "predicted"])
        == 32
    )


def test_lse_failure_propagates_without_alternative_backend(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("test attention kernel error")

    monkeypatch.setattr(policy, "attention", fail)
    layout = _layout(2)
    q = torch.zeros(layout["num_gen_tokens"], 2, 1)
    k = torch.zeros(layout["num_gen_tokens"], 1, 1)
    with pytest.raises(RuntimeError, match="test attention kernel error"):
        policy.action_aligned_future_profiles_with_lse(
            torch=torch,
            q_gen=q,
            k_ar=k[:3],
            k_gen=k,
            v_ar=k[:3],
            v_gen=k,
            scaling=1.0,
            token_layout=layout,
        )


def test_each_stack_updates_l0_and_restores_only_current_input():
    class Layer:
        def __call__(self, pack, *args, **kwargs):
            return (
                policy.from_und_gen_splits(
                    policy.get_und_seq(pack), policy.get_gen_seq(pack) + 1, pack
                ),
                {},
                None,
            )

    layers = [Layer() for _ in range(28)]
    net = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
    )
    controller = policy.Version1Controller(
        torch=torch, net=net, guidance=3.0, num_steps=4
    )
    controller._layout = _layout()
    controller.plan = policy.build_core_stable_plan(
        torch=torch, profile_records=_records()
    )
    selected = policy._selected_original_positions(
        torch, controller._layout, controller.plan["execution_mask"], "cpu"
    )
    for step in range(1, 4):
        for branch in ["conditional", "unconditional"]:
            value = 100 * step + (50 if branch == "unconditional" else 0)
            full = torch.full((3093, 2), float(value))
            pack = policy._make_sequence_pack(und_seq=torch.zeros(7, 2), gen_seq=full)
            controller._current = {"step": step, "branch": branch}
            controller.begin_stack(
                hidden_states=pack,
                position_embeddings=(pack, pack),
                natten_metadata_list=None,
            )
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
                assert torch.equal(
                    policy.get_gen_seq(result)[:340],
                    torch.full((340, 2), float(value + block + 1)),
                )
            result = controller.end_stack(result)
            expected = full.clone()
            expected[selected] += 28
            assert torch.equal(policy.get_gen_seq(result), expected)
            assert controller._side_buffer is None


def test_profile_bypasses_compiled_wrapper_and_cleans_callback_on_error():
    attention = SimpleNamespace(_attention_stats_capture_callback=None)
    calls = []

    class EagerLayer:
        self_attn = attention

        def __call__(self, *args, **kwargs):
            calls.append("eager")
            assert attention._attention_stats_capture_callback is not None
            raise RuntimeError("profile failure")

    class CompiledLayer:
        self_attn = attention
        _orig_mod = EagerLayer()

        def __call__(self, *args, **kwargs):
            raise AssertionError("profiling entered compiled graph")

    layer = CompiledLayer()
    net = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=[layer] * 28))
    )
    controller = policy.Version1Controller(
        torch=torch, net=net, guidance=3, num_steps=4
    )
    controller._position_embeddings = ({}, {})
    with pytest.raises(RuntimeError, match="profile failure"):
        controller._run_dense_profile_layer(
            block=0,
            decoder_layer=layer,
            hidden_states={},
            attention_mask=None,
            memory_value=None,
            gen_only=False,
        )
    assert calls == ["eager"]
    assert attention._attention_stats_capture_callback is None
    assert controller._profile_callback_block is None


def test_sparse_pack_preserves_logical_text_length_with_graph_padding():
    net = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=[None] * 28))
    )
    controller = policy.Version1Controller(
        torch=torch, net=net, guidance=3, num_steps=4
    )
    # Five real text tokens in an eight-row allocation. Padded rows must never
    # be counted as text attention keys after sparse repacking.
    pack = policy._make_sequence_pack(
        und_seq=torch.zeros(8, 2), gen_seq=torch.zeros(20, 2), und_tokens=5
    )
    controller._position_embeddings = (pack, pack)
    sparse, rope = controller._slice_pack_and_rope(pack, torch.tensor([0, 2, 7]))
    for item in [sparse, *rope]:
        assert item["_num_causal_tokens"] == 5
        assert item["_num_full_tokens"] == 3
        assert policy.get_und_seq(item).shape == (8, 2)


@pytest.mark.parametrize("spatial,heads,kv_heads", [(3, 4, 2), (360, 8, 2)])
def test_batched_profile_matches_independent_reference(
    monkeypatch, spatial, heads, kv_heads
):
    from cosmos_framework.inference import edge_core_stable_fast as fast

    monkeypatch.setattr(fast, "attention", _reference_attention)
    layout = _layout(spatial)
    rng = torch.Generator().manual_seed(192)
    q = torch.randn(layout["num_gen_tokens"], heads, 8, generator=rng)
    ka = torch.randn(5, kv_heads, 8, generator=rng)
    kg = torch.randn(layout["num_gen_tokens"], kv_heads, 8, generator=rng)
    actual = fast.action_aligned_future_profiles(
        q,
        ka,
        kg,
        torch.zeros_like(ka),
        torch.zeros_like(kg),
        0.25,
        fast.profile_geometry(layout),
    )
    expected = _reference_profile(q, ka, kg, layout, [1 / 6, 1 / 3, 1 / 3, 1 / 6])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_batched_profile_rejects_noncontiguous_layout():
    from cosmos_framework.inference.edge_core_stable_fast import profile_geometry

    layout = _layout(3)
    layout["latent_positions"]["L2"].reverse()
    with pytest.raises(ValueError, match="contiguous future"):
        profile_geometry(layout)


def test_deferred_profile_validation_rejects_nan():
    records = _records()
    records[12]["profiles"][2, 4] = float("nan")
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        policy.build_core_stable_plan(torch=torch, profile_records=records)
