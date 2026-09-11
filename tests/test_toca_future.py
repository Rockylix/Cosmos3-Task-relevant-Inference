from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq, init_sequence_pack
from cosmos_framework.inference import toca_future as toca


def pack(x):
    meta = init_sequence_pack(
        sample_lens=[len(x)], split_lens=[0, len(x)], attn_modes=["causal", "full"], device=x.device
    )
    return {
        **meta,
        "max_num_tokens": len(x),
        "causal_seq": x.new_empty(0, *x.shape[1:]),
        "full_only_seq": x,
        "is_sharded": False,
    }


def rope(q, k, cos, sin, unsqueeze_dim):
    def rotate(x):
        a, b = x.chunk(2, dim=-1)
        return torch.cat((-b, a), dim=-1)

    return (
        q * cos.unsqueeze(unsqueeze_dim) + rotate(q) * sin.unsqueeze(unsqueeze_dim),
        k * cos.unsqueeze(unsqueeze_dim) + rotate(k) * sin.unsqueeze(unsqueeze_dim),
    )


def reference_attention(*, query, key, value, **kwargs):
    assert kwargs["is_causal"] is False
    repeated = query.shape[2] // key.shape[2]
    k, v = key.repeat_interleave(repeated, 2), value.repeat_interleave(repeated, 2)
    p = torch.einsum("bqhd,bkhd->bhqk", query, k).mul(query.shape[-1] ** -0.5).softmax(-1)
    return torch.einsum("bhqk,bkhd->bqhd", p, v)


class ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm_moe_gen = torch.nn.LayerNorm(8)
        self.post_attention_layernorm_moe_gen = torch.nn.LayerNorm(8)
        self.mlp_moe_gen = torch.nn.Linear(8, 8)
        self.self_attn = torch.nn.Module()
        for name, width in (("q", 8), ("k", 4), ("v", 4), ("o", 8)):
            setattr(self.self_attn, f"{name}_proj_moe_gen", torch.nn.Linear(8, width, bias=False))
        self.self_attn.q_norm_moe_gen = torch.nn.Identity()
        self.self_attn.k_norm_moe_gen = torch.nn.Identity()
        self.self_attn.num_attention_heads = 4
        self.self_attn.num_key_value_heads = 2
        self.self_attn.head_dim = 2
        self.self_attn._apply_rotary_pos_emb = rope


def controller(config=None):
    net = torch.nn.Module()
    net.language_model = torch.nn.Module()
    net.language_model.model = torch.nn.Module()
    net.language_model.model.layers = torch.nn.ModuleList([ToyLayer() for _ in range(28)])
    control = toca.ToCaFutureController(net, config)
    control.layout = {
        "num_gen_tokens": 21,
        "latent_positions": {f"L{i}": [2 * i, 2 * i + 1] for i in range(9)},
        "action_positions": [18, 19, 20],
    }
    return control


def test_config_and_counts():
    config = toca.ToCaFutureConfig()
    assert config.fresh_count(0, 28, 2720) == 1020
    assert config.fresh_count(27, 28, 2720) == 340
    for options in ({"full_steps": (1, 2)}, {"full_steps": (0, 0)}, {"fresh_ratio": -1}, {"period": 0}):
        with pytest.raises(ValueError):
            toca.ToCaFutureConfig(**options)


def test_scan_grid_and_task_counts():
    from cosmos_framework.inference.toca_scan_plan import DIFFICULTY, TASKS, configurations

    grid = configurations()
    assert len(grid) == 25 and len(set(TASKS)) == 10
    assert list(DIFFICULTY.values()).count("simple") == 8
    assert list(DIFFICULTY.values()).count("moderate") == 2
    for values in list(grid.values())[1:]:
        config = toca.ToCaFutureConfig(**{**values, "full_steps": tuple(values["full_steps"])})
        for block in range(28):
            assert config.fresh_count(block, 28, 2720) == int(2720 * config.fresh_ratio * (1.5 - block / 27))


def test_spatial_windows_odd_grid_frame_partition():
    windows = toca.spatial_windows(8, 17, 20, "cpu")
    assert windows.shape == (720, 4)
    valid = windows[windows >= 0]
    assert torch.equal(valid.sort().values, torch.arange(2720))
    for window in windows:
        assert len((window[window >= 0] // 340).unique()) == 1


@pytest.mark.parametrize("count", [0, 1, 9, 30])
def test_spatial_bonus_matches_slow_reference_after_age(count):
    config = toca.ToCaFutureConfig(spatial_bonus=0.6, period=4)
    scores = torch.arange(1, 31).float()
    ages = torch.arange(30).remainder(3).float()
    expected = torch.nn.functional.normalize(scores, dim=0) + 0.25 * ages / 4
    for frame in range(2):
        for y in range(0, 3, 2):
            for x in range(0, 5, 2):
                cells = [frame * 15 + yy * 5 + xx for yy in range(y, min(y + 2, 3)) for xx in range(x, min(x + 2, 5))]
                winner = cells[int(expected[cells].argmax())]
                expected[winner] *= 1.6
    selected, age = toca.select_fresh(scores, ages, count, config, toca.spatial_windows(2, 3, 5, "cpu"))
    assert torch.equal(selected, expected.argsort(descending=True, stable=True)[:count])
    assert len(selected.unique()) == count
    assert torch.equal(ages, torch.arange(30).remainder(3).float())
    assert not age[selected].any()


def test_cfg_independent_selections_and_ages_do_not_leak():
    c = controller(toca.ToCaFutureConfig(full_steps=(0,), period=4, cfg_selection="independent"))
    c._init_positions(torch.zeros(21, 8))
    c.scores["conditional", 0] = torch.arange(16).float()
    c.scores["unconditional", 0] = torch.arange(16).flip(0).float()
    for branch in ("conditional", "unconditional"):
        c.ages[branch, 0] = torch.zeros(16)
    c.current = (1, "conditional")
    a = c._select_indices(0)
    cond_age = c.ages["conditional", 0].clone()
    assert not c.ages["unconditional", 0].any()
    c.current = (1, "unconditional")
    b = c._select_indices(0)
    assert not torch.isin(a, b).any()
    assert torch.equal(cond_age, c.ages["conditional", 0])
    c.current = (2, "conditional")
    c._select_indices(0)
    assert len(c.indices) == 3


def test_cfg_shared_selection_updates_age_once():
    c = controller(toca.ToCaFutureConfig(full_steps=(0,), period=4, spatial_bonus=0.6))
    c.layout["latent_shape_thw"] = (9, 1, 2)
    c._init_positions(torch.zeros(21, 8))
    c.scores["conditional", 0] = torch.arange(16).float()
    c.scores["unconditional", 0] = torch.ones(16)
    c.ages[0] = torch.zeros(16)
    c.current = (1, "conditional")
    a = c._select_indices(0)
    old_age = c.ages[0].clone()
    c.current = (1, "unconditional")
    assert torch.equal(a, c._select_indices(0))
    assert torch.equal(c.ages[0], old_age)


def test_score_gqa_full_denominator_and_tiling():
    g = torch.Generator().manual_seed(15)
    q, kg, ku = (
        torch.randn(11, 4, 6, generator=g),
        torch.randn(11, 2, 6, generator=g),
        torch.randn(5, 2, 6, generator=g),
    )
    chosen = torch.tensor([1, 2, 7, 9])
    keys = torch.cat((ku, kg)).repeat_interleave(2, dim=1)
    expected = torch.einsum("qhd,khd->hqk", q, keys).mul(6**-0.5).softmax(-1).mean((0, 1))[5 + chosen]
    for tile in (1, 3, 64):
        actual = toca.incoming_future_score(q, ku, kg, chosen, 6**-0.5, tile)
        torch.testing.assert_close(actual, expected)
    restricted = torch.einsum("qhd,khd->hqk", q, keys[5 + chosen]).mul(6**-0.5).softmax(-1).mean((0, 1))
    assert not torch.allclose(expected, restricted)


def test_selection_age_and_input_not_modified():
    scores, age = torch.ones(6), torch.zeros(6)
    chosen, next_age = toca.select_fresh(scores, age, 2, toca.ToCaFutureConfig())
    assert chosen.tolist() == [0, 1]
    assert age.tolist() == [0] * 6
    assert next_age.tolist() == [0, 0, 1, 1, 1, 1]
    chosen2, _ = toca.select_fresh(scores, next_age, 2, toca.ToCaFutureConfig())
    assert chosen2.tolist() == [2, 3]


def test_layout_protects_l0_and_all_actions():
    c = controller()
    c._init_positions(torch.zeros(21, 8))
    assert c.future.tolist() == list(range(2, 18))
    assert c.protected.tolist() == [0, 1, 18, 19, 20]
    assert sorted(c.future.tolist() + c.protected.tolist()) == list(range(21))


def test_original_rope_positions():
    layer = ToyLayer()
    positions = torch.tensor([0, 3, 8])
    q, k = torch.randn(9, 4, 2), torch.randn(9, 2, 2)
    angles = torch.arange(9).float()[:, None].repeat(1, 2)
    expected_q, expected_k = rope(q, k, angles.cos(), angles.sin(), 1)
    actual_q, actual_k = toca.apply_selected_rope(
        layer.self_attn, q[positions], k, angles.cos(), angles.sin(), positions
    )
    torch.testing.assert_close(actual_q, expected_q[positions])
    torch.testing.assert_close(actual_k, expected_k)


def test_cached_forward_reassembles_full_frame_and_true_row_counts(monkeypatch):
    monkeypatch.setattr(toca, "attention", reference_attention)
    c = controller(replace(toca.ToCaFutureConfig(), fresh_ratio=0, layer_slope=0))
    x = torch.randn(21, 8)
    before = x.clone()
    c._init_positions(x)
    c.current = (1, "conditional")
    old_attn, old_mlp = torch.randn(16, 8), torch.randn(16, 8)
    c.cache["conditional", 0] = {"attn": old_attn.clone(), "mlp": old_mlp.clone()}
    c.cache["unconditional", 0] = {"attn": old_attn.clone() + 10, "mlp": old_mlp.clone() + 20}
    c.scores["conditional", 0] = c.scores["unconditional", 0] = torch.ones(16)
    c.ages[0] = torch.zeros(16)
    memory = SimpleNamespace(und_k_cached=torch.randn(1, 3, 2, 2), und_v_cached=torch.randn(1, 3, 2, 2))
    rows = {}
    handles = []
    for name, module in [
        (name, getattr(c.layers[0].self_attn, f"{name}_proj_moe_gen")) for name in ("q", "k", "v", "o")
    ] + [("mlp", c.layers[0].mlp_moe_gen)]:
        handles.append(
            module.register_forward_pre_hook(lambda m, args, name=name: rows.update({name: args[0].shape[0]}))
        )
    angles = torch.arange(21).float()[:, None].repeat(1, 2)
    output, count = c._cached(0, pack(x), (pack(angles.cos()), pack(angles.sin())), memory, True)
    result = get_gen_seq(output[0])
    assert count == 0 and rows == {"q": 5, "k": 21, "v": 21, "o": 5, "mlp": 5}
    torch.testing.assert_close(result[c.future], x[c.future] + old_attn + old_mlp)
    torch.testing.assert_close(x, before)
    assert result.shape == x.shape and not torch.allclose(result[c.protected], x[c.protected])
    ages = c.ages[0].clone()
    c.current = (1, "unconditional")
    output2, _ = c._cached(0, pack(x), (pack(angles.cos()), pack(angles.sin())), memory, True)
    torch.testing.assert_close(c.ages[0], ages)
    torch.testing.assert_close(get_gen_seq(output2[0])[c.future], x[c.future] + old_attn + old_mlp + 30)
    for handle in handles:
        handle.remove()


def test_context_restores_forward_on_exception():
    c = controller()
    originals = [layer.forward for layer in c.layers]
    with pytest.raises(RuntimeError, match="injected"):
        with c:
            raise RuntimeError("injected")
    assert [layer.forward for layer in c.layers] == originals
    assert not hasattr(c.model, "_toca_future_controller")
    assert not c.net._forward_pre_hooks and not c.net._forward_hooks


def test_partial_mlp_only_updates_selected_future_rows(monkeypatch):
    monkeypatch.setattr(toca, "attention", reference_attention)
    c = controller(replace(toca.ToCaFutureConfig(), fresh_ratio=0.5, layer_slope=0))
    x = torch.randn(21, 8)
    c._init_positions(x)
    c.current = (1, "conditional")
    old_attn, old_mlp = torch.randn(16, 8), torch.randn(16, 8)
    c.cache["conditional", 0] = {"attn": old_attn.clone(), "mlp": old_mlp.clone()}
    c.scores["conditional", 0] = c.scores["unconditional", 0] = torch.arange(16).float()
    c.ages[0] = torch.zeros(16)
    memory = SimpleNamespace(und_k_cached=torch.randn(1, 3, 2, 2), und_v_cached=torch.randn(1, 3, 2, 2))
    angle = torch.arange(21).float()[:, None].repeat(1, 2)
    output, count = c._cached(0, pack(x), (pack(angle.cos()), pack(angle.sin())), memory, True)
    assert count == 8
    selected = c.indices[1, 0]
    assert set(selected.tolist()) == set(range(8, 16))
    cached = c.cache["conditional", 0]["mlp"]
    torch.testing.assert_close(cached[:8], old_mlp[:8])
    z = x[c.future] + old_attn
    expected_live = c.layers[0].mlp_moe_gen(c.layers[0].post_attention_layernorm_moe_gen(z[selected]))
    torch.testing.assert_close(cached[selected], expected_live)
    torch.testing.assert_close(get_gen_seq(output[0])[c.future], z + cached)


def test_forward_step_branch_schedule_and_layout_change(monkeypatch):
    c = controller()
    layout = c.layout
    c.layout = None
    monkeypatch.setattr(toca, "_action_query_layout", lambda *args: layout.copy())
    for index in range(8):
        c._pre(c.net, ({},), {})
        assert c.current == (index // 2, "conditional" if index % 2 == 0 else "unconditional")
        assert (c.current[0] in c.config.full_steps) == (index // 2 in (0, 2))
        c._block_sequence = list(range(28))
        c._post(c.net, (), {}, object())
        assert c.current is None
    with pytest.raises(RuntimeError, match="eight"):
        c._pre(c.net, ({},), {})
    fresh = controller()
    fresh._pre(fresh.net, ({},), {"und_only": True})
    assert fresh.calls == 0 and fresh.current is None
    monkeypatch.setattr(toca, "_action_query_layout", lambda *args: {**layout, "num_gen_tokens": 22})
    with pytest.raises(RuntimeError, match="layout changed"):
        fresh._pre(fresh.net, ({},), {})


def test_rejects_asi_overlap_and_invalid_gating():
    c = controller()
    c.model._robolab_version1_controller = object()
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        c.__enter__()
    with pytest.raises(ValueError, match="four steps"):
        toca.ToCaFutureController(c.net, num_steps=1)


def test_joint_dispatch_restored_on_full_step_failure():
    c = controller(replace(toca.ToCaFutureConfig(), attention_backend="joint"))
    c._init_positions(torch.zeros(21, 8))
    c.current = (0, "conditional")
    original_dispatch = object()
    c.layers[0].self_attn.dispatch_attention_fn = original_dispatch

    def fail(*args, **kwargs):
        assert c.layers[0].self_attn.dispatch_attention_fn is not original_dispatch
        raise RuntimeError("injected joint failure")

    with pytest.raises(RuntimeError, match="injected joint failure"):
        c._full(0, fail, None, None, None, None, False)
    assert c.layers[0].self_attn.dispatch_attention_fn is original_dispatch
    assert c.layers[0].self_attn._attention_stats_capture_callback is None
    assert not c.layers[0].self_attn._forward_hooks
    assert not c.layers[0].mlp_moe_gen._forward_hooks


def test_joint_backend_is_explicit():
    assert toca.ToCaFutureConfig().attention_backend == "reference"
    assert replace(toca.ToCaFutureConfig(), attention_backend="joint").attention_backend == "joint"
    with pytest.raises(ValueError, match="attention_backend"):
        toca.ToCaFutureConfig(attention_backend="sampled_queries")
