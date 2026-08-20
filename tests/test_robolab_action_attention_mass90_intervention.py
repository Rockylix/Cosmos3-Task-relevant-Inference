from __future__ import annotations

import torch

from cosmos_framework.scripts.robolab_action_attention_mass90_intervention import (
    ActionAttentionMass90Controller,
    make_sequence_pack,
    minimum_mass_mask,
    restore_terminal_hidden,
    select_future_mass_union,
)


def _layout() -> dict:
    # Small synthetic analogue of L0 + L1..L8 + q0 + q1..q32.
    latent_positions = {f"L{latent}": [latent * 3 + offset for offset in range(3)] for latent in range(9)}
    action_positions = list(range(27, 60))
    queries = [
        {
            "query_action_index": index,
            "gen_position": position,
            "query_role": "condition" if index == 0 else "predicted",
            "action_horizon": -1 if index == 0 else index - 1,
        }
        for index, position in enumerate(action_positions)
    ]
    return {
        "num_gen_tokens": 60,
        "latent_shape_thw": [9, 1, 3],
        "latent_positions": latent_positions,
        "action_positions": action_positions,
        "action_queries": queries,
    }


def test_minimum_mass_mask_is_minimal_per_query() -> None:
    weights = torch.tensor([[0.60, 0.25, 0.10, 0.05], [0.05, 0.05, 0.10, 0.80]])
    mask = minimum_mass_mask(weights, 0.9)
    assert mask.tolist() == [[True, True, True, False], [False, False, True, True]]
    coverage = (weights * mask).sum(-1) / weights.sum(-1)
    assert torch.all(coverage >= 0.9)


def test_selection_keeps_l0_and_all_actions_and_never_reintroduces_tokens() -> None:
    layout = _layout()
    active = torch.arange(60)
    torch.manual_seed(4)
    q = torch.randn(60, 4, 8)
    k_gen = torch.randn(60, 2, 8)
    k_ar = torch.randn(5, 2, 8)
    local, selected, rows = select_future_mass_union(
        torch=torch,
        q_gen=q,
        k_ar=k_ar,
        k_gen=k_gen,
        scaling=8**-0.5,
        token_layout=layout,
        active_original_positions=active,
        threshold=0.5,
    )
    assert torch.equal(selected, active.index_select(0, local))
    selected_set = set(selected.tolist())
    assert set(layout["latent_positions"]["L0"]).issubset(selected_set)
    assert set(layout["action_positions"]).issubset(selected_set)
    assert all(row["selected_after"] >= 1 for row in rows)
    selected_spatial_by_frame = []
    for latent in range(1, 9):
        frame = layout["latent_positions"][f"L{latent}"]
        selected_spatial_by_frame.append(
            [spatial for spatial, position in enumerate(frame) if position in selected_set]
        )
    assert all(spatial == selected_spatial_by_frame[0] for spatial in selected_spatial_by_frame)
    assert all(row["selected_spatial_positions"] == selected_spatial_by_frame[0] for row in rows)

    q2 = q.index_select(0, local)
    k2 = k_gen.index_select(0, local)
    _, selected2, _ = select_future_mass_union(
        torch=torch,
        q_gen=q2,
        k_ar=k_ar,
        k_gen=k2,
        scaling=8**-0.5,
        token_layout=layout,
        active_original_positions=selected,
        threshold=0.5,
    )
    assert set(selected2.tolist()).issubset(selected_set)


def test_selection_rejects_misaligned_future_frame_regions() -> None:
    layout = _layout()
    # L1 misses spatial index 0 while every other future frame remains full.
    active_list = [position for position in range(60) if position != layout["latent_positions"]["L1"][0]]
    active = torch.tensor(active_list)
    torch.manual_seed(8)
    q = torch.randn(len(active_list), 4, 8)
    k_gen = torch.randn(len(active_list), 2, 8)
    k_ar = torch.randn(5, 2, 8)
    try:
        select_future_mass_union(
            torch=torch,
            q_gen=q,
            k_ar=k_ar,
            k_gen=k_gen,
            scaling=8**-0.5,
            token_layout=layout,
            active_original_positions=active,
            threshold=0.9,
        )
    except RuntimeError as exc:
        assert "same active spatial coordinates" in str(exc)
    else:
        raise AssertionError("Expected mismatched future-frame spatial masks to be rejected")


def test_pack_and_rope_selection_preserve_original_order() -> None:
    und = torch.tensor([[100.0], [101.0]])
    gen = torch.arange(8.0).unsqueeze(-1)
    pack = make_sequence_pack(und_seq=und, gen_seq=gen)
    keep = torch.tensor([0, 3, 7])
    sparse = make_sequence_pack(
        und_seq=pack["causal_seq"],
        gen_seq=pack["full_only_seq"].index_select(0, keep),
    )
    assert sparse["_num_causal_tokens"] == 2
    assert sparse["_num_full_tokens"] == 3
    assert sparse["full_only_seq"].flatten().tolist() == [0.0, 3.0, 7.0]
    # The same keep index applied to position-derived cos/sin therefore retains
    # original position IDs rather than renumbering the sparse sequence.
    rope = torch.arange(80.0).view(8, 10)
    assert rope.index_select(0, keep)[:, 0].tolist() == [0.0, 30.0, 70.0]


def test_terminal_restore_uses_last_valid_hidden_for_dropped_tokens() -> None:
    side = torch.arange(12.0).view(6, 2)
    active_positions = torch.tensor([0, 3, 5])
    active = torch.tensor([[100.0, 101.0], [200.0, 201.0], [300.0, 301.0]])
    restored = restore_terminal_hidden(
        torch=torch,
        side_buffer=side,
        active_hidden=active,
        active_positions=active_positions,
    )
    assert torch.equal(restored.index_select(0, active_positions), active)
    assert torch.equal(restored[1], side[1])
    assert torch.equal(restored[2], side[2])
    assert torch.equal(restored[4], side[4])


def test_cfg_step_and_branch_gating_semantics() -> None:
    controller = object.__new__(ActionAttentionMass90Controller)
    controller.guidance = 3.0
    assert [controller._call_semantics(index) for index in range(8)] == [
        (0, "conditional"),
        (0, "unconditional"),
        (1, "conditional"),
        (1, "unconditional"),
        (2, "conditional"),
        (2, "unconditional"),
        (3, "conditional"),
        (3, "unconditional"),
    ]
    controller.guidance = 1.0
    assert [controller._call_semantics(index) for index in range(4)] == [
        (0, "conditional"),
        (1, "conditional"),
        (2, "conditional"),
        (3, "conditional"),
    ]
