from __future__ import annotations

import torch

from cosmos_framework.scripts.robolab_action_attention_l1_residual_intervention import (
    restore_with_l1_residual,
    select_l2_l8_shared_mass_union,
)


def _layout(spatial: int = 4) -> dict:
    latent_positions = {
        f"L{latent}": list(range(latent * spatial, (latent + 1) * spatial)) for latent in range(9)
    }
    action_positions = list(range(9 * spatial, 9 * spatial + 33))
    return {
        "num_gen_tokens": 9 * spatial + 33,
        "latent_shape_thw": [9, 1, spatial],
        "latent_positions": latent_positions,
        "action_positions": action_positions,
        "action_queries": [
            {
                "query_action_index": index,
                "gen_position": position,
                "query_role": "condition" if index == 0 else "predicted",
            }
            for index, position in enumerate(action_positions)
        ],
    }


def test_current_probe_selects_variable_shared_l2_l8_mask_and_keeps_l1() -> None:
    layout = _layout()
    torch.manual_seed(7)
    q = torch.randn(layout["num_gen_tokens"], 4, 8)
    k_gen = torch.randn(layout["num_gen_tokens"], 2, 8)
    k_ar = torch.randn(5, 2, 8)
    _, selected, rows, shared = select_l2_l8_shared_mass_union(
        torch=torch,
        q_gen=q,
        k_ar=k_ar,
        k_gen=k_gen,
        scaling=8**-0.5,
        token_layout=layout,
        threshold=0.5,
    )
    selected_set = set(selected.tolist())
    assert set(layout["latent_positions"]["L0"]) <= selected_set
    assert set(layout["latent_positions"]["L1"]) <= selected_set
    assert set(layout["action_positions"]) <= selected_set
    expected_spatial = torch.nonzero(shared, as_tuple=False).flatten().tolist()
    assert 0 < len(expected_spatial) <= 4
    for latent in range(2, 9):
        kept = [
            spatial
            for spatial, position in enumerate(layout["latent_positions"][f"L{latent}"])
            if position in selected_set
        ]
        assert kept == expected_spatial
    assert all(row["selected_spatial_positions"] == expected_spatial for row in rows)
    assert all(row["coverage_min"] >= 0.5 for row in rows)


def test_l1_residual_restores_every_omitted_l2_l8_position() -> None:
    layout = _layout()
    shared = torch.tensor([True, False, True, False])
    mandatory = (
        layout["latent_positions"]["L0"]
        + layout["latent_positions"]["L1"]
        + layout["action_positions"]
    )
    selected_l2_l8 = [
        position
        for latent in range(2, 9)
        for spatial, position in enumerate(layout["latent_positions"][f"L{latent}"])
        if shared[spatial]
    ]
    selected = torch.tensor(sorted(mandatory + selected_l2_l8))
    original = torch.arange(layout["num_gen_tokens"] * 2, dtype=torch.float32).reshape(-1, 2)
    sparse = original.index_select(0, selected).clone()
    sparse.add_(1.0)
    l1 = torch.tensor(layout["latent_positions"]["L1"])
    l1_offsets = torch.searchsorted(selected, l1)
    extra = torch.tensor([[10.0], [20.0], [30.0], [40.0]]).repeat(1, 2)
    sparse.index_add_(0, l1_offsets, extra)
    full, targets, l1_residual = restore_with_l1_residual(
        torch=torch,
        original_gen=original,
        sparse_output_gen=sparse,
        selected_positions=selected,
        token_layout=layout,
        shared_mask=shared,
    )
    assert full.shape == original.shape
    assert torch.equal(full.index_select(0, selected), sparse)
    assert torch.equal(l1_residual, torch.tensor([[11.0], [21.0], [31.0], [41.0]]).repeat(1, 2))
    omitted = torch.tensor([1, 3]).repeat(7)
    assert torch.equal(
        full.index_select(0, targets),
        original.index_select(0, targets) + l1_residual.index_select(0, omitted),
    )
    for latent in range(9):
        frame = full.index_select(0, torch.tensor(layout["latent_positions"][f"L{latent}"]))
        assert frame.shape == (4, 2)


def test_full_l2_l8_mask_needs_no_residual_recovery() -> None:
    layout = _layout()
    shared = torch.ones(4, dtype=torch.bool)
    selected = torch.arange(layout["num_gen_tokens"])
    original = torch.randn(layout["num_gen_tokens"], 3)
    sparse = original + 2
    full, targets, _ = restore_with_l1_residual(
        torch=torch,
        original_gen=original,
        sparse_output_gen=sparse,
        selected_positions=selected,
        token_layout=layout,
        shared_mask=shared,
    )
    assert targets.numel() == 0
    assert torch.equal(full, sparse)
