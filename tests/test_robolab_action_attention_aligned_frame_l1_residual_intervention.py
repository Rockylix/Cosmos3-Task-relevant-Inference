from __future__ import annotations

import torch

from cosmos_framework.scripts.robolab_action_attention_aligned_frame_l1_residual_intervention import (
    restore_with_per_frame_l1_residual,
    select_l2_l8_aligned_frame_mass,
)


def _layout(spatial: int = 4) -> dict:
    latent_positions = {
        f"L{latent}": list(range(latent * spatial, (latent + 1) * spatial)) for latent in range(9)
    }
    action_positions = list(range(9 * spatial, 9 * spatial + 33))
    return {
        "num_gen_tokens": 9 * spatial + 33,
        "latent_positions": latent_positions,
        "action_positions": action_positions,
        "action_queries": [
            {
                "query_action_index": index,
                "gen_position": position,
                "query_role": "condition" if index == 0 else "predicted",
                "action_horizon": -1 if index == 0 else index - 1,
            }
            for index, position in enumerate(action_positions)
        ],
    }


def test_aligned_selector_uses_independent_frame_masks_and_correct_horizons() -> None:
    layout = _layout(spatial=3)
    q = torch.zeros(layout["num_gen_tokens"], 2, 2)
    k_gen = torch.zeros(layout["num_gen_tokens"], 1, 2)
    k_ar = torch.zeros(1, 1, 2)
    for query in layout["action_queries"]:
        if query["query_role"] == "predicted":
            horizon = query["action_horizon"]
            q[query["gen_position"], :, horizon % 2] = 1.0
    for latent in range(2, 9):
        positions = layout["latent_positions"][f"L{latent}"]
        preferred = latent % 2
        k_gen[positions[preferred], :, (4 * (latent - 1)) % 2] = 8.0
    _, selected, rows, masks, alignment = select_l2_l8_aligned_frame_mass(
        torch=torch,
        q_gen=q,
        k_ar=k_ar,
        k_gen=k_gen,
        scaling=1.0,
        token_layout=layout,
        threshold=0.9,
    )
    selected_set = set(selected.tolist())
    assert set(layout["latent_positions"]["L0"]) <= selected_set
    assert set(layout["latent_positions"]["L1"]) <= selected_set
    assert set(layout["action_positions"]) <= selected_set
    assert masks.shape == (7, 3)
    assert [row["action_horizon_start"] for row in alignment] == [4, 8, 12, 16, 20, 24, 28]
    assert [row["action_horizon_end"] for row in alignment] == [7, 11, 15, 19, 23, 27, 31]
    for row in rows:
        assert row["selected_after"] >= 1
    assert all(row["aggregate_coverage"] >= 0.9 for row in alignment)


def test_per_frame_masks_restore_a_complete_frame_with_l1_residual() -> None:
    layout = _layout()
    masks = torch.tensor(
        [
            [True, False, True, False],
            [True, True, False, False],
            [False, True, True, False],
            [False, False, True, True],
            [True, False, False, True],
            [False, True, False, True],
            [True, True, True, False],
        ]
    )
    mandatory = layout["latent_positions"]["L0"] + layout["latent_positions"]["L1"] + layout["action_positions"]
    future = [
        position
        for mask_index, latent in enumerate(range(2, 9))
        for spatial, position in enumerate(layout["latent_positions"][f"L{latent}"])
        if masks[mask_index, spatial]
    ]
    selected = torch.tensor(sorted(mandatory + future))
    original = torch.arange(layout["num_gen_tokens"] * 2, dtype=torch.float32).reshape(-1, 2)
    sparse = original.index_select(0, selected) + 1.0
    l1_positions = torch.tensor(layout["latent_positions"]["L1"])
    l1_offsets = torch.searchsorted(selected, l1_positions)
    sparse[l1_offsets] += torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    full, targets, donor_spatial, l1_residual = restore_with_per_frame_l1_residual(
        torch=torch,
        original_gen=original,
        sparse_output_gen=sparse,
        selected_positions=selected,
        token_layout=layout,
        frame_masks=masks,
    )
    assert full.shape == original.shape
    assert torch.equal(full.index_select(0, selected), sparse)
    assert torch.equal(
        full.index_select(0, targets),
        original.index_select(0, targets) + l1_residual.index_select(0, donor_spatial),
    )
    assert int(targets.numel()) + sum(int(mask.sum()) for mask in masks) == 7 * 4


def test_bad_action_horizon_metadata_is_rejected() -> None:
    layout = _layout()
    layout["action_queries"][-1]["action_horizon"] = 30
    q = torch.randn(layout["num_gen_tokens"], 4, 8)
    k_gen = torch.randn(layout["num_gen_tokens"], 2, 8)
    k_ar = torch.randn(3, 2, 8)
    try:
        select_l2_l8_aligned_frame_mass(
            torch=torch,
            q_gen=q,
            k_ar=k_ar,
            k_gen=k_gen,
            scaling=8**-0.5,
            token_layout=layout,
            threshold=0.9,
        )
    except RuntimeError as error:
        assert "0..31" in str(error)
    else:
        raise AssertionError("Invalid horizon metadata was accepted")
