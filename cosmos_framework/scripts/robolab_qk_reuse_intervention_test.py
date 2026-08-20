# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.scripts.robolab_qk_reuse_intervention import (
    PostRopeQKReuseController,
    SparseProjectionQKReuseController,
    SparseProjectionTrace,
    prepare_qk_reuse_indices,
    project_gen_qk_with_reuse,
    replace_post_rope_qk,
    validate_reuse_plan,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _layout() -> dict[str, object]:
    positions = {f"L{latent}": list(range(latent * 2, latent * 2 + 2)) for latent in range(9)}
    return {
        "num_gen_tokens": 20,
        "latent_shape_thw": [9, 1, 2],
        "latent_positions": positions,
        "action_positions": [18, 19],
    }


def test_replace_post_rope_qk_preserves_non_targets_and_copies_exactly() -> None:
    q = torch.arange(20 * 4 * 3, dtype=torch.float32).reshape(20, 4, 3)
    k = torch.arange(20 * 2 * 3, dtype=torch.float32).reshape(20, 2, 3) + 1000
    q_before = q.clone()
    k_before = k.clone()
    rows = replace_post_rope_qk(
        torch=torch,
        q_rope=q,
        k_rope=k,
        token_layout=_layout(),
        pairs=((1, 2), (3, 4)),
    )
    assert torch.equal(q[4:6], q_before[2:4])
    assert torch.equal(k[4:6], k_before[2:4])
    assert torch.equal(q[8:10], q_before[6:8])
    assert torch.equal(k[8:10], k_before[6:8])
    assert torch.equal(q[0:4], q_before[0:4])
    assert torch.equal(k[18:20], k_before[18:20])
    assert all(row["q_post_copy_max_abs"] == 0.0 for row in rows)
    assert all(row["k_post_copy_max_abs"] == 0.0 for row in rows)


def test_validate_reuse_plan_rejects_overlap_and_invalid_bounds() -> None:
    validate_reuse_plan({(0, 1): ((1, 2), (3, 4))}, num_steps=4, num_blocks=28)
    with pytest.raises(ValueError, match="Overlapping"):
        validate_reuse_plan({(0, 1): ((1, 2), (2, 3))}, num_steps=4, num_blocks=28)
    with pytest.raises(ValueError, match="Invalid denoise step"):
        validate_reuse_plan({(4, 1): ((1, 2),)}, num_steps=4, num_blocks=28)


def test_sparse_projection_skips_targets_and_restores_post_rope_source() -> None:
    torch.manual_seed(7)
    hidden = torch.randn(20, 6)
    cos = torch.randn(20, 3)
    sin = torch.randn(20, 3)
    q_proj = torch.nn.Linear(6, 12, bias=False)
    k_proj = torch.nn.Linear(6, 6, bias=False)

    def rotary(q, k, selected_cos, selected_sin, unsqueeze_dim):
        assert unsqueeze_dim == 1
        offset = selected_cos.unsqueeze(1) + selected_sin.unsqueeze(1)
        return q + offset, k + offset

    dense, dense_stats = project_gen_qk_with_reuse(
        torch=torch,
        hidden_states=hidden,
        cos=cos,
        sin=sin,
        q_proj=q_proj,
        k_proj=k_proj,
        q_norm=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
        apply_rotary_pos_emb=rotary,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=3,
        token_layout=_layout(),
        pairs=((1, 2), (3, 4)),
        dense_control=True,
    )
    sparse, sparse_stats = project_gen_qk_with_reuse(
        torch=torch,
        hidden_states=hidden,
        cos=cos,
        sin=sin,
        q_proj=q_proj,
        k_proj=k_proj,
        q_norm=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
        apply_rotary_pos_emb=rotary,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=3,
        token_layout=_layout(),
        pairs=((1, 2), (3, 4)),
    )
    assert dense_stats["computed_tokens"] == 20
    assert sparse_stats == {
        "total_tokens": 20,
        "computed_tokens": 16,
        "skipped_tokens": 4,
        "pair_count": 2,
    }
    for dense_tensor, sparse_tensor in zip(dense, sparse, strict=True):
        assert torch.equal(sparse_tensor[4:6], dense_tensor[2:4])
        assert torch.equal(sparse_tensor[8:10], dense_tensor[6:8])
        # Different GEMM M dimensions may select different kernels, so kept
        # rows are mathematically equal but need not be bitwise identical.
        assert torch.allclose(sparse_tensor[18:20], dense_tensor[18:20], atol=1e-6, rtol=1e-6)


def test_optimized_sparse_projection_reuses_indices_and_only_restores_post_rope() -> None:
    hidden = torch.randn(20, 6)
    cos = torch.zeros(20, 3)
    sin = torch.zeros(20, 3)
    prepared = prepare_qk_reuse_indices(
        torch=torch,
        token_layout=_layout(),
        pairs=((1, 2), (3, 4)),
        device=hidden.device,
    )

    def rotary(q, k, selected_cos, selected_sin, unsqueeze_dim):
        del selected_cos, selected_sin, unsqueeze_dim
        return q, k

    (q_raw, k_raw, q_rope, k_rope), stats = project_gen_qk_with_reuse(
        torch=torch,
        hidden_states=hidden,
        cos=cos,
        sin=sin,
        q_proj=torch.nn.Linear(6, 12, bias=False),
        k_proj=torch.nn.Linear(6, 6, bias=False),
        q_norm=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
        apply_rotary_pos_emb=rotary,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=3,
        token_layout=_layout(),
        pairs=((1, 2), (3, 4)),
        prepared_indices=prepared,
        restore_raw_qk=False,
    )
    assert q_raw is None and k_raw is None
    assert q_rope.shape == (20, 4, 3)
    assert k_rope.shape == (20, 2, 3)
    assert stats["computed_tokens"] == 16
    assert torch.equal(q_rope[4:6], q_rope[2:4])
    assert torch.equal(k_rope[8:10], k_rope[6:8])


class _HookHandle:
    def remove(self) -> None:
        pass


class _Net:
    def __init__(self) -> None:
        layers = []
        for _ in range(28):
            layers.append(
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        _rope_qk_capture_callback=None,
                        _sparse_qk_projection_callback=None,
                    )
                )
            )
        self.language_model = SimpleNamespace(model=SimpleNamespace(layers=layers))

    def register_forward_pre_hook(self, hook, with_kwargs):
        del hook, with_kwargs
        return _HookHandle()

    def register_forward_hook(self, hook, with_kwargs):
        del hook, with_kwargs
        return _HookHandle()


def test_controller_is_request_scoped_and_step_branch_gated() -> None:
    net = _Net()
    controller = PostRopeQKReuseController(
        torch=torch,
        net=net,
        guidance=3.0,
        num_steps=4,
        plan={(2, 1): ((1, 2),)},
    )
    q = torch.randn(20, 4, 3)
    k = torch.randn(20, 2, 3)
    q_original = q.clone()
    controller._token_layout = _layout()
    with controller:
        assert net.language_model.model.layers[1].self_attn._rope_qk_capture_callback is not None
        controller._current = {"step": 1, "branch": "conditional"}
        controller._intervene(layer_index=1, q_raw=q, k_raw=k, q_rope=q, k_rope=k, cos=None, sin=None)
        assert torch.equal(q, q_original)
        controller._current = {"step": 2, "branch": "conditional"}
        controller._intervene(layer_index=1, q_raw=q, k_raw=k, q_rope=q, k_rope=k, cos=None, sin=None)
        assert torch.equal(q[4:6], q_original[2:4])
    assert net.language_model.model.layers[1].self_attn._rope_qk_capture_callback is None


def test_sparse_controller_is_default_off_and_reports_token_reduction() -> None:
    net = _Net()
    controller = SparseProjectionQKReuseController(
        torch=torch,
        net=net,
        guidance=3.0,
        num_steps=4,
        plan={(2, 1): ((1, 2),)},
    )
    attention = net.language_model.model.layers[1].self_attn
    assert attention._sparse_qk_projection_callback is None
    with controller:
        assert attention._sparse_qk_projection_callback is not None
        controller.trace.extend(
            [
                SparseProjectionTrace(2, "conditional", 1, "sparse", 20, 18, 2, 1),
                SparseProjectionTrace(2, "unconditional", 1, "sparse", 20, 18, 2, 1),
            ]
        )
        validation = controller.validate_complete()
        assert validation["actual_invocations"] == 2
        assert validation["selected_path_token_ratio"] == 0.9
    assert attention._sparse_qk_projection_callback is None
