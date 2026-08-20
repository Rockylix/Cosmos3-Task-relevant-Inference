# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Default-off post-RoPE Q/K replacement controller for Cosmos3 Edge experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from cosmos_framework.scripts.robolab_attention_stats_capture import _gen_token_layout

DEFAULT_REUSE_PLAN: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 27): ((7, 8),),
    (2, 1): ((1, 2), (3, 4), (5, 6), (7, 8)),
    (3, 1): ((1, 2), (3, 4), (5, 6), (7, 8)),
}


@dataclass(frozen=True)
class ReuseTrace:
    step: int
    branch: str
    block: int
    source_latent: int
    target_latent: int
    token_count: int
    q_post_copy_max_abs: float | None
    k_post_copy_max_abs: float | None
    finite: bool | None


@dataclass(frozen=True)
class SparseProjectionTrace:
    step: int
    branch: str
    block: int
    mode: str
    total_tokens: int
    computed_tokens: int
    skipped_tokens: int
    pair_count: int


@dataclass(frozen=True)
class PreparedQKReuseIndices:
    total_tokens: int
    keep_index: Any
    source_index: Any
    target_index: Any


def validate_reuse_plan(
    plan: Mapping[tuple[int, int], Sequence[tuple[int, int]]],
    *,
    num_steps: int,
    num_blocks: int,
) -> None:
    """Validate step/block bounds and require disjoint adjacent latent pairs."""

    for (step, block), pairs in plan.items():
        if step < 0 or step >= num_steps:
            raise ValueError(f"Invalid denoise step {step} for {num_steps} steps")
        if block < 0 or block >= num_blocks:
            raise ValueError(f"Invalid block {block} for {num_blocks} blocks")
        used: set[int] = set()
        for source, target in pairs:
            if source < 1 or target > 8 or target != source + 1:
                raise ValueError(f"Expected adjacent future latent pair in L1..L8, got {source}->{target}")
            if source in used or target in used:
                raise ValueError(f"Overlapping reuse pair at step={step}, block={block}: {source}->{target}")
            used.update((source, target))


def replace_post_rope_qk(
    *,
    torch: Any,
    q_rope: Any,
    k_rope: Any,
    token_layout: Mapping[str, Any],
    pairs: Sequence[tuple[int, int]],
    validate_copy: bool = True,
) -> list[dict[str, Any]]:
    """Copy source-frame post-RoPE Q/K to disjoint target-frame token slices."""

    num_gen_tokens = int(token_layout["num_gen_tokens"])
    if q_rope.ndim != 3 or k_rope.ndim != 3:
        raise RuntimeError("Q/K must be [tokens,heads,head_dim]")
    if int(q_rope.shape[0]) < num_gen_tokens or int(k_rope.shape[0]) < num_gen_tokens:
        raise RuntimeError("Q/K tensors are shorter than the live GEN token layout")
    traces = []
    for source, target in pairs:
        source_index = torch.tensor(
            token_layout["latent_positions"][f"L{source}"],
            device=q_rope.device,
            dtype=torch.long,
        )
        target_index = torch.tensor(
            token_layout["latent_positions"][f"L{target}"],
            device=q_rope.device,
            dtype=torch.long,
        )
        if source_index.shape != target_index.shape or source_index.numel() == 0:
            raise RuntimeError(f"Invalid live token layout for L{source}->L{target}")
        q_source = q_rope.index_select(0, source_index).clone()
        k_source = k_rope.index_select(0, source_index).clone()
        q_rope.index_copy_(0, target_index, q_source)
        k_rope.index_copy_(0, target_index, k_source)
        q_error = None
        k_error = None
        finite = None
        if validate_copy:
            q_after = q_rope.index_select(0, target_index)
            k_after = k_rope.index_select(0, target_index)
            q_error = float((q_after - q_source).detach().float().abs().max().cpu())
            k_error = float((k_after - k_source).detach().float().abs().max().cpu())
            finite = bool(torch.isfinite(q_after).all() and torch.isfinite(k_after).all())
        traces.append(
            {
                "source_latent": source,
                "target_latent": target,
                "token_count": int(source_index.numel()),
                "q_post_copy_max_abs": q_error,
                "k_post_copy_max_abs": k_error,
                "finite": finite,
            }
        )
    return traces


def project_gen_qk_with_reuse(
    *,
    torch: Any,
    hidden_states: Any,
    cos: Any,
    sin: Any,
    q_proj: Any,
    k_proj: Any,
    q_norm: Any,
    k_norm: Any,
    apply_rotary_pos_emb: Any,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    token_layout: Mapping[str, Any],
    pairs: Sequence[tuple[int, int]],
    dense_control: bool = False,
    prepared_indices: PreparedQKReuseIndices | None = None,
    restore_raw_qk: bool = True,
) -> tuple[tuple[Any, Any, Any, Any], dict[str, int]]:
    """Compute GEN Q/K only for kept tokens, then restore targets from source Q/K."""

    total_tokens = int(token_layout["num_gen_tokens"])
    if int(hidden_states.shape[0]) != total_tokens:
        raise RuntimeError(f"Hidden/token-layout mismatch: {hidden_states.shape[0]} != {total_tokens}")
    if dense_control:
        hidden_keep = hidden_states
        cos_keep = cos
        sin_keep = sin
        keep_index = None
    else:
        if prepared_indices is None:
            prepared_indices = prepare_qk_reuse_indices(
                torch=torch,
                token_layout=token_layout,
                pairs=pairs,
                device=hidden_states.device,
            )
        if prepared_indices.total_tokens != total_tokens:
            raise RuntimeError("Prepared sparse Q/K indices use a different token layout")
        keep_index = prepared_indices.keep_index
        hidden_keep = hidden_states.index_select(0, keep_index)
        cos_keep = cos.index_select(0, keep_index)
        sin_keep = sin.index_select(0, keep_index)

    q_raw_keep = q_proj(hidden_keep).view(-1, num_attention_heads, head_dim)
    k_raw_keep = k_proj(hidden_keep).view(-1, num_key_value_heads, head_dim)
    q_raw_keep = q_norm(q_raw_keep)
    k_raw_keep = k_norm(k_raw_keep)
    q_rope_keep, k_rope_keep = apply_rotary_pos_emb(
        q_raw_keep,
        k_raw_keep,
        cos_keep,
        sin_keep,
        unsqueeze_dim=1,
    )
    computed_tokens = int(hidden_keep.shape[0])
    stats = {
        "total_tokens": total_tokens,
        "computed_tokens": computed_tokens,
        "skipped_tokens": total_tokens - computed_tokens,
        "pair_count": len(pairs),
    }
    if dense_control:
        return (q_raw_keep, k_raw_keep, q_rope_keep, k_rope_keep), stats

    assert prepared_indices is not None and keep_index is not None

    def restore(kept: Any) -> Any:
        full = kept.new_empty((total_tokens, *kept.shape[1:]))
        full.index_copy_(0, keep_index, kept)
        full.index_copy_(
            0,
            prepared_indices.target_index,
            full.index_select(0, prepared_indices.source_index),
        )
        return full

    q_raw = restore(q_raw_keep) if restore_raw_qk else None
    k_raw = restore(k_raw_keep) if restore_raw_qk else None
    return (
        q_raw,
        k_raw,
        restore(q_rope_keep),
        restore(k_rope_keep),
    ), stats


def prepare_qk_reuse_indices(
    *,
    torch: Any,
    token_layout: Mapping[str, Any],
    pairs: Sequence[tuple[int, int]],
    device: Any,
) -> PreparedQKReuseIndices:
    """Build reusable keep/source/target indices for one step/block reuse pattern."""

    total_tokens = int(token_layout["num_gen_tokens"])
    source_positions = [
        position for source, _target in pairs for position in token_layout["latent_positions"][f"L{source}"]
    ]
    target_positions = [
        position for _source, target in pairs for position in token_layout["latent_positions"][f"L{target}"]
    ]
    if len(source_positions) != len(target_positions) or len(target_positions) != len(set(target_positions)):
        raise RuntimeError("Invalid or overlapping sparse Q/K source/target positions")
    source_index = torch.tensor(source_positions, device=device, dtype=torch.long)
    target_index = torch.tensor(target_positions, device=device, dtype=torch.long)
    keep_mask = torch.ones(total_tokens, device=device, dtype=torch.bool)
    keep_mask[target_index] = False
    keep_index = torch.nonzero(keep_mask, as_tuple=False).flatten()
    return PreparedQKReuseIndices(
        total_tokens=total_tokens,
        keep_index=keep_index,
        source_index=source_index,
        target_index=target_index,
    )


class PostRopeQKReuseController:
    """Install a request-scoped controller; normal inference remains unchanged by default."""

    def __init__(
        self,
        *,
        torch: Any,
        net: Any,
        guidance: float,
        num_steps: int,
        plan: Mapping[tuple[int, int], Sequence[tuple[int, int]]] = DEFAULT_REUSE_PLAN,
        selected_branches: Sequence[str] = ("conditional", "unconditional"),
        validate_copy: bool = True,
    ) -> None:
        self.torch = torch
        self.net = net
        self.guidance = float(guidance)
        self.num_steps = int(num_steps)
        self.plan = {key: tuple(pairs) for key, pairs in plan.items()}
        self.selected_branches = tuple(selected_branches)
        self.validate_copy = bool(validate_copy)
        layers = getattr(getattr(getattr(net, "language_model", None), "model", None), "layers", None)
        if layers is None:
            raise RuntimeError("Could not locate net.language_model.model.layers")
        self.layers = list(layers)
        validate_reuse_plan(self.plan, num_steps=self.num_steps, num_blocks=len(self.layers))
        if self.guidance == 1.0 and "unconditional" in self.selected_branches:
            raise ValueError("Unconditional intervention requires CFG guidance != 1")
        self.selected_blocks = sorted({block for _step, block in self.plan})
        self._network_handles: list[Any] = []
        self._attention_modules: list[Any] = []
        self._current: dict[str, Any] | None = None
        self._token_layout: dict[str, Any] | None = None
        self._forward_call_index = 0
        self.trace: list[ReuseTrace] = []

    def __enter__(self) -> "PostRopeQKReuseController":
        self._network_handles.append(self.net.register_forward_pre_hook(self._network_pre_hook, with_kwargs=True))
        self._network_handles.append(self.net.register_forward_hook(self._network_post_hook, with_kwargs=True))
        for block in self.selected_blocks:
            attention = getattr(self.layers[block], "self_attn", None)
            if attention is None or not hasattr(attention, "_rope_qk_capture_callback"):
                raise RuntimeError(f"Block {block} does not expose the RoPE Q/K hook")
            if attention._rope_qk_capture_callback is not None:
                raise RuntimeError(f"Block {block} already has a RoPE Q/K callback")
            attention._rope_qk_capture_callback = self._intervene
            self._attention_modules.append(attention)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for attention in self._attention_modules:
            attention._rope_qk_capture_callback = None
        self._attention_modules.clear()
        for handle in reversed(self._network_handles):
            handle.remove()
        self._network_handles.clear()
        self._current = None

    def _call_semantics(self, call_index: int) -> tuple[int, str]:
        if self.guidance == 1.0:
            return call_index, "conditional"
        return (
            call_index // 2,
            "conditional" if call_index % 2 == 0 else "unconditional",
        )

    def _network_pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        del module
        und_only = bool(kwargs.get("und_only", args[2] if len(args) > 2 else False))
        if und_only:
            self._current = None
            return
        packed_sequence = args[0] if args else kwargs.get("packed_seq")
        call_index = self._forward_call_index
        self._forward_call_index += 1
        step, branch = self._call_semantics(call_index)
        layout = _gen_token_layout(self.torch, packed_sequence)
        if self._token_layout is None:
            self._token_layout = layout
        elif layout != self._token_layout:
            raise RuntimeError("GEN token layout changed across calls")
        self._current = {"step": step, "branch": branch}

    def _network_post_hook(self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], output: Any) -> None:
        del module, args, kwargs, output
        self._current = None

    def _intervene(
        self,
        *,
        layer_index: int,
        q_raw: Any,
        k_raw: Any,
        q_rope: Any,
        k_rope: Any,
        cos: Any,
        sin: Any,
    ) -> None:
        del q_raw, k_raw, cos, sin
        if self._current is None or self._token_layout is None:
            return
        step = int(self._current["step"])
        branch = str(self._current["branch"])
        pairs = self.plan.get((step, int(layer_index)))
        if pairs is None or branch not in self.selected_branches:
            return
        rows = replace_post_rope_qk(
            torch=self.torch,
            q_rope=q_rope,
            k_rope=k_rope,
            token_layout=self._token_layout,
            pairs=pairs,
            validate_copy=self.validate_copy,
        )
        for row in rows:
            self.trace.append(ReuseTrace(step=step, branch=branch, block=int(layer_index), **row))

    def validate_complete(self) -> dict[str, Any]:
        expected = sum(len(pairs) for pairs in self.plan.values()) * len(self.selected_branches)
        finite = all(row.finite for row in self.trace) if self.validate_copy else None
        exact = (
            all(row.q_post_copy_max_abs == 0.0 and row.k_post_copy_max_abs == 0.0 for row in self.trace)
            if self.validate_copy
            else None
        )
        if len(self.trace) != expected or (self.validate_copy and (not finite or not exact)):
            raise RuntimeError(
                f"Incomplete/invalid Q/K intervention: trace={len(self.trace)}/{expected}, "
                f"finite={finite}, exact_copy={exact}"
            )
        return {
            "expected_replacement_pairs": expected,
            "actual_replacement_pairs": len(self.trace),
            "tokens_per_pair": sorted({row.token_count for row in self.trace}),
            "copy_checks_performed": self.validate_copy,
            "finite": finite,
            "exact_copy": exact,
        }


class SparseProjectionQKReuseController:
    """Request-scoped real Q/K projection skip using the profiled reuse plan."""

    _prepared_index_cache: dict[tuple[Any, ...], PreparedQKReuseIndices] = {}

    def __init__(
        self,
        *,
        torch: Any,
        net: Any,
        guidance: float,
        num_steps: int,
        plan: Mapping[tuple[int, int], Sequence[tuple[int, int]]] = DEFAULT_REUSE_PLAN,
        selected_branches: Sequence[str] = ("conditional", "unconditional"),
        mode: str = "sparse",
        record_cuda_events: bool = False,
    ) -> None:
        if mode not in {"sparse", "dense_control"}:
            raise ValueError(f"Unknown sparse Q/K mode: {mode}")
        self.torch = torch
        self.net = net
        self.guidance = float(guidance)
        self.num_steps = int(num_steps)
        self.plan = {key: tuple(pairs) for key, pairs in plan.items()}
        self.selected_branches = tuple(selected_branches)
        self.mode = mode
        self.record_cuda_events = bool(record_cuda_events)
        layers = getattr(getattr(getattr(net, "language_model", None), "model", None), "layers", None)
        if layers is None:
            raise RuntimeError("Could not locate net.language_model.model.layers")
        self.layers = list(layers)
        validate_reuse_plan(self.plan, num_steps=self.num_steps, num_blocks=len(self.layers))
        if self.guidance == 1.0 and "unconditional" in self.selected_branches:
            raise ValueError("Unconditional intervention requires CFG guidance != 1")
        self.selected_blocks = sorted({block for _step, block in self.plan})
        self._network_handles: list[Any] = []
        self._attention_modules: list[Any] = []
        self._current: dict[str, Any] | None = None
        self._token_layout: dict[str, Any] | None = None
        self._forward_call_index = 0
        self.trace: list[SparseProjectionTrace] = []
        self._cuda_events: list[tuple[Any, Any, int, str, int]] = []

    def __enter__(self) -> "SparseProjectionQKReuseController":
        self._network_handles.append(self.net.register_forward_pre_hook(self._network_pre_hook, with_kwargs=True))
        self._network_handles.append(self.net.register_forward_hook(self._network_post_hook, with_kwargs=True))
        for block in self.selected_blocks:
            attention = getattr(self.layers[block], "self_attn", None)
            if attention is None or not hasattr(attention, "_sparse_qk_projection_callback"):
                raise RuntimeError(f"Block {block} does not expose the sparse Q/K projection hook")
            if attention._sparse_qk_projection_callback is not None:
                raise RuntimeError(f"Block {block} already has a sparse Q/K projection callback")
            attention._sparse_qk_projection_callback = self._project
            self._attention_modules.append(attention)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for attention in self._attention_modules:
            attention._sparse_qk_projection_callback = None
        self._attention_modules.clear()
        for handle in reversed(self._network_handles):
            handle.remove()
        self._network_handles.clear()
        self._current = None

    def _call_semantics(self, call_index: int) -> tuple[int, str]:
        if self.guidance == 1.0:
            return call_index, "conditional"
        return (
            call_index // 2,
            "conditional" if call_index % 2 == 0 else "unconditional",
        )

    def _network_pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        del module
        und_only = bool(kwargs.get("und_only", args[2] if len(args) > 2 else False))
        if und_only:
            self._current = None
            return
        packed_sequence = args[0] if args else kwargs.get("packed_seq")
        call_index = self._forward_call_index
        self._forward_call_index += 1
        step, branch = self._call_semantics(call_index)
        layout = _gen_token_layout(self.torch, packed_sequence)
        if self._token_layout is None:
            self._token_layout = layout
        elif layout != self._token_layout:
            raise RuntimeError("GEN token layout changed across calls")
        self._current = {"step": step, "branch": branch}

    def _network_post_hook(self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], output: Any) -> None:
        del module, args, kwargs, output
        self._current = None

    def _project(self, *, layer_index: int, **kwargs: Any) -> tuple[Any, Any, Any, Any] | None:
        if self._current is None or self._token_layout is None:
            return None
        step = int(self._current["step"])
        branch = str(self._current["branch"])
        pairs = self.plan.get((step, int(layer_index)))
        if pairs is None or branch not in self.selected_branches:
            return None
        start_event = None
        end_event = None
        if self.record_cuda_events:
            start_event = self.torch.cuda.Event(enable_timing=True)
            end_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record()
        prepared_indices = None
        if self.mode == "sparse":
            hidden_states = kwargs["hidden_states"]
            cache_key = (
                str(hidden_states.device),
                tuple(self._token_layout["latent_shape_thw"]),
                int(self._token_layout["num_gen_tokens"]),
                tuple(pairs),
            )
            prepared_indices = self._prepared_index_cache.get(cache_key)
            if prepared_indices is None:
                prepared_indices = prepare_qk_reuse_indices(
                    torch=self.torch,
                    token_layout=self._token_layout,
                    pairs=pairs,
                    device=hidden_states.device,
                )
                self._prepared_index_cache[cache_key] = prepared_indices
        result, stats = project_gen_qk_with_reuse(
            torch=self.torch,
            token_layout=self._token_layout,
            pairs=pairs,
            dense_control=self.mode == "dense_control",
            prepared_indices=prepared_indices,
            restore_raw_qk=False,
            **kwargs,
        )
        if end_event is not None:
            end_event.record()
            assert start_event is not None
            self._cuda_events.append((start_event, end_event, step, branch, int(layer_index)))
        self.trace.append(
            SparseProjectionTrace(
                step=step,
                branch=branch,
                block=int(layer_index),
                mode=self.mode,
                **stats,
            )
        )
        return result

    def validate_complete(self) -> dict[str, Any]:
        expected_invocations = len(self.plan) * len(self.selected_branches)
        expected_pairs = sum(len(pairs) for pairs in self.plan.values()) * len(self.selected_branches)
        actual_pairs = sum(row.pair_count for row in self.trace)
        if len(self.trace) != expected_invocations or actual_pairs != expected_pairs:
            raise RuntimeError(
                f"Incomplete sparse Q/K projection: calls={len(self.trace)}/{expected_invocations}, "
                f"pairs={actual_pairs}/{expected_pairs}"
            )
        total_tokens = sum(row.total_tokens for row in self.trace)
        computed_tokens = sum(row.computed_tokens for row in self.trace)
        return {
            "mode": self.mode,
            "expected_invocations": expected_invocations,
            "actual_invocations": len(self.trace),
            "expected_pairs": expected_pairs,
            "actual_pairs": actual_pairs,
            "selected_path_total_tokens": total_tokens,
            "selected_path_computed_tokens": computed_tokens,
            "selected_path_token_ratio": computed_tokens / total_tokens,
            "selected_path_skipped_tokens": total_tokens - computed_tokens,
        }

    def cuda_timing_rows(self) -> list[dict[str, Any]]:
        if not self.record_cuda_events:
            return []
        return [
            {
                "step": step,
                "branch": branch,
                "block": block,
                "milliseconds": float(start.elapsed_time(end)),
            }
            for start, end, step, branch, block in self._cuda_events
        ]
