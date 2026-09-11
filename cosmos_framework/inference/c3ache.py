# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in cross-chunk GEN residual reuse for single-device RoboLab inference.

The model supplies explicit sampler-step/CFG identities. Only the pre-final-norm
GEN residual is retained; embeddings, heads, conditioning and sampling stay live.
This module is not imported by the training/model code: the server installs a
request-scoped, duck-typed context on the transformer while holding its lock.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch


def _freeze(value: Any) -> Any:
    """Snapshot small layout metadata, excluding observations and noisy token values.

    Tensor metadata is copied to CPU for exact comparisons. This conservative
    prototype pays synchronization overhead to reject same-size layout changes.
    """
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), str(value.dtype), tuple(value.detach().cpu().reshape(-1).tolist()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    return value


@dataclass(frozen=True)
class C3acheConfig:
    """Refresh every N chunks; keep the last dense_tail_steps fully computed."""

    refresh_period: int = 2
    dense_tail_steps: int = 2
    num_steps: int = 4
    max_sessions: int = 4

    def __post_init__(self) -> None:
        """Reject invalid periods and schedules without a protected final step."""
        if self.refresh_period < 1 or self.max_sessions < 1:
            raise ValueError("C3ache refresh_period and max_sessions must be positive")
        if not 1 <= self.dense_tail_steps <= self.num_steps:
            raise ValueError("C3ache requires at least one dense tail step")


@dataclass
class _Episode:
    """Bounded cache owned by exactly one client/environment/episode."""

    signature: Any
    last_chunk: int = -1
    entries: dict = field(default_factory=dict)


class C3acheRequest:
    """One serialized planning request with explicit branch and step contexts."""

    def __init__(self, state: _Episode, config: C3acheConfig, chunk: int, reason: str, branches: tuple) -> None:
        """Bind an episode cache and force dense computation after any invalidation."""
        self.state, self.config, self.chunk, self.branches = state, config, chunk, branches
        self.refresh = bool(reason) or chunk % config.refresh_period == 0
        self.reason = reason or ("periodic_refresh" if self.refresh else "reuse")
        self.active = None
        self.layout = None
        self.seen: set = set()
        self.stats = {"dense_forwards": 0, "cache_hits": 0, "cached_bytes": 0}

    @contextmanager
    def forward_context(self, step: int, branch: str, timestep: torch.Tensor):
        """Tag one actual velocity evaluation; never infer CFG identity by parity."""
        key = (branch, step)
        if self.active is not None or key in self.seen:
            raise RuntimeError("Duplicate or nested C3ache denoising call")
        if branch not in self.branches or not 0 <= step < self.config.num_steps:
            raise RuntimeError("Unsupported C3ache sampler/CFG schedule")
        self.active = (key, _freeze(timestep))
        self.layout = None
        try:
            yield
            if key not in self.seen:
                raise RuntimeError("C3ache did not reach the GEN transformer")
        finally:
            self.active = None
            self.layout = None

    def set_layout(self, packed: Any, temporal_causal: bool) -> None:
        """Record token roles, masks, geometry and text, never the observation content."""
        layout = {
            "sample_lens": packed.sample_lens,
            "split_lens": packed.split_lens,
            "attn_modes": packed.attn_modes,
            "text_ids": packed.text_ids,
            "text_indexes": packed.text_indexes,
            "position_ids": packed.position_ids,
            "temporal_causal": temporal_causal,
            "num_action_tokens_per_supertoken": packed.num_action_tokens_per_supertoken,
            "null_action_supertokens": packed.null_action_supertokens,
        }
        for name in ("vision", "action", "sound"):
            modality = getattr(packed, name)
            layout[name] = (
                None
                if modality is None
                else {
                    key: getattr(modality, key)
                    for key in ("token_shapes", "sequence_indexes", "condition_mask", "domain_id", "raw_action_dim")
                }
            )
        self.layout = _freeze(layout)

    def before_transformer(self, gen: torch.Tensor, position_ids: torch.Tensor):
        """Return (cached output, refresh ticket), guarding dtype/device/layout/step.

        A ticket owns a clone of h0 because decoder implementations may mutate
        their input. On a hit, all GEN rows (including conditions) use h0 + R.
        """
        if self.active is None or self.layout is None:
            raise RuntimeError("C3ache requires an explicit velocity context and packed layout")
        key, timestep = self.active
        if key in self.seen:
            raise RuntimeError("Multiple transformer stacks in one C3ache velocity call")
        self.seen.add(key)
        signature = (tuple(gen.shape), str(gen.dtype), str(gen.device), self.layout, _freeze(position_ids), timestep)
        entry = self.state.entries.get(key)
        if entry is not None and entry[0] != signature:
            self.state.entries.clear()
            self.refresh = True
            self.reason = "layout_or_timestep_changed"
            entry = None
        cacheable = key[1] < self.config.num_steps - self.config.dense_tail_steps
        if cacheable and not self.refresh and entry is not None:
            self.stats["cache_hits"] += 1
            return gen + entry[1], None
        self.stats["dense_forwards"] += 1
        ticket = (key, signature, gen.detach().clone()) if cacheable else None
        return None, ticket

    def after_transformer(self, ticket: Any, gen: torch.Tensor) -> None:
        """Store hL - h0 BEFORE final norm without changing the dense output."""
        if ticket is not None:
            key, signature, h0 = ticket
            self.state.entries[key] = (signature, (gen.detach() - h0).detach())

    def finish(self) -> dict:
        """Commit only a complete expected schedule and report measured reuse counts."""
        expected = {(branch, step) for branch in self.branches for step in range(self.config.num_steps)}
        if self.seen != expected:
            raise RuntimeError(f"Incomplete C3ache schedule: expected {expected}, received {self.seen}")
        self.state.last_chunk = self.chunk
        self.stats["cached_bytes"] = sum(t.numel() * t.element_size() for _, t in self.state.entries.values())
        return dict(self.stats, chunk_id=self.chunk, reason=self.reason, refresh_period=self.config.refresh_period)


class C3acheCache:
    """LRU-bounded session isolation; failures and discontinuities invalidate state."""

    def __init__(self, config: C3acheConfig) -> None:
        """Initialize an empty cache; the serving lock serializes all mutations."""
        self.config = config
        self.episodes: OrderedDict = OrderedDict()

    @staticmethod
    def identity(obs: dict) -> tuple[str, str]:
        """Require explicit nonempty session and episode IDs when caching is enabled."""
        values = tuple(obs.get(key) for key in ("session_id", "episode_id"))
        if not all(isinstance(value, str) and 0 < len(value) <= 256 for value in values):
            raise ValueError("C3ache requires nonempty session_id and episode_id strings (max 256 characters)")
        return values

    def reset(self, obs: dict) -> None:
        """Drop only the explicitly addressed session/episode; leave other clients intact."""
        self.episodes.pop(self.identity(obs), None)

    @contextmanager
    def request(self, obs: dict, *, signature: Any, transformer: Any, net: Any, branches: tuple):
        """Install the context for one request and restore model attributes on all exits.

        New episodes, changed configuration/task, skipped/duplicate chunk IDs,
        eviction or request failures force fresh dense residuals. Old episodes
        of the same per-environment session are released on an episode switch.
        """
        key = self.identity(obs)
        chunk = obs.get("chunk_id")
        if type(chunk) is not int or chunk < 0:
            raise ValueError("C3ache chunk_id must be a nonnegative integer")
        signature = _freeze(signature)
        state = self.episodes.get(key)
        reason = ""
        if state is None:
            reason = "new_episode_or_evicted"
        elif state.signature != signature:
            reason = "task_or_configuration_changed"
        elif chunk != state.last_chunk + 1:
            reason = "nonconsecutive_chunk"
        if reason:
            state = _Episode(signature)
        for previous in list(self.episodes):
            if previous[0] == key[0] and previous != key:
                self.episodes.pop(previous)
        self.episodes[key] = state
        self.episodes.move_to_end(key)
        while len(self.episodes) > self.config.max_sessions:
            self.episodes.popitem(last=False)
        request = C3acheRequest(state, self.config, chunk, reason, branches)
        targets = (transformer, net)
        previous_contexts = [getattr(target, "_c3ache_request", None) for target in targets]
        if any(value is not None for value in previous_contexts):
            raise RuntimeError("Nested C3ache request installation")
        try:
            for target in targets:
                target._c3ache_request = request
            yield request
            request.finish()
        except BaseException:
            self.episodes.pop(key, None)
            raise
        finally:
            for target, previous in zip(targets, previous_contexts):
                target._c3ache_request = previous
