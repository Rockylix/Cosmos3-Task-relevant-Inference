"""Observation-coordinate SpecPrune plan and per-forward exit hidden bookkeeping.

No ASI, velocity caching, model mutation, or cross-step hidden reuse.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExitConfig:
    local_k: int = 32
    global_k: int = 40
    global_blocks: tuple = (13, 27)
    update_blocks: tuple = (14, 19, 24)
    prune_blocks: tuple = (10, 15, 20, 25)
    low_change_cap: int = 313
    dynamic_threshold: float = 0.986
    keep_ratio: float = 0.9
    ema_beta: float = 0.2
    min_observation_tokens: int = 60
    controller: bool = False


def top_mask(score, k, candidates=None):
    if score.ndim != 1 or not torch.isfinite(score).all():
        raise ValueError("Expected finite observation scores")
    if candidates is None:
        candidates = torch.ones_like(score, dtype=torch.bool)
    ids = torch.where(candidates)[0]
    order = torch.argsort(score[ids], descending=True, stable=True)
    result = torch.zeros_like(candidates)
    result[ids[order[: max(0, int(k))]]] = True
    return result


class ObservationPlan:
    def __init__(self, config=ExitConfig()):
        self.config = config
        if config.controller:
            raise ValueError("Joint-position action controller is intentionally unsupported")
        self.reset()

    def reset(self):
        self.previous_rgb = None
        self.previous_global = {}
        self.confidence = {}
        self.pending_rgb = None

    def begin(self, rgb, nonvisual_tokens):
        if self.pending_rgb is not None:
            raise RuntimeError("Reset or finish previous chunk first")
        if rgb.ndim != 2 or rgb.shape[0] != 340 or not torch.isfinite(rgb).all():
            raise ValueError("Expected finite observation patches [340,features]")
        self.pending_rgb = rgb.detach().clone()
        self.nonvisual_tokens = int(nonvisual_tokens)
        self.active = torch.ones(340, dtype=torch.bool, device=rgb.device)
        self.global_mask = torch.zeros_like(self.active)
        self.dynamic_mask = torch.zeros_like(self.active)
        self.similarity = None
        if self.previous_rgb is not None:
            previous = self.previous_rgb.to(rgb.device)
            self.similarity = (rgb * previous).sum(-1) / (rgb.norm(dim=-1) * previous.norm(dim=-1) + 1e-8)
            stable = top_mask(
                self.similarity, self.config.low_change_cap, self.similarity >= self.config.dynamic_threshold
            )
            self.dynamic_mask = ~stable
            for b in self.config.global_blocks:
                self.global_mask |= top_mask(self.previous_global[b], self.config.global_k)
        self.importance = torch.zeros(340, device=rgb.device)
        self.dynamic_enabled = True
        self.masks, self.rows, self.local_scores, self.global_scores = {}, [], {}, {}
        self.action_scores = {}

    def local(self, block, score):
        self.local_scores[block] = score.detach().clone()
        if block == 0:
            self.local0_top = top_mask(score, self.config.local_k)
            self.active = top_mask(score, 2 * self.config.local_k) | self.global_mask | self.dynamic_mask
        elif block == 1:
            keep = top_mask(score, self.config.local_k, self.active) | self.local0_top
            self.active &= keep | self.global_mask | self.dynamic_mask
        else:
            raise ValueError("Local selection is B0/B1 only")
        self.save(block, "local")

    def update(self, block, action_to_observation):
        if action_to_observation.shape != (32, 340) or not torch.isfinite(action_to_observation).all():
            raise ValueError("Expected action q1..q32 to L0 probabilities")
        if (action_to_observation < 0).any():
            raise ValueError("Negative attention probability")
        ids = torch.where(self.active)[0]
        attn = action_to_observation[:, ids].float()
        self.action_scores[block] = action_to_observation.detach().clone()
        if not self.dynamic_enabled:
            return
        order = torch.argsort(attn.mean(0), descending=True, stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(len(ids), device=ids.device)
        weight = torch.sigmoid(-rank.float())
        weight /= weight.sum() + 1e-8
        if block not in self.confidence:
            prob = attn + 1e-8
            prob = prob / prob.sum(-1, keepdim=True)
            entropy = -(prob * prob.log()).sum(-1).mean() / torch.tensor(max(len(ids), 2), device=ids.device).log()
            self.confidence[block] = (1 / (entropy + 1e-8)).detach()
        beta = self.config.ema_beta
        self.importance[ids] = (1 - beta) * self.importance[ids] + beta * weight * self.confidence[block]

    def prune(self, block):
        count = int(self.active.sum())
        # Port total-sequence 0.9 to a virtual [observation | text | action]
        # sequence; never count eight copies of a spatial choice as eight votes.
        retain_total = max(
            int(self.config.keep_ratio * (count + self.nonvisual_tokens)),
            self.nonvisual_tokens + self.config.min_observation_tokens,
        )
        retain = min(count, retain_total - self.nonvisual_tokens)
        if retain >= count:
            self.dynamic_enabled = False
        self.active = top_mask(self.importance, retain, self.active)
        self.save(block, "dynamic")

    def save(self, block, kind):
        if self.masks and (self.active & ~next(reversed(self.masks.values()))).any():
            raise AssertionError("Layer plan reintroduced a deleted position")
        self.masks[block] = self.active.detach().clone()
        self.rows.append(
            dict(block=block, kind=kind, keep=int(self.active.sum()), scores_nonzero=int((self.importance != 0).sum()))
        )

    def finish(self):
        if set(self.global_scores) != set(self.config.global_blocks):
            raise ValueError("Incomplete historical observation attention")
        self.previous_rgb = self.pending_rgb
        self.pending_rgb = None
        self.previous_global = {b: s.detach().clone() for b, s in self.global_scores.items()}


class ExitHidden:
    """Each sequence position is written exactly once, within one forward only."""

    def __init__(self, reference):
        self.values = torch.empty_like(reference)
        self.written = torch.zeros(len(reference), dtype=torch.bool, device=reference.device)

    def put(self, ids, hidden):
        if ids.ndim != 1 or len(ids) != len(hidden) or ids.unique().numel() != len(ids):
            raise ValueError("Invalid exit coordinates")
        if self.written[ids].any():
            raise AssertionError("Hidden position restored twice")
        self.values.index_copy_(0, ids, hidden)
        self.written[ids] = True

    def finish(self):
        if not self.written.all() or not torch.isfinite(self.values).all():
            raise AssertionError("Incomplete or nonfinite full hidden reconstruction")
        return self.values
