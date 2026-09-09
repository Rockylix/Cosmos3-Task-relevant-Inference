"""Resolve the DROID GEN token layout used by the Edge sparse policy."""

from __future__ import annotations

import math
from typing import Any


def _gen_token_layout(torch: Any, packed_sequence: Any) -> dict[str, Any]:
    full_indexes: list[int] = []
    offset = 0
    for mode, split_len in zip(packed_sequence.attn_modes, packed_sequence.split_lens, strict=True):
        split_len = int(split_len)
        if mode == "full":
            full_indexes.extend(range(offset, offset + split_len))
        offset += split_len
    full_position = {original: position for position, original in enumerate(full_indexes)}

    vision = getattr(packed_sequence, "vision", None)
    action = getattr(packed_sequence, "action", None)
    vision_indexes = getattr(vision, "sequence_indexes", None)
    if vision is None or not torch.is_tensor(vision_indexes):
        raise RuntimeError("Attention statistics require packed vision tokens")
    vision_original = [int(value) for value in vision_indexes.detach().cpu().tolist()]
    action_indexes = getattr(action, "sequence_indexes", None)
    action_original = (
        [int(value) for value in action_indexes.detach().cpu().tolist()] if torch.is_tensor(action_indexes) else []
    )
    try:
        vision_positions = [full_position[index] for index in vision_original]
        action_positions = [full_position[index] for index in action_original]
    except KeyError as exc:
        raise RuntimeError(f"Modality token {int(exc.args[0])} is not in the GEN sequence") from exc

    token_shapes = getattr(vision, "token_shapes", None)
    if not token_shapes or len(token_shapes) != 1:
        raise RuntimeError(f"Expected exactly one vision token shape, got {token_shapes}")
    latent_shape = tuple(int(value) for value in token_shapes[0])
    if len(latent_shape) != 3 or math.prod(latent_shape) != len(vision_positions):
        raise RuntimeError(f"Vision token shape {latent_shape} does not match {len(vision_positions)} tokens")
    num_latents, height, width = latent_shape
    if num_latents != 9:
        raise RuntimeError(f"Expected condition L0 plus future L1..L8, got {num_latents} vision latents")
    spatial_tokens = height * width
    latent_positions = {
        f"L{latent}": vision_positions[latent * spatial_tokens : (latent + 1) * spatial_tokens]
        for latent in range(num_latents)
    }
    covered = [position for positions in latent_positions.values() for position in positions] + action_positions
    if len(covered) != len(full_indexes) or sorted(covered) != list(range(len(full_indexes))):
        raise RuntimeError("L0..L8 and action groups do not exactly partition GEN tokens")
    return {
        "num_gen_tokens": len(full_indexes),
        "latent_shape_thw": list(latent_shape),
        "latent_positions": latent_positions,
        "action_positions": action_positions,
    }


def _action_query_layout(torch: Any, packed_sequence: Any) -> dict[str, Any]:
    layout = _gen_token_layout(torch, packed_sequence)
    action = getattr(packed_sequence, "action", None)
    if action is None:
        raise RuntimeError("Action Query statistics require packed action tokens")
    action_positions = list(map(int, layout["action_positions"]))
    token_shapes = getattr(action, "token_shapes", None)
    if not token_shapes or len(token_shapes) != 1:
        raise RuntimeError(f"Expected one action token shape, got {token_shapes}")
    token_shape = tuple(int(value) for value in token_shapes[0])
    if math.prod(token_shape) != len(action_positions):
        raise RuntimeError(f"Action token shape {token_shape} does not match {len(action_positions)} positions")
    condition_mask = getattr(action, "condition_mask", None)
    if isinstance(condition_mask, (list, tuple)):
        if len(condition_mask) != 1:
            raise RuntimeError(f"Expected one sample's action condition mask, got {len(condition_mask)} entries")
        condition_mask = condition_mask[0]
    if not torch.is_tensor(condition_mask) or int(condition_mask.numel()) != len(action_positions):
        condition_shape = tuple(condition_mask.shape) if torch.is_tensor(condition_mask) else None
        raise RuntimeError(
            f"Expected an action condition mask with {len(action_positions)} entries, got "
            f"{condition_shape} ({type(condition_mask).__name__})"
        )
    condition_flags = [bool(value) for value in condition_mask.detach().reshape(-1).cpu().tolist()]
    predicted_horizon = 0
    queries = []
    for action_index, (gen_position, is_condition) in enumerate(zip(action_positions, condition_flags, strict=True)):
        horizon = -1 if is_condition else predicted_horizon
        if not is_condition:
            predicted_horizon += 1
        queries.append(
            {
                "query_action_index": action_index,
                "gen_position": gen_position,
                "query_role": "condition" if is_condition else "predicted",
                "action_horizon": horizon,
            }
        )
    layout["action_token_shape"] = list(token_shape)
    layout["action_queries"] = queries
    return layout
