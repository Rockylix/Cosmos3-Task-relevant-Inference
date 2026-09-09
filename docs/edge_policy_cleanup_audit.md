# Edge policy cleanup audit

Scope: the C80/S104 action-weighted release and its complete call path from the
canonical RoboLab server through sampling, Transformer dispatch, profiling,
selection, sparse packing, and restoration. Baseline reference: `324574a`;
pre-cleanup freeze: `7787fbe`. Changes are confined to the release worktree.
The original evaluated checkout and the existing release tag are preserved.

## Conditioning behavior

L0 is the observed frame. Its clean latent stays fixed during denoising: the
baseline `OmniMoTModel._compute_velocity_for_tokens` masks conditioned velocity
to zero. `Cosmos3VFMNetwork._embed_vision_tokens` applies timestep embeddings
only at noisy-frame indexes. These baseline functions are unchanged.

The Transformer still receives L0 in every CFG stack. `PackedAttentionMoT`
projects all GEN tokens through `q_proj_moe_gen`, `k_proj_moe_gen` and
`v_proj_moe_gen` in each decoder layer. The sparse selected-position list includes
all L0 and action rows. Thus L0's hidden states and projections are recomputed;
its input latent is not denoised. This is native model behavior, not an optional
compatibility mode.

Native understanding/text memory (`MemoryState`, `gen_only`) is separate from
L0, which belongs to GEN. Removing that baseline machinery would change the
model and is outside this cleanup. The explicit L0 inclusion in sparse packing
is necessary to preserve the baseline conditioning behavior.

## Removed from the current policy

- Alternative Core/Stable budgets, profile-layer count and CV-penalty arguments.
- Alternative action-horizon weight arguments and CFG=1 execution branch.
- `live_l0`, persistent-L0, velocity-cache, Adaptive/Fill and depth-narrowing
  compatibility fields and strategy-name suffixes. There was no executable
  velocity-cache implementation or enable switch in this Edge release.
- Broad exception fallback from the LSE attention profiler to a second full
  softmax implementation. Errors now propagate from the supported backend.
  The full-softmax reference exists only in the CPU tests.
- Request-history latency collection, forced timing synchronizations, and
  unconditional per-request selection output.
- The production import dependency on two research attention-statistics scripts;
  the required token-layout helpers now live in the inference package.
- Unused `memory_gen_only` controller argument, full-position identity remapping,
  and the dense profiling stack's unused full-input clone.

## Retained because they are required

- Action weighting `[1/6, 1/3, 1/3, 1/6]`; quality-weighted layer aggregation;
  Core-then-Stable selection and exact disjoint C80/S104 masks.
- Step-0 conditional dense profiling and the seven subsequent sparse stacks.
- Full input-buffer restoration at the END of each sparse stack. This buffer is
  freshly created from that stack's input and discarded immediately afterward.
  It is not a cross-step velocity or K/V cache.
- Selected-index reuse within a request: the mask is intentionally fixed after
  profiling, so these are integer indexes, not model state.
- Shape/finiteness/budget validations and attention-hook cleanup in `finally`.
  Positional/keyword hook argument handling and LSE shape normalization are
  normal interfaces of the supported framework, not alternative strategies.
- Dense dispatch when no sparse controller is installed, native AR/text memory,
  and all other baseline model/sampler behavior.
- Optional selection artifacts for explicitly requested mask visualization.

## Validation

Six focused CPU tests exercise fixed-budget selection, action weighting against
an independent softmax reference, rejection of ablation arguments, preservation
of all conditioning/action positions, attention-kernel error propagation, and
per-layer L0 updates/current-stack-only restoration.

The GPU probe in `tests/probe_edge_core_stable.py` runs the original evaluated
implementation and this cleaned implementation on two fixed observations, with
identical zero proprioception and identical seeds. It compares action and vision
outputs and all selection masks, counts GEN Q/K/V projections at every layer,
and checks that the L0 input latent is identical across all eight stacks.
This is a numerical regression, not a rollout success-rate test.

The summary schema is now version 2. Legacy ablation keyword arguments and
compatibility flags are intentionally absent; existing external benchmark wrappers
that assert the old schema should continue to use the preserved earlier tag.
The canonical server in this checkout is the entry point for the fixed policy.

Results at the cleanup commit:

- Six focused CPU tests passed; Ruff and whitespace checks passed.
- All returned plan tensors and masks were bitwise identical to `7787fbe` for
  20 independent random profile sets.
- GPU comparison is queued in tmux `edge_policy_cleanup_audit`. All eight user
  Slurm slots were occupied when queued. GPU output equality has not yet been
  established. The queued test writes its result to
  `/project/peilab/xiexinling/reports/edge_policy_cleanup_audit_20260909/result.json`.
