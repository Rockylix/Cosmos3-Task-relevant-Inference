# Edge C80/S104 action-weighted policy

This version makes the evaluated K184 C80/S104 policy the default of
`cosmos_framework.scripts.action_policy_server_robolab_version1`.
The former `version/version1` commit `d732f43` defaults to C64/S120; the recent
1200-rollout evaluation instead supplied C80/S104 explicitly through an external
runner. The first freeze captured that evaluated implementation. The cleanup release
now exposes one fixed C80/S104 policy and one attention profiling backend.

| Parameter | Fixed default |
| --- | --- |
| Core / Stable / total | 80 / 104 / 184 per future latent frame |
| Action horizon weights | `[1/6, 1/3, 1/3, 1/6]` |
| Core candidate blocks | B0 through B27, top 6 by quality |
| Block quality | mean attention mass × (1 − mean normalized entropy) |
| Stable score | weighted mean across future frames / (1 + CV) |
| Core and Stable overlap | none; Stable excludes the Core union |
| Profiling | step 0 conditional branch, all 28 blocks |
| Execution | one dense stack, seven sparse stacks; one mask stage |
| L0 | native conditioning semantics: fixed clean latent, recomputed Transformer projections |
| Sampler | native Edge UniPC |
| Sampling | CFG 3, 4 steps, shift 5 |
| Prompt | structured JSON |
| Policy RNG | seed 0, advancing request seeds; reset server per task |

The four action weights apply to the four action horizons aligned with each
future latent frame. Core layer weights and Stable layer aggregation are also
quality-weighted. The default policy is therefore both action-weighted and
Core-then-Stable. The production controller and selector accept no alternative
Core/Stable budgets or action weights; CFG 3 and four steps are required.

Use the canonical server from this checkout with the existing Cosmos environment:

```bash
python -m cosmos_framework.scripts.action_policy_server_robolab_version1 \
  --checkpoint-path /project/peilab/xiexinling/assets/Cosmos3-Edge-Policy-DROID \
  --guidance 3 --num-steps 4 --shift 5 --seed 0
```

Use the same local VAE overrides and cluster environment as the existing runner
when the checkpoint's configured VAE location is unavailable. Selection artifacts are disabled by default. Set `--version1-output-dir` only
when selection tensors are explicitly needed for inspection. There is no
per-request latency history or automatic aggregate-file rewrite.

The [machine-readable record](edge_c80_s104_action_weighted.json) records the
original evaluated controller hash, checkpoint metadata hash and policy settings.
Model weights are external assets and are not included in Git. The checkpoint
metadata hash is not a checksum of all weight shards.

Validation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -c /dev/null -q tests/test_robolab_version1.py
```

The focused CPU tests cover exact/disjoint selection, action weighting,
LSE/independent-softmax profile equivalence, invalid configuration rejection,
LSE error propagation, and native L0 updates/current-stack restoration. The
[cleanup audit](edge_policy_cleanup_audit.md) records the runtime checks.
Existing rollout results describe the pre-cleanup explicit-budget implementation;
regression probes do not constitute a new rollout evaluation.
