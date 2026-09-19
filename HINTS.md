# ASI system ablation constraints

- B0/B1 complete. User now approved cumulative testing of previously implemented eager optimizations without pausing between each, then one target-system optimization round; report before further target iterations.
- No compile/Graph, geometry/layout cache, sparse metadata cache, callback replacement or velocity cache in B1.
- Use project Edge venv; no downloads; no RoboLab closed-loop run requested.
- Entry: system-analysis; custom benchmark tools/benchmark_asi_batch_only.py.
- NVTX filter: asi.chunk.generate; warmed synchronized generation only, no load/warmup/decode/RPC.
- Formal timing is a separate profiler-free run, overriding skill's combined-run template.
- No automatic git commit/push/merge/reset, no git add -A. Original worktrees unchanged.
- Preserve actual backend LSE units and FP32 scoring; skill's generic base-2/0.02 threshold is not an override.
- No model-output caching across requests. Fresh deep-copied input and same reset RNG for paired runs.
- Save nsys reports and source hashes; keep all experiment artifacts ignored by Git.
- Review with independent skill reviewer, no concurrent GPU workloads.

## Compile evaluation authorized

User approved current B6 eager -> sparse-only decoder compile -> full decoder compile -> encode/decode-head compile -> CUDA Graph evaluation. Quantify BF16/selection/output differences rather than applying the previous eager bitwise-equivalence gate across modes. Same-mode repeated outputs and masks must remain stable; changing seeds must not replay stale output. Graph is explicitly authorized despite the generic skill anti-Graph guidance. No production default/history changes. Keep profiler-free paired timing and separate warmed NVTX captures; cold compile/capture latency separately recorded.
