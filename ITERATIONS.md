# Iteration Log

| Iter | Change | Status |
|---|---|---|
| B0 | ASI legacy, no compile/Graph | complete: median 578.036 ms |
| B1 | Only batched eight-frame scoring, all legacy setup and checks retained | complete: median 569.608 ms, 1.0148x vs B0; two-seed bitwise gate passed |

Original ASI HEAD: 87caf7454e203cb25ff768ac0a86d56d2dea10d6. First turn stopped after B1. User subsequently approved testing all existing mechanisms, then one target optimization.

Report: docs/asi_system_b1_batch_scoring_cn.md. nsys score kernels 3332 -> 2016; score memcpy/sync remains 784/784. Independent review passed. Stop for discussion before geometry-cache B2.

## Existing eager optimizations, cumulative B0-B5

2026-09-18: completed 20 alternating repetitions per mode; all two-seed Core/Stable/mask/action/latent exact vs legacy. B2-B5 scores differ up to 4.10e-8, no selected-token change. CPU 8/8 and independent review passed. Raw results: experiments/existing_system_banana_c3_v1/timing/results.json. nsys B2-B5 capture/parse complete. B2 includes contiguous slice/reshape scoring preparation, not only a Python layout dictionary cache. No compile/Graph/velocity cache.

## P0 / B6 - main attention LSE reuse

- Hypothesis: remove redundant 32-query attention and K/V concatenations by returning main GEN LSE.
- Strict paired gate: two seeds, all discrete selections and action/latent exact. Main attention O return_lse on/off exact; main vs old32Q LSE differs <=9.54e-6; scores <=6.34e-8 vs legacy.
- Timing: same-round B0 median 578.907572 ms; B5 548.663185 ms; B6 548.271810 ms. B6/B5 speedup 1.000714x, too small to claim a robust chunk benefit. B6/B0 1.055877x.
- CPU 18/18 tests pass; reviewer passed main path and safety guards corrected. No new mechanism added beyond P0.
- nsys paired B5/B6 complete: score kernels 616->504, score GPU 4.332->3.405 ms; extra LSE attention 28->0. Copies/sync unchanged. Reports main_lse_b5.nsys-rep / main_lse_b6.nsys-rep. Report docs/asi_system_existing_and_main_lse_cn.md; paused before P1.

## C0-C4 - cumulative compile and CUDA Graph evaluation

2026-09-18: formal v2 completed, five warmups and twenty alternating profiler-free generation samples per mode. Same inputs, noise, shift5/4 steps/guidance3; padding disabled throughout. Median ms: current B6 eager 553.527849; sparse-only decoder compile 486.494780; all decoder compile 466.034405; decoder+heads compile 465.095224; plus Graph 464.335563. Reference here is B6 ASI, not Dense or legacy ASI.

Two-seed same-mode repeat, seed switch, return-to-original-seed and all twenty repeated-output checks passed. Compiled-vs-eager outputs differ; compiled first dense/profile also changes some execution-mask entries. These are quantified differences, not a production-equivalence pass. CPU tests 29/29 passed. Formal data: experiments/compile_stages_banana_c3_v2/timing/results.json. Separate warmed per-mode nsys capture and independent review pending. No production defaults or Git history changed; no further optimization before user discussion.

Completion: all five nsys captures and SQLite audits finished. Eager/compile-only modes have zero GraphLaunch; all_graph has 264 actual launches, 224 decoder ranges (including first conditional dense/profile 28) and 40 outside layers. Kernel sum eager/all_compile/all_graph = 488.860/403.485/403.453 ms. Graph median marginal gain 0.760 ms (0.163%), not robust evidence of material extra speedup. Independent review found no blocker; report documents inherited sparse_compile summary label error, repeat gate boundaries and non-cold-cache first-call timings. No measured source or raw artifacts overwritten. Report: docs/asi_compile_stages_cn.md. Stop after this evaluation for user discussion.

## User-requested source snapshot — 2026-09-19

Commit the opt-in B1–B6 controllers, compiler evaluation, analysis tools, CPU tests and reports on `experiment/asi-system-ablation`; no production default changes, merge or push. Archive the Method/system design document here with branch-local evidence links. Independent pre-commit review found one reproducibility issue: the Dense/ASI profile tool required an ignored historical output. Make that comparison explicit and optional via `--reference-eager`, without changing inference or the mandatory current-run warmup/trace equality gate. Historical measured files are unchanged and remain ignored. New multi-strategy measurement awaits confirmation of compiler/comparison configuration; historical tables are not new results.

## Confirmed multi-strategy comparison — 2026-09-19

User confirmed Dense eager and ASI B6/ToCa/WorldCache/C3ache compile+Graph, C3ache period2 amortization. Snapshot commit `186a3e6`; new results `experiments/multistrategy_b6_graph_banana_c3_v1`. All workers exited 0; 5 warmups and 20 matched measurements, C3 20 refresh/hit cycles. Actual GEN3093, sparse1845, UND158/19; padding disabled. Median ms/speedup against in-process alternating Dense: ASI467.213/1.841x, ToCa687.901/1.266x, WorldCache570.643/1.519x, C3amortized576.390/1.511x. All inactive-strategy Dense outputs exact to pristine Dense source; repeat outputs tolerance1e-5, ASI repeat mask exact, changed-seed gates passed. Runtime GraphLaunch264/264/198/C3 264+152. Compiled-vs-eager masks/outputs differ as documented. Logical block matmul FLOPs60.732/65.681/75.142/75.142%, not hardware counts. Independent numerical/result/source review passed; 4 new CPU FLOPs tests passed. No new optimization, closed-loop test, production-default change, merge or push.
