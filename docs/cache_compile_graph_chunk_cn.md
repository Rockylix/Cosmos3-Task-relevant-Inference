# 四策略 Compile + CUDA Graph 单 chunk 测试（2026-09-13）

## 结果与边界

已完成 ASI、C3ache；**不是四个策略全部完成**。ToCa、WorldCache 的当前实现拒绝该模式，
未改缓存调度、评分公式、token 预算或生产源码，没有绕过保护检查。
兼容适配是否继续已交给用户确认。

RTX 4090、项目 Edge venv、Torch 2.10.0+cu130。固定 BananaInBowlTask 第3个 request
录制输入、seed=1097657232、4步 UniPC、shift=5、guidance=3。
每个模式5次预热、15次测量；各分支独立进程，每个进程内同模型与 Dense eager/Dense compile+Graph
交替计时。速度比使用同进程配对 Dense，不能把历史时间混作本轮测量。

| 策略 | Mean/s | Median/s | P90/s | 相对配对 Dense compile+Graph | 相对配对 Dense eager |
|---|---:|---:|---:|---:|---:|
| Dense (ASI 配对) | 0.737899 | 0.738457 | 0.741073 | 1.000× | 1.170× |
| ASI Core80+Stable104 | 0.471990 | 0.472423 | 0.474305 | 1.563× | 1.828× |
| Dense (C3ache 配对) | 0.740995 | 0.741084 | 0.743201 | 1.000× | 1.170× |
| C3ache 刷新 | 0.745012 | 0.745530 | 0.747252 | 0.994× | 1.163× |
| C3ache 命中 | 0.419069 | 0.418727 | 0.420436 | 1.770× | 2.071× |
| C3ache period=2 均摊 | 0.582041 | 0.582176 | 0.583837 | 1.273× | 1.489× |
| ToCa D/C/D/C r=0.25 | 不兼容 | — | — | — | — |
| WorldCache D/D/D/C | 不兼容 | — | — | — | — |

## 计时范围

CUDA 同步后计时 `generate_samples_from_batch`，包含 controller context、action profile、
token selection、缓存读写与完整去噪所需开销；输入 CPU deepcopy、RNG 初始化、controller 对象构造、
计时外 summary/有限性检查、CPU输出复制、future VAE decode、RPC、Isaac 不计入。
原生生成内部的 condition VAE encode 等准备工作仍包含在内。
模型加载、首次编译/捕获不计入稳定时间。profiler 审计单独运行，不计入上表。

C3ache：每2个chunk刷新，刷新为 D/D/D/D，命中为 C/C/D/D。
每个周期均摊时间 `(T_refresh + T_hit)/2`，再对15个周期统计 mean/median/P90。
这是同一观测回放、递增chunk id的性能测量，不是实际不同观测的跨chunk精度实验。

## CUDA Graph 的实际执行

采用 native decoder-layer/encode-decode-head `torch.compile(fullgraph=True, mode="reduce-overhead")`，
不是把全部 UniPC/CFG/Python控制器捕获成一个全局图。保持原 `compile_dynamic=True`。
逐chunk的mask选择、缓存调度等Python控制流仍在图外。

- ASI：dense_graph = 264次 cudaGraphLaunch, strategy_graph = 264次 cudaGraphLaunch。
- C3ache：dense_graph = 264次 cudaGraphLaunch, refresh_graph = 264次 cudaGraphLaunch, hit_graph = 152次 cudaGraphLaunch。

ASI：224层调用+40编码/输出头=264次；C3ache刷新同样264次，
命中为112层调用+40编码/输出头=152次，实测缓存统计为4次跳过整栈、4次dense。
最终action/vision有限、同输入重复结果稳定、改变seed输出会变化，排除了固定旧输出重放。
Compile会改变BF16运算顺序；上述检查不等于逐位数值等价，也不保证任务成功率不变。
ASI的优化版eager与compiled输出差异、C3ache刷新eager与compiled差异详见各自results.json。

## 两个阻塞项

- ToCa：实际报 `ToCa eager reference does not support padded GEN rows`。
  缓存步还有 `CUDA graphs are disabled for this reference experiment` 检查。
  动态hook捕获attention/MLP输出与score、替换attention dispatch、手写缓存步都需要专门适配；
  不能仅移除检查后宣称已编译。
- WorldCache：实际报 `WorldCache first version requires eager inference, no compile/CUDA graphs`。
  原实现通过llm2vae forward hook获取projection以保存缓存，需要核验编译图输出与历史缓存
  生命周期；本轮没有取消此保护。

## 文件

统一脚本：`tools/benchmark_cache_compile_graph.py`。
原始结果：`experiments/cache_compile_graph_chunk_v2/asi/results.json`、
`experiments/cache_compile_graph_chunk_v1/c3ache/results.json`。
ASI以最终统一配置脚本重跑v2；v1 ASI作为首次结果保留，不混入统计。
错误证据：v1下 `toca/failure.json`、`worldcache/failure.json`及各自log。
总表：`experiments/cache_compile_graph_chunk_v1/comparison.json`。

```bash
cd /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted
.venv/bin/python tools/benchmark_cache_compile_graph.py \
  --output experiments/cache_compile_graph_chunk_repeat \
  --modes asi worldcache c3ache toca --warmups 5 --repeats 15
```

测试脚本会继续测后续分支，即使某个分支失败；必须逐分支检查results.json或failure.json，
不能用主进程退出码推断四个策略全通过。当前分支拒绝编译的行为应在复现时保持。
本轮没有commit/merge/push，没有修改任何生产算法文件。
