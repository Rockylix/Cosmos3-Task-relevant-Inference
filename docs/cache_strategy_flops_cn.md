# Edge 缓存策略 FLOPs 对比（Baseline = 100%）

## 结果

对应 `experiments/cache_compile_adapt_final_v1/comparison.json` 的同输入单 chunk 测速。
本次仅离线计算 FLOPs，没有重新运行推理或修改模型。

| 策略 | 剩余 FLOPs | 减少 FLOPs | 等吞吐理论加速 | 实测加速 | 单 chunk median |
|---|---:|---:|---:|---:|---:|
| Dense eager | 100.00% | 0.00% | 1.000× | 1.000× | 配对基线 857.26–871.50 ms |
| ASI Core80+Stable104 | 60.76% | 39.24% | 1.646× | 1.832× | 467.84 ms |
| ToCa D/C/D/C，r=0.25 | 65.68% | 34.32% | 1.523× | 1.249× | 694.43 ms |
| WorldCache D/D/D/C | 75.14% | 24.86% | 1.331× | 1.498× | 577.89 ms |
| C3ache period=2，刷新/复用均摊 | 75.14% | 24.86% | 1.331× | 1.488× | 585.51 ms |

实测速度分别使用同进程、同输入的配对 Dense eager 为分母；四策略开启 compile + CUDA Graph。
所以实测加速同时包含减少运算和执行效率改善，不能仅由 FLOPs 比例预测。
ToCa 即使 FLOPs 少于 WorldCache，也可能因为评分、排序、索引与较小 GEMM 的开销而更慢；该表本身不做新的瓶颈定位。

## 统计边界

统计 28 个 Transformer block 的有效 token 主矩阵乘法：Q/K/V/O 投影、QK/AV、两层 MLP。
计入每个 chunk 两个 CFG 分支各一次 UND prefill，以及 ASI 首轮 28 个 block 的额外评分矩阵乘法。
一次乘加计 2 FLOPs。UND causal attention 使用有效下三角 pair 数。

不统计 VAE、输入/输出头、时间嵌入、UniPC、归一化、激活、softmax、排序、索引与缓存搬运。
不计编译 padding、kernel tile 冗余和重算。因此这是 **逻辑 Transformer 主矩阵乘法估算**，不是全 pipeline FLOPs，也不是 profiler 测出的 GPU 指令 FLOPs。
WorldCache/C3ache 的输出头执行次数不同，在此边界外；两者相同比例不表示所有运算完全相同。
绝对值约为 Dense 88.302、ASI 53.651、ToCa 57.997、WorldCache/C3ache 66.352 TFLOP/chunk（运算量，不是 TFLOP/s）。

## 维度与公式

本地权重 header 核对全部 28 层：hidden `D=2048`，query width `Q=2048`，KV width `K=1024`，MLP intermediate `I=9216`。
当前 Edge 是两层 Linear + ReLU²，**不是三矩阵 SwiGLU**。
GEN 全量 `N=9×340+33=3093`；保护 L0/action `P=340+33=373`。
ASI 每帧保留 184，稀疏 GEN `Ns=8×184+373=1845`。
本次条件 caption 经 chat template 是 156 token，无条件 17；packing 加 EOS/BOV 后 UND 为 `Uc=158, Uu=19`。
这些长度针对本次输入，不是所有任务固定长度。

单层 GEN 全量主矩阵乘法：

```
F(n,u) = 2nD(2Q+2K) + 4nDI + 4n(n+u)Q
```

单层 UND prefill：

```
U(u) = 2uD(2Q+2K) + 4uDI + 4[u(u+1)/2]Q
Tund = 28[U(Uc)+U(Uu)]
Pair = 28[F(N,Uc)+F(N,Uu)]
Dense = Tund + 4Pair
```

ASI：一次 conditional dense + 七次 sparse，包含首轮额外评分 attention 与逐帧对齐 QK。

```
Profile = 28[4×32×(N+Uc)Q + 2×32×340Q]
ASI = Tund + 28[F(N,Uc)+3F(Ns,Uc)+4F(Ns,Uu)] + Profile
```

ToCa：两个 dense step、两个 cache step，两分支独立。缓存步只为保护 token 重算 Q/O，全部 GEN 重算 K/V，MLP 重算保护 token 和 fresh future token。

```
fresh(l) = floor[0.25(1.5-l/27)×2720], l=0..27
C(u,l) = 4PDQ + 4NDK + 4P(N+u)Q + 4[P+fresh(l)]DI
ToCa = Tund + 2Pair + 2 sum_l[C(Uc,l)+C(Uu,l)]
```

fresh 从 B0 的 1020 降至 B27 的 340；不是每层固定 25%。joint attention 同时提供评分，不额外计一次完整 QK。

```
WorldCache = Tund + 3Pair
C3ache = [(Tund+4Pair) + (Tund+2Pair)]/2 = Tund+3Pair
normalized FLOPs = strategy / Dense × 100%
equal-throughput speedup = Dense / strategy
```

WorldCache/C3ache 略高于 75%，是因为每个 chunk 的 UND prefill 没有按 25% 减少。
C3ache 必须按 period=2 均摊，不能只报告复用 chunk。

## 复现与证据

```bash
cd /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted
.venv/bin/python tools/estimate_cache_strategy_flops.py
```

输出：`experiments/cache_compile_adapt_final_v1/flops_comparison.json` 和 `flops_comparison.csv`。
脚本核对 safetensors 全部层的矩阵形状、ASI dense/sparse 次数和预算、ToCa 配置与布局；不加载 GPU 模型。

计算路径：

- 本仓库 `cosmos_framework/inference/edge_core_stable_fast.py` 的 `action_aligned_future_profiles`。
- ToCa adapter worktree `cosmos_framework/inference/toca_compiled.py` 的 `cached_forward`，以及 `toca_future.py` 的 `fresh_count`。
- 本仓库 `cosmos_framework/data/generator/sequence_packing/modalities.py` 的 `add_special_tokens/compute_text_split_length`。
- 原始测速文件每个策略的 `results.json` 中 `last_controller` 给出实际调度；`comparison.json` 给出配对时间。

此表不增加任何闭环成功率结论。
