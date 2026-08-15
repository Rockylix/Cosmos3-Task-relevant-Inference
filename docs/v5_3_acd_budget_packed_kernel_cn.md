# V5.3 A/C/D 预算缩减与 packed 热路径优化

## 实验边界

V5.3 不改变 V5.2 A/C/D 的 mask 语义，只做两件事：

1. 将 K80 预算下探到 K75/K70；
2. 将实验型 Python 热路径改为 group-packed 执行，减少同步、索引重建和 side-buffer 回写。

本轮只用稳定单 chunk 时间做性能判断，不使用首请求、RPC、任务 wall time 或单次冷启动时间计算加速比。

## 稀疏的真实边界

每个 GEN hidden state 的完整 token layout 为：

\[
3093 = L0(340) + L1\ldots L8(8\times340) + q0\ldots q32(33)
\]

- UND/文本条件不稀疏；
- condition latent `L0` 和 33 个 action GEN tokens 始终保留；
- 只在 token 轴上筛选 `L1..L8` 的 spatial tokens；hidden dimension 2048 不剪枝；
- 缩短后的整个 hidden sequence 真实经过 Transformer block 的 Attention 和 SwiGLU MLP，不是计算完后才 mask；
- step0 全量 profile，后续 step 的 B0–B3 全量，B4–B27 按 G1/G2/G3 执行真实短序列；
- 每个 group 只 pack 一次，group 内的 8 个 block 都流通同一份稀疏 hidden state；
- 跳过的 future tokens 由 side buffer 保留，group 边界和最终恢复时合并成完整 GEN sequence；
- 后续 UniPC 积分中，背景 velocity 继续使用 step0 cache，ROI 使用当前 step velocity。

## 热路径优化

V5.3 相对 V5.2 reference 做了：

- step0 的 48 份 action-attention profile 在 GPU 上累积，最后一次批量 D2H；
- G1/G2/G3 的 original/local positions 每个 request 只构建一次；
- 用已排序 nested mask 的 `searchsorted` 替代通用 `isin` 搜索；
- side buffer 只在 G1→G2、G2→G3 和最终 restore 回写，不再每个 sparse block 回写；
- 性能测试关闭逐 block finite reduction 和分析型 trace clone，仅验证终端输出；
- 保留当前 FlashAttention 和 Transformer block kernel，没有替换 attention 实现。

这是 runtime/kernel-path 优化，不是新的 Triton/CUDA fused kernel。

## 计时协议

- GPU：RTX 4090；
- shift=5，4 个 UniPC denoise steps；
- BananaInBowlTask，chunk 3，seed `579362556`；
- compile 和 CUDA graphs 关闭；
- 同一已加载模型、同一 `data_batch` 和 seed；
- 每个 mode 3 轮预热，20 轮随机交替采样；
- `torch.cuda.synchronize()` 包围 `generate_samples_from_batch`；
- 主结果报告 median/P90。

## K80 reference 与 optimized 等价性

| Arm | Reference median (s) | Optimized median (s) | Runtime gain | Action max-abs | Vision max-abs |
|---|---:|---:|---:|---:|---:|
| A | 0.727812 | 0.684441 | 1.063x | 0 | 0 |
| C | 0.727215 | 0.684670 | 1.062x | 0 | 0 |
| D | 0.727560 | 0.684359 | 1.063x | 0 | 0 |

参考与优化路径在 action 和 vision tensor 上都是逐元素一致。

## 稳定单 chunk 时间

| Budget | G1/G2/G3 | Arm | Median (s) | P90 (s) | Speedup vs same-run Dense |
|---|---|---|---:|---:|---:|
| K80 | 192/160/144 | A | 0.684441 | 0.686225 | 1.258x |
| K80 | 192/160/144 | C | 0.684670 | 0.686431 | 1.258x |
| K80 | 192/160/144 | D | 0.684359 | 0.686284 | 1.258x |
| K75 | 180/150/135 | A | 0.676099 | 0.679575 | 1.269x |
| K75 | 180/150/135 | C | 0.676641 | 0.679647 | 1.268x |
| K75 | 180/150/135 | D | 0.677726 | 0.679725 | 1.266x |
| K70 | 168/140/126 | A | 0.664408 | 0.669185 | 1.299x |
| K70 | 168/140/126 | C | 0.664417 | 0.665667 | 1.299x |
| K70 | 168/140/126 | D | 0.664347 | 0.665559 | 1.299x |

K80/K75/K70 各自同轮 Dense median 分别为 `0.861155/0.857810/0.862989 s`。A/C/D 的 token shape 相同，因此性能应当接近；它们的区别主要是保留哪些 token，而不是保留多少 token。

## Token 计数

| Budget | Future K G1/G2/G3 | Sparse-group GEN retention G1/G2/G3 | 全 224 block-call GEN retention |
|---|---|---|---:|
| K80 | 192/160/144 | 61.72% / 53.44% / 49.30% | 70.96% |
| K75 | 180/150/135 | 58.62% / 50.86% / 46.98% | 69.24% |
| K70 | 168/140/126 | 55.51% / 48.27% / 44.65% | 67.52% |

全 block-call 统计包含 step0 全量与后续 B0–B3 全量路径，所以不会等于 sparse group 内的保留率。

## 固定 chunk 输出误差

| Budget | Arm | Action MSE | Action rel-L2 | Action cos | Vision rel-L2 | Vision cos |
|---|---|---:|---:|---:|---:|---:|
| K80 | A | 0.003143 | 0.04033 | 0.999229 | 0.35326 | 0.939108 |
| K80 | C | 0.006996 | 0.06017 | 0.998204 | 0.36346 | 0.935026 |
| K80 | D | 0.007047 | 0.06039 | 0.998252 | 0.35992 | 0.936279 |
| K75 | A | 0.004164 | 0.04642 | 0.999010 | 0.36171 | 0.936283 |
| K75 | C | 0.006356 | 0.05735 | 0.998405 | 0.37381 | 0.931497 |
| K75 | D | 0.008504 | 0.06634 | 0.998259 | 0.36864 | 0.933228 |
| K70 | A | 0.005763 | 0.05461 | 0.998581 | 0.37143 | 0.933132 |
| K70 | C | 0.006015 | 0.05579 | 0.998442 | 0.38011 | 0.929454 |
| K70 | D | 0.004482 | 0.04816 | 0.998840 | 0.37899 | 0.929852 |

Action cosine 在三档预算下都很高，但 vision cosine 随预算下降而持续恶化。因此 K70 只是性能上界，不能仅凭一个 chunk 的 action cosine 宣称可用。K80 仍是进入下一轮闭环的首选。

## NSys / NVTX 核对

K80/C 为每个 step、CFG branch、block 以及 group pack/commit/restore 添加了 NVTX。按 NVTX GPU projection 汇总：

| Range group | Instances | GPU projected time (ms) | Share of listed V5.3 ranges |
|---|---:|---:|---:|
| Step0 dense B0–B27 | 56 | 258.253 | 43.47% |
| Step1–3 dense B0–B3 | 24 | 81.940 | 13.79% |
| Step1–3 sparse B4–B27 | 144 | 245.953 | 41.40% |
| Group pack | 18 | 7.619 | 1.28% |
| Restore | 6 | 0.174 | 0.03% |
| Group commit | 12 | 0.118 | 0.02% |

已列 V5.3 ranges 合计约 `594.1 ms`。这是 GPU projection 时间，不与 Python wall time 或 CUDA kernel 全局占比混用。

整个 trace 的 CUDA kernel summary 中，三个最大 BF16 GEMM kernel family 合计约 `54.6%`，FlashAttention forward 约 `17.3%`。这支持“剩余路径仍由 Linear/GEMM 和 Attention 主导”，不单独证明 Roofline 意义上的 compute-bound。

结论：当前 pack/restore 已不是主要瓶颈。下一阶段若要明显超过 `1.30x`，需要减少 step0 profile 的全量 block 数，或减少后续 B0–B3 的全量路径；只继续优化 index/restore kernel 收益会很小。

NSys 产物：

```text
/root/robolab/experiments/preliminary/sparsity/velocity_cache/
ac_budget_packed_kernel_shift5_BananaInBowlTask_c3_k80_nsys_v1/
  c_opt_k80.nsys-rep
  stats_nvtx_gpu_proj_sum.csv
  stats_cuda_gpu_kern_sum.csv
```

## 复现命令

K80 完整 reference/optimized 等价性与稳定计时：

```bash
cd /root/robolab/worktrees/ac-budget-packed-kernel-v5-3
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 PYTHONPATH=. \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  tools/benchmark_robolab_v5_3_stable_chunk.py \
  --output-root /root/robolab/experiments/preliminary/sparsity/velocity_cache/\
ac_budget_packed_kernel_shift5_BananaInBowlTask_c3_k80_v2 \
  --group-token-budgets 192 160 144 \
  --modes dense a_ref a_opt c_ref c_opt d_ref d_opt \
  --warmup-rounds 3 --measure-rounds 20
```

K75/K70 将 `--group-token-budgets` 分别改为 `180 150 135` 和 `168 140 126`，并只测：

```text
--modes dense a_opt c_opt d_opt
```

CPU 回归：

```bash
cd /root/robolab/worktrees/ac-budget-packed-kernel-v5-3
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python -m pytest --capture=no \
  tests/test_robolab_v5_3_acd_packed_kernel_velocity_cache.py \
  tests/test_robolab_v5_2_motion_core_stable_adaptive.py \
  tests/test_robolab_grouped_temporal_closed_roi_velocity_cache.py
```

