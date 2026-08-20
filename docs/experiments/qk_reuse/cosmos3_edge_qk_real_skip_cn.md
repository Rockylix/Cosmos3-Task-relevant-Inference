# Cosmos3 Edge：基于 Q/K profile 的真实 projection 跳算实验

## 1. 实验语义

本实验不再采用“完整计算 Q/K 后覆盖目标帧”的伪稀疏方式。在 profile 选出的 9 个位置中，目标 future latent 的 hidden token 不进入以下计算：

```text
Q Linear / K Linear
QK RMSNorm
RoPE
```

仅对保留 token 执行上述路径，随后使用源帧的 post-RoPE Q/K 恢复目标帧位置，再进入正常的完整 attention。

```text
step 0 / B27: L7 -> L8

step 2 / B1: L1 -> L2, L3 -> L4, L5 -> L6, L7 -> L8
step 3 / B1: L1 -> L2, L3 -> L4, L5 -> L6, L7 -> L8
```

conditional/unconditional 使用相同策略。L0、所有 action token、text/condition token 和所有源 future latent 始终保留。V projection 和 attention matmul 仍然全量计算，因为当前 Q/K profile 没有证明 V 或 attention output 可以复用。

controller 默认关闭。没有安装 controller 时，原始 dense 路径及执行顺序不变。

## 2. 实际计算量

实时 GEN layout：

```text
3093 tokens = 9 × 340 video tokens + 33 action tokens
```

每个 generation 的选中路径共包含 6 个 CFG call：

| step/block | 每个 branch 的完整 token | 实际计算 token | 跳过 token |
|---|---:|---:|---:|
| step 0 / B27 | 3093 | 2753 | 340 |
| step 2 / B1 | 3093 | 1733 | 1360 |
| step 3 / B1 | 3093 | 1733 | 1360 |

两个 CFG branch 合计：

- 选中 Q/K 路径完整 token：18558
- 实际计算 token：12438
- 实际跳过 token：6120
- 选中路径计算比例：67.02%，减少 32.98%

但完整 DiT 包含 `4 steps × 2 CFG × 28 blocks = 224` 个 block forward。换算到全部 DiT Q/K projection：

- Q/K projection token 比例：99.1167%
- 全局只减少：0.8833%

因此不能把局部 32.98% 的减少解释为整个 DiT 的 32.98% 加速。

## 3. 配对正确性验证

每个 chunk 使用相同输入、state、conditioning image、seed、initial noise、CFG、shift 和 UniPC 参数运行：

1. Baseline：原始 dense 路径。
2. Post-Q/K Copy：上一轮完整计算后覆盖，用作误差参照。
3. Dense-Control：通过新 projection hook 计算全部 token，不做跳算。
4. Real-Skip：目标 token 不进入 Q/K projection/Norm/RoPE。

Dense-Control 与 Baseline 的 action、vision、future vision 均逐元素一致，MSE、relative-L2 和 max-abs 全部为 0。这验证了新增 hook 在不跳 token 时不会改变模型输出。

Real-Skip 每次触发 6 个选中 call、18 个 frame-pair reuse，实际跳过 6120 token-row。所有结果无 NaN/Inf。

## 4. Baseline 与 Real-Skip 输出误差

Action 使用 RoboLab 最终返回的 `[32,8]` action chunk 语义。

| chunk | Action MSE | MAE | relative L2 | cosine | max absolute error |
|---|---:|---:|---:|---:|---:|
| c3 | 2.0552e-4 | 0.009926 | 0.010723 | 0.999943 | 0.065996 |
| c5 | 1.1693e-4 | 0.007528 | 0.009507 | 0.999957 | 0.047734 |

逐 horizon 最大 MSE：

| chunk | horizon | MSE | relative L2 | cosine | max abs |
|---|---:|---:|---:|---:|---:|
| c3 | 12 | 7.10e-4 | 0.01972 | 0.999837 | 0.065996 |
| c5 | 3 | 3.45e-4 | 0.01587 | 0.999936 | 0.047734 |

32 个 horizon 的 relative-L2 均值：c3 为 `0.00996`，c5 为 `0.00907`。

Final vision latent：

| chunk | 范围 | relative L2 | cosine | max absolute error |
|---|---|---:|---:|---:|
| c3 | 全部 L0..L8 | 0.09645 | 0.99536 | 2.1572 |
| c3 | future L1..L8 | 0.09914 | 0.99511 | 2.1572 |
| c5 | 全部 L0..L8 | 0.07075 | 0.99750 | 1.1709 |
| c5 | future L1..L8 | 0.07281 | 0.99736 | 1.1709 |

真实跳算的 action 误差仍约为 1% relative-L2，与上一轮 Post-Q/K Copy 同量级，但两者不是逐元素一致：Copy 与 Real-Skip 的 action relative-L2 为 c3 `0.00822`、c5 `0.00916`。原因是 dense GEMM 与缩短后的 sparse GEMM 具有不同的矩阵 M 维，可能选择不同 CUDA kernel；即使保留 token 的数学表达式相同，BF16 舍入也不保证 bitwise 相同，误差还会经过后续 diffusion steps 放大。

## 5. CUDA timing

### 5.1 完整 generation chunk

双侧预热后 Baseline/Real-Skip 交替各运行 5 次：

| chunk | mode | mean | median | P90 |
|---|---|---:|---:|---:|
| c3 | Baseline | 852.47 ms | 852.59 ms | 853.52 ms |
| c3 | Real-Skip | 857.95 ms | 858.56 ms | 859.20 ms |
| c5 | Baseline | 857.73 ms | 857.81 ms | 858.93 ms |
| c5 | Real-Skip | 864.64 ms | 864.52 ms | 865.43 ms |

- c3：Real-Skip 慢 0.64%，增加 5.49 ms/chunk。
- c5：Real-Skip 慢 0.81%，增加 6.91 ms/chunk。

### 5.2 六个选中 Q/K 路径的总 CUDA 时间

| chunk | mode | mean | median | P90 |
|---|---|---:|---:|---:|
| c3 | Dense-Control | 3.394 ms | 3.395 ms | 3.399 ms |
| c3 | Real-Skip | 5.915 ms | 5.920 ms | 5.957 ms |
| c5 | Dense-Control | 3.423 ms | 3.423 ms | 3.428 ms |
| c5 | Real-Skip | 5.956 ms | 5.939 ms | 5.996 ms |

当前 PyTorch 实现虽然减少了 Linear/Norm/RoPE token 数，但 gather、`new_empty`、四个完整 Q/K tensor 的恢复以及多次 `index_copy` 开销超过了省下的 GEMM 时间，导致选中路径本身慢约 74%。

## 6. 结论

1. 本轮是真实 Q/K projection 跳算：目标 token 确实没有进入 Q/K Linear、Norm 和 RoPE。
2. profile 选出的 9 个点使最终 action 保持高 cosine，但 action 仍有约 1% relative-L2 和最高约 0.066 的绝对误差，不能声明 action 不变。
3. 只在 3 个 step/block 位置跳算，全部 DiT Q/K projection token 仅减少 0.883%，理论收益很小。
4. 当前 unfused gather/scatter 实现没有加速，反而使 chunk 慢 0.64%–0.81%。
5. 若要获得实际加速，需要扩大安全候选覆盖率，并将 token selection、Q/K projection 和恢复做成 fused kernel；或者进一步验证 V/O/attention query output 是否可复用。仅增加更多 Python 级 index-copy 不值得。
6. 本轮仍是冻结 chunk 的离线端到端输出验证，没有运行完整 RoboLab episode，因此没有任务成功率和任务完成时间结论。

## 7. 结果与复现

结果目录：

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/qk_optimization/real_skip/
  edge_qk_real_skip_BananaOnPlateTask_c3_c5_shift5_v1/
```

主要文件：

```text
metrics.json
timing.json
timing_samples.csv
qk_projection_timing_samples.csv
qk_projection_timing_summary.csv
real_skip_trace.csv
action_horizon_metrics.csv
baseline/
copy/
real_skip/
```

复现命令：

```bash
cd /root/robolab/cosmos-framework-edge

HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 \
PYTHONPATH=/root/robolab/cosmos-framework-edge:/root/robolab/cosmos-edge-overlay/tools \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/run_edge_qk_real_skip.py \
  --output-root /root/robolab/cosmos-framework-edge/experiments/preliminary/qk_optimization/real_skip/<new_run_name> \
  --timing-repeats 5
```
