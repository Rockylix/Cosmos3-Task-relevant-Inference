# Cosmos3 Edge Attention 真实 Group 输出实验

## 1. 实验目标

本实验只统计三个指标：

\[
\|O_g\|_2,
\qquad
\rho_g=\frac{\|O_g\|_2}{\sum_{j\in G_g}A_{ij}\|V_j\|_2+\epsilon},
\qquad
\alpha_g=\frac{\langle O_g,O\rangle}{\|O\|_2^2+\epsilon}.
\]

其中：

\[
O_g(i,h)=\sum_{j\in G_g}A_{ijh}V_{j,h_v},
\qquad
O(i,h)=\sum_gO_g(i,h).
\]

三个指标分别回答：

| 指标 | 含义 |
|---|---|
| \(\|O_g\|\) | group 真正形成的输出有多大 |
| \(\rho_g\) | `A×||V||` 有多少转化为真实向量，group 内部是否抵消 |
| \(\alpha_g\) | group 沿最终 per-head attention 输出方向是推动还是抵消 |

这里的 \(O_g\) 是 `o_proj_moe_gen` 之前的 per-head attention 输出，不包含 output projection、attention residual 或 MLP。

## 2. 实验设置

- task：`BananaOnPlateTask`；
- chunk：3；
- prompt：`Pick up the banana and put it on the plate`；
- seed：`1081530687`；
- guidance：3.0；
- denoise steps：4；
- Transformer blocks：0–27；
- CFG branch：conditional、unconditional；
- shift：1、5；
- future query：L1–L8；
- key groups：K_AR、K_L0–K_L8、K_action；
- Query heads：16；KV heads：8；head dimension：128；
- GQA：\(h_v=\lfloor h_q/2\rfloor\)。

每个 query token、Q head 和 key group 独立计算三个指标，然后才沿同一 future latent 的 340 个 spatial query tokens统计 mean、std、p10、p50、p90、min、max。原始 CSV 保留：

```text
shift, task, chunk, branch, step, timestep,
block, query_latent, head, key_group
```

## 3. 实现与真实输出验证

采集器使用 post-QK-norm/post-RoPE Q/K，以及 value projection 和 head reshape 后的真实 V。K_AR/V_AR 在 cache 命中后直接读取实际 `und_k_cached/und_v_cached`。

在 fused attention kernel 返回后捕获实际输出 \(O_{kernel}\)，并验证 FP32 dense 分组重建：

\[
O_{reconstructed}=\sum_gO_g\approx O_{kernel}.
\]

### 3.1 CPU 测试

结果：`5 passed`。

测试覆盖：

- GQA 连续映射 `[0,0,1,1]`；
- 均匀 attention、同方向 V 时 \(\rho_g=1\)；
- 均匀 attention 时 \(\alpha_g\) 等于 group mass；
- \(\sum_g\alpha_g=1\)；
- step/branch/block gating 和只读输出。

### 3.2 GPU 正确性

| 检查 | shift=1 | shift=5 |
|---|---:|---:|
| action relative L2 / max abs | 0 / 0 | 0 / 0 |
| vision relative L2 / max abs | 0 / 0 | 0 / 0 |
| call-block 数 | 224 | 224 |
| \(\sum_g\alpha_g\) 最大误差 | 7.15e-7 | 7.15e-7 |
| reconstructed/kernel relative L2 最大值 | 0.001970 | 0.001914 |
| reconstructed/kernel cosine 最小值 | 0.99999797 | 0.99999809 |
| reconstructed/kernel max abs | 0.00472 | 0.00486 |
| NaN/Inf | false | false |

FP32 dense 重算与 fused BF16 attention kernel 存在约 0.2% 的舍入差异，但方向 cosine 大于 0.9999979。step0 的 shift=1/5 三个指标最大绝对差均为 0。

## 4. Future frame 的真实输出

### 4.1 全部 block 聚合

shift=1 conditional：

| step | category | \(\|O_g\|\) | \(\rho_g\) | \(\alpha_g\) |
|---:|---|---:|---:|---:|
| 0/999 | Self | 0.1162 | 0.5115 | 0.2050 |
| 0/999 | Adjacent | 0.0548 | 0.4897 | 0.0990 |
| 0/999 | Far | 0.0303 | 0.4821 | 0.0534 |
| 3/249 | Self | 0.1756 | 0.6311 | 0.3316 |
| 3/249 | Adjacent | 0.0590 | 0.5969 | 0.1162 |
| 3/249 | Far | 0.0251 | 0.5745 | 0.0505 |

shift=5 conditional：

| step | category | \(\|O_g\|\) | \(\rho_g\) | \(\alpha_g\) |
|---:|---|---:|---:|---:|
| 0/999 | Self | 0.1162 | 0.5115 | 0.2050 |
| 0/999 | Adjacent | 0.0548 | 0.4897 | 0.0990 |
| 0/999 | Far | 0.0303 | 0.4821 | 0.0534 |
| 3/624 | Self | 0.1699 | 0.5975 | 0.3081 |
| 3/624 | Adjacent | 0.0567 | 0.5652 | 0.1053 |
| 3/624 | Far | 0.0266 | 0.5495 | 0.0502 |

unconditional 分支几乎相同。例如 shift=1/step3 的 `Self/Adjacent/Far`：

- \(\|O_g\|=0.1762/0.0597/0.0258\)；
- \(\rho_g=0.6305/0.5964/0.5743\)；
- \(\alpha_g=0.3310/0.1161/0.0505\)。

结论：

1. 三个真实输出指标在聚合层面都保持 `Self > Adjacent > Far`；
2. 去噪推进时 Self output norm 和 alpha 明显增长；
3. Adjacent 基本稳定；Far output norm 下降，alpha 维持约 0.05；
4. \(\rho\) 随去噪整体升高，说明 late denoise 的 future-frame V 方向更一致、内部抵消更少；
5. shift=1 的 late-step Self 贡献增长比 shift=5 更强。

该顺序不是每一个 head/block 都严格成立。在全部独立 `shift/branch/step/block/query/head` 样本上：

- \(\|O_g\|\) 严格 `Self > Adjacent > Far`：61.8%；
- \(\rho_g\) 严格成立：45.0%；
- \(\alpha_g\) 严格成立：62.9%。

因此这是稳定的聚合规律和 B17 规律，而不是所有 attention head 的硬约束。

### 4.2 B17

B17/shift=1/conditional：

| step | category | \(\|O_g\|\) | \(\rho_g\) | \(\alpha_g\) |
|---:|---|---:|---:|---:|
| 0/999 | Self | 0.2371 | 0.6674 | 0.3288 |
| 0/999 | Adjacent | 0.0912 | 0.6266 | 0.1213 |
| 0/999 | Far | 0.0363 | 0.6040 | 0.0456 |
| 3/249 | Self | 0.2827 | 0.7368 | 0.4484 |
| 3/249 | Adjacent | 0.0857 | 0.6831 | 0.1391 |
| 3/249 | Far | 0.0216 | 0.6449 | 0.0325 |

B17 的真实输出清楚支持前一轮 attention 结论：去噪深入后 Self 对最终 per-head attention 方向的平均贡献从 0.329 增至 0.448，Far 从 0.0456 降至 0.0325。

## 5. K_AR 与 Action

### 5.1 全局平均

| shift / branch | group | \(\|O_g\|\) | \(\rho_g\) | \(\alpha_g\) |
|---|---|---:|---:|---:|
| shift=1 conditional | K_AR | 0.0394 | 0.3212 | 0.0444 |
| shift=1 unconditional | K_AR | 0.0343 | 0.4746 | 0.0408 |
| shift=5 conditional | K_AR | 0.0401 | 0.3219 | 0.0425 |
| shift=5 unconditional | K_AR | 0.0332 | 0.4770 | 0.0379 |
| shift=1 conditional | Action | 0.00584 | 0.7591 | 0.00642 |
| shift=1 unconditional | Action | 0.00615 | 0.7600 | 0.00623 |
| shift=5 conditional | Action | 0.00628 | 0.7410 | 0.00760 |
| shift=5 unconditional | Action | 0.00663 | 0.7420 | 0.00733 |

K_AR 在上一轮具有很大的 `A×||V||`，但真实输出揭示了明显抵消：

- conditional K_AR 只有约 32% 的 magnitude upper bound 转化为 \(\|O_g\|\)；
- unconditional K_AR 转化约 47%–48%；
- Action 的 \(\rho\) 约 74%–76%，内部方向比 K_AR 一致，但其绝对输出和 alpha 都较小。

### 5.2 Block specialization

K_AR：

- 输出 norm 最大的主要是 B24、B1、B26；
- 正向 alpha 最大的是 B1、B26，其次才是 B3/B24 等；
- B24 可以形成很大的 \(\|O_g\|\)，但与总 attention 输出的方向一致性弱于 B1/B26。

Action：

- B9 的输出 norm 和 alpha 都是最高；
- B6、B7、B12 仍然属于主要贡献 block；
- 前一轮观察到的 B6–B12 specialization 在真实输出中继续成立。

QK mass 的 block 排名与真实输出的 Spearman 相关：

- K_AR：对 output norm 为 0.747–0.862，对 alpha 为 0.871–0.917；
- Action：对 output norm 为 0.823–0.829，对 alpha 为 0.743–0.869。

所以 QK mass 能定位主要 block，但不能准确替代真实输出大小或最终方向贡献。

按 `shift/branch/step/block/query/head` 的空间均值检查，K_AR 和 Action 各有约 29% 的行出现负 alpha；不过其 alpha p10 仅约 -0.0010 和 -0.0005，总体均值仍为正。Future-frame key group 出现负 mean alpha 的比例仅约 1%–2%。这说明 K_AR/Action 的部分专门化 head 会抵消总输出，而 future-frame group 通常沿总 attention 方向提供正贡献。

## 6. 最终结论

1. **future temporal localization 经真实输出验证成立。** 聚合层面和 B17 都稳定表现为 `Self > Adjacent > Far`。
2. **late denoise 的 Self 不只是 attention probability 更高。** 它形成更大的真实输出、内部抵消更少，并对最终 per-head attention 方向贡献更多。
3. **K_AR 的大 V 尺度不能直接解释为大真实贡献。** conditional K_AR 的 cancellation ratio 只有约 0.32，大量 `A×||V||` 在向量求和中抵消。
4. **Action 输出较小但方向更一致。** \(\rho\) 约 0.74–0.76，B9 仍是最主要 Action block。
5. **QK block specialization 基本保留，但排序会变化。** B24 是“输出大但最终方向贡献相对有限”的代表；B1/B26 的 K_AR 正向贡献更明确。

本实验只解释 `o_proj_moe_gen` 前的 per-head attention 输出，不声明 `W_O` 后、完整 block residual、最终 video/action 或任务成功率具有相同贡献关系。

## 7. 输出

根目录：

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/attention/output/
  edge_attention_output_BananaOnPlateTask_c3_shift1_shift5_v1/
```

核心文件：

- `true_output_group_metrics.csv`：保留完整轴和三个指标的 spatial 分布；
- `true_output_group_aggregate.csv`：按 step/block/branch/key group 聚合；
- `true_output_temporal_categories.csv`；
- `true_output_temporal_step_summary.csv`；
- `true_output_result_summary.json`；
- `true_output_offline_validation.json`；
- `paired_validation.json`；
- `figures/group_output_norm_shift1_shift5.png`；
- `figures/cancellation_ratio_shift1_shift5.png`；
- `figures/direction_contribution_shift1_shift5.png`；
- `figures/temporal_three_true_output_metrics.png`。
