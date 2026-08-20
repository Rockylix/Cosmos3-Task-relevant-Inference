# Cosmos3 Edge Attention V 尺度与 V-aware 加权实验

## 1. 实验目标与范围

本实验在第一轮 Q/K attention mass、enrichment 和 normalized entropy 的基础上，只增加两项验证：

1. 真正参与 attention 的 V 的尺度；
2. 使用 `A × ||V||` 判断 V 尺度如何调整原始 QK attention mass。

本轮仍不进行 value 向量方向抵消分析、输出投影 `W_O` 分析或 attention ablation。因此 `A × ||V||` 只能作为 contribution magnitude 的上界代理，不能等价为最终 block 输出贡献。

实验输入：

- task：`BananaOnPlateTask`；
- chunk：3；
- prompt：`Pick up the banana and put it on the plate`；
- seed：`1081530687`；
- guidance：3.0；
- denoise steps：4；
- blocks：0–27；
- CFG branch：conditional、unconditional；
- shift：1、5；
- future query：L1–L8。

## 2. 实际捕获位置

Q/K 使用 QK RMSNorm 和 RoPE 之后、真实 attention kernel 之前的张量。V 捕获位置严格为：

\[
V_{\mathrm{attn}}=\operatorname{reshape}(XW_V),
\]

即 value projection 与 head reshape 之后、attention matmul 之前；V 不应用 RoPE。

实际路径为：

```text
q_gen = reshape(q_proj_moe_gen(X_gen)) → QK norm → RoPE
k_gen = reshape(k_proj_moe_gen(X_gen)) → QK norm → RoPE
v_gen = reshape(v_proj_moe_gen(X_gen))

k_ar = 当前 forward 的 post-RoPE K，或实际 und_k_cached
v_ar = 当前 forward 的 projected/reshaped V，或实际 und_v_cached
```

模型配置为 16 个 Query heads、8 个 KV heads、head dimension 128。统计端与模型使用相同的连续 GQA 映射：

\[
h_v=\left\lfloor\frac{h_q}{2}\right\rfloor,
\]

等价于将 KV heads 沿 head 轴执行 `repeat_interleave(2)`：

```text
Q0,Q1   → KV0
Q2,Q3   → KV1
...
Q14,Q15 → KV7
```

## 3. 指标定义

### 3.1 V 尺度

对 key token \(j\) 和映射后的 KV head \(h_v\)：

\[
s_{j,h_q}=\|V_{j,h_v}\|_2,
\qquad
r_{j,h_q}=\frac{\|V_{j,h_v}\|_2}{\sqrt{d_v}}.
\]

`value_l2` 保存 \(s\)，`value_rms` 保存 \(r\)。按 `shift × branch × step × block × Q head × key group` 保存 mean、std、p10、p50、p90、min、max。

### 3.2 原始 QK attention mass

\[
A_{ijh}=\operatorname{softmax}_j\left(\frac{Q_{ih}K_{jh}^{\top}}{\sqrt{d_h}}\right),
\]

\[
M_g(i,h)=\sum_{j\in G_g}A_{ijh}.
\]

### 3.3 V-weighted magnitude 与 share

先计算未归一化的 magnitude upper bound：

\[
C_g(i,h)=\sum_{j\in G_g}A_{ijh}\|V_{j,h_v}\|_2.
\]

再归一化为所有 key group 之和为 1 的 V-weighted share：

\[
P_g^V(i,h)=
\frac{C_g(i,h)}
{\sum_j A_{ijh}\|V_{j,h_v}\|_2+\epsilon}.
\]

### 3.4 V capacity baseline 与 enrichment

只考虑 token 数和 V 尺度、不考虑 QK 选择时的 baseline：

\[
B_g^V(h)=
\frac{\sum_{j\in G_g}\|V_{j,h_v}\|_2}
{\sum_j\|V_{j,h_v}\|_2+\epsilon}.
\]

\[
E_g^V(i,h)=\frac{P_g^V(i,h)}{B_g^V(h)+\epsilon}.
\]

### 3.5 V 对 QK mass 的重加权比例

\[
R_g^V(i,h)=\frac{P_g^V(i,h)}{M_g(i,h)+\epsilon}.
\]

- \(R_g^V>1\)：该 group 的 QK mass 落在相对更大尺度的 V 上；
- \(R_g^V\approx1\)：V 尺度基本不改变 QK mass；
- \(R_g^V<1\)：V 尺度削弱该 group 的相对 share。

所有指标先对单个 query token/head 计算，再沿 spatial query token 聚合；没有先平均 Q、K、V 或 hidden。

## 4. 正确性验证

### 4.1 CPU 测试

结果：`5 passed`。

新增测试使用两个具有不同 V 尺度的 KV heads，验证 4 个 Q heads 严格映射为 `[0,0,1,1]`，并验证均匀 QK、head 内恒定 V 尺度时 `value_reweight_ratio=1`。

### 4.2 GPU 配对验证

| 检查 | shift=1 | shift=5 |
|---|---:|---:|
| action relative L2 / max abs | 0 / 0 | 0 / 0 |
| vision relative L2 / max abs | 0 / 0 | 0 / 0 |
| GQA repeat factor | 2 | 2 |
| value share sum 最大误差 | 5.96e-7 | 5.96e-7 |
| value enrichment identity 最大误差 | 5.96e-8 | 5.96e-8 |
| value reweight identity 最大误差 | 5.96e-8 | 5.96e-8 |
| sampled dense reference 最大误差 | 2.37e-6 | 1.88e-6 |
| NaN/Inf | false | false |
| call-block 数 | 224 | 224 |

捕获前后 action 与 vision 输出逐元素一致，说明 hook 为只读。

## 5. 实验结果

### 5.1 V 尺度不是所有 group 都相同

以下为所有 step、block 和 head 上的平均 `value_rms`：

| shift / branch | K_AR | future L1–L8 范围 | Action |
|---|---:|---:|---:|
| shift=1 conditional | 0.6355 | 0.0880–0.0900 | 0.0736 |
| shift=1 unconditional | 0.3545 | 0.0871–0.0897 | 0.0735 |
| shift=5 conditional | 0.6355 | 0.0930–0.0954 | 0.0784 |
| shift=5 unconditional | 0.3545 | 0.0922–0.0952 | 0.0783 |

主要现象：

1. conditional K_AR 的 V 尺度约为 future vision 的 6.7–7.2 倍；unconditional K_AR 约为 3.7–4.1 倍；
2. Action V 尺度约为 future vision 的 0.78–0.84 倍；
3. L1–L8 内部的 V 尺度接近。固定 `shift/branch/step/block/head` 后，八个 future frame 的 V RMS CV：mean=3.63%，p50=3.07%，p90=6.95%，max=15.17%。

因此，V 尺度可能明显改变 K_AR/Action 与 vision 之间的相对重要性，但不太可能单独制造 future L1–L8 内部的时间局部化。

### 5.2 V 加权没有推翻 future 时间局部化

按所有 block 聚合，step3 的 `value_weighted_share / QK mass`：

| shift / branch | Self | Adjacent | Far(gap>=2) |
|---|---:|---:|---:|
| shift=1 conditional | 1.025 | 1.021 | 1.015 |
| shift=1 unconditional | 1.068 | 1.063 | 1.057 |
| shift=5 conditional | 1.020 | 1.016 | 1.009 |
| shift=5 unconditional | 1.062 | 1.058 | 1.051 |

三类时间关系都受到近似共同的轻微重加权，Self 比 Far 只多约 0.6–1.1 个百分点。B17/step3 也保持相同结果：

- shift=1 conditional：QK enrichment `Self/Adjacent/Far = 3.950/1.340/0.404`；V reweight ratio `0.991/0.986/0.964`；
- shift=5 conditional：QK enrichment `3.658/1.240/0.424`；V reweight ratio `0.980/0.978/0.959`。

所以 `Self > Adjacent > Far` 主要来自 QK attention 的时间选择，而不是不同 future frame 的 V 尺度差异。V 加权略微增强 Self 相对 Far 的优势，但幅度很小。

### 5.3 V 会调整 K_AR 和 Action 的贡献判断

使用“整体平均 V-weighted share / 整体平均 QK mass”衡量：

| shift / branch | K_AR | Action |
|---|---:|---:|
| shift=1 conditional | 1.526 | 0.849 |
| shift=1 unconditional | 1.057 | 0.920 |
| shift=5 conditional | 1.569 | 0.861 |
| shift=5 unconditional | 1.047 | 0.933 |

因此：

- 只看 QK mass 会明显低估 conditional K_AR 的 magnitude share，低估约 53%–57%；
- unconditional K_AR 只被放大约 5%；
- 只看 QK mass 会轻度高估 Action，V 加权后相对 share 降低约 7%–15%。

block specialization 仍然稳定，但内部排序有所调整：

- K_AR：B1 仍是最高；V 加权后 B26/B24 上升，B3 相对下降；
- Action：B9 仍是最高，B6/B7/B12 仍在主要集合中；
- 跨 28 blocks，QK mass 与 V-weighted share 的 Spearman 排名相关系数为 0.910–0.977。

这说明原先的 K_AR/Action block specialization 不是 V 尺度造成的假象，但若要判断贡献强度，不能只使用 attention probability；V 尺度会显著改变 K_AR 的绝对判断。

## 6. 当前结论

1. **future temporal localization 结论保留。** L1–L8 的 V 尺度很接近，V 加权后 `Self > Adjacent > Far` 没有被推翻。
2. **K_AR 的重要性需要上调。** conditional K_AR 的 V 明显更大，QK mass 单独看会低估其 magnitude share。
3. **Action 的重要性略下调。** Action V 尺度低于 vision，V-aware share 比 QK mass 低约 7%–15%。
4. **block specialization 仍成立。** V 会调整 B3/B24/B26 的内部排序，但主要候选 block 集合基本不变。
5. **本轮仍不是最终 contribution。** `A×||V||` 忽略向量方向抵消，也没有经过 `W_O`；它不能直接说明某 group 对 block residual 或最终 action 的真实影响。

如果继续下一项判据，最直接的是计算：

\[
O_g(i,h)=\sum_{j\in G_g}A_{ijh}V_{j,h_v}
\]

以及 cancellation ratio：

\[
\rho_g(i,h)=
\frac{\|O_g(i,h)\|_2}
{\sum_{j\in G_g}A_{ijh}\|V_{j,h_v}\|_2+\epsilon}.
\]

它用于区分“V 尺度很大”与“这些 V 在向量求和后确实形成稳定输出”。

## 7. 输出文件

实验目录：

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/attention/value/
  edge_attention_value_BananaOnPlateTask_c3_shift1_shift5_v1/
```

关键文件：

- `shift_1/value_scale_stats.csv`、`shift_5/value_scale_stats.csv`：实际 V 尺度；
- `shift_*/group_attention_stats.csv`：QK 与 V-aware 指标明细；
- `value_scale_aggregate.csv`；
- `value_aware_group_aggregate.csv`；
- `temporal_value_categories.csv`；
- `self_frame_value_metrics.csv`；
- `paired_validation.json`、`value_offline_validation.json`；
- `value_result_summary.json`；
- `figures/group_value_rms_shift1_shift5.png`；
- `figures/group_value_reweight_log2_shift1_shift5.png`；
- `figures/self_frame_value_weighted_share_step_block_shift1_shift5.png`。
