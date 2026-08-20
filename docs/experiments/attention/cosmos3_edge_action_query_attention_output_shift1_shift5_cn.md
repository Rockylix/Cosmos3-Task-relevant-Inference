# Cosmos3 Edge Action Query 对 K groups 的关注与真实输出实验

## 1. 实验目标

本实验将上一轮 future-latent Query 的 group attention/真实输出分析扩展到 Action Query，回答：

1. Action Query 在不同 denoise step、Transformer block、CFG branch 下关注哪些 K group；
2. attention probability mass 有多少真正形成 group 输出；
3. 各 group 对最终 attention 输出方向是推动还是抵消；
4. `shift=1` 与 `shift=5` 是否改变上述规律；
5. 不同预测 action horizon 是否采用不同的信息来源。

## 2. 固定实验配置

- task：`BananaOnPlateTask`；
- chunk：3；
- seed：`1081530687`；
- guidance：3.0；
- denoise steps：4；
- shift：1、5；
- CFG branch：conditional、unconditional；
- block：`B0、B4、B8、B12、B16、B20、B24`；
- Query heads：16；
- KV heads：8；
- GQA：每两个连续 Query heads 共享一个 KV head；
- action token：33 个。

真实 Action layout 为：

- `A0`：当前机器人 state condition token；
- `A1...A32`：32 个预测 action token，对应报告中的 horizon 0...31。

所有主图和主聚合只统计预测 horizon 0...31。A0 单独保存到 `condition_action_query_group_aggregate.csv`，没有混入预测 horizon。

两个 shift 的真实 UniPC timestep：

| step | shift=1 | shift=5 |
|---:|---:|---:|
| 0 | 999 | 999 |
| 1 | 749 | 937 |
| 2 | 499 | 833 |
| 3 | 249 | 624 |

因此两个 shift 的 step 0 完全相同；后续相同 step index 不代表相同噪声强度。

## 3. K group 定义

对每个 Action Query 使用完整真实 key 集合：

\[
\mathcal G=
\{K_{AR},K_{L0},K_{L1},\ldots,K_{L8},K_{action}\}.
\]

- `K_AR`：instruction/text 等真实 AR KV cache；
- `K_L0`：condition vision latent；
- `K_L1...K_L8`：8 个 future vision latent；
- `K_action`：全部 33 个 Action key，包括 state condition 和预测 action key。

这些 group 无重叠并严格覆盖实际 attention 的全部 key。

## 4. 数学定义

### 4.1 Attention probability

固定 Action Query token \(i\) 和 Query head \(h_q\)：

\[
A_{ijh_q}
=
\operatorname{softmax}_j
\left(
\frac{Q_{ih_q}K_{j,h_v}^{\top}}{\sqrt d}
\right),
\qquad
h_v=\left\lfloor\frac{h_q}{2}\right\rfloor.
\]

Q/K 为 QK norm 后、RoPE 后真正参与 attention 的张量。V 为 value projection 与 head reshape 后真正送入 attention kernel 的张量。

### 4.2 Group attention mass

\[
m_g(i,h_q)=\sum_{j\in g}A_{ijh_q}.
\]

它回答该 Query/head 有多少 softmax probability 分给 group \(g\)。

### 4.3 Group enrichment

\[
e_g(i,h_q)
=
\frac{m_g(i,h_q)}{|g|/N_K}.
\]

- \(e_g=1\)：与均匀 attention 基准一致；
- \(e_g>1\)：相对 group token 数量发生富集；
- \(e_g<1\)：相对均匀基准被抑制。

热力图使用 `log2(enrichment)` 色标，但 CSV 保存原始 enrichment。

### 4.4 Normalized entropy

\[
H(i,h_q)
=
-\frac{\sum_j A_{ijh_q}\log(A_{ijh_q}+\epsilon)}{\log N_K}.
\]

范围约为 `[0,1]`。越接近 1 越均匀，越接近 0 越集中。

### 4.5 真实 group 输出大小

按照模型真实 GQA 映射：

\[
O_g(i,h_q)
=
\sum_{j\in g}A_{ijh_q}V_{j,h_v}.
\]

统计：

\[
\|O_g(i,h_q)\|_2.
\]

它回答该 group 最终形成了多大的 head output。

### 4.6 Group 内部方向一致性

\[
\rho_g(i,h_q)
=
\frac{
\|O_g(i,h_q)\|_2
}{
\sum_{j\in g}A_{ijh_q}\|V_{j,h_v}\|_2+\epsilon
}.
\]

- \(\rho_g\approx1\)：group 内 V 方向高度一致；
- \(\rho_g\ll1\)：虽然存在 \(A\|V\|\)，但 V 在求和时大量抵消。

### 4.7 对最终输出方向的贡献

\[
O(i,h_q)=\sum_g O_g(i,h_q),
\]

\[
\alpha_g(i,h_q)
=
\frac{
\langle O_g(i,h_q),O(i,h_q)\rangle
}{
\|O(i,h_q)\|_2^2+\epsilon
}.
\]

- \(\alpha_g>0\)：推动最终 attention 输出；
- \(\alpha_g<0\)：抵消其他 group；
- 对全部 group 求和应约等于 1。

### 4.8 真实输出边界说明

Flash/fused attention kernel 不直接返回按 K group 分解的输出，因此本实验：

1. 捕获真实 post-RoPE Q/K、真实 V；
2. 使用 FP32 重算 Action Query 的 softmax 和 \(O_g\)；
3. 将 \(\sum_gO_g\) 与 attention kernel 的真实 pre-`o_proj` 输出比较。

最坏重建结果为 relative L2 `0.001733`、cosine `0.99999845`。该小误差来自 BF16 fused kernel 与 FP32 重算的数值路径差异。

## 5. 代码与运行命令

代码：

- `cosmos_framework/scripts/robolab_action_query_attention_capture.py`；
- `cosmos_framework/scripts/robolab_action_query_attention_capture_test.py`；
- `/root/robolab/cosmos-edge-overlay/tools/run_edge_action_query_attention_paired.py`；
- `/root/robolab/cosmos-edge-overlay/tools/analyze_edge_action_query_attention.py`。

CPU 测试：

```bash
cd /root/robolab/cosmos-framework-edge

/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python -m pytest -q \
  cosmos_framework/scripts/robolab_action_query_attention_capture_test.py
```

采集：

```bash
cd /root/robolab/cosmos-framework-edge

PYTHONPATH=/root/robolab/cosmos-edge-overlay:/root/robolab/cosmos-framework-edge \
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
CUDA_VISIBLE_DEVICES=0 \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/run_edge_action_query_attention_paired.py
```

分析：

```bash
PYTHONPATH=/root/robolab/cosmos-framework-edge \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/analyze_edge_action_query_attention.py \
  /root/robolab/cosmos-framework-edge/experiments/preliminary/attention/action_query/edge_action_query_attention_output_BananaOnPlateTask_c3_shift1_shift5_b0to24s4_v2
```

## 6. 输出目录

正式结果目录：

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/attention/action_query/
└── edge_action_query_attention_output_BananaOnPlateTask_c3_shift1_shift5_b0to24s4_v2/
    ├── experiment.json
    ├── paired_outputs.pt
    ├── paired_validation.json
    ├── result_summary.json
    ├── action_query_group_aggregate.csv
    ├── action_query_horizon_metrics.csv
    ├── action_query_entropy_aggregate.csv
    ├── action_query_horizon_entropy.csv
    ├── condition_action_query_group_aggregate.csv
    ├── shift5_minus_shift1.csv
    ├── shift_1/
    │   ├── action_query_group_metrics.csv
    │   ├── action_query_entropy.csv
    │   ├── calls.csv
    │   └── validation.json
    ├── shift_5/
    │   └── ...
    └── figures/
        ├── action_query_mass.png
        ├── action_query_enrichment.png
        ├── action_query_normalized_entropy.png
        ├── action_query_group_output_norm.png
        ├── action_query_cancellation_ratio.png
        ├── action_query_direction_contribution.png
        ├── action_horizon_mass.png
        └── action_horizon_group_output_norm.png
```

`v1` 是第一次 live-layout 探测失败留下的不完整目录，不包含有效 capture；分析和引用必须使用 `v2`。

## 7. 正确性验证

CPU 测试：

```text
5 passed
```

覆盖：

- 均匀 attention 的 mass/enrichment/entropy 基准；
- 连续 GQA head 映射；
- A0 condition 与 A1...A32 horizon 映射；
- K group 完整划分；
- group output 重建真实 O；
- step/branch/block gating；
- hook 不改变 forward 输出。

GPU 验证：

| 验证项 | shift=1 | shift=5 |
|---|---:|---:|
| call-block | 56/56 | 56/56 |
| mass sum error max | 3.58e-7 | 4.77e-7 |
| alpha sum error max | 2.98e-7 | 2.98e-7 |
| O reconstruction relative L2 max | 0.001733 | 0.001731 |
| O reconstruction cosine min | 0.99999845 | 0.99999845 |
| NaN/Inf | 0 | 0 |

baseline 与 capture 配对：

- shift=1 action/vision：逐元素 max absolute error 为 0；
- shift=5 action/vision：逐元素 max absolute error 为 0。

因此采集 hook 对最终模型输出是零扰动。

## 8. 核心结果

以下结果均先逐 `Action Query × head × group` 计算，再跨预测 horizon、step、block 和 branch 聚合。

### 8.1 Action Query 最强读取 K_action

| shift | group | mass | enrichment | \(\|O_g\|\) | \(\rho_g\) | \(\alpha_g\) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | K_action | 0.2842 | 27.40 | 0.1589 | 0.8449 | 0.3313 |
| 5 | K_action | 0.2683 | 25.86 | 0.1613 | 0.8233 | 0.3116 |

K_action 只有 33 个 token，约占完整 key 集合的 1.04%，但获得约 27–28% attention mass。它不仅 probability enrichment 最大，而且 group output 最大、内部抵消最少，并贡献约 31–33% 的最终输出方向。

这说明 Action Query 存在很强的 action-to-action 协同/自一致计算。

### 8.2 K_AR 被明显富集，但真实输出贡献较小

| shift | group | mass | enrichment | \(\|O_g\|\) | \(\rho_g\) | \(\alpha_g\) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | K_AR | 0.0533 | 4.11 | 0.0353 | 0.4045 | 0.0331 |
| 5 | K_AR | 0.0506 | 3.95 | 0.0378 | 0.4035 | 0.0353 |

K_AR 相对 token 数量约有 4× enrichment，但 \(\rho\) 只有约 0.40，最终方向贡献约 3–4%。因此“相对 token 数被强关注”不等于“主导最终 Action attention 输出”。

### 8.3 Vision groups：L0 与 L8 最突出

| shift | group | mass | enrichment | \(\|O_g\|\) | \(\alpha_g\) |
|---:|---|---:|---:|---:|---:|
| 1 | K_L0 | 0.1023 | 0.957 | 0.0380 | 0.0839 |
| 1 | K_L8 | 0.1011 | 0.946 | 0.0291 | 0.0957 |
| 5 | K_L0 | 0.1074 | 1.005 | 0.0419 | 0.0939 |
| 5 | K_L8 | 0.1023 | 0.957 | 0.0295 | 0.0942 |

L0/L8 在 vision groups 中整体最突出；L1...L7 的整体 enrichment 多数为 `0.53–0.75`。不过 vision group 具有明显 block specialization，不能只看全局平均。

### 8.4 Block specialization 很强

K_action 跨 shift、branch、step 聚合：

| block | mass | enrichment | \(\|O_g\|\) | \(\rho_g\) | \(\alpha_g\) |
|---:|---:|---:|---:|---:|---:|
| B0 | 0.011 | 1.10 | 0.0056 | 0.930 | 0.019 |
| B4 | 0.102 | 9.81 | 0.0236 | 0.564 | 0.104 |
| B8 | 0.509 | 49.07 | 0.1948 | 0.816 | 0.541 |
| B12 | 0.307 | 29.56 | 0.1195 | 0.914 | 0.347 |
| B16 | 0.359 | 34.60 | 0.2375 | 0.887 | 0.536 |
| B20 | 0.090 | 8.71 | 0.0640 | 0.882 | 0.139 |
| B24 | 0.556 | 53.58 | 0.4755 | 0.847 | 0.564 |

主要层间规律：

- B0 几乎按 token 数量基准读取 K_action；
- B8、B16、B24 是最强 action-to-action block；
- B24 的 K_action mass、真实输出大小和方向贡献均最高；
- K_AR mass 在 B24 最高，但方向贡献在 B4 最高；
- L0 mass 在 B4 最高，方向贡献在 B8 最高；
- B20 对 L3/L5/L6/L8 更突出，其中 L8 在 B20 的 mass/alpha 约为 `0.272/0.286`。

因此 Action Query 并不是在所有层均匀读取 action、text 和 vision，而是存在明确的信息源 specialization。

### 8.5 Attention 随 denoise 推进变得更集中

Normalized entropy：

| shift | step 0 | step 1 | step 2 | step 3 |
|---:|---:|---:|---:|---:|
| 1 | 0.6996 | 0.6810 | 0.6734 | 0.6664 |
| 5 | 0.6996 | 0.6952 | 0.6849 | 0.6761 |

两个 shift 均随去噪推进下降。shift=1 因为后续 timestep 更低，集中速度更快。

按 block 聚合时：

- shift=1：B0 `0.977`，B24 `0.514`；
- shift=5：B0 `0.986`，B24 `0.508`。

B0 接近全局均匀混合，B8/B20/B24 已明显集中。

### 8.6 K_action 的 denoise step 趋势

| shift | step/timestep | mass | enrichment | \(\|O_g\|\) | \(\alpha_g\) |
|---:|---|---:|---:|---:|---:|
| 1 | 0/999 | 0.230 | 22.14 | 0.186 | 0.265 |
| 1 | 1/749 | 0.288 | 27.78 | 0.150 | 0.331 |
| 1 | 2/499 | 0.302 | 29.09 | 0.148 | 0.350 |
| 1 | 3/249 | 0.317 | 30.59 | 0.152 | 0.379 |
| 5 | 0/999 | 0.230 | 22.14 | 0.186 | 0.265 |
| 5 | 1/937 | 0.262 | 25.22 | 0.158 | 0.306 |
| 5 | 2/833 | 0.284 | 27.37 | 0.152 | 0.328 |
| 5 | 3/624 | 0.298 | 28.73 | 0.148 | 0.347 |

随着去噪推进，K_action mass/enrichment/alpha 增大，但 \(\|O_g\|\) 没有同步增大。step 0 的 group output norm 反而最高，说明 attention probability 和 V/output 尺度必须分开判断。

### 8.7 Horizon 规律

跨 step、block、head 聚合后：

- K_action mass 在 horizon 0...31 上相对稳定：shift=1 的相对范围约 5.1%，shift=5 约 6.4%；
- K_L8 mass 的 horizon 变化很强，相对范围约 119–123%；
- K_L0 也有约 42–45% 的 horizon 相对变化；
- K_action 在中间 horizon 略高，在 horizon 31 回落；
- conditional/unconditional 的 horizon 曲线非常接近。

这表明 action-to-action 是跨 horizon 稳定的基础路径，而 vision latent 的读取更具有 action-horizon 选择性。

## 9. Shift=1 与 shift=5

两者总体结构一致：

- K_action 始终是最强 group；
- K_AR 始终高 enrichment、低最终方向贡献；
- L0/L8 是最突出 vision group；
- B8/B16/B24 的 action specialization 稳定存在。

主要定量差异：

- shift=1 的 K_action mass/alpha 更高：`0.284/0.331`，shift=5 为 `0.268/0.312`；
- shift=5 的 K_action output norm 略高，但 \(\rho\) 略低；
- shift=5 的 L0 output/alpha 略高；
- 差异主要来自后续 step 使用不同 timestep，不能解释为 shift 参数在同噪声水平下的纯因果效应。

## 10. 结论边界

本实验验证了 Action Query 在一次固定 task/chunk 上的 attention 与真实 value-output 分解。它说明：

1. Action Query 最主要的真实输出路径是 K_action；
2. K_AR 虽被富集，但内部抵消较多、最终方向贡献较小；
3. vision 信息读取存在强 block 和 horizon specialization；
4. 只看 attention mass/enrichment 会遗漏 V 尺度和方向抵消；
5. shift=1/5 的结构规律一致，强度略有差异。

但这些结果不能单独证明某个 group 可以删除，也不能说明 ablation 后 action 或任务成功率保持不变。下一步如需验证因果重要性，应对代表性 block/group 做最小、配对的 output ablation，而不是只依据 attention mass。
