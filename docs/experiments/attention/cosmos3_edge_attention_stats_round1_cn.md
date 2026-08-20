# Cosmos3 Edge 未来帧 Attention 统计实验（第一轮）

## 1. 实验范围

本轮只验证以下三类统计量和实现正确性：

1. group attention mass；
2. group enrichment（同时保存 `log2_enrichment`）；
3. normalized entropy；
4. token 分组、softmax 重算和只读插桩的正确性验证。

未进行 value-aware、空间结构、attention ablation，也不据此声明 attention 质量、任务成功率或可稀疏性。

## 2. 固定输入和配对设置

- task：`BananaInBowlTask`
- action chunk：3
- prompt：`Pick up the banana and place it in the bowl`
- seed：`579362556`
- guidance：3.0
- denoise steps：4
- Transformer blocks：0–27
- CFG branch：conditional、unconditional
- future query：L1–L8
- query heads：16；KV heads：8；head dimension：128
- GEN tokens：3093 = 9 × 340 vision tokens + 33 action tokens
- conditional K_AR：158 tokens，总 K：3251
- unconditional K_AR：19 tokens，总 K：3112
- shift=1 timesteps：999、749、499、249
- shift=5 timesteps：999、937、833、624

shift=1 与 shift=5 在同一个已加载模型内依次运行；`data_batch`、conditioning observation、初始 noise seed、guidance 和采样步数完全相同。每次调用前重新复制 `data_batch` 并重置 Python、NumPy、PyTorch CPU/CUDA RNG。

## 3. 实际统计位置与公式

统计回调位于 Q/K norm 和 RoPE 之后、真实 attention kernel 之前。首个 denoise call 使用当前 forward 的文本 K；后续 call 使用 request-local text-KV cache 中实际传给 attention 的 K_AR。key 顺序严格按真实 kernel：

```text
K_all = [K_AR, K_L0, K_L1, ..., K_L8, K_action]
```

对每个 future query token 和 Q head，先用 FP32、小 query chunk 重算完整 dense softmax：

```text
A = softmax(Q K^T / sqrt(head_dim))
M_g = sum(A[..., key_group_g])
B_g = N_g / N_K
E_g = M_g / B_g
H_norm = -sum(A log A) / log(N_K)
```

每个 query token/head 的数值计算完成后，才沿 340 个 spatial query token 聚合 mean、std、p10、p50、p90、min 和 max。原始输出保留以下轴：

```text
shift, task, chunk, branch, step, timestep,
block, query_latent, head, key_group
```

不保存完整 attention matrix。

## 4. 五张图片的精确计算公式

### 4.1 统一符号与聚合规则

固定 shift \(\alpha\)、CFG branch \(b\)、denoise step \(s\) 和 Transformer block \(l\)。定义：

- future query latent：\(f\in\{1,\ldots,8\}\)；
- 每个 latent 的 spatial query token：\(i\in\{1,\ldots,N_S\}\)，其中 \(N_S=340\)；
- Q head：\(h\in\{1,\ldots,H\}\)，其中 \(H=16\)；
- attention key：\(j\in\{1,\ldots,N_K\}\)；conditional 时 \(N_K=3251\)，unconditional 时 \(N_K=3112\)；
- 原子 key group：\(G_g\in\{K_{AR},K_{L0},\ldots,K_{L8},K_{action}\}\)。

对单个 query token 和 Q head，首先按照真实 attention 计算：

\[
A^{\alpha,b,s,l}_{f,i,h,j}
=
\frac{
\exp\!\left(\langle Q_{f,i,h},K_{j,h}\rangle/\sqrt{d_h}\right)
}{
\sum_{k=1}^{N_K}
\exp\!\left(\langle Q_{f,i,h},K_{k,h}\rangle/\sqrt{d_h}\right)
},
\qquad d_h=128.
\]

式中的 \(K_{j,h}\) 表示按模型 GQA 映射将 8 个 KV head `repeat_interleave(2)` 到 16 个 Q head 后，与第 \(h\) 个 Q head 实际配对的 K。

所有图都先计算每个 \((f,i,h)\) 的指标，再做平均；没有先平均 Q、K 或 attention logits。图中的每个 panel 固定 shift 和 CFG branch，不跨 branch 或 shift 平均。

### 4.2 图一：`group_mass_shift1_shift5.png`

单个 query token/head 对 key group \(G_g\) 的 attention mass：

\[
M^{\alpha,b,s,l}_{f,i,h,g}
=
\sum_{j\in G_g} A^{\alpha,b,s,l}_{f,i,h,j}.
\]

CSV 中的 `mass_mean` 先沿同一 future latent 的 340 个 spatial query token 求均值：

\[
\overline M^{\alpha,b,s,l}_{f,h,g}
=
\frac{1}{N_S}
\sum_{i=1}^{N_S}M^{\alpha,b,s,l}_{f,i,h,g}.
\]

热力图一个像素的最终值再沿 8 个 future query latent 和 16 个 Q head 求均值：

\[
Y^{mass}_{\alpha,b,s,l,g}
=
\frac{1}{8H}
\sum_{f=1}^{8}\sum_{h=1}^{H}
\overline M^{\alpha,b,s,l}_{f,h,g}.
\]

图的横轴是 block 0–27，纵轴是 11 个原子 key group；四列对应四个 step，四行对应 `shift=1/5 × conditional/unconditional`。色值越大表示 future query 分配给该 key group 的 softmax 概率总量越大。绘图色轴下限为 0，上限使用所有原始 `mass_mean` 的 99.5% 分位数；精确数值以 CSV 为准。

### 4.3 图二：`group_log2_enrichment_shift1_shift5.png`

key group 按 token 数得到的随机均匀 baseline 为：

\[
B^{\alpha,b}_g=\frac{|G_g|}{N_K^{\alpha,b}}.
\]

对每个 query token/head 先计算 enrichment 和 log2 enrichment：

\[
E^{\alpha,b,s,l}_{f,i,h,g}
=
\frac{M^{\alpha,b,s,l}_{f,i,h,g}}
{B^{\alpha,b}_g+\varepsilon},
\]

\[
Z^{\alpha,b,s,l}_{f,i,h,g}
=
\log_2\!\left(\max(E^{\alpha,b,s,l}_{f,i,h,g},\varepsilon)\right),
\qquad \varepsilon=10^{-12}.
\]

CSV 的 `log2_enrichment_mean` 先沿 spatial query token 平均，图中再沿 future latent 和 Q head 平均：

\[
Y^{log2E}_{\alpha,b,s,l,g}
=
\frac{1}{8HN_S}
\sum_{f=1}^{8}\sum_{h=1}^{H}\sum_{i=1}^{N_S}
Z^{\alpha,b,s,l}_{f,i,h,g}.
\]

这里是“先逐 token 取 \(\log_2\)，再平均”，不是 \(\log_2\) 作用于平均 mass。\(Y=0\) 表示与 token-count baseline 相同；\(Y=1\) 表示约 2 倍富集；\(Y=-1\) 表示约为 baseline 的一半。布局与图一相同。可视化色轴固定为 \([-4,4]\)，超出范围的极端值会颜色饱和，但 CSV 中保留原值。

### 4.4 图三：`normalized_entropy_step_block_shift1_shift5.png`

对每个 future query token/head，在全部 \(N_K\) 个 key 上计算 normalized entropy：

\[
H^{\alpha,b,s,l}_{f,i,h}
=
-\frac{
\sum_{j=1}^{N_K}
A^{\alpha,b,s,l}_{f,i,h,j}
\log\!\left(\max(A^{\alpha,b,s,l}_{f,i,h,j},\varepsilon)\right)
}{\log N_K}.
\]

热力图一个像素是 8 个 future latent、340 个 spatial query token 和 16 个 Q head 的均值：

\[
Y^{entropy}_{\alpha,b,s,l}
=
\frac{1}{8HN_S}
\sum_{f=1}^{8}\sum_{h=1}^{H}\sum_{i=1}^{N_S}
H^{\alpha,b,s,l}_{f,i,h}.
\]

横轴是 block，纵轴是 `step/timestep`；四个 panel 对应 `shift=1/5 × conditional/unconditional`。色轴固定为 \([0,1]\)：接近 1 表示 attention 接近均匀分布，较低表示 attention 更集中，但不直接等价于更高语义质量。

### 4.5 图四：`self_frame_mass_mean_step_block_shift1_shift5.png`

对 query latent \(L_f\)，只选择与它具有相同 future-frame 编号的 key group \(K_{L_f}\)。单 token/head 的 matching-frame mass 为：

\[
M^{self,\alpha,b,s,l}_{f,i,h}
=
\sum_{j\in K_{L_f}}
A^{\alpha,b,s,l}_{f,i,h,j}.
\]

热力图像素为：

\[
Y^{self\_mass}_{\alpha,b,s,l}
=
\frac{1}{8HN_S}
\sum_{f=1}^{8}\sum_{h=1}^{H}\sum_{i=1}^{N_S}
M^{self,\alpha,b,s,l}_{f,i,h}.
\]

横轴是 block，纵轴是 `step/timestep`，四个 panel 对应两个 shift 和两个 CFG branch。色值表示 future query 分配给“同编号 future latent”的绝对 attention mass，色轴固定为 \([0,0.5]\)。它保留了 query latent 与 key latent 的对应关系，不会像图一那样在显示前把不同 query latent 对同一个固定 key group 混在一起。

### 4.6 图五：`self_frame_enrichment_mean_step_block_shift1_shift5.png`

每个 future latent 都有 340 个 key token，因此 matching-frame group 的 token-count baseline 为：

\[
B^{self}_{\alpha,b}=\frac{340}{N_K^{\alpha,b}}.
\]

单 token/head 的 matching-frame enrichment：

\[
E^{self,\alpha,b,s,l}_{f,i,h}
=
\frac{M^{self,\alpha,b,s,l}_{f,i,h}}
{340/N_K^{\alpha,b}}.
\]

热力图像素为：

\[
Y^{self\_enrichment}_{\alpha,b,s,l}
=
\frac{1}{8HN_S}
\sum_{f=1}^{8}\sum_{h=1}^{H}\sum_{i=1}^{N_S}
E^{self,\alpha,b,s,l}_{f,i,h}.
\]

\(Y=1\) 表示 matching-frame mass 与其 token 数占比一致；\(Y>1\) 表示 query 对同编号 future latent 有额外偏好。布局与图四相同，色轴固定为 \([0,4.5]\)。conditional/unconditional 的 \(N_K\) 不同，因此分别使用各自的 baseline 后才能比较 enrichment。

### 4.7 五张图的聚合维度汇总

| 图片 | 单样本先计算 | 图中继续平均 | 图中保留的轴 |
|---|---|---|---|
| group mass | 每个 query token/head 的 group probability sum | spatial token、future query latent、head | shift、branch、step、block、key group |
| group log2 enrichment | 每个 query token/head 的 `log2(mass / token_fraction)` | spatial token、future query latent、head | shift、branch、step、block、key group |
| normalized entropy | 每个 query token/head 在全部 key 上的归一化熵 | spatial token、future query latent、head | shift、branch、step、block |
| self-frame mass | 每个 query Lf token/head 对 `K_Lf` 的 mass | spatial token、future query latent、head | shift、branch、step、block |
| self-frame enrichment | 每个 self-frame mass 除以 `340/N_K` | spatial token、future query latent、head | shift、branch、step、block |

## 5. 运行命令

### 5.1 CPU 单元测试

```bash
cd /root/robolab/cosmos-framework-edge

PYTHONPATH=/root/robolab/cosmos-edge-overlay:/root/robolab/cosmos-framework-edge \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python -m pytest -q \
  -o cache_dir=/tmp/pytest-cache-attn \
  cosmos_framework/scripts/robolab_attention_stats_capture_test.py
```

结果：`4 passed`。

### 5.2 配对 GPU 实验

输出目录必须不存在：

```bash
cd /root/robolab/cosmos-framework-edge

PYTHONPATH=/root/robolab/cosmos-edge-overlay:/root/robolab/cosmos-framework-edge \
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
CUDA_VISIBLE_DEVICES=0 \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/run_edge_attention_stats_paired.py \
  --output-root \
  /root/robolab/cosmos-framework-edge/experiments/preliminary/attention/stats/edge_attention_mass_enrichment_entropy_BananaInBowlTask_c3_shift1_shift5_v1
```

### 5.3 聚合和绘图

```bash
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/analyze_edge_attention_stats.py \
  /root/robolab/cosmos-framework-edge/experiments/preliminary/attention/stats/edge_attention_mass_enrichment_entropy_BananaInBowlTask_c3_shift1_shift5_v1
```

## 6. 正确性结果

### 6.1 插桩不改变模型输出

| shift | output | relative L2 | max absolute error |
|---:|---|---:|---:|
| 1 | action | 0 | 0 |
| 1 | vision latent | 0 | 0 |
| 5 | action | 0 | 0 |
| 5 | vision latent | 0 | 0 |

### 6.2 数值与分组验证

| 检查 | shift=1 | shift=5 |
|---|---:|---:|
| group partition 完整且无重叠 | true | true |
| NaN/Inf | false | false |
| mass sum 最大误差 | 4.77e-7 | 5.36e-7 |
| enrichment identity 最大误差 | 5.96e-8 | 5.96e-8 |
| sampled dense reference 最大误差 | 2.74e-6 | 2.74e-6 |
| 完整 call-block 数 | 224 | 224 |

step0/999 的 shift=1 与 shift=5 对齐结果：

- `mass_mean` 最大绝对差：0；
- `enrichment_mean` 最大绝对差：0；
- `entropy_mean` 最大绝对差：0。

这证明 shift 配对、step/branch/block gating 和统计位置一致。

## 7. 第一轮观察结果

### 7.1 Future query 的注意力质量不是均匀随机分配

future key（L1–L8）占实际 K token 的大多数，因此总 mass 很高：

| shift / branch | future mass 均值 | future enrichment 均值 | matching-frame mass 均值 | matching-frame enrichment 均值 |
|---|---:|---:|---:|---:|
| shift=1 conditional | 0.7629 | 0.9118 | 0.2650 | 2.5336 |
| shift=1 unconditional | 0.7696 | 0.8805 | 0.2636 | 2.4128 |
| shift=5 conditional | 0.7483 | 0.8944 | 0.2437 | 2.3299 |
| shift=5 unconditional | 0.7553 | 0.8642 | 0.2424 | 2.2182 |

需要同时看 mass 和 enrichment：future 总 mass 约 0.75–0.77，但它总体略低于按 token 数得到的随机 baseline；相反，query Lf 对同一 `K_Lf` 的 matching-frame attention 明显富集，平均约为 baseline 的 2.22–2.53 倍。

### 7.2 同帧富集随去噪推进增强

按 step 聚合后的 matching-frame enrichment：

| shift / branch | step0 | step1 | step2 | step3 |
|---|---:|---:|---:|---:|
| shift=1 conditional | 1.788 | 2.626 | 2.787 | 2.934 |
| shift=1 unconditional | 1.635 | 2.523 | 2.675 | 2.818 |
| shift=5 conditional | 1.788 | 2.346 | 2.511 | 2.675 |
| shift=5 unconditional | 1.635 | 2.256 | 2.413 | 2.569 |

两种 shift 的 step0 完全一致；后续差异与两组 scheduler timestep 不同同时发生，不能把它解释成仅由 `shift` 参数直接改变了 attention。

### 7.3 B17 是最稳定的集中注意力位置

四组 shift/branch 中，normalized entropy 的全局最低点都在 step3、B17：

| shift / branch | step/timestep | block | entropy mean | matching-frame enrichment |
|---|---|---:|---:|---:|
| shift=1 conditional | 3/249 | 17 | 0.3806 | 4.1131 |
| shift=1 unconditional | 3/249 | 17 | 0.3824 | 3.9331 |
| shift=5 conditional | 3/624 | 17 | 0.3864 | 3.8038 |
| shift=5 unconditional | 3/624 | 17 | 0.3881 | 3.6365 |

B17 的低 entropy 与高 matching-frame enrichment 同时出现，且跨 CFG branch、shift 稳定，是第二轮空间结构或 value-aware 检查的首选代表 block。当前结果只说明 attention weight 更集中，尚不能证明其 value contribution 更重要或更准确。

### 7.4 entropy 随 denoise step 整体下降

conditional 分支按 block 平均：

- shift=1：0.6418 → 0.5917 → 0.5794 → 0.5692；
- shift=5：0.6418 → 0.6073 → 0.5909 → 0.5793。

unconditional 分支具有同样趋势。B0/step0 接近均匀（约 0.986–0.988），中后层与后期 step 更集中。

## 8. 输出文件

根目录：

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/attention/stats/
  edge_attention_mass_enrichment_entropy_BananaInBowlTask_c3_shift1_shift5_v1/
```

关键文件：

- `shift_1/group_attention_stats.csv`、`shift_5/group_attention_stats.csv`：保留完整实验轴的 mass/enrichment 分布；
- `shift_1/entropy_stats.csv`、`shift_5/entropy_stats.csv`：保留完整实验轴的 normalized entropy；
- `group_attention_aggregate.csv`：step × block × branch × key group 聚合；
- `entropy_aggregate.csv`：step × block × branch 聚合；
- `derived_group_metrics.csv`：future、condition、AR、action、matching-frame 派生统计；
- `future_query_to_vision_key_mass.csv`：query L1–L8 到 key L0–L8 的明细；
- `paired_validation.json`、`offline_validation.json`：正确性检查；
- `paired_outputs.pt`：baseline/capture 的 action 与 vision latent；
- `figures/group_mass_shift1_shift5.png`；
- `figures/group_log2_enrichment_shift1_shift5.png`；
- `figures/normalized_entropy_step_block_shift1_shift5.png`；
- `figures/self_frame_mass_mean_step_block_shift1_shift5.png`；
- `figures/self_frame_enrichment_mean_step_block_shift1_shift5.png`。

## 9. 当前结论边界

第一轮发现的稳定模式是“未来 query 对 matching future frame 的 key 有明显富集，且注意力随 denoise 推进在 B17 等 block 变得更集中”。这足以支持下一轮只针对代表性 step/block 增加 value-aware 或空间结构分析，但不能仅凭 mass、enrichment、entropy 判断某个 attention 是高质量对应、可以删除，或能够保持 action/vision 输出。

## 10. 第二任务交叉验证：BananaOnPlateTask

### 10.1 配对设置与正确性

为了判断上述模式是否只是 `BananaInBowlTask` 的个例，使用相同模型、chunk 3、guidance=3、4 个 denoise step 和同一套只读 attention hook，换成：

- task：`BananaOnPlateTask`；
- prompt：`Pick up the banana and put it on the plate`；
- seed：`1081530687`；
- state index：96；action 范围：96–127；
- shift=1 timesteps：999、749、499、249；
- shift=5 timesteps：999、937、833、624。

两种 shift 均通过以下验证：

| 检查 | shift=1 | shift=5 |
|---|---:|---:|
| action relative L2 / max abs | 0 / 0 | 0 / 0 |
| vision relative L2 / max abs | 0 / 0 | 0 / 0 |
| mass sum 最大误差 | 5.96e-7 | 5.96e-7 |
| enrichment identity 最大误差 | 5.96e-8 | 5.96e-8 |
| sampled dense reference 最大误差 | 2.07e-6 | 1.71e-6 |
| NaN/Inf | false | false |
| 完整 call-block 数 | 224 | 224 |

step0/999 的 shift=1/5 mass、enrichment 和 entropy 最大绝对差均为 0。

### 10.2 结论一：从近似均匀到时间局部化

该规律在第二任务上复现，但应准确表述为：**浅层 B0 在高噪声阶段接近均匀；B17 在 step0 已经具有明显局部性，随后随去噪继续增强。**

B0、step0 的 query-key 时间间距 gap=0 到 gap=7 的 enrichment：

| branch | gap=0 | gap=7 |
|---|---:|---:|
| conditional | 1.0460 | 1.0145 |
| unconditional | 1.0375 | 1.0064 |

B17 按 `Self / Adjacent / Far(gap>=2)` 聚合的 enrichment：

| shift / branch | step0 | step3 |
|---|---|---|
| shift=1 conditional | 2.822 / 1.184 / 0.527 | 3.950 / 1.340 / 0.404 |
| shift=1 unconditional | 2.590 / 1.104 / 0.521 | 3.779 / 1.283 / 0.388 |
| shift=5 conditional | 2.822 / 1.184 / 0.527 | 3.658 / 1.240 / 0.424 |
| shift=5 unconditional | 2.590 / 1.104 / 0.521 | 3.499 / 1.187 / 0.408 |

对 B17 的每个固定 query latent 分别检查，四个 step、两个 branch、两个 shift 中均为 `Self > Adjacent > Far`；每个组合都是 8/8 个 query latent 成立。B17 的 gap 与 enrichment 的 Spearman 相关系数为 -0.905 到 -0.976，说明整体随时间距离衰减，但不要求每个远距离 gap 都严格单调。

**判定：跨任务支持。** 需要否定的过强说法是“所有 block 在 step0 都近似均匀”；时间局部性具有明确的 block 依赖。

### 10.3 结论二：K_AR 和 Action 的 block/head specialization

按所有 step、future query 和 head 聚合，第二任务的高响应 block 与第一任务一致：

| group | shift=1 conditional 的主要 block（mass） | shift=5 conditional 的主要 block（mass） |
|---|---|---|
| K_AR | B1 0.533、B3 0.241、B24 0.162、B26 0.154 | B1 0.418、B3 0.214、B24 0.176、B26 0.165 |
| Action | B9 0.040、B7 0.025、B6 0.024、B12 0.023、B8 0.021、B10 0.019 | 排名仍为 B9、B7、B6、B12、B8、B10 |

unconditional 分支也保留相同 block 模式。K_AR 的 B3/B24 具有强 head 集中：top-4 head mass share 分别约 0.71/0.79；B1/B26 分别约 0.39/0.47，属于中等集中。Action 在 B6/B7/B9/B12 的 top-4 head share 约 0.55–0.68。

**判定：跨任务支持。** 更精确的结论是“block specialization 很强；head specialization 普遍存在但强度依 block 而异”，而不是所有这些 block 都只由极少数 head 读取。

### 10.4 结论三：去噪越深入，越依赖自身帧

对全部 28 个 block 聚合，第二任务得到：

| shift / branch | self enrichment step0→3 | adjacent enrichment step0→3 | far enrichment step0→3 |
|---|---|---|---|
| shift=1 conditional | 1.833→2.817 | 0.995→1.076 | 0.593→0.520 |
| shift=1 unconditional | 1.741→2.705 | 0.958→1.040 | 0.581→0.508 |
| shift=5 conditional | 1.833→2.603 | 0.995→0.998 | 0.593→0.529 |
| shift=5 unconditional | 1.741→2.501 | 0.958→0.967 | 0.581→0.517 |

全部 block 平均的 normalized entropy：

| shift / branch | step0 | step1 | step2 | step3 |
|---|---:|---:|---:|---:|
| shift=1 conditional | 0.6351 | 0.5854 | 0.5753 | 0.5665 |
| shift=1 unconditional | 0.6359 | 0.5829 | 0.5719 | 0.5627 |
| shift=5 conditional | 0.6351 | 0.5974 | 0.5856 | 0.5769 |
| shift=5 unconditional | 0.6359 | 0.5967 | 0.5835 | 0.5740 |

self enrichment 严格逐 step 上升的 block 为 24/28、24/28、24/28、25/28；entropy 严格逐 step 下降的 block 为 22/28、22/28、23/28、21/28（顺序为 shift1 conditional/unconditional、shift5 conditional/unconditional）。

**判定：聚合层面强支持，单个 block 不应写成绝对单调。** self 明显增强，far 整体受抑制，adjacent 维持在均匀 baseline 附近或轻度富集，entropy 整体下降。

### 10.5 交叉任务总结与边界

`BananaInBowlTask` 与 `BananaOnPlateTask` 在以下方面一致：

1. B0/step0 近似均匀，而 B17 具有稳定的 `Self > Adjacent > Far` 时间局部性；
2. K_AR 的 B1/B3/B24/B26 和 Action 的 B6–B12 specialization 稳定复现；
3. 去噪推进时 self enrichment 上升、far enrichment 降低、entropy 下降。

因此三条主结论均通过第二任务验证。当前证据仍只覆盖两个 banana 操作任务、各一个 chunk；它证明模式不是单个 task 的偶然结果，但尚不能替代更多物体、动作类型和 chunk 的分层抽样验证，也不能由 attention weight 直接推出 value contribution 或可稀疏性。

第二任务输出目录：

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/attention/stats/
  edge_attention_mass_enrichment_entropy_BananaOnPlateTask_c3_shift1_shift5_v1/
```

复现实验命令：

```bash
cd /root/robolab/cosmos-framework-edge

PYTHONPATH=/root/robolab/cosmos-edge-overlay:/root/robolab/cosmos-framework-edge \
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
CUDA_VISIBLE_DEVICES=0 \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/run_edge_attention_stats_paired.py \
  --output-root /root/robolab/cosmos-framework-edge/experiments/preliminary/attention/stats/edge_attention_mass_enrichment_entropy_BananaOnPlateTask_c3_shift1_shift5_v1 \
  --conditioning-image /root/robolab/cosmos-framework-edge/experiments/preliminary/representation/hidden_states/edge_hidden_banana_bowl_plate_c357_v1/task_pick_up_the_banana_and_put_it_on_the_plate_99d69b4f/chunk_000003/conditioning_observation.png \
  --episode-hdf5 /root/robolab/RoboLab/output/cosmos3_edge_hidden_banana_bowl_plate_c357_v1/BananaOnPlateTask/run_0.hdf5 \
  --task BananaOnPlateTask \
  --prompt 'Pick up the banana and put it on the plate' \
  --seed 1081530687 \
  --chunk 3 \
  --state-index 96 \
  --finger-joint-index 7
```
