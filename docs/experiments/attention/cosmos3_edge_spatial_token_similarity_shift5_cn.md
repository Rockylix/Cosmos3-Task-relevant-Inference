# Cosmos3 Edge 未来帧空间 token 相似度实验（shift=5）

## 1. 实验目的与口径

本实验观察相邻 future latent 在**相同空间位置**上的特征方向是否一致。实验不做 token 全匹配，不允许 token 在空间上移动；位置 `(y,x)` 只与下一 future latent 的同一个 `(y,x)` 比较。

固定配置：

- task：`BananaOnPlateTask`；
- chunk：3；
- seed：`1081530687`；
- guidance：3.0；
- UniPC steps：4；
- shift：5.0；
- denoise step/timestep：`0/999`、`1/937`、`2/833`、`3/624`；
- CFG branch：conditional、unconditional；
- block：`B0、B4、B8、B12、B16、B20、B24`；
- future latent：`L1...L8`；
- 每个 future latent 的空间 token 网格：`17 × 20 = 340`。

对表示类型

\[
R\in\{\mathrm{MLP},Q,K,V,O\}
\]

记固定 step、branch、block 下，第 \(f\) 个 future latent 的空间 token 为

\[
X^R_f[y,x]\in\mathbb{R}^{D_R},\qquad f=1,\ldots,8.
\]

先逐相邻帧、逐同坐标 token 计算：

\[
c^R_f[y,x]
=
\frac{
\langle X^R_f[y,x],X^R_{f+1}[y,x]\rangle
}{
\|X^R_f[y,x]\|_2\,\|X^R_{f+1}[y,x]\|_2+\epsilon
},
\qquad f=1,\ldots,7.
\]

最后对七个相邻 future pair 做算术平均，得到一张 `17 × 20` 热力图：

\[
H^R[y,x]
=
\frac{1}{7}\sum_{f=1}^{7}c^R_f[y,x].
\]

注意：先得到每个 pair、每个空间 token 的 cosine，再对七个 pair 求均值；没有先平均 hidden，也没有把 340 个空间 token 展平成一个全局向量。

## 2. 五种真实张量边界

| 表示 | 采集位置 | 每个空间 token 的 cosine 特征维 |
|---|---|---:|
| MLP | generation MLP 子层输出、尚未加 block residual | 2048 |
| Q | QK norm 后、RoPE 后，真实进入 attention 的 Q | `16 × 128 = 2048` |
| K | QK norm 后、RoPE 后，真实进入 attention 的 GEN K | `8 × 128 = 1024` |
| V | value projection 与 head reshape 后，真实进入 attention 的 GEN V | `8 × 128 = 1024` |
| O | attention kernel 的真实输出、`o_proj` 前 | `16 × 128 = 2048` |

Q/K/V/O 都将一个 token 的全部 head 与 head-dim 展平后计算 cosine。K/V 保留模型真实的 8 个 KV head，没有人为复制到 16 个 Query head。Q/K 是 post-RoPE；V 不经过 RoPE。

采集器只读取 `detach()` 后的张量，不替换 attention kernel，不修改 forward 输出，也不运行额外的 dense attention 重算。

## 3. 运行命令

代码：

- `cosmos_framework/scripts/robolab_spatial_token_similarity_capture.py`
- `cosmos_framework/scripts/robolab_spatial_token_similarity_capture_test.py`
- `/root/robolab/cosmos-edge-overlay/tools/run_edge_spatial_token_similarity.py`
- `/root/robolab/cosmos-edge-overlay/tools/analyze_edge_spatial_token_similarity.py`

CPU 测试：

```bash
cd /root/robolab/cosmos-framework-edge

/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python -m pytest -q \
  cosmos_framework/scripts/robolab_spatial_token_similarity_capture_test.py
```

实际采集：

```bash
cd /root/robolab/cosmos-framework-edge

PYTHONPATH=/root/robolab/cosmos-edge-overlay:/root/robolab/cosmos-framework-edge \
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
CUDA_VISIBLE_DEVICES=0 \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/run_edge_spatial_token_similarity.py
```

绘图：

```bash
PYTHONPATH=/root/robolab/cosmos-framework-edge \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/analyze_edge_spatial_token_similarity.py \
  /root/robolab/cosmos-framework-edge/experiments/preliminary/representation/spatial_token_similarity/edge_spatial_token_similarity_BananaOnPlateTask_c3_shift5_b0to24s4_v1
```

脚本要求输出目录是全新目录；重跑时请将默认目录末尾改成 `v2` 等新后缀。

## 4. 输出目录

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/spatial_token_similarity/
└── edge_spatial_token_similarity_BananaOnPlateTask_c3_shift5_b0to24s4_v1/
    ├── experiment.json
    ├── paired_outputs.pt
    ├── paired_validation.json
    ├── spatial_summary.csv
    ├── result_summary.json
    ├── capture/
    │   ├── spatial_similarity.csv
    │   ├── spatial_similarity_maps.pt
    │   └── validation.json
    └── figures/
        ├── mlp_spatial_adjacent_cosine.png
        ├── q_spatial_adjacent_cosine.png
        ├── k_spatial_adjacent_cosine.png
        ├── v_spatial_adjacent_cosine.png
        └── o_spatial_adjacent_cosine.png
```

每张 PNG 均为 `8 × 7` 面板：

- 行：conditional step 0...3，随后 unconditional step 0...3；
- 列：B0、B4、B8、B12、B16、B20、B24；
- 每个面板：`17 × 20` 空间 token 图；
- 所有图统一使用 cosine 色标 `[-1,1]`，避免隐藏负相似 token。

`spatial_similarity.csv` 保存七个相邻 pair 均值后的逐空间 token 数值。`spatial_similarity_maps.pt` 还保留全部七个 pair 的原始 `7 × 17 × 20` map。

## 5. 正确性验证

CPU 单元测试结果：

```text
4 passed
```

覆盖：

- 同坐标匹配与七个相邻 pair 的均值；
- 多 head/head-dim 展平；
- step/branch/block gating；
- MLP/Q/K/V/O 五种表示齐全；
- hook 不改变模型 forward 返回值。

GPU 配对实验在相同输入、seed、guidance、num_steps 和 shift 下比较 baseline 与 capture：

| 输出 | relative L2 | cosine | max absolute error |
|---|---:|---:|---:|
| action | 0 | 1.0 | 0 |
| vision | 0 | 1.000005（FP32 cosine 舍入） | 0 |

采集完整性：

- map：`5 × 4 × 2 × 7 = 280/280`；
- NaN/Inf：0；
- 零范数 token：0；
- 全部七 pair 原始 cosine 范围：`[-0.713754, 0.999876]`。

因此本次 hook 对最终 action/vision 是逐元素零扰动。

## 6. 首轮数值结果

### 6.1 五种表示的全部 map/空间位置均值

| 表示 | mean cosine |
|---|---:|
| MLP | 0.702866 |
| Q | 0.773349 |
| K | 0.757904 |
| V | 0.410655 |
| O | 0.754929 |

V 的同位置相邻 future token 相似度显著低于 Q/K/O/MLP。说明不能用 Q/K 的高相似度直接代替对 V 或真实 O 的判断。

### 6.2 按 denoise step 聚合

以下数值均已先计算单个 `branch × step × block × (y,x)` 样本，再跨 block、branch 和空间位置取均值。

| 表示 | step 0 / 999 | step 1 / 937 | step 2 / 833 | step 3 / 624 |
|---|---:|---:|---:|---:|
| MLP | 0.721945 | 0.688849 | 0.692209 | 0.708461 |
| Q | 0.750323 | 0.767447 | 0.780429 | 0.795198 |
| K | 0.729354 | 0.747614 | 0.766406 | 0.788242 |
| V | 0.405861 | 0.375339 | 0.394198 | 0.467222 |
| O | 0.790596 | 0.752520 | 0.738449 | 0.738150 |

在这个 task/chunk 上：

- Q/K 随去噪推进整体升高；
- V 在 step 1 先降低，step 3 升至四步最高，不是单调趋势；
- O 从 step 0 到 step 3 整体降低，说明 Q/K 方向相似度升高没有自动转化为 O 同位置相似度升高；
- MLP 也是非单调。

### 6.3 按 block 聚合

| 表示 | B0 | B4 | B8 | B12 | B16 | B20 | B24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.882014 | 0.784315 | 0.639448 | 0.609624 | 0.552231 | 0.649428 | 0.803001 |
| Q | 0.849662 | 0.824321 | 0.738354 | 0.707521 | 0.725885 | 0.694946 | 0.872757 |
| K | 0.594440 | 0.852040 | 0.806170 | 0.727699 | 0.732215 | 0.647015 | 0.945747 |
| V | 0.218709 | 0.299255 | 0.482481 | 0.514255 | 0.407175 | 0.245386 | 0.707324 |
| O | 0.944320 | 0.713317 | 0.766617 | 0.729860 | 0.685947 | 0.562723 | 0.881716 |

最明显的层间差异：

- B24 的 K/V 同位置相似度最高，O 也很高；
- B0 的 V 很低，但 O 是所有采样 block 中最高，说明 attention 混合可产生高度共享的输出；
- B20 的 V 与 O 都偏低；
- MLP 在 B16 最低，随后到 B24 明显回升。

空间图还显示出稳定的局部带状/区域结构，例如 V 在部分中间 block 出现纵向或横向的高相似区域；这些结构在 conditional/unconditional 间大体可复现，但当前只有一个 task/chunk，不能据此声称跨任务稳定。

## 7. 当前结论边界

本实验只回答“同一个空间 token 坐标在相邻 future latent 中的方向相似度”。它不能区分：

- token 语义真的改变；
- 语义内容移动到了邻近空间坐标；
- head 内部发生重排或抵消；
- 表示方向相似但模长发生变化。

因此该结果可用于定位值得进一步检查的 `step × block × spatial region`，但不能单独证明某个 future token 可以被跳过或复用，也不能说明 action 不受影响。
