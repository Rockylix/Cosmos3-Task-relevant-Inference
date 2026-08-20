# Cosmos3 Edge 未来帧空间 token 相似度：逐相邻帧对补充结果

## 1. 补充目标

原实验对七个相邻 future latent pair 的同位置 token cosine 取均值：

\[
H^R[y,x]
=
\frac{1}{7}\sum_{f=1}^{7}
\cos\left(X_f^R[y,x],X_{f+1}^R[y,x]\right).
\]

本补充实验不再对七个 pair 求均值，而是分别保留：

\[
H_f^R[y,x]
=
\cos\left(X_f^R[y,x],X_{f+1}^R[y,x]\right),
\qquad f=1,\ldots,7.
\]

即分别输出：

- L1→L2；
- L2→L3；
- L3→L4；
- L4→L5；
- L5→L6；
- L6→L7；
- L7→L8。

每个 pair 均输出 MLP、Q、K、V、O 五种表示，共 `7 × 5 = 35` 张热力图。

## 2. 实验参数

与原实验完全相同：

- task：`BananaOnPlateTask`；
- chunk：3；
- shift：5；
- denoise step/timestep：`0/999、1/937、2/833、3/624`；
- CFG：conditional、unconditional；
- block：`B0、B4、B8、B12、B16、B20、B24`；
- 空间 token：`17 × 20`；
- cosine 色标：固定 `[-1,1]`。

每张 PNG 仍使用原实验的 `8 × 7` 面板布局：

- 行：conditional step 0...3、unconditional step 0...3；
- 列：B0、B4、B8、B12、B16、B20、B24；
- 单个面板：固定相邻 future pair 的 `17 × 20` 同位置 token cosine。

## 3. 数据来源

本补充分析没有重新运行模型推理，直接读取原采集保存的：

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/spatial_token_similarity/
  edge_spatial_token_similarity_BananaOnPlateTask_c3_shift5_b0to24s4_v1/
  capture/spatial_similarity_maps.pt
```

该 artifact 对每个 `representation × step × branch × block` 保存：

- `pair_maps`：`[7,17,20]`；
- `mean_map`：`[17,20]`。

验证了：

\[
\max\left|
\operatorname{mean}_{pair}(\text{pair_maps})-	ext{mean_map}
\right|
=1.79\times10^{-7}.
\]

因此逐 pair 结果与原七对均值结果严格一致。

## 4. 运行命令

```bash
cd /root/robolab/cosmos-framework-edge

PYTHONPATH=/root/robolab/cosmos-framework-edge \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/analyze_edge_spatial_token_similarity_pairs.py \
  /root/robolab/cosmos-framework-edge/experiments/preliminary/representation/spatial_token_similarity/edge_spatial_token_similarity_BananaOnPlateTask_c3_shift5_b0to24s4_v1
```

脚本要求 `adjacent_pair_results/` 尚不存在；重算前应使用新的实验目录或显式处理旧结果。

## 5. 输出目录

```text
adjacent_pair_results/
├── validation.json
├── spatial_pair_similarity.csv
├── spatial_pair_summary.csv
├── spatial_pair_overall.csv
└── figures/
    ├── mlp/
    │   ├── mlp_l1_l2_spatial_cosine.png
    │   ├── ...
    │   └── mlp_l7_l8_spatial_cosine.png
    ├── q/
    ├── k/
    ├── v/
    └── o/
```

文件含义：

- `spatial_pair_similarity.csv`：666,400 行逐 `pair × step × branch × block × y × x` cosine；
- `spatial_pair_summary.csv`：1,960 行逐 map 空间统计；
- `spatial_pair_overall.csv`：35 行 representation × future pair 总体统计；
- `figures/`：35 张逐 pair 热力图。

## 6. 七个 pair 的总体结果

以下数值为每个 pair 先逐空间 token 计算 cosine，再跨空间、step、branch、block 聚合。

| representation | L1→L2 | L2→L3 | L3→L4 | L4→L5 | L5→L6 | L6→L7 | L7→L8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.6611 | 0.7179 | 0.7304 | **0.7511** | 0.6783 | 0.6864 | 0.6949 |
| Q | 0.7594 | 0.7938 | 0.7915 | **0.8068** | 0.7606 | 0.7527 | 0.7488 |
| K | 0.7439 | 0.7779 | 0.7706 | **0.7885** | 0.7469 | 0.7399 | 0.7376 |
| V | 0.3697 | 0.4359 | 0.4416 | **0.4534** | 0.3769 | 0.3881 | 0.4089 |
| O | 0.7088 | 0.7674 | 0.7861 | **0.8038** | 0.7360 | 0.7389 | 0.7435 |

## 7. 补充观察

逐 pair 展开后可以看到原均值图掩盖的时间结构：

1. 五种表示都在 L4→L5 达到七对中的最高平均 cosine；
2. L2→L3、L3→L4、L4→L5 构成连续的中段高相似区域；
3. L5→L6 在 MLP/Q/K/V/O 中同时明显下降，像是一个共同的 future-latent 边界；
4. MLP、V、O 的 L1→L2 最低，表示第一段 future transition 与中段不同；
5. Q/K 在 L6→L7、L7→L8 继续缓慢下降，而 V/O 在后段略有回升；
6. V 在所有 pair 上仍明显低于 Q/K/O，原实验的这一结论不是某一个 pair 单独造成的。

因此，七个相邻 pair 不应默认视为同分布。后续如果按 future latent 做稀疏近似或 anchor 选择，至少应把 L1→L2、L2→L5、L5→L6、L6→L8 分开检查，不能只依据七对平均 cosine。

当前仍只有一个 task/chunk，这一“中段高相似、L5→L6 下降”模式还需要跨 task/chunk 验证，不能直接解释为模型普遍存在固定的时间分段。

## 8. 完整性验证

- 原 artifact map：280/280；
- 展开 pair map：1,960/1,960；
- PNG：35/35，全部可正常解码；
- 逐 token CSV：666,400 行；
- NaN/Inf：0；
- 原始 cosine 范围：`[-0.713754,0.999876]`；
- 没有重新运行推理或修改模型。
