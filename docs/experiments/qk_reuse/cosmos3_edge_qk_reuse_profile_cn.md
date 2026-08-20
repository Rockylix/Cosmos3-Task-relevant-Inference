# Cosmos3 Edge 相邻未来帧 Q/K 复用候选 Profile（第一轮）

## 1. 本轮目标与边界

本轮只定位可供后续端到端干预验证的 Q/K 复用候选，不修改 attention 计算，不复制 Q/K，不评估任务成功率或加速比。

- Task：`BananaOnPlateTask`
- 真实任务 chunk：`c3`、`c5`
- shift：`5`
- denoise steps：4，对应 timestep `999 / 937 / 833 / 624`
- Transformer blocks：`B0..B27`
- CFG：conditional、unconditional 均采集
- future latent：`L1..L8`，检查 7 个相邻帧对
- checkpoint：`/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID`

输出目录：

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/qk_optimization/reuse_profile/
  edge_qk_reuse_profile_BananaOnPlateTask_c3_c5_shift5_allblocks_v1/
```

## 2. 捕获位置与指标

捕获的是真正送入 attention 的 GEN Q/K：完成线性投影、head reshape、QK RMSNorm 和 RoPE 之后，进入 attention kernel 之前。

- Q：16 个 query heads
- K：8 个 KV heads
- 单帧包含 340 个空间 token
- 相邻帧在相同空间位置逐 token 对齐比较

对表示 `Z`（分别取 Q、K），源帧 `a` 与目标帧 `b`：

### 2.1 整帧 cosine

```text
full_cosine(Za, Zb) = cosine(vec(Za), vec(Zb))
```

它适合快速筛选，但可能被大量高相似 token 掩盖，不能保证每个空间位置都可靠。

### 2.2 同位置 token cosine

对每个空间 token `j`：

```text
token_cosine_j = cosine(vec(Za[j]), vec(Zb[j]))
```

统计 mean、p10、p50、p90、min/max。其中 p10 表示至少 90% 的空间 token 不低于该值，比整帧 cosine 或 token mean 更保守。

### 2.3 Q/K 联合判据

```text
joint_full_cosine       = min(Q_full_cosine, K_full_cosine)
joint_token_mean        = min(Q_token_mean, K_token_mean)
joint_token_p10         = min(Q_token_p10, K_token_p10)
```

候选还必须在四个上下文中全部通过阈值：

```text
c3 conditional
c3 unconditional
c5 conditional
c5 unconditional
```

因此聚合表中的 `*_min4` 是四个上下文的最差值。

## 3. Profile 完整性验证

每个 chunk 捕获：

```text
4 steps × 2 CFG branches × 28 blocks = 224 call-blocks
224 × 7 adjacent pairs = 1568 rows
```

两个 chunk 共 3136 行原始数据；跨四上下文聚合后为：

```text
4 steps × 28 blocks × 7 adjacent pairs = 784 rows
```

所有数值均通过 NaN/Inf 检查。使用相同输入和 seed 分别运行无 hook baseline 与只读 profile，结果为：

| chunk | action relative L2 | action max abs | vision relative L2 | vision max abs |
|---|---:|---:|---:|---:|
| c3 | 0 | 0 | 0 | 0 |
| c5 | 0 | 0 | 0 | 0 |

这证明当前 collector 没有改变模型输出。

## 4. 候选统计

下面的数量单位均为 `(step, block, adjacent pair)`，并要求 c3/c5、conditional/unconditional 四个上下文全部通过。

| 稳定判据 | >= 0.85 | >= 0.90 | >= 0.95 |
|---|---:|---:|---:|
| Q/K 整帧 cosine 最小值 | 135 | 26 | 7 |
| Q/K token cosine mean 最小值 | 134 | 26 | 7 |
| Q/K token cosine p10 最小值 | 62 | 15 | 0 |

整帧 cosine >= 0.90 得到 26 个候选，但更保守的 token p10 >= 0.90 只保留 15 个。这说明部分候选虽然总体相似，仍有超过约 10% 的局部 token 未达到 0.90，不宜直接作为首轮复用点。

## 5. 保守候选列表

推荐判据：Q 和 K 的同位置 token cosine p10，在四个上下文中的最差值仍 >= 0.90。

| step/timestep | block | future pair | joint full cosine min4 | joint token mean min4 | joint token p10 min4 |
|---|---:|---|---:|---:|---:|
| 0 / 999 | B27 | L7->L8 | 0.9201 | 0.9186 | 0.9036 |
| 2 / 833 | B1 | L1->L2 | 0.9321 | 0.9323 | 0.9149 |
| 2 / 833 | B1 | L2->L3 | 0.9337 | 0.9340 | 0.9168 |
| 2 / 833 | B1 | L3->L4 | 0.9352 | 0.9354 | 0.9183 |
| 2 / 833 | B1 | L4->L5 | 0.9358 | 0.9359 | 0.9181 |
| 2 / 833 | B1 | L5->L6 | 0.9354 | 0.9355 | 0.9174 |
| 2 / 833 | B1 | L6->L7 | 0.9352 | 0.9352 | 0.9165 |
| 2 / 833 | B1 | L7->L8 | 0.9348 | 0.9348 | 0.9158 |
| 3 / 624 | B1 | L1->L2 | 0.9517 | 0.9518 | 0.9414 |
| 3 / 624 | B1 | L2->L3 | 0.9527 | 0.9527 | 0.9430 |
| 3 / 624 | B1 | L3->L4 | 0.9544 | 0.9545 | 0.9452 |
| 3 / 624 | B1 | L4->L5 | 0.9548 | 0.9548 | 0.9435 |
| 3 / 624 | B1 | L5->L6 | 0.9534 | 0.9534 | 0.9411 |
| 3 / 624 | B1 | L6->L7 | 0.9528 | 0.9529 | 0.9399 |
| 3 / 624 | B1 | L7->L8 | 0.9526 | 0.9526 | 0.9396 |

主要模式很集中：后两个去噪 step 的 B1 对全部相邻 future latent 都较相似；step 0 的 B27 只有 L7->L8 通过保守判据。

## 6. 首轮端到端干预建议

不能在同一个 `(step, block)` 中直接执行所有相邻复制，例如同时执行 `L1->L2` 和 `L2->L3` 会形成重叠依赖；后一操作的来源是否已经被替换会使实验定义含糊，也会放大误差。

第一轮建议采用互不重叠的 9 个复用操作：

```text
step 0 / B27: L7 -> L8

step 2 / B1:  L1 -> L2, L3 -> L4, L5 -> L6, L7 -> L8
step 3 / B1:  L1 -> L2, L3 -> L4, L5 -> L6, L7 -> L8
```

这只是 profile 给出的候选，并不等价于端到端安全。高 cosine 不保证 attention logits、softmax 后输出、最终 action 或任务成功率不变。

后续干预前还需明确一项实现语义：

- 若只把目标帧的 Q/K slice 替换为源帧 Q/K，V 仍使用目标帧自己的 V，则仍会完整执行 attention matmul，主要验证敏感性，未必节省时间。
- 若目标是“不计算第二帧”，则需要设计真正跳过目标 Q/K projection 或 attention token 计算的稀疏路径，并正确处理 V、attention mask、RoPE position、输出恢复与 GQA；不能用简单 tensor copy 直接声称加速。

因此推荐下一阶段先做“Q/K 替换敏感性实验”：Baseline/Sparse 使用同一输入、noise、seed、scheduler 状态，返回 Baseline action，逐 chunk 记录 action MSE/relative-L2/cosine/max-abs；确认误差可接受后，再实现真实跳算和 CUDA 计时。

## 7. 结果文件

```text
qk_profile_all.csv                     # c3/c5 的 3136 行原始 profile
qk_cross_chunk_branch_profile.csv      # 四上下文最差值/均值，784 行
qk_stable_candidates.csv               # 三种指标、三个阈值的全部候选
qk_recommended_conservative.csv        # p10 >= 0.90 的 15 个候选
qk_recommended_non_overlapping.csv     # 首轮建议的 9 个非重叠候选
profile_summary.json                   # 机器可读摘要
paired_validation.json                 # 只读一致性验证
chunk_000003/qk_adjacent_profile.csv
chunk_000005/qk_adjacent_profile.csv
figures/qk_full_cosine_worst_context.png
figures/qk_token_p10_worst_context.png
```

## 8. 复现实验命令

采集需要使用一个全新的输出目录：

```bash
cd /root/robolab/cosmos-framework-edge

PYTHONPATH=/root/robolab/cosmos-framework-edge \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/run_edge_qk_reuse_profile.py \
  --output-root /root/robolab/cosmos-framework-edge/experiments/preliminary/qk_optimization/reuse_profile/<new_run_name> \
  --threshold 0.90
```

聚合与绘图：

```bash
PYTHONPATH=/root/robolab/cosmos-framework-edge \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/analyze_edge_qk_reuse_profile.py \
  /root/robolab/cosmos-framework-edge/experiments/preliminary/qk_optimization/reuse_profile/<new_run_name>
```

注意：profile run 的 wall time 包含 GPU->CPU 统计、quantile 和 CSV 写入，不能用来代表原模型 chunk latency，也不能据此计算加速比。
