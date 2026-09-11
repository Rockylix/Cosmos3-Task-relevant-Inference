# Dense / ToCa / WorldCache / ASI：十任务汇总

统计日期：2026-09-11。ASI 十任务及十份配对计时/保真度测量均已完成；本次仅核验与汇总既有数据，没有重跑任务、修改策略或筛选成功样本。

## 1. 相同闭环协议

8 个简单任务 + 2 个普通任务，每任务 1 rollout；simulator seed=0，policy seed=0，deterministic_seed=False，每任务重置请求 RNG。4 denoise steps，shift=5，guidance=3，官方各任务步数上限。全部 native eager，关闭 compile/CUDA graphs。

ASI 是当前 `version/core80-stable104-action-weighted@62a09f7f95f1f0a54d5af9c2b8c530c862db9025`，不是早期 velocity-cache Version1：

- 每个 chunk 的 step0 conditional 全量 profile，Top-6 block 选 Core80，另选 Stable104。
- action horizon 权重 `[1/6, 1/3, 1/3, 1/6]`；每 future latent 保留 184/340 token。
- 同一 chunk 的其余 7 次 CFG Transformer forward 使用同一 mask，全部 28 个 block 真正缩短 Q/K/V/O/MLP 输入。
- 未选中的 hidden 在 stack 末尾用**本次 stack 输入**恢复，再正常运行输出头、CFG 和 UniPC；不使用 step0 velocity cache。

源码：[profile 与选区](../cosmos_framework/scripts/robolab_version1.py#L178)、[未选中 hidden 恢复](../cosmos_framework/scripts/robolab_version1.py#L516)。

## 2. 成功率、score 与稳定单 chunk 时间

| 策略 | 成功 | 简单 | 普通 | 平均 score | Chunk median(s) | 配对 Dense median(s) | 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 7/10 | 5/8 | 2/2 | 0.6000 | 0.864964 | 0.864964 | 1.000× |
| ToCa，D/C/D/C，r=0.25，bonus关，CFG独立 | 4/10 | 3/8 | 1/2 | 0.4667 | 0.787469 | 0.870994 | 1.106× |
| WorldCache，D/D/D/C，joint video/action | 5/10 | 5/8 | 0/2 | 0.4667 | 0.670914 | 0.863966 | 1.288× |
| ASI，Core80 + Stable104 | 5/10 | 5/8 | 0/2 | 0.4667 | 0.590330 | 0.864964 | 1.465× |

Dense 闭环成功率来自原十任务对照；表内 Dense 时间采用 ASI 本轮配对测量。各加速比均以各自同轮 Dense 为分母，不能用表内单个 Dense 秒数重新计算其他轮的比值。

计时：每任务/策略预热 5 次，测量 30 次，模式交替或轮转；先计算每任务统计量，再等权平均十任务。CUDA 同步，包含 generation、controller、策略 profile/cache 管理及最终 finite 检查；不含 VAE decode、模型加载、输入 CPU clone、输出 CPU copy、I/O 或仿真器耗时。

| 轮次 | 策略 | Mean(s) | Median(s) | P90(s) |
|---|---|---:|---:|---:|
| ToCa | Dense | 0.871018 | 0.870994 | 0.871639 |
| ToCa | ToCa | 0.787505 | 0.787469 | 0.788155 |
| WorldCache | Dense | 0.863989 | 0.863966 | 0.864859 |
| WorldCache | WorldCache | 0.671025 | 0.670914 | 0.671906 |
| ASI | Dense | 0.864919 | 0.864964 | 0.865975 |
| ASI | ASI | 0.590305 | 0.590330 | 0.591160 |

## 3. 相对配对 Dense 的输出保真度

每任务采样第三 chunk；每个样本先单独计算，再平均十任务。FP64 dot/norm；RGB 对完整 32 个 future frames 展平，固定 `[-1,1] → [0,1]`，不是逐帧归一化或先平均 hidden。Action 排除 q0；latent 排除 L0。

| 策略 | Action MSE ↓ | Action cosine ↑ | Latent cosine ↑ | Latent rel-L2 ↓ | RGB cosine ↑ | RGB rel-L2 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Dense 自身 | 0 | 1 | 1 | 0 | 1 | 0 |
| ToCa 固定版 | — | — | — | — | 0.992783 | 0.121688 |
| WorldCache | 0.00161489 | 0.999500 | 0.987977 | 0.154081 | 0.997963 | 0.063343 |
| ASI | 0.01534903 | 0.996577 | 0.533217 | 0.897266 | 0.919207 | 0.394712 |

ToCa 此次扫描只保留了 RGB 保真度，不把其他配置/其他输入的 action、latent 结果补入空格。ASI 的 action rel-L2 为 0.082251，WorldCache 为 0.030946。

重要边界：ToCa 在 Dense 轨迹的第三 chunk 上评估；WorldCache 和 ASI 各自在自己的闭环轨迹第三 chunk 上与 Dense 配对。WorldCache 前两个输入为经用户批准的同设置补采，其余八个为原闭环输入。因此各方法内部比较严格配对，但不是三种策略共享同一批 observation 的保真度横评。以上跨策略差异是目前样本的观察，不能当作完全受控的误差排名。

## 4. 逐任务成功与完成步数

单元格为 `success / episode_step`；失败行是终止步数，不是成功完成步数。

| 任务 | Dense | ToCa | WorldCache | ASI |
|---|---|---|---|---|
| BananaInBowlTask | True / 149 | True / 146 | True / 158 | True / 155 |
| BananaOnPlateTask | True / 145 | True / 226 | True / 125 | True / 145 |
| ButterAboveRaisinTask | True / 588 | False / 600 | True / 104 | True / 536 |
| BowlStackingLeftOnRightTask | True / 171 | False / 300 | True / 193 | True / 248 |
| GrabABagelTask | False / 450 | False / 450 | False / 450 | False / 450 |
| LargerObjectRaisinBoxInBinTask | False / 450 | False / 450 | False / 450 | False / 450 |
| MustardInLeftBinTask | False / 450 | False / 450 | False / 450 | False / 450 |
| RubiksCubeTask | True / 127 | True / 313 | True / 124 | True / 127 |
| RubiksCubeLeftOfBowlTask | True / 189 | False / 450 | False / 450 | False / 450 |
| MarkerInMugTask | True / 544 | True / 385 | False / 600 | False / 600 |

ASI 与 WorldCache 成功任务集合完全一致，均未完成两个普通任务，而 Dense 都完成。不能把 5/10 vs 5/10 解读为行为一致：例如 Butter 成功步数为 536 vs 104，BowlStacking 为 248 vs 193。成功样本平均步数 ASI 242.2、WorldCache 140.8；这不是 chunk 推理延迟。

`success` 与 `score` 按原字段独立统计：Butter 的 success=True/score=0 是官方结果，不自行修正；RubiksCubeLeftOfBowl 的部分 score≈0.6667 不算成功。

## 5. 讨论

1. **ASI 当前优势是计算延迟，不是已证明的成功率优势。** 本轮相对配对 Dense 减少约 31.75% 的稳定 chunk 时间（1.465×）。成功率与 WorldCache 相同、比 ToCa 多 1 个任务，但比 Dense 少 2 个。单 seed 十任务，且 ToCa 由该任务集调参选出，不足以支持泛化或统计显著性结论。
2. **ASI 当前没有保持完整未来视觉预测。** Latent cosine≈0.533、rel-L2≈0.897，RGB rel-L2≈0.395，不能称为基本无损。Action cosine≈0.9966 也不能单独证明动作保持：平均 action MSE≈0.01535，其中 MarkerInMug 单任务约 0.0910，误差分布明显不均匀。
3. **闭环成功与整体视觉保真度并非同一指标。** 本轮 ASI 和 WorldCache 成功集合相同，但整体视觉偏差相差明显。这支持继续分别评估任务行为与预测质量，不证明背景不重要或未来帧可以任意破坏。未选中 hidden 绕过 stack、使用输入恢复是解释视觉偏差的候选机制，尚未做分区误差或恢复方式消融，不能断言唯一原因。
4. **不能按“缓存几步”直接比较计算量。** WorldCache 只在 step3 缓存，但该步两个 CFG 分支都不执行 Transformer，真实完整 forward 从 8 降到 6。ToCa D/C/D/C 的缓存步仍执行全部 28 层，保留完整 K/V、实时 L0/action attention，以及部分 future MLP 更新，还有评分、选择和恢复成本。ASI 则在 7/8 次 forward 同时缩短 attention 与 MLP；这些不是同一种缓存单位，也不是等精度/等预算比较。
5. 下一步若要做严格策略结论，应固定同一批 observation/noise 同时测 Dense、ToCa、WorldCache、ASI，补全 ToCa action/latent 指标，并单独检查两个普通任务。这里只建议，不启动新实验。

## 6. 核验与数据入口

- ASI：10 个不同任务，118 个真实闭环请求；全部最终输出 finite，每请求 1 dense + 7 sparse stacks。
- GPU gate 对所有 28 层 Q/K/V/O/MLP 检查实际行数：Dense `[3093] × 8`，ASI `[3093] + [1845] × 7`，不是事后 mask。Gate 中间输出全部 finite，Dense 与已保存的 Dense 输出完全一致。
- 10 份 ASI 第三 chunk 重放均通过原闭环输出对照；600 个有效计时样本；FP64 指标无 NaN/Inf、cosine 均在 [-1,1]。
- 当前推理源码 SHA256 与运行 manifest 一致。未修改模型/选择逻辑；未新增推理、重跑失败任务、commit 或 push。

数据与原报告：

- [ASI summary](../experiments/asi_smoke10_s0_p0_v1/summary.json)、[逐任务结果](../experiments/asi_smoke10_s0_p0_v1/episodes.csv)、[配对测量](../experiments/asi_smoke10_s0_p0_v1/paired/)、[GPU gate](../experiments/asi_smoke10_s0_p0_v1/gpu_gate.json)、[manifest](../experiments/asi_smoke10_s0_p0_v1/manifest.json)。
- [ToCa 扫描汇总](../../worktrees/toca-future/experiments/toca_hparam_scan10_v1/summary.csv)、[固定配置](../../worktrees/toca-future/configs/toca_baseline.json)。
- [WorldCache summary](../../worktrees/worldcache/experiments/worldcache_dddc_smoke10_v2/summary.json)、[原报告](../../worktrees/worldcache/experiments/worldcache_dddc_smoke10_v2/report_cn.md)。

后台 `asi_smoke10:progress` 已正常结束（exit 0），保留窗口供查看，不代表任务仍在运行。
