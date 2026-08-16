# Core / Stable 参数选择工作流

## 1. 目标

目标不是在单个 Banana chunk 上找到最好看的参数，而是在满足动作精度和闭环能力的前提
下，找到稳定单 chunk 延迟最低的 Core / Stable / 总 token 组合。

当前假设是：**动作相关 Core 应适当增大，跨帧 Stable 环境区域可以减少。** 但 Core
本身来自 Action attention，仍可能包含角落和背景，因此不能只按 Core 数量越大越好。

最终采用分层筛选，而不是把所有参数同时网格搜索。

## 2. 固定项

第一轮只调整 Core 和 Stable，以下参数保持不变：

- `shift=5`，4 个 UniPC steps；
- Step 0 只有 conditional branch 全量；
- Step 0 unconditional 和 Step 1-3 均执行真实 packed-token 稀疏；
- block groups：G1=B0-B11、G2=B12-B19、G3=B20-B27；
- Core block range B4-B23、Core block count 6；
- Stable CV penalty、temporal smoothing、depth lookahead 不变；
- CFG 两个 branch 共用 mask，future frames 保持独立 mask；
- Torch compile 和 CUDA graphs 关闭；
- 相同输入、seed、initial noise、guidance 和 scheduler 状态。

## 3. 第一阶段：只改变 Core / Stable 分配

总 future-token budget 固定为 `K=(184,152,136)`。同时固定 Adaptive/Fill 的最大剩余
容量为 `(32,16,8)`，只把 Stable token 逐步转移给 Core：

| Candidate | Core | Stable G1/G2/G3 | Adaptive/Fill 剩余 | 总 K |
|---|---:|---:|---:|---:|
| A0：当前 V6-A | 64 | 88/72/64 | 32/16/8 | 184/152/136 |
| A1 | 80 | 72/56/48 | 32/16/8 | 184/152/136 |
| A2 | 96 | 56/40/32 | 32/16/8 | 184/152/136 |
| A3 | 112 | 40/24/16 | 32/16/8 | 184/152/136 |

这一阶段的真实计算 token 数完全相同，因此只比较表示质量和闭环行为，不比较加速。

Stable 均以 Core48 reference 为 forbidden scaffold 构造；扩大后的 Core 避开冻结 Stable。
这样能隔离 Core/Stable 配额变化。A0-A3 完成后，再对最优候选补做一次“按新 Core 重新
联合构造 Stable”的消融，确认冻结 scaffold 没有限制高分 Core。

## 4. 第二阶段：离线 mask 筛选

不运行模型，直接复用多个 task、chunk 的 Step-0 raw attention profile 重建候选 mask。
建议至少：

- 4 种任务：BananaInBowl、BananaOnPlate、RubiksCube、RubiksCubeAndBanana；
- 每个任务取早期、中期、后期各一个 chunk；
- tuning 与 validation chunk 分开，避免只适配某一轨迹。

每个候选必须先满足硬约束：

1. 每帧 token 数严格等于 K；
2. `G3 subset G2 subset G1`；
3. Core / Stable 不重叠；
4. conditional / unconditional 共用 mask；
5. 无 NaN/Inf，且 Adaptive 正分不足时只能由 Fill 补齐。

离线排序指标：

- Action-attention mass retention：报告 mean、P10、最差 task/chunk；
- Core 单独保留的 attention mass；
- 相邻 future-frame mask Jaccard；
- Stable token 的 temporal CV；
- Adaptive/Fill 实际数量；
- 边界与角落 token 比例，并人工检查 overlay。

淘汰规则：若候选只增加 Core 数量，却降低 validation attention retention、明显提高角落
占比，或让相邻帧 Jaccard 大幅下降，则不进入模型推理。

## 5. 第三阶段：配对单 chunk 精度

对离线筛选后的候选做 Baseline/V6-A/Candidate 配对推理。每个样本必须复用完全相同的
data batch、noise 和 RNG 状态。

主要指标：

- Action：cosine、relative-L2、逐 horizon/逐关节误差、max-absolute error；
- 动作动态：delta-action cosine、jerk cosine；
- Vision latent：cosine、relative-L2；
- 所有中间量 NaN/Inf 检查。

不要只看总体 action cosine。候选必须同时通过逐 horizon 误差和动作动态指标，否则可能
出现总体 cosine 很高、闭环却卡顿或用力错误。

建议以当前 V6-A 在多 task/chunk 上的分布作为门槛，而不是用单个固定数字：

- Candidate 的 action cosine P10 不低于 V6-A P10 超过 `0.001`；
- action relative-L2 P90 不高于 V6-A P90 的 `1.10x`；
- vision relative-L2 P90 不高于 V6-A P90 的 `1.10x`；
- 任一 task/chunk 出现明显 delta/jerk 退化则淘汰。

阈值是首轮工程门槛，积累更多闭环数据后再校准。

## 6. 第四阶段：压缩总 token

从 A0-A3 中选出 Core/Stable 分配最好的候选 `C* / S*`，再开始减少总 token。每次同步
减少 Stable 和总 K，保持 Core 以及 Adaptive/Fill 剩余容量 `(32,16,8)` 不变：

| Level | Core | Stable | 总 K |
|---|---:|---|---|
| T0 | C* | S* | 184/152/136 |
| T1 | C* | S* - 8 | 176/144/128 |
| T2 | C* | S* - 16 | 168/136/120 |

仅保留满足 `Core + Stable_g + AdaptiveReserve_g <= K_g` 的合法组合。若某组 Stable 已不足，
停止继续压缩，不把 Adaptive 自动压到零后仍宣称是同一实验。

这一步才能衡量真实 token 减少带来的精度/速度曲线。

## 7. 第五阶段：稳定单 chunk 计时

只对通过精度门槛的候选计时：

1. 同一个已加载模型；
2. 先全局预热；
3. 各模式再预热 3 次；
4. 使用相同 seeded input 随机交替测量至少 20 次；
5. CUDA synchronize 包围 `generate_samples_from_batch`；
6. 报告 median、P90 和 mean。

主速度指标只能使用上述稳定单 chunk。首请求、RPC、完整 task wall time只作为辅助信息。

## 8. 第六阶段：闭环晋级

按成本从低到高：

1. **3-task smoke**：BananaInBowl、RubiksCube、RubiksCubesInBin；
2. **9-task 单 seed**：使用 seed 579362556，与 V6-A 完全配对；
3. **9-task 五 seed**：0、1、42、1234、579362556，共 45 episodes；
4. 只让最多两个候选进入五 seed 测试。

统计 `episode_results.jsonl.success`，同时保存完成 steps、错误事件和任务难度分组。一个
task 一个 episode 只能形成 pooled success rate，不能当作稳定 per-task 成功率。

## 9. 最终选择规则

采用分层 Pareto 选择，不把所有指标混成一个任意加权总分：

1. 先满足 mask 正确性；
2. 再满足配对 action/vision 精度门槛；
3. 再要求闭环 success 不出现明确退化；
4. 在剩余候选中选择稳定单 chunk median 最低者；
5. 若速度差小于 1%，优先选择 P90 更低、闭环轨迹更稳定、Stable 更多的保守候选。

## 10. 推荐立即执行的顺序

1. 离线生成 A0-A3 的多 task/chunk mask 和 overlay；
2. 淘汰角落占比高或 attention retention 明显下降的候选；
3. 对剩余候选做多 task/chunk 配对输出验证；
4. 选 1-2 个候选做 T0/T1/T2 token 压缩；
5. 对 Pareto 前沿做稳定单 chunk timing；
6. 最后才跑 3-task、9-task 和五 seed 闭环。

第一轮不调整 Core block count、block quality 公式、Stable CV penalty 或 temporal smoothing。
这些属于第二轮算法参数，必须在 Core/Stable 配额确定后再单独消融。

## 11. 2026-08-16 双简单任务首轮筛选

固定 `K=(184,152,136)`、环境 seed 0、policy seed 579362556，在
`BananaInBowlTask` 与 `RubiksCubeTask` 上各跑一个 episode：

| 候选 | Core | Stable | 闭环结果 |
|---|---:|---:|---:|
| A0 | 64 | 88/72/64 | 1/2 |
| A1 | 80 | 72/56/48 | 1/2 |
| A2 | 96 | 56/40/32 | **2/2** |
| A3 | 112 | 40/24/16 | 1/2 |

本轮仅用于筛选。A2 是唯一完成两个任务的候选，说明扩大 Core 的收益在当前样本上不是单调的：
Core112 同时把 Stable 压到 40/24/16 后，RubiksCube 再次失败。下一轮优先复核 A2，暂不继续
增加 Core；完整报告位于
`experiments/preliminary/sparsity/velocity_cache/core_stable_alloc_2simple_seed579362556_v1/report_cn.md`。
