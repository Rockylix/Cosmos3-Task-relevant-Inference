# 任务首个 chunk 固定 Core Top-7，Stable 保持全局

## 当前行为

用户确认：**Stable 仍按当前 chunk 的全部 B0–B27、八个 future latent 全局评分选取**；不改为仅使用七层。

新增 `--task-core-top7`，默认关闭，旧的每 chunk Top-6 模式保留。

| 内容 | 任务首个 chunk | 同任务后续 chunk |
|---|---|---|
| Step0 conditional | 全量 forward + 28 层 attention profile | 全量 forward + 28 层 attention profile |
| R/H/Q | 当前 28 层重新计算 | 当前 28 层重新计算，供全局 Stable 使用 |
| Core 来源层 | 对 Q 做一次 Top-7，缓存层编号和顺序 | 直接使用首次七层，不再做层级 Top-K |
| Core 权重和 Core80 token | 当前 chunk 重算 | 在固定七层中按当前 Q 重新加权、重新选 token |
| Stable104 token | 当前 28 层全局评分、排除 Core union 后选取 | 同样重新计算，不缓存 Stable mask |
| 其余 CFG forward | 7 次真实稀疏 | 7 次真实稀疏 |
| Velocity cache | 保存本 chunk 的 ASI Step0 guided velocity | 为当前 chunk 重新生成缓存，不跨 chunk 复用 |

只有七个 **block ID** 跨 chunk 保存；不缓存 attention map、Core/Stable mask、hidden 或 velocity。每帧仍保留 Core80 + Stable104 = 184/340 个 token，L0 和全部 action 保留。全部 28 个 block 继续执行；这里缓存的是 **Core score 的来源层**，不是只执行七层 Transformer。

## 数学口径

设 c 为任务内 chunk，U 是原 action-aligned attention profile，Q 沿用原实现：

\[
R_l^{(c)}=\frac18\sum_f\sum_pU_{l,f,p}^{(c)},\qquad
Q_l^{(c)}=R_l^{(c)}(1-H_l^{(c)}).
\]

每帧对应四个 action horizon 仍使用 `[1/6,1/3,1/3,1/6]`。H 是八帧的空间归一化熵均值。

任务首次只选择一次：

\[
\mathcal B=\operatorname{Top7}_l(Q_l^{(1)}).
\]

后续不重新排序层，但七层权重仍随 chunk 更新：

\[
w_l^{(c)}=\frac{Q_l^{(c)}}{\sum_{j\in\mathcal B}Q_j^{(c)}+\epsilon},\quad
C_{f,p}^{(c)}=\sum_{l\in\mathcal B}w_l^{(c)}U_{l,f,p}^{(c)}.
\]

Stable 的全局分数仍是：

\[
G_{f,p}^{(c)}=\sum_{l=0}^{27}\frac{Q_l^{(c)}}{\sum_{j=0}^{27}Q_j^{(c)}+\epsilon}U_{l,f,p}^{(c)},\qquad
S_p^{(c)}=\frac{\operatorname{mean}_fG_{f,p}^{(c)}}{1+\operatorname{CV}_f(G_{f,p}^{(c)})}.
\]

实现沿用原分母 `clamp_min(eps)` 与零全局权重时均匀回退，不更改这些数值边界。Stable104 在当前 Core union 之外选择，因此全局 Stable **评分公式不变**，但 Core 从六层变成七层后，排除集合变化可能使最终 Stable mask 也变化。

## 实际节省边界

**本方案不减少 28 层 attention profile，也不省掉全局 R/H/Q。** Stable 每个 chunk 仍需要它们。省掉的只是后续 chunk 在 28 个 Q 值上的 Core 层 Top-7 排序，以及相应的新层编号提取；token-level Top-K 仍正常执行。

因此不能宣称 profile 从 28 层变为 7 层、节省 75%，也不应预期仅凭这项修改获得明显 chunk 加速。没有测独占、预热交替的速度；后续三任务闭环结果见下节。

## 状态生命周期

- 串行服务中，原始 task prompt 改变时清空层缓存，重置 task 内 chunk 计数与原请求 RNG。
- A→B→A 会为三个任务片段各重新选一次，不会捡回旧 A 的缓存。
- 层编号只在一次 generation 完成且输出 finite 后提交；失败的首次请求不提交半成品。
- 当前原生 RoboLab 请求不带 episode ID，连续同 prompt 的不同 rollout 无法仅从 prompt 自动区分。该模式目前用于串行不同任务；同名任务重新开始时需重启服务或显式调用 `TaskCoreLayerCache.begin_task(task, reset=True)` 并重置服务的 task/chunk 状态。不要把多任务并发请求混入同一状态。

## 验证结果（2026-09-11）

- 26 项 CPU 测试通过。覆盖旧 Top-6 默认行为、七层只选一次、权重/mask 刷新、Stable 继续依赖未入选层、184 token 预算和 Core/Stable 不交叠、任务切换/显式重置、缓存失败不提交。
- GPU 回归使用既有 BananaInBowl / BananaOnPlate **chunk3 输入**模拟 A1、A2、B1、A1 生命周期；这是请求重放，不是实际任务首个 chunk 的闭环采样。
- 四次 Core 层级 Top-K 调用数实测为 **`[1,0,1,1]`**；四次 profile 层数均为 **`[28,28,28,28]`**。
- 每次真实 Q/K/V/O/MLP GEN 行数均为 `[3093]+[1845]×7`；审计的 224 个 block 输出 finite，快速路径 action/vision 与同设置审计路径逐元素一致。
- 保留当前 velocity cache 的语义，不声称预测图像保真度提高。

[GPU 验证数据](../experiments/task_core_top7_global_stable_gpu_verify_v1/verification.json)；该目录 `gpu_gate.json` 是旧 Top-6 模式兼容性校验，Top-7 的四次验证在 `verification.json` 中。

## 使用

三任务运行器新增开关，以下命令已用于本页记录的闭环测试。输出目录已存在，不要覆盖；再次运行需使用新的输出目录：

```bash
cd /root/robolab/worktrees/asi-velocity-cache
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 COSMOS_TRAINING=0 LD_LIBRARY_PATH='' \
  PYTHONPATH="$PWD:/root/robolab/cosmos-edge-overlay" \
  /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -u tools/run_asi_velocity_cache_smoke3.py --task-core-top7 \
  --output experiments/task_core_top7_global_stable_smoke3_s0_p0_v1 --port 8029

/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  tools/summarize_asi_velocity_cache_smoke3.py \
  --run experiments/task_core_top7_global_stable_smoke3_s0_p0_v1
```

运行器自动传递 `--task-core-top7` 到 policy service，manifest 同步记录配置；直接启动服务时也要传同名开关和相符的 manifest。每个 chunk 日志包含七层编号、是否复用、当前七层权重和当前全 28 层 R/H/Q。

`chunk_topk_layers.csv` 的 `task_initial_rank_1..7` 保留首次层排序，不是当前 Q 的重新排名；`chunk_block_scores.csv` 中的 `current_core_weight` 是当前权重。汇总图为 `chunk_top7_layers.png`。

所有修改在 `experiment/asi-velocity-cache` worktree；原 ASI 和 baseline 未修改，未 commit、merge 或 push。

## 三任务闭环结果（2026-09-11）

使用上面的命令，各任务只运行一次，不重试失败 episode。Simulator seed=0、policy seed=0、`deterministic_seed=False`，任务切换重置 policy RNG，4 steps、CFG=3、shift=5，关闭 compile、CUDA graphs 和视频保存。

| Task | Success 原始字段 | Score | Episode steps | Chunks | Generation median (s)* |
|---|---:|---:|---:|---:|---:|
| BananaInBowlTask | True | 1.0 | 146 | 5 | 0.591975 |
| BananaOnPlateTask | True | 1.0 | 130 | 5 | 0.593393 |
| ButterAboveRaisinTask | True | 0.0 | 125 | 4 | 0.593839 |

原始 success **3/3**，平均 score **0.666667**。Butter 原始 reason 是 `success: object_grabbed(object=butter). advanced 1 step(s) to step 1 for butter.`，不能由此宣称完整放置子任务得分为 1。两个香蕉任务的 reason 均明确记录完成放置子任务。

*逐任务排除首个 chunk；11 个剩余 chunk 的 pooled median 为 **0.592954 s**。测量包含 generation、profile/选 token、velocity cache 和最后有限值检查，不含 score 转 CPU/CSV I/O。仿真器同时占用 GPU，这不是独占 GPU 的稳定配对 benchmark；本轮没有 Dense 对照，不报告加速比。

首个 chunk 选出的层，按首次 Q 降序：

- BananaInBowl：B21、B15、B19、B20、B23、B18、B12。
- BananaOnPlate：B21、B15、B19、B20、B18、B23、B22。
- ButterAboveRaisin：B21、B19、B15、B18、B20、B23、B22。

全部 **14/14 chunks** 的检查通过：3 次任务首次选层、11 次复用；每个任务始终只有一套七层编号；Stable profile 均为全部 28 层；每帧 Core80+Stable104；每 chunk 1 次全量、7 次稀疏、4 次缓存速度处理；action/vision 最终值 finite。运行前后源码 SHA256 一致。任务完成后本轮服务、仿真进程已退出，8029 端口和 GPU 计算进程已释放。

[完整实验报告](../experiments/task_core_top7_global_stable_smoke3_s0_p0_v1/report_cn.md) · [任务指标 CSV](../experiments/task_core_top7_global_stable_smoke3_s0_p0_v1/task_results.csv) · [逐 chunk 选层 CSV](../experiments/task_core_top7_global_stable_smoke3_s0_p0_v1/chunk_topk_layers.csv) · [原始 episode 结果](../experiments/task_core_top7_global_stable_smoke3_s0_p0_v1/simulator/episode_results.jsonl)。无仿真视频和大张量保存，输出约 1.2 MB。
