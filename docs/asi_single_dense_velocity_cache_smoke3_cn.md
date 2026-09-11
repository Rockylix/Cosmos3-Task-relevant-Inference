# ASI 单次全量 + velocity cache：三任务与逐 chunk Top-6

## 实测结果（2026-09-11）

3 episodes、15 个 chunk 已完成；各 chunk 都通过 1 dense / 7 sparse / 4 次 sampler 回调校验。服务器和仿真器已退出，未残留计算进程。

| 任务 | success | score | 步数 | chunk 数 | generation median(s)* |
|---|---:|---:|---:|---:|---:|
| BananaInBowlTask | True | 1 | 160 | 5 | 0.593438 |
| BananaOnPlateTask | True | 1 | 146 | 5 | 0.591916 |
| ButterAboveRaisinTask | True | 0 | 139 | 5 | 0.591353 |

按 RoboLab `success` 为 3/3，平均 score=0.666667。Butter 原始结果为 `success=True`、`score=0`，reason 为 `object_grabbed(object=butter)` 并推进到 step1；不把它自行改成满分，也不能据此称放置子任务全部完成。

*排除每任务 chunk1；其余四个 chunk 的 median。仿真器与服务器共享 GPU，不是独占、交替的稳定 benchmark，不用它与历史秒数推算加速比。

15 个 chunk 的入选次数：B18、B21 各 15/15；B15、B19、B20、B23 各 14/15；B22 为 4/15；其余 21 层为 0/15。B21 在 13/15 个 chunk 排第 1。两种香蕉任务各出现 3 种 Top-6 集合；黄油盒任务 5 个 chunk 集合不变，但排名变化。

| 任务 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 |
|---|---|---|---|---|---|
| BananaInBowlTask | 21,15,19,20,23,18 | 21,20,18,19,22,23 | 21,15,20,18,19,23 | 21,18,19,20,15,22 | 21,20,15,18,19,23 |
| BananaOnPlateTask | 21,15,19,18,20,23 | 21,20,23,18,15,22 | 21,15,20,18,19,23 | 21,18,19,15,23,22 | 21,15,18,19,20,23 |
| ButterAboveRaisinTask | 21,19,15,18,20,23 | 21,19,18,20,15,23 | 15,21,19,18,23,20 | 18,19,21,15,23,20 | 21,15,19,18,20,23 |

每格按 Q 降序，数字为 0-based block ID。已保存 90 个入选记录、420 个全层 R/H/Q 记录，源码哈希与运行 manifest 一致。

质量边界：独立 Banana c3 GPU gate 中，新缓存与原 ASI 的 action、保留区域 latent 逐元素相同；但**完整 vision latent（含 L0）相对原 ASI 的 cosine=0.066930、relative-L2=2.843428**，差异来自被移除位置。本轮未 VAE 解码评估，不能声称图像保持或沿用上一轮双分支全量缓存的质量改善。闭环成功与想象帧质量需要分开评价。

环境备注：新旧 ASI 仿真日志都出现 Warp `cuDeviceGetUuid` 初始化错误；本轮三 episodes 正常完成，未触发断线重试。本实验没有处理该既有环境问题，也不称环境日志完全无警告。

## 本轮定义

用户确认任务：`BananaInBowlTask`、`BananaOnPlateTask`、`ButterAboveRaisinTask`。每任务 1 episode；simulator seed=0、policy seed=0、`deterministic_seed=False`，每任务重置请求 RNG；4 steps、shift=5、guidance=3，官方步数上限，不重跑失败任务，不保存仿真视频。compile/CUDA graphs 关闭。

本轮只保留 **1 次全量 + 7 次稀疏 CFG forward**，不是上一轮的双分支全量 Step0 版本：

1. Step0 conditional：正常全量 28 个 block，同时得到原 ASI 的 action attention profile。
2. 按原 ASI 的 `Q_l=R_l(1-H_l)` 选 Top-6 来源层，选择 Core80、Stable104；每个 future latent 保留 184/340 token。权重与原方案不变：action 四个 horizon 为 `[1/6,1/3,1/3,1/6]`。
3. Step0 unconditional：仍然稀疏执行全部 28 个 block；保持原 position IDs，不把背景重新放回 attention。
4. 原生输出头、条件 mask、CFG 完成后，缓存该 **ASI Step0 guided velocity** 的被移除 future 坐标。
5. Step1/2/3 双分支都使用本 chunk 的同一 mask 稀疏执行；原生 CFG 后用缓存替换被移除区域的 velocity，再由原生 UniPC 积分。
6. 下一个 chunk 重新 profile、选 Top-6、选 mask、建 velocity cache，不跨 chunk 复用。

令 M 为保留位置，则：

\[
\widetilde v_0=v_0^{ASI},\qquad
\widetilde v_s=M\odot v_s^{ASI}+(1-M)\odot v_0^{ASI},\quad s=1,2,3.
\]

`v_0^{ASI}` 来自全量 conditional + 稀疏 unconditional 的**原生 CFG 公式**；不是仅 conditional prediction，也不是 `v_0^{Dense}`。因此不能沿用上一次双分支全量缓存实验的图像指标。

L0 与 action 不缓存；被移除位置按真实 2×2 patch 足迹映射到 33×40 latent，裁掉 padding，不做双线性缩放。被移除 hidden 仍按原 ASI 在 stack 末尾恢复；缓存仅在 CFG→UniPC 边界替换 velocity，不额外相加、不修改求解器历史。

## 轻量路径与验证

- `CacheSamplerAdapter(step0_source="asi_cfg", audit=False)` 明确拒绝 `dense_step0=True`，在首次 velocity 返回时核验计数为 1 dense / 1 sparse，最终为 1 dense / 7 sparse。
- mask 转为被移除位置的索引，只在 Step0 建一次；`index_select` 保存缓存，后三步 `clone + index_copy_` 替换。没有逐步 CPU velocity 拷贝或逐层输出审计。
- 每个 chunk 正常完成 generation 后检查 action/vision finite，再结束计时；R/H/Q 转 CPU 与 CSV I/O 在计时范围之外。
- GPU gate 独立执行，不计入三个闭环任务或 chunk 统计：原 ASI 与旧 capture 逐元素一致；新缓存的快路径与审计路径逐元素一致；所有 28 层 Q/K/V/O/MLP 的真实 GEN 行数为 `[3093]+[1845]×7`，224 个 block 输出 finite。
- 17 项 CPU 测试通过，含单次全量缓存的真实 UniPC 与独立显式参考逐元素一致；审计开/关两条路径都覆盖。

## 每个 chunk 的统计

本轮 Top-6 指 **Core score 的来源层**，不是只执行这六层，也不是每个 denoise step 都重选。每个 chunk 只根据 Step0 conditional 重选一次。

输出目录：`../experiments/asi_1d7s_velocity_cache_smoke3_s0_p0_v1/`。

- [结果报告](../experiments/asi_1d7s_velocity_cache_smoke3_s0_p0_v1/report_cn.md)
- [每 chunk 的 Top-6，按 Q 降序](../experiments/asi_1d7s_velocity_cache_smoke3_s0_p0_v1/chunk_topk_layers.csv)
- [每 chunk 全部 28 层的 R/H/Q](../experiments/asi_1d7s_velocity_cache_smoke3_s0_p0_v1/chunk_block_scores.csv)
- [按任务和全局汇总的入选次数/频率](../experiments/asi_1d7s_velocity_cache_smoke3_s0_p0_v1/layer_selection_frequency.csv)
- [逐 chunk 入选层热力图](../experiments/asi_1d7s_velocity_cache_smoke3_s0_p0_v1/chunk_top6_layers.png)
- [原始请求日志](../experiments/asi_1d7s_velocity_cache_smoke3_s0_p0_v1/requests.jsonl)、[GPU gate](../experiments/asi_1d7s_velocity_cache_smoke3_s0_p0_v1/gpu_gate.json)

`rank_1..rank_6` 按 Q 从高到低；所有 block ID 从 0 开始。入选率的分母是该任务 chunk 数，全局 pooled 按 chunk 加权；报告同时保留每任务统计，不把 chunk 较多的任务隐含当成任务等权。

闭环 generation 时间与仿真器共享 GPU，不能与历史独占 GPU benchmark 秒数直接相除得出加速比。成功率以 `success` 字段为准，`score` 单独报告。

## 运行与离线汇总

```bash
cd /root/robolab/worktrees/asi-velocity-cache
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 COSMOS_TRAINING=0 LD_LIBRARY_PATH='' \
  PYTHONPATH="$PWD:/root/robolab/cosmos-edge-overlay" \
  /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -u tools/run_asi_velocity_cache_smoke3.py \
  --output experiments/asi_1d7s_velocity_cache_smoke3_s0_p0_v1 --port 8029

/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  tools/summarize_asi_velocity_cache_smoke3.py \
  --run experiments/asi_1d7s_velocity_cache_smoke3_s0_p0_v1
```

运行器使用原项目环境、本地权重与原生 RoboLab 入口；完整 server/simulator 命令保存在结果目录 `commands.json`。目录存在时拒绝覆盖，重跑需要唯一新后缀并保留原结果。只清理本运行器启动的进程，不停止其他实验。

实验保留在 `experiment/asi-velocity-cache` worktree，未修改原 ASI 工作树或 baseline，未 commit/push/merge。
