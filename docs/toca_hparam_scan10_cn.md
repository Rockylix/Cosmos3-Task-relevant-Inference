# ToCa-Future 超参扫描：8 简单 + 2 普通任务

2026-09-10 更新：用户已停止后续扫描，并选定 `D/C/D/C, r=0.25, bonus=0, CFG independent` 作为固定 ToCa 对照。
后续运行请使用 [固定版本配置与结果](toca_fixed_baseline_cn.md)，不要继续遍历本页的历史扫描网格。旧结果保留，`toca_scan10` 已关闭。

## 范围与固定设置

本轮只扫描 ToCa-Future，不更改 ASI 和 Dense 工作树。分支为 `experiment/toca-future`，工作树为 `/root/robolab/worktrees/toca-future`。

| 因子 | 取值 |
|---|---|
| 去噪调度 | D/C/D/C、D/C/C/C |
| 空间 bonus | 关闭；开启（2×2 局部最高分 ×1.6） |
| 基础 fresh ratio r | 0.1、0.25、0.5 |
| CFG 选择 | shared；independent |

共 `2×2×3×2=24` 个 ToCa 配置，加一次 Dense 对照。每配置相同 10 任务各运行一次，共 **250 episodes**。D 表示全量计算，C 表示缓存计算；不重试失败，不按结果筛选任务。

固定：Edge/DROID 官方权重，shift=5、4 denoise steps、guidance=3、原生 eager，关闭 compile/CUDA graphs；policy seed=0、deterministic_seed=False，每个任务开始重置 policy RNG。RoboLab 使用本地当前评估路径的环境 seed=0、单环境、每任务一个 rollout，沿用各任务官方步数上限。具体运行配置保存在各任务 `env_cfg.json`。

## 任务列表

难度从本地 RoboLab `task_metadata.json` 核验，不根据本轮结果重新划分。

| 难度 | 任务 |
|---|---|
| simple | BananaInBowlTask |
| simple | BananaOnPlateTask |
| simple | ButterAboveRaisinTask |
| simple | BowlStackingLeftOnRightTask |
| simple | GrabABagelTask |
| simple | LargerObjectRaisinBoxInBinTask |
| simple | MustardInLeftBinTask |
| simple | RubiksCubeTask |
| moderate | RubiksCubeLeftOfBowlTask |
| moderate | MarkerInMugTask |

## 计算约束

仅 future vision 参与选择；condition L0、全部 action token 保留。缓存步复用 future attention 输出，future MLP 仅重算选中位置；K/V 保持完整上下文。不是把整个 Transformer 的有效长度统一减少到 K。

每帧 17×20=340 个空间 token，8 个 future latent 共 2720。GEN 共 3093 token（L0 340 + future 2720 + action/state 33）。B_l 的 fresh token 数为：

`K_l = floor(2720 * r * (1.5 - l/27)), l=0..27`。

空间 bonus 在 incoming-attention score 的 L2 normalization 和 age 项之后应用：每帧独立、不跨帧的 2×2 窗口中最高分乘 1.6，然后做相同预算 Top-K。17 行边界采用有效位置掩码，补齐位置不参与选择。bonus 不增加 K。

shared 对两个 CFG 分支的原始 incoming score 求均值后评分，同一步共用 fresh indices，age 更新一次。independent 分别维护 score、age 和 indices；两个分支的缓存张量始终分开。D/C/D/C 与 D/C/C/C 的 age period 分别为 2 与 4，age weight 固定 0.25，layer slope 固定 0.5。

## 指标及公平性

闭环只统计每配置真实返回策略 action 后的 `episode_results.jsonl`：

- Success：10 个 episodes 中 `success=True` 的数量与比例；不能用 score 替代。
- Score：10 个 episodes 的平均 score，同时保留逐任务结果以核查。
- 不保存仿真视频，不额外输出预测图片、hidden state 或 Q/K/V。

稳定单 chunk 时间与 future-frame 误差单独配对测量，不比较已经分叉的闭环观测：

1. Dense 每任务临时保留第 3 个 chunk 的同一组输入、seed、参数和输出；不足 3 个时取最后一个。
2. 在同一个加载好的模型中，25 个配置对同一输入运行。每次隔离输入修改、重置 RNG 和 ToCa controller/cache，正常重新建立 scheduler。
3. 每配置预热 5 次，之后 30 轮轮换运行顺序，CUDA 同步计时。计时包括 generation、ToCa 评分/bonus/选择/恢复及有效性检查，不包括输入 CPU 拷贝、VAE decode、文件 I/O、模型加载和仿真。
4. 每任务分别计算 mean、median、P90；汇总表对 10 个任务的相应统计量等权平均。`speedup = mean_task(Dense median) / mean_task(ToCa median)`，不是混合请求的全局 median。
5. 相同输入与 seed 下各配置的最终 latent 经相同 VAE decode。使用固定 `clip((x+1)/2,0,1)`，排除 condition 帧，比较后 32 张预测 RGB 帧。RGB cosine/relative-L2 对整个 future RGB tensor 展平；PSNR/SSIM 先逐帧再平均，最后等权聚合任务。Dense 自比较 PSNR 为无穷大，在 JSON/CSV 中记 null 并在逐任务 JSON 标注 `psnr_infinite`。

此设计测量单 seed 的超参筛选效果，不支持多 seed 泛化或任务成功率不变的结论。

## 验证

正式启动前完成：

- 30 项自动测试通过，包括真实 GPU joint-attention 与参考计算校验、token selection、空间 bonus、CFG 独立 age 和索引、图像误差函数。
- Dense + 24 个 ToCa 配置全部通过 GPU gate，检查实际 Q/K/V/O/MLP 输入行数和 NaN/Inf。
- 默认 D/C/D/C、r=0.25、无 bonus、共享 CFG 的输出与已有保存结果逐元素完全一致。
- gate 文件：`experiments/toca_hparam_scan10_v1/validation/summary.json`。未通过 25 项 gate，队列拒绝启动。

## 执行与查看

使用现有项目环境，不下载环境或权重。正式队列按 Dense 10任务 → 同输入配对计时/预测帧误差 → 24组 ToCa 闭环运行。

```bash
cd /root/robolab/worktrees/toca-future
PYTHONPATH="$PWD:/root/robolab/cosmos-edge-overlay" \
  /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -u tools/run_toca_scan.py \
  --run-dir experiments/toca_hparam_scan10_v1 --port 8019 --resume
```

本轮由 tmux 会话 `toca_scan10` 托管，查看：

```bash
tmux attach -t toca_scan10
```

界面打印当前配置、完成数、成功数和 score，以及配对计时阶段进度。异常停止不自动重试；断点续跑仅跳过已有完整 episode 结果，不会重新尝试已记录的失败。

## 输出与保留规则

目录：`experiments/toca_hparam_scan10_v1/`。

- `summary.csv`、`report_cn.md`：实时汇总；未完成项留空，不能视为 0。
- `manifest.json`：任务、配置、seed、步数上限、源文件 SHA256，防止断点续跑混入新代码。
- `progress.json`、`queue.log`：阶段/进度；最终只有 `phase=complete` 且 250 个结果齐全才视为完成。
- `paired/<Task>.json`：每任务、每配置的 timing 与 RGB 误差标量。
- `<mode>/simulator/episode_results.jsonl`：真实逐任务成功与 score。
- `<mode>/attempt_*/`：启动命令、小体积请求计时、运行配置、必要错误日志。
- `validation/summary.json`：预检证据；`cleanup.jsonl`：本轮临时数据清理记录。

只在本轮目录内临时保存 10 个 Dense 输入，配对指标持久化后删除；RoboLab 为计算 score 生成的本轮 HDF5/逐步 JSON 在结果写完且进程退出后删除。删除的原始张量不保留恢复副本，任务/配置/seed/逐 episode 结果与指标仍保留。既有实验不动。
