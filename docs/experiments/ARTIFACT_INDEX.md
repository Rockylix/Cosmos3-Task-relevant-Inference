# RoboLab 实验产物索引

整理日期：2026-08-06。

## 命名规则

- 项目身份由 `/root/robolab` 上层路径表达，实验族和 run 名不再添加 `cosmos3_edge_` 前缀。
- 除非 Edge 本身是对照变量，新 run 也不使用泛化的 `edge_` 前缀。
- 新 run 统一使用 `<主题>_<关键变量>_<任务或数据>_vN`。
- 当前已有的 legacy run ID 暂时保留，避免破坏报告、CSV 和脚本的历史引用。

示例：

```text
attention_mass_enrichment_banana_c3_shift1_shift5_v2
hidden_spatial_cosine_banana_c3_shift1_b0_8_16_24_v2
fis_interleaved_s3_banana_c3_v2
```

## 目录结构

所有探索性实验统一归档在 `preliminary/`：

```text
preliminary/
├── profiling/       # NVTX、Nsight Systems、耗时基线
├── representation/  # hidden、residual、RoPE、空间 token 与离线分析
├── attention/       # attention mass/value/output/action-query
├── qk_optimization/ # Q/K profile、复用、真实跳算、计划扫描
├── sparsity/        # FIS 与帧级/联合模态稀疏
├── evaluation/      # 闭环任务成功率等评测
└── _archive/        # 被新版替代但仍保留的历史版本
```

详细分类说明见 [PRELIMINARY_EXPERIMENTS.md](PRELIMINARY_EXPERIMENTS.md)。

## 大型原始数据

以下数据仍被离线分析依赖，未经确认不要删除：

- `preliminary/representation/hidden_states/edge_hidden_banana_bowl_plate_c357_v1/`：约 14 GB。
- `preliminary/representation/hidden_states/edge_hidden_BananaInBowlTask_c3_shift1_v1/`：约 2.7 GB。
- `preliminary/representation/hidden_states/edge_hidden_BananaOnPlateTask_c3_shift1_spatial_v1/`：约 2.7 GB。
- `preliminary/representation/block_residuals/edge_block_residual_BananaInBowlTask_c3_shift5_v1/`：约 2.4 GB。

上次归档与删除明细见
[`experiments/preliminary/_archive/2026-08-06/MANIFEST.md`](../../experiments/preliminary/_archive/2026-08-06/MANIFEST.md)。

原始数据实际位于 `/root/robolab/cosmos-framework-edge/experiments/preliminary/`，该目录由
Git 完整忽略。索引和关键实验结论只在 `docs/experiments/` 中提交。
