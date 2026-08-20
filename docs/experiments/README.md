# RoboLab / Cosmos3 Edge 文档索引

文档按用途分类。Baseline 实验文档提交到本目录；生成数据统一写入仓库根目录下被 Git
忽略的 `experiments/`。

## 环境配置

- [RoboLab 官方仿真环境配置](setup/cosmos3_robolab_official_setup_cn.md)

## 性能分析

- [Cosmos3 Edge eager baseline NVTX](profiling/cosmos3_edge_baseline_nvtx_cn.md)

## 数据采集

- [Hidden-state 采集实验](capture/cosmos3_edge_hidden_state_experiment_cn.md)
- [RoPE 前后 Q/K 采集实验](capture/cosmos3_edge_rope_qk_experiment_cn.md)

## Attention 分析

- [Round 1 attention mass / enrichment / entropy](attention/cosmos3_edge_attention_stats_round1_cn.md)
- [Value-aware attention](attention/cosmos3_edge_attention_value_round1_cn.md)
- [真实 attention output](attention/cosmos3_edge_attention_true_output_cn.md)
- [Action Query attention/output](attention/cosmos3_edge_action_query_attention_output_shift1_shift5_cn.md)
- [空间 token 相似度](attention/cosmos3_edge_spatial_token_similarity_shift5_cn.md)
- [逐相邻帧空间 token 相似度](attention/cosmos3_edge_spatial_token_similarity_pairs_cn.md)

## Q/K 复用与跳算

- [Q/K reuse profile](qk_reuse/cosmos3_edge_qk_reuse_profile_cn.md)
- [Q/K reuse intervention](qk_reuse/cosmos3_edge_qk_reuse_intervention_cn.md)
- [Q/K real skip](qk_reuse/cosmos3_edge_qk_real_skip_cn.md)
- [Q/K plan sweep](qk_reuse/cosmos3_edge_qk_plan_sweep_cn.md)

## Residual 与任务评测

- [Block residual 全量分析](residual/cosmos3_edge_block_residual_full_analysis_cn.md)
- [Shift 1/5 多任务评测](evaluation/cosmos3_edge_shift1_shift5_20tasks_eval_cn.md)

实验产物索引见 [ARTIFACT_INDEX.md](ARTIFACT_INDEX.md)，分类规则见
[PRELIMINARY_EXPERIMENTS.md](PRELIMINARY_EXPERIMENTS.md)。
