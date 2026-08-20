# Cosmos3 Edge Q/K 真跳算：路径扩展与优化实验

## 结论

已经把 profile 选用路径从 9 对扩展到 15、37、76 对，并优化索引准备、稀疏投影和批量恢复。目标 future token 不参与 Q/K Linear、QK Norm 和 RoPE；V 与 attention 仍按完整 token 计算。

优化后，所选 Q/K 路径本身在四档方案中均加速约 17%–39%（以 dense/sparse 比值定义）；但前三档只跳过全 DiT Q/K token 的 0.88%–3.63%，完整 chunk 没有稳定加速。压力档覆盖 7.46% 时，两个 chunk 均出现约 0.30%–0.36% 的完整 chunk 时间下降，说明跳过 Q/K 确实有效，但收益上限受 Q/K 在总计算中的占比以及 gather/restore 开销限制。

压力档只用于性能上限，不应视为质量安全策略。当前最适合作为后续质量实验起点的是保守档；如果目标是继续寻找可观加速，需要扩大可跳 block/step，或进一步融合 gather + projection + restore，而不是只放宽相似度阈值。

## 路径覆盖与输出误差

| 方案 | pair/branch | step-block | 全 DiT Q/K 跳过 | chunk | action rel-L2 | action cosine | future vision rel-L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 保守 p10>=0.90 | 9 | 3 | 0.88% | 3 | 0.01072 | 0.999943 | 0.09914 |
| 保守 p10>=0.90 | 9 | 3 | 0.88% | 5 | 0.00951 | 0.999957 | 0.07281 |
| 扩展 mean>=0.90 | 15 | 4 | 1.47% | 3 | 0.01309 | 0.999924 | 0.19934 |
| 扩展 mean>=0.90 | 15 | 4 | 1.47% | 5 | 0.02400 | 0.999732 | 0.13248 |
| 激进 p10>=0.85 | 37 | 16 | 3.63% | 3 | 0.02878 | 0.999663 | 0.22369 |
| 激进 p10>=0.85 | 37 | 16 | 3.63% | 5 | 0.02989 | 0.999560 | 0.25043 |
| 压力 full-cos>=0.85 | 76 | 26 | 7.46% | 3 | 0.04388 | 0.999342 | 0.29702 |
| 压力 full-cos>=0.85 | 76 | 26 | 7.46% | 5 | 0.03035 | 0.999561 | 0.31563 |

## 真实 CUDA 计时

每个 chunk/方案先预热，再交替运行 Baseline/Sparse 5 次。局部 Q/K 时间包含索引裁剪、稀疏 Q/K projection、QK norm、RoPE 和恢复；Dense-Control 使用相同 hook 和计时边界但不跳 token。正值表示 Sparse 更快。

| 方案 | chunk | Dense Q/K (ms) | Sparse Q/K (ms) | 局部加速 | Baseline chunk (ms) | Sparse chunk (ms) | chunk 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 保守 p10>=0.90 | 3 | 3.275 | 2.583 | 26.81% | 848.888 | 851.425 | -0.30% |
| 保守 p10>=0.90 | 5 | 3.324 | 2.635 | 26.11% | 868.292 | 870.234 | -0.22% |
| 扩展 mean>=0.90 | 3 | 4.395 | 3.171 | 38.61% | 852.294 | 853.862 | -0.18% |
| 扩展 mean>=0.90 | 5 | 4.436 | 3.231 | 37.29% | 867.596 | 868.745 | -0.13% |
| 激进 p10>=0.85 | 3 | 17.692 | 15.139 | 16.87% | 859.819 | 860.687 | -0.10% |
| 激进 p10>=0.85 | 5 | 17.842 | 15.258 | 16.94% | 868.100 | 869.419 | -0.15% |
| 压力 full-cos>=0.85 | 3 | 28.886 | 23.060 | 25.27% | 862.833 | 860.240 | 0.30% |
| 压力 full-cos>=0.85 | 5 | 29.041 | 23.153 | 25.43% | 868.455 | 865.318 | 0.36% |

## 实现优化

- GPU keep/source/target 索引按 layout 与计划缓存，不在每次 forward 重建。
- Dense-Control 直接使用原 hidden/cos/sin，避免无意义的全 token index_select。
- Sparse 对 hidden/cos/sin 各做一次 gather；目标 Q/K 使用一次批量 index_copy_ 恢复。
- 不再恢复后续未使用的 raw Q/K，只恢复真正进入 attention 的 post-RoPE Q/K。
- 默认 controller 关闭时仍走原始部署路径。

## 正确性与限制

- CPU 单元测试：6 passed；Ruff：All checks passed。
- Dense-Control 与 Baseline 在此前验证中逐元素一致；本轮所有输出均通过 NaN/Inf 检查。
- 本轮只有同一任务 BananaOnPlateTask 的 chunk 3/5、shift=5；没有运行完整 RoboLab episode，因此不声明成功率或任务完成时间不变。
- Q/K 被真实跳算，但 V projection、attention 矩阵与输出投影仍完整计算；完整 chunk 收益自然远小于局部 Q/K 收益。

## 复现命令

```bash
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 \
PYTHONPATH=/root/robolab/cosmos-framework-edge:/root/robolab/cosmos-edge-overlay/tools \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
/root/robolab/cosmos-edge-overlay/tools/run_edge_qk_plan_sweep.py \
  --output-root /root/robolab/cosmos-framework-edge/experiments/preliminary/qk_optimization/plan_sweep/edge_qk_plan_sweep_BananaOnPlateTask_c3_c5_shift5_v1 \
  --timing-repeats 5 --qk-timing-repeats 5
```

原始计时样本见 `timing_samples.csv` 与 `qk_timing_samples.csv`；聚合见 `plan_summary.csv` 与 `qk_timing_summary.csv`。
