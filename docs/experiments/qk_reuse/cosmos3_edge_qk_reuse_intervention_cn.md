# Cosmos3 Edge 相邻未来帧 Q/K 复制：端到端输出敏感性实验

## 1. 实验结论

本轮验证了：在 profile 给出的 9 个非重叠候选位置，将源 future latent 的 post-RoPE Q/K 直接覆盖目标 latent 的 Q/K，会使最终 RoboLab action 发生约 1% relative-L2 的变化，同时 final future vision latent 的 relative-L2 达到约 7%–10%。

这说明高 Q/K cosine 可以得到较小但非零的 action 扰动，不能仅凭 `cosine >= 0.9` 宣称复用安全或任务成功率不变。

当前实现仍执行完整 Q/K projection 和 attention matmul，只额外执行 clone/index-copy，因此没有产生加速，平均慢约 0.5%–0.6%。这轮是输出敏感性实验，不是真正的稀疏计算实验。

## 2. 实验配置

- Task：`BananaOnPlateTask`
- 冻结真实任务输入：chunk 3、chunk 5
- shift：5
- denoise steps：4，timestep `999 / 937 / 833 / 624`
- CFG：conditional 和 unconditional 使用相同策略
- compile：关闭
- CUDA graphs：关闭
- 每个 chunk：单独精确复制检查，双侧预热，随后 Baseline/Sparse 交替各运行 5 次
- Baseline/Sparse：相同 `data_batch`、conditioning image、state、seed、initial noise 和 sampler 参数

干预位置：Q/K projection、head reshape、QK RMSNorm 和 RoPE 之后，attention packing/kernel 之前。

```text
step 0 / B27: L7 -> L8

step 2 / B1:  L1 -> L2, L3 -> L4, L5 -> L6, L7 -> L8
step 3 / B1:  L1 -> L2, L3 -> L4, L5 -> L6, L7 -> L8
```

每次 generation 共执行：

```text
9 frame-pair operations × 2 CFG branches = 18 replacements
340 spatial tokens / replacement
```

源帧的 post-RoPE Q/K 被直接复制，因此目标帧拿到的是源帧原始 temporal position 对应的 RoPE 结果。目标帧 V、attention mask、其他 video/action/text token 均保持原值。

## 3. 实现正确性

- 18/18 个预期替换全部触发。
- Q target 与 Q source 复制后 max-abs error：0。
- K target 与 K source 复制后 max-abs error：0。
- 所有输出通过 NaN/Inf 检查。
- 5 次计时重复中，Baseline action 在相同 seed 下逐元素一致；Sparse action 也逐元素一致。
- controller 仅在实验 context 内挂载，退出后恢复为关闭状态，不改变默认部署路径。

CPU 单元测试覆盖：

- token 选择与精确复制；
- 非目标 token 和 action token 保持不变；
- 重叠 pair 拒绝；
- step/block/branch gating；
- controller 退出后 hook 清理。

测试结果：`3 passed`。

## 4. 最终 RoboLab action 误差

主指标严格按照 `RobolabPolicyService.infer()` 的返回语义：从模型原始 `[33,8]` action 删除第 0 个 state/history row，再执行 gripper `1-x`，比较最终 `[32,8]` action chunk。

| chunk | MSE | MAE | relative L2 | cosine | max absolute error |
|---|---:|---:|---:|---:|---:|
| c3 | 2.0865e-4 | 0.009764 | 0.010805 | 0.999942 | 0.077367 |
| c5 | 1.3829e-4 | 0.008400 | 0.010339 | 0.999947 | 0.047733 |

逐 horizon 中 MSE 最大的位置：

| chunk | horizon | MSE | relative L2 | cosine | max abs |
|---|---:|---:|---:|---:|---:|
| c3 | 12 | 9.81e-4 | 0.02318 | 0.999814 | 0.07737 |
| c5 | 0 | 6.43e-4 | 0.02165 | 0.999854 | 0.04773 |

32 个 horizon 的 relative-L2 均值分别为 c3 `0.00994`、c5 `0.00950`，最大值分别为 `0.02318`、`0.02165`。

逐关节 relative-L2 有两个异常高值：c3 joint 2 为 `0.716`，c5 joint 4 为 `0.643`。原因是对应 Baseline 轨迹本身接近零：L2 norm 分别只有 `0.0563` 和 `0.0830`；其 MSE 仍分别为 `5.09e-5`、`8.89e-5`。因此不能单看这些低能量维度的 relative-L2，但 max-abs 和逐关节绝对误差仍需在仿真阶段检查。

## 5. 最终 vision latent 误差

| chunk | 范围 | MSE | relative L2 | cosine | max absolute error |
|---|---|---:|---:|---:|---:|
| c3 | 全部 L0..L8 | 0.00901 | 0.09529 | 0.99547 | 2.0613 |
| c3 | future L1..L8 | 0.01013 | 0.09795 | 0.99522 | 2.0613 |
| c5 | 全部 L0..L8 | 0.00377 | 0.07095 | 0.99749 | 1.2185 |
| c5 | future L1..L8 | 0.00424 | 0.07302 | 0.99735 | 1.2185 |

vision cosine 仍较高，但 relative-L2 明显大于 action。c3 中误差最大的单个 future latent 是 L6，relative-L2 `0.1483`；c5 的 L7/L8 约为 `0.1014/0.1017`。这说明 Q/K 局部干预会通过后续 block 和 UniPC 累积传播，不能把初始 Q/K 高相似直接等价为最终视频不变。

## 6. Chunk timing

计时只包含 generation，使用 CUDA event 和同步 wall time；精确复制校验产生的 CPU 同步不进入正式计时。单位为秒。

| chunk | mode | mean | median | P90 |
|---|---|---:|---:|---:|
| c3 | Baseline | 0.84944 | 0.84975 | 0.85013 |
| c3 | Q/K copy | 0.85349 | 0.85376 | 0.85399 |
| c5 | Baseline | 0.85166 | 0.85166 | 0.85247 |
| c5 | Q/K copy | 0.85693 | 0.85718 | 0.85731 |

- c3：Q/K copy 比 Baseline 慢 `0.48%`，约 `4.06 ms/chunk`。
- c5：Q/K copy 比 Baseline 慢 `0.62%`，约 `5.27 ms/chunk`。
- 两个 chunk 的 mean：Baseline `0.85055 s`，Q/K copy `0.85521 s`。

该结果符合实现边界：当前是在完整计算之后覆盖 Q/K，没有减少 projection 或 attention token 数，新增的 clone/index-copy 只会增加开销。

## 7. 本轮不能回答的内容

本轮使用冻结的真实任务 chunk 做离线成对推理，没有启动 RoboLab simulator 完整 episode。因此：

- 没有任务成功/失败结论；
- 没有任务完成时间；
- 没有证明真实跳算可以加速；
- 没有证明 action 语义等价。

若继续进行完整仿真，建议先保留当前 9 点干预策略，对 Baseline/QK-copy 分别跑同一批 seeds，记录成功率、episode step、policy inference time、env step time 和 wall-total。只有输出敏感性可接受后，才值得实现真正跳过目标 Q/K projection 或稀疏 attention token 的 kernel 路径。

## 8. 输出文件

实验目录：

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/qk_optimization/reuse_intervention/
  edge_qk_reuse_BananaOnPlateTask_c3_c5_shift5_v1/
```

主要文件：

```text
experiment.json
metrics.json
timing.json
timing_samples.csv
action_overall_metrics.csv
action_horizon_metrics.csv
action_joint_metrics.csv
vision_latent_metrics.csv
intervention_trace.csv
chunk_000003_robolab_returned_action.pt
chunk_000005_robolab_returned_action.pt
baseline/chunk_000003.pt
baseline/chunk_000005.pt
sparse/chunk_000003.pt
sparse/chunk_000005.pt
```

复现命令必须显式使用已有 HF cache：

```bash
cd /root/robolab/cosmos-framework-edge

HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 \
PYTHONPATH=/root/robolab/cosmos-framework-edge \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/run_edge_qk_reuse_intervention.py \
  --output-root /root/robolab/cosmos-framework-edge/experiments/preliminary/qk_optimization/reuse_intervention/<new_run_name> \
  --timing-repeats 5
```
