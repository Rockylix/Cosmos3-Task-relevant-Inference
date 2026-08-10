# 当前 block Action Attention 90% 采样与 L1 residual 补全实验

## 1. 实验定义

本实验使用 Cosmos3 Edge、`shift=5`，验证以下双 pass、同 block 的离线 oracle 策略：

1. 每个 Transformer block 先用完整序列运行一次 probe；probe 输出不进入后续计算，只提取真正参与 attention 的 post-RoPE Action Q 和 Future K。
2. 对 32 个预测 action query 和未来帧 L2--L8，分别选择达到该帧 attention mass 90% 的最小空间 token 集合。
3. 将所有 query、所有 L2--L8 的集合在空间坐标上取并集，得到当前 block 的共享空间 mask。mask 大小随 block、step、CFG branch 和输入变化，不固定为 168 个 token。
4. 同一个 block 用相同输入再运行一次 committed sparse pass：
   - L0、L1、q0、q1--q32 全量计算；
   - L2--L8 使用同一份共享空间 mask；
   - UND/text token 不稀疏；
   - RoPE、hidden state 及按 token 排列的数据使用同一索引裁剪，保留原始位置编码。
5. 对 L2--L8 未计算的位置，直接复用当前 block 的 L1 同空间位置 residual：

   ```text
   R_L1[p] = H_sparse_out[L1,p] - H_in[L1,p]
   H_out[Lf,p] = H_in[Lf,p] + R_L1[p],  f=2..8
   ```

6. 每个 block 结束后恢复完整 GEN 序列，下一 block 再从完整序列开始 probe。不存在 token 累积删除，也不使用上一 block 的路由结果。

原始 GEN token 数为 3093。若共享 mask 保留 `k` 个空间位置，则 committed sparse pass 的 GEN token 数为：

```text
L0 + L1 + L2..L8 + action = 340 + 340 + 7*k + 33 = 713 + 7*k
```

其中 `1 <= k <= 340`。

## 2. 实现与测试

实验分支和 worktree：

```text
branch:   experiment/action-attn-current-block-oracle-shift5
worktree: /root/robolab/worktrees/action-attn-current-block-oracle-shift5
```

主要文件：

- `cosmos_framework/scripts/robolab_action_attention_l1_residual_intervention.py`：路由选择、双 pass controller、L1 residual 补全及 token 统计。
- `tools/run_robolab_action_attention_l1_residual_shift5.py`：固定 chunk 的 Baseline/Sparse 配对实验。
- `cosmos_framework/scripts/action_policy_server_robolab_action_l1_residual.py`：RoboLab 闭环服务入口。
- `tests/test_robolab_action_attention_l1_residual_intervention.py`：CPU 单元测试。

验证结果：

```text
ruff: passed
pytest: 9 passed
```

单元测试覆盖 L0/L1/action 必须保留、L2--L8 共享空间 mask、变长 token 选择、L1 同位置 residual 补全、完整帧恢复以及全 mask 边界条件。

## 3. 固定 chunk 配对结果

数据为 BananaInBowlTask、chunk 3；Baseline 与 Sparse 使用相同输入、seed、noise、`shift=5` 和 4 个 denoise step。

| 指标 | Sparse vs. Baseline |
|---|---:|
| Action MSE | 0.00069110 |
| Action relative-L2 | 0.018457 |
| Action cosine | 0.999830 |
| Action max absolute error | 0.102091 |
| Delta-action cosine | 0.283916 |
| Jerk cosine | 0.164244 |
| Final vision latent relative-L2 | 0.279109 |
| Final vision latent cosine | 0.960789 |
| Decoded RGB relative-L2 | 0.276099 |
| Decoded RGB cosine | 0.967197 |

所有 action、latent、RGB 和 block 中间输出均通过 NaN/Inf 检查。

配对产物：

```text
/root/robolab/experiments/preliminary/sparsity/action_attention_l1_residual/
  action_attention_current_block_mass90_l1_residual_shift5_BananaInBowlTask_c3_v1/
```

其中包含 `metrics.json`、逐 horizon/joint 指标、逐 block/frame token 统计、配对 tensor，以及 Baseline/Sparse 解码 future frames。

## 4. Token 统计

固定 chunk 共覆盖 4 step x 2 CFG branch x 28 block = 224 次 block committed pass：

| 项目 | 数值 |
|---|---:|
| 平均共享 L2--L8 空间 token `k` | 329.875 / 340 |
| 最小/最大 `k` | 260 / 340 |
| committed pass 平均保留 GEN 比例 | 97.7085% |
| committed pass 平均节省 GEN token | 70.875 / 3093 |
| 单 block 最大节省 GEN token | 560 / 3093 |
| 补全后传给下一 block 的 GEN token | 3093 / 3093 |

闭环仿真的 7 个 action chunk 平均每个 committed block pass 节省 69.129 GEN token，即相对 3093 token 约 2.235%。全部 1568 次 block 调用均完成完整序列恢复且数值有限。

这里的 90% 是“每个 action query、每个 future frame 的 attention mass 90%”，再跨 query/frame 取空间并集。该规则非常保守，因此最终保留的 token 比例明显高于 90%。

## 5. RoboLab 闭环结果

任务：BananaInBowlTask，1 次运行。

| 指标 | 结果 |
|---|---:|
| Success | 1/1 |
| Score | 1.0 |
| Episode step | 197 |
| 仿真时长 | 13.133 s |
| Wall time | 55.673 s |
| Action chunks | 7 |
| Policy inference total | 19.408 s |
| TARGET_OBJECT_DROPPED | 3 |

任务最终完成抓取并将香蕉放入碗中。单任务结果只能证明这次闭环可运行，不能代表总体成功率不变。

视频：

```text
/root/robolab/RoboLab/output/action_attention_current_block_mass90_l1_residual_shift5_BananaInBowlTask_sim_v1/
  BananaInBowlTask/Pick_up_the_banana_and_place_it_in_the_bowl_0.mp4
  BananaInBowlTask/Pick_up_the_banana_and_place_it_in_the_bowl_0_viewport.mp4
```

闭环逐 chunk token 统计：

```text
/root/robolab/experiments/preliminary/sparsity/action_attention_l1_residual/
  action_attention_current_block_mass90_l1_residual_shift5_BananaInBowlTask_sim_v1/server/
```

## 6. 运行命令

固定 chunk 配对实验：

```bash
cd /root/robolab/worktrees/action-attn-current-block-oracle-shift5
export PYTHONPATH=$PWD:/root/robolab/cosmos-edge-overlay
export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home
export HF_HUB_OFFLINE=1
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  tools/run_robolab_action_attention_l1_residual_shift5.py
```

启动策略服务：

```bash
cd /root/robolab/worktrees/action-attn-current-block-oracle-shift5
export PYTHONPATH=$PWD:/root/robolab/cosmos-edge-overlay
export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python -m \
  cosmos_framework.scripts.action_policy_server_robolab_action_l1_residual \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --format-prompt-as-json True --no-guardrails \
  --host 127.0.0.1 --port 8000 \
  --guidance 3 --num-steps 4 --shift 5 \
  --seed 579362556 --deterministic-seed \
  --action-attention-mass-threshold 0.9
```

另一个终端运行 RoboLab：

```bash
cd /root/robolab/RoboLab
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export OMNI_KIT_ACCEPT_EULA=Y
.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --task BananaInBowlTask --num-envs 1 --num-runs 1 \
  --headless --video-mode all \
  --output-folder-name \
  action_attention_current_block_mass90_l1_residual_shift5_BananaInBowlTask_sim_v1
```

## 7. 结论边界

- 功能上，L1 同位置 residual 能把 sparse pass 恢复为每层完整 8 帧，并在本次 BananaInBowlTask 闭环中成功。
- 输出 action 的整体 cosine 很高，但 delta-action 和 jerk cosine 较低；不能仅凭整体 action cosine 判断控制轨迹等价。
- 当前 90%-mass 并集只让 committed pass 平均减少约 2.2% GEN token。
- 每个 block 还额外执行一次全量 probe，所以该实现是 oracle/正确性实验，实际计算量高于 Baseline，不应宣称加速。
- 若下一步追求实际加速，需要取消同 block 全量 probe，改为可在线获得或跨层复用的低开销 mask，并重新验证动作动态与多任务成功率。
