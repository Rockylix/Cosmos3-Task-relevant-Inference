# Action Query 90% mass oracle 稀疏干预实验

## 实验语义

本实验是因果可行性验证，不是加速实现。每个 denoise step、CFG branch、Transformer block 都执行：

1. 对当前仍活跃的 token 做一次完整 probe block；
2. 使用 probe 中真实 post-RoPE Q/K，计算 32 个 predicted Action Query 对 L1..L8 空间 token 的 attention；
3. 对每个 `(action query, future frame)` 选择累计达到该帧 attention mass 90% 的最小 token 集合；
4. 每帧对 32 个 query 的集合取并集；
5. 从同一份 block 输入裁剪 hidden、RoPE 和 pack metadata，重新执行该 block；
6. 只提交第二次 sparse block 输出。被删 token 不进入后续 block。

L0、condition action q0、predicted action q1..q32 和 AR/text token 永不裁剪。RoPE 保留原位置，不重新编号。最后一个 block 后，用 last-valid-hidden side buffer 恢复固定 3093-token GEN 布局，随后才进入 final RMSNorm、vision/action head。

## 运行命令

单 chunk 精确配对：

```bash
cd /root/robolab/worktrees/action-attn-mass90
PYTHONPATH=/root/robolab/worktrees/action-attn-mass90:/root/robolab/cosmos-edge-overlay \
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  tools/run_robolab_action_attention_mass90.py
```

闭环 server：

```bash
cd /root/robolab/worktrees/action-attn-mass90
PYTHONPATH=/root/robolab/worktrees/action-attn-mass90:/root/robolab/cosmos-edge-overlay \
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab_action_mass90 \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --format-prompt-as-json True --no-guardrails \
  --host 127.0.0.1 --port 8000 \
  --guidance 3 --num-steps 4 --shift 1 \
  --seed 579362556 --deterministic-seed \
  --action-attention-mass-threshold 0.9
```

闭环 RoboLab：

```bash
cd /root/robolab/RoboLab
OMNI_KIT_ACCEPT_EULA=Y .venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --task BananaInBowlTask --num-envs 1 --num-runs 1 \
  --output-folder-name action_attention_mass90_BananaInBowlTask_sim_shift1_v1 \
  --video-mode all --headless
```

## 已完成结果

精确配对使用同一 data batch、seed、初始 noise、guidance、4 steps 和 shift=1。只统计实际返回 RoboLab 的 32x8 predicted actions，不包含 conditioned q0：

| 指标 | MSE | relative-L2 | cosine | max-abs |
|---|---:|---:|---:|---:|
| action | 0.049720 | 0.164767 | 0.992580 | 0.778313 |
| action delta | 0.002172 | 3.983800 | 0.029993 | 0.146838 |
| action jerk | 0.005798 | 6.061790 | -0.150330 | 0.268642 |

vision latent cosine 为 0.416768、relative-L2 为 0.914182；decoded RGB cosine 为 0.411707、relative-L2 为 0.912416。全部结果通过 NaN/Inf 检查。

单 chunk 的 224 个 `(step, branch, block)` committed sparse calls 平均节省 1921.75/3093 GEN token，平均保留比例 37.87%。逐 query/frame 的最小实测覆盖率为 0.900747。B27 八次 CFG/step forward 最终分别保留 412、409、434、434、444、439、441、438 token。

闭环 BananaInBowlTask 共执行 24 个 action chunks，平均逻辑节省 1883.61/3093 GEN token；末层平均保留 440.91 token，范围 405–484。任务结果为 0/1，750 steps 超时，失败原因是没有完成抓取香蕉。

视频：

- `/root/robolab/RoboLab/output/action_attention_mass90_BananaInBowlTask_sim_shift1_v1/BananaInBowlTask/Pick_up_the_banana_and_place_it_in_the_bowl_0.mp4`
- `/root/robolab/RoboLab/output/action_attention_mass90_BananaInBowlTask_sim_shift1_v1/BananaInBowlTask/Pick_up_the_banana_and_place_it_in_the_bowl_0_viewport.mp4`

## 结论边界

绝对 action cosine 0.9926 不能说明控制行为保持。delta/jerk 方向已经接近无关甚至反向，闭环也未完成任务。当前 per-query/per-frame 90% mass 的连续跨层取交集过于激进。

“节省 token”表示第二次 committed block 的逻辑输入缩短。每层还执行了一次完整 probe，因此物理总计算量高于 baseline，本实验不报告加速，也不能用其 wall time 推断可部署收益。
