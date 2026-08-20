# Action Query 90% mass oracle 稀疏干预实验

## 实验语义

本实验是因果可行性验证，不是加速实现。每个 denoise step、CFG branch、Transformer block 都执行：

1. 对当前仍活跃的 token 做一次完整 probe block；
2. 使用 probe 中真实 post-RoPE Q/K，计算 32 个 predicted Action Query 对 L1..L8 空间 token 的 attention；
3. 对每个 `(action query, future frame)` 选择累计达到该帧 attention mass 90% 的最小空间位置集合；
4. 对全部 32 个 query 和 L1..L8 在空间坐标维取总并集；
5. 将完全相同的 `(y,x)` mask 同时用于 L1..L8；
6. 从同一份 block 输入裁剪 hidden、RoPE 和 pack metadata，重新执行该 block；
7. 只提交第二次 sparse block 输出。被删 token 不进入后续 block。

L0、condition action q0、predicted action q1..q32 和 AR/text token 永不裁剪。L1..L8 每层保留的空间坐标严格相同，未来 token 数始终为 `8 × shared_spatial_tokens`。RoPE 保留原位置，不重新编号。最后一个 block 后，用 last-valid-hidden side buffer 恢复固定 3093-token GEN 布局，随后才进入 final RMSNorm、vision/action head。

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
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
OMNI_KIT_ACCEPT_EULA=Y .venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --task BananaInBowlTask --num-envs 1 --num-runs 1 \
  --output-folder-name action_attention_mass90_sharedmask_BananaInBowlTask_sim_shift1_v1 \
  --video-mode all --headless
```

本机若设置了 `HTTP_PROXY/HTTPS_PROXY`，必须保留上面的 `NO_PROXY/no_proxy`。`websockets>=16` 会自动读取代理；未排除 localhost 时，客户端可能在模型推理前就报 WebSocket handshake timeout。

## 已完成结果

精确配对使用同一 data batch、seed、初始 noise、guidance、4 steps 和 shift=1。只统计实际返回 RoboLab 的 32x8 predicted actions，不包含 conditioned q0：

| 指标 | MSE | relative-L2 | cosine | max-abs |
|---|---:|---:|---:|---:|
| action | 0.017877 | 0.098799 | 0.995911 | 0.381238 |
| action delta | 0.000414 | 1.739221 | 0.319854 | 0.062991 |
| action jerk | 0.000818 | 2.276429 | -0.146967 | 0.084645 |

vision latent cosine 为 0.616216、relative-L2 为 0.791029；decoded RGB cosine 为 0.609643、relative-L2 为 0.794707。全部结果通过 NaN/Inf 检查。

单 chunk 的 224 个 `(step, branch, block)` committed sparse calls 平均节省 789.93/3093 GEN token，平均保留比例 74.46%。逐 query/frame 的最小实测覆盖率为 0.944286。B27 八次 CFG/step forward 最终分别保留 1829、1805、2053、2061、1837、1877、1429、1389 个 GEN token。

闭环 BananaInBowlTask 共执行 24 个 action chunks，平均逻辑节省 906.42/3093 GEN token，平均 committed GEN 保留比例 70.69%。末层平均保留 1598.83 个 GEN token，对应每帧平均 153.23 个共享空间 token，范围 82–241。任务结果为 0/1，750 steps 超时；机器人曾多次抓起香蕉，但运输中掉落，最终未放入碗中。

闭环共检查 5376 个 `(request, step, branch, block)` 调用。每个调用的 L1..L8 均有完全相同的 `selected_spatial_positions`，并满足 `future_tokens_after = 8 × shared_future_spatial_tokens`；不一致计数为 0。闭环最小逐 query/frame 覆盖率为 0.933157。

视频：

- `/root/robolab/RoboLab/output/action_attention_mass90_sharedmask_BananaInBowlTask_sim_shift1_v1/BananaInBowlTask/Pick_up_the_banana_and_place_it_in_the_bowl_0.mp4`
- `/root/robolab/RoboLab/output/action_attention_mass90_sharedmask_BananaInBowlTask_sim_shift1_v1/BananaInBowlTask/Pick_up_the_banana_and_place_it_in_the_bowl_0_viewport.mp4`

## 结论边界

八帧共享空间 mask 后，绝对 action cosine 提高到 0.9959，但仍不能说明控制行为保持。delta cosine 只有 0.3199，jerk cosine 为 -0.1470，闭环也未完成任务。结果说明“所有 future frame 采用相同区域”修正了空间语义，但逐层永久丢弃 token 的误差仍会沿控制轨迹累积。

“节省 token”表示第二次 committed block 的逻辑输入缩短。每层还执行了一次完整 probe，因此物理总计算量高于 baseline，本实验不报告加速，也不能用其 wall time 推断可部署收益。
