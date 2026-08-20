# V5.1：Score-smoothed budgeted ROI velocity cache

## 1. 实验范围

- 模型：Cosmos3 Edge Policy DROID
- `num_steps=4, guidance=3, shift=5`
- eager inference，关闭 torch compile 和 CUDA graphs
- Step 0 全量计算并采集 Action-Q/Future-K 空间重要性
- Step 1--3：B0--B3 全量；B4--B27 分组稀疏
- 不改变原始 token position、RoPE、UniPC 或 velocity-cache 补齐路径

本实验只验证固定预算 mask 对单 chunk 误差、token 数、单次时间和一个闭环 episode 的影响。

## 2. V5.1 mask

对 Step 0 profile 在 CFG branch 和 block group 内取最大值，future frame 保持分开：

\[
S_{g,f,p}=\max_{b,l\in g}S_{b,l,f,p}.
\]

每张分数图先归一化，再在时间维做 score-level 平滑：

\[
\bar S_{g,f}=0.25S_{g,f-1}+0.5S_{g,f}+0.25S_{g,f+1},
\]

边界帧只使用存在的邻帧，并重新归一化权重。V5.1 不再对相邻帧的二值 mask 求并集。

令深度 look-ahead 衰减 `lambda=0.5`：

\[
D_1=\bar S_1+\lambda\bar S_2+\lambda^2\bar S_3,
\]

\[
D_2=\bar S_2+\lambda\bar S_3,\qquad D_3=\bar S_3.
\]

逐 future frame 选择：

\[
E_1=\operatorname{TopK}(D_1,240),
\]

\[
E_2=\operatorname{TopK}(D_2,200\mid E_1),
\]

\[
E_3=\operatorname{TopK}(D_3,180\mid E_2).
\]

因此严格满足：

\[
E_3\subseteq E_2\subseteq E_1,
\]

禁止 token 在缺失中间 block hidden state 后重新进入。

## 3. V5 与 V5.1 差异

| 项目 | V5 | V5.1 |
|---|---|---|
| 时间一致性 | 相邻二值 mask 并集 | 相邻连续 score 平滑 |
| 深度一致性 | 跨 group 二值并集 | 受约束的嵌套 Top-K |
| 选择规则 | 每图累计 90% attention mass | 固定 `240/200/180` token |
| token 数 | 由分布决定，容易膨胀 | 精确、可预测 |
| token 重入 | 禁止 | 禁止 |
| attention mass 保证 | 至少 90% | 不保证 90% |

## 4. Token 数

完整 GEN 序列为 3093 token，其中非 future token 为 373，future 为 `8*340=2720`。

| Blocks | Future token/frame | GEN tokens | Retained | Saved/block |
|---|---:|---:|---:|---:|
| B4--B11 | 240/340 | 2293/3093 | 74.14% | 800 |
| B12--B19 | 200/340 | 1973/3093 | 63.79% | 1120 |
| B20--B27 | 180/340 | 1813/3093 | 58.62% | 1280 |

真实 Step-0 artifact 验证了每个 future frame 都严格满足预算，且两级子集关系均成立。

## 5. Paired chunk 结果

任务为 `BananaInBowlTask / chunk 3`，seed `579362556`。Baseline 和 sparse 使用相同输入、初始 RNG、guidance、steps 与 shift。

| Strategy | Action MSE | Action cos | Action rel-L2 | Delta cos | Jerk cos | RGB cos | RGB rel-L2 | Latent cos | Latent rel-L2 | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0 | 1.000000 | 0 | 1.000000 | 1.000000 | 1.000000 | 0 | 1.000000 | 0 | 1.00x |
| V1 | 0.004775 | 0.998763 | 0.049733 | -0.033250 | -0.165619 | 0.967082 | 0.275822 | 0.944305 | 0.335609 | 1.47x |
| V5 | 0.000471 | 0.999880 | 0.015616 | 0.511664 | 0.459902 | 0.991365 | 0.169292 | 0.978886 | 0.205590 | 1.19x |
| V5.1 | 0.001893 | 0.999549 | 0.031314 | 0.135368 | 0.028085 | 0.970493 | 0.262428 | 0.948299 | 0.323499 | 1.43x |

时间是单次、非交替 benchmark，只用于初步比较。V5.1 的整体 action 指标优于 V1、弱于 V5；delta/jerk 明显弱于 V5。预测主体仍可辨认，但背景块状伪影比 V5 更明显。

## 6. 闭环功能验证

一次 `BananaInBowlTask`：

- `success=true`，`1/1`；
- episode step：147；
- 仿真任务时长：9.8 s；
- policy request：5；
- policy inference：6.505 s，平均 44.3 ms/仿真 step；
- server generation wall：`1.087, 0.756, 0.751, 0.748, 0.748 s`，warm request 均值约 0.751 s；
- viewport：`/root/robolab/RoboLab/output/grouped_score_smoothed_budgeted_roi_velocity_cache_shift5_BananaInBowlTask_sim_v1/BananaInBowlTask/Pick_up_the_banana_and_place_it_in_the_bowl_0_viewport.mp4`。

这是单 seed、单 episode 的功能验证，不代表稳定任务成功率。

## 7. 运行命令

Paired chunk：

```bash
cd /root/robolab/worktrees/grouped-temporal-closed-roi-velocity-cache-shift5-v5
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 PYTHONPATH=$PWD \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m tools.run_robolab_step0_fixed_roi_velocity_cache \
  --strategy-version v5.1 \
  --group-token-budgets 240 200 180 \
  --shift 5 \
  --output-root /root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/velocity_cache/<run-id>
```

Server：

```bash
cd /root/robolab/worktrees/grouped-temporal-closed-roi-velocity-cache-shift5-v5
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 LD_LIBRARY_PATH='' PYTHONPATH=$PWD \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab_grouped_temporal_closed_roi_velocity_cache \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --format-prompt-as-json True --no-guardrails \
  --guidance 3 --num-steps 4 --shift 5 \
  --deterministic-seed --seed 579362556 \
  --host 0.0.0.0 --port 8000 \
  --roi-tokens-g1 240 --roi-tokens-g2 200 --roi-tokens-g3 180 \
  --intervention-output-dir /root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/velocity_cache/<server-run-id>
```

RoboLab：

```bash
cd /root/robolab/RoboLab
OMNI_KIT_ACCEPT_EULA=Y NO_PROXY=127.0.0.1,localhost \
.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --task BananaInBowlTask --num-envs 1 --num-runs 1 \
  --output-folder-name <run-id> \
  --video-mode viewport --headless
```

## 8. 当前判断

V5.1 达成了预期的计算量控制，并在一个闭环 episode 中成功完成任务。它不是对 V5 的无损替代：V5.1 用更高稀疏度换来了更差的 delta/jerk 和视觉一致性。下一步若继续推进，应在 `(240,200,180)` 周围做预算消融，优先提高 G3 预算或降低深度收缩幅度，并使用多 seed 成功率决定是否保留，而不能只看整体 action cosine。
