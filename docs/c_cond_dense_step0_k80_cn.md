# C/K80：Step 0 仅 conditional 全量计算

## 策略

实验保持 C/K80、shift=5、4-step UniPC、guidance=3，compile/CUDA graphs 关闭。

- Step 0 conditional：B0--B27 全量，使用 conditional 的 B4--B27 attention 构造 C mask。
- Step 0 unconditional：B0--B11 使用 G1，B12--B19 使用 G2，B20--B27 使用 G3，全部为真实短序列计算。
- Step 1--3：保持原 C，B0--B3 全量，B4--B27 使用 G1/G2/G3。
- K80 每帧预算为 G1/G2/G3=`192/160/144`；L0、action 和文本条件始终保留。
- Step-0 unconditional 最终 future prediction 在 G3 外替换为 conditional prediction，即背景 `CFG delta=0`；G3 内使用真实 CFG。
- Step-0 完整 guided vision velocity 继续作为后续 step 的背景 velocity cache。

全 224 block call 的 GEN retention 从原 C 的 `70.96%` 降至 `65.43%`。新策略包含 52 次 full call、60 次 G1、56 次 G2、56 次 G3；Step-0 unconditional 的 28 个 block 全部稀疏。

## 冻结 chunk 结果

BananaInBowlTask/chunk 3，seed `579362556`，同一输入、noise 和推理参数；3 次预热，20 次交替计时：

| Mode | Median | P90 | Speedup vs Dense |
|---|---:|---:|---:|
| Dense | 0.856524 s | 0.860079 s | 1.000x |
| 原 C/K80 | 0.682474 s | 0.684376 s | 1.255x |
| C/K80 conditional-only Step0 | **0.613644 s** | **0.616065 s** | **1.396x** |

| Metric | 原 C vs Dense | 新 C vs Dense |
|---|---:|---:|
| action MSE | 0.006996 | 0.006383 |
| action relative-L2 | 0.060174 | 0.057475 |
| action cosine | 0.998204 | 0.998658 |
| delta-action cosine | -0.063942 | -0.028075 |
| jerk cosine | -0.186624 | -0.129969 |
| vision relative-L2 | 0.363464 | 0.357130 |
| vision cosine | 0.935026 | 0.934076 |

新策略整体 action 指标没有恶化，delta/jerk 也比原 C 略好，但两者相对 Dense 的局部动力学 cosine 都很差。单 chunk 整体 action cosine 不能替代闭环验证。

稳定 benchmark：

```bash
cd /root/robolab/worktrees/c-cond-dense-step0-k80
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=. \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  tools/benchmark_robolab_v5_3_stable_chunk.py \
  --output-root /root/robolab/experiments/preliminary/sparsity/velocity_cache/c_cond_dense_step0_k80_BananaInBowlTask_c3_v1 \
  --group-token-budgets 192 160 144 \
  --modes dense c_opt c_cond_opt \
  --warmup-rounds 3 --measure-rounds 20
```

## 单任务闭环

`BananaInBowlTask`、环境 seed 0、policy seed `579362556`：

- Success：`1/1`；
- 170 steps，score 1.0；
- 6 个 generation chunks；
- 排除冷请求后 generation median `0.616346 s`，P90 `0.623424 s`；
- 事件：`TARGET_OBJECT_DROPPED=2`。

Policy server：

```bash
cd /root/robolab/worktrees/c-cond-dense-step0-k80
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=. \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python -m \
  cosmos_framework.scripts.action_policy_server_robolab_v5_3_acd_packed_kernel \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --host 127.0.0.1 --port 8000 --format-prompt-as-json True --no-guardrails \
  --seed 579362556 --deterministic-seed --guidance 3.0 --num-steps 4 --shift 5.0 \
  --ablation-mode c_cond_step0 \
  --intervention-output-dir /root/robolab/experiments/preliminary/sparsity/velocity_cache/c_cond_dense_step0_k80_BananaInBowlTask_seed579362556_v1/server
```

RoboLab：

```bash
cd /root/robolab/RoboLab
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
OMNI_KIT_ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --task BananaInBowlTask --num-envs 1 --num-runs 1 \
  --headless --video-mode viewport \
  --output-folder-name c_cond_dense_step0_k80_BananaInBowlTask_seed579362556_v1
```

视频：

```text
/root/robolab/RoboLab/output/c_cond_dense_step0_k80_BananaInBowlTask_seed579362556_v1/BananaInBowlTask/Pick_up_the_banana_and_place_it_in_the_bowl_0_viewport.mp4
```

## 结论边界

该消融证明 Step-0 unconditional 全量计算不是 BananaInBowl 固定 seed 下的必要条件，并将稳定单 chunk 加速从原 C 的 `1.255x` 提高到 `1.396x`。但当前只有一个任务、一个环境 seed、一个 episode；且 delta/jerk 与 Dense 差异仍大，不能据此宣称多任务成功率不变。
