# V5.1 Top-K budget sweep：shift=5，三任务闭环

## 1. 目的与控制变量

本实验只缩减 V5.1 的逐 future-frame Top-K token 预算，不使用 raw mass，也不改变其他推理逻辑。

固定项：

- V5.1 帧内归一化 Action-Q/Future-K profile；
- Step 0 全量计算；Step 1--3 的 B0--B3 全量计算；
- block groups：G1=B4--B11、G2=B12--B19、G3=B20--B27；
- CFG branch/block-group max 聚合；
- 时间 score 平滑 `0.25/0.5/0.25`；
- depth look-ahead `lambda=0.5`；
- `E3 subset E2 subset E1`，禁止 token re-entry；
- velocity cache、`shift=5`、4 denoise steps、guidance=3；
- policy seed `579362556`，RoboLab environment seed `0`；
- eager inference，关闭 torch compile 和 CUDA graphs。

每个预算档运行相同三个任务各一次：

- `BananaInBowlTask`；
- `BananaOnPlateTask`；
- `RubiksCubeTask`。

总计 6 个预算档、18 个完整 episode。

## 2. 预算定义

`k100` 表示原始 V5.1 预算，而不是全量 340 token：

| Config | G1 | G2 | G3 |
|---|---:|---:|---:|
| k100 | 240 | 200 | 180 |
| k90 | 216 | 180 | 162 |
| k80 | 192 | 160 | 144 |
| k75 | 180 | 150 | 135 |
| k70 | 168 | 140 | 126 |
| k60 | 144 | 120 | 108 |

完整 GEN 序列为 3093 token，其中 future video 为 `8*340=2720`，其余文本/condition/action token 共 373，始终保留。

## 3. 三任务结果

| Config | Budget G1/G2/G3 | Success | Banana bowl | Banana plate | Rubiks cube | Mean episode steps |
|---|---:|---:|---:|---:|---:|---:|
| k100 | 240/200/180 | **3/3** | 171 ✓ | 172 ✓ | 498 ✓ | 280.3 |
| k90 | 216/180/162 | **3/3** | 265 ✓ | 170 ✓ | 380 ✓ | 271.7 |
| k80 | 192/160/144 | **3/3** | 202 ✓ | 252 ✓ | 206 ✓ | 220.0 |
| k75 | 180/150/135 | **2/3** | 157 ✓ | 363 ✓ | 600 ✗ | 373.3 |
| k70 | 168/140/126 | **2/3** | 750 ✗ | 161 ✓ | 385 ✓ | 432.0 |
| k60 | 144/120/108 | **2/3** | 587 ✓ | 159 ✓ | 600 ✗ | 448.7 |

失败任务均不是完全无法抓取：

- k75/k60 的 RubiksCube 已进入 `pick_and_place` 第 2/2 阶段，但没有完成放入碗并松手；
- k70 的 BananaInBowl 也已进入第 2/2 阶段，但没有完成放置。

因此主要退化出现在抓取后的精细放置与释放，而不是粗粒度接近目标。

## 4. Token 与实测 generation 时间

时间统计来自各档 server `aggregate.json`。每档排除第一个冷请求，对整个三任务 rollout 的后续请求统计 median 和 P90。

| Config | G1/G2/G3 GEN retained | Saved token/sparse block | Warm median | Warm P90 | Speedup vs k100 |
|---|---:|---:|---:|---:|---:|
| k100 | 74.14% / 63.79% / 58.62% | 1066.7 | 0.7519 s | 0.7589 s | 1.000x |
| k90 | 67.93% / 58.62% / 53.96% | 1232.0 | 0.7329 s | 0.7460 s | 1.026x |
| k80 | 61.72% / 53.44% / 49.30% | 1397.3 | 0.7025 s | 0.7100 s | **1.070x** |
| k75 | 58.62% / 50.86% / 46.98% | 1480.0 | 0.6935 s | 0.6979 s | 1.084x |
| k70 | 55.51% / 48.27% / 44.65% | 1562.7 | 0.6825 s | 0.6911 s | 1.102x |
| k60 | 49.30% / 43.10% / 39.99% | 1728.0 | 0.6577 s | 0.6644 s | 1.143x |

继续从 k80 降到 k75，只额外获得约 `1.3%` 的 warm generation speedup，却首次损失一个任务；降到 k60 也只有约 `1.143x`，说明当前耗时还包含 Step 0 全量、B0--B3 全量、非 future token、输出头与 sampler 等固定部分。

## 5. 结论

在这三个任务、这个固定 policy/environment seed 下：

- **保持 3/3 的最小已测预算是 k80：`(192,160,144)`**；
- k75、k70、k60 均只有 2/3；
- 成功结果不严格单调，例如 k70 Banana 失败而 k60 成功，说明离散 mask 会改变闭环轨迹，不能用单 episode 二分搜索；
- k80 相对原始 V5.1 的 warm generation median 加速约 `1.070x`；
- k80 是下一轮多 seed 验证的候选，不应称为稳定安全阈值。

推荐下一步只保留三个档位做多 seed：`k100`、`k80`、`k75`。k80 是当前候选，k75 是最近失败边界。

## 6. 输出

汇总目录：

```text
/root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/velocity_cache/
  v5_1_topk_budget_sweep_shift5_3tasks_seed579362556_v1/
```

关键文件：

- `episode_results.csv`：18 个 episode 明细与视频路径；
- `budget_summary.csv`：成功率、token 比例与 generation 时间；
- `experiment.json`：机器可读协议和汇总；
- `k*/server/aggregate.json`：逐 policy request 原始时间与 token 记录。

RoboLab 视频位于：

```text
/root/robolab/RoboLab/output/v5_1_topk_k*_shift5_3tasks_seed579362556_v1/
```

## 7. 复现实例

Server（以 k80 为例）：

```bash
cd /root/robolab/worktrees/v5-1-topk-budget-sweep-shift5
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 LD_LIBRARY_PATH='' PYTHONPATH=$PWD \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab_grouped_temporal_closed_roi_velocity_cache \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --format-prompt-as-json True --no-guardrails \
  --guidance 3 --num-steps 4 --shift 5 \
  --deterministic-seed --seed 579362556 \
  --host 127.0.0.1 --port 8000 \
  --roi-tokens-g1 192 --roi-tokens-g2 160 --roi-tokens-g3 144 \
  --intervention-output-dir <output>/k80/server
```

RoboLab：

```bash
cd /root/robolab/RoboLab
OMNI_KIT_ACCEPT_EULA=Y NO_PROXY=127.0.0.1,localhost \
.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --task BananaInBowlTask BananaOnPlateTask RubiksCubeTask \
  --num-envs 1 --num-runs 1 \
  --output-folder-name <run-id> \
  --video-mode viewport --headless
```
