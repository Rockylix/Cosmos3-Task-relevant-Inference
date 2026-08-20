# Version1：6-seed BIBD 闭环评估

## 1. 目的与策略

本实验比较三个 Cosmos3 Edge 路径：

- Dense baseline：所有 Transformer token 全量计算；
- Legacy ROI velocity cache：Step 0 profile、固定 ROI、后续背景复用 Step 0 velocity；
- Version1（原 V6-B / Direct Core64 + Stable）：每个 future frame 直接生成一次 Core Top-64，随后选择
  Stable/Adaptive/Fill；G1/G2/G3 每帧预算固定为 `184/152/136`。

三者均固定 `shift=5`、4 个 denoise steps、guidance 3，并关闭 Torch compile 和 CUDA
graphs。

## 2. 实验设计

使用 6 个 policy seed：`9573, 7685, 5732, 7640, 4701, 158`。每个 seed 从 A-D
任务池选取一个二元组合 `AB, AC, AD, BC, BD, CD`，并在简单、普通、复杂三个难度各取
对应两项。因此：

- 每个策略：`6 seeds x 6 tasks = 36 episodes`；
- 三个策略共 108 episodes；
- 每个具体任务在每个策略中出现 3 次；
- 策略之间在同一个 `policy seed x task` 单元严格配对。

policy seed 控制生成噪声；它不是独立的 simulator seed，仿真初始化仍由 RoboLab 任务配置
控制。复杂任务的评估上限不超过 1500 step；原本更低的官方上限保持不变。

## 3. 闭环结果

成功严格读取 `episode_results.jsonl.success`，不使用 `score`。

| Strategy | Success | Rate | Simple | Moderate | Complex | Success-step median |
|---|---:|---:|---:|---:|---:|---:|
| Dense baseline | 14/36 | 38.9% | 9/12 | 4/12 | 1/12 | 160.5 |
| Legacy ROI velocity cache | 13/36 | 36.1% | 8/12 | 3/12 | 2/12 | 178.0 |
| Version1 | 14/36 | 38.9% | 8/12 | 5/12 | 1/12 | 249.5 |

逐 seed-task 配对：

| Candidate | Both success | Candidate only | Dense only | Both fail | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Legacy ROI velocity cache | 10 | 3 | 4 | 19 | 1.0000 |
| Version1 | 9 | 5 | 5 | 17 | 1.0000 |

Version1 与 Dense 的总成功数相同，但不是逐 episode 等价：它在 5 个单元获得成功，同时在另
5 个单元丢失成功。36 个配对单元不足以证明成功率等价；当前结论只是没有观察到总体成功
数下降。成功样本的完成步数存在选择偏差，也不能脱离成功率单独比较。

## 4. 稳定单 chunk 时间与输出差异

主计时在一张 RTX 4090 上完成：同一已加载模型、同一输入，三个 mode 交替执行；每个 mode
预热 3 轮、正式测量 20 轮，CUDA 同步包围 `generate_samples_from_batch`。首请求、RPC 和
完整仿真 wall time 不进入加速比。

| Strategy | Median (s) | P90 (s) | Speedup | Action cosine | Vision latent cosine | Vision relative-L2 |
|---|---:|---:|---:|---:|---:|---:|
| Dense baseline | 0.857823 | 0.862805 | 1.000x | 1.000000000 | 1.000000000 | 0.000000 |
| Legacy ROI velocity cache | 0.718878 | 0.722885 | 1.193x | 0.999789596 | 0.953314841 | 0.305808 |
| Version1 | 0.571400 | 0.574525 | 1.501x | 0.996728420 | 0.927585185 | 0.373775 |

闭环服务器的 warm-request median（跨任务、仅作为稳定性辅助）分别为 0.8709 s、0.7165 s
和 0.5781 s，与受控单 chunk 排序一致。

## 5. Mask 与 attention 抽样

对 BananaInBowlTask、RubiksCubeLeftOfBowlTask、RubiksCubesInBinTask 各采集第一个 policy
request。每个任务均保存：

- G1/G2/G3 的 L1-L8 mask overlay；
- Version1 选择的 Core 来源 block 对 L1-L8 的 `17x20` raw Action-attention 空间图；
- 原始 mask artifact、未来预测帧和逐 block/frame 数值 CSV。

三个样本选出的 block 集合完全相同：`B15, B18, B19, B20, B21, B23`，只有质量排序略有
变化。这只是三次首请求的稳定性观察，不能直接推广到全部任务/chunk。

## 6. 输出与复现

总输出目录：

`/root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/evaluation/direct_core64_baseline_v1_bibd_6seeds_6tasks_v1/`

关键文件：

- `report_cn.md`：完整主报告和可点击图片索引；
- `episode_results.csv`：108 条 episode 明细；
- `paired_seed_task.csv`：36 个配对单元；
- `summary_by_strategy.csv`、`summary_by_difficulty.csv`、`summary_by_seed.csv`；
- `stable_chunk/metrics.json`、`stable_chunk/timing_samples.csv`；
- `visual_samples/`：三个难度的 mask、未来帧和 attention 图。

闭环 runner 可断点续跑，已完成单元会跳过：

```bash
cd /root/robolab/cosmos-framework-edge-version1
export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home
export HF_HUB_OFFLINE=1
export PYTHONPATH=/root/robolab/cosmos-edge-overlay:$PWD
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m tools.run_bibd_6seeds_6tasks_3strategies
```

重新聚合已有结果：

```bash
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m tools.summarize_bibd_6seeds_6tasks_3strategies \
  --stable-chunk-metrics \
  /root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/evaluation/direct_core64_baseline_v1_bibd_6seeds_6tasks_v1/stable_chunk/metrics.json
```

## 7. 数据提交边界

本实验的完整输出位于 Version1 worktree 的 ignored `/experiments/` 中，不提交 Git。本文件
仅提交实验协议、关键参数、汇总指标、复现命令和结论。
