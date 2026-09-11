# Dense / ASI / ToCa-Future：十任务与未来帧保真度

日期：2026-09-10。状态：全部完成，包括 Dense/ASI 各十任务闭环、十个同输入 chunk 的三方法重放与 VAE 解码、30 轮交替单 chunk 计时。

本页的 ToCa 为首版 `reference` 路径。后续共享 QK 的联合 attention/评分优化与新计时见 [优化报告](toca_joint_attention_optimization_cn.md)；不要将新 kernel 的时间与本页旧闭环成功率拼成同一轮结果。

结论：当前 ASI 相对 Dense 的 warmed generation median 加速 **1.462×**，但预测图存在明显块状伪影；ToCa-Future 的图像更接近 Dense，但当前参考移植没有产生加速。本轮闭环成功数为 Dense 5/10、ASI 3/10、ToCa 1/10，不能用预测图保真度或完整 action cosine 替代闭环验证。

## 比较对象与控制条件

| 方法 | 实际实现 |
|---|---|
| Dense | `/root/robolab/cosmos-framework-edge`，`cache@324574a`，原生 eager 路径 |
| ASI（我们的当前策略） | `version/core80-stable104-action-weighted@62a09f7`；Core80 + Stable104，每帧 K184，step 0 conditional 全量 profile，其余七次 CFG forward 固定 mask |
| ToCa-Future | `experiment/toca-future` 首版移植；D/C/D/C，future MLP fresh ratio 基值 0.25、层级 slope=0.5；L0/action 始终重算、当前 K/V 保持完整 |

ASI 在实验 worktree 中运行的策略代码、原 policy server、`unified_mot.py` SHA256 均与当前 ASI checkout 相同，没有修改 mask 选取、预算或恢复方式。ASI 未选 hidden 在每个 stack 末端从该 stack 的输入恢复；不要与早期 velocity-cache 版本混称。

统一本地 Edge DROID 权重、RoboLab 环境、simulation seed=0、policy seed=0、`deterministic_seed=False`、prompt 改变时重置 RNG、4 steps、shift=5、guidance=3、structured prompt。三者 compile/CUDA Graph 均关闭。每任务 1 episode，原任务 step limit，不重试、不按结果筛选，不保存仿真视频。

本轮补跑 Dense 与 ASI；ToCa 闭环沿用前一轮相同配置的十任务结果（非本轮重跑）：`experiments/toca_future_smoke10_s1_p0_n2_r25_v1/`。旧路径 `s1` 为命名错误，十个实际 env seed 均是 0，原 manifest 已说明。ToCa 的**保真度与单 chunk 配对计时则在本轮重新运行**。

## 数据与运行状态

本轮数据根目录：`experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/`。

```text
dense/                         原 Dense 十任务
  simulator/episode_results.jsonl
  server/requests.jsonl
  server/captures/*/sample.pt   每任务 Dense 第 3 个请求：输入、seed、Dense 输出
asi/                           ASI 十任务；独立闭环观测
  simulator/episode_results.jsonl
  server/requests.jsonl
fidelity/                      同输入配对：10 tasks × chunk 3 × 3 methods
  paired_metrics.csv           每 task、每方法相对 Dense 的指标
  frame_metrics.csv            逐 RGB future frame 的指标
  latent_frame_metrics.csv     逐 future latent 的指标
  dense_equivalence.json       原 Dense checkout 与本地 replay 等价校验
  timing_samples.csv           相同输入交替计时的原始记录
  summary.json
  <task>/comparison.png        Dense / ASI / ToCa 三行并排预测图
  <task>/<method>/outputs.pt   action + denoised video latent
  <task>/<method>/future_*.png  future RGB 4、8、…、32 的原分辨率图片
```

`_v1` 目录只包含启动导入失败日志（脚本目录 `hydra.py` 遮蔽依赖），没有 episode；通过 Python `-P` 修正启动搜索路径后使用 `_v2`，不是失败任务重试。

## 闭环结果

| 策略 | Success | 平均 Score | 全部 episode 平均步数（含超时） | 成功 episode 平均步数 |
|---|---:|---:|---:|---:|
| Dense | 5/10（50%） | 0.50 | 282.8 | 145.6 |
| ASI：Core80+Stable104 | 3/10（30%） | 0.40 | 383.1 | 227.0 |
| ToCa-Future | 1/10（10%） | 0.20 | 464.9 | 149.0 |

逐任务，单元格为 `success / episode steps`：

| Task | Dense | ASI | ToCa-Future |
|---|---:|---:|---:|
| BananaInBowlTask | 1 / 197 | 1 / 318 | 0 / 750 |
| BananaOnPlateTask | 1 / 145 | 1 / 240 | 1 / 149 |
| ButterAboveRaisinTask | 1 / 112 | 0 / 600 | 0 / 600 |
| BowlStackingLeftOnRightTask | 0 / 300 | 0 / 300 | 0 / 300 |
| GrabABagelTask | 0 / 450 | 0 / 450 | 0 / 450 |
| GrabAFruitTask | 0 / 450 | 0 / 450 | 0 / 450 |
| LargerObjectRaisinBoxInBinTask | 0 / 450 | 0 / 450 | 0 / 450 |
| MustardInLeftBinTask | 1 / 148 | 0 / 450 | 0 / 450 |
| PickGlassesTask | 0 / 450 | 0 / 450 | 0 / 450 |
| RubiksCubeTask | 1 / 126 | 1 / 123 | 0 / 600 |

成功样本集合不同，不能只比较成功平均步数。在 Dense/ASI **共同成功的三个任务**上，Dense 平均 156 steps、ASI 227 steps。Dense/ToCa 只有一个共同成功任务，分别为 145、149 steps。

闭环观测的 generation 耗时（各任务首 chunk 排除，包含对应实现的验证，不作为主配对 speedup）：

| 策略 | 全部请求 / warm 请求 | Mean | Median | P90 |
|---|---:|---:|---:|---:|
| Dense | 95 / 85 | 0.877072 s | 0.876753 s | 0.881133 s |
| ASI | 126 / 116 | 0.600471 s | 0.600367 s | 0.604684 s |
| ToCa-Future | 152 / 142 | 1.204766 s | 1.205006 s | 1.208571 s |

三组 task 列表、每任务实际 env seed=0、每任务相同 policy seed 序列、4 steps/shift/guidance/compile 配置与 224 block calls/request 均已核对。373 个请求全部通过有限值检查。原始逐任务 success/score/steps 汇总在根目录 `closed_loop_metrics.csv`，机器可读汇总为 `comparison_summary.json`。

Success 只按 `success` 字段计数，不能根据 score 替代。RoboLab 的任务成功判定和子任务 score 可能不一致，原值均保留。每任务一次只能作为 smoke，不是可靠的逐任务成功概率估计。

## 未来帧保真度协议

1. 从 **Dense 闭环**预先固定保存每任务第 3 个 request，不按成功/失败或图像好坏选样本。
2. 对同一 `data_batch`、初始噪声 seed、采样参数分别运行 Dense、ASI、ToCa。每次深拷贝输入、创建新的 controller；原生成函数重新创建 sampler/请求内缓存，不沿用上一方法的状态。
3. 首先验证实验 worktree 的 Dense replay 与原 Dense checkout 保存的输出 `allclose(rtol=1e-4,atol=1e-4)`；不通过则停止跨版本比较。
4. 解码完整 L0–L8 保持因果 VAE 上下文，但统计时排除 condition latent L0 和 decoded frame 0。统计 L1–L8，以及 32 张 RGB future frames；action 误差排除条件 q0。
5. 每次 VAE decode 清空 decoder cache；统一 `RGB=clip((decoded+1)/2,0,1)`，不对每种方法单独做 min–max normalization。指标使用量化前 float RGB，PNG 仅供观察。

对完整样本张量展平，定义：

\[
\mathrm{cos}(X,Y)=\frac{\langle X,Y\rangle}{\|X\|_2\|Y\|_2},\qquad
\mathrm{relL2}(Y,X)=\frac{\|Y-X\|_2}{\|X\|_2+\epsilon}.
\]

其中 X 为同输入 Dense 预测。RGB 全部像素/通道的 MSE 用于

\[
\mathrm{PSNR}=-10\log_{10}(\mathrm{MSE}),\quad \text{data range}=1.
\]

SSIM 逐 RGB frame 计算再平均：11×11 Gaussian window，sigma=1.5，population covariance，RGB channels 分别计算后平均（`skimage.metrics.structural_similarity`）。RGB cosine/relative-L2 和 latent cosine/relative-L2 先在每个 chunk 上计算，再对十个 chunk 做等权平均；不先平均图片或 latent。逐 frame/latent 数值保留在 CSV。

这里的“保真度”指相对 Dense 输出的保持程度，**不是与真实未来视频对比的预测准确率**。较高 RGB cosine 也可能主要来自公共背景，需结合 relative-L2、SSIM 和并排图片观察。

## 未来帧与 action 配对结果

十个任务各固定 Dense 第 3 个请求，实际噪声 seed 均为 `1097657232`。每个样本覆盖 8 个 future latent、32 张 future RGB；表中为十个样本的等权平均。

| 相对 Dense | RGB cosine ↑ | RGB relative-L2 ↓ | PSNR ↑ | SSIM ↑ | Latent cosine ↑ | Latent relative-L2 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| ASI：Core80+Stable104 | 0.924399 | 0.383467 | 13.440 dB | 0.434179 | 0.544734 | 0.883929 |
| ToCa-Future | 0.992555 | 0.122843 | 23.391 dB | 0.739884 | 0.951701 | 0.309674 |

| 相对 Dense | Action MSE ↓ | Action cosine ↑ |
|---|---:|---:|
| ASI：Core80+Stable104 | 0.00420758 | 0.998860 |
| ToCa-Future | 0.00394157 | 0.998638 |

Action 指标对 q1–q32 的完整 `[32,8]` 输出计算，未按关节重新标准化。因此 MSE 混合了模型输出的不同 action 分量，不能解读为物理单位统一的误差，也不是速度方向或 jerk 指标。

逐任务 SSIM（同输入 Dense 为参考）：

| Task | ASI | ToCa-Future |
|---|---:|---:|
| BananaInBowlTask | 0.4230 | 0.7008 |
| BananaOnPlateTask | 0.4114 | 0.7476 |
| ButterAboveRaisinTask | 0.3996 | 0.7451 |
| BowlStackingLeftOnRightTask | 0.4933 | 0.8090 |
| GrabABagelTask | 0.4034 | 0.6743 |
| GrabAFruitTask | 0.4424 | 0.7626 |
| LargerObjectRaisinBoxInBinTask | 0.4531 | 0.7656 |
| MustardInLeftBinTask | 0.5123 | 0.8033 |
| PickGlassesTask | 0.3698 | 0.6659 |
| RubiksCubeTask | 0.4335 | 0.7246 |

十个样本中，ToCa 的 SSIM 均较高，latent relative-L2 均较低。检查香蕉与魔方并排图，ASI 在桌面背景、多视角区域存在明显块状伪影，部分物体区域也受影响；ToCa 主要呈现细节模糊和局部变化。不能因为 ASI 的 RGB cosine 仍约 0.924，就称其图像基本无损。

当前 ASI 的恢复行为见 [robolab_version1.py](../cosmos_framework/scripts/robolab_version1.py) 的 `Version1Controller.end_stack`：以 `_side_buffer`（当前 stack 输入）的副本恢复未选位置，只写回真实计算的 selected hidden。这里**没有**用旧版本的 step-0 velocity 或 block residual 补全背景。这是本次源代码核对发现的机制差异，尚未通过恢复方式消融证明它是图像伪影的唯一原因；本轮不擅自更改策略。

ToCa 的 RGB/latent 更接近 Dense，但本轮闭环成功数更少、action MSE 也不能预测闭环排序。这些观测支持把“预测图保持程度”和“任务完成能力”分别报告，不能据此断言某个图像区域导致动作失败。

### 图片入口

每张对比图从上到下为 Dense、ASI、ToCa-Future，从左到右为 decoded future frame 4、8、…、32；它们不是仿真录像或不同闭环轨迹截图。

- [BananaInBowlTask](../experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/fidelity/BananaInBowlTask/comparison.png)
- [BananaOnPlateTask](../experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/fidelity/BananaOnPlateTask/comparison.png)
- [ButterAboveRaisinTask](../experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/fidelity/ButterAboveRaisinTask/comparison.png)
- [BowlStackingLeftOnRightTask](../experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/fidelity/BowlStackingLeftOnRightTask/comparison.png)
- [GrabABagelTask](../experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/fidelity/GrabABagelTask/comparison.png)
- [GrabAFruitTask](../experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/fidelity/GrabAFruitTask/comparison.png)
- [LargerObjectRaisinBoxInBinTask](../experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/fidelity/LargerObjectRaisinBoxInBinTask/comparison.png)
- [MustardInLeftBinTask](../experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/fidelity/MustardInLeftBinTask/comparison.png)
- [PickGlassesTask](../experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/fidelity/PickGlassesTask/comparison.png)
- [RubiksCubeTask](../experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/fidelity/RubiksCubeTask/comparison.png)

全精度统计见 [paired_metrics.csv](../experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/fidelity/paired_metrics.csv)。各策略目录中保留八张原分辨率 PNG，以及完整 action/latent 的 `outputs.pt`。

## 单 chunk 计时口径

主速度比较用同一加载模型、同一个 BananaInBowlTask Dense chunk-3 输入和 seed：每方法预热两次，随后循环轮换顺序运行 30 轮，每轮 Dense/ASI/ToCa 各一次。统计同步后的 generation Mean、Median、P90，速度比 `Dense median / method median`。

包括原生成函数、profile/评分、选择、cache、controller 和必要验证；排除输入的 Python 深拷贝、CPU 输出复制、VAE decode、文件保存、RPC 和仿真。ToCa 为当前未融合 FP32 全 GEN incoming-score 参考实现，其评分与检查开销不隐藏；本轮不能代表该方法所有可能的优化实现。

另列闭环观测 chunk 耗时。闭环各策略输入、结束步数、GPU 共驻负载不同，不把该观测值当作严格配对的主速度比。

RTX 4090，compile/CUDA Graph 均关闭，离线计时期间没有仿真器占用 GPU，每方法 30 个实测值：

| 策略 | Mean | Median | P90 | 相对 Dense 加速 |
|---|---:|---:|---:|---:|
| Dense | 0.868866 s | 0.869469 s | 0.870333 s | 1.000× |
| ASI：Core80+Stable104 | 0.594365 s | 0.594565 s | 0.595716 s | 1.462× |
| ToCa-Future | 1.197881 s | 1.198353 s | 1.199696 s | 0.726× |

ASI 的 generation median 减少约 31.6%；ToCa 当前参考实现反而增加约 37.8%。这是包含实际评分、控制和验证成本的结果，不是理论 FLOPs 比例；不能把 ToCa 参考移植的耗时当成官方 ToCa 的最佳性能。原始计时见 [timing_samples.csv](../experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/fidelity/timing_samples.csv)。

## 验证与修改范围

- 10/10 Dense replay 的 action/vision 与原 Baseline 保存张量逐元素完全一致，max absolute error 和 MSE 均为 0；比预设 `allclose` 门槛更严格。
- 配对结果完整：20 行 chunk 指标、640 行 RGB frame 指标、160 行 latent 指标、90 行计时记录；没有缺失样本或 NaN/Inf。
- 保存 10 张三行对比图、240 张原分辨率 future PNG、30 份 action/latent 输出和 10 份 ASI mask 数据。
- CPU 测试：`tests/test_future_fidelity_metrics.py` 与 `tests/test_toca_future.py`，13 passed。CPU 测试只验证工具逻辑，实际输出和时间来自本地 Edge 环境的 GPU 执行。
- 本轮新增 Python 文件的 Ruff lint 与 format check 均通过。
- 本轮只新增比较服务器、闭环 runner、配对生成/指标工具、测试与本文档；未修改 Dense、ASI 或上一轮 ToCa 的算法实现。
- 结束后原 Dense 为 `cache@324574a454a989f9b8f5392f7673ace487243e8d`，原 ASI 为 `version/core80-stable104-action-weighted@62a09f7f95f1f0a54d5af9c2b8c530c862db9025`，两者 `git status --porcelain` 均为空，与执行前一致。
- 实验分支仍为 `experiment/toca-future`，worktree `/root/robolab/worktrees/toca-future`；没有 commit、merge 或 push。GPU 测试进程已退出，8018 端口已释放。

## 复现

使用全新输出目录，不覆盖已有实验：

```bash
cd /root/robolab/worktrees/toca-future
env LD_LIBRARY_PATH='' \
  /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  tools/run_dense_asi_smoke.py \
  --run-dir /root/robolab/worktrees/toca-future/experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v3 \
  --port 8018
```

闭环全部结束、GPU 空闲后，再运行离线保真度和配对计时（下例指向本轮 `_v2` 的 Dense captures；若已存在 fidelity 输出则换一个新目录）：

```bash
cd /root/robolab/worktrees/toca-future
env CUDA_VISIBLE_DEVICES=0 COSMOS_TRAINING=0 LD_LIBRARY_PATH='' \
  HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/root/robolab/worktrees/toca-future:/root/robolab/cosmos-edge-overlay \
  /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -m cosmos_framework.scripts.paired_future_fidelity \
  --run-dir /root/robolab/worktrees/toca-future/experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2 \
  --output-dir /root/robolab/worktrees/toca-future/experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/fidelity \
  --timing-repeats 30
```

汇总并校验任务、实际 seed、请求序列、策略配置，合并闭环与配对结果：

```bash
cd /root/robolab/worktrees/toca-future
env LD_LIBRARY_PATH='' \
  /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  tools/aggregate_dense_asi_toca.py \
  --run-dir /root/robolab/worktrees/toca-future/experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2 \
  --toca-dir /root/robolab/worktrees/toca-future/experiments/toca_future_smoke10_s1_p0_n2_r25_v1
```

所有新增代码、测试、文档都在 `experiment/toca-future` 工作树；原 Dense/ASI 工作树不修改，不自动 commit/push。实验数据被 gitignore，不安装环境或额外模型。
