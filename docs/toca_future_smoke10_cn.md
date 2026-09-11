# ToCa-Future：RoboLab 十任务 smoke

日期：2026-09-10。状态：**10/10 episodes 已完成，success=1/10（10%），平均 score=0.20**；未重试任何任务。

## 本轮测什么

验证新分支的 **ToCa-Future（Edge adaptation）** 能否真实执行未来视觉 token 的模块缓存，并完成十任务闭环。不是已有 ASI/Core80 策略换名字，也不是官方 ToCa 对机器人模型的原版结果。

- 分支：`experiment/toca-future`。
- worktree：`/root/robolab/worktrees/toca-future`。
- 基础 HEAD：`62a09f7f95f1f0a54d5af9c2b8c530c862db9025`。
- 实验路径：`experiments/toca_future_smoke10_s1_p0_n2_r25_v1/`。
- 原 Dense 与 ASI checkout 不改动；本轮代码尚未 commit/push。

路径中的 `s1` 是启动时的命名错误，**实际 simulation seed 为 0**。RoboLab 注册配置的 seed=1 被 runner 调用 `create_env` 的默认 seed=0 覆盖（`RoboLab/robolab/eval/runner.py:207`、`RoboLab/robolab/core/environments/runtime.py:76`）；已逐任务检查 `env_cfg.json` 并纠正 `manifest.json`，保留目录名便于核对原日志。这不是先用 seed=1 跑失败后换 seed 重试。

## 固定配置与真实计算

| 参数 | 本轮设置 |
|---|---|
| GPU / 模型 | RTX 4090 / 本地 Cosmos3-Edge-Policy-DROID |
| Policy seed | 0，`deterministic_seed=False`；prompt 改变时重置 RNG，每个实际 request seed 均记录 |
| Simulation seed | 0，来自当前 runner 的实际环境配置 |
| 去噪 | UniPC，4 steps，shift=5，guidance=3，structured prompt |
| Full/cache 调度 | step 0/2 为 full；step 1/3 为 cached；两分支全部 B0–B27 |
| Future MLP fresh ratio | `rho_l=0.25*(1.5-l/27)`，`K_l=floor(2720*rho_l)` |
| ToCa 分数 | full step 的全 GEN-query incoming attention，CFG 分数平均，L2 归一化，年龄项 `0.25*age/2` |
| 编译 | torch compile=False，CUDA graphs=False |
| 闭环 | 固定十个任务，各 1 episode；使用任务原有 step limit，不重试、不按结果筛选 |
| 图像/视频 | 不生成仿真视频，不运行额外 VAE decode；保留首请求输入和配对 latent/action 供核查 |

调度中的 step 指执行序号。按当前未修改的 UniPC scheduler 在 CPU 重建：timestep 为 `[999,937,833,624]`，sigma 为 `[0.99979985,0.93726546,0.83305538,0.62468737,0]`；这是源码重建值，不是额外 GPU hook 采集值。

每个请求重新创建 ToCa cache，不跨 chunk 复用。每个 full step 更新 attention 投影后输出和 MLP 输出的缓存；cached step 将缓存输出加入**当前** hidden residual 主干，而不是拿上一步完整 hidden 覆盖当前输入。

cached step 中：

1. L0 和全部 action 的 Q/O、attention、MLP 真实重算。
2. 全部 GEN 的当前 K/V 真实重算；action attention 仍可看到完整 video key。
3. L1–L8 的 attention 输出复用最近 full step；MLP 仅重算 ToCa 选中的 future 行。
4. 未重算的 MLP 行复用缓存，按原位置拼回完整序列，进入下一层。
5. RoPE 沿用原位置，不重新编号；原输出头、CFG、UniPC 不变。

Incoming score 使用全部有效 UND+GEN key 的 softmax 分母，真实 GQA 16:8 映射，FP32 分块 QK/softmax 归约，额外评分开销计入时间。模型 attention 输出继续由原高效 kernel 计算；没有将全量 Q/MLP 先计算后丢弃。

在本轮 `D,C,D,C` 调度下，缓存步之前所有年龄刚被 full step 清零，因此年龄项不改变当次 Top-K 排名；不能据此声称已验证跨多个连续缓存步的年龄效果。

## 结果

| 任务 | Success | Score | 实际 episode steps |
|---|---:|---:|---:|
| BananaInBowlTask | 0 | 0.0 | 750 |
| BananaOnPlateTask | 1 | 1.0 | 149 |
| ButterAboveRaisinTask | 0 | 0.0 | 600 |
| BowlStackingLeftOnRightTask | 0 | 0.0 | 300 |
| GrabABagelTask | 0 | 1.0 | 450 |
| GrabAFruitTask | 0 | 0.0 | 450 |
| LargerObjectRaisinBoxInBinTask | 0 | 0.0 | 450 |
| MustardInLeftBinTask | 0 | 0.0 | 450 |
| PickGlassesTask | 0 | 0.0 | 450 |
| RubiksCubeTask | 0 | 0.0 | 600 |
| **总计 / 平均** | **1/10** | **0.20** | **464.9** |

成功 episode 的平均完成步数为 **149**，但只有一个成功样本，不构成稳定的效率估计。GrabABagelTask 虽然 score=1.0，官方结果的 `success=False`，因此不计为成功；其 reason 为 `No object completed. Best progress: object_grabbed(object=bagel_02) (step 1/1)`。保留原始判定，不用 score 修正 success。

**生成单 chunk 耗时**：152 个实际 ToCa 请求，排除各任务首个 chunk 后共 142 个样本。

| 边界 | Mean | Median | P90 |
|---|---:|---:|---:|
| 同步后的 generation wall time | 1.204766 s | 1.205006 s | 1.208571 s |

计时包含四步两分支推理、full-step 评分、Top-K、cache 读写、gather/scatter、controller 建立/恢复和有限值检查；不含仿真物理、RPC、VAE decode、首请求 Dense 验证与普通请求的日志落盘。首请求带额外 trace/验证，已排除。

这是**仿真共驻期间的闭环观测耗时**，不是相同输入预热交替的三策略 benchmark。本轮没有同轮 Dense/ASI 时间，因此不给 speedup，也不拿历史约 0.85 s 的 Dense 数字相除。当前评分是未融合的 FP32 分块参考实现，不能把“模块行数减少”直接当作实际加速。

本工作点的闭环成功数较低；不据此单独断言 ToCa 方法不适用，也不声称与 Dense 等价。要判断损失来源，需要同初态 Dense 配对，以及独立的调度/刷新率消融。

## 已通过的实现验证

1. CPU 单元测试 **10/10 通过**：候选/保护域分区、full-denominator incoming score 与分块/GQA 对照、确定性 Top-K/年龄更新、原始 RoPE 位置、真实缩短输入行数、部分 MLP cache 更新、CFG 共享选择但独立值缓存、当前 residual 主干不被旧 hidden 覆盖、step/branch gating、context 异常清理。
2. 首个实际 GPU 请求（BananaInBowlTask，chunk 1）先运行原生 Dense，再运行全部 full 的 ToCa observer：action 和 vision 的 **MSE=0、max absolute error=0**。这验证此 checkout 的原生路径和全量采集等价；不是另一个 checkout/历史图像的跨版本等价证明。
3. 同一请求、深拷贝 data_batch、相同显式 seed/采样参数，随后运行 ToCa，**向 RoboLab 返回 ToCa action**。原生成函数每次重新创建 scheduler 与 request-local UND cache。
4. 在 Q/K/V/O/MLP 模块实际入口采集行数，共 28 blocks × 5 modules × 8 forwards，全部与调度预期一致。证据：`server/actual_module_rows.json`。
5. 每个实际请求检查 8 次 GEN forward、224 次 block 调用及完整顺序，并检查每个 block 输出与最终 action/vision 的 NaN/Inf。最终 **152/152 请求通过**；总计 34,048 条 block 记录，full/cached 各 17,024 条。
6. 核对十个唯一任务、每任务恰好一条结果、十个实际 env seed 均为 0；控制器和服务器源文件 SHA256 与 run manifest 一致，未在运行中调整算法。Ruff 检查与格式检查通过。结束后 GPU 无计算进程，8017 无监听。

真实 GEN layout：L0=340，L1–L8=2720，action=33（condition q0 + 32 个预测 action），合计 3093。下表为 **GEN 模块行数**，不是整块 FLOPs 或含 UND 的总 token 比例。

| 模块 | Full forward 行数 | Cached forward 行数 | 全 8 次 forward 行数减少 |
|---|---:|---:|---:|
| Q projection / O projection | 3093 | 373 | 43.970% |
| K projection / V projection | 3093 | 3093 | 0% |
| MLP | 3093 | `373+K_l`，B0=1393，B27=713 | 32.985% |

首请求配对误差（只含预测部分，排除 action q0 与 video L0）：

| 输出 | MSE | Cosine | Relative L2 | Max absolute error |
|---|---:|---:|---:|---:|
| 32×8 future action | 0.00708204 | 0.998375 | 0.067755 | 0.282855 |
| L1–L8 denoised vision latent | 0.07966489 | 0.951708 | 0.311992 | 2.998745 |

Action 来自 `generate_samples_from_batch` 的 `ActionProcessor.postprocess_action` 后的 8 维 external action，整体 MSE 混合了不同 action 分量，不能解释为单一物理单位的误差；不等于客户端后续动作变换后的逐关节误差。Vision 是 latent，不是 RGB。原 `paired_chunk_metrics.json` 包含 q0/L0；本表对 `paired_outputs.pt` 离线排除条件部分后重新计算。

这只是一条初始请求的配对误差，不能推断整条闭环轨迹或其他任务精度。本轮没有额外采集 BF16 原生 kernel 与重算 AV 的 GPU 逐元素对照，也没有完成 Dense/ASI/ToCa 同轮正式性能评测；CPU 参考测试和全量 observer 等价不可替代这些扩展验证。

## 复现命令

在本机同样的现有环境/资产布局下：

```bash
cd /root/robolab/worktrees/toca-future
/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  tools/run_toca_future_smoke.py \
  --run-dir /root/robolab/worktrees/toca-future/experiments/toca_future_smoke10_s0_p0_n2_r25_v2 \
  --port 8017
```

例子采用一个新的、seed 标注正确的输出路径；若已存在则更换唯一后缀。脚本拒绝覆盖旧 run，检查本地权重/VAE/端口，不安装环境、不下载资产。它打印完成进度、success 和平均 score；只关闭自己创建的服务与仿真进程。十个任务名称及展开的两端命令见 `tools/run_toca_future_smoke.py` 和 run 内 `manifest.json`。

测试：

```bash
cd /root/robolab/worktrees/toca-future
env LD_LIBRARY_PATH='' COSMOS_TRAINING=0 \
  PYTHONPATH=/root/robolab/worktrees/toca-future \
  /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -m pytest tests/test_toca_future.py -q
```

## 文件与边界

- 原始成功率依据：run 内 `simulator/episode_results.jsonl`；逐任务 `env_cfg.json` 固定实际环境参数。
- 配置与核对：`manifest.json`、`server/runtime.json`、`audit.json`。
- 请求/计算量：`server/requests.jsonl`、`server/module_compute.csv`。
- 首请求验证：`server/all_full_equivalence.json`、`server/actual_module_rows.json`、`server/paired_input.pt`、`server/paired_outputs.pt`。
- 日志：`server.log`、`simulator.log`；最终汇总：`summary.json`。
- 数据位于 gitignored 的 `experiments/`，代码、测试和 docs 可提交；本轮没有自动 commit/push。

原工作树最终核对：

| Checkout | Branch | HEAD | `git status --short` |
|---|---|---|---|
| `/root/robolab/cosmos-framework-edge` | `cache` | `324574a454a989f9b8f5392f7673ace487243e8d` | 空，与开始时一致 |
| `/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted` | `version/core80-stable104-action-weighted` | `62a09f7f95f1f0a54d5af9c2b8c530c862db9025` | 空，与开始时一致 |

日志中启动时的 `opening handshake failed` 来自运行器 TCP 端口就绪探测后关闭连接，发生在仿真请求之前；不是一次失败的推理或额外 episode。未将该日志删除或改写。

十个任务各一条轨迹只能称 smoke，不能声称稳定成功率、与 ASI/Dense 持平、或取得加速。正式对比还需相同 task×seed 配对闭环，以及相同输入、预热交替运行的单 chunk benchmark。
