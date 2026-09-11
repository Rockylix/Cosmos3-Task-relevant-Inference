# ToCa D/C/C/C 单 chunk 加速测试

日期：2026-09-10。分支：`experiment/toca-future`。仅新增基准脚本，没有修改 controller、kernel 或默认部署配置。

## 结果

RTX 4090，BananaInBowlTask 的 Dense 第三个 request，三组在同一个进程、同一个加载模型上测试。每组预热 5 次，之后 30 轮轮换顺序；下表使用同轮 Dense 作分母，不拼接历史时间。

| 策略 | Mean | Median | P90 | 相对 Dense 加速 |
|---|---:|---:|---:|---:|
| Dense eager | 0.861975 s | 0.863006 s | 0.867317 s | 1.000× |
| ToCa joint D/C/D/C | 0.778385 s | 0.779886 s | 0.782751 s | 1.107× |
| ToCa joint D/C/C/C | 0.595685 s | 0.596389 s | 0.598955 s | **1.447×** |

D/C/C/C 相对 D/C/D/C 再快 **1.308×**，延迟减少约 **23.5%**；相对 Dense 延迟减少约 **30.9%**。这是稳定单 chunk generation 时间，不是完整任务或 RPC 加速。

## 配置与计时边界

- 原输入：`experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/dense/server/captures/request_000002/sample.pt`。
- task：BananaInBowlTask；chunk：3；扩散 seed：`1097657232`；4 steps，shift=5，guidance=3。
- 模型：现有 `RoboLab/Cosmos3-Edge-Policy-DROID`，现有项目 Edge venv；没有下载环境或权重。
- compile、CUDA Graph 均关闭；Dense 不安装任何稀疏 controller。
- D/C/D/C：`full_steps=(0,2)`、`period=2`；D/C/C/C：`full_steps=(0,)`、`period=4`。周期同时用于缓存年龄归一化，不改变年龄权重 0.25。
- 两种 ToCa 都用 `attention_backend="joint"`；基础 MLP 重算比例 0.25，层调度斜率 0.5。
- 两个 CFG 分支、全部 28 blocks 应用相同调度。缓存步 future attention residual 复用；L0/action 实时计算 Q/O，GEN K/V 始终完整，MLP 按既定分数部分重算。
- 每次调用前深拷贝输入、重置 RNG，新建 controller。原生 generation 每次重新生成同 seed 初始噪声、新建 UND KV cache 和 UniPC scheduler，不共享跨次或跨 chunk 缓存。
- CUDA synchronize 包围 generation，计入 controller、评分、选取、cache 更新、完整序列恢复及正常有限值检查；排除输入深拷贝、RNG 重置、CPU 输出复制、重复一致性对照、VAE decode、文件 IO 和仿真。

## 检查与误差边界

三种模式的 action/vision 均有限；两个 ToCa 的全部 block 输出均通过 NaN/Inf 检查。每种模式正式 30 次输出均与该模式自身的首次输出逐元素一致。Dense 重放通过历史输入对应 Dense 输出的数值一致性检查。

执行表检查：

- 每个 ToCa chunk 共 8 次 Transformer forward、224 次 block 调用。
- D/C/D/C：112 次 full block 调用、112 次 cached block 调用。
- D/C/C/C：56 次 full block 调用、168 次 cached block 调用。
- cached block 的 GEN K/V 行数为 3093，Q/O 行数为 373；非计算结束后才 mask。

同一个 chunk，相对当前 Dense 的输出误差：

| 策略 | Action MSE ↓ | Action cosine ↑ | Action rel-L2 ↓ | Future latent cosine ↑ | Future latent rel-L2 ↓ |
|---|---:|---:|---:|---:|---:|
| D/C/D/C | 0.00648760 | 0.998311 | 0.058586 | 0.938684 | 0.349513 |
| D/C/C/C | 0.01594103 | 0.996528 | 0.091836 | 0.902735 | 0.441420 |

Action 只取预测 q1–q32，vision 只取 future L1–L8；分别整体展平。这里是一份固定请求的数值差异，不是十任务平均。D/C/C/C 更快，但该样本误差增大；未做闭环成功率、VAE 解码或视频质量测试，不据此声称任务效果不变。

## 复现

在空闲 GPU 上执行；输出目录必须不存在：

```bash
cd /root/robolab/worktrees/toca-future
env CUDA_VISIBLE_DEVICES=0 COSMOS_TRAINING=0 LD_LIBRARY_PATH='' \
  HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/root/robolab/worktrees/toca-future:/root/robolab/cosmos-edge-overlay \
  /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -u -m cosmos_framework.scripts.benchmark_toca_schedules \
  --capture experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/dense/server/captures/request_000002/sample.pt \
  --output-dir experiments/toca_schedule_dccc_banana_c3_v2 \
  --warmups 5 --repeats 30
```

已完成结果在 `experiments/toca_schedule_dccc_banana_c3_v1/`：

- `summary.json`、`timing.csv`：三组统计与全部 90 次原始时间。
- `manifest.json`：环境、输入、seed、策略配置和源码 SHA256。
- `checks.json`、`module_compute_labeled.csv`：计算行数和完整调度记录。原 `module_compute.csv` 保留；其 mode 为 full/cached，按执行顺序前 224 行 D/C/D/C、后 224 行 D/C/C/C 补上 strategy 列，没有修改计算值。
- `benchmark_executed.py`：实际执行的脚本快照，SHA256 与 manifest 对应。当前脚本已修正上述 strategy 列命名，计时和推理逻辑不变。
- `fidelity.csv`、`{dense,toca_dcdc,toca_dccc}_outputs.pt`：误差和输出。
- 同级 `toca_schedule_dccc_banana_c3_v1.log`：完整执行日志。

源码：[benchmark_toca_schedules.py](../cosmos_framework/scripts/benchmark_toca_schedules.py)。原 Dense 和 ASI 工作树未修改，没有 commit 或 push。
