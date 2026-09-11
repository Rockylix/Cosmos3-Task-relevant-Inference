# ASI 被移除 token 的 velocity cache：单 chunk 未来帧对照

## 当前状态

2026-09-11 10:44（Asia/Shanghai）：**四组真实 GPU 推理和 VAE 解码全部完成**，每组保存 32 张 640×528 未来帧。13 项 CPU 测试重新通过；全部 GPU token 行数、有限值、配对输入与缓存替换检查通过。

用户重启后，内核模块、磁盘模块及用户态驱动统一为 `580.178.04`；项目原环境 PyTorch `2.10.0+cu130` 的 CUDA 分配正常。本轮使用 RTX 4090，不创建环境、不下载权重、不采用 LD_PRELOAD 兼容补丁。

此前阻塞是内核仍加载 `580.173.02`、用户态已更新到 `580.178.04` 引起的 NVML mismatch / CUDA 804；重启后已解决。本实验没有修改系统驱动。

## 实测结果

结果目录：`../experiments/asi_velocity_cache_banana_c3_s5_v1/`。

- [四组未来帧对照图](../experiments/asi_velocity_cache_banana_c3_s5_v1/comparison.png)
- [Velocity cache 的全部 32 帧预览](../experiments/asi_velocity_cache_banana_c3_s5_v1/asi_velocity_cache/all_future_frames.png)
- [Velocity cache 最后一张未来帧](../experiments/asi_velocity_cache_banana_c3_s5_v1/asi_velocity_cache/future_frames/frame_032.png)
- [原始指标](../experiments/asi_velocity_cache_banana_c3_s5_v1/metrics.json)、[缓存验证](../experiments/asi_velocity_cache_banana_c3_s5_v1/asi_velocity_cache/audit.json)、[完成标记](../experiments/asi_velocity_cache_banana_c3_s5_v1/completion.json)

所有误差都相对**本轮同输入 Dense**；action 只取预测的 32 个 horizon，latent 排除 L0，RGB 排除条件首帧，使用解码后固定映射到 `[0,1]` 的浮点图像（不是 PNG 量化值）。输入来自此前 ASI 闭环轨迹的 chunk 3，不是另一个 Dense 闭环轨迹。

| 策略 | Action MSE ↓ | Action cosine ↑ | Latent cosine ↑ | Latent rel-L2 ↓ | RGB cosine ↑ | RGB rel-L2 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0 | 1.000000 | 1.000000 | 0 | 1.000000 | 0 |
| ASI A：当前策略 | 0.00319143 | 0.999180 | 0.576678 | 0.851268 | 0.935322 | 0.356371 |
| ASI B：Step0 双分支全量 | 0.00765804 | 0.997948 | 0.599727 | 0.830229 | 0.938252 | 0.349345 |
| ASI C：B + velocity cache | 0.00765804 | 0.997948 | 0.933428 | 0.360988 | 0.992765 | 0.120085 |

本 chunk 的观察：

1. C 相比 B 明显减少了背景的块状伪影，但后期未来帧仍有模糊、物体形状偏差；不是恢复为 Dense 的精确视频。
2. **B/C action 逐元素完全相同**（两者互比 MSE=0、cosine=1）；保留区域的最终 vision latent 也逐元素相同。本轮变化只发生在被移除位置的最终 latent，而 VAE 解码可以传播其图像影响。
3. C 相比当前 ASI A 的 action MSE 反而更高；这在 B 中已经出现，不能归因于添加 velocity cache。A→B 额外把 Step0 unconditional 改为全量，是另一项干预。
4. 因此，本实验支持“缓存改善本 chunk 被移除区域的未来帧重建”，不支持“缓存改善 action”。这些位置在后续步仍不参与 Transformer attention，缓存不恢复背景 K/V。
5. 只验证一个离线 chunk，不宣称任务成功率提高，也没有进行稳定单 chunk 性能测量。

配对与真实计算验证：

- ASI A 的 action / vision 与原始 capture **逐元素完全一致**。
- B/C 的 Step0 guided velocity 与 Dense **逐元素完全一致**，最大绝对差 0。
- A/B/C mask 完全一致：每帧保留 184/340，移除 156/340；8 帧合计保留 1472、移除 1248 个 future tokens。
- 每个稀疏 forward 的真实 GEN 行数从 3093 降到 1845，含全量 L0=340、保留 future=1472、action=33；全部 28 层 Q/K/V/O/MLP 均核验，不是计算后 mask。
- C 的 Step1/2/3 每步精确替换 233088 个 velocity 标量；保留区域、L0、action 在替换操作前后完全不变。
- 每组 224 个 block 输出有限，四步 guided/mixed velocity 和最终 action/latent/RGB 无 NaN/Inf。
- 原 baseline 保持 `cache@324574a454a989f9b8f5392f7673ace487243e8d` 且干净；原 ASI 仍是 `version/core80-stable104-action-weighted@62a09f7f95f1f0a54d5af9c2b8c530c862db9025`，其已有 4 个未跟踪文件保留不变。

## 隔离与实验定义

- worktree：`/root/robolab/worktrees/asi-velocity-cache`
- branch：`experiment/asi-velocity-cache`
- 基于当前 ASI HEAD：`62a09f7f95f1f0a54d5af9c2b8c530c862db9025`
- 原 ASI 工作树及 baseline 不修改，不自动 commit/merge/push。
- 共用 `/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv`，不创建环境或下载权重。

固定已采集的 `BananaInBowlTask` 第 3 个 chunk：

`/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/experiments/asi_smoke10_s0_p0_v1/_temporary_inputs/BananaInBowlTask.pt`

相同 data_batch、初始 noise seed=`1097657232`、4 steps、shift=5、guidance=3。每组重置 RNG、controller、request-local cache，原生 UniPC 每次重新初始化；关闭 compile/CUDA graphs。

| 对照 | Dense / sparse CFG forwards | 跳过区域的 velocity |
|---|---|---|
| Dense | 8 / 0 | 原生全量预测 |
| ASI A，当前策略 | 1 / 7 | 当前 ASI 输出头预测 |
| ASI B，Step0 双分支全量 | 2 / 6 | 当前 ASI 输出头预测 |
| ASI C，B + velocity cache | 2 / 6 | Step0 全量 guided velocity |

所有 ASI 组只用 Step0 conditional profile 生成同一个 Core80+Stable104 mask；额外全量 unconditional 不重复 profile、不重复选择。B/C 单独控制额外一次全量计算的影响。

对 future token 的保留 mask 定义 M=1，被移除区域 M=0。Step0 保存真实双分支全量计算、原生 CFG 完成后的 velocity；后续 step1/2/3：

\[
\widetilde v_s=M\odot v_s^{ASI}+(1-M)\odot v_0^{Dense}.
\]

这是 **替换**，不是额外相加。只缓存被移除的 future 位置；L0 与全部 action 保持原生输出，不缓存 action，不跨 chunk 复用。

执行顺序：稀疏 Transformer → 原生输出头/unpatchify → 条件 velocity mask → 原生 CFG（及若启用的 normalization）→ **移除区域 velocity 替换** → **原生 UniPC**。

17×20 token mask 按真实 `patch_spatial` 展开到 latent patch 像素，裁去底部/右侧 padding，再按通道扩展；不是双线性缩放。仅维护完整 solver 状态，不修改求解器或多步历史。背景仍未参加中间层 attention，缓存不等价于恢复其 K/V。

## 运行命令

先确认 `nvidia-smi` 正常，且 GPU 无其他实验占用。再运行：

```bash
cd /root/robolab/worktrees/asi-velocity-cache
unset LD_PRELOAD
export CUDA_VISIBLE_DEVICES=0
export COSMOS_TRAINING=0
export LD_LIBRARY_PATH=''
export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/root/robolab/worktrees/asi-velocity-cache:/root/robolab/cosmos-edge-overlay

/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -m cosmos_framework.scripts.asi_velocity_cache_chunk \
  --output /root/robolab/worktrees/asi-velocity-cache/experiments/asi_velocity_cache_banana_c3_s5_v1
```

程序在 CUDA 不可用时提前退出，不生成看似已完成的图像目录。输出目录若已存在也拒绝覆盖；需要用新后缀或先确认已有结果状态。

## 已生成输出

- `comparison.png`：四组同一时间点的未来帧并排；列是 future RGB 帧 1、4、8、12、16、20、24、32，不把它们误标为 latent L1–L8。
- 每组 `future_frames/frame_001.png` 至 `frame_032.png`：完整分辨率未来帧，排除条件首帧。
- 每组 `all_future_frames.png`：32 帧预览；`outputs.pt`：action、latent、mask、Step0 guided velocity；`audit.json`：采样步骤、实际 token 数和 finite 校验。
- `metrics.json`：相对同输入 Dense 的 action、future latent、RGB 的 FP64 MSE/cosine/relative-L2。
- `manifest.json`：输入哈希、seed、参数、源码哈希；`completion.json` 仅全部检查通过后写出。

GPU 验证门槛：所有 28 层 Q/K/V/O/MLP 的真实行数符合 `[3093]×8` / `[3093]+[1845]×7` / `[3093]×2+[1845]×6`；全部 224 个 block 输出 finite；A 重放匹配原已采集 ASI 输出；A/B/C mask 完全一致；B/C Step0 guided velocity 与 Dense 匹配；缓存替换后保留区/action/L0 不变，被移除区等于缓存。

不报告未经预热的计时，不重跑闭环任务，不推断成功率。单 chunk 结果也不能证明该缓存适用于所有任务/step。

## CPU 测试

```bash
cd /root/robolab/worktrees/asi-velocity-cache
OMP_NUM_THREADS=1 PYTHONPATH="$PWD" \
  /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -m pytest --noconftest tests/test_robolab_version1.py tests/test_asi_velocity_cache.py -q
```

13 passed。包含真实 CPU UniPC 结果与独立显式混合 velocity 参考逐位一致；不是 GPU 模型推理测试。
