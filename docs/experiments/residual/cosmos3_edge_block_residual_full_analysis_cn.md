# Cosmos3 Edge future-frame block residual 全量分析

## 1. 目标与边界

本实验逐个保留 Cosmos3 Edge 的 denoise timestep、Transformer block 和 CFG path，不跨 timestep 或 branch 平均。对于 future vision latent `L1..L8`，采集每个 block 的真实输入和输出并定义：

\[
R_f^l = H_{f,\mathrm{out}}^l-H_{f,\mathrm{in}}^l
\]

每个 residual slice 的 shape 为：

```text
[future_latent=8, spatial_token=340, hidden_dim=2048]
```

对所有有序帧对 `(f,g)` 计算：

\[
C_{fg}^l=\cos(\operatorname{vec}(R_f^l),\operatorname{vec}(R_g^l))
\]

\[
E_{fg}^l=
\frac{\|R_f^l-R_g^l\|_F}{\|R_f^l\|_F+\epsilon}
\]

注意 `E_fg` 的分母只使用 `R_f`，所以 relative-L2 矩阵通常不对称。按 frame gap 汇总时，脚本统计全部 `f != g` 的有序 pair。

Residual 相对 block 输出的大小按下式实现：

\[
M_f^l=\frac{\|R_f^l\|_F}{\|H_{f,\mathrm{out}}^l\|_F+\epsilon}
\]

本实验不预设 anchor、不做插值、不修改 attention kernel，也不声称可以加速或保持 action 不变。

## 2. 实现

### 2.1 真实 block residual 采集

新增文件：

```text
/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/robolab_block_residual_capture.py
```

关键实现：

- network pre-hook 从实时 `SequencePack` 解析 vision token 位置和 `[T,H,W]`，然后排除第一个 latent frame `L0`。
- 每个 Transformer block 的 pre-hook 对 future token 的 `H_in` 做 `detach().clone()`；因此 block 内即使存在原地修改，也不会污染输入快照。
- 同一 block 的 post-hook 获取 `H_out`，严格计算 `H_out-H_in`。这也覆盖 B0，不使用“相邻 block 输出相减”的近似。
- raw residual 以 BF16/FP16 mmap 流式写盘；另存 residual Frobenius norm、block-output Frobenius norm 和二者比值。
- call profile 保存 `sampler_step_by_call`、源码真实 `timestep_by_call` 和 `cfg_branch_by_call`，conditional/unconditional 完全分开。
- 采集完整性、重复 hook、shape/dtype 改变、磁盘余量、NaN/Inf norm 都会触发错误。

代码位置：

- collector 和 planner：`/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/robolab_block_residual_capture.py:33`
- 实时 token layout 和 L0 排除：`/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/robolab_block_residual_capture.py:124`
- block pre-hook：`/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/robolab_block_residual_capture.py:257`
- block post-hook 与 residual：`/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/robolab_block_residual_capture.py:317`
- call/step/timestep/branch profile：`/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/robolab_block_residual_capture.py:357`
- artifact 对 future-only scope 的声明：`/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/robolab_block_residual_capture.py:462`

### 2.2 Policy server 开关

新增默认关闭的参数：

```text
--block-residual-capture-dir PATH
--block-residual-capture-chunks INT [INT ...]
--block-residual-capture-disk-reserve-gib FLOAT
```

只有同时传入目录和 chunk 列表时才启用。它与已有 hidden-state、RoPE Q/K capture 互斥。启用后 server 自动关闭 torch compile 和 CUDA graphs，保证 Python hook 能观察真实 block 调用；正常部署路径不受影响。

代码位置：

- 参数：`/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/action_policy_server_robolab.py:346`
- 参数校验和 capture 互斥：`/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/action_policy_server_robolab.py:411`
- planner 初始化：`/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/action_policy_server_robolab.py:539`
- 关闭 compile/CUDA graph：`/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/action_policy_server_robolab.py:610`
- infer 中安装 collector 和保存 artifact：`/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/action_policy_server_robolab.py:775`

### 2.3 全量分析脚本

```text
/root/robolab/cosmos-edge-overlay/tools/analyze_edge_block_residuals.py
```

脚本会：

1. 对 8 个 future latent 逐 block 构造完整 `8×8` cosine 和 directed relative-L2 矩阵。
2. 保持 task、chunk、branch、step、真实 timestep、block 字段，输出每一个原始矩阵元素。
3. 按 gap `1..7` 输出 mean、std 和样本数。
4. 输出每帧 residual norm、block-output norm 和 `M`。
5. 生成每个 branch/step 的 28-block 矩阵总图、gap 曲线和 block×frame 的 `M` heatmap。
6. 用同为 gap=1 的相邻 pair 做受控 boundary deficit，避免把“跨 cut 的 pair 平均更远”误判为自然分段。
7. 检查 raw、矩阵和 norm 的 NaN/Inf，并显式保证 `C_ff=1`、`E_ff=0`。

代码位置：

- pairwise relative-L2：`/root/robolab/cosmos-edge-overlay/tools/analyze_edge_block_residuals.py:226`
- gap 汇总：`/root/robolab/cosmos-edge-overlay/tools/analyze_edge_block_residuals.py:295`
- gap=1 受控 boundary：`/root/robolab/cosmos-edge-overlay/tools/analyze_edge_block_residuals.py:298`
- raw NPY/CSV 输出：`/root/robolab/cosmos-edge-overlay/tools/analyze_edge_block_residuals.py:342`
- 汇总与保守结构标签：`/root/robolab/cosmos-edge-overlay/tools/analyze_edge_block_residuals.py:479`

## 3. 测试

CPU 测试覆盖：

- B0 的真实 pre/post residual；
- L0 排除、L1..L8 保留；
- CFG branch 和 sampler-step gating；
- `und_only` forward 不进入 GEN residual；
- norm 与 `M`；
- server 参数配对、非负 chunk、capture 互斥；
- residual capture 自动关闭 compile/CUDA graph。

执行：

```bash
cd /root/robolab/cosmos-framework-edge

/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python -m pytest --capture=no \
  cosmos_framework/scripts/robolab_block_residual_capture_test.py \
  cosmos_framework/scripts/action_policy_server_robolab_test.py
```

本次结果：

```text
13 passed
```

Ruff：

```bash
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python -m ruff check \
  cosmos_framework/scripts/action_policy_server_robolab.py \
  cosmos_framework/scripts/action_policy_server_robolab_test.py \
  cosmos_framework/scripts/robolab_block_residual_capture.py \
  cosmos_framework/scripts/robolab_block_residual_capture_test.py
```

结果为 `All checks passed!`。

## 4. 启动采集

capture 根目录必须不存在或为空。单个 Edge artifact 的 raw residual 约 2.4 GiB；启动前先检查磁盘。

终端 A：

```bash
cd /root/robolab/cosmos-framework-edge

export EXP_ID=edge_block_residual_BananaInBowlTask_c3_shift5_v1
export EXP_ROOT=/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/block_residuals/$EXP_ID
export COSMOS_EDGE_CKPT=/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID
export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home
export HF_HUB_OFFLINE=1
export PYTHONPATH=/root/robolab/cosmos-edge-overlay:/root/robolab/cosmos-framework-edge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -f "$COSMOS_EDGE_CKPT/checkpoint.json"
test ! -e "$EXP_ROOT"
df -h /root/robolab

env -u LD_LIBRARY_PATH \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path "$COSMOS_EDGE_CKPT" \
  --format-prompt-as-json True \
  --no-guardrails \
  --seed 579362556 \
  --deterministic-seed \
  --guidance 3 \
  --num-steps 4 \
  --shift 5 \
  --block-residual-capture-dir "$EXP_ROOT" \
  --block-residual-capture-chunks 3 \
  --block-residual-capture-disk-reserve-gib 5 \
  --host 0.0.0.0 \
  --port 8000
```

终端 B：

```bash
cd /root/robolab/RoboLab
export OMNI_KIT_ACCEPT_EULA=Y

.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --task BananaInBowlTask \
  --num-envs 1 \
  --num-runs 1 \
  --output-folder-name edge_block_residual_BananaInBowlTask_c3_shift5_v1 \
  --video-mode viewport \
  --headless
```

命中 chunk 3 后，server 应输出：

```text
[robolab-block-residual-capture] completed ... raw_shape=[8, 28, 8, 340, 2048]
```

## 5. 离线分析命令

```bash
export ARTIFACT=/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/block_residuals/edge_block_residual_BananaInBowlTask_c3_shift5_v1/task_pick_up_the_banana_and_place_it_in_the_bowl_dc79626d/chunk_000003
export ANALYSIS=/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/offline_analysis/edge_block_residual_BananaInBowlTask_c3_shift5_v1

test -f "$ARTIFACT/gen_block_residual_raw.bin"
test ! -e "$ANALYSIS"

env -u LD_LIBRARY_PATH \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/analyze_edge_block_residuals.py \
  --artifact-dir "$ARTIFACT" \
  --output-dir "$ANALYSIS"
```

## 6. 输出结构

本次 raw artifact：

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/block_residuals/
└── edge_block_residual_BananaInBowlTask_c3_shift5_v1/
    ├── experiment.json
    ├── manifest.jsonl
    └── task_pick_up_the_banana_and_place_it_in_the_bowl_dc79626d/
        └── chunk_000003/
            ├── gen_block_residual_raw.bin
            ├── gen_block_residual_profile.json
            ├── residual_frobenius_norm.npy
            ├── block_output_frobenius_norm.npy
            ├── residual_to_output_ratio.npy
            ├── metadata.json
            ├── conditioning_observation.png
            ├── denoised_vision_latent.pt
            ├── future_vision_latent.pt
            └── predicted_future_frames/
```

本次分析目录：

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/offline_analysis/
└── edge_block_residual_BananaInBowlTask_c3_shift5_v1/
    ├── pairwise_cosine.npy
    ├── pairwise_relative_l2.npy
    ├── residual_to_output_ratio.npy
    ├── pairwise_metrics.csv
    ├── magnitude_metrics.csv
    ├── gap_summary.csv
    ├── step_block_summary.csv
    ├── candidate_boundary_metrics.csv
    ├── summary.json
    ├── matrices/
    │   ├── conditional/step_0/...step_3/
    │   └── unconditional/step_0/...step_3/
    └── figures/
```

数量：

- 448 个单独 matrix CSV：`8 calls × 28 blocks × 2 metrics`；
- `pairwise_metrics.csv`：每个 `8×8` 矩阵的全部元素；
- 26 张图：8 个 call 各 3 张，加 conditional/unconditional 两张 gap 汇总；
- 分析目录共 483 个文件，约 8.2 MiB；
- raw artifact 约 2.4 GiB。

## 7. 本次实测结果

固定输入：

- task：`BananaInBowlTask`
- source chunk：3
- seed：`579362556`
- guidance：3
- denoise steps：4
- shift：5
- timestep：`999, 937, 833, 624`
- CFG path：conditional 和 unconditional

采集结果：

```text
raw shape: [8, 28, 8, 340, 2048]
raw dtype: torch.bfloat16
raw bytes: 2,495,610,880
NaN/Inf: 0
```

矩阵不变量检查：

```text
cosine diagonal max error: 0
relative-L2 diagonal max error: 0
cosine symmetry max error: 0
relative-L2 mean directional asymmetry: 0.05175
```

### 7.1 每个 timestep 独立的总体结构

下面每一行只在同一个 CFG path、同一个 denoise timestep 内跨 28 个 block 汇总，没有跨 timestep 平均：

| path | step / timestep | 非对角 cosine | 相邻 cosine | gap>=4 cosine | 相邻-远距 | 非对角 relative-L2 | 相邻 relative-L2 | 远距 relative-L2 | M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cond | 0 / 999 | 0.6533 | 0.6895 | 0.5972 | 0.0923 | 0.8021 | 0.7556 | 0.8752 | 0.2302 |
| cond | 1 / 937 | 0.5948 | 0.6612 | 0.5270 | 0.1343 | 0.8718 | 0.7915 | 0.9528 | 0.2495 |
| cond | 2 / 833 | 0.5848 | 0.6520 | 0.5164 | 0.1355 | 0.8837 | 0.8031 | 0.9644 | 0.2544 |
| cond | 3 / 624 | 0.5910 | 0.6609 | 0.5224 | 0.1385 | 0.8774 | 0.7948 | 0.9563 | 0.2607 |
| uncond | 0 / 999 | 0.6682 | 0.6972 | 0.6186 | 0.0786 | 0.7843 | 0.7455 | 0.8523 | 0.2214 |
| uncond | 1 / 937 | 0.5955 | 0.6619 | 0.5277 | 0.1342 | 0.8706 | 0.7903 | 0.9517 | 0.2501 |
| uncond | 2 / 833 | 0.5855 | 0.6521 | 0.5176 | 0.1345 | 0.8820 | 0.8019 | 0.9623 | 0.2557 |
| uncond | 3 / 624 | 0.5917 | 0.6624 | 0.5229 | 0.1395 | 0.8752 | 0.7919 | 0.9543 | 0.2634 |

每个 timestep 都表现为相邻 cosine 高于远距 cosine、相邻 relative-L2 低于远距 relative-L2。step 0 的整体共享成分较强，step 1–3 的局部性更明显。两个 CFG path 在相同 timestep 上非常接近。

### 7.2 每个 timestep 的 gap 变化

用 gap 1 和 gap 7 展示首尾趋势；数值只在对应 path/step 内跨 block 汇总：

| path | step / timestep | gap1 cosine | gap1 relative-L2 | gap7 cosine | gap7 relative-L2 |
|---|---|---:|---:|---:|---:|
| cond | 0 / 999 | 0.6895 | 0.7556 | 0.4746 | 1.0217 |
| cond | 1 / 937 | 0.6612 | 0.7915 | 0.4231 | 1.0616 |
| cond | 2 / 833 | 0.6520 | 0.8031 | 0.4204 | 1.0638 |
| cond | 3 / 624 | 0.6609 | 0.7948 | 0.4308 | 1.0516 |
| uncond | 0 / 999 | 0.6972 | 0.7455 | 0.5005 | 1.0013 |
| uncond | 1 / 937 | 0.6619 | 0.7903 | 0.4234 | 1.0609 |
| uncond | 2 / 833 | 0.6521 | 0.8019 | 0.4210 | 1.0626 |
| uncond | 3 / 624 | 0.6624 | 0.7919 | 0.4322 | 1.0482 |

八个 path/step 切片都从 gap 1 到 gap 7 明显降低 cosine、提高 relative-L2，支持局部连续而不是全局一致。gap 2–6 的完整逐 block mean/std 保存在 `gap_summary.csv`。

### 7.3 每个 timestep 的 block 差异

| path | step | cosine 最高 block | cosine 最低 block | M 最低 block | M 最高 block |
|---|---:|---|---|---|---|
| cond | 0 | B1 / 0.974 | B3 / 0.360 | B0 / 0.070 | B24 / 0.389 |
| cond | 1 | B1 / 0.944 | B18 / 0.320 | B0 / 0.116 | B24 / 0.412 |
| cond | 2 | B0 / 0.978 | B18 / 0.290 | B7 / 0.139 | B27 / 0.437 |
| cond | 3 | B0 / 0.973 | B18 / 0.288 | B7 / 0.133 | B27 / 0.607 |
| uncond | 0 | B1 / 0.974 | B3 / 0.362 | B0 / 0.072 | B24 / 0.379 |
| uncond | 1 | B1 / 0.944 | B18 / 0.322 | B0 / 0.126 | B24 / 0.412 |
| uncond | 2 | B0 / 0.983 | B18 / 0.291 | B7 / 0.139 | B27 / 0.438 |
| uncond | 3 | B0 / 0.980 | B18 / 0.289 | B7 / 0.133 | B27 / 0.606 |

不同 timestep 的 block 排名会变化，但结论一致：高 residual cosine 和 residual 相对输出很小不是同一件事。尤其 step 2/3 的 B27 具有较大的 `M`，不能仅凭跨帧相似度判断可跳过。

### 7.4 每个 timestep 的自然分段诊断

同为 gap=1 的受控比较均把 `L1 | L2` 选为最明显 cut，但强度随 timestep 变化：

| path | step / timestep | L1 boundary deficit | 结构标签 |
|---|---|---:|---|
| cond | 0 / 999 | 0.0794 | 局部连续，边界未超过 0.08 阈值 |
| cond | 1 / 937 | 0.1261 | 局部连续 + 可能边界 |
| cond | 2 / 833 | 0.1327 | 局部连续 + 可能边界 |
| cond | 3 / 624 | 0.1157 | 局部连续 + 可能边界 |
| uncond | 0 / 999 | 0.0644 | 局部连续，边界未超过 0.08 阈值 |
| uncond | 1 / 937 | 0.1273 | 局部连续 + 可能边界 |
| uncond | 2 / 833 | 0.1316 | 局部连续 + 可能边界 |
| uncond | 3 / 624 | 0.1170 | 局部连续 + 可能边界 |

因此不能把 `L1 | L2..L8` 说成所有 timestep 都稳定存在的强分段：在 step 1–3 上明显，在 step 0 上只有较弱趋势。这只是一个 task、一个 chunk，当前仍不能据此直接确定 anchor 或 sparse approximation。

## 8. 结论

本次 full pairwise residual 实验的主要现象是：

- 在 4 个 timestep、两个 CFG path 的每一个独立切片中，冗余都以局部时间连续为主，不是全部 future frame 的全局共享。
- step 0 的全局共享成分比 step 1–3 强；不能跨 timestep 混成一个统计量。
- B0/B1 常有很高的跨帧 residual cosine，B18 在 step 1–3 的 frame-specific 成分最强；block 结构依赖 timestep。
- `L1 | L2` 是所有切片的最强候选 cut，但只在 step 1–3 超过当前诊断阈值，step 0 较弱。
- residual cosine 与 residual/output magnitude 必须联合判断；高 cosine 不代表 residual 足够小。

下一阶段如果要决定 anchor，应先在多个 task、chunk 上复现 `L1/L2` 边界及 block 结构，再讨论近似方法。本实验没有执行任何 anchor 插值。

## 9. 聚焦可视化：相邻 residual cosine 与复用误差

新增脚本：

```text
/root/robolab/cosmos-edge-overlay/tools/plot_edge_block_residual_focus.py
```

运行：

```bash
export ARTIFACT=/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/block_residuals/edge_block_residual_BananaInBowlTask_c3_shift5_v1/task_pick_up_the_banana_and_place_it_in_the_bowl_dc79626d/chunk_000003
export ANALYSIS=/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/offline_analysis/edge_block_residual_BananaInBowlTask_c3_shift5_v1

env -u LD_LIBRARY_PATH \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/robolab/cosmos-edge-overlay/tools/plot_edge_block_residual_focus.py \
  --artifact-dir "$ARTIFACT" \
  --analysis-dir "$ANALYSIS" \
  --representative-blocks 0 1 18 27
```

输出：

```text
$ANALYSIS/focused_figures/
├── figure1_adjacent_residual_cosine_heatmap.png
├── figure2_adjacent_residual_reuse_output_error_heatmap.png
├── adjacent_block_metrics.csv
├── adjacent_residual_cosine.npy
├── adjacent_residual_reuse_output_error.npy
├── focused_summary.json
├── representative_block_B00_pairwise.png
├── representative_block_B01_pairwise.png
├── representative_block_B18_pairwise.png
├── representative_block_B27_pairwise.png
└── representative_block_B*.npz
```

### 9.1 图 1

\[
C_{s,l}=\frac{1}{7}\sum_{f=1}^{7}
\cos(R_f^{s,l},R_{f+1}^{s,l})
\]

- 横轴：B0–B27。
- 纵轴：`0/999, 1/937, 2/833, 3/624`。
- conditional、unconditional 主面板固定色轴 `[0,1]`。
- 差值面板为 `conditional-unconditional`，因为存在负数，使用以 0 为中心的对称色轴。

文件：

```text
focused_figures/figure1_adjacent_residual_cosine_heatmap.png
```

### 9.2 图 2

\[
D_{s,l}=\frac{1}{7}\sum_{f=1}^{7}
\frac{\|R_{f+1}^{s,l}-R_f^{s,l}\|_F}
{\|H_{f+1,\mathrm{out}}^{s,l}\|_F}
\]

主面板使用同一个色轴，低误差为深色。差值面板仍为 `conditional-unconditional`。

文件：

```text
focused_figures/figure2_adjacent_residual_reuse_output_error_heatmap.png
```

### 9.3 聚焦定位结果

| block | 相邻 cosine 范围/均值 | 输出替换误差最大值/均值 | 解释 |
|---:|---:|---:|---|
| B0 | `min=0.786, mean=0.928` | `max=0.078, mean=0.048` | 输出误差最低，但 step 0 cosine 没有达到严格阈值 |
| B1 | `min=0.963, mean=0.976` | `max=0.103, mean=0.081` | 唯一跨全部 path/step 同时满足高 cosine、低误差的严格候选 |
| B18 | `min=0.335, mean=0.380` | `max=0.252, mean=0.233` | 低相似代表，不能复用相邻 residual |
| B23 | `min=0.584, mean=0.622` | `max=0.384, mean=0.357` | 全部 block 中输出替换误差最高 |
| B27 | `min=0.871, mean=0.910` | `max=0.304, mean=0.189` | cosine 较高但后期误差显著增大，是“高相似不等于安全”的代表 |

严格候选规则仅用于定位：

```text
所有 branch/step 的 cosine 最小值 >= 0.9
且所有 branch/step 的 D 最大值 <= 0.15
```

按此规则当前只有 B1。B0 是低输出误差候选，但其 step 0 cosine 约 `0.79`；B27 的相邻 cosine 一直较高，但 `D` 从 step 0 的约 `0.11` 增长到 step 3 的约 `0.30`，不适合作为“仅凭 cosine”选出的安全 block。

### 9.4 代表性 8×8 矩阵

只显示 B0、B1、B18、B27。每个 block 图按四个 timestep 分行，并列显示：

- conditional residual cosine；
- unconditional residual cosine；
- conditional residual-reuse output error；
- unconditional residual-reuse output error。

复用误差矩阵的行是 target future latent，列是被复用的 source residual：

\[
E_{\mathrm{reuse}}[\mathrm{target},\mathrm{source}]
=
\frac{\|R_{\mathrm{target}}-R_{\mathrm{source}}\|_F}
{\|H_{\mathrm{target,out}}\|_F}
\]

全量矩阵数值仍保留在 NPY/CSV 中，但默认观察只需要上述四个代表 block。
