# V5.2 Motion-Core / Stable-Adaptive K80 实验

## 目标

V5.2 保留 V5.1 的 Step-0 velocity cache 和真实短序列计算路径，只改变未来视觉 token mask 的生成方式。实验验证：跨 block Motion Core、跨未来帧 Stable Environment 和带阈值的 Adaptive replacement 是否能在相同 K80 预算下改善输出稳定性。

## 配对实验

| Arm | Mask 构造 |
|---|---|
| Dense | 340 个 future spatial token 全量计算 |
| A | V5.1 score，改为从 G3 向 G1 严格嵌套构造 |
| B | A + 逐 future frame Motion Core |
| C | B + Stable/Adaptive Environment |
| D | C + relative replacement threshold 和每帧替换上限 |

固定预算：

\[
K_{G1}=192,\qquad K_{G2}=160,\qquad K_{G3}=144
\]

所有 arm 必须满足：

\[
M_{G3,f}\subseteq M_{G2,f}\subseteq M_{G1,f}
\]

## Motion Core

真实 attention softmax probability 对 head 和与 future frame 对齐的四个 action queries 求平均：

\[
U^{(r,l)}_{f,p}
=
\frac{1}{H|\mathcal Q_f|}
\sum_h\sum_{a\in\mathcal Q_f}
A^{(r,l,h)}_{a,(f,p)}
\]

其中 \(\mathcal Q_f=\{4(f-1)+1,\ldots,4f\}\)。Future-video mass：

\[
R_l=\frac1{2\cdot8}\sum_r\sum_f\sum_pU^{(r,l)}_{f,p}
\]

仅为计算空间熵做归一化：

\[
\bar U^{(r,l)}_{f,p}
=\frac{U^{(r,l)}_{f,p}}{\sum_{p'}U^{(r,l)}_{f,p'}+\epsilon}
\]

\[
H_l=\frac1{2\cdot8}\sum_r\sum_f
\frac{-\sum_p\bar U^{(r,l)}_{f,p}\log(\bar U^{(r,l)}_{f,p}+\epsilon)}{\log340}
\]

Block quality 和权重：

\[
Q_l=R_l(1-H_l),\qquad
w_l=\frac{Q_l}{\sum_{j\in\mathcal B_{core}}Q_j+\epsilon}
\]

\[
G_{f,p}=\sum_{l\in\mathcal B_{core}}w_l\max_rU^{(r,l)}_{f,p},\qquad
C_f=\operatorname{TopK}(G_{f,:},K_{core})
\]

实际 Core score 始终使用 raw probability，空间归一化只参与熵。

## Stable/Adaptive Environment

三个 group 的 raw score 为：

\[
V_{g,f,p}=\sum_{l\in g}\omega_{g,l}\max_rU^{(r,l)}_{f,p}
\]

排除八帧 Core 并集后，计算：

\[
\mu_{g,p}=\frac18\sum_fV_{g,f,p},\qquad
CV_{g,p}=\frac{\sigma_{g,p}}{\mu_{g,p}+\epsilon}
\]

\[
T^{stable}_{g,p}=\frac{\mu_{g,p}}{1+\lambda_{cv}CV_{g,p}}
\]

Stable mask 在 L1-L8 使用相同空间位置，并从 G3 向 G1 扩展。Adaptive score：

\[
T^{adaptive}_{g,f,p}=\max(V_{g,f,p}-\mu_{g,p},0)
\]

只从严格正分候选中选择；不足预算的部分按照原始 \(V_{g,f,p}\) 补位，并单独标记为 `budget_fill`。

## D：Replacement

Motion Core、Stable、subset repair 和 budget fill 引起的变化记录为 forced replacement。普通 Adaptive token 只有满足：

\[
V_{new}>(1+\tau_{rel})V_{old}
\]

才允许替换，并限制每帧每 group 最多 \(R_{max}\) 次。

## 第一轮配置

| 参数 | 值 |
|---|---:|
| Shift | 5 |
| Denoise steps | 4 |
| Core block candidates | B4-B23 |
| Core block count | 6 |
| Core tokens | 48 |
| Stable G1/G2/G3 | 88/72/64 |
| CV penalty | 1.0 |
| Relative replacement threshold | 0.05 |
| Max replacements | 8 |

## 运行命令

```bash
cd /root/robolab/worktrees/v5-2-motion-core-stable-adaptive-k80

HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 \
LD_LIBRARY_PATH='' \
PYTHONPATH=$PWD \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m tools.run_robolab_v5_2_motion_core_stable_adaptive \
  --output-root /root/robolab/experiments/preliminary/sparsity/velocity_cache/\
v5_2_motion_core_stable_adaptive_k80_BananaInBowlTask_c3_v1
```

CPU 测试：

```bash
LD_LIBRARY_PATH='' /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m pytest -q \
  tests/test_robolab_v5_2_motion_core_stable_adaptive.py \
  tests/test_robolab_grouped_temporal_closed_roi_velocity_cache.py
```

## BananaInBowlTask chunk 3 第一轮结果

相同 seed `579362556`，A-D Step-0 raw profile 最大绝对差为 `0.0`。自动选择的 Core blocks 为：

```text
B15, B18, B19, B20, B21, B22
```

所有 arm 的 G3→G2 和 G2→G1 subset violation 都为 `0`。

| Arm | Action MSE | Action cos | Delta cos | Jerk cos | RGB cos | RGB rel-L2 | Latent cos | Latent rel-L2 | Wall(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.003241 | 0.999206 | -0.050834 | -0.184157 | 0.965570 | 0.280082 | 0.939108 | 0.353260 | 0.7976 |
| B | 0.004087 | 0.999040 | -0.028477 | -0.120695 | 0.965248 | 0.281187 | 0.938264 | 0.355270 | 0.7608 |
| C | 0.007215 | 0.998147 | -0.022211 | -0.124646 | 0.957700 | 0.306127 | 0.935026 | 0.363464 | 0.7689 |
| D | 0.007268 | 0.998199 | -0.094498 | -0.192725 | 0.959741 | 0.299359 | 0.936279 | 0.359923 | 0.7681 |

Raw attention-mass retention mean（G1/G2/G3）：

| Arm | G1 | G2 | G3 |
|---|---:|---:|---:|
| A | 0.7146 | 0.7791 | 0.6988 |
| B | 0.7158 | 0.7869 | 0.7001 |
| C | 0.6866 | 0.7608 | 0.6759 |
| D | 0.7188 | 0.7902 | 0.6930 |

相邻 future-frame mask Jaccard（G1/G2/G3）：

| Arm | G1 | G2 | G3 |
|---|---:|---:|---:|
| A | 0.8214 | 0.8040 | 0.7975 |
| B | 0.8262 | 0.7853 | 0.7581 |
| C | 0.5482 | 0.4804 | 0.4749 |
| D | 0.7473 | 0.6982 | 0.6866 |

D 总计记录 `458` 次 forced replacement；569 个 threshold proposal 中接受 125 个、拒绝 444 个。

## 五任务闭环筛选

闭环固定：

- policy seed：`579362556`，每个 request 都重置为相同 seed；
- RoboLab environment seed：`0`；
- prompt：官方 JSON 格式，所有 arm 都显式传入 `--format-prompt-as-json True`；
- shift：`5`，UniPC steps：`4`；
- compile/CUDA graph：关闭；
- 每个 arm、每个 task：一个 episode；
- viewport 视频：开启。

任务集合为：

```text
BananaInBowlTask
BananaOnPlateTask
RubiksCubeTask
RubiksCubeAndBananaTask
RubiksCubeLeftOfBowlTask
```

成功严格读取 `episode_results.jsonl` 的 `success` 布尔字段，不用 `score` 代替。

| Arm | 成功数 | 成功率 | Warm chunk median (s) | P90 (s) | 相对 Dense | 全 block-call GEN token 保留率 |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 3/5 | 60.0% | 0.8695 | 0.8778 | 1.000x | 100.00% |
| A | 3/5 | 60.0% | 0.7308 | 0.7438 | 1.190x | 70.96% |
| B | 2/5 | 40.0% | 0.7283 | 0.7402 | 1.194x | 70.96% |
| C | 4/5 | 80.0% | 0.7266 | 0.7350 | 1.197x | 70.96% |
| D | 3/5 | 60.0% | 0.7282 | 0.7372 | 1.194x | 70.96% |

逐任务结果：

| Task | Dense | A | B | C | D |
|---|:---:|:---:|:---:|:---:|:---:|
| BananaInBowlTask | Y | Y | Y | Y | Y |
| BananaOnPlateTask | Y | Y | Y | Y | N |
| RubiksCubeTask | N | N | N | Y | Y |
| RubiksCubeAndBananaTask | Y | Y | N | Y | Y |
| RubiksCubeLeftOfBowlTask | N | N | N | N | N |

稀疏臂在全 28 个 block call 上平均减少 `898.29 / 3093` 个 GEN token，即保留 `70.96%`。三个稀疏 group 的实际 GEN token 保留率分别为 `61.72%`、`53.44%`、`49.30%`。所有请求均通过 finite 检查，且 G3→G2、G2→G1 subset violation 为 `0`。

服务命令（将 `MODE` 设为 `dense`、`a`、`b`、`c` 或 `d`）：

```bash
cd /root/robolab/worktrees/v5-2-motion-core-stable-adaptive-k80
MODE=b
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 \
LD_LIBRARY_PATH='' \
PYTHONPATH=$PWD \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab_v5_2_motion_core_stable_adaptive \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --host 127.0.0.1 \
  --port 8000 \
  --seed 579362556 \
  --deterministic-seed \
  --format-prompt-as-json True \
  --ablation-mode "$MODE" \
  --intervention-output-dir \
    "/root/robolab/experiments/preliminary/sparsity/velocity_cache/\
v5_2_motion_core_stable_adaptive_k80_5tasks_seed579362556_json_v1/$MODE/server"
```

另一个终端运行 RoboLab：

```bash
cd /root/robolab/RoboLab
MODE=b
OMNI_KIT_ACCEPT_EULA=Y NO_PROXY=127.0.0.1,localhost \
.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --task BananaInBowlTask BananaOnPlateTask RubiksCubeTask \
    RubiksCubeAndBananaTask RubiksCubeLeftOfBowlTask \
  --num-envs 1 \
  --num-runs 1 \
  --output-folder-name \
    "v5_2_k80_${MODE}_shift5_5tasks_seed579362556_json_v1" \
  --video-mode viewport \
  --headless
```

统一汇总：

```bash
cd /root/robolab/worktrees/v5-2-motion-core-stable-adaptive-k80
LD_LIBRARY_PATH='' /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m tools.summarize_robolab_v5_2_closed_loop \
  --experiment-root /root/robolab/experiments/preliminary/sparsity/velocity_cache/\
v5_2_motion_core_stable_adaptive_k80_5tasks_seed579362556_json_v1 \
  --robolab-output-root /root/robolab/RoboLab/output \
  --seed 579362556 \
  --dense-run-name v5_2_k80_dense_shift5_5tasks_seed579362556_v1 \
  --sparse-run-suffix json_v1
```

闭环 CSV、中文报告和每个 request 的 token 数据位于：

```text
/root/robolab/experiments/preliminary/sparsity/velocity_cache/
v5_2_motion_core_stable_adaptive_k80_5tasks_seed579362556_json_v1/
```

每个 arm 的五个 viewport MP4 位于：

```text
/root/robolab/RoboLab/output/
v5_2_k80_<arm>_shift5_5tasks_seed579362556_json_v1/<task>/*_viewport.mp4
```

Dense 视频沿用已经完成、条件完全相同的：

```text
/root/robolab/RoboLab/output/
v5_2_k80_dense_shift5_5tasks_seed579362556_v1/<task>/*_viewport.mp4
```

配对审计时曾产生一批没有显式开启 JSON prompt 的 A-D 结果，位于不带 `_json_` 的旧目录；它与 Dense 的输入格式不一致，已明确排除，不进入上述表格和最终结论。

## 当前结论边界

1. Motion Core 轻微提高了部分 group 的 attention-mass retention，但没有在该 chunk 上降低 action/RGB 误差。
2. 当前 Stable/Adaptive 配额使 C 的整体 temporal Jaccard 明显下降，而不是上升；说明 32 个逐帧 Adaptive token 足以引入较大变化。
3. D 明显恢复了 temporal Jaccard，并略微改善 C 的 RGB/latent 指标，但 action MSE 没有改善。
4. 有效 JSON-prompt 五任务闭环中 Dense/A/B/C/D 为 `3/5`、`3/5`、`2/5`、`4/5`、`3/5`。A 与 Dense 的逐任务成败完全一致；C 在这个 seed 上最高，继续减 token 时应同时保留 A 和 C 两条基线。
5. Warm chunk median 显示约 `1.20x` 端到端 generation 加速；这是本次同机 eager 闭环请求测量，不外推为其他硬件或 compile/CUDA graph 配置的稳定加速。
6. 每个 task 只有一个固定 seed episode，因此 C 的 `4/5` 只比 Dense 多一个 episode，不能解释为成功率显著提升；这些数值是 pooled screening success，不是稳定的逐任务成功率估计。

完整结果位于：

```text
/root/robolab/experiments/preliminary/sparsity/velocity_cache/
v5_2_motion_core_stable_adaptive_k80_BananaInBowlTask_c3_v1/
```
