# Grouped Temporal-Closed ROI Velocity Cache V5

## 1. 实验定位

V5 从 Velocity Cache V1 的固定提交 `92ae3f8` 分支实现。目标是减小 V1 将 CFG 分支、B4--B27 和 L1--L8 全部压缩为一张 ROI 的聚合范围，同时避免逐帧独立 mask 造成相邻 future latent 的计算/缓存来源突然切换。

本实验固定：

- `num_steps=4`；
- `guidance=3`；
- `shift=5`；
- Step 0 全量；
- B0--B3 在所有 step 全量；
- eager inference，关闭 `torch.compile` 和 CUDA graphs；
- ROI threshold 为 0.9。

## 2. V5 方法

将稀疏层划分为：

\[
G_1=B4\ldots B11,\qquad G_2=B12\ldots B19,\qquad G_3=B20\ldots B27.
\]

Step 0 的 conditional/unconditional forward 全量运行。在每个 B4--B27 中，根据真实 post-RoPE Action-Q/Future-K attention，为每个 future latent 计算一个 `[340]` 空间分布。每 4 个 action horizons 对齐一个 future latent。

只在 CFG branch 和同一 block group 内做 max 聚合，不跨 future latent：

\[
S_{g,f}(p)=\max_{branch,\,l\in G_g}P_{branch,l,f}(p).
\]

对每个 `(group, frame)` 取累计 90% 分数的最小 mask：

\[
R_{g,f}=\operatorname{TopMass}_{0.9}(S_{g,f}).
\]

因此产生 `3 x 8 = 24` 张 raw masks。

### 2.1 相邻时间闭包

为避免同一空间位置在相邻 latent 间突然切换当前计算和 Step-0 cache：

\[
C_{g,f}=R_{g,f-1}\cup R_{g,f}\cup R_{g,f+1}.
\]

边界 frame 只合并存在的邻居。

### 2.2 深度嵌套

为禁止缺少中间 block hidden 的 token 在 B12/B20 重新进入：

\[
E_{1,f}=C_{1,f}\cup C_{2,f}\cup C_{3,f},
\]

\[
E_{2,f}=C_{2,f}\cup C_{3,f},\qquad E_{3,f}=C_{3,f}.
\]

于是对每个 frame 均有：

\[
E_{1,f}\supseteq E_{2,f}\supseteq E_{3,f}.
\]

B4、B12、B20 只会继续裁剪当前 active tokens。实现会显式检查目标集合是当前集合的子集，检测到 token re-entry 立即报错。

### 2.3 稀疏路径

Step 1--3：

1. B0--B3 全量；
2. B4--B11 对 L1--L8 分别使用 `E1,f`；
3. B12--B19 继续裁剪到 `E2,f`；
4. B20--B27 继续裁剪到 `E3,f`；
5. UND/text、L0、q0 和 q1--q32 始终保留；
6. hidden、RoPE 和 token metadata 使用相同原始索引；
7. B27 后 scatter 回完整 grid，再进入 RMSNorm 和 modality heads。

### 2.4 Guided velocity cache

Step 0 保存 CFG 后的完整 vision velocity。Step 1--3 对每个 future latent 独立合并：

\[
\widetilde V_f^{(s)}(p)=
\begin{cases}
V_f^{(s)}(p), & p\in E_{3,f},\\
V_f^{(0)}(p), & p\notin E_{3,f}.
\end{cases}
\]

L0 和 action velocity 不使用缓存。合并完成后才进入 UniPC。

## 3. CPU 验证

运行：

```bash
cd /root/robolab/worktrees/grouped-temporal-closed-roi-velocity-cache-shift5-v5
LD_LIBRARY_PATH='' PYTHONPATH=$PWD \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python -m pytest --capture=no \
  tests/test_robolab_grouped_temporal_closed_roi_velocity_cache.py \
  tests/test_robolab_step0_fixed_roi_velocity_cache.py
```

结果：`14 passed`。覆盖：

- future frame profile 不混合；
- radius-1 时间闭包；
- depth nesting；
- token re-entry 拒绝；
- L0/action 保留；
- frame-local mask 索引；
- token-grid 到 velocity-grid 扩展；
- frame-local Step-0 velocity merge；
- V1 回归测试。

## 4. Paired chunk 结果

任务为 `BananaInBowlTask / chunk 3`，seed `579362556`。Baseline 与先前 V1 paired artifact 的 baseline action/vision 均为 bit-exact。

| Strategy | Action MSE | Action cos | Action rel-L2 | Delta cos | Jerk cos | RGB cos | RGB rel-L2 | Latent cos | Latent rel-L2 | Generator speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0 | 1.000000 | 0 | 1.000000 | 1.000000 | 1.000000 | 0 | 1.000000 | 0 | 1.00x |
| V1 | 0.004775 | 0.998763 | 0.049733 | -0.033250 | -0.165619 | 0.967082 | 0.275822 | 0.944305 | 0.335609 | 1.47x |
| V5 | 0.000471 | 0.999880 | 0.015616 | 0.511664 | 0.459902 | 0.991365 | 0.169292 | 0.978886 | 0.205590 | 1.19x |

时间为单次 paired eager generation，不是交替多轮性能基准：

- Baseline：`1.1111 s`；
- V5：`0.9335 s`；
- 单次 speedup：`1.190x`。

V5 的 action、action delta、jerk、RGB 和 latent 都比 V1 更接近 Baseline，但速度收益下降。

## 5. Token 与时间一致性

### 5.1 Raw masks

| Block group | Mean tokens/frame | Adjacent Jaccard | Adjacent flip rate |
|---|---:|---:|---:|
| B4--B11 | 258.8 | 0.781 | 0.187 |
| B12--B19 | 170.9 | 0.528 | 0.308 |
| B20--B27 | 223.8 | 0.628 | 0.300 |

### 5.2 时间闭包后

| Block group | Mean tokens/frame | Adjacent Jaccard | Adjacent flip rate |
|---|---:|---:|---:|
| B4--B11 | 302.1 | 0.947 | 0.049 |
| B12--B19 | 242.9 | 0.892 | 0.083 |
| B20--B27 | 291.4 | 0.928 | 0.065 |

### 5.3 最终嵌套执行 mask

| Blocks | Mean tokens/frame | GEN tokens | GEN retained | Saved GEN/block | Adjacent flip rate |
|---|---:|---:|---:|---:|---:|
| B4--B11 | 329.1 | 3006 | 97.19% | 87 | 0.0185 |
| B12--B19 | 310.1 | 2854 | 92.27% | 239 | 0.0462 |
| B20--B27 | 291.4 | 2704 | 87.42% | 389 | 0.0647 |

时间闭包将相邻 frame mask flip 显著压低，但 depth nesting 使前两组接近全量。这解释了 V5 精度改善和 speedup 从 V1 的约 `1.47x` 降至约 `1.19x`。

## 6. RoboLab 单任务验证

V5 闭环运行 `BananaInBowlTask`：

- `success=true`；
- episode step：162；
- 任务时长：10.8 s；
- policy requests：6；
- policy inference：8.421 s；
- wall total：37.597 s；
- 输出 viewport 视频。

这只是 `1/1` 功能验证，不能视为成功率结论。

预测 future contact sheet 中仍能观察到背景块状颜色噪声。由于这一现象发生在最终 background velocity 复用后，当前证据更支持它与 Step-0 background velocity cache 有关；不能仅凭这次实验将其归因于逐帧 mask。

## 7. 输出路径

Paired chunk：

```text
/root/robolab/experiments/preliminary/sparsity/velocity_cache/
grouped_temporal_closed_roi_velocity_cache_shift5_BananaInBowlTask_c3_v1/
```

关键文件：

- `metrics.json`；
- `paired_outputs.pt`；
- `controller/step0_grouped_temporal_closed_roi.pt`；
- `controller/block_token_savings.csv`；
- `roi_visualizations/raw_masks_g*.png`；
- `roi_visualizations/closed_masks_g*.png`；
- `roi_visualizations/execution_masks_g*.png`；
- `roi_visualizations/execution_overlay_g*.png`；
- `baseline/contact_sheet.png`；
- `sparse/contact_sheet.png`。

闭环输出：

```text
/root/robolab/RoboLab/output/
grouped_temporal_closed_roi_velocity_cache_shift5_BananaInBowlTask_sim_v1/
```

视频：

```text
BananaInBowlTask/Pick_up_the_banana_and_place_it_in_the_bowl_0_viewport.mp4
```

## 8. 复现命令

Paired chunk：

```bash
cd /root/robolab/worktrees/grouped-temporal-closed-roi-velocity-cache-shift5-v5
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 LD_LIBRARY_PATH='' PYTHONPATH=$PWD \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  tools/run_robolab_step0_fixed_roi_velocity_cache.py \
  --strategy-version v5 \
  --output-root /root/robolab/experiments/preliminary/sparsity/velocity_cache/<new-run-id>
```

V5 server：

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
  --roi-mass-threshold 0.9 \
  --intervention-output-dir /root/robolab/experiments/preliminary/sparsity/velocity_cache/<server-run-id>
```

RoboLab：

```bash
cd /root/robolab/RoboLab
OMNI_KIT_ACCEPT_EULA=Y NO_PROXY=127.0.0.1,localhost \
.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --task BananaInBowlTask --num-envs 1 --num-runs 1 \
  --output-folder-name <new-run-id> \
  --video-mode viewport --headless
```

## 9. 当前结论

V5 已验证能够真实缩短 B4--B27 的 GEN sequence，并保持原始 token/position 对应关系。相邻时间闭包显著降低 mask flicker，单 chunk 精度和单任务动作执行均优于 V1 的风险表现；代价是嵌套执行 mask 很宽，实际 token savings 和速度收益有限。下一轮若继续优化，应优先减少 depth nesting 的膨胀，而不是直接移除时间闭包。
