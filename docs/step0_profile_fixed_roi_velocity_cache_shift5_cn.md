# Step 0 全量 Profile、固定 ROI 稀疏与背景 Velocity 缓存实验（shift=5）

## 1. 结论摘要

该策略已完成单 chunk 配对验证和一次完整 RoboLab 闭环：

- `BananaInBowlTask` 单次仿真成功，`1/1`；
- 每个 policy chunk 的 step 0、B0–B27 全量计算；
- step 1–3 的 B0–B3 全量，B4–B27 真正使用缩短后的 token 序列；
- future L1–L8、conditional/unconditional 和 B4–B27 使用同一空间 ROI；
- 后续 step 在 UniPC 前使用当前 ROI velocity，并用 step 0 guided velocity 补背景；
- 单 chunk 固定 ROI 为 `195/340`，B4–B27 的 GEN token 从 `3093` 降至 `1933`；
- 单次 paired eager generation 为 `1.116s → 0.757s`，即 `1.47×`，但这只是一次测量；
- Action 整体 cosine 为 `0.9988`，但 action delta/jerk cosine 为 `-0.033/-0.166`，说明局部动作动态仍然敏感；
- 最终 latent/RGB cosine 为 `0.9443/0.9671`，背景预测存在可见变化。

因此，本实验说明该方案在一个任务种子上可以闭环成功并产生真实 token 缩减，但不足以证明稳定成功率或普遍加速。

## 2. 真实执行流程

### 2.1 每个 policy chunk 的 step 0

四步 UniPC 的第一个 denoise step（timestep 999）执行完整模型：

```text
conditional:   B0 ───────────── B27，完整 3093 GEN tokens
unconditional: B0 ───────────── B27，完整 3093 GEN tokens
```

在两个 CFG branch 的 B4–B27 保存真实 post-RoPE Action-Q/Future-K attention profile，共：

\[
2\ \text{branches}\times24\ \text{blocks}=48\ \text{profiles}.
\]

对每个 profile，使用与 latent frame 对齐的四个 action horizons：

\[
L_f\leftrightarrow q_{4(f-1)},\ldots,q_{4f-1}.
\]

得到每个 future frame 的帧内空间分布 \(P_{b,l,f,p}\)，然后采用：

\[
S_p=\max_{b,l,f}P_{b,l,f,p}.
\]

选择覆盖 \(S\) 总质量 90% 的最小空间集合，形成一个固定的 `17×20` ROI。使用 max 聚合是为了保留任何 branch/block/frame 中出现的强热点，同时避免 mean/p90 聚合退化成近全图 mask。

step 0 的 conditional/unconditional velocity 按源码执行 CFG，得到完整 guided velocity：

\[
v_0^{guided}=v_0^{uncond}+g(v_0^{cond}-v_0^{uncond}).
\]

缓存其中 ROI 外的 future background velocity，再使用完整 \(v_0^{guided}\) 做第一次 UniPC 更新。

### 2.2 step 1–3

每个 CFG branch 的 B0–B3 仍接收完整序列。B4 开始只保留：

- 全部 text/K_AR；
- condition latent L0 的 340 个 token；
- future L1–L8 中完全相同的 ROI 空间位置；
- q0 和 32 个预测 action token。

本次 paired chunk 的 ROI 为 195 个 token，因此：

\[
N_{GEN}^{sparse}=340+8\times195+33=1933,
\]

而 baseline 为：

\[
N_{GEN}^{full}=9\times340+33=3093.
\]

B4 裁剪 hidden state 与 RoPE 后，B5–B27 连续接收同一个稀疏序列。没有逐层更换 mask，没有把 token 删除后又重新注入，也不是只在计算结束后做 attention mask。

最终输出头之前恢复完整 GEN 顺序：ROI/action 使用 B27 的真实输出；future background 的 hidden 只作为形状占位，其预测 velocity 随后会被缓存值覆盖。

### 2.3 CFG 后、UniPC 前的 velocity 补全

模型输出的 visual velocity grid 是 `[1,48,9,33,40]`，Transformer ROI 是 `17×20` patch grid。按 `patch_spatial=2` 将 ROI 最近邻扩展到 `34×40`，再裁剪成 `33×40`，保证一个 Transformer token 控制其对应 latent 区域。

后续 step 使用：

\[
\tilde v_t=M\odot v_t^{guided,sparse}+(1-M)\odot v_0^{guided},
\qquad t=1,2,3.
\]

完整形状的 \(\tilde v_t\) 才交给 UniPC。Action velocity 始终使用当前 step 的预测，从不读取 step 0 缓存。

## 3. 单 chunk 配对结果

固定输入：`BananaInBowlTask`、标记 chunk 3、state index 96、seed 579362556、guidance 3、4 steps、shift 5。Baseline 与 Sparse 使用相同 data batch、初始 RNG 和 eager 配置，compile/CUDA graphs 均关闭。

| 指标 | 结果 |
|---|---:|
| Action MSE | 0.004775 |
| Action relative-L2 | 0.04973 |
| Action cosine | 0.99876 |
| Action max absolute error | 0.23328 |
| Action delta cosine | -0.03325 |
| Action delta relative-L2 | 1.50936 |
| Action jerk cosine | -0.16562 |
| Action jerk relative-L2 | 1.58758 |
| Final vision latent cosine | 0.94430 |
| Final vision latent relative-L2 | 0.33561 |
| Decoded RGB cosine | 0.96708 |
| Decoded RGB relative-L2 | 0.27582 |

Action 的绝对值仍高度相似，但相邻 action 差分和二阶 jerk 已不相似。单看整体 action cosine 会明显低估控制风险。

### 3.1 背景 velocity 随 step 的失配

下表的 cosine/relative-L2 比较当前 sparse 模型的背景 velocity 与缓存的 step 0 背景 velocity：

| Step / timestep | cosine | relative-L2 | merged velocity vs dense baseline cosine |
|---|---:|---:|---:|
| 0 / 999 | 1.000 | 0.000 | 1.0000 |
| 1 / 937 | 0.7349 | 0.6805 | 0.9828 |
| 2 / 833 | 0.6833 | 0.7530 | 0.9749 |
| 3 / 624 | 0.5193 | 1.0155 | 0.9628 |

step 0 velocity 是明显陈旧的时间相关向量场。它在最后一步与当前背景预测的 relative-L2 已超过 1。完整 latent/RGB 仍保持较高 cosine，主要不能被解释为背景 velocity 本身稳定。

### 3.2 Token 与时间

| 项目 | Baseline | Sparse |
|---|---:|---:|
| Step 0 block calls | 全量 | 全量 |
| Step 1–3 B0–B3 | 全量 | 全量 |
| Step 1–3 B4–B27 GEN tokens | 3093 | 1933 |
| 每个 sparse block 节省 | 0 | 1160 |
| Sparse block calls / total | 0/224 | 144/224 |
| 平均到全部 block call 的 token 节省 | 0 | 745.7/3093，约 24.1% |
| Paired generation wall time | 1.116s | 0.757s |
| 单次 wall speedup | 1.00× | 1.47× |

时间没有预热和交替重复，不能作为稳定 benchmark；它只证明实际短序列没有被 Python/hook 开销完全抵消。

## 4. 完整 RoboLab 任务

运行：

- Task：`BananaInBowlTask`
- Runs：1
- Environments：1
- shift：5
- video mode：viewport
- policy chunks：5

结果：

| 指标 | 数值 |
|---|---:|
| Success | 1/1 |
| Score | 1.0 |
| Episode steps | 146 |
| Simulation duration | 9.73s |
| Wall total | 37.42s |
| Policy inference total | 11.18s |
| Policy inference avg / sim step | 76.6ms |
| EE path length | 1.014m |
| EE SPARC | -2.513 |
| Target dropped events | 1 |

5 个 policy chunk 的统计：

- ROI mean：190.6/340，range 186–196；
- 每个 sparse block 平均节省：1195.2 GEN tokens；
- 平均到全部 block calls 的 token 节省比例：约 24.84%；
- request generation wall time mean：0.819s。

该成功结果只是一条 episode，不能声明任务成功率保持不变。历史 baseline task 的服务器配置和 compile 状态不同，因此没有用它的 wall time 计算本策略端到端 speedup。

## 5. 输出目录

正式 paired 结果：

`/root/robolab/experiments/preliminary/sparsity/velocity_cache/step0_profile_fixed_roi_velocity_cache_shift5_BananaInBowlTask_c3_v3/`

关键文件：

- `metrics.json`：Action/latent/RGB/time/token 汇总；
- `paired_outputs.pt`：Baseline/Sparse action 与 vision latent；
- `action_horizon_metrics.csv`：逐 horizon Action 误差；
- `controller/step0_roi_profile.pt`：48 个 profile、aggregate score、固定 ROI；
- `controller/block_token_savings.csv`：全部 step/branch/block 的真实 token 数；
- `controller/velocity_cache_trace.json`：逐 step velocity cache 误差；
- `baseline/contact_sheet.png`、`sparse/contact_sheet.png`：解码未来帧对比；
- `baseline/future_frames/`、`sparse/future_frames/`：逐帧 PNG。

完整仿真：

`/root/robolab/RoboLab/output/step0_profile_fixed_roi_velocity_cache_shift5_BananaInBowlTask_sim_v1/`

Viewport 视频：

`BananaInBowlTask/Pick_up_the_banana_and_place_it_in_the_bowl_0_viewport.mp4`

服务器逐 chunk 结果：

`/root/robolab/experiments/preliminary/sparsity/velocity_cache/step0_profile_fixed_roi_velocity_cache_shift5_BananaInBowlTask_sim_v1/server/`

其中 `v1`、`v2` paired 目录是开发期间暴露 token-grid/velocity-grid 布局差异的失败运行；正式结果为 `v3`。

## 6. 复现命令

### 6.1 Paired chunk

```bash
cd /root/robolab/worktrees/step0-profile-fixed-roi-velocity-cache-shift5
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 \
PYTHONPATH=. \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  tools/run_robolab_step0_fixed_roi_velocity_cache.py \
  --output-root /root/robolab/experiments/preliminary/sparsity/velocity_cache/step0_profile_fixed_roi_velocity_cache_shift5_BananaInBowlTask_c3_v4
```

### 6.2 实验服务器

```bash
cd /root/robolab/worktrees/step0-profile-fixed-roi-velocity-cache-shift5
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
HF_HUB_OFFLINE=1 \
PYTHONPATH=. \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab_step0_fixed_roi_velocity_cache \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --format-prompt-as-json True \
  --no-guardrails \
  --guidance 3.0 \
  --num-steps 4 \
  --shift 5.0 \
  --deterministic-seed \
  --seed 579362556 \
  --host 0.0.0.0 \
  --port 8000 \
  --fixed-roi-mass-threshold 0.9 \
  --first-sparse-block 4 \
  --intervention-output-dir /root/robolab/experiments/preliminary/sparsity/velocity_cache/step0_profile_fixed_roi_velocity_cache_shift5_BananaInBowlTask_sim_v2/server
```

### 6.3 RoboLab

```bash
cd /root/robolab/RoboLab
OMNI_KIT_ACCEPT_EULA=Y .venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --task BananaInBowlTask \
  --num-envs 1 \
  --num-runs 1 \
  --output-folder-name step0_profile_fixed_roi_velocity_cache_shift5_BananaInBowlTask_sim_v2 \
  --video-mode viewport \
  --headless
```

## 7. 代码位置

- 控制器、mask 聚合与 velocity cache：`cosmos_framework/scripts/robolab_step0_fixed_roi_velocity_cache.py`；
- 实验服务器：`cosmos_framework/scripts/action_policy_server_robolab_step0_fixed_roi_velocity_cache.py`；
- paired runner：`tools/run_robolab_step0_fixed_roi_velocity_cache.py`；
- CPU 单元测试：`tests/test_robolab_step0_fixed_roi_velocity_cache.py`；
- whole-layer sparse dispatch：`cosmos_framework/model/generator/mot/unified_mot.py`。
