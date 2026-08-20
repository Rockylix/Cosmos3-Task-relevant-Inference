# 历史结果：Core-64、冻结 Stable、压缩 Adaptive 的 V6-A 实验

> 注意：本文件记录旧 V6-A（Core48→Stable→扩张 Core64）的历史结果。当前 worktree
> 已切换为 Version1（Direct Core64→Stable）；下述计时、误差和闭环结果不能代表 Version1。

## 1. 实验目的

在当前 `conditional Step0 dense + 后续全 block 稀疏 + velocity cache` 路径上，
验证以下预算重分配是否能在继续降低 token 的同时保持输出与闭环能力：

- Core：每个 future frame 从 48 增加到 64；
- Stable：数量和具体空间位置都保持原 Core-48 方案不变；
- 总预算：G1/G2/G3 从 `192/160/144` 降为 `184/152/136`；
- 由缩小后的剩余容量自动减少 Adaptive 和 Budget Fill。

固定 BananaInBowlTask、chunk 3、policy seed 579362556、guidance 3、4 个
UniPC step、shift 5，Torch compile 和 CUDA graphs 关闭。

## 2. Mask 构造

不能只把 `core_token_budget` 从 48 改成 64。旧实现先构造 Core，再让
Stable 避开所有 future frames 的 Core union；直接扩 Core 会改变 Stable 的
具体位置，无法隔离变量。

V6-A 按以下顺序构造：

1. 使用原来的 Core score 和 Core-48 得到 reference Core；
2. 使用 reference Core union 作为 forbidden 集合，生成并冻结原 Stable mask；
3. 每帧保留 reference Core-48，再从非 Stable 位置补入 16 个最高 Core score
   token，得到 Core-64；
4. 从 G3 向 G1 构造嵌套 mask，先合并 Core、Stable 和深组继承 token；
5. 剩余容量只选择正 Adaptive score；仍不足时才用 raw-mass Fill；
6. 严格验证 `G3 subset G2 subset G1`、精确预算、Core/Stable 无重叠及有限值。

## 3. 真实 profile 的 mask 结果

配对 benchmark 中同一份 Step-0 conditional profile 得到：

| Group | 方案 | K | Core | Stable | Adaptive（8 帧） | Fill（8 帧） |
|---|---|---:|---:|---:|---|---|
| G1 | 当前 C | 192 | 48 | 88 | 56/56/56/45/44/54/56/56 | 0/0/0/11/12/2/0/0 |
| G1 | V6-A | 184 | 64 | 88 | 32/32/32/25/25/32/32/32 | 0/0/0/7/7/0/0/0 |
| G2 | 当前 C | 160 | 48 | 72 | 每帧 40 | 每帧 0 |
| G2 | V6-A | 152 | 64 | 72 | 每帧 16 | 每帧 0 |
| G3 | 当前 C | 144 | 48 | 64 | 每帧 32 | 每帧 0 |
| G3 | V6-A | 136 | 64 | 64 | 每帧 8 | 每帧 0 |

离线重建额外确认：V6-A Stable mask 与原 Core-48 方案逐 token 完全一致，
reference Core 也完全一致；最终 Core 每帧均为 64，Core/Stable overlap 和两级
subset violation 均为 0。

全部 block call 的平均 GEN token 保留比例从 `0.613320` 降为 `0.595215`，
平均节省量从 `1196` 增加到 `1252 / 3093`，即每个 block call 平均再少 56 个
GEN token。

## 4. 稳定单 chunk 计时

同一已加载模型先全局预热；每个模式预热 3 次，再随机交替测量 20 次。CUDA
同步包围 `generate_samples_from_batch`，不使用首请求、RPC 或任务 wall time
计算加速比。

| Mode | Median (s) | P90 (s) | Mean (s) | Speedup vs Dense |
|---|---:|---:|---:|---:|
| Dense | 0.854751 | 0.857537 | 0.854400 | 1.000x |
| 当前 C / K192-160-144 | 0.575190 | 0.576898 | 0.574920 | 1.486x |
| V6-A / Core64 / K184-152-136 | 0.566990 | 0.569776 | 0.567523 | 1.508x |

V6-A 相对当前 C 的 median 延迟降低约 1.43%，吞吐提高约 1.014x。这是实测
收益；额外 token 缩减有限，而且 Step-0 conditional 仍全量，因此不会出现大幅
加速。

## 5. 单 chunk 输出误差

| 策略 | Action MSE | Action rel-L2 | Action cos | Delta cos | Jerk cos | Vision rel-L2 | Vision cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| 当前 C | 0.011288 | 0.076431 | 0.997106 | 0.018670 | -0.071234 | 0.386387 | 0.922680 |
| V6-A | 0.008897 | 0.067855 | 0.997825 | 0.042390 | -0.041258 | 0.380351 | 0.925148 |

该 chunk 上 V6-A 在 token 更少时，整体 action 和 vision 指标均略优于当前 C；
这支持“扩大高置信 Core 可以补偿一部分 Adaptive 尾部”的判断。但 Delta/Jerk
cosine 仍很低，不能仅凭整体 action cosine 宣称动力学无损。

## 6. 单任务闭环

BananaInBowlTask、环境 seed 0、policy seed 579362556：

- Success：`1/1`；
- 171 environment steps，score 1.0；
- 6 个 generation chunks；
- 排除冷请求后 generation median `0.585373 s`、P90 `0.586083 s`；
- `TARGET_OBJECT_DROPPED=1`；
- 完成原因：`Completed subtask 'pick_and_place' 1/1`。

这只是一条 episode 的功能检查，不能视为稳定成功率，也不能证明多任务或多
seed 不退化。

## 7. 输出位置

- 单 chunk：
  `/root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/velocity_cache/core64_stable_fixed_k184_152_136_BananaInBowlTask_c3_v1/`
- 闭环服务端：
  `/root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/velocity_cache/core64_stable_fixed_k184_152_136_BananaInBowlTask_seed579362556_v1/server/`
- viewport 视频：
  `/root/robolab/RoboLab/output/core64_stable_fixed_k184_152_136_BananaInBowlTask_seed579362556_v1/BananaInBowlTask/Pick_up_the_banana_and_place_it_in_the_bowl_0_viewport.mp4`

## 8. 复现命令

稳定单 chunk：

```bash
cd /root/robolab/cosmos-framework-edge-version1
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=. \
LD_LIBRARY_PATH= /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  tools/benchmark_robolab_v5_3_stable_chunk.py \
  --output-root /root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/velocity_cache/<new-run-id> \
  --modes dense c_cond_b0_sparse_opt c_core64_stable_fixed_opt \
  --warmup-rounds 3 --measure-rounds 20
```

闭环 policy server：

```bash
cd /root/robolab/cosmos-framework-edge-version1
HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=. \
LD_LIBRARY_PATH= NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python -m \
  cosmos_framework.scripts.action_policy_server_robolab_v5_3_acd_packed_kernel \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --host 127.0.0.1 --port 8000 --format-prompt-as-json True --no-guardrails \
  --seed 579362556 --deterministic-seed --guidance 3.0 \
  --num-steps 4 --shift 5.0 \
  --ablation-mode c_core64_stable_fixed_b0_sparse \
  --intervention-output-dir /root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/velocity_cache/<new-run-id>/server
```

RoboLab：

```bash
cd /root/robolab/RoboLab
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
OMNI_KIT_ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --task BananaInBowlTask --num-envs 1 --num-runs 1 \
  --headless --video-mode viewport --output-folder-name <new-run-id>
```

## 9. Seed 579362556 九任务闭环

相同环境 seed 0、policy seed 579362556、shift 5、4 steps、无 compile/CUDA
graphs，各 task 一个 episode：

| 策略 | Success | 稳定单 chunk median | Dense speedup |
|---|---:|---:|---:|
| Current C / K192-160-144 | 4/9 (44.4%) | 0.575190 s | 1.486x |
| V6-A / Core64 / K184-152-136 | 5/9 (55.6%) | 0.566990 s | 1.508x |

V6-A 额外成功 `RubiksCubesInBinTask`；其余八项胜负与 Current C 相同。但
`RubiksCubeTask` 的成功步数由 137 增至 578，因此不能从 pooled 结果推断轨迹
等价或稳定成功率提升。

完整逐任务表、闭环 warm 请求计时、token 统计与异常说明：

`/root/robolab/cosmos-framework-edge-version1/experiments/preliminary/sparsity/velocity_cache/core64_stable_fixed_9tasks_seed579362556_v1/report_cn.md`
