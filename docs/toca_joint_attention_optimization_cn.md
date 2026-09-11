# ToCa-Future 联合 attention / 评分优化

日期：2026-09-10。分支：`experiment/toca-future`。本轮只优化实现，不改 D/C/D/C 调度、0.25 基础 MLP 刷新比例或 Future-only 范围。

## 1. 官方评分路径核对

固定官方 commit：`e84096ffd85af4540a6a1f64e3334e428e1b7377`，只读源码在 `/tmp/toca-source-review-9SHbfd/ToCa`。

官方 FLUX 路径为：

```text
flux-ToCa/src/flux/math.py::attention
  → apply_rope
  → dot_product_attention
      QKᵀ / sqrt(d) + mask
      → P = softmax(...)
      → output = P V
      → incoming_score = P.mean(head).mean(query)
  → 保存 score 到 attn_map
modules/cache_functions/scores.py::attn_score
  → L2 normalize
modules/cache_functions/score_evaluate.py
  → 加缓存年龄项
modules/cache_functions/cache_cutfresh.py
  → descending argsort → fresh token → 更新年龄
```

来源：[官方联合输出/评分代码](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/flux-ToCa/src/flux/math.py)、[评分归一化](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/flux-ToCa/src/flux/modules/cache_functions/scores.py)、[年龄项](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/flux-ToCa/src/flux/modules/cache_functions/score_evaluate.py)、[选择与年龄更新](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/flux-ToCa/src/flux/modules/cache_functions/cache_cutfresh.py)。

所以用户指出的问题成立：官方从同一次显式 attention 的权重同时得到 AV 和评分，而首版 Edge 移植在 native attention 之外又做了一次 FP32 QK/softmax。旧实现的评分定义没有颠倒，但存在重复计算。

本轮保留已确认的 Edge 适配边界，不能与官方未修改 FLUX 混称：

\[
P_{h,q,j}=\operatorname{softmax}_{j\in UND_b\cup GEN}
\left(Q_{h,q}K_{m(h),j}^{\top}/\sqrt{128}\right),
\qquad
u_j=\frac{1}{16N_{GEN}}\sum_{h=1}^{16}\sum_{q\in GEN}P_{h,q,j}.
\]

`N_GEN=3093`，候选仅 L1–L8 的 2720 个 token；query 不是仅 action 的 32 行。GQA 为 `m(h)=floor(h/2)`。两分支 score 平均后在候选域 L2 normalize，再加原年龄项并选择；分支数值缓存独立。UND 长度与 key 集合按实时 metadata 处理。

## 2. 已实现的计算路径

`cosmos_framework/inference/toca_joint_attention.py` 提供推理专用 Triton kernel：

1. GEN Q/K/V 使用真实 post-RoPE 张量与 GQA head 映射，不重排位置编号。
2. 分块计算 QK **一次**，用在线 softmax 累积 AV；同时暂存该次计算的 FP32 logits 和最终行 normalizer。
3. 对指定 future key 列，用同一 logits/normalizer 恢复概率并归约 head/query，得到 incoming score；此阶段不再计算 QK。
4. 返回 GEN attention output 与 score。UND 自身的 causal attention 仍调用原 kernel，文本 KV cache 的写入/读取不变。
5. 缓存步仍使用原 attention kernel：L0/action 当前 Q，对当前完整 UND+GEN K/V；future attention residual 复用，MLP 按原 Top-K 部分刷新。

AV 的 Tensor Core 乘法遵循 BF16/FP16 数值精度；logits、softmax normalizer、评分归约是 FP32。与官方公式数学一致，不宣称与另一个 attention kernel 逐位相同。没有 query 抽样、future-only softmax 分母、均匀随机评分或降低 Dense 性能。

原 `reference` 路径保留。只有显式选择 `joint` 时替换 ToCa full block 内的 dispatcher，退出/异常时恢复；不修改原 `unified_mot.py`、Dense checkout 或 ASI checkout。

### 临时显存和尝试过的布局

联合实现暂存 `[16,N_GEN,N_UND+N_GEN]` FP32 logits，不跨 layer/step/chunk 缓存该矩阵。在 Banana conditional 的 `N_UND=158` 下，logits 理论占用约 0.60 GiB，另有很小的 row normalizer/分块归约缓冲。它不是 112 层调用同时驻留的 112 份矩阵。

也实测了“只保存 future 列”的紧凑缓冲：数值检查通过，但实际整 chunk median 约 1.068 s，比完整 key 轴的连续存储慢，因此未采用。结果保留在 `experiments/toca_joint_kernel_fidelity10_v2/`，其中 `kernel_future_columns_snapshot.py` 保存该尝试的源码。不能把分配显存减少直接当成延迟降低。

## 3. 验证协议与结果入口

最终执行目录：`experiments/toca_joint_kernel_fidelity10_v3/`。

- 同一批十任务 Dense 第三个 request，复用此前固定的 data_batch、seed、4 steps、shift=5、guidance=3；不重跑闭环、不按图像好坏筛选。
- 逐 full step、branch、block，在**相同 Q/K/V**上比较联合评分与完整 FP32 softmax 参考，以及联合 AV 与原 kernel AV。
- 在相同 Q/K 得到的两组分数上比较 Top-K 集合；这是评分实现验证，不是声称新旧完整轨迹的所有 mask 相同。
- 旧 reference ToCa 重放与上一轮保存的 action/vision 逐元素核对，确保参考路径没有被优化改变。
- 新旧 ToCa 和 Dense 的 final action、L1–L8 latent、32 个 future RGB 分别比较。RGB 固定映射 `clip((decoded+1)/2,0,1)`；每个 chunk 独立计算后做等权平均。
- 十样本之后，对同一个 Banana chunk：每方法预热两次，30 轮循环轮换 Dense/reference/joint 的顺序；compile/CUDA Graph 均关闭。
- 计时包含 generation、controller、评分、选择、cache、完整序列恢复及正常有限值检查；不包含额外的参考评分校验、VAE decode、CPU 输出复制、落盘和仿真。

原始数据：`block_correctness.csv`、`selection_correctness.csv`、`fidelity.csv`、`frame_metrics.csv`、`timing.csv`、`checks.json`、`summary.json`。每任务 `comparison.png` 从上到下为 Dense、旧 ToCa、新联合 kernel ToCa；单帧 PNG 与新 action/latent 的 `joint_outputs.pt` 同目录保存。

本轮不复用旧 ToCa 1/10 作为优化版闭环成功率。新 kernel 的 BF16 舍入可能传播到后续 hidden 和选择，必须由新的闭环实验确认任务表现。

### 3.1 最终数值检查

- 1120 次 full block attention/评分对照：评分最大 relative-L2 为 `1.03932e-6`，最大绝对误差 `1.11759e-8`。
- 相同 Q/K/V 下，联合 AV 对原 kernel 的最大 relative-L2 为 `0.00146580`（约 0.147%）。
- 560/560 次 Top-K **集合**与相同输入的 FP32 参考评分完全一致。
- 10/10 个旧 ToCa reference 重放与上一轮保存的 action/vision 逐元素完全一致。
- 全量列/紧凑 future 列两种布局在这十个样本上的最终 action/vision 逐元素一致，布局选择只依据实测速度；不是筛选更好看的输出。
- 没有 NaN/Inf；保存 10 张三行对比图、240 张 PNG，以及逐 block、逐 frame 的数值。

CPU/GPU 单元测试共 **22 passed**：覆盖 BF16/FP16、真实 Edge shape、GQA、完整 key 分母、非连续 future 索引、评分归约轴、backend 开关与异常后 dispatcher 恢复。

另通过真实 `ToCaPolicyService` 的单请求冒烟：实际 backend 为 `joint`，224 次 block 调用及实际模块 token 行数检查通过、结果有限。记录在 `experiments/toca_joint_server_smoke_gvi_3agh/`。这是服务推理路径检查，不是 RoboLab 闭环成功率测试；首请求带校验的冷启动耗时不计入下表。

### 3.2 正式 warmed generation 耗时

RTX 4090，同一 Banana chunk、seed、加载模型，预热后交替 30 轮：

| 策略 | Mean | Median | P90 | 相对 Dense 加速 |
|---|---:|---:|---:|---:|
| Dense 原生 eager | 0.874161 s | 0.875143 s | 0.876436 s | 1.000× |
| ToCa reference：独立评分 | 1.200288 s | 1.201070 s | 1.202184 s | 0.729× |
| ToCa joint：共享 QK | 0.785899 s | 0.786370 s | 0.787712 s | **1.113×** |

新版本比旧 ToCa 快约 **1.527×**，generation median 减少约 **34.5%**；相对高效 Dense 的延迟减少约 **10.1%**。这两个 speedup 的分母不同，不能混用。

这是实际 generation 加速，不是完整仿真任务加速，也不是论文在其他模型上的加速倍率。当前仍有两个 full steps；L0/action 必须实时计算、K/V 在缓存步仍完整，MLP 仅部分缓存，此外共享评分仍有 FP32 logits 读写和归约开销。因此尚未达到论文其他工作点的倍率。本轮没有为追求速度改变调度、刷新率或评分定义。

### 3.3 预测帧与动作

每任务 Dense 第三个请求，先逐 chunk 计算，再对十个 chunk 等权平均：

| 相对 Dense | RGB cosine ↑ | RGB rel-L2 ↓ | PSNR ↑ | SSIM ↑ | Latent cosine ↑ | Latent rel-L2 ↓ | Action MSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 ToCa reference | 0.992555 | 0.122843 | 23.391 dB | 0.739884 | 0.951701 | 0.309674 | 0.00394157 |
| 新 ToCa joint | 0.992525 | 0.123247 | 23.357 dB | 0.739346 | 0.951391 | 0.310583 | 0.00404597 |

直接以旧 ToCa 为参考，新 ToCa 的差异为：

| 输出 | Cosine ↑ | Relative-L2 ↓ | 其他 |
|---|---:|---:|---|
| Action | 0.999939 | 0.011023 | MSE=0.000152887 |
| Future latent | 0.998483 | 0.054766 | 不声称逐位一致 |
| Future RGB | 0.999702 | 0.024237 | SSIM=0.974469；PSNR=37.715 dB |

评分在数值容差内正确不等于整条推理输出逐位不变：AV 的 BF16 舍入差异会传播到后续 hidden，继而可能影响跨版本的评分与选择。因此报告保留上述差异，不宣称 action 不变或闭环成功率不变。

图片：[香蕉](../experiments/toca_joint_kernel_fidelity10_v3/BananaInBowlTask/comparison.png)、[魔方](../experiments/toca_joint_kernel_fidelity10_v3/RubiksCubeTask/comparison.png)。保真度是相对 Dense/旧 ToCa 的预测保持程度，不是真实未来视频的准确率。

## 4. 如何启用

为保留旧命令的可复现性，默认仍是 `reference`。在原 ToCa server 命令添加：

```bash
--toca-attention-backend joint
```

完整十任务启动器已支持同一开关（此命令供下一次闭环使用，本轮未执行）：

```bash
cd /root/robolab/worktrees/toca-future
env LD_LIBRARY_PATH='' \
  /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  tools/run_toca_future_smoke.py \
  --attention-backend joint \
  --run-dir experiments/toca_joint_smoke10_s0_p0_v1 \
  --port 8017
```

首请求的 `all_full_equivalence.json` 仍验证 **reference observer 与 Dense** 的精确一致性；不将其冒充 joint kernel 的逐位校验。`validation_scope.json` 明确记录这一区别，实际运行 backend 写入 `runtime.json` 和每个请求的 config。

重跑同输入数值校验与正式计时（换用不存在的新输出目录）：

```bash
cd /root/robolab/worktrees/toca-future
env CUDA_VISIBLE_DEVICES=0 COSMOS_TRAINING=0 LD_LIBRARY_PATH='' \
  HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/root/robolab/worktrees/toca-future:/root/robolab/cosmos-edge-overlay \
  /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -m cosmos_framework.scripts.benchmark_toca_joint \
  --source-run experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2 \
  --output-dir experiments/toca_joint_kernel_fidelity10_v4 \
  --tasks 10 --repeats 30 --decode
```

所有环境、权重均复用项目现有路径，未下载新环境/权重；没有 commit、merge 或 push。

## 5. 修改范围

- 新增 `cosmos_framework/inference/toca_joint_attention.py`：共享 QK 的 attention/评分 kernel。
- 更新 `cosmos_framework/inference/toca_future.py`：显式 backend 选择及局部 dispatcher 接入。
- 更新 `cosmos_framework/scripts/action_policy_server_robolab_toca_future.py`、`tools/run_toca_future_smoke.py`：启动参数与验证范围记录。
- 新增 `cosmos_framework/scripts/benchmark_toca_joint.py`、`tests/test_toca_joint_attention.py`，补充 `tests/test_toca_future.py`。
- 本报告及原迁移/比较文档增加新旧 backend 的说明。

全部修改保留在 `/root/robolab/worktrees/toca-future` 的 `experiment/toca-future` 分支。原 Dense checkout（`cache`，`324574a454a989f9b8f5392f7673ace487243e8d`）和原 ASI checkout（`version/core80-stable104-action-weighted`，`62a09f7f95f1f0a54d5af9c2b8c530c862db9025`）的 branch、HEAD 与 clean 状态均未改变。
