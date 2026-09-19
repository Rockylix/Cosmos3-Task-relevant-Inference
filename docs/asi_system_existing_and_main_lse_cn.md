# ASI 系统消融：已有优化逐项测试与主 attention LSE 复用

日期：2026-09-18。实验分支 `experiment/asi-system-ablation`，工作树 `/root/robolab/worktrees/asi-system-ablation`；起点 `87caf7454e203cb25ff768ac0a86d56d2dea10d6`。所有修改仅在该 worktree；不改生产默认策略，不合并、不推送。

按用户更新的 workflow：已有机制不中途暂停，全部累计测完；随后完成一轮目标优化，再讨论下一轮。使用项目 `kernel-optimizer` skill 的 NVTX 归因、逐项消融和独立审查流程；按用户约定，正式时间与 nsys 分开，不自动提交。

## 固定实验条件

- RTX 4090；项目原 `.venv`，PyTorch 2.10.0+cu130；没有下载环境或模型。
- 既有 BananaInBowlTask chunk 3 录制输入，SHA256 `64bbd643ec42a826cd607e36d594c1d0d859fd4824ac89a1b131f1bbcbd4ceaa`。
- shift=5、4 steps、guidance=3，Core80+Stable104、K184、Top6；1 dense conditional + 7 sparse stacks。
- 不启用 compile、CUDA Graph、velocity cache、跨 chunk mask 复用；基准是 ASI legacy，而非 Dense。
- 每个模式预热 5 次、正式 20 次；同一轮正序/倒序交替。每次重新 deepcopy 输入、重置 RNG、重建 controller；scheduler 与缓存 request-local。
- 时间为 CUDA synchronize 包裹的 `generate_samples_from_batch`，包括在线评分/选择/pack/restore；不含加载、输入复制、RNG 重置、CPU 结果转存、VAE decode、RPC 或仿真。
- seed=1097657232 与 seed+1 做严格输出检查；正式时间用原 seed。不同轮次各自重测 B0，不能混用分母。

## 1. 已有优化的累积结果

| 版本 | 本行累计新增的机制 | Mean ms | Median ms | P90 ms | 相对同轮 B0 |
|---|---|---:|---:|---:|---:|
| B0 | ASI legacy | 583.705 | 583.972 | 586.020 | 1.0000× |
| B1 | 八帧评分算术批处理 | 575.269 | 575.635 | 577.455 | 1.0145× |
| B2 | 连续布局 slice/reshape 评分准备、geometry/layout 复用 | 556.960 | 557.362 | 559.285 | 1.0477× |
| B3 | 逐层有限值检查延后至原 planner | 555.396 | 555.907 | 557.187 | 1.0505× |
| B4 | 稀疏 SequencePack metadata 缓存 | 553.699 | 554.005 | 555.964 | 1.0541× |
| B5 | decoder metadata 返回评分，即已有 optimized-eager | 553.051 | 552.956 | 554.801 | 1.0561× |

B2 不是“只缓存一个 Python 字典”：它完整采用已有 fast scorer 的连续布局准备，将逐帧 gather/stack 改成 slice/reshape，避免索引上传，权重改由设备算子准备，并 clone 每层 profile。此处作为一个布局/准备机制披露，不把收益全部归于 layout lookup。

布局缓存仅在 request 内同一 packed 对象被再次使用时命中；没有跨请求复用输出。B3 仍由 planner 检查 NaN/Inf，未删除安全验证。B4 使用 production 的真实 UND 长度；本轮无 Graph padding，因此与 legacy 等价，不能将此结论外推至任意 padded pack。B5 的内部参数名 `compile_profile_decoder=True` 只开启 metadata 接线，并不会自行调用 torch.compile。

### 正确性

- B1 raw profiles 与 legacy 逐位一致。
- B2–B5 的 raw profiles 最大绝对差：seed0 为 3.1665e-8，seed+1 为 4.0978e-8。
- 两个 seed 的 Top6、Core、Stable、execution mask 均一致，XOR=0。
- 两个 seed 最终 action 和 future latent 均逐位一致，MSE=0；每次计时还核对相对预热输出的一致性。
- CPU 已有机制测试 8/8，通过独立 review。

### nsys 证据

每个模式独立预热后捕获一个 `asi.chunk.generate`。使用 runtime launch correlation 关联异步 GPU kernel，而非硬切 CPU 时间窗；无未匹配 kernel、无 Graph launch。224 次 layer NVTX 明确标注 4 steps × 2 branches × 28 blocks。

| 版本 | chunk kernel 数 | chunk GPU memcpy 数 | chunk cudaStreamSynchronize 数 |
|---|---:|---:|---:|
| B0 | 23622 | 1420 | 1114 |
| B1 | 22306 | 1420 | 1114 |
| B2 | 21102 | 702 | 368 |
| B3 | 20906 | 646 | 312 |
| B4 | 20868 | 551 | 217 |
| B5 | 20868 | 551 | 217 |

B0/B1 trace 来自上一轮，B2–B5 来自本轮；用于结构计数，不把跨轮 profiler wall 当加速比。B2 的 `asi.score.fast` 不含外围逐层校验，其 host 时间不能直接与 B0 的整个 `asi.score` 比较。

已有 optimized-eager 的评分仍有 28 次额外 attention（32 action queries）；下一项只删除这部分，不同时改成融合 scorer。

## 2. 第一轮目标：P0 / B6 主 LSE 复用

### 数学保持

\[
U_{l,f,p}=\sum_{r=1}^4\alpha_r\frac1{16}\sum_h
\exp\big(s\langle Q_{a(f,r),h},K_{f,p,\lfloor h/2\rfloor}\rangle-\mathrm{LSE}_{a(f,r),h}\big),
\quad\alpha=(1/6,1/3,1/3,1/6).
\]

分母仍包含所有真实可见 key：UND/text、L0、future 和 action。仍使用 post-RoPE Q/K、原 GQA、原 FP32 局部 QK 计算和聚合顺序；不将 softmax 限制到 future frame。

唯一变化是 LSE 的来源：

```text
B5: 主 attention → O
    额外 32-query attention → 丢弃 O，取得 LSE → 局部评分

B6: 主 attention → O + GEN LSE
    截取 GEN-relative action rows → 局部评分
```

Flash2 wrapper 原本就调用 `return_attn_probs=True` 得到 O/LSE；之前通常只返回 O。因此此次不用换 kernel，只打开既有 LSE 返回路径，删除额外 attention 及其评分侧全 K/V 拼接。

实现：

- `mot/attention.py` 增加默认 false 的 `return_gen_lse`，仅标准 two-way full/GEN attention 返回 LSE。
- text-KV dispatcher 显式透传；cached-text 路径使用真实缓存 UND K/V 与当前 GEN K/V。
- 临时放在 SequencePack 私有键，`unified_mot.py` 立即 pop；以 GEN 内 action_start 切 `[1,32,16]`，不加 UND 偏移，不跨 block 保存。
- scorer 接受可选 action_lse；没有提供时原额外 attention 路径不变。
- `MainLSEController` 仅在上下文内启用，异常退出也恢复原属性；非 profile 层不请求 LSE。
- 显式拒绝训练、sharded/custom-parallel dispatcher、three-way、multi-control、NATTEN；只验证当前单样本标准 Edge 路径。
- LSE 使用当前后端的自然对数单位，不做 log2 变换。

### 数值边界

完整 GEN 查询与 32-query attention 可选择不同 tile/归约方式，数学等价不等于逐位相同。本轮额外在真实同一 Q/K 上独立审计（不计入时间）：

- 28 层主 attention 开/关 LSE 返回时，O 全部逐位一致。
- 复用 LSE 与重新执行完整 GEN attention 得到的对应 LSE 全部逐位一致。
- 主 LSE 与旧 32-query LSE 最大绝对差为 9.5367e-6。
- 同 Q/K 下两种 LSE 生成的 profiles 最大绝对差为 5.9605e-8。
- 两个 seed 的 B6 raw profiles 相对 legacy 最大绝对差分别为 6.3330e-8、5.4017e-8；Top6/Core/Stable/mask 均一致，action 与 future latent 均逐位一致。

这些是当前 chunk、两个 seed 的实测，不能保证所有输入的 Top-K 边界都不会翻转。

### 时间与新 trace

单独重测 B0/B5/B6，每条路径 20 次；这是同轮比较，不将前表的 583.972 ms 作为分母。

| 版本 | Mean ms | Median ms | P90 ms | 相对本轮 B0 |
|---|---:|---:|---:|---:|
| B0 ASI legacy | 579.451 | 578.908 | 581.024 | 1.0000× |
| B5 已有 optimized-eager | 549.089 | 548.663 | 551.020 | 1.0551× |
| B6 B5 + 主 LSE 复用 | 548.181 | 548.272 | 549.665 | 1.0559× |

B6 相对 B5 median 减少 0.391 ms（约 0.071%），1.000714×；**不能称为稳定显著的 chunk 加速**。当前样本的功能正确性和重复计算删除已经验证，但不是“大幅加速”。

两份目标轮 trace 同样只捕获 warmed `asi.chunk.generate`：

| nsys 指标 | B5 | B6 |
|---|---:|---:|
| 额外 32-query attention 调用 | 28 | 0 |
| 评分 GPU kernel 数 | 616 | 504 |
| 评分 GPU kernel 时间合计 | 4.332 ms | 3.405 ms |
| 评分 CPU NVTX 时长（含等待） | 11.867 ms | 5.313 ms |
| chunk kernel 数 | 20868 | 20756 |
| chunk GPU memcpy 数 | 551 | 551 |
| chunk cudaStreamSynchronize 数 | 217 | 217 |
| nsys chunk wall（非正式计时） | 703.339 ms | 703.408 ms |

`asi.score.lse_attention` 在 B6 完全消失；评分 kernel 减少 112 个，即每层去掉两次拼接和额外 attention 的两个 kernel。整体 nsys kernel GPU 总时间约 489.70 / 489.65 ms，几乎不变；不能将某一子范围降低直接当作整 chunk 降低。

B6 评分直接 GPU 时间现为约 3.4 ms/chunk。下一步融合 scorer 可以研究避免 KV head 显式展开、FP32 大中间量及多次 kernel launch，但不能预期从这 3.4 ms 的 GPU 工作直接省出数十毫秒。前三类 GEMM 仍合计约 266.67 ms，说明模型主体是重要成本；没有 ncu roofline 证据，不将此写成严格 compute-bound 结论。

特别注意：B5/B6 `asi.select` 的 host 时长约 27 ms，但选择 kernel 仅约 0.249 ms。它包含前面 GPU 工作的排队等待；删除逐层同步后，等待位置会移到 planner。不能声称 GPU planner 能直接节省 27 ms。

**本轮到此暂停。下一轮候选是融合局部 scorer（P1），或者优先进入模型主体 compile/launch 优化；属于不同归因方向，待用户决定，不自动叠加。**

## 3. 文件、测试和复现

新增累计控制器：`cosmos_framework/inference/asi_existing_ablation.py`；目标控制器：`cosmos_framework/inference/asi_main_lse.py`。基准脚本：`tools/benchmark_asi_batch_only.py`。对已有 scorer/attention 的改动默认关闭。

18 项 CPU 单元测试通过，包括原评分、GQA/全 key 归一化、布局缓存、原 planner 校验、metadata 不复用旧 hidden/RoPE、主 O 不变、normalized/cached UND keys、删除额外 attention、错误 LSE 形状、unsupported-path 拒绝、controller 异常恢复。

```bash
cd /root/robolab/worktrees/asi-system-ablation
export PYTHONPATH="$PWD:/root/robolab/cosmos-edge-overlay"
export COSMOS_TRAINING=0 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ASI_PY=/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python

# 新目录，拒绝覆盖历史结果；两个测试顺序执行。
"$ASI_PY" tools/benchmark_asi_batch_only.py --modes b0 b1 b2 b3 b4 b5 \
  --output experiments/existing_system_banana_c3_v2/timing
"$ASI_PY" tools/benchmark_asi_batch_only.py --modes b0 b5 b6 \
  --output experiments/main_lse_banana_c3_v2/timing

nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o experiments/nsys_reports/iter_6_v2 \
  "$ASI_PY" tools/benchmark_asi_batch_only.py --trace-mode b6 \
  --output experiments/main_lse_banana_c3_v2/trace_b6
"$ASI_PY" -m unittest discover -s tests -p 'test_asi_*.py' -v
```

当前结果路径（都被 Git 忽略）：

- `experiments/existing_system_banana_c3_v1/timing/results.json`：已有优化逐项时间、源码/input hashes、逐 seed gate、实际 controller summary。
- `experiments/main_lse_banana_c3_v1/timing/`：目标轮结果、same-QK audit、数值输出。
- `experiments/nsys_reports/baseline.nsys-rep`、`iter_1.nsys-rep` 至 `iter_5.nsys-rep`：已保存的逐项 trace，附 SQLite 和解析 JSON。
- `experiments/nsys_reports/main_lse_b5.nsys-rep`、`main_lse_b6.nsys-rep`：目标轮优化前后 trace，附 SQLite 和解析 JSON。

检查结束后 GPU 无本实验残留进程。原生产 ASI 工作树仍在 `version/core80-stable104-action-weighted`、原 HEAD 与原有 9 个未跟踪文档/工具保持不变。本实验没有启动 RoboLab，也没有改动 Dense 仓库。

没有进行闭环任务测试，不报告成功率，也不将理论 FLOPs 或 token 比例当作系统加速。
