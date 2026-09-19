# ASI 系统优化消融：B0 legacy → B1 八帧评分批处理

日期：2026-09-18。第一轮已完成；**停止在 B1，未继续 B2**。

## 1. 范围与基准

- 实验分支：`experiment/asi-system-ablation`。
- 工作树：`/root/robolab/worktrees/asi-system-ablation`。
- 起点 HEAD：`87caf7454e203cb25ff768ac0a86d56d2dea10d6`。
- B0 是 **ASI legacy**，不是 Dense，也不是当前 optimized-eager 的混合优化版本。
- B1 仅叠加八个 future frame 的评分算术批处理；没有加入几何缓存、稀疏 metadata 缓存、首轮 decoder metadata 回传、compile、CUDA Graph 或 velocity cache。
- 不改 token 选择、Core80+Stable104、Top-6 block、采样器、CFG、恢复规则；仍是每 chunk 1 dense + 7 sparse stacks，每帧保留 184 个 future token。
- 原 ASI 工作树的 branch、HEAD 和 `git status --short` 与实验前一致；没有 commit、merge 或 push。

## 2. 本轮只改什么

新增 `cosmos_framework/inference/asi_batch_only.py`，继承 legacy controller，只覆盖 `_capture_profile`。

原算法对八帧逐帧计算 action-to-future relevance。B1 将八份 Q/K/LSE 堆叠后，用一次批量 einsum 计算：

\[
A_{f,h,a,p}=\exp\{s\langle Q_{f,h,a},K_{f,h,p}\rangle-\mathrm{LSE}_{f,h,a}\},
\qquad U_{f,p}=\sum_{a=1}^{4}\alpha_a\frac1H\sum_h A_{f,h,a,p},
\quad \alpha=(1/6,1/3,1/3,1/6).
\]

LSE 仍来自全部可见 key，而不是仅 future key；GQA 仍按真实 head 映射展开。FP32 评分、先 head mean 再 action 加权保持一致。

刻意保留以下开销以避免混入下一项优化：

- 每层 action index / weight 创建和上传；
- 每帧 positions 创建、上传、gather 和 GQA 展开；
- 每层额外 32 action-query attention 调用获得 LSE；
- 每层 finite / negative 检查及其 host 同步；
- 原始 mask 选择、pack、restore 和 decoder 执行。

批处理新增的 `stack` 缓冲也是 B1 的实际成本，没有隐藏。

## 3. 测试口径

硬件 RTX 4090，驱动 580.178.04，PyTorch 2.10.0+cu130，项目原有 venv；未下载环境或权重。

输入：`BananaInBowlTask` chunk 3 的既有录制数据：

`/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/experiments/asi_smoke10_s0_p0_v1/_temporary_inputs/BananaInBowlTask.pt`

输入 SHA256：`64bbd643ec42a826cd607e36d594c1d0d859fd4824ac89a1b131f1bbcbd4ceaa`。

shift=5，4 denoise steps，guidance=3，原 seed=1097657232；另用 seed+1 做正确性验证。每次 fresh input deepcopy、重置 RNG、新 controller；原采样函数创建 request-local scheduler/缓存。不复用前一次输出。

正式时间：每条路径预热 5 次，然后 20 轮 B0/B1 与 B1/B0 交替；每条路径 20 次。计时范围为 `generate_samples_from_batch` 到最终 CUDA synchronize，不含加载、输入 deepcopy、RNG 重置、CPU 汇总、VAE decode、RPC 和仿真。没有 profiler。

nsys 独立进程分别预热 5 次，仅捕获一次 generation；不将 nsys 时间作为加速结论。

## 4. 正式无 profiler 时间

| 累积版本 | Mean | Median | P90 | 相对 B0 median 加速 |
|---|---:|---:|---:|---:|
| B0 ASI legacy | 577.992 ms | 578.036 ms | 579.452 ms | 1.000× |
| B1 B0 + 八帧评分批处理 | 569.801 ms | 569.608 ms | 570.707 ms | 1.0148× |

Median 减少 8.427 ms，即延迟下降约 1.46%。这是单输入的 warmed chunk 结果，不是闭环成功率，也不能套用历史 Dense 的时间作为本轮分母。

## 5. 数值与独立审查

- CPU 单元测试 4/4：多 GQA 布局、非连续位置、全 key 归一化、NaN 检查、controller 覆盖范围。
- 同一 Q/K 下，28 层 B0/B1 raw profile 全部逐位一致。
- 两个 seed 的 R/H/Q、Top-6 block、Core mask、Stable mask、最终 execution mask 全部一致；mask XOR=0。
- 最终 action 与 future vision latent 全部逐位一致，MSE=0、max absolute error=0；无 NaN/Inf。
- instrumented B0 与直接调用原 legacy controller 逐位一致；正式重复和 trace 相对预热输出均通过回归。
- 原 seed 的 Top-6 为 `[21,15,20,18,19,23]`。
- 独立 reviewer 审查确认只加入批处理；同意测量范围和 correlation-based nsys 归属。
- 这些是当前输入与两个 seed 的验证，不宣称所有输入均有 bitwise 一致保证。

正式计时完成后仅增强 harness 的 Core/Stable 显式 gate 和 summary 保存；原计时 JSON 不覆盖，其源码 hash 如实保留。其已有 Core/Stable XOR 字段另行检查均为 0；trace JSON 保存实际 controller summary。没有修改评分代码或计时区域。

## 6. nsys 归因

两份 trace 都是单个 `asi.chunk.generate`，含 224 个 step/branch/block NVTX 范围、28 次评分、1 次选择、7 次 pack 与 7 次 restore。没有 Graph launch。

| nsys 指标 | B0 | B1 |
|---|---:|---:|
| chunk 内 GPU kernel 数 | 23622 | 22306 |
| 评分 kernel 数 | 3332 | 2016 |
| 评分 GPU kernel 时间合计 | 7.676 ms | 5.549 ms |
| 评分 CPU NVTX 范围时长（含等待） | 109.985 ms | 98.245 ms |
| 评分内 cudaMemcpyAsync 次数 | 784 | 784 |
| 评分内 cudaStreamSynchronize 次数 | 784 | 784 |
| 全 chunk GPU kernel 时间合计 | 492.199 ms | 491.163 ms |
| 全 chunk GPU memcpy 次数 | 1420 | 1420 |
| nsys chunk wall（不用于加速比） | 751.491 ms | 752.611 ms |

收益定位：评分少了 1316 个 kernel（约 39.5%），评分 GPU 时间约下降 27.7%；拷贝和同步未减少。因此本轮端到端收益有限。两份单次 trace 的整体 wall 不显示加速，说明不能拿带 profiler 的单次 wall 替代正式重复计时。

剩余成本证据：

1. B1 耗时前三个 kernel 名称均为 BF16 GEMM，合计约 266.136 ms；支持“主要 GPU 计算集中在矩阵乘法”，不能仅凭 nsys 宣称严格 compute-bound。
2. 评分仍有 784 次拷贝和 784 次流同步；几何索引准备是下一项待验证候选，不能预先归因全部同步都可被缓存移除。
3. 评分用的额外 LSE attention 仍为 28 次调用、56 个 kernel，GPU 合计约 0.674 ms；本轮没有共享实际主 attention 的 LSE。

解析用 CUDA runtime launch correlation 归属 NVTX 阶段，不按异步 GPU 执行时间硬切 CPU 子范围。检查所有关联 kernel 落在同步后的根范围内，unmatched=0。inclusive 父子范围不可相加；GPU kernel sum、busy union、chunk wall 分开。wall 减 kernel busy 还包含复制和等待，不是纯 CPU 开销。

## 7. 产物与复现

- 正式结果：`experiments/batch_scoring_banana_c3_v1/timing/results.json`
- 数值明细及两个输出：同目录 `gate.json`、`outputs_seed_offset0.pt`、`outputs_seed_offset1.pt`
- B0 trace：`experiments/nsys_reports/baseline.nsys-rep`
- B1 trace：`experiments/nsys_reports/iter_1.nsys-rep`
- SQLite 与解析 JSON：同目录 `baseline.sqlite` / `iter_1.sqlite`、`*_summary.json`
- trace 回归/实际 token summary：`experiments/batch_scoring_banana_c3_v1/trace_b0/results.json` 和 `trace_b1/results.json`

所有数据在被 Git 忽略的 experiments 下。源文件与文档留在实验 worktree，未合并。

```bash
cd /root/robolab/worktrees/asi-system-ablation
export PYTHONPATH="$PWD:/root/robolab/cosmos-edge-overlay"
export COSMOS_TRAINING=0 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ASI_PY=/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python

# 输出目录必须是新的，脚本拒绝覆盖。
"$ASI_PY" tools/benchmark_asi_batch_only.py \
  --output experiments/batch_scoring_banana_c3_v2/timing

# 两次分别运行，切勿并发占用 GPU。
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o experiments/nsys_reports/baseline_v2 \
  "$ASI_PY" tools/benchmark_asi_batch_only.py --trace-mode b0 \
  --output experiments/batch_scoring_banana_c3_v2/trace_b0
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o experiments/nsys_reports/iter_1_v2 \
  "$ASI_PY" tools/benchmark_asi_batch_only.py --trace-mode b1 \
  --output experiments/batch_scoring_banana_c3_v2/trace_b1

nsys export --type sqlite --output experiments/nsys_reports/iter_1_v2.sqlite \
  experiments/nsys_reports/iter_1_v2.nsys-rep
"$ASI_PY" tools/analyze_asi_nsys.py \
  --sqlite experiments/nsys_reports/iter_1_v2.sqlite \
  --output experiments/nsys_reports/iter_1_v2_summary.json
"$ASI_PY" -m unittest discover -s tests -p test_asi_batch_only.py -v
```

## 8. 下一轮候选（未执行）

B2 = B1 + 评分几何/GPU 索引缓存，暂不加入稀疏 metadata 缓存、取消验证或编译。保持相同门槛，报告 B0、B1、B2 累积结果及 B2 相对 B1 的增量。待用户讨论确认后开始。
