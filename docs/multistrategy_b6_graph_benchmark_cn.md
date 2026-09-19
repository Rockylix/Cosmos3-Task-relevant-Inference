# Edge：B6 ASI 与 ToCa / WorldCache / C3ache 单 chunk 对比

日期：2026-09-19。本轮全部重新测量，没有将历史时间填入新表。优化代码提交为 `186a3e630378a208d0e4948563ac201a37822769`，分支 `experiment/asi-system-ablation`。此提交保存默认关闭的系统优化及实验入口；没有将 B6 或 compile/Graph 改为生产默认，没有 merge/push。

## 1. 实测结果

用户确认的配置：**Dense eager；其余四策略 compile + CUDA Graph。** ASI 使用本轮 B6 主 attention LSE 复用，不启用 velocity cache。

计时单位秒；FLOPs 是下一节定义的逻辑矩阵运算量，以 Dense=100%。

| 策略 | Mean | Median | P90 | 配对 Dense median | 实测加速 | FLOPs 剩余 |
|---|---:|---:|---:|---:|---:|---:|
| 干净 Dense eager 独立校验组 | 0.854361 | 0.853981 | 0.856181 | 0.853981 | 1.000× | 100.00% |
| ASI B6 C80/S104 | 0.467159 | 0.467213 | 0.469329 | 0.860358 | **1.841×** | **60.73%** |
| ToCa D/C/D/C，r=0.25 | 0.687218 | 0.687901 | 0.688926 | 0.870702 | 1.266× | 65.68% |
| WorldCache D/D/D/C | 0.570145 | 0.570643 | 0.571758 | 0.866910 | 1.519× | 75.14% |
| C3ache period=2，刷新/复用均摊 | 0.576529 | 0.576390 | 0.578044 | 0.870791 | 1.511× | 75.14% |

加速比 = **同进程交替测得的 Dense median / 策略 median**。各策略分支分进程顺序运行，其配对 Dense 有小幅波动，因此不能用第一行 0.853981 作为后面所有行的分母。每个分支关闭策略后的 Dense 输出都先与干净 checkout 的 Dense 输出逐元素对齐。

此表的加速包含算法减少计算和通用编译优化两种收益，不是同后端条件下的纯算法收益。没有增加 Dense compile 对照或闭环任务测试。

### C3ache 均摊方式

| 单元 | Mean | Median | P90 |
|---|---:|---:|---:|
| refresh chunk：D/D/D/D | 0.738221 | 0.737871 | 0.740487 |
| hit chunk：C/C/D/D | 0.414838 | 0.414803 | 0.415553 |
| period=2 均摊 | 0.576529 | 0.576390 | 0.578044 |

先对每个周期计算 `(refresh_time + hit_time) / 2`，再对20个周期求 mean/median/P90。不是只报 hit chunk，也不是简单平均两个 median。

这次用同一录制输入反复验证缓存开销和有效调用数，不代表真实观测连续变化时的跨 chunk 输出质量。

## 2. 配置与计时边界

- RTX 4090，驱动 `580.178.04`，PyTorch `2.10.0+cu130`。
- 只用项目现有环境：`/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv`；未下载环境、模型或数据。
- 输入为 `BananaInBowlTask` chunk 3，noise seed `1097657232`，4步 UniPC、shift=5、guidance=3、batch=1；另用 seed+1 做未重放陈旧输出检查。
- 输入 SHA256：`64bbd643ec42a826cd607e36d594c1d0d859fd4824ac89a1b131f1bbcbd4ceaa`。
- 录制文件位于生产 checkout 的 `experiments/asi_smoke10_s0_p0_v1/_temporary_inputs/BananaInBowlTask.pt`。该文件携带的是 ASI 输出，所以**没有将其中 outputs 错当作 Dense 参考**；先从干净 Dense 源码重新生成参考。
- 每次深拷贝输入、重置 Python/NumPy/Torch/CUDA RNG。ASI、ToCa controller 每次新建；WorldCache history 每次新建。只有 C3ache 按指定刷新/复用周期保留跨 chunk 状态。
- 每模式5次预热、20次正式测量；C3ache为5个预热周期、20个正式周期。Dense/策略交替，C3ache始终refresh紧接hit。
- 主时间是 CUDA 同步包围的 generation + controller context，包括在线评分、选择、稀疏执行、缓存查找和恢复。排除CPU输入深拷贝、summary/验证、输出拷回、VAE decode、RPC、仿真和编译预热。
- 正式计时不挂 profiler；Graph审计另运行一次。
- 所有模式 `pad_for_cuda_graphs=False`，使用实际序列长度。decoder/head 编译 `fullgraph=True, dynamic=True, mode=reduce-overhead`。这是 compiler 管理的 layer/head Graph，不是整个 chunk 的单一 Global Graph。
- 线程配置统一：Torch/OMP/OpenBLAS=4；缓存代码可复用，不缓存本次模型输出用于替代计算。

| 策略 | 本轮实际配置 |
|---|---|
| ASI | B6主LSE；正常step0 conditional采28层；Top6；Core80+Stable104；一次dense+七次sparse；同chunk固定K184 mask |
| ToCa | full_steps=[0,2]；fresh_ratio=0.25；layer_slope=0.5；age_weight=0.25；period=2；spatial_bonus=0；CFG selection=independent；joint attention评分 |
| WorldCache | D/D/D/C；独立CFG history；percentile_stable=.30，percentile_chaotic=.70，n_max=6；compile_compatible=True |
| C3ache | refresh_period=2；dense_tail_steps=2；整GEN栈residual；刷新8次dense，复用4次dense+4次cache |

注意 ToCa 的 `r=0.25` 不是每层固定重算25%：本配置 future MLP fresh 数由 B0 的1020递减到 B27 的340；缓存步future attention residual仍复用，保护token的Q/O和全部GEN的K/V仍计算。

## 3. FLOPs：逻辑主矩阵乘法，非硬件计数

按实际token布局和权重header检查，计算28层的 Q/K/V/O、QK/AV、两层MLP矩阵乘法，乘加=2 FLOPs。包含两分支各一次 UND prefill，以及 B6额外局部QK评分。

**不包含** VAE、输入/输出头、时间嵌入、UniPC、RMSNorm、ReLU²、softmax/exp、排序、索引和缓存搬运；不包含kernel tile冗余和attention重算。因此不是全pipeline FLOPs，也不是Nsight硬件浮点指令计数。尤其WorldCache/C3ache输出头调用方式不同，但位于此统计边界之外。

| 策略 | TFLOP/chunk | Dense比例 | 等吞吐理论加速 | 实测加速 |
|---|---:|---:|---:|---:|
| Dense | 88.301740 | 100.00% | 1.000× | 1.000× |
| ASI B6 | 53.627590 | 60.73% | 1.647× | 1.841× |
| ToCa | 57.997355 | 65.68% | 1.523× | 1.266× |
| WorldCache | 66.351758 | 75.14% | 1.331× | 1.519× |
| C3ache 均摊 | 66.351758 | 75.14% | 1.331× | 1.511× |

等吞吐理论值只是在相同有效吞吐率假设下的FLOPs比，不是实际加速上限。编译融合可提高执行效率；评分/排序与较小GEMM也可使实际收益低于FLOPs比。不能仅用本表推断各策略瓶颈。

### 维度与公式

从本次Dense首层8次forward的实时metadata确认：GEN `N=3093`，conditional UND `Uc=158`，unconditional UND `Uu=19`。前者含 `9×340+33`；保护token `P=340+33=373`，ASI稀疏GEN `Ns=P+8×184=1845`。

读取权重header验证28层共224个矩阵形状：hidden `D=2048`，query宽 `Q=2048`，KV宽 `K=1024`，MLP宽 `I=9216`。当前MLP是两层Linear + ReLU²，不是三矩阵SwiGLU。

```text
F(n,u) = 2nD(2Q+2K) + 4nDI + 4n(n+u)Q
U(u) = 2uD(2Q+2K) + 4uDI + 4[u(u+1)/2]Q
Tund = 28[U(Uc)+U(Uu)]
Pair = 28[F(N,Uc)+F(N,Uu)]

Dense = Tund + 4Pair
ASI = Tund + 28[F(N,Uc)+3F(Ns,Uc)+4F(Ns,Uu)] + Score
Score = 28×2×32×340×2048

fresh(l) = floor[0.25(1.5-l/27)×2720]
C(u,l) = 4PDQ + 4NDK + 4P(N+u)Q + 4[P+fresh(l)]DI
ToCa = Tund + 2Pair + 2 sum_l[C(Uc,l)+C(Uu,l)]

WorldCache = Tund + 3Pair
C3ache = [(Tund+4Pair)+(Tund+2Pair)]/2
```

ToCa真实controller累计Q/KV/O/MLP行数已与上述公式核对。joint attention同时产生输出/评分，不人为再加一次完整QK。

B6 `Score=1,247,805,440 FLOPs`；相对旧评分路径消除的额外32-query attention为 `23,862,444,032 FLOPs`。这个算术量仅占Dense约0.027%，因此主LSE优化不是本次整体加速的主要算术来源；它的主要目的还有减少重复操作和数据处理。

## 4. 正确性、Graph证据与局限

- 五个worker均正常退出，无 `failure.json`；四策略均通过 pristine-Dense exact gate、最终结果NaN/Inf检查及changed-seed gate。
- 每次正式输出与前次同模式输出按 `rtol=atol=1e-5` 检查，**不是逐位一致性声明**。ASI另外检查重复执行mask精确一致；本轮没有单独强制ToCa每次选择集合精确一致。
- ToCa和WorldCache的编译适配器在eager下与原eager结果exact；ToCa同时检查score/indices exact。
- 独立CPU profiler审计观察到的实际 `cudaGraphLaunch`：ASI264、ToCa264、WorldCache198、C3ache refresh264/hit152，与预期layer/head调用总数相容。本轮没有逐layer NVTX审计，因此这里不扩大为每个具体layer的Graph覆盖证明；ASI前轮逐层证据见[编译报告](asi_compile_stages_cn.md)。
- Graph与eager并非数值等价；ASI执行mask XOR=60（8×340布尔格的差异元素数，不是60个替换对）。不声称compile不影响动作、图像或闭环成功率。
- ToCa编译与eager的112组选区索引顺序均不完全相同，最低成员重合率约93.735%；因此不能只依据适配器eager exact就声称编译后的选择也不变。

以下只是同策略compiled-vs-eager的数值检查，使用原始 `result[action][0]` 和 `result[vision][0]` 全张量，包含condition位置；**不是排除condition后的32 horizon/action指标，也不是仅future latent或RGB保真度表**。

| 策略 | 原始action MSE | 原始vision rel-L2 |
|---|---:|---:|
| ASI | 0.00184173 | 0.264274 |
| ToCa | 0.00018620 | 0.054016 |
| WorldCache | 0.00206611 | 0.165334 |
| C3ache（refresh） | 0.00045249 | 0.131707 |

本轮只证明此录制输入上的稳定运行时间和代码路径，没有测试十任务成功率、变化观测的C3ache保真度或新输入shape的重新编译成本。

## 5. 复现与归档

- 工具：[benchmark_edge_multistrategy.py](../tools/benchmark_edge_multistrategy.py)、[summarize_edge_multistrategy.py](../tools/summarize_edge_multistrategy.py)。
- 参数、逐次时间、Graph审计、源码hash、输出gate：`experiments/multistrategy_b6_graph_banana_c3_v1/{dense,asi,toca,worldcache,c3ache}/results.json`。
- 总表：[comparison.csv](../experiments/multistrategy_b6_graph_banana_c3_v1/comparison.csv)、[comparison.json](../experiments/multistrategy_b6_graph_banana_c3_v1/comparison.json)。实验数据被Git忽略；只提交源码、测试和本文。

```bash
cd /root/robolab/worktrees/asi-system-ablation
EDGE_PY=/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python
# 使用新输出目录；工具顺序拉起分支进程，自动设置项目PYTHONPATH及offline环境。
"$EDGE_PY" -u tools/benchmark_edge_multistrategy.py \
  --output experiments/multistrategy_b6_graph_banana_c3_repeat --warmups 5 --repeats 20
"$EDGE_PY" tools/summarize_edge_multistrategy.py \
  --input experiments/multistrategy_b6_graph_banana_c3_repeat
```

这些脚本以本项目既有权重、录制输入和下列checkout为前提；clone本分支不会自动下载权重、复制capture或还原其他分支未提交的适配器。

| 用途 | checkout / 基础提交 |
|---|---|
| Dense | `/root/robolab/cosmos-framework-edge`，`cache / 1dec8b9`，干净 |
| ASI B6 | 本工作树，`experiment/asi-system-ablation / 186a3e6`；测速时新增工具未提交，模型源码已提交 |
| ToCa | `/root/robolab/worktrees/toca-compile-graph`，`f7cf5d6` + 已有未提交编译适配器 |
| WorldCache | `/root/robolab/worktrees/worldcache-compile-graph`，`763b6d1` + 已有未提交编译适配器 |
| C3ache | `/root/robolab/worktrees/c3ache`，`225dd58`，原有未跟踪文档/工具不影响本次模型 |

本轮没有修改后三者源码；其精确测量版本由每个results中的 `source_sha256` 标识，不只用基础HEAD。汇总工具重新核对输入hash、采样参数、source hash、布局、有效调用数、Graph/changed-seed gate和权重形状。

已有29项ASI CPU测试通过；新增4项逻辑FLOPs测试通过。独立审查已核对比较协议、缓存隔离和FLOPs边界。
