# ToCa-Future → Cosmos3 Edge：主 baseline 迁移与对比方案

日期：2026-09-10。状态：**已实现独立 ToCa-Future 入口，并执行十任务 smoke；实际配置、验证覆盖与结果见 [十任务测试报告](toca_future_smoke10_cn.md)**。本文保留完整迁移设计；未在测试报告标记完成的性能优化和扩展验证仍属于后续计划。

后续已完成不重复 QK 的联合 attention/评分实现，数值校验及实际加速见 [联合 kernel 优化报告](toca_joint_attention_optimization_cn.md)。通过 `--toca-attention-backend joint` 显式启用；旧 reference 路径和报告保留。

## 1. 结论和实验要回答的问题

ToCa 适合作为主要的 diffusion acceleration baseline。它与当前 ASI 的区别足够明确：ToCa 根据生成 token 对其他 token 的影响与缓存年龄分配重算；ASI 根据 action-to-future attention 选择持续参与计算的视觉区域。

应验证的问题是：**面向生成特征的跨 timestep 缓存，在取得同等实际加速时，能否同样保护动作和闭环任务？** 不预设答案，也不能用视频相似度代替任务成功率。

用户已确定仅采用 **ToCa-Future**，与 ASI 对齐为只减少未来视觉帧的计算；不实现或测试缓存 action 输出的 Joint 版本。

| Token 范围 | 缓存步的 attention | 缓存步的 MLP |
|---|---|---|
| Future L1–L8 | 复用最近 full step 的 attention 输出；当前 K/V 仍全量生成 | 按 ToCa 分数选中行重算，其余复用缓存输出 |
| L0、condition action、全部预测 action | 用当前完整 K/V 重算 | 始终重算 |
| UND/text | 保留原有条件处理和 UND KV cache | 不引入额外 ToCa 缓存或剪枝 |

不改输出头、CFG 或 UniPC。比较组固定为 **Dense、ASI、ToCa-Future**。

干预对象对齐不等于计算方式相同：ASI 使用 compact future token 序列；ToCa-Future 保留完整当前 K/V，以支持 L0/action 的实时 attention，主要减少 future query/output 路径与部分 MLP 计算。报告使用明确名称 **ToCa-Future（Edge adaptation）**，不宣称这是官方未修改的执行范围。

## 2. 审阅依据和版本固定

官方仓库已只读拉取至 `/tmp/toca-source-review-9SHbfd/ToCa`，未安装其环境或权重。固定 commit：`e84096ffd85af4540a6a1f64e3334e428e1b7377`。

实验分支：`experiment/toca-future`。

实验 worktree：`/root/robolab/worktrees/toca-future`。

起点：`62a09f7f95f1f0a54d5af9c2b8c530c862db9025`，来自 ASI 工作树 `/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted` 的 `version/core80-stable104-action-weighted` 分支。现有 Dense 工作树 HEAD 为 `324574a`。二者均保留，不合并、不推送。

本 worktree 后续推理共用项目现有 `/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python`，同时显式设置 `PYTHONPATH=/root/robolab/worktrees/toca-future`，并核验模块来源；不创建或下载新环境。

主要依据：

- [ICLR 2025 论文，方法、附录 A.1、A.6/Table 7](https://proceedings.iclr.cc/paper_files/paper/2025/file/bbe024e0517fe12ac3a8d388b19ff9fe-Paper-Conference.pdf)。Table 7 有 **4-step FLUX.1-schnell**：Dense 2.882 s，ToCa 1.890 s，作者报告 1.53×。这是 FLUX 图像生成结果，不是 Edge/机器人结果，也不能作为 Edge 的预计速度。
- [DiT 完整和缓存 block 实现](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/DiT-ToCa/models.py#L148-L190)。
- [DiT attention 分数](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/DiT-ToCa/cache_functions/scores.py#L5-L16)、[年龄分数](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/DiT-ToCa/cache_functions/score_evaluate.py#L64-L74)、[选择、局部 bonus、计数更新](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/DiT-ToCa/cache_functions/cache_cutfresh.py#L17-L82)。
- [FLUX joint attention 和统计实现](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/flux-ToCa/src/flux/math.py#L6-L80)、[block 实现](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/flux-ToCa/src/flux/modules/layers.py)。
- [FLUX fresh ratio 调度](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/flux-ToCa/src/flux/modules/cache_functions/fresh_ratio_scheduler.py#L42-L56)、[初始化参数](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/flux-ToCa/src/flux/modules/cache_functions/cache_init.py#L71-L79)、[前三步强制 full](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/flux-ToCa/src/flux/modules/cache_functions/cal_type.py#L3-L28)。
- [OpenSora 的实际 entropy score](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/Open-Sora/opensora/models/cache_functions/scores.py#L5-L22)、[强制刷新](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/Open-Sora/opensora/models/cache_functions/global_force_fresh.py#L8-L30)、[模块刷新周期](https://github.com/Shenyi-Z/ToCa/blob/e84096ffd85af4540a6a1f64e3334e428e1b7377/Open-Sora/opensora/models/cache_functions/force_scheduler.py#L13-L26)。

### 2.1 不能从名称推断的实现细节

1. **不是“Top-K token 进入整个 block，其他 token 原样跳过”。** DiT 的缓存步直接复用整个 attention 输出，只对选中行重新运行 MLP。当前输入仍走 residual add，恢复完整 hidden 后进入下一层。
2. 缓存 attention 的位置是其输出投影之后、residual add 之前；缓存 MLP 的位置是 MLP 输出、residual add 之前。DiT 原有 AdaLN gate 仍取当前步；Edge 的这些 residual 路径没有同样的 gate，不能凭空加上。
3. 评分不要求每个缓存 block 先跑一个 Dense oracle。它使用最近一次 full forward 留下的 attention 统计，以及持续更新的缓存年龄。
4. 官方各模型不是同一套启用项。DiT 启用 incoming attention、年龄和局部空间 bonus；FLUX 的启用路径主要是 incoming attention、年龄，局部 bonus 被注释；OpenSora 的当前评分主要用 cross-attention entropy，self-attention 分数被注释。
5. 官方 DiT/FLUX 的层级比例和 timestep 索引方向也不同。不能把某一配置文件的数字当成所有架构通用的最终超参。
6. 当前 FLUX 默认前三步强制 full，且配置周期为 4；它不等于论文 Table 7 的 `N=2` 四步配置。OpenSora 还将 SA/MLP 周期设为 3，直接用于四步时这两个模块会四步全量。少步数迁移必须显式记录执行表。

本方案采用 **DiT 的串行 attention→MLP 缓存边界，参考 FLUX 的 joint-attention incoming score**。这是针对 Edge 架构的 ToCa adaptation，不宣称逐行复现某个 FLUX/OpenSora checkpoint 的实现和指标。

## 3. Edge 的真实插入位置

以下源代码行号对应 `62a09f7`。

| 位置 | 已核实的作用 | 迁移要求 |
|---|---|---|
| `cosmos_framework/inference/edge_core_stable_layout.py:9` | 根据 packed metadata 定位 L0–L8/action | 复用布局解析，不复用 ASI 选择策略 |
| `cosmos_framework/model/generator/mot/unified_mot.py:720` | 取得实际 attention 的 Q/K/V，包括缓存 UND K/V | ToCa 统计使用真实 post-QK-norm、post-RoPE 张量及有效 mask |
| 同文件 `:1028` | Transformer controller 调度入口 | ToCa 默认关闭，与 ASI controller 互斥 |
| 同文件 `:1207` | pre-attention RMSNorm | 完全复用当前算子 |
| 同文件 `:1252`、`:1263` | attention 输出加到当前 hidden | attention cache 插在加法之前 |
| 同文件 `:1271`、`:1294` | pre-MLP RMSNorm | 对需要重算的当前 token 运行 |
| 同文件 `:1278`、`:1306` | GEN MLP | gather 后真实缩短输入，再 scatter 新输出 |
| 同文件 `:1290`、`:1319` | MLP 输出加 residual | 用完整缓存/重算拼合输出加当前 residual |
| `cosmos_framework/model/generator/omni_mot_model.py:2714` | 一次 solver velocity evaluation | 维护 denoise step，而非把两次 CFG forward 算两步 |
| 同文件 `:2787` | 实际 CFG 公式及后续可选 normalize | 不修改 |
| `cosmos_framework/model/generator/diffusion/samplers/unipc.py:83` | velocity→UniPC update | 不修改；每步始终完整 latent 输入/输出 |

当前 ASI 的关键设置见 `cosmos_framework/scripts/robolab_version1.py:4`、`:34`：step 0 conditional 全量 profile，Core80+Stable104，每帧 K184，其余七次 CFG forward 使用固定 compact mask；每个 stack 末端从该 stack 输入恢复未选 hidden。**这不是 ToCa 的跨 timestep 模块缓存，不应叠加使用。**

典型 DROID 输入为 L0–L8 共 9 个 latent、每帧 17×20=340 token，future 共 2720，action 共 33，因此 GEN 共 3093。实现必须读取 metadata 并验证，不能硬编码排布；UND 长度随实际条件变化，分别记录两分支有效长度和 padding。

Edge 配置例见 `cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json:4`：hidden 2048、28 层、16 Q heads、8 KV heads、head dimension 128。实际加载模型的参数和模块类型仍必须在实验 manifest 中导出，不能按其他 DiT 的 MHA/GELU 或其他 Cosmos 的 SwiGLU 估算。

## 4. 数学定义：缓存什么、如何更新

记 `s` 为执行顺序的 denoise step，`l` 为 block，`b` 为 CFG branch。先写单个 branch/block 的 GEN 路径。`X` 是当前步当前层输入，不是缓存 hidden。

全量计算：

\[
A_{s,l,b}=W_O\operatorname{Attention}(Q(X),K(X,UND),V(X,UND)),
\quad Z_{s,l,b}=X_{s,l,b}+A_{s,l,b}
\]

\[
F_{s,l,b}=\operatorname{MLP}(\operatorname{RMSNorm}(Z_{s,l,b})),
\quad Y_{s,l,b}=Z_{s,l,b}+F_{s,l,b}.
\]

`Q/K/V` 内含源码真实的输入 RMSNorm、投影、QK norm 和 RoPE。存储：

\[
C^A_{l,b}[V]\leftarrow A_{s,l,b}[V],\qquad C^F_{l,b}[V]\leftarrow F_{s,l,b}[V].
\]

其中 `V` 为 future token 集合，缓存只需为这些行分配存储，不保存 L0/action 的 ToCa 模块缓存。

### 4.1 Future 缓存步

记 `V` 为 L1–L8，`P` 为 L0 和全部 action token。先用**当前完整 GEN 输入**计算全量 K/V，用 `P` 的当前 Q 计算 attention：

\[
A[P]=W_O\operatorname{Attention}
\left(Q_P,[K_{UND},K_{GEN}],[V_{UND},V_{GEN}]\right),
\quad A[V]=C^A_{l,b}[V].
\]

之后 `Z=X+A`，MLP 在 `P ∪ I_V` 上重算，更新 future 选中行的缓存：

\[
F[P\cup I_V]=\operatorname{MLP}(\operatorname{RMSNorm}(Z[P\cup I_V])),
\quad C^F_{l,b}[I_V]\leftarrow F[I_V].
\]

`V\I_V` 的 MLP 输出取缓存，按原位置拼成完整 `F`，最后 `Y=Z+F`。未选行得到缓存 residual，不是零 residual；不能用旧 `Y` 覆盖当前 `X`，否则会丢失当前 step 的输入演化。

这保证 action query 不会因为本层 ToCa 选择而丢失任何当前 video key。但此前层的 hidden 已包含近似误差，**不意味着 action 输入或输出等于 Dense**。

计算行数必须按模块分别统计：

- Q 和 O projection：`N_P=340+33=373` 行；
- K/V projection：完整 `N_GEN=3093` 行；
- attention：373 个 query，对 `3093+N_UND` 个有效 key；
- MLP：`373+K_future` 行。

不能把 Future 模式估算为“整个 attention 都省掉”，也不能只用 373 行 K/V，否则变成另一种 token pruning。

## 5. ToCa 的 token 评分

### 5.1 Incoming attention：关注的是被读取程度

在最近一次 full step `r(s)`，取模型真正使用的 attention：

\[
P^{b,l,h}_{q,j}
=\operatorname{softmax}_{j\in\mathcal K_b}
\left(\frac{Q^{b,l,h}_q K^{b,l,m(h)\top}_j}{\sqrt{d_h}}+B_{q,j}\right),
\quad \mathcal K_b=UND_b\cup GEN.
\]

`m(h)` 是实际 GQA 映射；16:8 的当前配置下，每两个 Q heads 对应一个 KV head。不能混用 head 索引。

\[
u^{b,l}_j=\frac1{H_QN_{GEN}}\sum_h\sum_{q\in GEN}P^{b,l,h}_{q,j}.
\]

这是按 **query 维求和的 key-column 分数**。不能改成每个 query 对 key 求和，那在完整 softmax 下几乎恒为 1；也不能只采样 action 的 32 行然后称为 ToCa。

本文建议两分支用同一组重算索引。先对共同 GEN 位置的标量分数平均：

\[
\bar u_j=(u^{cond}_j+u^{uncond}_j)/2,
\qquad s_{1,j}=\frac{\bar u_j}{\|\bar u_{\mathcal E}\|_2+\epsilon}.
\]

唯一候选域为 `E=L1..L8`，L0/action 不参与 Top-K。评分的 query 仍来自全 GEN，而不是仅 action 行。这是显式的 Edge CFG/候选域适配。两分支 UND 长度可不同，不对 text 列做不合法的位置平均。softmax 分母始终包含全部可见 key，不改成 future 内归一化。

### 5.2 缓存年龄

令 `n_j` 为该层 MLP token 连续复用而未重算的次数，`N` 为 full refresh 周期：

\[
s_{3,j}=n_j/N,\qquad z_j=s_{1,j}+0.25s_{3,j}.
\]

`I=TopK(z,K_l)`。每个 token 重算后 `n_j=0`，其余 `n_j+=1`；full step 全部清零。CFG 共用索引/年龄时，一对 branch 只更新一次计数，不能加两次。数值缓存必须 branch 独立。

来源是官方 `score_evaluate.py` 和 `cache_cutfresh.py` 的实际实现。值得注意：若执行 `D,C,D,C`，每个缓存步之前刚刚全量刷新，第一次选取时所有年龄相同，年龄项不会改变排序。要观察其作用，需要 `D,C,C,C` 等连续缓存配置，不能把无效的 s3 消融当作负结果。

### 5.3 文本熵和空间 bonus 如何处理

官方 cross-attention 的 `s2` 是单个生成 token 对文本 key 的分布熵，高熵 token 更倾向重算。它**不是** ASI 的 action 空间熵，也不是“越低熵越重要”的 block quality。

Edge 没有 OpenSora 那种独立 cross-attention 模块；主迁移采用官方 FLUX 的有效项 `s1+s3`，不擅自创造 action entropy 项。若以后从 joint attention 的 UND 子矩阵构造 `s2`，需作为单独适配消融，明确是否重新对 UND 归一化。

FLUX 路径的局部 bonus 默认关闭；保留 DiT 风格的可选消融：每个 2×2 空间窗口的最高分乘 `1+0.6`，再全局 Top-K。这是软 bonus，不是保证每窗口一定保留一个 token。

17×20 不是正方形且高度为奇数，不能直接调用 DiT 的 `sqrt(N)` reshape。若启用该项，应逐 future frame 按真实 H/W 分窗，边界只包含有效格点，禁止 padding 被选中。它不对 action token 构造虚假的空间邻域。

### 5.4 不混入 ASI 的选择机制

主 ToCa 不使用 Core/Stable/Adaptive/Fill、不选 top-6 blocks、不加 `[1/6,1/3,1/3,1/6]` action 权重、不固定每帧相同 ROI，也不把当前 ASI mask 当作 ToCa mask。

ToCa 的 Top-K 在候选域整体选择，允许帧间数量不同。由于每层都以“当前 residual 主干 + 重算/缓存模块输出”恢复完整序列，不存在剪完后忘记原空间位置的问题。误差问题仍需实验检验。

## 6. 四步调度和最小参数集

`D` 表示两分支、全部 28 层正常计算并更新 future cache；`C` 表示 ToCa-Future 缓存计算，L0/action 仍每层重算。索引 0–3 表示执行顺序，不是原始噪声数值。每个 run 记录实际 timestep/sigma 列表。

| 配置 | step 0 | step 1 | step 2 | step 3 | 用途 |
|---|---|---|---|---|---|
| Dense | D | D | D | D | 数值基线 |
| 保守调试 | D | D | D | C | 验证一次跨步模块缓存 |
| 周期 N=2 | D | C | D | C | 主起点；对应论文四步实验的周期思想 |
| 周期 N=4 | D | C | C | C | 更高缓存率和缓存年龄作用 |
| 末步恢复，备选 | D | C | C | D | 若末步缓存敏感，单独标注的调度消融 |

主实验不借用 FIS 的“首层、末两层全量”规则，也不把末步默认强制 full。所有 block 使用统一缓存机制，通过刷新比例调深度；保护层方案应另列消融。

记 `rho` 为 **MLP 重算比例**，不是论文 `R` 的缓存比例。建议主配置的层级调度为：

\[
\rho_l=\operatorname{clip}\left[\rho_0
\left(1+0.5-2\cdot0.5\frac{l}{27}\right),0,1\right],
\quad K_l=\lfloor\rho_l|\mathcal E|\rfloor.
\]

这里移用 FLUX 的线性层级因子，去掉 Edge 不存在的 double/single-stream 倍率；倍率并入可调的 `rho0`。这是明确的迁移选择，不是论文所有模型统一使用的公式。另保留 `layer_slope=0` 的恒定比例校验。

参数筛选可采用 `N∈{2,4}`、`rho0∈{0.10,0.25,0.50}`，固定其余项。原建议用 `D,D,D,C` 做一次缓存调试；本轮十任务 smoke 实际固定主起点 `D,C,D,C`、`rho0=0.25`，不进行参数扫描或失败重试。如需同 token 数辅助对照，Future 模式可加恒定 `rho=184/340`，但 **MLP 行数相同不意味着 FLOPs 或延迟相同**。

约束：不增加 denoise steps 来帮助 ToCa，不更换 sampler/shift，不跨 chunk 保留模块输出 cache；每个新 chunk 从 full step 建立自己的 cache。仅缓存内存缓冲区可以复用分配。

## 7. 评分与高效 kernel：不可漏算的工程成本

官方 FLUX 的 `math.py` 显式计算 QK、softmax、AV；即使不启用 ToCa，其相关路径也调用手写 attention，SDPA 调用被注释。**不能让 Edge Dense 保留低效手写 attention，从而人为抬高 ToCa 加速比。**

建议分两阶段实现：

1. 数值参考：小样本上显式 FP32 attention 概率，验证分数、Top-K 和重算结果。此实现不作为正式性能版本。
2. 正式实现：模型输出继续用原生高效 attention；仅在后续将使用 cache 的 full step 获取 incoming score。若 native API 能返回每行 LSE，则分块计算 `exp(QK/sqrt(d)+mask-LSE)` 并直接归约 key-column 分数。不保存全量 attention map。

LSE 本身不能提供 column sum；仍有额外 QK 和归约成本。如果当前后端拿不到 LSE，则分块先计算完整有效 key 的 logsumexp，再计算需要的列分数。不能用只含 future 的分母，也不能假设原先 ASI 仅重算 action rows 的 helper 足够。

首版不修改核心 FlashAttention 内核。后续如实现融合统计 kernel，必须证明与参考分数等价，并独立报告优化前后。所有评分、Top-K、gather/scatter、cache 写回均计入延迟。

缓存占用估算，batch=1、BF16、28 层、2 分支、attention/MLP 两类输出：

\[
\text{bytes}=2\times28\times2\times N_{cached}\times2048\times2.
\]

ToCa-Future 仅保存 2720 行约 **1.162 GiB**。不含模型、UND cache、临时张量和 CUDA allocator。只保留必要的模块值、标量评分和年龄；不能保存所有 layer 的完整概率矩阵。4090 是否能与 Isaac Sim 同驻，需在实际模型加载后测峰值，不能仅凭此估算断言可行。

## 8. 具体实现拆分与数据流

已从上述固定 HEAD 建立独立 `experiment/toca-future` 分支和 `/root/robolab/worktrees/toca-future` 工作树；原 Dense 和 ASI 工作树保留。后续模型修改、测试和实验只在该实验 worktree 进行。实现时先验证“新分支所有 controller 关闭”与现有 Dense 的配对输出一致，再开始改算法。

实际新增文件：

- `cosmos_framework/inference/toca_future.py`：合并配置、分块 incoming score、CFG 合并、年龄、Top-K、模块缓存与临时 forward 分派，避免拆出多余模块。
- `cosmos_framework/scripts/action_policy_server_robolab_toca_future.py`：显式 ToCa 服务器入口、首请求 Dense/全量 observer 等价验证、真实模块行数 trace、逐请求记录。
- `tests/test_toca_future.py`：CPU 逻辑测试。
- `tools/run_toca_future_smoke.py`：固定十任务、现有环境、无视频、不重试的运行器；只关闭自己启动的进程。
- `docs/toca_future_smoke10_cn.md`：实际命令、参数、结果与证据范围。

未修改原模型、attention kernel 或原 Dense 服务器文件；只有显式使用 ToCa 入口才安装 controller，退出 context 后恢复原 forward。ToCa 工作树源代码通过 `PYTHONPATH` 使用，现有 ASI/Dense 工作树不改动。

最小模型接口只负责在上述 attention/MLP 边界分派。ToCa 和 ASI 同时启用应直接报错，不能静默叠加。

```text
新 request / chunk
  │  清空上一 chunk 的有效 cache；读取真实 layout、seed、sampler 参数
  ▼
每个 denoise step（0–3）
  ├─ conditional 分支：当前完整 latent → embedding
  │    └─ B0…B27
  │         D：完整 attention → cache/add → 完整 MLP → cache/add
  │         C：完整当前 K/V；L0/action attention 重算、future 输出取缓存 → add
  │              → L0/action 与选中 future 的 MLP → 拼合缓存输出 → add
  │         每个 block 输出始终为完整原序列
  ├─ unconditional 分支：相同调度和选择索引，独立数值 cache
  ├─ 两分支正常 final norm / heads
  ├─ 原 CFG（包括原配置可选 normalize）
  └─ 原 UniPC 积分，得到下一步完整 video/action latent
最终 action → 原反归一化/客户端；按需 VAE decode future frames
```

实现 CFG 时，full step 两分支都完成后才生成供下一缓存步使用的合并分数；不能在 conditional forward 中用上一周期的 uncond 分数。缓存步可提前从已有 score/age 生成本步两分支共用索引，不需要等待当前 Dense oracle。

RoPE、position ID 和原始 token 坐标不重新编号。保护 query 按原位置 gather，K/V 保持全部有效原位置。MLP 是逐 token 运算，选中行 scatter 回原位置即可。

## 9. 必须通过的验证

### 9.1 CPU / 小张量单元测试

- 候选域、保护域严格分区；固定输入可预测 Top-K，tie-breaking 确定。
- full step 正确初始化两个模块，partial step 更新指定行且不污染其他行。
- cache 存在独立存储，不 alias 当前输入；原地 scatter 不改上一块输入或对照数据。
- step/branch/block gating 正确；共享 CFG mask 不共享 feature，不重复累加年龄。
- 新 request、shape、dtype/device、layout 改变时清空有效状态；异常后不能继续用半写 cache。
- 缓存输出加入当前 residual；设计输入变化案例，排除错误整块 hidden 复用。
- `enabled=False` 和全 `D` 均回到 Dense；**只令 MLP rho=1 不是 Dense**，attention 仍可能缓存。
- 可选空间 bonus 支持 17×20、奇数边界，不选 padding、不打乱位置。

### 9.2 GPU 功能验证

固定 BananaInBowlTask 的第 3 个 request，保存同一 `data_batch` 和初始 noise，逐项比较：

1. 新分支 Dense 与原 Dense 的最终 action、latent，以及关键层输出。
2. 分块 score 对显式参考的误差与 Top-K overlap；真实 GQA、mask、UND 长度均覆盖。
3. Attention 输出按 `A V` 重算，与原 kernel 的实际输出比较；采用 BF16 合理容差，不要求概率逐元素 bitwise 相同。
4. 用 trace 证实缓存步没有先计算全部 future Q/attention/O 再丢弃；Q/O 行数为保护域、K/V 行数为完整 GEN，MLP 行数为 `|P|+K_l`。
5. L0/action 的 attention 和 MLP 每层均真实重算，不读其旧模块输出；当前 attention 的 key 集合完整。
6. 最终 heads、CFG、UniPC tensor shape 不变；所有中间和最终张量检查 NaN/Inf。

再做同一 request 配对推理，比较 action MSE、cosine、relative-L2、逐 horizon/逐维/max absolute error，future latent 和 RGB 的 cosine/relative-L2。分支、每层、实际 t/sigma、刷新率均写入 manifest。

## 10. 公平对比和实验顺序

### 10.1 比较组

主表固定为：`Dense`、`ASI（固定 62a09f7 配置）`、`ToCa-Future`。可以给 ToCa-Future 多个速度—成功率工作点，不用“固定相同 token 比例”冒充同等计算开销。不实现或加入 Joint 对照。

先统一 eager、同 attention backend 完成方法对照。之后加系统优化表，分别给各方法可用的 compile/graph 优化，逐行公开开关；历史 Dense eager vs ASI compile+graph 表不得当成本轮同后端结果。

统一 checkpoint、精度、4 steps、shift=5、guidance=3、输入处理、UND/prefill/offload 设置、action horizon/replan、仿真初始状态、step cap 与 seed 序列。服务器默认值依据 `cosmos_framework/scripts/action_policy_server_robolab.py:369`、`:376`：UniPC、policy seed=0、deterministic_seed=False、4 steps、shift=5。仿真 seed 不凭文档名称推断，由最终评测 manifest 显式指定。

### 10.2 配对离线与闭环各自回答什么

**离线配对**：来自 Dense 轨迹的同一请求分别运行各策略，深拷贝输入，重置 sampler、RNG、模块 cache，使用完全相同的初始 noise。它测量动作/视频误差传播。

**闭环**：同一个 task×rollout 的各策略从相同环境初态起跑，随后各自接收自己的观测。它测量任务表现。轨迹分叉后，不能把不同观测上的第 n 个 chunk action MSE 当作纯模型近似误差。

`deterministic_seed=False` 时，比较使用相同请求序号对应的预设噪声 seed 流，并记录每个实际 request seed；不能只是设置一个 server seed 后跨任务任意续接 RNG。失败不重试到成功，不筛除超时或异常。

### 10.3 分阶段规模

1. 一个配对 request 完成上述功能验证，不依赖闭环成功证明缓存正确。
2. 三个预先指定的简单任务用于筛选参数，各策略同配置同初态；记录失败，避免一次失败即宣布 ToCa 不适用。
3. 固定配置后进行原有 10-task smoke，并增加简单/中等/复杂的成对测试。校准任务不作为独立泛化证据。
4. 正式 benchmark 使用核验后的完整 120-task manifest，每任务 10 rollouts，即每策略 1200 episodes；三组共 3600 episodes。参数不再按测试结果调整。规模和任务清单需用户批准后执行。

同速对比的工作点在校准集选择：ToCa 和 ASI 的 warmed median chunk time 尽量落在 ±5% 范围；若无法匹配，则报告完整 Pareto 点，不删去 ToCa 更快但稍降成功率、或更慢但更稳的点。

### 10.4 计时口径

主指标是同步后的 warmed **`generate_samples_from_batch` 单 chunk wall time**，明确入口/出口，包含所有 4 steps、CFG、评分、选择和 cache 更新。先预热算子/编译，再交替随机顺序测至少 30 次相同配对请求。

每次计时必须从空的 request-local ToCa 有效 cache 开始，full refresh 也在计时内部；不能提前替 ToCa 跑 step 0，再只测后面缓存步。只复用显存缓冲区分配是允许的，但要给 Dense/ASI 相同机会。

记录 mean、median、P90、CUDA peak allocated/reserved；同时拆出 full forward、cached forward、score、gather/scatter、MLP、attention 时间。VAE decode、RPC、仿真物理耗时和整个 episode wall time另列，不混成“chunk 加速”。

compile/graph 首次 capture/compile 单列；静态 cache buffer 和固定 `K_l` 有利于后续优化，但首版不承诺全图无 graph break，也不把多编译分支成本隐藏。

### 10.5 闭环和质量指标

- Success：真实 `success` 字段；Score 单独统计，不能代替 success。
- 完成步数：各策略成功样本平均/中位数，加上双方都成功的配对子集；所有任务另给含 timeout cap 的步数，避免成功样本难度不同造成误读。
- 单 chunk median/P90 及相对 Dense 的同轮次 speedup。
- 配对 action：MSE、cosine、Δ-action cosine、Jerk cosine；后两项明确差分阶数、单位、归一化前后和是否包含 gripper。零范数结果标记而非硬填为 1。
- 配对 video：latent/RGB cosine 和 relative-L2；必要时 PSNR/SSIM。
- 按 task 及难度分层给成功率、成对胜负数和不确定区间；1200 episodes 不能忽略同 task rollout 的相关性。

## 11. 理论开销只能给边界，不能代替实测

设一次 Dense GEN forward 成本为 `T_D`，缓存 forward 为 `T_C`，两分支先假设相同成本。周期 `D,C,D,C` 的 Transformer 部分约为：

\[
S_{transformer}=\frac{8T_D}{4T_D+4T_C}
=\frac{2}{1+T_C/T_D}.
\]

这是理想化分解，实际 cond/uncond 分开计时，并加入评分成本、固定头部、prefill、solver 和内存访问。不能因只算 10% MLP token 就报 10× 端到端加速。

ToCa-Future 的 `T_C` 必须包含**完整 GEN 的 K/V projection**、保护 query 的 attention 与 O projection、保护 token 和选中 future 的 MLP，以及 cache 读写和 residual。不能套用“整个 attention 输出均复用”的成本公式；当前方案明确不缓存 action attention 输出。

粗略 FLOPs 应按实际 GQA Q/K/V 宽度、实际 MLP 模块和有效 UND 长度计算。不能照抄 DiT 的等宽 Q/K/V 或默认 4D FFN，也不能把 MLP token saving 当成整个 block saving。

## 12. 输出和验收

批准实施后，实验数据放新 worktree 的 `experiments/toca_baseline/<run_id>/` 并 gitignore；只提交代码、测试、manifest 模板和文档。建议包含：

```text
manifest.json                 # Edge/ToCa commit、权重、env、实际参数/seed
effective_schedule.csv        # step/branch/block 的 D/C 与实际 fresh count
module_compute.csv            # Q/K/V/O/MLP 行数、cache hits、字节数
paired_chunk_metrics.csv
timing_samples.csv
timing_summary.json
episode_results.jsonl
summary_by_task.csv
report_cn.md
```

功能完成标准：关闭开关无影响；全 D 与 Dense 对齐；缓存语义、CFG 和完整 grid 验证通过；trace 证明真实跳算；报告评分和缓存开销；闭环结果无重试筛选。

已确认的范围：**仅实现 ToCa-Future，不实现 Joint，不缓存 L0/action 的模块输出。** 用户随后要求直接进行十任务 smoke，本轮因此固定一个工作点，增加首请求配对验证并运行十个任务。尚未执行 Dense/ASI 的同轮十任务闭环或 warmed 配对速度评测；不能将本轮结果表述成三策略正式 benchmark。
