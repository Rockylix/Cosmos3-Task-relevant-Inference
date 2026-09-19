# ASI：评分机制与 Efficient Online Selection and Sparse Execution

更新日期：2026-09-19。本文面向论文 Method 的 **Efficient Online Selection and Sparse Execution** 小节，汇总已完成的系统消融、编译评估和 Dense 对照 profile。本文归档既有实验，不将历史数据表述为新一轮测试；明确区分生产默认、实验实现和未实现设计。

## 0. 版本与结论

生产接口所在仓库：`/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted`。

- 分支：`version/core80-stable104-action-weighted`。
- HEAD：`87caf7454e203cb25ff768ac0a86d56d2dea10d6`。
- 主算法：每 chunk step0 conditional 全量采集 28 层，Top6 来源层、Core80、Stable104、每帧 K184；余下七次 CFG forward 使用同一 mask。
- 这里的“profile”指在线采集选择统计，不是额外运行一次离线模型，也不是 Nsight 性能分析。
- 普通 ASI 与 ASI + velocity cache 必须区分；后者不是当前普通 ASI server 的默认功能。
- 此文件随实验分支归档；生产分支的推理默认参数不变。

实验实现位于 `/root/robolab/worktrees/asi-system-ablation`，分支 `experiment/asi-system-ablation`，基础 HEAD 与上面相同，本次将实验源码与文档一同提交；具体测量以结果文件的源码 SHA256 为准。**B6 主 LSE 复用与本轮编译阶梯没有合并为生产默认。** 不能在生产 checkout 只切 `--asi-execution` 就宣称复现了 B6。

系统主线：**借用正常首轮条件去噪提取最小统计；在请求内生成固定预算执行计划；后续 forward 复用原始索引并持续处理紧凑序列。** 张量主要驻留 GPU，但当前 planner 仍有 host 读取和同步，不能写成“选择已经完全 GPU-resident、无同步”。

| 层次 | 已验证的内容 | 不能混淆的边界 |
|---|---|---|
| ASI 选择/执行设计 | 正常首轮采集、一次选择、固定预算、整栈紧凑执行 | 与评分具体融合实现分开 |
| 已有 eager 系统优化 B1–B5 | 八帧批处理、连续布局准备、验证延后、metadata/索引复用 | 在固定chunk两seed上输出与legacy精确一致，不是全输入保证 |
| B6 / P0 实验实现 | 主 attention LSE复用，额外32-query attention从28次降到0 | scorer仍有局部QK与显式GQA展开；未成为生产默认 |
| compile / Graph评估 | decoder及评分张量计算可编译，Graph真实replay已确认 | 数值与mask可变化；通用模型融合不属于ASI独有算法贡献 |
| 待实现 | 无显式KV-head展开的局部融合scorer、无host读取的planner | 不能写成当前已实现的贡献 |

更新后的优化判断：B6评分GPU时间约3.4 ms，已不是主要GPU瓶颈。Dense本身的小算子约172.3 ms，ASI约113.3 ms；后者评分函数内的小算子仅约3.2 ms。下一步系统优化必须依据完整流水线，而非假设所有小算子来自评分。

## 1. 论文 Method 建议分为两节

### 1.1 Action-Relevant Future Token Selection

解释“选择谁”：动作时间对齐、层质量、逐帧 Core、共享 Stable，以及固定预算约束。该部分定义算法，不混入 compile、Graph 或 Python 开销。

### 1.2 Efficient Online Selection and Sparse Execution

解释“怎样低成本选择和执行”：在线最小统计提取、归一化统计复用、八帧批处理、请求内执行计划、固定尺寸稀疏整栈执行。可按“Online statistics → Execution plan → Compact execution”组织，不重复上一节Core/Stable的全部算法推导。

compile/CUDA Graph 放在实现细节或系统消融中，不应单独包装成新的token选择算法。已有编译器产生的算子融合，与尚未实现的专用单kernel/GQA原生scorer也要分开。

## 2. 当前评分机制

### 2.1 符号与实际布局

| 符号 | 当前 Edge 含义 |
|---|---|
| \(l\) | Transformer block，0–27 |
| \(f\) | future latent，L1–L8 |
| \(p\) | 每帧 17×20 网格中的位置，\(P=340\) |
| \(a\) | 32 个预测 action horizon；不含 condition action |
| \(H_q,H_{kv},d_h\) | 16 个 query heads、8 个 KV heads、head dimension 128 |
| \(\kappa(h)\) | 实际 GQA 映射；零基编号下为 \(\lfloor h/2\rfloor\) |
| \(s\) | 模型真实 attention scaling |
| \(\mathcal K\) | action query 实际可见的全部 keys，包括 UND/text、L0、future 和 action |

Q/K 取实际送入 attention 的版本，包含模型所需的 normalization 和 RoPE。UND 有缓存时，应使用 GEN 真正读取的 UND K，而不是误用另一路 K-normalization 或存储 padding。

布局由 packed metadata 解析，不应仅凭 32/8 猜测任意模型的时间对应关系。本实现针对既定 DROID 布局，将四个连续 action horizon 对应一个 future latent；迁移其他动作频率时需重新核对。

### 2.2 Action–future 原始概率

在 step0 conditional 中，对每层定义：

\[
z^{l,h}_{a,j}=s\,(q^{l,h}_a)^\top k^{l,\kappa(h)}_j,
\qquad
A^{l,h}_{a,j}=\frac{\exp(z^{l,h}_{a,j})}{\sum_{u\in\mathcal K}\exp(z^{l,h}_{a,u})}.
\]

分母必须保留所有真实可见 keys。只在当前 future frame 内做 softmax 会移除真实 frame mass 信息，改变后面的层质量与选择，不是等价优化。

用一基 action 编号 \(a(f,r)=4(f-1)+r\)，并取：

\[
\alpha=[1/6,1/3,1/3,1/6].
\]

每层输出的最小聚合统计是：

\[
U_{l,f,p}=\frac1{H_q}\sum_{h=1}^{H_q}\sum_{r=1}^4
\alpha_r A^{l,h}_{a(f,r),(f,p)}.
\]

语义：时间对应的四个动作对该帧各空间位置分配的原始 attention mass，中间两个动作权重更高。先逐 query 正确归一化，再平均 head、按动作权重聚合；不能先平均 Q/K 再 softmax。

该分数是相关性代理，不是动作因果重要性，也没有包含 \(A\|V\|\)、输出方向贡献或 V 的尺度。生产旧scorer读取V，是因为获取LSE的额外attention也计算了输出；B6使用主LSE后，局部评分不再需要读取V。主attention仍正常使用V，选择本身并未变成value-aware。

28 层均采集后：

\[
U\in\mathbb R^{28\times8\times340}.
\]

### 2.3 层质量与 Top6

以下使用与代码一致的分母下界 \(\max(x,\epsilon)\)，\(\epsilon\) 为评分 dtype 的机器 epsilon。

\[
m_{l,f}=\sum_pU_{l,f,p},\qquad
P_{l,f,p}=\frac{U_{l,f,p}}{\max(m_{l,f},\epsilon)}.
\]

\[
E_{l,f}=\frac{-\sum_pP_{l,f,p}\log\max(P_{l,f,p},\epsilon)}{\log P}.
\]

若 \(m_{l,f}\le\epsilon\)，代码将该帧熵设为 1。

\[
R_l=\frac18\sum_fm_{l,f},\qquad
H_l=\frac18\sum_fE_{l,f},\qquad
Q_l=R_l\max(1-H_l,0).
\]

\[
\mathcal B=\operatorname{TopK}_l(Q_l,6).
\]

解释：R 表示读取强度，1−H 表示空间集中度。注意当前 R 是“各帧对应 action 读取该帧的平均 mass”，不是“32 个 action 对所有 future keys 的总 mass”；Q 也不是 \(\operatorname{Mean}_f[m_{l,f}(1-E_{l,f})]\)，两者不可互换。

Top6 只限制 Core 来源层，不限制实际执行的 Transformer 层，也不限制 Stable 的 28 层输入。

### 2.4 Core 与 Stable 的分数

Core 使用 Top6 层的 raw profile：

\[
w_l=\frac{Q_l}{\max(\sum_{j\in\mathcal B}Q_j,\epsilon)},\qquad
C_{f,p}=\sum_{l\in\mathcal B}w_lU_{l,f,p}.
\]

Stable 使用全部 28 层：

\[
\bar w_l=\frac{Q_l}{\sum_jQ_j},\qquad
G_{f,p}=\sum_{l=0}^{27}\bar w_lU_{l,f,p}.
\]

当全层 Q 总和不大于 epsilon 时，Stable 的层权重退化为均匀权重。Core 不使用该均匀回退；全零质量会产生全零 Core score，因此并列 Top-K 行为需要单独测试。

\[
\mu_p=\frac18\sum_fG_{f,p},\qquad
\sigma_p=\sqrt{\frac18\sum_f(G_{f,p}-\mu_p)^2},
\]

\[
\mathrm{CV}_p=\frac{\sigma_p}{\max(\mu_p,\epsilon)},\qquad
S_p=\frac{\mu_p}{1+\mathrm{CV}_p}.
\]

Stable 倾向于持续受到关注、跨帧波动较小的位置，但不等价于背景语义分割。它比较同一网格坐标，不包含物体跟踪或光流对齐。

### 2.5 固定预算的真实选取顺序

代码并非两个互不关联的 Top-K：

\[
\mathcal P_{core}=\operatorname{TopK}_p(\max_f C_{f,p},340-104),
\]

\[
\mathcal C_f=\operatorname{TopK}_{p\in\mathcal P_{core}}(C_{f,p},80),
\]

\[
\mathcal S=\operatorname{TopK}_{p\notin\cup_f\mathcal C_f}(S_p,104),
\qquad \mathcal M_f=\mathcal C_f\cup\mathcal S.
\]

先用大小为 236 的共享候选池约束 Core，保证 Core 跨帧并集的补集中至少还有 104 个位置；然后才选择不与任何帧 Core 重叠的 Stable。每帧严格保留 184 个位置。

Core 可逐帧不同；Stable 跨帧共享。当前无 Adaptive、无额外 Fill、无逐层/逐 step 收窄。每 chunk 选一次，后续全部层及两个 CFG 分支使用该 mask；下一 chunk 重新选择。

## 3. 当前代码怎样得到评分

### 3.1 执行模式

| `--asi-execution` | 实际含义 |
|---|---|
| `legacy`，默认 | 原 Controller；首轮逐层 callback，逐 frame 评分；不启用 compile/Graph |
| `optimized-eager` | tensor-only 八帧批量评分，profile 由 decoder metadata 返回，布局/索引缓存；eager 执行 |
| `compile` | 优化 Controller，加 decoder 等模块编译 |
| `compile-graph` | 上述编译路径，并请求 CUDA Graph 优化 |

server 对优化路径设置 `compile_profile_decoder=True`、`compile_profile_kernel=False`：评分在 decoder 中参与编译，不是另启一套独立 compiled scorer。`compile_profile_decoder=True` 本身不代表 optimized-eager 也已开启 torch.compile。

### 3.2 生产路径与已验证的 B6 路径

生产既有 optimized-eager / B5 路径：

```text
真实 Q/K/V → 主 attention → 正常输出 O
       └→ 额外的 32-action-query attention(Q_action, K_all, V_all)
             ├→ 其 O 不使用
             └→ LSE [32,16]
                   + 对应帧的 Q_action/K_future
                   → FP32 局部 QK [8,16,4,340]
                   → exp(logit-LSE)
                   → head 平均、动作加权 → U_l [8,340]
```

实验 B6 路径（已实现与测试）：

```text
真实Q/K/V → 同一主attention → 正常输出O + GEN LSE
                                  └→ GEN-relative action rows [1,32,16]
                                       + 对应帧Q_action/K_future
                                       → FP32局部QK [8,16,4,340]
                                       → exp(logit-LSE)
                                       → head mean + action加权 → U_l [8,340]
```

B6删除了每层额外32-query attention及其评分侧全K/V拼接。当前Flash2 wrapper本来已产生LSE，改动是将已有返回结果通过dispatcher传至scorer，而不是重写attention内核。API先返回GEN所有query的LSE，再立即切action rows；不能声称底层只分配2 KiB的action LSE。

八帧批量化消除了scorer逐帧Python循环。B6仍有FP32准备、显式GQA `repeat_interleave`、局部QK及概率中间量。编译会融合其中一些周边算子，但没有完成原生GQA直接寻址、dot到最终U全部合并的专用scorer。

不能写“评分完全免费”或“已直接复用主 attention 的概率”。当前没有增加完整模型 forward，但增加了小规模统计计算。

### 3.3 输入输出与存储边界

| 阶段 | 输入 | 最小输出 |
|---|---|---|
| 评分 | 实际 action Q、future K、全 key 归一化量、GQA/时间映射 | 每层 `[8,340]` 的 U |
| 选择 | `[28,8,340]` 的 U、固定预算 | `[8,340]` mask，固定数量原始索引 |
| 稀疏执行 | 当前 hidden、原始位置对应的 RoPE、索引、metadata | 保留 token 的更新 hidden |
| 输出衔接 | 稀疏 hidden、本次 stack 输入的旁路副本 | 完整布局 hidden → 原生输出头 |

部署不必长期保存完整 attention、逐 head 热图或各层 Q/K/V。U 共 76,160 个 FP32 数，304,640 bytes，约 0.29 MiB；action-row LSE 每层仅 512 个数，FP32 约 2 KiB。这是逻辑张量大小，不是 allocator 峰值。

保留小型 U 张量有利于调试，没必要优先为省这 0.29 MiB 引入复杂流式 Top6。应先消除重复算子、同步和大中间量搬运。

## 4. 系统机制：已完成与后续候选

### P0 / B6：主 attention 导出 LSE（实验分支已验证）

逻辑接口（实现先返回GEN LSE再截取action rows）：

\[
\operatorname{Attention}(Q,K,V)\to(O,\mathrm{LSE}_{action}),
\qquad
\mathrm{LSE}_{a,h}=\log\sum_{j\in\mathcal K}\exp(z_{a,j,h}).
\]

主attention原本就需要softmax归一化。当前已验证的Flash2路径返回其自然对数LSE；scorer不再调用 `attention(Q_action,K_all,V_all)`，也不再为评分计算O或读取V。未推广到所有attention后端。

保留的局部重算为：

\[
U_{l,f,p}=\frac1{16}\sum_h\sum_{r=1}^4\alpha_r
\exp\big(z^{l,h}_{a(f,r),(f,p)}-\mathrm{LSE}^{l,h}_{a(f,r)}\big).
\]

只重算对应 frame 的 logits，不能将 softmax 分母缩成局部 frame。主 attention API 可能返回所有 query 的 LSE 再抽取 action rows，是否值得修改底层 kernel 必须测量，不能假设返回 LSE 没有任何开销。

需要核对：LSE 是自然对数还是 log2，是否包含 scaling/其他 logit 变换，返回维度与 padding、真实 UND 长度、cached UND K、GQA head 顺序。不能混用不同分支或不同层的归一化量。

后端接口若因 `return_lse` 切换到不同算法或低效路径，收益可能被抵消。也不能从 O 或 V 反推 LSE。

本次实现通过 `return_gen_lse` 默认关闭开关、text-KV dispatcher显式传递和SequencePack临时私有键完成。私有键立即pop；以GEN内部action起点切片，不加UND偏移。控制器退出/异常时恢复开关。不支持的训练、分片、自定义parallel、three-way/multi-control/NATTEN路径明确拒绝，不静默使用错误分母。

实测：额外attention调用28→0，评分GPU kernel616→504，4.332→3.405 ms；同轮chunk median548.663→548.272 ms，仅约0.391 ms，不能称显著整体加速。两seed的Top6/Core/Stable/mask及action/future latent与legacy精确一致；U最大差约6.34e-8。此验证范围不是所有输入的数学精确保证。

### P1：局部评分融合与原生 GQA 读取（专用kernel尚未实现）

目标小型 kernel 只消费 `Q_action`、`K_future`、`LSE_action`、geometry 和权重，输出 U；不新增投影。

1. 利用连续布局/固定 stride 批量读取 8 帧，避免逐层索引上传。
2. 按 \(\kappa(h)\) 直接读取共享 KV head，避免将全部 future K 复制为 16 heads。
3. 在寄存器/片上完成 dot、归一化、head/action reduction，尽量不把 `[8,16,4,340]` logits/probabilities 写入 HBM。
4. 每层写入独立 U 缓冲槽，避免 Graph 后续 replay 覆盖上一层记录。

这是设计目标，不保证一个大融合 kernel 一定快于当前小矩阵乘法。需要检查寄存器压力、occupancy、实际 kernel 数和评分 wall time。也不建议一开始重写整个 FlashAttention 来输出完整权重；优先采用主 LSE + 小型局部 scorer。

### P2：GPU 驻留固定预算选择

当前一次性 planner 仍有 Python 循环、`float/bool/int/.tolist()` 读取以及 `nonzero` 候选构造。未来可将 U→R/H/Q→Top6→Core/Stable→索引串为 tensor-only 路径：

- 保留同一公式、预算、Core 候选池和回退逻辑。
- 将不允许位置赋为 `-inf` 后做固定 K 的 Top-K，避免动态长度候选输出；必须先保证合法候选数量足够。
- 固定输出 `[8,184]` 位置索引，按原始位置排序后拼接 L0 和全部 action。
- 生产 hot path 不将 Top6 IDs、验证标量或 token 数回传 CPU；异步/运行外记录诊断信息，不直接删除必要安全检查。
- 并列 Top-K 和浮点 reduction 顺序必须建立统一策略；悄悄加 position bias 或 epsilon 打破并列不是无损实现替换。

P2收益需单独测量。已测planner/select的GPU kernel约0.248 ms，但host NVTX约26.9 ms，后者含前序GPU排队等待，不能全部当作可消除的CPU选择计算。不可按这个host数值预测GPU planner能省27 ms。

### P3：执行计划复用与 Graph 兼容

当前已有：请求内 selected indices 缓存、按真实 UND 长度/稀疏 GEN 长度/device 缓存 metadata、对同一 packed 对象缓存布局。RoPE 仍在每个 sparse stack 入口用索引切片，不能声称所有 gather 已只执行一次。

后续可探索：复用固定地址索引/输入/输出缓冲、对不变 RoPE 切片缓存、将 gather/restore 融合到适合的边界。hidden 每个 step 都变化，不能复用旧 hidden；CFG 分支文本长度/内容也不能误复用。

固定 token 数允许 mask 内容变化而不改变计算尺寸，但 Graph replay 仍需保证：索引作为 tensor 输入每请求更新、地址生命周期正确、无捕获旧 mask、各分支 metadata 正确、输出缓冲未被下一次 replay 覆盖。

当前实现是 decoder 级编译接线，不是整个四步 UniPC 的单一 Global Graph。不能仅凭参数开启就宣称所有算子 replay；需实测 `cudaGraphLaunch` 及 capture/replay 事件。Graph 收益也不能以减少 kernel launch 数直接替代端到端计时。

本轮在padding全程关闭时实测：264次GraphLaunch，其中224次分别落在全部step/branch/block，包含首次conditional的28层；另40次在layer范围外，未逐head精确归属。是分模块Graph，不是单chunk单图。compile-only与Graph使用不同编译mode，数值差异不能独归因于replay。

### 不应作为等价优化的改动

- 只采 Top6 层：Stable 依赖全部层，首次选 Top6 也需全层质量。
- 将层 ID 固定后声称省掉所有 profile：全局 Stable 和当前层权重仍需要统计。
- 跨 chunk 复用 mask、减少 head/action 采样、改变 layer/step、权重或 K：这些改变算法，需单独消融与用户决策。
- 用 pre-RoPE Q/K、frame-only softmax、先平均 query 后 softmax：改变评分定义。
- Velocity cache：改变被移除区域的去噪近似，不能归为纯执行优化。

## 5. 稀疏执行与去噪衔接

GEN 的逻辑长度：

\[
N_{dense}=340+8\cdot340+33=3093,\qquad
N_{sparse}=340+8\cdot184+33=1845.
\]

还存在实际文本/UND keys，上式不包含它们，也不包含 Graph padding，不能当作 attention 总长度或 FLOPs 比例。

```text
当前观测、指令、动作条件和初始噪声
  → step0 conditional：28 层 dense，正常输出 + U[28,8,340]
  → 一次 Core/Stable 选择，形成每帧 K184 的执行计划
  → step0 unconditional + step1/2/3 两分支，共七次 sparse stack
       Gather 当前 hidden 和原始 RoPE
       → 28 层始终保持紧凑序列（Q/K/V/O、attention、MLP 都变短）
       → stack 末尾 scatter 更新位置；未计算位置使用本 stack 输入 hidden
       → 最终归一化/输出头 → 原生 CFG → UniPC
  → action chunk；需要时 VAE 解码未来帧
```

该数据流不是全量算完再 mask，也不是只稀疏 attention。未选 future tokens 不在中间层重新注入 attention；全长恢复仅发生在 stack 末尾。输出头/积分等仍有全长开销。

每 chunk 一次 dense profile 的成本需要计入；它是正常八次 GEN forward 中的一次，不是额外第九次。UND/text prefill 应在单 chunk 计时中注明是否命中缓存，不能被“八次”表述掩盖。

### 5.1 将选择结果降低为可执行的数据结构

算法输出是 \(\mathcal M_f\)，执行器需要的是以下请求内计划，而不是每层重做一次布尔筛选：

\[
\Pi=\operatorname{Sort}\Big(\Pi_{L0}\cup\Pi_{action}
\cup\{\operatorname{pos}(f,p):p\in\mathcal M_f\}\Big).
\]

这里 \(\Pi\) 是GEN内的原始位置索引，长度1845；UND单独保留，不套用GEN偏移。计划包括GPU索引、mask、geometry和按真实UND长度/device区分的稀疏SequencePack metadata。Top6 ID用于选择与诊断，不是“只执行这6层”的调度表。

缓存作用域必须写清：mask和选择索引在当前chunk七次sparse forward间复用；布局缓存要求同一packed对象；metadata允许同一长度/device模板复用，但当前hidden、当前分支UND张量始终重新绑定。下一chunk重新采集与选择。不能把执行计划复用写成跨chunk的模型特征缓存。

### 5.2 紧凑整栈与恢复的数学接口

对一次sparse forward，令 \(H^0\in\mathbb R^{3093\times d}\) 为该次GEN stack输入，\(G_\Pi\) 为按索引提取行的算子：

\[
\widetilde H^0=G_\Pi H^0,\qquad
\widetilde H^{l+1}=B_l^{\Pi}(\widetilde H^l;\mathrm{UND},G_\Pi\mathrm{RoPE}),
\quad l=0,\ldots,27.
\]

全部28层持续处理1845行GEN；每层投影、attention和MLP均缩短，不只是attention mask置零。位置编码读取原始坐标的cos/sin，不能按压缩后的行号重编号。UND/text按原有路径计算或使用其合法缓存，未因未来帧选择而删掉。

stack边界恢复为：

\[
H^{out}=H^0+G_\Pi^\top\big(\widetilde H^{28}-G_\Pi H^0\big).
\]

该式表达scatter语义，实现是保存本stack输入旁路副本，再覆盖被选择行；不是要求额外实际执行式中的减法/加法。未选行保持本stack输入hidden，不是从上一chunk缓存，不是L1 residual插值，也不是置零；随后恢复完整布局进入最终norm、vision/action head、CFG、UniPC。

**这是ASI的近似边界，不是Dense等价变换。** 被选query的可见future keys也会减少，未选位置没有经历这一栈的28层更新；恢复形状不等于恢复Dense计算。固定mask可避免本chunk层间选择跳变，但不能证明所有未来帧语义/运动严格对齐。

### 5.3 精简伪代码

```text
for each chunk:
    initialize request-local noise, solver state and profile storage
    for each denoise step s:
        for b in [conditional, unconditional]:
            construct current packed input and native position metadata
            if s == 0 and b == conditional:
                run all 28 dense blocks once as part of normal generation
                collect U_l from actual post-RoPE Q/K and full-key LSE
                build Core/Stable mask and fixed-budget execution plan
            else:
                keep current full GEN hidden as a side buffer
                gather selected GEN rows and original RoPE using the plan
                run all 28 blocks on compact GEN plus unchanged UND pathway
                scatter selected outputs into the current side buffer
            apply normal final norm and prediction heads
        combine the two predictions with native CFG, then update with UniPC
    release request-local selection/cache state
```

伪代码描述B6逻辑；实际实现可能在第一轮profile结束与第一次sparse入口分别建立mask、索引，不能误读为所有gather/metadata/索引操作都在一个全局Graph中一次完成。

### 5.4 可选velocity cache：独立算法扩展

可选 ASI + velocity cache 在原生 CFG 后对被排除坐标使用首步缓存：

\[
\widetilde v_s=M\odot v_s^{ASI}+(1-M)\odot v_0^{ASI},\quad s>0.
\]

M 需按真实 patch 足迹扩展至 latent 并裁去 padding；L0/action 不替换。在 1 dense + 7 sparse 版本中，\(v_0^{ASI}\) 来自 dense conditional 与 sparse unconditional 的 CFG，不是全 Dense 速度。缓存是请求内的，不是跨 chunk 历史速度。

## 6. 已实现与待验证清单

| 设计 | 当前状态 |
|---|---|
| 正常 step0 conditional 提供全层 profile，不额外跑整模型 | 已实现 |
| 32 action rows、真实全 key 分母、时间加权 | 已实现 |
| 八帧批量评分、scorer 内移除逐层 host 读取 | 优化模式已实现，默认仍 legacy |
| profile 返回 decoder metadata、编译接线 | 已实现；不等于单 kernel 融合或整 chunk Graph |
| 固定 K184、GPU selected indices 请求内缓存 | 已实现 |
| 28 层紧凑序列、metadata 缓存 | 已实现 |
| 复用主 attention LSE，删除额外 attention | B6实验分支已实现，当前固定chunk两seed验证；未合并生产默认 |
| 编译器融合模型及评分周边算子 | 已测；不是专用单kernel scorer，且有数值差异 |
| CUDA Graph真实replay | 已测264次，224个decoder层范围全覆盖；非全chunk单图 |
| 无显式 KV-head 展开、dot到U的专用局部融合 | 尚未实现/验证 |
| planner 无 host 同步、固定尺寸 GPU 输出 | 待实现/验证 |
| 所有不变 RoPE 及固定地址 Graph buffer 的更大范围复用 | 待验证，不能归为当前已有 |

## 7. 已完成证据与下一步验证

以下数值来自2026-09-18至19日已保存的实验，本次未重测。全部使用RTX4090、同一Banana chunk3、4步/shift5/guidance3。稳定耗时每模式预热5次、20次正反序交替；nsys另起进程只采1个预热chunk。正式计时为同步的generation，不包含模型加载、warmup、输入深拷贝/RNG重置、CPU结果转存、VAE decode、RPC和仿真。

### 7.0 已完成系统消融

**A. 已有优化累积消融（同一轮，以ASI legacy为分母，不是Dense）：**

| 累积版本 | 本行新增机制 | Median ms | 对本轮legacy |
|---|---|---:|---:|
| B0 | ASI legacy | 583.972 | 1.000× |
| B1 | 八帧评分算术批处理 | 575.635 | 1.014× |
| B2 | 连续slice/reshape准备 + geometry/layout复用 | 557.362 | 1.048× |
| B3 | 逐层有限值检查延后至保留的planner检查 | 555.907 | 1.050× |
| B4 | 稀疏metadata缓存 | 554.005 | 1.054× |
| B5 | decoder metadata返回profile | 552.956 | 1.056× |

B2不只是Python字典缓存，包含评分输入准备方式变化；不能把全部改善归因一个lookup。两seed的选择与最终action/latent均与legacy精确一致；这是有限样本验证。另起的P0轮次B0/B5/B6 median为578.908/548.663/548.272 ms，必须使用该轮自己的分母；不能把本表与P0表拼接成一条严格配对计时。

**B. 编译阶梯（同轮reference是B6 ASI eager）：**

| 配置 | Median ms | 相对B6 eager |
|---|---:|---:|
| B6 eager | 553.528 | 1.000× |
| 仅后续7次sparse decoder compile | 486.495 | 1.138× |
| 全8次decoder compile | 466.034 | 1.188× |
| 再编译encode/decode heads | 465.095 | 1.190× |
| 再启用Graph | 464.336 | 1.192× |

全程关闭Graph padding。最后Graph相对compile-only仅0.760 ms（0.163%）改善，不声称稳定显著收益。两seed中，sparse-only compile的profile/mask与eager相同，但action relative-L2约1.28%–1.30%；首轮也编译时执行mask XOR为60–76，future latent relative-L2约26.1%–30.6%。相同模式重复、切seed再返回原seed验证通过，不意味着跨模式输出等价。没有这轮编译配置的闭环成功率证据。

**C. Dense对照与评分直接归因（单次nsys GPU时间，不是正式chunk median）：**

| GPU工作类别 | 原生Dense eager | B6 ASI eager |
|---|---:|---:|
| GEMM/GEMV相关 | 453.035 ms | 297.102 ms |
| Attention | 147.942 ms | 59.691 ms |
| 卷积 | 19.298 ms | 19.303 ms |
| 归一化/逐元素/转换/布局等小算子 | 172.347 ms | 113.301 ms |
| 全部kernel合计 | 792.621 ms | 489.396 ms |

Dense来自独立Baseline checkout，不含ASI controller。ASI score函数的504个kernel仅3.401 ms，其中小算子3.203 ms，占ASI全部小算子约2.83%；局部FP32 QK为0.198 ms。select范围GPU约0.248 ms。score范围不含主LSE导出的增量、外部clone、packing/restore等，因此不能宣称ASI全部附加成本只有3.4 ms。

**D. compile收益归因：** 另一个已保存trace中，ASI eager→all compile的小算子113.404→26.765 ms，总kernel数20,756→5,824；GEMM约297 ms、attention约60 ms基本不变。主要是RMSNorm、Q/K Norm与RoPE周边、ReLU²、residual以及布局/类型转换融合，并非“评分引入的开销被消除了”。本Edge MLP是ReLU²，不是SwiGLU。全量读写下降是生成代码支持的机制解释，未测NCU内存字节计数，不能写成定量HBM节省。

论文应分列 `Dense eager vs ASI eager` 与 `Dense compile vs ASI compile`；后者Dense compile尚未在这轮测量。不得将通用compile的1.190×全部归为ASI算法特有收益，或把不同轮次比值相乘当作一次实测。

### 7.1 计算正确性

先单独固定捕获的同一 Q/K/V 和布局比较 scorer，再比较完整推理，避免把 decoder 编译数值变化误归因于评分优化。

| 层级 | 核对内容 |
|---|---|
| Attention 接口 | Q/K normalization、RoPE、GQA、实际 mask/key 长度、LSE 底数与 shape |
| 评分 | 每层 U 的 max-abs/relative-L2、NaN/Inf、R/H/Q、零 mass/全零 Q 回退 |
| 选择 | Top6、逐帧 Core、共享 Stable、mask XOR、每帧恰好 184、L0/action 保留 |
| 稀疏执行 | 每层真实输入长度、原始 position ID、没有层间误注入、全保留回归 |
| 输出 | 同输入同 seed 的 action MSE/cos/max-abs、future latent 误差；不把 action 单位统一称作弧度 |
| Graph | 交替不同输入/seed/mask、不同 CFG 文本长度，排除旧索引和旧输出复用 |

“数学公式不变”不意味着低精度下逐位一致。局部 QK 当前使用 FP32，主 attention 内部的累加和 LSE 路径可能不同；优化后即使 U 很接近，Top-K 边界也可能翻转。应同时报告连续分数误差和离散选择变化。

### 7.2 性能

已有结果见§7.0；后续候选为Dense compile对照、专用融合scorer、GPU planner和更广输入/闭环质量验证，均待另行执行。compile和Graph保持独立维度，避免同时改算法与执行后无法归因。

每组同 GPU、同输入/seed、同模型与参数、同 UND cache 状态，充分预热后交替测量 `generate_samples_from_batch`，计时前后 CUDA synchronize，报告 mean/median/P90。选择与 gather/restore 开销必须包含；编译/capture 冷启动单列，VAE decode、RPC 和仿真不混入主要 chunk 指标。

Nsight/NVTX 建议范围：`step0/cond/profile`、`score/lse`、`score/local_reduce`、`select/quality`、`select/core_stable`、`pack`、`sparse_stack`、`restore`、`head`、`cfg`、`solver`。另查 CPU-GPU 同步、内存复制、实际 replay、峰值显存。详细 profiler 采集与稳定 latency benchmark 分开运行。

可用的时间分解是：

\[
T_{chunk}=T_{prepare}+T_{dense}+T_{score}+T_{select}
+\sum_{i=1}^7T_{sparse,i}+T_{pack/restore}+T_{head/CFG/solver}.
\]

其中各项按不重叠范围归因；score 编译/融合进 dense 后应通过消融判断增量，不能重复加总有包含关系的 NVTX 范围。没有实测前不预测固定毫秒收益，也不把 token 保留比例当加速比。

## 8. 可用于论文的组织与文字草稿

### Token selection（当前已实现）

我们利用每个 chunk 首次条件去噪中的动作—未来视觉注意力构造时间对齐的空间相关性。每个 future latent 聚合对应四个动作查询，并对中央两个 horizon 赋予更大权重。通过联合考虑读取强度与空间集中度的层质量分数，选择来源层构造逐帧 Core；同时聚合全层信息，以跨帧均值和变异系数选择共享 Stable。带候选池约束的互斥选择保证每帧相同预算，所得到的 mask 在当前 chunk 后续各层和去噪分支中固定使用。

### System design（对应B6实验实现，不暗示已合并生产）

我们将在线选择嵌入每个chunk正常的首次条件去噪，不增加额外模型forward。通过主attention返回的全key归一化统计，仅对时间对应的action–future位置批量重算局部logits，并输出每层聚合空间分数，避免导出完整attention矩阵及重复执行仅为取得LSE的attention。之后每chunk只构建一次固定预算计划，复用原始索引和适用的稀疏metadata；后续七次forward在整个Transformer栈中持续处理紧凑序列。原始RoPE位置与L0/action保持，未选择位置仅在stack输出边界从本次输入旁路恢复，然后使用原生输出头、CFG和UniPC。编译和Graph作为通用执行配置单独消融；不能将其浮点差异忽略为严格无损。

### 英文草稿：Efficient Online Selection and Sparse Execution

以下文本描述**B6已测试路径**。最终论文须确保实际发布/评测代码确实启用该路径；如果仍使用生产B5，应删除主LSE复用已实现的句子，改为对应实现描述。

**Online statistics from the normal denoising pass.** We obtain the statistics required for token selection from the first conditional denoising pass of each inference chunk, without introducing an additional network forward. Rather than materializing the full attention matrix, we reuse the log-sum-exp normalization statistics returned by the main attention operation. For each future frame, we recompute only the logits between its temporally aligned action queries and its spatial keys, while retaining the normalization over all visible keys. The computation uses the model's post-normalization, post-RoPE queries and keys and its original grouped-query head mapping. We batch the computation across future frames and aggregate the head and action dimensions into a compact score tensor \(U\in\mathbb{R}^{L\times F\times P}\). In our Edge configuration, this tensor contains 76,160 FP32 values, approximately 0.29 MiB. Reusing the main normalization statistics removes the auxiliary attention evaluation previously used only to recover the softmax denominator; the local relevance computation remains an explicit, nonzero cost.

**A request-local, fixed-budget execution plan.** The Core/Stable rule defined in the preceding section is evaluated once per chunk. Its masks are lowered to original-position GPU indices and compatible sparse packing metadata, which are reused across the remaining denoising forwards. A fixed per-frame budget maps irregular spatial selections to a regular compact sequence length. We retain all condition-frame and action tokens and leave the text/understanding pathway intact. In the evaluated configuration, the GEN sequence is reduced from 3,093 to 1,845 tokens for seven of the eight forwards; the first conditional forward remains dense. These lengths exclude text/understanding tokens and must not be interpreted as total attention lengths or a FLOP ratio. The plan is rebuilt for the next chunk, rather than reusing historical masks or model features.

**Compact execution across the Transformer stack.** At each sparse-forward boundary, we gather the current hidden states and their original positional encodings. All Transformer blocks then operate on the compact sequence, reducing the effective inputs to the projections, attention, and MLP instead of masking tokens after dense computation. We preserve a side buffer of the current full-length stack input and scatter the computed rows back only after the final block. Unselected rows retain their stack-input hidden states before the standard output heads, CFG combination, and solver update. This boundary restoration preserves the tensor interface to the denoising pipeline, but does not reproduce dense computation at omitted positions. We cache selection indices and eligible metadata, not changing hidden states, and retain the original temporal and spatial position IDs.

**Implementation and evaluation boundary.** We implement batched score extraction, request-local layout/metadata reuse, and main-attention LSE propagation. The current planner retains host-side validation and some scalar reads; a fully synchronization-free GPU planner and a custom scorer without explicit KV-head expansion remain future optimizations. We evaluate compilation and CUDA Graph execution separately from the selection algorithm. Profiling attributes their principal benefit to fusion of common model operations rather than an ASI-specific change in token selection. Since compilation can affect both floating-point predictions and discrete selection boundaries, performance results are accompanied by numerical comparisons rather than an assumption of exact equivalence.

### 系统设计贡献应该怎样表述

| 建议主文强调 | 可以实证支持的点 | 不要扩写成 |
|---|---|---|
| Online statistics without an extra network pass | 正常step0 conditional提供统计；B6复用全key LSE | 免费评分、没有任何局部重算 |
| Minimal statistics for selection | 只保留选择算法需要的U，不导出完整attention | 统计学意义上对最终action的充分统计量 |
| Fixed-budget execution plan | 索引/metadata请求内复用，mask内容可不规则但行数固定 | 全planner已无CPU同步、跨chunk缓存完全无误差 |
| Whole-stack compact execution | 投影/attention/MLP真实输入缩短，28层不重新注入未选token | Dense等价、被删token仍经历完整28层计算 |
| Separate generic backend optimization | compile及Graph单独消融，真实replay核验 | 通用compile加速是ASI算法独有贡献 |

主文可以保留上面前三段；第四段及§7数值建议放Implementation Details/System Ablations。不要在Method正文大量堆积特定GPU毫秒数，也不要只展示最快compile配置而省略输出变化。

## 9. 源码定位与相关论文

### 生产接口源码（未合并实验改动的上述 HEAD）

| 内容 | 文件与行号 |
|---|---|
| 常量、参考逐帧 LSE 评分 | [edge_core_stable.py](../cosmos_framework/inference/edge_core_stable.py)：34、112 |
| R/H/Q、Core/Stable、固定预算 | 同文件：178–238 |
| 编译 profile metadata 返回 | 同文件：456–479 |
| 索引与稀疏 metadata 缓存 | 同文件：501–542 |
| 整栈稀疏与末尾恢复 | 同文件：561–624 |
| 八帧批量 scorer、额外 attention | [edge_core_stable_fast.py](../cosmos_framework/inference/edge_core_stable_fast.py)：12–60 |
| 主 attention 与 scorer 的先后关系 | [unified_mot.py](../cosmos_framework/model/generator/mot/unified_mot.py)：732–781 |
| server 模式与默认值 | [action_policy_server_robolab_version1.py](../cosmos_framework/scripts/action_policy_server_robolab_version1.py)：33–100 |
| 已有优化验证及数值边界 | [robolab_asi_optimized_cn.md](robolab_asi_optimized_cn.md)：19–99 |

三个关键文件 SHA256，依上表名称顺序中的 controller、fast scorer、server：

```text
edge_core_stable.py
0d17ba88b0b34509a60b15e0d7e8f379907da25eeec39d2bcf611fff6c88d0d8
edge_core_stable_fast.py
fdf1520c5c00419a15096813d343ae6cf9c09c23841b7033fbc1c1e13819c250
action_policy_server_robolab_version1.py
80182e627f4e7047b0f889e369486ebe532153c3298590ec0ac3662c20ce26eb
```

### 实验源码与证据索引（B6及编译不在生产默认）

本文已归档到实验工作树的 `docs/`；下面链接指向同分支文件。原始测量仍以结果 manifest 的源码 SHA256 为准，不能用提交日期替代测量日期。生产源码表中的行号和 SHA256 对应上述基础 HEAD，而非本实验提交。

| 内容 | 证据 |
|---|---|
| B1–B5逐项累积、B6主LSE实现和安全边界 | [系统消融与LSE报告](asi_system_existing_and_main_lse_cn.md) |
| 编译阶梯、两seed误差、264次GraphLaunch | [编译评估报告](asi_compile_stages_cn.md) |
| 通用小算子融合、GEMM/Attention基本不变 | [compile归因](asi_compile_speedup_attribution_cn.md) |
| 真Dense对照、评分3.401 ms及范围限制 | [Dense与ASI profile](dense_vs_asi_small_ops_cn.md) |
| B6作用域开关与异常恢复 | [asi_main_lse.py](../cosmos_framework/inference/asi_main_lse.py) |
| main LSE回传、GEN-relative action行切片 | [unified_mot.py](../cosmos_framework/model/generator/mot/unified_mot.py) |
| 可选action_lse局部scorer | [edge_core_stable_fast.py](../cosmos_framework/inference/edge_core_stable_fast.py) |

原始结果均在该工作树被Git忽略的 `experiments/`：

- `existing_system_banana_c3_v1/timing/results.json`：B0–B5同轮结果。
- `main_lse_banana_c3_v1/timing/results.json`：B0/B5/B6独立同轮结果、源码/输入hash与误差。
- `compile_stages_banana_c3_v2/timing/results.json`：编译正式计时、两seed误差和return-to-seed核查。
- `compile_stages_banana_c3_v2/trace_all_graph/nsys_audit.json`：Graph运行证据。
- `dense_vs_asi_scoring_banana_c3_v1/{dense,asi}/{results.json,nsys_analysis.json}`：真实checkout来源、token数、评分调用范围及Dense/ASI profile。

当前材料没有新的编译闭环成功率、跨任务性能置信区间、Dense compile对照，也没有专用融合scorer/GPU planner实测。论文最终采用哪种执行配置，应与正式任务评估一致。

### 文献与对应启发

- [FlashAttention，§3.1、Appendix B](https://arxiv.org/html/2205.14135v2)：通过分块、归一化统计和重算避免完整 attention 矩阵的 HBM 读写。启发是重用 LSE、仅重算 ASI 所需概率，而不是为拿权重退回完整显式 attention。
- [ToCa，§3.5、Appendix A.4.2](https://arxiv.org/html/2410.05317v3)：将 token 选择和计算开销一并考虑。其选择开销结论不能直接搬到当前 Edge 后端；额外 scorer、同步及排序必须按实际实现测量。
- [FIS-DiT，§3.3.1](https://arxiv.org/html/2605.11869v1)：通过较短输入复用高效 dense attention kernel。ASI 可以借鉴紧凑执行的系统思路，但 ASI 的空间选择与 FIS 的帧级 anchor/插值并不是同一算法。

一句话：**第一节定义动作相关的计算预算；第二节说明怎样用最少统计和数据搬运，把该预算兑现成真正变短的 Transformer 计算。**
