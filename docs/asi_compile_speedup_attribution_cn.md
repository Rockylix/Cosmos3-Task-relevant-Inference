# 为什么 ASI compile 有明显收益

2026-09-19，只读诊断；未修改模型、推理配置或重新运行GPU。按项目kernel-optimizer skill，仅分析已保存的 `asi.chunk.generate` trace，kernel按同进程、同步root区间筛选，layer按runtime launch correlation归属。基于2026-09-18的固定Banana chunk3实验，不是新增多任务结论。

## 核心结论

**主要是小算子融合与中间张量物化减少，不是大GEMM变快，不是额外删token，也不是CUDA Graph。** 当前ASI eager到decoder+heads compile-only的正式median为553.528→465.095 ms（1.190×，延迟减少16.0%）；Graph只再减少0.760 ms。

## 1. 实际GPU工作量变化

从 `experiments/nsys_reports/compile_stages_v2_{eager,all_compile}.sqlite` 的全量kernel名称分组，单位为GPU kernel时间合计ms。不是各组独立干预实验，也不是profiler-free计时。

| kernel类别 | eager次数 | compile次数 | eager ms | compile ms |
|---|---:|---:|---:|---:|
| GEMM/GEMV及split-K相关 | 2,009 | 2,009 | 296.519 | 297.417 |
| Attention | 309 | 309 | 59.621 | 59.867 |
| cuDNN/卷积相关 | 100 | 100 | 19.317 | 19.436 |
| 其他ATen、逐元素、归约、布局 + Triton融合 | 18,338 | 3,406 | 113.404 | 26.765 |
| 合计 | 20,756 | 5,824 | 488.860 | 403.485 |

最后一组compile由1,118个原生kernel（6.604 ms）及2,288个Triton融合kernel（20.161 ms）组成。该组减少86.639 ms，基本解释GPU总时间减少85.375 ms；其他组略有波动，不应解释为相同数值的独立因果贡献。

分类规则按互斥顺序：名称含`cudnn/xmma_fprop`→卷积；`flash::/fmha_`→attention；`gemm/gemv/splitKreduce/triton_tem_fused`→矩阵类；剩余`triton_`与其他kernel合为最后一组。所有root kernel均被计入。309是chunk内attention kernel数，不是decoder forward数；主GEN FlashAttention仍为224次。

## 2. 融合了什么

### RMSNorm、Q/K归一化

源码 `cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/nemotron_3_dense_vl.py:55`：

`转FP32 → 平方 → mean → 加epsilon → rsqrt → 乘输入 → 乘weight → 转回输入dtype`。

eager把这些操作分开提交，并物化多个中间张量。trace中出现的 `triton_red_fused__to_copy_mean_mul_pow_rsqrt_0` 等将这些操作组合成融合归约kernel；这会减少独立启动及中间结果的全局内存往返。没有采NCU内存计数器，因此这里是根据生成代码解释机制，不是实测HBM字节下降比例。

### RoPE、拼接、类型/布局转换

trace存在含 `cat/mean/mul/pow/rsqrt/slice/unsqueeze/view` 的融合kernel，组合Q/K norm、位置旋转及attention输入准备中的若干操作。名称包含`_flash_attn_forward`不代表FlashAttention的矩阵计算被融合进同一个Triton kernel；对应代码仍单独调用外部FlashAttention。

### MLP非线性和residual

该Edge模型这里是 **ReLU² MLP，不是SwiGLU**：`down_proj(relu(up_proj(x))²)`。trace出现 `triton_poi_fused_pow_relu_4`，融合ReLU和平方，示例生成代码还复用激活buffer；residual加法可吸收到`addmm`。大矩阵计算仍由外部`mm/addmm`执行，因此大GEMM时间基本不变。

### 首轮评分

score的`exp、mean、按action权重求和`也进入融合kernel，但不是整轮加速的主要来源。只编译后续稀疏stack、不编译首轮评分，已经将median从553.528降到486.495 ms，省67.033 ms。首轮dense/profile一起编译再省20.460 ms，该增量同时包含首轮模型层和评分，不能全部称为评分收益。

## 3. 层级交叉验证

同一eager/decoder_compile trace，GPU归属使用launch correlation，不用CPU范围直接截GPU时间：

| 范围 | eager kernels | compile kernels | eager GPU ms | compile GPU ms |
|---|---:|---:|---:|---:|
| 首轮dense/profile全部28层 | 4,340 | 1,036 | 106.608 | 85.899 |
| 后续7个sparse stack全部层 | 14,724 | 3,336 | 354.236 | 289.510 |
| 示例step0 conditional B1 | 155 | 37 | 3.803 | 3.062 |
| 示例step1 conditional B1 | 64 | 14 | 1.777 | 1.456 |

这些层级值不是额外独立的加速来源，不能和第1节重复相加。

## 4. CPU与CUDA Graph为什么不是主因

eager→all_compile的trace wall为554.214→463.384 ms；GPU kernel busy为488.860→403.485 ms；wall减kernel busy仅65.353→59.899 ms。主要变化发生在GPU kernel工作量本身，而不是GPU空闲部分。

CPU kernel-launch API时间确实下降，但与GPU执行重叠，不能把其减少量再加到GPU节省量上。`cudaStreamSynchronize`累计等待反而增加也不代表代码更慢：CPU提交更快后会更早走到等待点。比如`asi.select` host范围26.967→67.114 ms，而自身GPU工作仍约0.25 ms；不能把这个host范围全部叫作选token算法耗时。

compile-only已将大量短kernel融合；剩下GPU时间主要是约297 ms矩阵计算和60 ms attention。Graph复用提交不能去掉这些计算，故264次真实GraphLaunch只带来很小额外收益。不是Graph没生效，也不能简单概括成“kernel总数太少”。

## 5. 证据限制

- 各模式token预算、4 steps、1 dense+7 sparse、padding关闭均相同；编译首轮会造成评分/TopK数值变化，但不是保留token数量减少。
- 同名kernel可对应多个Inductor cache版本；本机cache已包含不同实验和浮点融合设置。下述生成文件只作与trace kernel族及模型形状一致的机制示例，不声称唯一对应本轮加载的二进制。
- 示例 `/tmp/torchinductor_root/ml/cmlpphll2oae7osn3brc5y6rgtz7kxydqxpywsvvqtonf5qznny4.py`：81–124为RMSNorm融合；565–574为ReLU²；688为外部FlashAttention；698/719为residual addmm；709为MLP up projection mm。
- 因为cache版本不唯一，没有逐算子误差捕获，本诊断不能确定上一轮输出差异究竟由哪一次BF16舍入/FMA造成。不能声称编译严格无损。
- 正式计时、两seed数值差异和nsys文件链接见 [编译阶梯报告](asi_compile_stages_cn.md)。本次未改变源码、生产默认、Git历史，未运行闭环任务。
