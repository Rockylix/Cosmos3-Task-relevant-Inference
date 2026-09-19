# SpecPrune early-exit：Edge FLOPs 与单 chunk 实测

2026-09-19，RTX4090，现有项目环境，PyTorch2.10/cu130。固定 BananaInBowl 录制输入（文件来自chunk3），noise seed1097657232，4步、shift5、guidance3；5次预热、20次正式测量，Dense/eager/Graph交替，CPU输入复制与RNG准备在计时外，CUDA同步围住generation。没有把旧记录里的ASI输出当作Dense参考。

## 时间结果

| 模式 | Mean(s) | Median(s) | P90(s) | 同轮Dense加速 | 逻辑FLOPs比例 |
|---|---:|---:|---:|---:|---:|
| Dense eager | 0.858931 | 0.858667 | 0.861613 | 1.000× | 100.00% |
| SpecPrune eager，无历史 | 0.457503 | 0.457321 | 0.459183 | 1.878× | 25.04% |
| SpecPrune compile+Graph，无历史 | 0.372021 | 0.371844 | 0.373216 | 2.309× | 25.04% |
| SpecPrune eager，固定历史 | 0.533658 | 0.533797 | 0.535423 | 1.609× | 34.76% |
| SpecPrune compile+Graph，固定历史 | 0.454063 | 0.454009 | 0.455258 | 1.891× | 34.76% |

无历史：每次清空selector。固定历史：先用同一录制输入运行两次eager，保存计算所得RGB/Global/confidence，每次测量前恢复这份相同历史。**这不是实际仿真c1/c2观测，不是十任务平均延迟；不得把2.309×当成整个episode加速。**

无历史各层剪枝后保留位置：B0=64、B1/B10/B15/B20/B25=44，最终352/2720 future token。固定历史为134/118/87/60/60/60，最终480/2720。每帧共享位置，solver始终恢复完整状态。

计时包含原有在线评分、finite检查、CPU诊断拷贝、packing、稀疏执行、恢复、最终head、UniPC及输入VAE编码；**不扣除内部诊断开销**。排除额外Dense对照、文件保存、VAE解码、RPC、仿真和编译预热。

## 编译与正确性

- 默认eager算法不变，`compiled_layers=None`直接执行冻结十任务路径。
- 编译217次非采集层调用；step0 conditional的B0/B1/B13/B14/B19/B24/B27七次评分层维持eager，online评分不取消。
- fullgraph=True、dynamic=True、reduce-overhead；输出头/索引/选择/scheduler未改编译路径，不是单个Global Graph。
- 独立、非计时profiler审计：两模式均看到217次 `Torch-Compiled Region` 和217次 `cudaGraphLaunch`。更换noise seed输出改变，没有重放陈旧输出。
- 同模式20次正式重复action/vision逐位稳定；全保留eager与native Dense的形状和数值逐位一致。

编译存在浮点差异，不保证action/成功率不变：

| compiled相对同历史eager | 无历史 | 固定历史 |
|---|---:|---:|
| 六层mask XOR | 全部0 | 全部0 |
| Action MSE（33×8，含condition行） | 0.001609 | 0.000676 |
| Action max abs | 0.124354 | 0.091103 |
| Future latent rel-L2 | 0.10268 | 0.09790 |
| Future latent cosine | 0.99477 | 0.99522 |

已测mask不变不保证新场景也不变。远端默认eager；之前1/10成功率不属于Graph版本。新chunk token长度变化可能重新编译/捕获，需另测冷启动与闭环效果。

## FLOPs定义

与此前Edge策略总表相同的逻辑主矩阵范围，MAC=2 FLOPs。d=2048，q=2048，k=1024，MLP中间维m=9216，28层，Dense GEN n=3093，conditional/unconditional UND u=158/19。已核对实际权重维度。

\[
F_{GEN}(n,u)=2nd(2q+2k)+4ndm+4n(n+u)q.
\]

\[
F_{UND}(u)=2ud(2q+2k)+4udm+4\frac{u(u+1)}{2}q.
\]

按逐层/step/branch实际n累加。Dense UND每分支prefill一次；当前SpecPrune每step重新算UND，计4倍UND。显式评分QK及验证AV额外计入4*nq*(n+u)*q，四次nq=340，三次nq=32。

| 模式 | GEN TFLOP | UND TFLOP | 评分 TFLOP | 总 TFLOP/chunk | 剩余比例 | 等吞吐理论比 |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 87.79993 | 0.50181 | 0 | 88.30174 | 100% | 1.000× |
| SpecPrune无历史 | 20.08986 | 2.00725 | 0.01757 | 22.11468 | 25.04% | 3.993× |
| SpecPrune固定历史 | 28.66621 | 2.00725 | 0.02060 | 30.69406 | 34.76% | 2.877× |

**不是完整pipeline/硬件FLOPs。** 不含VAE、输入/输出头、`_view`重建metadata时重算后丢弃的输入投影、norm、softmax、排序、索引、搬运、UniPC、kernel tile浪费。时间包含这些现有实现开销，理论比不能当实际加速上限。本轮没有额外缓存/GEMM/kernel算法优化。

## 复现

[benchmark代码](../tools/benchmark_specprune_exit.py)；[部署及命令](specprune_release_cn.md)。本地原始结果 `experiments/specprune_compile_speed_banana_c3_v2/results.json`、逐次时间 `timing_samples.json`、输出 `outputs.pt` 不上传Git。输入SHA256为 `64bbd643ec42a826cd607e36d594c1d0d859fd4824ac89a1b131f1bbcbd4ceaa`，结果保存运行源码哈希。

首次尝试仅因FLOPs核验读取旧模块名而停止，修正为真实 `q_proj_moe_gen/k_proj_moe_gen` 后重跑；失败尝试不计入正式样本。
