# Dense Baseline 与 ASI：小算子是否来自评分？

2026-09-19，重新采集两个预热后单chunk nsys；不修改模型或生产配置。

## 结论

**不是主要由ASI评分导致。** 完全没有ASI的Dense Baseline本身就有172.347 ms的小算子；当前B6 ASI eager为113.301 ms，其中评分函数内的小算子仅3.203 ms（约2.83%）。所以上一轮compile的大头收益是模型共有的RMSNorm、Q/K Norm、RoPE、ReLU²、residual与布局/类型转换融合，不是修复ASI引入了上百毫秒的评分。

## 配置与来源

- Dense从 `/root/robolab/cosmos-framework-edge` 导入，HEAD `1dec8b988a3505eff0df6d4fe6b3d472079423cf`，不挂ASI controller。
- ASI从 `/root/robolab/worktrees/asi-system-ablation` 导入，MainLSE/B6 eager，C80/S104，每帧K184，1 dense+7 sparse，无velocity cache。
- 项目内Edge `.venv`，RTX4090，PyTorch2.10.0+cu130，无新增环境/权重下载。
- 同一BananaInBowlTask chunk3录制输入、seed1097657232、4steps、shift5、guidance3；两者均关闭compile/CUDA Graph/padding。
- 每组5次warmup，再捕获1次同步的 `asi.chunk.generate`。名字沿用既有parser约定；Dense的这个NVTX名称不表示开启了ASI。
- Dense的224次层调用均为3093 GEN token；ASI为28次3093、196次1845。conditional/unconditional的158/19条件token布局如实保留。
- 不启动仿真，不保存视频。trace的action/future latent与各自最后一次无插桩warmup精确一致；ASI也与上一轮eager保存结果精确一致。不是声明Dense与ASI输出一样。

## 全chunk GPU工作量

只统计同进程、同步root范围内的kernel。分类与前轮一致：卷积→attention→GEMM/GEMV/split-K→其余，互斥覆盖全部kernel。时间是单次nsys GPU kernel合计，不是正式median。

| 类别 | Dense次数 | ASI次数 | Dense GPU ms | ASI GPU ms |
|---|---:|---:|---:|---:|
| GEMM/GEMV相关 | 1,981 | 2,009 | 453.035 | 297.102 |
| Attention | 309 | 309 | 147.942 | 59.691 |
| 卷积相关 | 100 | 100 | 19.298 | 19.303 |
| 其他：逐元素/归约/类型转换/布局等 | 17,662 | 18,338 | 172.347 | 113.301 |
| 合计 | 20,052 | 20,756 | 792.621 | 489.396 |

ASI的kernel数量反而略多，但大量模型kernel处理的token更少，因此GPU耗时下降。两者相减是“新增评分/打包开销 + 稀疏计算节省 + 不同执行路径”的综合差，不能当作ASI评分成本。

## ASI直接归因

对真实 `action_aligned_future_profiles` 函数包裹NVTX；通过CUDA runtime的launch correlation关联GPU kernel，不用CPU区间直接截GPU时间。两份trace均0 unmatched kernel、0 GraphLaunch。

| ASI范围 | 调用数 | kernel数 | GPU合计 ms |
|---|---:|---:|---:|
| score函数全部 | 28 | 504 | 3.400912 |
| 其中局部FP32 QK矩阵计算 | — | 28 | 0.197618 |
| 其中转换/布局/概率/归约等小算子 | — | 476 | 3.203294 |
| select/plan范围 | 1 | 133 | 0.248271 |
| restore范围 | 7 | 7 | 0.069125 |

评分只发生于step0 conditional的B0–B27。函数包含FP32 Q/K准备、GQA展开、局部QK、减主attention LSE、exp、head/action加权归约。score总GPU时间约为ASI全部GPU kernel时间的0.695%；评分中的小算子约占ASI所有小算子的2.83%。

这些是**明确圈定的函数/范围成本**，不是ASI全部额外成本：不包含主attention返回LSE的增量、外部profile clone、controller布局处理、稀疏gather等。select GPU0.248 ms也不能被解释为完整端到端选mask只需0.248 ms。

`asi.score` host范围合计5.005 ms，`asi.select` host范围26.906 ms。后者含等待先前GPU工作的同步，不能全部归为评分CPU计算；GPU和host时间重叠，不能相加估计节省。

## 对compile对比的含义

1. 需要把“ASI减少token计算”和“compile优化通用模型执行”分开报告。
2. 上一轮1.190×是ASIB6 eager→ASI compile的系统收益，不是ASI算法独有收益。
3. 本轮没有测试Dense compile，不能推断其具体加速比。若要公平评估开启系统优化后的算法优势，需要另测相同compile配置的Dense与ASI；这里不自动扩展实验。
4. 本轮是各一个profile，不报告新的稳定chunk加速比或闭环成功率。

## 文件与复现

- 工具：[profile_dense_vs_asi.py](../tools/profile_dense_vs_asi.py)
- [Dense nsys](../experiments/nsys_reports/dense_vs_asi_dense_v1.nsys-rep)
- [ASI nsys](../experiments/nsys_reports/dense_vs_asi_asi_v1.nsys-rep)
- 两份原始结果和解析：`experiments/dense_vs_asi_scoring_banana_c3_v1/{dense,asi}/{results.json,nsys_analysis.json}`。

```bash
cd /root/robolab/worktrees/asi-system-ablation
EDGE_PY=/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python
export COSMOS_TRAINING=0 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 输出路径需新建，已有结果不会覆盖。两组顺序运行，不并行占GPU。
for mode in dense asi; do
  if [ "$mode" = dense ]; then
    source_root=/root/robolab/cosmos-framework-edge
  else
    source_root=/root/robolab/worktrees/asi-system-ablation
  fi
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o "experiments/nsys_reports/dense_vs_asi_${mode}_repeat" \
    env PYTHONPATH="$source_root:/root/robolab/cosmos-edge-overlay" \
    "$EDGE_PY" -u tools/profile_dense_vs_asi.py --mode "$mode" \
    --output "experiments/dense_vs_asi_scoring_banana_c3_repeat/$mode" || break
done
```

独立review核对了checkout来源、相同输入hash与参数、真实token数、28次评分范围和输出一致性。该轮采集时没有修改Dense源码、生产策略或Git历史。

提交整理（2026-09-19）：历史输出比对改为显式可选的 `--reference-eager <outputs_seed_offset0.pt>`（文件内需有 `eager` 键）。不传时仍强制检查本次 trace 与 warmup 输出精确相同，历史比对字段记为 `null`，而非宣称通过；传入时校验并记录参考文件 hash。既有报告中的历史 exact 结果保留不变，原始产物不提交。本工具仍以本项目现有 checkout、权重和录制输入路径为运行前提。
