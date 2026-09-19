# ASI 编译与 CUDA Graph 分阶段评估

日期：2026-09-18。实验分支 `experiment/asi-system-ablation`，基础 HEAD `87caf7454e203cb25ff768ac0a86d56d2dea10d6`。

## 结论

本轮主要收益来自 decoder 编译，而不是额外的 CUDA Graph。当前 B6 ASI eager 的单 chunk median 从 **553.528 ms** 降至 **464.336 ms**，相对加速 **1.192×**；其中全 decoder 编译已经达到 **466.034 ms**。

CUDA Graph 确实生效：预热后单 chunk 实测 **264 次 GraphLaunch**，其中 224 次分别位于 `4 steps × 2 CFG branches × 28 blocks`，首轮 dense/profile 的 28 层也全部 replay。不是“开了开关但没有捕获”。不过相对 decoder+heads compile-only，仅减少 **0.760 ms / 0.163%**，本轮不足以声称稳定的额外加速。

**编译不等于数值无损。** 仅编译后续稀疏栈时，首轮评分和 mask 完全不变，但最终 action/latent 仍有数值差异；首轮 dense/profile 也编译后，部分 mask 发生变化。未将任何配置设为生产默认，未运行闭环成功率评估。

## 1. 对比边界与配置

- 本表 reference 为 **当前 B6 ASI eager**：已有批量评分、布局/稀疏 metadata 缓存、metadata profile 路径及主 attention LSE 复用。
- **不是 Dense Baseline，也不是最初没有系统优化的 ASI legacy。** legacy 的先前测量见 [前轮报告](asi_system_existing_and_main_lse_cn.md)，不同轮次不能直接当作本轮严格配对结果。
- RTX 4090；项目内 Edge `.venv`；PyTorch `2.10.0+cu130`；无额外环境下载。
- 输入：`BananaInBowlTask` 第 3 个 chunk 的固定录制输入，路径为生产 checkout 下 `experiments/asi_smoke10_s0_p0_v1/_temporary_inputs/BananaInBowlTask.pt`。
- 4 denoise steps、shift=5、guidance=3、Core80+Stable104，每帧184个 future token、Top6 blocks、1个 dense stack + 7个 sparse stacks。无 velocity cache。
- 录制噪声 seed 为 `1097657232`；另用 `1097657233` 做数值与重复性验证。这里不是将录制请求的噪声 seed 强改为 policy server seed=0。
- 全模式 `pad_for_cuda_graphs=False`，不让 padding/token 数变化混入 Graph 对比。
- decoder `fullgraph=True, dynamic=True`；head 沿用源码的 `fullgraph=True, dynamic=True`。compile-only 用 default，Graph 用 reduce-overhead。
- 每种模式预热5次，再各20次正反序交替测量。每次深拷贝输入、重置 Python/NumPy/Torch/CUDA RNG、创建新 controller；未复用模型输出。
- 主计时为 CUDA 同步后的 `generate_samples_from_batch` wall time；不含模型加载、编译预热、输入深拷贝、结果搬回CPU/统计、VAE decode、RPC、仿真。Profiler 单独运行，不混入主计时。

| 模式 | 首轮 dense/profile | 后续7次 sparse forward | 5个 encode/decode heads | Graph |
|---|---|---|---|---|
| `eager` | eager | eager | eager | 关 |
| `sparse_compile` | eager，保留同一 metadata/LSE 评分路径 | compile | eager | 关 |
| `decoder_compile` | compile，包含评分张量计算 | compile | eager | 关 |
| `all_compile` | compile | compile | compile | 关 |
| `all_graph` | compile | compile | compile | 开 |

这里只对 layer/head 做编译和 compiler-managed Graph；不是整个 chunk、整个 decoder 栈或 UniPC 循环合成一个 global Graph。评分后的 TopK/选 mask 控制流程仍在图外。

## 2. 正式单 chunk 计时

数据：[timing/results.json](../experiments/compile_stages_banana_c3_v2/timing/results.json)。单位 ms，加速按 median 比值。

| 模式 | Mean | Median | P90 | 相对当前 B6 ASI eager |
|---|---:|---:|---:|---:|
| 当前 ASI eager | 553.151 | 553.528 | 554.537 | 1.000× |
| 仅 sparse decoder compile | 486.298 | 486.495 | 487.612 | 1.138× |
| 全 decoder compile | 465.684 | 466.034 | 467.857 | 1.188× |
| decoder + heads compile | 464.857 | 465.095 | 466.567 | 1.190× |
| 上述配置 + CUDA Graph | 464.223 | 464.336 | 465.977 | 1.192× |

逐项 median 减少：67.033 ms、20.460 ms、0.939 ms、0.760 ms。后两项都很小，不把单轮小差值称为稳定收益。

本进程各阶段第一次调用分别为 0.883 / 3.255 / 2.199 / 2.180 / 4.679 s。已有 Inductor 磁盘缓存，且不同阶段共享先前编译产物，wrapper安装在计时外；这些数值仅是**本次进程观测到的首次 generation 调用**，不是空缓存编译开销，也不能作为冷启动部署预算。Graph 第二次调用仍为0.664 s，正式统计在5次预热后。

## 3. 数值变化与重复性

以下全部相对同 seed 的 B6 eager；action 为32个预测 horizon（排除 condition），包含原接口 gripper 翻转；vision 为 final future latent（排除L0），不是RGB图片。max absolute error 是模型 action 数值单位，未自动换算成真机关节角单位。

`mask XOR` 是 `[8,340]` 布尔 mask 不同的元素总数，不是被替换 token 对数。所有模式都严格保留8×184个 future token，且本输入两seed的Top6 block列表一致。

### Seed 1097657232

| 模式 | mask XOR | Action MSE | Action max abs | Action cos | Latent cos | Latent rel-L2 |
|---|---:|---:|---:|---:|---:|---:|
| sparse_compile | 0 | 0.00029213 | 0.052838 | 0.999919 | 0.995082 | 0.099135 |
| decoder_compile | 64 | 0.00250009 | 0.173078 | 0.999299 | 0.955198 | 0.297809 |
| all_compile | 60 | 0.00291528 | 0.200219 | 0.999183 | 0.958351 | 0.287231 |
| all_graph | 60 | 0.00189929 | 0.187766 | 0.999470 | 0.961716 | 0.275665 |

### Seed 1097657233

| 模式 | mask XOR | Action MSE | Action max abs | Action cos | Latent cos | Latent rel-L2 |
|---|---:|---:|---:|---:|---:|---:|
| sparse_compile | 0 | 0.00029277 | 0.064245 | 0.999923 | 0.997123 | 0.075915 |
| decoder_compile | 60 | 0.00062993 | 0.075082 | 0.999838 | 0.965939 | 0.261065 |
| all_compile | 60 | 0.00159646 | 0.128906 | 0.999542 | 0.957729 | 0.290256 |
| all_graph | 76 | 0.00080443 | 0.112062 | 0.999776 | 0.952997 | 0.306272 |

验证通过：

- 两seed各模式重复执行，以及 `seed A → seed B → seed A` 返回后的 action/latent/mask 精确一致；切换seed确实改变输出，未发现 stale replay。
- 20次正式计时每次都核对该模式输出和mask；全部通过。
- sparse_compile 的 raw profile、Core/Stable/执行mask均与eager完全一致。
- 所有已采集结果（评分、输出、mask）检查NaN/Inf；不是对每个内部算子张量逐个插桩。
- 29项CPU测试通过，包括dense eager路由、LSE安全边界和Graph trace解析。

编译首轮会改变模型执行与评分的浮点计算，进而改变离散TopK边界；最终latent差异并不小。当前实验没有隔离“仅数值变化”和“mask变化”的各自贡献，也不能据此证明成功率不变。`all_graph` 与 `all_compile` 使用不同编译mode，输出差异不能单独归因于Graph replay。验证只覆盖同一输入、两种noise seed，未覆盖新任务/新文本长度的重编译与Graph管理。

## 4. nsys：Graph 真正执行了吗？

每模式独立进程预热5次，仅采集一次 `asi.chunk.generate`。使用 `--cuda-graph-trace=node`，不计入性能表。每份trace恰好一个root、224个layer NVTX。

| 模式 | Trace kernels | Kernel时间合计 ms | Trace wall ms | 实际 GraphLaunch |
|---|---:|---:|---:|---:|
| eager | 20,756 | 488.860 | 554.214 | 0 |
| sparse_compile | 9,368 | 424.150 | 484.832 | 0 |
| decoder_compile | 6,064 | 403.688 | 465.578 | 0 |
| all_compile | 5,824 | 403.485 | 463.384 | 0 |
| all_graph | 5,832 | 403.453 | 470.386 | 264 |

`all_graph` 的8个step/branch范围均观测到28次decoder GraphLaunch：包括step0 conditional首轮dense+profile。其余40次在layer范围外，与5个encode/decode入口×8次forward相容，但当前NVTX没有逐head标签，故不宣称已逐head精确归属。

为什么Graph收益小：全编译后 kernel GPU工作量已由488.860 ms降到403.485 ms，Graph约403.453 ms，基本没有进一步减少GPU计算；Graph的作用主要是提交复用，不会省掉GEMM/attention算术。本trace中全编译的GPU kernel busy占root约87.1%，它支持“剩余GPU计算占主导”，但不是所有60 ms空隙都能由Graph消除。root减kernel busy还含拷贝、同步等，不能直接标为Python开销。node级Graph trace本身有额外开销，因此不拿470.386 ms判定Graph比compile-only更慢。

报告与审计：

- [eager nsys](../experiments/nsys_reports/compile_stages_v2_eager.nsys-rep)
- [sparse compile nsys](../experiments/nsys_reports/compile_stages_v2_sparse_compile.nsys-rep)
- [decoder compile nsys](../experiments/nsys_reports/compile_stages_v2_decoder_compile.nsys-rep)
- [all compile nsys](../experiments/nsys_reports/compile_stages_v2_all_compile.nsys-rep)
- [all graph nsys](../experiments/nsys_reports/compile_stages_v2_all_graph.nsys-rep)
- [Graph审计JSON](../experiments/compile_stages_banana_c3_v2/trace_all_graph/nsys_audit.json)

## 5. 复现与代码

```bash
cd /root/robolab/worktrees/asi-system-ablation
export PYTHONPATH="$PWD:/root/robolab/cosmos-edge-overlay"
export COSMOS_TRAINING=0 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EDGE_PY=/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python

# output必须是新的目录，已有结果不会被覆盖
"$EDGE_PY" -u tools/benchmark_asi_compile_stages.py \
  --output experiments/compile_stages_banana_c3_repeat/timing

# 分别将mode替换为eager/sparse_compile/decoder_compile/all_compile/all_graph；顺序运行
mode=all_graph
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o "experiments/nsys_reports/compile_repeat_${mode}" \
  "$EDGE_PY" -u tools/benchmark_asi_compile_stages.py \
  --trace-mode "$mode" --output "experiments/compile_stages_banana_c3_repeat/trace_${mode}"
nsys export --type sqlite \
  --output "experiments/nsys_reports/compile_repeat_${mode}.sqlite" \
  "experiments/nsys_reports/compile_repeat_${mode}.nsys-rep"
"$EDGE_PY" tools/audit_asi_compile_trace.py \
  --sqlite "experiments/nsys_reports/compile_repeat_${mode}.sqlite" \
  --output "experiments/compile_stages_banana_c3_repeat/trace_${mode}/nsys_audit.json"

"$EDGE_PY" -m unittest discover -s tests -p 'test_asi_*.py'
```

新增工具：`tools/benchmark_asi_compile_stages.py`、`tools/audit_asi_compile_trace.py`；测试：`tests/test_asi_compile_stages.py`、`tests/test_asi_compile_trace.py`。主计时results/manifest保存了source SHA256、输入hash、HEAD、Python和GPU信息；未提交的实验源码状态不能只用HEAD标识。v1结果保留但以包含return-to-seed严格校验的v2为正式数据。

### 独立 review 与原始数据注意事项

独立 reviewer 已核对8项源码hash、输入hash及五份SQLite，重新解析结果与保存的audit JSON逐项一致；无阻断本次固定chunk报告的问题。另记录两项非阻断局限，保留原始结果不覆盖：

- `sparse_compile` 的继承summary字段 `dense_profile_execution="compiled_batched"` 不准确：它按wrapper是否存在标注，没有识别本实验的 `_orig_mod` 绕过。真实首轮是eager；同记录的 `dense_profile_eager=true`、调用源码和单元测试为准。本报告已按实际路径描述。
- 每次repeat gate强制验证action、future latent和执行mask，不是独立强制每个Core/Stable中间集合。`return_to_seed0` 保存的完整比较实际显示raw profile/score精确相同、Core/Stable/执行mask XOR均为0、Top6列表相同；不能扩大表述为每次重复都强制检查了所有中间统计。

同noise通过同输入seed及初始化源码核查保证（`arch_invariant_rand`使用每次新建的`RandomState(seed)`，噪声初始化不在编译区域），本轮没有另外保存初始noise逐元素hash。

## 6. 本轮边界与下一步待决策

若优先保持当前选token结果，`sparse_compile` 是本输入上更保守的候选；若优先追求吞吐，全decoder编译还省约20 ms，但需要先接受/验证mask和预测变化。两者都不是已经通过闭环验证的生产替代。

本轮只做经批准的编译阶梯评估，未继续调整算子精度或算法，未自动启用默认compile/Graph，未commit/merge/push。生产ASI checkout的branch、HEAD和9个原有untracked文件均未变化；Dense checkout仍为`cache / 1dec8b9`且工作区干净。后续优化或闭环质量验证待讨论后执行。
