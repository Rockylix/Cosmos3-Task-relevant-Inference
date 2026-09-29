# Edge 单 chunk 速度对比（2026-09-12）

RTX 4090、Torch 2.10.0+cu130；各分支原生 eager，compile/CUDA Graph 均关闭。
固定 BananaInBowlTask chunk 3 的同一输入，4 步 UniPC、shift=5、guidance=3、seed=1097657232。
各模式预热 5 次、测量 20 次，在各自进程内与 Dense 交替计时。

主表统一使用平均单 chunk 时间和配对 Dense mean 计算加速比。

| 策略 | 平均单 chunk / ms | 配对 Dense mean / ms | 配对加速比 |
|---|---:|---:|---:|
| Dense Baseline | 855.03 | 855.03 | 1.000× |
| ASI Core80+Stable104 | 587.44 | 860.06 | 1.464× |
| ToCa-Future D/C/D/C，r=0.25 | 782.88 | 866.63 | 1.107× |
| WorldCache D/D/D/C | 673.69 | 866.63 | 1.286× |
| C3ache period=2 均摊 | 679.49 | 869.78 | 1.280× |

C3ache 每 2 个 chunk 刷新一次，刷新/命中各半时，每 chunk **均摊 679.49 ms，1.280×**
（周期总时间除以 2；与配对 Dense mean 比较），不能把命中单次加速当成持续平均加速。
计时部分包含 20 次刷新、20 次命中，共 40 个 chunk：

\[
\overline T_{\mathrm{C3ache}}
=\frac{\sum_{i=1}^{20}T_{\mathrm{refresh},i}+\sum_{i=1}^{20}T_{\mathrm{hit},i}}{40}
=\frac{872.8045+486.1842}{2}\;\mathrm{ms}
=679.4943\;\mathrm{ms}.
\]

其他策略各 20 个计时 chunk；原始 median/P90 仍保留在各策略 `summary.json` 中。
本次仅重新汇总原有测量数据，没有重新推理。
这份输入下，ASI 的持续单 chunk 时间最低；C3ache 命中单次更快，但需要承担刷新成本。

## 测量与验证边界

- 包含 `generate_samples_from_batch`、控制器生命周期、profile/选择和缓存开销；前后 CUDA 同步。
- 不含模型加载、输入文件读取/复制、计时外验证与 CPU 输出复制、VAE decode、RPC、仿真。
- 每次复制输入并重置 RNG；不修改模型算法，不下载权重或环境。
- C3ache 使用同一观测重复回放、递增 chunk_id 来测缓存路径，不是相邻真实 chunk 的精度评测。
- 五个分支关闭策略的 Dense action/vision 与录制参考逐元素一致；C3ache 刷新也完全一致。
- 命中时完整 Transformer 调用实测由 8 降为 4，另外 4 次确实跳过整栈。所有模式最终输出有限。
- C3ache CPU 测试 18/18 通过；测试入口将写死的集群 Baseline 路径指向本机 Baseline，未改断言。
- ASI 是 `version/core80-stable104-action-weighted`，不是 `experiment/asi-velocity-cache`。
- ToCa：CFG 独立、spatial bonus=0、joint backend；WorldCache/C3ache 配置直接读取各分支冻结 JSON。

## 结果与复现

完整记录：[report_cn.md](../experiments/single_chunk_speed_comparison_20260912_v1/report_cn.md)。
同目录 `comparison.csv` / `comparison.json` 保存总表、HEAD、输入 SHA256 和配置；
各策略目录 `timings.csv` / `gate.json` / `summary.json` 保存原始计时及验证结果。
实验数据由 Git 忽略，本文和脚本可随分支保存。

```bash
cd /root/robolab/worktrees/c3ache
/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  tools/benchmark_edge_strategies.py \
  --output experiments/single_chunk_speed_comparison_repeat --warmups 5 --repeats 20
/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  tools/summarize_edge_strategy_benchmark.py experiments/single_chunk_speed_comparison_repeat
```

分支 HEAD：Baseline `1dec8b988a35`，ASI `7d3c42467188`，ToCa `f7cf5d693df0`，
WorldCache `763b6d145af0`，C3ache `225dd582180e`。
