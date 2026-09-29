"""Summarize native-branch paired benchmark results, with provenance checks."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    modes = ["baseline", "asi", "toca", "worldcache", "c3ache"]
    results = {m: json.loads((root / m / "summary.json").read_text()) for m in modes}
    dense = results["baseline"]
    for result in results.values():
        for key in ["input_sha256", "kwargs", "python", "torch", "gpu", "compile", "cuda_graphs"]:
            assert result[key] == dense[key], key
        assert result["gates"]["dense_hashes"] == dense["gates"]["dense_hashes"], "Dense output differs across branches"
        assert result["gates"]["finite"]
    mapping = [
        ("baseline", "dense", "Dense Baseline"),
        ("asi", "asi", "ASI Core80+Stable104"),
        ("toca", "toca", "ToCa-Future D/C/D/C r=0.25"),
        ("worldcache", "worldcache", "WorldCache D/D/D/C"),
        ("c3ache", "c3ache_period2_amortized", "C3ache period=2 均摊"),
    ]
    rows = []
    for branch, mode, label in mapping:
        t = results[branch]["timing"][mode]
        rows.append(
            {
                "strategy": label,
                "mean_ms": t["mean_s"] * 1000,
                "paired_dense_mean_ms": results[branch]["timing"]["dense"]["mean_s"] * 1000,
                "speedup_vs_paired_dense_mean": results[branch]["timing"]["dense"]["mean_s"] / t["mean_s"],
                "speedup_vs_baseline_mean": dense["timing"]["dense"]["mean_s"] / t["mean_s"],
                "n_chunks": t["n"] * (2 if branch == "c3ache" else 1),
                "branch": results[branch]["branch"],
                "head": results[branch]["head"],
            }
        )
    with (root / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cycle = results["c3ache"]["timing"]["c3ache_period2_amortized"]
    cycle_speedup = results["c3ache"]["timing"]["dense"]["mean_s"] / cycle["mean_s"]
    dense_times = [v["timing"]["dense"]["median_s"] for v in results.values()]
    spread = (max(dense_times) - min(dense_times)) / dense["timing"]["dense"]["median_s"] * 100
    summary = {
        "rows": rows,
        "dense_outputs_bitwise_equal_across_all_branches": True,
        "dense_median_spread_percent": spread,
        "c3ache_period2_per_chunk_mean_s": cycle["mean_s"],
        "c3ache_period2_speedup_vs_paired_dense_mean": cycle_speedup,
        "c3ache_cpu_tests": {
            "passed": 18,
            "total": 18,
            "test_runner_override": "test_c3ache.BASELINE=Path(/root/robolab/cosmos-framework-edge)",
        },
        "raw_results": results,
    }
    (root / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    table = ["| 策略 | 平均单 chunk (ms) | 配对 Dense mean (ms) | 配对加速比 |", "|---|---:|---:|---:|"]
    table += [
        f"| {r['strategy']} | {r['mean_ms']:.2f} | {r['paired_dense_mean_ms']:.2f} | {r['speedup_vs_paired_dense_mean']:.3f}× |"
        for r in rows
    ]
    c3gate = results["c3ache"]["gates"]
    report = f"""# Edge 单 chunk 同输入速度对比（2026-09-12）

## 结果

{chr(10).join(table)}

C3ache refresh_period=2：刷新与命中各占一半，实际测得**每 chunk 均摊 {cycle["mean_s"] * 1000:.2f} ms，{cycle_speedup:.3f}×**。
这是每对“刷新+命中”的总时间除以二；不能把 hit-only 加速比当作持续运行的平均加速比。
主表统一使用 mean，不混用 median；C3ache 包含 20 次刷新和 20 次命中，共 40 个 chunk。
其余策略各 20 个计时 chunk。刷新 mean 为 {results["c3ache"]["timing"]["c3ache_refresh"]["mean_s"] * 1000:.2f} ms，
命中 mean 为 {results["c3ache"]["timing"]["c3ache_hit"]["mean_s"] * 1000:.2f} ms。
不计模型加载冷启动，但均摊时间包含残差建立与 metadata 校验开销。

## 配置与计时边界

- GPU：RTX 4090 24 GB；驱动 580.178.04；Torch {dense["torch"]}。
- Python：`{dense["python"]}`。没有新增环境或下载权重。
- 权重：`/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID`。
- 同一个 BananaInBowlTask 第 3 个 chunk；4 步 UniPC、shift=5、guidance=3，记录的噪声 seed=1097657232。
- **所有分支均 eager；torch.compile=False、CUDA Graph=False。** 不与此前 compiled 真机数据混用。
- 每个模式预热 5 次，计时 20 次。各分支单独加载原生模型，在该进程内 Dense/策略交替运行；C3ache 保持 refresh→hit 顺序，交替把 Dense 放在这一对之前/之后。
- 每次 deep-copy 同一 data_batch，重置 Python/NumPy/Torch RNG，原生 generate 创建 request-local sampler/text KV。C3ache 仅保留预期跨请求的 residual cache。
- 计时前后 CUDA synchronize。包括 `generate_samples_from_batch`、控制器 setup/finish、VAE encode、打分、token 选择、gather/scatter、缓存 metadata 校验与写入。
- 不包括模型加载、warmup、输入文件读取/deepcopy、计时外有限值校验/CPU 输出拷贝、VAE decode、WebSocket/RoboLab；不是端到端完整任务时间。
- 各分支的 Dense median 范围为 {min(dense_times) * 1000:.2f}–{max(dense_times) * 1000:.2f} ms，跨度相对 Baseline 为 {spread:.2f}%。主表统一使用各自交替测得的 Dense mean 作分母；`comparison.csv` 同时保留共同 Baseline mean 分母。各模式原始 median/P90 保留在各自 `summary.json` 中，不把周期均摊的分位数当成原始 chunk 分位数。

## 方法边界

- ASI 使用 `version/core80-stable104-action-weighted`，不是 asi-velocity-cache：每帧 Core80 + 共享 Stable104，保留 K184/340；step0 conditional 全量 profile，之后 7 次 GEN forward 稀疏，每个 chunk 重新选 mask。
- ToCa 使用远端冻结配置：D/C/D/C，r=0.25，spatial bonus=0，CFG 独立，joint attention backend。
- WorldCache 使用 D/D/D/C，最后一步缓存预测；前三步、两个 CFG 分支均完整计算。
- C3ache 使用远端冻结配置：整栈 GEN pre-final-norm residual，refresh_period=2，dense_tail_steps=2。刷新8次完整 Transformer；命中4次完整、4次缓存，但 embeddings/final norm/heads/CFG/UniPC 仍执行。
- **C3ache 是重复同一观测、递增 cache chunk_id 的计算路径微基准。** 不是相邻真实 RoboLab chunk 的保真度或成功率评测；没有改变观测来伪造真实轨迹，也不能据此主张跨 chunk 动作精度。

## 验证

- 五个原生分支关闭策略的 Dense action/vision SHA256 完全一致，且与录制 Dense 输出逐元素一致。
- C3ache 刷新 action/vision 与本轮 Dense 逐元素完全一致；所有模式最终 action/vision 无 NaN/Inf。
- C3ache 真权重 GPU：refresh 的第0层调用 {c3gate["c3ache_refresh"]["counts"]["layer0"]} 次，hit 为 {c3gate["c3ache_hit"]["counts"]["layer0"]} 次；命中确实跳过整栈，不是只在输出端替换。
- C3ache CPU tests 18/18 通过。原测试的 Baseline 目录写死为集群名称，本机 runner 仅覆盖测试路径，不修改算法或测试断言。
- 没有修改现有 baseline/ASI/ToCa/WorldCache 源码；C3ache 新 worktree 仅新增本轮 benchmark/report 工具及报告，没有提交或推送。

## 代码与数据

| 策略分支 | HEAD | 工作树 |
|---|---|---|
"""
    for result in results.values():
        report += f"| `{result['branch']}` | `{result['head'][:12]}` | `{result['source']}` |\n"
    report += f"""
- 原始输入：`{dense["input"]}`。
- 输入 SHA256：`{dense["input_sha256"]}`。
- 每个目录的 `timings.csv` 为逐次原始时间；`gate.json` 为验证；`summary.json` 含配置和状态计数。
- 总表：`comparison.csv`；完整汇总：`comparison.json`。不保存额外视频或大张量。

复现（指定新的输出目录，防止覆盖已有结果）：

```bash
cd /root/robolab/worktrees/c3ache
/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \\
  tools/benchmark_edge_strategies.py \\
  --output experiments/single_chunk_speed_comparison_repeat --warmups 5 --repeats 20
/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \\
  tools/summarize_edge_strategy_benchmark.py experiments/single_chunk_speed_comparison_repeat
```

来源：`tools/benchmark_edge_strategies.py` 为测量边界；各分支 `configs/*baseline.json` 为冻结配置；
C3ache 实际缓存/调度见 `cosmos_framework/inference/c3ache.py`，真实跳栈在
`cosmos_framework/model/generator/mot/unified_mot.py`。
"""
    (root / "report_cn.md").write_text(report)
    print("\n".join(table))
    print(f"C3ache amortized: {cycle['mean_s'] * 1000:.2f} ms; {cycle_speedup:.3f}x")


if __name__ == "__main__":
    main()
