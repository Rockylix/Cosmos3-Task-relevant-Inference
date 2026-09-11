"""Offline layer-selection accounting for the completed three-task run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text())
    gate = json.loads((run / "gpu_gate.json").read_text())
    assert gate["passed"]
    manifest = json.loads((run / "manifest.json").read_text())
    tasks = manifest["tasks"]
    k = int(manifest.get("topk_blocks", 6))
    task_fixed = bool(manifest.get("task_core_top7", False))
    assert summary["phase"] == "complete" and summary["completed"] == 3
    requests = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines()]
    assert len({(r["task"], r["chunk"]) for r in requests}) == len(requests)
    assert set(r["task"] for r in requests) == set(tasks)
    episode_map = {r["task_name"]: r for r in summary["episodes"]}
    matrices, frequencies, task_results = {}, [], []
    for task in tasks:
        rows = [r for r in requests if r["task"] == task]
        assert [r["chunk"] for r in rows] == list(range(1, len(rows) + 1))
        rng = np.random.default_rng(0)
        matrix = np.zeros((len(rows), 28), dtype=np.int64)
        for i, row in enumerate(rows):
            assert row["seed"] == [int(rng.integers(0, 2**31))]
            assert (row["dense_stack_count"], row["sparse_stack_count"], row["cache_evaluations"]) == (1, 7, 4)
            assert row["all_final_finite"]
            selected = row["core_blocks"]
            assert len(selected) == len(set(selected)) == k
            q, r, h = (np.asarray(row[k]) for k in ("block_quality", "block_mass", "block_entropy"))
            assert all(v.shape == (28,) and np.isfinite(v).all() for v in (q, r, h))
            np.testing.assert_allclose(q, r * np.maximum(1 - h, 0), rtol=2e-6, atol=1e-9)
            if task_fixed:
                assert selected == rows[0]["core_blocks"]
                assert row["core_layers_reused"] == (i > 0)
                assert row["profiled_block_count"] == 28
                assert row["stable_profile_blocks"] == list(range(28))
            if not task_fixed or i == 0:
                assert np.all(np.diff(q[selected]) <= 1e-9)
                assert q[selected].min() >= np.delete(q, selected).max() - 1e-9
            matrix[i, selected] = 1
        matrices[task] = matrix
        times = [r["generation_s"] for r in rows if r["chunk"] > 1]
        task_results.append(
            {
                **episode_map[task],
                "chunks": len(rows),
                "generation_mean_s_excluding_chunk1": float(np.mean(times)),
                "generation_median_s_excluding_chunk1": float(np.median(times)),
                "generation_p90_s_excluding_chunk1": float(np.quantile(times, 0.9)),
                f"distinct_top{k}_sets": len({tuple(sorted(r["core_blocks"])) for r in rows}),
            }
        )
    for task in [*tasks, "ALL_CHUNKS_POOLED"]:
        rows = requests if task == "ALL_CHUNKS_POOLED" else [r for r in requests if r["task"] == task]
        for block in range(28):
            ranks = [row["core_blocks"].index(block) + 1 for row in rows if block in row["core_blocks"]]
            frequencies.append(
                {
                    "task": task,
                    "block": block,
                    "chunks": len(rows),
                    "selected_count": len(ranks),
                    "selection_rate": len(ranks) / len(rows),
                    "mean_rank_when_selected": float(np.mean(ranks)) if ranks else "",
                    "mean_Q_all_chunks": float(np.mean([r["block_quality"][block] for r in rows])),
                }
            )
    write_csv(run / "layer_selection_frequency.csv", frequencies)
    write_csv(run / "task_results.csv", task_results)
    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, max(len(v) for v in matrices.values()) * 0.27)))
    for ax, task in zip(axes, tasks, strict=True):
        matrix = matrices[task]
        ax.imshow(matrix, cmap=ListedColormap(["#eeeeee", "#d73027"]), vmin=0, vmax=1, aspect="auto")
        ax.set(title=task, xlabel="Transformer block", ylabel="Closed-loop chunk (1-based)")
        ax.set_xticks(range(0, 28, 2))
        ax.set_yticks(range(len(matrix)), range(1, len(matrix) + 1))
    fig.suptitle(f"Top-{k} Core-source layers; task-fixed={task_fixed}; red = used for current mask")
    fig.tight_layout()
    fig.savefig(run / f"chunk_top{k}_layers.png", dpi=170)
    plt.close(fig)
    summary.update(task_statistics=task_results, all_chunk_checks_passed=True, selected_layer_slots=k * len(requests))
    (run / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        f"# ASI 1D/7S + Step0 velocity cache：三任务与逐 chunk Top-{k}",
        "",
        "本轮只执行一次全量 conditional；Step0 unconditional 及其余六次 forward 保持 ASI 稀疏。",
        "缓存的是该 ASI Step0 的 guided velocity，不是双分支全量的 Dense velocity。每个 chunk 重新 profile、选区、建缓存。",
        "",
        "| Task | Success | Score | Steps | Chunks | Generation median(s)* |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in task_results:
        lines.append(
            f"| {row['task_name']} | {row['success']} | {row['score']:.4f} | {row['episode_step']} | {row['chunks']} | {row['generation_median_s_excluding_chunk1']:.6f} |"
        )
    lines += [
        "",
        f"成功 {summary['success_count']}/3，平均 score={summary['score_mean']:.6f}；共 {len(requests)} 个 chunk。",
        "",
        "*排除每任务 chunk1，测量生成区间（含选择与缓存、最终有限值检查；不含 score 转 CPU/CSV I/O）。仿真器同时占用 GPU，不是独占 GPU 的稳定配对 benchmark，不据此计算加速比。",
        "",
        f"## Top-{k} 统计口径",
        "",
        "R_l 为当前实现 action-aligned future mass 的八帧均值，H_l 为帧内空间归一化熵的八帧均值，Q_l=R_l(1-H_l)。",
        "action 四个 horizon 权重为 [1/6,1/3,1/3,1/6]；Stable 每个 chunk 始终使用全部 B0–B27 的当前 profile。",
        f"Core 层集合：{'仅每任务首个 chunk 选 Top-7，后续固定这些 IDs，但刷新其权重' if task_fixed else '每个 chunk 重选 Top-6'}。",
        "这些层是 Core score 的来源层，不是只计算这些层；全量/稀疏 forward 都经过全部 28 个 block。",
        "",
        f"![逐 chunk 来源层](chunk_top{k}_layers.png)",
        "",
        f"- [逐 chunk Top-{k} 来源层](chunk_topk_layers.csv)：{'task_initial_rank 保留任务首个 chunk 的排名，不是当前 Q 排名' if task_fixed else 'rank 按当前 Q 降序'}，block ID 从 0 开始。",
        "- [每 chunk 全部 28 层的 R/H/Q](chunk_block_scores.csv)。",
        "- [每任务及 pooled 入选频率](layer_selection_frequency.csv)：分母是该任务 chunk 数；pooled 按 chunk 加权，不是任务等权。",
        "- [原始请求记录](requests.jsonl)、[GPU gate](gpu_gate.json)、[任务结果](task_results.csv)。",
        "",
        f"## 每个 chunk 使用的层（{'任务首次排名顺序' if task_fixed else '当前 Q 降序'}）",
        "",
        f"| Task | Chunk | Top-{k} |",
        "|---|---:|---|",
    ]
    for row in requests:
        lines.append(f"| {row['task']} | {row['chunk']} | {', '.join('B' + str(b) for b in row['core_blocks'])} |")
    lines += [
        "",
        "本轮每任务一次，不筛选、不重跑失败 episode。成功以 success 字段为准，score 单独保留。",
        "GPU gate 为旧 Top-6 模式兼容性检查：全部 28×5 个 Q/K/V/O/MLP 模块行数 [3093]+[1845]×7、224 个 block 输出 finite、快速/审计路径逐元素一致。",
        "任务固定 Top-7 的 CPU/GPU 独立回归见 docs/task_core_top7_global_stable_cn.md；不要将上述 Top-6 gate 当作当前 Top-7 的保真度测试。" if task_fixed else "本轮 Core 来源层仍为每 chunk 重选 Top-6。",
        "不从单 chunk gate 的 action 一致推广为闭环任务成功率不变。",
        "",
        "质量边界（旧 Top-6 兼容性 gate）：新缓存与原 ASI 的完整 vision latent（含 L0）互比 "
        f"cosine={gate['cache_vs_asi']['vision']['cosine']:.6f}、"
        f"relative-L2={gate['cache_vs_asi']['vision']['relative_l2']:.6f}。"
        "action 与保留区域 latent 在该 gate 逐元素相同，差异位于被移除区域；本轮没有解码图像指标，不能沿用上一次双分支全量缓存的图像改善结论。",
        "保留各任务原始 success 和 score，不由 success 推断放置子任务得到满分。",
    ]
    (run / "report_cn.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
