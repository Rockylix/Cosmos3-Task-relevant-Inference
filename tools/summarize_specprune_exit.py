"""Summarize the complete, unfiltered matched ten-task test."""

import argparse
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

from run_specprune_exit_smoke10 import TASKS, rows


def csv_file(path, values):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    manifest_path = run / ("sparse_manifest.json" if (run / "sparse_manifest.json").exists() else "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    root = Path(__file__).resolve().parents[1]
    for name, sha in manifest["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Runtime source changed: {name}")
    episodes, chunks, blocks = [], [], []
    for strategy in ("dense", "specprune"):
        eps = rows(run / strategy / "simulator/episode_results.jsonl")
        reqs = rows(run / strategy / "requests.jsonl")
        if len(eps) != 10 or {r["task_name"] for r in eps} != set(TASKS):
            raise ValueError("Ten task results are incomplete or duplicated")
        for task in TASKS:
            e = next(r for r in eps if r["task_name"] == task)
            rs = [r for r in reqs if r["task"] == task]
            if [r["chunk"] for r in rs] != list(range(1, len(rs) + 1)) or not rs:
                raise ValueError("Incomplete request sequence")
            sparse = strategy == "specprune"
            saved = st.mean(r["mean_saved_gen_tokens_per_block_forward"] for r in rs) if sparse else 0
            episodes.append(
                dict(
                    strategy=strategy,
                    task=task,
                    success=e["success"],
                    score=e["score"],
                    steps=e["episode_step"],
                    chunks=len(rs),
                    mean_final_future_kept=st.mean(r["selected_future_tokens"] for r in rs) if sparse else 2720,
                    mean_saved_gen_tokens_per_block=saved,
                    mean_future_computed_per_block=2720 - saved,
                    paired_action_mse=st.mean(r["paired_action_error"]["mse"] for r in rs) if sparse else 0,
                    paired_action_cosine=st.mean(r["paired_action_error"]["cosine"] for r in rs) if sparse else 1,
                    paired_future_latent_cosine=st.mean(r["paired_vision_error"]["cosine"] for r in rs)
                    if sparse
                    else 1,
                    paired_future_latent_relative_l2=st.mean(r["paired_vision_error"]["relative_l2"] for r in rs)
                    if sparse
                    else 0,
                )
            )
            for r in rs:
                chunks.append(
                    dict(
                        strategy=strategy,
                        task=task,
                        chunk=r["chunk"],
                        seed=r["request_seed"],
                        final_future_kept=r.get("selected_future_tokens", 2720),
                        saved_gen_per_block=r.get("mean_saved_gen_tokens_per_block_forward", 0),
                        action_mse=r.get("paired_action_error", {}).get("mse"),
                        action_cosine=r.get("paired_action_error", {}).get("cosine"),
                        future_latent_cosine=r.get("paired_vision_error", {}).get("cosine"),
                        future_latent_relative_l2=r.get("paired_vision_error", {}).get("relative_l2"),
                    )
                )
                if sparse:
                    if len(r["solver_steps"]) != 4 or any(x["patch_tokens"] != 3060 for x in r["solver_steps"]):
                        raise AssertionError("Solver lost full latent state")
                    calls = json.loads(
                        (
                            run / strategy / "artifacts" / task / f"chunk_{r['chunk']:03d}" / "token_rows.json"
                        ).read_text()
                    )
                    if len(calls) != 224:
                        raise AssertionError("Unexpected block calls")
                    for call in calls:
                        blocks.append(dict(task=task, chunk=r["chunk"], **call))
    csv_file(run / "episodes.csv", episodes)
    csv_file(run / "chunks.csv", chunks)
    csv_file(run / "block_tokens.csv", blocks)
    summary = {}
    for strategy in ("dense", "specprune"):
        es = [r for r in episodes if r["strategy"] == strategy]
        summary[strategy] = dict(
            success=sum(r["success"] for r in es),
            episodes=len(es),
            score=st.mean(r["score"] for r in es),
            mean_steps=st.mean(r["steps"] for r in es),
            mean_success_steps=st.mean(r["steps"] for r in es if r["success"])
            if any(r["success"] for r in es)
            else None,
            task_macro_action_mse=st.mean(r["paired_action_mse"] for r in es),
            task_macro_action_cosine=st.mean(r["paired_action_cosine"] for r in es),
            task_macro_latent_cosine=st.mean(r["paired_future_latent_cosine"] for r in es),
            task_macro_latent_relative_l2=st.mean(r["paired_future_latent_relative_l2"] for r in es),
            task_macro_saved_gen_per_block=st.mean(r["mean_saved_gen_tokens_per_block"] for r in es),
        )
    (run / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    text = [
        "# 观测 SpecPrune + Early-exit hidden：十任务结果",
        "",
        "Dense 与实验策略各十次、每任务一次；不重试、不筛选失败。以下 success、score 使用原始独立字段。",
        "",
        "| 策略 | Success | Mean score | 平均完成/终止步数 | 成功 episode 平均步数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for s, r in summary.items():
        text.append(
            f"| {s} | {r['success']}/10 | {r['score']:.4f} | {r['mean_steps']:.1f} | {r['mean_success_steps']} |"
        )
    text += [
        "",
        "| 任务 | Dense success / score / steps | Sparse success / score / steps | Action MSE | Future latent cos | Future latent rel-L2 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for task in TASKS:
        d, s = [next(r for r in episodes if r["task"] == task and r["strategy"] == k) for k in ("dense", "specprune")]
        text.append(
            f"| {task} | {d['success']} / {d['score']} / {d['steps']} | {s['success']} / {s['score']} / {s['steps']} | {s['paired_action_mse']:.6f} | {s['paired_future_latent_cosine']:.5f} | {s['paired_future_latent_relative_l2']:.5f} |"
        )
    text += [
        "",
        "## 指标口径",
        "",
        "action 与 future latent 误差均对每个 Sparse chunk 的同输入、同 seed Dense reference 计算，先逐 chunk，再逐 task 聚合。Dense reference 不用于机器人动作、mask 或历史。跨策略闭环各走自身轨迹，不能直接将两条轨迹的 action 相减。",
        "Action 指标为 32×8 externalized 输出，排除 condition action；future latent 只统计 L1–L8，完整张量展平。未将这些指标当成任务成功率的替代。",
        "逐层真实计算 token 数见 block_tokens.csv；完整头与 UniPC 仍处理全部 latent。final_future_kept 是到最后 block 仍在计算的数量，不是 solver latent 数。",
        "本轮含显式 attention 重算、finite 检查、额外 Dense reference 和输出保存，不是优化后的稳定单 chunk 速度 benchmark，不给加速结论。",
        "",
        "## 方法与复现边界",
        "",
        "每 chunk step0 conditional 在 L0 上生成逐层空间计划，随后该计划按层用于所有 step/CFG 分支的 L1–L8。Local/Global 为 L0 Q→instruction K；逐层排名为 action Q→L0 K。L0、文本、action 始终计算。",
        "每次 forward 独立保存退出层 hidden，在最终 RMSNorm/输出头前补回；不把旧 hidden 带到下一步，不缓存 velocity。CFG/UniPC 始终保留完整状态。",
        "动作控制器关闭。静态参数沿用前次迁移：Local32、Global40、B13/B27，Dynamic阈值0.986、低变化上限313。动态更新B14/19/24，剪枝B10/15/20/25，EMA0.2，总序列保留率0.9。采用观测虚拟序列预算及至少60空间位置；同分按原位置ID稳定排序。",
        "保留官方B10在首次B14评分之前剪枝的调度：B10分数为零，不能将该次选择解释为动作重要性排名。32层→28层没有新增重采样，本轮保留可用的原动态层编号；Global保持此前已确认的B13/B27。",
        "本实验是有明确WAM补全机制的迁移，非完整原论文复现，也不预设VLA方法一定失败。L0的hidden仍通过joint attention读取future上下文；“观测评分”不表示网络对future噪声完全独立。静态候选少于60时不补点；达到动态下限后不再更新重要性和置信度，后续 attention 采集仅供诊断。",
        "",
        "## 文件",
        "",
        "[配置及源码SHA256](manifest.json) · [逐任务](episodes.csv) · [逐chunk](chunks.csv) · [逐层token](block_tokens.csv)",
        "若存在 sparse_manifest.json，则记录 Sparse 启动前修正停止条件后的门禁/指纹；原 manifest.json 保留 Dense 启动时版本，两阶段未重跑任何已完成 episode。",
        "chunk3 的真实观测评分与逐层mask：specprune/artifacts/<task>/chunk_003/observation_scores.npz；完整预测latent和action：output.pt。未保存仿真视频。",
    ]
    (run / "report_cn.md").write_text("\n".join(text) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
