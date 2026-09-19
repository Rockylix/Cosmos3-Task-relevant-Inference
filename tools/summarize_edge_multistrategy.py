"""Verify matched timing artifacts and estimate logical block matmul FLOPs.

Multiply-add counts as two operations. Not hardware-instruction or end-to-end
pipeline FLOPs; excludes elementwise operations, tile padding and recomputation.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from safetensors import safe_open

D, I, Q, KV, B = 2048, 9216, 2048, 1024, 28
N, P, NS, SPATIAL = 3093, 373, 1845, 340
CHECKPOINT = Path("/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID")
LABELS = {
    "dense": "Dense eager",
    "asi": "ASI B6 + compile/Graph",
    "toca": "ToCa D/C/D/C r=0.25 + compile/Graph",
    "worldcache": "WorldCache D/D/D/C + compile/Graph",
    "c3ache": "C3ache period=2 amortized + compile/Graph",
}


def full(n, u):
    return 2 * n * D * (2 * Q + 2 * KV) + 4 * n * D * I + 4 * n * (n + u) * Q


def und(u):
    return 2 * u * D * (2 * Q + 2 * KV) + 4 * u * D * I + 4 * (u * (u + 1) // 2) * Q


def cached(u, fresh):
    return 4 * P * D * Q + 4 * N * D * KV + 4 * P * (N + u) * Q + 4 * (P + fresh) * D * I


def verify_shapes():
    index_path = CHECKPOINT / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    shapes = {
        "mlp.up_proj.weight": [I, D],
        "mlp.down_proj.weight": [D, I],
        "mlp_moe_gen.up_proj.weight": [I, D],
        "mlp_moe_gen.down_proj.weight": [D, I],
        "self_attn.to_q.weight": [Q, D],
        "self_attn.to_k.weight": [KV, D],
        "self_attn.to_v.weight": [KV, D],
        "self_attn.to_out.weight": [D, Q],
    }
    for block in range(B):
        for suffix, shape in shapes.items():
            key = f"layers.{block}.{suffix}"
            with safe_open(CHECKPOINT / index[key], framework="pt", device="cpu") as file:
                assert list(file.get_slice(key).get_shape()) == shape, key
    return {"index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(), "verified_shapes": B * len(shapes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    reports = {s: json.loads((args.input / s / "results.json").read_text()) for s in LABELS}
    dense_report = reports["dense"]
    for name, report in reports.items():
        assert not (args.input / name / "failure.json").exists(), name
        assert report["input_sha256"] == dense_report["input_sha256"]
        assert report["kwargs"] == dense_report["kwargs"]
        assert report["benchmark_sha256"] == dense_report["benchmark_sha256"]
        assert report["boundary"] == dense_report["boundary"]
        assert report["pad_for_cuda_graphs"] is False and report["finite"] is True
        assert report["dense_layout_records"] == dense_report["dense_layout_records"]
        for filename, expected_hash in report["source_sha256"].items():
            assert hashlib.sha256((Path(report["source"]) / filename).read_bytes()).hexdigest() == expected_hash
        for mode, timing in report["timing"].items():
            assert timing["n"] == report["repeats"] == 20, (name, mode)
        if name != "dense":
            assert report["dense_matches_pristine_checkout_exact"] and report["changed_seed_gate"]
            assert all(
                any("cudaGraphLaunch" in k and n > 0 for k, n in events.items())
                for events in report["runtime_audit"].values()
            )
    layouts = dense_report["dense_layout_records"]
    assert len(layouts) == 8 and all(r["_num_full_tokens"] == N for r in layouts)
    uc, uu = layouts[0]["_num_causal_tokens"], layouts[1]["_num_causal_tokens"]
    assert (uc, uu) == (158, 19)
    asi = reports["asi"]["last_controller"]["strategy_graph"]
    assert (asi["dense_stack_count"], asi["sparse_stack_count"], asi["token_budget"]) == (1, 7, 184)
    assert reports["asi"]["asi_main_lse"]
    tc = reports["toca"]["last_controller"]["strategy_graph"]
    cfg = tc["config"]
    assert cfg["full_steps"] == [0, 2] and cfg["fresh_ratio"] == 0.25
    assert cfg["layer_slope"] == 0.5 and cfg["cfg_selection"] == "independent"
    assert cfg["spatial_bonus"] == 0 and cfg["attention_backend"] == "joint"
    assert tc["layout"]["num_gen_tokens"] == N
    wc = reports["worldcache"]["last_controller"]["strategy_graph"]
    assert (wc["full_forwards"], wc["cache_forwards"]) == (6, 2)
    for mode, counts in [("refresh_graph", (8, 0)), ("hit_graph", (4, 4))]:
        row = reports["c3ache"]["last_controller"][mode]
        assert (row["dense_forwards"], row["cache_hits"]) == counts
    assert reports["c3ache"]["config"]["refresh_period"] == 2
    fresh = [int(0.25 * (1.5 - block / 27) * 2720) for block in range(B)]
    assert tc["q"]["computed_rows"] == 4 * B * (N + P)
    assert tc["o"]["computed_rows"] == tc["q"]["computed_rows"]
    assert tc["kv"]["computed_rows"] == 8 * B * N
    assert tc["mlp"]["computed_rows"] == 4 * B * N + 4 * sum(P + f for f in fresh)
    prefill = B * (und(uc) + und(uu))
    pair = B * (full(N, uc) + full(N, uu))
    dense_flops = prefill + 4 * pair
    # B6 main attention supplies LSE. Only local aligned QK remains extra.
    score = B * 2 * 32 * SPATIAL * Q
    old_extra_attention = B * 4 * 32 * (N + uc) * Q
    totals = {
        "dense": dense_flops,
        "asi": prefill + B * (full(N, uc) + 3 * full(NS, uc) + 4 * full(NS, uu)) + score,
        "toca": prefill + 2 * pair + 2 * sum(cached(uc, f) + cached(uu, f) for f in fresh),
        "worldcache": prefill + 3 * pair,
        "c3ache": ((prefill + 4 * pair) + (prefill + 2 * pair)) / 2,
    }
    rows = []
    for name, report in reports.items():
        mode = "dense_eager" if name == "dense" else "amortized_graph" if name == "c3ache" else "strategy_graph"
        timing = report["timing"][mode]
        paired = report["timing"]["dense_eager"]["median_s"]
        rows.append(
            dict(
                strategy=LABELS[name],
                mean_s=timing["mean_s"],
                median_s=timing["median_s"],
                p90_s=timing["p90_s"],
                paired_dense_median_s=paired,
                measured_speedup=paired / timing["median_s"],
                logical_matmul_tflops=totals[name] / 1e12,
                baseline_flops_percent=100 * totals[name] / dense_flops,
                equal_throughput_speedup=dense_flops / totals[name],
            )
        )
    result = dict(
        rows=rows,
        scope=__doc__,
        weight_check=verify_shapes(),
        dimensions=dict(
            hidden=D,
            mlp=I,
            query=Q,
            kv=KV,
            blocks=B,
            dense_gen=N,
            sparse_gen=NS,
            protected=P,
            spatial=SPATIAL,
            conditional_und=uc,
            unconditional_und=uu,
        ),
        asi_scoring_flops=score,
        removed_extra_attention_flops=old_extra_attention,
        toca_fresh_per_block=fresh,
        und_prefill_flops=prefill,
        input_sha256=dense_report["input_sha256"],
        source_heads={k: v["head"] for k, v in reports.items()},
        raw_reports={k: hashlib.sha256((args.input / k / "results.json").read_bytes()).hexdigest() for k in reports},
    )
    (args.input / "comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    with (args.input / "comparison.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
