"""Offline logical Transformer matmul FLOPs; no GPU/model inference.

Run from this checkout with .venv/bin/python tools/estimate_cache_strategy_flops.py.
This deliberately excludes padding, elementwise work and all non-block modules.
"""
import csv
import json
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/cache_compile_adapt_final_v1"
CHECKPOINT = Path("/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID")
D, I, Q, KV, B = 2048, 9216, 2048, 1024, 28
N, P, NS = 3093, 373, 1845
# tokenize_caption gives 156/17 for the benchmark caption/empty caption.
# add_special_tokens + pack_text_tokens append EOS and BOV, not BOS.
UC, UU = 158, 19


def full(n, u):
    return 2 * n * D * (2 * Q + 2 * KV) + 4 * n * D * I + 4 * n * (n + u) * Q


def und(u):
    return 2 * u * D * (2 * Q + 2 * KV) + 4 * u * D * I + 4 * (u * (u + 1) // 2) * Q


def cached(u, fresh):
    return 4 * P * D * Q + 4 * N * D * KV + 4 * P * (N + u) * Q + 4 * (P + fresh) * D * I


def main():
    index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    shapes = {
        "mlp.up_proj.weight": [I, D], "mlp.down_proj.weight": [D, I],
        "mlp_moe_gen.up_proj.weight": [I, D], "mlp_moe_gen.down_proj.weight": [D, I],
        "self_attn.to_q.weight": [Q, D], "self_attn.to_k.weight": [KV, D],
        "self_attn.to_v.weight": [KV, D], "self_attn.to_out.weight": [D, Q],
    }
    for block in range(B):
        for suffix, expected in shapes.items():
            key = f"layers.{block}.{suffix}"
            with safe_open(CHECKPOINT / index[key], framework="pt", device="cpu") as f:
                assert list(f.get_slice(key).get_shape()) == expected, key
    data = {s: json.loads((OUT / s / "results.json").read_text()) for s in ("asi", "toca", "worldcache", "c3ache")}
    asi = data["asi"]["last_controller"]["strategy_graph"]
    assert (asi["dense_stack_count"], asi["sparse_stack_count"], asi["token_budget"]) == (1, 7, 184)
    tc = data["toca"]["last_controller"]["strategy_graph"]
    cfg = tc["config"]
    assert cfg["full_steps"] == [0, 2] and cfg["fresh_ratio"] == .25
    assert cfg["layer_slope"] == .5 and tc["layout"]["num_gen_tokens"] == N
    assert cfg["attention_backend"] == "joint"
    fresh = [int(.25 * (1.5 - l / 27) * 2720) for l in range(B)]
    prefill = B * (und(UC) + und(UU))
    pair = B * (full(N, UC) + full(N, UU))
    dense = prefill + 4 * pair
    profile = B * (4 * 32 * (N + UC) * Q + 2 * 32 * 340 * Q)
    totals = {
        "Dense Baseline": dense,
        "ASI Core80+Stable104": prefill + B * (full(N, UC) + 3 * full(NS, UC) + 4 * full(NS, UU)) + profile,
        "ToCa D/C/D/C r=0.25": prefill + 2 * pair + 2 * sum(cached(UC, f) + cached(UU, f) for f in fresh),
        "WorldCache D/D/D/C": prefill + 3 * pair,
        "C3ache period=2 均摊": ((prefill + 4 * pair) + (prefill + 2 * pair)) / 2,
    }
    measured = {r["strategy"]: r for r in json.loads((OUT / "comparison.json").read_text())["rows"]}
    rows = []
    for strategy, flops in totals.items():
        m = measured.get(strategy, {})
        rows.append(dict(strategy=strategy, logical_matmul_tflops=flops / 1e12,
                         baseline_percent=flops / dense * 100, reduction_percent=100 * (1-flops/dense),
                         equal_throughput_speedup=dense / flops, measured_speedup=m.get("speedup", 1.),
                         measured_median_s=m.get("median_s")))
    evidence = dict(scope="logical Transformer block matmul FLOPs; multiply-add=2; causal UND triangular",
                    dimensions=dict(hidden=D, intermediate=I, query_width=Q, kv_width=KV, blocks=B,
                                    dense_gen=N, sparse_gen=NS, protected=P, conditional_und=UC, unconditional_und=UU),
                    und_prefill_flops=prefill, asi_profile_matmul_flops=profile,
                    toca_fresh_per_block=fresh, rows=rows)
    (OUT / "flops_comparison.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    with (OUT / "flops_comparison.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
