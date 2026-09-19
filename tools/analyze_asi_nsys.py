"""Read-only NVTX/correlation-based parser for one warmed asi.chunk.generate.

Uses the kernel-optimizer skill's scoped analysis protocol, adapted to current
Nsight StringIds schema and asynchronous CPU launch/GPU execution attribution.
"""

import argparse
import collections
import json
import sqlite3
from pathlib import Path


def union_ns(intervals):
    total, left, right = 0, None, None
    for a, b in sorted(intervals):
        if left is None:
            left, right = a, b
        elif a > right:
            total += right - left
            left, right = a, b
        else:
            right = max(right, b)
    return total + (right - left if left is not None else 0)


def parse(path):
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    strings = dict(conn.execute("SELECT id,value FROM StringIds"))
    nvtx = [
        dict(r) for r in conn.execute("SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL")
    ]
    for r in nvtx:
        r["name"] = r["text"] or strings.get(r["textId"], "")
    roots = [r for r in nvtx if r["name"] == "asi.chunk.generate"]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one warmed root, got {len(roots)}")
    root = roots[0]
    start, end, tid = root["start"], root["end"], root["globalTid"]
    pid = tid & ~((1 << 24) - 1)
    duration = end - start
    sub = [r for r in nvtx if r["globalTid"] == tid and start <= r["start"] <= r["end"] <= end]
    runtime = [
        dict(r)
        for r in conn.execute(
            "SELECT start,end,correlationId,globalTid,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid=? AND start>=? AND start<?",
            (tid, start, end),
        )
    ]
    by_corr = collections.defaultdict(list)
    apis = collections.defaultdict(lambda: dict(count=0, cpu_ms=0))
    for r in runtime:
        by_corr[r["correlationId"]].append(r)
        a = apis[strings.get(r["nameId"], str(r["nameId"]))]
        a["count"] += 1
        a["cpu_ms"] += (r["end"] - r["start"]) / 1e6
    kernels = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? AND start>=? AND end<=?", (pid, start, end)
        )
    ]
    if not kernels:
        raise ValueError("No in-scope GPU kernels")
    # Fail if future callers accidentally remove the final CUDA synchronize.
    escaped = [
        dict(r)
        for r in conn.execute("SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=?", (pid,))
        if r["correlationId"] in by_corr and not (start <= r["start"] <= r["end"] <= end)
    ]
    if escaped:
        raise ValueError(f"Kernels launched in root escaped its synchronized boundary: {escaped[:5]}")
    inclusive = collections.defaultdict(lambda: dict(host_ms=0, gpu_sum_ms=0, kernels=0, range_count=0))
    exclusive = collections.defaultdict(lambda: dict(gpu_sum_ms=0, kernels=0))
    top = collections.defaultdict(lambda: dict(gpu_sum_ms=0, kernels=0))
    for r in sub:
        a = inclusive[r["name"]]
        a["host_ms"] += (r["end"] - r["start"]) / 1e6
        a["range_count"] += 1
    unmatched = []
    for k in kernels:
        ms = (k["end"] - k["start"]) / 1e6
        name = strings.get(k["demangledName"], str(k["demangledName"]))
        top[name]["gpu_sum_ms"] += ms
        top[name]["kernels"] += 1
        candidates = by_corr.get(k["correlationId"], [])
        if len(candidates) != 1:
            unmatched.append(dict(correlation=k["correlationId"], candidates=len(candidates)))
            continue
        launch = candidates[0]["start"]
        parents = [r for r in sub if r["start"] <= launch < r["end"]]
        # Inclusive counts once per named stage, even if identical names nest.
        for label in {r["name"] for r in parents}:
            inclusive[label]["gpu_sum_ms"] += ms
            inclusive[label]["kernels"] += 1
        leaf = min(parents, key=lambda r: r["end"] - r["start"])["name"]
        exclusive[leaf]["gpu_sum_ms"] += ms
        exclusive[leaf]["kernels"] += 1
    if unmatched:
        raise ValueError(f"Unmatched kernels: {unmatched[:5]} (total {len(unmatched)})")
    for table in (inclusive, exclusive, top):
        for v in table.values():
            v["gpu_sum_pct_chunk_wall"] = v["gpu_sum_ms"] / (duration / 1e6) * 100
    busy = union_ns([(k["start"], k["end"]) for k in kernels])
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    copies = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        copies = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE globalPid=? AND start>=? AND end<=?", (pid, start, end)
            )
        ]
    conn.close()
    return dict(
        nvtx_filter="asi.chunk.generate",
        match_count=1,
        wall_ms=duration / 1e6,
        kernel_count=len(kernels),
        kernel_sum_ms=sum(k["end"] - k["start"] for k in kernels) / 1e6,
        kernel_busy_union_ms=busy / 1e6,
        wall_minus_kernel_busy_ms=(duration - busy) / 1e6,
        note="Wall minus kernel busy also contains copies and synchronization; not a pure CPU overhead measure. Subrange GPU assignment uses runtime launch correlation, not GPU timestamp containment. Inclusive ranges overlap; do not sum parents and children.",
        unmatched_kernels=len(unmatched),
        cuda_graph_launches=sum(v["count"] for k, v in apis.items() if "GraphLaunch" in k),
        gpu_memcpy_count=len(copies),
        gpu_memcpy_bytes=sum(k.get("bytes", 0) for k in copies),
        cuda_apis=dict(apis),
        inclusive_ranges=dict(inclusive),
        exclusive_ranges=dict(exclusive),
        top_kernels=sorted([dict(name=k, **v) for k, v in top.items()], key=lambda r: -r["gpu_sum_ms"])[:20],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = parse(args.sqlite)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("cuda_apis", "inclusive_ranges", "exclusive_ranges", "top_kernels")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
