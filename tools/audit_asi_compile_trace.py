"""Read-only, synchronized chunk audit including Runtime/Driver graph launches.

Graph kernel correlation is intentionally not used for per-layer attribution:
the layer counts below describe actual host launch APIs, not GPU kernel counts.
"""

import argparse
import collections
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

ROOT = "asi.chunk.generate"
LAYER = re.compile(r"asi\.step([0-3])\.(conditional|unconditional)\.B(\d{2})$")
PID_MASK = ~((1 << 24) - 1)


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


def deduplicate_launches(runtime, driver):
    """Pair nested Runtime->Driver APIs once; preserve driver-only launches.

    Correlation IDs may differ across these API domains. Same-thread temporal
    containment is therefore the primary evidence for an underlying call pair.
    Do not deduplicate distinct launches merely because correlation IDs match.
    """
    used = set()
    launches, pairs = [], []
    for r in runtime:
        candidates = [
            (i, d)
            for i, d in enumerate(driver)
            if i not in used and d["globalTid"] == r["globalTid"] and r["start"] <= d["start"] <= d["end"] <= r["end"]
        ]
        if len(candidates) > 1:
            raise ValueError("Ambiguous graph launch: multiple Driver launches within one Runtime API")
        entry = dict(r, api_sources=["runtime"])
        if candidates:
            i, d = candidates[0]
            used.add(i)
            entry["api_sources"].append("driver")
            pairs.append({"runtime": r, "driver": d})
        launches.append(entry)
    launches.extend(dict(d, api_sources=["driver"]) for i, d in enumerate(driver) if i not in used)
    return sorted(launches, key=lambda r: r["start"]), pairs


def parse(path):
    with closing(sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        strings = dict(conn.execute("SELECT id,value FROM StringIds")) if "StringIds" in tables else {}
        nvtx = [dict(r) for r in conn.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL")]
        for r in nvtx:
            r["name"] = r.get("text") or strings.get(r.get("textId"), "")
        roots = [r for r in nvtx if r["name"] == ROOT]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one {ROOT} root, got {len(roots)}")
        root = roots[0]
        start, end, tid = root["start"], root["end"], root["globalTid"]
        pid = tid & PID_MASK
        if end <= start:
            raise ValueError("Invalid root interval")
        layers = [
            r
            for r in nvtx
            if (r["globalTid"] & PID_MASK) == pid
            and start <= r["start"] < r["end"] <= end
            and LAYER.fullmatch(r["name"])
        ]
        launches_by_source = {}
        for source in ("runtime", "driver"):
            table = f"CUPTI_ACTIVITY_KIND_{source.upper()}"
            rows = []
            if table in tables:
                for row in conn.execute(f"SELECT * FROM {table}"):
                    r = dict(row)
                    name = strings.get(r.get("nameId"), r.get("name", ""))
                    if "GraphLaunch" not in name:
                        continue
                    if (r["globalTid"] & PID_MASK) == pid and start <= r["start"] < end:
                        if r["end"] > end:
                            raise ValueError("Graph launch API escapes synchronized root")
                        rows.append(dict(r, name=name))
            launches_by_source[source] = rows
        launches, pairs = deduplicate_launches(launches_by_source["runtime"], launches_by_source["driver"])
        kernels = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? AND start>=? AND end<=?",
                (pid, start, end),
            )
        ]
        crossing = conn.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? AND start<? AND end>? AND (start<? OR end>?)",
            (pid, end, start, start, end),
        ).fetchone()[0]
        if crossing:
            raise ValueError(f"{crossing} GPU kernels cross root boundary; require CUDA synchronization")
    per_layer = collections.Counter({r["name"]: 0 for r in layers})
    unassigned = []
    for launch in launches:
        parents = [
            r for r in layers if r["globalTid"] == launch["globalTid"] and r["start"] <= launch["start"] < r["end"]
        ]
        if len(parents) > 1:
            raise ValueError("Ambiguous overlapping layer ranges for graph launch")
        if parents:
            launch["layer"] = parents[0]["name"]
            per_layer[launch["layer"]] += 1
        else:
            launch["layer"] = None
            unassigned.append(launch)
    step_branch = collections.Counter()
    for label, count in per_layer.items():
        match = LAYER.fullmatch(label)
        step_branch[f"step{match[1]}.{match[2]}"] += count
    busy = union_ns((k["start"], k["end"]) for k in kernels)
    expected_labels = {
        f"asi.step{s}.{branch}.B{b:02d}"
        for s in range(4)
        for branch in ("conditional", "unconditional")
        for b in range(28)
    }
    first = {f"asi.step0.conditional.B{b:02d}" for b in range(28)}
    return {
        "nvtx_filter": ROOT,
        "root_match_count": 1,
        "wall_ms": (end - start) / 1e6,
        "kernel_count": len(kernels),
        "kernel_sum_ms": sum(k["end"] - k["start"] for k in kernels) / 1e6,
        "kernel_busy_union_ms": busy / 1e6,
        "wall_minus_kernel_busy_ms": (end - start - busy) / 1e6,
        "raw_graph_launch_api_counts": {s: len(rows) for s, rows in launches_by_source.items()},
        "deduplicated_runtime_driver_pairs": len(pairs),
        "actual_graph_launch_count": len(launches),
        "layer_nvtx_range_count": len(layers),
        "graph_launches_by_layer": dict(sorted(per_layer.items())),
        "graph_launches_by_step_branch": dict(sorted(step_branch.items())),
        "layers_with_graph_launch": sum(n > 0 for n in per_layer.values()),
        "all_224_layers_observed_with_graph": expected_labels.issubset(per_layer)
        and all(per_layer[label] > 0 for label in expected_labels),
        "first_conditional_28_layers_observed_with_graph": first.issubset(per_layer)
        and all(per_layer[label] > 0 for label in first),
        "unassigned_graph_launch_count": len(unassigned),
        "graph_launches": launches,
        "notes": [
            "Kernel statistics are root-only, scoped by process and synchronized GPU time containment.",
            "Per-layer counts use actual host graph launch API timestamps, not Graph kernel correlations.",
            "Runtime/Driver duplicate launch APIs are paired by same-thread temporal containment.",
            "Wall minus kernel busy is not pure CPU overhead; it includes copies and synchronization.",
            "224-layer evidence requires all 4 steps x 2 branches x 28 named layer ranges to contain launches.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = parse(args.sqlite)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("graph_launches", "graph_launches_by_layer", "notes")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
