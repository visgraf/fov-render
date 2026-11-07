"""Issue #9 diagnostic (not a gate): compare the per-fixation point RGB of Playground-2 runs with
the frozen run, and say which fixations and objects are off.

    python docs/runs/issue-9/compare_runs.py RUN [RUN ...] --out docs/runs/issue-9/runs-vs-frozen.json

RUN: a run folder under f3d-vision/previews (fr2-baseline, fr2-baseline-wheel, ...). Geometry and
Object Index are asserted equal to the frozen run's (check_equivalence compares them; RGB it does
not). Object names come from the Classroom's legacy catalogue (ids 1..234).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIEWS = Path("/home/lvelho/rd/f3d-vision/previews")
FROZEN = "playground-sparse-lattice"
BIG = 0.1


def names() -> dict:
    p = HERE / "catalog-classroom-legacy.json"       # render_cell.py's catalog.json (legacy numbering)
    if not p.is_file():
        return {}
    return {e["id"]: e["name"] for e in json.loads(p.read_text())["instances"] if e["via_instance_of"] is None}


def compare(run: str, nm: dict) -> dict:
    rows, by_obj = [], collections.Counter()
    for f in sorted((PREVIEWS / FROZEN / "fixations").iterdir()):
        b, w = np.load(f / "points.npz"), np.load(PREVIEWS / run / "fixations" / f.name / "points.npz")
        if not (np.array_equal(b["xyz_h"], w["xyz_h"]) and np.array_equal(b["instance_id_oracle"], w["instance_id_oracle"])):
            raise SystemExit(f"{run} {f.name}: geometry or ids differ from the frozen run")
        d = w["rgb"].astype(np.float64) - b["rgb"].astype(np.float64)
        big = np.abs(d).max(-1) > BIG
        ids = b["instance_id_oracle"]
        for i, c in collections.Counter(ids[big].tolist()).items():
            by_obj[nm.get(i, f"id {i}")] += c
        rows.append({"n": int(f.name[4:]), "points": int(len(ids)), "max_abs": float(np.abs(d).max()),
                     "mean_signed": float(d.mean()), "points_gt_0.1": int(big.sum())})
    off = [r["n"] for r in rows if r["points_gt_0.1"]]
    return {"run": run, "fixations": len(rows), "fixations_with_points_gt_0.1": off,
            "points_gt_0.1": int(sum(r["points_gt_0.1"] for r in rows)),
            "max_abs": max(r["max_abs"] for r in rows), "objects_gt_0.1": dict(by_obj.most_common(12)),
            "per_fixation": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    nm = names()
    res = {"schema": "fov-render-issue9-runs-vs-frozen-v1", "frozen": FROZEN, "threshold": BIG,
           "runs": [compare(r, nm) for r in a.runs]}
    for r in res["runs"]:
        print(f"{r['run']:28s} max {r['max_abs']:.4f}  points > {BIG}: {r['points_gt_0.1']:6d}  "
              f"fixations: {r['fixations_with_points_gt_0.1']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
