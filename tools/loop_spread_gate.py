"""WITHDRAWN as #3's gate (Luiz, 2026-11-07). It cannot pass a correct loop: the legacy range comes
from runs that share fixation 0, the loop is chaotic, and the unchanged legacy loop fails it when its
IPD moves by 0.2 um (docs/runs/issue-3/part2/). Kept because that diagnosis runs it; do not use it as
an equivalence test.

Issue #3's statistical gate: fov_render runs of the active loop against the legacy spread.

    python tools/loop_spread_gate.py --spread docs/runs/issue-3/spread.json --config class_small \
        --legacy RUN_a RUN_b RUN_c --runs NATIVE_1 NATIVE_2 NATIVE_3 --fov3d /path/to/fov-3d-vision [--json OUT]

For each of the 17 Phase C numbers (spread.json), the gate is
    |mean(fov_render runs) - mean(legacy runs)| <= range(legacy runs) = max - min
(Luiz, #3: a single run inside min..max of three would fail by chance half the time per number).
A number whose legacy range is 0 must be matched exactly. The fov_render numbers come from
fov-3d-vision's own active_eval.py, run on the fov_render runs together, as part 1 did.

Also, from #9 (kept by Luiz's decision there): at fixation 0, where every run's poses are the same,
each eye's mean RGB may differ from each legacy run's by at most 1e-5 (an event otherwise).
Exit status 1 if any number fails.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

EVENT = 1e-5


def phase_c(runs, fov3d: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.run([sys.executable, str(Path(fov3d) / "tools" / "active_eval.py"), *map(str, runs), "--out", d],
                           capture_output=True, text=True)
        cj = Path(d) / "compare.json"
        if not cj.is_file():
            raise SystemExit(f"active_eval wrote no compare.json (exit {p.returncode}):\n{(p.stdout + p.stderr)[-2000:]}")
        out = json.loads(cj.read_text())
    return out


def fixation0_rgb(run: Path) -> dict:
    return {e: float(np.load(run / e / "f000" / "samples.npz")["value"].astype(np.float64).mean()) for e in ("L", "R")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spread", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="the fov_render runs (native mode)")
    ap.add_argument("--legacy", nargs="*", default=[], help="the legacy runs, for the fixation-0 RGB test")
    ap.add_argument("--fov3d", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    sp = json.loads(Path(a.spread).read_text())["configs"][a.config]["phase_c"]
    ev = phase_c(a.runs, a.fov3d)
    rows = ev["runs"]
    table, fails = {}, []
    for name, s in sp.items():
        leg = [float(v) for v in s["runs"].values()]
        new = [float(r[name]) for r in rows]
        dm, rng = abs(float(np.mean(new)) - float(np.mean(leg))), float(max(leg) - min(leg))
        ok = dm <= rng if rng > 0 else dm == 0.0
        table[name] = {"legacy": leg, "fov_render": new, "legacy_mean": float(np.mean(leg)),
                       "fov_render_mean": float(np.mean(new)), "abs_mean_difference": dm, "legacy_range": rng,
                       "ratio": dm / rng if rng > 0 else (0.0 if dm == 0 else float("inf")), "ok": ok}
        if not ok:
            fails.append(name)
    rgb = {}
    for r in a.legacy:
        for n in a.runs:
            la, nb = fixation0_rgb(Path(r)), fixation0_rgb(Path(n))
            for e in ("L", "R"):
                d = abs(la[e] - nb[e])
                rgb[f"{Path(r).name}~{Path(n).name}:{e}"] = d
                if d > EVENT:
                    fails.append(f"fixation-0 RGB {Path(r).name} vs {Path(n).name} eye {e}: {d:.2e}")
    rep = {"schema": "fov-render-issue3-gate-v1", "config": a.config, "runs": a.runs, "legacy": a.legacy,
           "active_eval_fails": ev.get("fails"), "numbers": table, "fixation0_abs_mean_drgb": rgb,
           "event_threshold": EVENT, "fails": fails, "pass": not fails and not ev.get("fails")}
    text = json.dumps(rep, indent=1, sort_keys=True) + "\n"
    if a.json:
        Path(a.json).write_text(text)
    worst = max(table.items(), key=lambda kv: kv[1]["ratio"])
    print(f"[loop-gate] {a.config}: {'PASS' if rep['pass'] else 'FAIL'}; {17 - len([f for f in fails if f in table])}/17 numbers "
          f"within the legacy range (worst {worst[0]}: {worst[1]['ratio']:.2f} of the range); fixation-0 RGB max "
          f"{max(rgb.values()) if rgb else float('nan'):.2e}; active_eval fails {ev.get('fails')}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
