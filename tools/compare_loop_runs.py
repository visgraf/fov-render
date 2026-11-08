"""Compare two runs of fov-3d-vision's active loop bit for bit (issue #3, the exact test).

    python tools/compare_loop_runs.py RUN_A RUN_B [--fov3d PATH] [--json OUT]

Compared, ignoring only timings and the run's own folder name:
    loop.json        every step (gaze, vergence, cells, every metric) and the final settings
    pairs.json       every pair (both eyes' gazes, centres, predictions, in-session measurements)
    *.npz            every array of every file: L/ and R/ samples, field/, belief.npz (bitwise)
    fix*.exr         every channel of every fixation render, both eyes (bitwise)
    Phase C table    with --fov3d: active_eval.py run on both runs together, its per-run numbers
Exit status 1 if anything differs. Timing keys (seconds, wall, rendered_at, warm-up) are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IGNORE = ("second", "wall", "rendered_at", "warmup", "warm_up")


def _strip(x, path=()):
    if isinstance(x, dict):
        return {k: _strip(v, path + (k,)) for k, v in x.items()
                if not any(s in k.lower() for s in IGNORE) and k not in ("out", "blend_dir")}
    if isinstance(x, list):
        return [_strip(v, path) for v in x]
    return x


def _diff(a, b, where="", out=None, limit=20):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{where}.{k}: only in {'A' if k in a else 'B'}")
            else:
                _diff(a[k], b[k], f"{where}.{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{where}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{where}[{i}]", out, limit)
    elif not (a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))):
        out.append(f"{where}: {a!r} vs {b!r}")
    return out


def compare(a: Path, b: Path, fov3d: str | None = None) -> dict:
    rep = {"run_a": str(a), "run_b": str(b), "json": {}, "npz": {"files": 0, "differ": []},
           "exr": {"files": 0, "differ": []}}
    for name in ("loop.json", "pairs.json"):
        ja, jb = (_strip(json.loads((r / name).read_text())) for r in (a, b))
        rep["json"][name] = _diff(ja, jb)
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.suffix in (".npz", ".exr") and "_fov_render" not in p.parts)
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.suffix in (".npz", ".exr") and "_fov_render" not in p.parts)
    rep["files_only_in_one"] = sorted(str(x) for x in set(files) ^ set(files_b))
    from fov_render.exr_lite import read_uncompressed_exr
    for rel in files:
        if rel not in files_b:
            continue
        if rel.suffix == ".npz":
            rep["npz"]["files"] += 1
            za, zb = np.load(a / rel), np.load(b / rel)
            bad = sorted(k for k in set(za.files) | set(zb.files)
                         if k not in za.files or k not in zb.files or za[k].dtype != zb[k].dtype
                         or za[k].shape != zb[k].shape or za[k].tobytes() != zb[k].tobytes())
            if bad:
                rep["npz"]["differ"].append({"file": str(rel), "arrays": bad})
        else:
            rep["exr"]["files"] += 1
            ca, cb = read_uncompressed_exr(str(a / rel)), read_uncompressed_exr(str(b / rel))
            bad = sorted(k for k in set(ca) | set(cb) if k not in ca or k not in cb or
                         np.asarray(ca[k]).tobytes() != np.asarray(cb[k]).tobytes())
            if bad:
                rep["exr"]["differ"].append({"file": str(rel), "channels": bad[:8]})
    if fov3d:
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.run([sys.executable, str(Path(fov3d) / "tools" / "active_eval.py"), str(a), str(b), "--out", d],
                               capture_output=True, text=True)
            rep["active_eval_exit"] = p.returncode
            cj = Path(d) / "compare.json"
            if cj.is_file():
                rows = json.loads(cj.read_text())["runs"]
                ra, rb = (_strip({k: v for k, v in r.items() if k not in ("run", "name", "path", "label")}) for r in rows)
                rep["phase_c"] = {"a": ra, "b": rb, "differences": _diff(ra, rb)}
            else:
                rep["phase_c"] = {"error": (p.stdout + p.stderr)[-1500:]}
    rep["identical"] = (not any(rep["json"].values()) and not rep["npz"]["differ"] and not rep["exr"]["differ"]
                        and not rep["files_only_in_one"] and rep["npz"]["files"] > 0 and rep["exr"]["files"] > 0
                        and not rep.get("phase_c", {}).get("differences") and "error" not in rep.get("phase_c", {}))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--fov3d", default=None, help="a fov-3d-vision checkout, to compare active_eval's Phase C table too")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    rep = compare(Path(a.run_a).resolve(), Path(a.run_b).resolve(), a.fov3d)
    text = json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n"
    if a.json:
        Path(a.json).write_text(text)
    print(f"[compare-loop] {'IDENTICAL' if rep['identical'] else 'DIFFERENT'}: loop.json {len(rep['json']['loop.json'])} "
          f"and pairs.json {len(rep['json']['pairs.json'])} differences; npz {len(rep['npz']['differ'])}/{rep['npz']['files']} "
          f"and exr {len(rep['exr']['differ'])}/{rep['exr']['files']} files differ; only in one: {len(rep['files_only_in_one'])}; "
          f"Phase C differences: {len(rep.get('phase_c', {}).get('differences', [])) if 'phase_c' in rep else 'not compared'}")
    return 0 if rep["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
