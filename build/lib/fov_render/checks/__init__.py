"""The gates of the migration plan, runnable in either interpreter:

    python -m fov_render.checks [--steps 1 2 3 4] [--work DIR] [--device CPU]      (bpy wheel)
    blender -b --python-exit-code 1 -P tools/run_checks.py -- [same options]       (workstation)

Every check has a known answer and a control that must fail. Exit status 1 if any check fails.
Checks that need bpy are SKIPPED without it, never passed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .common import ROOT, Result, have_bpy, write_json


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def run(steps, work: Path, device: str) -> list[Result]:
    from . import legacy_integrity, step2
    plan = [("0", legacy_integrity.CHECKS, False), ("2", step2.CHECKS, False)]
    if have_bpy():
        from . import step1, step3, step4
        ctx = {"work": work, "device": device}
        def bind(f):
            g = lambda: f(ctx)  # noqa: E731
            g.__name__ = f.__name__
            return g
        plan += [(m.__name__[-1], [bind(f) for f in m.CHECKS], True) for m in (step1, step3, step4)]
    else:
        plan += [(s, [], True) for s in ("1", "3", "4")]
    out = []
    for step, fns, needs_bpy in sorted(plan, key=lambda x: x[0]):
        if step not in steps and step != "0":
            continue
        if needs_bpy and not fns:
            out.append(Result(step, f"step {step} (needs bpy)", False, None, skipped="bpy not importable"))
            continue
        for fn in fns:
            t0 = time.perf_counter()
            try:
                r = fn()
            except Exception as e:  # noqa: BLE001
                r = Result(step, getattr(fn, "__name__", "check"), False, None,
                           {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]})
            r.detail["seconds"] = round(time.perf_counter() - t0, 2)
            out.append(r)
            print(f"[fov-render check] {r.status:4}  step {r.step}  {r.name}  ({r.detail['seconds']} s)", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fov_render.checks")
    ap.add_argument("--steps", nargs="*", default=["1", "2", "3", "4"])
    ap.add_argument("--work", default=str(ROOT / "previews" / "checks"))
    ap.add_argument("--device", default="CPU", choices=["CPU", "OPTIX", "CUDA"])
    ap.add_argument("--json", default=None, help="where to write the results (default: WORK/results.json)")
    a = ap.parse_args(argv)
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    results = run(set(a.steps), work, a.device)
    bv = None
    if have_bpy():
        import bpy
        bv = bpy.app.version_string
    summary = {"schema": "fov-render-checks-v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "commit": _git_head(), "python": platform.python_version(), "blender": bv, "host": platform.node(),
               "device": a.device, "passed": sum(r.status == "PASS" for r in results),
               "failed": sum(r.status == "FAIL" for r in results), "skipped": sum(r.status == "SKIP" for r in results),
               "results": [r.as_dict() for r in results]}
    write_json(Path(a.json) if a.json else work / "results.json", summary)
    print(f"[fov-render check] passed={summary['passed']} failed={summary['failed']} skipped={summary['skipped']}")
    return 1 if summary["failed"] else 0
