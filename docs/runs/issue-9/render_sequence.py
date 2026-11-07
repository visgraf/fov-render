"""Issue #9 diagnostic (not a gate): render a sequence of Playground-2 fixations in ONE session, as
the explorer does (persistent data on, preset ab1a), and record per-object means of Combined.

    blender -b CLASSROOM --python-exit-code 1 -P docs/runs/issue-9/render_sequence.py -- --mode cmdline --first 250 --last 266 --out DIR
    .venv/bin/python docs/runs/issue-9/render_sequence.py -- --mode open --first 250 --last 266 --out DIR

Same look per fixation as render_cell.py (gaze from the frozen run, calibration byte-identical to
it). --session server runs the looks through a background Session (fov_render.server), exactly as
tools/gfe_session.py; in_process (default) calls the handler directly.
Writes DIR/fix-NNNN/raw_{L,R}.exr (with --keep-exr), DIR/sequence.json (per fixation and eye: mean
Combined per Object Index and over the image).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import render_cell as RC  # noqa: E402

WATCH = (18, 15, 228, 229)          # Cube.014, Cube.011 (paintedBlind), woodWindow.002, .003 (issue #9)


def stats(exr: Path) -> dict:
    import numpy as np
    from fov_render.bl import engine
    p = engine.read_passes(str(exr))
    comb = np.asarray(p["combined"], np.float64)[..., :3]
    oid = np.rint(p["object_index"]).astype(int)
    out = {"image_mean": float(comb.mean())}
    for i in WATCH:
        m = oid == i
        if m.any():
            out[str(i)] = [float(comb[m].mean()), int(m.sum())]
    return out


def main() -> None:
    import bpy
    import numpy as np
    from fov_render import Gaze, HeadPose, Rig, Session, compat, preset
    from fov_render.bl import env
    from fov_render.sensors import Perspective
    env.prepare()
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cmdline", "open"])
    ap.add_argument("--session", default="in_process", choices=["in_process", "server"])
    ap.add_argument("--launcher", default="python", help="server sessions: blender or python")
    ap.add_argument("--device", default="OPTIX", choices=["CPU", "OPTIX"])
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    ap.add_argument("--frozen", default=str(RC.F3D / "previews" / "playground-sparse-lattice"))
    ap.add_argument("--blend", default=str(RC.F3D / "scenes" / "classroom" / "classroom_eye.blend"))
    ap.add_argument("--keep-exr", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    loaded = bpy.data.filepath
    if a.mode == "cmdline" and not (loaded and os.path.samefile(loaded, a.blend)):
        raise SystemExit(f"--mode cmdline needs the Classroom on the command line; loaded: {loaded!r}")
    frozen = Path(a.frozen)
    traj = json.loads((frozen / "trajectory.json").read_text())["trajectory"]
    cfg = preset("ab1a", device=a.device, spp=a.spp)
    rig = Rig(ipd_m=RC.IPD_M, roll=RC.ROLL)
    sensor = Perspective()
    model = sensor.model(cfg.profile)
    kw = dict(access="oracle", instance_ids="legacy", prepare=sensor, formats=("f3d_ab1a",), workdir=out / "session")
    if a.session == "server":
        s = Session(a.blend, cfg, rig, launcher=a.launcher, **kw)
    else:
        s = Session.in_process(cfg, rig, blend=None if a.mode == "cmdline" else a.blend, **kw)
    r = s.ready
    head = HeadPose(np.asarray(r["head"]["head_R_wh"], float), np.asarray(r["head"]["head_origin_w_m"], float))
    rows = []
    for n in range(a.first, a.last + 1):
        t = traj[n - 1]
        assert int(t["n"]) == n
        gaze = Gaze.yaw_pitch(float(t["yaw_deg"]), float(t["pitch_deg"]), RC.VERGENCE_M)
        calib = compat.f3d_json(compat.f3d_calibration(cfg.profile, model, rig, head, gaze))
        same = calib == (frozen / "fixations" / f"fix-{n:04d}" / "calibration.json").read_text()
        look = s.look(sensor, ("L", "R"), gaze, spp=a.spp, seeds=RC.SEEDS, out_dir=out / f"fix-{n:04d}")
        row = {"n": n, "kind": t.get("kind"), "calibration_equal": same,
               "eyes": {e: stats(p) for e, p in look.exr.items()}}
        if not a.keep_exr:
            for p in look.exr.values():
                p.unlink()
        rows.append(row)
        print(f"[issue-9 seq] fix {n} {t.get('kind')} calib {same} L {json.dumps(row['eyes']['L'])}", flush=True)
    rc = s.close()
    (out / "sequence.json").write_text(json.dumps({"schema": "fov-render-issue9-sequence-v1", "mode": a.mode,
                                                   "session": a.session, "launcher": getattr(s, "launcher", None),
                                                   "device": a.device, "spp": a.spp, "first": a.first, "last": a.last,
                                                   "ready_opened": r["scene"]["file"].get("opened"), "close_rc": rc,
                                                   "rows": rows}, indent=1, default=str) + "\n")
    if not all(x["calibration_equal"] for x in rows):
        raise SystemExit("a calibration is not the frozen run's")


if __name__ == "__main__":
    from fov_render.bl.env import loud
    loud(lambda: (main(), sys.stdout.flush(), os._exit(0)), "[issue-9 seq]")
