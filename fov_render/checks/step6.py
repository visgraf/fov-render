"""Step 6 gates (issue #3): fov-3d-vision's foveated active loop on a fov_render Session.

    6.1  compat.fov3d_gaze_matrix equals fov-3d-vision's render_foveated.gaze_matrix bit for bit
    6.2  the exact test, short: the legacy loop and tools/active_loop_session.py --mode exact give
         identical runs on the calibration room on CPU (every file, every Phase C number)
    6.3  native mode: fixation 0's records, built by fov_render, agree with the legacy loop's

6.2 and 6.3 need a fov-3d-vision checkout ($FOVRENDER_FOV3D_ROOT). The full runs (50 fixations,
the Classroom on OPTIX) are Code's, on the workstation (issue #3).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import ROOT, Result, load_legacy

STEP = "6"
FIX = 3


def gaze_matrix_bitwise(ctx) -> Result:
    import bpy
    from .. import compat
    from ..rig import Gaze, Rig
    from ..bl import scene as S
    from .step1 import calib_room
    rf = load_legacy("fov", "render_foveated.py")
    bpy.ops.wm.open_mainfile(filepath=calib_room(ctx))
    eye = bpy.data.objects["EYE"]
    eye.rotation_euler = (1.61, 0.013, -0.42)          # a head that is not axis-aligned
    bpy.context.view_layer.update()
    rng = np.random.default_rng(3)
    cases = [(float(y), float(p), off) for y, p in rng.uniform([-60, -40], [60, 40], size=(40, 2))
             for off in ((-0.0315, 0.0, 0.0), (0.0315, 0.0, 0.0), (0.0, 0.0, 0.0))]
    bad = sum(compat.fov3d_gaze_matrix(eye, y, p, off) != rf.gaze_matrix(eye, y, p, off) for y, p, off in cases)
    # control: fov_render's own float64 pose, stored in float32, is not bit-equal for every case
    head = S.resolve_head(bpy.context.scene, "EYE", None)[0]
    rig = Rig(0.063)
    own = 0
    for y, p, off in cases:
        e = "L" if off[0] < 0 else "R" if off[0] > 0 else "C"
        m = rig.pose(head, Gaze.per_eye(**{e: (y, p)}), e).matrix_world
        legacy = np.array(rf.gaze_matrix(eye, y, p, off), dtype=np.float32)
        own += m.astype(np.float32).tobytes() != legacy.tobytes()
    return Result(STEP, "6.1 compat.fov3d_gaze_matrix equals fov-3d-vision's gaze_matrix bit for bit", bad == 0, own > 0,
                  {"cases": len(cases), "not_equal": bad, "control_fov_render_pose_not_bitwise": own})


def _fov3d():
    r = os.environ.get("FOVRENDER_FOV3D_ROOT")
    return r if r and (Path(r) / "tools" / "active_loop.py").is_file() else None


def _loop(cmd_tool, out: Path, fov3d: str, blend: str, extra=()):
    shutil.rmtree(out, ignore_errors=True)
    cmd = [sys.executable, str(ROOT / "tools" / cmd_tool), "--", "--fov3d", fov3d, *extra, "--out", str(out),
           "--blend", blend, "--profile", "small", "--policy", "coverage", "--fixations", str(FIX), "--device", "CPU"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    (out.parent / f"{out.name}.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd_tool} {' '.join(extra)} failed ({p.returncode}): {(p.stdout + p.stderr)[-1500:]}")


def _compare(a: Path, b: Path, fov3d: str) -> dict:
    sys.path.insert(0, str(ROOT / "tools"))
    import compare_loop_runs as C
    return C.compare(a, b, fov3d)


def loop_exact(ctx) -> Result:
    name = f"6.2 the active loop on fov_render equals the legacy loop bit for bit (calibration room, CPU, {FIX} fixations)"
    fov3d = _fov3d()
    if fov3d is None:
        return Result(STEP, name, False, None, skipped="set FOVRENDER_FOV3D_ROOT to a fov-3d-vision checkout")
    from .step1 import calib_room
    work, room = Path(ctx["work"]) / "step6", calib_room(ctx)
    _loop("run_fov3d_loop.py", work / "legacy", fov3d, room)
    _loop("active_loop_session.py", work / "exact", fov3d, room, ("--mode", "exact"))
    ipd = repr(float(np.nextafter(0.063, 1.0)))
    _loop("active_loop_session.py", work / "control_ipd_ulp", fov3d, room, ("--mode", "exact", "--ipd", ipd))
    same = _compare(work / "legacy", work / "exact", fov3d)
    ctrl = _compare(work / "legacy", work / "control_ipd_ulp", fov3d)
    summ = lambda r: {"identical": r["identical"], "loop_json": r["json"]["loop.json"][:3], "npz_differ": len(r["npz"]["differ"]),
                      "npz_files": r["npz"]["files"], "exr_differ": len(r["exr"]["differ"]), "exr_files": r["exr"]["files"],
                      "phase_c_differences": len(r.get("phase_c", {}).get("differences", []))}
    return Result(STEP, name, same["identical"], not ctrl["identical"],
                  {"exact": summ(same), "control_ipd_plus_one_ulp": summ(ctrl), "fov3d": fov3d})


def loop_native(ctx) -> Result:
    name = "6.3 native mode: fixation 0's records from fov_render agree with the legacy loop's"
    fov3d = _fov3d()
    if fov3d is None:
        return Result(STEP, name, False, None, skipped="set FOVRENDER_FOV3D_ROOT to a fov-3d-vision checkout")
    from .step1 import calib_room
    work, room = Path(ctx["work"]) / "step6", calib_room(ctx)
    if not (work / "legacy" / "loop.json").is_file():
        _loop("run_fov3d_loop.py", work / "legacy", fov3d, room)
    _loop("active_loop_session.py", work / "native", fov3d, room, ("--mode", "native"))
    out, ok = {}, True
    for e in ("L", "R"):
        a = dict(np.load(work / "legacy" / e / "f000" / "samples.npz"))
        b = dict(np.load(work / "native" / e / "f000" / "samples.npz"))
        ka = np.lexsort((a["raster_index"][:, 1], a["raster_index"][:, 0]))
        kb = np.lexsort((b["raster_index"][:, 1], b["raster_index"][:, 0]))
        same_px = a["raster_index"][ka].tobytes() == b["raster_index"][kb].tobytes()
        da, db = a["direction"][ka].astype(np.float64), b["direction"][kb].astype(np.float64)
        da /= np.linalg.norm(da, axis=1, keepdims=True)
        db /= np.linalg.norm(db, axis=1, keepdims=True)
        # atan2(|a x b|, a . b): arccos of a dot product near 1 reads a norm error of 1e-7 as 0.03 deg
        ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(da, db), axis=1), (da * db).sum(-1)))
        hit = (a["distance"][ka] < 1e9) & (b["distance"][kb] < 1e9)
        dd = np.abs(a["distance"][ka][hit].astype(np.float64) - b["distance"][kb][hit])
        fr = np.abs(b["footprint"][kb] / a["footprint"][ka] - 1.0)
        drgb = abs(float(a["value"].astype(np.float64).mean() - b["value"].astype(np.float64).mean()))
        good = (same_px and float(np.percentile(ang, 99.9)) < 1e-4 and float(np.percentile(dd, 99)) < 1e-3
                and float(np.percentile(fr, 99)) < 1e-3 and drgb <= 1e-5
                and float(np.abs(a["origin"] - b["origin"]).max()) < 1e-6)
        ok &= good
        out[e] = {"same_pixels": same_px, "samples": int(len(ka)), "direction_p999_deg": float(np.percentile(ang, 99.9)),
                  "distance_p99_m": float(np.percentile(dd, 99)), "footprint_rel_p99": float(np.percentile(fr, 99)),
                  "mean_rgb_abs_difference": drgb, "origin_difference_m": float(np.abs(a["origin"] - b["origin"]).max()),
                  "ok": good}
    gz = [json.loads((work / r / "loop.json").read_text())["steps"] for r in ("legacy", "native")]
    same_gazes = [s["dir_head"] for s in gz[0]] == [s["dir_head"] for s in gz[1]]
    # control: the native run is not the exact one (the comparison sees fov_render's own geometry)
    ctrl = not _compare(work / "legacy", work / "native", fov3d)["identical"]
    return Result(STEP, name, ok, ctrl, {"fixation0": out, "same_gazes_all_fixations": same_gazes,
                                         "control_native_differs_from_legacy": ctrl})


CHECKS = [gaze_matrix_bitwise, loop_exact, loop_native]
