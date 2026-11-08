"""Step 1 gates: the foveated OSL camera, built by fov_render, renders what the legacy renderer
renders, and passes the legacy checker. Needs bpy. Runs on the generated calibration room
(Classroom gates are Phase 2, on the workstation)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .. import record
from ..rig import Gaze, Rig
from ..sensors import Foveated
from .common import FIXTURES, KNOWN, Result, run_blender_script

STEP = "1"
GAZE = (20.0, -5.0)
SPP = 16


def calib_room(ctx) -> str:
    out = Path(ctx["work"]) / "scenes" / "calib_room.blend"
    if not out.is_file():
        run_blender_script(str(FIXTURES / "make_calib_room.py"), ["--out", str(out)])
    return str(out)


def render_new_foveated(ctx, folder: Path, *, e2_shader: float = 2.0, seed: int = 0, profile: str = "small") -> dict:
    """Render one cyclopean fixation through fov_render and write fix.exr + meta.json in the
    format the legacy check_foveated reads."""
    import bpy
    from ..bl import cameras, config, engine, scene as S
    h = S.open_scene(calib_room(ctx), keep_passes=("combined", "z", "normal", "position"))
    cfg = config.preset("fov_fixation", profile=profile, device=ctx["device"], spp=SPP, seed=seed)
    model = Foveated().model(profile)
    backend = config.apply(h.scene, cfg, model.width, model.height, h.view_layer)
    cam = cameras.build(h.scene, model, cfg)
    if e2_shader != model.e2:
        cam.data.cycles_custom["E2"] = e2_shader
    pose = Rig().pose(h.head, Gaze.per_eye(C=GAZE), "C")
    cameras.pose(cam, pose)
    folder.mkdir(parents=True, exist_ok=True)
    secs = engine.render_once(h.scene, SPP, seed, str(folder / "fix.exr"))
    meta = {"warp": {"E2_deg": model.e2, "e_max_deg": model.emax, "s0_deg": model.s0_deg, "raster": model.n},
            "camera_position_m": pose.centre_w.tolist(), "camera_forward": (pose.R_wb @ [0, 0, -1.0]).tolist(),
            "device": backend, "spp": SPP, "seed": seed, "render_seconds": round(secs, 4),
            "settings": config.readback(h.scene, cam, h.view_layer), "eye_pose": pose.as_dict()}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def host_python() -> str:
    """A Python with numpy and OpenEXR for host-side legacy checkers: $FOVRENDER_HOST_PYTHON, else this
    interpreter when it has OpenEXR (the bpy wheel venv), else python3 on PATH."""
    import shutil
    if os.environ.get("FOVRENDER_HOST_PYTHON"):
        return os.environ["FOVRENDER_HOST_PYTHON"]
    try:
        import OpenEXR  # noqa: F401
        import sys
        return sys.executable
    except ImportError:
        return shutil.which("python3") or "python3"


def _check_foveated(folder: Path) -> dict:
    """Run fov-3d-vision's check_foveated (a host-side script) on a render folder; return its report."""
    import subprocess
    p = subprocess.run([host_python(), str(FIXTURES / "check_foveated.py"), str(folder)],
                       capture_output=True, text=True)
    text = p.stdout
    if "{" not in text:
        raise RuntimeError(f"check_foveated produced no report ({p.returncode}): {p.stderr[-1500:]}")
    report = json.loads(text[text.index("{"):text.rindex("}") + 1])
    report["exit_code"] = p.returncode
    return report


def legacy_checker_on_new_render(ctx) -> Result:
    """check_foveated (depth = ray distance, warp = formula, mask, ray saving) on a fov_render image."""
    work = Path(ctx["work"]) / "step1"
    good = _check_foveated(_folder(render_new_foveated(ctx, work / "new"), work / "new"))
    bad = _check_foveated(_folder(render_new_foveated(ctx, work / "control_e2", e2_shader=2.04), work / "control_e2"))
    return Result(STEP, "legacy check_foveated on a fov_render fixation", good["checks_failed"] == [],
                  bool(bad["checks_failed"]), {"report": good, "control_failures": bad["checks_failed"],
                                               "control": "shader E2 = 2.04 while the record says 2.0"})


def _folder(_meta, folder):
    return folder


def new_equals_legacy_render(ctx) -> Result:
    """Same scene, gaze, spp, seed, on CPU: fov_render's fixation against the legacy fixation, frozen in
    known/step1_legacy_fixation.npz (issue #16). The difference must stay at the pose-precision level
    (fov_render's float64 pose against the legacy float32 one): Position within 1e-4 m, Combined within
    5 % of what a seed change does. Control: that seed change itself.

    Up to v0.1.0 this gate also rendered with the legacy pose matrix and required equality bit for bit;
    that needed the legacy code, and CPU renders are not promised bitwise across machines (the frozen
    file comes from the Chat sandbox), so the gate keeps the tolerance part (PLAN, issue #16)."""
    try:
        import OpenEXR
    except ImportError:
        return Result(STEP, "fov_render fixation equals legacy fixation", False, None, skipped="OpenEXR not installed")
    cpu = dict(ctx, device="CPU")
    work = Path(ctx["work"]) / "step1" / "cpu"
    render_new_foveated(cpu, work / "new")
    render_new_foveated(cpu, work / "new_seed1", seed=1)

    def chans(p):
        with OpenEXR.File(str(p)) as f:
            return {k: np.asarray(v.pixels, dtype=np.float64) for part in f.parts for k, v in part.channels.items()}
    z = np.load(KNOWN / "step1_legacy_fixation.npz")
    leg = {k[3:]: z[k].astype(np.float64) for k in z.files if k.startswith("ch:")}
    own, s1 = (chans(work / d / "fix.exr") for d in ("new", "new_seed1"))
    keys = sorted(leg)

    def maxdiff(a, b):
        return {k: float(np.nanmax(np.abs(a[k] - b[k]))) for k in keys}
    own_d = maxdiff(leg, own)
    seed_d = maxdiff(own, s1)
    nm = json.loads((work / "new" / "meta.json").read_text())
    pose_diff = max(float(np.abs(z["camera_position_m"] - nm["camera_position_m"]).max()),
                    float(np.abs(z["camera_forward"] - nm["camera_forward"]).max()))
    comb = [k for k in keys if "Combined" in k][0]
    pos = [k for k in keys if "Position" in k]
    ok = (set(leg) == set(own) and pose_diff < 1e-6 and max(own_d[k] for k in pos) < 1e-4
          and own_d[comb] < 0.05 * seed_d[comb])
    return Result(STEP, "fov_render fixation equals the legacy fixation (frozen, pose precision)", ok,
                  seed_d[comb] > 0, {"channels": keys, "max_abs_diff_vs_legacy": own_d,
                                     "pose_diff_float32_vs_float64": pose_diff, "control_seed1_max_diff": seed_d,
                                     "spp": SPP, "gaze": GAZE, "device": "CPU", "requested_device": ctx["device"],
                                     "backend": nm["device"]})


def record_from_render(ctx) -> Result:
    """The SampleRecord built from a fov_render fixation: one entry per disc pixel, directions
    agree with Position within a pixel, distances equal the Depth pass (ray distance)."""
    from ..bl import engine
    work = Path(ctx["work"]) / "step1"
    folder = work / "record"
    from ..bl import cameras, config, scene as S
    h = S.open_scene(calib_room(ctx))
    cfg = config.RenderConfig(profile="small", device=ctx["device"], spp=SPP,
                              passes=("combined", "z", "position", "object_index"))
    model = Foveated().model("small")
    config.apply(h.scene, cfg, model.width, model.height, h.view_layer)
    cam = cameras.build(h.scene, model, cfg)
    pose = Rig().pose(h.head, Gaze.per_eye(C=GAZE), "C")
    cameras.pose(cam, pose)
    folder.mkdir(parents=True, exist_ok=True)
    engine.render_once(h.scene, SPP, 0, str(folder / "fix.exr"))
    p = engine.read_passes(str(folder / "fix.exr"))
    rec = record.build(model, p, pose.centre_w, pose.R_wb, h.head.R_wh, eye="C", look_id="f0", access="oracle")
    cons = record.sensor_consistency(rec)
    z = p["z"].reshape(-1)[rec.raster_index]
    hit = np.isfinite(rec.distance_m)
    dz = float(np.abs(z[hit] - rec.distance_m[hit]).max())
    rim_px_deg = model.s0_deg * (1 + model.emax / model.e2)
    # control: a record built with the wrong pose (mirrored yaw) disagrees with Position
    bad_pose = Rig().pose(h.head, Gaze.per_eye(C=(-GAZE[0], GAZE[1])), "C")
    bad = record.sensor_consistency(record.build(model, p, bad_pose.centre_w, bad_pose.R_wb, h.head.R_wh,
                                                 eye="C", look_id="x", access="oracle"))
    ok = (rec.n_samples == model.rays_per_eye and hit.all() and cons["p999_angle_deg"] <= rim_px_deg
          and dz < 1e-4)
    return Result(STEP, "SampleRecord from a foveated render agrees with Position and Depth", ok,
                  bad["p999_angle_deg"] > 10 * rim_px_deg,
                  {"n_samples": rec.n_samples, "rays_per_eye": model.rays_per_eye, "hits": int(hit.sum()),
                   "consistency": cons, "rim_spacing_deg": rim_px_deg, "position_vs_depth_max_m": dz,
                   "control_mirrored_pose": bad})


CHECKS = [legacy_checker_on_new_render, new_equals_legacy_render, record_from_render]
