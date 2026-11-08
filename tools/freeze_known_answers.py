"""Freeze what the legacy code gives the comparisons of steps 1-4 into fov_render/checks/known/ (issue #16).

    FOVRENDER_F3D_ROOT=... FOVRENDER_FOV3D_ROOT=... python tools/freeze_known_answers.py -- --work DIR   (bpy wheel, CPU)

The legacy code comes from checks.pinned: the files of known/pins.json, rebuilt from the two
checkouts' git history, the same bytes legacy/ held until v0.1.0. Each frozen answer is produced by
the same call the check made before the clean-up, so the checks can compare with the file instead.

    step2.npz                 warp tables (a fixed sample of pixels plus the whole mask), the three
                              historical pose formulas over the gaze grid, make_calibration's R_hc and
                              centres, the legacy sine-rule triangulation
    step1_legacy_fixation.npz the legacy foveated fixation of the calibration room, on CPU (channels)
    step3_readbacks.json      the settings each of the 7 legacy paths renders with (13 comparisons)
    step4.json                the legacy head readers and f3d-vision's object numbering
    freeze.json               when, from which commits, with which Blender, on which machine
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KNOWN = ROOT / "fov_render" / "checks" / "known"     # main() may point it elsewhere (--out)


def step2() -> dict:
    import numpy as np
    from fov_render import warp
    from fov_render.checks.common import load_legacy
    from fov_render.checks.step2 import CALIB_CASES, GAZES, TRIANGULATION_P
    lw = load_legacy("fov", "warp.py")
    lr = load_legacy("fov", "rig.py")
    lg = load_legacy("f3d", "fsg_geometry.py")
    out = {"gazes": np.array(GAZES, float)}
    for prof, s0 in (("small", 0.1), ("full", 0.05)):
        n = lw.raster_size(s0, 2.0, 45.0)
        a = lw.raster_samples(n, 2.0, 45.0)
        flat = np.flatnonzero(a["inside"].reshape(-1))
        idx = np.sort(np.random.default_rng(0).choice(flat, size=4000, replace=False))
        d = a["direction_cam"].reshape(-1, 3)[idx]
        rc, ins = lw.raster_of_direction(d, n, 2.0, 45.0)
        out.update({f"{prof}_n": np.int64(n), f"{prof}_idx": idx, f"{prof}_direction_cam": d,
                    f"{prof}_footprint": a["footprint"].reshape(-1)[idx], f"{prof}_rc": np.asarray(rc),
                    f"{prof}_inside_rc": np.asarray(ins), f"{prof}_mask": np.packbits(a["inside"].reshape(-1)),
                    f"{prof}_mask_size": np.int64(a["inside"].size)})
    out["gaze_rotation"] = np.array([lr.gaze_rotation(y, p) for y, p in GAZES])
    out["camera_rotation_h"] = np.array([lg.camera_rotation_h(lg.gaze_direction(y, p)) for y, p in GAZES])
    try:
        from types import SimpleNamespace
        from mathutils import Matrix
        rf = load_legacy("fov", "render_foveated.py")
        eye = SimpleNamespace(matrix_world=Matrix.Identity(4))
        out["gaze_matrix"] = np.array([np.array(rf.gaze_matrix(eye, y, p).to_3x3()) for y, p in GAZES])
    except ImportError:
        raise SystemExit("run under the bpy wheel or Blender: render_foveated.gaze_matrix needs mathutils")
    out["head_R_wh"], out["head_origin_w"] = np.asarray(lg.HEAD_R_WH, float), np.asarray(lg.HEAD_ORIGIN_W, float)
    rhc, cen = [], []
    for roll, frame, yaw, pitch in CALIB_CASES:
        c = lg.make_calibration("small", yaw, pitch, vergence_distance=2.10, tangent_frame=frame)
        rhc.append([e["R_hc"] for e in c["eyes"]])
        cen.append([e["centre_h_m"] for e in c["eyes"]])
    out["calib_R_hc"], out["calib_centre_h"] = np.array(rhc, float), np.array(cen, float)
    from fov_render import geometry
    from fov_render.rig import Rig
    rig = Rig()
    P = np.array(TRIANGULATION_P)
    cl, cr = rig.centre_h("L"), rig.centre_h("R")
    tl, _ = geometry.epipolar((P - cl) / np.linalg.norm(P - cl))
    tr, _ = geometry.epipolar((P - cr) / np.linalg.norm(P - cr))
    out["legacy_triangulate_m"] = np.float64(lr.triangulate(np.degrees(tl), np.degrees(tr), rig.ipd_m)[0])
    np.savez_compressed(KNOWN / "step2.npz", **out)
    return {"step2.npz": sorted(out)}


def render_legacy_foveated(ctx, folder: Path, profile: str = "small") -> None:
    """fov-3d-vision's render_foveated.py on the calibration room (moved from checks/step1.py, issue #16)."""
    from fov_render.bl import env
    from fov_render.checks.common import LEGACY, call_with_argv, load_legacy
    from fov_render.checks.step1 import GAZE, SPP, calib_room
    env.prepare()                     # the legacy script assumes the Blender binary's own oslquery
    rf = load_legacy("fov", "render_foveated.py")
    call_with_argv(rf.main, "render_foveated.py",
                   ["--blend", calib_room(ctx), "--out", str(folder), "--profile", profile, "--device", ctx["device"],
                    "--spp", SPP, "--yaw", GAZE[0], "--pitch", GAZE[1], "--shader", str(LEGACY["fov"] / "foveated_camera.osl")])


def step1(ctx) -> dict:
    import numpy as np
    import OpenEXR
    from fov_render.checks.step1 import GAZE, SPP
    cpu = dict(ctx, device="CPU")
    folder = Path(ctx["work"]) / "freeze" / "legacy_fixation"
    render_legacy_foveated(cpu, folder)
    with OpenEXR.File(str(folder / "fix.exr")) as f:
        ch = {k: np.asarray(v.pixels, dtype=np.float32) for part in f.parts for k, v in part.channels.items()}
    meta = json.loads((folder / "meta.json").read_text())
    np.savez_compressed(KNOWN / "step1_legacy_fixation.npz", **{"ch:" + k: v for k, v in ch.items()},
                        camera_position_m=np.array(meta["camera_position_m"]), camera_forward=np.array(meta["camera_forward"]))
    return {"step1_legacy_fixation.npz": {"channels": sorted(ch), "gaze": GAZE, "spp": SPP, "device": meta.get("device")}}


# ----------------------------------------------------------------------------- step 3: the legacy sides
# Moved here from fov_render/checks/step3.py in issue #16: each runs one legacy path on the room and
# returns the settings it rendered with (fov_render's readback).

from fov_render.checks.common import call_with_argv, load_legacy  # noqa: E402
from fov_render.checks.step3 import _open, _room  # noqa: E402

def _capture_first_render(fn) -> dict:
    """Run a legacy main() and read back the settings at the moment of its first render."""
    import bpy
    from fov_render.bl.config import readback
    got = {}

    @bpy.app.handlers.persistent               # survives the legacy script's own file load
    def handler(scene, *_):
        if "rb" not in got:
            got["rb"] = readback(scene, scene.camera, bpy.context.view_layer)
    bpy.app.handlers.render_pre.append(handler)
    try:
        fn()
    finally:
        if handler in bpy.app.handlers.render_pre:
            bpy.app.handlers.render_pre.remove(handler)
    if "rb" not in got:
        raise RuntimeError("the legacy script did not render")
    return got["rb"]


def legacy_fov_fixation(ctx, tmp):
    from fov_render.bl import env
    env.prepare()
    rf = load_legacy("fov", "render_foveated.py")
    return _capture_first_render(lambda: call_with_argv(rf.main, "render_foveated.py", [
        "--blend", _room(ctx), "--out", tmp / "fix", "--profile", "small", "--device", "CPU", "--spp", 1]))


def legacy_preview360(ctx, tmp):
    p = load_legacy("fov", "preview360.py")
    return _capture_first_render(lambda: call_with_argv(p.main, "preview360.py", [
        "--blend", _room(ctx), "--out", tmp / "pano", "--width", 64, "--spp", 1, "--device", "CPU", "--no-backface"]))


def legacy_fsg(ctx, tmp):
    from fov_render.bl.config import readback
    import bpy
    fr = load_legacy("fov", "fsg_render.py")
    c = fr.make_calibration("small", *fr.CASE_GAZE["fronto"])
    be = fr.BlenderBackend("CPU", False)
    be.prepare("fronto", c, tmp, 4)
    return readback(be.scene, be.cameras[0], bpy.context.view_layer)


def legacy_oracle1(ctx, tmp, profile="small", narrowed=False):
    from fov_render.bl.config import readback
    import bpy
    scene, vl = _open(_room(ctx))
    co = load_legacy("f3d", "classroom_oracle1_render.py")
    lg = load_legacy("f3d", "fsg_geometry.py")
    _eye, _note, o, r = co._head_pose()
    if narrowed:
        ar = load_legacy("f3d", "active_bootstrap/ab1a_render.py", "active_bootstrap")
        ar.SP.DEVICE = "CPU"                         # the spec pins OPTIX; the Chat sandbox has no GPU
        c = lg.make_calibration("full", 0.0, 0.0, 2.10, head_r_wh=r, head_origin_w=o, tangent_frame="baseline_projected")
        scene, cam, _b, _m = ar.configure(c, 256)
    else:
        c = lg.make_calibration(profile, 0.0, 0.0, 2.10, head_r_wh=r, head_origin_w=o)
        scene, cam, _b, _m = co._prepare_perspective_pair(c, "CPU", 4)
    return readback(scene, cam, bpy.context.view_layer)


def legacy_breadth1(ctx, tmp):
    from fov_render.bl.config import readback
    import bpy
    scene, vl = _open(_room(ctx))
    b = load_legacy("f3d", "classroom_oracle/breadth1_render.py", "classroom_oracle")
    from fov_render.bl import scene as S
    h = S.resolve_head(scene)[0]
    cam, _rb = b.configure(scene, h.R_wh, h.origin_w, "CPU")
    return readback(scene, cam, bpy.context.view_layer)


def legacy_reference_view(ctx, tmp):
    from fov_render.bl import scene as S
    scene, vl = _open(_room(ctx))
    h = S.resolve_head(scene)[0]
    seeds = tmp / "seeds.json"
    seeds.write_text(json.dumps(h.as_dict()))
    rv = load_legacy("f3d", "visual_language/render_reference_view.py", "visual_language")
    return _capture_first_render(lambda: call_with_argv(rv.main, "render_reference_view.py", [
        "--seeds", seeds, "--out", tmp / "ref", "--width", 64, "--height", 48, "--spp", 1, "--device", "CPU"]))


LEGACY_REPO = {"fov_fixation": ("fov", ()), "preview360": ("fov", ()), "fsg": ("fov", ()),
               "oracle1": ("f3d", ()), "ab1a": ("f3d", ("active_bootstrap",)),
               "breadth1": ("f3d", ("classroom_oracle",)), "reference_view": ("f3d", ("visual_language",))}




def _legacy_cases(ctx):
    """(name, legacy function, scene_path) for each of check step 3's cases."""
    from fov_render.checks import step3 as S3
    legacy = {"fov_fixation": legacy_fov_fixation, "preview360": legacy_preview360, "fsg": legacy_fsg,
              "oracle1": lambda c, t: legacy_oracle1(c, t), "ab1a": lambda c, t: legacy_oracle1(c, t, narrowed=True),
              "breadth1": legacy_breadth1, "reference_view": legacy_reference_view}
    return [(name, legacy[name], scene_path) for name, scene_path, *_ in S3._cases(ctx)]


def step3(ctx) -> dict:
    from fov_render.checks import step3 as S3
    from fov_render.checks.common import legacy_path
    out = {}
    variants = (("", None), (" [all passes on]", S3.calib_room_all_passes(ctx)))
    with tempfile.TemporaryDirectory() as d:
        for label, room in variants:
            vctx = dict(ctx, room=room) if room else ctx
            for name, legacy, scene_path in _legacy_cases(vctx):
                if room and scene_path is None:
                    continue
                key = name + label
                tmp = Path(d) / key.replace(" ", "_").replace("[", "").replace("]", "")
                tmp.mkdir()
                repo, subs = LEGACY_REPO[name]
                with legacy_path(repo, *subs):
                    out[key] = legacy(vctx, tmp)
    (KNOWN / "step3_readbacks.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return {"step3_readbacks.json": sorted(out)}


def step4(ctx) -> dict:
    import bpy
    from fov_render.bl import scene as S
    from fov_render.checks.common import legacy_path, load_legacy
    from fov_render.checks.scenes import make_audit_scene
    from fov_render.checks.step1 import calib_room
    bpy.ops.wm.open_mainfile(filepath=calib_room(ctx))
    with legacy_path("f3d"):
        co = load_legacy("f3d", "classroom_oracle1_render.py")
        _e, _n, lo, lr = co._head_pose()
    bpy.ops.wm.open_mainfile(filepath=calib_room(ctx))
    with legacy_path("f3d"):
        _by_id, by_name = co._assign_instance_ids()
    bpy.ops.wm.open_mainfile(filepath=make_audit_scene(ctx["work"]))
    with legacy_path("f3d"):
        co._assign_instance_ids()
    audit_live = {o.name_full: int(o.pass_index) for o in bpy.data.objects if o.type in S.RENDERABLE}
    out = {"head_origin_w": [float(x) for x in lo], "head_R_wh": [[float(x) for x in row] for row in lr],
           "calib_room_ids": dict(by_name), "audit_scene_ids_after_legacy": audit_live}
    (KNOWN / "step4.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return {"step4.json": sorted(out)}


def main(argv) -> int:
    import bpy
    from fov_render.bl import env
    from fov_render.checks.pinned import pins
    env.prepare()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", nargs="*", default=["1", "2", "3", "4"])
    ap.add_argument("--out", default=None, help="write here instead of fov_render/checks/known (to compare a re-freeze)")
    a = ap.parse_args(argv)
    global KNOWN
    if a.out:
        KNOWN = Path(a.out).resolve()
    ctx = {"work": Path(a.work).resolve(), "device": "CPU"}
    ctx["work"].mkdir(parents=True, exist_ok=True)
    KNOWN.mkdir(parents=True, exist_ok=True)
    made = {}
    for k, fn in (("2", step2), ("1", lambda: step1(ctx)), ("3", lambda: step3(ctx)), ("4", lambda: step4(ctx))):
        if k in a.only:
            made.update(fn())
            print(f"[freeze] step {k}: {list(made)[-1]}", flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rec = {"schema": "fov-render-known-answers-v1", "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "fov_render_commit": commit, "legacy": pins()["sources"], "blender": bpy.app.version_string,
           "machine": {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version()},
           "files": made, "note": "frozen in the Chat sandbox (bpy 5.2.1 wheel, CPU); renders are checked with "
                                  "tolerances, not bitwise, so other machines' CPUs agree"}
    (KNOWN / "freeze.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(f"[freeze] wrote {sorted(made)} -> {KNOWN}", flush=True)
    return 0


if __name__ == "__main__":
    import os
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    rc = main(argv)
    sys.stdout.flush()
    os._exit(rc)
