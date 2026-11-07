"""Step 4 gates: the scene layer on the generated calibration room, a synthetic world-only HDRI
scene and a synthetic audit scene. Needs bpy. The Classroom parts are Phase 2."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .. import record
from ..rig import Rig
from ..sensors import PanoramaModel
from .common import Result, legacy_path, load_legacy
from .scenes import AUDIT_EXPECTED, HDRI_W, make_audit_scene, make_hdri_scene
from .step1 import calib_room

STEP = "4"


def head_pose(ctx) -> Result:
    """The head comes from EYE: make_calib_room's eye at (0, 0, 1.6) looking +Y, up +Z; equal to the
    legacy readers (f3d-vision _head_pose, fov-3d-vision eye_record)."""
    import bpy
    from ..bl import scene as S
    h = S.open_scene(calib_room(ctx))
    expect_r = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])     # columns: right, up, -forward
    d_r = float(np.abs(h.head.R_wh - expect_r).max())
    d_o = float(np.abs(h.head.origin_w - [0, 0, 1.6]).max())
    with legacy_path("f3d"):
        co = load_legacy("f3d", "classroom_oracle1_render.py")
        _e, _n, lo, lr = co._head_pose()
    d_leg = max(float(np.abs(lr - h.head.R_wh).max()), float(np.abs(lo - h.head.origin_w).max()))
    other = S.resolve_head(bpy.context.scene, explicit=((0.0, 0.0, 1.6), 10.0, 0.0))[0]
    ctrl = float(np.abs(other.R_wh - h.head.R_wh).max()) > 1e-3
    return Result(STEP, "head pose from EYE (known answer and legacy readers)", max(d_r, d_o, d_leg) < 1e-6, ctrl,
                  {"eye_note": h.eye_note, "R_diff": d_r, "origin_diff": d_o, "vs_legacy": d_leg,
                   "control": "an explicit head yawed 10 deg differs"})


def instance_catalog(ctx) -> Result:
    """Numbering equals f3d-vision's _assign_instance_ids on the calibration room, is the same on
    reopening, and reaches objects that exist only inside collection instances."""
    import bpy
    from ..bl import scene as S
    h = S.open_scene(calib_room(ctx))
    ours = {e["name"]: e["id"] for e in h.catalog["instances"]}
    bpy.ops.wm.open_mainfile(filepath=calib_room(ctx))
    with legacy_path("f3d"):
        co = load_legacy("f3d", "classroom_oracle1_render.py")
        _by_id, by_name = co._assign_instance_ids()
    again = {e["name"]: e["id"] for e in S.open_scene(calib_room(ctx)).catalog["instances"]}
    a = S.open_scene(make_audit_scene(ctx["work"]))
    via = [e for e in a.catalog["instances"] if e["via_instance_of"]]
    ok = ours == dict(by_name) and ours == again and [e["name"] for e in via] == ["Prop"] and via[0]["id"] > a.catalog["direct"]
    renamed = dict(by_name)
    k = sorted(renamed)[0]
    renamed[k + "_x"] = renamed.pop(k)
    return Result(STEP, "instance catalogue (legacy numbering, stable, collection instances)", ok, renamed != ours,
                  {"calib_room_count": len(ours), "equal_to_legacy": ours == dict(by_name), "stable": ours == again,
                   "audit_scene": {k: a.catalog[k] for k in ("count", "direct", "via_instance")}, "via": via})


def calib_room_sanity(ctx) -> Result:
    """Closed room: no holes, no back-faces, floor and ceiling 1.600 m from the eye (manifest:
    0, 0, 1.60019 at 7200 px). Control: without the ceiling, holes appear."""
    import bpy
    from ..bl import config, scene as S
    cfg = config.RenderConfig(device=ctx["device"])
    h = S.open_scene(calib_room(ctx))
    chk = h.sanity(cfg, width=720, spp=4, out_dir=str(Path(ctx["work"]) / "step4" / "calib_sanity"))
    ok = (chk["hole_fraction"] == 0.0 and chk["backface_fraction"] == 0.0
          and abs(chk["nadir_depth_m"] - 1.6) < 2e-3 and abs(chk["zenith_depth_m"] - 1.6) < 2e-3)
    h2 = S.open_scene(calib_room(ctx))
    bpy.data.objects.remove(bpy.data.objects["ceiling"])
    bad = h2.sanity(cfg, width=720, spp=4, backface=False)
    return Result(STEP, "calibration room sanity (holes, back-faces, nadir, zenith)", ok, bad["hole_fraction"] > 0.1,
                  {k: v for k, v in chk.items() if k != "settings"} | {"control_no_ceiling_hole_fraction": bad["hole_fraction"]})


def _hdri_render(ctx, blend, yaw_deg=None):
    from ..bl import cameras, config, engine, scene as S
    eye = None
    if yaw_deg is not None:
        eye = ((0.0, 0.0, 0.0), yaw_deg, 0.0)
    h = S.open_scene(blend, eye=eye)
    model = PanoramaModel(HDRI_W)
    cfg = config.RenderConfig(device=ctx["device"], spp=16)
    config.apply(h.scene, cfg, model.width, model.height, h.view_layer)
    cam = cameras.build(h.scene, model, cfg)
    pose = Rig().fixed_pose(h.head, "C")
    cameras.pose(cam, pose)
    tmp = Path(tempfile.mkdtemp(prefix="fovrender-hdri-"))
    engine.render_once(h.scene, 16, 0, str(tmp / "pano.exr"))
    p = engine.read_passes(str(tmp / "pano.exr"))
    rec = record.build(model, p, pose.centre_w, pose.R_wb, h.head.R_wh, eye="C", look_id="hdri", access="oracle")
    return p["combined"][..., :3], rec


def _column_shift(render, src) -> int:
    a, b = render.mean(axis=(0, 2)), src.mean(axis=(0, 2))
    a, b = a - a.mean(), b - b.mean()
    corr = np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))
    s = int(np.argmax(corr))
    return s - len(a) if s > len(a) // 2 else s


def hdri_identity(ctx) -> Result:
    """World-only scene: the panorama reproduces the source image (column shift 0, small error), every
    pixel is a miss. Control: an eye turned 90 degrees shifts it by W/4."""
    blend, src = make_hdri_scene(ctx["work"])
    rgb, rec = _hdri_render(ctx, blend)
    shift = _column_shift(rgb, src)
    rel = np.abs(rgb - src).mean(-1) / np.maximum(src.mean(-1), 1e-4)
    holes = float(np.mean(~np.isfinite(rec.distance_m)))
    rgb90, _ = _hdri_render(ctx, blend, yaw_deg=0.0)       # make_hdri_scene's EYE has yaw -90
    shift90 = _column_shift(rgb90, src)
    ok = shift == 0 and float(np.median(rel)) < 0.01 and holes == 1.0
    return Result(STEP, "world-only HDRI scene reproduces its source", ok, abs(shift90) == HDRI_W // 4,
                  {"column_shift_px": shift, "rel_err_median": float(np.median(rel)),
                   "rel_err_p99": float(np.quantile(rel, 0.99)), "hole_fraction": holes,
                   "control_shift_px": shift90, "expected_control_shift_px": HDRI_W // 4})


def audit_and_sanitise(ctx) -> Result:
    """The audit flags exactly the four special materials and nothing in the calibration room; the
    animation on render settings is removed, and without that removal render_once refuses."""
    import bpy
    from ..bl import cameras, config, engine, scene as S
    path = make_audit_scene(ctx["work"])
    a = S.open_scene(path)
    flagged = a.audit["flagged_objects"]
    calib = S.open_scene(calib_room(ctx)).audit["flagged_objects"]
    removed = a.sanitised["animation_removed"]
    ok = flagged == AUDIT_EXPECTED and len(removed) == 2
    # control: the same file opened raw keeps its keyframed seed and render_once catches it
    bpy.ops.wm.open_mainfile(filepath=path)
    sc = bpy.context.scene
    model = PanoramaModel(32)
    cfg = config.RenderConfig(device=ctx["device"], spp=1)
    config.apply(sc, cfg, model.width, model.height)
    cameras.build(sc, model, cfg)
    caught = False
    try:
        engine.render_once(sc, 1, 3, str(Path(tempfile.mkdtemp()) / "x.exr"))
    except RuntimeError:
        caught = True
    return Result(STEP, "material audit and setting animation", ok and calib == {}, caught,
                  {"flagged": flagged, "expected": AUDIT_EXPECTED, "calib_room_flagged": calib,
                   "animation_removed": removed})


def instance_ids_vs_raycast(ctx) -> Result:
    """Object Index at sampled pixels equals the ray-cast object's catalogue id, instances included.
    Control: renumbering one object after the render breaks the agreement."""
    import bpy
    from ..bl import cameras, config, engine, scene as S
    h = S.open_scene(make_audit_scene(ctx["work"]))
    model = PanoramaModel(180)
    cfg = config.RenderConfig(device=ctx["device"], spp=1)
    config.apply(h.scene, cfg, model.width, model.height, h.view_layer)
    cam = cameras.build(h.scene, model, cfg)
    pose = Rig().fixed_pose(h.head, "C")
    cameras.pose(cam, pose)
    tmp = Path(tempfile.mkdtemp(prefix="fovrender-ids-"))
    engine.render_once(h.scene, 1, 0, str(tmp / "ids.exr"))
    p = engine.read_passes(str(tmp / "ids.exr"))
    rays = model.rays()
    ids = np.rint(p["object_index"]).astype(int)
    dg = bpy.context.evaluated_depsgraph_get()
    names = {e["name"]: e["id"] for e in h.catalog["instances"]}
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, model.height, 400), rng.integers(0, model.width, 400)
    agree = n = 0
    mismatch = []

    def raycast_id(i, j):
        d = pose.R_wb @ rays.direction[i, j]
        hit, _loc, _n, _f, obj, _m = h.scene.ray_cast(dg, pose.centre_w.tolist(), d.tolist())
        if not hit:
            return 0
        o = getattr(obj, "original", obj)
        if names.get(o.name_full) is None:
            return -1
        return int(bpy.data.objects[o.name_full].pass_index)     # the live number, not the catalogue

    def compare():
        agree = n = 0
        mism = []
        for i, j in zip(rows, cols):
            rid = raycast_id(i, j)
            if rid == 0 and ids[i, j] == 0:
                continue
            n += 1
            if rid == ids[i, j]:
                agree += 1
            else:
                mism.append([int(i), int(j), int(ids[i, j]), int(rid)])
        return agree, n, mism
    agree, n, mismatch = compare()
    # control: renumber the floor after the render; the rendered ids must stop agreeing
    bpy.data.objects["Floor"].pass_index = names["Floor"] + 100
    bad_agree, bad_n, _ = compare()
    bad = bad_n - bad_agree
    bpy.data.objects["Floor"].pass_index = names["Floor"]
    frac = agree / max(n, 1)
    return Result(STEP, "Object Index equals the ray-cast object (instances included)", n > 50 and frac >= 0.97,
                  bad > 0, {"sampled_hits": n, "agree_fraction": frac, "mismatches_first": mismatch[:10],
                            "ids_seen": sorted(set(int(x) for x in np.unique(ids))), "floor_id": names["Floor"],
                            "control_renumbered_floor_mismatches": bad,
                            "note": "pixels at object edges can disagree: the pass is one sample at spp 1, the "
                                    "ray cast goes through the pixel centre"})


CHECKS = [head_pose, instance_catalog, calib_room_sanity, hdri_identity, audit_and_sanitise, instance_ids_vs_raycast]
