"""Classroom gates (Phase 2, the workstation; issue #1): the parts of steps 1, 3 and 4 that need the
Classroom and the accepted f3d-vision records. Needs bpy.

    python -m fov_render.checks --steps classroom --device OPTIX

    1.C   legacy check_foveated on a fov_render fixation of the Classroom (profile small)
    3.C   preset("ab1a", device="OPTIX"), through the scene layer and on the raw file, equals AB1a's own configure, and
          AB1a's own readback of it equals the accepted AB1a record
    4.C1  the Classroom head pose from scene.resolve_head equals the accepted records
    4.C2  catalogue entries 1..234 equal the accepted Breadth-1 catalogue (names, ids, order); the
          legacy and complete numberings of the 116 collection-instance objects (PLAN D11)

Every file comes from the f3d-vision checkout the legacy specs name (their SHARED), or from
$FOVRENDER_F3D_ROOT, and is checked against the sha256 its spec pins before it is used. The specs and
the legacy code are f3d-vision's at the pinned commit, rebuilt from that checkout's git history
(checks.pinned, issue #16). Without them a gate is SKIPPED, never passed; a file whose hash differs
is a FAIL.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from ..rig import Gaze, Rig
from ..sensors import Foveated, PerspectiveModel
from .common import Result, legacy_path, load_legacy
from .step1 import GAZE, SPP, _check_foveated

STEP = "classroom"


class Missing(Exception):
    pass


def _specs():
    ab = load_legacy("f3d", "active_bootstrap/ab1a_spec.py", "active_bootstrap")
    b1 = load_legacy("f3d", "classroom_oracle/breadth1_spec.py", "classroom_oracle")
    ns = load_legacy("f3d", "north_star/ns1a_spec.py", "north_star")
    return ab, b1, ns


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pinned(path, sha: str, shared: Path) -> Path:
    """The pinned file under $FOVRENDER_F3D_ROOT (default: the spec's SHARED); Missing if absent,
    RuntimeError if its hash is not the pin."""
    p = Path(path)
    if p.is_absolute():
        p = p.relative_to(shared)
    root = Path(os.environ.get("FOVRENDER_F3D_ROOT") or shared)
    full = root / p
    if not full.is_file():
        raise Missing(str(full))
    got = _sha256(full)
    if got != sha:
        raise RuntimeError(f"{full}: sha256 {got} is not the pin {sha}")
    return full


def files() -> dict:
    ab, b1, ns = _specs()
    if b1.SOURCES["blend"][1] != ab.BLEND_SHA256 or b1.SOURCES["seeds"][1] != ab.HEAD_POSE_SOURCE[1]:
        raise RuntimeError("the Breadth-1 and AB1a specs pin different Classroom files")
    return {"blend": _pinned(ab.BLEND, ab.BLEND_SHA256, ab.SHARED),
            "seeds": _pinned(*ab.HEAD_POSE_SOURCE, ab.SHARED),
            "catalog": _pinned(*b1.SOURCES["catalog"], b1.SHARED),
            "ab1a_acquisition": _pinned(*ns.AB1A_ACQUISITION, ns.SHARED)}


def _need(name: str):
    from .pinned import LegacyUnavailable
    try:
        return files(), None
    except Missing as e:
        return None, Result(STEP, name, False, None,
                            skipped=f"Classroom file not found: {e} (set FOVRENDER_F3D_ROOT to an f3d-vision checkout)")
    except LegacyUnavailable as e:
        return None, Result(STEP, name, False, None, skipped=f"f3d-vision's pinned code: {e}")


# ----------------------------------------------------------------------------- 1.C

def _render_foveated(ctx, blend: Path, folder: Path, *, e2_shader: float = 2.0, profile: str = "small",
                     gaze=GAZE) -> Path:
    """step1.render_new_foveated on another scene: one cyclopean fixation, fix.exr + meta.json in the
    format the legacy check_foveated reads."""
    from ..bl import cameras, config, engine, scene as S
    h = S.open_scene(str(blend), keep_passes=("combined", "z", "normal", "position"), scale_length=None)
    cfg = config.preset("fov_fixation", profile=profile, device=ctx["device"], spp=SPP, seed=0)
    model = Foveated().model(profile)
    backend = config.apply(h.scene, cfg, model.width, model.height, h.view_layer)
    cam = cameras.build(h.scene, model, cfg)
    if e2_shader != model.e2:
        cam.data.cycles_custom["E2"] = e2_shader
    pose = Rig().pose(h.head, Gaze.per_eye(C=tuple(gaze)), "C")
    cameras.pose(cam, pose)
    folder.mkdir(parents=True, exist_ok=True)
    secs = engine.render_once(h.scene, SPP, 0, str(folder / "fix.exr"))
    meta = {"warp": {"E2_deg": model.e2, "e_max_deg": model.emax, "s0_deg": model.s0_deg, "raster": model.n},
            "camera_position_m": pose.centre_w.tolist(), "camera_forward": (pose.R_wb @ [0, 0, -1.0]).tolist(),
            "device": backend, "spp": SPP, "seed": 0, "render_seconds": round(secs, 4), "blend": str(blend),
            "settings": config.readback(h.scene, cam, h.view_layer), "eye_pose": pose.as_dict(),
            "scene_sanitised": h.sanitised["animation_removed"]}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    return folder


def classroom_check_foveated(ctx) -> Result:
    """The legacy checker (depth = ray distance, warp = formula, mask, ray saving) on a fov_render
    fixation of the Classroom. Control: shader E2 = 2.04 while the record says 2.0."""
    name = "1.C legacy check_foveated on a fov_render fixation of the Classroom"
    f, skip = _need(name)
    if skip:
        return skip
    work = Path(ctx["work"]) / "classroom" / "1C"
    good = _check_foveated(_render_foveated(ctx, f["blend"], work / "new"))
    bad = _check_foveated(_render_foveated(ctx, f["blend"], work / "control_e2", e2_shader=2.04))
    return Result(STEP, name, good["checks_failed"] == [], bool(bad["checks_failed"]),
                  {"report": good, "control_failures": bad["checks_failed"], "gaze": GAZE, "spp": SPP,
                   "profile": "small", "control": "shader E2 = 2.04 while the record says 2.0"})


# ----------------------------------------------------------------------------- 3.C

def _fresh_ab1a_render():
    """ab1a_render with a fresh ab1a_spec: step 3 replaces the spec's OPTIX with CPU in the module
    that `import ab1a_spec` caches, which would leak into this gate."""
    for m in ("ab1a_spec", "classroom_oracle1_render", "fsg_geometry", "bl_common", "render_foveated", "exr_lite"):
        sys.modules.pop(m, None)
    ar = load_legacy("f3d", "active_bootstrap/ab1a_render.py", "active_bootstrap")
    if ar.SP.DEVICE != "OPTIX":
        raise RuntimeError(f"AB1a spec device is {ar.SP.DEVICE!r}, expected its own 'OPTIX'")
    return ar


def ab1a_optix_settings(ctx) -> Result:
    """AB1a's own configure (OPTIX, the frozen gaze, the accepted head pose) against fov_render:
    open_scene, preset("ab1a", device="OPTIX"), the camera. Compared twice: the fov_render readback of
    both scenes (0 differences), and AB1a's own readback of the fov_render scene against the settings
    in the accepted AB1a record (equal). Control: Depth switched on with exact passes, and CPU."""
    import bpy
    from ..bl import cameras, config, scene as S
    from .step3 import _diff, _new, _open
    name = "3.C preset ab1a (OPTIX) equals AB1a's own configure and its accepted record"
    f, skip = _need(name)
    if skip:
        return skip
    rec = json.loads(f["ab1a_acquisition"].read_text())
    model = PerspectiveModel(256)
    with legacy_path("f3d", "active_bootstrap"):
        ar = _fresh_ab1a_render()
        _open(f["blend"])
        c = ar.calibration(ar.SP.GAZE_YAW_DEG, ar.SP.GAZE_PITCH_DEG, rec["eye_pose"])
        scene, cam, backend, _m = ar.configure(c, ar.SP.SPP)
        legacy_rb = config.readback(scene, cam, bpy.context.view_layer)
        legacy_own = json.loads(json.dumps(ar.readback(scene, cam)))

    def fov_render_side(cfg):
        h = S.open_scene(str(f["blend"]))
        used = config.apply(h.scene, cfg, model.width, model.height, h.view_layer)
        cam = cameras.build(h.scene, model, cfg)
        return used, config.readback(h.scene, cam, h.view_layer), json.loads(json.dumps(ar.readback(h.scene, cam)))

    def own_diff(a):
        return sorted(k for k in set(a) | set(rec["settings"]) if a.get(k) != rec["settings"].get(k))

    cfg = config.preset("ab1a", device="OPTIX")
    used, new_rb, new_own = fov_render_side(cfg)
    d_legacy, d_record = _diff(legacy_rb, new_rb), own_diff(new_own)
    controls = {}
    for label, bad_cfg in (("passes with Depth, exact", cfg.with_(passes=("combined", "z", "position", "object_index"),
                                                                    passes_exact=True)),
                           ("device CPU", cfg.with_(device="CPU"))):
        _u, b_rb, b_own = fov_render_side(bad_cfg)
        controls[label] = {"vs_legacy": _diff(legacy_rb, b_rb), "vs_record": own_diff(b_own)}
    # the preset without the scene layer (Phase 1's method in step 3): gated since the ab1a preset sets
    # its passes exactly (issue #1 item 3; before, Depth and Normal stayed on from the raw file)
    raw = _diff(legacy_rb, _new(f["blend"], cfg, model))
    ok = (backend == "OPTIX" and used == "OPTIX" and d_legacy == [] and d_record == [] and raw == []
          and legacy_own == rec["settings"])
    return Result(STEP, name, ok, all(v["vs_legacy"] and v["vs_record"] for v in controls.values()),
                  {"legacy_backend": backend, "fov_render_backend": used, "differences_vs_legacy": d_legacy,
                   "differences_vs_accepted_record": d_record,
                   "legacy_own_readback_equals_record": legacy_own == rec["settings"],
                   "record": {"path": str(f["ab1a_acquisition"]), "created_utc": rec.get("created_utc")},
                   "gaze_yaw_pitch_deg": [ar.SP.GAZE_YAW_DEG, ar.SP.GAZE_PITCH_DEG], "spp": ar.SP.SPP,
                   "controls": controls, "preset_on_raw_file_differences": raw,
                   "note": "compared twice: through scene.open_scene (which trims passes) and the preset alone "
                           "on the raw Classroom file (which has Depth and Normal on)"})


# ----------------------------------------------------------------------------- 4.C1

def _pose_diff(R, o, ref: dict) -> float:
    return max(float(np.abs(np.asarray(R) - np.asarray(ref["head_R_wh"], float)).max()),
               float(np.abs(np.asarray(o) - np.asarray(ref["head_origin_w_m"], float)).max()))


def classroom_head_pose(ctx) -> Result:
    """The head that scene.resolve_head finds in the Classroom (the EYE object) equals the accepted
    Controller-01 seeds record, the eye pose in the accepted AB1a record and the legacy reader
    (_head_pose), within the AB1a spec's EYE_POSE_TOL. Control: the accepted origin moved by 1 mm."""
    from ..bl import scene as S
    name = "4.C1 Classroom head pose equals the accepted records"
    f, skip = _need(name)
    if skip:
        return skip
    ab, _b1, _ns = _specs()
    tol = float(ab.EYE_POSE_TOL)
    h = S.open_scene(str(f["blend"]))
    R, o = h.head.R_wh, h.head.origin_w
    seeds = json.loads(f["seeds"].read_text())
    ab1a = json.loads(f["ab1a_acquisition"].read_text())["eye_pose"]
    with legacy_path("f3d"):
        co = load_legacy("f3d", "classroom_oracle1_render.py")
        _e, _n, lo, lr = co._head_pose()
    d = {"vs_seeds_record": _pose_diff(R, o, seeds), "vs_ab1a_record": _pose_diff(R, o, ab1a),
         "vs_legacy_reader": _pose_diff(R, o, {"head_R_wh": lr, "head_origin_w_m": lo})}
    shifted = dict(seeds, head_origin_w_m=(np.asarray(seeds["head_origin_w_m"], float) + [0.0, 0.0, 0.001]).tolist())
    d_ctrl = _pose_diff(R, o, shifted)
    ok = h.eye_note == "EYE object" and max(d.values()) <= tol
    return Result(STEP, name, ok, d_ctrl > tol,
                  {"eye_note": h.eye_note, "origin_w_m": o.tolist(), "R_wh": R.tolist(), "max_abs_diff": d,
                   "tolerance": tol, "control_origin_plus_1mm_diff": d_ctrl, "blend_sha256": h.info["sha256"],
                   "records": {"seeds": str(f["seeds"]), "ab1a": str(f["ab1a_acquisition"])}})


# ----------------------------------------------------------------------------- 4.C2

def _via_live_ids(scene) -> dict:
    """name_full -> live pass_index of every renderable object reached only through a collection
    instance. Read through the instancing collections: a linked object's name alone does not index
    bpy.data.objects."""
    from ..bl import scene as S
    direct = {o.name_full for o in scene.objects}
    out = {}
    for o in scene.objects:
        coll = o.instance_collection if o.instance_type == "COLLECTION" else None
        if coll is None:
            continue
        for x in coll.all_objects:
            if x.type in S.RENDERABLE and not x.hide_render and x.name_full not in direct:
                out.setdefault(x.name_full, int(x.pass_index))
    # particle sources hidden from render (issue #4): their instances render, and read their number
    for o in scene.objects:
        for m in ([] if o.hide_render else o.modifiers):
            if m.type != "PARTICLE_SYSTEM" or not m.show_render:
                continue
            st = m.particle_system.settings
            srcs = [st.instance_object] if st.render_type == "OBJECT" and st.instance_object else \
                (list(st.instance_collection.all_objects) if st.render_type == "COLLECTION" and st.instance_collection else [])
            for x in srcs:
                if x.type in S.RENDERABLE and (x.hide_render or x.name_full not in direct):
                    out.setdefault(x.name_full, int(x.pass_index))
    return out


def breadth1_catalog(ctx) -> Result:
    """Both numberings on the Classroom (PLAN D11). The Classroom's own renderable objects are 1..234
    exactly as the accepted Breadth-1 catalogue and the legacy _assign_instance_ids (names, ids,
    order), in both modes. Objects reached only through collection instances: under "legacy" every one keeps the pass index the legacy code leaves it, and the collisions with direct
    ids are listed; under "complete" they are 235.. with no gaps. Controls: the accepted catalogue
    with one entry dropped, and the scene with one object hidden from render, must both disagree."""
    import bpy
    from ..bl import scene as S
    name = "4.C2 catalogue: entries 1..234 equal the accepted Breadth-1 catalogue; both numberings"
    f, skip = _need(name)
    if skip:
        return skip
    _ab, b1, _ns = _specs()
    accepted = [(int(e["instance_id"]), e["object_name"]) for e in json.loads(f["catalog"].read_text())["instances"]]
    n = len(accepted)
    # the legacy numbering, live: every pass index as the legacy code leaves it
    bpy.ops.wm.open_mainfile(filepath=str(f["blend"]))
    with legacy_path("f3d"):
        co = load_legacy("f3d", "classroom_oracle1_render.py")
        by_id, _by_name = co._assign_instance_ids()
    live = sorted(by_id.items())
    legacy_via = _via_live_ids(bpy.context.scene)
    # fov_render, legacy (asked for explicitly since D13 made complete the default)
    h = S.open_scene(str(f["blend"]), instance_ids="legacy")
    lg = h.catalog
    lg_direct = [(e["id"], e["name"]) for e in lg["instances"] if e["via_instance_of"] is None]
    lg_via = {e["name"]: e["id"] for e in lg["instances"] if e["via_instance_of"] is not None}
    lg_via_live = _via_live_ids(h.scene)
    # fov_render, complete
    hc = S.open_scene(str(f["blend"]))                       # the default: complete (D13)
    cp = hc.catalog
    cp_all = [(e["id"], e["name"]) for e in cp["instances"]]
    cp_via = [e for e in cp["instances"] if e["via_instance_of"] is not None]
    cp_via_live = _via_live_ids(hc.scene)
    ok_legacy = (lg["mode"] == "legacy" and n == b1.CATALOG_COUNT and lg_direct == accepted and live == accepted
                 and lg_via == legacy_via and lg_via_live == legacy_via)
    ok_complete = (cp_all[:n] == accepted and [i for i, _ in cp_all] == list(range(1, len(cp_all) + 1))
                   and all(e["id"] > n for e in cp_via) and cp_via_live == {e["name"]: e["id"] for e in cp_via}
                   and cp["collisions"] == [] and set(lg_via) == {e["name"] for e in cp_via} and cp["mode"] == "complete")
    # controls
    k = n // 2
    dropped = accepted[:k] + accepted[k + 1:]
    h2 = S.open_scene(str(f["blend"]))
    bpy.data.objects[accepted[k][1]].hide_render = True
    hidden = [(e["id"], e["name"]) for e in S.assign_instance_ids(h2.scene)["instances"] if e["via_instance_of"] is None]
    ctrl = {"one_entry_dropped_detected": lg_direct != dropped, "one_object_hidden_detected": hidden != accepted,
            "complete_differs_from_legacy_on_via": {e["name"]: e["id"] for e in cp_via} != legacy_via}
    zero = sum(1 for v in lg_via.values() if v == 0)
    return Result(STEP, name, ok_legacy and ok_complete, all(ctrl.values()),
                  {"accepted_count": n, "spec_count": b1.CATALOG_COUNT, "direct": lg["direct"],
                   "via_instance": lg["via_instance"], "legacy": {"ok": ok_legacy, "via_id_zero": zero,
                   "via_nonzero_ids": sorted({v for v in lg_via.values() if v}), "collisions": len(lg["collisions"]),
                   "collisions_first": lg["collisions"][:5]},
                   "complete": {"ok": ok_complete, "count": cp["count"], "via_first": cp_via[:5]},
                   "via_parents": sorted({e["via_instance_of"] for e in cp_via}),
                   "controls": ctrl, "dropped": list(accepted[k]), "catalog_file": str(f["catalog"])})

CHECKS = [classroom_check_foveated, ab1a_optix_settings, classroom_head_pose, breadth1_catalog]
