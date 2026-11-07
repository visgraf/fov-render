"""Step 4 gates: the scene layer on the generated calibration room, a synthetic world-only HDRI
scene and a synthetic audit scene. Needs bpy. The Classroom parts are Phase 2."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from .. import record
from ..rig import Gaze, HeadPose, Rig
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
    """Both numberings (PLAN D11), complete the default (D13). Direct objects equal f3d-vision's _assign_instance_ids on the
    calibration room and are stable on reopening. On the audit scene, "legacy" leaves every pass index
    exactly where the legacy code leaves it (Prop keeps its own 2 and the collision with Floor is
    listed) and "complete" numbers Prop after the direct objects. Controls: one legacy name changed;
    the complete catalogue compared with the legacy pass indices."""
    import bpy
    from ..bl import scene as S
    h = S.open_scene(calib_room(ctx))
    ours = {e["name"]: e["id"] for e in h.catalog["instances"]}
    bpy.ops.wm.open_mainfile(filepath=calib_room(ctx))
    with legacy_path("f3d"):
        co = load_legacy("f3d", "classroom_oracle1_render.py")
        _by_id, by_name = co._assign_instance_ids()
    again = {e["name"]: e["id"] for e in S.open_scene(calib_room(ctx)).catalog["instances"]}

    audit = make_audit_scene(ctx["work"])

    def live_ids(cat):
        return {e["name"]: int(bpy.data.objects[e["name"]].pass_index) for e in cat["instances"]}

    bpy.ops.wm.open_mainfile(filepath=audit)
    with legacy_path("f3d"):
        co._assign_instance_ids()
    legacy_live = {o.name_full: int(o.pass_index) for o in bpy.data.objects if o.type in S.RENDERABLE}
    lg = S.open_scene(audit, instance_ids="legacy").catalog
    lg_ids = {e["name"]: e["id"] for e in lg["instances"]}
    lg_live = live_ids(lg)
    cp = S.open_scene(audit, instance_ids="complete").catalog
    cp_ids = {e["name"]: e["id"] for e in cp["instances"]}
    cp_live = live_ids(cp)
    via_c = [e for e in cp["instances"] if e["via_instance_of"]]
    try:
        S.open_scene(audit, instance_ids="nonsense")
        refused = False
    except ValueError:
        refused = True
    ok = (ours == dict(by_name) and ours == again and h.catalog["mode"] == "complete"
          and lg_ids == legacy_live and lg_live == legacy_live
          and lg_ids["Prop"] == 2 and lg["collisions"] == [{"name": "Prop", "id": 2, "same_id_as": "Floor"}]
          and cp_ids == cp_live and [e["name"] for e in via_c] == ["Prop"] and via_c[0]["id"] == cp["direct"] + 1
          and cp["collisions"] == [] and refused)
    renamed = dict(by_name)
    k = sorted(renamed)[0]
    renamed[k + "_x"] = renamed.pop(k)
    ctrl = {"renamed_detected": renamed != ours, "complete_differs_from_legacy": cp_ids != legacy_live}
    return Result(STEP, "instance catalogue (legacy and complete numbering, stable, collection instances)", ok,
                  all(ctrl.values()),
                  {"calib_room_count": len(ours), "equal_to_legacy": ours == dict(by_name), "stable": ours == again,
                   "audit_legacy": {"ids": lg_ids, "legacy_live": legacy_live, "collisions": lg["collisions"]},
                   "audit_complete": {"ids": cp_ids, "via": via_c}, "unknown_mode_refused": refused,
                   "controls": ctrl})


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
    h = S.open_scene(make_audit_scene(ctx["work"]), instance_ids="complete")
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
    return Result(STEP, "Object Index equals the ray-cast object (complete numbering, instances included)", n > 50 and frac >= 0.97,
                  bad > 0, {"sampled_hits": n, "agree_fraction": frac, "mismatches_first": mismatch[:10],
                            "ids_seen": sorted(set(int(x) for x in np.unique(ids))), "floor_id": names["Floor"],
                            "control_renumbered_floor_mismatches": bad,
                            "note": "pixels at object edges can disagree: the pass is one sample at spp 1, the "
                                    "ray cast goes through the pixel centre"})


CHECKS = [head_pose, instance_catalog, calib_room_sanity, hdri_identity, audit_and_sanitise, instance_ids_vs_raycast]


# ----------------------------------------------------------------------------- arbitrary files (issue #4)
# The Barcelona Pavilion's triage (issue #4, part A) found four ways an arbitrary .blend broke the
# scene layer. Each gate below builds a small scene that has the problem, and its control shows the
# previous behaviour failing on it.

def head_from_float32_camera(ctx) -> Result:
    """A camera rotation off orthonormal by float32 rounding (the Pavilion: 2.4e-7) is made
    orthonormal; one within tolerance is kept bit for bit (the Classroom's EYE, compared bitwise in
    #2); a sheared matrix is refused. Control: HeadPose on the raw rotation raises, as open_scene did."""
    from ..bl import scene as S
    from ..rig import rot_x, rot_y
    good = rot_y(-37.0) @ rot_x(12.5)
    off = good.copy()
    off[0, 1] += 2.4e-7
    off[2, 0] -= 1.3e-7
    fixed, dev = S.orthonormalise(off)
    kept, dev0 = S.orthonormalise(good)
    sheared = good @ np.array([[1.0, 2e-3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    try:
        S.orthonormalise(sheared)
        refused = False
    except ValueError:
        refused = True
    try:
        HeadPose(off, np.zeros(3))
        raw_raises = False
    except ValueError:
        raw_raises = True
    HeadPose(fixed, np.zeros(3))
    ok = (dev > 1e-7 and HeadPose.valid_rotation(fixed) and float(np.abs(fixed - off).max()) < 3e-7
          and kept.tobytes() == good.tobytes() and refused)
    return Result(STEP, "4.7 head from a float32 camera: made orthonormal only when needed", ok, raw_raises,
                  {"deviation_in": dev, "change": float(np.abs(fixed - off).max()), "kept_bitwise": kept.tobytes() == good.tobytes(),
                   "deviation_good": dev0, "sheared_refused": refused, "control_raw_rotation_raises": raw_raises})


def _two_layer_scene(work: Path) -> str:
    """The calibration room with a second view layer that does not render made the active one."""
    import bpy
    from .step1 import calib_room
    out = Path(work) / "scenes" / "two_layers.blend"
    if out.is_file():
        return str(out)
    bpy.ops.wm.open_mainfile(filepath=calib_room({"work": work}))
    s = bpy.context.scene
    other = s.view_layers.new("Inactive")
    other.use = False
    bpy.context.window.view_layer = other
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    return str(out)


def view_layer_that_renders(ctx) -> Result:
    """A file whose active view layer does not render: open_scene makes the rendering layer active
    and the only one, and a look's record has its Position and Object Index. Control: passes set on
    the file's active layer, as before, leave the written EXR without Position."""
    import bpy
    from .. import Session
    from ..bl import config, engine
    from ..sensors import Perspective
    path = _two_layer_scene(ctx["work"])
    bpy.ops.wm.open_mainfile(filepath=path)
    was = bpy.context.view_layer.name
    ip = Session.in_process(config.RenderConfig(profile="small", device=ctx["device"], spp=1), Rig(), blend=path,
                            access="oracle")
    vl = ip.ready["scene"]["file"]["view_layer"]
    look = ip.look(Perspective(), ("L",), Gaze.yaw_pitch(0.0, 0.0, 2.0), out_dir=Path(ctx["work"]) / "step4" / "4.8",
                   outputs=("exr", "records"))
    rec = look.records[0]
    hits = int(np.isfinite(rec.distance_m).sum())
    # control: the old behaviour
    bpy.ops.wm.open_mainfile(filepath=path)
    s = bpy.context.scene
    cfg = config.RenderConfig(profile="small", device=ctx["device"], spp=1)
    config.apply(s, cfg, 64, 32, bpy.context.view_layer)
    ctrl_exr = Path(ctx["work"]) / "step4" / "4.8" / "control.exr"
    from ..bl import cameras
    from ..sensors import PanoramaModel
    cameras.build(s, PanoramaModel(64), cfg)
    engine.render_once(s, 1, 0, str(ctrl_exr))
    ctrl_has_position = "position" in engine.read_passes(str(ctrl_exr))
    ok = was == "Inactive" and vl["view_layer"] == "ViewLayer" and vl["was_active"] == "Inactive" and hits > 1000
    return Result(STEP, "4.8 the view layer that renders gets the passes (active layer not rendering)", ok,
                  not ctrl_has_position, {"file_active_layer": was, "view_layer": vl, "record_hits": hits,
                                          "control_exr_has_position": ctrl_has_position})


def _particle_scene(work: Path) -> str:
    """Two emitters in front of the EYE: one whose particles render an object hidden from render, one
    whose particles render a collection that is not linked into the scene."""
    import bpy
    out = Path(work) / "scenes" / "particles-v2.blend"
    if out.is_file():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.unit_settings.system, s.unit_settings.scale_length = "METRIC", 1.0
    for name, x, kind in (("EmitterA", -0.6, "OBJECT"), ("EmitterB", 0.6, "COLLECTION")):
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(x, 3.0, 1.6), rotation=(math.radians(-90), 0, 0))
        em = bpy.context.object
        em.name = name
        bpy.ops.mesh.primitive_ico_sphere_add(radius=0.12, location=(0, -50, 0))
        src = bpy.context.object
        src.name, src.hide_render = f"Pebble{name[-1]}", True
        em.modifiers.new("ps", "PARTICLE_SYSTEM")
        st = em.particle_systems[0].settings
        st.type, st.count, st.hair_length = "HAIR", 30, 1.0
        if kind == "OBJECT":
            st.render_type, st.instance_object = "OBJECT", src
        else:                                   # a source collection not linked into the scene
            coll = bpy.data.collections.new(f"Pebbles{name[-1]}")
            coll.objects.link(src)
            for c in list(src.users_collection):
                if c != coll:
                    c.objects.unlink(src)
            src.hide_render = False
            st.render_type, st.instance_collection = "COLLECTION", coll
    cam = bpy.data.objects.new("EYE", bpy.data.cameras.new("EYE"))
    s.collection.objects.link(cam)
    cam.location, cam.rotation_euler = (0, 0, 1.6), (math.radians(90.0), 0.0, 0.0)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    return str(out)


def particle_instances_named(ctx) -> Result:
    """Particle instances of a source hidden from render (the Pavilion's pebbles) read the source's
    number under "complete". Control: under "legacy" they read 0."""
    from .. import Session
    from ..bl import config
    from ..sensors import Perspective
    path = _particle_scene(ctx["work"])
    out = {}
    for mode in ("complete", "legacy"):
        ip = Session.in_process(config.RenderConfig(profile="small", device=ctx["device"], spp=1), Rig(), blend=path,
                                access="oracle", instance_ids=mode)
        cat = ip.handler.scene_handle.catalog
        rec = ip.look(Perspective(), ("C",), Gaze.yaw_pitch(0.0, 0.0), out_dir=Path(ctx["work"]) / "step4" / "4.9" / mode,
                      outputs=("exr", "records")).records[0]
        ids = rec.instance_id[np.isfinite(rec.distance_m)]
        src = {e["name"]: e["id"] for e in cat["instances"] if e.get("via") == "particles"}
        emit = {e["name"]: e["id"] for e in cat["instances"] if e["name"].startswith("Emitter")}
        out[mode] = {"via_particles": cat["via_particles"], "sources": src, "emitters": emit,
                     "hits": int(ids.size), "hit_ids": {int(i): int((ids == i).sum()) for i in np.unique(ids)}}
    c, lg = out["complete"], out["legacy"]
    instance_px = sum(c["hit_ids"].get(v, 0) for v in c["sources"].values())
    ok = (c["via_particles"] == 2 and all(c["hit_ids"].get(v, 0) > 0 for v in c["sources"].values()))
    # control: under legacy the same pixels read Object Index 0, i.e. the record counts them as misses
    ctrl = instance_px > 0 and c["hits"] - lg["hits"] == instance_px and all(v == 0 for v in lg["sources"].values())
    return Result(STEP, "4.9 particle instances of sources outside the render are named (complete)", ok, ctrl,
                  {"complete": c, "legacy": lg, "instance_pixels": instance_px,
                   "control_legacy_reads_them_as_misses": ctrl})


def _audit_scene_v3(work: Path) -> str:
    """The audit scene with a glass object inside an instanced collection, an image texture whose
    file is missing, and one whose file exists."""
    import bpy
    from .scenes import _mat
    out = Path(work) / "scenes" / "audit-v3.blend"
    if out.is_file():
        return str(out)
    bpy.ops.wm.open_mainfile(filepath=make_audit_scene(work))
    s = bpy.context.scene
    inner = bpy.data.collections.new("Bottles")
    bpy.ops.mesh.primitive_cube_add(size=0.3, location=(0, 0, 0))
    b = bpy.context.object
    b.name = "Bottle"
    for c in list(b.users_collection):
        c.objects.unlink(b)
    inner.objects.link(b)
    b.data.materials.append(_mat("bottle_glass", "glass"))
    e = bpy.data.objects.new("BottlesA", None)
    e.instance_type, e.instance_collection, e.location = "COLLECTION", inner, (0, 2.5, 1.6)
    s.collection.objects.link(e)
    present = Path(work) / "scenes" / "present.png"
    img = bpy.data.images.new("present", 4, 4)
    img.filepath_raw, img.file_format = str(present), "PNG"
    img.save()
    for name, path, obj in (("missing_tex", str(Path(work) / "scenes" / "no-such-texture.png"), "Floor"),
                            ("present_tex", str(present), "Lamp")):
        m = bpy.data.materials.new(name)
        if hasattr(m, "use_nodes"):
            m.use_nodes = True
        t = m.node_tree.nodes.new("ShaderNodeTexImage")
        t.image = bpy.data.images.load(path, check_existing=False) if Path(path).is_file() else bpy.data.images.new(name, 2, 2)
        if not Path(path).is_file():
            t.image.source, t.image.filepath = "FILE", path
        m.node_tree.links.new(t.outputs["Color"], m.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
        bpy.data.objects[obj].data.materials.append(m)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    return str(out)


def audits_reach_instances_and_images(ctx) -> Result:
    """The material audit visits objects inside collection instances (a glass bottle), and the image
    audit lists exactly the texture whose file is missing, with the object that uses it. Control: the
    audit over the scene's own objects only (as before) misses the bottle."""
    from ..bl import scene as S
    path = _audit_scene_v3(ctx["work"])
    h = S.open_scene(path)
    a = h.audit
    old = sorted(o.name_full for o in h.scene.objects if o.type in S.RENDERABLE and not o.hide_render
                 and any(f for slot in o.material_slots for f in S.material_flags(slot.material)))
    miss = a["missing_images"]["missing"]
    ok = ("transmission" in a["flagged_objects"].get("Bottle", []) and list(miss) == ["missing_tex"]
          and miss["missing_tex"]["objects"] == ["Floor"] and a["by_flag"].get("transmission", 0) >= 2)
    return Result(STEP, "4.10 audits reach collection instances; missing images listed with their objects", ok,
                  "Bottle" not in old, {"flagged": a["flagged_objects"], "by_flag": a["by_flag"],
                                        "missing_images": a["missing_images"], "control_scene_objects_only": old})


def _backface_scene(work: Path) -> str:
    """Around the EYE: a cube with its normals flipped (a closed mesh showing back-faces) ahead, and a
    plane facing away (an open mesh, its back side seen) to the right, inside a large room."""
    import bpy
    out = Path(work) / "scenes" / "backfaces.blend"
    if out.is_file():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.unit_settings.system, s.unit_settings.scale_length = "METRIC", 1.0
    bpy.ops.mesh.primitive_cube_add(size=20.0, location=(0, 0, 1.6))
    room = bpy.context.object
    room.name = "Room"
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.normals_make_consistent(inside=True)        # the room faces inwards: no back-faces
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 3, 1.6))
    flipped = bpy.context.object
    flipped.name = "FlippedCube"
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.normals_make_consistent(inside=True)        # closed, normals inwards: seen as back-faces
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.mesh.primitive_plane_add(size=1.5, location=(3, 0, 1.6), rotation=(0, math.radians(90), 0))
    plane = bpy.context.object
    plane.name = "OpenPlane"                                  # normal +X: seen from the EYE, its back
    cam = bpy.data.objects.new("EYE", bpy.data.cameras.new("EYE"))
    s.collection.objects.link(cam)
    cam.location, cam.rotation_euler = (0, 0, 1.6), (math.radians(90.0), 0.0, 0.0)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    return str(out)


def backfaces_open_and_closed(ctx) -> Result:
    """Back-faces on an open mesh (a plane seen from behind) are reported, on a closed mesh (a cube
    with flipped normals) counted against the limit (issue #4: the Barbershop's open set dressing).
    Control: counting every back-face as the previous rule did also fails on the open plane."""
    from ..bl import config, scene as S
    h = S.open_scene(_backface_scene(ctx["work"]))
    c = h.sanity(config.RenderConfig(profile="small", device=ctx["device"]), width=360, spp=1,
                 out_dir=str(Path(ctx["work"]) / "step4" / "4.11"))
    v = S.backface_verdict(c, (), 0.0)
    only_plane = S.backface_verdict({k: c[k] for k in ("backface_by_object", "backface_closed")},
                                    ("FlippedCube",), 0.0)
    old_rule_on_plane = {"backface_by_object": {"OpenPlane": c["backface_by_object"].get("OpenPlane", 0.0)}}
    ctrl = not S.backface_verdict(old_rule_on_plane, (), 0.0)["ok"]
    ok = (c["backface_closed"].get("FlippedCube") is True and c["backface_closed"].get("OpenPlane") is False
          and "Room" not in c["backface_by_object"] and not v["ok"] and list(v["closed_not_allowed"]) == ["FlippedCube"]
          and only_plane["ok"] and only_plane["open_fraction"] > 0)
    return Result(STEP, "4.11 back-faces: open meshes reported, closed meshes counted", ok, ctrl,
                  {"backface_by_object": c["backface_by_object"], "closed": c["backface_closed"], "verdict": v,
                   "with_the_cube_allowed": only_plane})


CHECKS += [head_from_float32_camera, view_layer_that_renders, particle_instances_named, audits_reach_instances_and_images,
           backfaces_open_and_closed]
