"""The scene layer (report section 2, design step 4): open any .blend once, fix the head, and leave
the file in a state where only the framework's settings decide the render.

    h = scene.open_scene("room.blend")      # SceneHandle: head pose, instance catalogue, report
    h.sanity(cfg)                           # holes, back-faces, nadir and zenith depth

What `open_scene` does, in order: verify the file (sha256, Blender version, metric scale); resolve
the head (EYE object, an explicit pose, or the scene camera with a note); sanitise (remove
animation on render settings, switch off motion blur, compositing, sequencer, border, extra
passes); number the renderable objects for the Object Index pass; audit materials whose first
surface hit is not what the pixel shows.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from ..rig import HeadPose, Rig

EYE_NAME = "EYE"
RENDERABLE = {"MESH", "CURVE", "SURFACE", "FONT", "META"}
ANIMATED_PREFIXES = ("cycles.", "render.", "view_settings.")
MAX_PASS_INDEX = 32767


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- head

def rigid4(matrix) -> np.ndarray:
    loc, rot, _ = matrix.decompose()
    m = np.eye(4)
    m[:3, :3] = np.array(rot.to_matrix())
    m[:3, 3] = np.array(loc)
    return m


def resolve_head(scene, eye_name: str = EYE_NAME, explicit=None) -> tuple[HeadPose, str]:
    """explicit: None, a 4x4 world matrix, or (position_m, yaw_deg, pitch_deg) placed like
    fov-3d-vision bl_common.make_eye (yaw about world Z, 0 looks along +Y; pitch up positive)."""
    import bpy
    if explicit is not None:
        if isinstance(explicit, (tuple, list)) and len(explicit) == 3 and np.ndim(explicit[0]) == 1:
            pos, yaw, pitch = explicit
            from mathutils import Euler, Matrix
            rot = Euler((math.radians(90.0 + pitch), 0.0, math.radians(yaw)), "XYZ").to_matrix().to_4x4()
            m = Matrix.Translation(tuple(pos)) @ rot
            return HeadPose.from_matrix(rigid4(m)), f"explicit position {list(pos)}, yaw {yaw}, pitch {pitch}"
        return HeadPose.from_matrix(np.asarray(explicit, float)), "explicit matrix"
    bpy.context.view_layer.update()
    eye = bpy.data.objects.get(eye_name)
    if eye is not None:
        return HeadPose.from_matrix(rigid4(eye.matrix_world)), f"{eye_name} object"
    if scene.camera is not None:
        return HeadPose.from_matrix(rigid4(scene.camera.matrix_world)), \
            f"no {eye_name} object; using scene camera {scene.camera.name!r}"
    raise RuntimeError(f"no {eye_name} object and no scene camera: add a camera named {eye_name!r} at the vantage point")


# ----------------------------------------------------------------------------- sanitise

def _fcurves(act):
    """Every F-curve of an action, layered (Blender 4.4+) or legacy."""
    out = []
    for layer in getattr(act, "layers", []):
        for strip in layer.strips:
            for cb in strip.channelbags:
                out.extend((cb, fc) for fc in cb.fcurves)
    if hasattr(act, "fcurves") and not out:
        out.extend((act, fc) for fc in act.fcurves)
    return out


def remove_setting_animation(scene, prefixes=ANIMATED_PREFIXES) -> list[str]:
    """Remove F-curves and drivers on the scene's render settings, generalising fov-3d-vision's
    pin_seed: the Classroom keyframes cycles.seed, which silently overwrote a script's seed."""
    removed = []
    ad = scene.animation_data
    if ad is None:
        return removed
    if ad.action is not None:
        for owner, fc in _fcurves(ad.action):
            path = fc.data_path
            if path.startswith(prefixes):
                keys = [(round(k.co[0], 3), round(k.co[1], 6)) for k in fc.keyframe_points]
                owner.fcurves.remove(fc)
                removed.append(f"fcurve on {path} in action {ad.action.name!r}, keys {keys}")
    for d in list(ad.drivers):
        if d.data_path.startswith(prefixes):
            path = d.data_path
            ad.drivers.remove(d)
            removed.append(f"driver on {path}")
    return removed


def sanitise(scene, view_layer, keep_passes=("combined", "position", "object_index")) -> dict:
    from . import passes as P
    anim = remove_setting_animation(scene)
    r = scene.render
    before = {"motion_blur": r.use_motion_blur, "compositing": r.use_compositing,
              "sequencer": r.use_sequencer, "border": r.use_border}
    r.use_motion_blur = r.use_compositing = r.use_sequencer = r.use_border = False
    trimmed = P.trim_passes(view_layer, keep_passes)
    return {"animation_removed": anim, "switched_off": [k for k, v in before.items() if v],
            "passes": trimmed}


# ----------------------------------------------------------------------------- instances

INSTANCE_ID_MODES = ("legacy", "complete")


def assign_instance_ids(scene, mode: str = "complete") -> dict:
    """Number every renderable object for the Object Index pass (PLAN D11, D13).

    Objects in the scene are numbered 1.. by sorted name_full, exactly as f3d-vision's
    _assign_instance_ids, in both modes. Objects reached only through collection instances (e.g.
    linked libraries) are catalogued in both modes too, sorted by name:

    - "complete" (the default, D13): they are numbered after the direct objects, so none reads as a
      miss or as another object.
    - "legacy": they keep the pass index their file gives them, as in the legacy code. Often 0
      (reads as a miss) or a library's own number, which can collide with a direct object's id;
      collisions are listed in the catalogue. Ask for it when comparing with f3d-vision.

    Several instances of one object share its number: the Object Index pass is per object
    datablock (a documented limit)."""
    if mode not in INSTANCE_ID_MODES:
        raise ValueError(f"instance_ids must be one of {INSTANCE_ID_MODES}, got {mode!r}")
    direct = sorted([o for o in scene.objects if o.type in RENDERABLE and not o.hide_render],
                    key=lambda o: o.name_full)
    seen = {o.name_full for o in direct}
    via = {}
    for o in scene.objects:
        coll = o.instance_collection if o.instance_type == "COLLECTION" else None
        if coll is None:
            continue
        for x in coll.all_objects:
            if x.type in RENDERABLE and not x.hide_render and x.name_full not in seen:
                via.setdefault(x.name_full, (x, o.name_full))
    n_numbered = len(direct) + (len(via) if mode == "complete" else 0)
    if n_numbered >= MAX_PASS_INDEX:
        raise RuntimeError(f"{n_numbered} renderable objects; the Object Index pass holds at most {MAX_PASS_INDEX - 1}")
    entries = []
    for iid, obj in enumerate(direct, start=1):
        obj.pass_index = iid
        entries.append({"id": iid, "name": obj.name_full, "type": obj.type, "via_instance_of": None,
                        "renumbered": True})
    direct_ids = {e["id"]: e["name"] for e in entries}
    collisions = []
    for k, name in enumerate(sorted(via), start=len(direct) + 1):
        obj, parent = via[name]
        if mode == "complete":
            obj.pass_index = k
        iid = int(obj.pass_index)
        entries.append({"id": iid, "name": name, "type": obj.type, "via_instance_of": parent,
                        "renumbered": mode == "complete"})
        if mode == "legacy" and iid in direct_ids:
            collisions.append({"name": name, "id": iid, "same_id_as": direct_ids[iid]})
    return {"schema": "fov-render-instance-catalog-v1", "mode": mode, "count": len(entries),
            "direct": len(direct), "via_instance": len(via),
            "via_id_zero": sum(1 for e in entries if e["via_instance_of"] and e["id"] == 0),
            "collisions": collisions, "instances": entries}


# ----------------------------------------------------------------------------- materials

def _input_value(node, name):
    s = node.inputs.get(name)
    if s is None:
        return None, False
    v = getattr(s, "default_value", None)
    try:
        v = tuple(v)
    except TypeError:
        pass
    return v, s.is_linked


def material_flags(mat) -> list[str]:
    """Why a material's first hit may not be what the pixel shows. A heuristic over node types and
    inputs; it errs towards flagging."""
    flags = set()
    if mat is None or mat.node_tree is None:
        return []
    for n in mat.node_tree.nodes:
        t = n.bl_idname
        if t in ("ShaderNodeBsdfGlass", "ShaderNodeBsdfRefraction"):
            flags.add("transmission")
        elif t in ("ShaderNodeBsdfTransparent", "ShaderNodeHoldout"):
            flags.add("alpha")
        elif t in ("ShaderNodeEmission",):
            v, linked = _input_value(n, "Strength")
            if linked or (v is not None and v > 0):
                flags.add("emission")
        elif t in ("ShaderNodeVolumePrincipled", "ShaderNodeVolumeScatter", "ShaderNodeVolumeAbsorption"):
            flags.add("volume")
        elif t == "ShaderNodeBsdfPrincipled":
            v, linked = _input_value(n, "Transmission Weight")
            if linked or (v or 0) > 0:
                flags.add("transmission")
            v, linked = _input_value(n, "Alpha")
            if linked or (v is not None and v < 1.0):
                flags.add("alpha")
            v, linked = _input_value(n, "Emission Strength")
            c, clinked = _input_value(n, "Emission Color")
            if (linked or (v or 0) > 0) and (clinked or (c is not None and any(x > 0 for x in c[:3]))):
                flags.add("emission")
        elif t == "ShaderNodeOutputMaterial":
            s = n.inputs.get("Volume")
            if s is not None and s.is_linked:
                flags.add("volume")
    return sorted(flags)


def material_audit(scene) -> dict:
    objs = {}
    for o in scene.objects:
        if o.type not in RENDERABLE or o.hide_render:
            continue
        flags = sorted({f for slot in o.material_slots for f in material_flags(slot.material)})
        if flags:
            objs[o.name_full] = flags
    world_volume = False
    if scene.world is not None and scene.world.node_tree is not None:
        for n in scene.world.node_tree.nodes:
            if n.bl_idname == "ShaderNodeOutputWorld" and n.inputs.get("Volume") is not None and n.inputs["Volume"].is_linked:
                world_volume = True
    return {"flagged_objects": objs, "world_volume": world_volume,
            "rule": "first-hit truth differs from the pixel for these; experiments exclude or tag them"}


# ----------------------------------------------------------------------------- handle

@dataclass
class SceneHandle:
    path: str
    scene: object
    view_layer: object
    head: HeadPose
    eye_note: str
    info: dict
    sanitised: dict
    catalog: dict
    audit: dict
    checks: dict = field(default_factory=dict)

    def report(self) -> dict:
        return {"schema": "fov-render-scene-report-v1", "file": self.info, "head": self.head.as_dict(),
                "eye_note": self.eye_note, "sanitised": self.sanitised,
                "instances": {k: self.catalog[k] for k in ("mode", "count", "direct", "via_instance", "via_id_zero", "collisions")},
                "material_audit": self.audit, "checks": self.checks}

    def write(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "scene-report.json"), "w") as f:
            json.dump(self.report(), f, indent=1, sort_keys=True)
        with open(os.path.join(folder, "instance_catalog.json"), "w") as f:
            json.dump(self.catalog, f, indent=1)

    def sanity(self, cfg, width: int = 720, spp: int = 4, backface: bool = True, out_dir: str | None = None) -> dict:
        self.checks = sanity_checks(self, cfg, width, spp, backface, out_dir)
        return self.checks


def open_scene(path: str | None, *, eye=None, scale_length: float | None = 1.0,
               keep_passes=("combined", "position", "object_index"), instance_ids: str = "complete") -> SceneHandle:
    """Open a .blend (None: use the scene already loaded) and prepare it (see module docstring).
    scale_length: the unit scale the file must have; None accepts any (then it is recorded).
    instance_ids: "complete" (default) or "legacy", see assign_instance_ids (PLAN D11, D13)."""
    import bpy
    if path is not None:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(path))
    scene, vl = bpy.context.scene, bpy.context.view_layer
    info = {"path": bpy.data.filepath, "sha256": sha256(bpy.data.filepath) if bpy.data.filepath else None,
            "saved_with": ".".join(str(x) for x in bpy.data.version), "blender": bpy.app.version_string,
            "unit_system": scene.unit_settings.system, "scale_length": scene.unit_settings.scale_length,
            "frame_current": scene.frame_current}
    if scale_length is not None and not math.isclose(scene.unit_settings.scale_length, scale_length, rel_tol=1e-9):
        raise RuntimeError(f"unit scale_length is {scene.unit_settings.scale_length}, expected {scale_length}; "
                           f"pass scale_length=None to accept it")
    head, note = resolve_head(scene, explicit=eye)
    sanitised = sanitise(scene, vl, keep_passes)
    catalog = assign_instance_ids(scene, instance_ids)
    audit = material_audit(scene)
    return SceneHandle(info["path"], scene, vl, head, note, info, sanitised, catalog, audit)


# ----------------------------------------------------------------------------- sanity

def sanity_checks(h: SceneHandle, cfg, width: int = 720, spp: int = 4, backface: bool = True,
                  out_dir: str | None = None) -> dict:
    """A low-resolution panorama from the head origin: hole and back-face fractions (solid-angle
    weighted) and the nadir / zenith depth, median over the top and bottom rows (h // 100 rows,
    as fov-3d-vision inspect_preview)."""
    import tempfile
    import bpy
    from . import cameras, engine
    from .config import apply, readback
    from ..sensors import PanoramaModel
    from .. import record

    scene, vl = h.scene, h.view_layer
    model = PanoramaModel(width)
    c = cfg.with_(passes=("combined", "position", "object_index"), passes_exact=True, exr_codec="NONE",
                  multilayer=True, pixel_filter="BOX", filter_width=1.0, spp=spp, denoise=False)
    prev_cam = scene.camera
    apply(scene, c, model.width, model.height, vl)
    cam = cameras.build(scene, model, c, name="FOVRENDER_SANITY")
    pose = Rig().fixed_pose(h.head, "C")
    cameras.pose(cam, pose)
    tmp = out_dir or tempfile.mkdtemp(prefix="fovrender-sanity-")
    os.makedirs(tmp, exist_ok=True)
    exr = os.path.join(tmp, "sanity.exr")
    secs = engine.render_once(scene, spp, c.seed, exr)
    p = engine.read_passes(exr)
    rec = record.build(model, p, pose.centre_w, pose.R_wb, h.head.R_wh, eye="C", look_id="sanity", access="oracle")
    w = rec.footprint_sr / rec.footprint_sr.sum()
    hit = np.isfinite(rec.distance_m)
    rows = max(1, model.height // 100)
    dist = rec.as_raster("distance_m", fill=np.inf)
    hitr = np.isfinite(dist)
    nadir, zenith = dist[-rows:][hitr[-rows:]], dist[:rows][hitr[:rows]]
    out = {"width": width, "spp": spp, "render_seconds": round(secs, 3), "device": scene.cycles.device,
           "hole_fraction": float(w[~hit].sum()),
           "nadir_depth_m": float(np.median(nadir)) if nadir.size else None,
           "zenith_depth_m": float(np.median(zenith)) if zenith.size else None,
           "consistency": record.sensor_consistency(rec), "settings": readback(scene, cam, vl)}
    if backface:
        mat = bpy.data.materials.new("FOVRENDER_BACKFACE")
        if mat.node_tree is None and hasattr(mat, "use_nodes"):
            mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        geo, em, outn = (nt.nodes.new(t) for t in ("ShaderNodeNewGeometry", "ShaderNodeEmission", "ShaderNodeOutputMaterial"))
        nt.links.new(geo.outputs["Backfacing"], em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])
        world, override = scene.world, vl.material_override
        vl.material_override, scene.world = mat, None
        try:
            bexr = os.path.join(tmp, "backface.exr")
            engine.render_once(scene, 4, c.seed, bexr)
            bf = engine.read_passes(bexr)["combined"][..., 0].reshape(-1)[rec.raster_index]
        finally:
            vl.material_override, scene.world = override, world
            bpy.data.materials.remove(mat)
        out["backface_fraction"] = float(w[hit & (bf > 0.5)].sum())
    bpy.data.objects.remove(cam)
    scene.camera = prev_cam
    return out
