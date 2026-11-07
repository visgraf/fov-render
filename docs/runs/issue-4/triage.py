"""Issue #4, part A: triage a candidate scene for migration step 7 (not a gate).

    blender -b --python-exit-code 1 -P docs/runs/issue-4/triage.py -- --blend SCENE.blend --out DIR [--device OPTIX]

Opens the file through the scene layer (`scene.open_scene(path, scale_length=None)`, numbering
`complete`, D13) and reports, in DIR/triage.json:
  file        sha256, size, the Blender version it was saved with, units, frame range, render engine,
              linked libraries, images that do not load (has_data after img.size), Python the file
              would run if auto-run were on (scripted drivers, registered text blocks)
  head        EYE or the scene-camera fallback, position, forward / pitch / roll, and whether the
              head sits inside geometry (back-face first hits of rays cast from it)
  catalogue   count, direct, via_instance
  unnamed     render-visible objects the Object Index cannot number (CURVES, POINTCLOUD, VOLUME,
              GREASEPENCIL), particle systems that render, geometry-nodes instancers, vertex / face
              instancers, and the depsgraph's instances whose object carries pass index 0
  audit       flagged objects by flag; also the collection-instance objects, which material_audit
              does not visit; the 10 flagged objects that cover the most solid angle in the panorama
  animation   what sanitise removed
  sanity      h.sanity(...): 720 px, 4 spp
  looks       one look per sensor from the head, gaze (0, 0): foveated (small, C), perspective pair
              (small, L R, vergence 2.1 m), panorama 720 px (C); render time or the error. From the
              panorama: miss fraction and the solid angle of hits with Object Index 0
Scripts in the file never run: Blender's auto-run stays off (no -y).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

UNNUMBERED_TYPES = {"CURVES", "POINTCLOUD", "VOLUME", "GREASEPENCIL", "GPENCIL"}
INSTANCE_NODES = {"GeometryNodeInstanceOnPoints", "GeometryNodeCollectionInfo", "GeometryNodeObjectInfo",
                  "GeometryNodeRealizeInstances", "GeometryNodeDistributePointsOnFaces", "GeometryNodeInstancesToPoints"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reachable_objects(scene):
    """(object, via) for the scene's objects and those reached only through collection instances."""
    seen, out = set(), []
    for o in scene.objects:
        seen.add(o.name_full)
        out.append((o, None))
    for o in scene.objects:
        if o.instance_type == "COLLECTION" and o.instance_collection:
            for x in o.instance_collection.all_objects:
                if x.name_full not in seen:
                    seen.add(x.name_full)
                    out.append((x, o.name_full))
    return out


def node_types(tree, depth=0) -> set:
    out = set()
    if tree is None or depth > 8:
        return out
    for n in tree.nodes:
        out.add(n.bl_idname)
        if n.bl_idname == "GeometryNodeGroup" and n.node_tree is not None:
            out |= node_types(n.node_tree, depth + 1)
    return out


def file_report(path: str) -> dict:
    import bpy
    sc = bpy.context.scene
    images, bad = [], []
    for img in bpy.data.images:
        if img.source not in ("FILE", "SEQUENCE", "TILED", "MOVIE"):
            continue
        ab = bpy.path.abspath(img.filepath, library=img.library) if img.filepath else ""
        try:
            size = list(img.size)
        except Exception as e:  # noqa: BLE001
            size = f"error: {e}"
        entry = {"name": img.name_full, "filepath": img.filepath, "exists": bool(ab) and os.path.exists(ab),
                 "packed": img.packed_file is not None, "source": img.source, "size": size,
                 "has_data": bool(img.has_data), "colorspace": img.colorspace_settings.name}
        images.append(entry)
        if not entry["has_data"]:
            bad.append(entry)
    scripted = []
    for idb in list(bpy.data.objects) + list(bpy.data.materials) + list(bpy.data.scenes) + list(bpy.data.node_groups) \
            + list(bpy.data.shape_keys) + list(bpy.data.lights) + list(bpy.data.worlds):
        ad = getattr(idb, "animation_data", None)
        for d in (ad.drivers if ad else []):
            if d.driver.type == "SCRIPTED" and not getattr(d.driver, "is_simple_expression", False):
                scripted.append({"id": idb.name_full, "path": d.data_path, "expression": d.driver.expression[:80]})
    return {"path": path, "size_bytes": os.path.getsize(path), "sha256": sha256(path),
            "saved_with": ".".join(str(x) for x in bpy.data.version), "blender": bpy.app.version_string,
            "unit_system": sc.unit_settings.system, "scale_length": sc.unit_settings.scale_length,
            "length_unit": sc.unit_settings.length_unit,
            "frame_start": sc.frame_start, "frame_end": sc.frame_end, "frame_current": sc.frame_current,
            "render_engine_saved": sc.render.engine, "scenes": [s.name for s in bpy.data.scenes],
            "view_layers": [{"name": v.name, "renders": bool(v.use), "is_context": v == bpy.context.view_layer,
                             "passes_on": [p for p in ("combined", "z", "position", "object_index", "normal", "environment")
                                           if getattr(v, "use_pass_" + p, False)]} for v in sc.view_layers],
            "use_single_layer_saved": bool(sc.render.use_single_layer),
            "cameras": [o.name for o in sc.objects if o.type == "CAMERA"],
            "scene_camera": sc.camera.name if sc.camera else None,
            "libraries": [{"filepath": lib.filepath, "exists": os.path.exists(bpy.path.abspath(lib.filepath))}
                          for lib in bpy.data.libraries],
            "images": {"count": len(images), "packed": sum(i["packed"] for i in images),
                       "not_loading": bad, "missing_files": [i["name"] for i in images if not i["exists"] and not i["packed"]]},
            "scripts_not_run": {"auto_run": bool(bpy.context.preferences.filepaths.use_scripts_auto_execute),
                                "registered_texts": [t.name for t in bpy.data.texts if t.use_module],
                                "scripted_drivers": scripted[:20], "scripted_driver_count": len(scripted)},
            "objects": {"total": len(sc.objects), "by_type": dict(collections.Counter(o.type for o in sc.objects))}}


def head_report(h) -> dict:
    import bpy
    import numpy as np
    R, o = np.asarray(h.head.R_wh), np.asarray(h.head.origin_w)
    right, up, fwd = R[:, 0], R[:, 1], -R[:, 2]
    dg = bpy.context.evaluated_depsgraph_get()
    n = 256
    k = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * k / n), math.pi * (1 + 5 ** 0.5) * k
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)
    hits = back = 0
    dmin = math.inf
    for d in dirs:
        ok, loc, nrm, _i, _ob, _m = h.scene.ray_cast(dg, o.tolist(), d.tolist())
        if ok:
            hits += 1
            dmin = min(dmin, float(np.linalg.norm(np.asarray(loc) - o)))
            if float(np.dot(np.asarray(nrm), d)) > 0:
                back += 1
    return {"eye_note": h.eye_note, "origin_w_m": o.tolist(), "forward_w": fwd.tolist(),
            "pitch_deg": math.degrees(math.asin(max(-1.0, min(1.0, fwd[2])))),
            "roll_deg": math.degrees(math.asin(max(-1.0, min(1.0, right[2])))),
            "rays": n, "ray_hits": hits, "ray_backface_first_hits": back,
            "nearest_hit_m": None if dmin == math.inf else dmin,
            "inside_geometry": back > n // 2}


def unnamed_report(h) -> dict:
    import bpy
    sc = h.scene
    objs = reachable_objects(sc)
    types = [{"name": o.name_full, "type": o.type, "via": via} for o, via in objs
             if o.type in UNNUMBERED_TYPES and not o.hide_render]
    psys, gn, dupli = [], [], []
    for o, via in objs:
        if o.hide_render:
            continue
        for m in o.modifiers:
            if m.type == "PARTICLE_SYSTEM" and m.show_render:
                s = m.particle_system.settings
                psys.append({"object": o.name_full, "via": via, "type": s.type, "render_as": s.render_type,
                             "count": s.count, "children": s.child_type,
                             "instance": (s.instance_object.name_full if s.instance_object else None)
                             or (s.instance_collection.name_full if s.instance_collection else None)})
            elif m.type == "NODES" and m.show_render and m.node_group is not None:
                t = node_types(m.node_group)
                gn.append({"object": o.name_full, "via": via, "group": m.node_group.name_full,
                           "instance_nodes": sorted(t & INSTANCE_NODES)})
        if o.instance_type in ("VERTS", "FACES"):
            dupli.append({"object": o.name_full, "instance_type": o.instance_type,
                          "children": [c.name_full for c in o.children]})
    dg = bpy.context.evaluated_depsgraph_get()
    inst = collections.Counter()
    zero = collections.Counter()
    for oi in dg.object_instances:
        ob = oi.object.original
        if ob.type not in ("MESH", "CURVE", "SURFACE", "FONT", "META", *UNNUMBERED_TYPES):
            continue
        kind = "instance" if oi.is_instance else "object"
        par = oi.parent.original.type if oi.is_instance and oi.parent else None
        key = f"{kind}{'/' + par if par else ''}:{ob.type}"
        inst[key] += 1
        if int(ob.pass_index) == 0:
            zero[key] += 1
    return {"unnumbered_types": types, "particle_systems": psys, "geometry_nodes": gn,
            "vertex_face_instancers": dupli,
            "depsgraph_instances": dict(inst), "depsgraph_instances_pass_index_0": dict(zero)}


def audit_report(h) -> dict:
    from fov_render.bl import scene as S
    flagged = h.audit["flagged_objects"]
    by_flag = collections.Counter(f for flags in flagged.values() for f in flags)
    via = {}
    for o, parent in reachable_objects(h.scene):
        if parent is None or o.type not in S.RENDERABLE or o.hide_render:
            continue
        f = sorted({x for s in o.material_slots for x in S.material_flags(s.material)})
        if f:
            via[o.name_full] = f
    return {"flagged_count": len(flagged), "by_flag": dict(by_flag), "flagged": flagged,
            "world_volume": h.audit["world_volume"],
            "via_instance_flagged_count": len(via), "via_instance_by_flag": dict(collections.Counter(f for v in via.values() for f in v)),
            "via_instance_flagged": dict(list(via.items())[:30])}


def looks_report(path: str, device: str, out: Path, catalog: dict, flagged: dict, head_m=None) -> dict:
    import numpy as np
    from fov_render import Gaze, RenderConfig, Rig, Session
    from fov_render.rig import HeadPose
    from fov_render.sensors import Foveated, Panorama, Perspective
    res = {}
    t0 = time.perf_counter()
    s = Session.in_process(RenderConfig(profile="small", device=device), Rig(), blend=path, access="oracle",
                           head=None if head_m is None else HeadPose.from_matrix(head_m), workdir=out / "session")
    res["session_open_s"] = time.perf_counter() - t0
    plan = [("foveated", Foveated(), ("C",), Gaze.yaw_pitch(0.0, 0.0, 2.1)),
            ("perspective_pair", Perspective(), ("L", "R"), Gaze.yaw_pitch(0.0, 0.0, 2.1)),
            ("panorama_720", Panorama(width=720), ("C",), None)]
    names = {e["id"]: e["name"] for e in catalog["instances"]}
    for label, sensor, eyes, gaze in plan:
        t = time.perf_counter()
        try:
            look = s.look(sensor, eyes, gaze, outputs=("exr", "records"), out_dir=out / "looks" / label)
            r = {"ok": True, "wall_s": time.perf_counter() - t, "device": look.response["device"], "spp": look.response["spp"],
                 "render_s": {e: v["render_seconds"] for e, v in look.response["eyes"].items()}}
            rec = look.records[0]
            w = rec.footprint_sr / rec.footprint_sr.sum()
            hit = np.isfinite(rec.distance_m)
            ids = np.asarray(rec.instance_id)
            r.update({"samples": int(rec.n_samples), "miss_fraction_sr": float(w[~hit].sum()),
                      "hit_id0_fraction_sr": float(w[hit & (ids == 0)].sum())})
            if label == "panorama_720":
                per = collections.Counter()
                for i, wt in zip(ids[hit].tolist(), w[hit].tolist()):
                    per[i] += wt
                big = sorted(((names.get(i, f"id {i}"), v) for i, v in per.items() if names.get(i) in flagged),
                             key=lambda x: -x[1])[:10]
                r["largest_flagged_by_solid_angle"] = [{"object": n, "flags": flagged[n], "fraction_sr": v} for n, v in big]
                r["ids_seen"] = len(per)
            res[label] = r
        except Exception as e:  # noqa: BLE001
            res[label] = {"ok": False, "wall_s": time.perf_counter() - t, "error": f"{type(e).__name__}: {e}",
                          "traceback": traceback.format_exc()[-1500:]}
    s.close()
    return res


def main() -> None:
    import bpy
    from fov_render.bl import config, env, scene as S
    env.prepare()
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="OPTIX", choices=["OPTIX", "CPU"])
    ap.add_argument("--head-forward-m", type=float, default=0.0,
                    help="provisional head: the scene camera moved this far along its view (explicit head)")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    path = os.path.abspath(a.blend)
    rep = {"schema": "fov-render-issue4-triage-v1", "device": a.device, "timings_s": {}}
    t = time.perf_counter()
    bpy.ops.wm.open_mainfile(filepath=path)
    rep["timings_s"]["raw_open"] = time.perf_counter() - t
    rep["file"] = file_report(path)
    t = time.perf_counter()
    head_m = None
    try:
        if a.head_forward_m:
            import numpy as np
            m = S.rigid4(bpy.context.scene.camera.matrix_world)
            fwd = -m[:3, 2]
            rep["head_provisional"] = {"from": bpy.context.scene.camera.name, "forward_m": a.head_forward_m,
                                       "camera_origin_w_m": m[:3, 3].tolist()}
            m[:3, 3] = m[:3, 3] + a.head_forward_m * fwd
            head_m = m
        h = S.open_scene(path, scale_length=None, eye=head_m)
    except Exception as e:  # noqa: BLE001
        rep["open_scene_error"] = f"{type(e).__name__}: {e}"
        if "orthonormal" not in str(e) or bpy.context.scene.camera is None:
            (out / "triage.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
            print(f"[issue-4 triage] open_scene failed: {rep['open_scene_error']}", flush=True)
            return
        # Diagnostic workaround, not a fix: the scene camera's float32 rotation, made orthonormal in
        # float64 (nearest rotation, SVD), passed as an explicit head.
        import numpy as np
        m = S.rigid4(bpy.context.scene.camera.matrix_world)
        r0 = m[:3, :3].copy()
        u, _sv, vt = np.linalg.svd(r0)
        m[:3, :3] = u @ vt
        rep["head_workaround"] = {"why": rep["open_scene_error"], "camera": bpy.context.scene.camera.name,
                                  "max_abs_RtR_minus_I_before": float(np.abs(r0.T @ r0 - np.eye(3)).max()),
                                  "det_before": float(np.linalg.det(r0)),
                                  "max_abs_change": float(np.abs(m[:3, :3] - r0).max()),
                                  "max_abs_RtR_minus_I_after": float(np.abs(m[:3, :3].T @ m[:3, :3] - np.eye(3)).max())}
        h = S.open_scene(path, scale_length=None, eye=m)
        head_m = m
    rep["timings_s"]["open_scene"] = time.perf_counter() - t
    rep["head"] = head_report(h)
    rep["catalogue"] = {k: h.catalog[k] for k in ("mode", "count", "direct", "via_instance", "via_id_zero")}
    rep["unnamed"] = unnamed_report(h)
    rep["audit"] = audit_report(h)
    rep["animation"] = {"removed": h.sanitised["animation_removed"], "switched_off": h.sanitised["switched_off"]}
    t = time.perf_counter()
    try:
        chk = h.sanity(config.RenderConfig(device=a.device), width=720, spp=4, out_dir=str(out / "sanity"))
        rep["sanity"] = {k: v for k, v in chk.items() if k not in ("settings", "consistency")}
        rep["sanity"]["consistency_p999_deg"] = chk["consistency"].get("p999_angle_deg")
    except Exception as e:  # noqa: BLE001
        rep["sanity"] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-1500:]}
    rep["timings_s"]["sanity"] = time.perf_counter() - t
    t = time.perf_counter()
    rep["looks"] = looks_report(path, a.device, out, h.catalog, h.audit["flagged_objects"], head_m)
    rep["timings_s"]["looks"] = time.perf_counter() - t
    (out / "catalog.json").write_text(json.dumps(h.catalog, indent=1) + "\n")
    (out / "triage.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(f"[issue-4 triage] wrote {out / 'triage.json'}", flush=True)


if __name__ == "__main__":
    from fov_render.bl.env import loud
    loud(lambda: (main(), sys.stdout.flush(), os._exit(0)), "[issue-4 triage]")
