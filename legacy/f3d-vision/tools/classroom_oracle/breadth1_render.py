"""Breadth-1: the Blender side of the spherical glance (synthetic test, preflight, canonical render).

Contract: docs/classroom-oracle/breadth-1-spherical-glance-contract.md, sections 3 and 5.  Normally
run through ``breadth1_glance.py``:

    blender -b --factory-startup --python-exit-code 1 -P tools/classroom_oracle/breadth1_render.py \
        -- --mode synthetic --seeds SEEDS --out DIR
    blender -b scenes/classroom/classroom_eye.blend --python-exit-code 1 \
        -P tools/classroom_oracle/breadth1_render.py -- --mode preflight --seeds SEEDS --catalog CATALOG --out DIR
    blender -b scenes/classroom/classroom_eye.blend --python-exit-code 1 \
        -P tools/classroom_oracle/breadth1_render.py -- --mode canonical --seeds SEEDS --catalog CATALOG --out DIR

All three share one camera/pass configuration function.  ``preflight`` stops before ``render()``.
``canonical`` renders exactly once into a new or empty directory.  Object ids come from the accepted
``classroom_oracle1_render._assign_instance_ids`` (unchanged); the camera matrix is ``[head_R_wh |
head_origin_w_m]`` as in the accepted Visual Language 1 reference renderer.  Failures exit nonzero
(Blender alone would exit 0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent  # tools/: the accepted renderer helpers (read-only use)
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(HERE))

import breadth1_spec as SP  # noqa: E402

PREFIX = "[breadth1-render]"


def args_after_dashes() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def new_dir(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure(scene, r_wh: np.ndarray, o_w: np.ndarray, device: str) -> tuple[object, dict]:
    """The frozen glance configuration (contract section 3); returns the camera and Blender's read-back."""
    import bpy
    from mathutils import Matrix
    from bl_common import configure_multilayer_exr, ensure_cycles, pin_seed, setup_device

    ensure_cycles(scene)
    backend = setup_device(scene, device)
    if backend != device:
        raise RuntimeError(f"requested device {device} unavailable (got {backend}); no silent fallback")
    seed_animation_removed = pin_seed(scene)
    r = scene.render
    r.resolution_x, r.resolution_y = SP.WIDTH, SP.HEIGHT
    r.resolution_percentage = 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0
    r.film_transparent = False
    r.use_motion_blur = False
    r.use_compositing = False
    r.use_sequencer = False
    r.use_single_layer = True
    r.use_border = False
    c = scene.cycles
    c.samples = SP.SPP
    c.seed = SP.SEED
    c.use_adaptive_sampling = False
    c.pixel_filter_type = SP.PIXEL_FILTER
    c.filter_width = SP.FILTER_WIDTH
    c.use_denoising = False
    vl = bpy.context.view_layer
    vl.use_pass_combined = True
    vl.use_pass_position = True
    vl.use_pass_object_index = True
    configure_multilayer_exr(scene)
    r.image_settings.exr_codec = "NONE"  # exr_lite reads uncompressed EXR only

    data = bpy.data.cameras.new("BREADTH1_GLANCE")
    data.type = SP.CAMERA_TYPE
    data.panorama_type = SP.PANORAMA_TYPE
    data.longitude_min, data.longitude_max = SP.LONGITUDE
    data.latitude_min, data.latitude_max = SP.LATITUDE
    data.clip_start, data.clip_end = SP.CLIP_M
    data.dof.use_dof = False
    cam = bpy.data.objects.new("BREADTH1_GLANCE", data)
    scene.collection.objects.link(cam)
    m = np.eye(4)
    m[:3, :3] = r_wh  # camera axes = head axes: +X right, +Y up, -Z forward
    m[:3, 3] = o_w
    cam.matrix_world = Matrix(m.tolist())
    scene.camera = cam
    bpy.context.view_layer.update()
    return cam, readback(scene, cam, backend, seed_animation_removed)


def readback(scene, cam, backend: str, seed_animation_removed=None) -> dict:
    import bpy
    d = cam.data
    r, c = scene.render, scene.cycles
    mw = np.array([[cam.matrix_world[i][j] for j in range(4)] for i in range(4)], np.float64)
    vl = bpy.context.view_layer
    return {
        "engine": r.engine, "device_backend": backend, "cycles_device": c.device,
        "resolution_wh": [r.resolution_x, r.resolution_y], "resolution_percentage": r.resolution_percentage,
        "pixel_aspect": [r.pixel_aspect_x, r.pixel_aspect_y],
        "camera_type": d.type, "panorama_type": d.panorama_type,
        "longitude_rad": [d.longitude_min, d.longitude_max], "latitude_rad": [d.latitude_min, d.latitude_max],
        "clip_m": [d.clip_start, d.clip_end],
        "samples": c.samples, "seed": c.seed, "adaptive_sampling": bool(c.use_adaptive_sampling),
        "denoising": bool(c.use_denoising), "motion_blur": bool(r.use_motion_blur),
        "pixel_filter": c.pixel_filter_type, "filter_width": c.filter_width,
        "film_transparent": bool(r.film_transparent),
        "passes": {"combined": bool(vl.use_pass_combined), "position": bool(vl.use_pass_position),
                   "object_index": bool(vl.use_pass_object_index)},
        "file_format": r.image_settings.file_format, "exr_codec": r.image_settings.exr_codec,
        "color_depth": r.image_settings.color_depth,
        "camera_matrix_world": mw.tolist(),
        "seed_animation_removed": seed_animation_removed,
        "frame_current": scene.frame_current,
    }


def render_once(scene, exr: Path) -> tuple[Path, float]:
    import bpy
    scene.render.filepath = str(exr)
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    secs = time.perf_counter() - t0
    if not exr.is_file():
        raise RuntimeError(f"render produced no EXR at {exr}")
    if scene.cycles.seed != SP.SEED:
        raise RuntimeError(f"cycles.seed changed during render: {scene.cycles.seed}")
    return exr, secs


def pose(seeds_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    seeds = json.loads(seeds_path.read_text())
    return np.asarray(seeds["head_R_wh"], np.float64), np.asarray(seeds["head_origin_w_m"], np.float64), seeds


# ---------------------------------------------------------------- synthetic plumbing scene
SYN_BOX_HALF = 4.0
SYN_MARKER_DIST = 2.5
SYN_MARKER_HALF = 0.25
# (name, pass_index, yaw, pitch, emission rgb)
SYN_MARKERS = [
    ("m_forward", 11, 0.0, 0.0, (1.0, 1.0, 1.0)),
    ("m_right", 12, 90.0, 0.0, (0.2, 0.9, 0.9)),
    ("m_up", 13, 30.0, 70.0, (0.9, 0.5, 0.1)),
    ("m_down", 14, -45.0, -50.0, (0.5, 0.2, 0.9)),
    ("m_behind_seam", 15, 180.0, 10.0, (0.9, 0.9, 0.9)),
    ("m_index0", 0, -90.0, -20.0, (0.6, 0.6, 0.6)),
]
# (name, pass_index, axis, sign, emission rgb); the floor (-Y) is deliberately absent
SYN_WALLS = [
    ("w_right", 1, 0, +1, (0.9, 0.05, 0.05)),
    ("w_left", 2, 0, -1, (0.05, 0.9, 0.05)),
    ("w_ceiling", 3, 1, +1, (0.05, 0.05, 0.9)),
    ("w_front", 4, 2, -1, (0.8, 0.8, 0.05)),
    ("w_back", 5, 2, +1, (0.8, 0.05, 0.8)),
]


def _emission(name: str, rgb) -> object:
    import bpy
    from bl_common import node_material
    mat = node_material(name)
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = 1.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def _mesh_object(name: str, verts_h: np.ndarray, faces, r_wh, o_w, pass_index: int, rgb):
    import bpy
    verts_w = verts_h @ r_wh.T + o_w
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts_w], [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.pass_index = int(pass_index)
    me.materials.append(_emission("mat_" + name, rgb))
    return ob


def build_synthetic(r_wh: np.ndarray, o_w: np.ndarray) -> dict:
    import bpy
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    nt = world.node_tree
    if nt is not None:
        bg = next((n for n in nt.nodes if n.type == "BACKGROUND"), None)
        if bg is not None:
            bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            bg.inputs["Strength"].default_value = 0.0
    a = SYN_BOX_HALF
    walls = []
    for name, pid, axis, sign, rgb in SYN_WALLS:
        u, v = [k for k in range(3) if k != axis]
        quad = []
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = np.zeros(3)
            p[axis], p[u], p[v] = sign * a, su * a, sv * a
            quad.append(p)
        _mesh_object(name, np.asarray(quad), [(0, 1, 2, 3)], r_wh, o_w, pid, rgb)
        walls.append({"name": name, "pass_index": pid, "axis": axis, "sign": sign, "half_size_m": a, "rgb": rgb})
    markers = []
    s = SYN_MARKER_HALF
    corners = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)], np.float64)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    for name, pid, yaw, pitch, rgb in SYN_MARKERS:
        c = SP.direction_h(yaw, pitch) * SYN_MARKER_DIST
        _mesh_object(name, corners + c, faces, r_wh, o_w, pid, rgb)
        markers.append({"name": name, "pass_index": pid, "yaw_deg": yaw, "pitch_deg": pitch,
                        "center_h_m": c.tolist(), "half_size_m": s, "rgb": rgb})
    return {"walls": walls, "markers": markers, "floor": "absent (index-0 no-geometry region)"}


# ---------------------------------------------------------------- modes
def catalog_check(catalog_path: Path) -> tuple[dict, dict]:
    import classroom_oracle1_render as R  # _assign_instance_ids (unchanged)
    accepted = {int(e["instance_id"]): e["object_name"] for e in json.loads(catalog_path.read_text())["instances"]}
    if len(accepted) != SP.CATALOG_COUNT:
        raise RuntimeError(f"accepted catalog has {len(accepted)} objects, not {SP.CATALOG_COUNT}")
    by_id, _by_name = R._assign_instance_ids()
    live = {int(k): v for k, v in by_id.items()}
    if live != accepted:
        diff = sorted(set(live.items()) ^ set(accepted.items()))[:10]
        raise RuntimeError(f"live id assignment differs from the accepted catalog: {diff}")
    return live, {"accepted_count": len(accepted), "live_count": len(live), "equal": True}


def eye_check(r_wh: np.ndarray, o_w: np.ndarray) -> dict:
    import bpy
    from bl_common import find_eye, rigid
    eye, note = find_eye(bpy.context.scene)
    m = rigid(eye.matrix_world)
    r = np.array([[m[i][j] for j in range(3)] for i in range(3)], np.float64)
    o = np.array([m.translation[i] for i in range(3)], np.float64)
    dr, do = float(np.max(np.abs(r - r_wh))), float(np.max(np.abs(o - o_w)))
    if max(dr, do) > SP.EYE_POSE_TOL:
        raise RuntimeError(f"EYE pose differs from seeds.json: rotation {dr:.3g}, origin {do:.3g}")
    return {"eye_note": note, "max_abs_rotation_diff": dr, "max_abs_origin_diff": do, "tolerance": SP.EYE_POSE_TOL}


def main() -> None:
    import bpy
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["synthetic", "preflight", "canonical"])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=SP.DEVICE)
    a = ap.parse_args(args_after_dashes())
    if a.device != SP.DEVICE:
        raise RuntimeError(f"the frozen device is {SP.DEVICE}")
    seeds_path = Path(a.seeds).resolve()
    r_wh, o_w, _seeds = pose(seeds_path)
    scene = bpy.context.scene
    meta = {"schema": "Breadth1-render-metadata-v1", "mode": a.mode, "blender_version": bpy.app.version_string,
            "blend": bpy.data.filepath or None, "scene": scene.name, "seeds": str(seeds_path),
            "seeds_sha256": sha256(seeds_path),
            "head_R_wh_requested": r_wh.tolist(), "head_origin_w_m_requested": o_w.tolist(),
            "truth": {"Combined": SP.TRUTH_REFERENCE, "Position": SP.TRUTH_REFERENCE,
                      "Object Index": SP.TRUTH_ORACLE},
            "note": "one global Blender observation; no gaze, no controller, no foveation, no stereo growth"}
    if a.mode == "synthetic":
        out = new_dir(Path(a.out))
        meta["synthetic_scene"] = build_synthetic(r_wh, o_w)
        _cam, cfg = configure(scene, r_wh, o_w, a.device)
        meta["config"] = cfg
        exr, secs = render_once(scene, out / "synthetic.exr")
    else:
        if not a.catalog:
            raise RuntimeError("--catalog is required")
        catalog_path = Path(a.catalog).resolve()
        live, cat = catalog_check(catalog_path)
        meta.update(catalog=str(catalog_path), catalog_sha256=sha256(catalog_path), catalog_check=cat,
                    instance_names={str(k): v for k, v in sorted(live.items())}, eye_check=eye_check(r_wh, o_w))
        _cam, cfg = configure(scene, r_wh, o_w, a.device)
        meta["config"] = cfg
        if a.mode == "preflight":
            out = Path(a.out).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            meta["rendered"] = False
            write_json(out, meta)
            print(f"{PREFIX} PREFLIGHT ok: catalog {cat['live_count']} equal, EYE pose within "
                  f"{SP.EYE_POSE_TOL}, device {cfg['device_backend']}; no render -> {out}", flush=True)
            return
        out = new_dir(Path(a.out))
        exr, secs = render_once(scene, out / "canonical.exr")
        meta["config_after_render"] = readback(scene, _cam, cfg["device_backend"])
    meta.update(rendered=True, render_seconds=round(secs, 3), exr=exr.name, exr_bytes=exr.stat().st_size,
                exr_sha256=sha256(exr), render_invocations=1)
    write_json(out / ("render-metadata.json" if a.mode == "canonical" else "synthetic-metadata.json"), meta)
    print(f"{PREFIX} {a.mode.upper()} rendered {SP.WIDTH}x{SP.HEIGHT} at {SP.SPP} spp on "
          f"{cfg['device_backend']} in {secs:.2f} s -> {exr}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise
        traceback.print_exc()
        print(f"{PREFIX} FAILED", flush=True)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
