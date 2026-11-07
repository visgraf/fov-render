"""Issue #9 diagnostic (not a gate): render one fixation pair of the Playground-2 Classroom run one
way, and fingerprint the interpreter and the opened scene.

    # binary, Classroom loaded on the command line
    blender -b CLASSROOM --python-exit-code 1 -P docs/runs/issue-9/render_cell.py -- --mode cmdline --out DIR
    # binary, open_mainfile
    blender -b --python-exit-code 1 -P docs/runs/issue-9/render_cell.py -- --mode open --out DIR
    # wheel, open_mainfile
    .venv/bin/python docs/runs/issue-9/render_cell.py -- --mode open --out DIR

Options: --device CPU|OPTIX (CPU), --spp 16, --fixation 355, --frozen the frozen run, --blend the
Classroom. The look is what tools/gfe_session.py renders for that fixation: preset ab1a, seeds
L 2111 / R 2112, Rig(IPD 63 mm, baseline_projected), legacy Object Index numbering, the gaze from
the frozen run's trajectory. Its calibration must be byte-identical to the frozen run's.
Writes DIR/look/raw_{L,R}.exr, DIR/catalog.json, DIR/fingerprint.json, DIR/cell.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

F3D = Path("/home/lvelho/rd/f3d-vision")
SEEDS = {"L": 2111, "R": 2112}
IPD_M, VERGENCE_M, ROLL = 0.063, 2.10, "baseline_projected"


def _sha(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def _rna(obj, skip=("rna_type",)) -> dict:
    """Every readable scalar / enum / vector property of an RNA struct."""
    out = {}
    for p in obj.bl_rna.properties:
        k = p.identifier
        if k in skip or p.type in ("POINTER", "COLLECTION"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:  # noqa: BLE001
            continue
        try:
            v = list(v) if not isinstance(v, (str, bytes)) and hasattr(v, "__len__") else v
        except TypeError:
            pass
        if isinstance(v, (set, frozenset)):
            v = sorted(v)
        out[k] = v if isinstance(v, (bool, int, float, str, list, type(None))) else str(v)
    return out


def fingerprint(scene) -> dict:
    import bpy
    app = bpy.app
    libs = {}
    for name in ("ocio", "oiio", "openexr", "opensubdiv", "openvdb", "alembic", "usd", "sdl", "ffmpeg", "build_options"):
        m = getattr(app, name, None)
        if m is None:
            continue
        libs[name] = {k: str(getattr(m, k)) for k in dir(m) if not k.startswith("_") and not callable(getattr(m, k))}
    local = bpy.utils.resource_path("LOCAL")
    cm_dir = os.path.join(local, "datafiles", "colormanagement")
    ocio_cfg = os.environ.get("OCIO") or os.path.join(cm_dir, "config.ocio")
    vs, ds = scene.view_settings, scene.display_settings
    images = []
    for img in bpy.data.images:
        before = bool(img.has_data)
        try:
            size = list(img.size)            # forces a load
        except Exception as e:  # noqa: BLE001
            size = f"error: {e}"
        ab = bpy.path.abspath(img.filepath, library=img.library) if img.filepath else ""
        images.append({"name": img.name_full, "filepath": img.filepath, "abspath": os.path.normpath(ab) if ab else "",
                       "exists": bool(ab) and os.path.exists(ab), "library": img.library.filepath if img.library else None,
                       "source": img.source, "type": img.type, "packed": img.packed_file is not None,
                       "has_data_before": before, "size": size, "has_data_after": bool(img.has_data),
                       "colorspace": img.colorspace_settings.name, "is_float": bool(img.is_float),
                       "alpha_mode": img.alpha_mode, "file_format": img.file_format,
                       "users": img.users, "sha256": _sha(ab) if ab and os.path.isfile(ab) else None})
    materials = {}
    for m in bpy.data.materials:
        nodes = []
        if m.node_tree is not None:
            for n in m.node_tree.nodes:
                d = {"name": n.name, "type": n.bl_idname}
                if getattr(n, "image", None) is not None:
                    d["image"] = n.image.name_full
                    d["interpolation"] = getattr(n, "interpolation", None)
                    d["projection"] = getattr(n, "projection", None)
                if n.bl_idname == "ShaderNodeGroup" and n.node_tree is not None:
                    d["group"] = n.node_tree.name_full
                nodes.append(d)
        materials[m.name_full] = {"library": m.library.filepath if m.library else None, "nodes": nodes}
    groups = {g.name_full: [{"name": n.name, "type": n.bl_idname, "image": n.image.name_full if getattr(n, "image", None) else None}
                            for n in g.nodes] for g in bpy.data.node_groups}
    objects = {}
    for o in scene.objects:
        objects[o.name_full] = {"type": o.type, "pass_index": int(o.pass_index),
                                "materials": [s.material.name_full if s.material else None for s in o.material_slots]}
        if o.instance_type == "COLLECTION" and o.instance_collection:
            for x in o.instance_collection.all_objects:
                objects.setdefault(x.name_full, {"type": x.type, "pass_index": int(x.pass_index), "via": o.name_full,
                                                 "materials": [s.material.name_full if s.material else None for s in x.material_slots]})
    prefs = bpy.context.preferences
    cprefs = prefs.addons["cycles"].preferences if "cycles" in prefs.addons else None
    return {
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "bpy_app": {"version_string": app.version_string, "build_hash": str(app.build_hash), "build_date": str(app.build_date),
                    "build_branch": str(getattr(app, "build_branch", "")), "binary_path": app.binary_path,
                    "background": app.background, "factory_startup": bool(getattr(app, "factory_startup", False))},
        "libraries": libs,
        "resource_paths": {k: bpy.utils.resource_path(k) for k in ("LOCAL", "USER", "SYSTEM")},
        "colour_management": {"ocio_config": ocio_cfg, "ocio_env": os.environ.get("OCIO"), "ocio_config_sha256": _sha(ocio_cfg),
                              "display_device": ds.display_device, "view_transform": vs.view_transform, "look": vs.look,
                              "exposure": vs.exposure, "gamma": vs.gamma, "use_curve_mapping": vs.use_curve_mapping,
                              "sequencer_colorspace": scene.sequencer_colorspace_settings.name,
                              "working_space": getattr(scene.display_settings, "working_space", None)},
        "scene_render": _rna(scene.render), "scene_cycles": _rna(scene.cycles),
        "world": {"name": scene.world.name_full if scene.world else None,
                  "nodes": [n.bl_idname for n in scene.world.node_tree.nodes] if scene.world and scene.world.node_tree else None},
        "libraries_blend": [{"filepath": lib.filepath, "abspath": os.path.normpath(bpy.path.abspath(lib.filepath)),
                             "exists": os.path.exists(bpy.path.abspath(lib.filepath))} for lib in bpy.data.libraries],
        "images": images, "materials": materials, "node_groups": groups, "objects": objects,
        "preferences": {"system": _rna(prefs.system), "filepaths": _rna(prefs.filepaths), "edit": _rna(prefs.edit),
                        "experimental": _rna(prefs.experimental), "addons": sorted(a.module for a in prefs.addons),
                        "cycles": (_rna(cprefs) if cprefs else None),
                        "cycles_devices": ([{"name": d.name, "type": d.type, "use": d.use} for d in cprefs.devices] if cprefs else None)},
        "env": {k: v for k, v in os.environ.items() if k.startswith(("OCIO", "BLENDER", "CYCLES", "OIIO", "PYTHON", "FOVRENDER", "CUDA", "OPTIX"))},
    }


def main() -> None:
    import bpy
    from fov_render import Gaze, HeadPose, Rig, Session, compat, preset
    from fov_render.bl import env
    from fov_render.sensors import Perspective
    env.prepare()
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cmdline", "open"])
    ap.add_argument("--device", default="CPU", choices=["CPU", "OPTIX"])
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--fixation", type=int, default=355)
    ap.add_argument("--frozen", default=str(F3D / "previews" / "playground-sparse-lattice"))
    ap.add_argument("--blend", default=str(F3D / "scenes" / "classroom" / "classroom_eye.blend"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    loaded = bpy.data.filepath
    if a.mode == "cmdline" and not (loaded and os.path.samefile(loaded, a.blend)):
        raise SystemExit(f"--mode cmdline needs the Classroom loaded on the command line; loaded: {loaded!r}")
    if a.mode == "open" and loaded:
        raise SystemExit(f"--mode open needs no file on the command line; loaded: {loaded!r}")
    frozen = Path(a.frozen)
    traj = json.loads((frozen / "trajectory.json").read_text())["trajectory"]
    t = traj[a.fixation - 1]
    if int(t["n"]) != a.fixation:
        raise SystemExit(f"trajectory entry {a.fixation - 1} is fixation {t['n']}")
    cfg = preset("ab1a", device=a.device, spp=a.spp)
    rig = Rig(ipd_m=IPD_M, roll=ROLL)
    sensor = Perspective()
    model = sensor.model(cfg.profile)
    s = Session.in_process(cfg, rig, blend=None if a.mode == "cmdline" else a.blend, access="oracle",
                           instance_ids="legacy", prepare=sensor, formats=("f3d_ab1a",), workdir=out / "session")
    r = s.ready
    head = HeadPose(__import__("numpy").asarray(r["head"]["head_R_wh"], float),
                    __import__("numpy").asarray(r["head"]["head_origin_w_m"], float))
    gaze = Gaze.yaw_pitch(float(t["yaw_deg"]), float(t["pitch_deg"]), VERGENCE_M)
    calib = compat.f3d_json(compat.f3d_calibration(cfg.profile, model, rig, head, gaze))
    frozen_calib = (frozen / "fixations" / f"fix-{a.fixation:04d}" / "calibration.json").read_text()
    look = s.look(sensor, ("L", "R"), gaze, spp=a.spp, seeds=SEEDS, out_dir=out / "look")
    h = s.handler.scene_handle
    (out / "catalog.json").write_text(json.dumps(h.catalog, indent=1) + "\n")
    (out / "fingerprint.json").write_text(json.dumps(fingerprint(h.scene), indent=1, default=str) + "\n")
    cell = {"schema": "fov-render-issue9-cell-v1", "mode": a.mode, "device": a.device, "backend": look.response["device"],
            "spp": a.spp, "seeds": SEEDS, "fixation": a.fixation, "gaze_yaw_pitch_deg": [t["yaw_deg"], t["pitch_deg"]],
            "calibration_byte_identical_to_frozen": calib == frozen_calib, "opened": r["scene"]["file"].get("opened"),
            "interpreter": "binary" if os.path.basename(bpy.app.binary_path or "").startswith("blender") else "wheel",
            "exr": {e: str(p) for e, p in look.exr.items()},
            "render_seconds": {e: v["render_seconds"] for e, v in look.response["eyes"].items()},
            "settings": look.response["settings"]}
    (out / "cell.json").write_text(json.dumps(cell, indent=1, default=str) + "\n")
    print(f"[issue-9 cell] {cell['interpreter']} {a.mode} {a.device}: calibration equal {cell['calibration_byte_identical_to_frozen']}, "
          f"opened {cell['opened']!r}, wrote {out}", flush=True)
    if not cell["calibration_byte_identical_to_frozen"]:
        raise SystemExit("the calibration is not the frozen run's: wrong look")


if __name__ == "__main__":
    from fov_render.bl.env import loud
    loud(lambda: (main(), sys.stdout.flush(), os._exit(0)), "[issue-9 cell]")
