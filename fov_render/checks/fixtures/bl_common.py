"""Shared helpers for Blender-side scripts.

Run inside Blender 5.2 (`blender -b [file.blend] -P tools/<script>.py -- <args>`)
or with the `bpy` wheel (`python tools/<script>.py -- <args>`).

Conventions used by every script in this folder
-----------------------------------------------
* Units are metres. The head is fixed; the EYE object marks the eye centre.
* EYE is a camera object: local -Z is the primary gaze direction, local +Y is head-up.
  make_eye(yaw=0, pitch=0) looks along world +Y with world +Z up.
* Equirectangular pixel (u, v) in [0,1]^2, v = 0 at the top row, maps to the EYE-frame direction
      lon = (u - 0.5) * 2*pi,  lat = (0.5 - v) * pi
      d   = (sin(lon)*cos(lat), sin(lat), -cos(lon)*cos(lat))
  (verified against the Position pass in Blender 5.2.1).
"""
from __future__ import annotations

import argparse
import math
import sys

import bpy
from mathutils import Matrix, Vector

EYE_NAME = "EYE"

# Two configurations, one flag apart (CLAUDE.md, "Cost classes"). `small` is for building and
# debugging: its reference renders in about a minute and a fixation in well under a second.
# `full` is for the number that goes in a report. Geometry, sample record and checks are
# identical between them; only spacing and sample counts differ. The fixation spp is at most a
# sixteenth of the reference spp so D7 holds by construction (1/sqrt(16) = 0.25 <= 1/3).
PROFILES = {
    "small": {"s0": 0.1, "ref_width": 3600, "ref_spp": 1024, "fix_spp": 64, "filter": "BOX"},
    "full": {"s0": 0.05, "ref_width": 7200, "ref_spp": 8192, "fix_spp": 256, "filter": "BOX"},
}


def add_profile(ap: argparse.ArgumentParser, argv: list[str], **dest_to_field: str) -> None:
    """Add --profile to a parser. A profile only replaces *defaults*, so any flag given
    explicitly still wins. dest_to_field maps parser destinations to PROFILES fields, e.g.
    add_profile(ap, argv, width="ref_width", spp="ref_spp"). Must be called after the
    mapped arguments have been added and before parse_args."""
    ap.add_argument("--profile", choices=sorted(PROFILES), default=None,
                    help="small: build and debug; full: the reported numbers. Sets defaults for "
                         + ", ".join(f"--{d.replace('_', '-')}" for d in dest_to_field))
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--profile", choices=sorted(PROFILES), default=None)
    chosen, _ = pre.parse_known_args(argv)
    if chosen.profile:
        prof = PROFILES[chosen.profile]
        ap.set_defaults(**{dest: prof[field] for dest, field in dest_to_field.items()})


def script_args() -> list[str]:
    """Arguments after the '--' separator."""
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


def ensure_cycles(scene: bpy.types.Scene) -> None:
    if "cycles" not in bpy.context.preferences.addons:
        import addon_utils
        addon_utils.enable("cycles", default_set=True)
    scene.render.engine = "CYCLES"


def setup_device(scene: bpy.types.Scene, prefer: str = "OPTIX") -> str:
    """Use OptiX (or CUDA) if a GPU of that type exists, else CPU. Returns the backend used."""
    cprefs = bpy.context.preferences.addons["cycles"].preferences
    candidates = [] if prefer == "CPU" else [prefer] + (["CUDA"] if prefer != "CUDA" else [])
    for backend in candidates:
        try:
            cprefs.compute_device_type = backend
        except TypeError:
            continue
        cprefs.refresh_devices()
        gpus = [d for d in cprefs.devices if d.type == backend]
        if gpus:
            for d in cprefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            return backend
    scene.cycles.device = "CPU"
    return "CPU"


def make_eye(scene: bpy.types.Scene, location, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> bpy.types.Object:
    """Create the EYE camera. yaw about world Z (0 -> +Y, +90 -> -X, -90 -> +X); pitch up is positive."""
    cam = bpy.data.cameras.new(EYE_NAME)
    cam.lens = 18.0  # wide, only for looking through it in the GUI
    cam.clip_start = 0.001
    eye = bpy.data.objects.new(EYE_NAME, cam)
    scene.collection.objects.link(eye)
    eye.location = location
    eye.rotation_mode = "XYZ"
    eye.rotation_euler = (math.radians(90.0 + pitch_deg), 0.0, math.radians(yaw_deg))
    return eye


def find_eye(scene: bpy.types.Scene) -> tuple[bpy.types.Object, str]:
    """EYE object if present, else fall back to the scene camera (with a note)."""
    eye = bpy.data.objects.get(EYE_NAME)
    if eye is not None:
        return eye, "EYE object"
    if scene.camera is not None:
        return scene.camera, f"no EYE object; using scene camera '{scene.camera.name}'"
    raise SystemExit("No EYE object and no scene camera: add a camera named 'EYE' at the vantage point.")


def rigid(matrix: Matrix) -> Matrix:
    """Strip scale/shear from a world matrix, keeping rotation and translation."""
    loc, rot, _ = matrix.decompose()
    return Matrix.Translation(loc) @ rot.to_matrix().to_4x4()


def eye_record(eye: bpy.types.Object) -> dict:
    m = rigid(eye.matrix_world)
    r = m.to_3x3()
    fwd = r @ Vector((0.0, 0.0, -1.0))
    up = r @ Vector((0.0, 1.0, 0.0))
    q = m.to_quaternion()
    return {
        "object": eye.name,
        "position_m": [round(x, 6) for x in m.translation],
        "forward": [round(x, 6) for x in fwd],
        "up": [round(x, 6) for x in up],
        "quat_wxyz": [round(x, 6) for x in q],
    }


def configure_multilayer_exr(scene: bpy.types.Scene) -> None:
    """32-bit multilayer EXR, single part.

    Blender 5.2 writes one EXR *part per pass* by default; readers that only look at
    part 0 silently see just 'Combined'. Interleaving keeps all passes in one part.
    """
    im = scene.render.image_settings
    im.media_type = "MULTI_LAYER_IMAGE"
    im.file_format = "OPEN_EXR_MULTILAYER"
    im.color_depth = "32"
    im.exr_codec = "ZIP"
    im.use_exr_interleave = True


def pin_seed(scene: bpy.types.Scene) -> list[str]:
    """Remove any animation on `cycles.seed` so a seed set by a script actually takes.

    Blender's Classroom keyframes the scene's `cycles.seed` (one key at frame 1). Every
    render evaluates the frame, so the keyframe overwrote `c.seed` set a line earlier and two
    "different seed" renders came back identical (measured 2026-09-13, caught by
    noise_floor.py's guard). Drivers are removed for the same reason. Returns a description
    of what was removed, empty if nothing was. Callers that depend on the seed should still
    check `scene.cycles.seed` after rendering.
    """
    removed: list[str] = []
    ad = scene.animation_data
    if ad is None:
        return removed
    act = ad.action
    if act is not None:
        for layer in act.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    for fc in list(cb.fcurves):
                        if fc.data_path == "cycles.seed":
                            keys = [(k.co[0], k.co[1]) for k in fc.keyframe_points]
                            mods = [m.type for m in fc.modifiers]
                            cb.fcurves.remove(fc)
                            removed.append(f"fcurve on cycles.seed in action {act.name!r}, "
                                           f"keys {keys}, modifiers {mods}")
    for d in list(ad.drivers):
        if d.data_path == "cycles.seed":
            ad.drivers.remove(d)
            removed.append(f"driver on cycles.seed")
    return removed


def node_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None and hasattr(mat, "use_nodes"):  # use_nodes is deprecated in 5.x
        mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat
