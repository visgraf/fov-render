"""Make a derived copy of a .blend with an EYE camera: the fixed head of fov_render (issue #4).

The original file is never changed. The copy is saved next to it by default, so relative texture
paths keep working, with a JSON sidecar recording the source's sha256, the EYE pose, the command
and the copy's sha256.

    fov-render eye --blend IN.blend --out OUT.blend \
        (--position X Y Z --yaw DEG | --from-camera [NAME] --forward-m M) [--pitch DEG] [--roll DEG]
    blender -b --python-exit-code 1 -P tools/make_eye_scene.py -- (same options)   (by hand)

--from-camera places the EYE M metres along the named camera's view direction (the scene camera by
default) and takes the camera's yaw. The head is level (pitch and roll 0) unless --pitch / --roll
say otherwise. Yaw is about world +Z, 0 looks along +Y, positive turns left (fov-3d-vision's
make_eye). A file that already has an EYE is refused unless --replace-eye.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

PREFIX = "[make-eye-scene]"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def eye_rotation(yaw_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0):
    from mathutils import Euler, Matrix
    m = Euler((math.radians(90.0 + pitch_deg), 0.0, math.radians(yaw_deg)), "XYZ").to_matrix()
    if roll_deg:
        m = m @ Matrix.Rotation(math.radians(roll_deg), 3, "Z")       # about the view axis
    return m


def main(argv) -> int:
    import bpy
    from mathutils import Vector
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--position", nargs=3, type=float, metavar=("X", "Y", "Z"))
    g.add_argument("--from-camera", nargs="?", const="", default=None, metavar="NAME")
    ap.add_argument("--yaw", type=float, help="with --position: the head's yaw in degrees")
    ap.add_argument("--forward-m", type=float, default=0.0, help="with --from-camera")
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--replace-eye", action="store_true")
    a = ap.parse_args(argv)
    src, out = os.path.abspath(a.blend), os.path.abspath(a.out)
    if src == out:
        raise SystemExit(f"{PREFIX} the copy must not overwrite the original")
    bpy.ops.wm.open_mainfile(filepath=src)
    scene = bpy.context.scene
    old = bpy.data.objects.get("EYE")
    if old is not None:
        if not a.replace_eye:
            raise SystemExit(f"{PREFIX} {src} already has an EYE object (use --replace-eye)")
        bpy.data.objects.remove(old)
    if a.from_camera is not None:
        cam = bpy.data.objects.get(a.from_camera) if a.from_camera else scene.camera
        if cam is None:
            raise SystemExit(f"{PREFIX} no camera {a.from_camera or '(scene camera)'}")
        m = cam.matrix_world
        fwd = (m.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
        pos = m.translation + fwd * a.forward_m
        yaw = math.degrees(math.atan2(-fwd.x, fwd.y))
        how = {"from_camera": cam.name, "camera_position_m": list(m.translation), "camera_forward": list(fwd),
               "forward_m": a.forward_m}
    else:
        if a.yaw is None:
            raise SystemExit(f"{PREFIX} --position needs --yaw")
        pos, yaw, how = Vector(a.position), a.yaw, {"position_m": a.position}
    data = bpy.data.cameras.new("EYE")
    eye = bpy.data.objects.new("EYE", data)
    scene.collection.objects.link(eye)
    eye.matrix_world = eye_rotation(yaw, a.pitch, a.roll).to_4x4()
    eye.location = pos
    bpy.context.view_layer.update()
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=out, relative_remap=True, compress=False)
    side = {"schema": "fov-render-eye-scene-v1", "source": src, "source_sha256": _sha256(src), "out": out,
            "out_sha256": _sha256(out), "eye": {"position_m": list(pos), "yaw_deg": yaw, "pitch_deg": a.pitch,
                                                "roll_deg": a.roll, "matrix_world": [list(r) for r in eye.matrix_world]},
            "placed": how, "argv": argv, "blender": bpy.app.version_string}
    Path(out + ".json").write_text(json.dumps(side, indent=1) + "\n")
    print(f"{PREFIX} wrote {out}: EYE at {[round(x, 4) for x in pos]} m, yaw {yaw:.3f} deg, "
          f"pitch {a.pitch}, roll {a.roll}", flush=True)
    return 0
