"""Place the EYE camera in a scene that does not have one, headlessly.

    blender -b scene.blend -P tools/place_eye.py -- --dry-run
    blender -b scene.blend -P tools/place_eye.py -- --location 0 0 1.2 --yaw 30 --out scene_eye.blend
    blender -b scene.blend -P tools/place_eye.py -- --from-camera renderCam --height 1.2 --out scene_eye.blend

Tier-3 files come with a shot camera, not an eye: demo interiors are built to look right from
one viewpoint. README step 3 places EYE by hand in the GUI; this does the same thing from a
script, so the vantage point is a recorded number instead of a mouse gesture.

--dry-run prints the scene bounds and every existing camera's pose and exits, which is how you
pick the arguments for the real run. Writing to a new --out keeps the demo file pristine; save
it beside the original so its relative texture paths still resolve.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bl_common import EYE_NAME, eye_record, make_eye, rigid, script_args  # noqa: E402


def scene_bounds(scene: bpy.types.Scene):
    """World-space bounding box over visible mesh objects."""
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for ob in scene.objects:
        if ob.type != "MESH" or ob.hide_render:
            continue
        for corner in ob.bound_box:
            p = ob.matrix_world @ Vector(corner)
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
    return (None, None) if math.isinf(lo.x) else (lo, hi)


def report(scene: bpy.types.Scene) -> None:
    lo, hi = scene_bounds(scene)
    print(f"[place_eye] scene {scene.name!r}, unit scale {scene.unit_settings.scale_length}")
    if lo is None:
        print("[place_eye] no visible mesh objects")
    else:
        print(f"[place_eye] bounds min ({lo.x:.3f}, {lo.y:.3f}, {lo.z:.3f}) "
              f"max ({hi.x:.3f}, {hi.y:.3f}, {hi.z:.3f})")
        print(f"[place_eye] centre  ({(lo.x + hi.x) / 2:.3f}, {(lo.y + hi.y) / 2:.3f}, "
              f"{(lo.z + hi.z) / 2:.3f})  extent ({hi.x - lo.x:.3f}, {hi.y - lo.y:.3f}, {hi.z - lo.z:.3f})")
    for ob in scene.objects:
        if ob.type == "CAMERA":
            rec = eye_record(ob)
            print(f"[place_eye] camera {ob.name!r} at {rec['position_m']} forward {rec['forward']} "
                  f"yaw {yaw_of(ob):.2f} deg")


def yaw_of(cam: bpy.types.Object) -> float:
    """Yaw in make_eye's convention: forward = (-sin yaw, cos yaw, 0) at zero pitch."""
    f = rigid(cam.matrix_world).to_3x3() @ Vector((0.0, 0.0, -1.0))
    return math.degrees(math.atan2(-f.x, f.y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", help="open this file first (otherwise use the file Blender was started with)")
    ap.add_argument("--out", help="save here; required unless --dry-run")
    ap.add_argument("--location", type=float, nargs=3, metavar=("X", "Y", "Z"))
    ap.add_argument("--yaw", type=float, default=0.0, help="degrees about world Z; 0 -> +Y, -90 -> +X")
    ap.add_argument("--pitch", type=float, default=0.0, help="degrees, up positive; keep 0 for a level eye")
    ap.add_argument("--from-camera", help="take position and yaw from this camera")
    ap.add_argument("--height", type=float, help="override Z (seated eye height, ~1.2 m)")
    ap.add_argument("--dry-run", action="store_true", help="print bounds and cameras, change nothing")
    args = ap.parse_args(script_args())

    if args.blend:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))
    scene = bpy.context.scene
    report(scene)
    if args.dry_run:
        return
    if not args.out:
        raise SystemExit("--out is required unless --dry-run")

    loc, yaw = args.location, args.yaw
    if args.from_camera:
        src = bpy.data.objects.get(args.from_camera)
        if src is None or src.type != "CAMERA":
            raise SystemExit(f"no camera named {args.from_camera!r}")
        loc, yaw = list(rigid(src.matrix_world).translation), yaw_of(src)
    if loc is None:
        raise SystemExit("give --location or --from-camera")
    if args.height is not None:
        loc = [loc[0], loc[1], args.height]

    old = bpy.data.objects.get(EYE_NAME)  # idempotent: a second run replaces the first EYE
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)

    eye = make_eye(scene, Vector(loc), yaw_deg=yaw, pitch_deg=args.pitch)
    bpy.context.view_layer.update()
    scene["fov_kind"] = "mesh"
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print(f"[place_eye] EYE at {eye_record(eye)['position_m']} yaw {yaw:.2f} pitch {args.pitch:.2f}")
    print(f"[place_eye] saved {out}")


main()
