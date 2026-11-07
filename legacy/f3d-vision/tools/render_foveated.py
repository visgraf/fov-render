"""Render one fixation through the foveated OSL custom camera.

    blender -b scene.blend -P tools/render_foveated.py -- --out fix/f000 [--yaw 20 --pitch -5]

The pieces are importable (tools/fixation_sequence.py renders many gazes in one session):
setup_foveated_camera once, then set_gaze / render_fixation / save_render_result per gaze,
and fixation_meta for the record.

Writes to --out:
    fix.exr    Combined + Depth (ray distance) + Normal + Position, 32-bit, single part
    meta.json  camera pose actually rendered, warp parameters, raster size, device, timing

The raster is square and the warp fills its inscribed disc; everything outside the disc
gets zero throughput. Mask with the analytic r <= 1 test on the raster index, never with
alpha (see the note in tools/foveated_camera.osl).

Resolution follows from the warp: N = 2 * (E2/s0) * ln(1 + e_max/E2), so --s0 sets the
foveal spacing and the raster size together. Pass --n to override and let s0 follow instead.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bl_common import (add_profile, configure_multilayer_exr, ensure_cycles, eye_record,  # noqa: E402
                       find_eye, pin_seed, rigid, script_args, setup_device)

HERE = os.path.dirname(os.path.abspath(__file__))


from warp import raster_size, s0_of  # noqa: E402,F401  (moved to warp.py for Phase B; re-exported)


def gaze_matrix(eye, yaw_deg: float, pitch_deg: float, offset_local=(0.0, 0.0, 0.0)) -> Matrix:
    """Camera pose for a gaze: the EYE's rigid pose, then intrinsic yaw-then-pitch about the
    eye centre, so the resulting elevation equals pitch exactly; the Y sign is negated so
    +yaw turns right (verified in A3). No translation of the head: D3. offset_local is the
    eye centre in the EYE (head) frame, (0,0,0) for the cyclopean eye of Phase A and
    (+-ipd/2, 0, 0) for the two eyes of Phase B (D12); the rotation is about that centre.
    Same composition as rig.camera_pose."""
    return (rigid(eye.matrix_world)
            @ Matrix.Translation(Vector(offset_local))
            @ Matrix.Rotation(math.radians(-yaw_deg), 4, "Y")
            @ Matrix.Rotation(math.radians(pitch_deg), 4, "X"))


def setup_foveated_camera(scene, eye, shader_path: str, e2: float, emax: float, n: int,
                          spp: int, exr_codec: str = "ZIP"):
    """Create the OSL camera once and configure the scene for fixation renders: raster n x n,
    box filter, uniform sampling, seed pinned, Combined + Depth + Normal + Position passes in a
    single-part 32-bit multilayer EXR. Returns the camera object; call set_gaze before each
    render. exr_codec "NONE" lets exr_lite read the file back inside Blender."""
    text = bpy.data.texts.load(os.path.abspath(shader_path))
    cam_data = bpy.data.cameras.new("FOVEATED")
    cam_data.type = "CUSTOM"
    cam_data.custom_mode = "INTERNAL"
    cam_data.custom_shader = text
    cam_data.clip_start, cam_data.clip_end = 0.001, 1.0e5
    cam_data.dof.use_dof = False
    if not cam_data.custom_bytecode:
        raise RuntimeError("OSL camera shader did not compile; see the console above for the error.")
    for name, value in (("E2", e2), ("e_max", emax)):
        cam_data.cycles_custom[name] = value
    cam = bpy.data.objects.new("FOVEATED", cam_data)
    scene.collection.objects.link(cam)
    cam.matrix_world = gaze_matrix(eye, 0.0, 0.0)
    scene.camera = cam

    r = scene.render
    r.resolution_x = r.resolution_y = n
    r.resolution_percentage = 100
    r.film_transparent = r.use_motion_blur = False
    r.use_compositing = r.use_sequencer = False
    r.use_single_layer = True
    r.use_persistent_data = True          # a sequence renders many fixations in one session
    c = scene.cycles
    for what in pin_seed(scene):        # a keyframed seed would silently reuse one seed per session
        print(f"[foveated] removed {what}")
    c.samples, c.seed = spp, 0
    c.use_adaptive_sampling = False
    c.time_limit = 0.0
    c.pixel_filter_type, c.filter_width = "BOX", 1.0
    try:
        c.use_denoising = False
    except (AttributeError, TypeError):
        pass
    vl = bpy.context.view_layer
    vl.use_pass_combined = vl.use_pass_z = vl.use_pass_normal = vl.use_pass_position = True
    configure_multilayer_exr(scene)
    r.image_settings.exr_codec = exr_codec
    bpy.context.view_layer.update()
    return cam


def set_gaze(cam, eye, yaw_deg: float, pitch_deg: float, offset_local=(0.0, 0.0, 0.0)) -> None:
    cam.matrix_world = gaze_matrix(eye, yaw_deg, pitch_deg, offset_local)
    bpy.context.view_layer.update()


_LAST_SAMPLING: dict = {}


def render_fixation(scene, spp: int, out_path: str | None = None, seed: int = 0) -> float:
    """Render the current gaze at spp and return the wall seconds of the render call. With
    out_path the file is written inside the call (write_still, as in A3); without it the caller
    saves the Render Result itself, outside the timer.

    With persistent data Cycles re-reads the sample count only when the scene datablock is
    tagged for update: a change of `cycles.samples` alone was ignored across 16/256/4096 spp
    (all 16 ms, measured 2026-09-13) while a seed change or `scene.update_tag()` made it take.
    So the scene is tagged whenever spp or seed differ from the previous render. Afterwards the
    settings are asserted to still hold, since frame evaluation can overwrite them (A2)."""
    c = scene.cycles
    key = (spp, seed)
    if _LAST_SAMPLING.get(scene.name) != key:
        c.samples, c.seed = spp, seed
        scene.update_tag()
        _LAST_SAMPLING[scene.name] = key
    if out_path:
        scene.render.filepath = out_path
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=bool(out_path))
    dt = time.perf_counter() - t0
    got = (c.samples, c.seed, c.use_adaptive_sampling, c.time_limit)
    if got != (spp, seed, False, 0.0):
        raise RuntimeError(f"sampling settings did not take: (samples, seed, adaptive, time_limit) "
                           f"= {got} after the render, expected ({spp}, {seed}, False, 0.0)")
    return dt


def save_render_result(scene, out_path: str) -> None:
    """Save the live Render Result to out_path with the scene's image settings. Must be called
    before the next render: the buffer is shared (A2 audit)."""
    bpy.data.images["Render Result"].save_render(out_path, scene=scene)


def fixation_meta(scene, eye, eye_note: str, cam, backend: str, n: int, s0: float, e2: float,
                  emax: float, spp: int, yaw: float, pitch: float, seconds: float,
                  profile: str | None, seconds_include_write: bool) -> dict:
    m = cam.matrix_world.to_3x3()
    return {
        "blend": bpy.data.filepath, "scene": scene.name,
        "fov_kind": scene.get("fov_kind", "unknown"), "blender": bpy.app.version_string,
        "eye": eye_record(eye), "eye_note": eye_note,
        "profile": profile,
        "gaze_yaw_deg": yaw, "gaze_pitch_deg": pitch,
        "camera_position_m": [round(x, 6) for x in cam.matrix_world.translation],
        "camera_forward": [round(x, 6) for x in (m @ Vector((0.0, 0.0, -1.0)))],
        "camera_right": [round(x, 6) for x in (m @ Vector((1.0, 0.0, 0.0)))],
        "camera_up": [round(x, 6) for x in (m @ Vector((0.0, 1.0, 0.0)))],
        "warp": {"form": "e(r) = E2 * ((1 + e_max/E2)^r - 1), r = 2*hypot(u-0.5, v-0.5)",
                 "E2_deg": e2, "e_max_deg": emax, "s0_deg": s0, "raster": n},
        "spp": spp, "seed": 0, "adaptive_sampling": False, "device": backend,
        "pixel_filter": "BOX", "filter_width": 1.0,
        "render_seconds": round(seconds, 4),
        "render_seconds_includes_write": seconds_include_write,
        "depth_pass": "ray distance from the camera centre; background = 1e10",
        "mask": "valid samples are r <= 1; alpha is NOT a valid mask",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", help="open this file first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--e2", type=float, default=2.0, help="eccentricity at which spacing doubles, deg")
    ap.add_argument("--emax", type=float, default=45.0, help="half-field of the warp, deg")
    ap.add_argument("--s0", type=float, default=0.05, help="foveal spacing, deg/sample")
    ap.add_argument("--n", type=int, default=None, help="raster size override; s0 is then derived")
    ap.add_argument("--yaw", type=float, default=0.0, help="gaze, deg right, about the eye centre")
    ap.add_argument("--pitch", type=float, default=0.0, help="gaze, deg up, about the eye centre")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--device", default="OPTIX", choices=["OPTIX", "CUDA", "CPU"])
    ap.add_argument("--shader", default=os.path.join(HERE, "foveated_camera.osl"))
    add_profile(ap, script_args(), s0="s0", spp="fix_spp")
    args = ap.parse_args(script_args())

    if args.blend:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    scene = bpy.context.scene
    ensure_cycles(scene)
    backend = setup_device(scene, args.device)
    bpy.context.view_layer.update()  # rotation_euler does not reach matrix_world until the depsgraph runs
    eye, eye_note = find_eye(scene)

    n = args.n if args.n else raster_size(args.s0, args.e2, args.emax)
    s0 = s0_of(n, args.e2, args.emax)
    print(f"[foveated] {eye_note}; device {backend}; raster {n}x{n}; "
          f"s0 {s0:.4f} deg; E2 {args.e2} deg; e_max {args.emax} deg")

    cam = setup_foveated_camera(scene, eye, args.shader, args.e2, args.emax, n, args.spp)
    set_gaze(cam, eye, args.yaw, args.pitch)
    seconds = render_fixation(scene, args.spp, os.path.join(out, "fix.exr"))
    meta = fixation_meta(scene, eye, eye_note, cam, backend, n, s0, args.e2, args.emax, args.spp,
                         args.yaw, args.pitch, seconds, args.profile, seconds_include_write=True)
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[foveated] wrote {out} ({seconds:.2f}s on {backend})")


def run():
    """`blender -b -P` exits 0 even when the script raised (CLAUDE.md), so a tool that writes an
    artifact must make its own failure loud: traceback, a FAILED line, exit 1."""
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print("[foveated] FAILED", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    run()
