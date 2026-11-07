"""Render a 360-degree preview of a scene from its EYE vantage point.

    blender -b scene.blend -P tools/preview360.py -- --out previews/<scene_id> [--width 2048 --spp 64]

Writes to --out:
    pano.exr      Combined + Depth (ray distance) + Normal + Position, equirect, 32-bit, single part
    pano.png      display-transformed Combined, for looking at
    backface.exr  1 where the eye sees the back side of a surface (set dressing seen from behind)
    meta.json     eye pose, settings, timings, pixel->direction convention
Then run tools/inspect_preview.py on the folder.

The equirect centre is the EYE's primary direction. With a fixed head, this one image is
everything a rotating eye at this vantage point can ever see.
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
                       find_eye, node_material, pin_seed, rigid, script_args, setup_device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", help="open this file first (otherwise use the file Blender was started with)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=2048, help="height is width/2")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--device", default="OPTIX", choices=["OPTIX", "CUDA", "CPU"])
    ap.add_argument("--denoise", action="store_true", help="denoise the Combined pass (previews only)")
    ap.add_argument("--filter", default="BLACKMAN_HARRIS", choices=["BLACKMAN_HARRIS", "GAUSSIAN", "BOX"],
                    help="pixel reconstruction filter. BOX with --filter-width 1.0 makes a rendered "
                         "pixel the plain average over its own footprint, which is what the tier-1 "
                         "identity check and the sample footprint omega both assume.")
    ap.add_argument("--filter-width", type=float, default=None, help="default: 1.0 for BOX, else Cycles' own")
    ap.add_argument("--no-backface", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="Cycles seed; a second seed gives a noise measurement")
    ap.add_argument("--yaw-offset-px", type=float, default=0.0,
                    help="rotate the camera about the EYE's vertical by this fraction of a pixel of this "
                         "render (0.5 = half a pixel = s0/2 deg, +right); an off-grid reference for the "
                         "resampling floor")
    ap.add_argument("--eye-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                    help="render from this offset of the EYE centre, in the EYE (head) frame, metres: "
                         "(-0.0315 0 0) / (0.0315 0 0) are the L / R eyes of the 63 mm rig (D12). meta.json's "
                         "eye.position_m is then the offset centre, so check_sequence.py and integrate_sphere.py "
                         "read a per-eye sequence against it unchanged")
    ap.add_argument("--time-write", action="store_true",
                    help="after the render, re-save the multilayer EXR to a scratch path and time "
                         "it, so render_seconds (which includes the write) can be split; the "
                         "scratch file must match pano.exr in size and is then removed")
    add_profile(ap, script_args(), width="ref_width", spp="ref_spp", filter="filter")
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
    print(f"[preview360] eye: {eye_note}; device: {backend}")

    cam_data = bpy.data.cameras.new("PREVIEW360")
    cam_data.type, cam_data.panorama_type = "PANO", "EQUIRECTANGULAR"
    cam_data.latitude_min, cam_data.latitude_max = -math.pi / 2, math.pi / 2
    cam_data.longitude_min, cam_data.longitude_max = -math.pi, math.pi
    cam_data.clip_start, cam_data.clip_end = 0.001, 1.0e5
    cam_data.dof.use_dof = False
    cam = bpy.data.objects.new("PREVIEW360", cam_data)
    scene.collection.objects.link(cam)
    yaw_offset_deg = args.yaw_offset_px * 360.0 / args.width
    cam.matrix_world = (rigid(eye.matrix_world) @ Matrix.Translation(Vector(args.eye_offset))
                        @ Matrix.Rotation(math.radians(-yaw_offset_deg), 4, "Y"))  # +yaw right, offset then rotate, as gaze_matrix
    scene.camera = cam

    r = scene.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = args.width, args.width // 2, 100
    r.film_transparent, r.use_motion_blur, r.use_single_layer = False, False, True
    r.use_compositing, r.use_sequencer = False, False
    c = scene.cycles
    for what in pin_seed(scene):
        print(f"[preview360] removed {what}")
    c.samples, c.seed = args.spp, args.seed
    # Uniform sampling, so --spp is what every pixel gets. calib_room.blend ships with adaptive
    # sampling on (threshold 0.01): at 8192 spp the reference rendered in 135 s against 35 min
    # projected because most pixels stopped early (measured 2026-09-13). noise_floor.py and
    # render_foveated.py already force this off; a reference must match them.
    c.use_adaptive_sampling = False
    c.time_limit = 0.0
    c.pixel_filter_type = args.filter
    c.filter_width = args.filter_width if args.filter_width is not None else (1.0 if args.filter == "BOX" else c.filter_width)
    try:
        c.use_denoising = args.denoise
    except (AttributeError, TypeError):
        pass

    vl = bpy.context.view_layer
    vl.use_pass_combined = vl.use_pass_z = vl.use_pass_normal = vl.use_pass_position = True
    configure_multilayer_exr(scene)
    r.filepath = os.path.join(out, "pano.exr")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    t_render = time.time() - t0
    got = (c.samples, c.seed, c.use_adaptive_sampling, c.time_limit)
    if got != (args.spp, args.seed, False, 0.0):   # frame evaluation can overwrite any of these
        raise RuntimeError(f"sampling settings did not take: (samples, seed, adaptive, time_limit) "
                           f"= {got} after the render, expected ({args.spp}, {args.seed}, False, 0.0); is one animated?")

    t_write, write_note = None, None
    if args.time_write:
        scratch = os.path.join(out, "_write_timing.exr")
        t0 = time.time()
        bpy.data.images["Render Result"].save_render(scratch, scene=scene)
        t_write = time.time() - t0
        sz_ref, sz_scr = os.path.getsize(r.filepath), os.path.getsize(scratch)
        os.remove(scratch)
        write_note = f"re-saved Render Result as multilayer EXR: {sz_scr} bytes vs pano.exr {sz_ref} bytes"
        if sz_scr != sz_ref:
            raise RuntimeError(f"write timing invalid: {write_note}; the re-save is not the same file")
        print(f"[preview360] EXR write {t_write:.1f}s ({write_note})")

    im = r.image_settings
    im.media_type, im.file_format, im.color_mode, im.color_depth = "IMAGE", "PNG", "RGB", "8"
    bpy.data.images["Render Result"].save_render(os.path.join(out, "pano.png"), scene=scene)

    t_back = None
    if not args.no_backface:
        mat = node_material("BACKFACE_MASK")
        nt = mat.node_tree
        geo, em = nt.nodes.new("ShaderNodeNewGeometry"), nt.nodes.new("ShaderNodeEmission")
        outn = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(geo.outputs["Backfacing"], em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])
        vl.material_override = mat
        scene.world = None
        vl.use_pass_z = vl.use_pass_normal = vl.use_pass_position = False
        c.samples, c.use_adaptive_sampling = 4, False
        try:
            c.use_denoising = False
        except (AttributeError, TypeError):
            pass
        configure_multilayer_exr(scene)
        r.filepath = os.path.join(out, "backface.exr")
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        t_back = time.time() - t0

    meta = {
        "blend": bpy.data.filepath,
        "scene": scene.name,
        "fov_kind": scene.get("fov_kind", "unknown"),
        "blender": bpy.app.version_string,
        "eye": dict(eye_record(eye), position_m=[round(x, 6) for x in cam.matrix_world.translation]),
        "eye_offset_local_m": list(args.eye_offset),
        "head": eye_record(eye),
        "eye_note": eye_note,
        "unit_scale_length": scene.unit_settings.scale_length,
        "profile": args.profile,
        "width": args.width, "height": args.width // 2, "spp": args.spp,
        "denoise": args.denoise, "device": backend,
        "adaptive_sampling": False, "time_limit": 0.0, "seed": args.seed,
        "yaw_offset_px": args.yaw_offset_px, "yaw_offset_deg": yaw_offset_deg,
        "pixel_filter": args.filter, "filter_width": round(c.filter_width, 4),
        "render_seconds": round(t_render, 2),
        "render_seconds_includes": "the multilayer EXR write (write_still=True is inside the timer)",
        "exr_write_seconds": None if t_write is None else round(t_write, 2),
        "exr_write_note": write_note,
        "backface_seconds": None if t_back is None else round(t_back, 2),
        "equirect_convention": "lon=(u-0.5)*2pi, lat=(0.5-v)*pi, v=0 top; "
                               "d_eye=(sin lon cos lat, sin lat, -cos lon cos lat) in EYE frame (x right, y up, -z forward)",
        "depth_pass": "ray distance from the eye centre; background = 1e10",
    }
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[preview360] wrote {out} (render {t_render:.1f}s)")


def run():
    """`blender -b -P` exits 0 even when the script raised (CLAUDE.md), so a tool that writes an
    artifact must make its own failure loud: traceback, a FAILED line, exit 1."""
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print("[preview360] FAILED", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    run()
