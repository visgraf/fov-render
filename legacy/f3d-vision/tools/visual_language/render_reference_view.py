"""Visual Language 1: the single static REFERENCE / EVALUATION view (Blender side).

Contract: docs/methodology/visual-language-1-contract.md, section 3.  Run once:

    blender -b scenes/classroom/classroom_eye.blend --python-exit-code 1 \
        -P tools/visual_language/render_reference_view.py -- --seeds SEEDS.json --out DIR

One wide perspective view from the fixed head origin and orientation in ``seeds.json``
(Combined, Position and Object Index passes; object indices assigned exactly as the accepted
Classroom-Oracle renderer assigns them).  It is not an observation, it is never fed to the
controller, and it is not part of any measurement.  The host-side ``reference`` command resamples
it into the head chart and checks its alignment against the existing evaluation-only samples.
Failures exit nonzero (Blender alone would exit 0).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]  # tools/: the accepted renderer's helpers (read-only use)
sys.path.insert(0, str(ENGINE))


def args_after_dashes():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def main() -> None:
    import bpy
    from mathutils import Matrix
    from bl_common import configure_multilayer_exr, ensure_cycles, pin_seed, setup_device
    import classroom_oracle1_render as R  # _assign_instance_ids, _extract_exr (unchanged)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--half-fov-x-deg", type=float, default=27.0)
    ap.add_argument("--spp", type=int, default=512)
    ap.add_argument("--device", default="OPTIX")
    a = ap.parse_args(args_after_dashes())
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"reference output must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    seeds = json.loads(Path(a.seeds).read_text())
    r_wh = np.asarray(seeds["head_R_wh"], float)
    o_w = np.asarray(seeds["head_origin_w_m"], float)

    scene = bpy.context.scene
    ensure_cycles(scene)
    backend = setup_device(scene, a.device)
    if backend == "CPU" and a.device != "CPU":
        raise RuntimeError("requested GPU backend unavailable")
    pin_seed(scene)
    by_id, _by_name = R._assign_instance_ids()
    scene.render.resolution_x, scene.render.resolution_y = a.width, a.height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.film_transparent = False
    scene.render.use_motion_blur = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.use_single_layer = True
    scene.render.use_border = False
    scene.cycles.samples = a.spp
    scene.cycles.seed = 0
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    try:
        scene.cycles.use_denoising = False
    except (AttributeError, TypeError):
        pass
    vl = bpy.context.view_layer
    vl.use_pass_combined = True
    vl.use_pass_position = True
    vl.use_pass_object_index = True
    configure_multilayer_exr(scene)
    scene.render.image_settings.exr_codec = "NONE"
    f_px = (a.width / 2.0) / math.tan(math.radians(a.half_fov_x_deg))
    data = bpy.data.cameras.new("VL1_REFERENCE")
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.lens = f_px * data.sensor_width / a.width
    data.clip_start, data.clip_end = 0.01, 1000.0
    data.dof.use_dof = False
    cam = bpy.data.objects.new("VL1_REFERENCE", data)
    scene.collection.objects.link(cam)
    m = np.eye(4)
    m[:3, :3] = r_wh  # Blender camera axes = head axes: +X right, +Y up, -Z forward
    m[:3, 3] = o_w
    cam.matrix_world = Matrix(m.tolist())
    scene.camera = cam
    bpy.context.view_layer.update()
    exr = out / "reference.exr"
    scene.render.filepath = str(exr)
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    secs = time.perf_counter() - t0
    rgb, ids, pos = R._extract_exr(exr)
    np.savez_compressed(out / "reference.npz", rgb=rgb, instance=ids, position_w=pos)
    os.remove(exr)
    (out / "reference.json").write_text(json.dumps({
        "schema": "VisualLanguage1-reference-view-v1",
        "truth": "REFERENCE / EVALUATION: not an observation, never a controller input",
        "blend": bpy.data.filepath, "device": backend, "spp": a.spp, "seconds": round(secs, 2),
        "image_size_wh": [a.width, a.height], "f_px": f_px, "c_px": [(a.width - 1) / 2.0, (a.height - 1) / 2.0],
        "head_R_wh": r_wh.tolist(), "head_origin_w_m": o_w.tolist(),
        "camera_axes": "Blender camera: -Z forward, +Y up (= head frame)",
        "instance_names": {str(k): v for k, v in sorted(by_id.items())},
    }, indent=1, sort_keys=True) + "\n")
    print(f"[vl1-reference-view] rendered {a.width}x{a.height} at {a.spp} spp on {backend} in {secs:.1f} s -> {out}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise
        traceback.print_exc()
        print("[vl1-reference-view] FAILED", flush=True)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
