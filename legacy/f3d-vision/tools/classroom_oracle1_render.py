"""Blender-side acquisition for Classroom-Oracle-1.

Modes
-----
seeds
    Use ray casting only to enumerate instances visible inside the frozen FSG6f
    controller domain and write exactly one seed direction for each.  The dense
    hit cloud is written below ``evaluation_only/`` and is never a controller
    input.
fixation
    Render one binocular padded tangent pair at a requested gaze.  Each EXR
    contains Combined, Position and Object Index passes.  The host-side oracle
    matcher consumes only this current pair.

Run from the repository root, normally through classroom_oracle1_run.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import classroom_oracle1_public as public
from fsg_geometry import CV_TO_BLENDER, make_calibration


def _script_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _json_write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _gaze_direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y = math.radians(float(yaw_deg)); p = math.radians(float(pitch_deg))
    return np.array([math.sin(y) * math.cos(p), math.sin(p), -math.cos(y) * math.cos(p)], dtype=np.float64)


def _head_pose():
    import bpy
    from bl_common import find_eye, rigid
    eye, note = find_eye(bpy.context.scene)
    m = rigid(eye.matrix_world)
    r = np.array([[m[i][j] for j in range(3)] for i in range(3)], dtype=np.float64)
    o = np.array([m.translation[i] for i in range(3)], dtype=np.float64)
    return eye, note, o, r


def _renderable_objects() -> list:
    import bpy
    # Every renderable geometric Blender object is an ordinary scene entity.
    kinds = {"MESH", "CURVE", "SURFACE", "FONT", "META"}
    return sorted(
        [o for o in bpy.context.scene.objects if o.type in kinds and not o.hide_render],
        key=lambda o: o.name_full,
    )


def _assign_instance_ids() -> tuple[dict[int, str], dict[str, int]]:
    objs = _renderable_objects()
    if len(objs) >= 32767:
        raise RuntimeError("too many renderable objects for Blender Object Index pass")
    by_id: dict[int, str] = {}
    by_name: dict[str, int] = {}
    for iid, obj in enumerate(objs, start=1):
        obj.pass_index = iid
        by_id[iid] = obj.name_full
        by_name[obj.name_full] = iid
    return by_id, by_name


def _angles(d: np.ndarray) -> tuple[float, float]:
    d = np.asarray(d, float)
    d = d / max(float(np.linalg.norm(d)), 1e-15)
    yaw = math.degrees(math.atan2(float(d[0]), float(-d[2])))
    pitch = math.degrees(math.atan2(float(d[1]), float(math.hypot(d[0], d[2]))))
    return yaw, pitch


def seed_scan(args) -> None:
    import bpy
    import fsg6f_public as frozen

    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"seed output must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    evdir = out / "evaluation_only"
    evdir.mkdir()

    _eye, eye_note, head_origin, head_r_wh = _head_pose()
    by_id, by_name = _assign_instance_ids()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cfg = frozen.SURFACE_FRONTIER
    step = float(args.seed_step)
    yaw_values = np.arange(float(cfg["yaw_min_deg"]), float(cfg["yaw_max_deg"]) + 0.5 * step, step)
    pitch_values = np.arange(float(cfg["pitch_min_deg"]), float(cfg["pitch_max_deg"]) + 0.5 * step, step)

    hit_ids: list[int] = []
    hit_dirs: list[np.ndarray] = []
    hit_points_h: list[np.ndarray] = []
    hit_angles: list[tuple[float, float]] = []
    t0 = time.perf_counter()
    for pitch in pitch_values:
        for yaw in yaw_values:
            d_h = _gaze_direction(float(yaw), float(pitch))
            d_w = d_h @ head_r_wh.T
            hit, loc, _normal, _face, obj, _matrix = bpy.context.scene.ray_cast(
                depsgraph, head_origin, d_w, distance=float(args.max_range)
            )
            if not hit or obj is None:
                continue
            name = getattr(getattr(obj, "original", None), "name_full", None) or obj.name_full
            iid = by_name.get(name)
            if iid is None:
                # Some evaluated/instanced objects preserve only the evaluated name.
                iid = int(getattr(obj, "pass_index", 0))
            if iid <= 0 or iid not in by_id:
                continue
            p_w = np.array(loc[:], dtype=np.float64)
            p_h = (p_w - head_origin) @ head_r_wh
            hit_ids.append(int(iid))
            hit_dirs.append(d_h.astype(np.float64))
            hit_points_h.append(p_h)
            hit_angles.append((float(yaw), float(pitch)))

    if not hit_ids:
        raise RuntimeError("seed scan found no geometric instance in the controller domain")

    ids = np.asarray(hit_ids, np.int32)
    dirs = np.asarray(hit_dirs, np.float64)
    points_h = np.asarray(hit_points_h, np.float64)
    yaw_pitch = np.asarray(hit_angles, np.float64)
    seeds = []
    for iid in sorted(np.unique(ids).tolist()):
        idx = np.flatnonzero(ids == iid)
        mean = dirs[idx].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm <= 1e-12:
            best = int(idx[0])
        else:
            mean /= norm
            best = int(idx[np.argmax(dirs[idx] @ mean)])
        yaw, pitch = map(float, yaw_pitch[best])
        seeds.append({
            "instance_id": int(iid),
            "object_name": by_id[int(iid)],
            "seed_gaze_deg": [yaw, pitch],
        })

    # Only enumeration and one direction per visible instance are controller-visible.
    _json_write(out / "seeds.json", {
        "schema": "ClassroomOracle1-seeds-v1",
        "spec_id": public.SPEC_ID,
        "profile": args.profile,
        "eye_note": eye_note,
        "head_origin_w_m": head_origin.tolist(),
        "head_R_wh": head_r_wh.tolist(),
        "controller_domain_deg": {
            "yaw": [cfg["yaw_min_deg"], cfg["yaw_max_deg"]],
            "pitch": [cfg["pitch_min_deg"], cfg["pitch_max_deg"]],
        },
        "seed_scan_step_deg": step,
        "instances": seeds,
    })
    _json_write(out / "instance_catalog.json", {
        "schema": "ClassroomOracle1-instance-catalog-v1",
        "instances": [{"instance_id": int(k), "object_name": v} for k, v in sorted(by_id.items())],
    })
    # This file is evaluation truth and must not be opened by the controller.
    np.savez_compressed(
        evdir / "reachable_samples.npz",
        instance_id=ids,
        direction_h=dirs.astype(np.float32),
        xyz_h=points_h.astype(np.float32),
        yaw_pitch_deg=yaw_pitch.astype(np.float32),
    )
    _json_write(evdir / "manifest.json", {
        "schema": "ClassroomOracle1-evaluation-truth-v1",
        "sample_count": int(len(ids)),
        "visible_instance_count": int(len(seeds)),
        "seed_scan_step_deg": step,
        "note": "dense cyclopean first-hit samples; evaluation only; not a control input",
    })
    print("[classroom-oracle1-render] SEEDS", json.dumps({
        "instances": len(seeds), "samples": len(ids), "seconds": time.perf_counter() - t0
    }, sort_keys=True), flush=True)


def _channel(ch: dict[str, np.ndarray], suffix: str) -> np.ndarray:
    keys = [k for k in ch if k.endswith(suffix)]
    if len(keys) != 1:
        raise RuntimeError(f"expected exactly one EXR channel ending {suffix!r}, got {keys}")
    return np.asarray(ch[keys[0]])


def _extract_exr(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from exr_lite import read_uncompressed_exr
    ch = read_uncompressed_exr(str(path))
    rgb = np.stack([_channel(ch, f"Combined.{c}") for c in "RGB"], axis=-1).astype(np.float32)
    pos = np.stack([_channel(ch, f"Position.{c}") for c in "XYZ"], axis=-1).astype(np.float32)
    # Blender writes the scalar Object Index pass as X in a multilayer EXR.
    idx = np.rint(_channel(ch, "Object Index.X")).astype(np.int32)
    idx[idx < 0] = 0
    return rgb, idx, pos


def _prepare_perspective_pair(c: dict, device: str, spp: int):
    import bpy
    from mathutils import Matrix
    from bl_common import configure_multilayer_exr, ensure_cycles, pin_seed, setup_device

    scene = bpy.context.scene
    ensure_cycles(scene)
    backend = setup_device(scene, device)
    if backend == "CPU" and device != "CPU":
        raise RuntimeError("requested GPU backend unavailable; pass --device CPU only deliberately")
    # The Classroom .blend keyframes cycles.seed at frame 1, so every render evaluates the
    # frame and overwrites a script-set seed.  Same inherited bl_common repair the FSG blend
    # bridge already uses; the seed affects render noise only, never geometry or selection.
    pin_seed(scene)

    w, h = c["image_size_wh"]
    scene.render.resolution_x = int(w)
    scene.render.resolution_y = int(h)
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.film_transparent = False
    scene.render.use_motion_blur = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.use_single_layer = True
    scene.render.use_border = False
    scene.render.use_persistent_data = True
    scene.cycles.samples = int(spp)
    scene.cycles.seed = 0
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.time_limit = 0.0
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    try:
        scene.cycles.use_denoising = False
    except (AttributeError, TypeError):
        pass

    vl = bpy.context.view_layer
    vl.use_pass_combined = True
    vl.use_pass_z = True
    vl.use_pass_normal = True
    vl.use_pass_position = True
    vl.use_pass_object_index = True
    configure_multilayer_exr(scene)
    scene.render.image_settings.exr_codec = "NONE"

    data = bpy.data.cameras.new("CLASSROOM_ORACLE1")
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = data.sensor_height = 36.0
    data.shift_x = data.shift_y = 0.0
    data.lens = float(c["eyes"][0]["K"][0][0]) * data.sensor_width / float(w)
    data.clip_start = 0.01
    data.clip_end = 1000.0
    data.dof.use_dof = False
    cam = bpy.data.objects.new("CLASSROOM_ORACLE1", data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    bpy.context.view_layer.update()
    return scene, cam, backend, Matrix


def _eye_matrix(c: dict, eye: dict, Matrix):
    r_wh = np.asarray(c["head_R_wh"], float)
    r_hc = np.asarray(eye["R_hc"], float)
    rb = r_wh @ r_hc @ CV_TO_BLENDER
    pos = np.asarray(c["head_origin_w_m"], float) + np.asarray(eye["centre_h_m"], float) @ r_wh.T
    m = np.eye(4)
    m[:3, :3] = rb
    m[:3, 3] = pos
    return Matrix(m.tolist())


def fixation(args) -> None:
    import bpy
    import fsg6f_public as frozen
    from render_foveated import render_fixation

    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"fixation output must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    _eye, eye_note, head_origin, head_r_wh = _head_pose()
    _assign_instance_ids()
    c = make_calibration(
        args.profile,
        float(args.yaw), float(args.pitch), float(frozen.VERGENCE_DISTANCE_M),
        head_r_wh=head_r_wh, head_origin_w=head_origin,
    )
    _json_write(out / "calibration.json", c)
    scene, cam, backend, Matrix = _prepare_perspective_pair(c, args.device, args.spp)

    obs = {}
    timings = {}
    seeds_lr = {}
    for eye_i, eye in enumerate(c["eyes"]):
        side = eye["name"]
        cam.matrix_world = _eye_matrix(c, eye, Matrix)
        bpy.context.view_layer.update()
        # Stable, signed-32-safe physical render seed.  The target id affects noise only,
        # never geometry or selection.
        seed = int((104729 * max(1, int(args.object_id)) + 1009 * int(args.step) + eye_i) % 2147483647)
        path = out / f"raw_{side}.exr"
        sec = render_fixation(scene, int(args.spp), str(path), seed=seed)
        rgb, ids, pos = _extract_exr(path)
        w, h = c["image_size_wh"]
        if rgb.shape != (h, w, 3) or ids.shape != (h, w) or pos.shape != (h, w, 3):
            raise RuntimeError(f"bad extracted {side} pass shapes: {rgb.shape}, {ids.shape}, {pos.shape}")
        obs[f"rgb_{side}"] = rgb
        obs[f"instance_{side}"] = ids
        obs[f"position_w_{side}"] = pos
        timings[side] = float(sec)
        seeds_lr[side] = seed

    np.savez_compressed(out / "oracle_observation.npz", **obs)
    _json_write(out / "acquisition.json", {
        "schema": "ClassroomOracle1-fixation-acquisition-v1",
        "spec_id": public.SPEC_ID,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "blend": bpy.data.filepath,
        "profile": args.profile,
        "spp": int(args.spp),
        "device": backend,
        "eye_note": eye_note,
        "object_id_for_noise_seed_only": int(args.object_id),
        "step": int(args.step),
        "gaze_yaw_pitch_deg": [float(args.yaw), float(args.pitch)],
        "render_seeds_lr": seeds_lr,
        "render_seconds_lr": timings,
        "primary_camera_samples": int(2 * c["image_size_wh"][0] * c["image_size_wh"][1] * args.spp),
        "truth_passes": ["Position", "IndexOB"],
        "truth_scope": "this current tangent pair only",
        "complete": True,
    })
    print("[classroom-oracle1-render] FIXATION", json.dumps({
        "object": int(args.object_id), "step": int(args.step),
        "gaze": [float(args.yaw), float(args.pitch)], "device": backend,
        "seconds": timings,
    }, sort_keys=True), flush=True)


def parse_args(argv: list[str]):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("seeds", "fixation"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", choices=("small", "full"), default=public.DEFAULT_PROFILE)
    ap.add_argument("--device", choices=("OPTIX", "CUDA", "CPU"), default=public.DEFAULT_DEVICE)
    ap.add_argument("--spp", type=int, default=None)
    ap.add_argument("--seed-step", type=float, default=public.SEED_SCAN_STEP_DEG)
    ap.add_argument("--max-range", type=float, default=1000.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--object-id", type=int, default=0)
    ap.add_argument("--step", type=int, default=0)
    args = ap.parse_args(argv)
    if args.seed_step <= 0 or args.max_range <= 0:
        ap.error("--seed-step and --max-range must be positive")
    if args.spp is None:
        import fsg6f_public as frozen
        args.spp = int(frozen.DEFAULT_SPP[args.profile])
    if args.spp <= 0:
        ap.error("--spp must be positive")
    return args


def main() -> None:
    args = parse_args(_script_args())
    if args.mode == "seeds":
        seed_scan(args)
    else:
        fixation(args)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise
        traceback.print_exc()
        print("[classroom-oracle1-render] FAILED", flush=True)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
