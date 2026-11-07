"""Active Bootstrap-1a: the Blender side of the one binocular look (preflight, canonical acquisition, rehearsal).

Contract: docs/active-bootstrap/ab1a-first-natural-stereo-look-contract.md, sections 6 and 15.  Normally run through
``ab1a_run.py``:

    blender -b scenes/classroom/classroom_eye.blend --python-exit-code 1 \
        -P tools/active_bootstrap/ab1a_render.py -- --mode preflight --run RUN --head-pose SEEDS
    blender -b scenes/classroom/classroom_eye.blend --python-exit-code 1 \
        -P tools/active_bootstrap/ab1a_render.py -- --mode canonical --run RUN --head-pose SEEDS
    blender -b --factory-startup --python-exit-code 1 \
        -P tools/active_bootstrap/ab1a_render.py -- --mode rehearsal --run RUN --head-pose SEEDS

The accepted Classroom-Oracle-1 helpers are reused read-only (head pose, pass indices, camera / pass configuration,
eye matrices, EXR extraction) with ``render_foveated.render_fixation``.  ``preflight`` stops before any render.
``canonical`` renders exactly one L/R pair at frozen NB1c RGB gaze #1 into new or empty directories and splits the
result at once into the inference-visible RGB observation and the evaluation-only truth.  ``rehearsal`` builds a
synthetic factory-startup scene (never Classroom).  Every failure exits nonzero.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import ab1a_spec as SP  # noqa: E402
from fsg_geometry import make_calibration  # noqa: E402  (accepted, read-only)

PREFIX = "[ab1a-render]"


def args_after_dashes() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=True, allow_nan=False) + "\n")


def new_dir(path: Path) -> Path:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def accepted_head_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if sha256(path) != SP.HEAD_POSE_SOURCE[1]:
        raise RuntimeError(f"accepted head-pose record changed: {path}")
    d = json.loads(path.read_text())
    return np.asarray(d["head_R_wh"], np.float64), np.asarray(d["head_origin_w_m"], np.float64)


def eye_pose(head_pose: Path) -> dict:
    """The EYE pose in Blender, required equal to the accepted head-pose record."""
    import classroom_oracle1_render as CO
    _eye, note, o, r = CO._head_pose()
    r_acc, o_acc = accepted_head_pose(head_pose)
    dr, do = float(np.abs(r - r_acc).max()), float(np.abs(o - o_acc).max())
    if max(dr, do) > SP.EYE_POSE_TOL:
        raise RuntimeError(f"EYE pose differs from the accepted head pose: |dR| {dr}, |do| {do}")
    return {"eye_note": note, "head_R_wh": r.tolist(), "head_origin_w_m": o.tolist(),
            "max_abs_diff_from_accepted": {"R": dr, "origin": do}}


def calib_diff(a, b) -> float:
    """Max |a - b| over the numeric leaves of two calibrations; inf when their structure or text differs."""
    if isinstance(a, dict) and isinstance(b, dict):
        return max([calib_diff(a[k], b[k]) for k in a] or [0.0]) if set(a) == set(b) else math.inf
    if isinstance(a, list) and isinstance(b, list):
        return max([calib_diff(x, y) for x, y in zip(a, b)] or [0.0]) if len(a) == len(b) else math.inf
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b))
    return 0.0 if a == b else math.inf


def check_planned(run: Path, c: dict) -> float:
    planned = json.loads((run / "prelook/planned-calibration.json").read_text())
    diff = calib_diff(planned, json.loads(json.dumps(c)))
    if not diff <= SP.CALIBRATION_TOL:
        raise RuntimeError(f"the calibration built from the EYE differs from the planned calibration ({diff})")
    return diff


def calibration(yaw: float, pitch: float, pose: dict) -> dict:
    return make_calibration(SP.PROFILE, float(yaw), float(pitch), SP.VERGENCE_M, ipd=SP.IPD_M,
                            head_r_wh=np.asarray(pose["head_R_wh"]), head_origin_w=np.asarray(pose["head_origin_w_m"]),
                            tangent_frame=SP.TANGENT_FRAME)


def configure(c: dict, spp: int):
    """The accepted Classroom-Oracle-1 perspective-pair configuration, narrowed to the three declared passes."""
    import bpy
    import classroom_oracle1_render as CO
    scene, cam, backend, matrix = CO._prepare_perspective_pair(c, SP.DEVICE, spp)
    if backend != SP.DEVICE:
        raise RuntimeError(f"requested {SP.DEVICE}, got {backend}; no fallback")
    vl = bpy.context.view_layer
    vl.use_pass_z = False
    vl.use_pass_normal = False
    bpy.context.view_layer.update()
    return scene, cam, backend, matrix


def readback(scene, cam) -> dict:
    import bpy
    r, cy, d = scene.render, scene.cycles, cam.data
    vl = bpy.context.view_layer
    return {"engine": r.engine, "cycles_device": cy.device, "resolution_wh": [r.resolution_x, r.resolution_y],
            "resolution_percentage": r.resolution_percentage, "samples": cy.samples,
            "adaptive_sampling": bool(cy.use_adaptive_sampling), "denoising": bool(cy.use_denoising),
            "pixel_filter": cy.pixel_filter_type, "filter_width": cy.filter_width,
            "motion_blur": bool(r.use_motion_blur), "film_transparent": bool(r.film_transparent),
            "file_format": r.image_settings.file_format, "exr_codec": r.image_settings.exr_codec,
            "color_depth": r.image_settings.color_depth,
            "passes": {"combined": bool(vl.use_pass_combined), "position": bool(vl.use_pass_position),
                       "object_index": bool(vl.use_pass_object_index), "z": bool(vl.use_pass_z),
                       "normal": bool(vl.use_pass_normal)},
            "camera": {"type": d.type, "lens_mm": d.lens, "sensor_width_mm": d.sensor_width, "sensor_fit": d.sensor_fit,
                       "shift": [d.shift_x, d.shift_y], "clip_m": [d.clip_start, d.clip_end], "dof": bool(d.dof.use_dof)}}


def acquire_pair(c: dict, acq: Path, ev: Path, spp: int, extra: dict) -> dict:
    """Render the L/R pair and split it at once into the two truth domains."""
    import bpy
    import classroom_oracle1_render as CO
    from exr_lite import read_header
    from render_foveated import render_fixation
    write_json(acq / "calibration.json", c)
    scene, cam, backend, matrix = configure(c, spp)
    rgb, ref, seconds, channels, matrices = {}, {}, {}, {}, {}
    for eye in c["eyes"]:
        side = eye["name"]
        cam.matrix_world = CO._eye_matrix(c, eye, matrix)
        bpy.context.view_layer.update()
        matrices[side] = [[cam.matrix_world[i][j] for j in range(4)] for i in range(4)]
        path = ev / f"raw_{side}.exr"
        seconds[side] = float(render_fixation(scene, int(spp), str(path), seed=int(SP.SEEDS[side])))
        if scene.cycles.seed != SP.SEEDS[side] or scene.cycles.samples != spp:
            raise RuntimeError(f"render settings did not hold for {side}")
        channels[side] = [ch[0] for ch in read_header(path.read_bytes())[0]["channels"]]
        rgb_s, ids, pos = CO._extract_exr(path)
        w, h = c["image_size_wh"]
        if rgb_s.shape != (h, w, 3) or ids.shape != (h, w) or pos.shape != (h, w, 3):
            raise RuntimeError(f"bad extracted {side} shapes")
        rgb[f"rgb_{side}"] = rgb_s
        ref[f"instance_{side}"] = ids
        ref[f"position_w_{side}"] = pos
    np.savez_compressed(acq / "rgb-observation.npz", **rgb)
    np.savez_compressed(ev / "reference-observation.npz", **ref)
    record = {
        "schema": "AB1a-acquisition-v1", "truth": SP.TRUTH_ORACLE,
        "statement": "one binocular RGB acquisition; rgb-observation.npz is the only inference-visible scene data; "
                     "the truth passes live below evaluation_only/",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "blend": bpy.data.filepath, "blender": bpy.app.version_string,
        "gaze_yaw_pitch_deg": list(c["gaze_yaw_pitch_deg"]), "profile": c["profile"],
        "tangent_frame": c.get("tangent_frame"), "ipd_m": c["ipd_m"],
        "vergence_distance_m": c["prescribed_vergence_distance_m"], "spp": int(spp), "device": backend,
        "render_seeds_lr": dict(SP.SEEDS), "seed_rule": "declared constants; no object identity",
        "render_seconds_lr": seconds, "exr_channels_lr": channels, "camera_matrix_world_lr": matrices,
        "settings": readback(scene, cam), "calibration_sha256": sha256(acq / "calibration.json"),
        "rgb_observation_sha256": sha256(acq / "rgb-observation.npz"), "rgb_observation_arrays": sorted(rgb),
        "evaluation_only": ["raw_L.exr", "raw_R.exr", "reference-observation.npz", "instance-catalog.json"],
        "primary_camera_samples": int(2 * c["image_size_wh"][0] * c["image_size_wh"][1] * spp),
        "complete": True, **extra,
    }
    write_json(acq / "acquisition.json", record)
    return record


def preflight(a) -> None:
    import bpy
    run = Path(a.run).resolve()
    out = new_dir(run / "preflight")
    pose = eye_pose(Path(a.head_pose))
    c = calibration(SP.GAZE_YAW_DEG, SP.GAZE_PITCH_DEG, pose)
    diff = check_planned(run, c)
    import classroom_oracle1_render as CO
    scene, cam, backend, matrix = configure(c, SP.SPP)
    mats = {}
    for eye in c["eyes"]:
        cam.matrix_world = CO._eye_matrix(c, eye, matrix)
        bpy.context.view_layer.update()
        mats[eye["name"]] = [[cam.matrix_world[i][j] for j in range(4)] for i in range(4)]
    write_json(out / "preflight.json", {
        "schema": "AB1a-preflight-v1", "statement": "Classroom loaded and configured for gaze #1; NO render",
        "blend": bpy.data.filepath, "blender": bpy.app.version_string, "eye_pose": pose,
        "calibration_max_abs_diff_from_planned": diff, "device": backend, "settings": readback(scene, cam),
        "camera_matrix_world_lr": mats, "rendered": False})
    print(f"{PREFIX} COMPLETE preflight (no render)", flush=True)


def canonical(a) -> None:
    import classroom_oracle1_render as CO
    run = Path(a.run).resolve()
    acq, ev = run / "acquisition", run / "evaluation_only"
    for d in (acq, ev):
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(f"STOP {d} exists: the canonical gaze is rendered once")
    new_dir(acq)
    new_dir(ev)
    pose = eye_pose(Path(a.head_pose))
    c = calibration(SP.GAZE_YAW_DEG, SP.GAZE_PITCH_DEG, pose)
    diff = check_planned(run, c)
    by_id, _by_name = CO._assign_instance_ids()   # writes the evaluation-only Object Index pass only
    write_json(ev / "instance-catalog.json", {
        "schema": "AB1a-instance-catalog-v1", "truth": SP.TRUTH_REFERENCE,
        "instances": [{"instance_id": int(k), "object_name": v} for k, v in sorted(by_id.items())]})
    rec = acquire_pair(c, acq, ev, SP.SPP, {"action_source": SP.ACTION_SOURCE, "eye_pose": pose, "canonical": True,
                                            "calibration_max_abs_diff_from_planned": diff})
    print(f"{PREFIX} COMPLETE canonical " + json.dumps({"gaze": rec["gaze_yaw_pitch_deg"], "device": rec["device"],
                                                        "seconds": rec["render_seconds_lr"]}, sort_keys=True), flush=True)


# ------------------------------------------------------------------ rehearsal: a synthetic scene only
def _emission_noise(name: str, scale: float, seed_w: float):
    import bpy
    from bl_common import node_material
    mat = node_material(name)
    nt = mat.node_tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.noise_dimensions = "4D"
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = 10.0
    nz.inputs["Roughness"].default_value = 0.55
    nz.inputs["W"].default_value = seed_w
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    nt.links.new(nz.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def build_rehearsal_scene(head_pose: Path) -> None:
    import bpy
    from mathutils import Matrix
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.0, 0.0, 0.0)
    r_wh, o_w = accepted_head_pose(head_pose)
    eye_data = bpy.data.cameras.new("EYE")
    eye = bpy.data.objects.new("EYE", eye_data)
    scene.collection.objects.link(eye)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r_wh, o_w
    eye.matrix_world = Matrix(m.tolist())
    bpy.ops.mesh.primitive_cube_add(size=2 * SP.REHEARSAL_ROOM_HALF_M, location=tuple(o_w))
    room = bpy.context.active_object
    room.name = "rehearsal_room"
    room.data.materials.append(_emission_noise("room_noise", 20.0, 0.3))
    bpy.ops.mesh.primitive_plane_add(size=3.0)
    plane = bpy.context.active_object
    plane.name = "rehearsal_plane"
    pm = np.eye(4)
    pm[:3, :3] = r_wh                                    # plane +Z = head +Z: it faces the eye
    pm[:3, 3] = o_w + r_wh @ np.array([0.0, 0.0, -SP.REHEARSAL_PLANE_M])
    plane.matrix_world = Matrix(pm.tolist())
    plane.data.materials.append(_emission_noise("plane_noise", 40.0, 1.7))
    bpy.context.view_layer.update()


def rehearsal(a) -> None:
    import classroom_oracle1_render as CO
    run = Path(a.run).resolve()
    root = new_dir(run / "synthetic/blender-rehearsal")
    build_rehearsal_scene(Path(a.head_pose))
    pose = eye_pose(Path(a.head_pose))
    by_id, _ = CO._assign_instance_ids()
    done = {}
    for case, (yaw, pitch) in SP.REHEARSAL_GAZES.items():
        acq, ev = new_dir(root / case / "acquisition"), new_dir(root / case / "evaluation_only")
        write_json(ev / "instance-catalog.json", {"instances": [{"instance_id": int(k), "object_name": v}
                                                                 for k, v in sorted(by_id.items())]})
        c = calibration(yaw, pitch, pose)
        rec = acquire_pair(c, acq, ev, SP.REHEARSAL_SPP, {"rehearsal": True, "scene": "synthetic factory-startup room",
                                                          "eye_pose": pose})
        done[case] = rec["render_seconds_lr"]
    print(f"{PREFIX} COMPLETE rehearsal " + json.dumps(done, sort_keys=True), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("preflight", "canonical", "rehearsal"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--head-pose", required=True)
    a = ap.parse_args(args_after_dashes())
    {"preflight": preflight, "canonical": canonical, "rehearsal": rehearsal}[a.mode](a)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise
        traceback.print_exc()
        print(f"{PREFIX} FAILED", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
