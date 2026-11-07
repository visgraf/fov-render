"""f3d-vision's file formats, built from fov_render's own objects (issue #2).

The greedy explorer of f3d-vision (Playground-2) reads a calibration dict per fixation and records
its render server's settings; check_equivalence compares both with the frozen run byte for byte.
These functions write those formats from fov_render's Rig, sensor model and scene state, so the
explorer can run on a Session unchanged. Each one is checked against the legacy function it
replaces, bit for bit (fov_render.checks.step5).

numpy side:    f3d_calibration (fsg_geometry.make_calibration)
Blender side:  ab1a_readback (ab1a_render.readback), gfe_passes (greedy_foveal render_server.trim_passes)
"""
from __future__ import annotations

import json
import math

import numpy as np

from .rig import Gaze, HeadPose, Rig
from .sensors import PerspectiveModel

F3D_CALIBRATION_SCHEMA = "FSG1-camera-v1"
F3D_TANGENT_FRAME = {"upright": "legacy_upright", "baseline_projected": "baseline_projected"}
F3D_DEPTH_SEARCH_Z_RECT_M = [0.75, 4.5]


def f3d_calibration(profile: str, model: PerspectiveModel, rig: Rig, head: HeadPose, gaze: Gaze) -> dict:
    """make_calibration(profile, yaw, pitch, vergence, ipd, head_r_wh, head_origin_w, tangent_frame),
    built from fov_render's pose of each eye. Only a yaw-pitch gaze with a vergence distance has a
    make_calibration equivalent."""
    if gaze.mode != "yaw_pitch" or gaze.vergence_m is None:
        raise ValueError("an f3d-vision calibration needs Gaze.yaw_pitch(yaw, pitch, vergence_m)")
    if not isinstance(model, PerspectiveModel):
        raise TypeError("an f3d-vision calibration describes the perspective pair")
    if tuple(rig.baseline_h) != (1.0, 0.0, 0.0):
        raise ValueError("f3d-vision's baseline is head +X")
    k = model.K.tolist()
    eyes = []
    for name in ("L", "R"):
        p = rig.pose(head, gaze, name)
        eyes.append({"name": name, "centre_h_m": p.centre_h.tolist(), "R_hc": p.R_hc.tolist(), "K": k})
    out = {"schema": F3D_CALIBRATION_SCHEMA, "profile": profile, "image_size_wh": [model.width, model.height],
           "core_size": model.core, "nominal_core_fov_deg": model.core_fov,
           "raw_fov_deg": math.degrees(2 * math.atan(model.width / (2 * model.f))),
           "ipd_m": rig.ipd_m, "head_R_wh": np.asarray(head.R_wh).tolist(),
           "head_origin_w_m": np.asarray(head.origin_w).tolist(),
           "gaze_yaw_pitch_deg": [gaze.yaw_deg, gaze.pitch_deg],
           "prescribed_vergence_distance_m": gaze.vergence_m,
           "eyes": eyes, "depth_search_z_rect_m": list(F3D_DEPTH_SEARCH_Z_RECT_M),
           "pixel_convention": "integer pixel centres; rows top-down",
           "map_frame": "H: fixed head; +X right, +Y up, -Z forward; metres"}
    if rig.roll != "upright":
        out["tangent_frame"] = F3D_TANGENT_FRAME[rig.roll]
    return out


def f3d_json(data) -> str:
    """ab1a_render.write_json's text."""
    return json.dumps(data, indent=1, sort_keys=True, allow_nan=False) + "\n"


# ----------------------------------------------------------------------------- Blender side

def ab1a_readback(scene, cam, view_layer=None) -> dict:
    """ab1a_render.readback: the settings record of the AB1a pair, unrounded."""
    import bpy
    r, cy, d = scene.render, scene.cycles, cam.data
    vl = view_layer or bpy.context.view_layer
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


GFE_KEEP = ("use_pass_combined", "use_pass_position", "use_pass_object_index")
# AB1a's configure switches Depth and Normal off before the greedy server trims, so the server never
# lists them; fov_render's scene layer trims them itself and does list them.
GFE_NOT_LISTED = ("use_pass_z", "use_pass_normal")


def gfe_passes(sanitised: dict, view_layer) -> dict:
    """The greedy render server's trim_passes() report, from the pass report of fov_render's scene
    layer (scene.sanitise, which trims the same passes in the same dir() order)."""
    disabled = [x for x in sanitised["passes"]["disabled"] if x not in GFE_NOT_LISTED]
    return {"disabled": disabled, "kept": [n for n in GFE_KEEP if getattr(view_layer, n)]}


FORMATS = {"f3d_ab1a": "ab1a_readback and gfe_passes in the session's ready answer"}
