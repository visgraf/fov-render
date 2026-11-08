"""Step 2 gates: the numpy models (rig, sensors, record, geometry) agree with the legacy code, whose
answers are frozen in known/step2.npz (tools/freeze_known_answers.py, issue #16). numpy only."""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np

from .. import geometry, record, warp
from ..rig import HeadPose, Rig, Gaze, camera_rotation, gaze_angles, gaze_direction, gaze_rotation, CV_TO_BLENDER
from ..sensors import Foveated, Panorama, Perspective, ReferenceView, PanoramaModel
from .common import KNOWN, Result

STEP = "2"
GAZES = [(y, p) for y in range(-170, 171, 17) for p in range(-80, 81, 16)]


CALIB_CASES = [(roll, frame, y, p) for roll, frame in (("upright", "legacy_upright"), ("baseline_projected", "baseline_projected"))
               for y, p in [(0, 0), (14, -7), (-24, 11), (120, 45), (-120, -45), (179.9, -89.9)]
               if not (roll == "upright" and abs(p) > 89)]
TRIANGULATION_P = (0.3, 0.2, -2.7)


def _known():
    return np.load(KNOWN / "step2.npz")


def warp_matches_legacy() -> Result:
    """Directions, footprints, mask and inverse against fov-3d-vision's warp, frozen on 4000 pixels per
    profile plus the whole mask (known/step2.npz, issue #16)."""
    k = _known()
    worst, sizes = 0.0, {}
    for prof, s0 in (("small", 0.1), ("full", 0.05)):
        n = warp.raster_size(s0, 2.0, 45.0)
        sizes[prof] = [n, int(k[f"{prof}_n"])]
        m = Foveated().model(prof)
        r = m.rays()
        idx = k[f"{prof}_idx"]
        d = r.direction.reshape(-1, 3)[idx]
        worst = max(worst, float(np.abs(d * np.array([1, 1, -1]) - k[f"{prof}_direction_cam"]).max()),
                    float(np.abs(r.footprint_sr.reshape(-1)[idx] - k[f"{prof}_footprint"]).max()))
        mask = np.unpackbits(k[f"{prof}_mask"])[: int(k[f"{prof}_mask_size"])].astype(bool)
        if not np.array_equal(r.valid.reshape(-1), mask):
            worst = math.inf
        rc, ins = m.pixel_of(d)
        worst = max(worst, float(np.abs(rc - k[f"{prof}_rc"]).max()))
        if not np.array_equal(ins, k[f"{prof}_inside_rc"]):
            worst = math.inf
    ok = worst <= 1e-12 and all(a == b for a, b in sizes.values())
    ctrl = Foveated(E2_deg=2.02, n=126).model().rays().direction
    base = Foveated(n=126).model().rays().direction
    return Result(STEP, "new warp equals legacy warp (directions, footprints, mask, inverse; frozen)", ok,
                  float(np.abs(ctrl - base).max()) > 1e-6, {"max_abs_diff": worst, "raster_sizes": sizes})


def pose_agreement() -> Result:
    """The three historical pose formulas (frozen over the gaze grid) and the new rig agree, upright roll."""
    k = _known()
    if [tuple(x) for x in k["gazes"].tolist()] != [tuple(map(float, g)) for g in GAZES]:
        return Result(STEP, "pose agreement, upright roll, 3 legacy formulas", False, None,
                      {"error": "known/step2.npz was frozen on another gaze grid"})
    worst = {"rig.gaze_rotation": 0.0, "fsg_geometry.camera_rotation_h": 0.0, "render_foveated.gaze_matrix": 0.0}
    for i, (yaw, pitch) in enumerate(GAZES):
        new = camera_rotation(gaze_direction(yaw, pitch), "upright")
        worst["rig.gaze_rotation"] = max(worst["rig.gaze_rotation"], float(np.abs(new - k["gaze_rotation"][i]).max()))
        worst["fsg_geometry.camera_rotation_h"] = max(worst["fsg_geometry.camera_rotation_h"],
                                                      float(np.abs(new - k["camera_rotation_h"][i] @ CV_TO_BLENDER).max()))
        worst["render_foveated.gaze_matrix"] = max(worst["render_foveated.gaze_matrix"],
                                                   float(np.abs(new - k["gaze_matrix"][i]).max()))
    # numpy formulas agree to round-off; mathutils matrices are float32 (eps 6e-8), hence 1e-6 there
    tol = {"rig.gaze_rotation": 1e-12, "fsg_geometry.camera_rotation_h": 1e-12, "render_foveated.gaze_matrix": 1e-6}
    ok = all(v <= tol[key] for key, v in worst.items())
    # control: a yaw sign flip must be caught
    flipped = max(float(np.abs(camera_rotation(gaze_direction(-y, p), "upright") - k["gaze_rotation"][i]).max())
                  for i, (y, p) in enumerate(GAZES) if abs(y) > 1)
    return Result(STEP, "pose agreement, upright roll, 3 legacy formulas (frozen)", ok, flipped > 1e-3,
                  {"max_abs_diff": worst, "tolerance": tol, "gazes": len(GAZES)})


def calibration_agreement() -> Result:
    """Rig.pose reproduces f3d-vision make_calibration (R_hc and eye centres, frozen), both rolls."""
    k = _known()
    head = HeadPose(k["head_R_wh"], k["head_origin_w"])
    worst, n = 0.0, 0
    for c, (roll, _frame, yaw, pitch) in enumerate(CALIB_CASES):
        rig = Rig(roll=roll)
        g = Gaze.yaw_pitch(yaw, pitch, vergence_m=2.10)
        for j, eye in enumerate(("L", "R")):
            p = rig.pose(head, g, eye)
            worst = max(worst, float(np.abs(p.R_hc - k["calib_R_hc"][c, j]).max()),
                        float(np.abs(p.centre_h - k["calib_centre_h"][c, j]).max()))
            n += 1
    # control: upright roll at an extreme gaze breaks the baseline-horizontal property
    up = camera_rotation(gaze_direction(179.9, -89.9), "upright")
    bp = camera_rotation(gaze_direction(179.9, -89.9), "baseline_projected")
    ctrl = float(up[:, 0] @ np.array([1.0, 0, 0])) <= 0 < float(bp[:, 0] @ np.array([1.0, 0, 0]))
    return Result(STEP, "rig reproduces f3d-vision calibrations (R_hc, centres; frozen), both rolls", worst <= 1e-12, ctrl,
                  {"max_abs_diff": worst, "eyes_compared": n,
                   "control": "upright roll at (179.9, -89.9) points image +X against the baseline"})


def footprint_sums() -> Result:
    got = {}
    for prof in ("small", "full"):
        r = Foveated().model(prof).rays()
        got[f"foveated_{prof}_over_cap"] = float(r.footprint_sr[r.valid].sum() / warp.cap_sr(45.0))
        an = warp.footprint_analytic(Foveated().model(prof).n, 2.0, 45.0)
        rel = np.abs(an / r.footprint_sr - 1)[r.valid]
        got[f"foveated_{prof}_analytic_vs_numeric_max_rel"] = float(rel.max())     # the centre pixel of odd n
        got[f"foveated_{prof}_analytic_vs_numeric_p99_rel"] = float(np.quantile(rel, 0.99))
        got[f"foveated_{prof}_analytic_over_numeric_sum"] = float(an[r.valid].sum() / r.footprint_sr[r.valid].sum())
    pm = Perspective().model("full")
    got["perspective_full_over_closed_form"] = float(pm.rays().footprint_sr.sum() / pm.solid_angle_sr())
    rv = ReferenceView(240, 204).model()
    got["reference_view_over_closed_form"] = float(rv.rays().footprint_sr.sum() / rv.solid_angle_sr())
    for w in (720, 3600):
        got[f"panorama_{w}_over_4pi"] = float(PanoramaModel(w).rays().footprint_sr.sum() / (4 * math.pi))
    ok = (all(abs(got[f"foveated_{p}_over_cap"] - 1) <= 0.015 for p in ("small", "full"))   # rim ring, A4
          and all(got[f"foveated_{p}_analytic_vs_numeric_p99_rel"] <= 1e-3 for p in ("small", "full"))
          and all(abs(got[f"foveated_{p}_analytic_over_numeric_sum"] - 1) <= 1e-3 for p in ("small", "full"))
          and abs(got["perspective_full_over_closed_form"] - 1) <= 1e-5
          and abs(got["reference_view_over_closed_form"] - 1) <= 1e-4
          and all(abs(got[f"panorama_{w}_over_4pi"] - 1) <= 1e-4 for w in (720, 3600)))
    pr = PanoramaModel(720).rays()
    lat = PanoramaModel(720).lonlat()[1]
    no_cos = float((pr.footprint_sr / np.cos(lat)).sum() / (4 * math.pi))
    return Result(STEP, "footprints sum to the field's solid angle", ok, abs(no_cos - 1) > 0.1,
                  {**got, "control_panorama_without_cos_over_4pi": no_cos})


def inverse_maps() -> Result:
    worst = {}
    for spec, prof in ((Foveated(), "small"), (Perspective(), "small"), (Panorama(preset="breadth1"), "small"),
                       (ReferenceView(240, 204), "small")):
        m = spec.model(prof)
        r = m.rays()
        rc, ins = m.pixel_of(r.direction)
        h, w = m.shape
        ii, jj = np.mgrid[:h, :w]
        err = np.abs(rc - np.stack([ii, jj], -1))[r.valid].max()
        worst[m.name] = float(err) if ins[r.valid].all() else math.inf
    bad = Perspective().model("small")
    rc, _ = bad.pixel_of(bad.rays().direction * np.array([1, -1, 1]))
    return Result(STEP, "pixel_of inverts rays for every sensor", all(v <= 1e-6 for v in worst.values()),
                  float(np.abs(rc[..., 0] - np.mgrid[:bad.height, :bad.width][0]).max()) > 1, {"max_px_err": worst})


def record_and_geometry() -> Result:
    detail = {}
    m = Foveated(n=64).model()
    h, w = m.shape
    head = HeadPose(np.eye(3), np.zeros(3))
    pose = Rig().pose(head, Gaze.yaw_pitch(10, 5), "C")
    dirs_w = m.rays().direction @ pose.R_wb.T
    pos = pose.centre_w + 2.5 * dirs_w
    passes = {"combined": np.random.default_rng(0).random((h, w, 4)).astype(np.float32), "position": pos,
              "object_index": np.where(m.rays().valid, 7.0, 0.0)}
    rec = record.build(m, passes, pose.centre_w, pose.R_wb, head.R_wh, eye="C", look_id="t", access="oracle")
    detail["n_samples"] = rec.n_samples
    detail["distance_err"] = float(np.abs(rec.distance_m - 2.5).max())
    detail["consistency_deg"] = record.sensor_consistency(rec)["max_angle_deg"]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.npz")
        rec.save(p)
        back = record.SampleRecord.load(p, access="oracle")
        obs = record.SampleRecord.load(p)
        detail["round_trip"] = bool(np.array_equal(back.value, rec.value) and np.array_equal(back.distance_m, rec.distance_m))
        detail["as_raster"] = bool(np.array_equal(back.as_raster("value", 0)[m.rays().valid], rec.value))
        try:
            obs.position_w
            gate = False
        except record.AccessError:
            gate = True
    # triangulation: a known point, both formulas, and the legacy sine rule (frozen)
    rig = Rig()
    P = np.array(TRIANGULATION_P)
    cl, cr = rig.centre_h("L"), rig.centre_h("R")
    tl, _ = geometry.epipolar((P - cl) / np.linalg.norm(P - cl))
    tr, _ = geometry.epipolar((P - cr) / np.linalg.norm(P - cr))
    rl, _ = geometry.triangulate_sine(tl, tr, rig.baseline_m)
    X, gap = geometry.closest_approach(cl, P - cl, cr, P - cr)
    lrl = float(_known()["legacy_triangulate_m"])
    detail.update({"sine_rule_err_m": float(abs(rl - np.linalg.norm(P - cl))),
                   "midpoint_err_m": float(np.linalg.norm(X - P)), "gap_m": float(gap),
                   "vs_legacy_triangulate_m": float(abs(rl - lrl))})
    ok = (detail["distance_err"] < 1e-9 and detail["consistency_deg"] < 1e-5 and detail["round_trip"]
          and detail["as_raster"] and detail["sine_rule_err_m"] < 1e-9 and detail["midpoint_err_m"] < 1e-9
          and detail["vs_legacy_triangulate_m"] < 1e-12)
    # control: swapping the eyes makes the sine rule return no finite distance
    swapped, _ = geometry.triangulate_sine(tr, tl, rig.baseline_m)
    return Result(STEP, "record round trip, access gate, triangulation", ok and gate,
                  gate and not np.isfinite(swapped), {**detail, "oracle_gate_raises": gate})


CHECKS = [warp_matches_legacy, pose_agreement, calibration_agreement, footprint_sums, inverse_maps, record_and_geometry]
