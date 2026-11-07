"""FSG1 camera geometry. NumPy only; usable by both Python interpreters.

H: fixed head, +X right, +Y up, -Z forward; metres.
C: OpenCV camera, +X right, +Y down, +Z forward.
Matrices R_hc map C vectors into H; arrays of row vectors multiply R_hc.T.
Pixel centres have integer coordinates; the centred principal point is (N-1)/2.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
SCHEMA = "FSG1-camera-v1"
CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0])
CORE_SIZE = {"small": 128, "full": 256}
DEFAULT_IPD = 0.063
CORE_FOV_DEG = 12.0
# Debug/test definitions; production spp comes from bl_common.PROFILES.
HEAD_R_WH = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
HEAD_ORIGIN_W = np.array([0.3, -0.2, 1.6])

def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("zero direction")
    return x / n


def gaze_direction(yaw: float, pitch: float) -> np.ndarray:
    y, p = np.radians([yaw, pitch])
    return np.array([np.sin(y)*np.cos(p), np.sin(p), -np.cos(y)*np.cos(p)])

def camera_rotation_h(direction: np.ndarray,
                      horizontal_hint: np.ndarray | None = None) -> np.ndarray:
    """OpenCV-to-head rotation.

    With no hint this preserves the legacy zero-torsion repository convention.
    With a hint, the hint is projected into the tangent plane and defines camera
    +X.  The latter is the canonical 360-degree stereo tangent convention: use
    the physical head +X eye baseline as the hint so rectification keeps a
    horizontal left-to-right baseline in every non-degenerate look direction.
    """
    z = unit(direction)
    if horizontal_hint is None:
        x = unit(np.cross(z, np.array([0., 1., 0.])))
    else:
        h = unit(horizontal_hint)
        projected = h - np.dot(h, z) * z
        if np.linalg.norm(projected) < 1e-8:
            raise ValueError("view direction is degenerate with stereo baseline")
        x = unit(projected)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))

def make_calibration(profile: str, yaw: float = 0., pitch: float = 0.,
                     vergence_distance: float = 2., ipd: float = DEFAULT_IPD,
                     head_r_wh: np.ndarray = HEAD_R_WH,
                     head_origin_w: np.ndarray = HEAD_ORIGIN_W,
                     tangent_frame: str = "legacy_upright") -> dict[str, Any]:
    """Acquisition prescription, not a measured/ground-truth target depth.
    The raw 30-degree-ish raster provides generous search/rectification margins.
    Only the central core of the RECTIFIED raster is accepted as a measurement.

    ``baseline_projected`` is the omnidirectional stereo mode.  It preserves the
    physical left-to-right baseline as local image +X after projection into each
    eye's tangent plane.  Looking exactly along the baseline is intentionally
    rejected because the tangent horizontal and binocular depth cue are singular.
    """
    if profile not in CORE_SIZE or ipd <= 0 or vergence_distance <= 0:
        raise ValueError("invalid profile, baseline, or initial vergence distance")
    if tangent_frame not in ("legacy_upright", "baseline_projected"):
        raise ValueError("unsupported tangent frame")
    core = CORE_SIZE[profile]
    margin = 3 * core // 4
    n = core + 2 * margin
    f = core / (2 * math.tan(math.radians(CORE_FOV_DEG) / 2))
    k = np.array([[f, 0., (n-1)/2], [0., f, (n-1)/2], [0., 0., 1.]])
    target = gaze_direction(yaw, pitch) * vergence_distance
    horizontal_hint = np.array([1., 0., 0.]) if tangent_frame == "baseline_projected" else None
    eyes = []
    for name, x in (("L", -ipd/2), ("R", ipd/2)):
        centre = np.array([x, 0., 0.])
        eyes.append({"name": name, "centre_h_m": centre.tolist(),
                     "R_hc": camera_rotation_h(target-centre, horizontal_hint).tolist(), "K": k.tolist()})
    out = {"schema": SCHEMA, "profile": profile, "image_size_wh": [n, n],
           "core_size": core, "nominal_core_fov_deg": CORE_FOV_DEG,
           "raw_fov_deg": math.degrees(2 * math.atan(n/(2*f))),
           "ipd_m": ipd, "head_R_wh": np.asarray(head_r_wh).tolist(),
           "head_origin_w_m": np.asarray(head_origin_w).tolist(),
           "gaze_yaw_pitch_deg": [yaw, pitch],
           "prescribed_vergence_distance_m": vergence_distance,
           "eyes": eyes, "depth_search_z_rect_m": [0.75, 4.5],
           "pixel_convention": "integer pixel centres; rows top-down",
           "map_frame": "H: fixed head; +X right, +Y up, -Z forward; metres"}
    if tangent_frame != "legacy_upright":
        out["tangent_frame"] = tangent_frame
    validate_calibration(out)
    return out

def validate_calibration(c: dict) -> None:
    if c.get("schema") != SCHEMA:
        raise ValueError("unsupported calibration schema")
    w, h = c["image_size_wh"]
    if w < 32 or h < 32 or not 0 < c["core_size"] < min(w, h):
        raise ValueError("invalid raster/core sizes")
    if len(c["eyes"]) != 2:
        raise ValueError("two eyes required")
    for r in [c["head_R_wh"]] + [e["R_hc"] for e in c["eyes"]]:
        r = np.asarray(r, float)
        if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T@r, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(r), 1., atol=1e-7):
            raise ValueError("rotation must be proper orthonormal")
    for e in c["eyes"]:
        k = np.asarray(e["K"], float)
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("invalid intrinsics")
        if np.asarray(e["centre_h_m"]).shape != (3,):
            raise ValueError("invalid eye centre")
    centres = np.array([e["centre_h_m"] for e in c["eyes"]])
    expected = np.array([[-c["ipd_m"]/2, 0, 0], [c["ipd_m"]/2, 0, 0]])
    if not np.isfinite(centres).all() or c["ipd_m"] <= 0 or not np.allclose(centres, expected, atol=1e-9):
        raise ValueError("eyes must be at +/-ipd/2 on fixed head X")
    lo, hi = c["depth_search_z_rect_m"]
    if not (0 < lo < hi):
        raise ValueError("invalid declared depth search")

def pixels(w: int, h: int) -> np.ndarray:
    v, u = np.mgrid[:h, :w]
    return np.stack((u, v), axis=-1).astype(float)


def rays_h(eye: dict, uv: np.ndarray) -> np.ndarray:
    k, r = np.asarray(eye["K"]), np.asarray(eye["R_hc"])
    a = np.concatenate((uv, np.ones((*uv.shape[:-1], 1))), axis=-1)
    return unit((a @ np.linalg.inv(k).T) @ r.T)

def project_h(eye: dict, xyz_h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, k = np.asarray(eye["R_hc"]), np.asarray(eye["K"])
    xyz_c = (xyz_h - np.asarray(eye["centre_h_m"])) @ r
    a = xyz_c @ k.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = a[..., :2] / a[..., 2:3]
    return uv, xyz_c[..., 2]


def head_to_world(c: dict, xyz: np.ndarray) -> np.ndarray:
    return xyz @ np.asarray(c["head_R_wh"]).T + np.asarray(c["head_origin_w_m"])

def world_to_head(c: dict, xyz: np.ndarray) -> np.ndarray:
    return (xyz - np.asarray(c["head_origin_w_m"])) @ np.asarray(c["head_R_wh"])


def relative_pose(c: dict) -> tuple[np.ndarray, np.ndarray]:
    """X_R = R @ X_L + T, the convention required by stereoRectify."""
    l, r = c["eyes"]
    rl, rr = np.asarray(l["R_hc"]), np.asarray(r["R_hc"])
    return rr.T @ rl, rr.T @ (np.asarray(l["centre_h_m"]) - np.asarray(r["centre_h_m"]))

def crop_q(q: np.ndarray, left_xy: tuple[int, int], right_xy: tuple[int, int]) -> np.ndarray:
    """Q for potentially different horizontal crops; disparity is uL_crop-uR_crop.
    Do not just subtract the left principal point. The disparity offset changes
    by left_x-right_x. Unequal vertical crops would break the row convention.
    """
    if left_xy[1] != right_xy[1]:
        raise ValueError("rectified crops must share the same vertical origin")
    a = np.eye(4)
    a[0, 3], a[1, 3] = left_xy
    a[2, 3] = left_xy[0] - right_xy[0]
    return np.asarray(q) @ a

def reproject_q(q: np.ndarray, uv: np.ndarray, disparity: np.ndarray) -> np.ndarray:
    a = np.concatenate((uv, disparity[..., None], np.ones((*disparity.shape, 1))), axis=-1)
    b = a @ np.asarray(q).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return b[..., :3] / b[..., 3:4]


def rect_to_head(c: dict, r1: np.ndarray, xyz_rect: np.ndarray) -> np.ndarray:
    l = c["eyes"][0]
    return (xyz_rect @ r1) @ np.asarray(l["R_hc"]).T + np.asarray(l["centre_h_m"])

def json_write(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

def self_test() -> list[str]:
    fails = []
    for yaw, pitch in ((0, 0), (14, -7), (-24, 11)):
        c = make_calibration("small", yaw, pitch)
        uv = pixels(11, 9) + [145.3, 157.2]
        for eye in c["eyes"]:
            d = rays_h(eye, uv)
            xyz = np.asarray(eye["centre_h_m"]) + d * 2.3
            uv2, _ = project_h(eye, xyz)
            if not np.allclose(uv, uv2, atol=1e-10):
                fails.append("projection/ray round-trip")
            if not np.allclose(world_to_head(c, head_to_world(c, xyz)), xyz, atol=1e-12):
                fails.append("world/head round-trip")
        r, t = relative_pose(c)
        xyz = np.array([[0., 0., -2.], [.2, -.1, -3.]])
        l, ri = c["eyes"]
        xl = (xyz-np.asarray(l["centre_h_m"])) @ np.asarray(l["R_hc"])
        xr = (xyz-np.asarray(ri["centre_h_m"])) @ np.asarray(ri["R_hc"])
        if not np.allclose(xl @ r.T + t, xr, atol=1e-12):
            fails.append("relative extrinsics")
    for yaw, pitch in ((179.9, -89.9), (-120., -45.), (120., 45.)):
        c = make_calibration("small", yaw, pitch, tangent_frame="baseline_projected")
        for eye in c["eyes"]:
            x = np.asarray(eye["R_hc"], float)[:, 0]
            if not np.dot(x, np.array([1., 0., 0.])) > 0:
                fails.append("baseline-projected tangent axis")
    for yaw in (-90., 90.):
        try:
            make_calibration("small", yaw, 0., tangent_frame="baseline_projected")
        except ValueError:
            pass
        else:
            fails.append("baseline-degenerate view must be rejected")
    return fails

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", required=True)
    ap.parse_args()
    f = self_test()
    for line in f:
        print("[fsg-geometry] FAIL", line)
    print(f"[fsg-geometry] self-test {'FAILED' if f else 'PASS'}")
    raise SystemExit(bool(f))
