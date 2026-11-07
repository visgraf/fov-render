"""From matched rays to points: epipolar coordinates and triangulation (report section 6). numpy only."""
from __future__ import annotations

import numpy as np


def epipolar(d_head, degrees: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(theta, phi) of head-frame directions: theta from +X (the baseline), phi = atan2(d_y, -d_z).

    The atan2 form for theta (fov-3d-vision rig.epipolar): arccos(d_x) loses precision near the
    baseline, where a float32 direction's x is quantised."""
    d = np.asarray(d_head, dtype=np.float64)
    theta = np.arctan2(np.hypot(d[..., 1], d[..., 2]), d[..., 0])
    phi = np.arctan2(d[..., 1], -d[..., 2])
    return (np.degrees(theta), np.degrees(phi)) if degrees else (theta, phi)


def triangulate_sine(theta_l, theta_r, baseline_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Ray distances from C_L and C_R by the law of sines (eq. 20); radians in, metres out.
    Parallel or diverging rays (theta_R <= theta_L) give inf."""
    tl, tr = np.asarray(theta_l, dtype=np.float64), np.asarray(theta_r, dtype=np.float64)
    gamma = tr - tl
    with np.errstate(divide="ignore", invalid="ignore"):
        rl = np.where(gamma > 0, baseline_m * np.sin(tr) / np.sin(gamma), np.inf)
        rr = np.where(gamma > 0, baseline_m * np.sin(tl) / np.sin(gamma), np.inf)
    return rl, rr


def closest_approach(c_l, d_l, c_r, d_r) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint of the closest approach of two rays (eq. 22) and the gap between them (m)."""
    c_l, c_r = np.asarray(c_l, dtype=np.float64), np.asarray(c_r, dtype=np.float64)
    d_l = np.asarray(d_l, dtype=np.float64) / np.linalg.norm(d_l, axis=-1, keepdims=True)
    d_r = np.asarray(d_r, dtype=np.float64) / np.linalg.norm(d_r, axis=-1, keepdims=True)
    w = c_l - c_r
    k = np.sum(d_l * d_r, axis=-1)
    dl_w, dr_w = np.sum(d_l * w, axis=-1), np.sum(d_r * w, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (k * dr_w - dl_w) / (1.0 - k * k)
        t = (dr_w - k * dl_w) / (1.0 - k * k)
    p_l, p_r = c_l + s[..., None] * d_l, c_r + t[..., None] * d_r
    return 0.5 * (p_l + p_r), np.linalg.norm(p_l - p_r, axis=-1)
