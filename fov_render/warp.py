"""The foveated log-polar warp in numpy: raster -> eccentricity -> direction -> footprint.

Same formulas as `foveated_camera.osl` and fov-3d-vision `tools/warp.py` (frozen as a known answer
in checks/known/, issue #16). Spacing s(e) = s0 (1 + e/E2) integrates to the raster radius w(e) = (E2/s0) ln(1 + e/E2);
normalised by the rim, e(r) = E2 ((1 + e_max/E2)^r - 1) for r in [0, 1].

Frames: the shader frame is +X right, +Y up, +Z forward. Raster rows are top-down, so row i has
py = 0.5 - (i + 0.5)/n. Angles are in degrees unless a name says otherwise.
"""
from __future__ import annotations

import math

import numpy as np


def raster_size(s0_deg: float, e2_deg: float, emax_deg: float) -> int:
    """Pixels across so that the spacing at the centre is s0 (eq. N in the report)."""
    return int(round(2.0 * (e2_deg / s0_deg) * math.log(1.0 + emax_deg / e2_deg)))


def s0_of(n: int, e2_deg: float, emax_deg: float) -> float:
    """Foveal spacing (deg) that an n x n raster actually has."""
    return 2.0 * e2_deg * math.log(1.0 + emax_deg / e2_deg) / n


def eccentricity_of_r(r, e2: float, emax: float):
    return e2 * ((1.0 + emax / e2) ** np.asarray(r, dtype=np.float64) - 1.0)


def r_of_eccentricity(e_deg, e2: float, emax: float):
    return np.log(1.0 + np.asarray(e_deg, dtype=np.float64) / e2) / math.log(1.0 + emax / e2)


def warp_direction(px, py, e2: float, emax: float) -> np.ndarray:
    """Shader-frame unit direction for normalised raster offsets (px, py) from the centre."""
    px, py = np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64)
    r = 2.0 * np.hypot(px, py)
    e = np.radians(eccentricity_of_r(r, e2, emax))
    phi = np.arctan2(py, px)
    return np.stack([np.sin(e) * np.cos(phi), np.sin(e) * np.sin(phi), np.cos(e)], axis=-1)


def pixel_offsets(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(px, py) of every pixel centre of an n x n raster, rows top-down."""
    j, i = np.meshgrid(np.arange(n), np.arange(n))
    return (j + 0.5) / n - 0.5, 0.5 - (i + 0.5) / n


def footprint_numeric(n: int, e2: float, emax: float) -> np.ndarray:
    """Solid angle (sr) of each pixel by central differences of the warp (fov-3d-vision's method)."""
    px, py = pixel_offsets(n)
    h = 1.0 / n
    dx = warp_direction(px + h / 2, py, e2, emax) - warp_direction(px - h / 2, py, e2, emax)
    dy = warp_direction(px, py + h / 2, e2, emax) - warp_direction(px, py - h / 2, e2, emax)
    return np.linalg.norm(np.cross(dx, dy), axis=-1)


def footprint_analytic(n: int, e2: float, emax: float) -> np.ndarray:
    """Solid angle (sr) of each pixel from the report's eq. (10):
    omega = (sin e / rho) * de/drho, de/drho = (2/N) ln(1 + e_max/E2) (e + E2), angles in radians."""
    px, py = pixel_offsets(n)
    rho = np.hypot(px, py) * n                    # radius in pixels
    r = 2.0 * rho / n
    e = np.radians(eccentricity_of_r(r, e2, emax))
    de_drho = (2.0 / n) * math.log(1.0 + emax / e2) * (e + math.radians(e2))
    with np.errstate(divide="ignore", invalid="ignore"):
        tangential = np.where(rho > 0, np.sin(e) / rho, de_drho)
    return tangential * de_drho


def raster_of_direction(d_shader, n: int, e2: float, emax: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse warp: shader-frame unit directions -> continuous (row, col), pixel centres at
    integers, and the inside mask r <= 1. Directions behind the camera or beyond e_max are outside."""
    d = np.asarray(d_shader, dtype=np.float64)
    e = np.degrees(np.arccos(np.clip(d[..., 2], -1.0, 1.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(np.maximum(1.0 + e / e2, 1e-300)) / math.log(1.0 + emax / e2)
    phi = np.arctan2(d[..., 1], d[..., 0])
    px, py = 0.5 * r * np.cos(phi), 0.5 * r * np.sin(phi)
    return np.stack([(0.5 - py) * n - 0.5, (px + 0.5) * n - 0.5], axis=-1), r <= 1.0


def inside_mask(n: int) -> np.ndarray:
    px, py = pixel_offsets(n)
    return 2.0 * np.hypot(px, py) <= 1.0


def cap_sr(emax_deg: float) -> float:
    """Solid angle of the cone of half-angle e_max."""
    return 2.0 * math.pi * (1.0 - math.cos(math.radians(emax_deg)))
