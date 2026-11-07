"""What every sensor model provides (report sections 4-5).

A model is pure numpy and answers the geometric questions about one raster:
    rays()        Blender-camera-local unit directions (H, W, 3), footprints in sr (H, W),
                  valid mask (H, W)
    pixel_of(d)   inverse map: camera-local directions -> continuous (row, col) and an inside mask
Rows are top-down; (row, col) of a pixel centre are integers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rays:
    direction: np.ndarray     # (H, W, 3), Blender camera local: +X right, +Y up, -Z forward
    footprint_sr: np.ndarray  # (H, W)
    valid: np.ndarray         # (H, W) bool

    @property
    def shape(self) -> tuple[int, int]:
        return self.valid.shape


class SensorModel:
    name: str = "sensor"
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def rays(self) -> Rays:  # pragma: no cover - abstract
        raise NotImplementedError

    def pixel_of(self, d_local) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def rays_per_eye(self) -> int:
        return int(self.rays().valid.sum())

    def params(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError


def pinhole_rays(width: int, height: int, f: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    """Directions (Blender local) and per-pixel solid angle of a pinhole (report eqs. 12-13)."""
    v, u = np.mgrid[:height, :width].astype(np.float64)
    x, y = u - cx, v - cy
    d_cv = np.stack([x / f, y / f, np.ones_like(x)], axis=-1)
    d_cv /= np.linalg.norm(d_cv, axis=-1, keepdims=True)
    omega = f / (f * f + x * x + y * y) ** 1.5
    return d_cv * np.array([1.0, -1.0, -1.0]), omega


def pinhole_pixel_of(d_local, f: float, cx: float, cy: float, width: int, height: int):
    d = np.asarray(d_local, dtype=np.float64) * np.array([1.0, -1.0, -1.0])   # -> OpenCV
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cx + f * d[..., 0] / d[..., 2]
        v = cy + f * d[..., 1] / d[..., 2]
    inside = (d[..., 2] > 0) & (u > -0.5) & (u < width - 0.5) & (v > -0.5) & (v < height - 0.5)
    return np.stack([v, u], axis=-1), inside


def pinhole_solid_angle(a: float, b: float) -> float:
    """Solid angle of the rectangle [-a, a] x [-b, b] on the plane at unit distance:
    4 atan(ab / sqrt(1 + a^2 + b^2))."""
    return 4.0 * float(np.arctan(a * b / np.sqrt(1.0 + a * a + b * b)))
