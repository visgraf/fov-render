"""The four sensors. Each spec makes a numpy model for a profile (report sections 4-5, 8.2).

    eye = sensors.Foveated(E2_deg=2.0, e_max_deg=45.0)
    m = eye.model(profile="full")      # FoveatedModel, raster 253 x 253
    dirs, omega, valid = m.rays().direction, m.rays().footprint_sr, m.rays().valid

The Blender builders for these models are in fov_render.bl.cameras.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from .. import warp
from ..profiles import profile as _profile
from .base import Rays, SensorModel, pinhole_pixel_of, pinhole_rays, pinhole_solid_angle

OSL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foveated_camera.osl")
_SHADER_TO_LOCAL = np.array([1.0, 1.0, -1.0])     # OSL camera frame (+Z fwd) <-> Blender local (-Z fwd)


# ----------------------------------------------------------------------------- models

class FoveatedModel(SensorModel):
    name = "foveated"

    def __init__(self, n: int, e2_deg: float = 2.0, emax_deg: float = 45.0):
        if n < 8 or e2_deg <= 0 or emax_deg <= 0 or emax_deg >= 90:
            raise ValueError("invalid foveated raster")
        self.n, self.e2, self.emax = int(n), float(e2_deg), float(emax_deg)
        self.width = self.height = self.n

    @property
    def s0_deg(self) -> float:
        return warp.s0_of(self.n, self.e2, self.emax)

    def rays(self) -> Rays:
        px, py = warp.pixel_offsets(self.n)
        d = warp.warp_direction(px, py, self.e2, self.emax) * _SHADER_TO_LOCAL
        return Rays(d, warp.footprint_numeric(self.n, self.e2, self.emax), warp.inside_mask(self.n))

    def pixel_of(self, d_local):
        return warp.raster_of_direction(np.asarray(d_local) * _SHADER_TO_LOCAL, self.n, self.e2, self.emax)

    @property
    def rays_per_eye(self) -> int:
        return int(warp.inside_mask(self.n).sum())

    def params(self) -> dict:
        return {"sensor": self.name, "E2_deg": self.e2, "e_max_deg": self.emax, "raster": self.n,
                "s0_deg": self.s0_deg,
                "warp": "e(r) = E2 ((1 + e_max/E2)^r - 1), r = 2 hypot(u - 0.5, v - 0.5)",
                "mask": "valid samples are r <= 1; alpha is not a valid mask"}


class PinholeModel(SensorModel):
    """A square or rectangular pinhole with its principal point at the raster centre."""

    def __init__(self, width: int, height: int, f_px: float, name: str = "pinhole"):
        self.width, self.height, self.f = int(width), int(height), float(f_px)
        self.cx, self.cy = (self.width - 1) / 2.0, (self.height - 1) / 2.0
        self.name = name

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]])

    def rays(self) -> Rays:
        d, omega = pinhole_rays(self.width, self.height, self.f, self.cx, self.cy)
        return Rays(d, omega, np.ones(self.shape, dtype=bool))

    def pixel_of(self, d_local):
        return pinhole_pixel_of(d_local, self.f, self.cx, self.cy, self.width, self.height)

    @property
    def rays_per_eye(self) -> int:
        return self.width * self.height

    def solid_angle_sr(self) -> float:
        return pinhole_solid_angle(self.width / (2 * self.f), self.height / (2 * self.f))

    def lens_mm(self, sensor_width_mm: float = 36.0) -> float:
        return self.f * sensor_width_mm / self.width

    def params(self) -> dict:
        return {"sensor": self.name, "image_size_wh": [self.width, self.height], "f_px": self.f,
                "c_px": [self.cx, self.cy], "K": self.K.tolist(),
                "pixel_convention": "integer pixel centres; rows top-down"}


class PerspectiveModel(PinholeModel):
    """f3d-vision's tangent pair: a 12-degree core plus a 3/4-core margin on every side."""

    def __init__(self, core_px: int, core_fov_deg: float = 12.0):
        self.core, self.core_fov = int(core_px), float(core_fov_deg)
        margin = 3 * self.core // 4
        n = self.core + 2 * margin
        f = self.core / (2 * math.tan(math.radians(self.core_fov) / 2))
        super().__init__(n, n, f, name="perspective")

    def core_mask(self) -> np.ndarray:
        v, u = np.mgrid[:self.height, :self.width]
        h = self.core / 2.0
        return (np.abs(u - self.cx) < h) & (np.abs(v - self.cy) < h)

    def params(self) -> dict:
        p = super().params()
        p.update({"core_size": self.core, "nominal_core_fov_deg": self.core_fov,
                  "raw_fov_deg": math.degrees(2 * math.atan(self.width / (2 * self.f)))})
        return p


class PanoramaModel(SensorModel):
    """Equirectangular W x W/2; centre pixel = the camera's -Z (report eq. 14)."""
    name = "panorama"

    def __init__(self, width: int):
        if width < 8 or width % 2:
            raise ValueError("panorama width must be even")
        self.width, self.height = int(width), int(width) // 2

    def lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        v, u = np.mgrid[:self.height, :self.width].astype(np.float64)
        return ((u + 0.5) / self.width - 0.5) * 2 * math.pi, (0.5 - (v + 0.5) / self.height) * math.pi

    def rays(self) -> Rays:
        lon, lat = self.lonlat()
        d = np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], axis=-1)
        omega = (2 * math.pi / self.width) * (math.pi / self.height) * np.cos(lat)
        return Rays(d, omega, np.ones(self.shape, dtype=bool))

    def pixel_of(self, d_local):
        d = np.asarray(d_local, dtype=np.float64)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        lon = np.arctan2(d[..., 0], -d[..., 2])
        lat = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
        col = (lon / (2 * math.pi) + 0.5) * self.width - 0.5
        row = (0.5 - lat / math.pi) * self.height - 0.5
        return np.stack([row, col], axis=-1), np.ones(d.shape[:-1], dtype=bool)

    @property
    def rays_per_eye(self) -> int:
        return self.width * self.height

    def params(self) -> dict:
        return {"sensor": self.name, "image_size_wh": [self.width, self.height],
                "convention": "lon = (u - 0.5) 2 pi, lat = (0.5 - v) pi, v = 0 at the top row; "
                              "d = (sin lon cos lat, sin lat, -cos lon cos lat)"}


# ----------------------------------------------------------------------------- specs

@dataclass(frozen=True)
class Foveated:
    E2_deg: float = 2.0
    e_max_deg: float = 45.0
    s0_deg: float | None = None
    n: int | None = None
    name = "foveated"

    def model(self, profile: str = "small") -> FoveatedModel:
        if self.n is not None:
            return FoveatedModel(self.n, self.E2_deg, self.e_max_deg)
        s0 = self.s0_deg if self.s0_deg is not None else _profile(profile)["s0_deg"]
        return FoveatedModel(warp.raster_size(s0, self.E2_deg, self.e_max_deg), self.E2_deg, self.e_max_deg)


@dataclass(frozen=True)
class Perspective:
    core_fov_deg: float = 12.0
    core_px: int | None = None
    name = "perspective"

    def model(self, profile: str = "small") -> PerspectiveModel:
        return PerspectiveModel(self.core_px or _profile(profile)["core_px"], self.core_fov_deg)


@dataclass(frozen=True)
class Panorama:
    """preset "preview360": width from the profile's reference width; "breadth1": 720 x 360."""
    preset: str = "preview360"
    width: int | None = None
    yaw_offset_px: float = 0.0
    backface: bool = False
    name = "panorama"

    def model(self, profile: str = "small") -> PanoramaModel:
        if self.width is not None:
            return PanoramaModel(self.width)
        if self.preset == "breadth1":
            return PanoramaModel(720)
        if self.preset == "preview360":
            return PanoramaModel(_profile(profile)["ref_width"])
        raise ValueError(f"unknown panorama preset {self.preset!r}")

    def yaw_offset_deg(self, profile: str = "small") -> float:
        return self.yaw_offset_px * 360.0 / self.model(profile).width


@dataclass(frozen=True)
class ReferenceView:
    width: int = 2400
    height: int = 2048
    half_fov_x_deg: float = 27.0
    name = "reference_view"

    def model(self, profile: str = "small") -> PinholeModel:
        f = (self.width / 2.0) / math.tan(math.radians(self.half_fov_x_deg))
        return PinholeModel(self.width, self.height, f, name="reference_view")


__all__ = ["Foveated", "Perspective", "Panorama", "ReferenceView", "FoveatedModel", "PerspectiveModel",
           "PanoramaModel", "PinholeModel", "Rays", "SensorModel", "OSL_PATH"]
