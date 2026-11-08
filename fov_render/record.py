"""SampleRecord: one eye's samples from one look, the same for every sensor (report section 8.3).

Observation fields (an observer may read them):
    origin (3,) eye centre, world m · direction (K, 3) world unit · value (K, 3) linear RGB ·
    footprint_sr (K,) · raster_index (K,) flat row-major index into raster_shape
Oracle fields (only with access="oracle"):
    position_w (K, 3) · distance_m (K,) inf on a miss · instance_id (K,) 0 on a miss ·
    normal_w (K, 3) when the Normal pass was rendered

Range comes from Position, not from Cycles' Depth pass (report section 6.1).
numpy only; both interpreters import it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .geometry import epipolar as _epipolar

OBSERVATION_FIELDS = ("origin", "direction", "value", "footprint_sr", "raster_index")
ORACLE_FIELDS = ("position_w", "distance_m", "instance_id", "normal_w")
ACCESS = ("observation", "oracle")
SCHEMA = "fov-render-sample-record-v1"


class AccessError(PermissionError):
    """An oracle field was read from a record opened for observation only."""


@dataclass
class SampleRecord:
    eye: str
    look_id: str
    sensor: str
    raster_shape: tuple
    origin: np.ndarray
    direction: np.ndarray
    value: np.ndarray
    footprint_sr: np.ndarray
    raster_index: np.ndarray
    head_R_wh: np.ndarray
    access: str = "observation"
    meta: dict = field(default_factory=dict)
    _oracle: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.access not in ACCESS:
            raise ValueError(f"access must be one of {ACCESS}")

    # -- oracle gate ---------------------------------------------------------------
    def _oracle_field(self, name: str):
        if self.access != "oracle":
            raise AccessError(f"{name} is oracle data; open the session with access='oracle' to read it")
        if name not in self._oracle:
            raise KeyError(f"{name} was not rendered for this look")
        return self._oracle[name]

    position_w = property(lambda self: self._oracle_field("position_w"))
    distance_m = property(lambda self: self._oracle_field("distance_m"))
    instance_id = property(lambda self: self._oracle_field("instance_id"))
    normal_w = property(lambda self: self._oracle_field("normal_w"))

    @property
    def has_oracle(self) -> tuple:
        return tuple(sorted(self._oracle))

    # -- views ---------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.direction.shape[0])

    @property
    def direction_head(self) -> np.ndarray:
        return self.direction @ np.asarray(self.head_R_wh)

    def epipolar(self, idx=None) -> tuple[np.ndarray, np.ndarray]:
        """(theta, phi) in radians of the head-frame directions, optionally at indices idx."""
        d = self.direction_head if idx is None else self.direction_head[idx]
        return _epipolar(d)

    def as_raster(self, name: str, fill=np.nan) -> np.ndarray:
        a = getattr(self, name)
        h, w = self.raster_shape
        out = np.full((h * w,) + a.shape[1:], fill, dtype=np.result_type(a.dtype, np.asarray(fill).dtype))
        out[self.raster_index] = a
        return out.reshape((h, w) + a.shape[1:])

    # -- files ---------------------------------------------------------------------
    def save(self, path: str) -> None:
        arrays = {k: getattr(self, k) for k in OBSERVATION_FIELDS}
        arrays.update({f"oracle__{k}": v for k, v in self._oracle.items()})
        arrays["head_R_wh"] = np.asarray(self.head_R_wh)
        header = {"schema": SCHEMA, "eye": self.eye, "look_id": self.look_id, "sensor": self.sensor,
                  "raster_shape": list(self.raster_shape), "meta": self.meta}
        np.savez_compressed(path, __header__=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8), **arrays)

    @classmethod
    def load(cls, path: str, access: str = "observation") -> "SampleRecord":
        z = np.load(path)
        header = json.loads(bytes(z["__header__"]).decode())
        if header.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} file: {path}")
        oracle = {k[len("oracle__"):]: z[k] for k in z.files if k.startswith("oracle__")} if access == "oracle" else {}
        return cls(header["eye"], header["look_id"], header["sensor"], tuple(header["raster_shape"]),
                   *(z[k] for k in OBSERVATION_FIELDS), z["head_R_wh"], access, header["meta"], oracle)


def build(model, passes: dict, origin_w, R_wb, head_R_wh, *, eye: str, look_id: str, access: str = "observation",
          meta: dict | None = None) -> SampleRecord:
    """Make a record from rendered passes (each (H, W, ...) arrays, rows top-down).

    passes keys: "combined" (H, W, 3+), and optionally "position" (H, W, 3), "object_index" (H, W),
    "normal" (H, W, 3). Samples are the model's valid pixels. A pixel is a hit when its Object
    Index is positive (every renderable object is numbered by the scene layer); without that
    pass, when its Position is not exactly zero."""
    rays = model.rays()
    valid = rays.valid.reshape(-1)
    idx = np.flatnonzero(valid)
    origin_w = np.asarray(origin_w, dtype=np.float64)
    d_w = rays.direction.reshape(-1, 3)[idx] @ np.asarray(R_wb, dtype=np.float64).T
    rgb = np.asarray(passes["combined"])[..., :3].reshape(-1, 3)[idx].astype(np.float32)
    oracle = {}
    if "position" in passes:
        pos = np.asarray(passes["position"], dtype=np.float64).reshape(-1, 3)[idx]
        if "object_index" in passes:
            ids = np.rint(np.asarray(passes["object_index"])).reshape(-1)[idx].astype(np.int32)
            ids[ids < 0] = 0
            hit = ids > 0
            oracle["instance_id"] = ids
        else:
            hit = np.any(pos != 0.0, axis=-1)
        dist = np.where(hit, np.linalg.norm(pos - origin_w, axis=-1), np.inf)
        oracle["position_w"] = pos
        oracle["distance_m"] = dist
    if "normal" in passes:
        oracle["normal_w"] = np.asarray(passes["normal"], dtype=np.float32).reshape(-1, 3)[idx]
    rec = SampleRecord(eye, look_id, model.name, model.shape, origin_w, d_w, rgb,
                       rays.footprint_sr.reshape(-1)[idx], idx.astype(np.int64),
                       np.asarray(head_R_wh, dtype=np.float64), "oracle", dict(meta or {}), oracle)
    if access != "oracle":
        rec.access = "observation"
    return rec


def sensor_consistency(rec: SampleRecord) -> dict:
    """Max angle (deg) between each hit's direction and (Position - origin): the sensor model
    agrees with the renderer when this is small (report eq. 17). Needs oracle access."""
    hit = np.isfinite(rec.distance_m)
    if not hit.any():
        return {"hits": 0, "max_angle_deg": None, "p999_angle_deg": None}
    v = np.asarray(rec.position_w[hit], dtype=np.float64) - rec.origin
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    d = np.asarray(rec.direction[hit], dtype=np.float64)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    # atan2(|v x d|, v . d): arccos of a dot product near 1 turns a float32 norm error of 1e-7 into
    # an angle of ~0.03 deg (issue #3)
    ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(v, d), axis=-1), np.sum(v * d, axis=-1)))
    return {"hits": int(hit.sum()), "max_angle_deg": float(ang.max()),
            "p999_angle_deg": float(np.quantile(ang, 0.999))}
