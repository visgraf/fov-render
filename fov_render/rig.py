"""The binocular rig: a fixed head, two eyes on its X axis, a gaze per look, a roll convention.

numpy only; both interpreters import it. Report section 3.

Frames
    W  Blender world, metres, +Z up.
    H  head = the scene's EYE object: +X right (the baseline), +Y up, -Z forward.
    B  Blender camera local: +X right, +Y up, -Z forward (what `matrix_world` rotates).
    C  OpenCV camera: +X right, +Y down, +Z forward.  B = C @ diag(1, -1, -1).

One composition serves every sensor (report eq. 3):
    M_eye = T(o) R_WH T(+-b/2 x) R_HB,   R_HB = camera_rotation(gaze direction, roll).
With roll="upright" R_HB equals Ry(-yaw) Rx(pitch), fov-3d-vision's gaze_matrix; with
roll="baseline_projected" it is f3d-vision's 360-degree stereo tangent frame. Both are checked
against the legacy code (tools/run_checks.py, step 2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .profiles import DEFAULT_IPD_M

CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0])
_CV_SIGNS = np.array([1.0, -1.0, -1.0])     # R @ CV_TO_BLENDER as an exact column sign flip: a matrix
                                            # product would turn -0.0 into +0.0 (issue #2)


def cv_to_blender(r) -> np.ndarray:
    """R @ diag(1, -1, -1), exactly (its own inverse; keeps signed zeros)."""
    return np.asarray(r, dtype=np.float64) * _CV_SIGNS
ROLLS = ("upright", "baseline_projected")


def unit(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("zero-length direction")
    return x / n


def gaze_direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Head-frame unit direction of a gaze (report eq. 2). numpy's radians, sin and cos, in
    f3d-vision's order, so poses equal make_calibration's bit for bit (issue #2)."""
    y, p = np.radians([yaw_deg, pitch_deg])
    return np.array([np.sin(y) * np.cos(p), np.sin(p), -np.cos(y) * np.cos(p)])


def gaze_angles(d) -> tuple[float, float]:
    """Inverse of gaze_direction: (yaw, pitch) in degrees of a head-frame direction."""
    d = unit(d)
    return (math.degrees(math.atan2(d[0], -d[2])),
            math.degrees(math.atan2(d[1], math.hypot(d[0], d[2]))))


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def gaze_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """R_HB for the upright convention, composed as Ry(-yaw) @ Rx(pitch)."""
    return rot_y(-yaw_deg) @ rot_x(pitch_deg)


def camera_rotation(direction_h, roll: str = "upright", baseline_h=(1.0, 0.0, 0.0)) -> np.ndarray:
    """R_HB: Blender-camera-local -> head, for a camera looking along direction_h.

    upright             image x stays in the head's horizontal plane: x = unit(z x y_H).
                        Undefined straight up or down.
    baseline_projected  image x is the baseline projected into the image plane:
                        x = unit(h - (h.z) z). Undefined only along the baseline.
    """
    z = unit(direction_h)
    if roll == "upright":
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(x) < 1e-8:
            raise ValueError("upright roll is undefined looking straight up or down")
    elif roll == "baseline_projected":
        h = unit(baseline_h)
        x = h - float(h @ z) * z
        if np.linalg.norm(x) < 1e-8:
            raise ValueError("baseline-projected roll is undefined looking along the baseline")
    else:
        raise ValueError(f"unknown roll {roll!r}; expected one of {ROLLS}")
    x = unit(x)
    y = np.cross(z, x)
    r_hc = np.column_stack((x, y, z))
    return cv_to_blender(r_hc)


@dataclass(frozen=True)
class HeadPose:
    """The fixed head: R_wh maps head vectors to world, origin_w is the cyclopean point (metres)."""
    R_wh: np.ndarray
    origin_w: np.ndarray

    def __post_init__(self):
        r = np.asarray(self.R_wh, dtype=np.float64)
        if r.shape != (3, 3) or not np.allclose(r.T @ r, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(r), 1.0, atol=1e-7):
            raise ValueError("head rotation must be proper orthonormal")
        object.__setattr__(self, "R_wh", r)
        object.__setattr__(self, "origin_w", np.asarray(self.origin_w, dtype=np.float64).reshape(3))

    def to_world(self, p_h) -> np.ndarray:
        return np.asarray(p_h, dtype=np.float64) @ self.R_wh.T + self.origin_w

    def to_head(self, p_w) -> np.ndarray:
        return (np.asarray(p_w, dtype=np.float64) - self.origin_w) @ self.R_wh

    def dir_to_world(self, d_h) -> np.ndarray:
        return np.asarray(d_h, dtype=np.float64) @ self.R_wh.T

    def dir_to_head(self, d_w) -> np.ndarray:
        return np.asarray(d_w, dtype=np.float64) @ self.R_wh

    @classmethod
    def from_matrix(cls, m4) -> "HeadPose":
        m = np.asarray(m4, dtype=np.float64)
        return cls(m[:3, :3], m[:3, 3])

    def as_dict(self) -> dict:
        return {"head_R_wh": self.R_wh.tolist(), "head_origin_w_m": self.origin_w.tolist()}


@dataclass(frozen=True)
class Gaze:
    """Where the eyes point. Build with one of the three constructors."""
    mode: str
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    vergence_m: float | None = None
    point_w: tuple | None = None
    angles: dict = field(default_factory=dict)

    @classmethod
    def yaw_pitch(cls, yaw_deg: float, pitch_deg: float, vergence_m: float | None = None) -> "Gaze":
        """A cyclopean gaze; with vergence_m the eyes verge on the point at that distance from the
        head origin (f3d-vision's make_calibration), without it they look parallel."""
        return cls("yaw_pitch", float(yaw_deg), float(pitch_deg), None if vergence_m is None else float(vergence_m))

    @classmethod
    def verged_point(cls, point_w) -> "Gaze":
        return cls("verged_point", point_w=tuple(float(x) for x in point_w))

    @classmethod
    def per_eye(cls, **angles) -> "Gaze":
        return cls("per_eye", angles={k: (float(v[0]), float(v[1])) for k, v in angles.items()})

    def as_dict(self) -> dict:
        return {"mode": self.mode, "yaw_deg": self.yaw_deg, "pitch_deg": self.pitch_deg,
                "vergence_m": self.vergence_m, "point_w_m": self.point_w, "per_eye": self.angles}


@dataclass(frozen=True)
class EyePose:
    eye: str
    centre_h: np.ndarray
    centre_w: np.ndarray
    R_hb: np.ndarray            # Blender camera local -> head
    R_wb: np.ndarray            # Blender camera local -> world
    yaw_deg: float
    pitch_deg: float

    @property
    def matrix_world(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = self.R_wb, self.centre_w
        return m

    @property
    def R_hc(self) -> np.ndarray:
        """OpenCV camera -> head, as f3d-vision's calibrations store it."""
        return cv_to_blender(self.R_hb)

    def as_dict(self) -> dict:
        return {"eye": self.eye, "centre_h_m": self.centre_h.tolist(), "centre_w_m": self.centre_w.tolist(),
                "R_hb": self.R_hb.tolist(), "matrix_world": self.matrix_world.tolist(),
                "yaw_deg": self.yaw_deg, "pitch_deg": self.pitch_deg}


@dataclass(frozen=True)
class Rig:
    ipd_m: float = DEFAULT_IPD_M
    roll: str = "upright"
    baseline_h: tuple = (1.0, 0.0, 0.0)

    def __post_init__(self):
        if self.ipd_m <= 0:
            raise ValueError("ipd must be positive")
        if self.roll not in ROLLS:
            raise ValueError(f"unknown roll {self.roll!r}; expected one of {ROLLS}")

    @property
    def baseline_m(self) -> float:
        return self.ipd_m

    def centre_h(self, eye: str) -> np.ndarray:
        return {"L": np.array([-self.ipd_m / 2.0, 0.0, 0.0]), "R": np.array([self.ipd_m / 2.0, 0.0, 0.0]),
                "C": np.zeros(3)}[eye]

    def eye_direction_h(self, head: HeadPose, gaze: Gaze, eye: str) -> np.ndarray:
        c = self.centre_h(eye)
        if gaze.mode == "yaw_pitch":
            d = gaze_direction(gaze.yaw_deg, gaze.pitch_deg)
            return d if gaze.vergence_m is None or eye == "C" else d * gaze.vergence_m - c
        if gaze.mode == "verged_point":
            return head.to_head(np.asarray(gaze.point_w)) - c
        if gaze.mode == "per_eye":
            if eye not in gaze.angles:
                raise ValueError(f"per-eye gaze has no angles for eye {eye}")
            return gaze_direction(*gaze.angles[eye])
        raise ValueError(f"unknown gaze mode {gaze.mode!r}")

    def pose(self, head: HeadPose, gaze: Gaze, eye: str) -> EyePose:
        d = self.eye_direction_h(head, gaze, eye)
        r_hb = camera_rotation(d, self.roll, self.baseline_h)    # normalises once, as make_calibration
        yaw, pitch = gaze_angles(d)
        c_h = self.centre_h(eye)
        # R_WB = R_WH R_HC CV in f3d-vision's order (_eye_matrix), so world poses match bit for bit
        r_wb = (head.R_wh @ cv_to_blender(r_hb)) @ CV_TO_BLENDER
        return EyePose(eye, c_h, head.to_world(c_h), r_hb, r_wb, yaw, pitch)

    def fixed_pose(self, head: HeadPose, eye: str, yaw_offset_deg: float = 0.0) -> EyePose:
        """A camera whose axes are the head's (panoramas), optionally turned by yaw_offset_deg
        about head +Y, +yaw right, as preview360's --yaw-offset-px."""
        r_hb = rot_y(-yaw_offset_deg)
        c_h = self.centre_h(eye)
        return EyePose(eye, c_h, head.to_world(c_h), r_hb, head.R_wh @ r_hb, yaw_offset_deg, 0.0)

    def as_dict(self) -> dict:
        return {"ipd_m": self.ipd_m, "roll": self.roll, "baseline_h": list(self.baseline_h)}


def vergence_deg(point_h, centres_h) -> float:
    a, b = np.asarray(point_h) - centres_h[0], np.asarray(point_h) - centres_h[1]
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))
