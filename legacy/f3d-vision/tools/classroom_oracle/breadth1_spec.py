"""Breadth-1: the frozen canonical configuration and the cell geometry.

Contract: docs/classroom-oracle/breadth-1-spherical-glance-contract.md, sections 2-4.  Plain Python +
NumPy, imported by both the Blender-side renderer and the host-side tools.  The checker keeps its own
literal copy of these values, so changing one here cannot hide a configuration change.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

SHARED = Path("/home/lvelho/rd/f3d-vision")
BASE_COMMIT = "3aa0cc6777401ddef1eb6755bff60515d87bd2db"

# ---- section 3: the frozen canonical observation
WIDTH, HEIGHT = 720, 360
CELLS = WIDTH * HEIGHT
CELL_DEG = 0.5
CAMERA_TYPE = "PANO"
PANORAMA_TYPE = "EQUIRECTANGULAR"
LONGITUDE = (-math.pi, math.pi)
LATITUDE = (-math.pi / 2.0, math.pi / 2.0)
SPP = 512
DEVICE = "OPTIX"
SEED = 0
PIXEL_FILTER = "BOX"
FILTER_WIDTH = 1.0
CLIP_M = (0.01, 1000.0)
CATALOG_COUNT = 234
EYE_POSE_TOL = 1e-6

# ---- section 2: accepted sources (paths relative to SHARED)
SOURCES = {
    "seeds": ("previews/controller-01-full/bootstrap/seeds.json",
              "6ef833196de2462370ba4b2a2a8eed3fbde7f0a61e5aa6cbaf95f66947134c4f"),
    "catalog": ("previews/controller-01-full/bootstrap/instance_catalog.json",
                "be26594254b03fbda7216ac1a942a5c800cf4bdd8f79751a2b8c22d6bfb4c7ae"),
    "controller02_result": ("previews/controller-02-classroom-replay/result.json",
                            "a6dc4879f70216f99d098841f9d2c33dda0ceb71f01a4ab76320b0d08212dc55"),
    "blend": ("scenes/classroom/classroom_eye.blend",
              "dca66a3257b909aef00b0331652c55f62a93a8dff0665d070fba7efa436953cc"),
}

# ---- section 4: analysis constants (declared before the canonical run)
DOMAIN_YAW_DEG = (-25.0, 25.0)
DOMAIN_PITCH_DEG = (-20.0, 20.0)
SEED_FALLBACK_NORM = 1e-9
# Post-run numerical clarification of the representative-seed rule (contract section 11), authorized by
# Luiz/Chat on 2026-10-03 after the valid canonical observation.  Support cells whose float64 dot product
# to the normalized solid-angle-weighted mean satisfies max_dot - dot <= SEED_TIE_DOT_EPS are numerically
# tied and resolved by smaller row, then smaller column.  It removes round-off ambiguity in mathematically
# mirror-symmetric components; it changes only the DERIVED representative seed, never the observation,
# visibility, components, supports, ranges, ranks or accounting.
SEED_TIE_DOT_EPS = 1e-12
# Known-answer controls of that rule (exact mirror symmetry; contract section 11): (name, cells, seed).
# In the first, exact float64 equality would pick (60, 29): the round-off favours the larger column.
SEED_TIE_CONTROLS = [
    ("two-cell row, round-off favours the larger column", [(60, 28), (60, 29)], (60, 28)),
    ("four-way tie straddling the equator", [(179, 359), (179, 360), (180, 359), (180, 360)], (179, 359)),
    ("four-way tie across the longitude seam", [(179, 719), (179, 0), (180, 719), (180, 0)], (179, 0)),
]
SOLID_ANGLE_EDGES_SR = [10.0 ** (k / 2.0) for k in range(-14, 3)]
CELL_COUNT_EDGES = [2 ** k for k in range(0, 19)]
RANGE_EDGES_M = [0.0] + [2.0 ** (k / 2.0) for k in range(-6, 13)] + [math.inf]
RANGE_QUANTILES = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
TOP_K = [1, 5, 10, 25]

TRUTH_REFERENCE = "REFERENCE / EVALUATION"
TRUTH_ORACLE = "ORACLE INPUT"
TRUTH_DERIVED = "DERIVED"
TRUTH_CLASSES = (TRUTH_REFERENCE, TRUTH_ORACLE, TRUTH_DERIVED)


def yaw_centers_deg() -> np.ndarray:
    return -180.0 + CELL_DEG * (np.arange(WIDTH, dtype=np.float64) + 0.5)


def pitch_centers_deg() -> np.ndarray:
    return 90.0 - CELL_DEG * (np.arange(HEIGHT, dtype=np.float64) + 0.5)


def direction_h(yaw_deg, pitch_deg) -> np.ndarray:
    """Head-frame unit direction(s): +X right, +Y up, -Z forward."""
    y = np.radians(np.asarray(yaw_deg, np.float64))
    p = np.radians(np.asarray(pitch_deg, np.float64))
    return np.stack([np.sin(y) * np.cos(p), np.sin(p), -np.cos(y) * np.cos(p)], axis=-1)


def cell_directions_h() -> np.ndarray:
    """(HEIGHT, WIDTH, 3) unit directions of the cell centres."""
    yaw, pitch = np.meshgrid(yaw_centers_deg(), pitch_centers_deg())
    return direction_h(yaw, pitch)


def row_weights() -> np.ndarray:
    """Exact equirectangular solid angle of one cell in each row (contract 4.2)."""
    dlam = 2.0 * math.pi / WIDTH
    j = np.arange(HEIGHT, dtype=np.float64)
    phi_hi = math.pi / 2.0 - j * math.pi / HEIGHT
    phi_lo = math.pi / 2.0 - (j + 1.0) * math.pi / HEIGHT
    return dlam * (np.sin(phi_hi) - np.sin(phi_lo))


def solid_angle(row_counts: np.ndarray, weights: np.ndarray) -> float:
    """Sum_j n_j dOmega_j in float64, ascending j (the declared canonical summation)."""
    total = 0.0
    for n, w in zip(np.asarray(row_counts, np.int64).tolist(), np.asarray(weights, np.float64).tolist()):
        if n:
            total += n * w
    return float(total)


def yaw_pitch_deg(d_h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(d_h, np.float64)
    yaw = np.degrees(np.arctan2(d[..., 0], -d[..., 2]))
    pitch = np.degrees(np.arctan2(d[..., 1], np.hypot(d[..., 0], d[..., 2])))
    return yaw, pitch
