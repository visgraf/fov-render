"""The foveated warp, in numpy alone: raster -> eccentricity -> direction -> footprint.

Split out of fixation_sequence.py for Phase B so host-side checkers (check_pairs.py, the venv)
and Blender-side renderers import the same lines. No bpy here; importable by both interpreters.

    python tools/warp.py --self-test      # identities that must hold; exit 1 if one does not

The formula is the one in foveated_camera.osl: spacing s(e) = s0 (1 + e/E2) integrates to a
raster radius w(e) = (E2/s0) ln(1 + e/E2); normalised by the rim, e(r) = E2 ((1 + e_max/E2)^r - 1)
for r in [0, 1]. Raster rows are top-down as in the EXR file; the OSL raster y points up, so
row i has Py = 0.5 - (i + 0.5)/n (measured against the Position pass, 2026-09-13).

Check (a) in check_sequence.py / check_pairs.py is the live test vector between this file and
the OSL shader: on every sequence, the analytic direction is compared with the renderer's own
Position pass (0.018 reference pixels at p99.9 on the full profile, A4).
"""
from __future__ import annotations

import math
import sys

import numpy as np


def raster_size(s0_deg: float, e2_deg: float, emax_deg: float) -> int:
    """Pixels across so that spacing at the centre is s0."""
    return int(round(2.0 * (e2_deg / s0_deg) * math.log(1.0 + emax_deg / e2_deg)))


def s0_of(n: int, e2_deg: float, emax_deg: float) -> float:
    """Foveal spacing (deg) that an n x n raster actually has."""
    return 2.0 * e2_deg * math.log(1.0 + emax_deg / e2_deg) / n


def eccentricity_of_r(r: np.ndarray, e2: float, emax: float) -> np.ndarray:
    """Eccentricity in degrees at normalised raster radius r (1 at the rim)."""
    return e2 * ((1.0 + emax / e2) ** r - 1.0)


def r_of_eccentricity(e_deg: float, e2: float, emax: float) -> float:
    """Inverse of eccentricity_of_r: the raster radius (0..1) at which eccentricity e_deg falls."""
    return math.log(1.0 + e_deg / e2) / math.log(1.0 + emax / e2)


def warp_direction(px: np.ndarray, py: np.ndarray, e2: float, emax: float) -> np.ndarray:
    """Cycles camera-shader frame (+X right, +Y up, +Z forward) direction for normalised raster
    offsets (px, py) from the centre, r = 2*hypot in [0, 1] at the rim. Same formula as
    foveated_camera.osl; evaluated outside r <= 1 too (finite differences need it)."""
    r = 2.0 * np.hypot(px, py)
    e = np.radians(eccentricity_of_r(r, e2, emax))
    phi = np.arctan2(py, px)
    return np.stack([np.sin(e) * np.cos(phi), np.sin(e) * np.sin(phi), np.cos(e)], axis=-1)


def raster_samples(n: int, e2: float, emax: float) -> dict:
    """Pixel-centre directions (camera frame), footprints (sr), the inside mask and raster
    indices for an n x n raster."""
    j, i = np.meshgrid(np.arange(n), np.arange(n))
    px = (j + 0.5) / n - 0.5
    py = 0.5 - (i + 0.5) / n
    r = 2.0 * np.hypot(px, py)
    inside = r <= 1.0
    d = warp_direction(px, py, e2, emax)
    h = 1.0 / n                                     # one pixel, in normalised raster units
    dx = warp_direction(px + h / 2, py, e2, emax) - warp_direction(px - h / 2, py, e2, emax)
    dy = warp_direction(px, py + h / 2, e2, emax) - warp_direction(px, py - h / 2, e2, emax)
    omega = np.linalg.norm(np.cross(dx, dy), axis=-1)
    return {"direction_cam": d, "footprint": omega, "inside": inside,
            "raster_index": np.stack([i, j], axis=-1)}


def raster_of_direction(d_cam: np.ndarray, n: int, e2: float, emax: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of raster_samples: camera-shader-frame unit directions (+Z forward) -> continuous
    raster coordinates (row, col) with pixel centres at integers, plus the inside mask (r <= 1).
    Directions behind the camera or beyond e_max are outside. Used for ground-truth
    correspondence: where a world point falls in the other eye's raster."""
    d = np.asarray(d_cam, dtype=np.float64)
    e = np.degrees(np.arccos(np.clip(d[..., 2], -1.0, 1.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(np.maximum(1.0 + e / e2, 1e-300)) / math.log(1.0 + emax / e2)
    phi = np.arctan2(d[..., 1], d[..., 0])
    px, py = 0.5 * r * np.cos(phi), 0.5 * r * np.sin(phi)
    row = (0.5 - py) * n - 0.5
    col = (px + 0.5) * n - 0.5
    return np.stack([row, col], axis=-1), r <= 1.0


def centre_pixels(n: int) -> np.ndarray:
    """(row, col) of the pixel(s) nearest the raster centre: one for odd n, the 2x2 block for
    even n. Their mean direction is the foveal centre to within a quarter sample."""
    c = n // 2
    if n % 2 == 1:
        return np.array([[c, c]])
    return np.array([[c - 1, c - 1], [c - 1, c], [c, c - 1], [c, c]])


def cap_sr(emax_deg: float) -> float:
    return 2.0 * math.pi * (1.0 - math.cos(math.radians(emax_deg)))


# ----------------------------------------------------------------------------------------
# self-test: identities that must hold, each of which can fail
# ----------------------------------------------------------------------------------------

def self_test() -> list[str]:
    fails = []
    for e2, emax, s0 in ((2.0, 45.0, 0.1), (2.0, 45.0, 0.05), (1.0, 45.0, 0.1), (4.0, 30.0, 0.1)):
        n = raster_size(s0, e2, emax)
        # s0 round-trips through the raster size to within one pixel's worth
        if abs(s0_of(n, e2, emax) - s0) > s0 / n:
            fails.append(f"s0 round trip: {s0} -> n {n} -> {s0_of(n, e2, emax):.5f}")
        # eccentricity inverse
        for e in (0.0, 0.5, 2.0, 10.0, emax):
            r = r_of_eccentricity(e, e2, emax)
            if abs(float(eccentricity_of_r(np.array(r), e2, emax)) - e) > 1e-9:
                fails.append(f"e(r(e)) != e at e={e}")
        # spacing at the centre is s0: the direction step over one central pixel, in degrees
        rs = raster_samples(n, e2, emax)
        c = centre_pixels(n)
        d0, d1 = rs["direction_cam"][c[0][0], c[0][1]], rs["direction_cam"][c[-1][0], c[-1][1]]
        step = math.degrees(math.acos(float(np.clip(d0 @ d1, -1, 1)))) / (math.sqrt(2.0) if n % 2 == 0 else 1.0)
        if n % 2 == 0 and abs(step - s0_of(n, e2, emax)) > 0.02 * s0:
            fails.append(f"central spacing {step:.5f} deg, expected {s0_of(n, e2, emax):.5f} (n {n})")
        # the rim is at e_max
        rim = warp_direction(np.array(0.5), np.array(0.0), e2, emax)
        if abs(math.degrees(math.acos(float(rim[2]))) - emax) > 1e-9:
            fails.append(f"rim eccentricity != e_max for E2 {e2} e_max {emax}")
        # footprints integrate to the cap (the rim ring of straddling pixels adds about +0.6%, A4)
        total = float(rs["footprint"][rs["inside"]].sum())
        if abs(total / cap_sr(emax) - 1.0) > 0.015:
            fails.append(f"footprint sum {total:.5f} sr vs cap {cap_sr(emax):.5f} (n {n})")
        # rows are top-down: row 0 looks up (+Y), column 0 looks left (-X)
        if not (rs["direction_cam"][0, n // 2, 1] > 0.0 and rs["direction_cam"][n // 2, 0, 0] < 0.0):
            fails.append("raster orientation: row 0 should look up and column 0 left")
        # the inverse warp returns every inside pixel's own index
        rc, ins = raster_of_direction(rs["direction_cam"], n, e2, emax)
        ij = rs["raster_index"].astype(np.float64)
        if not np.array_equal(ins, rs["inside"]):
            fails.append(f"inverse warp inside mask differs on {int((ins != rs['inside']).sum())} pixels (n {n})")
        err = np.abs(rc - ij)[rs["inside"]].max()
        if err > 1e-6:
            fails.append(f"inverse warp round trip off by {err:.2e} px (n {n})")
        # unit directions
        norms = np.linalg.norm(rs["direction_cam"], axis=-1)
        if np.abs(norms - 1.0).max() > 1e-12:
            fails.append("directions are not unit")
    return fails


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        f = self_test()
        for x in f:
            print("[warp] FAIL", x)
        print(f"[warp] self-test {'FAILED' if f else 'ok'}")
        sys.exit(1 if f else 0)
    print(__doc__)
