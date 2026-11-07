"""The binocular rig and the gaze geometry, in numpy alone (no bpy; both interpreters import it).

    python tools/rig.py --self-test       # identities that must hold; exit 1 if one does not

Conventions (CLAUDE.md, D3, D12):
* The head is fixed. EYE, the Blender camera object, is the head frame and the cyclopean point:
  local +X right, +Y up, -Z primary gaze. `head_rot3` below is its rigid 3x3 with those local
  axes as columns (world = head_rot3 @ local); `head_origin` its position, metres.
* The two eye centres sit on the head's X axis at +-ipd/2: C_L = origin - (ipd/2) right,
  C_R = origin + (ipd/2) right. Eye 0 is L, eye 1 is R. Nothing in the scene changes.
* A gaze is (yaw, pitch) in degrees in the head frame, composed intrinsic yaw-then-pitch about
  the eye's own centre (Ry(-yaw) @ Rx(pitch), the A3 composition; +yaw right, +pitch up). No
  torsion in Phase B1: both eyes use this composition (assumed; Listing's law is a later
  refinement that changes this file, not the record).
* A verged pair fixates one world point P: each eye's gaze is P - C_i. The head-frame direction
  of a sample stays in the head frame for both eyes; only the origin differs.
"""
from __future__ import annotations

import math
import sys

import numpy as np

EYE_NAMES = ("L", "R")
DEFAULT_IPD_M = 0.063


# ----------------------------------------------------------------------------------------
# gaze composition (moved from fixation_sequence.py; same maths)
# ----------------------------------------------------------------------------------------

def gaze_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """3x3: Blender-camera-local -> head frame for a gaze, Ry(-yaw) @ Rx(pitch), the same
    composition as render_foveated.gaze_matrix."""
    y, p = math.radians(-yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0.0, math.sin(y)], [0.0, 1.0, 0.0], [-math.sin(y), 0.0, math.cos(y)]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(p), -math.sin(p)], [0.0, math.sin(p), math.cos(p)]])
    return ry @ rx


def to_eye_frame(direction_cam: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Camera-shader frame (+Z forward) -> Blender camera local (-Z forward) -> head frame."""
    local = direction_cam * np.array([1.0, 1.0, -1.0])
    return local @ gaze_rotation(yaw_deg, pitch_deg).T


def to_camera_frame(d_head: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Inverse of to_eye_frame: head-frame directions -> camera-shader frame (+Z forward) of the
    camera at this gaze, so warp.raster_of_direction can place them on its raster."""
    local = np.asarray(d_head, dtype=np.float64) @ gaze_rotation(yaw_deg, pitch_deg)
    return local * np.array([1.0, 1.0, -1.0])


def gaze_of_world_direction(d_world, head_rot3: np.ndarray) -> tuple[float, float]:
    """(yaw, pitch) in degrees that points a camera along d_world, given the head rotation."""
    d = head_rot3.T @ (np.asarray(d_world, dtype=np.float64) / np.linalg.norm(d_world))
    return math.degrees(math.atan2(d[0], -d[2])), math.degrees(math.asin(np.clip(d[1], -1.0, 1.0)))


def forward_of_gaze(yaw_deg: float, pitch_deg: float, head_rot3: np.ndarray) -> np.ndarray:
    """World unit vector the camera looks along for a gaze (local -Z through the composition)."""
    return head_rot3 @ (gaze_rotation(yaw_deg, pitch_deg) @ np.array([0.0, 0.0, -1.0]))


def camera_pose(head_origin: np.ndarray, head_rot3: np.ndarray, offset_local, yaw_deg: float,
                pitch_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(position, 3x3 rotation) of a camera at head-local offset_local looking along a gaze:
    world = T(origin) @ R_head @ T(offset) @ Ry(-yaw) @ Rx(pitch), the composition
    render_foveated.gaze_matrix builds with mathutils."""
    pos = head_origin + head_rot3 @ np.asarray(offset_local, dtype=np.float64)
    return pos, head_rot3 @ gaze_rotation(yaw_deg, pitch_deg)


# ----------------------------------------------------------------------------------------
# the rig
# ----------------------------------------------------------------------------------------

def eye_offsets_local(ipd_m: float) -> np.ndarray:
    """(2, 3) head-local offsets of eye L and R."""
    return np.array([[-ipd_m / 2.0, 0.0, 0.0], [ipd_m / 2.0, 0.0, 0.0]])


def eye_centres(head_origin: np.ndarray, head_rot3: np.ndarray, ipd_m: float) -> np.ndarray:
    """(2, 3) world positions of eye L and R."""
    return head_origin[None, :] + eye_offsets_local(ipd_m) @ head_rot3.T


def target_point(target: dict, head_origin: np.ndarray) -> np.ndarray | None:
    """World fixation point of a calib_room target: cards sit at head_origin + dir_world * distance
    (make_calib_room.place_card); wires are vertical at 1.5 m on the azimuth, taken at eye height.
    Returns None for a target with no usable geometry."""
    if "dir_world" in target and "distance_m" in target:
        d = np.asarray(target["dir_world"], dtype=np.float64)
        return head_origin + d / np.linalg.norm(d) * float(target["distance_m"])
    if "azimuth_deg" in target and "distance_m" in target:
        az = math.radians(float(target["azimuth_deg"]))
        return head_origin + float(target["distance_m"]) * np.array([math.sin(az), math.cos(az), 0.0])
    return None


def vergence_deg(p: np.ndarray, centres: np.ndarray) -> float:
    """Angle between the two eyes' lines of sight to p."""
    a, b = p - centres[0], p - centres[1]
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def pair_for_point(p: np.ndarray, head_origin: np.ndarray, head_rot3: np.ndarray, ipd_m: float,
                   verged: bool = True) -> dict:
    """Everything a fixation pair on p needs and everything the checks predict from it.

    verged=True: each eye looks at p. verged=False (the control): both eyes take the cyclopean
    gaze, so their lines of sight are parallel and miss p by a predictable amount.
    Returns a dict with per-eye gaze, centre, predicted central-ray distance and predicted miss
    (metres, from p to the closest point of the central ray, which for a card facing the
    cyclopean eye is also where the ray hits the card plane)."""
    p = np.asarray(p, dtype=np.float64)
    centres = eye_centres(head_origin, head_rot3, ipd_m)
    g_cyc = p - head_origin
    g_cyc = g_cyc / np.linalg.norm(g_cyc)
    eyes = []
    for k in range(2):
        c = centres[k]
        los = (p - c) if verged else g_cyc
        los = los / np.linalg.norm(los)
        yaw, pitch = gaze_of_world_direction(los, head_rot3)
        t = float((p - c) @ los)                      # parameter of the closest point
        closest = c + t * los
        eyes.append({"eye": EYE_NAMES[k], "eye_id": k, "centre_m": c.tolist(),
                     "yaw": yaw, "pitch": pitch,
                     "predicted_distance_m": t,      # ray distance to the card plane (= |p - c| if verged)
                     "predicted_miss_m": float(np.linalg.norm(p - closest))})
    return {"point_m": p.tolist(), "verged": verged,
            "cyclopean_distance_m": float(np.linalg.norm(p - head_origin)),
            "cyclopean_yaw": gaze_of_world_direction(g_cyc, head_rot3)[0],
            "cyclopean_pitch": gaze_of_world_direction(g_cyc, head_rot3)[1],
            "vergence_deg": vergence_deg(p, centres) if verged else 0.0,
            "vergence_if_verged_deg": vergence_deg(p, centres),
            "eyes": eyes}


# ----------------------------------------------------------------------------------------
# epipolar coordinates on the sphere (D13). Both centres lie on the head's X axis, so every
# epipolar plane contains that axis: a head-frame direction d is (theta, phi) with
#   theta = angle from +X (the baseline axis), in [0, pi]
#   phi   = the plane's rotation about X, atan2(d_y, -d_z): 0 is the forward horizontal
#           half-plane, +pi/2 straight up, pi backward.
# Corresponding directions in the two eyes share phi and differ in theta; the parallax
# theta_R - theta_L is positive for every finite point, and equals the vergence at the fixated
# point. Eye torsion does not enter: directions are in the head frame.
# ----------------------------------------------------------------------------------------

def epipolar(d_head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(theta_deg, phi_deg) of head-frame unit directions."""
    d = np.asarray(d_head, dtype=np.float64)
    # atan2 form, not arccos(d_x): arccos is ill-conditioned at the axis, where a float32
    # direction's x is quantised (-0.99999994 next to -1) and the error reaches 0.02 deg, more
    # than the parallax there; hypot(y, z) keeps the small components' full relative precision.
    theta = np.degrees(np.arctan2(np.hypot(d[..., 1], d[..., 2]), d[..., 0]))
    phi = np.degrees(np.arctan2(d[..., 1], -d[..., 2]))
    return theta, phi


def phi_distance_deg(phi_a: np.ndarray, phi_b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Great-circle distance across epipolar planes: the phi difference (wrapped) scaled by
    sin theta, which is what a direction error perpendicular to the epipolar line measures."""
    d = np.abs((phi_a - phi_b + 180.0) % 360.0 - 180.0)
    return d * np.sin(np.radians(theta))


def triangulate(theta_l_deg: np.ndarray, theta_r_deg: np.ndarray, ipd_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray distances (from C_L, from C_R) and the parallax (deg) of a point seen at theta_L from
    C_L and theta_R from C_R, by the sine rule in the triangle C_L C_R P:
        gamma = theta_R - theta_L,  |P - C_L| = ipd sin theta_R / sin gamma,  |P - C_R| = ipd sin theta_L / sin gamma.
    gamma <= 0 (parallel or diverging rays) gives inf."""
    tl, tr = np.radians(np.asarray(theta_l_deg, dtype=np.float64)), np.radians(np.asarray(theta_r_deg, dtype=np.float64))
    gamma = tr - tl
    with np.errstate(divide="ignore", invalid="ignore"):
        dl = np.where(gamma > 0, ipd_m * np.sin(tr) / np.sin(gamma), np.inf)
        dr = np.where(gamma > 0, ipd_m * np.sin(tl) / np.sin(gamma), np.inf)
    return dl, dr, np.degrees(gamma)


def depth_quantum_m(distance_m, s0_deg: float, ipd_m: float) -> np.ndarray:
    """Depth change for one sample spacing of parallax at this distance: D^2 s0 / ipd to first
    order (exact: d|P-C|/d gamma = -|P-C| cot gamma). The tolerance of the triangulation check."""
    D = np.asarray(distance_m, dtype=np.float64)
    gamma = 2.0 * np.arctan(ipd_m / 2.0 / D)
    return D * math.radians(s0_deg) / np.tan(gamma)


def head_rot3_of(eye_record: dict) -> np.ndarray:
    """From bl_common.eye_record's forward/up: columns right, up, -forward."""
    fwd, up = np.array(eye_record["forward"], float), np.array(eye_record["up"], float)
    right = np.cross(fwd, up)
    return np.stack([right, up, -fwd], axis=1)


# ----------------------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------------------

def self_test() -> list[str]:
    fails = []
    rng = np.random.default_rng(0)
    origin = np.array([0.0, 0.0, 1.6])
    # head frame of make_eye(yaw=0): local -Z -> world +Y, +Y -> +Z, +X -> +X
    head = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    if not np.allclose(head @ np.array([0, 0, -1.0]), [0, 1, 0]) or not np.allclose(head @ np.array([0, 1.0, 0]), [0, 0, 1]):
        fails.append("head frame of make_eye(0) is not (+X, +Z up, +Y forward)")
    ipd = DEFAULT_IPD_M
    c = eye_centres(origin, head, ipd)
    if not np.allclose(c[1] - c[0], [ipd, 0, 0]) or not np.allclose((c[0] + c[1]) / 2, origin):
        fails.append(f"eye centres {c.tolist()} are not +-ipd/2 about the origin on head +X")

    # 1. yaw/pitch composition: the gaze computed for a direction looks exactly along it
    for _ in range(200):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        yaw, pitch = gaze_of_world_direction(d, head)
        f = forward_of_gaze(yaw, pitch, head)
        if np.linalg.norm(f - d) > 1e-12:
            fails.append(f"forward_of_gaze(gaze_of_world_direction(d)) != d for d={d}")
            break
    # yaw sign: +yaw looks right (world +X for the level head)
    if forward_of_gaze(90.0, 0.0, head)[0] < 0.999:
        fails.append("+90 yaw does not look along world +X (right)")
    if forward_of_gaze(0.0, 90.0, head)[2] < 0.999:
        fails.append("+90 pitch does not look up")

    # 2. to_eye_frame at the raster centre equals forward_of_gaze in the head frame
    centre_cam = np.array([[0.0, 0.0, 1.0]])
    for yaw, pitch in ((0.0, 0.0), (30.0, -10.0), (-120.0, 45.0)):
        a = to_eye_frame(centre_cam, yaw, pitch)[0]
        b = head.T @ forward_of_gaze(yaw, pitch, head)
        if np.linalg.norm(a - b) > 1e-12:
            fails.append(f"to_eye_frame centre != gaze forward at ({yaw}, {pitch})")

    # 3. a verged pair on a random point: each eye's central ray passes through the point, the
    #    predicted distances are |P - C_i|, the vergence is the angle at P, symmetric on the midline
    for _ in range(100):
        p = origin + rng.uniform(-3, 3, size=3) * np.array([1, 1, 0.5]) + np.array([0, 1.5, 0])
        pr = pair_for_point(p, origin, head, ipd, verged=True)
        for e in pr["eyes"]:
            cpos, rot = camera_pose(origin, head, eye_offsets_local(ipd)[e["eye_id"]], e["yaw"], e["pitch"])
            los = rot @ np.array([0.0, 0.0, -1.0])
            miss = np.linalg.norm(np.cross(p - cpos, los))
            if miss > 1e-9 or abs(e["predicted_distance_m"] - np.linalg.norm(p - cpos)) > 1e-9 or e["predicted_miss_m"] > 1e-9:
                fails.append(f"verged eye {e['eye']} central ray misses p by {miss:.2e} m")
                break
    p_mid = origin + np.array([0.0, 2.0, 0.0])
    pr = pair_for_point(p_mid, origin, head, ipd)
    expect = 2.0 * math.degrees(math.atan(ipd / 2.0 / 2.0))
    if abs(pr["vergence_deg"] - expect) > 1e-9 or abs(pr["eyes"][0]["yaw"] + pr["eyes"][1]["yaw"]) > 1e-9:
        fails.append(f"midline vergence {pr['vergence_deg']:.6f} != {expect:.6f} or yaws not symmetric")

    # 4. the control: parallel gazes miss p by (ipd/2) sqrt(1 - (right . g)^2), and by exactly
    #    ipd/2 on the midline
    pc = pair_for_point(p_mid, origin, head, ipd, verged=False)
    for e in pc["eyes"]:
        if abs(e["predicted_miss_m"] - ipd / 2) > 1e-12 or abs(e["predicted_distance_m"] - 2.0) > 1e-12:
            fails.append(f"control on the midline: miss {e['predicted_miss_m']} distance {e['predicted_distance_m']}")
    for az in (30.0, 55.0, -70.0):
        p = origin + 1.2 * np.array([math.sin(math.radians(az)), math.cos(math.radians(az)), 0.0])
        pc = pair_for_point(p, origin, head, ipd, verged=False)
        g = (p - origin) / np.linalg.norm(p - origin)
        expect = ipd / 2 * math.sqrt(1.0 - float(head[:, 0] @ g) ** 2)
        if any(abs(e["predicted_miss_m"] - expect) > 1e-12 for e in pc["eyes"]):
            fails.append(f"control miss at azimuth {az}: {[e['predicted_miss_m'] for e in pc['eyes']]} != {expect}")

    # 5. target_point on the three calib_room kinds
    t_ring = {"dir_world": [0.0, 1.0, 0.0], "distance_m": 2.0}
    t_wire = {"azimuth_deg": -55.0, "distance_m": 1.5}
    if not np.allclose(target_point(t_ring, origin), [0, 2, 1.6]):
        fails.append("ring target point")
    pw = target_point(t_wire, origin)
    if pw is None or abs(np.linalg.norm(pw[:2]) - 1.5) > 1e-12 or pw[2] != 1.6:
        fails.append(f"wire target point {pw}")
    if target_point({"name": "x"}, origin) is not None:
        fails.append("target with no geometry should give None")

    # 6. epipolar coordinates and triangulation: a random point's directions from the two
    #    centres share phi, triangulate back to the right distances, and the parallax at the
    #    fixated midline point equals the vergence; parallel rays give inf
    for _ in range(200):
        p = origin + rng.uniform(-3, 3, size=3) * np.array([1, 1, 0.5]) + np.array([0, 1.5, 0])
        dl, dr_ = p - c[0], p - c[1]
        Dl, Dr = np.linalg.norm(dl), np.linalg.norm(dr_)
        hl, hr = head.T @ (dl / Dl), head.T @ (dr_ / Dr)
        tl, pl = epipolar(hl); tr, pr_ = epipolar(hr)
        if phi_distance_deg(pl, pr_, tl) > 1e-9:
            fails.append(f"epipolar phi differs between the eyes for p={p}: {pl} vs {pr_}")
            break
        Dl2, Dr2, gam = triangulate(tl, tr, ipd)
        if abs(Dl2 - Dl) > 1e-9 * Dl or abs(Dr2 - Dr) > 1e-9 * Dr or gam <= 0:
            fails.append(f"triangulation of p={p}: {Dl2:.6f}/{Dr2:.6f} vs {Dl:.6f}/{Dr:.6f}, parallax {gam}")
            break
    tl, _ = epipolar(head.T @ ((p_mid - c[0]) / np.linalg.norm(p_mid - c[0])))
    tr, _ = epipolar(head.T @ ((p_mid - c[1]) / np.linalg.norm(p_mid - c[1])))
    if abs(triangulate(tl, tr, ipd)[2] - pair_for_point(p_mid, origin, head, ipd)["vergence_deg"]) > 1e-9:
        fails.append("parallax at the fixated midline point != vergence")
    if np.isfinite(triangulate(80.0, 80.0, ipd)[0]) or np.isfinite(triangulate(80.0, 79.0, ipd)[0]):
        fails.append("parallel or diverging rays should triangulate to inf")
    if not (0.10 < float(depth_quantum_m(2.0, 0.1, ipd)) < 0.12):
        fails.append(f"depth quantum at 2 m, s0 0.1: {float(depth_quantum_m(2.0, 0.1, ipd)):.4f} m, expected ~0.111")
    # to_camera_frame inverts to_eye_frame
    dh = rng.normal(size=(50, 3)); dh /= np.linalg.norm(dh, axis=1, keepdims=True)
    if np.abs(to_eye_frame(to_camera_frame(dh, 33.0, -12.0), 33.0, -12.0) - dh).max() > 1e-12:
        fails.append("to_camera_frame is not the inverse of to_eye_frame")

    # 7. head_rot3_of round-trips the eye record
    rec = {"forward": (head @ [0, 0, -1.0]).tolist(), "up": (head @ [0, 1.0, 0]).tolist()}
    if not np.allclose(head_rot3_of(rec), head):
        fails.append("head_rot3_of(eye_record) != head rotation")
    return fails


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        f = self_test()
        for x in f:
            print("[rig] FAIL", x)
        print(f"[rig] self-test {'FAILED' if f else 'ok'}")
        sys.exit(1 if f else 0)
    print(__doc__)
