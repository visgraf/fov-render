"""Check a foveated render, and optionally compare two of them.

    python tools/check_foveated.py fix/f000
    python tools/check_foveated.py fix/optix --compare fix/cpu

Checks, each of which can fail:
    depth      the Depth pass equals |Position - camera centre|, i.e. ray distance
    warp       measured eccentricity per pixel matches e(r) = E2*((1+e_max/E2)^r - 1)
    mask       outside the disc, depth reads background; alpha is reported but not trusted
    rays       samples inside the disc, against uniform sampling at s0 over the same field

With --compare, also reports the radiance difference between the two renders and their
timing ratio. Use it for CPU against OptiX: agreement says the custom camera computes the
same thing on both, and the timing ratio says whether the GPU path is real or a silent
fallback to CPU.

Needs numpy and OpenEXR (see requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import OpenEXR


def read_exr(path: str) -> dict[str, np.ndarray]:
    chans: dict[str, np.ndarray] = {}
    with OpenEXR.File(path) as f:
        for part in f.parts:
            for name, ch in part.channels.items():
                chans[name] = np.asarray(ch.pixels)
    return chans


def find(chans, suffix):
    for k, v in chans.items():
        if k.endswith(suffix):
            return v
    raise KeyError(f"no channel ending in {suffix!r}; have {sorted(chans)}")


def vec3(chans, stem):
    try:
        v = find(chans, stem)
        if v.ndim == 3 and v.shape[2] >= 3:
            return v[..., :3]
    except KeyError:
        pass
    return np.stack([find(chans, f"{stem}.{a}") for a in "XYZ"], axis=-1)


def load(folder: str):
    meta = json.load(open(os.path.join(folder, "meta.json")))
    ch = read_exr(os.path.join(folder, "fix.exr"))
    z = find(ch, "Depth.Z")
    z = z[..., 0] if z.ndim == 3 else z
    return meta, ch, z


def radius_grid(h: int, w: int) -> np.ndarray:
    j, i = np.meshgrid(np.arange(w), np.arange(h))
    return 2.0 * np.hypot((j + 0.5) / w - 0.5, (i + 0.5) / h - 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--compare", help="a second render folder, e.g. the CPU one")
    ap.add_argument("--depth-tol", type=float, default=1e-4, help="metres")
    ap.add_argument("--warp-tol", type=float, default=0.02, help="degrees, applied to the p99.9")
    args = ap.parse_args()

    meta, ch, z = load(args.folder)
    warp = meta["warp"]
    e2, emax, s0 = warp["E2_deg"], warp["e_max_deg"], warp["s0_deg"]
    pos = vec3(ch, "Position")
    comb = find(ch, "Combined")
    h, w = z.shape
    centre = np.array(meta["camera_position_m"])
    forward = np.array(meta["camera_forward"])
    r = radius_grid(h, w)
    hit = z < 1e9
    inside = r <= 1.0
    report: dict[str, object] = {"folder": args.folder, "device": meta["device"],
                                 "raster": [w, h], "spp": meta["spp"],
                                 "render_seconds": meta["render_seconds"]}
    fails = []

    m = hit & inside
    d_err = float(np.abs(z[m] - np.linalg.norm(pos[m] - centre, axis=-1)).max()) if m.any() else float("nan")
    report["depth_max_err_m"] = d_err
    if not (d_err <= args.depth_tol):
        fails.append(f"depth pass is not ray distance (max error {d_err:.3e} m)")

    direction = pos[m] - centre
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    e_measured = np.degrees(np.arccos(np.clip(direction @ forward, -1.0, 1.0)))
    e_formula = e2 * ((1.0 + emax / e2) ** r[m] - 1.0)
    # The max is sensitive to the Position pass averaging across a depth discontinuity inside
    # one pixel, which at the rim spans a whole degree. The p99.9 measures the mapping itself.
    err = np.abs(e_measured - e_formula)
    w_err = float(np.percentile(err, 99.9)) if m.any() else float("nan")
    report["warp_p999_err_deg"] = w_err
    report["warp_max_err_deg"] = float(err.max()) if m.any() else float("nan")
    report["e_at_rim_deg"] = float(e_measured.max()) if m.any() else float("nan")
    if not (w_err <= args.warp_tol):
        fails.append(f"warp does not match the formula (p99.9 error {w_err:.4f} deg)")

    outside = ~inside
    report["outside_depth_finite_frac"] = float((z[outside] < 1e9).mean()) if outside.any() else 0.0
    report["outside_alpha_max"] = float(comb[..., 3][outside].max()) if (outside.any() and comb.ndim == 3
                                                                        and comb.shape[2] > 3) else None
    if report["outside_depth_finite_frac"] > 0.01:
        fails.append("rays outside the disc are still hitting geometry; clipping is not working")

    n_inside = int(inside.sum())
    n_uniform = math.pi * (emax / s0) ** 2
    report["samples_inside_disc"] = n_inside
    report["uniform_at_s0_same_field"] = int(n_uniform)
    report["ray_saving"] = round(n_uniform / n_inside, 1)

    if args.compare:
        meta_b, ch_b, z_b = load(args.compare)
        comb_b = find(ch_b, "Combined")
        differing = [k for k in ("gaze_yaw_deg", "gaze_pitch_deg", "spp", "blend")
                     if meta.get(k) != meta_b.get(k)] + \
                    [k for k in meta["warp"] if meta["warp"][k] != meta_b["warp"][k]]
        if comb_b.shape != comb.shape:
            fails.append("compared renders have different raster sizes")
        elif differing:
            fails.append(f"compared renders differ in {differing}; comparison is meaningless")
        else:
            a3, b3 = comb[..., :3][inside], comb_b[..., :3][inside]
            denom = np.maximum(np.abs(b3).mean(-1), 1e-4)
            rel = np.abs(a3 - b3).mean(-1) / denom
            report["compare"] = {
                "other": args.compare, "other_device": meta_b["device"],
                "other_seconds": meta_b["render_seconds"],
                "speedup": round(meta_b["render_seconds"] / max(meta["render_seconds"], 1e-9), 2),
                "rel_diff_median": float(np.median(rel)), "rel_diff_p99": float(np.percentile(rel, 99)),
            }

    report["checks_failed"] = fails
    print(json.dumps(report, indent=1))
    with open(os.path.join(args.folder, "check.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    raise SystemExit(1 if fails else 0)


main()
