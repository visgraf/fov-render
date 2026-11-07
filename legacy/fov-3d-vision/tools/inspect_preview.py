"""Inspect a preview360 folder: completeness + scale report and a contact sheet.

    python tools/inspect_preview.py previews/<scene_id> [--hdri source.exr]

Needs: numpy, OpenEXR (>= 3.3), pillow.

Report (report.json, also printed):
    hole_fraction      solid-angle fraction where rays escape to the background
    backface_fraction  solid-angle fraction of hits on back sides of surfaces
    depth_*            ray-distance statistics (m); nadir/zenith are along the EYE's down/up axis,
                       so they equal eye height / ceiling clearance only when the EYE is level
Contact sheet (sheet.png): RGB with eccentricity rings (10/20/40 deg) | log depth
                           normals | problems (holes red, backfaces magenta)
--hdri: for tier-1 scenes, compares the render with the source panorama (column shift + error).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import OpenEXR
from PIL import Image


def read_exr(path: str) -> dict[str, np.ndarray]:
    """All channels from all parts (Blender 5.2 may write one part per pass)."""
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


def pixel_directions(h, w):
    """EYE-frame unit directions for equirect pixel centres (x right, y up, -z forward)."""
    v, u = np.mgrid[0:h, 0:w]
    lon = ((u + 0.5) / w - 0.5) * 2 * np.pi
    lat = (0.5 - (v + 0.5) / h) * np.pi
    return np.stack([np.sin(lon) * np.cos(lat), np.sin(lat), -np.cos(lon) * np.cos(lat)], axis=-1), lat


def colormap(x):
    """0..1 -> RGB, dark blue -> cyan -> yellow -> red (no matplotlib needed)."""
    anchors = np.array([[0.05, 0.05, 0.3], [0.0, 0.6, 0.9], [0.95, 0.9, 0.1], [0.8, 0.1, 0.05]])
    t = np.clip(x, 0, 1) * (len(anchors) - 1)
    i = np.minimum(t.astype(int), len(anchors) - 2)
    f = (t - i)[..., None]
    return anchors[i] * (1 - f) + anchors[i + 1] * f


def to_u8(rgb):
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def hdri_check(render_rgb, hdri_path):
    src = read_exr(hdri_path)
    rgb = None
    for key in ("RGB", "RGBA"):
        if key in src:
            rgb = src[key][..., :3]
    if rgb is None:
        rgb = np.stack([find(src, c) for c in ("R", "G", "B")], axis=-1)
    h, w = render_rgb.shape[:2]
    fy, fx = rgb.shape[0] / h, rgb.shape[1] / w
    if abs(fx - round(fx)) < 1e-9 and abs(fy - round(fy)) < 1e-9:
        k = int(round(fx))
        small = rgb[: h * k, : w * k].reshape(h, k, w, k, 3).mean(axis=(1, 3))
    else:  # nearest sample at pixel centres
        ys = ((np.arange(h) + 0.5) * fy).astype(int)
        xs = ((np.arange(w) + 0.5) * fx).astype(int)
        small = rgb[ys][:, xs]
    lum_r = render_rgb.mean(axis=(0, 2))
    lum_s = small.mean(axis=(0, 2))
    a, b = lum_r - lum_r.mean(), lum_s - lum_s.mean()
    corr = np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))
    shift = int(np.argmax(corr))
    shift = shift - w if shift > w // 2 else shift
    rel = np.abs(render_rgb - small).mean(-1) / np.maximum(small.mean(-1), 1e-4)
    return {"column_shift_px": shift, "rel_err_median": float(np.median(rel)),
            "rel_err_p99": float(np.percentile(rel, 99)), "source_shape": list(rgb.shape[:2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--hdri", help="source panorama (EXR) for tier-1 identity check")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.folder, "meta.json")))
    ch = read_exr(os.path.join(args.folder, "pano.exr"))
    z = find(ch, "Depth.Z")
    if z.ndim == 3:
        z = z[..., 0]
    normal = vec3(ch, "Normal")
    h, w = z.shape
    dirs, lat = pixel_directions(h, w)
    weight = np.cos(lat)
    weight /= weight.sum()

    hit = z < 1e9
    report = {"scene": meta.get("blend"), "kind": meta.get("fov_kind"), "eye_note": meta.get("eye_note"),
              "eye_position_m": meta["eye"]["position_m"], "size": [w, h],
              "hole_fraction": float(weight[~hit].sum())}
    if meta.get("fov_kind") == "hdri":
        report["note"] = "holes expected: the whole scene is environment at infinity"
    backface = np.zeros_like(hit)
    bf_path = os.path.join(args.folder, "backface.exr")
    if os.path.exists(bf_path):
        bf = find(read_exr(bf_path), "Combined")
        bf = bf[..., 0] if bf.ndim == 3 else bf
        backface = hit & (bf > 0.5)
        report["backface_fraction"] = float(weight[backface].sum())
    if hit.any():
        zh = z[hit]
        rows = max(1, h // 100)
        nadir, zenith = z[-rows:][hit[-rows:]], z[:rows][hit[:rows]]
        report.update({
            "depth_min_m": float(zh.min()), "depth_p01_m": float(np.percentile(zh, 1)),
            "depth_median_m": float(np.median(zh)), "depth_max_m": float(zh.max()),
            "nadir_depth_m": float(np.median(nadir)) if nadir.size else None,
            "zenith_depth_m": float(np.median(zenith)) if zenith.size else None,
        })

    rgb_png = np.asarray(Image.open(os.path.join(args.folder, "pano.png")).convert("RGB")) / 255.0
    ecc = np.degrees(np.arccos(np.clip(-dirs[..., 2], -1, 1)))
    ring_px = np.zeros((h, w), bool)
    for e in (10.0, 20.0, 40.0):
        ring_px |= np.abs(ecc - e) < 0.8 * 360.0 / w
    panel_rgb = np.where(ring_px[..., None], [1.0, 0.85, 0.0], rgb_png)

    panel_depth = np.zeros((h, w, 3))
    if hit.any():
        lz = np.log(np.maximum(z, 1e-3))
        lo, hi = np.percentile(lz[hit], [1, 99])
        panel_depth[hit] = colormap((lz[hit] - lo) / max(hi - lo, 1e-6))
    panel_normal = np.where(hit[..., None], np.abs(normal), 0.0)
    gray = rgb_png.mean(-1, keepdims=True) * 0.6
    panel_prob = np.repeat(gray, 3, axis=-1)
    panel_prob[~hit] = [1.0, 0.1, 0.1]
    panel_prob[backface] = [1.0, 0.1, 1.0]
    sheet = np.concatenate([np.concatenate([panel_rgb, panel_depth], 1),
                            np.concatenate([panel_normal, panel_prob], 1)], 0)
    Image.fromarray(to_u8(sheet)).save(os.path.join(args.folder, "sheet.png"))

    if args.hdri:
        comb = find(ch, "Combined")[..., :3]
        report["hdri_check"] = hdri_check(comb, args.hdri)

    with open(os.path.join(args.folder, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))


main()
