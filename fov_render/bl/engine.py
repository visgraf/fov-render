"""The one render call (report section 7), fov-3d-vision's render_fixation with its two lessons:

* With persistent data, Cycles re-reads the sample count only when the scene datablock is tagged:
  a change of `cycles.samples` alone was ignored across 16/256/4096 spp (measured 2026-09-13,
  fov-3d-vision). So the scene is tagged whenever spp or seed change.
* Frame evaluation can overwrite settings (the Classroom keyframes its seed), so after the render
  (samples, seed, adaptive, time_limit) are asserted to still hold.
"""
from __future__ import annotations

import os
import time

_LAST: dict = {}


def render_once(scene, spp: int, seed: int, out_path: str | None = None) -> float:
    """Render the active camera; with out_path the file is written inside the call. Returns the
    wall seconds of the render call."""
    import bpy
    c = scene.cycles
    key = (int(spp), int(seed))
    if _LAST.get(scene.name) != key or c.samples != spp or c.seed != seed:
        c.samples, c.seed = key
        scene.update_tag()
        _LAST[scene.name] = key
    if out_path:
        scene.render.filepath = os.path.abspath(out_path)
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=bool(out_path))
    dt = time.perf_counter() - t0
    got = (c.samples, c.seed, c.use_adaptive_sampling, c.time_limit)
    if got != (int(spp), int(seed), False, 0.0):
        raise RuntimeError(f"sampling settings did not hold: (samples, seed, adaptive, time_limit) = {got} "
                           f"after the render, expected ({spp}, {seed}, False, 0.0); is one animated?")
    if out_path and not os.path.isfile(scene.render.filepath):
        raise RuntimeError(f"render wrote no file at {scene.render.filepath}")
    return dt


def read_passes(path: str) -> dict:
    """Read an uncompressed multilayer EXR written with exr_codec NONE into pass arrays
    (combined (H,W,4), position (H,W,3), normal (H,W,3), object_index (H,W), z (H,W))."""
    import numpy as np
    from ..exr_lite import read_uncompressed_exr
    ch = read_uncompressed_exr(path)

    def one(suffix):
        keys = [k for k in ch if k.endswith(suffix)]
        if len(keys) > 1:
            raise RuntimeError(f"several EXR channels end with {suffix!r}: {keys}")
        return np.asarray(ch[keys[0]]) if keys else None

    out = {}
    comb = [one(f"Combined.{c}") for c in "RGBA"]
    if comb[0] is not None:
        out["combined"] = np.stack([x if x is not None else np.ones_like(comb[0]) for x in comb], axis=-1)
    for name, stem, comps in (("position", "Position", "XYZ"), ("normal", "Normal", "XYZ")):
        parts = [one(f"{stem}.{c}") for c in comps]
        if all(p is not None for p in parts):
            out[name] = np.stack(parts, axis=-1)
    oi = one("Object Index.X") if one("Object Index.X") is not None else one("IndexOB.X")
    if oi is not None:
        out["object_index"] = oi
    z = one("Depth.Z")
    if z is not None:
        out["z"] = z
    return out
