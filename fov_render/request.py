"""The JSON requests between a Session and the request handler (report section 8.6).

numpy only: the host builds and validates requests, the Blender side validates them again and
answers. Two schemas:

    fov-render-session-v1   once per session: the scene, the RenderConfig, the rig, access,
                            the head, the Object Index numbering
    fov-render-request-v1   once per look: the sensor and its parameters, the eyes, the gaze,
                            optional rig override, spp, seeds, the output folder, what to write

Every field is checked and a malformed request is refused with all its reasons at once
(RequestError), before anything is rendered. JSON keeps float repr precision, so a request
round-trips bit for bit.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from . import sensors as S
from .rig import ROLLS, Gaze, HeadPose, Rig

SESSION_SCHEMA = "fov-render-session-v1"
REQUEST_SCHEMA = "fov-render-request-v1"
RESPONSE_SCHEMA = "fov-render-response-v1"
EYES = ("L", "R", "C")
ACCESS = ("observation", "oracle")
SENSOR_TYPES = {"foveated": S.Foveated, "perspective": S.Perspective, "panorama": S.Panorama,
                "reference_view": S.ReferenceView}
GAZE_MODES = ("yaw_pitch", "verged_point", "per_eye", "fixed")
OUTPUTS = ("exr", "records")


class RequestError(ValueError):
    """A request or session description that fails validation; .reasons lists every problem."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


# ----------------------------------------------------------------------------- small checks

def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _vec(x, n) -> bool:
    return isinstance(x, (list, tuple)) and len(x) == n and all(_num(v) for v in x)


def _unknown(d: dict, allowed, where: str, out: list) -> None:
    extra = sorted(set(d) - set(allowed))
    if extra:
        out.append(f"{where}: unknown field(s) {extra}")


# ----------------------------------------------------------------------------- sensors and gaze

def sensor_to_dict(spec) -> dict:
    for name, cls in SENSOR_TYPES.items():
        if isinstance(spec, cls):
            return {"type": name, **dataclasses.asdict(spec)}
    raise TypeError(f"not a sensor spec: {spec!r}")


def sensor_from_dict(d: dict):
    reasons = check_sensor(d)
    if reasons:
        raise RequestError(reasons)
    kw = {k: v for k, v in d.items() if k != "type"}
    return SENSOR_TYPES[d["type"]](**kw)


def check_sensor(d) -> list[str]:
    out = []
    if not isinstance(d, dict):
        return ["sensor: must be an object"]
    t = d.get("type")
    if t not in SENSOR_TYPES:
        return [f"sensor.type: {t!r} is not one of {sorted(SENSOR_TYPES)}"]
    fields = {f.name: f for f in dataclasses.fields(SENSOR_TYPES[t])}
    _unknown(d, set(fields) | {"type"}, f"sensor ({t})", out)
    for k, v in d.items():
        if k == "type" or k not in fields or v is None:
            continue
        if k in ("n", "core_px", "width", "height") and not (_int(v) and v > 0):
            out.append(f"sensor.{k}: must be a positive integer, got {v!r}")
        elif k == "preset" and v not in ("preview360", "breadth1"):
            out.append(f"sensor.preset: {v!r} is not preview360 or breadth1")
        elif k == "backface" and not isinstance(v, bool):
            out.append("sensor.backface: must be true or false")
        elif k not in ("n", "core_px", "width", "height", "preset", "backface") and not _num(v):
            out.append(f"sensor.{k}: must be a finite number, got {v!r}")
        elif k in ("E2_deg", "e_max_deg", "s0_deg", "core_fov_deg", "half_fov_x_deg") and v <= 0:
            out.append(f"sensor.{k}: must be positive, got {v!r}")
    if t == "foveated" and _num(d.get("e_max_deg")) and d["e_max_deg"] >= 90:
        out.append("sensor.e_max_deg: the foveated cone must stay below 90 degrees")
    if t == "panorama" and _int(d.get("width")) and d["width"] % 2:
        out.append("sensor.width: a panorama width must be even")
    return out


def gaze_to_dict(g: Gaze | None) -> dict:
    if g is None:
        return {"mode": "fixed"}
    if g.mode == "yaw_pitch":
        return {"mode": g.mode, "yaw_deg": g.yaw_deg, "pitch_deg": g.pitch_deg, "vergence_m": g.vergence_m}
    if g.mode == "verged_point":
        return {"mode": g.mode, "point_w": [float(x) for x in g.point_w]}
    if g.mode == "per_eye":
        return {"mode": g.mode, "angles": {k: [float(a), float(b)] for k, (a, b) in g.angles.items()}}
    raise ValueError(f"unknown gaze mode {g.mode!r}")


def gaze_from_dict(d: dict) -> Gaze | None:
    reasons = check_gaze(d)
    if reasons:
        raise RequestError(reasons)
    m = d["mode"]
    if m == "fixed":
        return None
    if m == "yaw_pitch":
        return Gaze.yaw_pitch(d["yaw_deg"], d["pitch_deg"], d.get("vergence_m"))
    if m == "verged_point":
        return Gaze.verged_point(d["point_w"])
    return Gaze.per_eye(**{k: tuple(v) for k, v in d["angles"].items()})


def check_gaze(d) -> list[str]:
    out = []
    if not isinstance(d, dict):
        return ["gaze: must be an object"]
    m = d.get("mode")
    if m not in GAZE_MODES:
        return [f"gaze.mode: {m!r} is not one of {list(GAZE_MODES)}"]
    allowed = {"yaw_pitch": ("yaw_deg", "pitch_deg", "vergence_m"), "verged_point": ("point_w",),
               "per_eye": ("angles",), "fixed": ("yaw_offset_deg",)}[m]
    _unknown(d, set(allowed) | {"mode"}, f"gaze ({m})", out)
    if m == "yaw_pitch":
        for k in ("yaw_deg", "pitch_deg"):
            if not _num(d.get(k)):
                out.append(f"gaze.{k}: must be a finite number")
        if _num(d.get("pitch_deg")) and abs(d["pitch_deg"]) > 90:
            out.append("gaze.pitch_deg: must lie in [-90, 90]")
        v = d.get("vergence_m")
        if v is not None and not (_num(v) and v > 0):
            out.append("gaze.vergence_m: must be positive or null")
    elif m == "verged_point":
        if not _vec(d.get("point_w"), 3):
            out.append("gaze.point_w: must be three finite numbers (world metres)")
    elif m == "per_eye":
        a = d.get("angles")
        if not isinstance(a, dict) or not a:
            out.append("gaze.angles: must map eyes to [yaw_deg, pitch_deg]")
        else:
            for k, v in a.items():
                if k not in EYES or not _vec(v, 2):
                    out.append(f"gaze.angles.{k}: must be an eye in {list(EYES)} with [yaw_deg, pitch_deg]")
    elif m == "fixed" and "yaw_offset_deg" in d and not _num(d["yaw_offset_deg"]):
        out.append("gaze.yaw_offset_deg: must be a finite number")
    return out


# ----------------------------------------------------------------------------- the request

def make_request(n: int, sensor, eyes, gaze: Gaze | None, out_dir, *, spp=None, seeds=None,
                 rig: Rig | None = None, outputs=("exr",), yaw_offset_deg: float | None = None) -> dict:
    g = gaze_to_dict(gaze)
    if gaze is None and yaw_offset_deg is not None:
        g["yaw_offset_deg"] = float(yaw_offset_deg)
    req = {"schema": REQUEST_SCHEMA, "n": int(n), "kind": "look", "sensor": sensor_to_dict(sensor),
           "eyes": list(eyes), "gaze": g, "spp": spp, "seeds": None if seeds is None else dict(seeds),
           "rig": None if rig is None else {"ipd_m": rig.ipd_m, "roll": rig.roll}, "out_dir": str(out_dir),
           "outputs": list(outputs)}
    validate_request(req)
    return req


def validate_request(req) -> dict:
    """Raise RequestError with every reason the request is malformed; return it unchanged if not."""
    out = []
    if not isinstance(req, dict):
        raise RequestError(["request: must be a JSON object"])
    _unknown(req, ("schema", "n", "kind", "sensor", "eyes", "gaze", "spp", "seeds", "rig", "out_dir", "outputs"),
             "request", out)
    if req.get("schema") != REQUEST_SCHEMA:
        out.append(f"schema: expected {REQUEST_SCHEMA!r}, got {req.get('schema')!r}")
    if not (_int(req.get("n")) and req["n"] >= 1):
        out.append("n: must be an integer >= 1 (the request's sequence number)")
    if req.get("kind") != "look":
        out.append(f"kind: only 'look' is supported, got {req.get('kind')!r}")
    out += check_sensor(req.get("sensor"))
    eyes = req.get("eyes")
    if not (isinstance(eyes, list) and eyes and all(e in EYES for e in eyes) and len(set(eyes)) == len(eyes)):
        out.append(f"eyes: must be a non-empty list of distinct eyes from {list(EYES)}, got {eyes!r}")
    out += check_gaze(req.get("gaze"))
    g = req.get("gaze") if isinstance(req.get("gaze"), dict) else {}
    st = req.get("sensor", {}).get("type") if isinstance(req.get("sensor"), dict) else None
    if g.get("mode") == "fixed" and st in ("foveated", "perspective"):
        out.append(f"gaze: a {st} look needs a gaze (yaw_pitch, verged_point or per_eye), not 'fixed'")
    if g.get("mode") == "per_eye" and isinstance(eyes, list) and isinstance(g.get("angles"), dict):
        missing = [e for e in eyes if e not in g["angles"]]
        if missing:
            out.append(f"gaze.angles: no angles for eye(s) {missing}")
    spp = req.get("spp")
    if spp is not None and not (_int(spp) and spp >= 1):
        out.append(f"spp: must be a positive integer or null, got {spp!r}")
    seeds = req.get("seeds")
    if seeds is not None:
        if not isinstance(seeds, dict):
            out.append("seeds: must map eyes to integer seeds, or be null")
        else:
            for k, v in seeds.items():
                if k not in EYES or not (_int(v) and 0 <= v < 2 ** 31):
                    out.append(f"seeds.{k}: must be an eye with an integer seed in [0, 2^31)")
            if isinstance(eyes, list):
                miss = [e for e in eyes if e not in seeds]
                if miss:
                    out.append(f"seeds: no seed for eye(s) {miss}")
    rig = req.get("rig")
    if rig is not None:
        if not isinstance(rig, dict):
            out.append("rig: must be an object or null")
        else:
            _unknown(rig, ("ipd_m", "roll"), "rig", out)
            if "ipd_m" in rig and not (_num(rig["ipd_m"]) and rig["ipd_m"] > 0):
                out.append("rig.ipd_m: must be positive")
            if "roll" in rig and rig["roll"] not in ROLLS:
                out.append(f"rig.roll: {rig['roll']!r} is not one of {list(ROLLS)}")
    if not (isinstance(req.get("out_dir"), str) and req["out_dir"]):
        out.append("out_dir: must be a non-empty path")
    outs = req.get("outputs")
    if not (isinstance(outs, list) and outs and all(o in OUTPUTS for o in outs)):
        out.append(f"outputs: must be a non-empty list from {list(OUTPUTS)}, got {outs!r}")
    if out:
        raise RequestError(out)
    return req


# ----------------------------------------------------------------------------- the session

def make_session(blend, cfg, rig: Rig, *, access="observation", head: HeadPose | None = None,
                 instance_ids="legacy", prepare=None, formats=()) -> dict:
    """prepare: a sensor to set up when the session opens, so the ready answer carries its
    settings (as a legacy server that configures its camera at start). formats: extra readbacks
    the handler also reports, e.g. ("f3d_ab1a",) (fov_render.compat)."""
    d = {"schema": SESSION_SCHEMA, "blend": None if blend is None else str(Path(blend).resolve()),
         "config": cfg.as_dict(), "rig": rig.as_dict(), "access": access,
         "head": None if head is None else head.as_dict(), "instance_ids": instance_ids,
         "prepare": None if prepare is None else sensor_to_dict(prepare), "formats": list(formats)}
    validate_session(d)
    return d


def validate_session(d) -> dict:
    out = []
    if not isinstance(d, dict):
        raise RequestError(["session: must be a JSON object"])
    _unknown(d, ("schema", "blend", "config", "rig", "access", "head", "instance_ids", "prepare", "formats"),
             "session", out)
    if d.get("schema") != SESSION_SCHEMA:
        out.append(f"schema: expected {SESSION_SCHEMA!r}, got {d.get('schema')!r}")
    if d.get("blend") is not None and not isinstance(d["blend"], str):
        out.append("blend: must be a path or null (the scene already loaded)")
    if not isinstance(d.get("config"), dict):
        out.append("config: must be a RenderConfig as a JSON object")
    r = d.get("rig")
    if not (isinstance(r, dict) and _num(r.get("ipd_m")) and r["ipd_m"] > 0 and r.get("roll") in ROLLS):
        out.append(f"rig: needs a positive ipd_m and a roll in {list(ROLLS)}")
    if d.get("access") not in ACCESS:
        out.append(f"access: {d.get('access')!r} is not one of {list(ACCESS)}")
    h = d.get("head")
    if h is not None and not (isinstance(h, dict) and isinstance(h.get("head_R_wh"), list)
                              and len(h["head_R_wh"]) == 3 and all(_vec(row, 3) for row in h["head_R_wh"])
                              and _vec(h.get("head_origin_w_m"), 3)):
        out.append("head: must be null (the EYE object) or {head_R_wh: 3x3, head_origin_w_m: 3}")
    if d.get("instance_ids") not in ("legacy", "complete"):
        out.append("instance_ids: must be 'legacy' or 'complete'")
    if d.get("prepare") is not None:
        out += check_sensor(d["prepare"])
    from .compat import FORMATS
    if not (isinstance(d.get("formats"), list) and all(x in FORMATS for x in d["formats"])):
        out.append(f"formats: must be a list of names from {sorted(FORMATS)}")
    if out:
        raise RequestError(out)
    return d


def dumps(d: dict) -> str:
    return json.dumps(d, indent=1, sort_keys=True, allow_nan=False) + "\n"
