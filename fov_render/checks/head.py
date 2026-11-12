"""Step "head": the head moves between looks (issue #20, PLAN D19).

numpy only:
    head.1  requests with a head round-trip bit for bit; a malformed head is refused with its reason;
            a request without a head is byte for byte what main built before #20 (known answers)
    head.2  HeadPose.moved / translated against their definitions; the rig places and aims the eyes
            of a moved head
    head.3  Session.move_head fills in exactly the requests it should (client side, no Blender)
needs bpy (calibration room, CPU):
    head.4  a look whose head is the session's renders bit for bit as a look without a head
    head.5  from moved heads, every sensor's record sits on the requested head: origins, the sample
            nearest the expected gaze, and agreement with Position. Control: a handler that ignores
            the request's head fails it
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

from ..rig import Gaze, HeadPose, Rig, gaze_direction, rot_x, rot_y, rot_z, turn_rotation
from ..session import _Base
from .common import Result

STEP = "head"
POINT = [0.4, 2.1, 1.1]        # the world point cli.2 verges on, in the calibration room
VERGENCE_M = 2.1

# sha256 of step 5.1's four good requests, as main built them at aee89ff (before #20): Chat sandbox
KNOWN_NO_HEAD_SHA256 = ["c5111c2f55b8576b903573005d96006e15aee9be6b3a99e89df75ea524f27189",
                        "46d36fd2dff622c9d64246a190614380aca9231bcc4bdce6c7b076579b4c48c0",
                        "402af115a6fad5bb0b0cd3d617f28e2a25f596065d48c42672bf5bd32b3f4d7f",
                        "da191e40d0979236723f14a7691af141ce6471afe14a4b7c1a14e8109404fa4d"]


def _base_head() -> HeadPose:
    """An arbitrary head: turned about world Z and tilted, away from the origin."""
    return HeadPose(rot_z(37.0) @ rot_x(80.0) @ rot_y(-12.0), np.array([0.3, -1.2, 1.55]))


def _angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b)))


# ----------------------------------------------------------------------------- head.1

def request_head(_ctx=None) -> Result:
    from .. import request as Q, sensors as S
    good = [Q.make_request(1, S.Perspective(), ["L", "R"], Gaze.yaw_pitch(10.123456789012345, -3.25, 2.1), "/tmp/a",
                           seeds={"L": 2111, "R": 2112}, spp=64),
            Q.make_request(2, S.Foveated(E2_deg=2.0, e_max_deg=45.0), ["C"], Gaze.per_eye(C=(20.0, -5.0)), "/tmp/b",
                           outputs=("exr", "records")),
            Q.make_request(3, S.Panorama(preset="breadth1"), ["L"], None, "/tmp/c", yaw_offset_deg=0.1),
            Q.make_request(4, S.ReferenceView(64, 48, 27.0), ["R"], Gaze.verged_point([0.4, 2.1, 1.1]), "/tmp/d",
                           rig=Rig(0.07, "baseline_projected"))]
    sha = [hashlib.sha256(Q.dumps(r).encode()).hexdigest() for r in good]
    no_head_same = sha == KNOWN_NO_HEAD_SHA256 and all("head" not in r for r in good)
    h = _base_head().moved((0.1, -0.02, -0.3), 23.123456789012345, -7.5, 3.0)
    with_head = Q.make_request(5, S.Foveated(), ["L", "R"], Gaze.yaw_pitch(5.0, 0.0, 2.1), "/tmp/e", head=h)
    back = json.loads(Q.dumps(with_head))
    hb = HeadPose.from_dict(back["head"])
    round_trip = (back == with_head and np.array_equal(hb.R_wh, h.R_wh) and np.array_equal(hb.origin_w, h.origin_w)
                  and bool(Q.validate_request(back)))
    hd = h.as_dict()
    refl = (np.asarray(hd["head_R_wh"]) @ np.diag([1.0, 1.0, -1.0])).tolist()
    sheared = (np.asarray(hd["head_R_wh"]) + 1e-4 * np.eye(3)).tolist()
    bad = {
        "not an object": ([1, 2, 3], "head:"),
        "2x3 rotation": (dict(hd, head_R_wh=hd["head_R_wh"][:2]), "head.head_R_wh"),
        "NaN in the rotation": (dict(hd, head_R_wh=[[float("nan"), 0, 0], [0, 1, 0], [0, 0, 1]]), "head.head_R_wh"),
        "not orthonormal": (dict(hd, head_R_wh=sheared), "proper rotation"),
        "a reflection": (dict(hd, head_R_wh=refl), "proper rotation"),
        "origin of two numbers": (dict(hd, head_origin_w_m=[0.0, 1.0]), "head.head_origin_w_m"),
        "infinite origin": (dict(hd, head_origin_w_m=[0.0, float("inf"), 1.0]), "head.head_origin_w_m"),
        "unknown field": (dict(hd, head_up=[0, 0, 1]), "unknown field"),
    }
    refused = {}
    for label, (head, needle) in bad.items():
        try:
            Q.validate_request(dict(with_head, head=head))
            refused[label] = "ACCEPTED"
        except Q.RequestError as e:
            refused[label] = "ok" if any(needle in x for x in e.reasons) else f"wrong reason: {e.reasons}"
    # control: a good head, and an explicit null head (the session's), are accepted
    try:
        Q.validate_request(with_head)
        Q.validate_request(dict(with_head, head=None))
        accepts_good = True
    except Q.RequestError:
        accepts_good = False
    ok = no_head_same and round_trip and all(v == "ok" for v in refused.values())
    return Result(STEP, "head.1 requests with a head round-trip; malformed heads are refused; no head = main's bytes",
                  ok, accepts_good, {"no_head_sha256_as_main": no_head_same, "sha256": sha, "round_trip": round_trip,
                                     "refused": refused, "control_good_head_accepted": accepts_good})


# ----------------------------------------------------------------------------- head.2

def _moved_ok(turn) -> dict:
    """The definitions of moved(), with `turn` as the head-frame turn (the control swaps it)."""
    h0 = _base_head()
    t = np.array([0.12, -0.05, -0.4])
    out = {}
    m = h0.moved(t)
    out["translate: origin = to_world(t), rotation kept"] = float(np.max(np.abs(m.origin_w - h0.to_world(t)))) \
        + float(np.max(np.abs(m.R_wh - h0.R_wh)))
    for yaw, pitch in ((25.0, 0.0), (0.0, -15.0), (-40.0, 12.0)):
        m = HeadPose(h0.R_wh @ turn(yaw, pitch, 0.0), h0.origin_w)
        fwd = m.dir_to_world([0.0, 0.0, -1.0])
        out[f"yaw {yaw}, pitch {pitch}: forward = old head's gaze_direction"] = \
            _angle_deg(fwd, h0.dir_to_world(gaze_direction(yaw, pitch)))
    m = HeadPose(h0.R_wh @ turn(0.0, 0.0, 10.0), h0.origin_w)
    out["roll 10: up tips toward +X (right ear down)"] = _angle_deg(m.dir_to_world([0.0, 1.0, 0.0]),
                                                                    h0.dir_to_world([math.sin(math.radians(10.0)),
                                                                                     math.cos(math.radians(10.0)), 0.0]))
    return out


def head_moves(_ctx=None) -> Result:
    tol_m, tol_deg = 1e-12, 1e-9
    defs = _moved_ok(turn_rotation)
    ok_defs = all(v <= (tol_m if "translate" in k else tol_deg) for k, v in defs.items())
    h0 = _base_head()
    back = h0.moved((0.0, 0.0, 0.0), 33.0).moved((0.0, 0.0, 0.0), -33.0)
    there = h0.moved((0.2, 0.1, -0.3)).moved((-0.2, -0.1, 0.3))
    trans = h0.translated([0.5, -0.25, 0.1])
    inverse = {"turn and back: rotation": float(np.max(np.abs(back.R_wh - h0.R_wh))),
               "move and back: origin": float(np.max(np.abs(there.origin_w - h0.origin_w))),
               "translated: origin": float(np.max(np.abs(trans.origin_w - (h0.origin_w + [0.5, -0.25, 0.1])))),
               "translated: rotation kept": float(np.max(np.abs(trans.R_wh - h0.R_wh)))}
    ok_inv = all(v <= 1e-13 for v in inverse.values())
    # the rig with a moved head: eyes on the moved head; verged_point aims every eye at the world point;
    # yaw_pitch turns with the head
    rig = Rig(roll="baseline_projected")
    hm = h0.moved((0.3, 0.0, -0.5), 25.0, -10.0, 5.0)
    P = hm.to_world([0.2, 0.1, -2.0])
    aim = {}
    for eye in ("L", "R", "C"):
        p = rig.pose(hm, Gaze.verged_point(P), eye)
        aim[f"{eye} centre"] = float(np.max(np.abs(p.centre_w - hm.to_world(rig.centre_h(eye)))))
        aim[f"{eye} verged on P (deg)"] = _angle_deg(p.R_wb @ [0.0, 0.0, -1.0], P - p.centre_w)
    pc = rig.pose(hm, Gaze.yaw_pitch(10.0, -5.0), "C")
    aim["C yaw_pitch turns with the head (deg)"] = _angle_deg(pc.R_wb @ [0.0, 0.0, -1.0],
                                                             hm.dir_to_world(gaze_direction(10.0, -5.0)))
    pf = rig.fixed_pose(hm, "L")
    aim["fixed pose: camera axes = head axes"] = float(np.max(np.abs(pf.R_wb - hm.R_wh)))
    ok_aim = all(v <= (1e-9 if "deg" in k else 1e-12) for k, v in aim.items())
    # control: the yaw sign flipped (Ry(+yaw)) breaks the definitions
    ctrl = _moved_ok(lambda y, p, r: rot_y(y) @ rot_x(p) @ rot_z(-r))
    ctrl_fails = not all(v <= (tol_m if "translate" in k else tol_deg) for k, v in ctrl.items())
    return Result(STEP, "head.2 HeadPose.moved/translated as defined; the rig places and aims a moved head's eyes",
                  ok_defs and ok_inv and ok_aim, ctrl_fails,
                  {"definitions": defs, "inverses": inverse, "rig": aim, "control_yaw_sign_flipped": ctrl})


# ----------------------------------------------------------------------------- head.3

class _Recorder(_Base):
    """A Session without Blender: it records the requests _Base.look builds."""

    def __init__(self, head: HeadPose):
        self.ready, self.n, self.sent = {"head": head.as_dict()}, 0, []
        self.workdir = Path("/tmp/fov-render-head3")

    def request(self, req):
        self.n = req["n"]
        self.sent.append(req)
        return {"out_dir": req["out_dir"], "eyes": {}}

    def close(self):
        return 0


def move_head_requests(_ctx=None) -> Result:
    from .. import sensors as S
    from ..request import validate_request
    h0 = _base_head()
    s = _Recorder(h0)
    g = Gaze.yaw_pitch(5.0, 0.0, VERGENCE_M)
    other = h0.translated([1.0, 0.0, 0.0])
    s.look(S.Foveated(), ("L", "R"), g)                                   # 1: no move
    s.move_head(translate_h=(0.0, 0.0, -0.1))
    s.look(S.Foveated(), ("L", "R"), g)                                   # 2: moved
    s.move_head(yaw_deg=15.0)
    s.look(S.Foveated(), ("L", "R"), g)                                   # 3: moved again, from the current head
    s.look(S.Foveated(), ("L", "R"), g, head=other)                       # 4: explicit head wins
    s.move_head(other)
    s.look(S.Panorama(width=64), ("C",), None)                            # 5: an absolute move
    s.reset_head()
    s.look(S.Foveated(), ("L", "R"), g)                                   # 6: back to the session's
    try:
        s.move_head(other, yaw_deg=3.0)
        both = "ACCEPTED"
    except ValueError:
        both = "refused"
    want = [None, h0.moved((0, 0, -0.1)), h0.moved((0, 0, -0.1)).moved(yaw_deg=15.0), other, other, None]
    got = [r.get("head") for r in s.sent]
    cases = {}
    for k, (w, r) in enumerate(zip(want, s.sent)):
        if w is None:
            cases[k + 1] = "ok" if "head" not in r else f"has a head: {r['head']}"
        else:
            cases[k + 1] = "ok" if r.get("head") == w.as_dict() else "wrong head"
        validate_request(r)
    ok = len(s.sent) == 6 and all(v == "ok" for v in cases.values()) and both == "refused" \
        and s.head.as_dict() == h0.as_dict()
    # control: composing the second move from the session's head (not the current one) is told apart
    wrong = h0.moved(yaw_deg=15.0).as_dict()
    return Result(STEP, "head.3 Session.move_head fills in exactly the requests it should (client side)", ok,
                  got[2] != wrong, {"cases": cases, "head_and_move_together": both, "requests": len(s.sent)})


# ----------------------------------------------------------------------------- bpy gates

def _session(ctx, work: Path, ignore_head: bool = False):
    from .. import Session
    from ..bl.config import RenderConfig
    from .step1 import calib_room
    s = Session.in_process(RenderConfig(profile="small", device=ctx["device"]), Rig(roll="baseline_projected"),
                           blend=calib_room(ctx), access="oracle", workdir=work)
    if ignore_head:      # the control: a handler that never reads the request's head
        s.handler.look_head = lambda req: s.handler.head
    return s


def same_head_same_look(ctx) -> Result:
    from .. import sensors as S
    from .cli import _arrays, _same
    work = Path(ctx["work"]) / "head" / "4"
    shutil.rmtree(work, ignore_errors=True)
    s = _session(ctx, work / "q")
    g, seeds = Gaze.verged_point(POINT), {"L": 11, "R": 12}
    kw = dict(spp=4, seeds=seeds, outputs=("exr", "records"))
    a = s.look(S.Foveated(), ("L", "R"), g, out_dir=work / "none", **kw)
    b = s.look(S.Foveated(), ("L", "R"), g, out_dir=work / "same", head=s.session_head, **kw)
    c = s.look(S.Foveated(), ("L", "R"), g, out_dir=work / "1mm", head=s.session_head.translated([0.001, 0, 0]), **kw)
    A, B, C = _arrays(work / "none"), _arrays(work / "same"), _arrays(work / "1mm")
    # the records' headers hold look_id, the request's sequence number (1 and 2 here): compared without it
    looks_ids = {}
    for arrays, tag in ((A, "none"), (B, "same")):
        for k in [k for k in arrays if k.endswith(":__header__")]:
            hd = json.loads(bytes(arrays[k]).decode())
            looks_ids.setdefault(tag, set()).add(hd.pop("look_id"))
            arrays[k] = np.frombuffer(json.dumps(hd, sort_keys=True).encode(), dtype=np.uint8)
    same, diff = _same(A, B)
    sources = (a.response["head_source"], b.response["head_source"])
    ok = same and "head" not in a.request and sources == ("session", "request") \
        and a.response["head"] == b.response["head"]
    key = "raw_L:ViewLayer.Position.X"
    ctrl = bool(key in A and key in C and A[key].tobytes() != C[key].tobytes())
    return Result(STEP, "head.4 a look with the session's head as its own renders bit for bit as one without a head",
                  ok, ctrl, {"arrays": len(A), "differ": diff[:20], "head_source": sources,
                             "look_ids": {k: sorted(v) for k, v in looks_ids.items()},
                             "control_1mm_position_differs": ctrl})


def _nearest_deg(dirs, d) -> float:
    """The smallest angle (deg) between unit directions dirs (K, 3) and the unit direction d."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    dirs = np.asarray(dirs, float)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    return float(np.degrees(np.min(np.arctan2(np.linalg.norm(np.cross(dirs, d), axis=-1), dirs @ d))))


def _looks(s, heads):
    from .. import sensors as S
    out = []
    for k, h in enumerate(heads):
        for name, spec, eyes, gaze in (("foveated", S.Foveated(), ("L", "R"), Gaze.yaw_pitch(10.0, -5.0, VERGENCE_M)),
                                       ("perspective", S.Perspective(), ("L", "R"), Gaze.verged_point(POINT)),
                                       ("panorama", S.Panorama(width=256), ("C",), None)):
            out.append((k, name, spec, h, gaze, s.look(spec, eyes, gaze, head=h, spp=2, outputs=("exr", "records"),
                                                       out_dir=s.workdir.parent / f"h{k}" / name)))
    return out


def _judge(looks, rig: Rig, profile: str) -> dict:
    """Per look and eye: the record's head and origin are the requested ones; the sample nearest the
    expected gaze is as near as the sensor's own nearest ray to its axis; the record agrees with
    Position within a pixel (S4's limit)."""
    from .. import record
    out = {}
    for k, name, spec, h, gaze, look in looks:
        model = spec.model(profile)
        rays = model.rays()
        d_loc = rays.direction.reshape(-1, 3)[rays.valid.reshape(-1)]
        axis0 = _nearest_deg(d_loc, [0.0, 0.0, -1.0])
        px = (model.s0_deg * (1 + model.emax / model.e2) if name == "foveated" else
              360.0 / model.width if name == "panorama" else math.degrees(1.0 / model.f))
        for rec in look.records:
            eye = rec.eye
            d_exp = h.dir_to_world([0.0, 0.0, -1.0]) if gaze is None else \
                h.dir_to_world(rig.eye_direction_h(h, gaze, eye))
            d_exp = d_exp / np.linalg.norm(d_exp)
            near = _nearest_deg(rec.direction, d_exp)
            c = record.sensor_consistency(rec)
            e = {"origin_err_m": float(np.max(np.abs(rec.origin - h.to_world(rig.centre_h(eye))))),
                 "head_origin_err_m": float(np.max(np.abs(rec.head_origin_w - h.origin_w))),
                 "head_R_err": float(np.max(np.abs(rec.head_R_wh - h.R_wh))),
                 "nearest_to_gaze_deg": near, "sensor_nearest_to_axis_deg": axis0,
                 "consistency_p999_deg": c["p999_angle_deg"], "pixel_deg": px, "hits": c["hits"]}
            e["ok"] = (e["origin_err_m"] <= 1e-12 and e["head_origin_err_m"] == 0.0 and e["head_R_err"] == 0.0
                       and near <= axis0 + 1e-4 and c["hits"] > 0 and c["p999_angle_deg"] <= px)
            out[f"head{k}_{name}_{eye}"] = e
    return out


def moved_heads(ctx) -> Result:
    work = Path(ctx["work"]) / "head" / "5"
    shutil.rmtree(work, ignore_errors=True)
    s = _session(ctx, work / "run" / "q")
    h0 = s.session_head
    heads = [h0, h0.moved((0.3, 0.0, 0.0)), h0.moved((0.0, 0.0, -0.5), 25.0),
             h0.moved((0.0, 0.1, 0.0), 0.0, -15.0, 10.0), h0.moved((-0.2, -0.05, 0.2), -40.0, 10.0, -5.0)]
    rig = s.rig
    res = _judge(_looks(s, heads), rig, s.cfg.profile)
    ok = all(v["ok"] for v in res.values())
    c = _session(ctx, work / "control" / "q", ignore_head=True)
    ctrl = _judge(_looks(c, heads[1:]), rig, c.cfg.profile)
    # every moved head must be caught, by every sensor
    caught = {kk: not v["ok"] for kk, v in ctrl.items()}
    return Result(STEP, "head.5 from moved heads every sensor's record sits on the requested head", ok,
                  all(caught.values()),
                  {"looks": res, "control_ignores_head": {kk: {"caught": caught[kk], "origin_err_m": v["origin_err_m"],
                                                               "nearest_to_gaze_deg": v["nearest_to_gaze_deg"]}
                                                          for kk, v in ctrl.items()}})


CHECKS_NP = [request_head, head_moves, move_head_requests]
CHECKS = [same_head_same_look, moved_heads]
