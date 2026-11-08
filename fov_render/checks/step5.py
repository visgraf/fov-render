"""Step 5 gates: the request schema, Session, the render server and the client.

numpy only:
    5.1  requests and sessions round-trip through JSON bit for bit; every malformed one is refused
         with its reason
needs bpy:
    5.4  the server protocol: a refused request leaves the server serving; a failed render stops it
         with exit 1; a session that cannot open fails at start

Up to v0.1.0 this step also held 5.2 (f3d-vision's calibrations byte-identical), 5.3 (its greedy
render server reproduced bit for bit) and 5.5 (its explorer on a Session); they went with the
adapters they tested (Luiz, issue #16). Their record is in docs/HISTORY.md.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..rig import Gaze, Rig
from .common import Result

STEP = "5"
VERGENCE_M = 2.10


# ----------------------------------------------------------------------------- 5.1

def request_schema(_ctx=None) -> Result:
    from .. import request as Q, sensors as S
    from ..bl.config import RenderConfig, preset
    good = [Q.make_request(1, S.Perspective(), ["L", "R"], Gaze.yaw_pitch(10.123456789012345, -3.25, 2.1), "/tmp/a",
                           seeds={"L": 2111, "R": 2112}, spp=64),
            Q.make_request(2, S.Foveated(E2_deg=2.0, e_max_deg=45.0), ["C"], Gaze.per_eye(C=(20.0, -5.0)), "/tmp/b",
                           outputs=("exr", "records")),
            Q.make_request(3, S.Panorama(preset="breadth1"), ["L"], None, "/tmp/c", yaw_offset_deg=0.1),
            Q.make_request(4, S.ReferenceView(64, 48, 27.0), ["R"], Gaze.verged_point([0.4, 2.1, 1.1]), "/tmp/d",
                           rig=Rig(0.07, "baseline_projected"))]
    sess = Q.make_session("/tmp/x.blend", preset("ab1a", device="CPU", spp=4), Rig(0.063, "baseline_projected"),
                          prepare=S.Perspective())
    round_trip = all(json.loads(Q.dumps(r)) == r and Q.validate_request(json.loads(Q.dumps(r))) for r in good)
    g = Q.gaze_from_dict(json.loads(Q.dumps(good[0]))["gaze"])
    floats_exact = g.yaw_deg == 10.123456789012345 and g.vergence_m == 2.1
    specs_back = [Q.sensor_from_dict(r["sensor"]) for r in good] == [S.Perspective(), S.Foveated(2.0, 45.0),
                                                                     S.Panorama(preset="breadth1"), S.ReferenceView(64, 48, 27.0)]
    cfg_back = RenderConfig.from_dict(json.loads(Q.dumps(sess))["config"]) == preset("ab1a", device="CPU", spp=4)
    r0 = good[0]
    bad = {
        "wrong schema": (dict(r0, schema="fov-render-request-v0"), "schema"),
        "n = 0": (dict(r0, n=0), "n:"),
        "unknown field": (dict(r0, colour="red"), "unknown field"),
        "unknown sensor": (dict(r0, sensor={"type": "fisheye"}), "sensor.type"),
        "sensor field of another sensor": (dict(r0, sensor={"type": "perspective", "E2_deg": 2.0}), "unknown field"),
        "negative E2": (dict(r0, sensor={"type": "foveated", "E2_deg": -1.0}), "sensor.E2_deg"),
        "cone >= 90": (dict(r0, sensor={"type": "foveated", "e_max_deg": 95.0}), "below 90"),
        "odd panorama": (dict(r0, sensor={"type": "panorama", "width": 721}, gaze={"mode": "fixed"}), "even"),
        "duplicate eye": (dict(r0, eyes=["L", "L"]), "eyes"),
        "unknown eye": (dict(r0, eyes=["X"]), "eyes"),
        "fixed gaze for a perspective look": (dict(r0, gaze={"mode": "fixed"}), "needs a gaze"),
        "pitch beyond 90": (dict(r0, gaze={"mode": "yaw_pitch", "yaw_deg": 0.0, "pitch_deg": 91.0}), "pitch"),
        "NaN yaw": (dict(r0, gaze={"mode": "yaw_pitch", "yaw_deg": float("nan"), "pitch_deg": 0.0}), "yaw_deg"),
        "per-eye gaze missing an eye": (dict(r0, gaze={"mode": "per_eye", "angles": {"L": [0.0, 0.0]}}), "no angles"),
        "seed missing for an eye": (dict(r0, seeds={"L": 1}), "no seed"),
        "negative seed": (dict(r0, seeds={"L": -1, "R": 2}), "seeds.L"),
        "spp 0": (dict(r0, spp=0), "spp"),
        "spp as text": (dict(r0, spp="64"), "spp"),
        "bad roll": (dict(r0, rig={"roll": "sideways"}), "rig.roll"),
        "empty out_dir": (dict(r0, out_dir=""), "out_dir"),
        "unknown output": (dict(r0, outputs=["png"]), "outputs"),
    }
    refused = {}
    for label, (req, needle) in bad.items():
        try:
            Q.validate_request(req)
            refused[label] = "ACCEPTED"
        except Q.RequestError as e:
            refused[label] = "ok" if any(needle in x for x in e.reasons) else f"wrong reason: {e.reasons}"
    sess_bad = {"unknown field": dict(sess, formats=["nope"]), "bad access": dict(sess, access="god"),
                "bad numbering": dict(sess, instance_ids="random"), "bad head": dict(sess, head={"head_R_wh": [1, 2]})}
    for label, s in sess_bad.items():
        try:
            Q.validate_session(s)
            refused["session: " + label] = "ACCEPTED"
        except Q.RequestError:
            refused["session: " + label] = "ok"
    # control: the validators accept every good request and the good session (they do not refuse everything)
    try:
        for r in good:
            Q.validate_request(r)
        Q.validate_session(sess)
        accepts_good = True
    except Q.RequestError:
        accepts_good = False
    ok = round_trip and floats_exact and specs_back and cfg_back and all(v == "ok" for v in refused.values())
    return Result(STEP, "5.1 requests round-trip bit for bit; malformed ones are refused with their reason", ok,
                  accepts_good, {"round_trip": round_trip, "floats_exact": floats_exact, "sensor_specs_back": specs_back,
                                 "config_back": cfg_back, "refused": refused, "cases": len(refused),
                                 "control_good_requests_accepted": accepts_good})


# ----------------------------------------------------------------------------- 5.4

def server_protocol(ctx) -> Result:
    from .. import RequestError, Session, SessionError, preset
    from ..sensors import Perspective
    from .step1 import calib_room
    work = Path(ctx["work"]) / "step5" / "5.4"
    shutil.rmtree(work, ignore_errors=True)
    room = calib_room(ctx)
    cfg = preset("ab1a", device="CPU", spp=1, profile="small")
    g = Gaze.yaw_pitch(5.0, 0.0, VERGENCE_M)
    detail = {}
    s = Session(room, cfg, Rig(), launcher="python", workdir=work / "q1")
    try:
        a = s.look(Perspective(), ("L",), g, out_dir=work / "a")
        raw = s.request_raw('{"schema": "fov-render-request-v1", "n": 2, "kind": "look", "eyes": ["Q"]}')
        detail["raw_refused_reasons"] = raw.get("reasons")
        try:
            s.request(dict(a.request, n=3, spp=-1))
            detail["bad_spp"] = "ACCEPTED"
        except RequestError as e:
            detail["bad_spp"] = e.reasons
        b = s.look(Perspective(), ("L",), g, out_dir=work / "b")
        detail["served_after_refusals"] = b.response["ok"] and b.response["n"] == 4
        bad = dict(b.request, n=5, out_dir="/proc/fov-render-cannot-write-here")
        try:
            s.request(bad)
            detail["render_failure"] = "no error"
        except SessionError as e:
            detail["render_failure"] = str(e).splitlines()[0]
        rc = s.p.wait(timeout=60) if s.p is not None else None
        detail["exit_after_failure"] = rc
    finally:
        s.close()
    try:            # valid on the host, fails in the server: the file does not exist
        Session(work / "no-such-scene.blend", cfg, Rig(), launcher="python", workdir=work / "q2", ready_timeout_s=120)
        detail["bad_session"] = "opened"
    except SessionError as e:
        detail["bad_session"] = str(e).splitlines()[0][:160]
    ok = (bool(detail["raw_refused_reasons"]) and isinstance(detail["bad_spp"], list) and detail["served_after_refusals"]
          and detail["render_failure"].startswith("render failed") and detail["exit_after_failure"] == 1
          and detail["bad_session"].startswith("the render server died at start"))
    # control: a valid request is answered (the refusals are not the server refusing everything)
    return Result(STEP, "5.4 server protocol: refusals keep it serving, a failed render exits 1, a bad session fails",
                  ok, bool(detail.get("served_after_refusals")), detail)


CHECKS_NP = [request_schema]
CHECKS = [server_protocol]
