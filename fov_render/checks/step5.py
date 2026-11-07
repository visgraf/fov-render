"""Step 5 gates (issue #2): the request schema, Session, the render server and the client.

numpy only:
    5.1  requests and sessions round-trip through JSON bit for bit; every malformed one is refused
         with its reason
    5.2  compat.f3d_calibration is byte-identical to f3d-vision's make_calibration, and fov_render's
         eye matrices bitwise equal to _eye_matrix, over a gaze grid, both rolls, both profiles
needs bpy:
    5.3  fov_render reproduces f3d-vision's greedy render server bit for bit on CPU: the ready
         record (pass report, AB1a settings, head), every fixation's calibration.json and both eyes'
         EXR channels, through Session.in_process and through the background server
    5.4  the server protocol: a refused request leaves the server serving; a failed render stops it
         with exit 1; a session that cannot open fails at start
    5.5  tools/gfe_session.py drives f3d-vision's unchanged explorer on a Session (smoke, two
         fixations on the calibration room); needs an f3d-vision checkout ($FOVRENDER_F3D_ROOT)

The equivalence gate on the Classroom (check_equivalence 23/23 against the frozen Playground-2
run) needs the GPU and the frozen run: it is Code's, on the workstation (issue #2).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..rig import CV_TO_BLENDER, Gaze, HeadPose, Rig
from .common import ROOT, Result, legacy_path, load_legacy

STEP = "5"
SEEDS = {"L": 2111, "R": 2112}
VERGENCE_M = 2.10
GRID = [(float(y), float(p)) for p in (-84.5, -61.0, -33.25, -7.5, 0.0, 4.75, 29.0, 58.5, 88.0)
        for y in np.arange(-180.0, 180.0, 13.7)]


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
                          prepare=S.Perspective(), formats=("f3d_ab1a",))
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
    sess_bad = {"unknown format": dict(sess, formats=["nope"]), "bad access": dict(sess, access="god"),
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


# ----------------------------------------------------------------------------- 5.2

def _heads():
    out = [HeadPose(np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]), np.array([0.3, -0.2, 1.6]))]
    rng = np.random.default_rng(5)
    for _ in range(2):
        q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
        out.append(HeadPose(q * np.sign(np.linalg.det(q)), rng.normal(size=3)))
    return out


def f3d_calibration_bytes(_ctx=None) -> Result:
    from .. import compat
    from ..sensors import Perspective
    with legacy_path("f3d"):
        fg = load_legacy("f3d", "fsg_geometry.py")
        co = load_legacy("f3d", "classroom_oracle1_render.py")
    counts = {}
    for roll, tf in (("baseline_projected", "baseline_projected"), ("upright", "legacy_upright")):
        n = bad_json = bad_m = degenerate = 0
        for prof in ("full", "small"):
            model, rig = Perspective().model(prof), Rig(0.063, roll)
            for head in _heads():
                for yaw, pitch in GRID:
                    try:
                        c = fg.make_calibration(prof, yaw, pitch, VERGENCE_M, ipd=0.063, head_r_wh=head.R_wh,
                                                head_origin_w=head.origin_w, tangent_frame=tf)
                    except ValueError:
                        degenerate += 1
                        continue
                    g = Gaze.yaw_pitch(yaw, pitch, VERGENCE_M)
                    n += 1
                    bad_json += compat.f3d_json(compat.f3d_calibration(prof, model, rig, head, g)) != compat.f3d_json(c)
                    for e in c["eyes"]:
                        legacy_m = co._eye_matrix(c, e, np.array)          # Matrix replaced by np.array
                        bad_m += rig.pose(head, g, e["name"]).matrix_world.tobytes() != legacy_m.tobytes()
        counts[roll] = {"calibrations": n, "json_not_identical": bad_json, "eye_matrices_not_bitwise": bad_m,
                        "degenerate_skipped_by_legacy": degenerate}
    # control: one ULP of IPD changes the bytes
    head, model = _heads()[1], Perspective().model("full")
    c = fg.make_calibration("full", 31.0, 7.0, VERGENCE_M, ipd=0.063, head_r_wh=head.R_wh, head_origin_w=head.origin_w,
                            tangent_frame="baseline_projected")
    ulp = compat.f3d_calibration("full", model, Rig(float(np.nextafter(0.063, 1.0)), "baseline_projected"), head,
                                 Gaze.yaw_pitch(31.0, 7.0, VERGENCE_M))
    ctrl = compat.f3d_json(ulp) != compat.f3d_json(c)
    ok = all(v["calibrations"] > 1000 and v["json_not_identical"] == 0 and v["eye_matrices_not_bitwise"] == 0
             for v in counts.values())
    return Result(STEP, "5.2 f3d calibrations byte-identical, eye matrices bitwise (gaze grid, both rolls)", ok, ctrl,
                  {"counts": counts, "grid": len(GRID), "heads": 3, "control_ipd_plus_one_ulp_detected": ctrl})


# ----------------------------------------------------------------------------- 5.3

def calib_room_rich(ctx) -> str:
    """The calibration room with what the Classroom carries: Depth and Normal on, light passes, a
    Cycles denoising-data pass, an AOV and a keyframed seed."""
    import bpy
    from .step1 import calib_room
    out = Path(ctx["work"]) / "scenes" / "calib_room_rich.blend"
    if out.is_file():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=calib_room(ctx))
    s, vl = bpy.context.scene, bpy.context.view_layer
    for a in ("use_pass_combined", "use_pass_z", "use_pass_normal", "use_pass_position", "use_pass_object_index",
              "use_pass_diffuse_direct", "use_pass_mist", "use_pass_emit"):
        setattr(vl, a, True)
    vl.cycles.denoising_store_passes = True
    vl.aovs.add().name = "extra"
    s.cycles.seed = 7
    s.keyframe_insert(data_path="cycles.seed", frame=1)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    return str(out)


GAZES_5_3 = [(0.0, 0.0), (37.5, -12.25), (-101.0, 23.0)]
SPP_5_3 = 2


def _legacy_server_route(blend: str, out: Path) -> dict:
    """f3d-vision's greedy render server, its ready record and its per-fixation loop, with the
    legacy functions themselves (render_server.trim_passes, ab1a_render.configure / readback /
    calibration / write_json, classroom_oracle1_render._eye_matrix, render_foveated.render_fixation).
    Two substitutions, because the sandbox has neither: CPU for the spec's OPTIX, and the head read
    from the EYE object by the legacy reader instead of ns1a_render.eye_pose's comparison with the
    accepted AB1a file (whose head it returns unchanged)."""
    import bpy
    for m in ("ab1a_spec", "ab1a_render", "classroom_oracle1_render", "render_foveated", "bl_common", "fsg_geometry",
              "ns1a_spec", "ns1a_render", "exr_lite"):
        sys.modules.pop(m, None)
    with legacy_path("f3d", "active_bootstrap", "north_star", "greedy_foveal"):
        bpy.ops.wm.open_mainfile(filepath=blend)
        import ab1a_render as AR
        import classroom_oracle1_render as CO
        import render_foveated as RF
        rs = load_legacy("f3d", "greedy_foveal/render_server.py", "active_bootstrap", "north_star")
        AR.SP.DEVICE = "CPU"
        _e, note, o, r = CO._head_pose()
        pose = {"head_R_wh": r.tolist(), "head_origin_w_m": o.tolist()}
        CO._assign_instance_ids()
        c0 = AR.calibration(0.0, 0.0, pose)
        scene, cam, backend, matrix = AR.configure(c0, SPP_5_3)
        passes = rs.trim_passes()
        settings = AR.readback(scene, cam)
        RF._LAST_SAMPLING.clear()
        for k, (yaw, pitch) in enumerate(GAZES_5_3, start=1):
            fd = out / f"fix-{k:04d}"
            fd.mkdir(parents=True, exist_ok=True)
            c = AR.calibration(yaw, pitch, pose)
            AR.write_json(fd / "calibration.json", c)
            for eye in c["eyes"]:
                cam.matrix_world = CO._eye_matrix(c, eye, matrix)
                bpy.context.view_layer.update()
                RF.render_fixation(scene, SPP_5_3, str(fd / f"raw_{eye['name']}.exr"), seed=int(rs.AS.SEEDS[eye["name"]]))
    return {"passes": passes, "settings": json.loads(json.dumps(settings)), "head": pose, "backend": backend,
            "seeds": dict(rs.AS.SEEDS)}


def _fov_route(session, out: Path, gazes=GAZES_5_3, seeds=SEEDS) -> None:
    from .. import compat
    from ..sensors import Perspective
    rig = session.rig
    head = HeadPose(np.asarray(session.ready["head"]["head_R_wh"], float),
                    np.asarray(session.ready["head"]["head_origin_w_m"], float))
    model = Perspective().model(session.cfg.profile)
    for k, (yaw, pitch) in enumerate(gazes, start=1):
        fd = out / f"fix-{k:04d}"
        fd.mkdir(parents=True, exist_ok=True)
        g = Gaze.yaw_pitch(yaw, pitch, VERGENCE_M)
        (fd / "calibration.json").write_text(compat.f3d_json(compat.f3d_calibration(session.cfg.profile, model, rig, head, g)))
        session.look(Perspective(), ("L", "R"), g, spp=SPP_5_3, seeds=seeds, out_dir=fd)


def _exr_channels(path: Path) -> dict:
    from ..exr_lite import read_uncompressed_exr
    return {k: np.asarray(v) for k, v in read_uncompressed_exr(str(path)).items()}


def _compare_runs(a: Path, b: Path) -> dict:
    out = {"calibration_not_identical": [], "channels_not_bitwise": {}, "channel_names_differ": []}
    for k in range(1, len(GAZES_5_3) + 1):
        fa, fb = a / f"fix-{k:04d}", b / f"fix-{k:04d}"
        if (fa / "calibration.json").read_bytes() != (fb / "calibration.json").read_bytes():
            out["calibration_not_identical"].append(k)
        for eye in ("L", "R"):
            ca, cb = _exr_channels(fa / f"raw_{eye}.exr"), _exr_channels(fb / f"raw_{eye}.exr")
            if sorted(ca) != sorted(cb):
                out["channel_names_differ"].append([k, eye, sorted(ca), sorted(cb)])
                continue
            bad = [c for c in ca if ca[c].dtype != cb[c].dtype or ca[c].tobytes() != cb[c].tobytes()]
            if bad:
                out["channels_not_bitwise"][f"{k}{eye}"] = {c: float(np.nanmax(np.abs(ca[c].astype(np.float64) -
                                                                                     cb[c].astype(np.float64)))) for c in bad}
    out["equal"] = not (out["calibration_not_identical"] or out["channels_not_bitwise"] or out["channel_names_differ"])
    return out


def legacy_server_vs_session(ctx) -> Result:
    from .. import Session, compat, preset
    from ..sensors import Perspective
    work = Path(ctx["work"]) / "step5" / "5.3"
    shutil.rmtree(work, ignore_errors=True)          # a Session refuses a queue folder that was used
    rich = calib_room_rich(ctx)
    legacy = _legacy_server_route(rich, work / "legacy")
    cfg = preset("ab1a", device="CPU", spp=SPP_5_3)
    rig = Rig(0.063, "baseline_projected")
    kw = dict(access="oracle", instance_ids="legacy", prepare=Perspective(), formats=("f3d_ab1a",))
    ip = Session.in_process(cfg, rig, blend=rich, **kw)
    _fov_route(ip, work / "in_process")
    with Session(rich, cfg, rig, launcher="python", workdir=work / "queue", **kw) as srv:
        _fov_route(srv, work / "server")
        ready_srv = srv.ready
    rc = srv.returncode
    ready = {"in_process": ip.ready, "server": ready_srv}
    rec = {}
    for name, r in ready.items():
        f = r["formats"]["f3d_ab1a"]
        rec[name] = {"passes_equal": f["passes"] == legacy["passes"], "settings_equal": f["settings"] == legacy["settings"],
                     "head_equal": r["head"] == legacy["head"], "device": r["device"],
                     "settings_diff": sorted(k for k in set(f["settings"]) | set(legacy["settings"])
                                             if f["settings"].get(k) != legacy["settings"].get(k))}
    cmp_ip = _compare_runs(work / "legacy", work / "in_process")
    cmp_srv = _compare_runs(work / "legacy", work / "server")
    # controls: swapped seeds (Combined), a gaze 0.01 deg off (Position, calibration), the pass report
    # without dropping Depth and Normal
    _fov_route(ip, work / "ctrl_seeds", seeds={"L": SEEDS["R"], "R": SEEDS["L"]})
    c_seed = _compare_runs(work / "legacy", work / "ctrl_seeds")
    _fov_route(ip, work / "ctrl_gaze", gazes=[(y + 0.01, p) for y, p in GAZES_5_3])
    c_gaze = _compare_runs(work / "legacy", work / "ctrl_gaze")
    raw = ip.handler.scene_handle.sanitised["passes"]["disabled"]
    c_pass = raw != legacy["passes"]["disabled"]
    seed_hit = any(any("Combined" in ch for ch in v) for v in c_seed["channels_not_bitwise"].values())
    gaze_hit = bool(c_gaze["calibration_not_identical"]) and any(
        any("Position" in ch for ch in v) for v in c_gaze["channels_not_bitwise"].values())
    ok = (all(v["passes_equal"] and v["settings_equal"] and v["head_equal"] for v in rec.values())
          and cmp_ip["equal"] and cmp_srv["equal"] and rc == 0 and legacy["passes"]["disabled"] != [])
    return Result(STEP, "5.3 Session reproduces f3d-vision's greedy render server bit for bit (CPU, in-process and server)",
                  ok, seed_hit and gaze_hit and c_pass,
                  {"gazes": GAZES_5_3, "spp": SPP_5_3, "legacy_passes": legacy["passes"], "ready": rec,
                   "in_process_vs_legacy": cmp_ip, "server_vs_legacy": cmp_srv, "server_exit": rc,
                   "controls": {"swapped_seeds_combined_differs": seed_hit, "gaze_plus_0.01deg_differs": gaze_hit,
                                "unfiltered_pass_report_differs": c_pass},
                   "substitutions": "CPU for OPTIX; head from the legacy EYE reader (no accepted AB1a file here)"})


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


# ----------------------------------------------------------------------------- 5.5

def gfe_adapter_smoke(ctx) -> Result:
    name = "5.5 f3d-vision's explorer runs unchanged on a Session (tools/gfe_session.py, 2 fixations, CPU)"
    f3d = os.environ.get("FOVRENDER_F3D_ROOT")
    if not f3d or not (Path(f3d) / "tools" / "greedy_foveal" / "run.py").is_file():
        return Result(STEP, name, False, None, skipped="set FOVRENDER_F3D_ROOT to an f3d-vision checkout")
    from ..bl import scene as S
    from .step1 import calib_room
    work = Path(ctx["work"]) / "step5" / "5.5"
    work.mkdir(parents=True, exist_ok=True)
    room = calib_room(ctx)
    head = S.open_scene(room).head
    (work / "head.json").write_text(json.dumps(head.as_dict()))
    run = work / "run"
    shutil.rmtree(run, ignore_errors=True)
    cmd = [sys.executable, str(ROOT / "tools" / "gfe_session.py"), "--f3d", f3d, "--launcher", "python", "--device", "CPU",
           "--blend", room, "--head-json", str(work / "head.json"), "explore", "--run", str(run), "--max-fix", "2",
           "--spp", "2", "--keep-raw"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    (work / "explore.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        return Result(STEP, name, False, None, {"exit": p.returncode, "log_tail": (p.stdout + p.stderr)[-2000:]})
    tr = json.loads((run / "trajectory.json").read_text())
    with legacy_path("f3d"):
        fg = load_legacy("f3d", "fsg_geometry.py")
    from .. import compat
    cal_ok, cal_ctrl = [], []
    for e in tr["trajectory"]:
        fd = run / "fixations" / f"fix-{e['n']:04d}"
        legacy_c = fg.make_calibration("full", e["yaw_deg"], e["pitch_deg"], VERGENCE_M, ipd=0.063,
                                       head_r_wh=head.R_wh, head_origin_w=head.origin_w, tangent_frame="baseline_projected")
        cal_ok.append((fd / "calibration.json").read_text() == compat.f3d_json(legacy_c))
        upright = fg.make_calibration("full", e["yaw_deg"], e["pitch_deg"], VERGENCE_M, ipd=0.063,
                                      head_r_wh=head.R_wh, head_origin_w=head.origin_w)
        cal_ctrl.append((fd / "calibration.json").read_text() != compat.f3d_json(upright))
    srv = tr["server"]
    ok = (tr["fixations"] == 2 and tr["server_exit"] == 0 and all(cal_ok) and srv["seeds"] == SEEDS
          and srv["settings"]["samples"] == 2 and (run / "freeze.json").is_file()
          and all((run / "fixations" / f"fix-{k:04d}" / "points.npz").is_file() for k in (1, 2)))
    return Result(STEP, name, ok, all(cal_ctrl),
                  {"fixations": tr["fixations"], "stop_reason": tr["stop_reason"], "server_exit": tr["server_exit"],
                   "calibration_equals_make_calibration": cal_ok, "server_record": {k: srv[k] for k in ("device", "spp", "seeds", "passes")},
                   "points": [e["correspondence"]["valid"] for e in tr["trajectory"]],
                   "control_upright_calibration_differs": cal_ctrl, "f3d_checkout": f3d})


CHECKS_NP = [request_schema, f3d_calibration_bytes]
CHECKS = [legacy_server_vs_session, server_protocol, gfe_adapter_smoke]
