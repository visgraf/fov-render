"""Step "cli": the `fov-render` command line (issue #18).

numpy only:
    cli.1  malformed arguments are refused with their reasons and exit 2, before Blender starts
needs bpy:
    cli.2  a look through the command line equals the same look through Session, bit for bit
    cli.3  a scene Blender cannot open exits 1 from look, info and check, with every launcher found
    cli.4  `check --write-spec` writes a spec the calibration room passes and a broken room fails
    cli.5  `eye` places the EYE where asked, as tools/make_eye_scene.py did at v0.1.0

Every Blender run here is a separate process (the command line's own path), so the step costs
about as many Blender starts as it has runs.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import ROOT, Result

STEP = "cli"
NO_BLENDER = "/nonexistent/fov-render-no-blender"


def _cli(argv) -> tuple[int, str]:
    """Run the command line in this process: (exit status, what it printed to stderr)."""
    from .. import cli
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = cli.main([str(a) for a in argv])
    return rc, err.getvalue()


def _launcher() -> str:
    """The launcher the gates use: the Blender binary they run in, else this Python (the wheel)."""
    import bpy
    b = bpy.app.binary_path
    return b if b and os.path.basename(b).lower().startswith("blender") else "python"


def _launchers() -> list[str]:
    out = [_launcher()]
    b = os.environ.get("FOVRENDER_BLENDER") or shutil.which("blender")
    if b and b not in out:
        out.append(b)
    return out


# ----------------------------------------------------------------------------- cli.1

def refusals(ctx) -> Result:
    work = Path(ctx["work"]) / "cli" / "1"
    work.mkdir(parents=True, exist_ok=True)
    blend = work / "exists.blend"
    blend.write_bytes(b"not read: nothing starts")
    bad_file = work / "looks.json"
    bad_file.write_text(json.dumps([{"sensor": {"type": "fisheye"}, "eyes": ["L"]},
                                    {"sensor": {"type": "panorama", "width": 721}, "eyes": ["C", "C"]}]))
    spec = work / "spec.json"
    spec.write_text(json.dumps({"schema": "fov-render-scene-spec-v1", "expect": {"max_hole_fraction": None,
                                                                                 "max_holes": 0.1}}))
    to_measure = work / "to_measure.json"
    to_measure.write_text(json.dumps({"schema": "fov-render-scene-spec-v1", "expect": {}, "to_measure": ["spp"]}))
    L = ["--launcher", NO_BLENDER]
    look = ["look", "--blend", blend, "--out", work / "o", *L]
    cases = {
        "no such scene": (["look", "--blend", work / "none.blend", "--sensor", "perspective", "--gaze", 0, 0,
                           "--out", work / "o", *L], ["--blend: no such file"]),
        "unknown sensor": (look + ["--sensor", "fisheye"], ["invalid choice"]),
        "foveated without a gaze": (look + ["--sensor", "foveated"], ["needs a gaze"]),
        "every reason at once": (look + ["--sensor", "foveated", "--gaze", 0, 95, "--param", "E2_deg=-1", "n=3.5",
                                         "e_max_deg=91"],
                                 ["sensor.E2_deg", "sensor.n", "below 90", "pitch_deg"]),
        "field of another sensor": (look + ["--sensor", "perspective", "--gaze", 0, 0, "--param", "E2_deg=2"],
                                    ["unknown field"]),
        "param not KEY=VALUE": (look + ["--sensor", "perspective", "--gaze", 0, 0, "--param", "E2_deg"],
                                ["not KEY=VALUE"]),
        "vergence without --gaze": (look + ["--sensor", "perspective", "--point", 0, 1, 1, "--vergence", 2],
                                    ["--vergence"]),
        "seed for a missing eye": (look + ["--sensor", "perspective", "--gaze", 0, 0, "--seeds", "L=1"],
                                   ["no seed for eye(s) ['R']"]),
        "two gazes": (look + ["--sensor", "perspective", "--gaze", 0, 0, "--point", 0, 1, 1], ["not allowed with"]),
        "odd panorama": (["panorama", "--blend", blend, "--width", 721, "--out", work / "o", *L], ["even"]),
        "bad config": (look + ["--sensor", "perspective", "--gaze", 0, 0, "--profile", "huge"], ["render config"]),
        "requests file": (["requests", "--blend", blend, bad_file, "--out", work / "o", *L],
                          ["looks[0].sensor.type", "looks[1].sensor.width", "looks[1].eyes"]),
        "eye without yaw": (["eye", "--blend", blend, "--out", work / "e.blend", "--position", 0, 0, 1, *L],
                            ["--position needs --yaw"]),
        "eye overwrites": (["eye", "--blend", blend, "--out", blend, "--from-camera", *L], ["must not overwrite"]),
        "spec typo and null": (["check", "--blend", blend, "--spec", spec, *L],
                               ["unknown expectation(s) ['max_holes']", "['max_hole_fraction'] are null"]),
        "spec to measure": (["check", "--blend", blend, "--spec", to_measure, *L], ["still to be measured"]),
        "draft without S2": (["check", "--blend", blend, "--only", "S1", "--write-spec", work / "d.json", *L],
                             ["needs S1 and S2"]),
    }
    out, ok = {}, True
    for name, (argv, want) in cases.items():
        rc, err = _cli(argv)
        good = rc == 2 and all(w in err for w in want)
        ok &= good
        out[name] = {"exit": rc, "ok": good, "want": want, "stderr": err.strip().splitlines()[-6:]}
    # control: the same arguments made valid get past every refusal and fail only at the launch (exit 1)
    rc_ctrl, err_ctrl = _cli(look + ["--sensor", "perspective", "--gaze", 0, 0])
    # the console module exits with the same status
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT)] + ([os.environ["PYTHONPATH"]]
                                                                      if os.environ.get("PYTHONPATH") else [])))
    p = subprocess.run([sys.executable, "-m", "fov_render.cli", *[str(a) for a in cases["odd panorama"][0]]],
                       capture_output=True, text=True, env=env, cwd=work)
    ok &= p.returncode == 2 and "even" in p.stderr
    return Result(STEP, "cli.1 malformed arguments are refused with every reason, exit 2, nothing started", ok,
                  rc_ctrl == 1, {"cases": out, "control_valid_args": {"exit": rc_ctrl, "stderr": err_ctrl.strip()[-300:]},
                                 "python_-m_exit": p.returncode})


# ----------------------------------------------------------------------------- cli.2

def _arrays(folder: Path) -> dict:
    from ..exr_lite import read_uncompressed_exr
    out = {}
    for f in sorted(folder.glob("raw_*.exr")):
        for k, v in read_uncompressed_exr(str(f)).items():
            out[f"{f.stem}:{k}"] = np.asarray(v)
    for f in sorted(folder.glob("record_*.npz")):
        with np.load(f, allow_pickle=False) as z:
            for k in z.files:
                out[f"{f.stem}:{k}"] = np.asarray(z[k])
    return out


def _same(a: dict, b: dict) -> tuple[bool, list]:
    diff = sorted(k for k in set(a) | set(b)
                  if k not in a or k not in b or a[k].shape != b[k].shape or a[k].dtype != b[k].dtype
                  or a[k].tobytes() != b[k].tobytes())
    return not diff and bool(a), diff


def look_equals_session(ctx) -> Result:
    from .. import Session, sensors as SN
    from ..bl.config import RenderConfig
    from ..rig import Gaze, Rig
    from .step1 import calib_room
    work = Path(ctx["work"]) / "cli" / "2"
    shutil.rmtree(work, ignore_errors=True)
    room, launcher = calib_room(ctx), _launcher()
    point, seeds = [0.4, 2.1, 1.1], {"L": 11, "R": 12}
    with Session(room, RenderConfig(profile="small", device="CPU"), Rig(), launcher=launcher,
                 workdir=work / "q") as s:
        s.look(SN.Foveated(), ("L", "R"), Gaze.verged_point(point), spp=4, seeds=seeds, out_dir=work / "session",
               outputs=("exr", "records"))
    base = ["--blend", room, "--device", "CPU", "--profile", "small", "--launcher", launcher]
    rc, err = _cli(["look", *base, "--sensor", "foveated", "--eyes", "L", "R", "--point", *point, "--spp", 4,
                    "--seeds", "L=11", "R=12", "--records", "--out", work / "cli"])
    a, b = _arrays(work / "session"), _arrays(work / "cli")
    same, diff = _same(a, b)
    ra = json.loads((work / "session" / "meta.json").read_text())["request"]
    rb = json.loads((work / "cli" / "meta.json").read_text())["request"]
    req_same = {k: v for k, v in ra.items() if k != "out_dir"} == {k: v for k, v in rb.items() if k != "out_dir"}
    # controls, through `requests` (one session): seeds swapped (Combined differs), the point 1 mm away
    # (Position differs)
    looks = [{"sensor": {"type": "foveated"}, "eyes": ["L", "R"], "gaze": {"mode": "verged_point", "point_w": point},
              "spp": 4, "seeds": {"L": 12, "R": 11}, "outputs": ["exr", "records"], "out_dir": "seeds"},
             {"sensor": {"type": "foveated"}, "eyes": ["L", "R"],
              "gaze": {"mode": "verged_point", "point_w": [point[0] + 0.001, point[1], point[2]]},
              "spp": 4, "seeds": seeds, "outputs": ["exr", "records"], "out_dir": "point"}]
    (work / "looks.json").write_text(json.dumps(looks))
    rc2, err2 = _cli(["requests", *base, work / "looks.json", "--out", work / "ctrl"])
    ctrl = {}
    for name, key in (("seeds", "raw_L:ViewLayer.Combined.R"), ("point", "raw_L:ViewLayer.Position.X")):
        c = _arrays(work / "ctrl" / name)
        ctrl[name] = {"key": key, "differs": bool(key in a and key in c and a[key].tobytes() != c[key].tobytes()),
                      "all_same": _same(a, c)[0]}
    run = json.loads((work / "cli" / "fov-render-run.json").read_text()) if rc == 0 else {}
    ready_device = (run.get("ready") or {}).get("device")
    return Result(STEP, "cli.2 a look through the command line equals the same look through Session, bit for bit",
                  rc == 0 and same and req_same and ready_device == "CPU", rc2 == 0 and all(v["differs"] for v in ctrl.values()),
                  {"exit": rc, "stderr": err.strip()[-500:], "arrays": len(a), "differ": diff[:20],
                   "request_same_but_out_dir": req_same, "controls": ctrl, "controls_exit": rc2,
                   "controls_stderr": err2.strip()[-500:], "launcher": launcher,
                   "ready_device": ready_device})


# ----------------------------------------------------------------------------- cli.3

def failures_exit_1(ctx) -> Result:
    from .step1 import calib_room
    work = Path(ctx["work"]) / "cli" / "3"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    bad = work / "not_a_scene.blend"
    bad.write_text("this is not a .blend file\n")
    room = calib_room(ctx)
    out, ok = {}, True
    for launcher in _launchers():
        L = ["--launcher", launcher]
        runs = {"look": ["look", "--blend", bad, "--sensor", "perspective", "--gaze", 0, 0, "--device", "CPU",
                         "--out", work / "o", *L],
                "info": ["info", "--blend", bad, *L],
                "check": ["check", "--blend", bad, "--device", "CPU", "--only", "S1", "--work", work / "g", *L]}
        res = {k: _cli(v) for k, v in runs.items()}
        good = all(rc == 1 for rc, _e in res.values())
        ok &= good
        out[launcher] = {k: {"exit": rc, "stderr": e.strip()[-300:]} for k, (rc, e) in res.items()}
        out[launcher]["ok"] = good
    # control: the same `info` on the calibration room exits 0
    rc_ctrl, _ = _cli(["info", "--blend", room, "--json", work / "info.json", "--launcher", _launcher()])
    return Result(STEP, "cli.3 a scene Blender cannot open exits 1 from look, info and check (every launcher)",
                  ok, rc_ctrl == 0, {"runs": out, "control_info_on_room_exit": rc_ctrl})


# ----------------------------------------------------------------------------- cli.4

def _gate(blend, spec, work, launcher, draft=None) -> tuple[int, dict]:
    argv = ["check", "--blend", blend, "--device", "CPU", "--only", "S1", "S2", "--work", work,
            "--json", Path(work) / "results.json", "--launcher", launcher]
    if spec:
        argv += ["--spec", spec]
    if draft:
        argv += ["--write-spec", draft]
    rc, _ = _cli(argv)
    p = Path(work) / "results.json"
    res = json.loads(p.read_text()) if p.exists() else {"results": []}
    return rc, {r["name"][:2]: r["status"] for r in res["results"]}


def write_spec(ctx) -> Result:
    from .common import run_blender_script
    from .step1 import calib_room
    work = Path(ctx["work"]) / "cli" / "4"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    room, launcher = calib_room(ctx), _launcher()
    script = str(Path(__file__).resolve().parent / "mutate_room.py")
    rooms = {"room": Path(room)}
    for m in ("broken", "flipped"):
        rooms[m] = work / f"calib_room_{m}.blend"
        run_blender_script(script, ["--blend", room, "--out", rooms[m], "--mutation", m])
    d = {}
    for name in ("room", "flipped"):
        rc, st = _gate(rooms[name], None, work / f"draft_{name}", launcher, draft=work / f"{name}.spec.json")
        d[f"draft_{name}"] = {"exit": rc, "status": st}
    spec_room, spec_flip = work / "room.spec.json", work / "flipped.spec.json"
    for name, blend, spec in (("room_on_its_draft", rooms["room"], spec_room),
                              ("flipped_on_its_draft", rooms["flipped"], spec_flip),
                              ("broken_on_room_draft", rooms["broken"], spec_room),
                              ("flipped_on_room_draft", rooms["flipped"], spec_room)):
        rc, st = _gate(blend, spec, work / name, launcher)
        d[name] = {"exit": rc, "status": st}
    sr = json.loads(spec_room.read_text()) if spec_room.exists() else {}
    sf = json.loads(spec_flip.read_text()) if spec_flip.exists() else {}
    d["room_draft_expect"] = sr.get("expect")
    d["flipped_draft_backface_allowed"] = sf.get("expect", {}).get("backface_allowed")
    ok = (d["room_on_its_draft"]["exit"] == 0 and d["flipped_on_its_draft"]["exit"] == 0
          and sr.get("draft") is True and sr["expect"]["missing_images"] == []
          and d["flipped_draft_backface_allowed"] == ["fov_flipped_cube"])
    ctrl = (d["broken_on_room_draft"]["exit"] == 1 and d["broken_on_room_draft"]["status"] == {"S1": "FAIL", "S2": "FAIL"}
            and d["flipped_on_room_draft"]["exit"] == 1 and d["flipped_on_room_draft"]["status"].get("S2") == "FAIL")
    return Result(STEP, "cli.4 --write-spec: the room passes its draft; a broken room fails it (S1, S2)", ok, ctrl, d)


# ----------------------------------------------------------------------------- cli.5

def _head_yaw(head: dict) -> tuple[np.ndarray, float, float]:
    R = np.asarray(head["head_R_wh"], float)
    f = R @ np.array([0.0, 0.0, -1.0])                 # H: -Z forward
    return np.asarray(head["head_origin_w_m"], float), math.degrees(math.atan2(-f[0], f[1])), float(f[2])


def eye(ctx) -> Result:
    from .step1 import calib_room
    work = Path(ctx["work"]) / "cli" / "5"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    room, launcher = calib_room(ctx), _launcher()
    pos, yaw = [0.1, 0.2, 1.4], 30.0
    rc, err = _cli(["eye", "--blend", room, "--out", work / "eye30.blend", "--position", *pos, "--yaw", yaw,
                    "--replace-eye", "--launcher", launcher])
    rc_i, _ = _cli(["info", "--blend", work / "eye30.blend", "--json", work / "eye30.json", "--launcher", launcher])
    o, y, fz = _head_yaw(json.loads((work / "eye30.json").read_text())["head"]) if rc_i == 0 else (None, None, None)
    placed = (rc == 0 and rc_i == 0 and float(np.max(np.abs(o - pos))) < 1e-6 and abs(y - yaw) < 1e-5 and abs(fz) < 1e-6)
    refused_rc, _ = _cli(["eye", "--blend", room, "--out", work / "again.blend", "--from-camera", "--launcher", launcher])
    # behaviour before structure (CLAUDE.md rule 6): tools/make_eye_scene.py as it was at v0.1.0, on the same input
    old = {"compared": False}
    tag = subprocess.run(["git", "-C", str(ROOT), "show", "v0.1.0:tools/make_eye_scene.py"], capture_output=True,
                         text=True)
    if tag.returncode == 0:
        from .common import run_blender_script
        script = work / "make_eye_scene_v0.1.0.py"
        script.write_text(tag.stdout.replace("sys.path.insert(0, str(Path(__file__).resolve().parents[1]))",
                                             f"sys.path.insert(0, {str(ROOT)!r})"))
        run_blender_script(str(script), ["--blend", room, "--out", work / "eye30_old.blend", "--position", *pos,
                                         "--yaw", yaw, "--replace-eye"])
        a = json.loads((work / "eye30.blend.json").read_text())["eye"]
        b = json.loads((work / "eye30_old.blend.json").read_text())["eye"]
        old = {"compared": True, "eye_block_identical": a == b}
    ok = placed and refused_rc == 1 and old.get("eye_block_identical", True)
    # control: the room's own EYE (before `eye`) is not at the asked pose
    rc_c, _ = _cli(["info", "--blend", room, "--json", work / "room.json", "--launcher", launcher])
    oc, yc, _ = _head_yaw(json.loads((work / "room.json").read_text())["head"])
    ctrl = rc_c == 0 and not (float(np.max(np.abs(oc - pos))) < 1e-6 and abs(yc - yaw) < 1e-5)
    return Result(STEP, "cli.5 eye places the EYE where asked (as tools/make_eye_scene.py at v0.1.0)", ok, ctrl,
                  {"exit": rc, "stderr": err.strip()[-300:], "info_exit": rc_i,
                   "origin": None if o is None else o.tolist(), "yaw_deg": y, "forward_z": fz,
                   "existing_eye_refused_exit": refused_rc, "v0.1.0_tool": old,
                   "control_room_head": {"origin": oc.tolist(), "yaw_deg": yc}})


CHECKS_NP = [refusals]
CHECKS = [look_equals_session, failures_exit_1, write_spec, eye]
