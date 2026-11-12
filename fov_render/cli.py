"""The `fov-render` command line (report section 8.6; issue #18).

    fov-render look      --blend B --sensor foveated --eyes L R --point X Y Z --out DIR [--device CPU]
                         [--head-move DX DY DZ] [--head-turn YAW PITCH ROLL]   the head moved for this look
    fov-render panorama  --blend B [--preset preview360 | --width W] [--eye C] --out DIR
    fov-render requests  --blend B FILE.json --out DIR       many looks, the scene opened once
    fov-render info      --blend B [--json OUT]               the scene report: head, view layer, catalogue, audits
    fov-render eye       --blend IN --out OUT (--position X Y Z --yaw D | --from-camera [NAME] --forward-m M)
    fov-render check     --blend B [--spec S] [--only S1 S2] [--write-spec DRAFT]   the scene gate S1-S6
    fov-render selftest  [--steps 1 2 3 4 5 cli] [--device CPU]                     the package's gates

`look`, `panorama` and `requests` send the same requests a fov_render.Session sends; `info`, `eye`,
`check` and `selftest` run fov_render.blcli in Blender. Both find Blender the same way
(fov_render.launch, --launcher): $FOVRENDER_BLENDER, else `blender` on PATH, else this Python with
the bpy wheel.

Exit status: 0 done; 1 a render, a gate or Blender failed; 2 the arguments or a request were
refused (every reason is printed, and nothing is started).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from . import launch
from .request import EYES, REQUEST_SCHEMA, RequestError, check_gaze, check_sensor, dumps, gaze_from_dict, \
    gaze_to_dict, sensor_from_dict, sensor_to_dict, validate_request
from .rig import ROLLS

BLCLI_SCRIPT = Path(__file__).resolve().parent / "blcli.py"
SENSOR_NAMES = {"foveated": "foveated", "perspective": "perspective", "panorama": "panorama",
                "reference_view": "reference_view", "reference": "reference_view"}
PREFIX = "fov-render"


class Refused(Exception):
    """Arguments refused before anything starts (exit 2); .reasons lists every problem."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


class _Parser(argparse.ArgumentParser):
    def error(self, message):                     # argparse's own refusals: also exit 2, with our prefix
        self.print_usage(sys.stderr)
        print(f"{PREFIX}: refused: {message}", file=sys.stderr)
        raise SystemExit(2)


# ----------------------------------------------------------------------------- parsing helpers

def _value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _pairs(items, what: str, reasons: list, parse=_value) -> dict:
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep or not k:
            reasons.append(f"{what}: {it!r} is not KEY=VALUE")
            continue
        try:
            out[k] = parse(v)
        except ValueError:
            reasons.append(f"{what}: {it!r} has a value that does not parse")
    return out


def _floats(text: str):
    return [float(x) for x in text.split(",")]


def _blend(path, reasons: list, what="--blend") -> Path | None:
    p = Path(path)
    if not p.is_file():
        reasons.append(f"{what}: no such file {path}")
        return None
    return p.resolve()


def _config(a, reasons: list):
    from .bl.config import PRESET_NAMES, RenderConfig, preset
    kw = {"device": a.device, "device_policy": a.device_policy}
    if a.profile is not None:
        kw["profile"] = a.profile
    if a.config_seed is not None:
        kw["seed"] = a.config_seed
    try:
        if a.config_preset:
            if a.config_preset not in PRESET_NAMES:
                reasons.append(f"--config-preset: {a.config_preset!r} is not one of {list(PRESET_NAMES)}")
                return None
            return preset(a.config_preset, **kw)
        return RenderConfig(**kw)
    except (ValueError, KeyError) as e:
        reasons.append(f"render config: {e}")
        return None


def _rig(a, reasons: list):
    from .rig import Rig
    try:
        return Rig(**({"ipd_m": a.ipd} if a.ipd is not None else {}), roll=a.roll)
    except ValueError as e:
        reasons.append(f"rig: {e}")
        return None


def _gaze(a, reasons: list) -> dict:
    if a.point is not None:
        d = {"mode": "verged_point", "point_w": a.point}
    elif a.gaze is not None:
        d = {"mode": "yaw_pitch", "yaw_deg": a.gaze[0], "pitch_deg": a.gaze[1], "vergence_m": a.vergence}
    elif a.per_eye:
        d = {"mode": "per_eye", "angles": _pairs(a.per_eye, "--per-eye", reasons, _floats)}
    else:
        d = {"mode": "fixed"}
        if a.yaw_offset is not None:
            d["yaw_offset_deg"] = a.yaw_offset
    if a.vergence is not None and d["mode"] != "yaw_pitch":
        reasons.append("--vergence: only with --gaze YAW PITCH")
    if a.yaw_offset is not None and d["mode"] != "fixed":
        reasons.append("--yaw-offset: only with a fixed look (no --point, --gaze or --per-eye)")
    reasons += check_gaze(d)
    return d


# ----------------------------------------------------------------------------- the parser

def _session_options(p):
    g = p.add_argument_group("the session (opened once)")
    g.add_argument("--blend", required=True, help="the scene, with an EYE camera (fov-render eye)")
    g.add_argument("--profile", default=None, help="small | full | ... (default: the config's, small)")
    g.add_argument("--device", default="OPTIX", choices=["OPTIX", "CUDA", "CPU"])
    g.add_argument("--device-policy", default="strict", choices=["strict", "fallback"],
                   help="strict: fail rather than fall back to the CPU")
    g.add_argument("--config-preset", default=None, help="a RenderConfig preset (fsg, oracle1, ab1a, ...)")
    g.add_argument("--config-seed", type=int, default=None, help="the config's seed (per-eye seeds: --seeds)")
    g.add_argument("--ipd", type=float, default=None, help="metres (default 0.063)")
    g.add_argument("--roll", default="upright", choices=list(ROLLS))
    g.add_argument("--access", default="observation", choices=["observation", "oracle"])
    g.add_argument("--instance-ids", default="complete", choices=["complete", "legacy"])
    g.add_argument("--launcher", default="auto", help="auto | blender | python | an executable")
    g.add_argument("--workdir", default=None, help="the session's queue folder (default: a new temporary one)")


def _look_options(p, *, panorama: bool):
    g = p.add_argument_group("the look")
    if not panorama:
        g.add_argument("--sensor", required=True, choices=sorted(SENSOR_NAMES))
        g.add_argument("--param", nargs="+", default=[], metavar="KEY=VALUE",
                       help="sensor fields, e.g. E2_deg=2.0 e_max_deg=45 (fov_render.sensors)")
        g.add_argument("--eyes", nargs="+", default=["L", "R"], choices=list(EYES))
        gz = g.add_mutually_exclusive_group()
        gz.add_argument("--point", nargs=3, type=float, metavar=("X", "Y", "Z"), help="verge on a world point (m)")
        gz.add_argument("--gaze", nargs=2, type=float, metavar=("YAW", "PITCH"), help="head-frame degrees")
        gz.add_argument("--per-eye", nargs="+", metavar="EYE=YAW,PITCH")
        gz.add_argument("--fixed", action="store_true", help="no gaze: the eyes look along the head (panorama, reference)")
        g.add_argument("--vergence", type=float, default=None, help="with --gaze: verge at this distance (m)")
    else:
        g.add_argument("--preset", default="preview360", choices=["preview360", "breadth1"])
        g.add_argument("--width", type=int, default=None, help="overrides the preset's width (even)")
        g.add_argument("--yaw-offset-px", type=float, default=None)
        g.add_argument("--backface", action="store_true", help="also write the back-face pass")
        g.add_argument("--eye", nargs="+", dest="eyes", default=["C"], choices=list(EYES))
    g.add_argument("--yaw-offset", type=float, default=None, help="a fixed look's yaw offset (deg)")
    g.add_argument("--head-move", nargs=3, type=float, default=None, metavar=("DX", "DY", "DZ"),
                   help="move the head from the session's, in its own frame (m; +X right, +Y up, -Z forward)")
    g.add_argument("--head-turn", nargs=3, type=float, default=None, metavar=("YAW", "PITCH", "ROLL"),
                   help="then turn it (deg; +yaw right, +pitch up, +roll right ear down)")
    g.add_argument("--spp", type=int, default=None, help="samples per pixel (default: the profile's)")
    g.add_argument("--seeds", nargs="+", default=None, metavar="EYE=N")
    g.add_argument("--records", action="store_true", help="also write the sample records (.npz)")
    g.add_argument("--out", required=True, help="the look's folder")


def parser() -> argparse.ArgumentParser:
    from . import __version__
    ap = _Parser(prog=PREFIX, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"fov-render {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, parser_class=_Parser)

    p = sub.add_parser("look", help="one look: any sensor, any gaze")
    _session_options(p)
    _look_options(p, panorama=False)

    p = sub.add_parser("panorama", help="a fixed equirectangular look (report 8.6's shorthand)")
    _session_options(p)
    _look_options(p, panorama=True)

    p = sub.add_parser("requests", help="many looks from a JSON file through one session")
    _session_options(p)
    p.add_argument("file", help="a JSON list of looks: {sensor, eyes, gaze, spp, seeds, rig, head, out_dir, outputs}")
    p.add_argument("--out", required=True, help="the folder relative out_dirs are taken from")

    p = sub.add_parser("info", help="the scene report as JSON")
    p.add_argument("--blend", required=True)
    p.add_argument("--instance-ids", default="complete", choices=["complete", "legacy"])
    p.add_argument("--catalogue", action="store_true", help="include every catalogue entry")
    p.add_argument("--json", default=None, help="write here instead of printing")
    p.add_argument("--launcher", default="auto")

    p = sub.add_parser("eye", help="a copy of a .blend with an EYE camera")
    p.add_argument("--blend", required=True)
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--position", nargs=3, type=float, metavar=("X", "Y", "Z"))
    g.add_argument("--from-camera", nargs="?", const="", default=None, metavar="NAME")
    p.add_argument("--yaw", type=float, default=None, help="with --position (deg; 0 looks along +Y)")
    p.add_argument("--forward-m", type=float, default=None, help="with --from-camera")
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--roll", type=float, default=0.0)
    p.add_argument("--replace-eye", action="store_true")
    p.add_argument("--launcher", default="auto")

    p = sub.add_parser("check", help="the scene gate S1-S6")
    p.add_argument("--blend", required=True)
    p.add_argument("--spec", default=None, help="docs/scenes/NAME.json")
    p.add_argument("--device", default="OPTIX", choices=["OPTIX", "CUDA", "CPU"])
    p.add_argument("--work", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--only", nargs="+", default=None, choices=[f"S{k}" for k in range(1, 7)])
    p.add_argument("--write-spec", default=None, metavar="DRAFT", help="a draft spec from S1 and S2, for review")
    p.add_argument("--launcher", default="auto")

    p = sub.add_parser("selftest", help="the package's gates (python -m fov_render.checks)")
    p.add_argument("--steps", nargs="+", default=None)
    p.add_argument("--device", default="CPU", choices=["OPTIX", "CUDA", "CPU"])
    p.add_argument("--work", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--launcher", default="auto")
    return ap


# ----------------------------------------------------------------------------- looks

def _look_request(a, panorama: bool, reasons: list) -> dict:
    """The request (without n and out_dir) the arguments describe; every problem goes to reasons."""
    if panorama:
        sensor = {"type": "panorama", "preset": a.preset}
        if a.width is not None:
            sensor["width"] = a.width
        if a.yaw_offset_px is not None:
            sensor["yaw_offset_px"] = a.yaw_offset_px
        if a.backface:
            sensor["backface"] = True
        a.point = a.gaze = a.per_eye = a.vergence = None
    else:
        sensor = {"type": SENSOR_NAMES[a.sensor], **_pairs(a.param, "--param", reasons)}
    reasons += check_sensor(sensor)
    for opt in ("head_move", "head_turn"):
        v = getattr(a, opt, None)
        if v is not None and not all(math.isfinite(x) for x in v):
            reasons.append(f"--{opt.replace('_', '-')}: must be finite numbers")
    gaze = _gaze(a, reasons)
    seeds = None if a.seeds is None else _pairs(a.seeds, "--seeds", reasons)
    return {"sensor": sensor, "eyes": list(a.eyes), "gaze": gaze, "spp": a.spp, "seeds": seeds, "rig": None,
            "outputs": ["exr", "records"] if a.records else ["exr"]}


def _normalised(look: dict) -> dict:
    """The sensor and gaze written out in full, as Session.look writes them (defaults included), when
    they are valid; left as they are otherwise (validation reports them)."""
    look = dict(look)
    if isinstance(look.get("sensor"), dict) and not check_sensor(look["sensor"]):
        look["sensor"] = sensor_to_dict(sensor_from_dict(look["sensor"]))
    g = look.get("gaze")
    if isinstance(g, dict) and not check_gaze(g) and g.get("mode") != "fixed":
        look["gaze"] = gaze_to_dict(gaze_from_dict(g))
    return look


def _full(n: int, look: dict, out_dir: Path) -> dict:
    return {"schema": REQUEST_SCHEMA, "n": n, "kind": "look", "spp": None, "seeds": None, "rig": None,
            "outputs": ["exr"], **_normalised(look), "out_dir": str(out_dir)}


def _validate(req: dict, where: str, reasons: list) -> None:
    try:
        validate_request(req)
    except RequestError as e:
        reasons += [f"{where}{r}" for r in e.reasons if f"{where}{r}" not in reasons]


def _looks_from_file(path: str, out: Path, reasons: list) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        reasons.append(f"{path}: {e}")
        return []
    looks = data.get("looks") if isinstance(data, dict) else data
    if not isinstance(looks, list) or not looks:
        reasons.append(f"{path}: must be a non-empty JSON list of looks (or {{\"looks\": [...]}})")
        return []
    reqs = []
    for k, item in enumerate(looks):
        if not isinstance(item, dict):
            reasons.append(f"looks[{k}]: must be an object")
            continue
        item = dict(item)
        item.setdefault("gaze", {"mode": "fixed"})
        od = Path(item.pop("out_dir", f"look-{k + 1:04d}"))
        req = _full(k + 1, item, od if od.is_absolute() else out / od)
        _validate(req, f"looks[{k}].", reasons)
        reqs.append(req)
    return reqs


def _session(a, blend: Path, reasons: list, first: dict | None):
    """The session; `first` (the first look's request) is set up when the session opens, so the ready
    answer, and the run record, carry the device Cycles chose (issue #18)."""
    cfg, rig = _config(a, reasons), _rig(a, reasons)
    if reasons:
        raise Refused(reasons)
    from .session import Session
    prepare = None if first is None else sensor_from_dict(first["sensor"])
    return Session(blend, cfg, rig, access=a.access, instance_ids=a.instance_ids, prepare=prepare,
                   launcher=a.launcher, workdir=a.workdir)


def _summary(resp: dict) -> str:
    eyes = ", ".join(f"{e}: {v['exr']}" for e, v in resp["eyes"].items())
    return f"look {resp['n']} done in {resp.get('wall_seconds', 0):.2f} s -> {eyes}"


def _moved_head(a, s) -> dict | None:
    """--head-move / --head-turn: the session's head moved, for this look (issue #20)."""
    if a.head_move is None and a.head_turn is None:
        return None
    yaw, pitch, roll = a.head_turn or (0.0, 0.0, 0.0)
    return s.session_head.moved(a.head_move or (0.0, 0.0, 0.0), yaw, pitch, roll).as_dict()


def cmd_look(a, panorama: bool) -> int:
    reasons: list = []
    blend = _blend(a.blend, reasons)
    out = Path(a.out).resolve()
    look = _look_request(a, panorama, reasons)
    _validate(_full(1, look, out), "", reasons)
    s = _session(a, blend, reasons, _full(1, look, out))
    try:
        head = _moved_head(a, s)
        if head is not None:
            look["head"] = head
        resp = s.request(_full(1, look, out))
        _write_cli(out, a, s, [resp])
        print(f"{PREFIX}: {_summary(resp)}", flush=True)
    finally:
        s.close()
    return 0


def cmd_requests(a) -> int:
    reasons: list = []
    blend = _blend(a.blend, reasons)
    out = Path(a.out).resolve()
    reqs = _looks_from_file(a.file, out, reasons)
    s = _session(a, blend, reasons, reqs[0] if reqs else None)
    done = []
    try:
        for req in reqs:
            done.append(s.request(req))
            print(f"{PREFIX}: {_summary(done[-1])}", flush=True)
    finally:
        _write_cli(out, a, s, done, planned=len(reqs))
        s.close()
    return 0


def _write_cli(out: Path, a, s, responses, planned=None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rec = {"schema": "fov-render-cli-run-v1", "argv": sys.argv[1:] if a.argv is None else a.argv,
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "launcher": s.launcher,
           "session": s.session, "ready": s.ready, "responses": responses,
           "looks_planned": len(responses) if planned is None else planned}
    (out / "fov-render-run.json").write_text(dumps(rec))


# ----------------------------------------------------------------------------- Blender-side subcommands

def _check_spec(path: str, reasons: list) -> None:
    from .checks.scene import spec_problems       # numpy only at import
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        reasons.append(f"--spec: {e}")
        return
    reasons += [f"--spec: {r}" for r in spec_problems(d)]


def _blender_argv(a) -> list[str]:
    reasons: list = []
    c = a.command
    if c in ("info", "eye", "check"):
        blend = _blend(a.blend, reasons)
    if c == "info":
        args = ["--blend", blend, "--instance-ids", a.instance_ids] + (["--catalogue"] if a.catalogue else []) \
            + (["--json", Path(a.json).resolve()] if a.json else [])
    elif c == "eye":
        if blend is not None and Path(a.out).resolve() == blend:
            reasons.append("--out: the copy must not overwrite the original")
        if a.position is not None and a.yaw is None:
            reasons.append("--position needs --yaw")
        if a.position is None and a.yaw is not None:
            reasons.append("--yaw: only with --position (--from-camera takes the camera's yaw)")
        if a.position is not None and a.forward_m is not None:
            reasons.append("--forward-m: only with --from-camera")
        args = ["--blend", blend, "--out", Path(a.out).resolve(), "--pitch", a.pitch, "--roll", a.roll]
        if a.position is not None:
            args += ["--position", *a.position, "--yaw", a.yaw]
        else:
            args += ["--from-camera"] + ([a.from_camera] if a.from_camera else []) + \
                    ["--forward-m", a.forward_m if a.forward_m is not None else 0.0]
        if a.replace_eye:
            args.append("--replace-eye")
    elif c == "check":
        if a.spec:
            _check_spec(a.spec, reasons)
        if a.write_spec and a.only and not {"S1", "S2"} <= set(a.only):
            reasons.append("--write-spec needs S1 and S2 (add them to --only)")
        args = ["--blend", blend, "--device", a.device]
        for k in ("spec", "work", "json", "write_spec"):
            if getattr(a, k):
                args += [f"--{k.replace('_', '-')}", Path(getattr(a, k)).resolve()]
        if a.only:
            args += ["--only", *a.only]
    else:
        args = ["--device", a.device] + (["--steps", *a.steps] if a.steps else [])
        for k in ("work", "json"):
            if getattr(a, k):
                args += [f"--{k}", Path(getattr(a, k)).resolve()]
    if reasons:
        raise Refused(reasons)
    return [c, *[str(x) for x in args]]


def cmd_blender(a) -> int:
    argv = _blender_argv(a)
    cmd, env, _ = launch.command(a.launcher, BLCLI_SCRIPT, "fov_render.blcli", argv)
    if a.command == "info" and not a.json:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        text = p.stdout
        if p.returncode == 0 and "FOVRENDER-JSON-BEGIN\n" in text:
            sys.stdout.write(text.split("FOVRENDER-JSON-BEGIN\n", 1)[1].split("FOVRENDER-JSON-END\n", 1)[0])
        else:
            sys.stderr.write(text[-4000:] + p.stderr[-4000:])
            print(f"{PREFIX}: info failed (exit {p.returncode or 1})", file=sys.stderr)
        return p.returncode if p.returncode else (0 if "FOVRENDER-JSON-BEGIN\n" in text else 1)
    return subprocess.run(cmd, env=env).returncode


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    from .session import SessionError
    try:
        a = parser().parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    a.argv = None if argv is None else [str(x) for x in argv]
    try:
        if a.command in ("look", "panorama"):
            return cmd_look(a, a.command == "panorama")
        if a.command == "requests":
            return cmd_requests(a)
        return cmd_blender(a)
    except (Refused, RequestError) as e:
        for r in e.reasons:
            print(f"{PREFIX}: refused: {r}", file=sys.stderr)
        return 2
    except (SessionError, launch.LaunchError, OSError) as e:
        print(f"{PREFIX}: failed: {e}", file=sys.stderr)
        return 1


def entry() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entry()
