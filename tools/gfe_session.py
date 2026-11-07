"""Run f3d-vision's greedy foveal explorer (Playground-2) on a fov_render Session (issue #2, step 5).

f3d-vision is not changed (PLAN D8). This script imports its tools/greedy_foveal/run.py, replaces
the one class that talks to Blender (run.Server, which starts render_server.py) with SessionServer
below, and runs run.py's own command line:

    .venv/bin/python tools/gfe_session.py --f3d /path/to/f3d-vision [--launcher blender] \\
        explore --run RUN --max-fix 600 --spp 64

Only `explore` renders; `evaluate`, `checks` and `visualize` are run.py's, unchanged (run them with
f3d-vision's own run.py, or through this script). Then f3d-vision's check_equivalence.py compares
RUN with the frozen Playground-2 run.

What fov_render does per fixation, in place of the legacy server: the eye poses come from
fov_render's Rig (baseline-projected roll, IPD 63 mm, vergence 2.10 m), calibration.json from
compat.f3d_calibration, and the two renders from one Session request (preset "ab1a", OPTIX, the
explorer's spp, seeds L 2111 / R 2112, legacy Object Index numbering, D11). The server's ready
record is written in the legacy format by compat.ab1a_readback and compat.gfe_passes. Each piece is
checked against the legacy function it replaces in fov_render.checks.step5.

Options for a smoke test without the Classroom (Chat sandbox): --blend another scene, --head-json
a file with head_R_wh / head_origin_w_m to use as the "accepted" head, --device CPU.
--control-ipd-ulp makes the control run of the gate: the IPD one ULP above 63 mm, which changes
every calibration.json and eye position, so check_equivalence must fail on it. (A seed change is
no control here: seeds change only RGB, which check_equivalence reports but does not compare.)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fov_render import Gaze, HeadPose, Rig, Session, compat, preset  # noqa: E402
from fov_render.rig import cv_to_blender  # noqa: E402
from fov_render.sensors import Perspective  # noqa: E402

PREFIX = "[gfe-session]"
SEEDS = {"L": 2111, "R": 2112}          # ab1a_spec.SEEDS
IPD_M, VERGENCE_M, ROLL = 0.063, 2.10, "baseline_projected"     # ab1a_spec IPD_M, VERGENCE_M, TANGENT_FRAME
OPT = argparse.Namespace(launcher="auto", device="OPTIX", blend=None, head_json=None, control_ipd_ulp=False)


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


class SessionServer:
    """run.Server's interface (ready, start_seconds, render, stop) on a fov_render Session."""

    def __init__(self, run: Path, spp: int) -> None:
        import run as GR
        q = run / "queue"
        q.mkdir(parents=True)                        # as run.Server: refuses an existing queue
        self.log = open(run / "blender.log", "w")
        self.spp = int(spp)
        self.cfg = preset("ab1a", device=OPT.device, spp=self.spp)
        ipd = float(np.nextafter(IPD_M, 1.0)) if OPT.control_ipd_ulp else IPD_M
        self.rig = Rig(ipd_m=ipd, roll=ROLL)
        self.sensor = Perspective()
        self.model = self.sensor.model(self.cfg.profile)
        blend = Path(OPT.blend) if OPT.blend else GR.BLEND
        self.s = Session(blend, self.cfg, self.rig, access="oracle", instance_ids="legacy", prepare=self.sensor,
                         formats=("f3d_ab1a",), launcher=OPT.launcher, workdir=q, log=self.log)
        r = self.s.ready
        f = r["formats"]["f3d_ab1a"]
        self.head = HeadPose(np.asarray(r["head"]["head_R_wh"], float), np.asarray(r["head"]["head_origin_w_m"], float))
        self.ready = {"blend": r["scene"]["file"]["path"], "blender": r["blender"], "device": r["device"],
                      "spp": self.spp, "seeds": dict(SEEDS),
                      "eye_pose": {"eye_note": r["scene"]["eye_note"], **r["head"]},
                      "passes": f["passes"], "settings": f["settings"], "pid": r["pid"]}
        self.start_seconds = self.s.start_seconds
        (run / "fov-render-session.json").write_text(json.dumps({
            "schema": "gfe-on-fov-render-v1", "fov_render_commit": _git("rev-parse", "HEAD"),
            "fov_render_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "launcher": self.s.launcher, "session": self.s.session, "ready": r}, indent=1, sort_keys=True) + "\n")

    def render(self, n: int, yaw: float, pitch: float, out: Path) -> dict:
        t0 = time.perf_counter()
        gaze = Gaze.yaw_pitch(yaw, pitch, VERGENCE_M)
        c = compat.f3d_calibration(self.cfg.profile, self.model, self.rig, self.head, gaze)
        out.mkdir(parents=True, exist_ok=True)
        (out / "calibration.json").write_text(compat.f3d_json(c))
        look = self.s.look(self.sensor, ("L", "R"), gaze, spp=self.spp, seeds=SEEDS, out_dir=out)
        for e in c["eyes"]:                          # the server rendered the calibration's poses, bit for bit
            got = cv_to_blender(np.asarray(look.response["eyes"][e["name"]]["pose"]["R_hb"], float))
            if got.tobytes() != np.asarray(e["R_hc"], float).tobytes():
                raise RuntimeError(f"fixation {n} eye {e['name']}: the rendered pose is not the calibration's")
        return {"n": n, "render_seconds_lr": {k: v["render_seconds"] for k, v in look.response["eyes"].items()},
                "wall_seconds": time.perf_counter() - t0}

    def stop(self) -> int:
        rc = self.s.close()
        self.log.close()
        return rc


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f3d", required=True, help="an f3d-vision checkout (its tools/greedy_foveal/run.py is used)")
    ap.add_argument("--launcher", default="auto", help="auto, blender, python or an executable (fov_render.Session)")
    ap.add_argument("--device", default="OPTIX", choices=["OPTIX", "CUDA", "CPU"])
    ap.add_argument("--blend", default=None, help="smoke tests only: another scene than run.py's Classroom")
    ap.add_argument("--head-json", default=None, help="smoke tests only: the head to treat as accepted")
    ap.add_argument("--control-ipd-ulp", action="store_true",
                    help="control run: IPD one ULP above 63 mm; check_equivalence must FAIL on it")
    a, rest = ap.parse_known_args(argv)
    OPT.launcher, OPT.device, OPT.blend, OPT.head_json = a.launcher, a.device, a.blend, a.head_json
    OPT.control_ipd_ulp = a.control_ipd_ulp
    gf = Path(a.f3d).resolve() / "tools" / "greedy_foveal"
    if not (gf / "run.py").is_file():
        raise SystemExit(f"{PREFIX} no tools/greedy_foveal/run.py under {a.f3d}")
    sys.path.insert(0, str(gf))
    import run as GR
    GR.Server = SessionServer
    if a.head_json:
        h = json.loads(Path(a.head_json).read_text())
        GR.accepted_head_pose = lambda: (h["head_R_wh"], h["head_origin_w_m"])
    sys.argv = [str(gf / "run.py"), *rest]
    print(f"{PREFIX} run.py {' '.join(rest)} on a fov_render Session ({a.launcher}, {a.device})", flush=True)
    GR.main()


if __name__ == "__main__":
    main()
