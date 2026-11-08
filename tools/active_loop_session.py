"""Run fov-3d-vision's foveated active loop with fov_render rendering it (issue #3, migration step 6).

fov-3d-vision is not changed. This script imports its tools/active_loop.py and replaces what talks to
Blender: fixation_pairs.PairRenderer's constructor (the legacy OSL camera and scene set-up) and three
functions of render_foveated (set_gaze, render_fixation, save_render_result). Every fixation is then
one request to a fov_render Session.in_process. Everything else runs as fov-3d-vision's own code:
the policy, the belief, the stereo field, the record bookkeeping, the files it writes, and
active_eval.py on the result.

    blender -b SCENE.blend --python-exit-code 1 -P tools/active_loop_session.py -- \
        --fov3d /path/to/fov-3d-vision --mode exact|native  <active_loop.py's own arguments>
    python tools/active_loop_session.py -- --blend SCENE.blend --fov3d ... --mode ... <...>   (the bpy wheel)

--mode exact    the camera is posed by compat.fov3d_gaze_matrix (fov-3d-vision's float32 mathutils
                composition) and the records are built by fov-3d-vision's warp tables, so on CPU a
                run must equal the legacy loop's bit for bit (#3's exact test).
--mode native   fov_render's own Rig poses the camera (float64) and the records come from
                fov_render's SampleRecord (directions, footprints, range from Position), converted
                to the loop's D1 format. NOT validated statistically (Luiz, #3): its fixation-0
                records agree with legacy's (gate 6.3), but the loop is chaotic and whether its
                Phase C numbers are unbiased is not established.
The run folder also gets fov-render-loop.json: the mode, fov-render's commit and the session.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREFIX = "[loop-session]"


def _take_options() -> dict:
    """Remove this script's own options from sys.argv (active_loop parses the rest strictly)."""
    if "--" not in sys.argv:
        raise SystemExit(f"{PREFIX} arguments go after --")
    i = sys.argv.index("--")
    rest, mine, k = sys.argv[i + 1:], {"fov3d": None, "mode": None}, 0
    keep = []
    while k < len(rest):
        if rest[k] in ("--fov3d", "--mode") and k + 1 < len(rest):
            mine[rest[k][2:]] = rest[k + 1]
            k += 2
        else:
            keep.append(rest[k])
            k += 1
    if not mine["fov3d"] or mine["mode"] not in ("exact", "native"):
        raise SystemExit(f"{PREFIX} need --fov3d PATH and --mode exact|native")
    sys.argv[i + 1:] = keep
    return mine


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> None:
    opt = _take_options()
    from fov_render.bl import env
    env.prepare()
    tools = Path(opt["fov3d"]).resolve() / "tools"
    sys.path.insert(0, str(tools))
    import numpy as np
    import bpy
    import active_loop as AL
    import fixation_pairs as FP
    import render_foveated as RF
    from fov_render import Gaze, Rig, Session, preset, sensors
    exact = opt["mode"] == "exact"

    class SessionPairRenderer(FP.PairRenderer):
        """PairRenderer with a fov_render Session in place of the legacy camera set-up. Its other
        methods (pair_for, warm_up, grab, render_pair, seed_pass, finish) are fov-3d-vision's,
        unchanged; they reach Blender only through the three functions patched below."""

        def __init__(self, args, out):
            from bl_common import eye_record, find_eye, rigid
            from rig import EYE_NAMES, eye_centres, eye_offsets_local
            from warp import centre_pixels, raster_samples, raster_size, s0_of
            self.args, self.out = args, os.path.abspath(out)
            self.verged = args.vergence == "on"
            for e in EYE_NAMES:
                os.makedirs(os.path.join(self.out, e), exist_ok=True)
            self.t_seq0 = time.perf_counter()
            self.scene = bpy.context.scene
            self.eye, self.eye_note = find_eye(self.scene)
            n_legacy = args.n if args.n else raster_size(args.s0, args.e2, args.emax)
            self.spec = sensors.Foveated(E2_deg=args.e2, e_max_deg=args.emax, s0_deg=args.s0, n=n_legacy)
            self.cfg = preset("fov_fixation", profile=args.profile or "small", device=args.device, spp=args.spp,
                              seed=0, exr_codec="NONE")
            self.session = Session.in_process(self.cfg, Rig(ipd_m=args.ipd), blend=None, access="oracle",
                                              prepare=self.spec, pose_convention="fov3d" if exact else "fov_render",
                                              workdir=os.path.join(self.out, "_fov_render"))
            model = self.session.handler.model
            if model.n != n_legacy:
                raise RuntimeError(f"fov_render's raster {model.n} is not the loop's {n_legacy}")
            self.backend = self.session.ready["device"]
            if exact:
                eye_m = rigid(self.eye.matrix_world)
                self.head_rot3 = np.array([[eye_m[r][c] for c in range(3)] for r in range(3)], dtype=np.float64)
                self.head_origin = np.array([eye_m.translation[k] for k in range(3)], dtype=np.float64)
                self.centres = eye_centres(self.head_origin, self.head_rot3, args.ipd)
            else:
                head = self.session.handler.head
                self.head_rot3, self.head_origin = np.asarray(head.R_wh, float), np.asarray(head.origin_w, float)
                rig = self.session.rig
                self.centres = np.array([head.to_world(rig.centre_h(e)) for e in EYE_NAMES])
            self.offsets = eye_offsets_local(args.ipd)
            self.n = n_legacy
            self.s0 = s0_of(self.n, args.e2, args.emax)
            self.rs = raster_samples(self.n, args.e2, args.emax)
            self.inside = self.rs["inside"]
            self.n_inside = int(self.inside.sum())
            self.cp = centre_pixels(self.n)
            self.cam = self.session.handler.cam
            self.seq_fix = {e: [] for e in EYE_NAMES}
            self.warmup = None
            self.eye_record = eye_record
            self.blend = bpy.data.filepath
            self.blender = bpy.app.version_string
            self._pending = None
            self._last = None
            self.looks = 0
            STATE["renderer"] = self
            (Path(self.out) / "fov-render-loop.json").write_text(json.dumps({
                "schema": "fov3d-loop-on-fov-render-v1", "mode": opt["mode"], "fov3d_tools": str(tools),
                "fov_render_commit": _git("rev-parse", "HEAD"),
                "fov_render_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
                "session": self.session.session, "ready_device": self.backend,
                "head": self.session.ready["head"]}, indent=1, sort_keys=True, default=float) + "\n")

        def _render(self, yaw, pitch, offset, spp, seed) -> float:
            en = "L" if offset[0] < 0 else "R" if offset[0] > 0 else "C"
            outputs = ("exr",) if exact else ("exr", "records")
            look = self.session.look(self.spec, (en,), Gaze.per_eye(**{en: (float(yaw), float(pitch))}), spp=int(spp),
                                     seeds={en: int(seed)}, outputs=outputs)
            self.looks += 1
            self._last = (en, look)
            return float(look.response["eyes"][en]["render_seconds"])

        def grab(self, pid, k, g, point_m, exr_path):
            rec, measured = super().grab(pid, k, g, point_m, exr_path)
            if exact:
                return rec, measured
            # native: the record from fov_render's SampleRecord, in the loop's D1 format
            en, look = self._last
            fr = look.records[0]
            r, c = np.divmod(np.asarray(fr.raster_index), fr.raster_shape[1])
            dist = np.where(np.isfinite(fr.distance_m), fr.distance_m, 1e10)
            m = int(fr.n_samples)
            rec.update({"origin": np.asarray(fr.origin, np.float32), "direction": fr.direction_head.astype(np.float32),
                        "value": np.asarray(fr.value)[:, :3].astype(np.float32),
                        "footprint": np.asarray(fr.footprint_sr, np.float32), "distance": dist.astype(np.float32),
                        "fixation_id": np.full(m, pid, np.int32), "raster_index": np.stack([r, c], -1).astype(np.int32),
                        "eye_id": np.full(m, k, np.int32), "pair_id": np.full(m, pid, np.int32)})
            return rec, measured

    STATE: dict = {}

    def set_gaze(cam, eye, yaw_deg, pitch_deg, offset_local=(0.0, 0.0, 0.0)):
        STATE["renderer"]._pending = (yaw_deg, pitch_deg, tuple(offset_local))

    def render_fixation(scene, spp, out_path=None, seed=0):
        R = STATE["renderer"]
        if R._pending is None:
            raise RuntimeError("render_fixation before set_gaze")
        secs = R._render(*R._pending, spp, seed)
        if out_path:
            save_render_result(scene, out_path)
        return secs

    def save_render_result(scene, out_path):
        R = STATE["renderer"]
        en, look = R._last
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        shutil.move(str(look.exr[en]), out_path)            # one copy on disk, where the loop wants it

    RF.set_gaze, RF.render_fixation, RF.save_render_result = set_gaze, render_fixation, save_render_result
    AL.PairRenderer = SessionPairRenderer
    print(f"{PREFIX} fov-3d-vision's active loop ({tools}) on a fov_render Session, mode {opt['mode']}", flush=True)
    AL.main()


def run():
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print(f"{PREFIX} FAILED", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    run()
