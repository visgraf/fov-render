"""Greedy Foveal Explorer, official baseline: the Blender side, a persistent binocular render server.

Started by ``run.py``; never run by hand except for debugging:

    blender -b scenes/classroom/classroom_eye.blend --python-exit-code 1 \
        -P tools/greedy_foveal/render_server.py -- --queue RUN/queue --spp 64

Classroom is loaded once.  The accepted acquisition helpers are reused read-only: ``ns1a_render.eye_pose`` (the fixed
physical head, checked against the accepted AB1a calibration), ``ab1a_render.calibration`` / ``configure`` (the accepted
perspective pair, OPTIX, box filter, no denoise) and ``classroom_oracle1_render._assign_instance_ids`` / ``_eye_matrix``
with ``render_foveated.render_fixation``.  Only the eye cameras move between renders (fixed head H0, eye gaze only).

Protocol: the host writes ``QUEUE/req-NNNN.json`` ({"n", "yaw_deg", "pitch_deg", "out_dir"}); the server writes
``OUT/calibration.json``, ``OUT/raw_L.exr``, ``OUT/raw_R.exr`` and then ``QUEUE/done-NNNN.json``.  ``QUEUE/stop`` ends
the loop.  Any failure writes ``QUEUE/fail-NNNN.json`` and exits nonzero.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "active_bootstrap", HERE.parent / "north_star", HERE.parent):
    sys.path.insert(0, str(_p))

import ab1a_render as AR  # noqa: E402  (accepted AB1a acquisition helpers, read-only)
import ab1a_spec as AS  # noqa: E402
import ns1a_render as NR  # noqa: E402  (accepted NS1a driver helpers, read-only)

PREFIX = "[gfe-render]"
KEEP_PASSES = ("use_pass_combined", "use_pass_position", "use_pass_object_index")


def write_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def trim_passes() -> dict:
    """Keep only Combined, Position and Object Index (the Classroom view layer enables extra light passes)."""
    import bpy
    vl = bpy.context.view_layer
    off = []
    for name in dir(vl):
        if name.startswith("use_pass_") and name not in KEEP_PASSES:
            try:
                if getattr(vl, name) is True:
                    setattr(vl, name, False)
                    off.append(name)
            except (AttributeError, TypeError):
                pass
    cyc = getattr(vl, "cycles", None)
    for name in (dir(cyc) if cyc is not None else []):
        if name.startswith("use_pass_") or name == "denoising_store_passes":
            try:
                if getattr(cyc, name) is True:
                    setattr(cyc, name, False)
                    off.append("cycles." + name)
            except (AttributeError, TypeError):
                pass
    for aov in list(getattr(vl, "aovs", [])):
        off.append("aov:" + aov.name)
    vl.update()
    return {"disabled": off, "kept": [n for n in KEEP_PASSES if getattr(vl, n)]}


def serve(a) -> None:
    import bpy
    import classroom_oracle1_render as CO
    import render_foveated as RF
    q = Path(a.queue).resolve()
    q.mkdir(parents=True, exist_ok=True)
    spp = int(a.spp)
    pose = NR.eye_pose(strict=True)                      # fixed physical head == accepted AB1a head pose
    CO._assign_instance_ids()                            # Object Index oracle labels (visualization only downstream)
    c0 = AR.calibration(0.0, 0.0, pose)
    scene, cam, backend, matrix = AR.configure(c0, spp)
    passes = trim_passes()
    write_atomic(q / "server-ready.json", {
        "blend": bpy.data.filepath, "blender": bpy.app.version_string, "device": backend, "spp": spp,
        "seeds": dict(AS.SEEDS), "eye_pose": pose, "passes": passes, "settings": AR.readback(scene, cam),
        "pid": os.getpid()})
    print(f"{PREFIX} READY spp={spp} device={backend} passes_disabled={len(passes['disabled'])}", flush=True)
    n = 1
    while True:
        if (q / "stop").exists():
            print(f"{PREFIX} STOP after {n - 1} requests", flush=True)
            return
        req_p = q / f"req-{n:04d}.json"
        if not req_p.exists():
            time.sleep(0.01)
            continue
        t0 = time.perf_counter()
        req = json.loads(req_p.read_text())
        try:
            if int(req["n"]) != n:
                raise RuntimeError(f"request number {req['n']} != {n}")
            out = Path(req["out_dir"])
            out.mkdir(parents=True, exist_ok=True)
            c = AR.calibration(float(req["yaw_deg"]), float(req["pitch_deg"]), pose)
            AR.write_json(out / "calibration.json", c)
            secs = {}
            for eye in c["eyes"]:
                side = eye["name"]
                cam.matrix_world = CO._eye_matrix(c, eye, matrix)
                bpy.context.view_layer.update()
                secs[side] = float(RF.render_fixation(scene, spp, str(out / f"raw_{side}.exr"),
                                                      seed=int(AS.SEEDS[side])))
                if scene.cycles.samples != spp:
                    raise RuntimeError("the sample count did not hold")
            write_atomic(q / f"done-{n:04d}.json", {"n": n, "render_seconds_lr": secs,
                                                    "wall_seconds": time.perf_counter() - t0})
        except BaseException:
            write_atomic(q / f"fail-{n:04d}.json", {"n": n, "traceback": traceback.format_exc()})
            raise
        n += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--spp", type=int, required=True)
    serve(ap.parse_args(AR.args_after_dashes()))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise
        traceback.print_exc()
        print(f"{PREFIX} FAILED", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
