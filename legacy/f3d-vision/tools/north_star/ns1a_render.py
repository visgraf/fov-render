"""North Star-1a: the Blender side (no-render preflight, the one canonical six-gaze acquisition, rehearsal).

Contract: docs/north-star/ns1a-perfect-bootstrap-round-contract.md, sections 4b, 5 and 6.  Normally run through
``ns1a_run.py``:

    blender -b scenes/classroom/classroom_eye.blend --python-exit-code 1 \
        -P tools/north_star/ns1a_render.py -- --mode preflight --run RUN
    blender -b scenes/classroom/classroom_eye.blend --python-exit-code 1 \
        -P tools/north_star/ns1a_render.py -- --mode canonical --run RUN
    blender -b --factory-startup --python-exit-code 1 \
        -P tools/north_star/ns1a_render.py -- --mode rehearsal --run DEV_RUN --spp 16

A thin driver over the accepted acquisition helpers, imported read-only: ``ab1a_render`` (``calibration``,
``configure``, ``readback``, ``acquire_pair``, ``calib_diff``, ``write_json``, ``new_dir``, ``_emission_noise``) and
``classroom_oracle1_render`` (``_head_pose``, ``_assign_instance_ids``, ``_eye_matrix``).  No accepted renderer is
modified.  The head pose comes from the accepted AB1a calibration (NOT from the Controller-01 seeds file).  ``canonical``
renders the six frozen NB1c gazes once each, in frozen order, at 4096 spp, into new directories, and splits each pair
at once into the INFERENCE / SENSORY observation and the ORACLE AID.  The instance catalog is written to a sealed
evaluation-only file whose sha256 Blender records.  ``rehearsal`` builds a synthetic factory-startup scene (never
Classroom).  Every failure exits nonzero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "active_bootstrap", HERE.parent, HERE):
    sys.path.insert(0, str(_p))

import ab1a_render as AR  # noqa: E402  (accepted AB1a Blender acquisition helpers, read-only)
import ns1a_spec as SP  # noqa: E402

PREFIX = "[ns1a-render]"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def max_abs(a, b) -> float:
    return max(abs(float(x) - float(y)) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def head_source() -> tuple[dict, bytes]:
    """The accepted AB1a calibration (fixed head), hash-verified; its exact bytes for the rank-1 identity."""
    path, want = SP.HEAD_POSE_SOURCE
    b = Path(path).read_bytes()
    if hashlib.sha256(b).hexdigest() != want:
        raise RuntimeError(f"STOP the accepted head-pose source changed: {path}")
    return json.loads(b), b


def eye_pose(strict: bool = True) -> dict:
    """The Blender EYE pose, required equal to the head pose of the accepted AB1a calibration."""
    import classroom_oracle1_render as CO
    _eye, note, o, r = CO._head_pose()
    src, _ = head_source()
    r_acc, o_acc = np.asarray(src["head_R_wh"], np.float64), np.asarray(src["head_origin_w_m"], np.float64)
    dr, do = float(np.abs(r - r_acc).max()), float(np.abs(o - o_acc).max())
    if strict and max(dr, do) > SP.EYE_POSE_TOL:
        raise RuntimeError(f"STOP the EYE pose differs from the accepted head pose: |dR| {dr}, |do| {do}")
    return {"eye_note": note, "head_R_wh": r.tolist(), "head_origin_w_m": o.tolist(),
            "head_pose_source": str(SP.HEAD_POSE_SOURCE[0]), "head_pose_source_sha256": SP.HEAD_POSE_SOURCE[1],
            "max_abs_diff_from_accepted": {"R": dr, "origin": do}}


def gaze_list(run: Path) -> list[dict]:
    """The frozen six from the host-side ``source`` record, verified again against the literal table."""
    d = json.loads((run / "source/nb1c-gaze-list.json").read_text())
    got = [(g["rank"], g["row"], g["col"], g["yaw_deg"], g["pitch_deg"]) for g in d["gazes"]]
    if got != [tuple(g) for g in SP.GAZES]:
        raise RuntimeError(f"STOP the gaze list differs from the frozen NB1c six: {got}")
    return d["gazes"]


def prepare(run: Path, pose: dict, strict: bool) -> list[tuple[dict, dict, dict]]:
    """Every calibration built from the EYE, against the host plan; rank 1 byte-identical to AB1a (strict)."""
    _, ab1a_bytes = head_source()
    out = []
    for g in gaze_list(run):
        r = int(g["rank"])
        c = AR.calibration(g["yaw_deg"], g["pitch_deg"], pose)
        plan = json.loads((run / "plan" / SP.rank_dir(r) / "planned-calibration.json").read_text())
        diff = AR.calib_diff(plan, json.loads(json.dumps(c)))
        b = (json.dumps(c, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
        rep = {"rank": r, "gaze_yaw_pitch_deg": list(c["gaze_yaw_pitch_deg"]),
               "calibration_max_abs_diff_from_planned": diff,
               "calibration_sha256": hashlib.sha256(b).hexdigest(),
               "byte_identical_to_planned": b == (run / "plan" / SP.rank_dir(r) / "planned-calibration.json").read_bytes()}
        if r == 1:
            rep["rank1_byte_identical_to_ab1a"] = b == ab1a_bytes
        ok = diff <= SP.CALIBRATION_TOL and (r != 1 or rep["rank1_byte_identical_to_ab1a"])
        rep["ok"] = bool(ok)
        if strict and not ok:
            raise RuntimeError("STOP BEFORE RENDERING: calibration identity failed " + json.dumps(rep, sort_keys=True))
        out.append((g, c, rep))
    return out


def accepted_settings() -> dict:
    path, want = SP.AB1D2_SETTINGS_SOURCE
    if sha256(Path(path)) != want:
        raise RuntimeError(f"STOP the accepted AB1d2 acquisition record changed: {path}")
    return json.loads(Path(path).read_text())["settings"]


def ab1a_matrices() -> dict:
    path, want = SP.AB1A_ACQUISITION
    if sha256(Path(path)) != want:
        raise RuntimeError(f"STOP the accepted AB1a acquisition record changed: {path}")
    return json.loads(Path(path).read_text())["camera_matrix_world_lr"]


def configure_and_check(c: dict, spp: int, strict: bool, rank: int) -> dict:
    """Configure the accepted perspective pair; settings must equal the accepted 4096-spp readback (strict)."""
    import bpy
    import classroom_oracle1_render as CO
    scene, cam, backend, matrix = AR.configure(c, spp)
    mats = {}
    for eye in c["eyes"]:
        cam.matrix_world = CO._eye_matrix(c, eye, matrix)
        bpy.context.view_layer.update()
        mats[eye["name"]] = [[cam.matrix_world[i][j] for j in range(4)] for i in range(4)]
    rb = AR.readback(scene, cam)
    rep = {"device": backend, "settings": rb, "camera_matrix_world_lr": mats}
    if strict:
        acc = accepted_settings()
        diff = {k: [acc.get(k), rb.get(k)] for k in sorted(set(acc) | set(rb)) if acc.get(k) != rb.get(k)}
        rep["settings_differences_from_accepted_4096"] = diff
        if diff or backend != SP.DEVICE or rb["samples"] != SP.SPP:
            raise RuntimeError(f"STOP rank {rank} settings differ from the accepted 4096-spp readback: {diff}")
        if rank == 1:
            m1 = ab1a_matrices()
            md = {s: max_abs(mats[s], m1[s]) for s in ("L", "R")}
            rep["rank1_camera_matrix_max_abs_diff_vs_ab1a"] = md
            if max(md.values()) > SP.EYE_POSE_TOL:
                raise RuntimeError(f"STOP rank-1 camera matrices differ from the AB1a record: {md}")
    return rep


def check_record(rec: dict, strict: bool, rank: int) -> dict:
    """After ``acquire_pair``: its own settings readback and camera matrices against the accepted records (strict)."""
    rep = {"device": rec["device"], "samples": rec["settings"]["samples"]}
    if strict:
        acc = accepted_settings()
        rb = rec["settings"]
        diff = {k: [acc.get(k), rb.get(k)] for k in sorted(set(acc) | set(rb)) if acc.get(k) != rb.get(k)}
        rep["settings_differences_from_accepted_4096"] = diff
        if diff or rec["device"] != SP.DEVICE or rec["spp"] != SP.SPP:
            raise RuntimeError(f"STOP rank {rank} settings differ from the accepted 4096-spp readback: {diff}")
        if rank == 1:
            m1 = ab1a_matrices()
            md = {s: max_abs(rec["camera_matrix_world_lr"][s], m1[s]) for s in ("L", "R")}
            rep["rank1_camera_matrix_max_abs_diff_vs_ab1a"] = md
            if max(md.values()) > SP.EYE_POSE_TOL:
                raise RuntimeError(f"STOP rank-1 camera matrices differ from the AB1a record: {md}")
    return rep


def exr_samples(path: Path) -> str | None:
    from exr_lite import read_header
    with open(path, "rb") as f:
        attrs = read_header(f.read(65536))[0]
    v = attrs.get(SP.EXR_SAMPLES_ATTR)
    return None if v is None else (v.decode() if isinstance(v, bytes) else str(v))


class Budget:
    """Project the total render time after every eye; a projection above the budget is a STOP."""

    def __init__(self, eyes_total: int, budget_s: float) -> None:
        self.total, self.budget, self.done, self.seconds, self.t0 = eyes_total, budget_s, 0, [], time.perf_counter()

    def eye_done(self, s: float) -> None:
        self.done += 1
        self.seconds.append(float(s))
        elapsed = time.perf_counter() - self.t0
        projected = elapsed / self.done * self.total
        print(f"{PREFIX} eye {self.done}/{self.total}: {s:.1f} s; projected total {projected:.0f} s", flush=True)
        if projected > self.budget:
            raise RuntimeError(f"STOP the projected render time {projected:.0f} s exceeds {self.budget:.0f} s")

    def record(self) -> dict:
        return {"eyes_rendered": self.done, "render_seconds": self.seconds,
                "wall_seconds": round(time.perf_counter() - self.t0, 3), "budget_s": self.budget}


def acquisition(run: Path, spp: int, strict: bool, extra_record: dict) -> dict:
    """The six frozen gazes, once each, in frozen order (both canonical and rehearsal use this path)."""
    import classroom_oracle1_render as CO
    import render_foveated as RF
    obs = run / "observations"
    if obs.exists() and any(obs.iterdir()):
        raise FileExistsError(f"STOP {obs} exists: each frozen gaze is observed once; never re-rendered")
    pose = eye_pose(strict=True)
    prepared = prepare(run, pose, strict=strict)        # all six, before any scene change
    print(f"{PREFIX} calibration identity verified for all six gazes before rendering", flush=True)
    ev_all = AR.new_dir(obs / "evaluation_only")
    by_id, _by_name = CO._assign_instance_ids()          # the Object Index oracle mechanism (unchanged)
    AR.write_json(ev_all / "instance-catalog.json", {
        "schema": "NS1a-instance-catalog-v1", "truth": SP.TRUTH_REFERENCE,
        "statement": "SEALED until the NS1a seed freeze: opened only by `evaluate`",
        "instances": [{"instance_id": int(k), "object_name": v} for k, v in sorted(by_id.items())]})
    catalog_sha = sha256(ev_all / "instance-catalog.json")
    budget = Budget(2 * len(prepared), SP.RENDER_BUDGET_S)
    original = RF.render_fixation

    def timed(scene, spp_, out_path=None, seed=0):
        s = original(scene, spp_, out_path, seed=seed)
        budget.eye_done(s)
        return s

    RF.render_fixation = timed                           # in-process timing wrapper only; the accepted file is unchanged
    per_rank = []
    try:
        for g, c, rep in prepared:
            r = int(g["rank"])
            acq = AR.new_dir(obs / SP.rank_dir(r) / "acquisition")
            aid = AR.new_dir(obs / SP.rank_dir(r) / "oracle_aid")
            rec = AR.acquire_pair(c, acq, aid, spp, {
                "experiment": SP.EXPERIMENT, "action_source": g["action_source"], "nb1c_rank": r,
                "nb1c_row_col": [g["row"], g["col"]], "eye_pose": pose, "canonical": bool(strict),
                "calibration_max_abs_diff_from_planned": rep["calibration_max_abs_diff_from_planned"],
                "oracle_aid_dir": "oracle_aid", "product_domains": {
                    "acquisition": "INFERENCE / SENSORY: calibration.json, rgb-observation.npz",
                    "oracle_aid": "ORACLE AID (ORACLE INPUT): raw_L.exr, raw_R.exr, reference-observation.npz"},
                **extra_record})
            cfg = check_record(rec, strict, r)
            cal_sha = sha256(acq / "calibration.json")
            if cal_sha != rep["calibration_sha256"]:
                raise RuntimeError(f"rank {r}: the written calibration differs from the verified one")
            per_rank.append({
                "rank": r, "gaze_yaw_pitch_deg": rec["gaze_yaw_pitch_deg"], "spp": rec["spp"], "device": rec["device"],
                "render_seconds_lr": rec["render_seconds_lr"], "calibration_sha256": cal_sha,
                "rgb_observation_sha256": sha256(acq / "rgb-observation.npz"),
                "reference_observation_sha256": sha256(aid / "reference-observation.npz"),
                "exr_samples_lr": {s: exr_samples(aid / f"raw_{s}.exr") for s in ("L", "R")},
                "configuration": cfg,
                "calibration_identity": rep})
            print(f"{PREFIX} observed rank {r} " + json.dumps(rec["render_seconds_lr"], sort_keys=True), flush=True)
    finally:
        RF.render_fixation = original
    out = {"schema": "NS1a-acquisition-run-v1", "truth": SP.TRUTH_ORACLE, "experiment": SP.EXPERIMENT,
           "statement": "the six frozen NB1c gazes, once each, in frozen order; one Blender process; fixed head, "
                        "static scene; each pair split at once into INFERENCE / SENSORY and ORACLE AID",
           "blend": __import__("bpy").data.filepath, "blender": __import__("bpy").app.version_string,
           "spp": int(spp), "ranks_in_order": [p["rank"] for p in per_rank], "per_rank": per_rank,
           "catalog_seal": {"path": SP.CATALOG_REL, "sha256": catalog_sha, "instances": len(by_id),
                            "statement": "the id -> name catalog is sealed until the seed freeze"},
           "budget": budget.record(), **extra_record}
    AR.write_json(run / SP.ACQ_RUN_REL, out)
    return out


def preflight(a) -> None:
    import bpy
    run = Path(a.run).resolve()
    out = run / "preflight"
    if (out / "preflight.json").exists():
        raise FileExistsError(f"STOP {out / 'preflight.json'} exists: the Classroom preflight runs once")
    out.mkdir(parents=True, exist_ok=True)
    pose = eye_pose(strict=True)
    gazes = []
    for g, c, rep in prepare(run, pose, strict=True):
        cfg = configure_and_check(c, SP.SPP, True, int(g["rank"]))
        gazes.append({**rep, **cfg})
    AR.write_json(out / "preflight.json", {
        "schema": "NS1a-preflight-v1", "truth": SP.TRUTH_DERIVED,
        "statement": "Classroom loaded and configured at 4096 spp for the six frozen gazes; EYE pose, calibrations, "
                     "settings and rank-1 camera matrices checked against the accepted records; NO render",
        "blend": bpy.data.filepath, "blend_realpath": os.path.realpath(bpy.data.filepath),
        "blender": bpy.app.version_string, "eye_pose": pose, "spp": SP.SPP, "gazes": gazes, "rendered": False})
    print(f"{PREFIX} COMPLETE preflight (no render)", flush=True)


def canonical(a) -> None:
    run = Path(a.run).resolve()
    rec = acquisition(run, SP.SPP, strict=True, extra_record={"observation_quality": {
        "spp": SP.SPP, "statement": "the accepted synthetic natural-stereo reference observation quality"}})
    print(f"{PREFIX} COMPLETE canonical " + json.dumps({"ranks": rec["ranks_in_order"], "budget": rec["budget"]},
                                                       sort_keys=True), flush=True)


# ------------------------------------------------------------------ rehearsal: a synthetic factory-startup scene only
def build_rehearsal_scene() -> None:
    """A noise-textured room with one textured slab per frozen gaze direction and one collection-instanced
    (instance-0, non-catalog) slab at rank 3; never Classroom."""
    import bpy
    from mathutils import Matrix
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.0, 0.0, 0.0)
    src, _ = head_source()
    r_wh, o_w = np.asarray(src["head_R_wh"], float), np.asarray(src["head_origin_w_m"], float)
    eye = bpy.data.objects.new("EYE", bpy.data.cameras.new("EYE"))
    scene.collection.objects.link(eye)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r_wh, o_w
    eye.matrix_world = Matrix(m.tolist())
    bpy.ops.mesh.primitive_cube_add(size=8.0, location=tuple(o_w))
    room = bpy.context.active_object
    room.name = "rehearsal_room"
    room.data.materials.append(AR._emission_noise("room_noise", 20.0, 0.3))
    # (rank, distance m, size m, name): rank 6 is deliberately small (< 100 core points expected)
    slabs = [(1, 2.5, 0.40, "slab_a"), (2, 1.2, 0.20, "slab_b"), (4, 1.4, 0.30, "slab_c"), (5, 1.9, 0.30, "slab_d"),
             (6, 3.0, 0.02, "slab_e")]
    hidden = bpy.data.collections.new("noncatalog_source")      # not linked to the scene: its object is no
    for rank, dist, size, name in slabs + [(3, 1.0, 0.30, "noncatalog_slab")]:   # scene object -> pass index 0
        yaw, pitch = SP.gaze_yaw_pitch(*[(g[1], g[2]) for g in SP.GAZES if g[0] == rank][0])
        y, p = math.radians(yaw), math.radians(pitch)
        d_h = np.array([math.sin(y) * math.cos(p), math.sin(p), -math.cos(y) * math.cos(p)])
        z_h = -d_h                                              # the slab faces the head
        x_h = np.cross([0.0, 1.0, 0.0], z_h)
        x_h = x_h / np.linalg.norm(x_h) if np.linalg.norm(x_h) > 1e-6 else np.array([1.0, 0.0, 0.0])
        y_h = np.cross(z_h, x_h)
        rot = r_wh @ np.column_stack((x_h, y_h, z_h))
        pm = np.eye(4)
        pm[:3, :3], pm[:3, 3] = rot, o_w + r_wh @ (d_h * dist)
        bpy.ops.mesh.primitive_plane_add(size=size)
        ob = bpy.context.active_object
        ob.name = name
        ob.matrix_world = Matrix(pm.tolist())
        ob.data.materials.append(AR._emission_noise(f"{name}_noise", 60.0 + 7 * rank, 1.0 + rank))
        if name == "noncatalog_slab":
            for col in list(ob.users_collection):
                col.objects.unlink(ob)
            hidden.objects.link(ob)
            inst = bpy.data.objects.new("noncatalog_instance", None)
            inst.instance_type = "COLLECTION"
            inst.instance_collection = hidden
            scene.collection.objects.link(inst)
    bpy.context.view_layer.update()


def rehearsal(a) -> None:
    run = Path(a.run).resolve()
    spp = int(a.spp)
    if spp >= SP.SPP:
        raise ValueError("the rehearsal is a low-spp plumbing run")
    build_rehearsal_scene()
    rec = acquisition(run, spp, strict=False, extra_record={"rehearsal": True,
                                                            "scene": "synthetic factory-startup room (never Classroom)"})
    print(f"{PREFIX} COMPLETE rehearsal " + json.dumps({"ranks": rec["ranks_in_order"], "budget": rec["budget"]},
                                                       sort_keys=True), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("preflight", "canonical", "rehearsal"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--spp", type=int, default=None)
    a = ap.parse_args(AR.args_after_dashes())
    if (a.mode == "rehearsal") != (a.spp is not None):
        raise ValueError("--spp is a rehearsal option only, and the rehearsal requires it")
    {"preflight": preflight, "canonical": canonical, "rehearsal": rehearsal}[a.mode](a)


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
