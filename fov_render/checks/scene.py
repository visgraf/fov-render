"""The scene gate: every fov_render check that a new .blend can run (issue #4, step 7; report §8.6's
`fov-render check --blend`).

    fov-render check --blend SCENE_eye.blend [--spec docs/scenes/NAME.json] [--device OPTIX] \
        [--work previews/scene-NAME] [--json OUT] [--only S1 S2 ...] [--write-spec DRAFT.json]
    python -m fov_render.checks.scene (same options)                        (the bpy wheel, by hand)

The spec (optional) states what is expected of this scene: hole fraction, the objects allowed to
show back-faces, the nearest surface, the images known to be missing, the gazes. Without a spec the
defaults below apply and nothing is known in advance.

    S1  the scene opens: head from EYE, the view layer that renders, the catalogue names every hit
    S2  sanity panorama: holes, back-faces only on the allowed objects, clearance
    S3  fov-3d-vision's check_foveated on fov_render fixations at the spec's gazes
    S4  every sensor's record agrees with its Position pass within a pixel
    S5  Object Index equals the ray-cast object
    S6  the material and image audits, by solid angle from the head (reported for review, not gated)

Exit status 1 if any gate fails.

--write-spec DRAFT writes a draft spec from what S1 and S2 measured (issue #18), for a person to
review before it becomes docs/scenes/NAME.json: the missing images as found, the closed meshes
whose back-faces exceed the limit as allowed, a hole limit of twice the measured fraction (at
least 0.001, never above the default 0.02), the default clearance. A draft accepts the scene as it
is; what it records is what a later change of the scene must keep.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from ..rig import Gaze, Rig
from .common import Result, write_json

STEP = "scene"
DEFAULTS = {"max_hole_fraction": 0.02, "backface_allowed": [], "max_backface_fraction_other": 0.001,
            "min_clearance_m": 0.25, "missing_images": None, "gazes": [[0.0, 0.0], [35.0, -10.0], [-70.0, 8.0]],
            "spp": 16, "panorama_width": 720}


SPEC_SCHEMA = "fov-render-scene-spec-v1"


def spec_problems(d) -> list[str]:
    """Why a spec cannot be used as it is (numpy only; `fov-render check` refuses it with these)."""
    if not isinstance(d, dict) or d.get("schema") != SPEC_SCHEMA:
        return [f"schema must be {SPEC_SCHEMA!r}"]
    out = []
    expect = d.get("expect", {})
    extra = sorted(set(expect) - set(DEFAULTS))
    if extra:
        out.append(f"unknown expectation(s) {extra}; known: {sorted(DEFAULTS)}")
    nulls = sorted(k for k, v in expect.items() if v is None and k != "missing_images")
    if nulls:
        out.append(f"expectation(s) {nulls} are null")
    if d.get("to_measure"):
        out.append(f"expectation(s) {sorted(d['to_measure'])} are still to be measured "
                   f"(`fov-render check --write-spec`, then review)")
    return out


def _spec(path) -> dict:
    s = dict(DEFAULTS)
    if path:
        d = json.loads(Path(path).read_text())
        bad = spec_problems(d)
        if bad:
            raise SystemExit(f"[fov-render scene] {path}: " + "; ".join(bad))
        s.update(d.get("expect", {}))
    return s


def _open(ctx):
    from ..bl import scene as S
    return S.open_scene(ctx["blend"], scale_length=None)


# ----------------------------------------------------------------------------- S1

def opens(ctx) -> Result:
    h = _open(ctx)
    sp, cat, a = ctx["spec"], h.catalog, h.audit
    miss = sorted(a["missing_images"]["missing"])
    expected_missing = sp["missing_images"]
    ok = (h.eye_note.startswith("EYE object") and cat["count"] < 32767
          and (expected_missing is None or miss == sorted(expected_missing)))
    # control: the same scene against an expectation one image off (one fewer, or one more when none
    # is expected) fails
    ctrl = None if expected_missing is None else (
        miss != (sorted(expected_missing)[1:] if expected_missing else ["<an image expected missing>"]))
    return Result(STEP, "S1 the scene opens: head from EYE, rendering view layer, catalogue, audits", ok, ctrl,
                  {"file": {k: h.info.get(k) for k in ("path", "sha256", "saved_with", "blender", "unit_system",
                                                       "scale_length", "view_layer")},
                   "head": h.head.as_dict(), "eye_note": h.eye_note, "sanitised": h.sanitised,
                   "catalogue": {k: cat.get(k) for k in ("mode", "count", "direct", "via_instance", "via_particles",
                                                         "gn_instancers")},
                   "flagged_by_flag": a["by_flag"], "missing_images": miss, "expected_missing": expected_missing})


# ----------------------------------------------------------------------------- S2

def sanity(ctx) -> Result:
    from ..bl import config
    h = _open(ctx)
    sp = ctx["spec"]
    cfg = config.RenderConfig(profile="small", device=ctx["device"])
    c = h.sanity(cfg, width=sp["panorama_width"], spp=4, out_dir=str(Path(ctx["work"]) / "S2"))
    from ..bl.scene import backface_verdict
    v = backface_verdict(c, sp["backface_allowed"], sp["max_backface_fraction_other"])
    ok = (c["hole_fraction"] <= sp["max_hole_fraction"] and v["ok"]
          and (c["nearest_surface_m"] or 0) >= sp["min_clearance_m"])
    # control: with every object's back-faces counted as if closed and none allowed, a scene that has
    # back-faces fails the same limit (None when it has none)
    worst = backface_verdict({"backface_by_object": c["backface_by_object"]}, (), sp["max_backface_fraction_other"])
    ctrl = None if c["backface_fraction"] <= sp["max_backface_fraction_other"] else not worst["ok"]
    return Result(STEP, "S2 sanity panorama: holes, back-faces on closed meshes only where allowed, clearance",
                  ok, ctrl, {"sanity": {k: v for k, v in c.items() if k != "settings"}, "backface_verdict": v,
                             "spec": {k: sp[k] for k in ("max_hole_fraction", "backface_allowed",
                                                         "max_backface_fraction_other", "min_clearance_m")}})


# ----------------------------------------------------------------------------- S3

def check_foveated(ctx) -> Result:
    from .classroom import _render_foveated
    from .step1 import _check_foveated
    work = Path(ctx["work"]) / "S3"
    reports, failures = [], []
    for k, g in enumerate(ctx["spec"]["gazes"]):
        r = _check_foveated(_render_foveated(ctx, Path(ctx["blend"]), work / f"g{k}", gaze=g))
        reports.append({"gaze": g, "failed": r["checks_failed"], "ray_saving": r.get("ray_saving")})
        failures += r["checks_failed"]
    bad = _check_foveated(_render_foveated(ctx, Path(ctx["blend"]), work / "control_e2", e2_shader=2.04,
                                           gaze=ctx["spec"]["gazes"][0]))
    return Result(STEP, "S3 fov-3d-vision's check_foveated on fov_render fixations", not failures,
                  bool(bad["checks_failed"]), {"fixations": reports, "control_failures": bad["checks_failed"],
                                                "control": "shader E2 = 2.04 while the record says 2.0"})


# ----------------------------------------------------------------------------- S4

def records_agree(ctx) -> Result:
    from .. import Session, record, sensors as SN
    from ..bl import config
    sp = ctx["spec"]
    cfg = config.RenderConfig(profile="small", device=ctx["device"], spp=sp["spp"])
    ip = Session.in_process(cfg, Rig(roll="baseline_projected"), blend=ctx["blend"], access="oracle")
    y, p = sp["gazes"][1]
    looks = {"foveated": (SN.Foveated(), ("L", "R"), Gaze.yaw_pitch(y, p, 2.0)),
             "perspective": (SN.Perspective(), ("L", "R"), Gaze.yaw_pitch(y, p, 2.0)),
             "panorama": (SN.Panorama(width=sp["panorama_width"]), ("C",), None),
             "reference_view": (SN.ReferenceView(640, 544, 27.0), ("C",), Gaze.yaw_pitch(y, p))}
    out, ok = {}, True
    for name, (spec, eyes, gaze) in looks.items():
        look = ip.look(spec, eyes, gaze, out_dir=Path(ctx["work"]) / "S4" / name, outputs=("exr", "records"))
        model = spec.model(cfg.profile)
        px = (model.s0_deg * (1 + model.emax / model.e2) if name == "foveated" else
              360.0 / model.width if name == "panorama" else math.degrees(1.0 / model.f))
        for rec in look.records:
            c = record.sensor_consistency(rec)
            good = c["hits"] > 0 and c["p999_angle_deg"] <= px
            ok &= good
            out[f"{name}_{rec.eye}"] = {**c, "pixel_deg": px, "ok": good,
                                        "seconds": look.response["eyes"][rec.eye]["render_seconds"]}
    # control: the perspective look's record built with the other eye's pose disagrees
    look = ip.look(SN.Perspective(), ("L", "R"), Gaze.yaw_pitch(y, p, 2.0), out_dir=Path(ctx["work"]) / "S4" / "ctrl",
                   outputs=("exr", "records"))
    from ..bl import engine
    passes = engine.read_passes(str(look.exr["L"]))
    pr = look.response["eyes"]["R"]["pose"]
    bad = record.sensor_consistency(record.build(SN.Perspective().model(cfg.profile), passes, pr["centre_w_m"],
                                                 np.asarray(pr["matrix_world"])[:3, :3], ip.handler.head.R_wh,
                                                 eye="L", look_id="x", access="oracle"))
    px = math.degrees(1.0 / SN.Perspective().model(cfg.profile).f)
    return Result(STEP, "S4 every sensor's record agrees with its Position pass within a pixel", ok,
                  bad["p999_angle_deg"] > px, {"looks": out, "control_other_eye_pose": bad})


# ----------------------------------------------------------------------------- S5

def ids_vs_raycast(ctx) -> Result:
    import bpy
    from ..bl import cameras, config, engine, scene as S
    from ..sensors import PanoramaModel
    h = _open(ctx)
    model = PanoramaModel(360)
    cfg = config.RenderConfig(device=ctx["device"], spp=1)
    config.apply(h.scene, cfg, model.width, model.height, h.view_layer)
    cam = cameras.build(h.scene, model, cfg)
    pose = Rig().fixed_pose(h.head, "C")
    cameras.pose(cam, pose)
    exr = Path(ctx["work"]) / "S5" / "ids.exr"
    exr.parent.mkdir(parents=True, exist_ok=True)
    engine.render_once(h.scene, 1, 0, str(exr))
    ids = np.rint(engine.read_passes(str(exr))["object_index"]).astype(int)
    rays = model.rays()
    dg = bpy.context.evaluated_depsgraph_get()
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, model.height, 600), rng.integers(0, model.width, 600)
    flagged = set(h.audit["flagged_objects"])

    def chain(origin, d, steps=12):
        """The objects a ray meets up to and including the first one the audit does not flag: the
        render's Object Index may stop on a flagged (non-opaque) one or pass through it (issue #4:
        fog volumes, glass)."""
        out, o = [], np.asarray(origin, float)
        for _ in range(steps):
            hit, loc, _n, _f, obj, _m = h.scene.ray_cast(dg, o.tolist(), d.tolist())
            if not hit:
                return out, False
            obj = getattr(obj, "original", obj)
            out.append(obj)
            if obj.name_full not in flagged:
                return out, True
            o = np.asarray(loc) + 1e-4 * d
        return out, False

    hits, through = [], 0
    for i, j in zip(rows, cols):
        d = pose.R_wb @ rays.direction[i, j]
        objs, opaque = chain(pose.centre_w, d)
        if objs:
            hits.append((int(i), int(j), objs))
            through += len(objs) > 1

    def agrees(i, j, objs):
        return int(ids[i, j]) in {int(o.pass_index) for o in objs}
    agree = sum(agrees(i, j, objs) for i, j, objs in hits)
    first_only = sum(int(ids[i, j]) == int(objs[0].pass_index) for i, j, objs in hits)
    frac = agree / max(len(hits), 1)
    # control: renumber the most-hit object after the render; its pixels stop agreeing
    from collections import Counter
    top = Counter(objs[-1].name_full for _i, _j, objs in hits).most_common(1)
    ctrl = None
    if top:
        o = next(objs[-1] for _i, _j, objs in hits if objs[-1].name_full == top[0][0])
        keep = o.pass_index
        o.pass_index = keep + 30000 if keep + 30000 < 32767 else keep + 1
        bad = sum(agrees(i, j, objs) for i, j, objs in hits)
        o.pass_index = keep
        ctrl = bad < agree
    return Result(STEP, "S5 Object Index equals the ray-cast object (through flagged non-opaque objects)",
                  len(hits) > 50 and frac >= 0.97, ctrl,
                  {"sampled_hits": len(hits), "agree_fraction": frac, "agree_first_hit_only": first_only / max(len(hits), 1),
                   "rays_through_flagged": through, "most_hit": top,
                   "note": "a pixel agrees when its Object Index is one of the objects the ray meets up to the first "
                           "unflagged one; edge pixels can disagree (one sample at spp 1 against the pixel-centre ray)"})


# ----------------------------------------------------------------------------- S6

def audit_coverage(ctx) -> Result:
    """What fraction of the head's sphere (solid angle) shows flagged materials or missing images:
    the material audit to review with Luiz. Reported, not gated."""
    from .. import Session, sensors as SN
    from ..bl import config
    sp = ctx["spec"]
    ip = Session.in_process(config.RenderConfig(profile="small", device=ctx["device"], spp=4), Rig(),
                            blend=ctx["blend"], access="oracle")
    h = ip.handler.scene_handle
    rec = ip.look(SN.Panorama(width=sp["panorama_width"]), ("C",), None, out_dir=Path(ctx["work"]) / "S6",
                  outputs=("exr", "records")).records[0]
    w = rec.footprint_sr / rec.footprint_sr.sum()
    names = {e["id"]: e["name"] for e in h.catalog["instances"]}
    flagged = h.audit["flagged_objects"]
    miss_objs = {o for m in h.audit["missing_images"]["missing"].values() for o in m["objects"]}
    per_obj, per_flag, miss = {}, {}, 0.0
    for i in np.unique(rec.instance_id):
        if i == 0:
            continue
        n = names.get(int(i), f"id {int(i)}")
        f = float(w[rec.instance_id == i].sum())
        if n in flagged:
            per_obj[n] = {"flags": flagged[n], "solid_angle_fraction": f}
            for fl in flagged[n]:
                per_flag[fl] = per_flag.get(fl, 0.0) + f
        if n in miss_objs:
            miss += f
    top = dict(sorted(per_obj.items(), key=lambda kv: -kv[1]["solid_angle_fraction"])[:25])
    return Result(STEP, "S6 audit coverage from the head (for review, not gated)", True, None,
                  {"flagged_objects": len(flagged), "flagged_seen": len(per_obj),
                   "flagged_solid_angle_fraction": float(sum(v["solid_angle_fraction"] for v in per_obj.values())),
                   "by_flag_solid_angle_fraction": per_flag, "missing_image_objects_solid_angle_fraction": miss,
                   "top_flagged": top, "missing_images": h.audit["missing_images"]})


CHECKS = [opens, sanity, check_foveated, records_agree, ids_vs_raycast, audit_coverage]
NAMES = {f"S{k + 1}": fn for k, fn in enumerate(CHECKS)}


def _round_up(x: float, digits: int = 1) -> float:
    """x rounded up to `digits` significant figures."""
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x)) - (digits - 1)
    return math.ceil(x / 10 ** e - 1e-9) * 10 ** e


def draft_spec(blend: str, s1: Result, s2: Result, base: dict) -> dict:
    """A spec that this scene passes on S1 and S2 as it is now (see the module docstring)."""
    san = s2.detail["sanity"]
    limit = base["max_backface_fraction_other"]
    closed = dict(sorted(s2.detail["backface_verdict"]["closed_not_allowed"].items(), key=lambda kv: -kv[1]))
    allowed, rest = list(base["backface_allowed"]), sum(closed.values())
    for name, f in closed.items():
        if rest <= limit:
            break
        allowed.append(name)
        rest -= f
    h = float(san["hole_fraction"])
    hole = min(DEFAULTS["max_hole_fraction"], max(0.001, _round_up(2.0 * h)))
    expect = dict(base, missing_images=sorted(s1.detail["missing_images"]), backface_allowed=allowed,
                  max_hole_fraction=hole)
    near = san.get("nearest_surface_m")
    notes = {"how": "written by `fov-render check --write-spec` from S1 and S2; review every line before it becomes "
                    "docs/scenes/NAME.json",
             "measured": {"hole_fraction": h, "nearest_surface_m": near,
                          "backface_by_object": san.get("backface_by_object", {}),
                          "backface_closed": san.get("backface_closed", {}),
                          "missing_images": s1.detail["missing_images"]},
             "backface_allowed": {n: closed[n] for n in allowed if n in closed},
             "will_fail": [m for m, bad in (
                 (f"holes {h:.4g} above the default limit {DEFAULTS['max_hole_fraction']}", h > hole),
                 (f"nearest surface {near} m nearer than {base['min_clearance_m']} m",
                  near is not None and near < base["min_clearance_m"])) if bad]}
    return {"schema": "fov-render-scene-spec-v1", "name": Path(blend).stem, "draft": True,
            "derived": {"file": Path(blend).name, "sha256": s1.detail["file"]["sha256"]},
            "expect": expect, "notes": notes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blend", required=True)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--device", default="OPTIX", choices=["CPU", "OPTIX", "CUDA"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", nargs="+", choices=sorted(NAMES), default=None, help="run only these gates")
    ap.add_argument("--write-spec", default=None, metavar="DRAFT", help="write a draft spec from S1 and S2")
    a = ap.parse_args(argv)
    only = a.only or sorted(NAMES)
    if a.write_spec and not {"S1", "S2"} <= set(only):
        ap.error("--write-spec needs S1 and S2")
    work = Path(a.work or Path("previews") / f"scene-{Path(a.blend).stem}").resolve()
    work.mkdir(parents=True, exist_ok=True)
    ctx = {"blend": str(Path(a.blend).resolve()), "spec": _spec(a.spec), "device": a.device, "work": work}
    out = []
    for key in sorted(only):
        fn = NAMES[key]
        t0 = time.perf_counter()
        try:
            r = fn(ctx)
        except Exception as e:  # noqa: BLE001
            r = Result(STEP, f"{key} {fn.__name__}", False, None, {"error": f"{type(e).__name__}: {e}",
                                                        "traceback": traceback.format_exc()[-3000:]})
        r.detail["seconds"] = round(time.perf_counter() - t0, 2)
        out.append(r)
        print(f"[fov-render scene] {r.status:4}  {r.name}  ({r.detail['seconds']} s)", flush=True)
    if a.write_spec:
        by = {r.name[:2]: r for r in out}
        if "error" in by["S1"].detail or "error" in by["S2"].detail:
            print("[fov-render scene] no draft spec: S1 or S2 raised", flush=True)
            return 1
        write_json(Path(a.write_spec), draft_spec(ctx["blend"], by["S1"], by["S2"], ctx["spec"]))
        print(f"[fov-render scene] wrote the draft spec {a.write_spec}", flush=True)
    from . import _git_head
    import bpy
    write_json(Path(a.json) if a.json else work / "results.json",
               {"schema": "fov-render-scene-gate-v1", "blend": ctx["blend"], "spec": a.spec, "spec_values": ctx["spec"],
                "only": only,
                "device": a.device, "commit": _git_head(), "blender": bpy.app.version_string,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "passed": sum(r.status == "PASS" for r in out), "failed": sum(r.status == "FAIL" for r in out),
                "results": [r.as_dict() for r in out]})
    print(f"[fov-render scene] passed={sum(r.status == 'PASS' for r in out)} "
          f"failed={sum(r.status == 'FAIL' for r in out)}", flush=True)
    return 1 if any(r.status == "FAIL" for r in out) else 0


if __name__ == "__main__":
    sys.exit(main())
