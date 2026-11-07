"""Issue #1 diagnostics (not gates): the measurements behind the issue comment.

    .venv/bin/python docs/runs/issue-1/diagnostics.py --work previews/issue-1-diag

1. repeatability: gate 1.2's four renders (legacy, fov_render with the legacy pose, fov_render's own
   pose; calibration room, small, 16 spp) rendered twice with identical inputs, on CPU and on OPTIX.
   CPU is the control: every repeat must be bitwise equal there. Also the distribution of the
   own-pose Combined difference that gate 1.2 tests by its max.
2. linked_ids: on the Classroom, what the Object Index pass reads on objects reached only through
   collection instances (linked libraries), with fov_render's numbering and with the legacy
   numbering (_assign_instance_ids on the raw file). Panorama 720 px, 1 spp, OPTIX, from the head.

Writes repeatability.json and linked-ids.json next to this file.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import OpenEXR  # noqa: E402


def _chans(p):
    with OpenEXR.File(str(p)) as f:
        return {k.split(".", 1)[1]: np.asarray(v.pixels, dtype=np.float64) for part in f.parts for k, v in part.channels.items()}


def _maxdiff(a, b):
    return {k: float(np.nanmax(np.abs(a[k] - b[k]))) for k in sorted(a)}


def repeatability(work: Path) -> dict:
    from fov_render.checks import step1
    out = {"spp": step1.SPP, "gaze": step1.GAZE}
    for dev in ("CPU", "OPTIX"):
        w = work / "repeat" / dev
        ctx = {"work": w, "device": dev}
        for rep in ("a", "b"):
            step1.render_legacy_foveated(ctx, w / f"legacy_{rep}")
            step1.render_new_foveated(ctx, w / f"legacy_pose_{rep}", legacy_pose=True)
            step1.render_new_foveated(ctx, w / f"own_pose_{rep}")
        c = {f"{n}_{r}": _chans(w / f"{n}_{r}" / "fix.exr") for n in ("legacy", "legacy_pose", "own_pose") for r in "ab"}
        d = {f"{n}: repeat a vs b": _maxdiff(c[f"{n}_a"], c[f"{n}_b"]) for n in ("legacy", "legacy_pose", "own_pose")}
        d["legacy_a vs legacy_pose_a"] = _maxdiff(c["legacy_a"], c["legacy_pose_a"])
        d["legacy_a vs own_pose_a"] = _maxdiff(c["legacy_a"], c["own_pose_a"])
        px = np.abs(c["legacy_a"]["Combined"][..., :3] - c["own_pose_a"]["Combined"][..., :3]).max(-1)
        z = c["legacy_a"]["Depth.Z"]
        worst = []
        for k in np.argsort(px.ravel())[::-1][:5]:
            i, j = divmod(int(k), px.shape[1])
            win = z[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
            win = win[win < 1e9]
            worst.append({"row": i, "col": j, "diff": float(px[i, j]),
                          "depth_range_3x3_m": float(win.max() - win.min()) if win.size else None})
        d["own_pose_combined_per_pixel"] = {
            "pixels": int(px.size), "median": float(np.median(px)), "p99": float(np.quantile(px, 0.99)),
            "p999": float(np.quantile(px, 0.999)), "max": float(px.max()),
            "count_gt_1e-3": int((px > 1e-3).sum()), "count_gt_1e-2": int((px > 1e-2).sum()),
            "count_gt_0.0474_gate_limit": int((px > 0.05 * 0.948268860578537).sum()), "worst5": worst}
        out[dev] = d
    return out


def linked_ids(work: Path) -> dict:
    import bpy
    from fov_render import record
    from fov_render.bl import cameras, config, engine, scene as S
    from fov_render.checks import classroom
    from fov_render.checks.common import legacy_path, load_legacy
    from fov_render.rig import Rig
    from fov_render.sensors import PanoramaModel
    blend = str(classroom.files()["blend"])
    h = S.open_scene(blend)
    model, cfg = PanoramaModel(720), config.RenderConfig(device="OPTIX", spp=1)

    def render():
        sc, vl = bpy.context.scene, bpy.context.view_layer
        config.apply(sc, cfg, model.width, model.height, vl)
        cam = cameras.build(sc, model, cfg)
        pose = Rig().fixed_pose(h.head, "C")
        cameras.pose(cam, pose)
        exr = Path(tempfile.mkdtemp(dir=work)) / "ids.exr"
        engine.render_once(sc, 1, 0, str(exr))
        p = engine.read_passes(str(exr))
        rec = record.build(model, p, pose.centre_w, pose.R_wb, h.head.R_wh, eye="C", look_id="ids", access="oracle")
        ids = np.rint(p["object_index"]).astype(int).reshape(-1)[rec.raster_index]
        return ids, rec.footprint_sr / rec.footprint_sr.sum(), np.isfinite(rec.distance_m)

    direct = h.catalog["direct"]
    ids_new, w, hit = render()
    on_linked = hit & (ids_new > direct)
    bpy.ops.wm.open_mainfile(filepath=blend)
    S.remove_setting_animation(bpy.context.scene)
    with legacy_path("f3d"):
        load_legacy("f3d", "classroom_oracle1_render.py")._assign_instance_ids()
    linked = [o for o in bpy.data.objects if o.library is not None]
    ids_leg, _w, _hit = render()
    vals = np.unique(ids_leg[on_linked])
    return {"blend": blend, "width": model.width, "spp": 1, "device": "OPTIX",
            "catalog": {k: h.catalog[k] for k in ("count", "direct", "via_instance")},
            "libraries": len(bpy.data.libraries), "linked_objects": len(linked),
            "linked_raw_pass_index_values": sorted({int(o.pass_index) for o in linked}),
            "solid_angle_fraction": {
                "misses": float(w[~hit].sum()), "fov_render_ids_1_234": float(w[hit & ~on_linked].sum()),
                "fov_render_ids_above_234": float(w[on_linked].sum()),
                "distinct_ids_above_234_seen": int(len(set(ids_new[on_linked].tolist()))),
                "legacy_id_on_those_pixels": {int(v): float(w[on_linked & (ids_leg == v)].sum()) for v in vals}},
            "elsewhere_legacy_equals_fov_render": bool(np.array_equal(ids_leg[hit & ~on_linked], ids_new[hit & ~on_linked]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "previews" / "issue-1-diag"))
    a = ap.parse_args()
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    rep = repeatability(work)
    (HERE / "repeatability.json").write_text(json.dumps(rep, indent=1) + "\n")
    ids = linked_ids(work)
    (HERE / "linked-ids.json").write_text(json.dumps(ids, indent=1) + "\n")
    cpu_zero = all(v == 0.0 for k, d in rep["CPU"].items() if "repeat" in k or "legacy_pose_a" in k for v in d.values())
    print(f"[issue-1 diag] CPU control bitwise: {cpu_zero}; wrote repeatability.json, linked-ids.json")
    sys.exit(0 if cpu_zero else 1)


main()
