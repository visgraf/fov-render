"""Issue #4 part B: diagnose the scene gate's S1 and S5 failures on the Barbershop (not a gate).

    .venv/bin/python docs/runs/issue-4/diagnose_step7.py --blend BARBERSHOP_EYE.blend --s5-exr previews/scene-barbershop/S5/ids.exr

S1: for each image in the spec's missing list, its users and whether any rendered object uses it.
S5: the gate's own 600 samples (rng seed 0, 360 px panorama from the head) against its saved
Object Index render, split by whether the ray-cast object is flagged by the material audit.
Writes step7-diagnosis.json next to this file.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))


def main() -> None:
    import bpy
    import numpy as np
    from fov_render.bl import engine, scene as S
    from fov_render.rig import Rig
    from fov_render.sensors import PanoramaModel
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--s5-exr", required=True)
    ap.add_argument("--spec", default=str(HERE.parents[2] / "docs" / "scenes" / "barbershop.json"))
    a = ap.parse_args(argv)
    h = S.open_scene(a.blend, scale_length=None)
    in_scene = {o.name_full for o in h.scene.objects}
    mat_users = collections.defaultdict(list)
    for o in bpy.data.objects:
        for s in o.material_slots:
            if s.material:
                mat_users[s.material.name_full].append(o)
    expected = json.loads(Path(a.spec).read_text())["expect"]["missing_images"]
    um = bpy.data.user_map(subset=[bpy.data.images[n] for n in expected if n in bpy.data.images])
    images = {}
    for img, users in um.items():
        rows, rendering = [], 0
        for u in users:
            if type(u).__name__ == "Material":
                objs = mat_users.get(u.name_full, [])
                r = sum(1 for o in objs if not o.hide_render and o.name_full in in_scene)
                rendering += r
                rows.append({"material": u.name_full, "objects": len(objs), "rendering": r})
            else:
                rows.append({type(u).__name__: u.name_full})
        images[img.name_full] = {"missing_on_disk": S.image_missing(img), "users": rows,
                                 "used_by_rendered_objects": rendering > 0}
    ids = np.rint(engine.read_passes(a.s5_exr)["object_index"]).astype(int)
    model = PanoramaModel(360)
    rays = model.rays()
    pose = Rig().fixed_pose(h.head, "C")
    dg = bpy.context.evaluated_depsgraph_get()
    rng = np.random.default_rng(0)
    rows_, cols = rng.integers(0, model.height, 600), rng.integers(0, model.width, 600)
    flags = h.audit["flagged_objects"]
    tally, objs = collections.Counter(), collections.Counter()
    for i, j in zip(rows_, cols):
        d = pose.R_wb @ rays.direction[i, j]
        hit, _l, _n, _f, obj, _m = h.scene.ray_cast(dg, pose.centre_w.tolist(), d.tolist())
        if not hit:
            continue
        o = getattr(obj, "original", obj)
        ok = int(ids[i, j]) == int(o.pass_index)
        tally["flagged" if o.name_full in flags else "opaque", "agree" if ok else "differ"] += 1
        if not ok:
            objs[o.name_full] += 1
    opaque = tally["opaque", "agree"] + tally["opaque", "differ"]
    out = {"schema": "fov-render-issue4-step7-diagnosis-v1", "blend": a.blend,
           "S1_images": images,
           "S1_used_by_rendered_objects": sorted(k for k, v in images.items() if v["used_by_rendered_objects"]),
           "S5": {"hits": sum(tally.values()), "tally": {f"{k[0]} {k[1]}": v for k, v in tally.items()},
                  "agree_fraction_all": (tally["opaque", "agree"] + tally["flagged", "agree"]) / max(sum(tally.values()), 1),
                  "agree_fraction_opaque": tally["opaque", "agree"] / max(opaque, 1),
                  "mismatched_ray_cast_objects": {k: {"n": v, "flags": flags.get(k, [])} for k, v in objs.most_common()}}}
    (HERE / "step7-diagnosis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"S1_used": out["S1_used_by_rendered_objects"], "S5": {k: out["S5"][k] for k in ("tally", "agree_fraction_opaque")}}))


if __name__ == "__main__":
    main()
