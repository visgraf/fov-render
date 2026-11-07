"""Issue #9: summarise the cell renders of render_cell.py into results.json (host side, OpenEXR).

    .venv/bin/python docs/runs/issue-9/summarise.py --work previews/issue-9

Cells: {cpu, optix} x {binary-cmdline, binary-open, wheel-open}, each rendered twice (suffix -2).
For each: pixel-for-pixel Combined differences against binary-cmdline and against its own repeat,
and the mean Combined over the image and over the objects issue #9 found darkened. Also the
fingerprint diff between the interpreters and between the two ways of opening the file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import OpenEXR

HERE = Path(__file__).resolve().parent
WATCH = {18: "Cube.014", 15: "Cube.011", 228: "woodWindow.002", 229: "woodWindow.003"}
CELLS = ("binary-cmdline", "binary-open", "wheel-open")


def read(p: Path):
    with OpenEXR.File(str(p)) as f:
        d = {k: np.asarray(v.pixels, np.float64) for part in f.parts for k, v in part.channels.items()}
    comb = next(v for k, v in d.items() if k.endswith("Combined"))[..., :3]
    other = {k.split(".", 1)[1]: v for k, v in d.items() if not k.endswith("Combined")}
    return comb, np.rint(other["Object Index.X"]).astype(int), other


def pix(a, b) -> dict:
    d = np.abs(a - b).max(-1)
    return {"max": float(d.max()), "pixels_differing": int((d > 0).sum()), "pixels": int(d.size),
            "mean_signed": float((a - b).mean())}


def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, p + str(k) + ".")
    elif isinstance(o, list) and o and isinstance(o[0], dict) and "name" in o[0]:
        for x in o:
            yield from flat(x, p + str(x["name"]) + ".")
    else:
        yield p[:-1], o


def fp_diff(a: dict, b: dict) -> list:
    fa, fb = dict(flat(a)), dict(flat(b))
    return [[k, fa.get(k), fb.get(k)] for k in sorted(set(fa) | set(fb))
            if fa.get(k) != fb.get(k) and not k.startswith("scene_render.filepath")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(HERE.parents[2] / "previews" / "issue-9"))
    a = ap.parse_args()
    w = Path(a.work)
    out = {"schema": "fov-render-issue9-results-v1", "fixation": 355, "watch": WATCH}
    for dev in ("cpu", "optix"):
        res = {}
        for eye in "LR":
            r = {c + s: read(w / f"{dev}-{c}{s}" / "look" / f"raw_{eye}.exr") for c in CELLS for s in ("", "-2")}
            ref, oid, ref_other = r["binary-cmdline"]
            e = {}
            for k, (comb, _o, other) in r.items():
                e[k] = {"vs_binary_cmdline": pix(comb, ref), "image_mean": float(comb.mean()),
                        "object_means": {n: float(comb[oid == i].mean()) for i, n in WATCH.items() if (oid == i).any()},
                        "position_and_index_bitwise_vs_binary_cmdline": all(np.array_equal(other[c], ref_other[c]) for c in other)}
                if not k.endswith("-2"):
                    e[k]["vs_own_repeat"] = pix(comb, r[k + "-2"][0])
            res[eye] = e
        cells = {c: json.loads((w / f"{dev}-{c}" / "cell.json").read_text()) for c in CELLS}
        out[dev] = {"spp": cells["binary-cmdline"]["spp"], "backend": {c: v["backend"] for c, v in cells.items()},
                    "opened": {c: v["opened"] for c, v in cells.items()},
                    "calibration_byte_identical_to_frozen": {c: v["calibration_byte_identical_to_frozen"] for c, v in cells.items()},
                    "eyes": res}
    fps = {c: json.loads((w / f"cpu-{c}" / "fingerprint.json").read_text()) for c in CELLS}
    out["fingerprint_diff"] = {"binary-cmdline vs binary-open": fp_diff(fps["binary-cmdline"], fps["binary-open"]),
                               "binary-cmdline vs wheel-open": fp_diff(fps["binary-cmdline"], fps["wheel-open"])}
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    for dev in ("cpu", "optix"):
        for eye in "LR":
            e = out[dev]["eyes"][eye]
            print(dev, eye, {k: (v["vs_binary_cmdline"]["pixels_differing"], round(v["image_mean"], 6)) for k, v in e.items()},
                  "repeats:", {k: v["vs_own_repeat"]["pixels_differing"] for k, v in e.items() if "vs_own_repeat" in v})
    print("fingerprint diffs:", {k: len(v) for k, v in out["fingerprint_diff"].items()})


if __name__ == "__main__":
    main()
