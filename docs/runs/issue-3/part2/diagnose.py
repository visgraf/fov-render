"""Issue #3 part 2: why the statistical gate fails (a diagnosis, not a gate; host side, numpy only).

    .venv/bin/python docs/runs/issue-3/part2/diagnose.py      (after run_part2.sh and run_perturbed.sh)

From the runs in previews/issue-3 (part 1's legacy a, b, c) and previews/issue-3b (this part's) and the
JSONs next to this file:
  fixation0       per-pixel |dRGB| at fixation 0 (eye L, every sample) of each run against legacy a:
                  legacy b (OptiX noise only), native, the legacy loop with the IPD 0.2/0.4/0.6 um larger
  records0        fixation 0's records, legacy a against native 1, field by field
  shift_in_ranges the calibration room on CPU (step 1): the legacy CPU run and the IPD + 1 ULP control,
                  each Phase C number's distance from part 1's OPTIX legacy mean, in OPTIX legacy ranges
  gate_side_by_side  loop_spread_gate's ratio |dmean| / legacy range for native and for perturbed legacy
  trajectories    class_full's fine-level numbers at fixations 0..49 for every run
  look_time       time per look outside the render, from the Session's own meta.json
Writes diagnosis.json next to this file.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FR = HERE.parents[3]
P1, P2 = FR / "previews" / "issue-3", FR / "previews" / "issue-3b"
CONFIGS = ("class_small", "class_full")


def runs(c: str) -> dict:
    out = {f"legacy {r}": P1 / f"{c}-{r}" for r in "abc"}
    out.update({f"legacy ipd+{2 * k}e-7": P2 / f"{c}-legacy-ipd{k}" for k in (1, 2, 3)})
    out.update({f"native {k}": P2 / f"{c}-native-{k}" for k in (1, 2, 3)})
    return out


def samples(run: Path, eye="L") -> dict:
    return dict(np.load(run / eye / "f000" / "samples.npz"))


def fixation0() -> dict:
    out = {}
    for c in CONFIGS:
        rs = runs(c)
        a = samples(rs["legacy a"])["value"].astype(np.float64)
        out[c] = {}
        for name, r in rs.items():
            if name == "legacy a":
                continue
            b = samples(r)["value"].astype(np.float64)
            d = np.abs(b - a).max(1)
            out[c][name] = {"p50": float(np.percentile(d, 50)), "p99": float(np.percentile(d, 99)), "max": float(d.max()),
                            "frac_above_1e-3": float((d > 1e-3).mean()), "abs_mean_drgb": float(abs((b - a).mean()))}
    return out


def records0() -> dict:
    out = {}
    for c in CONFIGS:
        for eye in "LR":
            a, b = samples(P1 / f"{c}-a", eye), samples(P2 / f"{c}-native-1", eye)
            row = {"samples": int(len(a["value"])), "same_raster_order": bool(np.array_equal(a["raster_index"], b["raster_index"]))}
            for k in sorted(a):
                d = np.abs(a[k].astype(np.float64) - b[k].astype(np.float64))
                row[k] = {"max_abs_difference": float(d.max()), "n_different": int((d > 0).sum())}
            out[f"{c} {eye}"] = row
    return out


def shift_in_ranges() -> dict:
    sp = json.loads((HERE.parent / "spread.json").read_text())["configs"]["calib_small"]["phase_c"]
    out = {}
    for f in ("exact", "exact-control"):
        pc = json.loads((HERE / f"{f}.json").read_text())["phase_c"]
        for k, s in sp.items():
            rng, mean = s["max"] - s["min"], float(np.mean(list(s["runs"].values())))
            a, b = pc["a"][k], pc["b"][k]
            row = out.setdefault(k, {"optix_legacy_range": rng, "optix_legacy_mean": mean, "cpu_legacy": a,
                                     "cpu_legacy_minus_optix_mean_in_ranges": abs(a - mean) / rng if rng else (0.0 if a == mean else None)})
            row[f"{f}_minus_cpu_legacy_in_ranges"] = abs(b - a) / rng if rng else (0.0 if a == b else None)
    return out


def gate_side_by_side() -> dict:
    out = {}
    for c in CONFIGS:
        n = json.loads((HERE / f"gate-{c}.json").read_text())["numbers"]
        p = json.loads((HERE / f"diag-perturbed-{c}.json").read_text())["numbers"]
        rows = {}
        for k in n:
            env = n[k]["legacy"] + p[k]["fov_render"]
            rows[k] = {"native_ratio": n[k]["ratio"], "native_ok": n[k]["ok"], "perturbed_ratio": p[k]["ratio"],
                       "perturbed_ok": p[k]["ok"], "legacy": n[k]["legacy"], "perturbed": p[k]["fov_render"],
                       "native": n[k]["fov_render"],
                       "native_mean_inside_legacy_and_perturbed": min(env) <= n[k]["fov_render_mean"] <= max(env)}
        out[c] = {"numbers": rows, "native_fail": sum(not r["native_ok"] for r in rows.values()),
                  "perturbed_fail": sum(not r["perturbed_ok"] for r in rows.values()),
                  "native_mean_inside_six_run_envelope": sum(r["native_mean_inside_legacy_and_perturbed"] for r in rows.values())}
    return out


def trajectories() -> dict:
    out = {}
    for name, r in runs("class_full").items():
        s = json.loads((r / "loop.json").read_text())["steps"]
        out[name] = {m: [x[m] for x in s] for m in ("fine_gross_frac", "fine_outlier_frac", "fine_rho_err_median", "fine_cells",
                                                     "gated", "z_hat_m")}
    return out


def look_time() -> dict:
    out = {}
    for run in ["calib-cpu-exact"] + [f"{c}-native-{k}" for c in CONFIGS for k in (1, 2, 3)]:
        w = r = 0.0
        n = 0
        for m in sorted(glob.glob(str(P2 / run / "_fov_render" / "looks" / "*" / "meta.json"))):
            j = json.loads(Path(m).read_text())["response"]
            n += 1
            w += j["wall_seconds"]
            r += sum(e["render_seconds"] for e in j["eyes"].values())
        out[run] = {"looks": n, "look_wall_s": w, "render_s": r, "outside_render_s": w - r,
                    "outside_render_ms_per_look": 1000 * (w - r) / max(n, 1)}
    loops = {}
    for c in CONFIGS:
        loops[c] = {"legacy": [json.loads((P1 / f"{c}-{r}" / "loop.json").read_text())["wall_seconds"] for r in "abc"],
                    "native": [json.loads((P2 / f"{c}-native-{k}" / "loop.json").read_text())["wall_seconds"] for k in (1, 2, 3)],
                    "legacy_ipd": [json.loads((P2 / f"{c}-legacy-ipd{k}" / "loop.json").read_text())["wall_seconds"] for k in (1, 2, 3)]}
    loops["calib_small_cpu"] = {r: json.loads((P2 / f"calib-cpu-{r}" / "loop.json").read_text())["wall_seconds"]
                                for r in ("legacy", "exact", "ctrl")}
    return {"per_look": out, "loop_wall_seconds": loops}


def main() -> None:
    out = {"schema": "fov-render-issue3-part2-diagnosis-v1", "fixation0_pixels_vs_legacy_a": fixation0(),
           "fixation0_records_legacy_a_vs_native_1": records0(), "calib_cpu_shift_in_optix_legacy_ranges": shift_in_ranges(),
           "gate_native_vs_perturbed_legacy": gate_side_by_side(), "class_full_trajectories": trajectories(),
           "time": look_time()}
    (HERE / "diagnosis.json").write_text(json.dumps(out, indent=1) + "\n")
    g = out["gate_native_vs_perturbed_legacy"]
    for c in CONFIGS:
        print(f"{c}: gate fails native {g[c]['native_fail']}/17, perturbed legacy {g[c]['perturbed_fail']}/17; "
              f"native mean inside the 6-run envelope {g[c]['native_mean_inside_six_run_envelope']}/17")
        for name, v in out["fixation0_pixels_vs_legacy_a"][c].items():
            print(f"   fixation 0 vs legacy a: {name:18s} p50 {v['p50']:.1e} p99 {v['p99']:.1e} >1e-3 {v['frac_above_1e-3']:.4f}")
    for k, v in out["calib_cpu_shift_in_optix_legacy_ranges"].items():
        print(f"   calib CPU {k:24s} cpu legacy vs OPTIX mean {v['cpu_legacy_minus_optix_mean_in_ranges']}  "
              f"control vs cpu legacy {v['exact-control_minus_cpu_legacy_in_ranges']}")
    for k, v in out["time"]["per_look"].items():
        print(f"   {k:22s} {v['outside_render_ms_per_look']:.0f} ms per look outside the render")


if __name__ == "__main__":
    main()
