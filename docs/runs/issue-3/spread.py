"""Issue #3 part 1: the legacy spread from the runs of run_spread.sh (host side, numpy only).

    .venv/bin/python docs/runs/issue-3/spread.py --work previews/issue-3 [--runs a b c]

Per configuration (calib_small, class_small, class_full) and its runs:
  settings     each run's loop.json settings against the accepted run's (fov-3d-vision previews/
               loop2/calib_coverage, loop3/class_coverage, loop3/class_coverage_full), "out" aside
  phase_c      the numbers active_eval.py reports per run (compare.json of all the runs); per number
               min, max, spread = max - min, spread / mean; active_eval's checks and its failures
  per pair:
  trajectory   gaze directions (dir_head) and vergence distances (z_hat_m): equal bit for bit at
               which fixations
  events       issue #3 (from #9): |mean dRGB| over both eyes' samples, per fixation. Judged against
               1e-5 only where the pose is the same (direction and vergence): there two renders differ
               only by OptiX's noise, which is what #9's threshold measured. Where the loop's vergence
               has diverged, the renders differ because the eyes do; reported with what a #9-type
               event would show (a sustained, one-signed darkening) and how the difference follows the
               vergence difference.
Writes spread.json next to this file.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FOV = Path("/home/lvelho/rd/fov-3d-vision")
ACCEPTED = {"calib_small": "loop2/calib_coverage", "class_small": "loop3/class_coverage",
            "class_full": "loop3/class_coverage_full"}
EVENT = 1e-5
PHASE_C = ("coverage_any", "coverage_fine", "visited_fine_unmeasured", "rho_err_median", "fine_rho_err_median",
           "depth_err_median_m", "fine_depth_err_median_m", "vergence_err_median_m", "gross_frac", "fine_gross_frac",
           "z_rms", "outlier_frac", "coarse_frac", "gated", "u_start", "u_end", "u_fixed_cells")


def pair(w: Path, c: str, ra: str, rb: str) -> dict:
    la, lb = (json.loads((w / f"{c}-{r}" / "loop.json").read_text())["steps"] for r in (ra, rb))
    n = min(len(la), len(lb))
    same_dir = [la[k]["dir_head"] == lb[k]["dir_head"] for k in range(n)]
    same_pose = [same_dir[k] and la[k]["z_hat_m"] == lb[k]["z_hat_m"] for k in range(n)]
    signed, absm, dz = [], [], []
    for k in range(n):
        s, m = 0.0, 0.0
        for eye in "LR":
            za, zb = (np.load(w / f"{c}-{r}" / eye / f"f{k:03d}" / "samples.npz") for r in (ra, rb))
            if not np.array_equal(za["raster_index"], zb["raster_index"]):
                raise SystemExit(f"{c} {ra}/{rb} fixation {k} eye {eye}: not the same rays")
            d = float((zb["value"].astype(np.float64) - za["value"].astype(np.float64)).mean())
            s += d / 2
            m = max(m, abs(d))
        signed.append(s)
        absm.append(m)
        dz.append(abs(la[k]["z_hat_m"] - lb[k]["z_hat_m"]) / la[k]["z_hat_m"])
    signed, absm, dz = np.array(signed), np.array(absm), np.array(dz)
    sp, sd = np.array(same_pose), np.array(same_dir)
    vd = sd & ~sp                        # same direction, the vergence diverged: comparable views
    events = [{"k": int(k), "abs_mean_drgb": float(absm[k])} for k in np.where(sp & (absm > EVENT))[0]]

    def longest_run(sign):               # consecutive comparable fixations darker (-1) / brighter (+1) by > EVENT
        best = run = 0
        for k in range(n):
            run = run + 1 if (sd[k] and sign * signed[k] > EVENT) else 0
            best = max(best, run)
        return best
    nz = vd & (dz > 0)
    corr = float(np.corrcoef(np.log10(dz[nz]), np.log10(absm[nz] + 1e-15))[0, 1]) if nz.sum() > 2 else None
    return {"runs": [ra, rb], "fixations": n,
            "direction_equal_at": int(sd.sum()), "direction_differs_at": [int(k) for k in np.where(~sd)[0]],
            "pose_equal_at": [int(k) for k in np.where(sp)[0]],
            "first_vergence_difference": next((k for k in range(n) if same_dir[k] and not same_pose[k]), None),
            "same_pose": {"fixations": int(sp.sum()), "max_abs_mean_drgb": float(absm[sp].max()) if sp.any() else None,
                          "events_above_threshold": events},
            "vergence_diverged": {"fixations": int(vd.sum()),
                                  "abs_mean_drgb_max": float(absm[vd].max()) if vd.any() else None,
                                  "abs_mean_drgb_median": float(np.median(absm[vd])) if vd.any() else None,
                                  "rel_vergence_diff_median": float(np.median(dz[nz])) if nz.any() else None,
                                  "rel_vergence_diff_max": float(dz[vd].max()) if vd.any() else None,
                                  "corr_log_vergence_diff_log_drgb": corr,
                                  "signed_positive": int((signed[vd] > 0).sum()), "signed_negative": int((signed[vd] < 0).sum()),
                                  "longest_run_darker_in_second": longest_run(-1),
                                  "longest_run_brighter_in_second": longest_run(+1)},
            "direction_diverged": {"fixations": int((~sd).sum()),
                                   "abs_mean_drgb_max": float(absm[~sd].max()) if (~sd).any() else None,
                                   "note": "different gaze directions: different views, not compared"},
            "per_fixation": [{"k": k, "same_direction": bool(sd[k]), "same_pose": bool(sp[k]),
                              "signed_mean_drgb": float(signed[k]), "abs_mean_drgb": float(absm[k]),
                              "rel_vergence_diff": float(dz[k])} for k in range(n)]}


def config(w: Path, c: str, runs) -> dict:
    loops = {r: json.loads((w / f"{c}-{r}" / "loop.json").read_text()) for r in runs}
    acc = json.loads((FOV / "previews" / ACCEPTED[c] / "loop.json").read_text())["settings"]
    def diff(s):
        return {k: [s.get(k), acc.get(k)] for k in set(s) | set(acc) if k != "out" and s.get(k) != acc.get(k)}
    cmp_dir = w / (f"{c}-compare3" if len(runs) > 2 else f"{c}-compare")
    cmp_ = json.loads((cmp_dir / "compare.json").read_text())
    rows = {Path(r["run"]).name: r for r in cmp_["runs"]}
    pc = {}
    for k in PHASE_C:
        v = [rows[f"{c}-{r}"].get(k) for r in runs]
        if all(isinstance(x, (int, float)) for x in v):
            mean = float(np.mean(v))
            pc[k] = {"runs": dict(zip(runs, v)), "min": min(v), "max": max(v), "spread": max(v) - min(v),
                     "rel_spread": (max(v) - min(v)) / abs(mean) if mean else None}
    checks = {r: {k: rows[f"{c}-{r}"].get(k) for k in ("replay", "learning", "z_rms")} for r in runs}
    acc_cmp = {"calib_small": "loop2/calib_compare", "class_small": "loop3/class_compare",
               "class_full": "loop3/class_compare_full"}[c]
    acc_row = next(r for r in json.loads((FOV / "previews" / acc_cmp / "compare.json").read_text())["runs"]
                   if r["policy"] == "coverage")
    for k, p in pc.items():
        v = acc_row.get(k)
        if isinstance(v, (int, float)):
            p["accepted_phase_c"] = v
            p["accepted_inside_spread"] = p["min"] <= v <= p["max"]
    return {"settings_differences_vs_accepted": {r: diff(loops[r]["settings"]) for r in runs},
            "accepted_compare": acc_cmp,
            "accepted_run": ACCEPTED[c], "wall_seconds": {r: loops[r]["wall_seconds"] for r in runs},
            "eval": {"compare": str(cmp_dir.name), "fails": cmp_.get("fails")}, "checks": checks, "phase_c": pc,
            "pairs": [pair(w, c, a, b) for a, b in itertools.combinations(runs, 2)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(HERE.parents[2] / "previews" / "issue-3"))
    ap.add_argument("--runs", nargs="+", default=["a", "b", "c"])
    a = ap.parse_args()
    w = Path(a.work)
    out = {"schema": "fov-render-issue3-spread-v2", "event_threshold_abs_mean_drgb": EVENT, "runs": a.runs,
           "configs": {c: config(w, c, a.runs) for c in ACCEPTED}}
    (HERE / "spread.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    for c, r in out["configs"].items():
        print(f"== {c}: settings diffs {r['settings_differences_vs_accepted']['a']} | eval fails {r['eval']['fails']}")
        for k in ("coverage_any", "coverage_fine", "rho_err_median", "fine_rho_err_median", "depth_err_median_m",
                  "fine_depth_err_median_m", "vergence_err_median_m", "z_rms"):
            p = r["phase_c"][k]
            print(f"   {k:24s} " + "  ".join(f"{x} {v:.6g}" for x, v in p["runs"].items()) + f"  | spread {p['spread']:.3g} ({100 * (p['rel_spread'] or 0):.2f} %)")
        for p in r["pairs"]:
            s, v = p["same_pose"], p["vergence_diverged"]
            print(f"   pair {p['runs']}: direction equal {p['direction_equal_at']}/{p['fixations']} (differs at {p['direction_differs_at']}), "
                  f"pose equal at {p['pose_equal_at']} "
                  f"(max |mean dRGB| {s['max_abs_mean_drgb']:.2g}, events {len(s['events_above_threshold'])}) | vergence diverged: max {v['abs_mean_drgb_max']:.2g}, "
                  f"corr {v['corr_log_vergence_diff_log_drgb']:.2f}, signs +{v['signed_positive']}/-{v['signed_negative']}, "
                  f"longest darker run {v['longest_run_darker_in_second']}, brighter run {v['longest_run_brighter_in_second']}")


if __name__ == "__main__":
    main()
