# Issue #3, part 2: fov-3d-vision's active loop on a fov_render Session

Code, 2026-11-07 (local time). Every number is **measured on the workstation** (`euclid`: RTX 4090,
driver 595.99.02, Blender 5.2.1 LTS binary build `9e2066aef7ef`; the gates with the bpy 5.2.1 wheel).
fov-render branch `issue-3-active-loop` at `327d428` (Chat's part 2; the runs used it unchanged).
fov-3d-vision `main` at `48a3139`, unchanged and only read; every run went to fov-render's ignored
`previews/issue-3b/`.

```
FOVRENDER_F3D_ROOT=/home/lvelho/rd/f3d-vision FOVRENDER_FOV3D_ROOT=/home/lvelho/rd/fov-3d-vision \
  .venv/bin/python -m fov_render.checks --device OPTIX --work previews/ws-i3   # step 0, 36 s
bash docs/runs/issue-3/part2/run_part2.sh                                       # steps 1 and 2, 9 min
bash docs/runs/issue-3/part2/run_perturbed.sh                                   # the diagnosis runs, 6 min
.venv/bin/python docs/runs/issue-3/part2/diagnose.py                            # diagnosis.json
```

## Results

| Step | Result | Record |
|---|---|---|
| 0. Gates, OPTIX | **32/32 PASS**, including 6.1–6.3 | [`ws-i3.json`](ws-i3.json) |
| 1. Exact, CPU, calibration room, 50 fixations | **IDENTICAL**: 153 npz files, 102 EXRs, `loop.json`, `pairs.json`, the Phase C table | [`exact.json`](exact.json) |
| 1. Control, IPD + 1 ULP | **DIFFERENT**: 119/153 npz, 50/102 EXRs, `loop.json`, `pairs.json`, the Phase C table (`compare_loop_runs` lists at most 20 differences per file) | [`exact-control.json`](exact-control.json) |
| 2. Statistical, OPTIX, `class_small`, 3 native runs | **FAIL**: 6/17 numbers within the legacy range | [`gate-class_small.json`](gate-class_small.json) |
| 2. Statistical, OPTIX, `class_full`, 3 native runs | **FAIL**: 7/17 numbers within the legacy range | [`gate-class_full.json`](gate-class_full.json) |

`active_eval` reports 0 failures on each set of native runs. The fixation-0 RGB test passes: at most
4.5e-6 (`class_small`) and 5.0e-6 (`class_full`), against 1e-5.

**Step 2, each number's ratio |mean(runs) − mean(legacy)| / legacy range** (bold: above 1, or a
legacy range of 0 not matched). Columns "native" are the gate as Chat specified it; columns
"perturbed legacy" are the diagnosis below, the same tool on the unchanged legacy loop.

| number | `class_small` native | perturbed legacy | `class_full` native | perturbed legacy |
|---|---|---|---|---|
| `coarse_frac` | **1.33** | 0.55 | **1.20** | **1.56** |
| `coverage_any` | **4.29** | **3.34** | **1.89** | **1.61** |
| `coverage_fine` | **2.99** | **1.73** | 0.31 | 0.18 |
| `depth_err_median_m` | 0.08 | 0.15 | 0.80 | **3.00** |
| `fine_depth_err_median_m` | **1.88** | **1.20** | **3.97** | **1.84** |
| `fine_gross_frac` | 0.75 | 0.83 | **34.25** | **11.91** |
| `fine_rho_err_median` | 0.46 | 0.02 | **4.46** | 0.70 |
| `gated` | **1.39** | **1.50** | **1.58** | 0.79 |
| `gross_frac` | 0.74 | 0.19 | 0.43 | **1.83** |
| `outlier_frac` | **5.25** | **3.18** | **82.37** | **26.80** |
| `rho_err_median` | **1.75** | 0.53 | 0.05 | **1.70** |
| `u_end` | 0.71 | **1.21** | 0.74 | 0.28 |
| `u_fixed_cells` | **inf** (5628 vs 5627) | 0.00 | **inf** (27338 vs 27427) | **inf** |
| `u_start` | **7432** | **10311** | **2049** | **2606** |
| `vergence_err_median_m` | **3.81** | **6.41** | 0.31 | 0.16 |
| `visited_fine_unmeasured` | **2.79** | **2.13** | 0.02 | **2.33** |
| `z_rms` | 0.01 | 0.45 | **1.99** | 0.57 |
| **fail** | 11/17 | 9/17 | 10/17 | 11/17 |

The worst ratios are `u_start` and `u_fixed_cells`, which are set at fixation 0; among the numbers the
whole run shapes, `outlier_frac` (5.25 on `class_small`, 82.4 on `class_full`) and `fine_gross_frac`
(34.3 on `class_full`). No tolerance was changed.

## Why it fails: the reference is narrower than the loop's own spread

Records: [`diagnosis.json`](diagnosis.json), [`diag-perturbed-class_small.json`](diag-perturbed-class_small.json),
[`diag-perturbed-class_full.json`](diag-perturbed-class_full.json).

1. **Native mode's fixation 0 is not the legacy one.** The records line up (same pixels in the same
   order, identical footprints and origin; directions within one float32 step, 1.19e-7; distances
   within 2e-5 m), but the renders differ: per pixel, median 2.6e-6 (`class_small`) and 5.8e-6
   (`class_full`) against legacy a, with 2.0 and 3.2 % of the pixels above 1e-3 (at most 0.08). Legacy
   b against a: median 7.5e-9, none above 1e-3. That is the same samples traced along rays one float32
   step apart (fov_render's float64 rig against fov-3d-vision's float32 composition), with paths that
   part in a few per cent of the pixels. The three native runs share their own fixation 0 (OptiX
   noise only), as the three legacy runs share theirs.
2. **The legacy range only measures OptiX noise after a shared fixation 0.** The groups of runs that
   share a fixation 0 are apart from fixation 0 on, while the runs within a group stay together
   (`class_full_trajectories` in `diagnosis.json`: `fine_gross_frac` at fixation 0 is 0.0685 for all
   three legacy runs, 0.0652, 0.0684, 0.0711 for the perturbed ones below, 0.0696 for all three native
   runs; within a group it agrees to four digits until fixation 10–20). The numbers set at fixation 0
   (`u_start`, `u_fixed_cells`) have a legacy range of about 0, so any change of fixation 0 fails them.
3. **The unchanged legacy loop fails the same gate when fixation 0 moves by native's amount.**
   `run_perturbed.sh`: fov-3d-vision's `active_loop.py` with the IPD 0.2, 0.4, 0.6 µm larger (each eye
   0.1–0.3 µm further out, one to three float32 steps at the head's 1.6 m). At fixation 0 they differ
   from legacy a by a per-pixel median of 1.6–4.4e-6 (`class_small`) and 4.1–9.7e-6 (`class_full`),
   1.6–4.6 % of the pixels above 1e-3: native's size. `loop_spread_gate` on these three runs: **FAIL**,
   9/17 and 11/17 numbers outside (native: 11 and 10), with the same kind of ratios (table above).
4. **On the calibration room, CPU against OPTIX is the same story.** The legacy loop on CPU (step 1)
   against part 1's three OPTIX legacy runs: vergence error 0.587 against 0.570 m (2935 OPTIX legacy
   ranges), depth error 0.872 against 0.844 m (76), `u_fixed_cells` 10824 against 10820. Even the IPD +
   1 ULP control, whose first renders are bit for bit the legacy ones, moves 5 numbers by 1.27–2.49
   ranges.
5. **Against the six legacy runs** (a–c and the three perturbed ones), native's mean is inside min..max
   for 14/17 numbers on `class_small` and 6/17 on `class_full`. On `class_full` native's fine-level
   numbers lean the same way: `fine_gross_frac` 0.236–0.243 against 0.228–0.235, `outlier_frac`
   0.1047–0.1059 against 0.1031–0.1048, `fine_rho_err_median` 0.01845–0.01849 against 0.01759–0.01838.
   Four independent fixation-0 draws on the legacy side and one on the native side cannot tell that
   lean from chance, nor rule it out.
6. **The fixation-0 RGB test** passes, but it now measures the pose difference, not noise: native's
   fixation 0 is not posed bit for bit as legacy's, and the perturbed legacy runs reach 9.4e-6 against
   the 1e-5 threshold, which #9 set for identical poses.

## Wall time: does the Session add cost?

From each run's `loop.json` (`wall_seconds`) and the Session's `meta.json` per look (`time` in
`diagnosis.json`); with Blender's start, [`walltimes.txt`](walltimes.txt).

| Run | Legacy | On fov_render | Session time outside the render, per look |
|---|---|---|---|
| Calibration room, CPU, exact | 30.7 s | 30.9 s (+0.7 %) | 1 ms |
| `class_small`, OPTIX, native | 29.6–34.9 s (part 1) | 33.5–35.7 s (+8 %) | 40 ms (12 492 samples per eye) |
| `class_full`, OPTIX, native | 70.5–71.0 s (part 1) | 84.5–88.9 s (+22 %) | 154 ms (50 269 samples per eye) |

The Session itself costs nothing measurable; native mode's cost is building and writing fov_render's
records (`outputs=("exr", "records")`, a `record_<eye>.npz` per look, 342 MB per `class_full` run),
103 looks per run: 4 s and 16 s, against observed differences of the means of 3 s and 15 s.

## For Chat

1. **The exact test passes**: over 50 fixations on CPU, the loop on a Session is the legacy loop, bit
   for bit, and the control fails it.
2. **The statistical gate cannot pass a correct loop.** The unchanged legacy loop fails it when its
   fixation 0 moves by native's amount (finding 3). A reference that could work needs runs that differ
   at fixation 0, on both sides: for example N legacy runs with distinct small IPD perturbations against
   N native runs with the same perturbations, compared on their means with the spread between runs.
   A legacy and a native run take about 65 s together on `class_small` and 155 s on `class_full`; N = 10
   on both configurations is about 40 min. Which reference, if any, is Luiz's decision.
3. `u_start` and `u_fixed_cells` are set at fixation 0; no range of runs that share a fixation 0 can
   judge them.
4. Native's records cost 154 ms per look on `full` (22 % of the loop's time).
