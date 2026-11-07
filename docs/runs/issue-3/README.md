# Issue #3, part 1: the legacy spread

Code, 2026-11-07. Every number is **measured on the workstation** (`euclid`: RTX 4090, driver
595.99.02, Blender 5.2.1 LTS binary build `9e2066aef7ef`, OPTIX). fov-3d-vision `main` at
`48a3139`, unchanged and only read (its tracked files untouched; the runs written to fov-render's
ignored `previews/issue-3/`). fov-render branch `issue-3-legacy-spread` from `main` at `ff8ff99`.

```
bash docs/runs/issue-3/run_spread.sh                                   # the runs and active_eval, batch, ~7 min
.venv/bin/python docs/runs/issue-3/spread.py --work previews/issue-3   # spread.json
```

**What ran.** fov-3d-vision's `tools/active_loop.py --policy coverage --fixations 50`, in each of
the three configurations Phase C reports for coverage, **three times** with identical arguments
(a, b, c), then `tools/active_eval.py` on the three runs together:

| Configuration | Scene | Arguments | Loop wall time per run (`loop.json`; with Blender's start, [`walltimes.txt`](walltimes.txt)) |
|---|---|---|---|
| `calib_small` | `scenes/calib_room/calib_room.blend` | `--profile small` | 18.1–18.6 s |
| `class_small` | `scenes/classroom/classroom_eye.blend` | `--profile small` | 29.6–34.9 s |
| `class_full` | `scenes/classroom/classroom_eye.blend` | `--profile full --kappa 3.8` | 70.5–71.0 s |

Each run's settings equal the accepted Phase C run's (`previews/loop2/calib_coverage`,
`loop3/class_coverage`, `loop3/class_coverage_full` in fov-3d-vision), apart from two options the
loop gained later, at their defaults: `fusion: mean`, `nb_drop_isolated: false`.

**Checks.** `active_eval` reports 0 failures for every configuration: (s) replay ok on all 9 runs;
(t) z RMS 0.54, 0.61–0.64, 1.01, inside [0.4, 2.5]; (u) the fixation-0 cells improve in every run.
(v) needs a random-policy run and does not apply.

## The spread (3 runs) and the published Phase C numbers

Records: [`spread.json`](spread.json), [`eval/`](eval/) (`compare.json` and the log of each
three-run evaluation).

| `calib_small` | a | b | c | spread (rel) | Phase C |
|---|---|---|---|---|---|
| `coverage_fine` | 0.1752 | 0.1753 | 0.1753 | 8.1e-05 (0.05 %) | 0.1379 (outside) |
| `rho_err_median` | 0.1063 | 0.1064 | 0.1064 | 0.00014 (0.13 %) | 0.1052 (outside) |
| `fine_rho_err_median` | 0.02436 | 0.02433 | 0.02433 | 3.1e-05 (0.13 %) | 0.02488 (outside) |
| `depth_err_median_m` | 0.8434 | 0.8438 | 0.8438 | 0.00037 (0.04 %) | 0.8363 (outside) |
| `fine_depth_err_median_m` | 0.1123 | 0.1116 | 0.1116 | 0.00065 (0.58 %) | 0.127 (outside) |
| `vergence_err_median_m` | 0.5701 | 0.5701 | 0.5701 | 5.9e-06 (0.00 %) | 0.5075 (outside) |
| `z_rms` | 0.5375 | 0.5389 | 0.5389 | 0.0014 (0.25 %) | 0.4928 (outside) |

| `class_small` | a | b | c | spread (rel) | Phase C |
|---|---|---|---|---|---|
| `coverage_fine` | 0.06683 | 0.06647 | 0.06713 | 0.00067 (1.00 %) | 0.06621 (outside) |
| `rho_err_median` | 0.08772 | 0.08682 | 0.08855 | 0.0017 (1.98 %) | 0.08518 (outside) |
| `fine_rho_err_median` | 0.05193 | 0.05204 | 0.05305 | 0.0011 (2.12 %) | 0.05336 (outside) |
| `depth_err_median_m` | 0.7075 | 0.6917 | 0.7091 | 0.017 (2.47 %) | 0.7058 |
| `fine_depth_err_median_m` | 0.3985 | 0.4009 | 0.4124 | 0.014 (3.44 %) | 0.3871 (outside) |
| `vergence_err_median_m` | 0.8734 | 0.8632 | 0.8882 | 0.025 (2.86 %) | 0.7694 (outside) |
| `z_rms` | 0.6122 | 0.6109 | 0.6407 | 0.03 (4.80 %) | 0.6237 |

| `class_full` | a | b | c | spread (rel) | Phase C |
|---|---|---|---|---|---|
| `coverage_fine` | 0.09431 | 0.09375 | 0.09371 | 0.0006 (0.63 %) | 0.0939 |
| `rho_err_median` | 0.04255 | 0.04211 | 0.04206 | 0.00049 (1.16 %) | 0.04274 (outside) |
| `fine_rho_err_median` | 0.01789 | 0.01774 | 0.01777 | 0.00015 (0.84 %) | 0.01765 (outside) |
| `depth_err_median_m` | 0.3485 | 0.3471 | 0.3469 | 0.0016 (0.46 %) | 0.3507 (outside) |
| `fine_depth_err_median_m` | 0.1367 | 0.1354 | 0.1358 | 0.0012 (0.89 %) | 0.1343 (outside) |
| `vergence_err_median_m` | 0.3763 | 0.3503 | 0.3503 | 0.026 (7.23 %) | 0.3763 |
| `z_rms` | 1.013 | 1.007 | 1.007 | 0.0065 (0.65 %) | 1.006 (outside) |

The full list (17 numbers per run, including `coverage_any`, gross and outlier fractions) is in
`spread.json`.

**The published Phase C numbers are outside today's spread** for most numbers: 15 of 17 on
`calib_small` (fine coverage 0.138 then, 0.175 now), 10 of 17 on `class_small`, 11 of 17 on
`class_full`. The loop's code changed after Phase C: the last commit touching `active_loop.py`,
`belief.py`, `stereo_field.py` or `fixation_pairs.py` is `c991231` (2026-09-18, D23 / D5, the
consensus belief), and the accepted runs' settings lack the options that commit added. **Part 2's
reference is therefore today's legacy code at `48a3139`, not the Phase C table.**

## Trajectories and the event check

- **Gaze directions** are bit-for-bit equal across the three runs at every fixation, except run c
  of `class_small`, which chose other directions at fixations 38–40 (47 of 50 equal).
- **The vergence distance diverges from fixation 1 on**, in every pair: it comes from the loop's
  belief, which fuses each run's own OptiX-noisy renders (relative differences: median 7e-7 to
  2e-4 per configuration, maximum 1.3–14 %). So after fixation 0 the two runs' eyes are posed
  differently, and their images differ for that reason, not by noise alone.
- **The #9 event test** (|mean dRGB| per fixation above 1e-5) assumes identical poses. Applied
  where the pose is identical, fixation 0 of each of the 9 pairs, it finds no event: at most
  1.8e-7 (OptiX noise, as in #9). Applied as written to every fixation, it flags 22–24 per pair,
  all where the vergence differs. Those are not #9-type events:
  - the colour difference follows the vergence difference (log-log correlation 0.88–0.97 in the
    a/b and a/c pairs; 0.31–0.45 between b and c, which are nearly identical, differences at
    noise level);
  - its sign is mixed (for example 27 brighter / 22 darker);
  - the longest run of consecutive fixations darker by more than 1e-5 is 2–3, or 6 in
    `class_small`'s pairs with run c (fixations 28–33, −3.5e-5 to −1.3e-4, with vergence
    differences of 0.2–5 %). #9's event was 8 and 7 consecutive fixations darker by 7.9e-4 to 0.14.
- **Run c was made as the issue asks** when events appear. It adds a third run to the spread and
  shows the same pattern; the fixations whose directions differ (38–40) are left out of the
  colour comparison.

## For part 2

1. Judge the fov_render loop against **today's legacy runs**, these three per configuration, not
   against the Phase C table.
2. The spread with three runs is small on `calib_small` (≤ 0.6 %), 1–5 % on
   `class_small`, ≤ 1.2 % on `class_full` except the vergence error (7.2 %, two of three runs
   identical). Three runs make a narrow interval; more runs would widen it.
3. **An event check for the loop cannot compare the runs' own renders after fixation 0**, because
   their poses diverge. The witness look of #9 (proposal 2: a fixed look re-rendered in the same
   session, compared with its first render) is what would catch an event inside a loop run.
