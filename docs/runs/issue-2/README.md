# Issue #2: the equivalence gate on the workstation

Code, 2026-11-07. Every number here is **measured on the workstation** (`euclid`: RTX 4090,
driver 595.99.02, Blender 5.2.1 LTS binary `/usr/local/bin/blender` build `9e2066aef7ef`) unless
labelled "Chat sandbox" ([`chat-sandbox.json`](chat-sandbox.json)). fov-render at `32cd5eb`
(code `1fe65cb`), branch `issue-2-session`. f3d-vision at `11bfbf6` (`main`, no tracked changes;
nothing edited there, runs written to its `previews/`). Frozen run: `playground-sparse-lattice`,
code `70d992ee` (branch `playground/sparse-lattice`). Commands: Chat's hand-over comment in #2,
with `F3D=/home/lvelho/rd/f3d-vision`, `RUNS=$F3D/previews`.

## Results

| Step | Run | Result | Wall |
|---|---|---|---|
| 0 | gates, wheel, OPTIX ([`workstation-gates.json`](workstation-gates.json)) | 24/24 PASS | 18.7 s |
| — | sanity, `check_equivalence.py source` ([`source-sanity.json`](source-sanity.json)) | 11/12, see below | — |
| 1 | `fr2-prefix75`, binary, 75 fixations ([`equivalence-prefix75.json`](equivalence-prefix75.json)) | `run.py checks` PASS; equivalence 17/17 PASS | 75.9 s |
| 2 | `fr2-control`, `--control-ipd-ulp`, 3 fixations ([`equivalence-control-ipd-ulp.json`](equivalence-control-ipd-ulp.json)) | `run.py checks` PASS; equivalence **FAIL 14/17**, as required | 4.5 s |
| **3** | **`fr2-baseline`, binary, the gate** ([`equivalence-baseline.json`](equivalence-baseline.json)) | **23/23 PASS, known answer included** | 419.9 s + evaluate 27.5 s |
| 4 | `fr2-baseline-wheel`, wheel launcher, report only ([`equivalence-baseline-wheel.json`](equivalence-baseline-wheel.json)) | 23/23 PASS | 430.7 s + evaluate 27.2 s |

Wall times of each command: [`walltimes.txt`](walltimes.txt) (whole process, from the shell).

**Step 2 fails for the right reason only:** `calibration.json` byte-identical 1..3 (all three
differ), `points.npz` (`err_mm` differs at 1..3; `xyz_h` and `instance_id_oracle` equal), and the
per-fixation records (`err_vs_left_position_mm_diagnostic` at 1..3, `fusion` at 2). Nothing else
fails; `run.py checks` passes.

**Step 3, the known answer:** 356 fixations, stop "no valid unseen fixation", 315 local / 40
global, seen 100.00 %, depth 99.26 %, cells 98.93 % (253,023 / 255,758), solid angle 98.82 %.
Final map 16,943,073 surfels, as the frozen run. `coverage.npz` and
`final-map-oracle-segmentation.ply` are byte-identical to the frozen run's (their freeze hashes
match); `final-map.npz`, `final-map.ply` and `trajectory.json` differ only in RGB and run metadata.

## Times (from each run's `trajectory.json`, measured by `run.py`)

| Run | Fixations | Explorer wall | Render sum | Server ready |
|---|---|---|---|---|
| frozen `playground-sparse-lattice` (legacy server, binary) | 356 | 416.40 s | 270.7 s | 0.502 s |
| `fr2-baseline` (Session, binary) | 356 | 416.40 s | 273.3 s | 0.502 s |
| `fr2-baseline-wheel` (Session, wheel) | 356 | 427.15 s | 284.1 s | 0.553 s |

The equal walls (416.39735 s vs 416.39766 s) are a coincidence, checked: the per-stage sums differ
(render +2.6 s, fusion −2.0 s, correspondence −0.15 s). Server ready times are quantised: run.py
and Session both poll the ready file every 0.05 s; the three binary runs read 0.502–0.503 s.
Inside the server (Session's own `start_seconds`): 0.126 s with the scene loaded on the command
line, 0.418 s with `open_mainfile` (wheel).

## RGB (presentation only, not compared by check_equivalence)

[`rgb-binary-vs-wheel.json`](rgb-binary-vs-wheel.json): `points.npz` RGB of all 356 fixations
against the frozen run, 66,248,853 values.

| | max | mean | values > 1e-3 | values > 0.1 | fixations with a value > 0.1 |
|---|---|---|---|---|---|
| binary (gate) | 0.030 | 2.1e-8 | 0.0003 % | 0 | 0 |
| wheel | 2.00 | 3.1e-4 (all darker) | 1.2 % | 15,509 | 15 (13,211 in `fix-0355`) |

The binary's differences are OptiX run-to-run noise (#1). The wheel's are systematic: every
differing value is darker, concentrated in a few fixations. The two runs' ready records are equal
except the launcher, pid, start time and how the scene was opened (command line vs
`open_mainfile`); neither server log has a warning. Cause not established (report-only step).

## Findings

1. **5.5 needs Pillow** in the interpreter that runs the checks: it runs `tools/gfe_session.py`,
   which imports f3d-vision's `run.py` → `visuals.py` → `PIL`. The first step-0 run was 23/24
   (`No module named 'PIL'`); installing f3d-vision's pin `pillow==12.3.0` into `.venv` gave 24/24.
   `pyproject.toml`'s `[checks]` extra lists only OpenEXR.
2. **`check_equivalence.py source` is 11/12 here** with the command given. The failing check,
   "every loaded repository module is tracked", counts every loaded file under the repository
   root, and f3d-vision's `.venv` is inside it: the 114 "untracked" files are all `.venv`
   site-packages (`PIL`, `cv2`, `numpy`). The 16 repository files in the closure are all tracked,
   none is NS1e lineage; the other 11 checks (tags, byte-identical sources) pass. f3d-vision is at
   its accepted code. `run` mode does not include this check.
3. **The wheel's RGB differs from the binary's** on the Classroom (above); geometry, instance ids,
   calibrations and every compared product are bitwise equal either way.
