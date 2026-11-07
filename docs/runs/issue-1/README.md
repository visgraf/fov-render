# Issue #1: Phase 1 gates on the workstation, and the Classroom gates

Code, 2026-11-07. Every number here is **measured on the workstation** (`euclid`) at commit
`2641260` on branch `issue-1-workstation-gates`, unless it is labelled "Chat sandbox" (those come
from [`docs/phase-1-report.md`](../../phase-1-report.md)). The re-run after Chat's review
(D10–D12, commit `46078d8`) is the [last section](#re-run-after-chats-review-46078d8): all PASS.

**Machine.** NVIDIA GeForce RTX 4090 (24 GB), driver 595.99.02, CUDA 13.2; Intel Core i9-14900KF;
Linux 7.0.0-42-generic. Blender 5.2.1 LTS binary (`/usr/local/bin/blender`, build `9e2066aef7ef`,
Python 3.13.13, numpy 2.3.4, no OpenEXR). `bpy` 5.2.1 wheel (same build `9e2066aef7ef`) in `.venv`
on Python 3.13.12 (miniconda), OpenEXR 3.4.15, numpy 2.5.4. Classroom inputs: f3d-vision checkout
at `11bfbf6`, files checked against their pins (below).

## Runs

```
/home/lvelho/miniconda/bin/python3.13 -m venv .venv && . .venv/bin/activate
pip install -e ".[bpy,checks]"
python -m fov_render.checks --device CPU   --work previews/ws-cpu                                # ws-cpu
python -m fov_render.checks --device OPTIX --work previews/ws-optix                              # ws-optix
FOVRENDER_HOST_PYTHON=$PWD/.venv/bin/python \
  blender -b --python-exit-code 1 -P tools/run_checks.py -- --device OPTIX --work previews/ws-bin # ws-bin
python -m fov_render.checks --device OPTIX --steps 1 2 3 4 classroom --work previews/ws-all      # ws-all
FOVRENDER_HOST_PYTHON=$PWD/.venv/bin/python blender -b --python-exit-code 1 -P tools/run_checks.py \
  -- --device OPTIX --steps classroom --work previews/ws-bin-classroom                           # ws-bin-classroom
python docs/runs/issue-1/diagnostics.py --work previews/issue-1-diag                            # diagnostics
```

| Run | Interpreter, device | Result | Exit | Wall time |
|---|---|---|---|---|
| [`ws-cpu`](ws-cpu.json) | wheel, CPU | 19 / 19 PASS | 0 | 4.1 s |
| [`ws-optix`](ws-optix.json) | wheel, OPTIX | 18 PASS, **1 FAIL (1.2)** | 1 | 8.0 s |
| [`ws-bin`](ws-bin.json) | binary, OPTIX | 18 PASS, 1 SKIP (1.2: no OpenEXR) | 0 | 6.8 s |
| [`ws-all`](ws-all.json) | wheel, OPTIX, steps 1–4 + classroom in one process | 22 PASS, **1 FAIL (1.2)** | 1 | 10.5 s |
| [`ws-bin-classroom`](ws-bin-classroom.json) | binary, OPTIX, classroom | 5 / 5 PASS | 0 | 3.1 s |

Wall times: [`walltimes.txt`](walltimes.txt), from a clean `--work` folder each time.

**OPTIX in the wheel.** `setup_device` with `device_policy="strict"` returns `OPTIX` and uses the
RTX 4090 (`cycles.device = GPU`, the 4090 the only device in use), in the wheel and in the binary.
Control: with `CUDA_VISIBLE_DEVICES=` the strict policy raises instead of falling back to CPU.

## Gate 1.2 fails under OPTIX: OptiX is not bit-reproducible

| Gate 1.2 quantity | Chat sandbox, CPU | workstation, CPU | workstation, OPTIX | limit in the gate |
|---|---|---|---|---|
| legacy pose vs legacy image, max over 8 channels | 0 | 0 | 3.6e-7 (Combined), 1.8e-7 (Normal) | `== 0` |
| own float64 pose vs legacy, Combined max | 0.016 | 0.0159 | **0.0839** | < 0.05 × seed control = 0.0474 |
| seed 1 vs seed 0 (control), Combined max | 0.948 | 0.948 | 0.948 | > 0 |

[`diagnostics.py`](diagnostics.py) → [`repeatability.json`](repeatability.json): each of the four
renders done twice with identical inputs.

- On CPU every repeat is bitwise equal (the control), and legacy = fov_render-with-legacy-pose bitwise.
- On OPTIX the **legacy renderer differs from itself** by up to 4.8e-7 in Combined and 1.8e-7 in
  Normal between two identical runs; Position and Depth repeat exactly. Legacy vs fov_render with
  the legacy pose differs by the same amount. The bitwise clause cannot hold on OPTIX for any code.
- The own-pose Combined difference is the same on both OPTIX repeats (0.08387): it is not noise, it
  is the max over 15,876 pixels, decided by 2 of them. Median 1.3e-7, p99.9 2.0e-5 on OPTIX (CPU:
  1.2e-7 and 2.9e-5). The worst pixel is not on a depth edge (0.28 mm depth range around it): a
  3.5e-7 rad ray change sends one 16-spp path down another branch.

The check and its thresholds are unchanged (rule 2). The gate is Chat's to redesign; see the issue comment.

## Under the Blender binary

`FOVRENDER_HOST_PYTHON` fixes 1.1 (check_foveated runs in the `.venv` Python). Control: without it,
1.1 fails under the binary ("No module named 'OpenEXR'" from `python3` on PATH) and the runner exits
1. It does not fix 1.2, which imports OpenEXR inside Blender's own Python: 1.2 SKIPs.

## Classroom gates (`--steps classroom`, new: `fov_render/checks/classroom.py`)

Inputs, each checked against the sha256 pinned in `legacy/` before use (absent → SKIP, hash differs → FAIL):
`scenes/classroom/classroom_eye.blend` (`ab1a_spec.BLEND_SHA256`), Controller-01
`bootstrap/seeds.json` (`ab1a_spec.HEAD_POSE_SOURCE`), `bootstrap/instance_catalog.json`
(`breadth1_spec.SOURCES`), the AB1a `acquisition/acquisition.json` (`ns1a_spec.AB1A_ACQUISITION`).
Controls of that: with `FOVRENDER_F3D_ROOT` an empty folder all four SKIP (exit 0); with one byte
added to `seeds.json` all four FAIL (exit 1).

| Gate | Known answer, measured (OPTIX) | Control, measured |
|---|---|---|
| 1.C `check_foveated` on a fov_render fixation of the Classroom (`small`, 16 spp, gaze (20°, −5°)) | no failure: depth = ray distance to 0.84 µm, warp p99.9 0.0016°, 12,492 disc samples, outside-disc hits 0, ray saving 50.7× | E2 = 2.04: "warp does not match the formula (p99.9 error 0.0806 deg)" |
| 3.C `open_scene` + `preset("ab1a", device="OPTIX")` vs AB1a's own `configure` (OPTIX, gaze #1, 256 spp) | 0 differences; AB1a's own readback of the fov_render scene equals the accepted record's settings; legacy readback equals it too; both back-ends OPTIX | Depth pass on: caught by both comparisons; CPU: `cycles_device` caught by both |
| 4.C1 head pose | `resolve_head` vs seeds record, AB1a record and legacy `_head_pose`: 0.0, 0.0, 0.0 (tolerance `EYE_POSE_TOL` 1e-6) | origin + 1 mm: 1.0e-3 |
| 4.C2 catalogue | ids 1–234 equal the accepted Breadth-1 catalogue and live `_assign_instance_ids` (names, ids, order); 116 more entries 235–350 (below) | one entry dropped; one object hidden from render: both caught |

## Flags

1. **3.C needs the scene layer.** `preset("ab1a")` on the raw Classroom file (step 3's method) has 1
   difference: the file has Depth and Normal on, AB1a's `configure` turns them off, the preset only
   switches passes on (`passes_exact=False`). Through `open_scene`, which trims passes, 0
   differences. Phase 1's 0 on the calibration room held because that file has them off.
2. **The Classroom catalogue has 350 entries, not 234.** 116 objects reached only through
   collection instances (linked libraries: chairs, books, lamps, …; 17 parents) are numbered
   235–350. [`linked-ids.json`](linked-ids.json): from the head they cover 16.7 % of the sphere's
   solid angle; under the legacy numbering those pixels read id 0 (15.4 %) or the library files'
   own pass indices 10, 20, 2 (1.3 %), which are the ids of *other* catalogue entries. Elsewhere
   the two numberings give identical ids. Object Index rasters therefore differ from legacy on
   those pixels.
3. **Numbers vs the Chat sandbox.** Every Phase 1 number in `phase-1-report.md` §2 is reproduced
   on CPU and OPTIX within its gate tolerance (1.1 depth 0.63 µm, warp 0.0016°; 1.3 0.0198°;
   3.2 ratio 4.241 CPU / 4.241 OPTIX; 4.3 nadir and zenith 1.60014 m; 4.4 median 0.019 %;
   4.6 209 hits, 100 %), except gate 1.2 under OPTIX above.
4. **Set-up.** The shell's default `python3` here is 3.10 (conda `world2018`); `pyproject` needs
   ≥ 3.11 and the wheel 3.13, so `.venv` is built from miniconda's 3.13. `build/lib/` (a copy of
   `fov_render/` from an install) is tracked in git since `c52c731` and not in `.gitignore`.

## Re-run after Chat's review (`46078d8`)

Chat's changes: gate 1.2 on CPU whatever `--device` says (D10); both instance numberings,
`open_scene(instance_ids="legacy"|"complete")`, legacy the default (D11); checks with the wheel
(D12); the `ab1a` preset sets its passes exactly; step 3 also on an all-passes copy of the room;
3.C also gates the preset on the raw Classroom file. Same machine and versions as above.

```
python -m fov_render.checks --device CPU   --work previews/ws2-cpu
python -m fov_render.checks --device OPTIX --work previews/ws2-optix
python -m fov_render.checks --steps 1 2 3 4 classroom --device OPTIX --work previews/ws2-all
```

| Run | Result | Exit | Wall time |
|---|---|---|---|
| [`ws2-cpu`](ws2-cpu.json) | 19 / 19 PASS | 0 | 4.6 s |
| [`ws2-optix`](ws2-optix.json) | 19 / 19 PASS | 0 | 8.0 s |
| [`ws2-all`](ws2-all.json) | 23 / 23 PASS (integrity + 22) | 0 | 10.7 s |

- **1.2** under `--device OPTIX`: `requested_device` OPTIX, all four renders on CPU, legacy pose
  bitwise equal (0.0), own pose Combined 0.0159 (limit 0.0474).
- **3.1**: 13 comparisons (7 paths, 6 also on the all-passes room), 0 differences, every control caught.
- **3.C**: `preset_on_raw_file_differences: []`, through the scene layer `[]`, against the
  accepted record `[]`; both back-ends OPTIX.
- **4.C2** legacy: `via_id_zero` 112, `via_nonzero_ids` [2, 10, 20] (as in
  [`linked-ids.json`](linked-ids.json)), 4 collisions (e.g. `ChamferBox01 [lib.003]` id 20 =
  `Cube.016`). Independent count on the raw file after the legacy `_assign_instance_ids`: 116
  collection-instance objects, pass indices {0: 112, 2: 1, 10: 2, 20: 1}, 4 equal to a direct id;
  the 117th linked object is a light (`Point.001 [lib]`). Complete: `count` 350, ids 235–350 with
  no gaps, no collisions. Controls: one entry dropped, one object hidden, complete ≠ legacy: all caught.
- **4.2** audit scene: legacy `Prop` = 2 collides with `Floor`; complete `Prop` = 6; controls caught.
