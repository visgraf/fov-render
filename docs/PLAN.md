# Plan

The design is the Claude Doc *Unified Rendering Support for fov-3d-vision and f3d-vision* and the
technical report in [lvelho/fov-render-tr](https://github.com/lvelho/fov-render-tr). This file
holds what was decided and the two phases.

## Decisions (Luiz, 2026-11-07)

| # | Question | Decision |
|---|---|---|
| D1 | Where the package lives | Its own repository, `visgraf/fov-render`, importable as a submodule (the design's proposal, f3d-vision, was not taken) |
| D2 | Default roll for perspective pairs | `upright` until the legacy baselines are reproduced, then `baseline_projected` for 360° work |
| D3 | Device policy | `strict` everywhere except sandbox smoke tests |
| D4 | Record storage | samples plus `raster_index`; rasters rebuilt on demand |
| D5 | Non-opaque materials | tagged by the scene audit; each experiment decides |
| D6 | Colour | `measurement` for records; `scene` only for figure renders, marked DERIVED |
| D7 | Python package name | `fov_render` (f3d-vision already has a `fov3d` package) |
| D8 | Working agreement | the N-Surface method, lighter, with GitHub Issues between Chat and Code; in fov-render only, f3d-vision unchanged |
| D9 | Classroom gates | deferred to Phase 2: the Classroom files cannot be downloaded into Chat's sandbox |

Added in Phase 2 (Luiz, 2026-11-07, recorded in #1):

| # | Question | Decision |
|---|---|---|
| D10 | Gate 1.2 (fov_render image = legacy image, bitwise) on a GPU | always rendered on CPU, whatever `--device` says: OptiX is not bit-reproducible (two legacy runs differ, #1), and the claim is about builder, shader and settings. OptiX's run-to-run variation is measured in #3 |
| D11 | Object Index of objects reached only through collection instances | both numberings, `open_scene(instance_ids="legacy"\|"complete")`. `legacy` (default) leaves them at their file's pass index as the legacy code does (on the Classroom: 116 objects, id 0 or a library's 10, 20, 2, which collide with direct ids); `complete` numbers them after the direct objects. Default stays `legacy` until #2 passes `check_equivalence`, then a Decision issue on switching it |
| D12 | Interpreter for the checks | the `bpy` 5.2.1 wheel (same build as the binary, OptiX under the strict policy, has OpenEXR; needs Python 3.13). The binary stays supported for rendering; under it gate 1.2 SKIPs |
| D13 | Default Object Index numbering, after #2 (Luiz, 2026-11-07, #10) | `complete`. Comparisons with f3d-vision ask for `instance_ids="legacy"` explicitly (`tools/gfe_session.py`, gates 4.2, 4.C2, 5.3) |

## Phase 1: Chat, local repository, CPU (done, 2026-11-07)

Steps 1–4 of the migration plan, in Chat's sandbox with the `bpy` 5.2.1 wheel on CPU. Report:
[`docs/phase-1-report.md` (in v0.1.0)](https://github.com/visgraf/fov-render/blob/v0.1.0/docs/phase-1-report.md).

| Step | Gate | Status |
|---|---|---|
| 1 Foveated camera | legacy `check_foveated` passes on a fov_render fixation; fov_render equals legacy bit for bit with the legacy pose | passed, calibration room (Classroom → Phase 2) |
| 2 numpy models | pose agreement with the 3 legacy formulas; legacy self-tests; footprint sums; inverse maps; record and triangulation | passed |
| 3 Config, engine, presets | 7 presets reproduce their legacy paths' settings; spp changes take effect; animated settings are caught | passed (AB1a with OPTIX replaced by CPU) |
| 4 Scene layer | head pose, instance catalogue, sanity, world-only HDRI, material audit, Object Index vs ray cast | passed on the calibration room and two synthetic scenes (Classroom → Phase 2) |

## Phase 2: GitHub, Chat and Code, the workstation

1. **Set up** (done, 2026-11-07). Create `visgraf/fov-render`, push `main`, create the labels `for-code`, `for-chat`,
   `for-luiz`; open one issue per step below.
2. **Phase 1 on the workstation** (done, 2026-11-07, #1). Code runs `tools/run_checks.py` with `--device OPTIX` and adds the
   Classroom gates deferred from steps 1 and 4: `check_foveated` on the Classroom, the Classroom head
   pose against the accepted AB1a record, the 234-entry catalogue of Breadth-1, AB1a's own OPTIX spec.
   Result: 19/19 on CPU and OPTIX with the wheel, 23/23 with the Classroom gates; decisions D10–D12;
   report [`docs/runs/issue-1/README.md`](https://github.com/visgraf/fov-render/blob/v0.1.0/docs/runs/issue-1/README.md).
3. **Step 5. Session, server and client** (done, 2026-11-07, #2) (report §8), then the greedy explorer through
   `client.Session`; gate: f3d-vision `check_equivalence.py` 23/23 against the frozen Playground-2 run.
   Built in #2 (Chat): `request.py` (schemas `fov-render-session-v1`, `-request-v1`, `-response-v1`),
   `bl/handler.py`, `server.py` (queue protocol of f3d-vision's greedy server), `session.py`
   (`Session`, `Session.in_process`, `Look`), `compat.py` (f3d-vision's calibration, AB1a readback
   and pass report, each checked bit for bit against the legacy function) and `tools/gfe_session.py`
   (the explorer, unchanged, with `run.Server` replaced). Gates 5.1–5.5 (`--steps 5`). Changes on
   the way: the rig computes poses in `make_calibration`'s order (numpy trigonometry, one
   normalisation, an exact sign flip for the OpenCV ↔ Blender frames), so poses equal f3d-vision's bit
   for bit, signed zeros included; before, 275 of 714 eye poses differed in the last bits. The
   control for the gate is the IPD one ULP higher (`--control-ipd-ulp`): a seed changes only RGB,
   which `check_equivalence` reports but does not compare. Not built: the `fov-render` command
   line of report §8.6. Result (Code, workstation): `check_equivalence` 23/23 with the known answer
   through the binary (and through the wheel); the IPD-ULP control fails 14/17; gates 24/24 on
   OPTIX; report [`docs/runs/issue-2/README.md`](https://github.com/visgraf/fov-render/blob/v0.1.0/docs/runs/issue-2/README.md).
4. **Step 6. The foveated sensor in the same handler**: fov-3d-vision's `active_loop.py --policy
   coverage` with `Session(in_process=True)`; gate: checks (s)–(v) and the Phase C numbers within the
   spread of one repeated legacy run (to be measured first). In #3: part 1 (Code) measured the
   spread, 3 runs × 3 configurations at fov-3d-vision `48a3139`; the Phase C table itself no longer
   reproduces (the loop changed after it). Gate (Luiz, #3), replacing "inside min..max", which fails
   by chance: (1) exact, CPU, calibration room: the loop on fov_render (`tools/active_loop_session.py
   --mode exact`) equals the legacy loop in every file and Phase C number; (2) statistical, OPTIX,
   Classroom small and full: 3 runs in `--mode native` (fov_render's own rig and records), each Phase
   C number's mean within one legacy range of the legacy mean (`tools/loop_spread_gate.py`). Only
   the fixation-0 RGB test of #9 is kept. Sandbox gates 6.1–6.3.
   **Outcome (Luiz, #3): accepted on the exact test** (50 fixations, CPU, bit for bit; control
   differs). The statistical gate is withdrawn. It was Chat's design error: its range came from
   runs sharing fixation 0, while the loop is chaotic, and the unchanged legacy loop fails it when
   its IPD moves by 0.2 µm (Code's diagnosis, `runs/issue-3/part2/`). Native mode is kept but not
   validated statistically.
5. **Step 7. A new scene** neither repository has used, through every sensor; gate: every check
   passes and the material audit is reviewed. Scene (Luiz, #4): the Agent 327 Barbershop
   (Blender Studio, CC-BY); the Barcelona Pavilion was triaged and discarded, and its triage found
   four scene-layer bugs, fixed in #4 with gates 4.7–4.10: a float32 camera's rotation is made
   orthonormal when needed; the view layer that renders gets the passes; particle instances of
   sources outside the render are named (`complete`); the audits reach collection instances and
   list missing images. The scene gate for any file is `python -m fov_render.checks.scene`
   (S1–S6); the Barbershop's expectations are `docs/scenes/barbershop.json`, its head comes from
   `tools/make_eye_scene.py`. Settled by Chat, at Luiz's request (#4): the EYE goes 1.5 m along
   the shot camera's view, level, with its yaw (the camera stands behind a wall); the set walls'
   back-faces are accepted as the scene is (allowed by name, any other back-face fails S2); the
   missing images are listed and their objects' solid angle reported, the material audit by solid
   angle from the head (S6) is reviewed with Luiz.

## Clean-up (#16, after the tag `v0.1.0`)

`v0.1.0` (tag object `0df9ff0`, at `ca30e29`) is the end of the migration: the last commit with
`legacy/`, the migration's run records and every comparison against the legacy code.

Decisions (Luiz, 2026-11-07, recorded in #16):

| # | Question | Decision |
|---|---|---|
| D14 | What replaces the comparisons against `legacy/` | mixed. Steps 1–4's comparisons use known answers frozen from the legacy code (`fov_render/checks/known/`, by `tools/freeze_known_answers.py`). The Classroom gates use f3d-vision's pinned files rebuilt from a checkout's git history (`fov_render/checks/pinned.py`, `known/pins.json`) |
| D15 | The migration's history | in the tag. `docs/runs/` and `docs/phase-1-report.md` left `main`; `docs/HISTORY.md` summarises each issue with links into `v0.1.0` |
| D16 | The migration adapters | removed: `compat.py`, `tools/gfe_session.py`, `tools/active_loop_session.py`, `compare_loop_runs.py`, `run_fov3d_loop.py`, `loop_spread_gate.py`, the session fields `formats` and `pose_convention`, gates 5.2, 5.3, 5.5 and 6.1–6.3. Recoverable from `v0.1.0`, though they may need repair against a later fov_render |
| D17 | The tag | `v0.1.0` at `ca30e29`, created before anything was removed |

What changed in the gates that stayed:
- **Gate 1.2** compares fov_render's fixation with the frozen legacy fixation within the pose-precision
  tolerance. Its bitwise clause (rendering with the legacy pose matrix) needed the legacy code, and CPU
  renders are not promised bitwise across machines, so it ran up to `v0.1.0` only.
- **Step 2's legacy self-tests** went: they tested the legacy code itself.
- **Three files from `legacy/` are fixtures now** (`fov_render/checks/fixtures/`, byte-identical,
  hash-checked by gate 0): the calibration-room and HDRI scene builders, with their `bl_common`, and
  fov-3d-vision's `check_foveated`.
- **The migration rule** (CLAUDE.md rule 6, "show it behaves the same, by a check against the copies in
  `legacy/`") now reads against the known answers and the pinned files. Only Luiz changes CLAUDE.md
  (proposed wording in the PR of #16).

## The command line and the course's scene specs (#18)

What #18 adds, in two steps (Chat on CPU, then Code on the workstation):
- **`fov-render`** (`fov_render/cli.py`, console script): `look`, `panorama`, `requests` (host side,
  through `Session`), and `info`, `eye`, `check`, `selftest` (`fov_render/blcli.py` in Blender). Exit
  status 0 / 1 failed / 2 refused before Blender starts. Gates cli.1–cli.5 (step `cli`, in the default
  set).
- **`fov_render/launch.py`**: finding Blender and building its command, shared by `Session` (whose
  command is unchanged, gates 5.4 and cli.2) and the command line.
- **`tools/make_eye_scene.py`** moved into the package (`fov_render/bl/eye_scene.py`); the tool is a
  wrapper. Gate cli.5 checks that `fov-render eye` writes the same EYE as the tool at `v0.1.0`.
- **Scene specs:** `--write-spec` drafts one (to review); a spec with unknown or null expectations, or a
  non-empty `to_measure`, is refused. `docs/scenes/calib_room.json`, `docs/scenes/classroom.json` (to be
  measured).
- **S1's control** was undefined for a scene expected to have no missing images (it compared against
  the list minus its first entry, the same empty list, so S1 always FAILED on such a scene). It now
  compares against one image more when none is expected. Found by gate cli.4 on the calibration room.

- **The run record's device:** `ready.device` was null for the command line's looks (Code, #18), since
  Cycles' device is chosen when a sensor is first set up. The command line now opens its session with
  the first look's sensor prepared (`Session(prepare=…)`); gate cli.2 checks that `ready.device` is set
  and that the images are still bit for bit those of a `Session` opened without it.

**D18 (Luiz, #18): scenes may have their own limits.** The Classroom's window panes open onto a
featureless world: 2.063 % holes (Code, workstation, OPTIX), above the default 0.02. Its spec sets
`max_hole_fraction` 0.025, about a fifth above the measured value, with the reason in its notes. A
draft never raises a limit above the default by itself; a person does, in the scene's spec.
