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

## Phase 1: Chat, local repository, CPU (done, 2026-11-07)

Steps 1–4 of the migration plan, in Chat's sandbox with the `bpy` 5.2.1 wheel on CPU. Report:
[`docs/phase-1-report.md`](phase-1-report.md).

| Step | Gate | Status |
|---|---|---|
| 1 Foveated camera | legacy `check_foveated` passes on a fov_render fixation; fov_render equals legacy bit for bit with the legacy pose | passed, calibration room (Classroom → Phase 2) |
| 2 numpy models | pose agreement with the 3 legacy formulas; legacy self-tests; footprint sums; inverse maps; record and triangulation | passed |
| 3 Config, engine, presets | 7 presets reproduce their legacy paths' settings; spp changes take effect; animated settings are caught | passed (AB1a with OPTIX replaced by CPU) |
| 4 Scene layer | head pose, instance catalogue, sanity, world-only HDRI, material audit, Object Index vs ray cast | passed on the calibration room and two synthetic scenes (Classroom → Phase 2) |

## Phase 2: GitHub, Chat and Code, the workstation

1. **Set up** (done, 2026-11-07). Create `visgraf/fov-render`, push `main`, create the labels `for-code`, `for-chat`,
   `for-luiz`; open one issue per step below.
2. **Phase 1 on the workstation.** Code runs `tools/run_checks.py` with `--device OPTIX` and adds the
   Classroom gates deferred from steps 1 and 4: `check_foveated` on the Classroom, the Classroom head
   pose against the accepted AB1a record, the 234-entry catalogue of Breadth-1, AB1a's own OPTIX spec.
3. **Step 5. Session, server and client** (report §8), then the greedy explorer through
   `client.Session`; gate: f3d-vision `check_equivalence.py` 23/23 against the frozen Playground-2 run.
4. **Step 6. The foveated sensor in the same handler**: fov-3d-vision's `active_loop.py --policy
   coverage` with `Session(in_process=True)`; gate: checks (s)–(v) and the Phase C numbers within the
   spread of one repeated legacy run (to be measured first).
5. **Step 7. A new scene** neither repository has used, through every sensor; gate: every check
   passes and the material audit is reviewed.
