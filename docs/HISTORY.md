# History: the migration, 2026-11-07

fov-render was built by migrating the rendering code of
[fov-3d-vision](https://github.com/visgraf/fov-3d-vision) and
[f3d-vision](https://github.com/visgraf/f3d-vision) into one package, in seven steps (`PLAN.md`), each
gated by checks with controls that must fail. The tag **`v0.1.0`** is the end of the migration. It is
the last commit with the legacy copies (`legacy/`), the adapters that ran the two repositories' code
on fov_render, and the run records this page links to.

The work was shared by Luiz (decisions, acceptance), Chat (design and code, in a CPU sandbox) and Code
(the RTX 4090 workstation), through GitHub Issues (`CLAUDE.md`).

| Issue | What | Result |
|---|---|---|
| Phase 1 | steps 1–4 in Chat's sandbox: foveated camera, numpy models, config and presets, scene layer | 19/19 gates on CPU; [report](https://github.com/visgraf/fov-render/blob/v0.1.0/docs/phase-1-report.md) |
| #1 | the Phase 1 gates on the workstation, plus the Classroom gates | 19/19 on CPU and OPTIX, 23/23 with the Classroom. Gate 1.2 runs on CPU (D10); two numberings for collection instances (D11); checks with the `bpy` wheel (D12); the `ab1a` preset's passes fixed. [record](https://github.com/visgraf/fov-render/blob/v0.1.0/docs/runs/issue-1/README.md) |
| #2 | step 5: `Session`, the render server, JSON requests | f3d-vision's greedy explorer ran unchanged on a Session and reproduced the frozen Playground-2 run: `check_equivalence` 23/23 with the known answer (356 fixations, 98.82 % of the sphere). The rig was made bitwise equal to f3d-vision's (275 of 714 poses had differed in the last bits). [record](https://github.com/visgraf/fov-render/blob/v0.1.0/docs/runs/issue-2/README.md) |
| #7 | Pillow in the `[checks]` extra | merged (PR #8) |
| #9 | the `bpy` wheel rendered the Classroom darker than the binary | a one-off in two time windows of one run, not the wheel; ignored (Luiz). CPU renders of the Classroom are not bit-reproducible. [record](https://github.com/visgraf/fov-render/blob/v0.1.0/docs/runs/issue-9/README.md) |
| #10 | `complete` as the default Object Index numbering | D13; gates re-run on the workstation |
| #4 | step 7: a new scene | the Barbershop (Blender Studio, CC-BY) through the scene gate S1–S6; the Barcelona Pavilion's triage found four scene-layer bugs, fixed with gates 4.7–4.11 (float32 camera rotations, the view layer that renders, particle instances, audits of instanced objects and missing images, back-faces on open meshes). [record](https://github.com/visgraf/fov-render/blob/v0.1.0/docs/runs/issue-4/README.md) |
| #3 | step 6: fov-3d-vision's foveated active loop on a Session | part 1 measured the legacy loop's spread (the published Phase C table no longer reproduces). Part 2: over 50 fixations on CPU the loop on fov_render equalled the legacy loop bit for bit in every file; the IPD + 1 ULP control differed. The statistical gate was withdrawn as a design error: the loop is chaotic, and the unchanged legacy loop fails it when its IPD moves by 0.2 µm. [part 1](https://github.com/visgraf/fov-render/blob/v0.1.0/docs/runs/issue-3/README.md), [part 2](https://github.com/visgraf/fov-render/blob/v0.1.0/docs/runs/issue-3/part2/README.md) |
| #16 | the clean-up | `legacy/`, the adapters and the run records left `main`; the comparisons became known answers and pinned files (D14–D17 in `PLAN.md`) |

## What `v0.1.0` has that `main` does not

- **`legacy/`**: byte-identical copies of the 31 legacy files, now pinned in
  `fov_render/checks/known/pins.json` and rebuilt from the two repositories' history on demand.
- **The adapters:**
  - `fov_render/compat.py`, with f3d-vision's calibration, AB1a readback and server pass report, and
    fov-3d-vision's gaze matrix;
  - `tools/gfe_session.py`, f3d-vision's explorer on a Session;
  - `tools/active_loop_session.py`, fov-3d-vision's active loop on a Session, in exact and native
    modes (native was never validated statistically);
  - `tools/compare_loop_runs.py`, `tools/run_fov3d_loop.py` and the withdrawn
    `tools/loop_spread_gate.py`.
- **The gates that tested those adapters:** 5.2, 5.3, 5.5 and 6.1–6.3. Also gate 1.2's bitwise
  clause and step 2's legacy self-tests.
- **`docs/runs/`**: every run record of #1–#4, #9 and #10, and the Phase 1 report.

To get any of it back: `git checkout v0.1.0 -- <path>`. Code restored this way may need repair
against a later fov_render.
