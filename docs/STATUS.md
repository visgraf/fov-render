# Status

Rewritten at each milestone; earlier versions are in git.

**Now:** the migration is complete (tag `v0.1.0`) and the clean-up is in #16 (PR for Luiz once Code
has re-run the gates). `main` has no `legacy/`, no adapters and no run records; their history is in
[`HISTORY.md`](HISTORY.md), the decisions in [`PLAN.md`](PLAN.md) (D14–D17).

**The gates** (`python -m fov_render.checks`): 25 in the Chat sandbox (`bpy` wheel, CPU), 8 of them
numpy only, plus the 4 Classroom gates on the workstation. Every comparison with the legacy code still
runs: against known answers in `fov_render/checks/known/` (steps 1–4), or against f3d-vision's pinned
files rebuilt from a checkout (the Classroom). The scene gate for any `.blend` is
`python -m fov_render.checks.scene`; the Barbershop's expectations are in `scenes/barbershop.json`.

**Open:** #18, the `fov-render` command line of report §8.6 and the course's scene specs
(`docs/scenes/calib_room.json`, `classroom.json`): Chat's CPU step is on `issue-18-cli`, Code's
workstation step is next.

**Known limits:**
- **OptiX:** not bit-reproducible. Position and Object Index repeat exactly; Combined does not. CPU
  repeats bit for bit on the calibration room, but not on the Classroom (#9).
- **Known answers come from one machine** (the Chat sandbox's CPU). Renders are compared with
  tolerances, never bitwise, so other CPUs agree; numpy answers within 1e-12.
- **Object Index:** instances of one object share its number; geometry-nodes instancers are listed,
  not numbered.
- **The Classroom gates need the workstation:** the Classroom files and an f3d-vision checkout whose
  history has the pinned commit.
