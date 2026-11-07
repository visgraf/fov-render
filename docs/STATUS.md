# Status

Rewritten at each milestone; earlier versions are in git.

**Now:** Phase 2 set up (2026-11-07). `main` is on GitHub at `visgraf/fov-render` with the Phase 1
code: migration steps 1–4, 19/19 gates in Chat's sandbox (bpy 5.2.1 wheel, CPU),
[`phase-1-report.md`](phase-1-report.md). Labels `for-code`, `for-chat` and `for-luiz` exist, and
there is one issue per remaining step of [`PLAN.md`](PLAN.md):

| Issue | Step | Whose turn |
|---|---|---|
| #1 | Phase 1 gates on the workstation with OPTIX, plus the Classroom gates | Code |
| #2 | Step 5: `Session`, server and client; `check_equivalence` 23/23 | Chat (Code for the gate) |
| #3 | Step 6: foveated active loop through `Session`; part 1 measures the legacy spread | Code (part 1) |
| #4 | Step 7: choose the new scene | Luiz |

**Next:** Code takes #1 and part 1 of #3 ("work on #1"); Chat starts #2; Luiz answers #4.

**Open for Luiz:** #4.

**Known limits** (in the report, §5): instances of one object share its Object Index; the
foveated pose is float64 here and float32 in the legacy code (≤ 8.6 µm at 2 m); gates 1.1 and 1.2
need OpenEXR, which the Blender binary's Python lacks (#1 settles it); `Session`, the server and
the JSON requests are not built yet (#2).
