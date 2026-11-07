# Status

Rewritten at each milestone; earlier versions are in git.

**Now:** #2, step 5, built by Chat on `issue-2-session`, waiting for Code's gate on the workstation.

- **Done in #2:**
  - `Session`, `Session.in_process` and `Look`, the render server, the JSON request schemas,
    `compat` (f3d-vision's formats) and `tools/gfe_session.py`.
  - Gates 5.1–5.5 pass in the Chat sandbox (`bpy` 5.2.1 wheel, CPU): 24/24 with steps 1–4
    ([`runs/issue-2/chat-sandbox.json`](runs/issue-2/chat-sandbox.json)).
  - 5.3 reproduces f3d-vision's greedy render server bit for bit on CPU, in-process and through the
    server: ready record, calibrations, all EXR channels.
- **Before that, #1** (accepted 2026-11-07): the Phase 1 gates pass on the workstation (RTX 4090, the
  wheel), 19/19 on CPU and OPTIX and 23/23 with the Classroom gates; decisions D10–D12 in
  [`PLAN.md`](PLAN.md).

| Issue | Step | Whose turn |
|---|---|---|
| #1 | Phase 1 gates on the workstation with OPTIX, plus the Classroom gates | done |
| #2 | Step 5: `Session`, server and client; `check_equivalence` 23/23 | Code (the gate) |
| #3 | Step 6: foveated active loop through `Session`; part 1 measures the legacy spread | Code (part 1) |
| #4 | Step 7: choose the new scene | Luiz |

**Next:**
- Code runs the Playground-2 explorer through `tools/gfe_session.py` and `check_equivalence` (#2),
  and part 1 of #3.
- Luiz answers #4.

**Open for Luiz:** #4.

**Known limits:**
- **OptiX:** not bit-reproducible (two identical legacy runs differ by up to 4.8e-7 in Combined), so
  equality with legacy on a GPU means "within the spread" (#3). Position and Object Index do repeat
  exactly, which is what Playground-2's equivalence compares.
- **Object Index:**
  - Under the `legacy` numbering, objects reached only through collection instances keep their
    library's pass index. On the Classroom, 112 read 0 and 4 collide with direct ids. `complete`
    fixes this; the default switches only after #2 (D11).
  - Instances of one object share its Object Index.
- **Blender binary:** gate 1.2 SKIPs under it (no OpenEXR).
- **Not built yet:** the `fov-render` command line of report §8.6.
