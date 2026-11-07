# Status

Rewritten at each milestone; earlier versions are in git.

**Now:** #2 accepted (Luiz, 2026-11-07). Step 5 is done: `Session`, the render server, the JSON
requests, `compat` and `tools/gfe_session.py`. f3d-vision's greedy explorer runs unchanged on a
fov_render Session and reproduces the frozen Playground-2 run: `check_equivalence` 23/23 with the
known answer (356 fixations, 315 local / 40 global, 98.82 % sr), through the Blender binary on the
workstation; the IPD-one-ULP control fails 14/17. Gates 24/24 on OPTIX. Report:
[`runs/issue-2/README.md`](runs/issue-2/README.md). Before that, #1: the Phase 1 gates on the
workstation and the Classroom gates; decisions D10–D12 in [`PLAN.md`](PLAN.md).

| Issue | Step | Whose turn |
|---|---|---|
| #1 | Phase 1 gates on the workstation with OPTIX, plus the Classroom gates | done |
| #2 | Step 5: `Session`, server and client; `check_equivalence` 23/23 | done |
| #3 | Step 6: foveated active loop through `Session`; part 1 measures the legacy spread | Code (part 1) |
| #4 | Step 7: choose the new scene | Luiz |

**Next:**
- Chat opens the Decision on D11: whether `complete` becomes the default Object Index numbering.
- Code runs part 1 of #3 (the legacy spread, overnight if needed).
- Luiz answers #4.

**Open from #2** (in its report, for Chat): `pyproject.toml`'s `[checks]` extra does not declare
Pillow, which gate 5.5 needs (f3d-vision's `run.py` imports it); f3d-vision's
`check_equivalence.py source` gives 11/12 on the workstation because its import-closure check
counts f3d-vision's own `.venv` as untracked modules; the wheel renders the Classroom's RGB darker
than the binary (geometry and ids bitwise equal), cause not established.

**Open for Luiz:** #4.

**Known limits:**
- **OptiX:** not bit-reproducible (two identical legacy runs differ by up to 4.8e-7 in Combined), so
  equality with legacy on a GPU means "within the spread" (#3). Position and Object Index repeat
  exactly, which is what Playground-2's equivalence compares.
- **Object Index:** under the `legacy` numbering, objects reached only through collection instances
  keep their library's pass index (on the Classroom 112 read 0, 4 collide with direct ids);
  `complete` fixes this (D11). Instances of one object share its Object Index.
- **RGB:** measure with the Blender binary until the wheel's difference is understood (D12: the
  binary stays supported for rendering). Under the binary gate 1.2 SKIPs (no OpenEXR).
- **Not built yet:** the `fov-render` command line of report §8.6.
