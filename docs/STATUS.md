# Status

Rewritten at each milestone; earlier versions are in git.

**Now:** #1 accepted (Luiz, 2026-11-07). The Phase 1 gates pass on the workstation (RTX 4090,
`bpy` 5.2.1 wheel): 19/19 on CPU and on OPTIX, and with the Classroom gates (1.C, 3.C, 4.C1, 4.C2,
`--steps classroom`) 23/23 on OPTIX. Report: [`runs/issue-1/README.md`](runs/issue-1/README.md).
Decisions from #1 are D10–D12 in [`PLAN.md`](PLAN.md): gate 1.2 renders on CPU; both Object Index
numberings, `legacy` the default; checks run with the wheel.

| Issue | Step | Whose turn |
|---|---|---|
| #1 | Phase 1 gates on the workstation with OPTIX, plus the Classroom gates | done |
| #2 | Step 5: `Session`, server and client; `check_equivalence` 23/23 | Chat (Code for the gate) |
| #3 | Step 6: foveated active loop through `Session`; part 1 measures the legacy spread | Code (part 1) |
| #4 | Step 7: choose the new scene | Luiz |

**Next:** Code runs part 1 of #3 (the legacy spread, overnight if needed); Chat builds #2; Luiz
answers #4.

**Open for Luiz:** #4.

**Known limits:** OptiX is not bit-reproducible (two identical legacy runs differ by up to 4.8e-7 in
Combined), so equality with legacy on a GPU means "within the spread" (#3). Under the `legacy`
numbering, objects reached only through collection instances keep their library's pass index (on
the Classroom 112 read 0, 4 collide with direct ids); `complete` fixes this, and the default
switches only after #2 (D11). Instances of one object share its Object Index. The foveated pose is
float64 here and float32 in the legacy code (≤ 8.6 µm at 2 m). Under the Blender binary gate 1.2
SKIPs (no OpenEXR). `Session`, the server and the JSON requests are not built yet (#2).
