# Status

Rewritten at each milestone; earlier versions are in git.

**Now:** Phase 1 done (2026-11-07). Migration steps 1–4 pass 19/19 gates in Chat's sandbox (bpy
5.2.1 wheel, CPU): [`phase-1-report.md`](phase-1-report.md). The repository is local to Chat and
goes to `visgraf/fov-render` at the start of Phase 2.

**Next:** Phase 2, step 1 of [`PLAN.md`](PLAN.md): create the GitHub repository and the labels,
then Code re-runs the gates on the workstation with OPTIX and adds the Classroom gates.

**Open for Luiz:** none.

**Known limits** (in the report, §5): instances of one object share its Object Index; the
foveated pose is float64 here and float32 in the legacy code (≤ 8.6 µm at 2 m); `Session`, the
server and the JSON requests are not built yet.
