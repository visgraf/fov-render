# Issue #9: why the wheel rendered the Classroom darker

Code, 2026-11-07. Every number is **measured on the workstation** (`euclid`: RTX 4090, driver
595.99.02, Blender 5.2.1 LTS binary build `9e2066aef7ef`; `bpy` 5.2.1 wheel, same build, in
`.venv`). Branch `issue-9-wheel-rgb` from `main` at `0e7e433` (the issue says `a0ec916`; `main`
had moved by the Pillow fix of #7, nothing else). f3d-vision `main` at `11bfbf6`, no tracked
changes; one run folder written to its ignored `previews/`, as in #2.

## Answer

**The wheel is not the cause.** The darkening in #2's `fr2-baseline-wheel` was a one-off event
during that run, confined to two windows of wall-clock time. None of the following reproduces it:

| Experiment | Result |
|---|---|
| Fixation 355's pair alone, CPU 16 spp, {binary, wheel} × {command line, `open_mainfile`}, each twice | every mean equal (below); no wheel or load-path effect |
| The same on OPTIX, 64 spp | every mean equal to 6 decimals |
| Fixations 250–267 in one session: wheel in-process; binary, command line | image means equal at every fixation |
| Fixations 1–267 through a wheel **server** (`Session`, `launcher="python"`), as the explorer | equal to the binary at 250–267 |
| **#2's step 4 again, the same command** (`fr9-baseline-wheel-2`) | `check_equivalence` 23/23 with the known answer; RGB vs frozen max 0.046, **no point off by > 0.1** |

The cause of the event itself is not established (see "Open").

## What the event was

[`runs-vs-frozen.json`](runs-vs-frozen.json) ([`compare_runs.py`](compare_runs.py)): point RGB
per fixation against the frozen run. Geometry and Object Index are bitwise equal in every run.

| Run | max \|ΔRGB\| | points off > 0.1 | fixations |
|---|---|---|---|
| `fr2-baseline` (binary, #2) | 0.030 | 0 | — |
| `fr2-baseline-wheel` (wheel, #2) | 2.00 | 6,179 | 259–266, 350–356 |
| `fr9-baseline-wheel-2` (wheel, this issue) | 0.046 | 0 | — |

- Fixations 259–266 were rendered from 17:25:49.9 to 17:25:59.1; 350–356 from 17:27:53.4 to
  17:28:01.2, the end of the run (file times). Fixations before, between and after are clean.
- In those fixations many surfaces are darker by 0.08–0.10 (walls, corkboard, beams, pipes, a
  drawing) and the two blinds (`Cube.014`, `Cube.011`, material `paintedBlind`: a plain Diffuse
  BSDF, no texture) go almost black in fixation 355: point mean 0.0017 against 0.974. One light's
  contribution missing fits; which one is not shown. The blinds rendered correctly in that same
  run at fixations 88–97, 286 and 313–314.
- The gazes do not explain it: 259–266 look at yaw −90° to −57°, 355 at +96°.
- Render times inside the windows are unchanged.

## The 2×2 and the fingerprints

[`results.json`](results.json) ([`render_cell.py`](render_cell.py), [`summarise.py`](summarise.py)):
fixation 355, the explorer's look (preset `ab1a`, seeds 2111/2112, calibration byte-identical to
the frozen run's).

- **CPU is not bit-reproducible on the Classroom** (it was on the calibration room, #1). The same
  cell rendered twice differs in 2,000–3,500 of 409,600 pixels per eye, max 0.086, spread thinly
  (under 1 % of any object's pixels; id 0 objects, `woodBase`, walls, window frames). So the
  cell-to-cell pixel differences (2,040–4,022 pixels) are of the repeat's size and attribute nothing.
- The means do: image mean and the means over both blinds and both window frames are identical to
  5–6 decimals in all six CPU renders, and in all six OPTIX renders. Position and Object Index are
  bitwise equal everywhere.
- **Fingerprints** ([binary](fingerprint-binary-cmdline.json), [wheel](fingerprint-wheel-open.json)):
  command line vs `open_mainfile` differ in one field (the order of
  `preferences.edit.key_insert_channels`). Binary vs wheel differ in 16, none render-relevant:
  paths (resource path, Python, OCIO config path, the `OCIO` variable the binary sets; the two
  config files have the same sha256), Python 3.13.13 vs 3.13.12, `factory_startup` True under
  the wheel, audio and NDOF build options, audio device, UI scale, `compute_device_type` saved in
  the user's preferences. Images: the same 52 entries in both; every file-backed one exists and
  loads, with the same file, sha256, colour space and size. Scene, Cycles and
  colour-management settings are equal.

## A check that would have caught it

The event shows in a per-fixation statistic with three orders of magnitude to spare:

| | per-fixation \|mean ΔRGB\| vs frozen |
|---|---|
| clean: `fr2-baseline`, `fr9-baseline-wheel-2` (712 fixations) and the 341 clean fixations of `fr2-baseline-wheel` | ≤ 4.4e-7 |
| the 15 darkened fixations | ≥ 7.9e-4 |

Proposal, for Chat to build:
1. **Against a frozen run** (as #2): a fov-render gate, `compare_runs.py` made into a check, that
   fails when any fixation's \|mean ΔRGB\| exceeds 1e-5. It fails `fr2-baseline-wheel` at exactly
   its 15 fixations and passes the two clean runs.
2. **Without a frozen run** (#3, new scenes): a witness look, re-rendered every K looks in the same
   session, whose image mean must stay within the OptiX spread of its first render (measure the
   spread first; here identical renders agree in the mean to 6 decimals). And each look's `meta`
   records its image mean and per-object means, so a scan like the one above needs no points.

An environment fingerprint in `ready` would not have caught this: the fingerprints are equal.

## Open

- **The cause of the event.** Candidates: a transient GPU or driver fault, interference on the
  shared GPU (it also drives the display), or a rare race in Cycles' persistent-data update. The
  kernel log would show a fault, but this account cannot read it; with `sudo`:
  `journalctl -k --since "2026-11-07 17:20" --until "2026-11-07 17:30" | grep -i -E "xid|nvrm"`.
- **One event in four full-length runs** (the frozen legacy run and `fr2-baseline` on the binary,
  `fr2-baseline-wheel` and `fr9-baseline-wheel-2` on the wheel; 1,424 fixations) plus 303 replayed
  fixations. That it happened under the wheel cannot be told apart from chance.
- **For #3**, which measures Combined on OPTIX: an event like this inside a spread measurement
  would read as spread. The witness check (2) belongs in it.
