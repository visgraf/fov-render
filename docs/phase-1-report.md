# Phase 1 report: migration steps 1–4

Chat, 2026-11-07. All numbers are **MEASURED in the Chat sandbox** (2 vCPU Intel Xeon @ 2.10 GHz,
Python 3.13.18, `bpy` 5.2.1 LTS wheel, Cycles on CPU) unless marked otherwise. None is a GPU,
timing or Classroom number for the project: those come from Code on the workstation in Phase 2.

## 1. What was run

```
pip install bpy==5.2.1 numpy OpenEXR==3.4.15
python -m fov_render.checks --work previews/checks          # 19 gates, 14 s from a clean folder
python -m fov_render.checks --steps 2                       # without bpy: 8 PASS, 3 SKIP
```

Sources: fov-3d-vision `main` at `15eedee`, f3d-vision `main` at `bbee8ef`. The 31 files the checks
compare against are byte-identical copies in `legacy/`, hashed in `legacy/PROVENANCE.json` (gate 0).
Results file: `previews/checks/results.json` (not committed; regenerate with the command above).

Control of the controls: with the rig's yaw sign flipped in a copy of the package, gates 2.3 and
2.4 fail and the runner exits 1.

## 2. Gates

| Gate | Known answer, measured | Control (must fail), measured |
|---|---|---|
| 0 Legacy comparators | 31 files match their hashes; the OSL shader and `exr_lite.py` equal their sources | a one-byte change alters the hash |
| 1.1 Legacy `check_foveated` on a fov_render fixation (calibration room, `small`, 16 spp) | no failure: depth = ray distance to 0.63 µm, warp p99.9 error 0.0016°, 12,492 disc samples, ray saving 50.7× | shader E2 = 2.04 with the record saying 2.0: "warp does not match the formula (p99.9 error 0.0806 deg)" |
| 1.2 fov_render image vs legacy image | with the legacy pose matrix, all 8 channels **bitwise equal** | seed 1 instead of 0: Combined differs by up to 0.948 |
| 1.3 SampleRecord from that render | 12,492 samples = rays per eye; direction vs Position within 0.020°; range from Position vs Depth 0.63 µm | mirrored yaw: 40.0° disagreement |
| 2.1 Legacy self-tests (warp, rig, fsg_geometry) | all pass | E2 perturbed by 1 % inside the legacy warp is caught by its own self-test |
| 2.2 New warp vs legacy warp | directions, footprints, mask and inverse: max difference 0.0; N = 126 (`small`), 253 (`full`) | E2 = 2.02 changes the directions |
| 2.3 Pose agreement, upright roll, 231 gazes | `rig.gaze_rotation` 2.2e-16, `fsg_geometry.camera_rotation_h` 0.0, `render_foveated.gaze_matrix` 3.4e-7 (float32) | yaw sign flipped: caught |
| 2.4 Rig vs f3d-vision `make_calibration`, both rolls, 22 eyes | R_hc and centres to 2.2e-16 | upright roll at (179.9°, −89.9°) turns image +X against the baseline |
| 2.5 Footprint sums | foveated / cap: 1.0059 (`small`), 0.9998 (`full`); perspective / closed form 1 + 1.4e-7; reference view 1 + 2.9e-6; panorama / 4π 1 + 3.2e-6 (720), 1 + 1.3e-7 (3600); analytic eq. (10) vs numeric: p99 0.013 % (`full`) | panorama without cos β: 1.571 × 4π |
| 2.6 `pixel_of` inverts `rays` | all four sensors within 7e-12 px | a flipped direction axis is off by more than a pixel |
| 2.7 Record, access gate, triangulation | npz round trip exact; an oracle field read without oracle access raises; sine rule 1.4e-14 m, midpoint 7.9e-13 m, legacy `rig.triangulate` difference 0 | eyes swapped: no finite distance |
| 3.1 Presets vs legacy settings, 7 paths | `fov_fixation`, `preview360`, `fsg`, `oracle1`, `ab1a`, `breadth1`, `reference_view`: **0 differences** each, over 32 read-back settings and the camera | one changed field per preset is detected, 7 of 7 |
| 3.2 `render_once` | noise ratio 2 vs 32 spp under persistent data: 4.24 (expected 4) | a keyframed `cycles.samples` is caught; after removal the render holds |
| 4.1 Head pose | EYE of the calibration room = (0, 0, 1.6) m, R_wh as built, to 6e-8; equal to f3d-vision `_head_pose` | an explicit head yawed 10° differs |
| 4.2 Instance catalogue | calibration room: 101 objects, numbering equal to f3d-vision `_assign_instance_ids`, stable on reopening; audit scene: 5 direct + 1 reached through collection instances | a renamed object changes the map |
| 4.3 Calibration-room sanity, 720 px, 4 spp | holes 0, back-faces 0, nadir 1.60014 m, zenith 1.60014 m (manifest: 0, 0, 1.60019 at 7200 px) | ceiling removed: 28.1 % holes |
| 4.4 World-only HDRI scene (synthetic 256 × 128 source, legacy `make_hdri_scene`) | column shift 0, median relative error 0.019 %, all pixels misses | eye turned 90°: shift 64 px = W/4 |
| 4.5 Material audit and setting animation | flags exactly Glass→transmission, Cutout→alpha, Lamp→emission, Fog→volume; nothing in the calibration room; removes the keyframes on `cycles.seed` and `view_settings.exposure` | the file opened raw: `render_once` refuses the animated seed |
| 4.6 Object Index vs ray cast | 209 sampled hits, 100 % agree, ids 1–6 seen including the instanced Prop | floor renumbered after the render: 196 mismatches |

## 3. Deviations from the design plan

1. **Classroom gates deferred to Phase 2** (D9). Steps 1 and 4 ran on the generated calibration room
   and two synthetic scenes; the Classroom head pose, 234-entry catalogue and `check_foveated` on
   the Classroom are open.
2. **AB1a compared with OPTIX replaced by CPU.** Its spec pins OPTIX and the sandbox has no GPU; the
   comparison patches only `SP.DEVICE`.
3. **Pose tolerance.** The plan said 10⁻⁹ for all three formulas. `mathutils` matrices are float32,
   so `gaze_matrix` agrees to 3.4e-7; the numpy formulas are held to 10⁻¹² and `mathutils` to 10⁻⁶.
4. **Bitwise equality needs the legacy pose.** fov_render poses cameras in float64; the legacy code
   in float32. With the legacy matrix the images are identical; with fov_render's own pose Position
   moves by at most 8.6 µm and Combined by 0.016 (a seed change: 0.948). The float64 pose is kept.
5. **Analytic footprint gate uses p99 and the sum.** The one exception is the centre pixel of an odd
   raster (1.2 % at `full`), where the analytic limit and a finite difference across the origin
   differ.

## 4. Changes to the design, made while building

- The `bpy` wheel keeps `oslquery` outside `sys.path`; `fov_render.bl.env.prepare()` adds it. Without
  this no custom OSL camera compiles in the wheel.
- Legacy paths only switch passes on; presets do the same (`passes_exact=False`), new configs set
  exactly their passes.
- `trim_passes` lists AOVs and removes them only on request, as the legacy server did.
- For the perspective sensor every raster pixel is a valid sample; the 12° core is
  `PerspectiveModel.core_mask()`, used by the matcher.
- `check_foveated` calls `main()` at import, so it runs as a host-side subprocess
  (`$FOVRENDER_HOST_PYTHON`, else the current interpreter if it has OpenEXR).

## 5. Known limits and open items for Phase 2

- Instances of one object datablock share its Object Index: the pass is per object.
- Gates 1.1 and 1.2 read EXRs with OpenEXR, which Blender's bundled Python lacks. Under the
  Blender binary 1.2 reports SKIP and 1.1 needs `FOVRENDER_HOST_PYTHON` set to the repository
  `.venv`. Phase 2 either installs the `bpy` wheel in the `.venv` (to be checked: OptiX in the
  wheel) or moves 1.2 to `exr_lite` with an uncompressed fixation.
- Not built yet: `Session`, the server and client, JSON requests and the CLI of the report's §8
  (step 5).
