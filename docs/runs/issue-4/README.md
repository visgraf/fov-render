# Issue #4, part A: scene triage for step 7

Code, 2026-11-07. Every number is **measured on the workstation** (`euclid`: RTX 4090, driver
595.99.02, Blender 5.2.1 LTS binary build `9e2066aef7ef`, OPTIX) with
[`triage.py`](triage.py) at the commit of this file, branch `issue-4-scene-triage` from `main`
at `66b67e7` (`complete` numbering, D13). The scenes live outside the repo under
`/home/lvelho/rd/scenes/`.

```
blender -b --python-exit-code 1 -P docs/runs/issue-4/triage.py -- --blend FILE --out DIR [--head-forward-m 1.5]
```

**Luiz discarded the Barcelona Pavilion during the triage** (2026-11-07, "giving too much
trouble"). What it found is kept below because three of its findings are general bugs in the
scene layer that step 7 must fix for any arbitrary file.

## Agent 327 Barbershop

`/home/lvelho/rd/scenes/barbershop/barbershop_interior.blend`, from
`svn.blender.org/.../lib/benchmarks/cycles/barbershop_interior/`. **CC-BY**, by the Blender Studio
(blender.org demo-files page: "Agent 327 Barbershop … (CC-BY, 280MB)"). Reports:
[from the scene camera](triage-barbershop.json), [from a provisional head 1.5 m ahead](triage-barbershop-head1.5.json).

| Item | Measured |
|---|---|
| file | 287,574,804 bytes, sha256 `95972b56180462cac47ec82f3a755bd9111ec18ca37a6196a319c013db994130`; saved with Blender 2.78.5; units `NONE`, scale 1.0; frames 1–3000; Cycles; 1 scene, 1 view layer (renders, is the context layer); no linked libraries |
| images | 197, 190 packed; **7 do not load** (files missing): `agent_face_bruises_color`, `agent_face_bump`, `agent_face_color`, `agent_face_spec`, `generic_scratches.png` (`//__/lib/chars/maps/`), `Dirt_Patch_512.png`, `guilder_ornament.png` (`//__/lib/maps/brushes/`) |
| scripts not run | auto-run off: one registered text, `generate_customprops.py` (BlenRig's bone-hiding UI helper: no effect on renders); no scripted drivers |
| objects | 1,819: 1,155 mesh, 624 empty, 16 armature, 15 light, 8 curve, 1 camera |
| head | no `EYE`: the scene camera `Camera` at (2.968, 9.955, 1.545) m, pitch −2.96°, roll −0.38°. **It stands 0.53 m behind the shop's wall** (`GEO-barbershop_walls`) and sees through it only because its clip start is **1.1 m**; fov_render's sensors clip at 1 mm |
| catalogue | `count` 1,130 = `direct` 1,130, `via_instance` 0 |
| unnamed | no CURVES / POINTCLOUD / VOLUME / GREASEPENCIL objects; 4 hair particle systems rendered as strands (shaving brushes, broom: they take their emitter's id); geometry-nodes modifiers are all "Auto Smooth" (no instancing); 6 curve instances; the 33 meshes with pass index 0 are all hidden from render. Hits with Object Index 0: **0** |
| material audit | **393 of 1,130 objects flagged**: alpha 377, transmission 68, emission 21, volume 3; world volume no. Largest by solid angle from the provisional head: `GEO-cabinet_medium` (alpha, transmission, 1.1 %), `GEO-Lamp_large.003` and `GEO-Lamp_large` (alpha, emission), `GEO-Counter.001`, `GEO-barbershop_door`, a picture frame, the cash register (emission) |
| animation | none on render settings |

| Render (OPTIX) | From the scene camera | From the provisional head (1.5 m ahead: 2.543, 8.519, 1.468) |
|---|---|---|
| sanity, 720 px, 4 spp | holes 29.2 %, back-faces 0.5 %, nadir 1.55 m, zenith 1.09 m, 6.8 s | holes 0.01 %, **back-faces 13.7 %**, nadir 1.47 m, zenith 1.16 m, 6.3 s |
| Combined | **black**: 0.0 on every hit of the foveated and perspective looks (the back of the wall, unlit) | lit: panorama mean 0.044 |
| foveated, small, 64 spp | ran, 8.3 s | ran, 6.1 s |
| perspective pair, small, 64 spp | ran, L 6.1 s / R 1.8 s; sees only the wall | ran, L 6.2 s / R 1.8 s |
| panorama 720 px, 64 spp | ran, 6.2 s; 16 objects seen | ran, 6.2 s; **761 objects seen**; misses 0.01 % |

The file itself renders correctly in Blender 5.2 with its own camera and settings (25 %, 16 spp,
OPTIX, 6.6 s; mean 0.094): the black renders come only from the head's position.

The 13.7 % back-faces are the set: `GEO-barbershop_walls` is 99 % back-face seen from inside, and
`wall_panels.005` / `.003` make up most of the rest. The walls were built to be seen from the camera's
side, outside the room. Opening, raw: 0.8 s; `open_scene`: 1.0 s.

**Verdict: usable for step 7; no blocker in the file.** It needs an `EYE` inside the shop (the
scene camera is outside a wall), and step 7 has to decide how to treat its 13.7 % back-facing walls
(the calibration room's gate asks for 0), its 393 flagged objects and its 7 missing textures.

## Barcelona Pavilion (discarded by Luiz)

`pabellon_barcelona_v1.scene_.zip` from `mirror.blender.org/demo/test/` (24,661,092 bytes, sha256
`0a85bf51…`); `3d/pavillon_barcelone_v1.2.blend` 8,485,449 bytes, sha256 `92162db8…`; saved with
Blender 2.66.6; 6 scenes, 2 view layers. Report: [`triage-pavilion.json`](triage-pavilion.json).

1. **`open_scene` fails: "head rotation must be proper orthonormal".** There is no `EYE`; the scene
   camera's rotation, from Blender's float32 matrix, has ‖RᵀR − I‖∞ = 2.4e-7, and `HeadPose` allows
   1e-7 off the diagonal. Five of the file's seven cameras exceed it (1.2e-7 to 2.4e-7). The triage continued
   with the rotation made orthonormal in float64 (change ≤ 1.2e-7) as an explicit head.
2. **The renders lack Position and Object Index.** The file opens with `RenderLayer.001` as the
   context view layer, which does not render (`use` False), while `RenderLayer` renders and
   `use_single_layer` is off. fov_render sets its passes on the context layer; Blender writes
   `RenderLayer` with the file's own passes (Combined, Depth, Environment). The sanity check and
   all three looks then fail building the record ("distance_m was not rendered").
3. **20,622 particle instances carry pass index 0** (lotus 300, pebbles 20,000, trees 322; rendered
   as collections). The `complete` numbering covers collection-instance empties only.
4. One texture missing (`//textures/water bump.jpg`).

## For step 7 (scene-layer findings, not fixed here)

- **Head from a float32 camera** (Pavilion 1): `rigid4` should return an orthonormal float64
  rotation, or `HeadPose` should accept float32 precision. Any arbitrary camera can hit this.
- **Passes on the view layer that renders** (Pavilion 2): set the passes on the layer Blender will
  render (or make it the context layer and switch the others off), and check the written channels.
- **Instances the catalogue cannot name** (Pavilion 3): particle-system and geometry-nodes instances.
- **The audit does not visit collection-instance objects** (`material_audit` walks `scene.objects`
  only); and on the Barbershop it flags a third of the objects, alpha mostly, so its heuristic
  needs the review the issue asks for.
- **The head inherits the scene camera's pitch and roll** (Barbershop: −3.0°, −0.4°) and position,
  and the camera's own clip start hides what it stands behind. Step 7 should use an explicit `EYE`.
