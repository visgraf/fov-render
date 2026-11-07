# fov-render

Blender Cycles rendering for gaze-directed binocular observers. Two eyes on a fixed head, four
sensors behind one rig, and one sample record whatever the sensor:

| Sensor | Camera | Used for |
|---|---|---|
| `Foveated` | OSL log-polar warp, `CUSTOM` | fixations in an active loop |
| `Perspective` | pinhole tangent pair, 12° core + margin | local stereo, oracle correspondence |
| `Panorama` | equirectangular | references, noise floor, evaluation |
| `ReferenceView` | wide pinhole | figures |

It unifies the render support of [fov-3d-vision](https://github.com/visgraf/fov-3d-vision) and
[f3d-vision](https://github.com/visgraf/f3d-vision). The design and its mathematics are in the
technical report *Foveated Stereo Rendering in Blender: A Unified Sensor Framework*
([lvelho/fov-render-tr](https://github.com/lvelho/fov-render-tr)).

**Status:** Phase 1 (migration steps 1–4) done in a CPU sandbox; Phase 2 (server, active loop, new
scene, GPU) is next. See [`docs/STATUS.md`](docs/STATUS.md) and [`docs/PLAN.md`](docs/PLAN.md).

## Use it as a submodule

```
git submodule add https://github.com/visgraf/fov-render external/fov-render
pip install -e external/fov-render            # numpy side; add [bpy] for the bpy wheel
```

The numpy side imports anywhere (host scripts, notebooks, Blender):

```python
from fov_render import sensors, Rig, Gaze, HeadPose

m = sensors.Foveated(E2_deg=2.0, e_max_deg=45.0).model(profile="full")   # 253 x 253, 50,269 rays
rays = m.rays()                     # Blender-camera directions, footprints (sr), valid mask
pose = Rig(roll="baseline_projected").pose(head, Gaze.yaw_pitch(10, -5, vergence_m=2.1), "L")
```

The Blender side (`fov_render.bl`) needs `bpy`: Blender 5.2.1 on the workstation, or
`pip install bpy==5.2.1` on any machine for CPU rendering.

```python
from fov_render.bl import cameras, config, engine, scene

h = scene.open_scene("room.blend")                      # head from EYE, sanitised, objects numbered
cfg = config.RenderConfig(profile="small", device="OPTIX")
config.apply(h.scene, cfg, m.width, m.height)
cam = cameras.build(h.scene, m, cfg)
cameras.pose(cam, Rig().pose(h.head, Gaze.yaw_pitch(0, 0), "C"))
engine.render_once(h.scene, cfg.resolved_spp(), 0, "look/C.exr")
```

`Session`, the persistent render server and the JSON requests of the report's §8 arrive in
Phase 2.

## Checks

Every module ships a check with a control that must fail. Run them all:

```
python -m fov_render.checks --device CPU                                   # bpy wheel, or numpy only
blender -b --python-exit-code 1 -P tools/run_checks.py -- --device OPTIX   # the workstation
```

Without `bpy` the Blender gates are reported SKIP, never PASS. Results go to
`previews/checks/results.json`. The checks compare against byte-identical copies of the legacy
code in `legacy/` (provenance and hashes in `legacy/PROVENANCE.json`).

## Layout

```
fov_render/            numpy side: profiles, warp, rig, geometry, sensors/, record, exr_lite
fov_render/bl/         Blender side: env, config (RenderConfig, presets), engine, passes, cameras, scene
fov_render/checks/     the gates of the migration plan
legacy/                comparators, never edited
tools/run_checks.py    gates under the Blender binary
docs/                  PLAN, STATUS, reports
```

How we work on it (Luiz, Chat, Code, GitHub Issues): [`CLAUDE.md`](CLAUDE.md).

Licence: CC0 1.0, like both source repositories.
