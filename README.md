# fov-render

Blender Cycles rendering for gaze-directed binocular observers. Two eyes on a head that may move
between looks, four sensors behind one rig, and one sample record whatever the sensor:

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

**Status:** the migration from fov-3d-vision and f3d-vision is complete (tag `v0.1.0`), and the
migration scaffolding is gone from `main` (#16). See [`docs/STATUS.md`](docs/STATUS.md),
[`docs/PLAN.md`](docs/PLAN.md) and [`docs/HISTORY.md`](docs/HISTORY.md).

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

The Blender side (`fov_render.bl`) needs `bpy`: Blender 5.2.1, or the `bpy` 5.2.1 wheel
(`pip install -e ".[bpy]"`, which needs **Python 3.13**; it renders on CPU anywhere and on OptiX
where there is an NVIDIA GPU). The numpy side needs Python 3.11 or later.

```python
from fov_render.bl import cameras, config, engine, scene

h = scene.open_scene("room.blend")                      # head from EYE, sanitised, objects numbered
# instance_ids="complete" (default) or "legacy": objects reached only through collection instances
# are numbered after the scene's own, or keep their file's pass index as f3d-vision (PLAN D11, D13)
cfg = config.RenderConfig(profile="small", device="OPTIX")
config.apply(h.scene, cfg, m.width, m.height)
cam = cameras.build(h.scene, m, cfg)
cameras.pose(cam, Rig().pose(h.head, Gaze.yaw_pitch(0, 0), "C"))
engine.render_once(h.scene, cfg.resolved_spp(), 0, "look/C.exr")
```

A `Session` opens the scene once and answers one look per request: in a background Blender (the
persistent render server, `fov_render/server.py`) or, inside a Blender script, in the same process:

```python
from fov_render import Session, RenderConfig, Rig, Gaze, sensors

with Session("room.blend", RenderConfig(profile="small", device="OPTIX"), rig=Rig(roll="baseline_projected"),
             access="oracle") as s:                       # launcher="auto": $FOVRENDER_BLENDER, blender, or the wheel
    look = s.look(sensors.Perspective(), eyes=("L", "R"), gaze=Gaze.yaw_pitch(10, -5, vergence_m=2.1),
                  outputs=("exr", "records"))
    L, R = look.records                                   # SampleRecord per eye; look.meta, look.exr

s = Session.in_process(RenderConfig(...), rig=Rig(), blend=None)   # inside Blender: the loaded scene
```

**The head moves between looks** (issue #20). A look may carry its own head pose; without one it
uses the session's (the scene's `EYE`). The server keeps no head state: the head is a field of each
request. `move_head` is a client-side convenience that fills it in:

```python
s.move_head(translate_h=(0, 0, -0.10), yaw_deg=15)      # 10 cm forward in the head frame, then turn 15 deg right
s.look(sensors.Foveated(), gaze=Gaze.yaw_pitch(0, 0, vergence_m=2.1))   # the request carries the moved head
s.look(sensors.Foveated(), gaze=Gaze.verged_point(P), head=s.session_head.translated([0.2, 0, 0]))  # this look only
s.reset_head()                                          # back to the session's head
```

`HeadPose.moved(translate_h, yaw_deg, pitch_deg, roll_deg)` moves in the head's own frame (+X right,
+Y up, −Z forward; +yaw right, +pitch up, +roll lowers the right ear); `HeadPose.translated(dt_w)`
moves in world metres. A `yaw_pitch` or `per_eye` gaze is head-relative and turns with the head; a
`verged_point` gaze is a world point and the eyes stay on it. Both eyes of a look share its head
(nothing moves within a look), and each record stores it (`head_R_wh`, `head_origin_w`). There is
no body frame: a caller with a body or neck model composes it into the head pose (PLAN D20).

Each look is a `fov-render-request-v1` JSON request (`fov_render/request.py`); its folder holds
`request.json`, `raw_<eye>.exr`, `record_<eye>.npz` when asked, and `meta.json` (poses, the
settings Cycles used, timings). A malformed request is refused with every reason, before anything
renders.

## Command line

`pip install -e .` puts `fov-render` on the PATH (also `python -m fov_render.cli`). It finds Blender
as `Session` does (`--launcher auto`: `$FOVRENDER_BLENDER`, `blender` on PATH, or the `bpy` wheel).

```
fov-render eye      --blend scene.blend --out scene_eye.blend --from-camera --forward-m 1.5   # add the head
fov-render info     --blend scene_eye.blend                       # head, view layer, catalogue, audits (JSON)
fov-render check    --blend scene_eye.blend --only S1 S2 --write-spec draft.json             # a first spec
fov-render check    --blend scene_eye.blend --spec docs/scenes/NAME.json --device OPTIX      # the scene gate
fov-render look     --blend scene_eye.blend --sensor foveated --eyes L R --point 0.4 2.1 1.1 \
                    --profile full --records --out runs/demo/f0000
fov-render look     --blend scene_eye.blend --sensor perspective --gaze 10 -5 --vergence 2.1 \
                    --seeds L=2111 R=2112 --out runs/demo/p0000
fov-render look     --blend scene_eye.blend --sensor foveated --gaze 0 0 --vergence 2.1 \
                    --head-move 0 0 -0.5 --head-turn 20 0 0 --out runs/demo/moved    # the head moved for this look
fov-render panorama --blend scene_eye.blend --preset preview360 --eye C --out previews/ref_C
fov-render requests --blend scene_eye.blend looks.json --out runs/demo     # many looks, one session
fov-render selftest --device OPTIX                                         # the gates below
```

`look`, `panorama` and `requests` send exactly the requests `Session.look` sends (gate cli.2 checks it
bit for bit); every folder also gets `fov-render-run.json` (the command, the session, the answers).
Sensor fields go in `--param` (`E2_deg=2.0 e_max_deg=45`), the gaze is one of `--point`,
`--gaze YAW PITCH [--vergence M]`, `--per-eye L=YAW,PITCH R=…`, or none (a fixed look). A
`requests` file is a JSON list of looks, `{sensor, eyes, gaze, spp, seeds, rig, head, out_dir, outputs}`,
with `out_dir` relative to `--out` and `head` an absolute pose `{head_R_wh, head_origin_w_m}` (omitted:
the session's). `--head-move DX DY DZ` and `--head-turn YAW PITCH ROLL` move the session's head for
one look, in its own frame (`HeadPose.moved`).

Exit status: **0** done, **1** a render, a gate or Blender failed, **2** the arguments, a request or a
spec were refused: every reason is printed, and Blender is not started.

A scene's spec (`docs/scenes/NAME.json`) says what the scene gate expects of it. `--write-spec`
drafts one from what S1 and S2 measure: the missing images as found, the closed meshes with
back-faces as allowed, a hole limit of twice the measured fraction (at least 0.001, at most the
default 0.02). A draft accepts the scene as it is, so it is reviewed before it is committed; a spec
that still lists `to_measure` is refused. Specs so far: `calib_room.json` (the fixture, everything
known), `barbershop.json` (#4), `classroom.json` (to be measured on the workstation, #18).

## Checks

Every module ships a check with a control that must fail. Run them all:

```
pip install -e ".[bpy,checks]"                                  # Python 3.13: bpy wheel + OpenEXR
python -m fov_render.checks --device OPTIX                      # or CPU; numpy only without bpy (= fov-render selftest)
python -m fov_render.checks --steps classroom --device OPTIX    # the workstation: Classroom files + f3d-vision
```

A new scene gets its own gate, every check that an arbitrary file can run (S1 opens, S2 sanity,
S3 `check_foveated`, S4 every sensor's record against Position, S5 Object Index against a ray cast,
S6 the audits by solid angle):

```
fov-render check --blend scene_eye.blend [--spec docs/scenes/NAME.json] --device OPTIX   # python -m fov_render.checks.scene
```

The checks run with the `bpy` wheel (PLAN D12): under the Blender binary
(`blender -b --python-exit-code 1 -P tools/run_checks.py -- …`) gate 1.2 SKIPs, because Blender's
own Python has no OpenEXR. Gate 1.2 renders on CPU whatever `--device` says (D10). Without `bpy`
the Blender gates are reported SKIP, never PASS. Results go to
`previews/checks/results.json`.

The comparisons with the code fov_render replaced (fov-3d-vision, f3d-vision) use **known answers**:
what the legacy code gave, frozen in `fov_render/checks/known/` by `tools/freeze_known_answers.py`.
The legacy files themselves are pinned in `known/pins.json` (repository, commit, path, sha256) and
rebuilt on demand from a checkout's git history (`fov_render/checks/pinned.py`, with
`$FOVRENDER_F3D_ROOT` and `$FOVRENDER_FOV3D_ROOT`): the Classroom gates use them, and so does the
freeze if it ever has to be re-run. Until `v0.1.0` they were copies in `legacy/`; the migration's
history is in [`docs/HISTORY.md`](docs/HISTORY.md).

## Layout

```
fov_render/            numpy side: profiles, warp, rig, geometry, sensors/, record, exr_lite,
                       request (JSON schemas), session (Session, Look)
fov_render/cli.py      the `fov-render` command line; launch.py finds Blender for it and for Session
fov_render/server.py   the persistent render server (Blender side)
fov_render/blcli.py    the Blender side of `fov-render info | eye | check | selftest`
fov_render/bl/         Blender side: env, config (RenderConfig, presets), engine, passes, cameras, scene, handler,
                       eye_scene (`fov-render eye`)
fov_render/checks/     the gates; known/ (frozen answers, pins), fixtures/ (scene builders, check_foveated)
tools/run_checks.py    gates under the Blender binary
tools/make_eye_scene.py  `fov-render eye` by hand: a derived copy of a .blend with an EYE, the original untouched
tools/freeze_known_answers.py  re-freezes known/ from the pinned legacy code (needs both checkouts)
docs/scenes/           what each scene used in the project is expected to give (fov_render.checks.scene)
docs/                  PLAN, STATUS, HISTORY
```

How we work on it (Luiz, Chat, Code, GitHub Issues): [`CLAUDE.md`](CLAUDE.md).

Licence: CC0 1.0, like both source repositories.
