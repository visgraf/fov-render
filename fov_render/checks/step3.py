"""Step 3 gates: each RenderConfig preset sets what its legacy script set (compared through one
readback in Blender, against the legacy readbacks frozen in known/, issue #16), and the render call
keeps its two guarantees. Needs bpy."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from ..rig import Gaze, HeadPose, Rig
from ..sensors import Foveated, PanoramaModel, PerspectiveModel, ReferenceView
from .common import KNOWN, Result
from .step1 import calib_room

STEP = "3"
IGNORE = {"blender", "frame_current"}


def _diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        if k in IGNORE:
            continue
        if k not in a or k not in b:
            out.append(f"{prefix}{k}: only in {'legacy' if k in a else 'fov_render'}")
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            out += _diff(a[k], b[k], prefix + k + ".")
        elif a[k] != b[k]:
            out.append(f"{prefix}{k}: legacy {a[k]!r} vs fov_render {b[k]!r}")
    return out


def _open(path):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(path))
    bpy.context.view_layer.update()
    return bpy.context.scene, bpy.context.view_layer


def _new(scene_path, cfg, model, pose_fn=None):
    """fov_render side: apply the preset and build the camera; returns readback."""
    import bpy
    from ..bl import cameras, config, scene as S
    if scene_path is None:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene, vl = bpy.context.scene, bpy.context.view_layer
    else:
        scene, vl = _open(scene_path)
    config.apply(scene, cfg, model.width, model.height, vl)
    cam = cameras.build(scene, model, cfg)
    if pose_fn is not None:
        cameras.pose(cam, pose_fn(S))
    return config.readback(scene, cam, vl)


def _head(S):
    import bpy
    return S.resolve_head(bpy.context.scene)[0]


# ------------------------------------------------------------------- legacy sides

def _room(ctx) -> str:
    """The scene the comparison runs on: the calibration room, or ctx["room"] (the all-passes copy)."""
    return ctx.get("room") or calib_room(ctx)


def calib_room_all_passes(ctx) -> str:
    """The calibration room with all five passes switched on. A preset that leaves on a pass its
    legacy path switches off (issue #1: ab1a's Depth and Normal) agrees on the plain room, whose
    file has them off, and disagrees here."""
    import bpy
    from ..bl import passes as P
    out = Path(ctx["work"]) / "scenes" / "calib_room_all_passes.blend"
    if out.is_file():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=calib_room(ctx))
    P.set_passes(bpy.context.view_layer, tuple(P.PASS_ATTR))
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    return str(out)

# ------------------------------------------------------------------- the comparisons

def _cases(ctx):
    """(name, scene, cfg, model, pose_fn, mutation) per preset; the legacy side of each is frozen in
    known/step3_readbacks.json (tools/freeze_known_answers.py, issue #16)."""
    from ..bl import config
    room = _room(ctx)
    fixed = lambda S: Rig().fixed_pose(_head(S), "C")   # noqa: E731
    return [
        ("fov_fixation", room, config.preset("fov_fixation", device="CPU", spp=1),
         Foveated().model("small"), lambda S: Rig().pose(_head(S), Gaze.per_eye(C=(0.0, 0.0)), "C"),
         dict(pixel_filter="GAUSSIAN")),
        ("preview360", room, config.preset("preview360", device="CPU", spp=1), PanoramaModel(64), fixed,
         dict(exr_codec="NONE")),
        ("fsg", None, config.preset("fsg", device="CPU", spp=4), PerspectiveModel(128), None, dict(color="scene")),
        ("oracle1", room, config.preset("oracle1", device="CPU", spp=4), PerspectiveModel(128), None,
         dict(clip_m=(0.001, 1.0e5))),
        ("ab1a", room, config.preset("ab1a", device="CPU"), PerspectiveModel(256), None,
         dict(passes=("combined", "z", "position", "object_index"), passes_exact=True)),
        ("breadth1", room, config.preset("breadth1", device="CPU"), PanoramaModel(720), fixed, dict(spp=256)),
        ("reference_view", room, config.preset("reference_view", device="CPU", spp=1), ReferenceView(64, 48).model(),
         fixed, dict(filter_width=1.5)),
    ]


def presets_match_legacy(ctx) -> Result:
    """Each preset against the settings its legacy path rendered with (frozen, known/step3_readbacks.json),
    on the calibration room and on a copy with all five passes on (fsg builds its own scene, so it runs
    once). Control: one setting mutated per preset."""
    known = json.loads((KNOWN / "step3_readbacks.json").read_text())
    per, controls = {}, {}
    variants = (("", None), (" [all passes on]", calib_room_all_passes(ctx)))
    for label, room in variants:
        vctx = dict(ctx, room=room) if room else ctx
        for name, scene_path, cfg, model, pose_fn, mutation in _cases(vctx):
            if room and scene_path is None:
                continue
            key = name + label
            try:
                a = known[key]
                per[key] = _diff(a, _new(scene_path, cfg, model, pose_fn))
                controls[key] = bool(_diff(a, _new(scene_path, cfg.with_(**mutation), model, pose_fn)))
            except Exception as e:  # noqa: BLE001
                per[key] = [f"error: {type(e).__name__}: {e}"]
                controls[key] = None
    ok = all(v == [] for v in per.values()) and len(per) == len(known)
    return Result(STEP, f"presets reproduce the legacy settings ({len(per)} comparisons, 7 paths, frozen)", ok,
                  all(controls.values()), {"differences": per, "control_detected": controls, "known": len(known),
                                           "note": "ab1a compared with its spec's OPTIX replaced by CPU here; "
                                                   "its OPTIX spec is gate 3.C (classroom)"})


def render_call_guarantees(ctx) -> Result:
    """render_once: a changed spp takes effect under persistent data (noise falls ~ 1/sqrt(n)), and an
    animated setting that overwrites the request is caught."""
    import bpy
    from ..bl import cameras, config, engine, scene as S
    h = S.open_scene(_room(ctx))
    model = PanoramaModel(96)
    cfg = config.RenderConfig(device=ctx["device"], spp=2, persistent_data=True)
    config.apply(h.scene, cfg, model.width, model.height, h.view_layer)
    cam = cameras.build(h.scene, model, cfg)
    cameras.pose(cam, Rig().fixed_pose(h.head, "C"))
    tmp = Path(tempfile.mkdtemp(prefix="fovrender-step3-"))
    rgb = {}
    for spp in (2, 32):
        for seed in (1, 2):
            engine.render_once(h.scene, spp, seed, str(tmp / f"r{spp}_{seed}.exr"))
            rgb[(spp, seed)] = engine.read_passes(str(tmp / f"r{spp}_{seed}.exr"))["combined"][..., :3]
    noise = {spp: float(np.std(rgb[(spp, 1)] - rgb[(spp, 2)])) for spp in (2, 32)}
    ratio = noise[2] / noise[32]
    # control: an animated cycles.samples overwrites the request; render_once must raise
    sc = h.scene
    sc.cycles.samples = 3
    sc.keyframe_insert(data_path="cycles.samples", frame=1)
    caught = False
    try:
        engine.render_once(sc, 5, 1, str(tmp / "anim.exr"))
    except RuntimeError:
        caught = True
    removed = S.remove_setting_animation(sc)
    engine.render_once(sc, 5, 1, str(tmp / "after.exr"))       # must hold once the animation is removed
    return Result(STEP, "render_once: spp change takes effect; animated settings are caught", 2.5 <= ratio <= 6.5,
                  caught and bool(removed), {"noise_std": noise, "ratio_2_vs_32_spp": ratio,
                                             "expected_ratio": 4.0, "animation_removed": removed})


CHECKS = [presets_match_legacy, render_call_guarantees]
