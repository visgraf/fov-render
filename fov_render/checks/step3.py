"""Step 3 gates: each RenderConfig preset sets what its legacy script sets (compared through one
readback in Blender), and the render call keeps its two guarantees. Needs bpy."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from ..rig import Gaze, HeadPose, Rig
from ..sensors import Foveated, PanoramaModel, PerspectiveModel, ReferenceView
from .common import Result, call_with_argv, legacy_path, load_legacy
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


def _capture_first_render(fn) -> dict:
    """Run a legacy main() and read back the settings at the moment of its first render."""
    import bpy
    from ..bl.config import readback
    got = {}

    @bpy.app.handlers.persistent               # survives the legacy script's own file load
    def handler(scene, *_):
        if "rb" not in got:
            got["rb"] = readback(scene, scene.camera, bpy.context.view_layer)
    bpy.app.handlers.render_pre.append(handler)
    try:
        fn()
    finally:
        if handler in bpy.app.handlers.render_pre:
            bpy.app.handlers.render_pre.remove(handler)
    if "rb" not in got:
        raise RuntimeError("the legacy script did not render")
    return got["rb"]


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

def legacy_fov_fixation(ctx, tmp):
    from ..bl import env
    env.prepare()
    rf = load_legacy("fov", "render_foveated.py")
    return _capture_first_render(lambda: call_with_argv(rf.main, "render_foveated.py", [
        "--blend", _room(ctx), "--out", tmp / "fix", "--profile", "small", "--device", "CPU", "--spp", 1]))


def legacy_preview360(ctx, tmp):
    p = load_legacy("fov", "preview360.py")
    return _capture_first_render(lambda: call_with_argv(p.main, "preview360.py", [
        "--blend", _room(ctx), "--out", tmp / "pano", "--width", 64, "--spp", 1, "--device", "CPU", "--no-backface"]))


def legacy_fsg(ctx, tmp):
    from ..bl.config import readback
    import bpy
    fr = load_legacy("fov", "fsg_render.py")
    c = fr.make_calibration("small", *fr.CASE_GAZE["fronto"])
    be = fr.BlenderBackend("CPU", False)
    be.prepare("fronto", c, tmp, 4)
    return readback(be.scene, be.cameras[0], bpy.context.view_layer)


def legacy_oracle1(ctx, tmp, profile="small", narrowed=False):
    from ..bl.config import readback
    import bpy
    scene, vl = _open(_room(ctx))
    co = load_legacy("f3d", "classroom_oracle1_render.py")
    lg = load_legacy("f3d", "fsg_geometry.py")
    _eye, _note, o, r = co._head_pose()
    if narrowed:
        ar = load_legacy("f3d", "active_bootstrap/ab1a_render.py", "active_bootstrap")
        ar.SP.DEVICE = "CPU"                         # the spec pins OPTIX; the Chat sandbox has no GPU
        c = lg.make_calibration("full", 0.0, 0.0, 2.10, head_r_wh=r, head_origin_w=o, tangent_frame="baseline_projected")
        scene, cam, _b, _m = ar.configure(c, 256)
    else:
        c = lg.make_calibration(profile, 0.0, 0.0, 2.10, head_r_wh=r, head_origin_w=o)
        scene, cam, _b, _m = co._prepare_perspective_pair(c, "CPU", 4)
    return readback(scene, cam, bpy.context.view_layer)


def legacy_breadth1(ctx, tmp):
    from ..bl.config import readback
    import bpy
    scene, vl = _open(_room(ctx))
    b = load_legacy("f3d", "classroom_oracle/breadth1_render.py", "classroom_oracle")
    from ..bl import scene as S
    h = S.resolve_head(scene)[0]
    cam, _rb = b.configure(scene, h.R_wh, h.origin_w, "CPU")
    return readback(scene, cam, bpy.context.view_layer)


def legacy_reference_view(ctx, tmp):
    from ..bl import scene as S
    scene, vl = _open(_room(ctx))
    h = S.resolve_head(scene)[0]
    seeds = tmp / "seeds.json"
    seeds.write_text(json.dumps(h.as_dict()))
    rv = load_legacy("f3d", "visual_language/render_reference_view.py", "visual_language")
    return _capture_first_render(lambda: call_with_argv(rv.main, "render_reference_view.py", [
        "--seeds", seeds, "--out", tmp / "ref", "--width", 64, "--height", 48, "--spp", 1, "--device", "CPU"]))


# ------------------------------------------------------------------- the comparisons

def _cases(ctx):
    from ..bl import config
    room = _room(ctx)
    fixed = lambda S: Rig().fixed_pose(_head(S), "C")   # noqa: E731
    return [
        ("fov_fixation", legacy_fov_fixation, room, config.preset("fov_fixation", device="CPU", spp=1),
         Foveated().model("small"), lambda S: Rig().pose(_head(S), Gaze.per_eye(C=(0.0, 0.0)), "C"),
         dict(pixel_filter="GAUSSIAN")),
        ("preview360", legacy_preview360, room, config.preset("preview360", device="CPU", spp=1),
         PanoramaModel(64), fixed, dict(exr_codec="NONE")),
        ("fsg", legacy_fsg, None, config.preset("fsg", device="CPU", spp=4), PerspectiveModel(128), None,
         dict(color="scene")),
        ("oracle1", lambda c, t: legacy_oracle1(c, t), room, config.preset("oracle1", device="CPU", spp=4),
         PerspectiveModel(128), None, dict(clip_m=(0.001, 1.0e5))),
        ("ab1a", lambda c, t: legacy_oracle1(c, t, narrowed=True), room, config.preset("ab1a", device="CPU"),
         PerspectiveModel(256), None, dict(passes=("combined", "z", "position", "object_index"), passes_exact=True)),
        ("breadth1", legacy_breadth1, room, config.preset("breadth1", device="CPU"), PanoramaModel(720), fixed,
         dict(spp=256)),
        ("reference_view", legacy_reference_view, room, config.preset("reference_view", device="CPU", spp=1),
         ReferenceView(64, 48).model(), fixed, dict(filter_width=1.5)),
    ]


LEGACY_REPO = {"fov_fixation": ("fov", ()), "preview360": ("fov", ()), "fsg": ("fov", ()),
               "oracle1": ("f3d", ()), "ab1a": ("f3d", ("active_bootstrap",)),
               "breadth1": ("f3d", ("classroom_oracle",)), "reference_view": ("f3d", ("visual_language",))}


def presets_match_legacy(ctx) -> Result:
    """Each preset against its legacy path, on the calibration room and on a copy with all five passes
    on (fsg builds its own scene, so it runs once). Control: one setting mutated per preset."""
    per, controls = {}, {}
    variants = (("", None), (" [all passes on]", calib_room_all_passes(ctx)))
    with tempfile.TemporaryDirectory() as d:
        for label, room in variants:
            vctx = dict(ctx, room=room) if room else ctx
            for name, legacy, scene_path, cfg, model, pose_fn, mutation in _cases(vctx):
                if room and scene_path is None:
                    continue
                key = name + label
                tmp = Path(d) / key.replace(" ", "_").replace("[", "").replace("]", "")
                tmp.mkdir()
                try:
                    repo, subs = LEGACY_REPO[name]
                    with legacy_path(repo, *subs):
                        a = legacy(vctx, tmp)
                    b = _new(scene_path, cfg, model, pose_fn)
                    per[key] = _diff(a, b)
                    controls[key] = bool(_diff(a, _new(scene_path, cfg.with_(**mutation), model, pose_fn)))
                except Exception as e:  # noqa: BLE001
                    per[key] = [f"error: {type(e).__name__}: {e}"]
                    controls[key] = None
    ok = all(v == [] for v in per.values())
    return Result(STEP, f"presets reproduce the legacy settings ({len(per)} comparisons, 7 paths)", ok,
                  all(controls.values()), {"differences": per, "control_detected": controls,
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
