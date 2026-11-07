"""RenderConfig: one frozen set of Cycles settings, presets that reproduce the legacy paths,
`apply` and `readback` (report sections 2 and 7).

A field set to None is left as the .blend file has it; every legacy path leaves some things
alone, and a preset that reproduces it must leave the same things alone.

Presets (name -> where its settings come from):
    fov_fixation  fov-3d-vision tools/render_foveated.py setup_foveated_camera
    preview360    fov-3d-vision tools/preview360.py (defaults: Blackman-Harris, no denoise)
    fsg           fov-3d-vision tools/fsg_render.py BlenderBackend.prepare
    oracle1       f3d-vision tools/classroom_oracle1_render.py _prepare_perspective_pair
    ab1a          f3d-vision tools/active_bootstrap/ab1a_render.py configure (oracle1, three passes)
    breadth1      f3d-vision tools/classroom_oracle/breadth1_render.py configure
    reference_view f3d-vision tools/visual_language/render_reference_view.py
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from ..profiles import profile as _profile

DEVICES = ("OPTIX", "CUDA", "CPU")
DEVICE_POLICIES = ("strict", "fallback")
COLORS = ("measurement", "scene")


@dataclass(frozen=True)
class RenderConfig:
    profile: str = "small"
    device: str = "OPTIX"
    device_policy: str = "strict"          # strict: no silent CPU fallback (decision 3)
    spp: int | None = None                 # None: the profile's fixation spp
    seed: int = 0
    pixel_filter: str = "BOX"
    filter_width: float | None = 1.0       # None: Cycles' own width for the filter
    denoise: bool | None = False
    persistent_data: bool | None = True
    color: str = "measurement"             # measurement: Standard, no look, exposure 0, gamma 1 (decision 6)
    multilayer: bool = True
    color_mode: str | None = None          # for single-layer EXR ("RGBA" in FSG)
    exr_codec: str = "NONE"                # NONE: readable by exr_lite inside Blender
    clip_m: tuple = (0.001, 1.0e5)
    passes: tuple = ("combined", "position", "object_index")
    passes_exact: bool = True              # False: switch the named passes on, leave others (legacy paths)
    single_layer: bool | None = True
    pixel_aspect: bool = True              # force square pixels
    use_border: bool | None = False
    preset: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        _profile(self.profile)
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}")
        if self.device_policy not in DEVICE_POLICIES:
            raise ValueError(f"device_policy must be one of {DEVICE_POLICIES}")
        if self.color not in COLORS:
            raise ValueError(f"color must be one of {COLORS}")

    def resolved_spp(self, kind: str = "fixation") -> int:
        if self.spp is not None:
            return int(self.spp)
        p = _profile(self.profile)
        return int(p["ref_spp"] if kind == "reference" else p["fix_spp"])

    def with_(self, **kw) -> "RenderConfig":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["clip_m"], d["passes"] = list(self.clip_m), list(self.passes)
        return d


_PRESETS = {
    "fov_fixation": dict(seed=0, pixel_filter="BOX", filter_width=1.0, denoise=False, persistent_data=True,
                         color="scene", multilayer=True, exr_codec="ZIP", clip_m=(0.001, 1.0e5),
                         passes=("combined", "z", "normal", "position"), single_layer=True,
                         pixel_aspect=False, use_border=None),
    "preview360": dict(seed=0, pixel_filter="BLACKMAN_HARRIS", filter_width=None, denoise=False,
                       persistent_data=None, color="scene", multilayer=True, exr_codec="ZIP",
                       clip_m=(0.001, 1.0e5), passes=("combined", "z", "normal", "position"),
                       single_layer=True, pixel_aspect=False, use_border=None, extra={"kind": "reference"}),
    "fsg": dict(seed=None, pixel_filter="BOX", filter_width=1.0, denoise=False, persistent_data=True,
                color="measurement", multilayer=False, color_mode="RGBA", exr_codec="ZIP", clip_m=(0.01, 100.0),
                passes=None, single_layer=None, pixel_aspect=True, use_border=False),
    "oracle1": dict(seed=0, pixel_filter="BOX", filter_width=1.0, denoise=False, persistent_data=True,
                    color="scene", multilayer=True, exr_codec="NONE", clip_m=(0.01, 1000.0),
                    passes=("combined", "z", "normal", "position", "object_index"), single_layer=True,
                    pixel_aspect=True, use_border=False),
    "ab1a": dict(seed=0, pixel_filter="BOX", filter_width=1.0, denoise=False, persistent_data=True,
                 color="scene", multilayer=True, exr_codec="NONE", clip_m=(0.01, 1000.0),
                 passes=("combined", "position", "object_index"), single_layer=True,
                 pixel_aspect=True, use_border=False, profile="full", spp=256),
    "breadth1": dict(seed=0, pixel_filter="BOX", filter_width=1.0, denoise=False, persistent_data=None,
                     color="scene", multilayer=True, exr_codec="NONE", clip_m=(0.01, 1000.0),
                     passes=("combined", "position", "object_index"), single_layer=True,
                     pixel_aspect=True, use_border=False, spp=512),
    "reference_view": dict(seed=0, pixel_filter="BOX", filter_width=1.0, denoise=False, persistent_data=None,
                           color="scene", multilayer=True, exr_codec="NONE", clip_m=(0.01, 1000.0),
                           passes=("combined", "position", "object_index"), single_layer=True,
                           pixel_aspect=True, use_border=False, spp=512),
}
PRESET_NAMES = tuple(_PRESETS)


def preset(name: str, **overrides) -> RenderConfig:
    """A RenderConfig reproducing one legacy path; keyword overrides win (profile, device, spp, ...)."""
    if name not in _PRESETS:
        raise ValueError(f"unknown preset {name!r}; expected one of {PRESET_NAMES}")
    kw = {k: v for k, v in _PRESETS[name].items()}
    kw.setdefault("passes_exact", False)       # every legacy path only switches passes on
    kw.update(overrides)
    if kw.get("seed") is None:
        kw["seed"] = 0
        kw.setdefault("extra", {})
        kw["extra"] = dict(kw["extra"], seed_untouched=True)
    return RenderConfig(preset=name, **kw)


# ----------------------------------------------------------------------------- Blender side

def setup_device(scene, cfg: RenderConfig) -> str:
    """Use the requested GPU back-end (OptiX also tries CUDA); strict policy raises instead of
    falling back to CPU. Returns the back-end used."""
    import bpy
    if "cycles" not in bpy.context.preferences.addons:
        import addon_utils
        addon_utils.enable("cycles", default_set=True)
    scene.render.engine = "CYCLES"
    cprefs = bpy.context.preferences.addons["cycles"].preferences
    candidates = [] if cfg.device == "CPU" else [cfg.device] + (["CUDA"] if cfg.device != "CUDA" else [])
    for backend in candidates:
        try:
            cprefs.compute_device_type = backend
        except TypeError:
            continue
        cprefs.refresh_devices()
        gpus = [d for d in cprefs.devices if d.type == backend]
        if gpus:
            for d in cprefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            return backend
    if cfg.device != "CPU" and cfg.device_policy == "strict":
        raise RuntimeError(f"requested {cfg.device} but no such GPU is available; strict device policy "
                           f"(use device='CPU' deliberately, or device_policy='fallback' for smoke tests)")
    scene.cycles.device = "CPU"
    return "CPU"


def apply(scene, cfg: RenderConfig, width: int, height: int, view_layer=None) -> str:
    """Apply cfg to scene and its view layer for a width x height raster. Returns the device used.
    Seed animation is the scene layer's job (fov_render.bl.scene.sanitise); render_once asserts it."""
    import bpy
    from . import passes as P
    vl = view_layer or bpy.context.view_layer
    backend = setup_device(scene, cfg)
    r, c = scene.render, scene.cycles
    r.resolution_x, r.resolution_y, r.resolution_percentage = int(width), int(height), 100
    if cfg.pixel_aspect:
        r.pixel_aspect_x = r.pixel_aspect_y = 1.0
    r.film_transparent = False
    r.use_motion_blur = False
    r.use_compositing = False
    r.use_sequencer = False
    if cfg.single_layer is not None:
        r.use_single_layer = cfg.single_layer
    if cfg.use_border is not None:
        r.use_border = cfg.use_border
    if cfg.persistent_data is not None:
        r.use_persistent_data = cfg.persistent_data
    c.samples = cfg.resolved_spp(cfg.extra.get("kind", "fixation"))
    if not cfg.extra.get("seed_untouched"):
        c.seed = cfg.seed
    c.use_adaptive_sampling = False
    c.time_limit = 0.0
    c.pixel_filter_type = cfg.pixel_filter
    if cfg.filter_width is not None:
        c.filter_width = cfg.filter_width
    if cfg.denoise is not None:
        try:
            c.use_denoising = cfg.denoise
        except (AttributeError, TypeError):
            pass
    if cfg.passes is not None:
        if cfg.passes_exact:
            P.set_passes(vl, cfg.passes)
        else:
            P.add_passes(vl, cfg.passes)
    im = r.image_settings
    if cfg.multilayer:
        im.media_type = "MULTI_LAYER_IMAGE"
        im.file_format = "OPEN_EXR_MULTILAYER"
        im.color_depth = "32"
        im.exr_codec = cfg.exr_codec
        im.use_exr_interleave = True     # one part; Blender 5.2 otherwise writes one part per pass
    else:
        if hasattr(im, "media_type"):
            im.media_type = "IMAGE"
        im.file_format = "OPEN_EXR"
        if cfg.color_mode:
            im.color_mode = cfg.color_mode
        im.color_depth = "32"
        im.exr_codec = cfg.exr_codec
    if cfg.color == "measurement":
        v = scene.view_settings
        v.view_transform, v.look, v.exposure, v.gamma = "Standard", "None", 0.0, 1.0
    vl.update()
    return backend


def apply_camera_data(cam_data, cfg: RenderConfig) -> None:
    cam_data.clip_start, cam_data.clip_end = cfg.clip_m
    cam_data.dof.use_dof = False


def readback(scene, cam=None, view_layer=None) -> dict:
    """What Cycles will actually use: the settings a run is checked against, stored with every look."""
    import bpy
    from . import passes as P
    vl = view_layer or bpy.context.view_layer
    r, c, im, v = scene.render, scene.cycles, scene.render.image_settings, scene.view_settings
    out = {
        "blender": bpy.app.version_string, "engine": r.engine, "cycles_device": c.device,
        "resolution_wh": [r.resolution_x, r.resolution_y], "resolution_percentage": r.resolution_percentage,
        "pixel_aspect": [r.pixel_aspect_x, r.pixel_aspect_y],
        "samples": c.samples, "seed": c.seed, "adaptive_sampling": bool(c.use_adaptive_sampling),
        "time_limit": c.time_limit, "denoising": bool(c.use_denoising),
        "pixel_filter": c.pixel_filter_type, "filter_width": round(float(c.filter_width), 6),
        "motion_blur": bool(r.use_motion_blur), "film_transparent": bool(r.film_transparent),
        "compositing": bool(r.use_compositing), "sequencer": bool(r.use_sequencer),
        "single_layer": bool(r.use_single_layer), "border": bool(r.use_border),
        "persistent_data": bool(r.use_persistent_data),
        "file_format": im.file_format, "exr_codec": im.exr_codec, "color_depth": im.color_depth,
        "color_mode": im.color_mode, "exr_interleave": bool(getattr(im, "use_exr_interleave", False)),
        "passes": P.enabled(vl),
        "view_transform": v.view_transform, "look": v.look, "exposure": round(float(v.exposure), 6),
        "gamma": round(float(v.gamma), 6), "frame_current": scene.frame_current,
    }
    if cam is not None:
        d = cam.data
        out["camera"] = {"type": d.type, "clip_m": [round(d.clip_start, 6), round(d.clip_end, 3)],
                         "dof": bool(d.dof.use_dof)}
        if d.type == "PERSP":
            out["camera"].update({"lens_mm": round(d.lens, 9), "sensor_width_mm": d.sensor_width,
                                  "sensor_fit": d.sensor_fit, "shift": [d.shift_x, d.shift_y]})
        elif d.type == "PANO":
            out["camera"].update({"panorama_type": d.panorama_type,
                                  "longitude": [round(d.longitude_min, 9), round(d.longitude_max, 9)],
                                  "latitude": [round(d.latitude_min, 9), round(d.latitude_max, 9)]})
        elif d.type == "CUSTOM":
            out["camera"].update({"custom_mode": d.custom_mode,
                                  "custom": {k: round(float(d.cycles_custom[k]), 9) for k in ("E2", "e_max")
                                             if k in d.cycles_custom}})
    return out
