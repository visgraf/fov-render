"""Blender builders for the sensor models: create a camera once, pose it before each render.

    cam = build(scene, model, cfg)            # once per sensor and session
    pose(cam, eye_pose)                       # before each render (fov_render.rig.EyePose)
"""
from __future__ import annotations

import os

from ..sensors import OSL_PATH, FoveatedModel, PanoramaModel, PinholeModel
from .config import RenderConfig, apply_camera_data


def build(scene, model, cfg: RenderConfig, name: str | None = None, shader_path: str = OSL_PATH):
    import bpy
    from . import env
    name = name or f"FOVRENDER_{model.name.upper()}"
    data = bpy.data.cameras.new(name)
    if isinstance(model, FoveatedModel):
        env.prepare()                                   # oslquery for the bpy wheel
        text = bpy.data.texts.load(os.path.abspath(shader_path))
        data.type = "CUSTOM"
        data.custom_mode = "INTERNAL"
        data.custom_shader = text
        if not data.custom_bytecode:
            raise RuntimeError("OSL camera shader did not compile; see the console above")
        data.cycles_custom["E2"], data.cycles_custom["e_max"] = model.e2, model.emax
    elif isinstance(model, PanoramaModel):
        import math
        data.type, data.panorama_type = "PANO", "EQUIRECTANGULAR"
        data.longitude_min, data.longitude_max = -math.pi, math.pi
        data.latitude_min, data.latitude_max = -math.pi / 2, math.pi / 2
    elif isinstance(model, PinholeModel):
        data.type = "PERSP"
        data.sensor_fit = "HORIZONTAL"
        data.sensor_width = data.sensor_height = 36.0
        data.shift_x = data.shift_y = 0.0
        data.lens = model.lens_mm(36.0)
    else:
        raise TypeError(f"no builder for {type(model).__name__}")
    apply_camera_data(data, cfg)
    cam = bpy.data.objects.new(name, data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def pose(cam, eye_pose) -> None:
    import bpy
    from mathutils import Matrix
    cam.matrix_world = Matrix(eye_pose.matrix_world.tolist())
    bpy.context.view_layer.update()
