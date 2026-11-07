"""Synthetic scenes for the gates, generated in Blender (no downloads): a world-only HDRI scene with
a known source image, and an audit scene with every material case the scene layer must flag,
collection instances and animated render settings."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .common import LEGACY, run_blender_script

HDRI_W, HDRI_H = 256, 128


def hdri_pattern() -> np.ndarray:
    """(H, W, 3) top-down: a longitude sawtooth (unique circular alignment), a latitude ramp, a
    smooth longitude wave."""
    v, u = np.mgrid[:HDRI_H, :HDRI_W].astype(np.float64)
    lon_frac = (u + 0.5) / HDRI_W
    lat_frac = (v + 0.5) / HDRI_H
    r = 0.1 + 0.8 * lon_frac
    g = 0.15 + 0.7 * lat_frac
    b = 0.5 + 0.3 * np.sin(2 * math.pi * 2 * lon_frac)
    return np.stack([r, g, b], axis=-1)


def make_hdri_scene(work: Path) -> tuple[str, np.ndarray]:
    import bpy
    folder = Path(work) / "scenes"
    folder.mkdir(parents=True, exist_ok=True)
    exr, blend = folder / "synthetic_hdri.exr", folder / "hdri_synthetic.blend"
    pat = hdri_pattern()
    if not blend.is_file():
        img = bpy.data.images.new("synthetic_hdri", HDRI_W, HDRI_H, alpha=True, float_buffer=True)
        rgba = np.concatenate([pat[::-1], np.ones((HDRI_H, HDRI_W, 1))], axis=-1)   # Blender rows are bottom-up
        img.pixels.foreach_set(rgba.astype(np.float32).ravel())
        img.filepath_raw = str(exr)
        img.file_format = "OPEN_EXR"
        img.save()
        run_blender_script(str(LEGACY["fov"] / "make_hdri_scene.py"),
                           ["--hdri", str(exr), "--out", str(blend), "--interpolation", "Linear"])
    return str(blend), pat


def _mat(name, kind):
    import bpy
    m = bpy.data.materials.new(name)
    if m.node_tree is None and hasattr(m, "use_nodes"):
        m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes.get("Principled BSDF")
    if p is None:
        nt.nodes.clear()
        p = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(p.outputs[0], out.inputs["Surface"])
    out = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"][0]
    if kind == "glass":
        p.inputs["Transmission Weight"].default_value = 1.0
    elif kind == "cutout":
        p.inputs["Alpha"].default_value = 0.5
    elif kind == "emitter":
        p.inputs["Emission Color"].default_value = (1.0, 0.9, 0.8, 1.0)
        p.inputs["Emission Strength"].default_value = 5.0
    elif kind == "fog":
        vol = nt.nodes.new("ShaderNodeVolumeScatter")
        nt.links.new(vol.outputs[0], out.inputs["Volume"])
    return m


def _cube(name, loc, size, mat=None, coll=None):
    import bpy
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=size)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    if mat is not None:
        me.materials.append(mat)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


AUDIT_EXPECTED = {"Glass": ["transmission"], "Cutout": ["alpha"], "Lamp": ["emission"], "Fog": ["volume"]}


def make_audit_scene(work: Path) -> str:
    """Room-free scene, EYE at (0, 0, 1.6) looking +Y. Objects: a diffuse Floor, the four flagged
    materials, a collection 'Props' (not in the scene) instanced twice, animation on cycles.seed and
    view_settings.exposure."""
    import bpy
    from ..bl import config as C
    out = Path(work) / "scenes" / "audit.blend"
    if out.is_file():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    C.setup_device(s, C.RenderConfig(device="CPU"))
    s.unit_settings.system, s.unit_settings.scale_length = "METRIC", 1.0
    world = bpy.data.worlds.new("grey")
    s.world = world
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.3, 0.3, 0.3, 1.0)
    floor = _cube("Floor", (0, 0, -0.5), 1.0, _mat("diffuse", "diffuse"))
    floor.scale = (20, 20, 1)
    _cube("Glass", (0, 3, 1.6), 1.0, _mat("glass", "glass"))
    _cube("Cutout", (2.5, 3, 1.6), 1.0, _mat("cutout", "cutout"))
    _cube("Lamp", (-2.5, 3, 1.6), 1.0, _mat("emitter", "emitter"))
    _cube("Fog", (0, -3, 1.6), 1.0, _mat("fog", "fog"))
    props = bpy.data.collections.new("Props")
    _cube("Prop", (0, 0, 0), 0.6, _mat("prop", "diffuse"), coll=props)
    for name, loc in (("PropsA", (3, 0, 1.6)), ("PropsB", (-3, 0, 1.6))):
        e = bpy.data.objects.new(name, None)
        e.instance_type, e.instance_collection, e.location = "COLLECTION", props, loc
        s.collection.objects.link(e)
    cam = bpy.data.cameras.new("EYE")
    eye = bpy.data.objects.new("EYE", cam)
    s.collection.objects.link(eye)
    eye.location = (0, 0, 1.6)
    eye.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    s.cycles.seed = 7
    s.keyframe_insert(data_path="cycles.seed", frame=1)
    s.view_settings.exposure = 0.5
    s.keyframe_insert(data_path="view_settings.exposure", frame=1)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    return str(out)
