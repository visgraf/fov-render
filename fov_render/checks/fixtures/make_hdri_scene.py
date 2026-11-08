"""Wrap an equirectangular HDRI as a scene (tier 1) and save it as a .blend.

    blender -b -P tools/make_hdri_scene.py -- --hdri scenes/hdri/<slug>/<file>.exr \
                                             --out scenes/hdri/<slug>/scene.blend

The environment is at infinity, so the scene is exactly a function on the sphere.
EYE sits at the origin looking along world +X (up +Z): with Blender's environment mapping
this makes a full equirect render from EYE reproduce the source image pixel for pixel
(verified in 5.2.1), so the HDRI file itself is ground truth for every eye rotation.
"""
from __future__ import annotations

import argparse
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bl_common import ensure_cycles, make_eye, script_args  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdri", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--interpolation", default="Linear", choices=["Closest", "Linear", "Cubic", "Smart"])
    args = ap.parse_args(script_args())
    hdri, out = os.path.abspath(args.hdri), os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    ensure_cycles(scene)
    scene.unit_settings.system, scene.unit_settings.scale_length = "METRIC", 1.0

    world = bpy.data.worlds.new("hdri")
    scene.world = world
    nt = world.node_tree
    nt.nodes.clear()
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri, check_existing=True)
    env.interpolation = args.interpolation
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = args.strength
    wout = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], wout.inputs["Surface"])

    make_eye(scene, (0.0, 0.0, 0.0), yaw_deg=-90.0)  # forward +X
    scene["fov_kind"] = "hdri"
    scene["fov_source"] = hdri
    bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print(f"saved {out}")


main()
