"""Build the calibration room (tier 2) and save it as a .blend plus a targets sidecar JSON.

    blender -b -P tools/make_calib_room.py -- --out scenes/calib_room/calib_room.blend

Contents, all procedural (no external assets):
* closed 6 x 7.5 x 3.2 m room with noisy checker walls (texture everywhere, for later stereo);
* Siemens-star cards on rings of eccentricity around the primary gaze direction, at 2 m.
  Card size follows alpha(e) = alpha0 * (1 + e / E2t), so a foveation warp with E2 = E2t
  should render every ring about equally sharp; label under each card gives e in degrees;
* depth ladder on the right (constant 3 deg cards from 0.5 to 2.6 m), for Phase B disparities;
* wire harp on the left (vertical wires, 0.5-6 mm radius at 1.5 m), an aliasing stress test;
* three textured solids beyond the ring plane, for occlusion boundaries.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bl_common import ensure_cycles, make_eye, node_material, script_args  # noqa: E402

ROOM = {"x": (-3.0, 3.0), "y": (-2.5, 5.0), "z": (0.0, 3.2)}
# (eccentricity deg, number of meridians); sizes come from alpha0 * (1 + e / E2t)
RINGS = [(0.0, 1), (2.5, 4), (6.0, 8), (12.0, 8), (24.0, 8), (40.0, 8)]
LADDER_AZ = [55.0, 62.0, 69.0, 76.0, 83.0]
LADDER_DIST = [0.5, 0.8, 1.2, 1.8, 2.6]
WIRE_AZ = [-55.0 - 4.0 * i for i in range(8)]
WIRE_RADIUS_MM = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


# ---------------------------------------------------------------- materials
def link(nt, a, b):
    nt.links.new(a, b)


def math_node(nt, op, a=None, b=None, c=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    for i, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            link(nt, v, n.inputs[i])
    return n.outputs[0]


def principled(nt, base_color_socket=None, roughness=0.8):
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = roughness
    if base_color_socket is not None:
        link(nt, base_color_socket, bsdf.inputs["Base Color"])
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    link(nt, bsdf.outputs["BSDF"], out.inputs["Surface"])
    return bsdf


def mat_star(spokes: int = 18):
    mat = node_material("siemens_star")
    nt = mat.node_tree
    uv = nt.nodes.new("ShaderNodeTexCoord").outputs["UV"]
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    link(nt, uv, sep.inputs[0])
    cx = math_node(nt, "SUBTRACT", sep.outputs["X"], 0.5)
    cy = math_node(nt, "SUBTRACT", sep.outputs["Y"], 0.5)
    theta = math_node(nt, "ARCTAN2", cy, cx)
    star = math_node(nt, "GREATER_THAN", math_node(nt, "SINE", math_node(nt, "MULTIPLY", theta, float(spokes))), 0.0)
    r2 = math_node(nt, "ADD", math_node(nt, "MULTIPLY", cx, cx), math_node(nt, "MULTIPLY", cy, cy))
    inside = math_node(nt, "LESS_THAN", r2, 0.47 ** 2)
    star_albedo = math_node(nt, "MULTIPLY_ADD", star, 0.86, 0.04)
    albedo = math_node(nt, "ADD", math_node(nt, "MULTIPLY", star_albedo, inside),
                       math_node(nt, "MULTIPLY", math_node(nt, "SUBTRACT", 1.0, inside), 0.5))
    principled(nt, albedo)
    return mat


def mat_noisy_checker(name, checker_scale, base, k_checker, k_noise, noise_scale=3.0):
    mat = node_material(name)
    nt = mat.node_tree
    coord = nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]
    chk = nt.nodes.new("ShaderNodeTexChecker")
    chk.inputs["Scale"].default_value = checker_scale
    link(nt, coord, chk.inputs["Vector"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = noise_scale
    noise.inputs["Detail"].default_value = 8.0
    link(nt, coord, noise.inputs["Vector"])
    a = math_node(nt, "MULTIPLY_ADD", chk.outputs["Fac"], k_checker, base)
    albedo = math_node(nt, "ADD", a, math_node(nt, "MULTIPLY", noise.outputs["Fac"], k_noise))
    principled(nt, albedo)
    return mat


def mat_flat(name, albedo, roughness=0.8):
    mat = node_material(name)
    bsdf = principled(mat.node_tree, None, roughness)
    bsdf.inputs["Base Color"].default_value = (albedo, albedo, albedo, 1.0)
    return mat


def mat_noise_color(name, scale):
    mat = node_material(name)
    nt = mat.node_tree
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 6.0
    principled(nt, noise.outputs["Color"], 0.5)
    return mat


# ---------------------------------------------------------------- geometry
def add_mesh(coll, name, verts, faces, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def inward_quad(p0, p1, p2, p3, centre):
    """Order the quad so its normal points toward the room centre."""
    a, b, d = Vector(p0), Vector(p1), Vector(p3)
    n = (b - a).cross(d - a)
    face_c = (Vector(p0) + Vector(p1) + Vector(p2) + Vector(p3)) / 4
    return [p0, p1, p2, p3] if n.dot(centre - face_c) > 0 else [p3, p2, p1, p0]


def build_room(coll, m_wall, m_floor, m_ceiling):
    (x0, x1), (y0, y1), (z0, z1) = ROOM["x"], ROOM["y"], ROOM["z"]
    c = Vector(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    quads = {
        "floor": ([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], m_floor),
        "ceiling": ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], m_ceiling),
        "wall_front": ([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], m_wall),
        "wall_back": ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], m_wall),
        "wall_left": ([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], m_wall),
        "wall_right": ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], m_wall),
    }
    for name, (pts, mat) in quads.items():
        add_mesh(coll, name, inward_quad(*pts, c), [(0, 1, 2, 3)], mat)


def card_mesh():
    me = bpy.data.meshes.new("card_unit")
    me.from_pydata([(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name="UVMap")
    for loop in me.loops:
        x, y, _ = me.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (x + 0.5, y + 0.5)
    me.update()
    return me


def place_card(coll, me, mat_label, eye_pos, direction, dist, size_deg, name, label):
    loc = eye_pos + direction * dist
    size = 2.0 * dist * math.tan(math.radians(size_deg) / 2.0)
    q = (eye_pos - loc).to_track_quat("Z", "Y")  # card faces the eye, stays upright
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    ob.location, ob.rotation_mode, ob.rotation_quaternion, ob.scale = loc, "QUATERNION", q, (size, size, 1.0)

    cu = bpy.data.curves.new(name + "_label", "FONT")
    cu.body, cu.align_x, cu.align_y, cu.size = label, "CENTER", "CENTER", 0.22 * size
    cu.materials.append(mat_label)
    lab = bpy.data.objects.new(name + "_label", cu)
    coll.objects.link(lab)
    down = q @ Vector((0.0, -1.0, 0.0))
    lab.location, lab.rotation_mode, lab.rotation_quaternion = loc + down * 0.68 * size, "QUATERNION", q
    return loc


def wire(coll, name, x, y, radius, height, mat, segments=16):
    verts, faces = [], []
    for i in range(segments):
        t = 2 * math.pi * i / segments
        verts += [(x + radius * math.cos(t), y + radius * math.sin(t), 0.0),
                  (x + radius * math.cos(t), y + radius * math.sin(t), height)]
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((2 * i, 2 * j, 2 * j + 1, 2 * i + 1))
    add_mesh(coll, name, verts, faces, mat)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--eye-height", type=float, default=1.6)
    ap.add_argument("--ring-distance", type=float, default=2.0)
    ap.add_argument("--alpha0", type=float, default=1.0, help="card size at e=0, degrees")
    ap.add_argument("--e2t", type=float, default=4.0, help="card-size doubling eccentricity, degrees")
    args = ap.parse_args(script_args())
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    ensure_cycles(scene)
    scene.unit_settings.system, scene.unit_settings.scale_length = "METRIC", 1.0
    scene.cycles.samples, scene.cycles.max_bounces = 128, 6
    coll = scene.collection

    world = bpy.data.worlds.new("dim")
    scene.world = world
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)

    m_wall = mat_noisy_checker("wall", 5.0, 0.18, 0.25, 0.3)
    m_floor = mat_noisy_checker("floor", 2.0, 0.10, 0.20, 0.25, noise_scale=6.0)
    m_ceiling = mat_noisy_checker("ceiling", 1.0, 0.55, 0.0, 0.15)
    m_star, m_label = mat_star(), mat_flat("label", 0.03)
    m_wire, m_solid = mat_flat("wire", 0.06, 0.4), mat_noise_color("solid", 8.0)
    build_room(coll, m_wall, m_floor, m_ceiling)

    for name, loc, energy in (("key", (0.0, 1.5, 3.15), 150.0), ("fill", (0.0, -1.5, 3.15), 60.0)):
        light = bpy.data.lights.new(name, "AREA")
        light.shape, light.size, light.energy = "SQUARE", 2.5, energy
        ob = bpy.data.objects.new(name, light)
        coll.objects.link(ob)
        ob.location = loc  # area lights emit along local -Z, i.e. downward

    eye_pos = Vector((0.0, 0.0, args.eye_height))
    make_eye(scene, eye_pos, yaw_deg=0.0)
    f, r, u = Vector((0, 1, 0)), Vector((1, 0, 0)), Vector((0, 0, 1))
    me = card_mesh()
    me.materials.append(m_star)
    targets = []

    for e, n in RINGS:
        size = args.alpha0 * (1.0 + e / args.e2t)
        for k in range(n):
            m = 360.0 * k / n
            d = (f * math.cos(math.radians(e)) + (r * math.cos(math.radians(m)) + u * math.sin(math.radians(m)))
                 * math.sin(math.radians(e))).normalized()
            name = f"ring_e{e:g}_m{m:g}"
            place_card(coll, me, m_label, eye_pos, d, args.ring_distance, size, name, f"{e:g}°")
            targets.append({"name": name, "kind": "ring", "e_deg": e, "meridian_deg": m,
                            "distance_m": args.ring_distance, "size_deg": size, "dir_world": list(d)})

    for az, dist in zip(LADDER_AZ, LADDER_DIST):
        d = Vector((math.sin(math.radians(az)), math.cos(math.radians(az)), 0.0))
        name = f"ladder_{dist:g}m"
        place_card(coll, me, m_label, eye_pos, d, dist, 3.0, name, f"{dist:g} m")
        targets.append({"name": name, "kind": "ladder", "azimuth_deg": az, "distance_m": dist,
                        "size_deg": 3.0, "dir_world": list(d)})

    for i, (az, rad) in enumerate(zip(WIRE_AZ, WIRE_RADIUS_MM)):
        x, y = 1.5 * math.sin(math.radians(az)), 1.5 * math.cos(math.radians(az))
        wire(coll, f"wire_{rad:g}mm", x, y, rad / 1000.0, ROOM["z"][1], m_wire)
        targets.append({"name": f"wire_{rad:g}mm", "kind": "wire", "azimuth_deg": az,
                        "distance_m": 1.5, "radius_mm": rad})

    for op, kw in ((bpy.ops.mesh.primitive_uv_sphere_add, {"radius": 0.35, "location": (-0.9, 3.6, 0.35)}),
                   (bpy.ops.mesh.primitive_torus_add, {"major_radius": 0.3, "minor_radius": 0.1,
                                                       "location": (0.0, 3.9, 0.5), "rotation": (1.2, 0.0, 0.4)}),
                   (bpy.ops.mesh.primitive_cube_add, {"size": 0.6, "location": (0.9, 3.5, 0.3),
                                                      "rotation": (0.0, 0.0, 0.5)})):
        op(**kw)
        bpy.context.object.data.materials.append(m_solid)

    scene["fov_kind"] = "calib"
    bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups
    bpy.ops.wm.save_as_mainfile(filepath=out)
    sidecar = os.path.splitext(out)[0] + ".targets.json"
    with open(sidecar, "w") as fh:
        json.dump({"eye_position_m": list(eye_pos), "primary_forward": [0, 1, 0], "up": [0, 0, 1],
                   "alpha0_deg": args.alpha0, "e2t_deg": args.e2t, "targets": targets}, fh, indent=1)
    print(f"saved {out} and {sidecar} ({len(targets)} targets)")


main()
