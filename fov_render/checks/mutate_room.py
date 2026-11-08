"""Broken copies of the calibration room for the CLI gates (issue #18). Blender side:

    python fov_render/checks/mutate_room.py -- --blend calib_room.blend --out OUT.blend --mutation broken|flipped

broken   the back wall deleted (the panorama sees out: holes) and a missing image on the floor's
         material (Cycles would render it magenta)
flipped  a closed 0.3 m cube 1 m in front of the head with its normals inward (back-faces on a
         closed mesh, S2's rule)
"""
import argparse
import os
import sys


def main(argv):
    import bpy
    from mathutils import Vector
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mutation", required=True, choices=["broken", "flipped"])
    a = ap.parse_args(argv)
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
    scene = bpy.context.scene
    if a.mutation == "broken":
        bpy.data.objects.remove(bpy.data.objects["wall_back"])
        mat = bpy.data.objects["floor"].active_material
        node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        img = bpy.data.images.new("fov_missing_texture.png", 4, 4)
        img.source = "FILE"
        img.filepath = "//fov_missing_texture.png"
        node.image = img
    else:
        eye = bpy.data.objects["EYE"]
        m = eye.matrix_world
        c = m.translation + (m.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized() * 1.0
        h = 0.15
        verts = [(c.x + sx * h, c.y + sy * h, c.z + sz * h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        me = bpy.data.meshes.new("fov_flipped_cube")
        me.from_pydata(verts, [], faces)
        me.update()
        # from_pydata winding above is outward for some faces: make all outward, then flip all inward
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(bpy.data.objects["floor"].active_material)
        ob = bpy.data.objects.new("fov_flipped_cube", me)
        scene.collection.objects.link(ob)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out), compress=False)
    print(f"[mutate-room] wrote {a.out} ({a.mutation})", flush=True)
    return 0


if __name__ == "__main__":
    try:
        rc = main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    except BaseException:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    os._exit(rc)
