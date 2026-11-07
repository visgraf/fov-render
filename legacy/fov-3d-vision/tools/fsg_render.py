"""FSG1: acquire three independent local perspective stereo calibration pairs.

    blender -b --python-exit-code 1 -P tools/fsg_render.py -- \
        --out previews/fsg1/small --profile small --device OPTIX

No peripheral preview, fusion, surface growth, or original OSL sensor in this step.
Requires only Blender, its NumPy, and repository rig/bl_common. Host OpenCV is NOT
imported. Synthetic tests inject a backend into acquire(); production has no fake flag.
"""
from __future__ import annotations
import argparse
import datetime as dt
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsg_geometry import (CV_TO_BLENDER, head_to_world, world_to_head, json_write,
                          make_calibration, pixels, project_h, rays_h)
from fsg_scene import CASES, CASE_GAZE, case_objects, ray_mesh, texture


class BlenderBackend:
    source = "blender_cycles"

    def __init__(self, device: str, save_blend: bool):
        import bpy
        self.bpy = bpy
        self.preferred_device = device
        self.save_blend = save_blend
        self.version = bpy.app.version_string
        self.device = "not configured"

    def prepare(self, case: str, calibration: dict, folder: Path, spp: int) -> dict:
        import bpy
        from mathutils import Matrix
        from bl_common import ensure_cycles, setup_device, make_eye, node_material
        import rig
        bpy.ops.wm.read_factory_settings(use_empty=True)
        s = bpy.context.scene
        ensure_cycles(s)
        self.device = setup_device(s, self.preferred_device)
        if self.device == "CPU" and self.preferred_device != "CPU":
            raise RuntimeError("No requested GPU backend available; use --device CPU explicitly only after reviewing runtime cost")
        s.unit_settings.system = "METRIC"
        s.unit_settings.scale_length = 1.0
        s.render.resolution_x, s.render.resolution_y = calibration["image_size_wh"]
        s.render.resolution_percentage = 100
        s.render.pixel_aspect_x = s.render.pixel_aspect_y = 1.0
        s.render.film_transparent = s.render.use_motion_blur = False
        s.render.use_compositing = s.render.use_sequencer = False
        s.render.use_border = False
        s.render.use_persistent_data = True
        s.cycles.samples = spp
        s.cycles.use_adaptive_sampling = False
        s.cycles.use_denoising = False
        s.cycles.time_limit = 0.0
        s.cycles.pixel_filter_type = "BOX"
        s.cycles.filter_width = 1.0
        im = s.render.image_settings
        if hasattr(im, "media_type"):
            im.media_type = "IMAGE"
        im.file_format = "OPEN_EXR"
        im.color_mode = "RGBA"
        im.color_depth = "32"
        im.exr_codec = "ZIP"
        s.view_settings.view_transform = "Standard"
        s.view_settings.look = "None"
        s.view_settings.exposure = 0.0
        s.view_settings.gamma = 1.0
        world = bpy.data.worlds.new("FSG_WORLD")
        s.world = world
        if world.node_tree is None:
            world.use_nodes = True
        world.node_tree.nodes.clear()
        bg = world.node_tree.nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = (.7,.7,.7,1)
        bg.inputs["Strength"].default_value = .6
        out = world.node_tree.nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(bg.outputs["Background"],out.inputs["Surface"])
        # Fixed EYE is the map frame, never the current gaze.
        e = make_eye(s, calibration["head_origin_w_m"])
        h = np.eye(4); h[:3,:3] = calibration["head_R_wh"]; h[:3,3] = calibration["head_origin_w_m"]
        e.matrix_world = Matrix(h.tolist())
        self.head = e
        for desc in case_objects(case):
            mesh = bpy.data.meshes.new(desc["name"])
            xyz_w = head_to_world(calibration, desc["vertices_h"])
            mesh.from_pydata(xyz_w.tolist(), [], [(0,1,2,3)])
            mesh.update()
            uv = mesh.uv_layers.new(name="UVMap")
            for poly in mesh.polygons:
                for li in poly.loop_indices:
                    uv.data[li].uv = desc["uv"][mesh.loops[li].vertex_index].tolist()
            obj = bpy.data.objects.new(desc["name"], mesh)
            s.collection.objects.link(obj)
            obj.pass_index = int(desc["instance_id"])
            obj["fsg_surface"] = True
            a = texture(obj.pass_index)
            image = bpy.data.images.new(desc["name"]+"_texture", width=len(a), height=len(a), alpha=True, float_buffer=True)
            image.colorspace_settings.name = "Non-Color"
            rgba = np.concatenate((a,np.ones((*a.shape[:2],1),np.float32)),axis=-1)
            image.pixels.foreach_set(rgba.ravel())
            image.update(); image.pack()
            mat = node_material(desc["name"]+"_mat")
            tex = mat.node_tree.nodes.new("ShaderNodeTexImage"); tex.image = image
            tex.interpolation = "Linear"; tex.extension = "EXTEND"
            diff = mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
            diff.inputs["Roughness"].default_value = 0.0
            output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
            mat.node_tree.links.new(tex.outputs["Color"], diff.inputs["Color"])
            mat.node_tree.links.new(diff.outputs[0], output.inputs["Surface"])
            obj.data.materials.append(mat)
        # Broad area light; all geometry remains opaque and diffuse.
        light_data = bpy.data.lights.new("FSG_KEY", "AREA")
        light_data.energy = 100.; light_data.shape = "DISK"; light_data.size = 3.
        light = bpy.data.objects.new("FSG_KEY",light_data); s.collection.objects.link(light)
        light.matrix_world = Matrix(h.tolist())
        light.location = head_to_world(calibration, np.array([0.,1.,.2])).tolist()
        # Build actual eye rotations through the repository rig, then assert our
        # calibration prescription agrees. No changes to the original rig code.
        from fsg_geometry import gaze_direction
        target_h = gaze_direction(*calibration["gaze_yaw_pitch_deg"])*calibration["prescribed_vergence_distance_m"]
        pair = rig.pair_for_point(head_to_world(calibration,target_h),
                                  np.asarray(calibration["head_origin_w_m"]),
                                  np.asarray(calibration["head_R_wh"]),calibration["ipd_m"])
        self.cameras = []
        for k, eye in enumerate(calibration["eyes"]):
            g = pair["eyes"][k]
            pos, rb = rig.camera_pose(np.asarray(calibration["head_origin_w_m"]),
                                     np.asarray(calibration["head_R_wh"]),
                                     rig.eye_offsets_local(calibration["ipd_m"])[k], g["yaw"], g["pitch"])
            actual_r = np.asarray(calibration["head_R_wh"]).T @ rb @ CV_TO_BLENDER
            if not np.allclose(actual_r, eye["R_hc"], atol=1e-9):
                raise RuntimeError("repository rig and FSG camera conventions disagree")
            if not np.allclose(world_to_head(calibration,pos),eye["centre_h_m"],atol=1e-9):
                raise RuntimeError("head-frame eye centres disagree")
            data = bpy.data.cameras.new("FSG_"+eye["name"])
            data.type = "PERSP"; data.sensor_fit = "HORIZONTAL"; data.sensor_width = 36.
            data.sensor_height = 36.; data.shift_x = data.shift_y = 0.
            data.lens = eye["K"][0][0]*data.sensor_width/calibration["image_size_wh"][0]
            data.clip_start = .01; data.clip_end = 100.
            data.dof.use_dof = False
            cam = bpy.data.objects.new("FSG_"+eye["name"], data); s.collection.objects.link(cam)
            m = np.eye(4); m[:3,:3] = rb; m[:3,3] = pos
            cam.matrix_world = Matrix(m.tolist()); self.cameras.append(cam)
        bpy.context.view_layer.update()
        self.scene = s
        self.mesh = self.export_mesh(calibration)
        self.checks = self.check_geometry(calibration)
        if self.save_blend:
            s.camera = self.cameras[0]
            bpy.ops.wm.save_as_mainfile(filepath=str(folder/"fixture.blend"))
        return self.mesh

    def export_mesh(self, c: dict) -> dict:
        """Read evaluated Blender mesh triangles, not the procedural specification."""
        bpy = self.bpy; dg = bpy.context.evaluated_depsgraph_get()
        triangles, ids = [], []
        for obj in self.scene.objects:
            if obj.type != "MESH" or not obj.get("fsg_surface",False): continue
            evaluated = obj.evaluated_get(dg); mesh = evaluated.to_mesh()
            try:
                mesh.calc_loop_triangles()
                for tri in mesh.loop_triangles:
                    pts = np.array([list(evaluated.matrix_world @ mesh.vertices[i].co) for i in tri.vertices])
                    triangles.append(world_to_head(c,pts)); ids.append(int(obj.pass_index))
            finally:
                evaluated.to_mesh_clear()
        if not triangles or any(i<=0 for i in ids):
            raise RuntimeError("empty or unlabelled Blender evaluation geometry")
        return {"triangles_h": np.array(triangles,float), "instance_ids": np.array(ids,np.int32)}

    def check_geometry(self, c: dict) -> dict:
        """Independent Blender projection and Scene.ray_cast cross-checks."""
        from mathutils import Vector
        from bpy_extras.object_utils import world_to_camera_view
        w,h = c["image_size_wh"]
        rng = np.random.default_rng(812)
        uv = rng.uniform([7,7],[w-8,h-8],(121,2))
        max_pixel, max_hit, count = 0.,0.,0
        dg = self.bpy.context.evaluated_depsgraph_get()
        for eye, cam in zip(c["eyes"], self.cameras):
            d = rays_h(eye,uv); origin = np.asarray(eye["centre_h_m"])
            xyz_h = origin + 2.2*d
            for p, expected in zip(head_to_world(c,xyz_h),uv):
                ndc = world_to_camera_view(self.scene,cam,Vector(p.tolist()))
                got = np.array([ndc.x*w-.5, (1-ndc.y)*h-.5])
                max_pixel = max(max_pixel,float(np.max(np.abs(got-expected))))
            reference = ray_mesh(origin,d,self.mesh)
            ow = head_to_world(c,origin)
            for i, dw in enumerate(d @ np.asarray(c["head_R_wh"]).T):
                hit, loc, normal, index, obj, matrix = self.scene.ray_cast(dg,Vector(ow.tolist()),Vector(dw.tolist()),distance=100.)
                want = int(reference["instance_id"][i])
                got_id = int(obj.pass_index) if hit else 0
                if got_id != want:
                    raise RuntimeError(f"Blender ray-cast object ID {got_id} != exported-mesh ID {want}")
                if hit:
                    err = np.linalg.norm(world_to_head(c,np.array(loc))-reference["position_h"][i])
                    max_hit = max(max_hit,float(err)); count += 1
        if max_pixel > .002:
            raise RuntimeError(f"Blender projection check FAIL: {max_pixel:.6g} pixels > .002")
        if max_hit > 2e-5 or count < 100:
            raise RuntimeError(f"Blender truth check FAIL: max {max_hit:.6g} m; hits {count}")
        return {"projection_max_error_px":max_pixel,"raycast_max_error_m":max_hit,
                "raycast_hits_checked":count,"independent_blender_checks":True}

    def render_eye(self, eye_id: int, c: dict, folder: Path, spp: int, seed: int) -> tuple[np.ndarray,float]:
        bpy = self.bpy; s = self.scene
        s.camera = self.cameras[eye_id]
        s.cycles.samples = spp; s.cycles.seed = seed; s.update_tag()
        bpy.context.view_layer.update()
        t0 = time.perf_counter()
        bpy.ops.render.render()
        elapsed = time.perf_counter()-t0
        if (s.cycles.samples,s.cycles.seed,s.cycles.use_adaptive_sampling,s.cycles.time_limit) != (spp,seed,False,0.):
            raise RuntimeError("Cycles sampling settings changed during rendering")
        p = folder/(c["eyes"][eye_id]["name"]+".exr")
        bpy.data.images["Render Result"].save_render(str(p),scene=s)
        loaded = bpy.data.images.load(str(p),check_existing=False)
        try:
            w,h = c["image_size_wh"]
            if tuple(loaded.size) != (w,h): raise RuntimeError("EXR dimensions disagree")
            n = len(loaded.pixels); channels = n//(w*h)
            if channels < 3 or n != w*h*channels: raise RuntimeError("EXR RGB buffer unavailable")
            a = np.empty(n,np.float32); loaded.pixels.foreach_get(a)
            rgb = a.reshape(h,w,channels)[::-1,:,:3].copy()
        finally:
            bpy.data.images.remove(loaded)
        if not np.isfinite(rgb).all() or rgb.std() < 1e-5:
            raise RuntimeError("rendered RGB is empty, non-finite, or constant")
        return rgb, elapsed


def acquire(args: argparse.Namespace, backend) -> dict:
    """Shared production/stub orchestration; never calls stereo or reads its result."""
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output must be new or empty: {out}")
    out.mkdir(parents=True,exist_ok=True)
    t_all = time.perf_counter()
    results = []
    names = CASES if args.case == "all" else (args.case,)
    for ci, name in enumerate(names):
        t_case = time.perf_counter(); folder = out/name; folder.mkdir()
        ev = folder/"evaluation_only"; ev.mkdir()
        c = make_calibration(args.profile,*CASE_GAZE[name])
        mesh = backend.prepare(name,c,folder,args.spp)
        np.savez_compressed(ev/"mesh.npz", **mesh)
        obs = {}; seconds=[]; seeds=[]
        w,h = c["image_size_wh"]; uv = pixels(w,h)
        t_oracle = 0.
        for eye_id, eye in enumerate(c["eyes"]):
            seed = 10000*args.seed + 100*CASES.index(name) + eye_id
            rgb, secs = backend.render_eye(eye_id,c,folder,args.spp,seed)
            if rgb.shape != (h,w,3) or not np.isfinite(rgb).all(): raise RuntimeError("bad RGB buffer")
            obs["rgb_"+eye["name"]] = rgb.astype(np.float32)
            t0 = time.perf_counter()
            oracle = ray_mesh(np.asarray(eye["centre_h_m"]),rays_h(eye,uv),mesh)
            obs["instance_"+eye["name"]] = oracle["instance_id"]
            t_oracle += time.perf_counter()-t0
            seconds.append(secs); seeds.append(seed)
        if seeds[0] == seeds[1]: raise RuntimeError("inter-eye render seeds must differ")
        np.savez_compressed(folder/"observation.npz",**obs)
        json_write(folder/"calibration.json",c)
        data = {"schema":"FSG1-acquisition-v1", "source":backend.source,
                "case":name, "profile":args.profile, "spp":args.spp, "seeds_lr":seeds,
                "profile_default_spp":getattr(args,"profile_default_spp",{"small":64,"full":256}[args.profile]),
                "blender_version":backend.version, "device":backend.device,
                "render_seconds_lr":seconds, "oracle_seconds":t_oracle,
                "case_seconds":time.perf_counter()-t_case,
                "primary_camera_samples":2*w*h*args.spp if backend.source=="blender_cycles" else 0,
                "nominal_camera_samples":2*w*h*args.spp,
                "adaptive_sampling":False, "optics":"pinhole; no DOF/motion blur",
                "segmentation":"oracle first-hit Blender pass_index; pixel centre; not antialiased",
                "checks":backend.checks}
        json_write(folder/"acquisition.json",data)
        results.append(data)
        print(f"[fsg-render] {name} source={backend.source} raster={w}x{h} spp={args.spp} "
              f"primary_samples={data['primary_camera_samples']} render_seconds={sum(seconds):.4f}",flush=True)
    run = {"schema":"FSG1-run-v1", "source":backend.source,"complete":True,
           "created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"cases":list(names),
           "profile":args.profile,"spp":args.spp,"seed":args.seed,
           "profile_default_spp":getattr(args,"profile_default_spp",{"small":64,"full":256}[args.profile]),
           "primary_camera_samples":sum(x["primary_camera_samples"] for x in results),
           "total_wall_seconds":time.perf_counter()-t_all}
    json_write(out/"run.json",run)  # completion marker written last, never on failure
    print(f"[fsg-render] COMPLETE {out}",flush=True)
    return run


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",required=True)
    ap.add_argument("--profile",choices=("small","full"),default="small")
    ap.add_argument("--case",choices=("all",)+CASES,default="all")
    ap.add_argument("--device",choices=("OPTIX","CUDA","CPU"),default="OPTIX")
    ap.add_argument("--spp",type=int,default=None)
    ap.add_argument("--seed",type=int,default=17)
    ap.add_argument("--save-blend",action="store_true")
    args = ap.parse_args(argv)
    if args.spp is not None and args.spp < 1: ap.error("--spp must be positive")
    if not 0 <= args.seed <= 100000: ap.error("--seed must be in [0,100000]")
    return args


def main() -> None:
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    args = parse_args(argv)
    from bl_common import PROFILES
    args.profile_default_spp = int(PROFILES[args.profile]["fix_spp"])
    if args.spp is None: args.spp = args.profile_default_spp
    acquire(args,BlenderBackend(args.device,args.save_blend))


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        if isinstance(e,SystemExit) and e.code in (0,None): raise
        traceback.print_exc()
        print("[fsg-render] FAILED",flush=True)
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
