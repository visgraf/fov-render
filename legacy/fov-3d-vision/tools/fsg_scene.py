"""FSG1 procedural geometry/annotation utilities. Never imported by the stereo estimator.

All three fixtures are opaque, textured, planar polygons. ray_mesh is a small
vectorised triangle intersector for this calibration experiment, not a scalable
scene renderer. Exported evaluated Blender triangles are the evaluation asset.
"""
from __future__ import annotations
import argparse
import numpy as np
from fsg_geometry import camera_rotation_h, gaze_direction, unit

CASES = ("fronto", "tilted", "step")
CASE_GAZE = {"fronto": (0., 0.), "tilted": (14., -7.), "step": (0., 0.)}


def quad(centre: np.ndarray, right: np.ndarray, up: np.ndarray,
         width: float, height: float, instance: int, name: str) -> dict:
    c, r, u = np.asarray(centre), np.asarray(right)*width/2, np.asarray(up)*height/2
    return {"name": name, "vertices_h": np.array([c-r-u, c+r-u, c+r+u, c-r+u]),
            "instance_id": instance, "uv": np.array([[0.,0.], [1.,0.], [1.,1.], [0.,1.]])}


def case_objects(name: str) -> list[dict]:
    if name == "fronto":
        return [quad([0,0,-2.], [1,0,0], [0,1,0], 3., 3., 1, "fronto_surface")]
    if name == "tilted":
        g = gaze_direction(*CASE_GAZE[name]); r = camera_rotation_h(g)[:, 0]
        u = -camera_rotation_h(g)[:, 1]
        a = np.radians(25.)
        return [quad(g*2., r*np.cos(a)+g*np.sin(a), u, 3., 3., 1, "tilted_surface")]
    if name == "step":
        return [quad([0,0,-3.4], [1,0,0], [0,1,0], 4., 4., 2, "background"),
                quad([-.75,0,-1.6], [1,0,0], [0,1,0], 1.5, 3., 1, "foreground")]
    raise ValueError(f"unknown case {name}")


def triangulate_objects(objects: list[dict]) -> dict[str, np.ndarray]:
    ts, ids, uvs = [], [], []
    for obj in objects:
        for ix in ([0,1,2], [0,2,3]):
            ts.append(obj["vertices_h"][ix]); uvs.append(obj["uv"][ix]); ids.append(obj["instance_id"])
    return {"triangles_h": np.array(ts, float), "instance_ids": np.array(ids, np.int32),
            "triangle_uv": np.array(uvs, float)}


def ray_mesh(origins: np.ndarray, directions: np.ndarray, mesh: dict) -> dict[str, np.ndarray]:
    """Closest positive hit. Two-sided triangles; masks/background are not antialiased."""
    shape = directions.shape[:-1]
    d = np.asarray(directions, float).reshape(-1,3)
    o = np.broadcast_to(origins, (*shape,3)).reshape(-1,3)
    best = np.full(len(d), np.inf); tri = np.full(len(d), -1, np.int32)
    bary = np.zeros((len(d),3)); normals = np.zeros((len(d),3))
    for k, t in enumerate(mesh["triangles_h"]):
        e1, e2 = t[1]-t[0], t[2]-t[0]
        p = np.cross(d, e2); det = p @ e1
        inv = np.divide(1., det, out=np.zeros_like(det), where=np.abs(det)>1e-12)
        rel = o-t[0]; a = np.einsum("ij,ij->i", rel,p)*inv
        q = np.cross(rel,e1); b = np.einsum("ij,ij->i", d,q)*inv
        dist = (q @ e2)*inv
        hit = (np.abs(det)>1e-12) & (a>=-1e-9) & (b>=-1e-9) & (a+b<=1+1e-9) & (dist>1e-6) & (dist<best)
        best[hit] = dist[hit]; tri[hit] = k
        bary[hit] = np.column_stack((1-a[hit]-b[hit], a[hit], b[hit]))
        normals[hit] = unit(np.cross(e1,e2))
    hit = tri >= 0
    xyz = np.full_like(d, np.nan); xyz[hit] = o[hit]+best[hit,None]*d[hit]
    ids = np.zeros(len(d), np.int32); ids[hit] = mesh["instance_ids"][tri[hit]]
    return {"position_h": xyz.reshape(*shape,3), "range_m": best.reshape(shape),
            "instance_id": ids.reshape(shape), "triangle": tri.reshape(shape),
            "barycentric": bary.reshape(*shape,3), "normal_h": normals.reshape(*shape,3)}


def texture(instance: int, size: int = 512) -> np.ndarray:
    """Repeatable aperiodic linear-RGB texture. One texture per object, never per eye."""
    rng = np.random.default_rng(7100+int(instance))
    noise = rng.normal(size=(size,size))
    fy = np.fft.fftfreq(size)[:,None]; fx = np.fft.rfftfreq(size)[None,:]
    freq2 = fx*fx+fy*fy
    a = np.fft.irfft2(np.fft.rfft2(noise)*np.exp(-2*np.pi**2*1.3**2*freq2), s=noise.shape)
    b = np.fft.irfft2(np.fft.rfft2(noise)*np.exp(-2*np.pi**2*5.0**2*freq2), s=noise.shape)
    a = .7*a/max(a.std(),1e-12)+.3*b/max(b.std(),1e-12)
    a = .12 + .76/(1+np.exp(-a))
    return np.stack((a, a*.94, a*.86),axis=-1).astype(np.float32)


def synthetic_rgb(hit: dict, mesh: dict) -> np.ndarray:
    """Texture-only TEST image, NOT stochastic photorealistic rendering."""
    ids = hit["instance_id"]; tr = hit["triangle"]
    rgb = np.full((*ids.shape,3), .05, np.float32)
    for i in np.unique(ids):
        if i <= 0: continue
        m = ids == i; t = texture(int(i)); n = len(t)
        uv = np.einsum("ni,nij->nj", hit["barycentric"][m], mesh["triangle_uv"][tr[m]])
        # Generated Blender pixels are bottom-up: row zero has UV v=0.
        x = np.clip(uv[:,0]*n-.5, 0, n-1-1e-6); y = np.clip(uv[:,1]*n-.5, 0, n-1-1e-6)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        ax, ay = (x-x0)[:,None], (y-y0)[:,None]
        rgb[m] = ((1-ay)*((1-ax)*t[y0,x0]+ax*t[y0,x0+1])+
                  ay*((1-ax)*t[y0+1,x0]+ax*t[y0+1,x0+1]))
    return rgb


def self_test() -> list[str]:
    mesh = triangulate_objects(case_objects("step"))
    o = np.zeros(3)
    h = ray_mesh(o, unit(np.array([[-.1,0,-1.],[.1,0,-1.],[0,0,1.]])),mesh)
    f = []
    if not np.array_equal(h["instance_id"], [1,2,0]): f.append("step first-hit IDs")
    if not np.allclose(h["position_h"][:2,2], [-1.6,-3.4]): f.append("known step depths")
    if not np.isinf(h["range_m"][2]): f.append("background miss")
    return f


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--self-test",action="store_true",required=True); ap.parse_args()
    f=self_test()
    for a in f: print("[fsg-scene] FAIL",a)
    print(f"[fsg-scene] self-test {'FAILED' if f else 'PASS'}")
    raise SystemExit(bool(f))
