"""The request handler: one scene, one RenderConfig, one look per request (report section 2).

Both entry points use it unchanged: the render server (fov_render.server) feeds it requests from
a queue folder, and Session.in_process calls it directly inside a Blender script.

    h = Handler(session_dict)        # opens the scene once; head from EYE; objects numbered
    h.ready                          # what the session runs with (settings, passes, head, device)
    response = h.handle(request)     # renders each eye into request["out_dir"]

A look's folder holds request.json, raw_<eye>.exr per eye, record_<eye>.npz when asked, and
meta.json (poses, the settings Cycles used, timings). Only the cameras move between looks; a
change of sensor re-applies the resolution.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .. import record
from ..profiles import DEFAULT_SEEDS
from ..request import (RESPONSE_SCHEMA, dumps, gaze_from_dict, sensor_from_dict, validate_request,
                       validate_session)
from ..rig import HeadPose, Rig


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class Handler:
    def __init__(self, session: dict, reuse_loaded: bool = False):
        """reuse_loaded: the session's .blend is already loaded, untouched, from the Blender command
        line (the server launched by the binary); otherwise it is opened here. blend None always
        uses the scene already loaded, as it is."""
        import bpy
        from . import config, scene as S
        validate_session(session)
        self.session = session
        self.cfg = config.RenderConfig.from_dict(session["config"])
        r = session["rig"]
        self.rig = Rig(ipd_m=r["ipd_m"], roll=r["roll"], baseline_h=tuple(r.get("baseline_h", (1.0, 0.0, 0.0))))
        self.access = session["access"]
        blend = session["blend"]
        loaded = bpy.data.filepath
        if reuse_loaded and blend is not None and not (loaded and os.path.exists(loaded) and os.path.samefile(loaded, blend)):
            raise RuntimeError(f"reuse_loaded: the loaded file {loaded!r} is not the session's {blend!r}")
        reuse = blend is None or reuse_loaded
        head = session["head"]
        explicit = None
        if head is not None:
            m = np.eye(4)
            m[:3, :3], m[:3, 3] = np.asarray(head["head_R_wh"], float), np.asarray(head["head_origin_w_m"], float)
            explicit = m
        self.scene_handle = S.open_scene(None if reuse else blend, eye=explicit,
                                         keep_passes=tuple(self.cfg.passes or ("combined", "position", "object_index")),
                                         instance_ids=session["instance_ids"])
        self.scene_handle.info["opened"] = "already loaded (command line)" if reuse and blend else \
            ("already loaded" if reuse else "open_mainfile")
        self.head: HeadPose = self.scene_handle.head
        from . import engine
        engine._LAST.pop(self.scene_handle.scene.name, None)     # a freshly opened scene: no cached sampling
        self.backend = None
        self._sensor_key = None
        self._models: dict = {}
        self._cams: dict = {}
        self.cam = None
        self.model = None
        self.n_handled = 0
        if session.get("prepare") is not None:
            self._use_sensor(session["prepare"])
        self.ready = self._ready()

    # ------------------------------------------------------------------ sensors
    def _use_sensor(self, sensor_dict: dict):
        """Make the sensor's camera active, applying the config for its raster when it changes."""
        from . import cameras, config
        key = json.dumps(sensor_dict, sort_keys=True)
        if key == self._sensor_key:
            return self.model, self.cam
        h = self.scene_handle
        spec = sensor_from_dict(sensor_dict)
        model = self._models.get(key) or spec.model(self.cfg.profile)
        self.backend = config.apply(h.scene, self.cfg, model.width, model.height, h.view_layer)
        if key not in self._cams:
            self._cams[key] = cameras.build(h.scene, model, self.cfg)
        cam = self._cams[key]
        h.scene.camera = cam
        h.view_layer.update()
        self._models[key], self._sensor_key, self.model, self.cam, self.spec = model, key, model, cam, spec
        return model, cam

    def _ready(self) -> dict:
        import bpy
        from . import config
        h = self.scene_handle
        out = {"schema": "fov-render-ready-v1", "pid": os.getpid(), "blender": bpy.app.version_string,
               "session": self.session, "scene": h.report(), "head": self.head.as_dict(),
               "device": self.backend, "prepared": self.session.get("prepare"),
               "settings": None if self.cam is None else config.readback(h.scene, self.cam, h.view_layer),
               "formats": {}}
        if "f3d_ab1a" in self.session["formats"]:
            from .. import compat
            if self.cam is None:
                raise ValueError("format f3d_ab1a needs a prepared sensor (session 'prepare')")
            out["formats"]["f3d_ab1a"] = {"settings": compat.ab1a_readback(h.scene, self.cam, h.view_layer),
                                          "passes": compat.gfe_passes(h.sanitised, h.view_layer)}
        return out

    # ------------------------------------------------------------------ one look
    def handle(self, req: dict) -> dict:
        from . import cameras, config, engine
        t0 = time.perf_counter()
        validate_request(req)
        h = self.scene_handle
        model, cam = self._use_sensor(req["sensor"])
        rig = self.rig
        if req.get("rig"):
            rig = Rig(ipd_m=req["rig"].get("ipd_m", rig.ipd_m), roll=req["rig"].get("roll", rig.roll),
                      baseline_h=rig.baseline_h)
        gaze = gaze_from_dict(req["gaze"])
        spp = int(req["spp"]) if req.get("spp") is not None else self.cfg.resolved_spp(self.cfg.extra.get("kind", "fixation"))
        seeds = req.get("seeds") or {e: DEFAULT_SEEDS[e] for e in req["eyes"]}
        out = Path(req["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        _write_atomic(out / "request.json", dumps(req))
        look_id = f"{int(req['n']):06d}"
        eyes = {}
        for eye in req["eyes"]:
            if gaze is None:
                pose = rig.fixed_pose(self.head, eye, float(req["gaze"].get("yaw_offset_deg", 0.0)))
            else:
                pose = rig.pose(self.head, gaze, eye)
            cameras.pose(cam, pose)
            exr = out / f"raw_{eye}.exr"
            secs = engine.render_once(h.scene, spp, int(seeds[eye]), str(exr))
            e = {"exr": str(exr), "seed": int(seeds[eye]), "render_seconds": float(secs), "pose": pose.as_dict(),
                 "matrix_world_blender": [[float(cam.matrix_world[i][j]) for j in range(4)] for i in range(4)]}
            if "records" in req["outputs"]:
                passes = engine.read_passes(str(exr))
                rec = record.build(model, passes, pose.centre_w, pose.R_wb, self.head.R_wh, eye=eye,
                                   look_id=look_id, access=self.access,
                                   meta={"spp": spp, "seed": int(seeds[eye]), "access": self.access})
                rec.save(str(out / f"record_{eye}.npz"))
                e["record"] = str(out / f"record_{eye}.npz")
            eyes[eye] = e
        settings = config.readback(h.scene, cam, h.view_layer)
        self.n_handled += 1
        resp = {"schema": RESPONSE_SCHEMA, "n": int(req["n"]), "ok": True, "out_dir": str(out), "eyes": eyes,
                "spp": spp, "device": self.backend, "settings": settings, "sensor_params": model.params(),
                "rig": rig.as_dict(), "access": self.access, "wall_seconds": time.perf_counter() - t0}
        meta = {"schema": "fov-render-look-meta-v1", "request": req, "response": resp, "head": self.head.as_dict(),
                "scene": {k: h.info[k] for k in ("path", "sha256", "blender", "opened") if k in h.info},
                "instance_ids": h.catalog["mode"]}
        _write_atomic(out / "meta.json", json.dumps(meta, indent=1, sort_keys=True, default=float) + "\n")
        return resp
