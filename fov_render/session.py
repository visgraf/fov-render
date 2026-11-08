"""Session: open a scene once, ask for looks, get records back (report sections 2 and 8).

    from fov_render import Session, RenderConfig, Rig, Gaze, sensors

    with Session("scene.blend", RenderConfig(profile="small", device="OPTIX"), rig=Rig()) as s:
        look = s.look(sensors.Foveated(), eyes=("L", "R"), gaze=Gaze.yaw_pitch(10, -5, 2.1))
        L, R = look.records          # needs outputs=("exr", "records")

Session(...) runs Blender as a persistent server in the background (fov_render.server) and sends
one JSON request per look through a queue folder; Session.in_process(...) inside a Blender script
calls the same handler directly. Both return the same Look.

Which Blender: launcher="auto" uses $FOVRENDER_BLENDER, else `blender` on PATH, else this Python
if it has the bpy wheel; "blender", "python" or an executable path choose explicitly. With the
Blender binary the scene is loaded on its command line, as f3d-vision's server does.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from . import launch
from .bl.config import RenderConfig
from .record import SampleRecord
from .request import RequestError, dumps, make_request, make_session
from .rig import Rig

PACKAGE_ROOT = launch.PACKAGE_ROOT
SERVER_SCRIPT = Path(__file__).resolve().parent / "server.py"


class SessionError(RuntimeError):
    pass


class Look:
    """One look: the request, the handler's response, its folder, and the records."""

    def __init__(self, request: dict, response: dict, access: str):
        self.request, self.response, self.access = request, response, access
        self.out_dir = Path(response["out_dir"])
        self._records = None

    @property
    def meta(self) -> dict:
        return json.loads((self.out_dir / "meta.json").read_text())

    @property
    def exr(self) -> dict:
        return {e: Path(v["exr"]) for e, v in self.response["eyes"].items()}

    @property
    def records(self) -> list[SampleRecord]:
        if self._records is None:
            missing = [e for e, v in self.response["eyes"].items() if "record" not in v]
            if missing:
                raise SessionError(f"no record written for eye(s) {missing}: ask with outputs=('exr', 'records')")
            self._records = [SampleRecord.load(v["record"], access=self.access) for v in self.response["eyes"].values()]
        return self._records


class _Base:
    access = "observation"

    def _next_out(self, out_dir):
        if out_dir is not None:
            return Path(out_dir).resolve()
        return self.workdir / "looks" / f"look-{self.n + 1:04d}"

    def look(self, sensor, eyes=("L", "R"), gaze=None, *, spp=None, seeds=None, out_dir=None,
             outputs=("exr",), rig: Rig | None = None, yaw_offset_deg=None) -> Look:
        req = make_request(self.n + 1, sensor, eyes, gaze, self._next_out(out_dir), spp=spp, seeds=seeds,
                           rig=rig, outputs=outputs, yaw_offset_deg=yaw_offset_deg)
        return Look(req, self.request(req), self.access)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Session(_Base):
    def __init__(self, blend, cfg: RenderConfig, rig: Rig | None = None, *, access: str = "observation",
                 head=None, instance_ids: str = "complete", prepare=None, launcher: str = "auto",
                 workdir=None, log=None, ready_timeout_s: float = 600.0, poll_s: float = 0.003):
        self.cfg, self.rig, self.access = cfg, rig or Rig(), access
        self.blend = None if blend is None else Path(blend).resolve()
        if self.blend is None:
            raise SessionError("a background Session needs a .blend (Session.in_process uses the loaded scene)")
        self.workdir = Path(workdir).resolve() if workdir else Path(tempfile.mkdtemp(prefix="fov-render-session-"))
        self.q = self.workdir
        self.q.mkdir(parents=True, exist_ok=True)
        if any(self.q.glob("req-*.json")) or (self.q / "server-ready.json").exists():
            raise SessionError(f"{self.q} already holds a session's queue")
        self.session = make_session(self.blend, cfg, self.rig, access=access, head=head, instance_ids=instance_ids,
                                    prepare=prepare)
        (self.q / "session.json").write_text(dumps(self.session))
        self.poll_s, self.n = poll_s, 0
        cmd, env, self.launcher = self._command(launcher)
        self._own_log = log is None or isinstance(log, (str, Path))
        self.log = open(log or (self.q / "server.log"), "w") if self._own_log else log
        t0 = time.perf_counter()
        try:
            self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        except OSError as e:
            self.p = None
            if self._own_log:
                self.log.close()
            raise SessionError(f"cannot start {cmd[0]}: {e}") from None
        while not (self.q / "server-ready.json").exists():
            if (self.q / "fail-0000.json").exists() or self.p.poll() is not None:
                self._died("at start")
            if time.perf_counter() - t0 > ready_timeout_s:
                self.p.kill()
                raise SessionError(f"the render server did not become ready in {ready_timeout_s} s; see {self.log.name}")
            time.sleep(0.05)
        self.ready = json.loads((self.q / "server-ready.json").read_text())
        self.start_seconds = time.perf_counter() - t0

    @classmethod
    def in_process(cls, cfg: RenderConfig, rig: Rig | None = None, blend=None, **kw) -> "InProcessSession":
        return InProcessSession(blend, cfg, rig, **kw)

    def _command(self, launcher: str):
        try:
            return launch.command(launcher, SERVER_SCRIPT, "fov_render.server",
                                  ["--queue", self.q, "--session", self.q / "session.json"],
                                  blend=self.blend, blender_only_args=["--reuse-loaded"])
        except launch.LaunchError as e:
            raise SessionError(str(e)) from None

    def _died(self, when: str):
        f0 = self.q / "fail-0000.json"
        detail = json.loads(f0.read_text()).get("traceback", "") if f0.exists() else ""
        rc = self.p.poll()
        raise SessionError(f"the render server died {when} (exit {rc}); see {self.log.name}\n{detail}")

    def request(self, req: dict) -> dict:
        n = int(req["n"])
        if n != self.n + 1:
            raise SessionError(f"request n = {n}, expected {self.n + 1}")
        tmp = self.q / f"req-{n:04d}.tmp"
        tmp.write_text(dumps(req))
        os.replace(tmp, self.q / f"req-{n:04d}.json")
        self.n = n
        done, fail = self.q / f"done-{n:04d}.json", self.q / f"fail-{n:04d}.json"
        while not done.exists():
            if fail.exists():
                f = json.loads(fail.read_text())
                if f.get("reasons"):
                    raise RequestError(f["reasons"])
                raise SessionError("render failed:\n" + f.get("traceback", ""))
            if self.p.poll() is not None:
                self._died(f"during request {n}")
            time.sleep(self.poll_s)
        return json.loads(done.read_text())

    def request_raw(self, text: str) -> dict:
        """Send text as the next request file unchanged (for checks of the server's validation)."""
        n = self.n + 1
        tmp = self.q / f"req-{n:04d}.tmp"
        tmp.write_text(text)
        os.replace(tmp, self.q / f"req-{n:04d}.json")
        self.n = n
        done, fail = self.q / f"done-{n:04d}.json", self.q / f"fail-{n:04d}.json"
        while not (done.exists() or fail.exists()):
            if self.p.poll() is not None:
                self._died(f"during request {n}")
            time.sleep(self.poll_s)
        return json.loads((done if done.exists() else fail).read_text())

    def close(self, timeout_s: float = 60.0) -> int:
        if getattr(self, "p", None) is None:
            return 0
        (self.q / "stop").touch()
        try:
            rc = self.p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.p.kill()
            rc = self.p.wait()
        if self._own_log:
            self.log.close()
        self.p, self.returncode = None, rc
        return rc

    stop = close


class InProcessSession(_Base):
    """The same handler, called directly inside a Blender script (no second process)."""

    def __init__(self, blend, cfg: RenderConfig, rig: Rig | None = None, *, access: str = "observation", head=None,
                 instance_ids: str = "complete", prepare=None, workdir=None):
        from .bl.handler import Handler
        self.cfg, self.rig, self.access = cfg, rig or Rig(), access
        self.workdir = Path(workdir).resolve() if workdir else Path(tempfile.mkdtemp(prefix="fov-render-session-"))
        self.session = make_session(blend, cfg, self.rig, access=access, head=head, instance_ids=instance_ids,
                                    prepare=prepare)
        t0 = time.perf_counter()
        self.handler = Handler(self.session)
        self.ready = dict(self.handler.ready, start_seconds=time.perf_counter() - t0)
        self.n = 0
        self.launcher = {"kind": "in_process"}

    def request(self, req: dict) -> dict:
        n = int(req["n"])
        if n != self.n + 1:
            raise SessionError(f"request n = {n}, expected {self.n + 1}")
        self.n = n
        return self.handler.handle(json.loads(dumps(req)))    # the same JSON round trip as the server

    def close(self) -> int:
        return 0

    stop = close
