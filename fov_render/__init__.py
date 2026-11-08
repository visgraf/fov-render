"""fov-render: Blender Cycles rendering for gaze-directed binocular observers.

numpy side (both interpreters):  profiles, warp, rig, geometry, sensors, record, exr_lite,
                                 request (JSON schemas), session (Session, Look)
Blender side (needs bpy):        fov_render.bl  (config, engine, passes, cameras, scene, handler),
                                 fov_render.server (the persistent render server)
"""
from __future__ import annotations

__version__ = "0.1.0.dev0"

from . import geometry, profiles, record, rig, sensors, warp  # noqa: F401
from .profiles import PROFILES  # noqa: F401
from .record import AccessError, SampleRecord  # noqa: F401
from .rig import EyePose, Gaze, HeadPose, Rig  # noqa: F401
from .bl.config import RenderConfig, preset  # noqa: F401,E402  (plain dataclass; no bpy at import)
from .request import RequestError  # noqa: F401,E402
from .session import Look, Session, SessionError  # noqa: F401,E402
