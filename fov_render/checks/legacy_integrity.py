"""Gate 0: the vendored files are the files they claim to be. The package's OSL shader and EXR reader,
and the check fixtures, have the sha256 of their pinned legacy sources (known/pins.json, issue #16)."""
from __future__ import annotations

import hashlib

from .common import ROOT, Result
from .pinned import pins

VENDORED = {"fov_render/sensors/foveated_camera.osl": ("fov-3d-vision", "tools/foveated_camera.osl"),
            "fov_render/exr_lite.py": ("f3d-vision", "tools/exr_lite.py"),
            "fov_render/checks/fixtures/make_calib_room.py": ("fov-3d-vision", "tools/make_calib_room.py"),
            "fov_render/checks/fixtures/make_hdri_scene.py": ("fov-3d-vision", "tools/make_hdri_scene.py"),
            "fov_render/checks/fixtures/bl_common.py": ("fov-3d-vision", "tools/bl_common.py"),
            "fov_render/checks/fixtures/check_foveated.py": ("fov-3d-vision", "tools/check_foveated.py")}


def provenance() -> Result:
    p = pins()
    sha = {(f["repo"], f["path"]): f["sha256"] for f in p["files"]}
    bad = [a for a, key in VENDORED.items() if hashlib.sha256((ROOT / a).read_bytes()).hexdigest() != sha[key]]
    # control: a one-byte change to a file must change its hash
    first = next(iter(VENDORED))
    ctrl = hashlib.sha256((ROOT / first).read_bytes() + b" ").hexdigest() != sha[VENDORED[first]]
    return Result("0", "vendored files unchanged (shader, EXR reader, check fixtures)", not bad, ctrl,
                  {"files": len(VENDORED), "hash_mismatch": bad, "sources": p["sources"]})


CHECKS = [provenance]
