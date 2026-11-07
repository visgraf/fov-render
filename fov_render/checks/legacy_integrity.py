"""Gate 0: the legacy comparators are the files they claim to be (sha256 against legacy/PROVENANCE.json),
and the package's OSL shader and EXR reader are byte-identical to their legacy sources."""
from __future__ import annotations

import hashlib
import json

from .common import ROOT, Result


def provenance() -> Result:
    prov = json.loads((ROOT / "legacy" / "PROVENANCE.json").read_text())
    bad = [f["path"] for f in prov["files"]
           if hashlib.sha256((ROOT / f["path"]).read_bytes()).hexdigest() != f["sha256"]]
    vendored = {"fov_render/sensors/foveated_camera.osl": "legacy/fov-3d-vision/tools/foveated_camera.osl",
                "fov_render/exr_lite.py": "legacy/f3d-vision/tools/exr_lite.py"}
    diff = [a for a, b in vendored.items() if (ROOT / a).read_bytes() != (ROOT / b).read_bytes()]
    # control: a one-byte change to a copy must change its hash
    p = ROOT / prov["files"][0]["path"]
    ctrl = hashlib.sha256(p.read_bytes() + b" ").hexdigest() != prov["files"][0]["sha256"]
    return Result("0", "legacy comparators and vendored files unchanged", not bad and not diff, ctrl,
                  {"files": len(prov["files"]), "hash_mismatch": bad, "vendored_differs": diff,
                   "sources": prov["sources"]})


CHECKS = [provenance]
