"""The legacy files fov_render replaced, rebuilt from a checkout's git history (issue #16).

Until v0.1.0 they were copies in legacy/. Now known/pins.json lists each one (repository, path,
sha256) with the commit it was copied from, and materialize() writes exactly those files, read with
`git show <commit>:<path>` from a checkout of the original repository. So the checkout can be at
any commit, as long as its history has the pinned one. Every file's sha256 is checked.

    $FOVRENDER_F3D_ROOT     a checkout of visgraf/f3d-vision
    $FOVRENDER_FOV3D_ROOT   a checkout of visgraf/fov-3d-vision

The Classroom gates use them (they also need the workstation's Classroom files), and
tools/freeze_known_answers.py can be re-run from them.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

PINS = Path(__file__).resolve().parent / "known" / "pins.json"
REPO = {"f3d": ("f3d-vision", "FOVRENDER_F3D_ROOT"), "fov": ("fov-3d-vision", "FOVRENDER_FOV3D_ROOT")}


class LegacyUnavailable(RuntimeError):
    """The pinned legacy files cannot be rebuilt here (no checkout, or its history lacks the commit)."""


def pins() -> dict:
    return json.loads(PINS.read_text())


def materialize(repo: str, out: Path | None = None) -> Path:
    """The pinned tools/ folder of repo ("f3d" or "fov"), written under out (default: a cache folder);
    returns its path. Raises LegacyUnavailable with the reason."""
    name, env = REPO[repo]
    root = os.environ.get(env)
    if not root or not (Path(root) / ".git").exists():
        raise LegacyUnavailable(f"set {env} to a git checkout of visgraf/{name}")
    p = pins()
    commit = p["sources"][name]["commit"]
    base = Path(out) if out else Path(os.environ.get("FOVRENDER_CACHE", Path(tempfile.gettempdir()) / "fov-render-pinned"))
    dest = base / f"{name}-{commit[:12]}"
    done = dest / ".complete"
    if done.is_file():
        return dest / "tools"
    have = subprocess.run(["git", "-C", root, "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True)
    if have.returncode != 0:
        raise LegacyUnavailable(f"{root} has no commit {commit} (git fetch it, or use a full clone)")
    for f in p["files"]:
        if f["repo"] != name:
            continue
        data = subprocess.run(["git", "-C", root, "show", f"{commit}:{f['path']}"], capture_output=True)
        if data.returncode != 0:
            raise LegacyUnavailable(f"{name} {commit[:12]}: {f['path']} not found ({data.stderr.decode()[-200:]})")
        got = hashlib.sha256(data.stdout).hexdigest()
        if got != f["sha256"]:
            raise LegacyUnavailable(f"{name} {commit[:12]}: {f['path']} has sha256 {got}, pinned {f['sha256']}")
        target = dest / f["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data.stdout)
    done.write_text(commit + "\n")
    return dest / "tools"
