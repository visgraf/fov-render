"""Shared machinery for the gates: results, legacy imports, running Blender-side scripts."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
KNOWN = Path(__file__).resolve().parent / "known"


class _Legacy:
    """LEGACY["f3d"] / LEGACY["fov"]: the pinned legacy tools folder, rebuilt from a checkout's git
    history by checks.pinned (legacy/ was removed in issue #16). Raises pinned.LegacyUnavailable
    when no checkout is configured; the checks that need it SKIP with that reason."""

    def __getitem__(self, repo: str) -> Path:
        from .pinned import materialize
        return materialize(repo)


LEGACY = _Legacy()


@dataclass
class Result:
    step: str
    name: str
    ok: bool
    control_detected: bool | None
    detail: dict = field(default_factory=dict)
    skipped: str | None = None

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        return "PASS" if self.ok and self.control_detected is not False else "FAIL"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status
        return d


def have_bpy() -> bool:
    try:
        import bpy  # noqa: F401
        return True
    except ImportError:
        return False


@contextlib.contextmanager
def legacy_path(repo: str, *subdirs: str):
    """Put a legacy tools folder (and subfolders) first on sys.path; drop cached modules that
    exist in more than one legacy tree (fsg_geometry differs between the two repositories)."""
    dirs = [str(LEGACY[repo])] + [str(LEGACY[repo] / s) for s in subdirs]
    clash = {"fsg_geometry", "rig", "warp", "bl_common", "render_foveated", "exr_lite"}
    for name in clash:
        sys.modules.pop(name, None)
    old = list(sys.path)
    sys.path[:0] = dirs
    try:
        yield
    finally:
        sys.path[:] = old
        for name in clash:
            sys.modules.pop(name, None)


def load_legacy(repo: str, rel: str, *subdirs: str):
    """Import a legacy module by path under a unique name, with its folder on sys.path."""
    path = LEGACY[repo] / rel
    name = f"legacy_{repo}_{rel.replace('/', '_').removesuffix('.py')}"
    with legacy_path(repo, *subdirs, *([str(Path(rel).parent)] if "/" in rel else [])):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def call_with_argv(fn, script: str, args: list[str]):
    old = sys.argv
    sys.argv = [script, "--"] + [str(a) for a in args]
    try:
        return fn()
    finally:
        sys.argv = old


def blender_command(script: str, args: list[str], blend: str | None = None) -> list[str]:
    """Command that runs a Blender-side script: the Blender binary, or this Python with the bpy wheel."""
    import bpy
    binary = bpy.app.binary_path
    if binary and os.path.basename(binary).lower().startswith("blender"):
        cmd = [binary, "-b"] + ([blend] if blend else []) + ["--python-exit-code", "1", "-P", script, "--"]
    else:
        cmd = [sys.executable, script, "--"]
    return cmd + [str(a) for a in args]


def run_blender_script(script: str, args: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    from fov_render.bl import env as fenv
    fenv.prepare()
    keep = [str(ROOT)] + [x for x in env.get("PYTHONPATH", "").split(os.pathsep) if x]
    env["PYTHONPATH"] = os.pathsep.join(keep)
    p = subprocess.run(blender_command(script, args), capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"{script} failed ({p.returncode}):\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")
