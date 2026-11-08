"""Which Blender runs the Blender-side code, and the command line that runs it (numpy only).

    launcher="auto"     $FOVRENDER_BLENDER, else `blender` on PATH, else this Python with the bpy wheel
    launcher="blender"  $FOVRENDER_BLENDER or `blender` on PATH, nothing else
    launcher="python"   this Python (it must import bpy)
    any other string    that executable (a Blender binary if its name starts with "blender")

The two interpreters are not interchangeable (CLAUDE.md): the Blender binary runs a script with
`-b [file.blend] --python-exit-code 1 -P script.py -- ARGS`, a Python runs `-m module ARGS`.
`--python-exit-code 1` makes an uncaught exception exit 1; the scripts also exit nonzero
themselves (bl.env.loud), because `blender -b -P` exits 0 otherwise.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class LaunchError(RuntimeError):
    pass


def find(launcher: str = "auto") -> str:
    """The executable for launcher (see the module docstring)."""
    if launcher == "auto":
        exe = os.environ.get("FOVRENDER_BLENDER") or shutil.which("blender")
        if exe is None:
            try:
                import bpy  # noqa: F401
            except ImportError:
                raise LaunchError("no Blender: set FOVRENDER_BLENDER, put blender on PATH, or install the bpy wheel")
            exe = sys.executable
        return exe
    if launcher == "blender":
        exe = os.environ.get("FOVRENDER_BLENDER") or shutil.which("blender")
        if exe is None:
            raise LaunchError("launcher='blender' but no blender on PATH and FOVRENDER_BLENDER unset")
        return exe
    if launcher == "python":
        return sys.executable
    return launcher


def is_blender(exe: str) -> bool:
    return Path(exe).name.lower().startswith("blender")


def command(launcher: str, script: Path, module: str, args, *, blend=None, blender_only_args=()):
    """(argv, env, info) running `script` in the Blender binary or `module` in a Python.

    blend: loaded on the Blender binary's command line (a Python opens it itself);
    blender_only_args: appended for the Blender binary only (e.g. the server's --reuse-loaded)."""
    exe = find(launcher)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    args = [str(a) for a in args]
    if is_blender(exe):
        return ([exe, "-b", *([str(blend)] if blend is not None else []), "--python-exit-code", "1", "-P", str(script),
                 "--", *args, *blender_only_args], env, {"kind": "blender", "executable": exe})
    return [exe, "-m", module, *args], env, {"kind": "python", "executable": exe}
