"""Running Blender-side code under either interpreter.

Two ways to run it, not interchangeable (CLAUDE.md):
    blender -b [file.blend] --python-exit-code 1 -P script.py -- ARGS   (the workstation)
    python script.py -- ARGS        with the `bpy` wheel, pip install bpy==5.2.1 (CPU sandboxes)

The wheel ships OSL support but keeps the `oslquery` module in a bundled site-packages that is
not on sys.path, so a custom OSL camera fails with "No module named 'oslquery'" (measured in the
Chat sandbox, 2026-11-07). `prepare()` adds that folder.
"""
from __future__ import annotations

import os
import sys
import traceback


def prepare() -> None:
    import bpy
    try:
        import oslquery  # noqa: F401
        return
    except ImportError:
        pass
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cand = os.path.join(bpy.utils.resource_path("LOCAL"), "python", "lib", py, "site-packages")
    if os.path.isdir(os.path.join(cand, "oslquery")) and cand not in sys.path:
        sys.path.append(cand)


def script_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def loud(main, prefix: str = "[fov-render]"):
    """Run main(); on any exception print the traceback and a FAILED line and exit 1.
    `blender -b -P` exits 0 even when the script raised, so a tool must make its own failure loud."""
    try:
        return main()
    except BaseException:  # noqa: BLE001 - SystemExit included on purpose
        traceback.print_exc()
        print(f"{prefix} FAILED", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
