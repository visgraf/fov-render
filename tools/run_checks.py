"""Run the gates inside the Blender binary (the workstation):

    blender -b --python-exit-code 1 -P tools/run_checks.py -- --device OPTIX [--steps 1 2 3 4]

With the bpy wheel use `python -m fov_render.checks` instead. Exit status 1 if any check fails.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fov_render.bl.env import loud, script_args  # noqa: E402
from fov_render.checks import main  # noqa: E402


def _run():
    rc = main(script_args())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


loud(_run, "[fov-render check]")
