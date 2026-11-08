"""Make a derived copy of a .blend with an EYE camera (issue #4). The code is fov_render.bl.eye_scene,
also `fov-render eye`; this script runs it by hand:

    blender -b --python-exit-code 1 -P tools/make_eye_scene.py -- --blend IN.blend --out OUT.blend ...
    python tools/make_eye_scene.py -- ...                                   (the bpy wheel)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fov_render.bl.env import loud, script_args  # noqa: E402
from fov_render.bl.eye_scene import PREFIX, main  # noqa: E402


def _run():
    rc = main(script_args() if "--" in sys.argv else sys.argv[1:])
    sys.stdout.flush()
    os._exit(rc)


loud(_run, PREFIX)
