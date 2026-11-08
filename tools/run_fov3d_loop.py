"""fov-3d-vision's tools/active_loop.py, unchanged, under either interpreter (issue #3).

    python tools/run_fov3d_loop.py -- --fov3d /path/to/fov-3d-vision <active_loop.py's arguments>

The legacy reference for the exact test. Under the bpy wheel the OSL camera needs oslquery on the
path (fov_render.bl.env.prepare); under the Blender binary run active_loop.py itself.
"""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fov_render.bl import env  # noqa: E402

env.prepare()
i = sys.argv.index("--")
k = sys.argv.index("--fov3d", i)
tools = str(Path(sys.argv[k + 1]).resolve() / "tools")
del sys.argv[k:k + 2]
sys.path.insert(0, tools)
sys.argv[0] = os.path.join(tools, "active_loop.py")
runpy.run_path(sys.argv[0], run_name="__main__")
