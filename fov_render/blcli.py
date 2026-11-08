"""The Blender side of the `fov-render` subcommands that need bpy (info, eye, check, selftest).

Started by fov_render.cli through fov_render.launch, by hand only for debugging:

    blender -b --python-exit-code 1 -P <path>/fov_render/blcli.py -- SUBCOMMAND ARGS   (the workstation)
    python -m fov_render.blcli SUBCOMMAND ARGS                                         (the bpy wheel)

The host has already checked the arguments. Exit status: the subcommand's, 1 on any exception
(the Blender binary would exit 0 otherwise, CLAUDE.md).
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

if __package__ in (None, ""):                      # run as a script: blender -P .../fov_render/blcli.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PREFIX = "[fov-render]"


def info(argv) -> int:
    import argparse
    from fov_render.bl import env
    from fov_render.bl import scene as S
    env.prepare()
    ap = argparse.ArgumentParser(prog="fov-render info")
    ap.add_argument("--blend", required=True)
    ap.add_argument("--instance-ids", default="complete", choices=["complete", "legacy"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--catalogue", action="store_true", help="include every catalogue entry")
    a = ap.parse_args(argv)
    h = S.open_scene(a.blend, scale_length=None, instance_ids=a.instance_ids)
    rep = h.report()
    if a.catalogue:
        rep["instances"]["entries"] = h.catalog.get("instances")
    text = json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n"
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(text)
        print(f"{PREFIX} wrote {a.json}", flush=True)
    else:
        sys.stdout.write("FOVRENDER-JSON-BEGIN\n" + text + "FOVRENDER-JSON-END\n")
    return 0


def eye(argv) -> int:
    from fov_render.bl.eye_scene import main
    return main(argv)


def check(argv) -> int:
    from fov_render.bl import env
    from fov_render.checks.scene import main
    env.prepare()
    return main(argv)


def selftest(argv) -> int:
    from fov_render.checks import main
    try:
        from fov_render.bl import env
        env.prepare()
    except ImportError:
        pass
    return main(argv)


COMMANDS = {"info": info, "eye": eye, "check": check, "selftest": selftest}


def run(argv) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(f"{PREFIX} expected one of {sorted(COMMANDS)}, got {argv[:1]}", file=sys.stderr, flush=True)
        return 2
    try:
        rc = COMMANDS[argv[0]](argv[1:])
    except SystemExit as e:                       # the tools' own refusals: SystemExit("message")
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            rc = 1
        else:
            rc = int(e.code or 0)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        rc = 1
    if rc:
        print(f"{PREFIX} {argv[0]} FAILED (exit {rc})", flush=True)
    return int(rc or 0)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    code = run(args)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
