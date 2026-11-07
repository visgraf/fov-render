"""The persistent render server: the Blender side of a Session (report section 2).

Started by fov_render.Session, by hand only for debugging. Either interpreter:

    blender -b scene.blend --python-exit-code 1 -P <path>/fov_render/server.py -- --queue Q --session S --reuse-loaded
    python -m fov_render.server --queue Q --session S          (the bpy wheel)

S is a fov-render-session-v1 JSON file. The scene opens once; with --reuse-loaded the file the
Blender binary loaded from its command line is used as it is, as f3d-vision's server does. Protocol, as f3d-vision's greedy render server:

    the server writes  Q/server-ready.json  once ready
    the host writes    Q/req-NNNN.json      (fov-render-request-v1, n = NNNN, from 1)
    the server writes  Q/done-NNNN.json     (fov-render-response-v1), or
                       Q/fail-NNNN.json     {"n", "error", "reasons" | "traceback"}
    the host touches   Q/stop               the server exits 0

A request that fails validation is refused (fail file with its reasons) and the server goes on;
any other failure writes the fail file and exits 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):                      # run as a script: blender -P .../fov_render/server.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fov_render.request import RequestError, dumps  # noqa: E402

PREFIX = "[fov-render server]"


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(dumps(data) if isinstance(data, dict) else data)
    os.replace(tmp, path)


def serve(queue: str, session_path: str, poll_s: float = 0.01, reuse_loaded: bool = False) -> int:
    from fov_render.bl.handler import Handler
    q = Path(queue).resolve()
    q.mkdir(parents=True, exist_ok=True)
    session = json.loads(Path(session_path).read_text())
    t0 = time.perf_counter()
    try:
        handler = Handler(session, reuse_loaded=reuse_loaded)
    except BaseException:
        _write(q / "fail-0000.json", {"n": 0, "error": "session failed to open", "traceback": traceback.format_exc()})
        raise
    ready = dict(handler.ready, start_seconds=time.perf_counter() - t0)
    _write(q / "server-ready.json", ready)
    print(f"{PREFIX} READY device={ready['device']} scene={ready['scene']['file']['path']}", flush=True)
    n = 1
    while True:
        if (q / "stop").exists():
            print(f"{PREFIX} STOP after {n - 1} requests", flush=True)
            return 0
        req_p = q / f"req-{n:04d}.json"
        if not req_p.exists():
            time.sleep(poll_s)
            continue
        try:
            req = json.loads(req_p.read_text())
            if not isinstance(req, dict) or req.get("n") != n:
                raise RequestError([f"n: request file {req_p.name} holds n = "
                                    f"{req.get('n') if isinstance(req, dict) else None!r}, expected {n}"])
            resp = handler.handle(req)
            _write(q / f"done-{n:04d}.json", resp)
        except RequestError as e:
            _write(q / f"fail-{n:04d}.json", {"n": n, "error": "request refused", "reasons": e.reasons})
            print(f"{PREFIX} REFUSED request {n}: {e}", flush=True)
        except json.JSONDecodeError as e:
            _write(q / f"fail-{n:04d}.json", {"n": n, "error": "request refused", "reasons": [f"not JSON: {e}"]})
        except BaseException:
            _write(q / f"fail-{n:04d}.json", {"n": n, "error": "render failed", "traceback": traceback.format_exc()})
            raise
        n += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--reuse-loaded", action="store_true",
                    help="the session's .blend was loaded on the Blender command line; do not reopen it")
    a = ap.parse_args(argv)
    return serve(a.queue, a.session, reuse_loaded=a.reuse_loaded)


if __name__ == "__main__":
    from fov_render.bl.env import loud, script_args
    argv = script_args() if "--" in sys.argv else sys.argv[1:]

    def _run():
        rc = main(argv)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    loud(_run, PREFIX)
