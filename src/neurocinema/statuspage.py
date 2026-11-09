"""Local status page for staff: one page, a few buttons, no internet needed.

    GET  /             the page
    GET  /api/engine   the engine's state, calibration and "put down" numbers (JSON)
    POST /api/engine/recalibrate   calibrate now (visitor looks at the middle screen)
    POST /api/engine/idle          the headset is off (on its stand)
    POST /api/engine/reset         re-centre yaw (visitor faces the middle)
    GET  /api/status   the headset manager's snapshot, or {"managed": false}
    POST /api/select   {"label": "B"}   use headset B (staff then lift it)
    POST /api/release  stop using the current headset
    POST /api/reconnect

The headset routes need source.kind = "muse_direct". Runs on its own thread;
commands are queued to the main loop.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

from .app import PAGE_COMMANDS, PageLink
from .headsets import HeadsetManager

log = logging.getLogger(__name__)

MAX_BODY = 1024


class StatusPage:
    def __init__(self, manager: HeadsetManager | None, host: str, port: int, engine: PageLink | None = None):
        self.manager = manager
        page = files("neurocinema").joinpath("static/status.html").read_bytes()
        handler = _make_handler(manager, engine, page)
        self.server = ThreadingHTTPServer((host, port), handler)
        self.server.daemon_threads = True
        self.address = f"http://{host}:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, name="status-page", daemon=True)

    def start(self) -> None:
        self._thread.start()
        log.info("status page on %s", self.address)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _make_handler(manager: HeadsetManager | None, engine: PageLink | None, page: bytes):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the engine log clean
            log.debug("%s " + fmt, self.address_string(), *args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data: dict) -> None:
            self._send(code, json.dumps(data).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self._json(200, {**manager.snapshot(), "managed": True} if manager else {"managed": False})
            elif self.path == "/api/engine" and engine is not None:
                self._json(200, engine.status())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._json(413, {"error": "too large"})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._json(400, {"error": "bad json"})
                return
            command = self.path.removeprefix("/api/engine/")
            if engine is not None and command != self.path and command in PAGE_COMMANDS:
                engine.submit(command)
                self._json(200, {"ok": True})
                return
            if manager is None:
                self._json(404, {"error": "not found"})
                return
            if self.path == "/api/select":
                label = body.get("label") if isinstance(body, dict) else None
                if label not in {h["label"] for h in manager.snapshot().get("headsets", [])}:
                    self._json(400, {"error": "unknown headset"})
                    return
                manager.submit("select", label)
            elif self.path == "/api/release":
                manager.submit("release")
            elif self.path == "/api/reconnect":
                manager.submit("reconnect")
            else:
                self._json(404, {"error": "not found"})
                return
            self._json(200, {"ok": True})

    return Handler
