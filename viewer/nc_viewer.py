"""`nc-viewer`: a 3D view of the room, the visitor's gaze and the targets.

Listens to the engine's OSC output (like nc-monitor) and relays it to a
browser page over Server-Sent Events. Standard library + python-osc only.

    python viewer/nc_viewer.py            # then open http://127.0.0.1:8080

Port 7000 can only be bound by one program. To run alongside TouchDesigner
or nc-monitor, add "127.0.0.1:7003" to osc.destinations and pass --osc-port 7003.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import webbrowser
from dataclasses import asdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from neurocinema.config import ConfigError, load_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room_geometry import RoomError, load_room

STATIC = Path(__file__).resolve().parent / "static"
CONFIG_FILES = (ROOT / "config/installation.toml", ROOT / "config/local.toml")
ROOM_FILE = ROOT / "config/room.toml"


class Hub:
    """Fans each OSC message out to every connected browser."""

    def __init__(self, prefix: str):
        self.prefix = prefix.rstrip("/") + "/"
        self.clients: set[queue.Queue] = set()
        self.lock = threading.Lock()
        # Last value per address, so a page opened late starts in the right state.
        self.latest: dict[str, str] = {}

    def handle(self, address: str, *args: object) -> None:
        name = address[len(self.prefix):] if address.startswith(self.prefix) else address
        msg = json.dumps({"a": name, "v": list(args)})
        with self.lock:
            if name != "event":
                self.latest[name] = msg
            for q in self.clients:
                if q.qsize() < 200:  # a stalled tab must not grow memory
                    q.put(msg)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            for msg in self.latest.values():
                q.put(msg)
            self.clients.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            self.clients.discard(q)


def make_handler(hub: Hub, config_paths: list[Path], room_path: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(STATIC), **kw)

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/config.json":
                # Re-read on every page load, so a reload picks up edited angles.
                try:
                    body = json.dumps({"targets": asdict(load_config(config_paths).targets),
                                       "room": load_room(room_path)})
                except (ConfigError, RoomError, OSError) as e:
                    self.send_error(500, str(e))
                    return
                self._send(body.encode(), "application/json")
            elif self.path == "/events":
                self._stream()
            else:
                super().do_GET()

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            q = hub.subscribe()
            try:
                while True:
                    try:
                        msg = q.get(timeout=5)
                        self.wfile.write(f"data: {msg}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.unsubscribe(q)

    return Handler


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nc-viewer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--osc-host", default="127.0.0.1")
    p.add_argument("--osc-port", type=int, default=7000, help="port the engine sends to (default 7000)")
    p.add_argument("--prefix", default="/nc")
    p.add_argument("--http-port", type=int, default=8080, help="port of the web page (default 8080)")
    p.add_argument("-c", "--config", type=Path, action="append",
                   help="TOML config file (repeatable; default config/installation.toml then config/local.toml)")
    p.add_argument("--room", type=Path, default=ROOM_FILE,
                   help="room/screen geometry TOML (default config/room.toml)")
    p.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    args = p.parse_args(argv)

    config_paths = args.config or [c for c in CONFIG_FILES if c.exists()]
    try:
        load_config(config_paths)
        load_room(args.room)
    except (ConfigError, RoomError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1

    hub = Hub(args.prefix)
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(hub.handle)
    try:
        osc = ThreadingOSCUDPServer((args.osc_host, args.osc_port), dispatcher)
    except OSError as e:
        print(f"cannot listen for OSC on {args.osc_host}:{args.osc_port}: {e}\n"
              "If TouchDesigner or nc-monitor already uses this port, add another destination "
              "in the config (osc.destinations) and pass --osc-port.", file=sys.stderr)
        return 1
    threading.Thread(target=osc.serve_forever, daemon=True).start()

    http = ThreadingHTTPServer(("127.0.0.1", args.http_port), make_handler(hub, config_paths, args.room))
    http.daemon_threads = True
    url = f"http://127.0.0.1:{args.http_port}"
    print(f"OSC in on {args.osc_host}:{args.osc_port}, viewer at {url} (Ctrl-C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
