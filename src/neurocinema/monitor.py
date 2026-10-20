"""`nc-monitor`: print what the engine sends, without TouchDesigner.

Discrete messages (state, target, events, errors) are printed as lines;
continuous ones (pose, dwell, status) are folded into one live status line.
"""

from __future__ import annotations

import argparse
import sys
import time

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

CONTINUOUS = ("pose", "dwell", "calibration", "status", "status/stream", "heartbeat")
# Re-sent every second by the snapshot; only print them when they change.
SNAPSHOT = ("state", "phase", "target")


class Monitor:
    def __init__(self, prefix: str, raw: bool):
        self.prefix = prefix.rstrip("/") + "/"
        self.raw = raw
        self.tty = sys.stdout.isatty()
        self.live: dict[str, tuple] = {}
        self.last: dict[str, tuple] = {}
        self._last_draw = 0.0

    def handle(self, address: str, *args: object) -> None:
        if self.raw:
            self._line(f"{address} {' '.join(map(str, args))}")
            return
        name = address[len(self.prefix):] if address.startswith(self.prefix) else address
        if name in CONTINUOUS:
            self.live[name] = args
            self._draw()
            return
        if name in SNAPSHOT:
            if self.last.get(name) == args:
                return
            self.last[name] = args
        self._line(f"{name:<14} {' '.join(map(str, args))}")

    def _line(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        if self.tty:
            sys.stdout.write("\r\x1b[2K")
        print(f"{stamp}  {text}", flush=True)
        self._draw(force=True)

    def _draw(self, force: bool = False) -> None:
        if not self.tty:
            return
        now = time.monotonic()
        if not force and now - self._last_draw < 0.05:
            return
        self._last_draw = now
        parts = []
        if "pose" in self.live:
            y, p, r = self.live["pose"]
            parts.append(f"yaw {y:+6.1f}  pitch {p:+6.1f}  roll {r:+5.1f}")
        if "dwell" in self.live:
            d = float(self.live["dwell"][0])
            parts.append("dwell " + "#" * int(d * 10) + "." * (10 - int(d * 10)))
        if "calibration" in self.live and self.last.get("state") == ("calibrating",):
            parts.append(f"calibrating {float(self.live['calibration'][0]) * 100:3.0f}%")
        if "status" in self.live:
            c, age, rate, motion = self.live["status"]
            parts.append(f"{'link ok' if c else 'NO LINK'} {rate:4.1f} Hz age {age} ms motion {motion}")
        if "status/stream" in self.live:
            st, fails = self.live["status/stream"]
            parts.append(f"muselsl {st} ({fails} fail)")
        sys.stdout.write("\r\x1b[2K" + "  |  ".join(parts))
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nc-monitor", description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7000, help="port the engine sends to (default 7000)")
    p.add_argument("--prefix", default="/nc")
    p.add_argument("--raw", action="store_true", help="print every message")
    args = p.parse_args(argv)

    mon = Monitor(args.prefix, args.raw)
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(mon.handle)
    try:
        server = BlockingOSCUDPServer((args.host, args.port), dispatcher)
    except OSError as e:
        print(f"cannot listen on {args.host}:{args.port}: {e}\n"
              "If TouchDesigner already uses this port, add another destination in the "
              "config (osc.destinations) and pass --port.", file=sys.stderr)
        return 1
    print(f"listening on {args.host}:{args.port} (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
