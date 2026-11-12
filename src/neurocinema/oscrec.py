"""Record the engine's OSC output to a file, and play it back later.

For working on the TouchDesigner side without a headset or a visitor: record a
real session once, then replay it to TouchDesigner as often as needed, with the
original timing.

    nc-osc-record                          # listen on 7005, Ctrl-C to stop and save
    nc-osc-record --name nods-left-top     # recordings/osc/<date-time>_nods-left-top.jsonl
    nc-osc-play --latest                   # newest recording -> 127.0.0.1:7000
    nc-osc-play FILE --loop --speed 2      # loop it, twice as fast
    nc-osc-play FILE --start 30 --end 90   # one stretch (seconds into the recording)
    nc-osc-play FILE --info                # what is in it, without sending

The engine sends to every address in [osc] destinations, so to record next to
TouchDesigner add "127.0.0.1:7005" there (config/local.toml).

Only what the engine sends on OSC is recorded (pose, targets, gestures, state,
events, /nc/cam/...): no raw sensor samples, no EEG, nothing about the visitor.

File format: JSON Lines, written as messages arrive (a crash loses nothing).
The first line is {"meta": {...}}, then one {"t", "address", "args", "types"}
per message (t = seconds since the recording started; types = OSC type tags, so
playback sends exactly the same types), and on a clean stop a last
{"end": {...}} line with a summary. nc-osc-play also reads the older
{"meta", "messages"} JSON files of the NeuroTheater osc_recorder.py.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import signal
import socket
import sys
import time
from collections import Counter
from pathlib import Path

from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_packet import OscPacket, ParseError

from .config import parse_host_port

DEFAULT_DIR = Path("recordings/osc")
DEFAULT_RECORD_PORT = 7005  # 7000 TD, 7001 engine commands, 7002 monitor, 7003 viewer, 7004 camera
SCHEMA = "nc-osc-recording/1"
# Type tags replayed as recorded; anything else (arrays, MIDI, ...) is inferred from the value.
_PLAIN_TAGS = set("ifdhsTFNbc")


# ---- recording ------------------------------------------------------------

def _type_tags(dgram: bytes) -> str:
    """The OSC type tag string of one message datagram, without the leading ','."""
    end = dgram.index(b"\x00")
    i = (end + 4) & ~3  # address is padded to a multiple of 4
    if i >= len(dgram) or dgram[i:i + 1] != b",":
        return ""
    return dgram[i + 1:dgram.index(b"\x00", i)].decode("ascii", "replace")


def _jsonable(value: object) -> object:
    if isinstance(value, bytes):
        return {"blob": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)  # MIDI, timetags: kept readable, replayed as a string


class Recorder:
    def __init__(self, path: Path, host: str, port: int, prefixes: list[str], quiet: bool):
        self.path = path
        self.prefixes = prefixes
        self.quiet = quiet
        self.counts: Counter[str] = Counter()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")
        self.t0_mono = time.monotonic()
        self.started = dt.datetime.now().astimezone()
        self.first_t: float | None = None
        self.last_t = 0.0
        self._write({"meta": {
            "schema": SCHEMA,
            "recorded_at": self.started.isoformat(timespec="seconds"),
            "listen": f"{host}:{port}",
            "filter": prefixes,
        }})
        self._last_draw = 0.0

    def _write(self, obj: dict) -> None:
        self.file.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def run(self, stop: list[bool]) -> None:
        while not stop[0]:
            try:
                data, _ = self.sock.recvfrom(65536)
            except socket.timeout:
                self._draw()
                continue
            t = time.monotonic() - self.t0_mono
            try:
                packet = OscPacket(data)
            except ParseError:
                continue
            for timed in packet.messages:
                self._record(t, timed.message)
            self.file.flush()
            self._draw()

    def _record(self, t: float, msg) -> None:
        address = msg.address
        if self.prefixes and not any(address.startswith(p) for p in self.prefixes):
            return
        if self.first_t is None:
            self.first_t = t
        self.last_t = t
        self.counts[address] += 1
        self._write({"t": round(t, 4), "address": address,
                     "args": [_jsonable(a) for a in msg.params], "types": _type_tags(msg.dgram)})
        if not self.quiet and not sys.stdout.isatty():
            print(f"{t:9.3f}  {address}  {' '.join(map(str, msg.params))}")

    def _draw(self) -> None:
        if self.quiet or not sys.stdout.isatty():
            return
        now = time.monotonic()
        if now - self._last_draw < 0.25:
            return
        self._last_draw = now
        n = sum(self.counts.values())
        el = now - self.t0_mono
        sys.stdout.write(f"\r\x1b[2K  recording {el:7.1f} s   {n:7d} messages   "
                         f"{len(self.counts)} addresses")
        sys.stdout.flush()

    def close(self) -> dict:
        n = sum(self.counts.values())
        duration = (self.last_t - self.first_t) if self.first_t is not None else 0.0
        summary = {
            "message_count": n,
            "duration_s": round(duration, 3),
            "streams": [{"address": a, "count": c} for a, c in self.counts.most_common()],
        }
        self._write({"end": summary})
        self.file.close()
        self.sock.close()
        return summary


def record_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nc-osc-record", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0 = all)")
    p.add_argument("--port", type=int, default=DEFAULT_RECORD_PORT,
                   help=f"port to listen on (default {DEFAULT_RECORD_PORT}; add it to osc.destinations)")
    p.add_argument("--out", type=Path, help="output file (default recordings/osc/<date-time>[_name].jsonl)")
    p.add_argument("--name", help="short label added to the file name, e.g. nods-left-top")
    p.add_argument("--filter", action="append", default=[], metavar="PREFIX",
                   help="only record addresses starting with this (repeatable), e.g. /nc/cam")
    p.add_argument("-q", "--quiet", action="store_true", help="no live counter")
    args = p.parse_args(argv)

    if args.out is None:
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        label = f"_{args.name}" if args.name else ""
        args.out = DEFAULT_DIR / f"{stamp}{label}.jsonl"
    try:
        rec = Recorder(args.out, args.host, args.port, args.filter, args.quiet)
    except OSError as e:
        print(f"cannot listen on {args.host}:{args.port}: {e}\n"
              "Another program has this port (nc-osc-record already running?). Pass --port.",
              file=sys.stderr)
        return 1

    stop = [False]

    def _stop(*_: object) -> None:
        stop[0] = True

    signal.signal(signal.SIGTERM, _stop)
    print(f"recording OSC on {args.host}:{args.port} -> {args.out}")
    print(f"the engine must send there: osc.destinations includes \"127.0.0.1:{args.port}\"")
    print("Ctrl-C to stop and save")
    try:
        rec.run(stop)
    except KeyboardInterrupt:
        pass
    summary = rec.close()
    if sys.stdout.isatty() and not args.quiet:
        print()
    if summary["message_count"] == 0:
        print("nothing received; the (empty) file is kept:", args.out)
        print(f"check that the engine is running and sends to 127.0.0.1:{args.port}")
        return 0
    print(f"saved {args.out}")
    _print_summary(summary)
    return 0


# ---- playback -------------------------------------------------------------

def load_recording(path: Path) -> tuple[dict, list[dict]]:
    """(meta, messages sorted by time, t starting at 0) from either file format."""
    text = path.read_text(encoding="utf-8")
    meta: dict = {}
    messages: list[dict] = []
    try:
        whole = json.loads(text)  # NeuroTheater osc_recorder.py: one object, epoch times
    except json.JSONDecodeError:
        whole = None  # JSON Lines
    if isinstance(whole, dict) and isinstance(whole.get("messages"), list):
        meta = whole.get("meta", {})
        messages = [m for m in whole["messages"] if "address" in m]
    else:
        for n, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # A recording cut off mid-line (power loss): keep what came before.
                print(f"{path}:{n}: unreadable line skipped", file=sys.stderr)
                continue
            if "meta" in obj:
                meta = obj["meta"]
            elif "address" in obj:
                messages.append(obj)
    messages.sort(key=lambda m: m["t"])
    if messages:
        t0 = messages[0]["t"]
        for m in messages:
            m["t"] = m["t"] - t0
    return meta, messages


def _arg_value(value: object) -> object:
    if isinstance(value, dict) and "blob" in value:
        return base64.b64decode(value["blob"])
    return value


def build_message(entry: dict) -> bytes:
    builder = OscMessageBuilder(address=entry["address"])
    args = entry.get("args", [])
    tags = entry.get("types")
    exact = isinstance(tags, str) and len(tags) == len(args) and set(tags) <= _PLAIN_TAGS
    for i, a in enumerate(args):
        v = _arg_value(a)
        if exact:
            builder.add_arg(v, tags[i])
        else:
            builder.add_arg(v)
    return builder.build().dgram


def select(messages: list[dict], start: float, end: float | None, prefixes: list[str],
           exclude: list[str]) -> list[dict]:
    out = []
    for m in messages:
        if m["t"] < start or (end is not None and m["t"] > end):
            continue
        a = m["address"]
        if prefixes and not any(a.startswith(p) for p in prefixes):
            continue
        if any(a.startswith(p) for p in exclude):
            continue
        out.append(m)
    return out


def latest_recording(folder: Path) -> Path | None:
    files = [f for f in folder.glob("*.json*") if f.is_file()]
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def _print_summary(summary: dict) -> None:
    n = summary["message_count"]
    d = summary["duration_s"]
    rate = n / d if d > 0 else 0.0
    print(f"  {n} messages over {_fmt_time(d)} ({rate:.1f}/s)")
    for s in summary["streams"][:30]:
        print(f"    {s['address']:<28} {s['count']}")
    if len(summary["streams"]) > 30:
        print(f"    ... and {len(summary['streams']) - 30} more addresses")


def _fmt_time(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


def _summary_of(messages: list[dict]) -> dict:
    counts = Counter(m["address"] for m in messages)
    return {
        "message_count": len(messages),
        "duration_s": round(messages[-1]["t"] - messages[0]["t"], 3) if messages else 0.0,
        "streams": [{"address": a, "count": c} for a, c in counts.most_common()],
    }


# Sent continuously (headset and camera): left out of --info's event list.
_CONTINUOUS = ("/pose", "/aim", "/dwell", "/calibration", "/calib", "/status", "/status/stream",
               "/heartbeat")
# Re-sent regularly by the snapshot: --info shows them only when they change.
_SNAPSHOT = ("/state", "/phase", "/target", "/setup", "/screen", "/half", "/headset")


def _print_events(messages: list[dict]) -> None:
    """The discrete messages (events, state changes, selections), for picking --start/--end."""
    last: dict[str, list] = {}
    for m in messages:
        a, args = m["address"], m.get("args", [])
        if a.endswith(_CONTINUOUS):
            continue
        if a.endswith(_SNAPSHOT):
            if last.get(a) == args:
                continue
            last[a] = args
        print(f"  {_fmt_time(m['t']):>8}  {a} {' '.join(map(str, args))}")


def play_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nc-osc-play", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path, nargs="?", help="a recording (.jsonl, or NeuroTheater .json)")
    p.add_argument("--latest", action="store_true", help="the newest file in recordings/osc/")
    p.add_argument("--to", action="append", metavar="HOST:PORT",
                   help="where to send (repeatable; default 127.0.0.1:7000, TouchDesigner)")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (2 = twice as fast)")
    p.add_argument("--loop", action="store_true", help="start again at the end, until Ctrl-C")
    p.add_argument("--gap", type=float, default=1.0, help="pause between loops, seconds")
    p.add_argument("--start", type=float, default=0.0, help="start this many seconds into the recording")
    p.add_argument("--end", type=float, help="stop this many seconds into the recording")
    p.add_argument("--filter", action="append", default=[], metavar="PREFIX",
                   help="only send addresses starting with this (repeatable), e.g. /nc/cam")
    p.add_argument("--exclude", action="append", default=[], metavar="PREFIX",
                   help="do not send addresses starting with this (repeatable), e.g. /nc/cam")
    p.add_argument("--info", action="store_true",
                   help="print what the recording holds (addresses, events with their times) and exit")
    p.add_argument("-v", "--verbose", action="store_true", help="print every message sent")
    args = p.parse_args(argv)

    path = args.file
    if args.latest:
        path = latest_recording(DEFAULT_DIR)
        if path is None:
            print(f"no recordings in {DEFAULT_DIR}/", file=sys.stderr)
            return 1
    if path is None:
        p.error("give a recording file, or --latest")
    if not path.is_file():
        print(f"not found: {path}", file=sys.stderr)
        return 1
    if args.speed <= 0:
        p.error("--speed must be above 0")

    meta, messages = load_recording(path)
    chosen = select(messages, args.start, args.end, args.filter, args.exclude)
    print(f"{path}" + (f"  (recorded {meta['recorded_at']})" if meta.get("recorded_at") else ""))
    if args.info:
        _print_summary(_summary_of(messages))
        print("  events and changes:")
        _print_events(messages)
        return 0
    if not chosen:
        print("nothing to send (empty recording, or --start/--end/--filter left nothing)",
              file=sys.stderr)
        return 1

    dests = []
    for d in args.to or ["127.0.0.1:7000"]:
        dests.append(parse_host_port(d))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packets = [(m["t"], build_message(m), m) for m in chosen]
    span = packets[-1][0] - packets[0][0]
    print(f"sending {len(packets)} messages ({_fmt_time(span)} at x{args.speed:g}) to "
          + ", ".join(f"{h}:{pt}" for h, pt in dests)
          + (" in a loop" if args.loop else "") + ". Ctrl-C to stop.")

    tty = sys.stdout.isatty() and not args.verbose
    loops = 0
    try:
        while True:
            loops += 1
            t_first = packets[0][0]
            wall0 = time.monotonic()
            last_draw = 0.0
            for t, dgram, m in packets:
                due = wall0 + (t - t_first) / args.speed
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                for dest in dests:
                    try:
                        sock.sendto(dgram, dest)
                    except OSError:
                        pass  # nobody listening yet: UDP, carry on
                if args.verbose:
                    print(f"{_fmt_time(t):>8}  {m['address']} {' '.join(map(str, m.get('args', [])))}")
                elif tty and time.monotonic() - last_draw > 0.2:
                    last_draw = time.monotonic()
                    sys.stdout.write(f"\r\x1b[2K  {_fmt_time(t)} / {_fmt_time(packets[-1][0])}"
                                     + (f"   loop {loops}" if args.loop else ""))
                    sys.stdout.flush()
            if not args.loop:
                break
            time.sleep(args.gap)
    except KeyboardInterrupt:
        pass
    if tty:
        print()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(play_main())
