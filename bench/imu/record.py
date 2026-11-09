"""Record a bench session: Muse ACC + GYRO, with spoken prompts.

IMU only (no EEG), for comparing orientation backends offline with replay.py.
This is not part of nc-engine, which never records anything.

    python bench/imu/record.py --participant P01
    python bench/imu/record.py --participant P01 --quick --no-voice   # 2-minute dry run
    python bench/imu/record.py --participant P01 --source muse_direct --headset A
    python bench/imu/record.py --participant T1 --fake-headset --quick  # no Muse needed

Two ways to reach the headset, through the engine's own sources, so samples
and timestamps are exactly what the engine would see:

  muse         the LSL streams (`source.kind = "muse"`). nc-engine (and
               nc-monitor) can run at the same time.
  muse_direct  connects to the headset in-process (`source.kind =
               "muse_direct"`, `[headsets] registry`). Stop nc-engine first: a
               Muse accepts one Bluetooth connection.

The default is the config's `source.kind`. Writes
bench/imu/recordings/<participant>_<date-time>/ (see recording.py).
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

import protocol  # noqa: E402
from recording import Event, EventWriter, ImuWriter, timing, write_meta  # noqa: E402

from neurocinema.config import ConfigError, load_config  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--participant", required=True, help="a code such as P01 (no names)")
    p.add_argument("--out", type=Path, default=HERE / "recordings")
    p.add_argument("-c", "--config", type=Path, action="append",
                   help="engine config (default config/installation.toml, then config/local.toml)")
    p.add_argument("--yaw", help="left,middle,right target yaw in degrees (default: [targets] in the config)")
    p.add_argument("--top-pitch", type=float, default=15.0, help="top targets, degrees above the calibration gaze")
    p.add_argument("--bottom-pitch", type=float, default=-15.0, help="bottom targets (negative = below)")
    p.add_argument("--stand-s", type=float, help="time on the stand at the start (default 120, 20 with --quick)")
    p.add_argument("--quick", action="store_true", help="short protocol (about 2 minutes) for a dry run")
    p.add_argument("--no-voice", action="store_true", help="do not speak the prompts (macOS `say`)")
    p.add_argument("--wait-s", type=float, default=90.0, help="give up if the stream is not found in time")
    p.add_argument("--source", choices=("muse", "muse_direct"),
                   help="LSL streams or in-process connection (default: source.kind in the config)")
    p.add_argument("--headset", help="muse_direct: headset label from [headsets] registry "
                   "(default: the only one there)")
    p.add_argument("--fake-headset", action="store_true",
                   help="muse_direct with a simulated headset, to try the recorder without a Muse")
    return p.parse_args(argv)


def open_lsl(cfg):
    try:
        from neurocinema.sources.lsl import LslImuSource
    except ImportError as e:
        print(f"LSL support is not installed ({e}); run: pip install -e '.[muse]'", file=sys.stderr)
        return 4
    print(f"Looking for the Muse LSL streams (name '{cfg.muse.stream_name}')... "
          "start nc-engine, or `muselsl stream --acc --gyro --lsltime --disable-eeg`, if nothing happens.")
    return LslImuSource(cfg.muse)


def open_direct(cfg, label: str | None, fake: bool):
    from neurocinema.cli import SIM_BATTERY, make_headset_manager

    if fake:
        cfg.headsets.registry = [f"{x}=Muse-SIM{x}" for x in SIM_BATTERY]
        label = label or "B"
    try:
        labels = [x for x, _ in cfg.headsets.entries()]
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if label is None and len(labels) == 1:
        label = labels[0]
    if label not in labels:
        print(f"--headset must be one of {labels or 'the [headsets] registry (empty)'}; "
              "e.g. registry = [\"A=Muse-41D2\"] in config/local.toml", file=sys.stderr)
        return 2
    manager = make_headset_manager(cfg, simulated=fake)
    if manager is None:
        return 4
    print(f"Connecting to headset {label} in-process (nc-engine must not be connected to it)...")
    manager.submit("select", label)
    return manager


def _battery(source) -> float | None:
    snapshot = getattr(source, "snapshot", None)
    if snapshot is None:
        return None
    snap = snapshot()
    h = next((h for h in snap.get("headsets", []) if h["label"] == snap.get("current")), None)
    return None if h is None else h["battery"]


class Voice:
    def __init__(self, enabled: bool):
        self.cmd = shutil.which("say") if enabled else None
        self._proc: subprocess.Popen | None = None

    def say(self, text: str) -> None:
        if not self.cmd:
            return
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = subprocess.Popen([self.cmd, text])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,16}", args.participant):
        print("--participant must be a short code such as P01 (letters, digits, - or _)", file=sys.stderr)
        return 2
    paths = args.config or [p for p in (ROOT / "config/installation.toml", ROOT / "config/local.toml") if p.exists()]
    try:
        cfg = load_config(paths)
    except (ConfigError, OSError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    t = cfg.targets
    yaws = [t.left_yaw_deg, t.middle_yaw_deg, t.right_yaw_deg]
    if args.yaw:
        yaws = [float(v) for v in args.yaw.split(",")]
        if len(yaws) != 3:
            print("--yaw needs three values: left,middle,right", file=sys.stderr)
            return 2
    layout = protocol.Layout(*yaws, top_pitch_deg=args.top_pitch, bottom_pitch_deg=args.bottom_pitch)
    if args.stand_s is None:
        args.stand_s = 20.0 if args.quick else 120.0
    steps = protocol.build(layout, stand_s=args.stand_s, quick=args.quick)

    kind = "muse_direct" if args.fake_headset else (
        args.source or (cfg.source.kind if cfg.source.kind in ("muse", "muse_direct") else "muse"))
    opened = open_lsl(cfg) if kind == "muse" else open_direct(cfg, args.headset, args.fake_headset)
    if isinstance(opened, int):
        return opened
    source = opened
    deadline = time.monotonic() + args.wait_s
    last_message = None
    while not source.connected():
        source.poll()
        message = getattr(source, "message", None)  # headset manager: connecting / pick it up
        if message and message != last_message:
            print(f"  {message}")
            last_message = message
        if time.monotonic() > deadline:
            print("No IMU data in time.", file=sys.stderr)
            source.close()
            return 3
        time.sleep(0.05)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = args.out / f"{args.participant}_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    imu = ImuWriter(folder / "imu.csv")
    events = EventWriter(folder / "events.csv")
    voice = Voice(not args.no_voice)
    t0 = source.now()
    meta = {
        "participant": args.participant,
        "started": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        # source clock (LSL local_clock, or time.monotonic for muse_direct), as in imu.csv / events.csv
        "clock_t0": t0,
        "time_time_t0": time.time(),
        "monotonic_t0": time.monotonic(),
        "source": kind,
        "headset": source.headset_id(),
        "simulated": args.fake_headset,
        "protocol": {"version": 1, "quick": args.quick, "stand_s": args.stand_s,
                     "prompt_every_s": protocol.PROMPT_EVERY_S},
        "layout": asdict(layout),
        "config_files": [str(p) for p in paths],
        "tool": "bench/imu/record.py",
    }
    if (battery := _battery(source)) is not None:
        meta["battery_start"] = battery
    write_meta(folder / "meta.json", meta)
    print(f"Recording to {folder} ({protocol.duration_s(steps) / 60:.1f} min). Ctrl-C stops.\n")

    idx = 0
    was_connected = True
    next_status = next_flush = 0.0
    aborted = False
    samples = []
    try:
        while idx < len(steps):
            for s in source.poll():
                imu.write(s)
                samples.append(s)
            now = source.now()
            while idx < len(steps) and now - t0 >= steps[idx].t:
                st = steps[idx]
                events.write(Event(now, st.kind, st.name, st.column, st.row, st.yaw_deg, st.pitch_deg, st.text))
                label = st.name.upper() if st.kind == "phase" else ">>"
                print(f"\r[{now - t0:6.1f} s] {label:9s} {st.text}{' ' * 20}")
                voice.say(st.text)
                idx += 1
            connected = source.connected()
            if connected != was_connected:
                events.write(Event(now, "status", "stream_back" if connected else "stream_lost"))
                print(f"\r[{now - t0:6.1f} s] stream {'back' if connected else 'LOST'}{' ' * 30}")
                was_connected = connected
            if now >= next_status and idx < len(steps):
                next_status = now + 0.5
                nxt = steps[idx].t - (now - t0)
                print(f"\r  gyro {source.rate_hz():5.1f} Hz   next in {nxt:4.1f} s ", end="", flush=True)
            if now >= next_flush:
                next_flush = now + 1.0
                imu.flush()
            time.sleep(0.01)
    except KeyboardInterrupt:
        aborted = True
        events.write(Event(source.now(), "phase", "end", text="aborted"))
        print("\nStopped.")
    finally:
        imu.close()
        events.close()
        source.close()

    rep = timing(samples)
    battery = _battery(source)
    if battery is not None:
        meta["battery_end"] = battery
    meta.update({
        "aborted": aborted,
        "duration_s": round(source.now() - t0, 1),
        "samples": imu.counts,
        "rate_hz": {k: round(v, 2) for k, v in rep.rate_hz.items()},
        "duplicate_timestamps": rep.duplicates,
        "gaps_over_100ms": rep.gaps,
    })
    write_meta(folder / "meta.json", meta)
    print(f"\nSaved {folder}")
    print(f"  samples {imu.counts}, rate {meta['rate_hz']} Hz, "
          f"duplicate timestamps {rep.duplicates}, gaps > 0.1 s {rep.gaps}")
    print(f"  compare backends:  python bench/imu/replay.py {folder}")
    return 1 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
