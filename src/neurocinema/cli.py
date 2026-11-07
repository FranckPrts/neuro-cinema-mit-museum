"""`nc-engine`: run the head-gesture engine."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import signal
import sys
from pathlib import Path

from .app import App
from .config import Config, ConfigError, load_config
from .osc import OscCommandServer, OscOutput

log = logging.getLogger("neurocinema")

DEFAULT_CONFIG = Path("config/installation.toml")
LOCAL_CONFIG = Path("config/local.toml")


def setup_logging(cfg: Config, level_override: str | None) -> None:
    level = getattr(logging, (level_override or cfg.logging.level).upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)
    if cfg.logging.file:
        path = Path(cfg.logging.file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=5)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
            root.addHandler(fh)
        except OSError as e:
            log.warning("cannot write log file %s: %s", path, e)


# Starting battery of the simulated headsets (--sim-headsets). A starts low so
# the "swap soon" state shows up within a few minutes.
SIM_BATTERY = {"A": 22.0, "B": 96.0, "C": 71.0, "D": 100.0}


def make_headset_manager(cfg: Config, simulated: bool):
    from .ble import BleWorker
    from .headsets import HeadsetManager

    if simulated:
        from .ble import FakeDriver

        driver = FakeDriver({f"Muse-SIM{label}": pct for label, pct in SIM_BATTERY.items()})
    else:
        if not cfg.headsets.registry:
            log.error("source.kind = 'muse_direct' needs headsets.registry, e.g. "
                      "registry = [\"A=Muse-41D2\"] in config/local.toml (names from `muselsl list`)")
            return None
        try:
            from .ble import MuselslDriver

            driver = MuselslDriver()
        except ImportError as e:
            log.error("headset support is not installed (%s); run: pip install -e '.[muse]'", e)
            return None
    log.info("headsets: %s", ", ".join(cfg.headsets.registry))
    return HeadsetManager(cfg.headsets, BleWorker(driver))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nc-engine", description=__doc__)
    p.add_argument("-c", "--config", type=Path, action="append",
                   help=f"TOML config file (repeatable; default {DEFAULT_CONFIG} then {LOCAL_CONFIG} if present)")
    p.add_argument("--sim", action="store_true", help="use the virtual head instead of the headset")
    p.add_argument("--no-keyboard", action="store_true", help="simulator without keyboard control")
    p.add_argument("--unmanaged", action="store_true",
                   help="do not launch muselsl; read LSL streams published by something else")
    p.add_argument("--sim-headsets", action="store_true",
                   help="in-process headset manager with four simulated headsets (try the status page)")
    p.add_argument("--log-level", help="DEBUG, INFO, WARNING ...")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = args.config or [p for p in (DEFAULT_CONFIG, LOCAL_CONFIG) if p.exists()]
    try:
        cfg = load_config(paths)
    except (ConfigError, OSError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if args.sim:
        cfg.source.kind = "sim"
    if args.unmanaged:
        cfg.muse.manage_stream = False
    if args.sim_headsets:
        cfg.source.kind = "muse_direct"
        cfg.headsets.registry = [f"{label}=Muse-SIM{label}" for label in SIM_BATTERY]
    setup_logging(cfg, args.log_level)
    log.info("config: %s", ", ".join(map(str, paths)) or "built-in defaults")

    out = OscOutput(cfg.osc.destinations)
    log.info("sending OSC to %s", ", ".join(cfg.osc.destinations))
    try:
        commands = OscCommandServer(cfg.osc.listen_host, cfg.osc.listen_port, cfg.osc.prefix)
    except OSError as e:
        log.error("cannot listen on %s:%d (%s); is another engine running?",
                  cfg.osc.listen_host, cfg.osc.listen_port, e)
        return 3

    closers = []
    if cfg.source.kind == "sim":
        from .sources.sim import SimSource

        source = SimSource()
    elif cfg.source.kind == "muse_direct":
        source = make_headset_manager(cfg, simulated=args.sim_headsets)
        if source is None:
            return 4
        if cfg.status_page.enabled:
            from .statuspage import StatusPage

            try:
                page = StatusPage(source, cfg.status_page.host, cfg.status_page.port)
            except OSError as e:
                log.error("cannot serve the status page on %s:%d (%s)",
                          cfg.status_page.host, cfg.status_page.port, e)
                return 3
            page.start()
            closers.append(page.close)
    else:
        try:
            from .sources.lsl import LslImuSource
        except ImportError as e:
            log.error("headset support is not installed (%s); run: pip install -e '.[muse]'", e)
            return 4
        source = LslImuSource(cfg.muse)

    app = App(cfg, source, out, commands=commands)

    if cfg.source.kind == "sim" and not args.no_keyboard and sys.stdin.isatty():
        from .sim.keyboard import SimKeyboard

        kb = SimKeyboard(source, cfg.targets, app.stop_event.set)
        app.hooks.append(kb)
        closers.append(kb.close)

    if cfg.source.kind == "muse" and cfg.muse.manage_stream:
        from .supervisor import MuseStreamSupervisor

        app.supervisor = MuseStreamSupervisor(cfg.muse, on_restart=source.reset, on_error=app.on_stream_error)

    def stop(signum, _frame):
        log.info("received signal %d", signum)
        app.stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        app.run()
    finally:
        for close in closers:
            close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
