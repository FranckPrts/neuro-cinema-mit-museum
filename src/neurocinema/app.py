"""The main loop: source -> engine -> OSC, plus status, commands and supervision."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from .config import Config
from .engine import Engine, Output
from .osc import OscCommandServer
from .sources.base import ImuSource
from .supervisor import MuseStreamSupervisor

log = logging.getLogger(__name__)

LOOP_HZ = 200.0


class App:
    def __init__(
        self,
        cfg: Config,
        source: ImuSource,
        out: Output,
        commands: OscCommandServer | None = None,
        supervisor: MuseStreamSupervisor | None = None,
        hooks: list[Callable[[float], None]] | None = None,
    ):
        self.cfg = cfg
        self.source = source
        self.out = out
        self.commands = commands
        self.supervisor = supervisor
        self.hooks = hooks or []
        self.engine = Engine(cfg, out)
        self.p = cfg.osc.prefix
        self.stop_event = threading.Event()
        self._started = time.monotonic()
        self._next_status = 0.0
        self._next_heartbeat = 0.0

    def run(self) -> None:
        log.info("engine running (source=%s)", self.cfg.source.kind)
        self.out.send(f"{self.p}/event", "engine_started")
        if self.commands:
            self.commands.start()
        if self.supervisor:
            self.supervisor.start()
        period = 1.0 / LOOP_HZ
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                self.step()
                time.sleep(max(0.0, period - (time.monotonic() - started)))
        finally:
            self.shutdown()

    def step(self) -> None:
        source = self.source
        for sample in source.poll():
            self.engine.process(sample)
        now = source.now()
        connected = source.connected()
        if connected != self.engine.connected:
            self.engine.on_connection(connected, now)
        if self.supervisor:
            self.supervisor.tick(connected)
        if self.commands:
            for name, args in self.commands.drain():
                self.handle_command(name, args, now)
        for hook in self.hooks:
            hook(now)
        self.engine.tick(now)
        self._emit_status(now)

    def handle_command(self, name: str, args: tuple, now: float) -> None:
        log.info("command %s %s", name, " ".join(map(str, args)))
        if name == "restart_stream":
            if self.supervisor:
                self.supervisor.restart("command")
            else:
                self.source.reset()
        else:
            self.engine.command(name, args, now)

    def _emit_status(self, now: float) -> None:
        mono = time.monotonic()
        if mono >= self._next_heartbeat:
            self._next_heartbeat = mono + 1.0
            self.out.send(f"{self.p}/heartbeat", round(mono - self._started, 1))
        if mono >= self._next_status:
            self._next_status = mono + 1.0 / self.cfg.osc.status_rate_hz
            last = self.source.last_receive()
            age_ms = -1 if last is None else int((now - last) * 1000)
            motion = self.engine.fusion.stillness.gyro_range()
            self.out.send(
                f"{self.p}/status",
                int(self.source.connected()),
                age_ms,
                round(self.source.rate_hz(), 1),
                -1.0 if motion is None else round(motion, 2),
            )
            if self.supervisor:
                self.out.send(
                    f"{self.p}/status/stream",
                    self.supervisor.state.value,
                    self.supervisor.failures,
                )

    def on_stream_error(self, code: str, message: str) -> None:
        if code == "ok":
            log.info("stream healthy again")
        else:
            log.error("%s: %s", code, message)
        self.out.send(f"{self.p}/status/error", code, message)

    def shutdown(self) -> None:
        log.info("shutting down")
        if self.supervisor:
            self.supervisor.stop()
        if self.commands:
            self.commands.close()
        self.source.close()
        self.out.send(f"{self.p}/event", "engine_stopped")
