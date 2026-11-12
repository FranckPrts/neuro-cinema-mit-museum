"""The main loop: source -> engine -> OSC, plus the camera relay, status, commands and supervision."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

from .camera import CameraRelay
from .config import Config
from .engine import Engine, Output
from .heartrate import HeartRate, HeartReading
from .osc import OscCommandServer
from .setups import SelectionSetup
from .sources.base import ImuSource
from .supervisor import MuseStreamSupervisor

log = logging.getLogger(__name__)

LOOP_HZ = 200.0

# Engine commands the status page may send (see Engine.command).
PAGE_COMMANDS = ("recalibrate", "idle", "reset")


class PageLink:
    """Between the status page's thread and the main loop: the engine's status
    goes out (copied a few times a second), staff commands come in."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status: dict = {}
        self._commands: queue.SimpleQueue[str] = queue.SimpleQueue()

    def publish(self, status: dict) -> None:
        with self._lock:
            self._status = status

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def submit(self, command: str) -> None:
        if command not in PAGE_COMMANDS:
            raise ValueError(f"unknown engine command {command!r}")
        self._commands.put(command)

    def drain(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self._commands.get_nowait())
            except queue.Empty:
                return out


class App:
    def __init__(
        self,
        cfg: Config,
        source: ImuSource,
        out: Output,
        commands: OscCommandServer | None = None,
        supervisor: MuseStreamSupervisor | None = None,
        hooks: list[Callable[[float], None]] | None = None,
        setups: dict[str, SelectionSetup] | None = None,
        camera: CameraRelay | None = None,
    ):
        self.cfg = cfg
        self.source = source
        self.out = out
        self.commands = commands
        self.supervisor = supervisor
        self.camera = camera
        self.hooks = hooks or []
        self.engine = Engine(cfg, out, setups)
        self.page = PageLink()
        self.p = cfg.osc.prefix
        self.stop_event = threading.Event()
        self._started = time.monotonic()
        self._next_status = 0.0
        self._next_heartbeat = 0.0
        # Heart rate from the headset PPG (muse_direct only: the source has poll_ppg).
        self.heart: HeartRate | None = None
        if cfg.heart_rate.enabled and hasattr(source, "poll_ppg"):
            self.heart = HeartRate(cfg.heart_rate)
        self._heart_headset: str | None = None
        self._heart_reading: HeartReading | None = None

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
            self.engine.on_connection(connected, now, headset=source.headset_id() if connected else None)
        if self.heart:
            self._step_heart(now, connected)
        if self.supervisor:
            self.supervisor.tick(connected)
        if self.commands:
            for name, args in self.commands.drain():
                self.handle_command(name, args, now)
        page_commands = self.page.drain()
        for name in page_commands:
            log.info("status page: %s", name)
            self.handle_command(name, (), now)
        if self.camera:
            self.camera.poll(self.engine.phase.value)
        for hook in self.hooks:
            hook(now)
        self.engine.tick(now)
        self._emit_status(now)
        if page_commands:
            self.page.publish(self._page_status(now))  # show the result at once

    def _page_status(self, now: float) -> dict:
        """Engine status for the staff page, plus the PPG on-a-head check.

        `pulse`: "worn", "not_worn", "waiting" (not enough signal yet), or None
        when there is no PPG. Only this word goes to the page, never the heart rate.
        """
        status = self.engine.status(now)
        pulse = None
        if self.heart is not None:
            worn = self._heart_reading.worn if self._heart_reading is not None else None
            pulse = "waiting" if worn is None else ("worn" if worn else "not_worn")
        status["pulse"] = pulse
        return status

    def _step_heart(self, now: float, connected: bool) -> None:
        headset = self.source.headset_id() if connected else None
        if headset != self._heart_headset:
            self._heart_headset = headset
            self.heart.reset()  # never mix two heads (or a gap) in one window
        for r in self.source.poll_ppg():
            self.heart.add(r.t, r.ambient, r.infrared, r.red)
        self.heart.motion(now, self.engine.fusion.stillness.gyro_range())
        self._heart_reading = self.heart.update(now)

    def handle_command(self, name: str, args: tuple, now: float) -> None:
        log.info("command %s %s", name, " ".join(map(str, args)))
        if name == "restart_stream":
            if self.supervisor:
                self.supervisor.restart("command")
            else:
                self.source.reset()
        else:
            self.engine.command(name, args, now)
            if name == "snapshot" and self.camera:
                self.camera.snapshot()

    def _emit_status(self, now: float) -> None:
        mono = time.monotonic()
        if mono >= self._next_heartbeat:
            self._next_heartbeat = mono + 1.0
            self.out.send(f"{self.p}/heartbeat", round(mono - self._started, 1))
        if mono >= self._next_status:
            self._next_status = mono + 1.0 / self.cfg.osc.status_rate_hz
            self.page.publish(self._page_status(now))
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
            if self._heart_reading is not None:
                h = self._heart_reading
                self.out.send(
                    f"{self.p}/heart",
                    -1.0 if h.bpm is None else round(h.bpm, 1),
                    round(h.quality, 2),
                    -1.0 if h.age_s is None else round(h.age_s, 1),
                    int(h.moving),
                )
            snapshot = getattr(self.source, "snapshot", None)
            if snapshot is not None:  # headset manager (muse_direct)
                self._emit_headset(snapshot())

    def _emit_headset(self, snap: dict) -> None:
        """/nc/headset label battery minutes_left phase alert (-1 = unknown)."""
        label = snap.get("current") or ""
        h = next((h for h in snap.get("headsets", []) if h["label"] == label), None)
        battery = h["battery"] if h and h["battery"] is not None else -1
        left = h["minutes_left"] if h and h["minutes_left"] is not None else -1
        self.out.send(
            f"{self.p}/headset",
            label,
            float(battery),
            float(left),
            snap.get("phase", ""),
            snap.get("alert", ""),
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
        if self.camera:
            self.camera.close()
        self.source.close()
        self.out.send(f"{self.p}/event", "engine_stopped")
