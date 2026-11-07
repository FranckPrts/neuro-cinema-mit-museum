"""Headset manager: which headset is in use, its battery, and staff swaps.

It is also the engine's IMU source (`source.kind = "muse_direct"`): samples
reach the engine only from the headset staff chose and confirmed.

    NO_HEADSET ──staff tap──► CONNECTING ──connected──► CONFIRMING ──lifted──► ACTIVE
        ▲                        │ failed n times          │ timeout            │ no data
        └────────────────────────┴─────────────────────────┘                    ▼
                                                                         RECONNECTING
                                                         (same headset, until data returns
                                                          or staff choose another one)

Staff confirm a swap by tapping the headset's sticker label on the status
page and lifting that headset: the engine only uses it once it sees it move,
so the wrong headset can never be locked in. Headsets outside the registry
are never connected.

Status for the page, kept to three states: `ok`, `swap_soon` (battery or
estimated time left is low) and `offline` (no headset in use for a while).
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from .ble import Seen
from .config import HeadsetsConfig
from .samples import ImuSample
from .sources.base import RateMeter

log = logging.getLogger(__name__)

BATTERY_WINDOW_S = 20 * 60.0  # drain-rate estimate uses the last 20 minutes...
BATTERY_MIN_SPAN_S = 5 * 60.0  # ...once it covers at least 5


class Phase(StrEnum):
    NO_HEADSET = "no_headset"
    CONNECTING = "connecting"
    CONFIRMING = "confirming"
    ACTIVE = "active"
    RECONNECTING = "reconnecting"


class Alert(StrEnum):
    OK = "ok"
    SWAP_SOON = "swap_soon"
    OFFLINE = "offline"


@dataclass
class Headset:
    label: str
    name: str
    address: str | None = None
    last_seen: float | None = None
    battery: float | None = None
    battery_at: float | None = None
    history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=400))

    def record_battery(self, percent: float, t: float) -> None:
        self.battery, self.battery_at = percent, t
        self.history.append((t, percent))
        while self.history and self.history[0][0] < t - BATTERY_WINDOW_S:
            self.history.popleft()

    def minutes_left(self) -> float | None:
        """Linear fit of recent battery readings; None until there is enough."""
        if len(self.history) < 3 or self.history[-1][0] - self.history[0][0] < BATTERY_MIN_SPAN_S:
            return None
        ts = [t for t, _ in self.history]
        ps = [p for _, p in self.history]
        mt, mp = sum(ts) / len(ts), sum(ps) / len(ps)
        var = sum((t - mt) ** 2 for t in ts)
        if var == 0:
            return None
        slope = sum((t - mt) * (p - mp) for t, p in zip(ts, ps, strict=True)) / var  # % per s
        if slope >= 0:
            return None
        return max(0.0, ps[-1] / -slope / 60.0)


class HeadsetManager:
    """Drives a `ble.BleWorker`; implements the engine's `ImuSource`."""

    def __init__(self, cfg: HeadsetsConfig, worker, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.worker = worker
        self.clock = clock
        self.headsets = {label: Headset(label, name) for label, name in cfg.entries()}
        self._by_name = {h.name: h for h in self.headsets.values()}
        self.unknown: dict[str, float] = {}  # unregistered Muses seen: name -> last seen
        now = clock()
        self.phase = Phase.NO_HEADSET
        self.phase_since = now
        self.current: str | None = None  # label chosen / in use
        self.auto_adopted = False
        self.message = "Choose a headset to start."
        self._attempts = 0
        self._next_scan = now
        self._next_connect = now
        self._confirm_deadline = 0.0
        self._last_receive: float | None = None
        self._rate = RateMeter()
        self._commands: queue.Queue[tuple] = queue.Queue()
        self._snapshot_lock = threading.Lock()
        self._snapshot: dict = {}
        self._update_snapshot(now)

    # --- staff commands (any thread) -------------------------------------------------------

    def submit(self, command: str, arg: str | None = None) -> None:
        """"select" <label>, "release", "reconnect"."""
        self._commands.put((command, arg))

    def snapshot(self) -> dict:
        with self._snapshot_lock:
            return dict(self._snapshot)

    # --- ImuSource ---------------------------------------------------------------------------

    def now(self) -> float:
        return self.clock()

    def poll(self) -> list[ImuSample]:
        now = self.clock()
        while True:
            try:
                command, arg = self._commands.get_nowait()
            except queue.Empty:
                break
            self._command(command, arg, now)
        for event in self.worker.events():
            self._event(event, now)
        samples = self._readings(now)
        self._timers(now)
        self._update_snapshot(now)
        return samples

    def connected(self) -> bool:
        return (
            self.phase == Phase.ACTIVE
            and self._last_receive is not None
            and self.clock() - self._last_receive <= self.cfg.stale_after_s
        )

    def last_receive(self) -> float | None:
        return self._last_receive

    def rate_hz(self) -> float:
        return self._rate.rate(self.clock())

    def reset(self) -> None:
        """`/nc/cmd/restart_stream`: reconnect the current headset."""
        self.submit("reconnect")

    def close(self) -> None:
        self.worker.close()

    # --- internals -----------------------------------------------------------------------------

    def _set_phase(self, phase: Phase, now: float, message: str) -> None:
        if phase != self.phase:
            log.info("headset %s -> %s (%s)", self.current or "-", phase.value, message)
            self.phase, self.phase_since = phase, now
        self.message = message
        if phase in (Phase.NO_HEADSET, Phase.CONNECTING, Phase.CONFIRMING):
            self._last_receive = None
            self._rate.clear()

    def _command(self, command: str, arg: str | None, now: float) -> None:
        log.info("staff command: %s %s", command, arg or "")
        if command == "select":
            if arg not in self.headsets:
                log.warning("unknown headset label %r", arg)
                return
            self._start_connect(arg, now, auto=False)
        elif command == "release":
            self.worker.request_disconnect()
            self.current = None
            self._set_phase(Phase.NO_HEADSET, now, "Headset released. Choose a headset to start.")
        elif command == "reconnect":
            if self.current is None:
                return
            self.worker.request_disconnect()
            if self.phase == Phase.ACTIVE or self.phase == Phase.RECONNECTING:
                self._set_phase(Phase.RECONNECTING, now, f"Reconnecting to headset {self.current}…")
                self._next_connect = now
            else:
                self._start_connect(self.current, now, auto=self.auto_adopted)

    def _start_connect(self, label: str, now: float, auto: bool) -> None:
        h = self.headsets[label]
        self.worker.request_disconnect()
        self.current = label
        self.auto_adopted = auto
        self._attempts = 1
        self.worker.request_connect(label, h.name, h.address)
        self._set_phase(Phase.CONNECTING, now, f"Connecting to headset {label}…")

    def _event(self, event: tuple, now: float) -> None:
        kind = event[0]
        if kind == "scan":
            seen: list[Seen] = event[1]
            for s in seen:
                h = self._by_name.get(s.name)
                if h is not None:
                    h.address, h.last_seen = s.address, now
                else:
                    self.unknown[s.name] = now
        elif kind == "battery":
            _, label, percent, t = event
            if label in self.headsets:
                self.headsets[label].record_battery(percent, t)
        elif kind == "connected":
            label = event[1]
            if label != self.current:
                self.worker.request_disconnect()  # a stale request finished late
                return
            if self.phase == Phase.RECONNECTING or self.auto_adopted:
                self._set_phase(Phase.ACTIVE, now, f"Headset {label} connected.")
            else:
                self._confirm_deadline = now + self.cfg.confirm_timeout_s
                self._set_phase(Phase.CONFIRMING, now, f"Pick up headset {label} to confirm.")
        elif kind == "connect_failed":
            _, label, reason = event
            if label != self.current:
                return
            if self.phase == Phase.CONNECTING:
                if self._attempts < self.cfg.connect_attempts:
                    self._attempts += 1
                    self.worker.request_connect(label, self.headsets[label].name, None)
                else:
                    self.current = None
                    self._set_phase(Phase.NO_HEADSET, now,
                                    f"Could not reach headset {label}. Is it switched on and charged?")
            elif self.phase == Phase.RECONNECTING:
                self._next_connect = now + self.cfg.reconnect_interval_s
                self.message = f"Headset {label} lost. Retrying… ({reason})"
        elif kind == "disconnected":
            label = event[1]
            if label == self.current and self.phase == Phase.ACTIVE:
                self._lost(now)

    def _readings(self, now: float) -> list[ImuSample]:
        out: list[ImuSample] = []
        for r in self.worker.readings():
            if r.label != self.current:
                continue
            if self.phase == Phase.CONFIRMING:
                if r.kind == "gyro" and math.sqrt(r.x * r.x + r.y * r.y + r.z * r.z) >= self.cfg.confirm_dps:
                    self._set_phase(Phase.ACTIVE, now, f"Headset {r.label} in use.")
                continue
            if self.phase in (Phase.ACTIVE, Phase.RECONNECTING):
                if self.phase == Phase.RECONNECTING:
                    self._set_phase(Phase.ACTIVE, now, f"Headset {r.label} back.")
                out.append(ImuSample(r.t, r.kind, r.x, r.y, r.z))
                if r.kind == "gyro":
                    self._rate.add(now)
        if out:
            self._last_receive = now
            out.sort(key=lambda s: s.t)
        return out

    def _lost(self, now: float) -> None:
        self.worker.request_disconnect()
        self._next_connect = now + 0.5
        self._set_phase(Phase.RECONNECTING, now, f"Headset {self.current} lost. Reconnecting…")

    def _timers(self, now: float) -> None:
        c = self.cfg
        if self.phase == Phase.ACTIVE:
            ref = self._last_receive if self._last_receive is not None else self.phase_since
            if now - ref > c.stale_after_s:
                self._lost(now)
        elif self.phase == Phase.CONFIRMING and now > self._confirm_deadline:
            label = self.current
            self.worker.request_disconnect()
            self.current = None
            self._set_phase(Phase.NO_HEADSET, now,
                            f"Headset {label} was not picked up, so it was not used. Choose again.")
        elif self.phase == Phase.RECONNECTING and now >= self._next_connect and not self.worker.busy:
            h = self.headsets[self.current]
            self._next_connect = now + c.reconnect_interval_s
            self.worker.request_connect(h.label, h.name, None)  # rescan: the address may change
        elif self.phase == Phase.NO_HEADSET:
            if now >= self._next_scan and not self.worker.busy:
                self._next_scan = now + c.scan_interval_s
                self.worker.request_scan(c.scan_timeout_s)
            if c.auto_adopt_after_s > 0 and now - self.phase_since >= c.auto_adopt_after_s:
                fresh = [h for h in self.headsets.values()
                         if h.last_seen is not None and now - h.last_seen <= 2 * c.scan_interval_s]
                if len(fresh) == 1:
                    log.warning("nobody chose a headset for %.0f s; using the only one in range: %s",
                                now - self.phase_since, fresh[0].label)
                    self._start_connect(fresh[0].label, now, auto=True)

    def alert(self, now: float) -> Alert:
        if self.phase != Phase.ACTIVE:
            return Alert.OFFLINE if now - self.phase_since >= self.cfg.offline_after_s else Alert.OK
        h = self.headsets[self.current]
        left = h.minutes_left()
        if (h.battery is not None and h.battery < self.cfg.swap_soon_percent) or (
                left is not None and left < self.cfg.swap_soon_minutes):
            return Alert.SWAP_SOON
        return Alert.OK

    def _headline(self, now: float, alert: Alert) -> str:
        h = self.headsets.get(self.current) if self.current else None
        if alert == Alert.OFFLINE:
            return "Offline: no headset in use"
        if self.phase == Phase.ACTIVE and h is not None:
            battery = "battery unknown" if h.battery is None else f"{h.battery:.0f} %"
            left = h.minutes_left()
            extra = "" if left is None else f", about {_duration(left)} left"
            if alert == Alert.SWAP_SOON:
                return f"Swap headset {h.label} soon ({battery}{extra})"
            return f"All good: headset {h.label}, {battery}{extra}"
        return self.message

    def _update_snapshot(self, now: float) -> None:
        alert = self.alert(now)
        headsets = []
        for h in self.headsets.values():
            headsets.append({
                "label": h.label,
                "name": h.name,
                "in_use": h.label == self.current and self.phase == Phase.ACTIVE,
                "chosen": h.label == self.current,
                "seen_s_ago": None if h.last_seen is None else round(now - h.last_seen),
                "battery": None if h.battery is None else round(h.battery),
                "battery_s_ago": None if h.battery_at is None else round(now - h.battery_at),
                "minutes_left": None if h.minutes_left() is None else round(h.minutes_left()),
            })
        snap = {
            "alert": alert.value,
            "headline": self._headline(now, alert),
            "message": self.message,
            "phase": self.phase.value,
            "phase_s": round(now - self.phase_since),
            "current": self.current,
            "auto_adopted": self.auto_adopted,
            "receiving": self.connected(),
            "rate_hz": round(self.rate_hz(), 1),
            "headsets": headsets,
            "unknown": sorted(n for n, t in self.unknown.items() if now - t <= 5 * self.cfg.scan_interval_s),
        }
        with self._snapshot_lock:
            self._snapshot = snap


def _duration(minutes: float) -> str:
    if minutes >= 90:
        return f"{minutes / 60:.1f} h"
    return f"{minutes:.0f} min"
