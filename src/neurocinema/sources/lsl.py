"""IMU samples from the Muse ACC + GYRO LSL streams published by muselsl.

Stream discovery runs on a background thread so the main loop never blocks.
When samples stop for `stale_after_s` the inlets are dropped and discovery
starts again; this covers muselsl being restarted (new outlets) as well as
streams published by something we do not manage.
"""

from __future__ import annotations

import logging
import threading

import pylsl

try:  # where LostError lives moved between pylsl versions
    from pylsl.util import LostError
except ImportError:  # pragma: no cover
    LostError = getattr(pylsl, "LostError", RuntimeError)

from ..config import MuseConfig
from ..samples import ImuSample
from .base import RateMeter

log = logging.getLogger(__name__)

_KINDS = {"ACC": "acc", "GYRO": "gyro"}

# A freshly opened inlet can take a second or two (clock sync) before the
# first sample arrives; do not mistake that for a dead stream.
FIRST_DATA_GRACE_S = 5.0


class LslImuSource:
    def __init__(self, cfg: MuseConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._pending: dict[str, pylsl.StreamInlet] | None = None
        self._inlets: dict[str, pylsl.StreamInlet] | None = None
        self._source_id = ""
        self._opened_at: float | None = None
        self._last_receive: float | None = None
        self._rate = RateMeter()
        self._want = threading.Event()
        self._want.set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._resolver, name="lsl-resolver", daemon=True)
        self._thread.start()

    # --- ImuSource --------------------------------------------------------------

    def now(self) -> float:
        return pylsl.local_clock()

    def poll(self) -> list[ImuSample]:
        now = self.now()
        with self._lock:
            if self._pending is not None:
                self._close_inlets()
                self._inlets, self._pending = self._pending, None
                self._opened_at = now
                self._last_receive = None
                self._rate.clear()
        if self._inlets is None:
            return []

        samples: list[ImuSample] = []
        for stype, inlet in self._inlets.items():
            kind = _KINDS[stype]
            try:
                data, stamps = inlet.pull_chunk(timeout=0.0)
            except LostError:
                log.warning("LSL %s stream lost", stype)
                self.reset()
                return []
            for values, ts in zip(data, stamps, strict=True):
                samples.append(ImuSample(ts, kind, values[0], values[1], values[2]))
                if kind == "gyro":
                    self._rate.add(now)
        if samples:
            self._last_receive = now
            samples.sort(key=lambda s: s.t)

        if self._last_receive is not None:
            silent, limit = now - self._last_receive, self.cfg.stale_after_s
        elif self._opened_at is not None:
            silent, limit = now - self._opened_at, max(self.cfg.stale_after_s, FIRST_DATA_GRACE_S)
        else:
            silent, limit = 0.0, 1.0
        if silent > limit:
            log.warning("no IMU samples for %.1f s from %s; looking for the stream again",
                        silent, self._source_id or "?")
            self.reset()
        return samples

    def connected(self) -> bool:
        return (
            self._inlets is not None
            and self._last_receive is not None
            and self.now() - self._last_receive <= self.cfg.stale_after_s
        )

    def last_receive(self) -> float | None:
        return self._last_receive

    def rate_hz(self) -> float:
        return self._rate.rate(self.now())

    def reset(self) -> None:
        with self._lock:
            self._pending = None
        self._close_inlets()
        self._inlets = None
        self._opened_at = None
        self._last_receive = None
        self._rate.clear()
        self._want.set()

    def close(self) -> None:
        self._stop.set()
        self._want.set()
        self._close_inlets()
        self._inlets = None

    # --- discovery ----------------------------------------------------------------

    def _close_inlets(self) -> None:
        if self._inlets:
            for inlet in self._inlets.values():
                try:
                    inlet.close_stream()
                except Exception:  # noqa: BLE001 - closing is best effort
                    pass

    def _resolver(self) -> None:
        while not self._stop.is_set():
            self._want.wait()
            if self._stop.is_set():
                return
            try:
                found = self._find()
            except Exception:  # noqa: BLE001 - keep the thread alive no matter what
                log.exception("LSL discovery failed")
                found = None
            if found is None:
                continue
            source_id, infos = found
            try:
                inlets = {
                    stype: pylsl.StreamInlet(
                        info,
                        max_buflen=10,
                        recover=False,
                        processing_flags=pylsl.proc_clocksync | pylsl.proc_monotonize,
                    )
                    for stype, info in infos.items()
                }
                for inlet in inlets.values():
                    inlet.open_stream(timeout=2.0)
            except Exception:  # noqa: BLE001
                log.exception("could not open LSL inlets for %s", source_id)
                continue
            log.info("receiving IMU from LSL source %s", source_id)
            with self._lock:
                self._pending = inlets
                self._source_id = source_id
                self._want.clear()

    def _find(self) -> tuple[str, dict[str, pylsl.StreamInfo]] | None:
        infos = pylsl.resolve_streams(wait_time=self.cfg.resolve_wait_s)
        by_source: dict[str, dict[str, pylsl.StreamInfo]] = {}
        for info in infos:
            if info.type() not in _KINDS or not info.name().startswith(self.cfg.stream_name):
                continue
            if self.cfg.source_id and info.source_id() != self.cfg.source_id:
                continue
            by_source.setdefault(info.source_id(), {})[info.type()] = info
        complete = {sid: s for sid, s in by_source.items() if set(s) == set(_KINDS)}
        if not complete:
            return None
        # Prefer the newest outlet if an old one is still being torn down.
        sid = max(complete, key=lambda k: complete[k]["GYRO"].created_at())
        return sid, complete[sid]
