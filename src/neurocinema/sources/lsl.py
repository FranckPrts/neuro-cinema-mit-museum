"""IMU samples from the Muse ACC + GYRO LSL streams published by muselsl.

Stream discovery runs on a background thread so the main loop never blocks.
When samples stop for `stale_after_s` the inlets are dropped and discovery
starts again; this covers muselsl being restarted (new outlets) as well as
streams published by something we do not manage.

muselsl stamps the 3 samples of each Muse IMU packet with the same arrival
time. `Retimer` spreads them back out (as `ble.MuselslDriver` does for
`muse_direct`); otherwise `Fusion` drops 2 of every 3 samples as duplicates.
"""

from __future__ import annotations

import logging
import threading

import pylsl

try:  # where LostError lives moved between pylsl versions
    from pylsl.util import LostError
except ImportError:  # pragma: no cover
    LostError = getattr(pylsl, "LostError", RuntimeError)

from ..ble import IMU_RATE_HZ
from ..config import MuseConfig
from ..samples import ImuSample
from .base import RateMeter

log = logging.getLogger(__name__)

_KINDS = {"ACC": "acc", "GYRO": "gyro"}

# A freshly opened inlet can take a second or two (clock sync) before the
# first sample arrives; do not mistake that for a dead stream.
FIRST_DATA_GRACE_S = 5.0


class Retimer:
    """Makes one stream's timestamps strictly increasing.

    A run of samples sharing a timestamp T (one muselsl packet) is spread back
    over 1 / rate steps so its last sample lands on T. If that would reach back
    to or before the previous sample (packets that arrived in a burst), the run
    is spread evenly over the time since the previous sample instead. Samples
    with distinct timestamps pass through unchanged.

    A run's length is taken as the larger of what this pull holds and the
    length of the previous complete run, so a packet split across two pulls
    gets the same times as if it had arrived whole.
    """

    def __init__(self, rate_hz: float = IMU_RATE_HZ):
        self.dt = 1.0 / rate_hz
        self.clear()

    def clear(self) -> None:
        self._last: float | None = None  # last time handed out
        self._run_t: float | None = None  # raw stamp of the current run
        self._run_k = 0  # samples of the current run handed out so far
        self._run_start = 0.0
        self._run_step = 0.0
        self._size = 1  # length of the last complete run

    def __call__(self, stamps: list[float]) -> list[float]:
        out: list[float] = []
        for i, ts in enumerate(stamps):
            if ts != self._run_t:
                self._start_run(ts, stamps, i)
            t = self._run_start + self._run_k * self._run_step
            self._run_k += 1
            self._last = t
            out.append(t)
        return out

    def _start_run(self, ts: float, stamps: list[float], i: int) -> None:
        if self._run_t is not None:
            self._size = self._run_k
        here = 1
        while i + here < len(stamps) and stamps[i + here] == ts:
            here += 1
        n = max(self._size, here)
        start, step = ts - (n - 1) * self.dt, self.dt
        last = self._last
        if last is not None and start <= last:
            if ts > last:  # burst: spread over the time since the previous sample
                step = (ts - last) / n
                start = last + step
            else:  # stamp went back (or the previous run overshot): keep increasing
                start = last + self.dt
        self._run_t, self._run_k = ts, 0
        self._run_start, self._run_step = start, step


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
        self._retimers = {stype: Retimer() for stype in _KINDS}
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
                self._clear_retimers()
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
            for values, ts in zip(data, self._retimers[stype](stamps), strict=True):
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
        self._clear_retimers()
        self._want.set()

    def close(self) -> None:
        self._stop.set()
        self._want.set()
        self._close_inlets()
        self._inlets = None

    # --- discovery ----------------------------------------------------------------

    def _clear_retimers(self) -> None:
        for retimer in self._retimers.values():
            retimer.clear()

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
