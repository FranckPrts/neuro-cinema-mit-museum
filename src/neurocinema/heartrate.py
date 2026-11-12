"""Heart rate from the Muse 2 forehead PPG.

The headset only sends raw light levels (ambient, infrared, red at 64 Hz).
Every `update_every_s` the last `window_s` of one channel is band-passed and
autocorrelated; the beat period is the autocorrelation peak between
60/`max_bpm` and 60/`min_bpm` seconds. `quality` is the correlation at that
lag (0..1); below `min_quality` the estimate is not trusted.

Head movement wrecks forehead PPG. While the head moves (gyro range above
`motion_dps`) and for one window after, no new estimate is made and the last
good one is held, so its age grows. After `hold_s` without a good estimate
the heart rate is reported as unknown.

`worn` is the staff page's on-a-head check, from the same estimates: True once
a pulse is found, False after `NOT_WORN_AFTER` clear windows in a row
without one (on the stand, or not touching the skin), None while there is
not enough signal to tell. Head movement freezes it like the heart rate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import HeartRateConfig

CHANNELS = ("ambient", "infrared", "red")
# Estimates in a row that find no pulse before the headset counts as not worn
# (one estimate per update_every_s): one bad window on a head is not enough.
NOT_WORN_AFTER = 3


@dataclass(frozen=True)
class HeartReading:
    bpm: float | None  # None = unknown
    quality: float  # of the last estimate attempt, 0..1
    age_s: float | None  # since `bpm` was measured; None when unknown
    moving: bool  # estimates are paused by head movement
    worn: bool | None = None  # pulse found / clearly none / cannot tell yet


class HeartRate:
    def __init__(self, cfg: HeartRateConfig, rate_hz: float = 64.0):
        if cfg.channel not in CHANNELS:
            raise ValueError(f"heart_rate.channel must be one of {CHANNELS}, got {cfg.channel!r}")
        self.cfg = cfg
        self.rate_hz = rate_hz
        self._channel = CHANNELS.index(cfg.channel)
        self._t: deque[float] = deque(maxlen=int(cfg.window_s * rate_hz * 1.5))
        self._v: deque[float] = deque(maxlen=self._t.maxlen)
        self.reset()

    def reset(self) -> None:
        """Forget everything (headset changed or link lost)."""
        self._t.clear()
        self._v.clear()
        self._bpm: float | None = None
        self._bpm_at: float | None = None
        self._quality = 0.0
        self._last_motion: float | None = None
        self._next_update: float | None = None
        self._worn: bool | None = None
        self._misses = 0

    def add(self, t: float, ambient: float, infrared: float, red: float) -> None:
        self._t.append(t)
        self._v.append((ambient, infrared, red)[self._channel])

    def motion(self, now: float, gyro_range_dps: float | None) -> None:
        """Called every loop with the engine's recent gyro range (None = unknown)."""
        if gyro_range_dps is not None and gyro_range_dps > self.cfg.motion_dps:
            self._last_motion = now

    def moving(self, now: float) -> bool:
        return self._last_motion is not None and now - self._last_motion < self.cfg.window_s

    def update(self, now: float) -> HeartReading:
        if self._next_update is None or now >= self._next_update:
            self._next_update = now + self.cfg.update_every_s
            if not self.moving(now):
                self._estimate(now)
        if self._bpm_at is not None and now - self._bpm_at > self.cfg.hold_s:
            self._bpm = self._bpm_at = None
        age = None if self._bpm_at is None else max(0.0, now - self._bpm_at)
        return HeartReading(self._bpm, self._quality, age, self.moving(now), self._worn)

    def _estimate(self, now: float) -> None:
        if len(self._t) < 2:
            self._quality = 0.0
            return
        t = np.fromiter(self._t, float)
        keep = t >= t[-1] - self.cfg.window_s
        t, v = t[keep], np.fromiter(self._v, float)[keep]
        if now - t[-1] > 1.0:  # no PPG coming in: cannot tell
            self._quality, self._worn, self._misses = 0.0, None, 0
            return
        # Need (nearly) a full window without a gap in it.
        if t[-1] - t[0] < 0.9 * self.cfg.window_s or np.max(np.diff(t)) > 0.5:
            self._quality = 0.0
            return
        bpm, quality = estimate_bpm(v, self.rate_hz, self.cfg.min_bpm, self.cfg.max_bpm)
        self._quality = quality
        if bpm is not None and quality >= self.cfg.min_quality:
            self._bpm, self._bpm_at = bpm, now
            self._worn, self._misses = True, 0
        else:
            self._misses += 1
            if self._misses >= NOT_WORN_AFTER:
                self._worn = False


def estimate_bpm(values: np.ndarray, rate_hz: float, min_bpm: float, max_bpm: float
                 ) -> tuple[float | None, float]:
    """Pulse rate of one PPG channel: (bpm or None, quality 0..1).

    Band-pass (0.6-4 Hz) by FFT mask, then autocorrelation. The beat period
    is the shortest-lag autocorrelation peak within 85 % of the highest one
    in the [60/max_bpm, 60/min_bpm] s range (picking the highest outright
    often lands on two beats). Quality is the lag-corrected correlation there:
    about 0.9 for a clean pulse; noise alone rarely gets past 0.65.
    """
    n = len(values)
    if n < 16:
        return None, 0.0
    x = values - np.mean(values)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))  # zero-pad: no circular wrap
    spec = np.fft.rfft(x, nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / rate_hz)
    spec[(freqs < 0.6) | (freqs > 4.0)] = 0.0
    y = np.fft.irfft(spec, nfft)[:n]
    ac = np.fft.irfft(np.abs(np.fft.rfft(y, nfft)) ** 2, nfft)[:n]
    if ac[0] <= 0.0:
        return None, 0.0
    biased = ac / ac[0]
    lo = int(np.floor(rate_hz * 60.0 / max_bpm))
    hi = min(n - 2, int(np.ceil(rate_hz * 60.0 / min_bpm)))
    if hi - lo < 3:
        return None, 0.0
    seg = biased[lo:hi + 1]
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]]
    if not peaks:
        return None, 0.0
    best = max(seg[i] for i in peaks)
    if best <= 0.0:
        return None, 0.0
    i = next(i for i in peaks if seg[i] >= 0.85 * best)
    a, b, c = seg[i - 1:i + 2]
    denom = a - 2 * b + c
    offset = 0.5 * (a - c) / denom if denom != 0 else 0.0  # parabolic interpolation
    lag = lo + i + offset
    quality = float(np.clip(seg[i] * n / (n - (lo + i)), 0.0, 1.0))
    return float(60.0 * rate_hz / lag), quality
