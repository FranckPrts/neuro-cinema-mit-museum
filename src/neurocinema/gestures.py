"""Nod and shake detection from angular velocity.

A nod is pitch velocity going down-then-up (two lobes); a shake is yaw velocity
alternating left/right (three or more lobes). A lobe is a stretch where the
rate stays above `exit_dps` with one sign, after first crossing `enter_dps`.
A gesture fires when enough alternating lobes, each peaking above `peak_dps`
and separated by at most `max_gap_s`, have accumulated, and the motion is
dominated by this axis rather than the other one. A lobe longer than
`max_lobe_s` is a deliberate turn and breaks the sequence. `first_sign` can
require the first lobe's direction (a nod starts downwards).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import OscillationConfig


@dataclass(frozen=True)
class Gesture:
    kind: str  # "nod" | "shake"
    onset: float  # start of the first lobe
    t: float  # when it was recognised
    lobes: int
    peak_dps: float


@dataclass
class _Lobe:
    sign: int
    start: float
    end: float
    peak: float
    primary: float  # integral of |primary rate|, degrees
    other: float  # integral of |other rate|, degrees
    too_long: bool = False


class OscillationDetector:
    def __init__(self, kind: str, cfg: OscillationConfig):
        self.kind = kind
        self.cfg = cfg
        self._last_emit: float | None = None
        self.reset()

    def reset(self) -> None:
        self._t: float | None = None
        self._lobe: _Lobe | None = None
        self._seq: list[_Lobe] = []
        self._emitted = False

    @property
    def busy(self) -> bool:
        """True while a possible gesture is in progress."""
        return self._lobe is not None or bool(self._seq)

    def update(self, t: float, primary: float, other: float) -> Gesture | None:
        dt = 0.0 if self._t is None else min(max(t - self._t, 0.0), 0.1)
        self._t = t
        cfg = self.cfg

        if self._lobe is None and self._seq and t - self._seq[-1].end > cfg.max_gap_s:
            self._clear_sequence()

        mag = abs(primary)
        sign = 1 if primary > 0 else -1
        gesture = None

        lobe = self._lobe
        if lobe is not None:
            if sign == lobe.sign and mag > cfg.exit_dps:
                lobe.peak = max(lobe.peak, mag)
                lobe.primary += mag * dt
                lobe.other += abs(other) * dt
                lobe.end = t
                if t - lobe.start > cfg.max_lobe_s:
                    lobe.too_long = True
                return None
            gesture = self._close(lobe, t)
            self._lobe = None

        if mag > cfg.enter_dps:
            self._lobe = _Lobe(sign, t, t, mag, mag * dt, abs(other) * dt)
        return gesture

    def _clear_sequence(self) -> None:
        self._seq = []
        self._emitted = False

    def _close(self, lobe: _Lobe, t: float) -> Gesture | None:
        cfg = self.cfg
        lobe.end = t
        if lobe.too_long or lobe.peak < cfg.peak_dps:
            self._clear_sequence()
            return None
        if self._seq:
            prev = self._seq[-1]
            if prev.sign == lobe.sign or lobe.start - prev.end > cfg.max_gap_s:
                self._clear_sequence()
        if not self._seq and cfg.first_sign and lobe.sign != cfg.first_sign:
            return None  # cannot start a gesture in this direction
        self._seq.append(lobe)

        if self._emitted or len(self._seq) < cfg.min_lobes:
            return None
        primary = sum(l.primary for l in self._seq)
        other = sum(l.other for l in self._seq)
        if primary < cfg.dominance * other:
            return None
        if self._last_emit is not None and t - self._last_emit < cfg.cooldown_s:
            return None
        self._emitted = True
        self._last_emit = t
        return Gesture(
            kind=self.kind,
            onset=self._seq[0].start,
            t=t,
            lobes=len(self._seq),
            peak_dps=max(l.peak for l in self._seq),
        )
