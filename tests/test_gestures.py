import math

from neurocinema.config import Config
from neurocinema.gestures import OscillationDetector

RATE = 52.0


def run(detector, fn, seconds, other=lambda t: 0.0):
    found = []
    for i in range(int(seconds * RATE)):
        t = i / RATE
        g = detector.update(t, fn(t), other(t))
        if g:
            found.append(g)
    return found


def nod_rate(t, amp=16.0, period=0.6, t0=0.5):
    tau = t - t0
    if not 0 <= tau <= period:
        return 0.0
    # Derivative of -amp * (1 - cos(2 pi tau / T)) / 2
    return -amp * math.pi / period * math.sin(2 * math.pi * tau / period)


def shake_rate(t, amp=14.0, f=2.5, dur=1.2, t0=0.5):
    tau = t - t0
    if not 0 <= tau <= dur:
        return 0.0
    env = math.sin(math.pi * tau / dur) ** 2
    denv = 2 * math.sin(math.pi * tau / dur) * math.cos(math.pi * tau / dur) * math.pi / dur
    return amp * (2 * math.pi * f * math.cos(2 * math.pi * f * tau) * env + math.sin(2 * math.pi * f * tau) * denv)


def test_nod_detected_once_with_onset_at_start():
    d = OscillationDetector("nod", Config().gestures.nod)
    found = run(d, nod_rate, 2.0)
    assert len(found) == 1
    assert abs(found[0].onset - 0.5) < 0.1


def test_shake_detected_by_shake_detector():
    d = OscillationDetector("shake", Config().gestures.shake)
    assert len(run(d, shake_rate, 2.5)) == 1


def test_nod_motion_is_not_a_shake_and_vice_versa():
    gc = Config().gestures
    # Shake detector sees the nod on its "other" axis only.
    assert run(OscillationDetector("shake", gc.shake), lambda t: 0.0, 2.0, other=nod_rate) == []
    # Nod detector watching pitch while the head shakes (pitch quiet).
    assert run(OscillationDetector("nod", gc.nod), lambda t: 0.0, 2.5, other=shake_rate) == []


def test_diagonal_motion_fails_dominance():
    d = OscillationDetector("nod", Config().gestures.nod)
    # Same magnitude on both axes: not a clean nod.
    assert run(d, nod_rate, 2.0, other=lambda t: 1.2 * nod_rate(t)) == []


def test_small_nods_while_talking_are_ignored():
    d = OscillationDetector("nod", Config().gestures.nod)
    assert run(d, lambda t: nod_rate(t, amp=5.0), 2.0) == []


def test_a_turn_and_return_glance_is_not_a_shake():
    def glance(t):
        # Turn right then straight back: two lobes only.
        if 0.5 <= t < 0.8:
            return 150 * math.sin(math.pi * (t - 0.5) / 0.3)
        if 0.85 <= t < 1.15:
            return -150 * math.sin(math.pi * (t - 0.85) / 0.3)
        return 0.0

    d = OscillationDetector("shake", Config().gestures.shake)
    assert run(d, glance, 2.0) == []


def test_slow_sustained_turn_breaks_the_sequence():
    d = OscillationDetector("nod", Config().gestures.nod)

    def slow(t):
        return 70.0 if 0.2 <= t < 1.4 else (-70.0 if 1.4 <= t < 1.7 else 0.0)

    assert run(d, slow, 2.5) == []


def test_busy_during_gesture_and_clear_after():
    d = OscillationDetector("nod", Config().gestures.nod)
    busy = []
    for i in range(int(2.0 * RATE)):
        t = i / RATE
        d.update(t, nod_rate(t), 0.0)
        busy.append((t, d.busy))
    assert any(b for t, b in busy if 0.6 < t < 1.0)
    assert not any(b for t, b in busy if t > 1.6)


def test_nod_must_start_downwards():
    d = OscillationDetector("nod", Config().gestures.nod)
    # Up-then-down: the mirror image of a nod.
    assert run(d, lambda t: -nod_rate(t), 2.0) == []
