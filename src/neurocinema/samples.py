"""The one data type every source produces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SampleKind = Literal["acc", "gyro"]


@dataclass(frozen=True, slots=True)
class ImuSample:
    """One accelerometer or gyroscope reading in the sensor frame.

    ``t`` is in seconds on the source clock (LSL ``local_clock`` for the headset).
    Accelerometer values are in g, gyroscope values in degrees per second.
    """

    t: float
    kind: SampleKind
    x: float
    y: float
    z: float
