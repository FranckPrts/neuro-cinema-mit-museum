"""Named selection setups: which yaw range picks which column.

A setup gives each column (left / middle / right) its own [lo, hi] yaw range,
in degrees (+ = visitor's right), independently of the room geometry, so
ranges may overlap (the head stays in its column across the shared zone) or
leave gaps (no target there). One setup is active at a time; the engine can
switch at runtime (/<prefix>/cmd/setup <name>).

The built-in setup needs no entry: "room" (targets.from_room, the screens'
edges) or "manual" (halfway between the three targets.*_yaw_deg), widened by
half of yaw_hysteresis_deg on the inner edges, which is the dead band the
targets always had. The others come from config/selection_setups.toml:

    [narrow-centre]
    left   = [-60, -12]
    middle = [-18, 18]
    right  = [12, 60]
    centres = [-30, 0, 30]     # optional, default = middle of each range
    split_pitch_deg = 0.0      # optional, default = [targets]
    max_pitch_deg = 40.0       # optional, default = [targets]

Standard library only (the viewer reads it too).
"""

from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, TargetsConfig, resolve_path

log = logging.getLogger(__name__)

COLUMNS = ("left", "middle", "right")
BUILTIN_NAMES = ("room", "manual")
_KEYS = (*COLUMNS, "centres", "split_pitch_deg", "max_pitch_deg")
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class SetupError(ConfigError):
    pass


@dataclass(frozen=True)
class SelectionSetup:
    name: str
    ranges: dict[str, tuple[float, float]]  # column -> (lo, hi), degrees
    centres: dict[str, float]               # re-anchoring on selection; sim keys 1 2 3
    split_pitch_deg: float
    max_pitch_deg: float
    builtin: bool = False

    def osc_args(self) -> tuple:
        """/setup name l_lo l_hi m_lo m_hi r_lo r_hi c_l c_m c_r split max."""
        r = self.ranges
        nums = (*r["left"], *r["middle"], *r["right"],
                *(self.centres[c] for c in COLUMNS), self.split_pitch_deg, self.max_pitch_deg)
        return (self.name, *(round(float(v), 2) for v in nums))

    def as_json(self) -> dict:
        return {"name": self.name, "builtin": self.builtin,
                "ranges": {c: list(self.ranges[c]) for c in COLUMNS},
                "centres": dict(self.centres),
                "split_pitch_deg": self.split_pitch_deg, "max_pitch_deg": self.max_pitch_deg}

    def describe(self) -> str:
        return "  ".join(f"{c} {lo:+.0f}..{hi:+.0f}" for c, (lo, hi) in self.ranges.items())


def builtin_name(t: TargetsConfig) -> str:
    return "room" if t.from_room else "manual"


def builtin_setup(t: TargetsConfig) -> SelectionSetup:
    """Today's columns: the bounds, with the dead band as the overlap."""
    lo, lm, mr, hi = t.bounds()
    h2 = t.yaw_hysteresis_deg / 2
    return SelectionSetup(
        name=builtin_name(t),
        ranges={"left": (lo, lm + h2), "middle": (lm - h2, mr + h2), "right": (mr - h2, hi)},
        centres={"left": t.left_yaw_deg, "middle": t.middle_yaw_deg, "right": t.right_yaw_deg},
        split_pitch_deg=t.split_pitch_deg,
        max_pitch_deg=t.max_pitch_deg,
        builtin=True,
    )


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SetupError(f"{what}: expected a number, got {value!r}")
    return float(value)


def _numbers(value: object, n: int, what: str) -> list[float]:
    if not isinstance(value, list) or len(value) != n:
        raise SetupError(f"{what}: expected {n} numbers, got {value!r}")
    return [_number(v, what) for v in value]


def parse_setup(name: str, data: object, t: TargetsConfig) -> SelectionSetup:
    """One table of the setups file; raises SetupError (message names the setup)."""
    where = f"[{name}]"
    if not _NAME.match(name):
        raise SetupError(f"{where}: a setup name is letters, digits, '-', '_' or '.'")
    if name in BUILTIN_NAMES:
        raise SetupError(f"{where}: '{name}' is the built-in setup's name")
    if not isinstance(data, dict):
        raise SetupError(f"{where}: expected a table")
    for key in data:
        if key not in _KEYS:
            raise SetupError(f"{where}: unknown key '{key}' (expected {', '.join(_KEYS)})")
    ranges: dict[str, tuple[float, float]] = {}
    for col in COLUMNS:
        if col not in data:
            raise SetupError(f"{where}: missing '{col}'")
        lo, hi = _numbers(data[col], 2, f"{where} {col}")
        if not lo < hi:
            raise SetupError(f"{where} {col}: expected [lo, hi] with lo < hi, got [{lo:g}, {hi:g}]")
        ranges[col] = (lo, hi)
    (l_lo, l_hi), (m_lo, m_hi), (r_lo, r_hi) = (ranges[c] for c in COLUMNS)
    if not (l_lo < m_lo < r_lo and l_hi < m_hi < r_hi):
        raise SetupError(f"{where}: columns out of order (left must start and end before middle, "
                         "middle before right)")
    if l_hi > r_lo:
        raise SetupError(f"{where}: left and right overlap; a column may only overlap its neighbour")
    if "centres" in data:
        values = _numbers(data["centres"], 3, f"{where} centres")
        centres = dict(zip(COLUMNS, values))
        for col, c in centres.items():
            lo, hi = ranges[col]
            if not lo <= c <= hi:
                raise SetupError(f"{where} centres: {col} centre {c:g} is outside its range [{lo:g}, {hi:g}]")
    else:
        centres = {col: (lo + hi) / 2 for col, (lo, hi) in ranges.items()}
    split = _number(data.get("split_pitch_deg", t.split_pitch_deg), f"{where} split_pitch_deg")
    max_pitch = _number(data.get("max_pitch_deg", t.max_pitch_deg), f"{where} max_pitch_deg")
    if max_pitch <= 0:
        raise SetupError(f"{where} max_pitch_deg must be positive")
    if not m_lo <= 0 < m_hi:
        log.warning("selection setup %s: yaw 0 (the calibration gaze) is outside middle [%g, %g]",
                    name, m_lo, m_hi)
    return SelectionSetup(name, ranges, centres, split, max_pitch)


def setups_path(t: TargetsConfig) -> Path | None:
    return resolve_path(t.setups_file) if t.setups_file else None


def load_setups(t: TargetsConfig, path: Path | str | None = None) -> dict[str, SelectionSetup]:
    """The built-in setup, then every setup in the file (`path`, default
    targets.setups_file). A missing file leaves only the built-in one."""
    setups = {builtin_name(t): builtin_setup(t)}
    path = Path(path) if path is not None else setups_path(t)
    if path is None:
        return setups
    if not path.exists():
        log.warning("selection setups file %s not found; only the built-in setup", path)
        return setups
    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise SetupError(f"{path}: {e}") from e
    for name, table in data.items():
        try:
            setups[name] = parse_setup(name, table, t)
        except SetupError as e:
            raise SetupError(f"{path}: {e}") from e
    return setups


def startup_name(t: TargetsConfig, setups: dict[str, SelectionSetup]) -> str:
    """The setup to start with (targets.setup, or the built-in one)."""
    name = t.setup or builtin_name(t)
    if name not in setups:
        raise SetupError(f"targets.setup: no selection setup named {name!r} "
                         f"(have {', '.join(setups)})")
    return name
