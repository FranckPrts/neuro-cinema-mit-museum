"""Configuration: dataclass defaults, overridden by one or more TOML files.

The dataclass defaults are the source of truth. ``config/installation.toml``
mirrors them with comments so the values can be tuned on site; unknown keys are
rejected so a typo never silently falls back to a default.
"""

from __future__ import annotations

import importlib.util
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar

AXIS_NAMES = ("x", "-x", "y", "-y", "z", "-z")


class ConfigError(ValueError):
    pass


@dataclass
class SourceConfig:
    # "muse" reads the headset through LSL; "sim" runs a virtual head (no hardware);
    # "muse_direct" connects to the headset in-process (battery, staff headset swaps).
    kind: str = "muse"


@dataclass
class MuseConfig:
    # Launch and supervise `muselsl stream` ourselves. Set false if something
    # else (BlueMuse, a manual muselsl) already publishes the LSL streams.
    manage_stream: bool = True
    # Headset identity. Empty = first Muse found. On macOS this is the
    # CoreBluetooth UUID shown by `muselsl list`, not a MAC address.
    address: str = ""
    name: str = ""
    backend: str = "auto"
    # Layer A only needs the IMU; EEG stays off to keep the BLE link light.
    stream_eeg: bool = False
    extra_args: list[str] = field(default_factory=list)
    # LSL stream selection.
    stream_name: str = "Muse"
    source_id: str = ""
    # Recovery timings.
    startup_grace_s: float = 45.0
    stale_after_s: float = 2.0
    resolve_wait_s: float = 1.0
    backoff_initial_s: float = 2.0
    backoff_max_s: float = 60.0
    escalate_after_failures: int = 5
    # Optional command run every `escalate_after_failures` consecutive
    # failures, e.g. to restart Bluetooth. Argument list, no shell.
    escalation_command: list[str] = field(default_factory=list)
    pidfile: str = "run/muselsl.pid"


@dataclass
class HeadsetsConfig:
    """In-process headset connection (source.kind = "muse_direct")."""

    # Registered headsets, "LABEL=BLE name", e.g. ["A=Muse-41D2", "B=Muse-0C3A"].
    # The label is the sticker on the headset. Only these are ever connected.
    registry: list[str] = field(default_factory=list)
    # Looking for headsets while none is in use.
    scan_interval_s: float = 10.0
    scan_timeout_s: float = 5.0
    connect_attempts: int = 3
    # After staff choose a headset, it is used only once it moves (they lift it):
    # gyro above this, within the timeout.
    confirm_dps: float = 40.0
    confirm_timeout_s: float = 120.0
    # In use: no samples this long = lost; then retry the same headset this often.
    stale_after_s: float = 2.0
    reconnect_interval_s: float = 5.0
    # Status: "offline" after this long without a headset in use; "swap soon"
    # below this battery level or estimated time left.
    offline_after_s: float = 60.0
    swap_soon_percent: float = 20.0
    swap_soon_minutes: float = 30.0
    # Fallback when nobody confirms: after this long with no headset, use the
    # only registered headset in range without confirmation. 0 = off.
    auto_adopt_after_s: float = 0.0

    def entries(self) -> list[tuple[str, str]]:
        """(label, BLE name) pairs; raises ConfigError if malformed."""
        out = []
        for item in self.registry:
            label, sep, name = (part.strip() for part in item.partition("="))
            if not sep or not label or not name:
                raise ConfigError(f"headsets.registry: expected 'LABEL=Muse-XXXX', got {item!r}")
            out.append((label, name))
        labels = [label for label, _ in out]
        names = [name for _, name in out]
        if len(set(labels)) != len(labels) or len(set(names)) != len(names):
            raise ConfigError("headsets.registry: labels and names must be unique")
        return out


@dataclass
class StatusPageConfig:
    # Local web page for staff (muse_direct only). host "0.0.0.0" makes it
    # reachable from a tablet on the same network.
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8081  # 8080 is the 3D viewer


@dataclass
class ImuConfig:
    # Orientation backend: "complementary" (the filter below) or "vqf" (needs
    # the optional extra: pip install -e '.[vqf]'). VQF only replaces how "up"
    # and the gyro bias are estimated; yaw, gestures and targets are shared.
    backend: str = "complementary"
    # Sensor axis pointing out of the forehead. Verify on the real headset.
    forward_axis: str = "x"
    invert_yaw: bool = False
    invert_pitch: bool = False
    # Complementary filter on gravity: larger = trust the gyro longer.
    gravity_tau_s: float = 1.0
    # Accelerometer only corrects gravity when |acc| is within 1 g +/- this.
    acc_trust_band_g: float = 0.15
    # Stillness: over `still_window_s`, every gyro axis varies by less than
    # `still_gyro_range_dps` and |acc| by less than `still_acc_range_g`.
    still_window_s: float = 0.5
    still_gyro_range_dps: float = 3.0
    still_acc_range_g: float = 0.03
    # Gyro bias keeps being re-estimated while still, unless the mean rate is
    # further than this from the current bias (that is a slow turn, not bias).
    bias_tau_s: float = 8.0
    bias_update_max_dps: float = 2.0
    max_bias_dps: float = 20.0
    # While still, stop integrating yaw (zero-velocity update).
    freeze_yaw_when_still: bool = True
    # Gyro bias learned while the headset lies still off the head (engine idle):
    # a running mean over `idle_bias_tau_s`, usable after `idle_bias_min_s` of
    # stillness. Calibration uses it instead of the 2 s worn mean if it is
    # younger than `idle_bias_max_age_s` (0 = off) and within
    # `bias_update_max_dps` of that mean. Kept across a dropout of the same headset.
    idle_bias_tau_s: float = 10.0
    idle_bias_min_s: float = 5.0
    idle_bias_max_age_s: float = 600.0
    # VQF only (https://vqf.readthedocs.io, VQFParams): accelerometer low-pass
    # time constant; rest detection thresholds and minimum duration; the
    # largest gyro bias it can represent (above it, rest is never detected).
    vqf_tau_acc_s: float = 3.0
    vqf_rest_th_gyr_dps: float = 4.0
    vqf_rest_th_acc_ms2: float = 1.0
    vqf_rest_min_s: float = 1.0
    vqf_bias_clip_dps: float = 5.0
    # Gyro bias with VQF: "vqf" (its own estimate, learned at rest and during
    # motion; calibration keeps it under the idle-bias rules above and if its
    # uncertainty is below `vqf_bias_max_sigma_dps`, otherwise starts from the
    # worn mean) or "stillness" (the idle / calibration / still-update bias above).
    vqf_bias_from: str = "vqf"
    vqf_bias_max_sigma_dps: float = 0.1


@dataclass
class CalibrationConfig:
    # Continuous stillness needed, looking at the middle of the middle screen.
    still_s: float = 2.0
    # After this long without success, send a `calibration_slow` event once.
    slow_after_s: float = 15.0


@dataclass
class TargetsConfig:
    # true: the screens' yaws (centres and edges) come from the room file, so
    # the engine, the viewer and the camera bench share one room. false: use
    # the three *_yaw_deg keys below (column boundaries halfway between them).
    from_room: bool = True
    # Relative paths are tried from the current folder, then the repo root.
    room_file: str = "config/room.toml"
    # Yaw of each screen's centre, degrees, + = visitor's right. Filled from
    # the room file when from_room is true (setting them then is an error).
    left_yaw_deg: float = -40.0
    middle_yaw_deg: float = 0.0
    right_yaw_deg: float = 40.0
    # Width of the dead band around each column boundary.
    yaw_hysteresis_deg: float = 6.0
    # from_room = false only: beyond the outer screens' centres by more than
    # this = no target. With the room, no target = past the side screens' ends.
    outer_margin_deg: float = 30.0
    # Pitch of the horizontal divider relative to the calibration gaze.
    split_pitch_deg: float = 0.0
    # Width of the dead band around the divider.
    pitch_hysteresis_deg: float = 4.0
    # Further than this above/below the divider = no target.
    max_pitch_deg: float = 40.0

    # Set by load_config from the room (not a config key): (left outer end,
    # left|middle, middle|right, right outer end), degrees.
    column_bounds: ClassVar[tuple[float, float, float, float] | None] = None
    room_hash: ClassVar[str] = ""  # first 12 hex digits of the room file's SHA-1

    def bounds(self) -> tuple[float, float, float, float]:
        if self.column_bounds is not None:
            return self.column_bounds
        m = self.outer_margin_deg
        return (self.left_yaw_deg - m, (self.left_yaw_deg + self.middle_yaw_deg) / 2,
                (self.middle_yaw_deg + self.right_yaw_deg) / 2, self.right_yaw_deg + m)


@dataclass
class OscillationConfig:
    """A nod or shake: alternating lobes of angular velocity on one axis."""

    enter_dps: float = 30.0
    exit_dps: float = 12.0
    peak_dps: float = 60.0
    min_lobes: int = 2
    max_gap_s: float = 0.35
    max_lobe_s: float = 0.6
    # Primary-axis motion must exceed the other axis by this factor.
    dominance: float = 1.5
    cooldown_s: float = 0.8
    # Direction the first lobe must have: -1 negative, 1 positive, 0 either.
    first_sign: int = 0


@dataclass
class GesturesConfig:
    # A "yes" nod starts downwards; this also keeps a quick upward glance
    # followed by a dip from being read as a nod.
    nod: OscillationConfig = field(default_factory=lambda: OscillationConfig(
        enter_dps=30.0, exit_dps=12.0, peak_dps=55.0, min_lobes=2, first_sign=-1,
    ))
    shake: OscillationConfig = field(default_factory=lambda: OscillationConfig(
        enter_dps=35.0, exit_dps=15.0, peak_dps=70.0, min_lobes=3,
    ))


@dataclass
class InteractionConfig:
    # A nod only selects after looking at the target this long (also the
    # time for /dwell to reach 1.0).
    min_dwell_s: float = 0.4
    # Time spent in SELECTED before tracking resumes.
    selected_hold_s: float = 2.0
    # On selection, pull yaw this fraction of the way to the screen centre.
    reanchor_gain: float = 0.5
    # After a shake: wait for stillness this long (or the timeout), then zero yaw.
    reset_settle_s: float = 0.3
    reset_timeout_s: float = 3.0
    # IDLE (headset not worn): gyro range below `idle_gyro_range_dps` for
    # `idle_still_s`, or tilted more than `idle_tilt_deg` from the calibration
    # pose for `idle_tilt_s`.
    idle_still_s: float = 20.0
    idle_gyro_range_dps: float = 0.6
    idle_tilt_deg: float = 70.0
    idle_tilt_s: float = 3.0
    # Leaving IDLE: any gyro reading above this starts calibration.
    wake_dps: float = 30.0
    start_phase: str = "choosing"


@dataclass
class OscConfig:
    prefix: str = "/nc"
    # Every message goes to each destination ("host:port"). Add the monitor
    # here (e.g. "127.0.0.1:7002") to watch alongside TouchDesigner.
    destinations: list[str] = field(default_factory=lambda: ["127.0.0.1:7000"])
    listen_host: str = "127.0.0.1"
    listen_port: int = 7001
    pose_rate_hz: float = 30.0
    status_rate_hz: float = 2.0
    snapshot_rate_hz: float = 1.0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/neurocinema.log"


@dataclass
class Config:
    source: SourceConfig = field(default_factory=SourceConfig)
    muse: MuseConfig = field(default_factory=MuseConfig)
    headsets: HeadsetsConfig = field(default_factory=HeadsetsConfig)
    status_page: StatusPageConfig = field(default_factory=StatusPageConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    gestures: GesturesConfig = field(default_factory=GesturesConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    osc: OscConfig = field(default_factory=OscConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        if self.source.kind not in ("muse", "sim", "muse_direct"):
            raise ConfigError(
                f"source.kind must be 'muse', 'sim' or 'muse_direct', got {self.source.kind!r}")
        self.headsets.entries()
        if self.headsets.connect_attempts < 1:
            raise ConfigError("headsets.connect_attempts must be at least 1")
        if self.imu.forward_axis not in AXIS_NAMES:
            raise ConfigError(f"imu.forward_axis must be one of {AXIS_NAMES}")
        if self.imu.backend not in ("complementary", "vqf"):
            raise ConfigError(f"imu.backend must be 'complementary' or 'vqf', got {self.imu.backend!r}")
        if self.imu.vqf_bias_from not in ("vqf", "stillness"):
            raise ConfigError("imu.vqf_bias_from must be 'vqf' or 'stillness'")
        if self.imu.backend == "vqf" and importlib.util.find_spec("vqf") is None:
            raise ConfigError("imu.backend = 'vqf' needs the vqf package: pip install -e '.[vqf]'")
        t = self.targets
        if not t.left_yaw_deg < t.middle_yaw_deg < t.right_yaw_deg:
            raise ConfigError("targets: expected left_yaw_deg < middle_yaw_deg < right_yaw_deg")
        if self.interaction.start_phase not in ("choosing", "watching"):
            raise ConfigError("interaction.start_phase must be 'choosing' or 'watching'")
        for dest in self.osc.destinations:
            parse_host_port(dest)
        for name in ("nod", "shake"):
            g: OscillationConfig = getattr(self.gestures, name)
            if not g.exit_dps < g.enter_dps <= g.peak_dps:
                raise ConfigError(f"gestures.{name}: expected exit_dps < enter_dps <= peak_dps")
            if g.min_lobes < 2:
                raise ConfigError(f"gestures.{name}.min_lobes must be at least 2")
            if g.first_sign not in (-1, 0, 1):
                raise ConfigError(f"gestures.{name}.first_sign must be -1, 0 or 1")
        if not self.osc.prefix.startswith("/"):
            raise ConfigError("osc.prefix must start with '/'")


def parse_host_port(value: str) -> tuple[str, int]:
    host, sep, port = value.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ConfigError(f"expected 'host:port', got {value!r}")
    return host, int(port)


def _coerce(current: Any, value: Any, name: str) -> Any:
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{name}: expected true/false, got {value!r}")
        return value
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{name}: expected a number, got {value!r}")
        return float(value)
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{name}: expected an integer, got {value!r}")
        return value
    if isinstance(current, str):
        if not isinstance(value, str):
            raise ConfigError(f"{name}: expected a string, got {value!r}")
        return value
    if isinstance(current, list):
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{name}: expected a list of strings, got {value!r}")
        return list(value)
    raise ConfigError(f"{name}: unsupported setting type")


def _merge(obj: Any, data: dict[str, Any], path: str) -> None:
    known = {f.name for f in fields(obj)}
    for key, value in data.items():
        name = f"{path}{key}"
        if key not in known:
            raise ConfigError(f"unknown config key '{name}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ConfigError(f"{name}: expected a [{name}] table")
            _merge(current, value, f"{name}.")
        else:
            setattr(obj, key, _coerce(current, value, name))


_YAW_KEYS = ("left_yaw_deg", "middle_yaw_deg", "right_yaw_deg")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(paths: list[Path] | tuple[Path, ...] = (), apply_room: bool = True) -> Config:
    """Defaults, then each file in order (later files win), then the screen
    yaws from the room file when targets.from_room is true."""
    cfg = Config()
    yaw_set_in: Path | None = None
    for path in paths:
        with open(path, "rb") as f:
            try:
                data = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise ConfigError(f"{path}: {e}") from e
        if any(k in data.get("targets", {}) for k in _YAW_KEYS):
            yaw_set_in = Path(path)
        try:
            _merge(cfg, data, "")
        except ConfigError as e:
            raise ConfigError(f"{path}: {e}") from e
    if apply_room and cfg.targets.from_room:
        if yaw_set_in is not None:
            raise ConfigError(f"{yaw_set_in}: targets.*_yaw_deg are set but targets.from_room = true "
                              "(the room file gives them); set from_room = false to use them")
        apply_room_file(cfg.targets)
    cfg.validate()
    return cfg


def apply_room_file(t: TargetsConfig) -> None:
    """Screen centres and edges from the room file, into `t`."""
    from .room import RoomError, load_room, screen_layout

    path = Path(t.room_file)
    if not path.is_absolute() and not path.exists():
        path = _REPO_ROOT / path
    if not path.exists():
        raise ConfigError(f"targets.room_file: {t.room_file} not found (or set targets.from_room = false)")
    import hashlib

    try:
        layout = screen_layout(load_room(path))
    except RoomError as e:
        raise ConfigError(f"targets.room_file: {e}") from e
    t.left_yaw_deg = layout.centre["left"]
    t.middle_yaw_deg = layout.centre["middle"]
    t.right_yaw_deg = layout.centre["right"]
    t.column_bounds = layout.bounds
    t.room_hash = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
