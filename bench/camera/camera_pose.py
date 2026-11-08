"""Camera position in the room (camera.toml) -> the bearing and tilt used by pointing.cam_to_room.

Room frame as in config/room.toml: metres, origin on the floor under the
visitor's eyes, x = visitor's right, y = up, z = behind the visitor.
"""

from __future__ import annotations

import copy
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULTS = {
    "camera": {"x_m": 0.0, "y_m": 1.3, "z_m": -0.6},   # desk test: laptop in front of you
    "aim": {"x_m": 0.0, "y_m": 1.3, "z_m": 0.0},       # the visitor's chest on the standing spot
}


class CameraConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CameraPose:
    position: tuple[float, float, float]
    aim: tuple[float, float, float]
    bearing_deg: float   # direction from the aim point toward the camera, + = visitor's right
    tilt_deg: float      # how far the camera looks down

    @property
    def distance_m(self) -> float:
        return math.dist(self.position, self.aim)


def pose_from_points(position, aim) -> CameraPose:
    """A camera at `position` looking at `aim`, with no roll."""
    f = [a - p for a, p in zip(aim, position)]  # forward
    n = math.sqrt(sum(c * c for c in f))
    if n < 1e-6:
        raise CameraConfigError("the camera position and its aim point are the same")
    fx, fy, fz = (c / n for c in f)
    if math.hypot(fx, fz) < 1e-3:
        raise CameraConfigError("a camera looking straight up or down is not supported")
    bearing = math.degrees(math.atan2(-fx, fz))
    tilt = math.degrees(math.asin(-fy))
    return CameraPose(tuple(map(float, position)), tuple(map(float, aim)), bearing + 0.0, tilt + 0.0)


def load_camera(path: Path | None) -> CameraPose:
    """Defaults, overlaid with the file if it exists. Unknown keys are errors."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None and Path(path).exists():
        with open(path, "rb") as f:
            try:
                data = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise CameraConfigError(f"{path}: {e}") from e
        for section, values in data.items():
            if section not in cfg or not isinstance(values, dict):
                raise CameraConfigError(f"{path}: unknown section [{section}]")
            for key, value in values.items():
                if key not in cfg[section]:
                    raise CameraConfigError(f"{path}: unknown key '{section}.{key}'")
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise CameraConfigError(f"{path}: {section}.{key} must be a number")
                cfg[section][key] = float(value)
    c, a = cfg["camera"], cfg["aim"]
    return pose_from_points((c["x_m"], c["y_m"], c["z_m"]), (a["x_m"], a["y_m"], a["z_m"]))
