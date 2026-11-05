"""Room and screen geometry for the gaze projection, shared by the viewer.

Standard library only, so the engine or TouchDesigner scripts can import it
later. Values come from config/room.toml (missing keys keep the defaults).

Room frame: metres, y up, origin on the floor under the visitor's eyes,
x = visitor's right, -z = straight ahead (yaw 0). Engine convention:
+yaw = right, +pitch = up.

Each wall has a 2D frame as seen from inside the room, facing it:
u runs left to right from the wall's bottom-left corner, v runs up from the floor.
"""

from __future__ import annotations

import copy
import math
import tomllib
from pathlib import Path

DEFAULTS: dict = {
    "room": {
        "width_m": 16.5 * 0.3048,
        "depth_m": 16.5 * 0.3048,
        "ceiling_m": 10 * 0.3048,      # unconfirmed
        "eye_height_m": 1.6,           # unconfirmed
        "visitor_inset_m": 2 * 0.3048,
    },
    "screens": {
        "walls": ["front", "left", "right"],  # unconfirmed
        "inset_m": 1.44 * 0.0254,
    },
    "gaze": {"cone_deg": 5.0},
}

WALL_NAMES = ("front", "right", "back", "left")


class RoomError(ValueError):
    pass


def load_room(path: Path | None) -> dict:
    """Defaults, overlaid with the file if it exists. Unknown keys are errors."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is None or not Path(path).exists():
        return cfg
    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise RoomError(f"{path}: {e}") from e
    for section, values in data.items():
        if section not in cfg or not isinstance(values, dict):
            raise RoomError(f"{path}: unknown section [{section}]")
        for key, value in values.items():
            if key not in cfg[section]:
                raise RoomError(f"{path}: unknown key '{section}.{key}'")
            default = cfg[section][key]
            if isinstance(default, list):
                if not isinstance(value, list) or any(v not in WALL_NAMES for v in value):
                    raise RoomError(f"{path}: {section}.{key} must be a list of {WALL_NAMES}")
            elif not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise RoomError(f"{path}: {section}.{key} must be a non-negative number")
            cfg[section][key] = list(value) if isinstance(default, list) else float(value)
    return cfg


def walls(cfg: dict) -> dict[str, dict]:
    """Each wall: plane origin (bottom-left), u axis, inward normal, width, height (room frame)."""
    r = cfg["room"]
    hw, front, back, h = r["width_m"] / 2, -(r["depth_m"] - r["visitor_inset_m"]), r["visitor_inset_m"], r["ceiling_m"]
    return {
        "front": {"origin": (-hw, 0.0, front), "u": (1.0, 0.0, 0.0), "normal": (0.0, 0.0, 1.0), "width": r["width_m"], "height": h},
        "right": {"origin": (hw, 0.0, front), "u": (0.0, 0.0, 1.0), "normal": (-1.0, 0.0, 0.0), "width": r["depth_m"], "height": h},
        "back": {"origin": (hw, 0.0, back), "u": (-1.0, 0.0, 0.0), "normal": (0.0, 0.0, -1.0), "width": r["width_m"], "height": h},
        "left": {"origin": (-hw, 0.0, back), "u": (0.0, 0.0, -1.0), "normal": (1.0, 0.0, 0.0), "width": r["depth_m"], "height": h},
    }


def gaze_dir(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float]:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return (math.cos(p) * math.sin(y), math.sin(p), -math.cos(p) * math.cos(y))


def hit(cfg: dict, yaw_deg: float, pitch_deg: float = 0.0) -> dict:
    """Where the gaze ray from the eye meets the room (first surface)."""
    r, s = cfg["room"], cfg["screens"]
    eye = (0.0, r["eye_height_m"], 0.0)
    d = gaze_dir(yaw_deg, pitch_deg)
    best: tuple[float, str] | None = None
    for name, w in walls(cfg).items():
        n = w["normal"]
        denom = sum(a * b for a, b in zip(d, n))
        if denom < -1e-9:
            t = sum((o - e) * k for o, e, k in zip(w["origin"], eye, n)) / denom
            if t > 0 and (best is None or t < best[0]):
                best = (t, name)
    if d[1] < -1e-9:
        t = -eye[1] / d[1]
        if best is None or t < best[0]:
            best = (t, "floor")
    elif d[1] > 1e-9:
        t = (r["ceiling_m"] - eye[1]) / d[1]
        if best is None or t < best[0]:
            best = (t, "ceiling")
    t, name = best  # a closed box always gives a hit
    p = tuple(e + t * k for e, k in zip(eye, d))
    out = {"surface": name, "point": p, "distance": t, "yaw_deg": yaw_deg, "pitch_deg": pitch_deg,
           "on_screen": False}
    if name in ("floor", "ceiling"):
        return out
    w = walls(cfg)[name]
    u = sum((a - o) * k for a, o, k in zip(p, w["origin"], w["u"]))
    v = p[1]
    i = s["inset_m"]
    sw, sh = w["width"] - 2 * i, w["height"] - 2 * i
    out.update(u=u, v=v, su=(u - i) / sw, sv=(v - i) / sh)
    out["on_screen"] = name in s["walls"] and 0 <= out["su"] <= 1 and 0 <= out["sv"] <= 1
    return out


if __name__ == "__main__":  # quick table: python viewer/room_geometry.py
    cfg = load_room(Path(__file__).resolve().parent.parent / "config/room.toml")
    for yaw in (-90, -60, -29.64, -29.5, 0, 29.5, 29.64, 60, 90, 150, 180):
        h = hit(cfg, yaw)
        where = f"screen {h['surface']}" if h["on_screen"] else h["surface"]
        print(f"yaw {yaw:7.2f}  {where:13s} d={h['distance']:.3f}  "
              f"u={h.get('u', float('nan')):.3f} su={h.get('su', float('nan')):.3f}")
