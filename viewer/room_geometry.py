"""Moved to `neurocinema.room` (the engine uses it too). Kept so the viewer and
the camera bench can still `import room_geometry`."""

from neurocinema.room import *  # noqa: F403
from neurocinema.room import DEFAULTS, WALL_NAMES, RoomError, gaze_dir, hit, load_room, walls  # noqa: F401
