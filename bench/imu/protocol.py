"""The recording protocol: what the participant is asked to do, and when.

About 6 minutes (`quick` is about 2 minutes, for a dry run):

  stand     headset still on its stand (VQF and the idle bias learn the gyro bias)
  put_on    put it on, look at the middle of the middle screen, keep still (calibration)
  targets   face one of the six targets, prompted every 5 s (ground truth for yaw and pitch)
  gestures  face a target and nod (a selection), or face the middle and shake (a reset)
  free      look around freely, no nods or shakes (false gestures, drift)
  check     face the middle screen (yaw drift after free looking)
  off       take the headset off and put it back on its stand

Ground-truth angles come from the target layout: the screens' yaw from
`[targets]` (or `--yaw`), and `top_pitch` / `bottom_pitch` above / below the
calibration gaze. Put markers at those angles around the participant.
"""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_EVERY_S = 5.0

COLUMNS = ("left", "middle", "right")
ROWS = ("top", "bottom")

# Fixed orders (no target twice in a row) so every session is comparable.
_TARGET_ORDER = [
    ("middle", "top"), ("left", "bottom"), ("right", "top"), ("middle", "bottom"),
    ("left", "top"), ("right", "bottom"), ("left", "bottom"), ("middle", "top"),
    ("right", "bottom"), ("left", "top"), ("middle", "bottom"), ("right", "top"),
    ("middle", "top"), ("right", "bottom"), ("left", "top"), ("middle", "bottom"),
    ("right", "top"), ("left", "bottom"),
]
_NOD_ORDER = [
    ("left", "top"), ("right", "bottom"), ("middle", "top"),
    ("left", "bottom"), ("right", "top"), ("middle", "bottom"),
]


@dataclass(frozen=True)
class Step:
    t: float  # seconds after the start
    kind: str  # "phase" or "prompt"
    name: str  # phase name, or face / nod / shake
    text: str  # what is said and shown
    column: str = ""
    row: str = ""
    yaw_deg: float | None = None
    pitch_deg: float | None = None


@dataclass(frozen=True)
class Layout:
    left_yaw_deg: float = -40.0
    middle_yaw_deg: float = 0.0
    right_yaw_deg: float = 40.0
    top_pitch_deg: float = 15.0
    bottom_pitch_deg: float = -15.0

    def angles(self, column: str, row: str) -> tuple[float, float]:
        yaw = getattr(self, f"{column}_yaw_deg")
        pitch = self.top_pitch_deg if row == "top" else self.bottom_pitch_deg
        return yaw, pitch


def build(layout: Layout, stand_s: float = 120.0, quick: bool = False) -> list[Step]:
    steps: list[Step] = []
    t = 0.0

    def phase(name: str, text: str, duration: float) -> None:
        nonlocal t
        steps.append(Step(t, "phase", name, text))
        t += duration

    def prompt(name: str, text: str, column: str = "", row: str = "") -> None:
        nonlocal t
        yaw, pitch = layout.angles(column, row) if column and row else (None, None)
        if column and not row:  # shake: facing the middle, any height
            yaw = layout.angles(column, "top")[0]
        steps.append(Step(t, "prompt", name, text, column, row, yaw, pitch))
        t += PROMPT_EVERY_S

    targets = _TARGET_ORDER[:6] if quick else _TARGET_ORDER
    nods = _NOD_ORDER[:3] if quick else _NOD_ORDER
    shakes = 1 if quick else 3

    phase("stand", "Leave the headset still on its stand.", stand_s)
    phase("put_on", "Put the headset on. Look at the middle of the middle screen and keep still.", 15.0)
    phase("targets", "Now face each target as it is called out.", 3.0)
    for column, row in targets:
        prompt("face", f"{column} {row}", column, row)
    phase("gestures", "Now face each target and nod once to choose it.", 3.0)
    for column, row in nods:
        prompt("nod", f"{column} {row}, and nod", column, row)
    for _ in range(shakes):
        prompt("shake", "face the middle, and shake your head", "middle", "")
    phase("free", "Look around freely for a minute. No nods, no shakes.", 20.0 if quick else 60.0)
    prompt("face", "middle top", "middle", "top")
    phase("off", "Take the headset off and put it back on its stand.", 10.0 if quick else 20.0)
    steps.append(Step(t, "phase", "end", "Done, thank you."))
    return steps


def duration_s(steps: list[Step]) -> float:
    return steps[-1].t
