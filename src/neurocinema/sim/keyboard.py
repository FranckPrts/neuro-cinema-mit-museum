"""Drive the virtual head from the keyboard (macOS, Linux and Windows terminals)."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..sources.sim import SimSource

if TYPE_CHECKING:
    from ..engine import Engine

HELP = """\
simulator keys:
  arrows / a d w x   turn left/right, look up/down
  1 2 3              face left / middle / right column (active setup's centres)
  s                  next selection setup
  t b                look at top / bottom half
  0                  look straight ahead (calibration pose)
  n                  nod (confirm)
  h                  head shake (reset)
  o                  take headset off / put it on
  c                  drop / restore the Bluetooth connection
  ?                  this help
  q                  quit
"""


class KeyReader:
    """Non-blocking single-key reads. Returns names like 'a', 'up', 'left'."""

    def __init__(self) -> None:
        self._old = None
        if os.name != "nt" and sys.stdin.isatty():
            import termios
            import tty

            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def read(self) -> list[str]:
        if os.name == "nt":
            return self._read_windows()
        return self._read_posix()

    def _read_posix(self) -> list[str]:
        import select

        keys: list[str] = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
            if ch == "\x1b":
                seq = os.read(sys.stdin.fileno(), 2).decode(errors="ignore")
                keys.append({"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(seq, "esc"))
            elif ch:
                keys.append(ch)
        return keys

    def _read_windows(self) -> list[str]:
        import msvcrt

        keys: list[str] = []
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                code = msvcrt.getwch()
                keys.append({"H": "up", "P": "down", "M": "right", "K": "left"}.get(code, "?"))
            else:
                keys.append(ch)
        return keys

    def close(self) -> None:
        if self._old is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)
            self._old = None


class SimKeyboard:
    """An App hook: reads keys each loop and moves the virtual head."""

    COLUMN_KEYS = {"1": "left", "2": "middle", "3": "right"}

    def __init__(self, source: SimSource, engine: Engine, quit_: Callable[[], None]):
        self.source = source
        self.engine = engine
        self.head = source.head
        self.quit = quit_
        self.reader = KeyReader()
        print(HELP, file=sys.stderr)

    def __call__(self, now: float) -> None:
        for key in self.reader.read():
            self._handle(key, now)

    def _say(self, text: str) -> None:
        print(f"[sim] {text}", file=sys.stderr)

    def _handle(self, key: str, now: float) -> None:
        h = self.head
        if key in ("left", "a"):
            h.look(yaw=h.target_yaw - 10)
        elif key in ("right", "d"):
            h.look(yaw=h.target_yaw + 10)
        elif key in ("up", "w"):
            h.look(pitch=h.target_pitch + 6)
        elif key in ("down", "x"):
            h.look(pitch=h.target_pitch - 6)
        elif key in self.COLUMN_KEYS:
            h.look(yaw=self.engine.targets.column_centre(self.COLUMN_KEYS[key]))
        elif key == "s":
            names = list(self.engine.setups)
            i = names.index(self.engine.setup_name) if self.engine.setup_name in names else -1
            self.engine.command("setup", (names[(i + 1) % len(names)],), now)
            self._say(f"setup {self.engine.setup_name}: {self.engine.targets.setup.describe()}")
            return
        elif key == "t":
            h.look(pitch=12)
        elif key == "b":
            h.look(pitch=-12)
        elif key == "0":
            h.look(yaw=0, pitch=0)
        elif key == "n":
            h.nod(now)
            self._say("nod")
        elif key == "h":
            h.shake(now)
            self._say("shake")
        elif key == "o":
            h.set_worn(not h.worn, now)
            self._say("headset on" if h.worn else "headset off")
            return
        elif key == "c":
            self.source.streaming = not self.source.streaming
            self._say("connection restored" if self.source.streaming else "connection dropped")
            return
        elif key == "?":
            print(HELP, file=sys.stderr)
            return
        elif key == "q":
            self.quit()
            return
        else:
            return
        if key not in ("n", "h"):
            self._say(f"looking at yaw {h.target_yaw:+.0f}, pitch {h.target_pitch:+.0f}")

    def close(self) -> None:
        self.reader.close()
