"""Runs `muselsl stream` and exits if the engine that launched it dies.

Without this, a crashed engine would leave an orphaned muselsl holding the
Bluetooth connection, and the restarted engine could not reach the headset.

    python -m neurocinema.musestream --parent-pid 1234 -- --acc --gyro ...
"""

from __future__ import annotations

import os
import sys
import threading
import time

import psutil


def _watch_parent(pid: int) -> None:
    while True:
        time.sleep(1.0)
        if not psutil.pid_exists(pid):
            print("parent engine is gone; stopping muselsl", flush=True)
            os._exit(0)


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 3 or argv[0] != "--parent-pid" or argv[2] != "--":
        print("usage: python -m neurocinema.musestream --parent-pid PID -- [muselsl stream args]")
        sys.exit(2)
    parent = int(argv[1])
    threading.Thread(target=_watch_parent, args=(parent,), daemon=True).start()

    from muselsl.__main__ import main as muselsl_main

    sys.argv = ["muselsl", "stream", *argv[3:]]
    muselsl_main()


if __name__ == "__main__":
    main()
