"""Make a synthetic bench session with the virtual head following the protocol.

For trying replay.py (and testing it) without a headset:

    python bench/imu/simulate.py --out bench/imu/recordings/SIM01
    python bench/imu/replay.py bench/imu/recordings/SIM01

`--duplicates` stamps every 3 samples alike, as `muselsl stream` does.
`--shake-deg 28 --turn-dps 400` makes a vigorous participant whose head goes
past the gyro's 245 deg/s full scale.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict
from pathlib import Path

import protocol
from recording import NOMINAL_RATE_HZ, Event, EventWriter, ImuWriter, write_meta

from neurocinema.samples import ImuSample
from neurocinema.sim.head import VirtualHead

T0 = 1000.0  # sample clock at the start (any value works)
REACT_S = 0.7  # participant starts turning this long after a prompt
GESTURE_S = 2.2  # ...and nods / shakes this long after it


def simulate(folder: Path, layout: protocol.Layout | None = None, quick: bool = False,
             stand_s: float | None = None, seed: int = 1, duplicates: bool = False,
             lever_arm_m: tuple[float, float, float] = (0.10, 0.0, 0.12),
             shake_deg: float = 14.0, turn_dps: float = 120.0) -> Path:
    layout = layout or protocol.Layout()
    folder.mkdir(parents=True, exist_ok=True)
    stand_s = (20.0 if quick else 120.0) if stand_s is None else stand_s
    steps = protocol.build(layout, stand_s=stand_s, quick=quick)
    head = VirtualHead(seed=seed, worn=False, lever_arm_m=lever_arm_m)
    head.max_speed_dps = turn_dps
    rng = random.Random(seed)

    # Actions of the virtual participant, as (time after start, callable).
    actions: list[tuple[float, object]] = []
    for i, st in enumerate(steps):
        nxt = steps[i + 1].t if i + 1 < len(steps) else st.t
        if st.kind == "phase" and st.name == "put_on":
            actions.append((st.t + 2.0, lambda t: head.set_worn(True, t)))
        elif st.kind == "phase" and st.name == "free":
            t = st.t + REACT_S
            while t < nxt - 1.0:
                y, p = rng.uniform(-55, 55), rng.uniform(-25, 25)
                actions.append((t, lambda _t, y=y, p=p: head.look(yaw=y, pitch=p)))
                t += rng.uniform(1.5, 3.0)
        elif st.kind == "phase" and st.name == "off":
            actions.append((st.t + 3.0, lambda t: head.set_worn(False, t)))
        elif st.kind == "prompt":
            yaw = st.yaw_deg if st.yaw_deg is not None else 0.0
            pitch = st.pitch_deg if st.pitch_deg is not None else 0.0
            actions.append((st.t + REACT_S, lambda _t, y=yaw, p=pitch: head.look(yaw=y, pitch=p)))
            if st.name == "nod":
                actions.append((st.t + GESTURE_S, lambda t: head.nod(t)))
            elif st.name == "shake":
                actions.append((st.t + GESTURE_S, lambda t: head.shake(t, shake_deg)))
    actions.sort(key=lambda a: a[0])

    imu = ImuWriter(folder / "imu.csv")
    events = EventWriter(folder / "events.csv")
    end = steps[-1].t
    pending: dict[str, list[ImuSample]] = {"acc": [], "gyro": []}
    n = 0
    a_idx = s_idx = 0
    while True:
        rel = n / NOMINAL_RATE_HZ
        if rel > end:
            break
        t = T0 + rel
        while s_idx < len(steps) and steps[s_idx].t <= rel:
            st = steps[s_idx]
            events.write(Event(t, st.kind, st.name, st.column, st.row, st.yaw_deg, st.pitch_deg, st.text))
            s_idx += 1
        while a_idx < len(actions) and actions[a_idx][0] <= rel:
            actions[a_idx][1](t)
            a_idx += 1
        acc, gyro = head.step(t)
        for kind, v in (("acc", acc), ("gyro", gyro)):
            s = ImuSample(t, kind, *map(float, v))
            if duplicates:  # muselsl: a packet's 3 samples get its arrival time
                pending[kind].append(s)
                if len(pending[kind]) == 3:
                    for p in pending[kind]:
                        imu.write(ImuSample(s.t, p.kind, p.x, p.y, p.z))
                    pending[kind] = []
            else:
                imu.write(s)
        n += 1
    imu.close()
    events.close()
    write_meta(folder / "meta.json", {
        "participant": folder.name, "simulated": True, "seed": seed, "duplicates": duplicates,
        "clock_t0": T0, "duration_s": round(end, 1), "headset": "sim", "lever_arm_m": list(lever_arm_m),
        "shake_deg": shake_deg, "turn_dps": turn_dps,
        "true_gyro_bias_dps_at_end": [round(float(b), 4) for b in head.bias],
        "protocol": {"version": 1, "quick": quick, "stand_s": stand_s,
                     "prompt_every_s": protocol.PROMPT_EVERY_S},
        "layout": asdict(layout), "tool": "bench/imu/simulate.py",
    })
    return folder


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="session folder to create")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--duplicates", action="store_true", help="stamp 3 samples alike, like muselsl stream")
    p.add_argument("--no-lever-arm", action="store_true", help="no linear acceleration from head rotation")
    p.add_argument("--shake-deg", type=float, default=14.0,
                   help="shake amplitude (default 14, peaks near 220 deg/s; the gyro clips at 245)")
    p.add_argument("--turn-dps", type=float, default=120.0, help="fastest head turn, deg/s (default 120)")
    args = p.parse_args(argv)
    lever = (0.0, 0.0, 0.0) if args.no_lever_arm else (0.10, 0.0, 0.12)
    folder = simulate(args.out, quick=args.quick, seed=args.seed, duplicates=args.duplicates, lever_arm_m=lever,
                      shake_deg=args.shake_deg, turn_dps=args.turn_dps)
    print(f"wrote {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
