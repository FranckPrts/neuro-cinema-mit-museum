# Neurocinema: head-gesture engine

Layer A of the Neurocinema installation. A visitor wearing a Muse 2 faces one of
three screens, each split into a **top** and **bottom** half (six targets), and
**nods** to choose. A **head shake** while facing the middle screen re-centres
the system. The engine reads the headset's accelerometer + gyroscope and sends
everything to TouchDesigner over OSC. It recovers from dropouts and crashes by
itself.

```
Muse 2 ──BLE──► muselsl (child process, supervised)
                   │ LSL: ACC + GYRO, 52 Hz
                   ▼
               nc-engine ── fusion ─ gestures ─ targets ─ state machine
                   │  ▲
         OSC :7000 ▼  │ OSC :7001
                TouchDesigner
```

EEG is not used yet (Layer C will add live EEG on a separate `/nc/eeg/...`
namespace; nothing is ever recorded).

## Setup

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[muse,dev]"
pytest                             # about 15 s
```

If `pylsl` cannot find the LSL library, install it (macOS:
`brew install labstreaminglayer/tap/lsl`).

## Running

**Without a headset** (virtual head, keyboard controlled):

```bash
nc-engine --sim
```

Keys: arrows to turn and look up/down, `1 2 3` face left/middle/right, `t b`
top/bottom, `n` nod, `h` shake, `o` take the headset off/put it on, `c` drop
the connection, `q` quit. The virtual head starts idle: move it (or press `o`
twice) to start calibration, then keep still for 2 s.

**With the headset:**

```bash
nc-engine
```

This launches `muselsl stream` itself, finds the LSL streams and restarts
everything as needed. Use `--unmanaged` if something else (BlueMuse, a manual
`muselsl stream --acc --gyro --lsltime`) already publishes the streams.

**See what is sent**, without TouchDesigner:

```bash
nc-monitor
```

Port 7000 can only be bound by one program. To watch alongside
TouchDesigner, add a second destination in `config/local.toml`:

```toml
[osc]
destinations = ["127.0.0.1:7000", "127.0.0.1:7002"]
```

and run `nc-monitor --port 7002`.

## Everything together: engine, camera, viewer

One OSC port can be received by one program, but any number can send to it.
So the engine and the camera bench both send to the viewer:

```bash
nc-engine                                                  # IMU -> /nc/... on 7000
python bench/camera/camera_bench.py --osc 127.0.0.1:7000   # camera -> /nc/cam/... on 7000
python viewer/nc_viewer.py                                 # receives both, http://127.0.0.1:8080
```

To keep TouchDesigner (or `nc-monitor`) on 7000, send the engine to a second
port too (`[osc] destinations = ["127.0.0.1:7000", "127.0.0.1:7003"]`), send
the camera to 7003, and run `python viewer/nc_viewer.py --osc-port 7003`.

In the viewer: white = IMU, gold = camera; the calibration card (top right)
shows what each one was set up with; `r` runs the room check (each backend's
error at each screen). Everything reads `config/room.toml`, so measure the
room there first. See `viewer/README.md`.

## First session with the real headset

1. `muselsl list` and put the headset's address in `config/local.toml`
   (`[muse] address = "..."`). On macOS this is a CoreBluetooth UUID, not a MAC.
2. Run `nc-engine` and `nc-monitor`. Put the headset on, look at the centre of
   the middle screen and keep still until `calibrated`.
3. **Check the axes.** Turn right: `yaw` must go **positive**. Look up: `pitch`
   must go **positive**. If not, set `imu.invert_yaw` / `imu.invert_pitch`. If
   pitch barely moves when looking up and down, the headset's forward axis is
   not `x`: try the others in `imu.forward_axis` (and recalibrate).
4. Measure the room and put it in `config/room.toml` (room size, where the
   visitor stands, eye height, which walls have screens). The engine works out
   the screens' yaws and edges from it (`[targets] from_room = true`), and the
   viewer and camera bench use the same file; `python -m neurocinema.room`
   prints the angles. Set `split_pitch_deg` if the screens' divider is not at
   the calibration gaze height.
5. Tune gestures with real visitors (see the checklist below). The
   `nod_ignored` events and the `motion` value in `nc-monitor` help.

## Status page: calibrate, put down, re-centre

`nc-engine` serves a local page for staff at http://127.0.0.1:8081, whatever
the source (`[status_page]`; `host = "0.0.0.0"` to open it from a tablet). Its
Engine card shows what the engine is doing, when it last calibrated and where
the gyro bias came from, and three buttons:

- **Calibrate now**: the visitor wears the headset, looks at the centre of the
  middle screen and keeps still for 2 s; that gaze becomes straight ahead and
  level. Calibration also starts by itself when the headset is picked up, at
  the first 2 s of stillness, wherever the visitor is looking then. Press it
  when the height looks off.
- **Headset is on its stand**: the engine goes idle. It notices by itself when
  the headset is tilted more than 70° from how it was worn for 3 s, or
  completely still (motion below 0.6 °/s) for 20 s; the card shows both
  numbers live. While idle and still it learns the gyro bias ("On the stand":
  5 s minimum, 10 s better), and the next calibration starts from it.
- **Re-centre**: the visitor faces the middle screen; this resets the
  left/right drift, like a head shake.

The same actions are OSC commands: `/nc/cmd/recalibrate`, `/nc/cmd/idle`,
`/nc/cmd/reset`.

## Headset manager (proof of concept)

`source.kind = "muse_direct"` connects to the headset inside the engine
(muselsl's `Muse` class, no `muselsl stream`, no LSL), so the engine can read
the battery and switch headsets without a restart. The status page then also
shows the headsets, their battery and the swap buttons.

**Try it without hardware** (four simulated headsets, A starts at 22 %):

```bash
nc-engine --sim-headsets
# open http://127.0.0.1:8081, tap "Use headset B"; the fake headset is "lifted" after 3 s
```

**With real headsets**, in `config/local.toml`:

```toml
[source]
kind = "muse_direct"

[headsets]
registry = ["A=Muse-41D2", "B=Muse-0C3A", "C=Muse-....", "D=Muse-...."]  # names from `muselsl list`

[status_page]
host = "0.0.0.0"   # reach it from a tablet on the same network
```

Swap: switch the old headset off, switch a charged one on, tap its letter on
the page, pick it up. The engine only uses a headset once it sees it move, so
the wrong one is never locked in, and headsets outside `registry` are never
connected. Status has three states: all good, swap soon (under 20 % or 30 min
left), offline (no headset in use for 60 s). Settings are in `[headsets]`.

Not verified on hardware yet: muselsl's battery reading (raw / 512) against
the Muse app, how often the Muse sends battery telemetry (the engine asks once
after 60 s if none arrived), and bleak running on a background thread on macOS.
The existing `muse` source (supervised `muselsl stream`) is unchanged and stays
the default.

## Interaction

| State | Meaning | Leaves when |
|---|---|---|
| `disconnected` | no data from the headset | data returns (to `calibrating` if a visitor was mid-session, otherwise `idle`) |
| `idle` | headset not worn (attract mode) | strong motion (someone picks it up), or `/cmd/start` |
| `calibrating` | "look at the middle screen and keep still" | 2 s of stillness |
| `active` | tracking: `/target` and `/dwell` update | nod (`selected`), shake (`resetting`), headset off (`idle`) |
| `selected` | a choice was made; target frozen | after 2 s, back to `active` |
| `resetting` | shake seen; waiting for the head to settle | still for 0.3 s (or 3 s), yaw zeroed, `active` |

- A nod only selects after looking at the target for `min_dwell_s` (0.4 s), and
  only in the `choosing` phase. TouchDesigner sets the phase from the film
  timeline (`/nc/cmd/phase watching|choosing`).
- The selected target is the one faced **when the nod began**.
- Each selection pulls yaw halfway back to that screen's centre, so drift
  rarely builds up. The shake is the visitor's manual fix.
- Pitch (top/bottom) is measured against gravity and does not drift.
- While the headset lies still on its stand (`idle`), the engine learns the
  gyro bias. Calibration starts from it, so a visitor who sways or drifts a
  little during "keep still" doesn't spoil it. It is kept across a dropout of
  the same headset and dropped when the headset changes (`imu.idle_bias_*`).

### Orientation backend (experimental)

`imu.backend = "vqf"` swaps the complementary filter for
[VQF](https://vqf.readthedocs.io), which estimates "up" and the gyro bias with
a different method (accelerometer filtered in a near-inertial frame, Kalman
bias estimate). Only those two estimates change: yaw, gestures, targets and
re-centring are shared. It needs `pip install -e ".[vqf]"`. The complementary
filter stays the default until recordings from the headset show VQF is better
(`bench/imu/`). VQF can only follow a gyro bias up to `imu.vqf_bias_clip_dps`
per axis (10 °/s; headset C has about 6 °/s on one axis). Above it pitch is
off by a few degrees, and the engine logs a warning at calibration.

## OSC reference

Engine → TouchDesigner (default `127.0.0.1:7000`):

| Address | Arguments | When |
|---|---|---|
| `/nc/heartbeat` | uptime_s (float) | 1 Hz, always. No heartbeat for ~3 s means the engine is down. |
| `/nc/status` | connected (0/1), packet_age_ms (int, -1 = never), rate_hz (float), motion_dps (float, -1 = unknown) | 2 Hz |
| `/nc/status/stream` | state (`stopped` `starting` `streaming` `backoff`), consecutive_failures (int) | 2 Hz, headset mode |
| `/nc/status/error` | code (`stream_unrecoverable` / `ok`), message | after repeated stream failures; `ok` once recovered |
| `/nc/headset` | label (string, `""` = none), battery_percent (float, -1 = unknown), minutes_left (float, -1 = unknown), phase (`no_headset` `connecting` `confirming` `active` `reconnecting`), alert (`ok` `swap_soon` `offline`) | 2 Hz, `muse_direct` only |
| `/nc/state` | `disconnected` `idle` `calibrating` `active` `selected` `resetting` | on change, and every second |
| `/nc/phase` | `choosing` / `watching` | on change, and every second |
| `/nc/target` | column (`left` `middle` `right` `none`), row (`top` `bottom` `none`) | on change, and every second |
| `/nc/selected` | column, row | when a nod selects |
| `/nc/dwell` | 0.0 to 1.0 | 30 Hz in `active`; reaches 1.0 when a nod would be accepted |
| `/nc/calibration` | 0.0 to 1.0 | 30 Hz in `calibrating` |
| `/nc/pose` | yaw, pitch, roll (degrees) | 30 Hz while tracking. + = right / up / right ear down |
| `/nc/event` | name [args] | `engine_started`, `engine_stopped`, `calibrated`, `calibration_slow`, `nod`, `shake`, `reset`, `selected <col> <row>`, `nod_ignored <watching\|no_target\|dwell>` |
| `/nc/calib/imu` | backend, bias_source (`idle` `worn` `vqf` `none`), bias_dps (float), since_s (float, -1 = not calibrated), yaw_corrected_deg (float: re-anchoring + shake resets since calibration; large = drift), clipped (int: gyro samples at full scale since calibration) | on calibration, `/cmd/snapshot`, and every 5 s |
| `/nc/calib/room` | from_room (0/1), room_hash (string), left, middle, right screen centre yaws, then column bounds lo, left\|middle, middle\|right, hi (degrees) | same |

TouchDesigner → engine (default `127.0.0.1:7001`):

| Address | Effect |
|---|---|
| `/nc/cmd/start`, `/nc/cmd/recalibrate` | start calibration now |
| `/nc/cmd/reset` | zero yaw now (the visitor faces the middle screen) |
| `/nc/cmd/idle` | back to attract mode |
| `/nc/cmd/phase` `choosing\|watching` | allow or block selections |
| `/nc/cmd/restart_stream` | restart muselsl (not counted as a failure) |
| `/nc/cmd/ping` [args] | replies `/nc/pong` [args] |
| `/nc/cmd/snapshot` | re-send state, phase and target now |

TouchDesigner tips: string messages (`/state`, `/target`, `/event`) need an
**OSC In DAT**; numeric ones (`/pose`, `/dwell`, `/calibration`, `/status`)
work well in an **OSC In CHOP**. `/state`, `/phase` and `/target` repeat every
second, so a TouchDesigner that starts after the engine catches up by itself.

## Recovery

| Failure | What happens |
|---|---|
| Headset drops out / out of range | samples stop → `disconnected` → muselsl is restarted with backoff (2, 4, 8 … 60 s) → on return, the visitor is recalibrated |
| muselsl crashes or stalls | same; its output is in the engine log as `muselsl:` lines |
| Repeated failures (5) | `/nc/status/error stream_unrecoverable` (show "call staff"), optional `muse.escalation_command` (e.g. restart Bluetooth), retries continue |
| Engine crashes | muselsl notices its parent is gone and exits, freeing the headset; on restart, a leftover muselsl in `run/muselsl.pid` is stopped |
| Headset taken off | no motion for 20 s or tilted > 70° for 3 s → `idle`; picking it up starts calibration |

Restarting the engine itself after a crash, starting at login and nightly
reboots are handled by the operating system (milestone A5, once the host OS is
chosen). Logs: `logs/neurocinema.log` (rotated, 5 × 2 MB). No sensor data is
ever written to disk.

## Field-test checklist

- [ ] Axes correct (turn right → yaw +, look up → pitch +)
- [ ] Faced-target accuracy after 5+ minutes without a shake (drift)
- [ ] False selections while visitors talk, laugh and look around
- [ ] Shake reset lands within a few degrees of 0
- [ ] Time from putting the headset on to first selection
- [ ] Tall and short visitors, glasses, long hair, loose fit
- [ ] `idle` triggers when the headset is put down, and not while a visitor
      watches still (tune `idle_gyro_range_dps` using `motion` in nc-monitor
      or the status page's put-down check)
- [ ] Calibrations log `from idle`: the headset rests on its stand, switched
      on, at least 10 s before the next visitor
- [ ] Recovery: headset switched off and on, out of range and back, muselsl killed, engine killed

## Code map

| Path | What |
|---|---|
| `src/neurocinema/fusion.py` | gravity + gyro → yaw / pitch / roll, bias tracking, calibration |
| `src/neurocinema/fusion_vqf.py` | optional VQF backend for "up" and the gyro bias (`imu.backend = "vqf"`) |
| `src/neurocinema/gestures.py` | nod and shake detection |
| `src/neurocinema/targets.py` | yaw/pitch → six targets, with hysteresis |
| `src/neurocinema/room.py` | room and screens from `config/room.toml`: screen yaws for the targets, gaze hits for the viewer |
| `src/neurocinema/engine.py` | the state machine and OSC messages |
| `src/neurocinema/app.py` | main loop, status, commands |
| `src/neurocinema/supervisor.py`, `musestream.py` | running and restarting muselsl |
| `src/neurocinema/sources/` | LSL (headset) and simulator inputs |
| `src/neurocinema/ble.py`, `headsets.py` | in-process headset link, registry, swaps, battery (`muse_direct`) |
| `src/neurocinema/statuspage.py`, `static/status.html` | local status page for staff |
| `src/neurocinema/sim/` | virtual head and keyboard control |
| `bench/imu/` | IMU-only recording and replay, to compare orientation backends on real data |
| `config/installation.toml` | every setting, documented (mirrors the code defaults) |

## Not done yet

- A4: live-reloading the config while running; a live debug plot.
- A5: OS integration (auto-start, supervisor service, nightly reboot).
- Swapping headsets without a restart: proof of concept (`muse_direct`), not yet tested on hardware.
- Layer C: live EEG during `watching`.
