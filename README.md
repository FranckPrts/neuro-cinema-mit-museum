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

## First session with the real headset

1. `muselsl list` and put the headset's address in `config/local.toml`
   (`[muse] address = "..."`). On macOS this is a CoreBluetooth UUID, not a MAC.
2. Run `nc-engine` and `nc-monitor`. Put the headset on, look at the centre of
   the middle screen and keep still until `calibrated`.
3. **Check the axes.** Turn right: `yaw` must go **positive**. Look up: `pitch`
   must go **positive**. If not, set `imu.invert_yaw` / `imu.invert_pitch`. If
   pitch barely moves when looking up and down, the headset's forward axis is
   not `x`: try the others in `imu.forward_axis` (and recalibrate).
4. Measure the room and set the screen angles in `[targets]`. For a screen
   centre offset sideways by `d` metres at `D` metres from the standing spot,
   the yaw is `atan(d / D)` in degrees. Set `split_pitch_deg` if the screens'
   divider is not at the calibration gaze height.
5. Tune gestures with real visitors (see the checklist below). The
   `nod_ignored` events and the `motion` value in `nc-monitor` help.

## Headset manager and status page (proof of concept)

`source.kind = "muse_direct"` connects to the headset inside the engine
(muselsl's `Muse` class, no `muselsl stream`, no LSL), so the engine can read
the battery and switch headsets without a restart. Staff use a local web page.

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
      watches still (tune `idle_gyro_range_dps` using `motion` in nc-monitor)
- [ ] Recovery: headset switched off and on, out of range and back, muselsl killed, engine killed

## Code map

| Path | What |
|---|---|
| `src/neurocinema/fusion.py` | gravity + gyro → yaw / pitch / roll, bias tracking, calibration |
| `src/neurocinema/gestures.py` | nod and shake detection |
| `src/neurocinema/targets.py` | yaw/pitch → six targets, with hysteresis |
| `src/neurocinema/engine.py` | the state machine and OSC messages |
| `src/neurocinema/app.py` | main loop, status, commands |
| `src/neurocinema/supervisor.py`, `musestream.py` | running and restarting muselsl |
| `src/neurocinema/sources/` | LSL (headset) and simulator inputs |
| `src/neurocinema/ble.py`, `headsets.py` | in-process headset link, registry, swaps, battery (`muse_direct`) |
| `src/neurocinema/statuspage.py`, `static/status.html` | local status page for staff |
| `src/neurocinema/sim/` | virtual head and keyboard control |
| `config/installation.toml` | every setting, documented (mirrors the code defaults) |

## Not done yet

- A4: live-reloading the config while running; a live debug plot.
- A5: OS integration (auto-start, supervisor service, nightly reboot).
- Swapping headsets without a restart: proof of concept (`muse_direct`), not yet tested on hardware.
- Layer C: live EEG during `watching`.
