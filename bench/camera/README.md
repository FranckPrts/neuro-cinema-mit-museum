# Camera bench: pick a screen without the IMU

First attempt at using a camera instead of the Muse IMU for screen selection.
The Muse is not used here at all. MediaPipe Pose finds the visitor and
`pointing.py` turns a raised, straight arm into an eye-to-fingertip ray in the
room frame. `viewer/room_geometry.py` (with `config/room.toml`) then says which
screen the ray hits, and `panels.py` says whether it lands in the top or bottom
half. Hold on one screen half for 1.5 s to select it.

```bash
source .venv/bin/activate
pip install -r bench/camera/requirements.txt
python bench/camera/camera_bench.py
```

The first run downloads the pose model (about 9 MB) into `bench/camera/models/`.
On macOS, allow camera access for your terminal app (System Settings > Privacy &
Security > Camera) if the camera doesn't open.

## Trying it at a desk

With the laptop camera in front of you, the room maths assume that you're
standing on the visitor's spot and the camera is on the front (middle) screen.
- Point roughly at the camera: **middle** screen.
- Point well off to your right (more than about 30°): **right** screen. Same on the left.
- Drop your arm: nothing is selected after 0.5 s.

- Point a little above the camera: **top** half. Point a little below: **bottom** half.

Press `h` to switch to head direction and `a` to switch back. Head direction
here comes from the body-pose model (nose against the ears). It's rough and
reads a turned head as less turned than it is. If head aiming matters, the next
step is MediaPipe's Face Landmarker.

## Where the camera is: `camera.toml`

The camera's position and the point it's aimed at go in
[`camera.toml`](camera.toml), in the same room frame as `config/room.toml`:
metres, origin on the floor under the visitor's eyes, x = visitor's right,
y = up, z = behind the visitor (the front wall is at z = −4.42). The file lists
the wall positions and examples (desk, front corners, above the middle screen).

At startup the bench prints the camera's position and the **bearing** and
**tilt** it works out from it. Bearing is the direction from the visitor to the
camera (+ = right), and tilt is how far the camera looks down. The camera is
drawn on the top-down map as a yellow triangle, with a dotted line to its aim
point. The default file is a desk test: a level camera 0.6 m in front of you.

Only the camera's *direction* enters the maths. MediaPipe gives the body's
shape relative to the hips, not where the body is in the room, so the
position is used only to work out which way the camera faces. The camera must
actually be aimed at the `[aim]` point, with no sideways roll. Use
`--camera-config other.toml` to switch setups. `--bearing` / `--tilt` still
override the file.

## Top and bottom halves

The ray starts at the eye and goes through the fingertip, so "up" and "down"
are measured from eye height, not from the shoulder. With the room's current
numbers (10 ft screens, 1.6 m eye height, both still unconfirmed), the divider
on the front screen is about 1° below eye level. The top half covers roughly
0° to +18° of pitch and the bottom half 0° to −19°. The side screens are
closer, so their halves cover wider angles and are easier to hit. An arm held
level with the shoulder points slightly down from the eye, so it lands in the
bottom half.

- A dead band (`--band`, 3° by default) around the divider keeps the half from
  flickering when someone points right at the line.
- `--split 0.5` puts the divider at mid-height. Change it if the content isn't
  split in the middle.
- The window shows a **side view** of the wall being aimed at (above the
  top-down map). It shows the eye, the aim ray (white, smoothed; grey, raw) and
  the two halves.
- **Pitch calibration:** the camera's tilt and MediaPipe's depth estimate can
  bias the pitch. Press `c`, and a 5-second countdown gives you time to step
  into place. Then point at the middle of the middle screen (at a desk,
  straight at the camera) and hold while "HOLD" shows. The pitch is averaged
  over half a second. Press `c` again to cancel, and change the delay with
  `--calib-delay 8`. The offset is printed as `--pitch-offset`, so you can
  reuse it. `x` clears it.
- Head mode doesn't measure pitch, so it only picks the screen, not the half.

## Options

| Flag | What it does |
|---|---|
| `--camera-config corner.toml` | Another camera position file (default `bench/camera/camera.toml`). |
| `--bearing 29.6` `--tilt 15` | Override the direction worked out from the camera file. |
| `--roi x0,y0,x1,y1` | Only look inside this box, in fractions of the frame (the standing zone). Press `r` to draw it with the mouse, and it prints the flag to reuse. |
| `--split 0.5` `--band 3` | Top/bottom divider (fraction of the screen height) and its dead band in degrees. |
| `--no-halves` | Screens only, like the first version. |
| `--pitch-offset 4.5` | Degrees added to the arm's pitch (what `c` prints). |
| `--select head` | Head direction drives the selection instead of the arm. |
| `--dwell 1.5` `--hold 0.25` `--smooth 0.15` | Seconds to select, seconds before a new screen lights up, aim smoothing. |
| `--osc 127.0.0.1:7004` | Sends `/nc/cam/`: `pose`, `target <col> <row>`, `selected`, `dwell`, `state`, `event`, `calib`, `heartbeat`, same arguments as nc-engine's (see `osc_out.py`), to nc-engine's camera port (`camera.listen_port`, 7004). The engine re-publishes them to TouchDesigner and the viewer, blocks selections while watching, and reports `camera_connected` / `camera_lost`. The older `/nc/cam/aim`, `/nc/cam/highlight` and `/nc/cam/select` are still sent but the engine drops them. Off by default. |
| `--log logs/cam.csv` | One CSV row per frame (aim, degrees from the divider, highlight, events), to compare runs. |
| `--camera 1` or `--camera clip.mp4` | Another camera, or replay a video. |
| `--model lite|full|heavy` | Speed against accuracy. `full` is the default. |

## Tests

```bash
python -m pytest bench/camera
```

These test the maths only, with no camera or MediaPipe needed.

## Known limits of this first pass

- One person only (`num_poses=1`). Use `--roi` to keep it on the standing zone.
- MediaPipe's 3D (world) landmarks are estimates from a single camera. Depth
  (toward or away from the camera) is the weakest axis, so pointing straight
  at or straight away from the camera is the least precise.
- Nothing is recorded. Frames stay in memory.
