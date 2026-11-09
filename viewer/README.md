# nc-viewer

A 3D view of the room in the browser: the visitor's head (with the Muse band),
a gaze arrow, the six target regions painted on the walls from
`config/installation.toml` (+ `config/local.toml`), and the floor rays at each
column boundary (faint rays = hysteresis band, yellow = screen centres).
The faced target lights up and fills with dwell; a selection flashes; events
are listed in the panel. Separate from `nc-monitor`, which is unchanged.

```bash
nc-engine --sim                    # or nc-engine with the headset
python viewer/nc_viewer.py         # opens http://127.0.0.1:8080
```

It listens on OSC port 7000, like `nc-monitor`, so only one of them (or
TouchDesigner) can run on that port. To run it alongside, add a destination in
`config/local.toml` and point the viewer at it:

```toml
[osc]
destinations = ["127.0.0.1:7000", "127.0.0.1:7003"]
```

```bash
python viewer/nc_viewer.py --osc-port 7003
```

**Calibration card** (top right): the room in use, then, once the engine
sends `/nc/calib/imu`, the orientation backend, the gyro bias and where it came
from (orange when it came from the 2 s worn mean instead of the stand), time
since calibration, the yaw corrected since then (re-anchoring plus shake
resets; orange above 12°, red above 30° = drift), and gyro samples at full
scale. The dashed grey line is the calibration gaze (yaw 0, pitch 0). When the
camera bench sends to the same port (`camera_bench.py --osc 127.0.0.1:7000`),
the camera appears as a gold cone at its `camera.toml` position, aimed at its
aim point, with its bearing, tilt, eye height and divider in the card. With both streams, the camera's gaze is a gold arrow next to the IMU's white
one, and its target is hatched in gold (a whole column in head mode). The
panel shows the camera's target, the share of time both picked the same target
while both were tracking, and how many times they differed for more than 0.5 s
(each one is logged). `b` makes the visitor's view (`v`) follow the camera
instead of the IMU. A
banner warns when the engine's screen angles differ from the viewer's, or the
camera's eye height differs from `room.toml`.

**Room check** (`r`, once per setup or after moving anything): the viewer
says and shows "look at middle top", then the other five screen halves, then
middle top again (2 s to turn, 2.5 s to hold each). For each, it averages
every running backend's yaw and pitch and subtracts the true direction to the
centre of that half, worked out from `room.toml`. The table shows the errors
(orange above 6°, red above 15°) and the median per backend; the last row,
middle top again, shows drift over the check. `c` copies it as CSV. Heads turn
less than eyes, so a few degrees of yaw error toward the middle is normal for
the IMU; a large error on one screen only usually means a wrong room number.

Press `v` to toggle the visitor's own view. Reload the page after editing the
target angles. Room: 16.5 x 16.5 ft, visitor centred on one edge, 2 ft in,
yaw 0 straight across the room. Ceiling (10 ft) and eye height (1.6 m) are
assumptions at the top of `static/index.html`. Three.js loads from a CDN, so
the page needs internet the first time.

## Gaze projection (proof of concept)

The visitor's gaze is also drawn as a cone footprint on the wall screens
(`static/gaze_projection.js`). Each surface has a `THREE.ShaderMaterial` that
works out, per pixel, the angle between the ray from the eye to that pixel and
the gaze direction. The reticle (gold spot, ring at the cone edge, white
centre dot) is drawn from that angle, so its footprint is the exact cone/wall
intersection: it stretches on oblique walls and carries on across screen
edges and corners. Around the gaze point the picture brightens (more as dwell
builds) and a faint ripple runs outward, to show the uniforms can drive
content. Bare surfaces (the back wall, the 1.44 in gaps around each screen,
floor, ceiling) show the footprint hatched in red, so "off-screen" stands out.

The panel shows what the gaze ray hits: the screen (green) or a gap/bare wall
(red), `u v` in metres from the wall's bottom-left (as seen from inside the room,
facing it), `u v` from 0 to 1 across the screen (outside 0-1 = in the gap),
distance from the eye, and yaw/pitch of the hit point. Floor and ceiling hits
show the room-frame `x` and forward distance.

Controls: `[` / `]` cone half-angle -/+ 1°; "load file" in the panel puts an
image or video on every screen ("test card" goes back to the grid). URL
parameters: `?cone=8`, `?tex=<image or video URL>`, `?view=visitor`.

**Geometry** lives in `config/room.toml`, read by `neurocinema.room` (the
engine uses it too, for the screens' yaws and edges; `room_geometry.py` here
just re-exports it). `python -m neurocinema.room` prints the screen angles and
a check table. Pass another file with `--room`; the engine reads
`[targets] room_file`, so keep the two the same.

| key | default | |
|---|---|---|
| `room.width_m`, `room.depth_m` | 5.0292 (16.5 ft) | |
| `room.ceiling_m` | 3.048 (10 ft) | **unconfirmed** |
| `room.eye_height_m` | 1.6 | **unconfirmed** |
| `room.visitor_inset_m` | 0.6096 (2 ft) | from the back wall, centred left/right |
| `screens.walls` | front, left, right | **unconfirmed**; back wall bare |
| `screens.inset_m` | 0.0366 (1.44 in) | from every wall edge |
| `gaze.cone_deg` | 5 | cone half-angle |

Room frame: metres, y up, origin on the floor under the visitor's eyes, x to
the visitor's right, -z straight ahead (yaw 0). With the defaults the front wall
is 4.420 m away, the side walls 2.515 m to each side, and the front/side corners
are at ±29.64° yaw.

**Uniforms** (all in the room frame, shared by every surface):

| uniform | type | |
|---|---|---|
| `uEye` | vec3 | eye position |
| `uGazeDir` | vec3 | unit gaze direction, from `/nc/pose` yaw and pitch |
| `uConeAngle` | float | cone half-angle, radians |
| `uTex` | sampler2D | screen content (screens only), uv (0,0) = bottom-left |
| `uTime` | float | seconds |
| `uHighlight` | float | `/nc/dwell` while a target is faced, else 0 |
| `uSelected` | float | 1 at `/nc/selected`, fading to 0 over 2 s |
| `uOrigin` | vec3 | world position of the room origin (the vertex shader subtracts it) |

Screenshots: `screenshots/` (room view, visitor's view at a corner, back wall).
