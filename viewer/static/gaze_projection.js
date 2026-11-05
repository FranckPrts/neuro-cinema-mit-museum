// Gaze projection: the visitor's gaze cone drawn onto the wall screens.
//
// Every surface of the room (screens, bare walls, floor, ceiling) gets a
// ShaderMaterial that computes, per pixel, the angle between the ray from the
// eye to that pixel and the gaze direction. The reticle is drawn from that
// angle, so its footprint is the exact cone/plane intersection: a circle
// straight ahead, a stretched ellipse on an oblique wall, continuous across
// screen edges and corners.
//
// All shader maths is in the room frame: metres, y up, origin on the floor
// under the visitor's eyes, x = visitor's right, -z = straight ahead.
// The Three.js scene is centred on the room, so the room origin sits at
// `roomOrigin` in world coordinates and the vertex shader subtracts it.
import * as THREE from "three";

const D2R = Math.PI / 180;

// Shared uniforms (same objects in every material, so one update drives all).
function makeUniforms(roomOrigin, eye, coneDeg) {
  return {
    uOrigin: { value: roomOrigin.clone() },      // world position of the room origin
    uEye: { value: eye.clone() },                // eye position, room frame
    uGazeDir: { value: new THREE.Vector3(0, 0, -1) },  // unit gaze direction, room frame
    uConeAngle: { value: coneDeg * D2R },        // cone half-angle, radians
    uTime: { value: 0 },                         // seconds
    uHighlight: { value: 0 },                    // dwell 0..1 while a target is faced
    uSelected: { value: 0 },                     // 1 at a selection, decays to 0 over 2 s
  };
}

const VERT = /* glsl */ `
  uniform vec3 uOrigin;
  varying vec3 vRoomPos;
  varying vec2 vUv;
  void main() {
    vUv = uv;
    vec4 w = modelMatrix * vec4(position, 1.0);
    vRoomPos = w.xyz - uOrigin;
    gl_Position = projectionMatrix * viewMatrix * w;
  }`;

// Common part: angle from the gaze axis, in units of the cone half-angle.
const CONE = /* glsl */ `
  uniform vec3 uEye;
  uniform vec3 uGazeDir;
  uniform float uConeAngle;
  uniform float uTime;
  uniform float uHighlight;
  uniform float uSelected;
  varying vec3 vRoomPos;
  varying vec2 vUv;

  // r = angle / cone half-angle: 0 on the gaze axis, 1 on the cone edge.
  float coneR() {
    vec3 ray = normalize(vRoomPos - uEye);
    // atan(|a x b|, a.b) stays accurate near 0, unlike acos.
    float ang = atan(length(cross(ray, uGazeDir)), dot(ray, uGazeDir));
    return ang / uConeAngle;
  }
  // Soft reticle: inside-cone fill, ring at the cone edge, centre dot.
  vec3 reticle(float r) {
    float aa = fwidth(r) * 1.5 + 1e-4;
    float fill = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, r);
    float ring = 1.0 - smoothstep(0.02, 0.02 + 2.0 * aa, abs(r - 1.0));
    float dotc = 1.0 - smoothstep(0.07 - aa, 0.07 + aa, r);
    return vec3(fill, ring, dotc);
  }`;

const SCREEN_FRAG = /* glsl */ `
  uniform sampler2D uTex;
  ${CONE}
  void main() {
    float r = coneR();
    vec3 ret = reticle(r);
    vec3 col = texture2D(uTex, vUv).rgb;

    // Gaze-driven content: the picture brightens around the gaze point (more
    // as dwell builds) and a faint ripple runs outward from it.
    float glow = exp(-r * r * 0.25);
    float ripple = sin(r * 6.2832 - uTime * 4.0) * exp(-0.35 * max(r - 1.0, 0.0)) * smoothstep(0.9, 1.4, r);
    col *= 0.45 + (0.55 + 0.5 * uHighlight) * glow + 0.07 * ripple;

    vec3 gold = vec3(0.925, 0.898, 0.847);  // paper
    col = mix(col, gold, ret.x * 0.18);
    col = mix(col, gold, ret.y * 0.9);
    col = mix(col, vec3(1.0), ret.z * 0.9);
    col = mix(col, vec3(1.0), ret.x * uSelected * 0.7);  // selection flash
    gl_FragColor = vec4(col, 1.0);
  }`;

// Bare surfaces (back wall, the 1.44 in gaps, floor, ceiling): nearly
// transparent, with the footprint hatched in red so "off-screen" is obvious.
const BARE_FRAG = /* glsl */ `
  uniform float uBaseAlpha;
  ${CONE}
  void main() {
    float r = coneR();
    vec3 ret = reticle(r);
    float hatch = step(0.5, fract((gl_FragCoord.x + gl_FragCoord.y) / 12.0));
    vec3 red = vec3(0.973, 0.443, 0.443);  // danger
    vec3 col = vec3(0.545, 0.514, 0.467);  // dim
    float a = uBaseAlpha;
    col = mix(col, red, max(ret.x, ret.y));
    a = max(a, ret.x * (0.18 + 0.25 * hatch));
    a = max(a, ret.y * 0.85);
    a = max(a, ret.z * 0.9);
    gl_FragColor = vec4(col, a);
  }`;

// A quad from `origin` spanning `a * la` and `b * lb`; its front face points
// along a x b, which is always into the room here.
function quad(origin, a, b, la, lb, lift) {
  const n = new THREE.Vector3().crossVectors(a, b).normalize();
  const o = origin.clone().addScaledVector(n, lift);
  const p = [o, o.clone().addScaledVector(a, la), o.clone().addScaledVector(a, la).addScaledVector(b, lb),
    o.clone().addScaledVector(b, lb)];
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(p.flatMap((v) => [v.x, v.y, v.z]), 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
  g.setIndex([0, 1, 2, 0, 2, 3]);
  return g;
}

// Test card: 25 cm grid, 1 m grid, circles and diagonals so oblique
// foreshortening is easy to read, wall name and corner labels.
// Same earth tones as the columns in index.html; the back wall in dim.
const WALL_COLOR = { front: "#a89cc8", left: "#d09a6a", right: "#86b39f", back: "#8b8377" };
function testPattern(name, wM, hM) {
  const W = 1536, H = Math.round(W * hM / wM), px = W / wM;
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, W, H);
  grad.addColorStop(0, "#1a1512"); grad.addColorStop(1, "#241e1b");
  g.fillStyle = grad; g.fillRect(0, 0, W, H);
  const line = (x0, y0, x1, y1) => { g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); };
  for (const [step, color, width] of [[0.25, "#2e2723", 1.5], [1, "#5c544b", 3]]) {
    g.strokeStyle = color; g.lineWidth = width;
    for (let x = 0; x <= wM + 1e-6; x += step) line(x * px, 0, x * px, H);
    for (let y = 0; y <= hM + 1e-6; y += step) line(0, H - y * px, W, H - y * px);
  }
  g.strokeStyle = WALL_COLOR[name]; g.lineWidth = 6;
  for (const rr of [0.5, 1.0]) { g.beginPath(); g.arc(W / 2, H / 2, rr * px, 0, 2 * Math.PI); g.stroke(); }
  g.globalAlpha = 0.35; line(0, 0, W, H); line(W, 0, 0, H); g.globalAlpha = 1;
  g.strokeRect(3, 3, W - 6, H - 6);
  g.fillStyle = WALL_COLOR[name]; g.textAlign = "center"; g.textBaseline = "middle";
  g.font = `500 ${Math.round(px * 0.28)}px "DM Mono", ui-monospace, Menlo, monospace`;
  g.fillText(name.toUpperCase(), W / 2, H / 2 - 1.25 * px);
  g.font = `${Math.round(px * 0.1)}px "DM Mono", ui-monospace, Menlo, monospace`;
  g.fillStyle = "#8b8377";
  g.textAlign = "left"; g.fillText("(0,0)", 12, H - 30);
  g.textAlign = "right"; g.fillText("(1,1)", W - 12, 30);
  g.textAlign = "center"; g.fillText(`${wM.toFixed(3)} x ${hM.toFixed(3)} m · 25 cm grid`, W / 2, H - 30);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

export class GazeProjection {
  // `room` is the /config.json "room" object (see config/room.toml).
  constructor(scene, room, roomOrigin, params = new URLSearchParams()) {
    this.cfg = room;
    const R = room.room, S = room.screens;
    this.eye = new THREE.Vector3(0, R.eye_height_m, 0);
    const cone = Number(params.get("cone")) || room.gaze.cone_deg;
    this.uniforms = makeUniforms(roomOrigin, this.eye, cone);
    this.group = new THREE.Group();
    this.group.position.copy(roomOrigin);  // children are built in room coordinates
    scene.add(this.group);

    const hw = R.width_m / 2, front = -(R.depth_m - R.visitor_inset_m), back = R.visitor_inset_m, H = R.ceiling_m;
    const V = (x, y, z) => new THREE.Vector3(x, y, z);
    const up = V(0, 1, 0);
    // Same walls and (u, v) frames as viewer/room_geometry.py.
    this.walls = {
      front: { origin: V(-hw, 0, front), u: V(1, 0, 0), normal: V(0, 0, 1), width: R.width_m },
      right: { origin: V(hw, 0, front), u: V(0, 0, 1), normal: V(-1, 0, 0), width: R.depth_m },
      back: { origin: V(hw, 0, back), u: V(-1, 0, 0), normal: V(0, 0, -1), width: R.width_m },
      left: { origin: V(-hw, 0, back), u: V(0, 0, -1), normal: V(1, 0, 0), width: R.depth_m },
    };
    this.height = H;
    this.inset = S.inset_m;
    this.screenWalls = new Set(S.walls);

    const bare = (alpha) => new THREE.ShaderMaterial({
      uniforms: { ...this.uniforms, uBaseAlpha: { value: alpha } },
      vertexShader: VERT, fragmentShader: BARE_FRAG,
      transparent: true, depthWrite: false, side: THREE.FrontSide,
    });
    // Bare wall surfaces (whole walls: the screens sit 4 mm in front of them).
    for (const w of Object.values(this.walls)) {
      this.group.add(new THREE.Mesh(quad(w.origin, w.u, up, w.width, H, 0.001), bare(0.05)));
    }
    this.group.add(new THREE.Mesh(quad(V(-hw, 0, back), V(1, 0, 0), V(0, 0, -1), R.width_m, R.depth_m, 0.002), bare(0)));
    this.group.add(new THREE.Mesh(quad(V(-hw, H, front), V(1, 0, 0), V(0, 0, 1), R.width_m, R.depth_m, 0.001), bare(0.03)));

    // Screens.
    this.screens = {};
    for (const name of S.walls) {
      const w = this.walls[name], i = S.inset_m;
      const sw = w.width - 2 * i, sh = H - 2 * i;
      const tex = testPattern(name, sw, sh);
      const mat = new THREE.ShaderMaterial({
        uniforms: { ...this.uniforms, uTex: { value: tex } },
        vertexShader: VERT, fragmentShader: SCREEN_FRAG, side: THREE.FrontSide,
      });
      const o = w.origin.clone().addScaledVector(w.u, i).add(V(0, i, 0));
      const mesh = new THREE.Mesh(quad(o, w.u, up, sw, sh, 0.004), mat);
      this.group.add(mesh);
      this.screens[name] = { mesh, mat, pattern: tex, width: sw, height: sh };
    }
    if (params.get("tex")) this.loadTexture(params.get("tex"));
    this.info = null;
    this._buildHud();
  }

  get coneDeg() { return this.uniforms.uConeAngle.value / D2R; }
  set coneDeg(v) { this.uniforms.uConeAngle.value = Math.min(45, Math.max(0.5, v)) * D2R; }

  // Image or video (by extension or MIME type) on every screen; "" = test card.
  loadTexture(url, mime = "") {
    let tex = null;
    if (url) {
      if (/^video\//.test(mime) || /\.(mp4|webm|mov|m4v)(\?|$)/i.test(url)) {
        const v = document.createElement("video");
        Object.assign(v, { src: url, loop: true, muted: true, playsInline: true, crossOrigin: "anonymous" });
        v.play().catch(() => {});
        tex = new THREE.VideoTexture(v);
      } else {
        tex = new THREE.TextureLoader().load(url);
      }
      tex.colorSpace = THREE.SRGBColorSpace;
    }
    for (const s of Object.values(this.screens)) s.mat.uniforms.uTex.value = tex ?? s.pattern;
  }

  // Ray from the eye along the gaze: first surface hit (walls, floor, ceiling).
  intersect(d) {
    const e = this.eye;
    let best = null;
    for (const [name, w] of Object.entries(this.walls)) {
      const denom = d.dot(w.normal);
      if (denom < -1e-9) {
        const t = w.origin.clone().sub(e).dot(w.normal) / denom;
        if (t > 0 && (!best || t < best.t)) best = { t, name };
      }
    }
    if (d.y < -1e-9) { const t = -e.y / d.y; if (!best || t < best.t) best = { t, name: "floor" }; }
    if (d.y > 1e-9) { const t = (this.height - e.y) / d.y; if (!best || t < best.t) best = { t, name: "ceiling" }; }
    const p = e.clone().addScaledVector(d, best.t);
    const out = { surface: best.name, onScreen: false, distance: best.t, point: p.toArray(),
      yaw: Math.atan2(p.x - e.x, -(p.z - e.z)) / D2R,
      pitch: Math.atan2(p.y - e.y, Math.hypot(p.x - e.x, p.z - e.z)) / D2R };
    const w = this.walls[best.name];
    if (w) {
      const i = this.inset, s = this.screens[best.name];
      out.u = p.clone().sub(w.origin).dot(w.u);
      out.v = p.y;
      out.su = (out.u - i) / (w.width - 2 * i);
      out.sv = (out.v - i) / (this.height - 2 * i);
      out.onScreen = !!s && out.su >= 0 && out.su <= 1 && out.sv >= 0 && out.sv <= 1;
    }
    return out;
  }

  // Call every frame. yaw/pitch in degrees (engine convention), time in s.
  update({ yaw, pitch, time, highlight = 0, selected = 0 }) {
    const y = yaw * D2R, p = pitch * D2R;
    const d = new THREE.Vector3(Math.cos(p) * Math.sin(y), Math.sin(p), -Math.cos(p) * Math.cos(y));
    const u = this.uniforms;
    u.uGazeDir.value.copy(d);
    u.uTime.value = time;
    u.uHighlight.value = highlight;
    u.uSelected.value = selected;
    this.info = this.intersect(d);
    window.ncGaze = this.info;  // for tests and the console
    this._hud();
  }

  _buildHud() {
    const hud = document.getElementById("hud");
    const box = document.createElement("div");
    box.id = "gp";
    box.style.cssText = "margin-top:0";
    const rows = [["gpSurf", "gaze hits"], ["gpUV", "u v (m)"], ["gpN", "u v (screen 0-1)"],
      ["gpD", "distance"], ["gpYP", "yaw pitch at hit"], ["gpCone", "cone ½-angle"]];
    box.innerHTML = `<div class="term-rule"></div>` + rows.map(([id, k]) => `<div class="row"><span class="k">${k}</span><span id="${id}">-</span></div>`).join("")
      + `<div class="row"><span class="k">screen content</span><span><a href="#" id="gpLoad" >load file</a>`
      + ` · <a href="#" id="gpReset" >test card</a></span></div>`
      + `<input type="file" id="gpFile" accept="image/*,video/*" style="display:none">`;
    hud.insertBefore(box, document.getElementById("log").previousElementSibling);
    const file = box.querySelector("#gpFile");
    box.querySelector("#gpLoad").onclick = (e) => { e.preventDefault(); file.click(); };
    box.querySelector("#gpReset").onclick = (e) => { e.preventDefault(); this.loadTexture(""); };
    file.onchange = () => { const f = file.files[0]; if (f) this.loadTexture(URL.createObjectURL(f), f.type); };
    addEventListener("keydown", (e) => {
      if (e.key === "[") this.coneDeg -= 1;
      if (e.key === "]") this.coneDeg += 1;
    });
  }

  _hud() {
    const h = this.info, $ = (id) => document.getElementById(id), f = (x, n = 3) => x.toFixed(n);
    let surf;
    if (h.onScreen) surf = `screen ${h.surface}`;
    else if (this.walls[h.surface]) surf = this.screens[h.surface] ? `${h.surface} wall (gap)` : `${h.surface} wall (bare)`;
    else surf = h.surface;
    $("gpSurf").textContent = surf;
    $("gpSurf").style.color = h.onScreen ? "var(--color-success)" : "var(--color-danger)";
    if (h.u !== undefined) {
      $("gpUV").textContent = `${f(h.u)} ${f(h.v)}`;
      $("gpN").textContent = this.screens[h.surface] ? `${f(h.su)} ${f(h.sv)}` : "-";
    } else {
      // Floor / ceiling: room-frame x (right) and forward distance.
      $("gpUV").textContent = `x ${f(h.point[0])} fwd ${f(-h.point[2])}`;
      $("gpN").textContent = "-";
    }
    $("gpD").textContent = `${f(h.distance)} m`;
    $("gpYP").textContent = `${f(h.yaw, 1)} / ${f(h.pitch, 1)}`;
    $("gpCone").textContent = `${f(this.coneDeg, 1)}°  [ ]`;
  }
}
