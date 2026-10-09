// Rain or Pumps map UI. Data from simulator/ui/build_ui_data.py, simulator/geo/ and
// simulator/forecast/; What if and Future call the FastAPI service in simulator/api.

const D = "data/";
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const DAY_MS = 86400000;

const S = {
  mode: "replay", day: 0, playing: false, speed: 6, years: new Map(), pending: new Map(),
  grid: null, cellAt: null, today: null, national: null, wells: [], camps: [], campMs: [],
  obsByCamp: [], curCamp: -1, wellIx: new Map(), api: false, pressure: null, distFeatures: [],
};

// ---------------------------------------------------------------- utilities
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const fmtM = (x, d = 2) => (x == null || Number.isNaN(x)) ? "–" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(d)} m`;
const iso = (ms) => new Date(ms).toISOString().slice(0, 10);
const loading = (on, text) => { $("#loading").hidden = !on; if (text) $("#loading-text").textContent = text; };
const getJSON = (p) => fetch(D + p).then((r) => { if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); });
const dayOf = (isoDate) => Math.round((Date.parse(isoDate + "T00:00:00Z") - S.grid.t0) / DAY_MS);

function season(ms) {
  const m = new Date(ms).getUTCMonth() + 1;
  return m >= 6 && m <= 9 ? "south-west monsoon" : m >= 10 ? "post-monsoon" : m <= 2 ? "winter" : "pre-monsoon";
}

// ---------------------------------------------------------------- map
const map = new maplibregl.Map({
  container: "map",
  style: { version: 8, glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf", sources: {},
    layers: [{ id: "bg", type: "background", paint: { "background-color": "#e7e1d3" } }] },
  center: [80.5, 22.5], zoom: 4.1, minZoom: 3.2, maxZoom: 11, pitchWithRotate: false, dragRotate: false,
  maxBounds: [[52, -4], [112, 44]], attributionControl: { compact: true },
});
map.touchZoomRotate.disableRotation();
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

function worldMask(india) {
  const holes = [];
  for (const f of india.features) {
    const g = f.geometry, polys = g.type === "Polygon" ? [g.coordinates] : g.coordinates;
    for (const p of polys) holes.push(p[0].slice().reverse());
  }
  return { type: "Feature", geometry: { type: "Polygon", coordinates: [[[40, -20], [130, -20], [130, 55], [40, 55], [40, -20]], ...holes] } };
}

function stateLabels(districts) {
  const acc = new Map();
  for (const f of districts.features) {
    const g = f.geometry, pts = (g.type === "Polygon" ? g.coordinates[0] : g.coordinates.flatMap((p) => p[0]));
    let x0 = 180, y0 = 90, x1 = -180, y1 = -90;
    for (const [x, y] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const a = (x1 - x0) * (y1 - y0), k = f.properties.state, s = acc.get(k) || { x: 0, y: 0, a: 0 };
    s.x += (x0 + x1) / 2 * a; s.y += (y0 + y1) / 2 * a; s.a += a; acc.set(k, s);
  }
  return { type: "FeatureCollection", features: [...acc].map(([name, s]) => ({
    type: "Feature", properties: { name, area: s.a }, geometry: { type: "Point", coordinates: [s.x / s.a, s.y / s.a] } })) };
}

// rain colour scale, by quantised byte q (mm = e^(q/33) - 1)
const RAIN_LUT = (() => {
  // light rain is a pale wash, heavy rain a saturated plum: it has to read on paper
  const stops = [[0.6, [120, 170, 220, 0]], [2.5, [120, 170, 220, 70]], [15, [62, 128, 196, 140]],
    [35, [38, 92, 168, 180]], [64.5, [58, 58, 150, 205]], [115, [112, 40, 132, 225]], [204.5, [160, 20, 100, 240]], [400, [110, 0, 60, 250]]];
  const lut = new Uint8ClampedArray(256 * 4);
  for (let q = 0; q < 256; q++) {
    const mm = Math.exp(q / 33) - 1;
    let c = [0, 0, 0, 0];
    if (mm >= stops[0][0]) {
      c = stops[stops.length - 1][1];
      for (let i = 1; i < stops.length; i++) if (mm < stops[i][0]) {
        const [a, ca] = stops[i - 1], [b, cb] = stops[i], t = (Math.log(mm) - Math.log(a)) / (Math.log(b) - Math.log(a));
        c = ca.map((v, k) => v + (cb[k] - v) * t); break;
      }
    }
    lut.set(c, q * 4);
  }
  return lut;
})();
const rainSmall = document.createElement("canvas");   // one pixel per IMD cell
const rainCanvas = document.createElement("canvas");  // 4x, smoothed, what the map draws
const UPSCALE = 4;

function damIcon() {
  const c = document.createElement("canvas"), n = 26, x = c.getContext("2d");
  c.width = c.height = n;
  x.beginPath(); x.moveTo(n / 2, 3); x.lineTo(n - 3, n - 4); x.lineTo(3, n - 4); x.closePath();
  x.fillStyle = "rgba(255,253,247,0.95)"; x.strokeStyle = "#4b4437"; x.lineWidth = 2.2; x.fill(); x.stroke();
  return x.getImageData(0, 0, n, n);
}
function panelPad() {
  return innerWidth > 760 ? { left: 380, top: 20, right: 280, bottom: 20 } : { left: 0, top: 60, right: 0, bottom: Math.round(innerHeight * 0.45) };
}

// ---------------------------------------------------------------- rain replay
function yearStartIndex(y) { return Math.round((Date.UTC(y, 0, 1) - S.grid.t0) / DAY_MS); }

function loadYear(y) {
  if (S.years.has(y)) return Promise.resolve(S.years.get(y));
  if (!S.pending.has(y)) {
    S.pending.set(y, fetch(`${D}rain/${y}.bin`).then((r) => r.arrayBuffer()).then((b) => {
      const a = new Uint8Array(b); S.years.set(y, a); S.pending.delete(y); return a;
    }));
  }
  return S.pending.get(y);
}

function dayRow(day) {
  const y = new Date(S.grid.t0 + day * DAY_MS).getUTCFullYear(), a = S.years.get(y);
  if (!a) return null;
  const n = S.grid.n_cells, i = day - yearStartIndex(y);
  return a.subarray(i * n, (i + 1) * n);
}

function drawRain(row) {
  const g = S.grid, W = g.lon.length, H = g.lat.length, ctx = rainSmall.getContext("2d");
  const img = ctx.createImageData(W, H);
  let sum = 0, heavy = 0, max = 0;
  for (let c = 0; c < g.n_cells; c++) {
    const q = row[c], p = ((H - 1 - g.lat_idx[c]) * W + g.lon_idx[c]) * 4;
    img.data[p] = RAIN_LUT[q * 4]; img.data[p + 1] = RAIN_LUT[q * 4 + 1];
    img.data[p + 2] = RAIN_LUT[q * 4 + 2]; img.data[p + 3] = RAIN_LUT[q * 4 + 3];
    const mm = Math.exp(q / g.quant) - 1; sum += mm; if (mm >= 64.5) heavy++; if (mm > max) max = mm;
  }
  ctx.putImageData(img, 0, 0);
  const big = rainCanvas.getContext("2d");
  big.clearRect(0, 0, rainCanvas.width, rainCanvas.height);
  big.imageSmoothingEnabled = true; big.imageSmoothingQuality = "high";
  big.drawImage(rainSmall, 0, 0, rainCanvas.width, rainCanvas.height);
  const src = map.getSource("rain");
  if (src) { src.play(); src.pause(); }      // pause() re-reads the canvas once
  S.today = row; fxDirty = true;
  $("#st-mean").textContent = `${(sum / g.n_cells).toFixed(1)} mm`;
  $("#st-heavy").textContent = heavy.toLocaleString();
  $("#st-max").textContent = `${max.toFixed(0)} mm`;
}

function rainAt(lng, lat) {
  if (!S.today) return 0;
  const g = S.grid, i = Math.round((lat - g.lat[0]) / 0.25), j = Math.round((lng - g.lon[0]) / 0.25);
  if (i < 0 || j < 0 || i >= g.lat.length || j >= g.lon.length) return 0;
  const c = S.cellAt[i * g.lon.length + j];
  return c < 0 ? 0 : Math.exp(S.today[c] / g.quant) - 1;
}

let dayReq = 0;
async function setDay(day) {
  const d = clamp(Math.round(day), 0, S.grid.days - 1), id = ++dayReq;
  const ms = S.grid.t0 + d * DAY_MS, y = new Date(ms).getUTCFullYear();
  S.day = d;
  $("#date-out").textContent = iso(ms);
  $("#season-out").textContent = season(ms);
  $("#day").value = d;
  drawTimeline();
  if (!S.years.has(y)) { loading(true, `Loading ${y} rainfall`); await loadYear(y); loading(false); if (id !== dayReq) return; }
  if (new Date(ms).getUTCMonth() >= 10 && y < 2022) loadYear(y + 1);   // prefetch
  drawRain(dayRow(d));
  setCampaign(lastCampaign(ms));
}

function lastCampaign(ms) {
  let lo = -1;
  for (let i = 0; i < S.campMs.length; i++) if (S.campMs[i] <= ms) lo = i; else break;
  return lo;
}

function setCampaign(c) {
  if (c === S.curCamp || S.mode !== "replay") return;
  S.curCamp = c;
  const vals = c >= 0 ? S.obsByCamp[c] : null;
  let n = 0, rise = 0, fall = 0;
  if (vals) for (const v of vals) if (!Number.isNaN(v)) { n++; v > 0 ? fall++ : rise++; }
  const cm = c >= 0 ? S.camps[c] : null;
  $("#camp-out").textContent = cm
    ? `${cm.season[0] + cm.season.slice(1).toLowerCase()} ${cm.year}: ${n.toLocaleString()} wells read. Colour is the change since each well's previous reading.`
    : "No reading yet.";
  $("#camp-rise").style.flexGrow = rise; $("#camp-fall").style.flexGrow = fall;
  $("#camp-rise-l").textContent = `${rise.toLocaleString()} rose`; $("#camp-fall-l").textContent = `${fall.toLocaleString()} fell`;
  paintWells((i) => (vals ? vals[i] : NaN));
}

// ---------------------------------------------------------------- timeline
const tl = $("#tl"), tlx = tl.getContext("2d");
let tlBg = null;
function buildTimeline() {
  const r = tl.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  tl.width = Math.round(r.width * dpr); tl.height = Math.round(r.height * dpr);
  const W = tl.width, H = tl.height, n = S.national.length;
  tlBg = document.createElement("canvas"); tlBg.width = W; tlBg.height = H;
  const c = tlBg.getContext("2d"), top = 6 * dpr, base = H - 16 * dpr;
  const bins = new Float32Array(W);
  for (let x = 0; x < W; x++) {
    const a = Math.floor(x / W * n), b = Math.max(a + 1, Math.floor((x + 1) / W * n));
    let s = 0; for (let i = a; i < b; i++) s += S.national[i]; bins[x] = s / (b - a);
  }
  const mx = Math.max(...bins);
  c.font = `${10 * dpr}px IBM Plex Mono`; c.textBaseline = "bottom";
  for (let y = 1998; y <= 2022; y++) {
    const x = (Date.UTC(y, 0, 1) - S.grid.t0) / DAY_MS / n * W;
    if (y % 2) { c.fillStyle = "rgba(60,50,30,0.035)"; c.fillRect(x, 0, (365 / n) * W, base); }
    if (y % 4 === 2 && x < W - 34 * dpr) { c.fillStyle = "#7a7364"; c.fillText(String(y), x + 2 * dpr, H - 2 * dpr); }
  }
  const grad = c.createLinearGradient(0, top, 0, base);
  grad.addColorStop(0, "rgba(38,92,168,0.9)"); grad.addColorStop(1, "rgba(38,92,168,0.18)");
  c.beginPath(); c.moveTo(0, base);
  for (let x = 0; x < W; x++) c.lineTo(x, base - Math.sqrt(bins[x] / mx) * (base - top));
  c.lineTo(W, base); c.closePath(); c.fillStyle = grad; c.fill();
  c.strokeStyle = "rgba(60,50,30,0.25)"; c.beginPath(); c.moveTo(0, base + 0.5); c.lineTo(W, base + 0.5); c.stroke();
  drawTimeline();
}
function drawTimeline() {
  if (!tlBg) return;
  const W = tl.width, H = tl.height, dpr = W / Math.max(1, tl.getBoundingClientRect().width), x = S.day / (S.grid.days - 1) * W;
  tlx.clearRect(0, 0, W, H); tlx.drawImage(tlBg, 0, 0);
  tlx.fillStyle = "rgba(180,68,26,0.55)";     // the quarterly well readings
  for (const ms of S.campMs) { const cx = (ms - S.grid.t0) / DAY_MS / (S.grid.days - 1) * W; tlx.fillRect(cx, H - 16 * dpr, 1, 4 * dpr); }
  tlx.fillStyle = "rgba(255,253,247,0.45)"; tlx.fillRect(x, 0, W - x, H - 16 * dpr);
  tlx.fillStyle = "#1d232b"; tlx.fillRect(x - dpr, 0, 2 * dpr, H - 14 * dpr);
  tlx.beginPath(); tlx.arc(x, 5 * dpr, 4 * dpr, 0, Math.PI * 2); tlx.fill();
}

// ---------------------------------------------------------------- playback
let playT = 0, playAcc = 0;
function playLoop(t) {
  if (!S.playing) return;
  const dt = playT ? (t - playT) / 1000 : 0; playT = t;
  playAcc += dt * S.speed;
  if (playAcc >= 1) {
    const step = Math.floor(playAcc); playAcc -= step;
    const next = S.day + step;
    if (next >= S.grid.days) { togglePlay(false); return; }
    const y = new Date(S.grid.t0 + next * DAY_MS).getUTCFullYear();
    if (!S.years.has(y)) { loadYear(y); playAcc = 0; } else setDay(next);
  }
  requestAnimationFrame(playLoop);
}
function togglePlay(on = !S.playing) {
  S.playing = on; playT = 0; playAcc = 0;
  $("#play").classList.toggle("playing", on);
  $("#play").setAttribute("aria-label", on ? "Pause" : "Play");
  if (on) requestAnimationFrame(playLoop);
}

// ---------------------------------------------------------------- falling rain
const fx = $("#rainfx"), fctx = fx.getContext("2d");
const GX = 36, GY = 22, MAX_DROPS = 1400, ALPHA = [0.3, 0.5, 0.7, 0.85];
let drops = [], fxDirty = true, fxGrid = new Float32Array(GX * GY), fxSampledAt = 0;
function sizeFx() { const r = fx.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 2); fx.width = r.width * d; fx.height = r.height * d; fctx.setTransform(d, 0, 0, d, 0, 0); fxDirty = true; }
function sampleFx(w, h) {
  for (let gy = 0; gy < GY; gy++) for (let gx = 0; gx < GX; gx++) {
    const ll = map.unproject([(gx + 0.5) * w / GX, (gy + 0.5) * h * 0.9 / GY]);
    fxGrid[gy * GX + gx] = rainAt(ll.lng, ll.lat);
  }
  fxDirty = false; fxSampledAt = performance.now();
}
function fxLoop() {
  requestAnimationFrame(fxLoop);
  const on = !REDUCED && !document.hidden && S.mode === "replay" && $("#ly-rain").checked && map.getZoom() >= 4.8 && S.today;
  if (!on && !drops.length) return;
  const w = fx.clientWidth, h = fx.clientHeight;
  fctx.clearRect(0, 0, w, h);
  if (on) {
    if (fxDirty && performance.now() - fxSampledAt > 120) sampleFx(w, h);
    const tries = Math.round(60 + (map.getZoom() - 4.8) * 40);
    for (let k = 0; k < tries && drops.length < MAX_DROPS; k++) {
      const c = (Math.random() * GX * GY) | 0, mm = fxGrid[c];
      if (mm > 1 && Math.random() < clamp(mm / 60, 0.02, 1) * 0.5) {
        drops.push({ x: ((c % GX) + Math.random()) * w / GX, y: (((c / GX) | 0) + Math.random()) * h * 0.9 / GY,
          v: 7 + Math.random() * 5 + Math.min(mm, 200) / 25, l: 6 + Math.min(mm, 150) / 10, life: 14 + Math.random() * 18, b: Math.min(3, (mm / 40) | 0) });
      }
    }
  } else drops.length = 0;
  const paths = ALPHA.map(() => new Path2D());
  let n = 0;
  for (const d of drops) {
    d.y += d.v; d.x -= d.v * 0.18; d.life--;
    if (d.life <= 0 || d.y >= h) continue;
    paths[d.b].moveTo(d.x, d.y); paths[d.b].lineTo(d.x + d.l * 0.18, d.y - d.l);
    drops[n++] = d;
  }
  drops.length = n;
  fctx.lineWidth = 1; fctx.lineCap = "round";
  paths.forEach((pth, i) => { fctx.strokeStyle = `rgba(30,70,135,${ALPHA[i] * 0.8})`; fctx.stroke(pth); });
}

// ---------------------------------------------------------------- wells
// one setData per repaint; per-feature state updates were too slow on mode switch
function paintWells(valueOf, scale = 1) {
  const features = [];
  for (let i = 0; i < S.wells.length; i++) {
    const v = valueOf(i);
    if (Number.isNaN(v)) continue;
    const w = S.wells[i];
    features.push({ type: "Feature", id: i, properties: { i, r: v, v: v * scale, a: Math.min(Math.abs(v * scale), 4) },
      geometry: { type: "Point", coordinates: [w.lon, w.lat] } });
  }
  map.getSource("wells")?.setData({ type: "FeatureCollection", features });
}

// ---------------------------------------------------------------- api
async function api(path, opt) {
  const r = await fetch(path, opt);
  if (!r.ok) {
    let m = `${r.status} ${r.statusText}`;
    try { const d = (await r.json()).detail; m = typeof d === "string" ? d : d.map((x) => x.msg).join("; "); } catch { }
    throw new Error(m);
  }
  return r.json();
}
const NO_API = "Start the service to use this: make api, then open http://localhost:8000";
const fmtDate = (d) => new Date(d + "T00:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
const cm = (m) => `${Math.abs(m * 100).toFixed(Math.abs(m) < 0.1 ? 1 : 0)} cm`;

// ---------------------------------------------------------------- place (What if, Future)
// q is what the API gets: {}, {state}, {district, state}, {well} or {lat, lon, radius_km}
S.place = { q: {}, label: "All India" };

function circle(lng, lat, km) {
  const pts = [];
  for (let k = 0; k <= 64; k++) {
    const a = k / 64 * 2 * Math.PI;
    pts.push([lng + km / (111.32 * Math.cos(lat * Math.PI / 180)) * Math.cos(a), lat + km / 110.57 * Math.sin(a)]);
  }
  return { type: "Feature", geometry: { type: "Polygon", coordinates: [pts] } };
}

function placeShape(q) {
  if (q.lat != null) return [circle(q.lon, q.lat, q.radius_km)];
  if (q.well != null) { const w = S.wells[S.wellIx.get(q.well)]; return [circle(w.lon, w.lat, 6)]; }
  const norm = (s) => s.toLowerCase();
  if (q.district) return S.distFeatures.filter((f) => norm(f.properties.name) === norm(q.district) && (!q.state || norm(f.properties.state) === norm(q.state)));
  if (q.state) return S.distFeatures.filter((f) => norm(f.properties.state) === norm(q.state));
  return [];
}

function setPlace(q, label, { fly = false, name = null } = {}) {
  S.place = { q, label, name };
  $("#pl-now").textContent = label;
  const shape = placeShape(q);
  map.getSource("place")?.setData({ type: "FeatureCollection", features: shape });
  if (fly && shape.length) {
    const c = shape.flatMap((f) => f.geometry.type === "Polygon" ? f.geometry.coordinates[0] : f.geometry.coordinates.flatMap((p) => p[0]));
    const xs = c.map((p) => p[0]), ys = c.map((p) => p[1]);
    map.fitBounds([[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]], { padding: 80, maxZoom: 8.5, duration: REDUCED ? 0 : 1000 });
  }
  if (S.mode === "forecast") runForecast();
  if (S.mode === "whatif") { wiKey = ""; $("#wi-result").hidden = true; paintWells(() => NaN); }
}

// ---------------------------------------------------------------- search
// states, districts and towns (places.json); a town resolves to its district
const fold = (t) => t.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
let SEARCH = [], qItems = [], qSel = -1;

function buildSearch(towns) {
  const n = new Map();
  for (const w of S.wells) { n.set(`${w.district}|${w.state}`, (n.get(`${w.district}|${w.state}`) || 0) + 1); n.set(w.state, (n.get(w.state) || 0) + 1); }
  S.wellsIn = (k) => n.get(k) || 0;
  const states = [...new Set(S.distFeatures.map((f) => f.properties.state))];
  SEARCH = [
    ...states.map((st) => ({ kind: "state", name: st, state: st, keys: [fold(st)], order: 0 })),
    ...S.distFeatures.map((f, id) => ({ kind: "district", name: f.properties.name, state: f.properties.state, id, keys: [fold(f.properties.name)], order: 1 })),
    ...towns.map(([name, alts, district, state, lat, lon, pop]) => ({ kind: "town", name, alts, district, state, lat, lon, pop, keys: [fold(name), ...alts.map(fold)], order: 2 })),
  ];
}

function findPlaces(q) {
  const f = fold(q);
  if (f.length < 2) return [];
  const hits = [];
  for (const e of SEARCH) {
    let best = -1, via = 0;
    e.keys.forEach((k, i) => {
      const sc = k === f ? 3 : k.startsWith(f) ? 2 : k.includes(" " + f) ? 1 : -1;
      if (sc > best || (sc === best && i === 0)) { best = sc; via = i; }
    });
    if (best >= 0) hits.push({ e, best, via });
  }
  hits.sort((a, b) => b.best - a.best || a.e.order - b.e.order || (b.e.pop || 0) - (a.e.pop || 0));
  // drop a town when its district of the same name is already listed
  const seen = new Set(), out = [];
  for (const h of hits) {
    const k = h.e.kind === "state" ? h.e.name : `${fold(h.e.kind === "town" ? h.e.district : h.e.name)}|${h.e.state}|${fold(h.e.name)}`;
    if (seen.has(k)) continue;
    seen.add(k); out.push(h);
    if (out.length === 8) break;
  }
  return out;
}

const esc = (t) => t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function showSuggestions() {
  const q = $("#q").value;
  qItems = findPlaces(q); qSel = qItems.length ? 0 : -1;
  const list = $("#q-list");
  if (fold(q).length < 2) { list.hidden = true; $("#q").setAttribute("aria-expanded", "false"); return; }
  list.innerHTML = qItems.length ? qItems.map(({ e, via }, i) => {
    const wells = e.kind === "state" ? S.wellsIn(e.state) : S.wellsIn(`${e.kind === "town" ? e.district : e.name}|${e.state}`);
    const what = e.kind === "state" ? "state" : e.kind === "district" ? `district · ${e.state}` : `in ${e.district} district, ${e.state}`;
    const also = e.kind === "town" && via > 0 ? ` <span class="via">(${esc(e.alts[via - 1])})</span>` : "";
    return `<li role="option" id="q-o${i}" aria-selected="${i === qSel}" data-i="${i}"><span>${esc(e.name)}${also}<span class="what">${esc(what)}</span></span>` +
      `<span class="n">${wells ? `${wells} well${wells === 1 ? "" : "s"}` : "no wells"}</span></li>`;
  }).join("") : `<li aria-disabled="true"><span class="st">Nothing called “${esc(q)}” in India's towns, districts or states</span></li>`;
  list.hidden = false; $("#q").setAttribute("aria-expanded", "true");
}

function districtId(name, state) {
  return S.distFeatures.findIndex((f) => f.properties.name === name && f.properties.state === state);
}

// districts with no wells (Jaipur, Hyderabad, Kolkata...) use the smallest circle
// around the town that has at least 3 wells
const RADII = [25, 50, 100, 150, 200, 300];
function nearestRadius(lat, lon, need = 3) {
  const km = S.wells.map((w) => {
    const dl = (w.lat - lat) * Math.PI / 180, dg = (w.lon - lon) * Math.PI / 180;
    const a = Math.sin(dl / 2) ** 2 + Math.cos(lat * Math.PI / 180) * Math.cos(w.lat * Math.PI / 180) * Math.sin(dg / 2) ** 2;
    return 12742 * Math.asin(Math.sqrt(a));
  });
  for (const r of RADII) { const n = km.filter((d) => d <= r).length; if (n >= need) return [r, n]; }
  return [RADII.at(-1), km.filter((d) => d <= RADII.at(-1)).length];
}
function goTo(e) {
  $("#q-list").hidden = true; $("#q").setAttribute("aria-expanded", "false");
  $("#q").value = e.kind === "town" ? `${e.name}, ${e.district}` : e.name;
  $("#q").blur();
  if (e.kind === "state") { setPlace({ state: e.state }, e.state, { fly: true }); return; }
  const dname = e.kind === "town" ? e.district : e.name, id = districtId(dname, e.state);
  const label = e.kind === "town" ? `${e.name} → ${dname} district, ${e.state}` : `${dname}, ${e.state}`;
  if (S.wellsIn(`${dname}|${e.state}`)) setPlace({ district: dname, state: e.state }, label, { fly: true, name: `${dname}, ${e.state}` });
  else {
    let lat = e.lat, lon = e.lon;
    if (lat == null) {      // district picked directly: centre on its namesake town if there is one
      const t = SEARCH.find((x) => x.kind === "town" && x.district === e.name && x.state === e.state && fold(x.name) === fold(e.name));
      if (t) { lat = t.lat; lon = t.lon; }
    }
    if (lat == null) {
      const g = S.distFeatures[id].geometry, c = g.type === "Polygon" ? g.coordinates[0] : g.coordinates.flatMap((x) => x[0]);
      lon = (Math.min(...c.map((x) => x[0])) + Math.max(...c.map((x) => x[0]))) / 2; lat = (Math.min(...c.map((x) => x[1])) + Math.max(...c.map((x) => x[1]))) / 2;
    }
    const [r, n] = nearestRadius(lat, lon);
    $("#pl-radius").value = r;
    setPlace({ lat: +lat.toFixed(4), lon: +lon.toFixed(4), radius_km: r }, `${label}: no wells there, so the ${n} within ${r} km`,
      { name: `the area around ${e.name}` });
    map.flyTo({ center: [lon, lat], zoom: r <= 50 ? 7.4 : r <= 150 ? 6.4 : 5.6, duration: REDUCED ? 0 : 1000 });
  }
  if (S.mode === "pressure" && id >= 0) showDistrict(id);
}

// ---------------------------------------------------------------- charts
// series: {name, color, pts: [[t, y]], band?: [[t, lo, hi]], dash?, dots?}; invert for depth
function bandChart(series, { invert = false, zero = false, unit = "m", shadeFrom = null, marker = null, yfmt = (v) => v.toFixed(1) } = {}) {
  const W = 330, H = 170, P = { l: 38, r: 8, t: 10, b: 20 };
  const ys = series.flatMap((s) => [...s.pts.map((p) => p[1]), ...(s.band || []).flatMap((b) => [b[1], b[2]])]);
  const xs = series.flatMap((s) => s.pts.map((p) => p[0]));
  if (!xs.length) return "";
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (zero) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
  const pad = Math.max((y1 - y0) * 0.08, 0.02); y0 -= pad; y1 += pad;
  if (x1 === x0) x1 = x0 + DAY_MS * 90;
  const X = (x) => P.l + (x - x0) / (x1 - x0) * (W - P.l - P.r);
  const Y = (y) => invert ? P.t + (y - y0) / (y1 - y0) * (H - P.t - P.b) : H - P.b - (y - y0) / (y1 - y0) * (H - P.t - P.b);
  const xy = (x, y) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`;
  let svg = "";
  if (shadeFrom != null && shadeFrom < x1) svg += `<rect x="${X(Math.max(shadeFrom, x0))}" y="${P.t}" width="${X(x1) - X(Math.max(shadeFrom, x0))}" height="${H - P.t - P.b}" fill="url(#hatch)"/><text x="${X(x1) - 2}" y="${P.t + 10}" text-anchor="end" fill="#8a8270">untested</text>`;
  svg += [y0 + pad, (y0 + y1) / 2, y1 - pad].map((v) => `<text x="${P.l - 6}" y="${Y(v) + 3}" text-anchor="end">${yfmt(v)}</text><line x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}" class="g"/>`).join("");
  const yr0 = new Date(x0).getUTCFullYear(), yr1 = new Date(x1).getUTCFullYear(), every = Math.max(1, Math.ceil((yr1 - yr0) / 5));
  for (let y = yr0 + 1; y <= yr1; y += every) { const t = Date.UTC(y, 0, 1); svg += `<text x="${X(t)}" y="${H - 4}" text-anchor="middle">${y}</text><line x1="${X(t)}" x2="${X(t)}" y1="${H - P.b}" y2="${H - P.b + 3}" stroke="#8a8270"/>`; }
  if (zero) svg += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="#8a8270" stroke-dasharray="2 3"/>`;
  if (marker != null) svg += `<line x1="${X(marker)}" x2="${X(marker)}" y1="${P.t}" y2="${H - P.b}" stroke="#1d232b" stroke-dasharray="3 3"/>`;
  for (const s of series) {
    if (s.band?.length > 1) svg += `<path d="M${s.band.map((b) => xy(b[0], b[1])).join(" L")} L${s.band.slice().reverse().map((b) => xy(b[0], b[2])).join(" L")}Z" fill="${s.color}" fill-opacity=".14"/>`;
    if (s.pts.length > 1) svg += `<polyline fill="none" stroke="${s.color}" stroke-width="1.8" stroke-linejoin="round" ${s.dash ? 'stroke-dasharray="4 3"' : ""} points="${s.pts.map((p) => xy(p[0], p[1])).join(" ")}"/>`;
    if (s.dots || s.pts.length === 1) svg += s.pts.map((p) => `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="2" fill="${s.color}"/>`).join("");
  }
  const key = series.map((s, i) => `<tspan fill="${s.color}" dx="${i ? 10 : 0}">— ${s.name}</tspan>`).join("");
  return `<svg viewBox="0 0 ${W} ${H + 14}" font-family="IBM Plex Mono" font-size="10" fill="#7a7364"><defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="rgba(60,50,30,.03)"/><line x1="0" y1="0" x2="0" y2="6" stroke="rgba(60,50,30,.09)" stroke-width="2"/></pattern></defs><style>.g{stroke:rgba(60,50,30,.1)}</style>${svg}<text x="${P.l}" y="${H + 12}">${key}<tspan dx="8">(${unit})</tspan></text></svg>`;
}
const tms = (d) => Date.parse(d + "T00:00:00Z");
const placeName = (r) => S.place.name || r.place;

// ---------------------------------------------------------------- what if
// the API returns depth effects (positive = deeper); the page talks about level, so flip
let wiKey = "", wiRun = 0, wiLast = null;
function wiBody() {
  const kind = $('input[name="wi-kind"]:checked').value, start = $("#wi-start").value;
  const days = +$("#wi-days").value;
  const end = start ? new Date(tms(start) + (days - 1) * DAY_MS).toISOString().slice(0, 10) : start;
  return { ...S.place.q, start, end, rain_pct: kind === "pct" ? +$("#wi-pct").value : 0, add_mm: kind === "mm" ? +$("#wi-mm").value : 0 };
}
function wiLabel(b) {
  if (b.add_mm) return `${b.add_mm} mm more rain a day`;
  return b.rain_pct === -100 ? "no rain at all" : `${Math.abs(b.rain_pct)}% ${b.rain_pct > 0 ? "more" : "less"} rain`;
}
function whenLabel(b) {
  return b.start === b.end ? fmtDate(b.start) : `${fmtDate(b.start)} to ${fmtDate(b.end)}`;
}
async function runWhatIf() {
  const res = $("#wi-result");
  if (!S.api) { res.hidden = false; $("#wi-big").textContent = "–"; $("#wi-sub").textContent = NO_API; return; }
  const body = wiBody();
  if (!body.start) { res.hidden = false; $("#wi-big").textContent = "–"; $("#wi-sub").textContent = "Pick a date first."; return; }
  if (!body.add_mm && !body.rain_pct) { res.hidden = false; $("#wi-big").textContent = "no change"; $("#wi-sub").textContent = "Pick how much rain to add or take away."; return; }
  const key = JSON.stringify(body), id = ++wiRun;
  if (key === wiKey && wiLast) { renderWhatIf(wiLast, body); return; }
  const future = body.end > "2022-12-31" || body.start > "2020-11-15";
  $("#wi-run").disabled = true; res.classList.add("pending");
  loading(true, future ? "Working it out for 23 kinds of rain year" : "Working it out");
  try {
    const r = await api("api/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: key });
    if (id !== wiRun) return;
    wiKey = key; wiLast = r; renderWhatIf(r, body);
  } catch (e) {
    if (id !== wiRun) return;
    res.hidden = false; $("#wi-big").textContent = "–"; $("#wi-arrow").className = "arrow"; $("#wi-sub").textContent = e.message;
    $("#wi-chart").innerHTML = ""; $("#wi-kv").innerHTML = "";
  } finally {
    if (id === wiRun) { loading(false); $("#wi-run").disabled = false; res.classList.remove("pending"); }
  }
}
function renderWhatIf(r, body) {
  $("#wi-result").hidden = false;
  if (!r.peak) { $("#wi-big").textContent = "–"; $("#wi-sub").textContent = "No well reading falls within two years of those days."; return; }
  const e = r.peak.depth_effect_m, past = r.past, rise = -e.p50, none = Math.abs(rise) < 0.0005;
  $("#wi-big").textContent = none ? "no change" : `${rise > 0 ? "rises" : "falls"} ${cm(rise)}`;
  $("#wi-arrow").className = "arrow" + (none ? "" : rise > 0 ? " up" : " down");
  $("#wi-sub").textContent = none ? `With ${wiLabel(body)} on ${whenLabel(body)}, the wells in ${placeName(r)} barely move.`
    : `With ${wiLabel(body)} on ${whenLabel(body)}, the water level in ${placeName(r)} ${rise > 0 ? "rises" : "falls"} by about ${cm(rise)} at the ${fmtDate(r.peak.date)} well reading` +
      (past ? ", compared with the rain that actually fell." : `. Depending on the rain around it: ${cm(-e.p90)} to ${cm(-e.p10)}.`);
  const pts = r.readings.map((x) => [tms(x.date), -x.depth_effect_m.p50 * 100]);
  const band = past ? null : r.readings.map((x) => [tms(x.date), -x.depth_effect_m.p90 * 100, -x.depth_effect_m.p10 * 100]);
  $("#wi-chart").innerHTML = bandChart([{ name: "water level change", color: "#265ca8", pts, band, dots: true }],
    { zero: true, unit: "cm, up = rises", marker: tms(body.start), yfmt: (v) => v.toFixed(Math.abs(v) < 2 ? 1 : 0) });
  $("#wi-chart").setAttribute("aria-label", `Water level change at ${r.readings.length} well readings after the rain; largest ${cm(rise)} on ${r.peak.date}`);
  const c = r.rain_context;
  const kv = [["Season", c.season], ["Rain added", `${c.added_mm >= 0 ? "" : "−"}${Math.abs(c.added_mm).toLocaleString()} mm`],
    ["Usual rain on those days", `${c.normal_mm.toLocaleString()} mm`]];
  if (c.recorded_mm != null) kv.push(["Rain that actually fell", `${c.recorded_mm.toLocaleString()} mm`]);
  if (c.added_vs_normal != null) kv.push(["Added, against usual", `${c.added_vs_normal}×`]);
  kv.push(["Wells used", r.wells.toLocaleString()], ["Readings affected", r.readings.length]);
  if (!past) kv.push(["Rain years tried", r.members]);
  $("#wi-kv").innerHTML = kv.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  const by = new Map(r.per_well_at_peak.map((w) => [w.index, w.depth_effect_m]));
  wiPaint = () => paintWells((i) => (by.has(i) ? by.get(i) : NaN), 10);
  wiPaint();
}
let wiPaint = null;

// ---------------------------------------------------------------- future
const level = (depthChange) => (Math.abs(depthChange) < 0.005 ? "no change" : `${depthChange > 0 ? "falls" : "rises"} ${Math.abs(depthChange).toFixed(2)} m`);
let fcRun = 0, fcKey = "", fcLast = null;
async function runForecast() {
  const to = +$("#fc-year").value, pct = +$("#fc-rain").value;
  $("#fc-year-out").textContent = to;
  $("#fc-k").textContent = `Water level in November ${to}, compared with November 2022`;
  if (!S.api) { $("#fc-big").textContent = "–"; $("#fc-sub").textContent = NO_API; return; }
  const q = new URLSearchParams({ to_year: to, rain_pct: pct, ...Object.fromEntries(Object.entries(S.place.q).filter(([, v]) => v != null)) });
  const key = q.toString(), id = ++fcRun;
  if (key === fcKey && fcLast) { renderForecast(fcLast); return; }
  $("[data-panel='forecast'] .result").classList.add("pending");
  try {
    const r = await api(`api/forecast?${key}`);
    if (id !== fcRun) return;
    fcKey = key; fcLast = r; renderForecast(r);
  } catch (e) {
    if (id !== fcRun) return;
    $("#fc-big").textContent = $("#fc-trend").textContent = "–"; $("#fc-sub").textContent = e.message; $("#fc-chart").innerHTML = "";
  } finally {
    if (id === fcRun) $$(".pending").forEach((el) => el.classList.remove("pending"));
  }
}
function renderForecast(r) {
  const s = r.summary, last = r.forecast.at(-1), d0 = r.start.depth_m;
  const showRain = $("#fc-show-rain").checked, showTrend = $("#fc-show-trend").checked;
  $("#fc-big").textContent = level(s.rain_only_change_m);
  $("#fc-trend").textContent = level(s.rain_plus_trend_change_m);
  $(".ans-rain").classList.toggle("off", !showRain); $(".ans-trend").classList.toggle("off", !showTrend);
  const lo = level(last.rain_only.p10 - d0), hi = level(last.rain_only.p90 - d0);
  $("#fc-sub").textContent = `${placeName(r)}, ${r.wells.toLocaleString()} well${r.wells === 1 ? "" : "s"}. ` +
    `Rain alone: ${lo === hi ? lo : `${lo} to ${hi}`}, depending on the years. ` +
    (last.beyond_tested_skill ? "This far ahead, trust the direction more than the number." : "");
  const series = [{ name: "measured", color: "#1d232b", pts: r.history.map((h) => [tms(h.date), h.depth_m]), dots: true }];
  const start = [tms(r.start.date), d0];
  if (showRain) series.push({ name: "rain only", color: "#265ca8", pts: [start, ...r.forecast.map((x) => [tms(x.date), x.rain_only.p50])],
    band: [[start[0], d0, d0], ...r.forecast.map((x) => [tms(x.date), x.rain_only.p10, x.rain_only.p90])] });
  if (showTrend) series.push({ name: "with pumping", color: "#b4441a", dash: true, pts: [start, ...r.forecast.map((x) => [tms(x.date), x.rain_plus_trend.p50])],
    band: [[start[0], d0, d0], ...r.forecast.map((x) => [tms(x.date), x.rain_plus_trend.p10, x.rain_plus_trend.p90])] });
  const untested = r.forecast.find((x) => x.beyond_tested_skill);
  $("#fc-chart").innerHTML = bandChart(series, { invert: true, unit: "m deep", marker: start[0], shadeFrom: untested ? tms(untested.date) : null });
  $("#fc-chart").setAttribute("aria-label", `Water level, measured to 2022 and forecast to ${r.to_year}: rain only ${level(s.rain_only_change_m)}, with recent pumping ${level(s.rain_plus_trend_change_m)}`);
  const by = new Map(r.per_well.map((w) => [w.index, w.change_m]));
  fcPaint = () => paintWells((i) => (by.has(i) ? by.get(i) : NaN));
  fcPaint();
}
let fcPaint = null;

// ---------------------------------------------------------------- pumping hotspots
let prKey = "";
let prWorst = true;
function renderPressure() {
  const a = Math.min(+$("#pr-from").value, +$("#pr-to").value), b = Math.max(+$("#pr-from").value, +$("#pr-to").value);
  const key = `${a}-${b}-${prWorst}`;
  $("#pr-warn").hidden = a >= 2015;
  $$("[data-years]").forEach((c) => c.setAttribute("aria-checked", String(c.dataset.years === `${a},${b}`)));
  if (prKey === key) return;
  prKey = key;
  const rank = [];
  S.distFeatures.forEach((f, id) => {
    const rec = S.pressure[`${f.properties.name}|${f.properties.state}`];
    let s = 0, k = 0;
    if (rec) for (let y = a; y <= b; y++) if (rec[y]) { s += rec[y][0]; k++; }
    const v = k ? s / k : null;
    map.setFeatureState({ source: "districts", id }, { p: v });
    if (v != null && k >= Math.min(3, b - a + 1)) rank.push([f.properties.name, f.properties.state, v, id]);
  });
  rank.sort((x, y) => prWorst ? y[2] - x[2] : x[2] - y[2]);
  const top = rank.slice(0, 10), mx = Math.max(0.01, ...top.map((r) => Math.abs(r[2])));
  $("#pr-title").textContent = prWorst ? `Falling most beyond rain, ${a}–${b}` : `Holding up best against rain, ${a}–${b}`;
  $("#pr-rank").classList.toggle("better", !prWorst);
  $("#pr-rank").innerHTML = top.map(([n, s, v, id]) =>
    `<li tabindex="0" data-id="${id}"><span class="nm">${n} <span class="st">${s}</span></span>` +
    `<span class="val">${Math.abs(v).toFixed(2)} m ${v > 0 ? "more fall" : "less fall"}</span>` +
    `<span class="bar-in" style="width:${Math.max(4, Math.abs(v) / mx * 100)}%"></span></li>`).join("");
}

// ---------------------------------------------------------------- detail panel
function openDetail() {
  $("#detail").hidden = false;
  $("#detail").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "nearest" });
}

function showWell(i) {
  const w = S.wells[i], hs = w.h;
  const depth = hs.map((h) => [S.campMs[h[0]], h[1]]);
  const mObs = hs.reduce((s, h) => s + h[2], 0) / hs.length, mExp = hs.reduce((s, h) => s + h[3], 0) / hs.length;
  $("#detail-kind").textContent = "Well";
  $("#detail-title").textContent = w.district;
  $("#detail-sub").textContent = `${w.state} · ${w.lat.toFixed(3)}°N ${w.lon.toFixed(3)}°E · depth to water below ground; down is deeper`;
  $("#detail-chart").innerHTML = bandChart([{ name: "depth to water", color: "#265ca8", pts: depth }], { invert: true, unit: "m" });
  $("#detail-chart").setAttribute("aria-label", `Depth to water over ${hs.length} readings, from ${depth[0][1]} m to ${depth.at(-1)[1]} m`);
  const kv = [["Readings", hs.length], ["Aquifer", w.aquifer], ["Well type", w.type], ["Well depth", w.well_depth_m != null ? `${w.well_depth_m} m` : "–"],
    ["Specific yield", w.sy ?? "–"], ["Normal annual rain", w.rain_normal_annual_mm != null ? `${w.rain_normal_annual_mm.toLocaleString()} mm` : "–"],
    ["Mean change per reading", fmtM(mObs)], ["Rain-expected", fmtM(mExp)], ["Unexplained", fmtM(mObs - mExp)]];
  $("#detail-kv").innerHTML = kv.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  openDetail();
}

function showDistrict(id) {
  const f = S.distFeatures[id], rec = S.pressure[`${f.properties.name}|${f.properties.state}`];
  $("#detail-kind").textContent = "District";
  $("#detail-title").textContent = f.properties.name;
  if (!rec) { $("#detail-sub").textContent = `${f.properties.state} · not enough monitored wells`; $("#detail-chart").innerHTML = ""; $("#detail-kv").innerHTML = ""; openDetail(); return; }
  const ys = Object.keys(rec).map(Number).sort();
  const pt = (k) => ys.map((y) => [Date.UTC(y, 6, 1), rec[y][k]]);
  S.prDistrict = id;
  const a = +$("#pr-from").value, b = +$("#pr-to").value, span = a === b ? `In ${a}` : `In ${a}–${b}`;
  const held = ys.filter((y) => y >= a && y <= b), all = held.map((y) => rec[y][0]), mean = all.length ? all.reduce((x, y) => x + y, 0) / all.length : NaN;
  $("#detail-sub").textContent = `${f.properties.state} · water fall per well reading, each year. ` + (Number.isNaN(mean) ? `No readings in ${a}–${b}.` :
    mean > 0.02 ? `${span} it fell ${Math.abs(mean).toFixed(2)} m more per reading than rain explains.` :
    mean < -0.02 ? `${span} it fell ${Math.abs(mean).toFixed(2)} m less per reading than rain explains.` : `${span} it moved about as rain explains.`);
  $("#detail-chart").innerHTML = bandChart([
    { name: "measured", color: "#b4441a", pts: pt(1) },
    { name: "expected from rain", color: "#265ca8", pts: pt(2), dash: true }], { zero: true, unit: "m fall", yfmt: (v) => v.toFixed(2) });
  $("#detail-kv").innerHTML = [["Years with data", ys.length], ["Wells (latest year)", rec[ys.at(-1)][3]], [`Fall beyond rain, ${a}–${b}`, Number.isNaN(mean) ? "–" : fmtM(mean)]]
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  openDetail();
}

// ---------------------------------------------------------------- modes and layers
const VIS = { replay: ["rain", "wells"], whatif: ["wells", "place"], forecast: ["wells", "place"], pressure: ["district-fill"] };
const WELL_LEGEND = {
  replay: ["Water level at the last well reading", ["rose 3 m", "no change", "fell 3 m"]],
  whatif: ["Water level change from the extra rain", ["rises 30 cm", "none", "falls 30 cm"]],
  forecast: ["Water level by the chosen year, if only rain mattered", ["rises 3 m", "no change", "falls 3 m"]],
};
function setMode(mode) {
  S.mode = mode;
  $$(".modes button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  $$(".mode-panel").forEach((p) => (p.hidden = p.dataset.panel !== mode));
  $("#detail").hidden = true;
  if (mode !== "replay") togglePlay(false);
  history.replaceState(null, "", `#${mode}`);
  const usesPlace = mode === "whatif" || mode === "forecast";
  $("#place").hidden = !usesPlace;
  if (usesPlace) $(`[data-panel="${mode}"] .place-slot`).append($("#place"));
  $('[data-lg="rain"]').hidden = mode !== "replay";
  $('[data-lg="wells"]').hidden = mode === "pressure";
  $('[data-lg="pressure"]').hidden = mode !== "pressure";
  $('[data-lg="place"]').hidden = !usesPlace;
  if (WELL_LEGEND[mode]) {
    $("#lg-wells-title").textContent = WELL_LEGEND[mode][0];
    $("#lg-wells-labels").innerHTML = WELL_LEGEND[mode][1].map((t) => `<span>${t}</span>`).join("");
  }
  applyLayers();
  if (mode === "replay") { S.curCamp = -2; setCampaign(lastCampaign(S.grid.t0 + S.day * DAY_MS)); }
  if (mode === "whatif") { if (wiPaint && wiKey) wiPaint(); else paintWells(() => NaN); }
  if (mode === "forecast") runForecast();
  if (mode === "pressure") renderPressure();
}
function vis(id, on) { if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none"); }
function applyLayers() {
  const m = VIS[S.mode];
  vis("hillshade", $("#ly-terrain").checked);
  vis("rain", $("#ly-rain").checked && m.includes("rain"));
  vis("rivers", $("#ly-rivers").checked);
  vis("dams", $("#ly-dams").checked);
  vis("district-line", $("#ly-districts").checked);
  vis("state-labels", $("#ly-labels").checked);
  vis("district-fill", m.includes("district-fill"));
  vis("wells", $("#ly-wells").checked && m.includes("wells"));
  vis("place-fill", m.includes("place"));
  vis("place-line", m.includes("place"));
}

// ---------------------------------------------------------------- boot
map.on("load", async () => {
  try {
    loading(true, "Loading data");
    const [grid, national, wells, camps, india, districts, rivers, dams, pressure, towns] = await Promise.all([
      getJSON("rain/grid.json"), getJSON("rain/national.json"), getJSON("wells.json"), getJSON("campaigns.json"),
      getJSON("india.geojson"), getJSON("districts.geojson"), getJSON("rivers.geojson"), getJSON("dams.geojson"),
      getJSON("pressure.json"), getJSON("places.json").catch(() => [])]);

    grid.t0 = Date.parse(grid.first_day + "T00:00:00Z");
    grid.days = national.length;
    Object.assign(S, { grid, national, wells, camps, pressure, distFeatures: districts.features });
    S.campMs = camps.map((c) => Date.parse(c.date + "T00:00:00Z"));
    wells.forEach((w, i) => S.wellIx.set(w.id, i));
    buildSearch(towns);
    S.cellAt = new Int32Array(grid.lat.length * grid.lon.length).fill(-1);
    grid.lat_idx.forEach((li, c) => (S.cellAt[li * grid.lon.length + grid.lon_idx[c]] = c));
    S.obsByCamp = camps.map(() => new Float32Array(wells.length).fill(NaN));
    wells.forEach((w, i) => w.h.forEach(([c, , d]) => { S.obsByCamp[c][i] = d; }));
    rainSmall.width = grid.lon.length; rainSmall.height = grid.lat.length;
    rainCanvas.width = grid.lon.length * UPSCALE; rainCanvas.height = grid.lat.length * UPSCALE;
    $("#day").max = grid.days - 1;

    map.addSource("dem", { type: "raster-dem", tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
      encoding: "terrarium", tileSize: 256, maxzoom: 10, attribution: "Relief: AWS Terrain Tiles" });
    map.addSource("mask", { type: "geojson", data: worldMask(india) });
    map.addSource("india", { type: "geojson", data: india });
    map.addSource("districts", { type: "geojson", data: districts, generateId: true });
    map.addSource("states", { type: "geojson", data: stateLabels(districts) });
    map.addSource("rivers", { type: "geojson", data: rivers, attribution: "Rivers: HydroSHEDS HydroRIVERS" });
    map.addSource("dams", { type: "geojson", data: dams, attribution: "Dams: GeoDAR v1.1 (CC BY 4.0)" });
    map.addSource("place", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addSource("wells", { type: "geojson", data: { type: "FeatureCollection", features: [] }, attribution: "Wells: CGWB via figshare (CC BY 4.0) · Rain: IMD · Towns: GeoNames (CC BY 4.0)" });
    const h = 0.125;
    map.addSource("rain", { type: "canvas", canvas: rainCanvas, animate: false, coordinates: [
      [grid.lon[0] - h, grid.lat.at(-1) + h], [grid.lon.at(-1) + h, grid.lat.at(-1) + h],
      [grid.lon.at(-1) + h, grid.lat[0] - h], [grid.lon[0] - h, grid.lat[0] - h]] });

    // layers, bottom to top
    map.addLayer({ id: "india-fill", type: "fill", source: "india", paint: { "fill-color": "#f7f3ea" } });
    map.addLayer({ id: "hillshade", type: "hillshade", source: "dem", paint: {
      "hillshade-shadow-color": "#5e5545", "hillshade-highlight-color": "#fffdf7", "hillshade-accent-color": "#8f8672",
      "hillshade-exaggeration": 0.32, "hillshade-illumination-direction": 315 } });
    map.addLayer({ id: "mask", type: "fill", source: "mask", paint: { "fill-color": "#e7e1d3", "fill-opacity": 0.86 } });
    map.addLayer({ id: "district-fill", type: "fill", source: "districts", layout: { visibility: "none" }, paint: {
      "fill-color": ["case", ["==", ["coalesce", ["feature-state", "p"], -999], -999], "rgba(0,0,0,0)",
        ["interpolate", ["linear"], ["feature-state", "p"], -0.5, "#2c64a8", -0.15, "#a9c6e3", 0, "#ede7da", 0.15, "#e7a457", 0.5, "#9e2a1a"]],
      "fill-opacity": 0.9 } });
    map.addLayer({ id: "rivers", type: "line", source: "rivers",
      filter: ["any", [">=", ["zoom"], 5.5], [">=", ["get", "dis"], 120]], layout: { "line-cap": "round", "line-join": "round" }, paint: {
      "line-color": ["interpolate", ["linear"], ["ln", ["get", "dis"]], 2.7, "#9fc0e0", 8, "#3f78b5"],
      "line-width": ["interpolate", ["linear"], ["zoom"], 3, ["interpolate", ["linear"], ["ln", ["get", "dis"]], 2.7, 0.2, 6, 0.7, 9.5, 2],
        8, ["interpolate", ["linear"], ["ln", ["get", "dis"]], 2.7, 0.8, 6, 2, 9.5, 5]],
      "line-opacity": ["interpolate", ["linear"], ["ln", ["get", "dis"]], 2.7, 0.55, 6, 0.95] } });
    map.addLayer({ id: "district-line", type: "line", source: "districts", paint: {
      "line-color": "#d2c9b6", "line-width": ["interpolate", ["linear"], ["zoom"], 3, 0.25, 8, 0.9] } });
    map.addLayer({ id: "india-line", type: "line", source: "india", paint: { "line-color": "#4b4437", "line-width": 1 } });
    map.addLayer({ id: "rain", type: "raster", source: "rain", paint: { "raster-opacity": 0.8, "raster-resampling": "linear", "raster-fade-duration": 0 } });
    map.addImage("dam", damIcon(), { pixelRatio: 2 });
    // dams: the 334 with a recorded storage from regional zoom, sized by it; the rest only up close
    map.addLayer({ id: "dams", type: "symbol", source: "dams", minzoom: 5.5,
      filter: ["any", [">=", ["zoom"], 8.5], [">", ["coalesce", ["get", "storage_mcm"], 0], 0]], layout: {
      "icon-image": "dam", "icon-allow-overlap": false, "icon-padding": 2,
      "symbol-sort-key": ["-", 0, ["coalesce", ["get", "storage_mcm"], 0]],
      "icon-size": ["case", [">", ["coalesce", ["get", "storage_mcm"], 0], 0], ["interpolate", ["linear"], ["sqrt", ["get", "storage_mcm"]], 0, 0.55, 100, 1.3], 0.42] },
      paint: { "icon-opacity": ["interpolate", ["linear"], ["zoom"], 5.5, 0, 6.2, 0.9] } });
    map.addLayer({ id: "place-fill", type: "fill", source: "place", layout: { visibility: "none" }, paint: { "fill-color": "#1d232b", "fill-opacity": 0.06 } });
    map.addLayer({ id: "place-line", type: "line", source: "place", layout: { visibility: "none" }, paint: { "line-color": "#1d232b", "line-width": 1.6, "line-dasharray": [3, 2] } });
    map.addLayer({ id: "wells", type: "circle", source: "wells", paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"],
        3.5, ["+", 1.4, ["*", 0.55, ["get", "a"]]],
        6, ["+", 2.6, ["*", 1.0, ["get", "a"]]],
        10, ["+", 5, ["*", 1.6, ["get", "a"]]]],
      "circle-color": ["interpolate", ["linear"], ["get", "v"], -3, "#1d4e91", -1, "#5b93cf", 0, "#d9d2c3", 1, "#e0913a", 3, "#a83a14"],
      "circle-opacity": 0.95,
      "circle-stroke-color": "#fffdf7", "circle-stroke-width": 0.7 } });
    map.addLayer({ id: "state-labels", type: "symbol", source: "states", layout: {
      "text-field": ["upcase", ["get", "name"]], "text-font": ["Open Sans Semibold"],
      "text-size": ["interpolate", ["linear"], ["zoom"], 4, 9, 7, 13], "text-letter-spacing": 0.14,
      "text-max-width": 7, "symbol-sort-key": ["-", 0, ["get", "area"]] },
      paint: { "text-color": "#7a7262", "text-halo-color": "#f7f3ea", "text-halo-width": 1.6 } });

    // hotspot year pickers
    const prYears = [...new Set(Object.values(pressure).flatMap((r) => Object.keys(r).map(Number)))].sort((x, y) => x - y);
    for (const id of ["#pr-from", "#pr-to"]) $(id).innerHTML = prYears.map((y) => `<option>${y}</option>`).join("");
    $("#pr-from").value = 2015; $("#pr-to").value = prYears.at(-1);

    // What if and Future need the API; a plain http.server has no /api
    S.api = await fetch("api/health").then((r) => r.ok && r.headers.get("content-type")?.includes("json")).catch(() => false);
    const today = new Date().toISOString().slice(0, 10);
    $("#wi-start").value = today;

    map.setPadding(panelPad());
    addEventListener("resize", () => { map.setPadding(panelPad()); sizeFx(); buildTimeline(); });
    map.on("move", () => (fxDirty = true));
    sizeFx(); requestAnimationFrame(fxLoop);
    buildTimeline();
    await setDay(dayOf("2019-08-01"));
    loading(false);
    setMode(["replay", "whatif", "forecast", "pressure"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "replay");
  } catch (e) {
    console.error(e);
    loading(true, `Could not load data: ${e.message}. Run build_ui_data.py and serve this folder over http.`);
    $(".spin").hidden = true;
  }
});

// ---------------------------------------------------------------- events
$$(".modes button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
$("#play").addEventListener("click", () => togglePlay());
$("#day").addEventListener("input", (e) => setDay(+e.target.value));
$("#speed").addEventListener("change", (e) => (S.speed = +e.target.value));
const PLACES = { "2005-07-26": [73.2, 19.0, 7], "2013-06-16": [79.1, 30.3, 7], "2015-12-01": [79.9, 12.9, 7.2], "2018-08-15": [76.4, 10.1, 7], "2009-07-15": [80.5, 22.5, 4.1] };
$$(".jump .chip").forEach((b) => b.addEventListener("click", () => {
  $$(".jump .chip").forEach((c) => c.setAttribute("aria-pressed", String(c === b)));
  const [lng, lat, z] = PLACES[b.dataset.jump];
  setDay(dayOf(b.dataset.jump));
  map.flyTo({ center: [lng, lat], zoom: z, duration: REDUCED ? 0 : 1600 });
}));
// search ("/" focuses it)
$("#q").addEventListener("input", showSuggestions);
$("#q").addEventListener("focus", () => { if ($("#q").value) { $("#q").select(); showSuggestions(); } });
$("#q").addEventListener("keydown", (e) => {
  if (e.key === "Escape") { $("#q-list").hidden = true; $("#q").blur(); return; }
  if ($("#q-list").hidden || !qItems.length) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault(); qSel = (qSel + (e.key === "ArrowDown" ? 1 : -1) + qItems.length) % qItems.length;
    $$("#q-list li").forEach((li, i) => li.setAttribute("aria-selected", String(i === qSel)));
    $("#q").setAttribute("aria-activedescendant", `q-o${qSel}`);
  } else if (e.key === "Enter") { e.preventDefault(); goTo(qItems[qSel].e); }
});
$("#q-list").addEventListener("mousedown", (e) => { const li = e.target.closest("li[data-i]"); if (li) { e.preventDefault(); goTo(qItems[+li.dataset.i].e); } });
$("#q").addEventListener("blur", () => setTimeout(() => ($("#q-list").hidden = true), 120));
$("#pl-search").addEventListener("click", () => $("#q").focus());
$("#pl-all").addEventListener("click", () => { setPlace({}, "All India"); map.flyTo({ center: [80.5, 22.5], zoom: 4.1, duration: REDUCED ? 0 : 1000 }); });
$("#pl-radius").addEventListener("change", () => { const q = S.place.q; if (q.lat != null) setPlace({ ...q, radius_km: +$("#pl-radius").value }, `Within ${$("#pl-radius").value} km of ${q.lat.toFixed(2)}°N ${q.lon.toFixed(2)}°E`); });

// what if
const pctText = (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v)}%`;
$("#wi-pct").addEventListener("input", () => ($("#wi-pct-out").textContent = pctText(+$("#wi-pct").value)));
$("#wi-mm").addEventListener("input", () => ($("#wi-mm-out").textContent = `${$("#wi-mm").value} mm`));
$$('input[name="wi-kind"]').forEach((r) => r.addEventListener("change", () => {
  const k = $('input[name="wi-kind"]:checked').value;
  $$("[data-kind]").forEach((f) => (f.hidden = f.dataset.kind !== k));
}));
function choose(group, chip) { $$(`${group} .chip`).forEach((c) => c.setAttribute("aria-checked", String(c === chip))); }
$$("[data-when]").forEach((b) => b.addEventListener("click", () => {
  $("#wi-start").value = b.dataset.when === "today" ? new Date().toISOString().slice(0, 10) : b.dataset.when;
  $("#wi-days").value = b.dataset.days;
  $$("[data-when]").forEach((c) => c.setAttribute("aria-pressed", String(c === b)));
}));
for (const id of ["#wi-start", "#wi-days"]) $(id).addEventListener("change", () => $$("[data-when]").forEach((c) => c.setAttribute("aria-pressed", "false")));
$$("[data-amt]").forEach((b) => b.addEventListener("click", () => {
  const [kind, v] = b.dataset.amt.split(":");
  $(`input[name="wi-kind"][value="${kind}"]`).checked = true;
  $$("[data-kind]").forEach((f) => (f.hidden = f.dataset.kind !== kind));
  $(kind === "mm" ? "#wi-mm" : "#wi-pct").value = v;
  $("#wi-mm-out").textContent = `${$("#wi-mm").value} mm`; $("#wi-pct-out").textContent = pctText(+$("#wi-pct").value);
  choose('[data-panel="whatif"] .choice', b);
}));
for (const id of ["#wi-mm", "#wi-pct"]) $(id).addEventListener("input", () => choose('[data-panel="whatif"] .choice', null));
$$('input[name="wi-kind"]').forEach((r) => r.addEventListener("change", () => choose('[data-panel="whatif"] .choice', null)));
$("#wi-run").addEventListener("click", runWhatIf);

// future
let fcT;
$("#fc-year").addEventListener("input", () => {
  $("#fc-year-out").textContent = $("#fc-year").value;
  clearTimeout(fcT); fcT = setTimeout(runForecast, 200);
});
$$("[data-rain]").forEach((b) => b.addEventListener("click", () => {
  $("#fc-rain").value = b.dataset.rain; choose('[data-panel="forecast"] .choice', b); runForecast();
}));
for (const id of ["#fc-show-rain", "#fc-show-trend"]) $(id).addEventListener("change", () => fcLast && renderForecast(fcLast));
$$("[data-years]").forEach((b) => b.addEventListener("click", () => {
  [$("#pr-from").value, $("#pr-to").value] = b.dataset.years.split(",");
  renderPressure();
}));
// From after To gets swapped
for (const id of ["#pr-from", "#pr-to"]) $(id).addEventListener("change", () => {
  const a = +$("#pr-from").value, b = +$("#pr-to").value;
  if (a > b) { $("#pr-from").value = b; $("#pr-to").value = a; }
  renderPressure();
  if (!$("#detail").hidden && S.prDistrict != null) showDistrict(S.prDistrict);
});
$$("[data-show]").forEach((b) => b.addEventListener("click", () => {
  prWorst = b.dataset.show === "worst";
  $$("[data-show]").forEach((c) => c.setAttribute("aria-checked", String(c === b))); renderPressure();
}));
$("#pr-rank").addEventListener("click", (e) => { const li = e.target.closest("li"); if (li) focusDistrict(+li.dataset.id); });
$("#pr-rank").addEventListener("keydown", (e) => { const li = e.target.closest("li"); if (li && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); focusDistrict(+li.dataset.id); } });
function focusDistrict(id) {
  const f = S.distFeatures[id], c = f.geometry.type === "Polygon" ? f.geometry.coordinates[0] : f.geometry.coordinates.flat(1);
  const xs = c.map((p) => p[0]), ys = c.map((p) => p[1]);
  map.fitBounds([[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]], { padding: 60, duration: REDUCED ? 0 : 1200, maxZoom: 7 });
  showDistrict(id);
}
$$(".layers input").forEach((i) => i.addEventListener("change", applyLayers));
$("#detail-close").addEventListener("click", () => ($("#detail").hidden = true));
addEventListener("keydown", (e) => {
  if (e.target.matches("select, textarea, input")) return;   // the timeline input handles its own arrows
  if (e.key === "/") { e.preventDefault(); $("#q").focus(); return; }
  if (e.code === "Space" && S.mode === "replay") { e.preventDefault(); togglePlay(); }
  if (S.mode === "replay" && (e.key === "ArrowRight" || e.key === "ArrowLeft"))
    setDay(S.day + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 30 : 1));
});

// hover card on wells, click for the full history
const hover = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 10, maxWidth: "240px" });
map.on("mousemove", "wells", (e) => {
  const f = e.features[0];
  if (!f) { hover.remove(); map.getCanvas().style.cursor = ""; return; }
  map.getCanvas().style.cursor = "pointer";
  const w = S.wells[f.properties.i], st = f.properties;
  const v = st.r, line = S.mode === "whatif" ? `${Math.abs(v) < 0.0005 ? "no change" : `${v < 0 ? "rises" : "falls"} ${cm(v)}`} <span class="pop-s">from the extra rain</span>`
    : S.mode === "forecast" ? `${level(v)} <span class="pop-s">by ${$("#fc-year").value}, rain only</span>`
    : `${v > 0 ? "fell" : "rose"} ${Math.abs(v).toFixed(2)} m <span class="pop-s">since previous reading</span>`;
  hover.setLngLat(e.lngLat).setHTML(`<div class="pop-t">${w.district}</div><div class="pop-s">${w.state} · ${w.type}</div><div class="pop-v">${line}</div>`).addTo(map);
});
map.on("mouseleave", "wells", () => { hover.remove(); map.getCanvas().style.cursor = ""; });
map.on("click", (e) => {
  const well = map.getLayer("wells") && map.queryRenderedFeatures(e.point, { layers: ["wells"] })[0];
  if (S.mode === "whatif" || S.mode === "forecast") {
    if (map.queryRenderedFeatures(e.point, { layers: ["dams"] }).length) return;
    if (well) { const w = S.wells[well.properties.i]; setPlace({ well: w.id }, `One well, ${w.district}, ${w.state}`); showWell(well.properties.i); return; }
    const km = +$("#pl-radius").value, { lng, lat } = e.lngLat;
    setPlace({ lat: +lat.toFixed(4), lon: +lng.toFixed(4), radius_km: km }, `Within ${km} km of ${lat.toFixed(2)}°N ${lng.toFixed(2)}°E`);
    return;
  }
  if (well) { showWell(well.properties.i); return; }
  const d = S.mode === "pressure" && map.queryRenderedFeatures(e.point, { layers: ["district-fill"] })[0];
  if (d) showDistrict(d.id);
});
map.on("click", "dams", (e) => {
  const p = e.features[0].properties;
  new maplibregl.Popup({ closeButton: false, offset: 8 }).setLngLat(e.lngLat).setHTML(
    `<div class="pop-t">Dam</div><div class="pop-s">GeoDAR ${p.id}</div><div class="pop-v">${p.storage_mcm && p.storage_mcm !== "null" ? `${(+p.storage_mcm).toLocaleString()} million m³ storage` : "storage not recorded"}</div>`).addTo(map);
});
for (const id of ["dams", "district-fill"]) {
  map.on("mouseenter", id, () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", id, () => (map.getCanvas().style.cursor = ""));
}
