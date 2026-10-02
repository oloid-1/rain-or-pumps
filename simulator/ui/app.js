// Rain or Pumps simulator: replay recorded rain, run rain scenarios through the
// simulator model in the browser, and map the part of the fall rain does not explain.
// Data comes from simulator/ui/build_ui_data.py and simulator/geo/build_geo.py.

const D = "data/";
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const DAY_MS = 86400000;

const S = {
  mode: "replay", day: 0, playing: false, speed: 6, years: new Map(), pending: new Map(),
  grid: null, cellAt: null, today: null, national: null, wells: [], camps: [], campMs: [],
  obsByCamp: [], depthByCamp: [], curCamp: -1, scen: null, scenByWell: null, sess: null,
  meta: null, pressure: null, distFeatures: [],
};

// ---------------------------------------------------------------- utilities
const half = (() => {           // float16 -> float32 lookup
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    t[h] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return t;
})();
const f16 = (buf, off, n) => { const u = new Uint16Array(buf, off, n), o = new Float32Array(n); for (let i = 0; i < n; i++) o[i] = half[u[i]]; return o; };
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
  paintWells((i) => (vals ? vals[i] : NaN), () => false);
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
// one setData per repaint, carrying only the wells that have a value. Per-feature
// state updates (one per well) were what made switching modes feel slow.
function paintWells(valueOf, ringOf) {
  const features = [];
  for (let i = 0; i < S.wells.length; i++) {
    const v = valueOf(i);
    if (Number.isNaN(v)) continue;
    const w = S.wells[i];
    features.push({ type: "Feature", id: i, properties: { i, v, a: Math.min(Math.abs(v), 4), ring: !!ringOf(i) },
      geometry: { type: "Point", coordinates: [w.lon, w.lat] } });
  }
  map.getSource("wells")?.setData({ type: "FeatureCollection", features });
}

// ---------------------------------------------------------------- scenario
// The model runs in model-worker.js, off the main thread.
let worker = null, workerReady = null, jobId = 0;
const jobs = new Map();
function initModel() {
  if (workerReady) return workerReady;
  worker = new Worker("model-worker.js");
  workerReady = new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "ready") resolve();
      else if (m.type === "progress" && m.id === jobId && S.mode === "scenario") loading(true, `Running the model · ${Math.round(100 * m.done / m.n)}%`);
      else if (m.type === "done") { jobs.get(m.id)?.resolve(m.out); jobs.delete(m.id); }
      else if (m.type === "error") { (jobs.get(m.id) || { reject })?.reject(new Error(m.message)); jobs.delete(m.id); }
    };
    worker.onerror = (e) => reject(new Error(e.message || "model worker failed"));
  });
  worker.postMessage({ type: "init", url: new URL(D + "sim.onnx", location.href).href });
  return workerReady;
}

// Opening a year needs only the index: the model's prediction at recorded rain was
// computed when the data was built. The model and its inputs load only when the
// rain is changed, so switching to Scenario never waits on inference.
function openScenarioYear(y) {
  const idx = S.scenIndex.years[y], [, T, C] = S.scenIndex.layout.seq;
  S.scen = { y, n: idx.rows, T, C, nn: S.scenIndex.layout.num[1], nc: S.scenIndex.layout.cat[1],
    idx, base: Float32Array.from(idx.base), seq: null, inputs: null };
  scenKey = "";
}

function scenarioInputs() {
  const sc = S.scen;
  sc.inputs ??= fetch(`${D}scenario/${sc.y}.bin`).then((r) => r.arrayBuffer()).then((buf) => {
    const { n, T, C, nn, nc } = sc, offN = n * T * C * 2;
    sc.seq = f16(buf, 0, n * T * C);
    sc.num = new Float32Array(buf.slice(offN, offN + n * nn * 4));
    const catU = new Uint8Array(buf, offN + n * nn * 4, n * nc);
    sc.cat = new BigInt64Array(n * nc);
    for (let i = 0; i < catU.length; i++) sc.cat[i] = BigInt(catU[i]);
  });
  return sc.inputs;
}

function scaledSeq(f, rows) {
  const { seq, T, C } = S.scen, out = seq.slice(), m = S.meta, ch = m.channels;
  const r = ch.indexOf("rain_mm"), a = ch.indexOf("rain_anom_mm");
  const rs = m.channel_scaling.rain_mm, as = a >= 0 ? m.channel_scaling.rain_anom_mm : null;
  for (const i of rows) for (let t = 0; t < T; t++) {
    const o = (i * T + t) * C;
    const rain = Math.max(0, Math.expm1(seq[o + r] * rs.std + rs.mean));
    out[o + r] = (Math.log1p(rain * f) - rs.mean) / rs.std;
    if (as) {
      const u = seq[o + a] * as.std + as.mean;
      const an = Math.sign(u) * Math.expm1(Math.abs(u)) + (f - 1) * rain;
      out[o + a] = (Math.sign(an) * Math.log1p(Math.abs(an)) - as.mean) / as.std;
    }
  }
  return out;
}

function predict(f, rows) {
  const { n, T, C, nn, nc } = S.scen;
  const seq = scaledSeq(f, rows);                 // a fresh copy, handed to the worker
  const id = ++jobId;
  return new Promise((resolve, reject) => {
    jobs.set(id, { resolve, reject });
    worker.postMessage({ type: "run", id, n, T, C, nn, nc, seq, num: S.scen.num.slice(), cat: S.scen.cat.slice() }, [seq.buffer]);
  });
}

let scenRun = 0, scenKey = "", scenPaint = null;
async function runScenario() {
  if (!S.scen) return;
  const pct = +$("#sc-rain").value, f = 1 + pct / 100, region = $("#sc-region").value, id = ++scenRun;
  const key = `${S.scen.y}|${pct}|${region}`;
  if (key === scenKey && scenPaint) { scenPaint(); return; }      // same inputs: repaint, don't re-run
  $("#sc-pct").textContent = `${pct > 0 ? "+" : pct < 0 ? "−" : ""}${Math.abs(pct)}%`;
  $$(".presets .chip").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.pct === pct)));
  const { idx, n } = S.scen;
  const rows = [...Array(n).keys()].filter((i) => !region || S.wells[idx.well[i]].state === region);
  let pred = S.scen.base;
  // never show the last scenario's numbers under the new slider value
  $$(".result, [data-panel='scenario'] .kpis").forEach((el) => el.classList.toggle("pending", f !== 1));
  if (f !== 1) {
    $("#sc-diff").textContent = "computing…"; $("#sc-arrow").className = "arrow";
    $("#sc-diff-sub").textContent = `running the model on every well with ${pct > 0 ? "+" : "−"}${Math.abs(pct)}% rain`;
    loading(true, "Running the model");
    await initModel(); await scenarioInputs();
    if (id !== scenRun) return;
    pred = await predict(f, rows);
    loading(false);
  }
  if (id !== scenRun) return;
  const byWell = new Map(); let sum = 0, diff = 0, fall = 0, rise = 0, wl = 0, wlBase = 0;
  for (const i of rows) {
    const p = pred[i], w = idx.well[i], after = idx.depth_before[i] + p;
    byWell.set(w, { p, ring: after <= 2, i });
    sum += p; diff += p - S.scen.base[i]; p > 0 ? fall++ : rise++;
    if (after <= 2) wl++; if (idx.depth_before[i] + S.scen.base[i] <= 2) wlBase++;
  }
  $$(".pending").forEach((el) => el.classList.remove("pending"));
  const k = rows.length || 1, dm = diff / k;
  // dm is in delta_h_m (positive = fell); the headline speaks in water level, so flip it
  $("#sc-diff").textContent = f === 1 ? "as recorded" : `${-dm >= 0 ? "+" : "−"}${Math.abs(dm * 100).toFixed(1)} cm`;
  $("#sc-arrow").className = "arrow" + (f === 1 ? "" : dm < 0 ? " up" : " down");
  $("#sc-diff-sub").textContent = f === 1 ? "move the slider to change the rain" :
    `water level ${dm < 0 ? "higher" : "lower"} on average than with the rain that actually fell, across ${rows.length.toLocaleString()} wells`;
  $("#sc-mean").textContent = fmtM(sum / k);
  $("#sc-split").textContent = `${fall.toLocaleString()} ↓  ${rise.toLocaleString()} ↑`;
  $("#sc-wl").textContent = `${wl}${f === 1 ? "" : ` (${wl - wlBase >= 0 ? "+" : "−"}${Math.abs(wl - wlBase)})`}`;
  S.scenByWell = byWell;
  scenKey = key;
  scenPaint = () => paintWells((i) => (byWell.has(i) ? byWell.get(i).p : NaN), (i) => byWell.get(i)?.ring);
  scenPaint();
}

// load the model and the base predictions in the background, so Scenario opens at once
let warm = null;
function warmScenario() {
  // fetch and compile only; nothing runs until the rain slider moves
  warm ??= (async () => { await initModel(); if (!S.scen) openScenarioYear(+$("#sc-year").value); await scenarioInputs(); })();
  return warm;
}

// ---------------------------------------------------------------- pressure
let prKey = "";
function renderPressure() {
  const a = Math.min(+$("#pr-from").value, +$("#pr-to").value), b = Math.max(+$("#pr-from").value, +$("#pr-to").value);
  $("#pr-out").textContent = `${a}–${b}`;
  if (prKey === `${a}-${b}`) return;
  prKey = `${a}-${b}`;
  const rank = [];
  S.distFeatures.forEach((f, id) => {
    const rec = S.pressure[`${f.properties.name}|${f.properties.state}`];
    let s = 0, k = 0;
    if (rec) for (let y = a; y <= b; y++) if (rec[y]) { s += rec[y][0]; k++; }
    const v = k ? s / k : null;
    map.setFeatureState({ source: "districts", id }, { p: v });
    if (v != null && k >= Math.min(3, b - a + 1)) rank.push([f.properties.name, f.properties.state, v, id]);
  });
  rank.sort((x, y) => y[2] - x[2]);
  const top = rank.slice(0, 10), mx = Math.max(0.01, ...top.map((r) => r[2]));
  $("#pr-rank").innerHTML = top.map(([n, s, v, id]) =>
    `<li tabindex="0" data-id="${id}"><span class="nm">${n} <span class="st">${s}</span></span><span class="val">${fmtM(v)}</span>` +
    `<span class="bar-in" style="width:${Math.max(4, v / mx * 100)}%"></span></li>`).join("");
}

// ---------------------------------------------------------------- detail panel
function chart(series, { invert = false, unit = "m", zero = false } = {}) {
  const W = 330, H = 150, P = { l: 36, r: 8, t: 10, b: 20 };
  const pts = series.flatMap((s) => s.pts);
  if (!pts.length) return "";
  let [x0, x1] = [Math.min(...pts.map((p) => p[0])), Math.max(...pts.map((p) => p[0]))];
  let [y0, y1] = [Math.min(...pts.map((p) => p[1])), Math.max(...pts.map((p) => p[1]))];
  if (zero) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
  if (y1 - y0 < 0.5) { y0 -= 0.25; y1 += 0.25; }
  if (x1 === x0) x1 = x0 + 1;
  const X = (x) => P.l + (x - x0) / (x1 - x0) * (W - P.l - P.r);
  const Y = (y) => invert ? P.t + (y - y0) / (y1 - y0) * (H - P.t - P.b) : H - P.b - (y - y0) / (y1 - y0) * (H - P.t - P.b);
  const ticks = [y0, (y0 + y1) / 2, y1].map((v) => `<text x="${P.l - 6}" y="${Y(v) + 3}" text-anchor="end">${v.toFixed(1)}</text><line x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}" class="g"/>`).join("");
  const xt = [x0, x1].map((v, i) => `<text x="${X(v)}" y="${H - 4}" text-anchor="${i ? "end" : "start"}">${new Date(v).getUTCFullYear()}</text>`).join("");
  const z = zero ? `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="#8a8270" stroke-dasharray="2 3"/>` : "";
  const s0 = series[0], edge = invert ? P.t : H - P.b;
  const area = s0.fill ? `<path d="M${X(s0.pts[0][0])},${edge} ${s0.pts.map((p) => `L${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ")} L${X(s0.pts.at(-1)[0])},${edge}Z" fill="url(#ga)"/>` : "";
  const lines = series.map((s) => `<polyline fill="none" stroke="${s.color}" stroke-width="1.8" stroke-linejoin="round" ${s.dash ? 'stroke-dasharray="4 3"' : ""} points="${s.pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ")}"/>`).join("");
  const key = series.map((s, i) => `<tspan fill="${s.color}" dx="${i ? 12 : 0}">— ${s.name}</tspan>`).join("");
  return `<svg viewBox="0 0 ${W} ${H + 14}" font-family="IBM Plex Mono" font-size="10" fill="#7a7364"><defs><linearGradient id="ga" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#265ca8" stop-opacity=".18"/><stop offset="1" stop-color="#265ca8" stop-opacity="0"/></linearGradient></defs><style>.g{stroke:rgba(60,50,30,.1)}</style>${ticks}${xt}${z}${area}${lines}<text x="${P.l}" y="${H + 12}">${key}<tspan dx="8">(${unit})</tspan></text></svg>`;
}

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
  $("#detail-chart").innerHTML = chart([{ name: "depth to water", color: "#265ca8", pts: depth, fill: true }], { invert: true });
  $("#detail-chart").setAttribute("aria-label", `Depth to water over ${hs.length} readings, from ${depth[0][1]} m to ${depth.at(-1)[1]} m`);
  const kv = [["Readings", hs.length], ["Aquifer", w.aquifer], ["Well type", w.type], ["Well depth", w.well_depth_m != null ? `${w.well_depth_m} m` : "–"],
    ["Specific yield", w.sy ?? "–"], ["Normal annual rain", w.rain_normal_annual_mm != null ? `${w.rain_normal_annual_mm.toLocaleString()} mm` : "–"],
    ["Mean change per reading", fmtM(mObs)], ["Rain-expected", fmtM(mExp)], ["Unexplained", fmtM(mObs - mExp)]];
  if (S.mode === "scenario" && S.scenByWell?.has(i)) {
    const r = S.scenByWell.get(i);
    kv.push(["Scenario change", fmtM(r.p)], ["Recorded change", fmtM(S.scen.idx.observed[r.i])], ["Depth after scenario", `${(S.scen.idx.depth_before[r.i] + r.p).toFixed(2)} m`]);
  }
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
  $("#detail-sub").textContent = `${f.properties.state} · mean change per reading, by year; positive means the water fell`;
  $("#detail-chart").innerHTML = chart([
    { name: "observed", color: "#b4441a", pts: pt(1) },
    { name: "rain-expected", color: "#265ca8", pts: pt(2), dash: true }], { zero: true });
  const all = ys.map((y) => rec[y][0]), mean = all.reduce((a, b) => a + b, 0) / all.length;
  $("#detail-kv").innerHTML = [["Years with data", ys.length], ["Wells (latest year)", rec[ys.at(-1)][3]], ["Mean unexplained", fmtM(mean)]]
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  openDetail();
}

// ---------------------------------------------------------------- modes and layers
const VIS = { replay: ["rain", "wells"], scenario: ["wells"], pressure: ["district-fill"] };
function setMode(mode) {
  S.mode = mode;
  $$(".modes button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  $$(".mode-panel").forEach((p) => (p.hidden = p.dataset.panel !== mode));
  $("#detail").hidden = true;
  if (mode !== "replay") togglePlay(false);
  history.replaceState(null, "", `#${mode}`);
  $('[data-lg="rain"]').hidden = mode !== "replay";
  $('[data-lg="wells"]').hidden = mode === "pressure";
  $('[data-lg="pressure"]').hidden = mode !== "pressure";
  $('[data-lg="ring"]').hidden = mode !== "scenario";
  $("#lg-wells-title").textContent = mode === "scenario" ? "Predicted change in water level" : "Well level since previous reading";
  applyLayers();
  if (mode === "replay") { S.curCamp = -2; setCampaign(lastCampaign(S.grid.t0 + S.day * DAY_MS)); }
  if (mode === "scenario") (async () => { if (!S.scen) openScenarioYear(+$("#sc-year").value); await runScenario(); })()
    .catch((e) => { loading(false); $("#sc-note").textContent = `The model could not load: ${e.message}`; });
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
}

// ---------------------------------------------------------------- boot
map.on("load", async () => {
  try {
    loading(true, "Loading data");
    const [grid, national, wells, camps, india, districts, rivers, dams, pressure, scenIndex, meta] = await Promise.all([
      getJSON("rain/grid.json"), getJSON("rain/national.json"), getJSON("wells.json"), getJSON("campaigns.json"),
      getJSON("india.geojson"), getJSON("districts.geojson"), getJSON("rivers.geojson"), getJSON("dams.geojson"),
      getJSON("pressure.json"), getJSON("scenario/index.json"), getJSON("sim_meta.json")]);

    grid.t0 = Date.parse(grid.first_day + "T00:00:00Z");
    grid.days = national.length;
    Object.assign(S, { grid, national, wells, camps, pressure, scenIndex, meta, distFeatures: districts.features });
    S.campMs = camps.map((c) => Date.parse(c.date + "T00:00:00Z"));
    S.cellAt = new Int32Array(grid.lat.length * grid.lon.length).fill(-1);
    grid.lat_idx.forEach((li, c) => (S.cellAt[li * grid.lon.length + grid.lon_idx[c]] = c));
    S.obsByCamp = camps.map(() => new Float32Array(wells.length).fill(NaN));
    S.depthByCamp = camps.map(() => new Float32Array(wells.length).fill(NaN));
    wells.forEach((w, i) => w.h.forEach(([c, dep, d]) => { S.obsByCamp[c][i] = d; S.depthByCamp[c][i] = dep; }));
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
    map.addSource("wells", { type: "geojson", data: { type: "FeatureCollection", features: [] }, attribution: "Wells: CGWB via figshare (CC BY 4.0) · Rain: IMD" });
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
    map.addLayer({ id: "wells", type: "circle", source: "wells", paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"],
        3.5, ["+", 1.4, ["*", 0.55, ["get", "a"]]],
        6, ["+", 2.6, ["*", 1.0, ["get", "a"]]],
        10, ["+", 5, ["*", 1.6, ["get", "a"]]]],
      "circle-color": ["interpolate", ["linear"], ["get", "v"], -3, "#1d4e91", -1, "#5b93cf", 0, "#d9d2c3", 1, "#e0913a", 3, "#a83a14"],
      "circle-opacity": 0.95,
      "circle-stroke-color": ["case", ["get", "ring"], "#1d232b", "#fffdf7"],
      "circle-stroke-width": ["case", ["get", "ring"], 2, 0.7] } });
    map.addLayer({ id: "state-labels", type: "symbol", source: "states", layout: {
      "text-field": ["upcase", ["get", "name"]], "text-font": ["Open Sans Semibold"],
      "text-size": ["interpolate", ["linear"], ["zoom"], 4, 9, 7, 13], "text-letter-spacing": 0.14,
      "text-max-width": 7, "symbol-sort-key": ["-", 0, ["get", "area"]] },
      paint: { "text-color": "#7a7262", "text-halo-color": "#f7f3ea", "text-halo-width": 1.6 } });

    // scenario controls
    $("#sc-year").innerHTML = Object.keys(scenIndex.years).map((y) => `<option value="${y}">November ${y}</option>`).join("");
    $("#sc-year").value = Object.keys(scenIndex.years).at(-1);
    $("#sc-region").innerHTML += [...new Set(wells.map((w) => w.state))].sort().map((s) => `<option>${s}</option>`).join("");
    const rv = meta.response_valid || {};
    if (rv["x1.2"]) $("#sc-note").textContent =
      `On held-out readings, +20% rain moves the mean prediction by ${fmtM(rv["x1.2"].mean, 3)}, and ${(100 * rv["x1.2"].wrong_way).toFixed(1)}% of readings still move the wrong way. Read differences under a few centimetres as noise.`;

    map.setPadding(panelPad());
    addEventListener("resize", () => { map.setPadding(panelPad()); sizeFx(); buildTimeline(); });
    map.on("move", () => (fxDirty = true));
    sizeFx(); requestAnimationFrame(fxLoop);
    buildTimeline();
    await setDay(dayOf("2019-08-01"));
    loading(false);
    setMode(["replay", "scenario", "pressure"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "replay");
    const idle = window.requestIdleCallback || ((f) => setTimeout(f, 1500));
    idle(() => warmScenario().catch(() => {}), { timeout: 4000 });
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
let scT;
$("#sc-rain").addEventListener("input", () => {
  clearTimeout(scT); const v = +$("#sc-rain").value;
  $("#sc-pct").textContent = `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v)}%`; scT = setTimeout(runScenario, 220);
});
$("#sc-region").addEventListener("change", runScenario);
$("#sc-year").addEventListener("change", () => { openScenarioYear(+$("#sc-year").value); runScenario(); });
$$(".presets .chip").forEach((b) => b.addEventListener("click", () => { $("#sc-rain").value = b.dataset.pct; runScenario(); }));
["#pr-from", "#pr-to"].forEach((s) => $(s).addEventListener("input", renderPressure));
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
$("#about-btn").addEventListener("click", () => $("#about").showModal());
addEventListener("keydown", (e) => {
  if (e.target.matches("select, textarea, input")) return;   // the timeline input handles its own arrows
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
  const what = S.mode === "scenario" ? "predicted" : "since previous reading";
  hover.setLngLat(e.lngLat).setHTML(`<div class="pop-t">${w.district}</div><div class="pop-s">${w.state} · ${w.type}</div>` +
    `<div class="pop-v">${st.v > 0 ? "fell" : "rose"} ${Math.abs(st.v).toFixed(2)} m <span class="pop-s">${what}</span></div>`).addTo(map);
});
map.on("mouseleave", "wells", () => { hover.remove(); map.getCanvas().style.cursor = ""; });
map.on("click", "wells", (e) => { const f = e.features[0]; if (f) showWell(f.properties.i); });
map.on("click", "district-fill", (e) => { if (S.mode === "pressure" && e.features[0]) showDistrict(e.features[0].id); });
map.on("click", "dams", (e) => {
  const p = e.features[0].properties;
  new maplibregl.Popup({ closeButton: false, offset: 8 }).setLngLat(e.lngLat).setHTML(
    `<div class="pop-t">Dam</div><div class="pop-s">GeoDAR ${p.id}</div><div class="pop-v">${p.storage_mcm && p.storage_mcm !== "null" ? `${(+p.storage_mcm).toLocaleString()} million m³ storage` : "storage not recorded"}</div>`).addTo(map);
});
for (const id of ["dams", "district-fill"]) {
  map.on("mouseenter", id, () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", id, () => (map.getCanvas().style.cursor = ""));
}
