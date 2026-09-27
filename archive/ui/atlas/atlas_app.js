const A = window.ATLAS, META = A.meta, D = A.districts, RAIN = A.rain || {};
const $ = (id) => document.getElementById(id);
const YEARS = META.years;                              // wells: 2000 onward
const RAIN_YEARS = META.rainYears || META.years;       // rain: 1998 onward, and far more districts
const state = { mode: "depth", year: 2019, delta: 0.2, picked: null, hover: null };

// Rain comes from the grid, so it exists wherever the grid reaches - not only where a well was
// drilled. Wells and rain therefore have different coverage and different year ranges, and each
// mode says which it is drawing.
const rainAt = (name, key, year) => {
  const series = RAIN[name];
  const index = RAIN_YEARS.indexOf(year);
  return series && index >= 0 ? series[key][index] : undefined;
};

const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(d));
const pct = (d) => `${d > 0 ? "+" : ""}${Math.round(d * 100)}%`;
const gainOf = (district, delta) => (district.gains ? district.gains[String(delta)] : undefined);

// ---- what each mode paints -----------------------------------------------------------------
const RAMPS = {
  depth: ["#f7ecd0", "#e6bd73", "#c1843a", "#8a511c", "#4a2a10"],
  rain: ["#f4efd6", "#cfe6bf", "#79c3ae", "#3591b8", "#2b5ea7"],
  diverging: ["#7a3410", "#c98b4a", "#efe7d8", "#8fb56a", "#3f6f1e"],
};
const interpolate = (stops) => (t) => {
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const [a, b] = [stops[i], stops[i + 1]].map((hex) =>
    [1, 3, 5].map((p) => parseInt(hex.slice(p, p + 2), 16)));
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
};
const RAMP = { depth: interpolate(RAMPS.depth), rain: interpolate(RAMPS.rain), diverging: interpolate(RAMPS.diverging) };

// `source` decides which coverage a mode draws from: the rain grid, or the wells.
const MODES = {
  depth: { title: "How deep the water is", legend: "metres down", ramp: "depth", timed: true, source: "wells",
           explain: "How far down you hit water in November. Bigger number means deeper water, so darker is worse.",
           value: (name) => D[name]?.depth[YEARS.indexOf(state.year)], domain: [0, 20], ticks: [0, 5, 10, 15, 20] },
  jjas: { title: "Monsoon rain", legend: "June–September, mm", ramp: "rain", timed: true, source: "rain",
          explain: "Rain that fell during the monsoon, June to September.",
          value: (name) => rainAt(name, "jjas", state.year), domain: [0, 2000], ticks: [0, 500, 1000, 1500, 2000] },
  annual: { title: "Rain for the year", legend: "mm", ramp: "rain", timed: true, source: "rain",
            explain: "Rain over the whole year, monsoon and everything else.",
            value: (name) => rainAt(name, "annual", state.year), domain: [0, 2500], ticks: [0, 600, 1200, 1800, 2500] },
  decline: { title: "How fast the water is dropping", legend: "metres a year · red = dropping", ramp: "diverging",
             timed: false, flip: true, source: "wells",
             explain: "How much the water level moved each year since 2000. Red means it is dropping. Green means it is coming back.",
             value: (name) => D[name]?.decline, domain: [-0.3, 0.3], ticks: [-0.3, -0.15, 0, 0.15, 0.3] },
  scenario: { title: "How much more rain lifts the water", legend: "metres gained", ramp: "rain", timed: false,
              source: "wells",
              explain: "If the monsoon were the size you picked, how much higher the water would sit. This one is the model's answer, not a measurement.",
              value: (name) => (D[name] ? gainOf(D[name], state.delta) : undefined),
              domain: [0, 1.2], ticks: [0, 0.3, 0.6, 0.9, 1.2] },
  recharge: { title: "How much rain soaks in", legend: "% of the monsoon kept", ramp: "rain", timed: false,
              source: "wells",
              explain: "Out of all the monsoon rain, how much stays in the ground instead of running off. Where more soaks in, ponds and check dams have more to catch.",
              value: (name) => D[name]?.infiltration, domain: [0, 30], ticks: [0, 8, 15, 22, 30] },
  net: { title: "Is that rain enough?", legend: "metres a year · red = not enough", ramp: "diverging", timed: false,
         source: "wells", explain: "The lift from extra rain, minus what the district loses in a normal year. Red means even that rain does not keep up.",
         value: (name) => { const d = D[name]; if (!d) return undefined;
                            const g = gainOf(d, state.delta); return g === undefined ? undefined : g - d.decline; },
         domain: [-0.3, 0.3], ticks: [-0.3, -0.15, 0, 0.15, 0.3] },
};

function colour(name) {
  const mode = MODES[state.mode];
  const value = mode.value(name);
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  const [lo, hi] = mode.domain;
  const t = (value - lo) / (hi - lo);
  return RAMP[mode.ramp](mode.flip ? 1 - t : t);
}

// ---- map ------------------------------------------------------------------------------------
const canvas = $("canvas"), ctx = canvas.getContext("2d");
const features = A.boundaries.features;
const RAD = Math.PI / 180;
const py = (lat) => -Math.log(Math.tan(Math.PI / 4 + (lat * RAD) / 2)) / RAD;
const VIEW = { lonMin: 67.6, lonMax: 97.8, top: py(37.4), bottom: py(6.3) };
let view = { w: 1, h: 1, dpr: 1, s: 1, ox: 0, oy: 0 };

const paths = features.map((feature) => {
  const path = new Path2D();
  const polygons = feature.geometry.type === "Polygon" ? [feature.geometry.coordinates] : feature.geometry.coordinates;
  for (const polygon of polygons) {
    for (const ring of polygon) {
      ring.forEach(([lon, lat], i) => (i ? path.lineTo(lon, py(lat)) : path.moveTo(lon, py(lat))));
      path.closePath();
    }
  }
  return { path, name: feature.properties.n, state: feature.properties.s };
});

function fit() {
  const rect = canvas.parentElement.getBoundingClientRect();
  view.dpr = window.devicePixelRatio || 1;
  view.w = rect.width; view.h = rect.height;
  canvas.width = Math.round(rect.width * view.dpr);
  canvas.height = Math.round(rect.height * view.dpr);
  const spanX = VIEW.lonMax - VIEW.lonMin, spanY = VIEW.bottom - VIEW.top;
  view.s = Math.min(view.w / spanX, view.h / spanY);
  view.ox = (view.w - view.s * spanX) / 2;
  view.oy = (view.h - view.s * spanY) / 2;
  draw();
}

function draw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const styles = getComputedStyle(document.documentElement);
  const token = (name) => styles.getPropertyValue(name).trim();
  ctx.fillStyle = token("--sea");
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const k = view.s * view.dpr;
  ctx.setTransform(k, 0, 0, k, view.dpr * (view.ox - VIEW.lonMin * view.s), view.dpr * (view.oy - VIEW.top * view.s));

  const missing = token("--raised");
  for (const entry of paths) {
    ctx.fillStyle = colour(entry.name) ?? missing;
    ctx.fill(entry.path);
  }
  ctx.lineWidth = 0.35 / view.s;
  ctx.strokeStyle = token("--rule");
  for (const entry of paths) ctx.stroke(entry.path);

  // the selected district gets a white underlay so its outline reads against any fill colour
  for (const highlight of [state.hover, state.picked]) {
    if (!highlight) continue;
    const entry = paths.find((p) => p.name === highlight);
    if (!entry) continue;
    if (highlight === state.picked) {
      ctx.lineWidth = 5 / view.s;
      ctx.strokeStyle = token("--surface");
      ctx.stroke(entry.path);
    }
    ctx.lineWidth = (highlight === state.picked ? 2.4 : 1.2) / view.s;
    ctx.strokeStyle = token("--ink");
    ctx.stroke(entry.path);
  }
}

function districtAt(event) {
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left - view.ox) / view.s + VIEW.lonMin;
  const y = (event.clientY - rect.top - view.oy) / view.s + VIEW.top;
  const k = view.s * view.dpr;
  ctx.setTransform(k, 0, 0, k, view.dpr * (view.ox - VIEW.lonMin * view.s), view.dpr * (view.oy - VIEW.top * view.s));
  for (const entry of paths) if (ctx.isPointInPath(entry.path, x, y)) return entry;
  return null;
}

canvas.addEventListener("pointermove", (event) => {
  const entry = districtAt(event);
  state.hover = entry ? entry.name : null;
  const mode = MODES[state.mode];
  const value = entry ? mode.value(entry.name) : undefined;
  const absent = mode.source === "rain" ? "no rain data here" : "no wells here";
  $("readout").textContent = !entry ? ""
    : value === undefined || value === null ? `${entry.name}, ${entry.state}\n${absent}`
      : `${entry.name}, ${entry.state}\n${mode.title}: ${num(value)}`;
  $("readout").style.whiteSpace = "pre";
  draw();
});
canvas.addEventListener("pointerleave", () => { state.hover = null; $("readout").textContent = ""; draw(); });
canvas.addEventListener("click", (event) => {
  const entry = districtAt(event);
  if (entry && D[entry.name]) { state.picked = entry.name; $("pick").value = entry.name; renderDetail(); draw(); }
});

// ---- legend ---------------------------------------------------------------------------------
function legend() {
  const mode = MODES[state.mode];
  const bar = $("legendBar"), g = bar.getContext("2d");
  for (let i = 0; i < bar.width; i++) {
    const t = i / (bar.width - 1);
    g.fillStyle = RAMP[mode.ramp](mode.flip ? 1 - t : t);
    g.fillRect(i, 0, 1, bar.height);
  }
  const [lo, hi] = mode.domain;
  $("legendTicks").replaceChildren(...mode.ticks.map((value, i) => {
    const span = document.createElement("span");
    span.textContent = value;
    const t = (value - lo) / (hi - lo);
    span.style.left = `${t * 100}%`;
    span.style.transform = i === 0 ? "none" : i === mode.ticks.length - 1 ? "translateX(-100%)" : "translateX(-50%)";
    return span;
  }));
  $("legendTitle").textContent = mode.legend;
  $("mapTitle").textContent = mode.title;
  // the masthead counter follows the mode, or it contradicts the panel beside it
  $("coverage").textContent = mode.source === "rain"
    ? `${META.rainDistricts} districts · ${RAIN_YEARS[0]}–${RAIN_YEARS[RAIN_YEARS.length - 1]} · rain`
    : `${META.districts} districts · ${YEARS[0]}–${YEARS[YEARS.length - 1]} · wells`;
  const covered = mode.source === "rain"
    ? `${META.rainDistricts ?? "—"} districts with rain data`
    : `${META.districts} districts with wells`;
  $("mapNote").textContent = `${mode.timed ? state.year : state.mode === "decline" ? "2000–2022" : `monsoon ${pct(state.delta)}`} · ${covered}`;
  $("explain").textContent = mode.explain;

  // the year range follows the data: rain reaches back to 1998, wells only to 2000
  const years = mode.source === "rain" ? RAIN_YEARS : YEARS;
  const slider = $("year");
  slider.min = years[0];
  slider.max = years[years.length - 1];
  if (state.year < years[0]) { state.year = years[0]; slider.value = state.year; $("yearOut").textContent = state.year; }
  slider.disabled = !mode.timed;
}

// ---- charts ---------------------------------------------------------------------------------
function timelineChart(district) {
  const w = 340, h = 170, l = 34, r = 30, t = 10, b = 20;
  const depth = district.depth;
  // one axis of years for both series: rain reaches back further than the wells, so the well
  // series is padded at the front rather than stretched across a different span
  const offset = YEARS[0] - RAIN_YEARS[0];
  const rain = RAIN_YEARS.map((year) => rainAt(district.name, "jjas", year) ?? null);
  const known = depth.filter((v) => v !== null);
  if (!known.length && !rain.some((v) => v !== null)) return "";
  const dMax = Math.max(...known, 1) * 1.1, rMax = Math.max(...rain.filter((v) => v !== null), 1);
  const x = (i) => l + (i / (RAIN_YEARS.length - 1)) * (w - l - r);
  const yd = (v) => t + (v / dMax) * (h - t - b);          // depth grows downward
  const yr = (v) => h - b - (v / rMax) * (h - t - b) * 0.55;

  const bars = rain.map((v, i) => v === null ? "" :
    `<rect x="${x(i) - 3}" y="${yr(v)}" width="6" height="${h - b - yr(v)}" fill="var(--rain)" opacity=".28"/>`).join("");
  let line = "", open = false;
  depth.forEach((v, i) => {
    if (v === null) { open = false; return; }
    line += `${open ? "L" : "M"}${x(i + offset).toFixed(1)},${yd(v).toFixed(1)}`;
    open = true;
  });
  const marker = MODES[state.mode].timed && RAIN_YEARS.includes(state.year)
    ? `<line x1="${x(RAIN_YEARS.indexOf(state.year))}" x2="${x(RAIN_YEARS.indexOf(state.year))}" y1="${t}" y2="${h - b}"
         stroke="var(--ink)" stroke-dasharray="3 3" opacity=".5"/>` : "";
  return `${bars}${marker}<path d="${line}" fill="none" stroke="var(--gw)" stroke-width="1.8"/>
    <text x="${l}" y="${h - 6}">${RAIN_YEARS[0]}</text>
    <text x="${w - r}" y="${h - 6}" text-anchor="end">${RAIN_YEARS[RAIN_YEARS.length - 1]}</text>
    <text x="2" y="${t + 8}">0 m</text><text x="2" y="${h - b}">${dMax.toFixed(0)} m</text>
    <text x="${w - 2}" y="${t + 8}" text-anchor="end" style="fill:var(--rain)">${Math.round(rMax)} mm</text>
    <text x="${w - 2}" y="${t + 19}" text-anchor="end" style="fill:var(--rain)">rain</text>
    <text x="2" y="${t + 19}" style="fill:var(--gw)">water depth</text>`;
}

function scenarioChart(district) {
  const w = 340, h = 130, l = 34, r = 12, t = 24, b = 24;
  const deltas = META.deltas.filter((d) => gainOf(district, d) !== undefined);
  if (!deltas.length) return "";
  const gains = deltas.map((d) => gainOf(district, d));
  const lo = Math.min(0, district.decline, ...gains), hi = Math.max(0.05, district.decline, ...gains);
  const x = (d) => l + ((d - deltas[0]) / (deltas[deltas.length - 1] - deltas[0] || 1)) * (w - l - r);
  const y = (v) => h - b - ((v - lo) / (hi - lo)) * (h - t - b);
  const line = deltas.map((d, i) => `${i ? "L" : "M"}${x(d).toFixed(1)},${y(gains[i]).toFixed(1)}`).join(" ");
  const dots = deltas.map((d, i) => `<circle cx="${x(d)}" cy="${y(gains[i])}" r="${d === state.delta ? 4.5 : 2.5}"
      fill="${d === state.delta ? "var(--ink)" : "var(--rain)"}"/>`).join("");
  // Label on the left, where the curve is lowest: anchored right it lands on the line's own
  // right-hand end, which is exactly where the gain curve arrives.
  const labelBelow = y(district.decline) < t + 16;
  const declineLine = district.decline > 0
    ? `<line x1="${l}" x2="${w - r}" y1="${y(district.decline)}" y2="${y(district.decline)}" stroke="var(--gw)"
         stroke-width="1.4" stroke-dasharray="5 4"/>
       <text x="${l + 2}" y="${y(district.decline) + (labelBelow ? 12 : -5)}"
         style="fill:var(--gw)">what it loses in a year</text>` : "";
  const ticks = deltas.map((d) => `<text x="${x(d)}" y="${h - 6}" text-anchor="middle">${pct(d)}</text>`).join("");
  return `${declineLine}<path d="${line}" fill="none" stroke="var(--rain)" stroke-width="2"/>${dots}${ticks}
    <text x="${l}" y="12">metres the water rises ↑</text>`;
}

// ---- detail ----------------------------------------------------------------------------------
function renderDetail() {
  const district = D[state.picked];
  if (!district) { $("detailTitle").textContent = "Pick a district"; return; }
  const gain = gainOf(district, state.delta), net = gain === undefined ? undefined : gain - district.decline;
  const index = YEARS.indexOf(state.year);
  $("detailTitle").textContent = `${district.name}, ${district.state}`;
  $("detailWells").textContent = `${district.wells} wells`;
  $("facts").innerHTML = [
    [`Water in ${state.year}`, index >= 0 ? `${num(district.depth[index])} m down` : "–"],
    [`Monsoon ${state.year}`, `${num(rainAt(district.name, "jjas", state.year), 0)} mm`],
    ["Dropping by", `${district.decline >= 0 ? "+" : ""}${num(district.decline)} m a year`],
    ["Wells range from", `${num(district.ci_low)} to ${num(district.ci_high)}`],
    [`If monsoon ${pct(state.delta)}`, gain === undefined ? "–" : `lifts ${num(gain)} m`],
    ["After the year's loss", net === undefined ? "–" : `${net >= 0 ? "+" : ""}${num(net)} m`],
    ["Rain needed to break even", district.decline <= 0 ? "none needed"
      : district.break_even === null ? "more than +100%" : pct(district.break_even)],
  ].map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");

  const falling = district.decline > 0;
  const covered = net !== undefined && net >= 0;
  $("verdict").hidden = false;
  $("verdict").className = `verdict ${falling && !covered ? "falling" : "holding"}`;
  $("verdictHead").textContent = !falling
    ? "The water here is not dropping"
    : covered ? `${pct(state.delta)} more rain is enough`
      : `${pct(state.delta)} more rain is not enough`;
  $("verdictBody").textContent = !falling
    ? "Extra rain would still help, but there is nothing to make up here."
    : covered ? `It lifts the water about ${num(gain)} m, and the district loses ${num(district.decline)} m a year.`
      : `It lifts about ${num(gain)} m, but the district loses ${num(district.decline)} m a year${
          district.break_even === null ? ". No realistic monsoon closes that gap."
            : `. It would take ${pct(district.break_even)} more rain to break even.`}`;

  $("timeline").innerHTML = timelineChart(district);
  $("scenario").innerHTML = scenarioChart(district);
  $("detailNote").textContent = district.cgwb.length > 1
    ? `The data lists this district under two names, ${district.cgwb.join(" and ")}. Both are counted here.`
    : "";
}

// ---- controls ---------------------------------------------------------------------------------
$("delta").innerHTML = META.deltas.map((d) =>
  `<option value="${d}"${d === 0.2 ? " selected" : ""}>${d === 0 ? "as normal" : pct(d)}</option>`).join("");
$("pick").innerHTML = ['<option value="">—</option>', ...Object.values(D)
  .sort((a, b) => a.name.localeCompare(b.name))
  .map((d) => `<option value="${d.name}">${d.name}, ${d.state}</option>`)].join("");

$("mode").addEventListener("change", (e) => { state.mode = e.target.value; legend(); draw(); renderDetail(); });
$("year").addEventListener("input", (e) => {
  state.year = Number(e.target.value); $("yearOut").textContent = state.year;
  legend(); draw(); renderDetail();
});
$("delta").addEventListener("change", (e) => { state.delta = Number(e.target.value); legend(); draw(); renderDetail(); });
$("pick").addEventListener("change", (e) => { state.picked = e.target.value || null; renderDetail(); draw(); });

let timer = null;
$("play").addEventListener("click", () => {
  if (timer) { clearInterval(timer); timer = null; }
  else timer = setInterval(() => {
    state.year = state.year >= YEARS[YEARS.length - 1] ? YEARS[0] : state.year + 1;
    $("year").value = state.year; $("yearOut").textContent = state.year;
    legend(); draw(); renderDetail();
  }, 850);
  $("play").textContent = timer ? "Pause" : "Play";
  $("play").setAttribute("aria-pressed", String(Boolean(timer)));
});

$("caveat").innerHTML = `<b>These are estimates, not forecasts.</b> We do not predict next year's rain — nobody can,
  from rainfall records alone. You choose the monsoon size and the model answers for that.
  <b>Grey areas have no data.</b> Rain is shown for ${META.rainDistricts} districts from ${RAIN_YEARS[0]}; water levels
  only for the ${META.districts} districts that have monitored wells, from ${YEARS[0]}. Many states, including
  Rajasthan, have no wells in this dataset at all.
  <b>One thing is left out on purpose.</b> We tried to split each district's fall into "caused by weather" and
  "caused by pumping". That split gave different answers on different halves of the record, so it is not shown here.`;

new ResizeObserver(fit).observe($("map"));
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
const first = Object.values(D).sort((a, b) => b.decline - a.decline)[0];
if (first) { state.picked = first.name; $("pick").value = first.name; }
fit(); legend(); renderDetail();
