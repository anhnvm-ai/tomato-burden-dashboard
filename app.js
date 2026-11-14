/* Results explorer: static port of the Dash dashboard (Code/demo/dashboard.py). Same data, filters, views,
   insight sentences and drill-down; everything is computed in the browser from data/results.json. */
'use strict';

const INK = '#1b1b1f', MUTED = '#5b5b66', GRID = '#e6e6ea', SURFACE = '#ffffff';
const BLUE = '#2a78d6', ORANGE = '#eb6834';
const RAMP = ['#eaf2fc', '#c3d9f4', '#8fb9e8', '#5795da', '#2a78d6', '#1b5aa6', '#0f3b72'];
const BAND_COLORS = ['#c3d9f4', '#8fb9e8', '#5795da', '#2a78d6', '#0f3b72'];
const BAND_EDGES = [10, 25, 50, 75];
const BAND_LABELS = ['0 to 10%', '10 to 25%', '25 to 50%', '50 to 75%', '75 to 100%'];
const HEAT = {gh: ['GH MAE (pp)', 'gh', 'low'], e2e: ['E2E MAE (pp)', 'e2e', 'low'], dice: ['Lesion Dice', 'dice', 'high']};
const METRICS = ['dice', 'host', 'gh', 'e2e', 'drift', 'agree'];
const LEGEND = 'Cyan = host, red = lesion inside the host, orange = lesion outside the host. ' +
  'Host error map: blue = reference host the network missed, orange = extra predicted host.';
const CONFIG = {displayModeBar: false, responsive: true};

let D, COST, ROWS, NETS, DISEASES, GROUP_DISEASES;
const state = {network: 'MNv4-FPN', precision: 'INT8', group: 'All', disease: 'All', metric: 'gh', seed: 20260924, picked: null};

/* ------------------------------------------------------------------ small helpers */
const valid = x => x !== null && x !== undefined && !Number.isNaN(x);
const mean = a => { const v = a.filter(valid); return v.length ? v.reduce((s, x) => s + x, 0) / v.length : NaN; };
const std = a => {
  const v = a.filter(valid); if (v.length < 2) return NaN;
  const m = mean(v); return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - 1));
};
const pct = x => Math.round(x).toString();            // like Python f"{x:.0f}" for the shares used here
const f = (x, d) => x.toFixed(d);
function signed(value, digits = 3) {
  const text = (value >= 0 ? '+' : '') + value.toFixed(digits);
  return Number(text) === 0 ? '+' + Math.abs(value).toFixed(digits) : text;   // no "-0.000"
}
function groupBy(rows, key) {
  const m = new Map();
  for (const r of rows) { const k = key(r); if (!m.has(k)) m.set(k, []); m.get(k).push(r); }
  return m;
}
const argBest = (entries, better) => entries.filter(([, v]) => valid(v))
  .reduce((best, e) => (best === null || (better === 'low' ? e[1] < best[1] : e[1] > best[1]) ? e : best), null);
const el = id => document.getElementById(id);
function setInsight(id, rq, text) {
  const p = el(id); p.textContent = '';
  if (rq) { const s = document.createElement('span'); s.className = 'rq'; s.textContent = rq; p.appendChild(s); }
  p.appendChild(document.createTextNode(text));
}

/* ------------------------------------------------------------------ data */
function subset(precision, group, disease) {
  return ROWS.filter(r => r.prec === precision && (group === 'All' || r.group === group) &&
    (disease === 'All' || r.disease === disease));
}
function seedSummary(rows) {
  /* mean over images within each seed, then mean and SD over the seeds (as in dashboard.seed_summary) */
  const out = new Map();
  for (const net of NETS) {
    const netRows = rows.filter(r => r.net === net);
    if (!netRows.length) continue;
    const perSeed = {}; METRICS.forEach(k => { perSeed[k] = []; });
    for (const rs of groupBy(netRows, r => r.seed).values()) METRICS.forEach(k => perSeed[k].push(mean(rs.map(r => r[k]))));
    const s = {}; METRICS.forEach(k => { s[k] = mean(perSeed[k]); s[k + 'Sd'] = std(perSeed[k]); });
    out.set(net, s);
  }
  return out;
}
function band(v) { let b = 0; for (const e of BAND_EDGES) if (v >= e) b += 1; return b; }

/* ------------------------------------------------------------------ figures */
function baseLayout(title, height = 340) {
  return {
    title: {text: title, x: 0, xanchor: 'left', font: {size: 15, color: INK}}, height,
    margin: {l: 10, r: 16, t: 48, b: 10}, paper_bgcolor: SURFACE, plot_bgcolor: SURFACE,
    font: {family: 'Segoe UI, Arial, sans-serif', size: 12, color: MUTED},
    legend: {orientation: 'h', y: 1.02, yanchor: 'bottom', x: 1, xanchor: 'right'},
    hoverlabel: {bgcolor: 'white', font: {color: INK}},
    xaxis: {gridcolor: GRID, zeroline: false, linecolor: GRID, automargin: true},
    yaxis: {gridcolor: GRID, zeroline: false, linecolor: GRID, automargin: true},
  };
}
function emptyFigure(div, title, message, height = 340) {
  const layout = baseLayout(title, height);
  layout.xaxis.visible = false; layout.yaxis.visible = false;
  layout.annotations = [{text: message, x: 0.5, y: 0.5, xref: 'paper', yref: 'paper', showarrow: false,
    font: {color: MUTED, size: 13}}];
  Plotly.react(div, [], layout, CONFIG);
}

function kpiCards(rows, summary) {
  const box = el('kpis'); box.textContent = '';
  const card = (label, value, note) => {
    const d = document.createElement('div'); d.className = 'kpi';
    [['kpi-label', label], ['kpi-value', value], ['kpi-note', note]].forEach(([c, t]) => {
      const x = document.createElement('div'); x.className = c; x.textContent = t; d.appendChild(x);
    });
    box.appendChild(d);
  };
  if (!rows.length) { card('Images', '0', 'no image matches these filters'); return; }
  const net = state.network, prec = state.precision;
  if (!summary.has(net)) { card(net, 'n/a', `no ${prec} model`); return; }
  const s = summary.get(net);
  const driftRows = rows.filter(r => r.net === net && valid(r.drift));
  const large = driftRows.length ? 100 * driftRows.filter(r => r.drift > 5).length / driftRows.length : NaN;
  const cost = COST[net];
  const latency = cost ? cost[prec === 'INT8' ? 'int8_1t_ms' : 'fp32_1t_ms'] : NaN;
  const images = new Set(rows.map(r => r.sample)).size;
  card('Lesion Dice', f(s.dice, 3), `${prec}, ${images} images`);
  card('Host Dice', f(s.host, 3), 'vs reference host');
  card('GH MAE', `${f(s.gh, 2)} pp`, 'reference host');
  card('E2E MAE', `${f(s.e2e, 2)} pp`, 'predicted host');
  card('GH drift', valid(s.drift) ? `${f(s.drift, 2)} pp` : 'n/a', valid(large) ? `${pct(large)}% of pairs > 5 pp` : 'FP32 vs INT8');
  card('Latency, 1 thread', valid(latency) ? `${Math.round(latency)} ms` : 'n/a', `${prec} on CPU`);
}

function errorBars(summary) {
  const names = [...summary.keys()].reverse();
  const traces = [['e2e', 'Predicted host (E2E)', ORANGE], ['gh', 'Reference host (GH)', BLUE]].map(([c, label, color]) => ({
    type: 'bar', orientation: 'h', y: names, x: names.map(n => summary.get(n)[c]), name: label,
    marker: {color, cornerradius: 4},
    error_x: {type: 'data', array: names.map(n => summary.get(n)[c + 'Sd']), color: MUTED, thickness: 1},
    hovertemplate: '%{y}<br>' + label + ': %{x:.2f} pp<extra></extra>',
  }));
  const layout = baseLayout(`Burden error by network (${state.precision})`);
  Object.assign(layout, {barmode: 'group', bargap: 0.3, bargroupgap: 0.08});
  layout.legend.traceorder = 'reversed';
  layout.xaxis.title = {text: 'Burden MAE (pp), mean of three seeds'};
  Plotly.react('g-error', traces, layout, CONFIG);
  const entries = k => [...summary.entries()].map(([n, s]) => [n, s[k]]);
  const [bestGh, vGh] = argBest(entries('gh'), 'low'), [bestE2e, vE2e] = argBest(entries('e2e'), 'low');
  let text = `Lowest GH MAE: ${bestGh} (${f(vGh, 2)} pp); lowest E2E MAE: ${bestE2e} (${f(vE2e, 2)} pp).`;
  if (summary.has(state.network)) {
    const s = summary.get(state.network);
    text += ` For ${state.network} the predicted host changes the MAE by ${signed(s.e2e - s.gh, 2)} pp.`;
  }
  setInsight('i-error', 'RQ1', text);
}

function heatmap() {
  const [label, col, better] = HEAT[state.metric];
  const rows = subset(state.precision, state.group, 'All');
  const diseases = [...new Set(rows.map(r => r.disease))].sort();
  const nets = NETS.filter(n => rows.some(r => r.net === n));
  const cell = (d, n) => {
    const rs = rows.filter(r => r.disease === d && r.net === n);
    return rs.length ? mean([...groupBy(rs, r => r.seed).values()].map(v => mean(v.map(r => r[col])))) : NaN;
  };
  const z = diseases.map(d => nets.map(n => cell(d, n)));
  const fmt = state.metric === 'dice' ? v => f(v, 3) : v => f(v, 1);
  const flat = z.flat().filter(valid), low = Math.min(...flat), high = Math.max(...flat);
  const annotations = [];
  diseases.forEach((d, i) => nets.forEach((n, j) => {
    const v = z[i][j]; if (!valid(v)) return;
    annotations.push({x: n, y: d, text: fmt(v), showarrow: false,
      font: {size: 11, color: (v - low) / Math.max(high - low, 1e-9) > 0.55 ? 'white' : INK}});
  }));
  const outline = {color: ORANGE, width: 2}, shapes = [];
  const j = nets.indexOf(state.network), i = diseases.indexOf(state.disease);
  if (j >= 0) shapes.push({type: 'rect', xref: 'x', yref: 'paper', x0: j - 0.5, x1: j + 0.5, y0: 0, y1: 1, line: outline});
  if (i >= 0) shapes.push({type: 'rect', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: i - 0.5, y1: i + 0.5, line: outline});
  const trace = {type: 'heatmap', z: z.map(r => r.map(v => (valid(v) ? v : null))), x: nets, y: diseases, xgap: 2, ygap: 2,
    colorscale: RAMP.map((c, k) => [k / (RAMP.length - 1), c]),
    colorbar: {title: {text: label, side: 'right'}, thickness: 12, outlinewidth: 0},
    hovertemplate: '%{y}<br>%{x}<br>' + label + ': %{z:.3f}<extra></extra>'};
  const layout = baseLayout(`${label.split(' (')[0]} by disease and network (${state.precision})`, 360);
  layout.annotations = annotations; layout.shapes = shapes;
  Object.assign(layout.xaxis, {tickangle: -30, showgrid: false});
  Object.assign(layout.yaxis, {showgrid: false, autorange: 'reversed'});
  Plotly.react('g-heatmap', [trace], layout, CONFIG);

  const diseaseMeans = z.map(r => mean(r));
  const networkMeans = [...seedSummary(rows).values()].map(s => s[state.metric]);
  let text = '';
  if (j >= 0) {
    const worst = argBest(diseases.map((d, k) => [d, z[k][j]]), better === 'low' ? 'high' : 'low');
    text = `${state.network} is weakest on ${worst[0]} (${fmt(worst[1])}). `;
  }
  const dMin = Math.min(...diseaseMeans.filter(valid)), dMax = Math.max(...diseaseMeans.filter(valid));
  const nMin = Math.min(...networkMeans.filter(valid)), nMax = Math.max(...networkMeans.filter(valid));
  text += `The disease moves the ${label.split(' (')[0]} ${dMax - dMin > nMax - nMin ? 'more' : 'less'} than the ` +
    `network does (${fmt(dMin)} to ${fmt(dMax)} across diseases, ${fmt(nMin)} to ${fmt(nMax)} across networks with the diseases pooled).`;
  setInsight('i-heatmap', 'RQ1', text);
}

function latencyScatter() {
  const all = seedSummary(subset('INT8', state.group, state.disease));
  const nets = [...all.keys()].filter(n => COST[n]);
  const trace = {type: 'scatter', mode: 'markers+text', x: nets.map(n => COST[n].int8_1t_ms), y: nets.map(n => all.get(n).dice),
    text: nets, textposition: nets.map(n => (n === 'MNv4-FPN (IN)' ? 'bottom center' : 'top center')),
    textfont: {color: INK, size: 11},
    marker: {size: 11, color: nets.map(n => (n === state.network ? ORANGE : BLUE)), line: {color: 'white', width: 2}},
    customdata: nets.map(n => [all.get(n).e2e, COST[n].int8_size_mb]),
    hovertemplate: '%{text}<br>Lesion Dice %{y:.3f}<br>%{x:.0f} ms, 1 thread<br>E2E MAE %{customdata[0]:.2f} pp' +
      '<br>%{customdata[1]:.1f} MB<extra></extra>'};
  const layout = baseLayout('Accuracy and CPU latency after INT8 PTQ');
  Object.assign(layout.xaxis, {title: {text: 'INT8 latency per image, 1 thread (ms, log scale)'}, type: 'log',
    range: [Math.log10(40), Math.log10(700)]});
  layout.yaxis.title = {text: 'Lesion Dice (INT8)'};
  Plotly.react('g-latency', [trace], layout, CONFIG);
  let text;
  if (nets.includes(state.network)) {
    const fp32 = seedSummary(subset('FP32', state.group, state.disease)).get(state.network);
    const fast = COST[state.network].int8_1t_ms, slow = COST[state.network].fp32_1t_ms;
    text = `${state.network}: INT8 runs ${f(slow / fast, 1)} times faster than FP32 (${Math.round(slow)} to ${Math.round(fast)} ms on one ` +
      `thread, ${f(COST[state.network].int8_size_mb, 1)} MB) for a lesion Dice change of ${signed(all.get(state.network).dice - fp32.dice)}.`;
  } else {
    text = `${state.network} has no INT8 model, so it is not on this chart.`;
  }
  setInsight('i-latency', 'RQ2', text);
}

function radar() {
  const all = seedSummary(subset(state.precision, state.group, state.disease));
  const nets = [...all.keys()].filter(n => COST[n]);
  const speed = state.precision === 'INT8' ? 'int8_1t_ms' : 'fp32_1t_ms';
  const value = (n, c) => (c in COST[n] ? COST[n][c] : all.get(n)[c]);
  const axes = [['Lesion Dice', 'dice', 'high'], ['Host Dice', 'host', 'high'], ['GH MAE', 'gh', 'low'], ['E2E MAE', 'e2e', 'low'],
    ['Speed', speed, 'low']];
  if (state.precision === 'INT8') axes.push(['Drift', 'drift', 'low'], ['Size', 'int8_size_mb', 'low']);
  const scores = new Map(nets.map(n => [n, axes.map(([, c, dir]) => {
    const column = nets.map(m => value(m, c));
    return dir === 'high' ? value(n, c) / Math.max(...column) : Math.min(...column) / value(n, c);
  })]));
  const theta = [...axes.map(a => a[0]), axes[0][0]];
  const traces = [];
  nets.filter(n => n !== state.network).forEach((n, k) => {
    const r = scores.get(n);
    traces.push({type: 'scatterpolar', r: [...r, r[0]], theta, mode: 'lines', name: 'Other networks', legendgroup: 'others',
      showlegend: k === 0, line: {color: '#a9abb6', width: 1}, hovertemplate: n + '<br>%{theta}: %{r:.2f} of the best<extra></extra>'});
  });
  if (scores.has(state.network)) {
    const r = scores.get(state.network);
    traces.push({type: 'scatterpolar', r: [...r, r[0]], theta, mode: 'lines+markers', name: state.network, fill: 'toself',
      fillcolor: 'rgba(235, 104, 52, 0.16)', line: {color: ORANGE, width: 2}, marker: {size: 8, line: {color: 'white', width: 2}},
      hovertemplate: state.network + '<br>%{theta}: %{r:.2f} of the best<extra></extra>'});
  }
  const layout = baseLayout(`Network profile, ${state.precision} (1 = best of the six networks)`);
  layout.margin = {l: 40, r: 40, t: 60, b: 24};
  layout.polar = {bgcolor: SURFACE,
    radialaxis: {range: [0, 1.05], tickvals: [0.5, 1], gridcolor: GRID, angle: 25, tickangle: 25, linecolor: GRID,
      tickfont: {size: 9, color: MUTED}},
    angularaxis: {gridcolor: GRID, linecolor: GRID, tickfont: {size: 12, color: INK}}};
  Plotly.react('g-radar', traces, layout, CONFIG);
  let text;
  if (scores.has(state.network)) {
    const s = scores.get(state.network);
    const best = axes.filter((a, k) => s[k] >= 0.999).map(a => a[0]);
    const w = s.indexOf(Math.min(...s));
    const lead = best.length ? `${state.network} is the best of the six on ${best.join(', ')}` : `${state.network} leads on no axis`;
    text = `${lead}, and weakest on ${axes[w][0]} (${pct(100 * s[w])}% of the best network).`;
  } else {
    text = `${state.network} has no latency benchmark or INT8 model, so it is not on the radar.`;
  }
  setInsight('i-radar', 'RQ2', text);
}

function driftHistogram() {
  const rows = subset('INT8', state.group, state.disease).filter(r => r.net === state.network);
  if (!rows.length) {
    emptyFigure('g-drift', `${state.network}: per-image GH drift`, 'No INT8 model for this network');
    setInsight('i-drift', 'RQ3', `${state.network} has no INT8 model.`); return;
  }
  const drift = rows.map(r => r.drift);
  const share = 100 * drift.filter(v => v > 5).length / drift.length, m = mean(drift);
  const trace = {type: 'histogram', x: drift, xbins: {start: 0, size: 1},
    marker: {color: BLUE, line: {color: 'white', width: 2}}, hovertemplate: '%{x} pp: %{y} image-seed pairs<extra></extra>'};
  const layout = baseLayout(`${state.network}: per-image GH drift, FP32 vs INT8 (mean ${f(m, 2)} pp)`);
  layout.shapes = [{type: 'line', x0: 5, x1: 5, y0: 0, y1: 1, yref: 'paper', line: {color: INK, dash: 'dash', width: 1}}];
  layout.annotations = [{x: 5, y: 1, yref: 'paper', xanchor: 'left', showarrow: false, text: `  ${pct(share)}% above 5 pp`,
    font: {color: INK}}];
  layout.xaxis.title = {text: 'Absolute burden change per image and seed (pp)'};
  layout.yaxis.title = {text: 'Image-seed pairs'};
  Plotly.react('g-drift', [trace], layout, CONFIG);
  const fp32 = seedSummary(subset('FP32', state.group, state.disease)).get(state.network);
  const int8 = seedSummary(rows).get(state.network);
  setInsight('i-drift', 'RQ3', `Mean lesion Dice changes by ${signed(int8.dice - fp32.dice)} after INT8, yet ${pct(share)}% of ` +
    `image-seed pairs move by more than 5 pp (mean ${f(m, 2)} pp), ` +
    (share >= 5 ? 'so a kept mean Dice does not guarantee a kept per-image burden.' : 'so the per-image burden of this network is stable.'));
}

function sankey() {
  const key = r => `${r.seed}|${r.sample}`;
  const fp = new Map(subset('FP32', state.group, state.disease).filter(r => r.net === state.network).map(r => [key(r), r.ghPred]));
  const pairs = subset('INT8', state.group, state.disease).filter(r => r.net === state.network && fp.has(key(r)))
    .map(r => [band(fp.get(key(r))), band(r.ghPred)]);
  const title = `${state.network}: burden band in FP32 and INT8 (reference host)`;
  if (!pairs.length) {
    emptyFigure('g-sankey', title, 'No INT8 model for this network');
    setInsight('i-sankey', 'RQ3', `${state.network} has no INT8 model.`); return;
  }
  const counts = Array.from({length: 5}, () => Array(5).fill(0));
  pairs.forEach(([a, b]) => { counts[a][b] += 1; });
  const left = [0, 1, 2, 3, 4].filter(i => counts[i].some(v => v)), right = [0, 1, 2, 3, 4].filter(j => counts.some(r => r[j]));
  const nodes = [...left.map(i => ['FP32', i]), ...right.map(j => ['INT8', j])];
  const pos = new Map(nodes.map(([side, b], k) => [`${side}${b}`, k]));
  const link = {source: [], target: [], value: [], color: []};
  left.forEach(i => right.forEach(j => {
    if (!counts[i][j]) return;
    link.source.push(pos.get(`FP32${i}`)); link.target.push(pos.get(`INT8${j}`)); link.value.push(counts[i][j]);
    link.color.push(i === j ? 'rgba(169, 171, 182, 0.35)' : (j > i ? 'rgba(235, 104, 52, 0.6)' : 'rgba(42, 120, 214, 0.6)'));
  }));
  link.hovertemplate = '%{source.label} to %{target.label}: %{value} pairs<extra></extra>';
  const trace = {type: 'sankey', arrangement: 'snap',
    node: {label: nodes.map(([side, b]) => `${side} ${BAND_LABELS[b]}`), pad: 12, thickness: 14,
      color: nodes.map(([, b]) => BAND_COLORS[b]), line: {color: 'white', width: 1},
      hovertemplate: '%{label}: %{value} image-seed pairs<extra></extra>'},
    link, textfont: {color: INK, size: 11}};
  Plotly.react('g-sankey', [trace], baseLayout(title), CONFIG);
  const moved = pairs.filter(([a, b]) => a !== b).length, up = pairs.filter(([a, b]) => b > a).length;
  setInsight('i-sankey', 'RQ3', `INT8 moves ${moved} of ${pairs.length} image-seed pairs (${pct(100 * moved / pairs.length)}%) to another ` +
    `burden band: ${up} up (orange) and ${moved - up} down (blue).`);
}

function perImageFp32() {
  /* FP32 errors of the selected network averaged over seeds, one entry per image */
  const rows = subset('FP32', state.group, state.disease).filter(r => r.net === state.network);
  return [...groupBy(rows, r => r.sample).entries()].map(([sample, rs]) => ({
    sample, disease: rs[0].disease, group: rs[0].group, gh: mean(rs.map(r => r.gh)), e2e: mean(rs.map(r => r.e2e))}));
}

function hostEffect(perImage) {
  const title = `${state.network}: error with the reference vs the predicted host (FP32)`;
  if (!perImage.length) {
    emptyFigure('g-host', title, 'No image matches these filters', 600);
    setInsight('i-host', 'RQ4', 'No image to show.'); return;
  }
  const traces = [['Focal', BLUE], ['Virus', ORANGE]].map(([name, color]) => {
    const part = perImage.filter(p => p.group === name);
    return part.length ? {type: 'scatter', mode: 'markers', name, x: part.map(p => p.gh), y: part.map(p => p.e2e),
      marker: {size: 9, color, opacity: 0.85, line: {color: 'white', width: 1}}, customdata: part.map(p => [p.sample, p.disease]),
      hovertemplate: '%{customdata[0]} (%{customdata[1]})<br>reference host %{x:.1f} pp<br>predicted host %{y:.1f} pp' +
        '<br>click to open<extra></extra>'} : null;
  }).filter(Boolean);
  const top = Math.max(...perImage.map(p => Math.max(p.gh, p.e2e)), 1);
  traces.push({type: 'scatter', mode: 'lines', x: [0, top], y: [0, top], line: {color: MUTED, dash: 'dot', width: 1},
    hoverinfo: 'skip', showlegend: false});
  const layout = baseLayout(title, 600);
  layout.xaxis.title = {text: 'Burden error with the reference host (pp)'};
  layout.yaxis.title = {text: 'Burden error with the predicted host (pp)'};
  Plotly.react('g-host', traces, layout, CONFIG);
  const lower = 100 * perImage.filter(p => p.e2e < p.gh).length / perImage.length;
  const change = mean(perImage.map(p => p.e2e)) - mean(perImage.map(p => p.gh));
  setInsight('i-host', 'RQ4', `The predicted host changes the mean error by ${signed(change, 2)} pp, yet lowers it on ${pct(lower)}% ` +
    'of images (points below the diagonal), so lesion and host errors interact rather than add.');
}

function summaryTable(summary) {
  const header = ['Network', 'Lesion Dice', 'Host Dice', 'GH MAE (pp)', 'E2E MAE (pp)', 'GH drift (pp)'];
  const t = document.createElement('table');
  const head = t.createTHead().insertRow();
  header.forEach(h => { const th = document.createElement('th'); th.textContent = h; head.appendChild(th); });
  const body = t.createTBody();
  for (const [n, s] of summary) {
    const row = body.insertRow();
    [n, f(s.dice, 3), f(s.host, 3), f(s.gh, 2), f(s.e2e, 2), valid(s.drift) ? f(s.drift, 2) : 'n/a']
      .forEach(v => { row.insertCell().textContent = v; });
  }
  const box = el('summary-table'); box.textContent = ''; box.appendChild(t);
}

/* ------------------------------------------------------------------ drill-down (stored masks, no model run) */
const imageCache = new Map();
function loadImage(src) {
  if (!imageCache.has(src)) {
    imageCache.set(src, new Promise((resolve, reject) => {
      const img = new Image(); img.onload = () => resolve(img); img.onerror = reject; img.src = src;
    }));
  }
  return imageCache.get(src);
}
function pixels(img) {
  const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
  const g = c.getContext('2d', {willReadFrequently: true}); g.drawImage(img, 0, 0);
  return g.getImageData(0, 0, c.width, c.height);
}
function tint(base, layers) {
  const out = new Uint8ClampedArray(base);
  for (const [mask, [r, g, b], a] of layers) {
    for (let i = 0; i < mask.length; i++) if (mask[i]) {
      const k = 4 * i;
      out[k] = out[k] * (1 - a) + r * a; out[k + 1] = out[k + 1] * (1 - a) + g * a; out[k + 2] = out[k + 2] * (1 - a) + b * a;
    }
  }
  return out;
}
function panel(data, w, h, caption) {
  const d = document.createElement('div'); d.className = 'panel';
  if (!data) { d.className = 'panel empty'; d.textContent = caption; return d; }
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  c.getContext('2d').putImageData(new ImageData(data, w, h), 0, 0);
  const cap = document.createElement('div'); cap.className = 'caption'; cap.textContent = caption;
  d.append(c, cap); return d;
}
let drillToken = 0;
async function drill(perImage) {
  const box = el('drill');
  if (!perImage.length) { box.innerHTML = '<p class="hint">No image matches these filters.</p>'; return; }
  let picked = perImage.find(p => p.sample === state.picked);
  if (!picked) {   // default: the image where the predicted host lowers the error most (RQ4 interaction)
    const gains = perImage.map(p => p.gh - p.e2e), best = Math.max(...gains);
    picked = best > 0 ? perImage[gains.indexOf(best)] : perImage[gains.map(Math.abs).indexOf(Math.max(...gains.map(Math.abs)))];
  }
  const token = ++drillToken, net = state.network, seed = state.seed, sid = picked.sample;
  const rows = ROWS.filter(r => r.net === net && r.seed === seed && r.sample === sid);
  const fp32 = rows.find(r => r.prec === 'FP32'), int8 = rows.find(r => r.prec === 'INT8');
  const [photoImg, maskImg] = await Promise.all([loadImage(`data/images/${sid}.jpg`), loadImage(`data/masks/${sid}.png`)]);
  if (token !== drillToken) return;
  const photo = pixels(photoImg), masks = pixels(maskImg), w = photo.width, h = photo.height;
  const tile = name => {
    const k = D.maskTiles.indexOf(name); if (k < 0) return null;
    const lesion = new Uint8Array(w * h), host = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) { const p = 4 * (k * w * h + i); lesion[i] = masks.data[p] > 127; host[i] = masks.data[p + 1] > 127; }
    return {lesion, host};
  };
  const overlay = m => tint(photo.data, [[m.host.map((v, i) => v && !m.lesion[i]), [0, 200, 220], 0.35],
    [m.lesion.map((v, i) => v && m.host[i]), [230, 30, 30], 0.55], [m.lesion.map((v, i) => v && !m.host[i]), [255, 150, 0], 0.55]]);
  const gold = tile('gold'), fpMask = tile(`${net}|${seed}|FP32`), q8Mask = int8 ? tile(`${net}|${seed}|INT8`) : null;
  const faded = new Uint8ClampedArray(photo.data.length);
  for (let i = 0; i < faded.length; i++) faded[i] = (i % 4 === 3) ? 255 : photo.data[i] * 0.45 + 255 * 0.55;
  const hostError = tint(faded, [[gold.host.map((v, i) => v && fpMask.host[i]), [0, 200, 220], 0.30],
    [gold.host.map((v, i) => v && !fpMask.host[i]), [42, 120, 214], 0.85],
    [fpMask.host.map((v, i) => v && !gold.host[i]), [235, 104, 52], 0.85]]);
  const panels = [
    panel(overlay(gold), w, h, `Reference masks: burden ${f(fp32.gold, 1)}%`),
    panel(overlay(fpMask), w, h, `FP32: E2E ${f(fp32.e2ePred, 1)}%, GH ${f(fp32.ghPred, 1)}%, lesion Dice ${f(fp32.dice, 3)}`),
    q8Mask ? panel(overlay(q8Mask), w, h, `INT8: E2E ${f(int8.e2ePred, 1)}%, GH ${f(int8.ghPred, 1)}%, GH drift ${f(int8.drift, 1)} pp`)
      : panel(null, 0, 0, 'No INT8 model for this network'),
    panel(hostError, w, h, `FP32 host vs reference host: host Dice ${f(fp32.host, 3)}`),
  ];
  const meta = document.createElement('div'); meta.className = 'drill-meta';
  const b = document.createElement('b'); b.textContent = sid;
  meta.append(b, ` · ${picked.disease} · ${picked.group} · ${net} · seed ${seed}`);
  const grid = document.createElement('div'); grid.className = 'drill-grid'; grid.append(...panels);
  const insight = document.createElement('p'); insight.className = 'insight'; insight.id = 'i-drill';
  const legend = document.createElement('p'); legend.className = 'hint'; legend.textContent = LEGEND;
  box.textContent = ''; box.append(meta, grid, insight, legend);
  setInsight('i-drill', 'RQ4', `Here the predicted host ${fp32.e2e < fp32.gh ? 'lowers' : 'raises'} the burden error from ` +
    `${f(fp32.gh, 1)} to ${f(fp32.e2e, 1)} pp.`);
}

/* ------------------------------------------------------------------ page */
function syncDiseaseOptions() {
  const allowed = GROUP_DISEASES[state.group];
  if (state.disease !== 'All' && !allowed.includes(state.disease)) state.disease = 'All';
  const sel = el('f-disease'); sel.textContent = '';
  [['All', 'All diseases'], ...allowed.map(d => [d, d])].forEach(([v, t]) => sel.add(new Option(t, v, false, v === state.disease)));
}
function render() {
  const rows = subset(state.precision, state.group, state.disease);
  const summary = seedSummary(rows);
  kpiCards(rows, summary);
  if (!rows.length) return;
  errorBars(summary); heatmap(); latencyScatter(); radar(); driftHistogram(); sankey();
  const perImage = perImageFp32();
  hostEffect(perImage); summaryTable(summary);
  drill(perImage).catch(e => { el('drill').textContent = `Drill-down could not load: ${e}`; });
}
function radioValue(name) { return document.querySelector(`input[name="${name}"]:checked`).value; }

async function initExplorer() {
  [D, COST] = await Promise.all([fetch('data/results.json?v=20261114b').then(r => r.json()), fetch('data/cost.json?v=20261114b').then(r => r.json())]);
  NETS = D.networks; DISEASES = D.diseases;
  ROWS = D.net.map((_, i) => {
    const s = D.sample[i];
    const row = {net: D.networks[D.net[i]], seed: D.seeds[D.seed[i]], prec: D.precisions[D.prec[i]], sample: D.samples[s],
      disease: D.diseases[D.sampleDisease[s]], group: D.groups[D.sampleGroup[s]]};
    for (const k of ['dice', 'host', 'gold', 'ghPred', 'e2ePred', 'gh', 'e2e', 'drift', 'driftE2e', 'agree']) row[k] = D[k][i] === null ? NaN : D[k][i];
    return row;
  });
  GROUP_DISEASES = {All: DISEASES};
  for (const g of D.groups) GROUP_DISEASES[g] = [...new Set(ROWS.filter(r => r.group === g).map(r => r.disease))].sort();
  const net = el('f-network'); NETS.forEach(n => net.add(new Option(n, n, false, n === state.network)));
  syncDiseaseOptions();
  net.addEventListener('change', () => { state.network = net.value; render(); });
  el('f-disease').addEventListener('change', e => { state.disease = e.target.value; render(); });
  for (const [name, key] of [['precision', 'precision'], ['group', 'group'], ['metric', 'metric']]) {
    document.querySelectorAll(`input[name="${name}"]`).forEach(i => i.addEventListener('change', () => {
      state[key] = radioValue(name); if (key === 'group') syncDiseaseOptions(); render();
    }));
  }
  document.querySelectorAll('input[name="seed"]').forEach(i => i.addEventListener('change', () => {
    state.seed = Number(radioValue('seed')); drill(perImageFp32());
  }));
  render();
  el('g-heatmap').on('plotly_click', ev => {        // drill-down: filter the page to that network and disease
    const p = ev.points[0];
    if (NETS.includes(p.x)) { state.network = p.x; el('f-network').value = p.x; }
    if (GROUP_DISEASES[state.group].includes(p.y)) state.disease = p.y;
    syncDiseaseOptions(); render();
  });
  el('g-host').on('plotly_click', ev => {           // drill-down: open the clicked image
    const data = ev.points[0].customdata;
    if (data) { state.picked = data[0]; drill(perImageFp32()); }
  });
}

/* tabs; ?tab=estimator opens the estimator, &sample=<0..3> runs a sample (as in the Dash app) */
function showTab(name) {
  for (const t of ['explorer', 'estimator']) {
    el(`page-${t}`).classList.toggle('hidden', t !== name);
    el(`tab-${t}`).classList.toggle('active', t === name);
    el(`tab-${t}`).setAttribute('aria-selected', String(t === name));
  }
  if (name === 'explorer') window.dispatchEvent(new Event('resize'));
}
el('tab-explorer').addEventListener('click', () => showTab('explorer'));
el('tab-estimator').addEventListener('click', () => showTab('estimator'));
const params = new URLSearchParams(location.search);
if (params.get('tab') === 'estimator') showTab('estimator');
window.explorerReady = initExplorer().catch(e => { el('kpis').textContent = `Could not load the data: ${e}`; throw e; });
