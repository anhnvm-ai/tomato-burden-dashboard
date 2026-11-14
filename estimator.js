/* Burden estimator in the browser: the same pipeline as Code/demo/burden_core.py and service.py, run with
   ONNX Runtime Web (WebAssembly). 640 letterbox input with Pillow-style bilinear resampling and ImageNet
   normalization, logits moved to the 384 letterbox space, threshold at 0, burden = 100 * |S and H| / |H|. */
'use strict';

const EST = (() => {
  const DEPLOY = 640, EVAL_SIZE = 384, MAX_SIDE = 900;
  const MEAN = [0.485, 0.456, 0.406].map(Math.fround), STD = [0.229, 0.224, 0.225].map(Math.fround);
  const DRIFT_ALERT_PP = 5.0, OUTSIDE_HOST_ALERT = 0.20;
  const SAMPLES = ['val1_septoria_leaf_spot.jpg', 'val2_early_blight.jpg', 'val3_leaf_mold.jpg', 'val4_septoria_leaf_spot.jpg'];
  const STATUS = {error: '✖', warning: '⚠', info: 'ℹ'};
  let sessions = null, last = null;

  /* ---------------------------------------------------------------- geometry and resampling */
  function pyRound(x) {                       // Python round(): halves go to the even neighbour
    const r = Math.round(x);
    return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
  }
  function letterboxGeometry(width, height, size) {
    const scale = size / Math.max(width, height);
    const rw = Math.max(1, pyRound(width * scale)), rh = Math.max(1, pyRound(height * scale));
    return {size, width, height, rw, rh, left: Math.floor((size - rw) / 2), top: Math.floor((size - rh) / 2)};
  }
  const PREC = 22;                            // Pillow 8-bit resampling: PRECISION_BITS = 32 - 8 - 2
  function coeffs(inSize, outSize) {
    const scale = inSize / outSize, filterscale = Math.max(scale, 1), support = filterscale;
    const ksize = Math.ceil(support) * 2 + 1, bounds = new Int32Array(outSize * 2), k = new Float64Array(outSize * ksize);
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale, ss = 1 / filterscale;
      let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize; xmax -= xmin;
      let ww = 0;
      for (let x = 0; x < xmax; x++) {
        let t = Math.abs((x + xmin - center + 0.5) * ss); const w = t < 1 ? 1 - t : 0;
        k[xx * ksize + x] = w; ww += w;
      }
      for (let x = 0; x < xmax; x++) if (ww !== 0) k[xx * ksize + x] /= ww;
      bounds[2 * xx] = xmin; bounds[2 * xx + 1] = xmax;
    }
    for (let i = 0; i < k.length; i++) { const v = k[i] * (1 << PREC); k[i] = v < 0 ? Math.trunc(v - 0.5) : Math.trunc(v + 0.5); }
    return {ksize, bounds, k};
  }
  const clip8 = v => (v >= 255 * 2 ** PREC + 2 ** PREC ? 255 : v <= 0 ? 0 : Math.min(255, Math.floor(v / 2 ** PREC)));
  function resizeBilinear8(src, sw, sh, dw, dh) {
    /* Pillow Image.resize(BILINEAR) for RGB: horizontal pass, then vertical pass, 8-bit fixed point */
    let img = src, w = sw;
    if (dw !== sw) {
      const {ksize, bounds, k} = coeffs(sw, dw), out = new Uint8ClampedArray(dw * sh * 3);
      for (let y = 0; y < sh; y++) for (let x = 0; x < dw; x++) {
        const xmin = bounds[2 * x], n = bounds[2 * x + 1];
        for (let c = 0; c < 3; c++) {
          let s = 2 ** (PREC - 1);
          for (let i = 0; i < n; i++) s += img[(y * w + xmin + i) * 3 + c] * k[x * ksize + i];
          out[(y * dw + x) * 3 + c] = clip8(s);
        }
      }
      img = out; w = dw;
    }
    if (dh !== sh) {
      const {ksize, bounds, k} = coeffs(sh, dh), out = new Uint8ClampedArray(w * dh * 3);
      for (let y = 0; y < dh; y++) {
        const ymin = bounds[2 * y], n = bounds[2 * y + 1];
        for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
          let s = 2 ** (PREC - 1);
          for (let i = 0; i < n; i++) s += img[((ymin + i) * w + x) * 3 + c] * k[y * ksize + i];
          out[(y * w + x) * 3 + c] = clip8(s);
        }
      }
      img = out;
    }
    return img;
  }
  function preprocess(rgb, width, height) {
    const g = letterboxGeometry(width, height, DEPLOY);
    const resized = resizeBilinear8(rgb, width, height, g.rw, g.rh);
    const plane = DEPLOY * DEPLOY, x = new Float32Array(3 * plane);
    for (let c = 0; c < 3; c++) x.fill(Math.fround((0 - MEAN[c]) / STD[c]), c * plane, (c + 1) * plane);
    for (let y = 0; y < g.rh; y++) for (let xx = 0; xx < g.rw; xx++) {
      const o = (g.top + y) * DEPLOY + g.left + xx;
      for (let c = 0; c < 3; c++) {
        const v = Math.fround(resized[(y * g.rw + xx) * 3 + c] / 255);
        x[c * plane + o] = Math.fround(Math.fround(v - MEAN[c]) / STD[c]);
      }
    }
    return {tensor: new ort.Tensor('float32', x, [1, 3, DEPLOY, DEPLOY]), geometry: g};
  }
  function axis(outSize, inSize) {           // torch interpolate(bilinear, align_corners=False) sampling
    const low = new Int32Array(outSize), high = new Int32Array(outSize), frac = new Float32Array(outSize);
    for (let i = 0; i < outSize; i++) {
      const s = Math.max((i + 0.5) * (inSize / outSize) - 0.5, 0);
      low[i] = Math.min(Math.floor(s), inSize - 1); high[i] = Math.min(low[i] + 1, inSize - 1); frac[i] = s - low[i];
    }
    return {low, high, frac};
  }
  function toCanvas(logits, g) {
    /* crop the content of the 640 letterbox, resize it bilinearly into the 384 letterbox, pad with -20 */
    const goal = letterboxGeometry(g.width, g.height, EVAL_SIZE);
    const ay = axis(goal.rh, g.rh), ax = axis(goal.rw, g.rw);
    const at = (y, x) => logits[(g.top + y) * DEPLOY + g.left + x];
    const mask = new Uint8Array(EVAL_SIZE * EVAL_SIZE);
    for (let y = 0; y < goal.rh; y++) {
      const wy = ay.frac[y];
      for (let x = 0; x < goal.rw; x++) {
        const wx = ax.frac[x];
        const top = Math.fround(at(ay.low[y], ax.low[x]) * (1 - wx) + at(ay.low[y], ax.high[x]) * wx);
        const bottom = Math.fround(at(ay.high[y], ax.low[x]) * (1 - wx) + at(ay.high[y], ax.high[x]) * wx);
        mask[(goal.top + y) * EVAL_SIZE + goal.left + x] = Math.fround(top * (1 - wy) + bottom * wy) >= 0 ? 1 : 0;
      }
    }
    return {mask, goal};
  }
  function burden(s, h) {
    let inter = 0, area = 0;
    for (let i = 0; i < h.length; i++) { area += h[i]; inter += s[i] & h[i]; }
    return area === 0 ? 0 : 100 * inter / area;
  }

  /* ---------------------------------------------------------------- models */
  async function loadSessions(setStatus) {
    if (sessions) return sessions;
    ort.env.wasm.wasmPaths = new URL('lib/ort/', location.href).href;
    ort.env.wasm.numThreads = 1;
    setStatus('Loading the INT8 and FP32 models (about 38 MB, only the first time)…');
    const options = {executionProviders: ['wasm'], graphOptimizationLevel: 'all'};
    const [int8, fp32] = await Promise.all([
      ort.InferenceSession.create('models/mnv4_fpn_s20260924_int8.onnx', options),
      ort.InferenceSession.create('models/mnv4_fpn_s20260924_fp32.onnx', options)]);
    sessions = {int8, fp32};
    return sessions;
  }
  async function run(session, tensor, g) {
    const start = performance.now();
    const out = await session.run({[session.inputNames[0]]: tensor});
    const latency = performance.now() - start;
    const [sName, hName] = session.outputNames;
    const s = toCanvas(out[sName].data, g), h = toCanvas(out[hName].data, g);
    return {symptom: s.mask, host: h.mask, goal: s.goal, burden: burden(s.mask, h.mask), latency};
  }
  async function decode(blob) {
    const bitmap = await createImageBitmap(blob, {imageOrientation: 'from-image', colorSpaceConversion: 'none',
      premultiplyAlpha: 'none'});
    const c = document.createElement('canvas'); c.width = bitmap.width; c.height = bitmap.height;
    const ctx = c.getContext('2d', {willReadFrequently: true}); ctx.drawImage(bitmap, 0, 0);
    const rgba = ctx.getImageData(0, 0, c.width, c.height).data, rgb = new Uint8ClampedArray(c.width * c.height * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) { rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2]; }
    return {bitmap, rgb, width: c.width, height: c.height};
  }
  async function predict(blob, name, setStatus) {
    const s = await loadSessions(setStatus);
    setStatus('Running the models…');
    const picture = await decode(blob);
    const {tensor, geometry} = preprocess(picture.rgb, picture.width, picture.height);
    const int8 = await run(s.int8, tensor, geometry), fp32 = await run(s.fp32, tensor, geometry);
    let both = 0, sum = 0;
    for (let i = 0; i < int8.symptom.length; i++) { both += int8.symptom[i] & fp32.symptom[i]; sum += int8.symptom[i] + fp32.symptom[i]; }
    return {file: name, picture, int8, fp32, drift: Math.abs(int8.burden - fp32.burden), agreement: 2 * both / Math.max(1, sum)};
  }

  /* ---------------------------------------------------------------- alerts and view (as service.py) */
  function alerts(r, threshold) {
    const out = [], int8 = r.int8;
    let hostPixels = 0, lesion = 0, outside = 0;
    for (let i = 0; i < int8.host.length; i++) {
      hostPixels += int8.host[i]; lesion += int8.symptom[i]; outside += int8.symptom[i] & (1 - int8.host[i]);
    }
    if (hostPixels === 0) return [{level: 'error', code: 'NO_HOST', message: 'No leaf or fruit was detected, so the burden is undefined.'}];
    if (r.drift > DRIFT_ALERT_PP) out.push({level: 'warning', code: 'INT8_DRIFT', message: `INT8 and FP32 burdens differ by ` +
      `${r.drift.toFixed(1)} pp (> ${DRIFT_ALERT_PP.toFixed(0)} pp); check this image with the FP32 model.`});
    if (lesion && outside / lesion > OUTSIDE_HOST_ALERT) out.push({level: 'warning', code: 'LESION_OUTSIDE_HOST',
      message: `${Math.round(100 * outside / lesion)}% of the lesion pixels fall outside the predicted host, so the host mask may be too narrow.`});
    if (int8.burden >= threshold) out.push({level: 'info', code: 'HIGH_BURDEN',
      message: `Burden ${int8.burden.toFixed(1)}% is at or above the user threshold of ${Number(threshold).toFixed(0)}%.`});
    return out;
  }
  function overlayCanvas(r) {
    const {bitmap, width, height} = r.picture, scale = Math.min(1, MAX_SIDE / Math.max(width, height));
    const w = Math.max(1, Math.round(width * scale)), h = Math.max(1, Math.round(height * scale));
    const c = document.createElement('canvas'); c.width = w; c.height = h; c.className = 'overlay-canvas';
    const ctx = c.getContext('2d'); ctx.imageSmoothingQuality = 'high'; ctx.drawImage(bitmap, 0, 0, w, h);
    const img = ctx.getImageData(0, 0, w, h), d = img.data, g = r.int8.goal;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const gx = g.left + Math.min(g.rw - 1, Math.floor((x + 0.5) * g.rw / w));
      const gy = g.top + Math.min(g.rh - 1, Math.floor((y + 0.5) * g.rh / h));
      const s = r.int8.symptom[gy * EVAL_SIZE + gx], hh = r.int8.host[gy * EVAL_SIZE + gx];
      const layer = s && hh ? [[230, 30, 30], 0.55] : s ? [[255, 150, 0], 0.55] : hh ? [[0, 200, 220], 0.35] : null;
      if (!layer) continue;
      const k = 4 * (y * w + x), [[cr, cg, cb], a] = layer;
      d[k] = d[k] * (1 - a) + cr * a; d[k + 1] = d[k + 1] * (1 - a) + cg * a; d[k + 2] = d[k + 2] * (1 - a) + cb * a;
    }
    ctx.putImageData(img, 0, 0);
    return c;
  }
  function kpi(label, value, note) {
    const d = document.createElement('div'); d.className = 'kpi';
    [['kpi-label', label], ['kpi-value', value], ['kpi-note', note]].forEach(([c, t]) => {
      const x = document.createElement('div'); x.className = c; x.textContent = t; d.appendChild(x);
    });
    return d;
  }
  function show() {
    const box = document.getElementById('prediction'); box.textContent = '';
    if (!last) return;
    const threshold = Number(document.getElementById('threshold').value);
    const list = alerts(last, threshold);
    const shown = list.length ? list : [{level: 'info', code: 'OK', message: 'No alert for this image.'}];
    const result = document.createElement('div'); result.className = 'result';
    const side = document.createElement('div'); side.className = 'result-side';
    const title = document.createElement('h3'); title.textContent = last.file;
    const cards = document.createElement('div'); cards.className = 'kpis small';
    cards.append(kpi('Burden (INT8)', `${last.int8.burden.toFixed(1)}%`, 'deployed model'),
      kpi('Burden (FP32)', `${last.fp32.burden.toFixed(1)}%`, 'reference model'),
      kpi('INT8 drift', `${last.drift.toFixed(1)} pp`, `lesion agreement ${last.agreement.toFixed(3)}`),
      kpi('Browser latency', `${Math.round(last.int8.latency)} ms`, `FP32 ${Math.round(last.fp32.latency)} ms; native INT8: 68 ms`));
    const alertBox = document.createElement('div');
    shown.forEach(a => {
      const x = document.createElement('div'); x.className = `alert ${a.level}`;
      x.textContent = `${STATUS[a.level] || STATUS.info} ${a.code}: ${a.message}`; alertBox.appendChild(x);
    });
    const note = document.createElement('p'); note.className = 'footnote';
    note.textContent = 'Overlay: cyan = predicted host, red = lesion inside the host, orange = lesion outside the host.';
    side.append(title, cards, alertBox, note);
    result.append(overlayCanvas(last), side);
    box.appendChild(result);
  }

  /* ---------------------------------------------------------------- page wiring */
  const setStatus = t => { document.getElementById('status').textContent = t; };
  async function handle(blob, name) {
    try {
      last = await predict(blob, name, setStatus);
      setStatus(''); show();
    } catch (e) {
      last = null; show();
      setStatus(`✖ The image could not be processed: ${e.message || e}`);
    }
  }
  async function runSample(index) {
    const name = SAMPLES[index];
    const blob = await fetch(`samples/${name}`).then(r => r.blob());
    await handle(blob, name);
  }
  function init() {
    const samples = document.getElementById('samples');
    SAMPLES.forEach((name, i) => {
      const b = document.createElement('button'); b.className = 'sample';
      b.textContent = name.replace('.jpg', '').replace(/_/g, ' ');
      b.addEventListener('click', () => runSample(i)); samples.appendChild(b);
    });
    const file = document.getElementById('file'), zone = document.getElementById('dropzone');
    document.getElementById('choose').addEventListener('click', e => { e.preventDefault(); file.click(); });
    zone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); file.click(); } });
    file.addEventListener('change', () => { if (file.files[0]) handle(file.files[0], file.files[0].name); });
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
    zone.addEventListener('drop', e => {
      e.preventDefault(); zone.classList.remove('drag');
      const f = e.dataTransfer.files[0]; if (f) handle(f, f.name);
    });
    const slider = document.getElementById('threshold'), out = document.getElementById('threshold-out');
    slider.addEventListener('input', () => { out.textContent = slider.value; show(); });   // re-checks the alert, no new run
    const params = new URLSearchParams(location.search), sample = params.get('sample');
    if (params.get('tab') === 'estimator' && sample !== null && /^\d+$/.test(sample) && Number(sample) < SAMPLES.length) {
      runSample(Number(sample));
    }
  }
  init();
  return {predict, runSample, letterboxGeometry, resizeBilinear8};
})();
