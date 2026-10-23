/*
 * Движок Monte Carlo для браузера (Web Worker). Повторяет brm/engine.py:
 * ставка считается от базы на текущем шаге и сразу разыгрывается, ставка не больше
 * живого банка, рынок и исходы идут из отдельного генератора (common random numbers),
 * просадка — максимум (пик - банк) / пик после каждой ставки.
 */
'use strict';

let DATA = null;
const SYN = new Map();

// ---------- генераторы случайных чисел ----------
function makeRng(seed, stream) {
  let s = ((seed >>> 0) ^ Math.imul(stream + 1, 0x9e3779b9)) >>> 0;
  const sm = () => {
    s = (s + 0x9e3779b9) | 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return (z ^ (z >>> 16)) >>> 0;
  };
  let a = sm(), b = sm(), c = sm(), d = sm();
  const r = () => {
    a |= 0; b |= 0; c |= 0; d |= 0;
    const t = (((a + b) | 0) + d) | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
  for (let i = 0; i < 15; i++) r();
  return r;
}

function gauss(r) {
  let u = 0;
  while (u === 0) u = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

function gamma(r, k) {
  if (k < 1) return gamma(r, k + 1) * Math.pow(r(), 1 / k);
  const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = gauss(r); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = r();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

// Синтетика: кэф из Beta на [lo, hi] с заданным средним, ROI из смеси ТЗ 3–6 / 6–12 / 12–20% (85/10/5).
// По умолчанию — генератор из ТЗ: [1.5, 3.5], среднее 2.8.
function syntheticPool(lo = 1.5, hi = 3.5, mean = 2.8, conc = 4, size = 20000, seed = 7) {
  const r = makeRng(seed, 9);
  if (!(lo > 1)) throw new Error('Минимальный кэф должен быть больше 1');
  if (!(hi > lo) || !(mean > lo && mean < hi)) throw new Error('Средний кэф должен лежать строго между минимальным и максимальным');
  const m = (mean - lo) / (hi - lo);
  const odds = new Float64Array(size), ev = new Float64Array(size);
  const ranges = [[0.03, 0.06], [0.06, 0.12], [0.12, 0.20]];
  for (let i = 0; i < size; i++) {
    const x = gamma(r, m * conc), y = gamma(r, (1 - m) * conc);
    odds[i] = lo + (hi - lo) * x / (x + y);
    const u = r();
    const [a, b] = ranges[u < 0.85 ? 0 : u < 0.95 ? 1 : 2];
    ev[i] = a + (b - a) * r();
  }
  return { odds, ev };
}

// ---------- рынок: что видит стратегия и что правда ----------
function buildMarket(c) {
  let pool = DATA;
  if (c.source === 'synthetic') {
    const sp = c.synth || { lo: 1.5, hi: 3.5, mean: 2.8, conc: 4 };
    const key = [sp.lo, sp.hi, sp.mean, sp.conc].join('|');
    if (!SYN.has(key)) SYN.set(key, syntheticPool(sp.lo, sp.hi, sp.mean, sp.conc));
    pool = SYN.get(key);
  }
  const n = pool.odds.length;
  const est = new Float64Array(n), tru = new Float64Array(n);
  let k = 1, meanEst = c.roi;
  if (c.evModel === 'flat') {
    est.fill(c.roi);
    tru.fill(c.roi);
  } else {
    let sum = 0;
    for (let i = 0; i < n; i++) { est[i] = Math.min(pool.ev[i], c.estCap); sum += est[i]; }
    const mean = sum / n;
    meanEst = mean;
    if (!(mean > 0)) throw new Error('Средняя оценка EV в данных не положительна');
    k = c.roi / mean;
    for (let i = 0; i < n; i++) tru[i] = est[i] * k;
    if (c.calibrated) est.set(tru);
  }
  let so = 0;
  for (let i = 0; i < n; i++) so += pool.odds[i];
  return { odds: pool.odds, est, tru, n, k, meanEst, noise: c.noise, avgOdds: so / n };
}

// ---------- стратегии: доля базы на ставку ----------
const kellyF = (o, e) => (e > 0 ? e / (o - 1) : 0);
const clip01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

function makeStrategy(key, p, nSims) {
  switch (key) {
    case 'fixed': { const f = p.pct / 100; return { frac: () => f }; }
    case 'flat': { const a = p.pct / 100; return { frac: (o, e, b, pk, base, b0) => a * b0 / Math.max(base, 1e-12) }; }
    case 'kelly_half': return { frac: (o, e) => p.mult * kellyF(o, e) };
    case 'kelly_guard':
      return {
        frac: (o, e, b, pk) => {
          const dd = (pk - b) / pk;
          const c1 = Math.min(p.cut1, p.cut2) / 100, c2 = Math.max(p.cut1, p.cut2) / 100;
          return p.mult * kellyF(o, e) * (dd > c2 ? 0.25 : dd > c1 ? 0.5 : 1);
        },
      };
    case 'floor_kelly': {
      // подушка от базы ставки и пика баз (как в brm.strategies.FloorKelly): при снимке банка раз в K
      // ставок стратегия знает банк только на момент снимка
      const fl = p.floor / 100, pkb = new Float64Array(nSims);
      return {
        frac: (o, e, b, pk, base, b0, s) => {
          if (base > pkb[s]) pkb[s] = base;
          return Math.min(p.mult * kellyF(o, e), 1) * Math.max(base - fl * pkb[s], 0) / Math.max(base, 1e-12);
        },
      };
    }
    case 'log_formula': {
      const cap = p.edgeCap / 100;
      return { frac: (o, e) => { const ee = Math.min(e, cap); const q = 1 - (1 + ee) / o; return ee > 0 && q > 0 ? -Math.log10(q) / p.risk : 0; } };
    }
    case 'linear_roi':
      return { frac: (o, e) => { const roi = e * 100; return roi > 0 ? Math.min(p.basePct * roi / p.baseRoi, p.maxPct) / 100 : 0; } };
    case 'sqrt_roi':
      return { frac: (o, e) => { const roi = e * 100; return roi > 0 ? Math.min(p.basePct * Math.sqrt(roi / p.baseRoi), p.maxPct) / 100 : 0; } };
    case 'constant_profit':
      return { frac: (o) => Math.min(p.target / 100 / Math.max(o - 1, 1e-9), p.maxPct / 100) };
    case 'adaptive_constant_profit':
      return {
        frac: (o, e) => {
          const k = clip01((e * 100 - 3) / (20 - 3));
          const t = p.minT + (p.maxT - p.minT) * k;
          const f = o > 1.05 ? t / 100 / (o - 1) : 0.01;
          return Math.min(f, 0.15);
        },
      };
    case 'linear_roi_odds':
      return {
        frac: (o, e) => {
          const pct = p.basePct * (e * 100 / 5) * (1 - p.penalty * clip01((o - 1.5) / 3.5));
          return Math.min(Math.max(pct, 0), 10) / 100;
        },
      };
    case 'dynamic_percentage':
      return {
        frac: (o, e, b, pk, base, b0) => {
          const r = b / b0;
          let m = r > 1.2 ? 1.2 : 1;
          if (r < 0.8) m = 0.7;
          if (r < 0.6) m = 0.5;
          return p.pct * m / 100;
        },
      };
    case 'drawdown_guard':
      return { frac: (o, e, b, pk) => p.pct / 100 * ((pk - b) / pk > p.cut / 100 ? p.factor : 1) };
    case 'anti_martingale': {
      const pct = new Float64Array(nSims).fill(p.basePct), streak = new Int32Array(nSims);
      return {
        frac: (o, e, b, pk, base, b0, s) => pct[s] / 100,
        update: (s, won) => {
          if (won) {
            streak[s]++;
            if (streak[s] < p.maxStreak) pct[s] = Math.min(pct[s] * p.mult, 10);
          } else { streak[s] = 0; pct[s] = p.basePct; }
        },
      };
    }
    case 'portfolio_theory':
      return { frac: (o) => p.basePct * (o < 3.5 ? 0.4 : 0.2) / 0.3 / 100 };
    default:
      throw new Error('Неизвестная стратегия: ' + key);
  }
}

// ---------- статистика ----------
function quantileSorted(a, q) {
  const pos = (a.length - 1) * q, lo = Math.floor(pos), hi = Math.min(lo + 1, a.length - 1);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}
const sortedCopy = (src) => Float64Array.from(src).sort();

function summarize(r, cfg) {
  const N = r.final.length, b0 = cfg.bank0;
  const profit = new Float64Array(N), dd = new Float64Array(N), lg = new Float64Array(N);
  let sumP = 0, loss = 0, d20 = 0, d50 = 0, d80 = 0, half = 0, ruin = 0, sumProfit = 0, sumTurn = 0, stakeAvg = 0;
  for (let s = 0; s < N; s++) {
    profit[s] = (r.final[s] / b0 - 1) * 100;
    dd[s] = r.maxDD[s] * 100;
    lg[s] = Math.log(Math.max(r.final[s], 1e-300) / b0) / cfg.nBets;
    sumP += profit[s];
    if (profit[s] < 0) loss++;
    if (dd[s] >= 20) d20++;
    if (dd[s] >= 50) d50++;
    if (dd[s] >= 80) d80++;
    if (r.minBank[s] < 0.5 * b0) half++;
    if (r.ruined[s]) ruin++;
    sumProfit += r.final[s] - b0;
    sumTurn += r.turnover[s];
    stakeAvg += r.stakeSum[s] / Math.max(r.nPlaced[s], 1);
  }
  const sp = sortedCopy(profit), sd = sortedCopy(dd), sl = sortedCopy(lg), ls = sortedCopy(r.maxLoss);
  return {
    median: quantileSorted(sp, 0.5), mean: sumP / N,
    p5: quantileSorted(sp, 0.05), p25: quantileSorted(sp, 0.25), p75: quantileSorted(sp, 0.75), p95: quantileSorted(sp, 0.95),
    pLoss: loss / N * 100, medianDD: quantileSorted(sd, 0.5), p95DD: quantileSorted(sd, 0.95),
    pDD20: d20 / N * 100, pDD50: d50 / N * 100, pDD80: d80 / N * 100,
    pHalf: half / N * 100, pRuin: ruin / N * 100,
    roiTurnover: sumTurn > 0 ? sumProfit / sumTurn * 100 : 0,
    avgStake: stakeAvg / N * 100,
    growth: quantileSorted(sl, 0.5) * 100,
    medianLossStreak: quantileSorted(ls, 0.5), maxLossStreak: ls[ls.length - 1],
  };
}

// ---------- сам прогон ----------
const LEGACY_VAR = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15];

function simulate(strat, mk, cfg, opts = {}) {
  const N = cfg.nSims, T = cfg.nBets, n = mk.n, b0 = cfg.bank0, cap = opts.cap ?? cfg.cap;
  const scale = opts.scale ?? 1, full = !!opts.full;
  const rm = makeRng(cfg.seed, 1), rs = makeRng(cfg.seed, 2);
  const odds = mk.odds, est = mk.est, tru = mk.tru, noise = mk.noise;
  const bank = new Float64Array(N).fill(b0), peak = new Float64Array(N).fill(b0);
  const minBank = new Float64Array(N).fill(b0), maxDD = new Float64Array(N);
  const turnover = new Float64Array(N), stakeSum = new Float64Array(N);
  const nPlaced = new Int32Array(N), lossRun = new Int32Array(N), maxLoss = new Int32Array(N);
  const ruined = new Uint8Array(N);
  const ruinAbs = 0.01 * b0;

  let snap = null, since = null, period = null;
  if (cfg.recalc !== 'live') {
    snap = new Float64Array(N).fill(b0); since = new Int32Array(N); period = new Int32Array(N);
    for (let s = 0; s < N; s++) period[s] = cfg.recalc === 'r30' ? 30 : 30 + Math.floor(rs() * 41);
  }
  const redraw = cfg.recalc === 'r3070';   // случайный период тянется заново после каждого снимка
  const vr = cfg.variation;

  const stride = Math.max(1, Math.round(T / 120));
  const steps = [];
  const q = [[], [], [], [], []], mean = [];
  const nPaths = full ? Math.min(12, N) : 0;
  const paths = Array.from({ length: nPaths }, () => []);
  const scratch = full ? new Float64Array(N) : null;
  const QS = [0.05, 0.25, 0.5, 0.75, 0.95];
  const record = (t) => {
    steps.push(t);
    scratch.set(bank);
    scratch.sort();
    for (let i = 0; i < 5; i++) q[i].push(quantileSorted(scratch, QS[i]) / b0);
    let m = 0;
    for (let s = 0; s < N; s++) m += bank[s];
    mean.push(m / N / b0);
    for (let i = 0; i < nPaths; i++) paths[i].push(bank[i] / b0);
  };
  if (full) record(0);

  const frac = strat.frac, upd = strat.update;
  for (let t = 0; t < T; t++) {
    for (let s = 0; s < N; s++) {
      const idx = (rm() * n) | 0;
      const o = odds[idx], e = est[idx];
      let et = tru[idx];
      if (noise > 0) et += noise * gauss(rm);
      const won = rm() < (1 + et) / o;

      const b = bank[s];
      let base = b;
      if (snap !== null) {
        if (since[s] >= period[s]) { snap[s] = b; since[s] = 0; if (redraw) period[s] = 30 + Math.floor(rs() * 41); }
        base = snap[s];
        since[s]++;
      }
      let f = frac(o, e, b, peak[s], base, b0, s) * scale;
      if (!(f > 0)) f = 0; else if (f > cap) f = cap;
      let stake = f * base;
      if (vr === 'uniform') stake *= 0.8 + 0.4 * rs();
      else if (vr === 'legacy') stake *= LEGACY_VAR[(rs() * 17) | 0];
      if (cfg.roundTo > 0) stake = Math.round(stake / cfg.roundTo) * cfg.roundTo;
      if (stake > b) stake = b;                 // сначала не больше живого банка,
      if (stake < cfg.minStake) stake = 0;      // потом минимум: банк меньше минимальной ставки — ставки нет
      if (ruined[s]) stake = 0;

      let nb = won ? b + stake * (o - 1) : b - stake;
      if (nb < 0) nb = 0;
      bank[s] = nb;
      if (stake > 0) {
        turnover[s] += stake;
        nPlaced[s]++;
        stakeSum[s] += stake / b;
        if (won) lossRun[s] = 0; else { lossRun[s]++; if (lossRun[s] > maxLoss[s]) maxLoss[s] = lossRun[s]; }
      }
      if (nb > peak[s]) peak[s] = nb;
      const d = (peak[s] - nb) / peak[s];
      if (d > maxDD[s]) maxDD[s] = d;
      if (nb < minBank[s]) minBank[s] = nb;
      if (nb < ruinAbs) ruined[s] = 1;
      if (upd) upd(s, won);
    }
    if (full && ((t + 1) % stride === 0 || t + 1 === T)) record(t + 1);
  }
  const res = { final: bank, maxDD, minBank, turnover, stakeSum, nPlaced, maxLoss, ruined };
  const out = { summary: summarize(res, cfg) };
  if (full) {
    out.steps = steps; out.q = q; out.mean = mean; out.paths = paths;
    out.final = Float32Array.from(bank, (x) => x / b0);
    out.maxDD = Float32Array.from(maxDD);
  }
  return out;
}

// ---------- калибровка под риск-профиль ----------
const PROFILES = {
  conservative: ['pDD50', 0], cautious: ['pDD50', 3.5], balanced: ['pDD50', 10], risky: ['pDD80', 5], extreme: ['pDD80', 50],
};

// Келли с полом: две ручки — пол и доля Келли от подушки; как в brm.optimize.calibrate_floor_kelly
function calibrateFloorKelly(mk, cfg, profile, iters) {
  const run = (fl) => {
    const r = calibrate('floor_kelly', { floor: Math.round(fl * 1000) / 10 }, mk, cfg, profile, iters);
    return r ? { ...r, floor: Math.round(fl * 1000) / 1000 } : null;
  };
  let best = null;
  for (const fl of [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) { const r = run(fl); if (r && (!best || r.median > best.median)) best = r; }
  if (!best) return null;
  for (const fl of [best.floor - 0.05, best.floor + 0.05]) {
    if (fl >= 0.05 && fl <= 0.95) { const r = run(fl); if (r && r.median > best.median) best = r; }
  }
  return best;
}

function calibrate(key, params, mk, cfg, profile, iters = 12) {
  const [metric, limit] = PROFILES[profile];
  const floorKelly = key === 'floor_kelly';
  const scaleHi = floorKelly ? 2 : 64;
  const ev = (s) => {
    const strat = floorKelly ? makeStrategy(key, { mult: s, floor: params.floor }, cfg.nSims) : makeStrategy(key, params, cfg.nSims);
    return simulate(strat, mk, cfg, { scale: floorKelly ? 1 : s }).summary;
  };
  const feasible = new Map();
  let lo = 0, hi = Math.min(1, scaleHi);
  let m = ev(hi);
  if (m[metric] <= limit) {
    feasible.set(hi, m);
    let broke = false;
    while (hi < scaleHi) {
      lo = hi; hi = Math.min(hi * 2, scaleHi);
      m = ev(hi);
      if (m[metric] > limit) { broke = true; break; }
      feasible.set(hi, m);
    }
    if (!broke) lo = hi;
  }
  if (lo !== hi) {
    for (let i = 0; i < iters; i++) {
      const mid = (lo + hi) / 2;
      m = ev(mid);
      if (m[metric] <= limit) { lo = mid; feasible.set(mid, m); } else hi = mid;
    }
  }
  if (!feasible.size) return null;
  const sMax = Math.max(...feasible.keys());
  for (const k of [0.4, 0.55, 0.7, 0.85]) {
    const s = sMax * k;
    if (!feasible.has(s)) { m = ev(s); if (m[metric] <= limit) feasible.set(s, m); }
  }
  let best = null;
  for (const [s, mm] of feasible) if (!best || mm.median > best[1].median) best = [s, mm];
  return { scale: best[0], scaleMax: sMax, ...best[1] };
}

// ---------- обработчик сообщений ----------
self.onmessage = (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'init') {
      DATA = { odds: Float64Array.from(m.odds), ev: Float64Array.from(m.ev) };
      return;
    }
    const t0 = performance.now();
    const mk = buildMarket(m.market);
    if (m.type === 'simulate') {
      const out = simulate(makeStrategy(m.key, m.params, m.cfg.nSims), mk, m.cfg, { full: true });
      out.ms = performance.now() - t0;
      out.marketK = mk.k;
      out.avgOdds = mk.avgOdds;
      self.postMessage({ type: 'simulate', id: m.id, out }, [out.final.buffer, out.maxDD.buffer]);
    } else if (m.type === 'compare') {
      const rows = [];
      m.items.forEach((it, i) => {
        const r = it.key === 'floor_kelly' ? calibrateFloorKelly(mk, m.cfg, m.profile, 12) : calibrate(it.key, it.params, mk, m.cfg, m.profile);
        if (r) rows.push({ key: it.key, ...r });
        self.postMessage({ type: 'progress', id: m.id, done: i + 1, total: m.items.length, label: it.key });
      });
      rows.sort((a, b) => b.median - a.median);
      self.postMessage({ type: 'compare', id: m.id, rows, ms: performance.now() - t0 });
    } else if (m.type === 'kelly') {
      const cases = {
        calibrated: { ...m.market, evModel: 'scaled', calibrated: true, noise: 0 },
        overstated: { ...m.market, evModel: 'scaled', calibrated: false, noise: 0 },
        overstated_noise: { ...m.market, evModel: 'scaled', calibrated: false, noise: 0.05 },
      };
      const out = {};
      const total = Object.keys(cases).length * m.mults.length;
      let done = 0;
      for (const [name, mc] of Object.entries(cases)) {
        const mkc = buildMarket(mc);
        out[name] = m.mults.map((k) => {
          const s = simulate(makeStrategy('kelly_half', { mult: k }, m.cfg.nSims), mkc, m.cfg, { cap: 1 }).summary;
          done++;
          self.postMessage({ type: 'progress', id: m.id, done, total, label: name });
          return { mult: k, ...s };
        });
      }
      // k — во сколько раз истинный EV меньше оценки модели (k < 1 — модель завышает)
      const over = buildMarket(cases.overstated);
      self.postMessage({ type: 'kelly', id: m.id, out, k: over.k, meanEst: over.meanEst, ms: performance.now() - t0 });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String((err && err.message) || err) });
  }
};
