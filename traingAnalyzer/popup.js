'use strict';

// ═══════════════════════════════════════════════════════════════════
// STARK-OS IMPORTS — Hexagonal Architecture Modules
// Los módulos de la nueva arquitectura se importan aquí.
// popup.js actúa como orquestador fino (Thin Shell).
// ═══════════════════════════════════════════════════════════════════
import { initNeuralLab } from './neural-lab.js';
import { initWhaleTracker } from './whale-tracker.js';

// ═══════════════════════════════════════════════════════════════════
// ML MODEL — Weighted Scoring Engine (Logistic Regression)
// Entrenado con Python en tv-analyzer/train/
// El JSON exportado se carga automáticamente si existe en la extensión
// o manualmente desde el botón de importar en el Brain.
// ═══════════════════════════════════════════════════════════════════

let _mlModel = null; // { coefficients, intercept, scaler_mean, scaler_std, feature_names, cv_auc, importance }

async function mlLoadModel() {
  // 1. Intentar cargar model_weights.json desde la carpeta de la extensión
  try {
    const url  = chrome.runtime.getURL('model_weights.json');
    const resp = await fetch(url);
    if (resp.ok) {
      _mlModel = await resp.json();
      console.log('[ML] Modelo cargado desde extensión — AUC:', _mlModel.cv_auc);
      return true;
    }
  } catch {}
  // 2. Intentar desde chrome.storage (importado manualmente)
  try {
    const data = await new Promise(res => chrome.storage.local.get('mlModel', res));
    if (data.mlModel?.coefficients) {
      _mlModel = data.mlModel;
      console.log('[ML] Modelo cargado desde storage — AUC:', _mlModel.cv_auc);
      return true;
    }
  } catch {}
  return false;
}

function mlSigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, x)))); }

function mlExtractFeatures(candles, indicators) {
  const n = candles.length;
  if (n < 200) return null;
  const closes  = candles.map(c => c.close);
  const highs   = candles.map(c => c.high);
  const lows    = candles.map(c => c.low);
  const opens   = candles.map(c => c.open);
  const volumes = candles.map(c => c.volume || 0);

  // Recalcular series completas (misma lógica que build_features.py)
  const rsiArr  = calcRSI(closes, 14);
  const { hist: histArr } = calcMACD(closes);
  const e20Arr  = calcEMA(closes, 20);
  const e50Arr  = calcEMA(closes, 50);
  const e200Arr = calcEMA(closes, 200);
  const bbArr   = calcBB(closes, 20, 2);
  const atrArr  = calcATR(candles, 14);
  const stochArr = calcStoch(candles, 14);

  const rsi   = rsiArr[n-1]  ?? 50;
  const rsiP3 = rsiArr[n-4]  ?? rsi;
  const hist  = histArr[n-1] ?? 0;
  const histP = histArr[n-2] ?? 0;
  const e20   = e20Arr[n-1];
  const e50   = e50Arr[n-1];
  const e200  = e200Arr[n-1];
  const e20P5 = e20Arr[n-6] ?? e20;
  const e50P5 = e50Arr[n-6] ?? e50;
  const bb    = bbArr[n-1];
  const atr   = atrArr[n-1] ?? (closes[n-1] * 0.005);
  const stk   = stochArr.k[n-1] ?? 50;
  const price = closes[n-1];
  const open  = opens[n-1];
  const high  = highs[n-1];
  const low   = lows[n-1];

  const avgVol20 = volumes.slice(-20).reduce((a,b)=>a+b,0) / 20 || 1;
  const volR     = volumes[n-1] / avgVol20;
  const body     = Math.abs(price - open);
  const rng      = (high - low) || 1e-10;
  const upperW   = high - Math.max(price, open);
  const lowerW   = Math.min(price, open) - low;

  const hour = candles[n-1].time
    ? new Date(candles[n-1].time).getUTCHours()
    : 12;

  const bbP = bb?.lower != null ? (price - bb.lower) / (bb.upper - bb.lower + 1e-10) : 0.5;
  const bbW = bb?.width ?? 0.05;

  // DEBE coincidir con FEATURE_NAMES en train_model.py (mismo orden)
  return [
    (rsi - 50) / 50,                                   // rsi_norm
    (rsi - rsiP3) / 3 / 50,                            // rsi_slope
    rsi > 70 ? 1 : 0,                                  // rsi_ob
    rsi < 30 ? 1 : 0,                                  // rsi_os
    rsi > 50 ? 1 : 0,                                  // rsi_bull
    hist / (atr || 1),                                 // macd_hist_norm
    hist > 0 ? 1 : 0,                                  // macd_bull
    (hist > 0 && histP <= 0) ? 1 : 0,                  // macd_cross_up
    (hist < 0 && histP >= 0) ? 1 : 0,                  // macd_cross_dn
    e20 ? price / e20 - 1 : 0,                         // price_ema20
    e50 ? price / e50 - 1 : 0,                         // price_ema50
    e200 ? price / e200 - 1 : 0,                       // price_ema200
    e20P5 ? e20 / e20P5 - 1 : 0,                       // ema20_slope
    e50P5 ? e50 / e50P5 - 1 : 0,                       // ema50_slope
    (e20 > e50 && e50 > e200) ? 1 : 0,                 // ema_bull
    (e20 < e50 && e50 < e200) ? 1 : 0,                 // ema_bear
    price > e200 ? 1 : 0,                              // above_ema200
    Math.min(1, Math.max(0, bbP)),                     // bb_pos
    bbW,                                               // bb_width
    bbW < 0.025 ? 1 : 0,                               // bb_squeeze
    atr / price,                                       // atr_norm
    stk / 100,                                         // stoch_norm
    stk < 20 ? 1 : 0,                                  // stoch_os
    stk > 80 ? 1 : 0,                                  // stoch_ob
    Math.min(3, Math.log1p(volR)),                     // vol_ratio
    volR > 2 ? 1 : 0,                                  // vol_spike
    price > open ? 1 : 0,                              // bull_candle
    Math.min(1, body / rng),                           // body_ratio
    Math.min(1, upperW / rng),                         // upper_wick
    Math.min(1, lowerW / rng),                         // lower_wick
    Math.sin(2 * Math.PI * hour / 24),                 // hour_sin
    Math.cos(2 * Math.PI * hour / 24),                 // hour_cos
  ];
}

function mlInfer(candles) {
  if (!_mlModel) return null;
  const feat = mlExtractFeatures(candles);
  if (!feat) return null;
  // Normalizar con los parámetros del scaler entrenado en Python
  const norm = feat.map((v, i) =>
    (_mlModel.scaler_std[i] > 0)
      ? (v - _mlModel.scaler_mean[i]) / _mlModel.scaler_std[i]
      : 0
  );
  // Producto punto + bias → sigmoide
  const z = norm.reduce((s, v, i) => s + v * _mlModel.coefficients[i], 0) + _mlModel.intercept;
  const prob = mlSigmoid(z);
  return {
    bullPct:    Math.round(prob * 100),
    bearPct:    Math.round((1 - prob) * 100),
    raw:        prob,
    auc:        _mlModel.cv_auc,
    samples:    _mlModel.n_samples,
    importance: _mlModel.importance,
    trainedAt:  _mlModel.trained_at,
  };
}

// ═══════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════

let currentSymbol = 'BTCUSDT';
let currentTf     = '15m';
let currentPrice  = 0;
let candles       = [];       // OHLCV array
let lastAnalysis  = null;
let autoRefresh   = null;

// Paper trading
let ptBalance   = 10000;   // USD
let ptStart     = 10000;
let ptPositions = [];
let ptPeakEq    = 10000;
let ptSolPrice  = 1;       // dummy, we use USD directly

// Watchlist
const WATCHLIST_DEFAULT = ['BTCUSDT','ETHUSDT','XRPUSDT','SOLUSDT','BNBUSDT','DOGEUSDT'];
let watchlist = [...WATCHLIST_DEFAULT];
let watchData = {};

// WebSocket live feed
let liveWS  = null;
let wsTimer = null;

// TradingView tab tracking
let tvTabId = null;

// ═══════════════════════════════════════════════════════════════════
// WEBSOCKET LIVE FEED  (Binance miniTicker — ~500ms push)
// ═══════════════════════════════════════════════════════════════════

const isCrypto = s => /USDT$|USDC$|BTC$|ETH$|BNB$|BUSD$|SOL$/i.test(s);

function wsApply() {
  const syms = [...new Set([
    currentSymbol,
    ...ptPositions.filter(p => p.status === 'open').map(p => p.symbol),
  ])].filter(s => s && isCrypto(s)); // WebSocket only for Binance crypto pairs

  // Kill old connection
  if (liveWS) {
    liveWS.onclose = null;
    liveWS.onerror = null;
    if (liveWS.readyState !== WebSocket.CLOSED) liveWS.close();
    liveWS = null;
  }
  if (wsTimer) { clearTimeout(wsTimer); wsTimer = null; }
  if (!syms.length) return;

  const streams = syms.map(s => s.toLowerCase() + '@miniTicker').join('/');
  liveWS = new WebSocket(`wss://stream.binance.com:9443/stream?streams=${streams}`);

  liveWS.onmessage = (e) => {
    try {
      const msg   = JSON.parse(e.data);
      const data  = msg.data;
      if (!data?.c) return;
      const sym   = data.s;
      const price = parseFloat(data.c);
      if (!sym || !(price > 0)) return;

      // Price display update
      if (sym === currentSymbol) {
        currentPrice = price;
        const pe = document.getElementById('aPrice');
        if (pe && document.getElementById('aContent')?.style.display !== 'none')
          pe.textContent = '$' + fmtPrice(price);
        const dp = document.getElementById('dTpPrice');
        if (dp) dp.textContent = '$' + fmtPrice(price);
        const ts = document.getElementById('aLastUpdate');
        if (ts) ts.textContent = new Date().toLocaleTimeString('es',
          { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      }

      // Open positions — update price & check TP/SL/liq
      for (const pos of ptPositions.filter(p => p.status === 'open' && p.symbol === sym)) {
        pos.currentPrice = price;
        const isLong = pos.direction !== 'short';
        if (pos.liqPrice && (isLong ? price <= pos.liqPrice : price >= pos.liqPrice))
          { ptSellDemo(pos.id, 100, 'liquidated'); continue; }
        if (pos.tpPrice  && (isLong ? price >= pos.tpPrice  : price <= pos.tpPrice))
          { ptSellDemo(pos.id, 100, 'tp');         continue; }
        if (pos.slPrice  && (isLong ? price <= pos.slPrice  : price >= pos.slPrice))
          { ptSellDemo(pos.id, 100, 'sl'); }
      }
    } catch (_) {}
  };

  liveWS.onclose = () => { wsTimer = setTimeout(wsApply, 3000); };
  liveWS.onerror = () => {};  // onclose fires after onerror
}

// ═══════════════════════════════════════════════════════════════════
// BINANCE API
// ═══════════════════════════════════════════════════════════════════

const BINANCE = 'https://api.binance.com/api/v3';
const TF_MAP  = { '1m':'1m','3m':'3m','5m':'5m','15m':'15m','30m':'30m',
                  '1h':'1h','2h':'2h','4h':'4h','1d':'1d','1w':'1w' };

async function fetchKlines(symbol, interval = '15m', limit = 200) {
  const iv = TF_MAP[interval] || '15m';
  const r  = await fetch(`${BINANCE}/klines?symbol=${symbol}&interval=${iv}&limit=${limit}`);
  if (!r.ok) throw new Error(`Binance ${r.status}`);
  const raw = await r.json();
  return raw.map(k => ({
    time:   k[0],
    open:   parseFloat(k[1]),
    high:   parseFloat(k[2]),
    low:    parseFloat(k[3]),
    close:  parseFloat(k[4]),
    volume: parseFloat(k[5]),
  }));
}

async function fetchTicker(symbol) {
  const r = await fetch(`${BINANCE}/ticker/24hr?symbol=${symbol}`);
  if (!r.ok) return null;
  return r.json();
}

async function fetchMultiTicker(symbols) {
  const results = await Promise.allSettled(symbols.map(s => fetchTicker(s)));
  const out = {};
  symbols.forEach((s, i) => {
    if (results[i].status === 'fulfilled' && results[i].value)
      out[s] = results[i].value;
  });
  return out;
}

// ── Yahoo Finance (stocks, forex, indices) ────────────────────────────────────
const YAHOO       = 'https://query1.finance.yahoo.com/v8/finance/chart';
const CRYPTOCOMP  = 'https://min-api.cryptocompare.com/data/v2';
const FF_CAL      = 'https://nfs.faireconomy.media/ff_calendar_thisweek.json';

// ── Session ranges ────────────────────────────────────────────────────────────

// All times UTC (summer, EDT = UTC-4 / CEST = UTC+2)
//   Asia:          00:00–09:00 UTC  = 02:00–11:00 España
//   Londres:       07:00–16:00 UTC  = 09:00–18:00 España
//   NY Pre-market: 13:00–13:30 UTC  = 15:00–15:30 España  (30 min antes apertura)
//   NY Regular:    13:30–20:00 UTC  = 15:30–22:00 España
//   NY Post-market:20:00–24:00 UTC  = 22:00–02:00 España
const SESSIONS = [
  { key:'asia',    name:'Asia (Tokio)',   code:'JP', start:0,    end:9,    color:'#e3b341' },
  { key:'london',  name:'Londres',        code:'EU', start:7,    end:16,   color:'#00d4ff' },
  { key:'ny_pre',  name:'NY Pre-mercado', code:'US', start:13,   end:13.5, color:'#a78bfa', sub:true },
  { key:'ny',      name:'Nueva York',     code:'US', start:13.5, end:20,   color:'#00ff41' },
  { key:'ny_post', name:'NY Post-mercado',code:'US', start:20,   end:24,   color:'#6b7280', sub:true },
];

function fmtHour(h) {
  const hh = Math.floor(h), mm = Math.round((h - hh) * 60);
  return `${String(hh).padStart(2,'0')}:${String(mm).padStart(2,'0')}`;
}

// Convert UTC decimal hour to local time string using browser offset
function fmtHourLocal(h) {
  const offsetH = -new Date().getTimezoneOffset() / 60; // e.g. +2 for CEST
  const local = (h + offsetH + 24) % 24;
  return fmtHour(local);
}

async function fetchSessionRanges(symbol) {
  try {
    // Always fetch 1h candles for today's session ranges regardless of selected TF
    const h1 = await fetchKlinesUniversal(symbol, '1h', 30);
    return computeSessionRanges(h1);
  } catch (_) { return computeSessionRanges([]); }
}

function computeSessionRanges(candles1h) {
  const now      = Date.now();
  const dayStart = new Date(); dayStart.setUTCHours(0,0,0,0);
  const dayMs    = dayStart.getTime();

  return SESSIONS.map(s => {
    const startMs = dayMs + s.start * 3600000;
    const endMs   = dayMs + s.end   * 3600000;
    const sc      = candles1h.filter(c => c.time >= startMs && c.time < Math.min(endMs, now));

    let status, remaining = '';
    if (now < startMs) {
      status = 'pending';
      const mins = Math.round((startMs - now) / 60000);
      remaining = mins < 60 ? `Abre en ${mins}m` : `Abre en ${Math.floor(mins/60)}h${mins%60?` ${mins%60}m`:''}`;
    } else if (now >= endMs) {
      status = 'closed';
    } else {
      status = 'open';
      const mins = Math.round((endMs - now) / 60000);
      remaining = mins < 60 ? `Cierra en ${mins}m` : `Cierra en ${Math.floor(mins/60)}h${mins%60?` ${mins%60}m`:''}`;
    }

    if (!sc.length) return { ...s, status, remaining, high:null, low:null, rangePct:null };
    const high     = Math.max(...sc.map(c => c.high));
    const low      = Math.min(...sc.map(c => c.low));
    const rangePct = ((high - low) / low * 100).toFixed(2);
    return { ...s, status, remaining, high, low, rangePct };
  });
}

// ── Economic calendar — full week, high impact only (Forex Factory) ──────────

const DAY_NAMES = ['Dom','Lun','Mar','Mié','Jue','Vie','Sáb'];
const COUNTRY_FLAG = {
  USD:'🇺🇸', EUR:'🇪🇺', GBP:'🇬🇧', JPY:'🇯🇵', AUD:'🇦🇺',
  CAD:'🇨🇦', CHF:'🇨🇭', CNY:'🇨🇳', NZD:'🇳🇿',
};

async function fetchEcoCalendar() {
  try {
    // Try primary source (Forex Factory via faireconomy.media)
    // Also try with cache-busting to avoid stale weekly data
    const r = await fetch(FF_CAL + '?_=' + new Date().toISOString().slice(0,10));
    if (!r.ok) throw new Error('cal fail');
    const all = await r.json();
    if (!Array.isArray(all) || all.length === 0) throw new Error('empty');
    const todayStr = new Date().toISOString().slice(0,10);

    const high = all.filter(e => e.country === 'USD' && (e.impact === 'High' || e.impact === 'Medium'));
    const byDay = {};
    // isDST: detect if browser is in summer time (affects ET→UTC offset)
    const browserDST = new Date().getTimezoneOffset() < new Date(new Date().getFullYear(), 0, 1).getTimezoneOffset();
    const etOffset   = browserDST ? 4 : 5; // EDT=+4h to UTC, EST=+5h

    for (const e of high) {
      let dateKey, timeStr;

      if (typeof e.date === 'string' && e.date.includes('T')) {
        // Full ISO: "2026-04-27T13:00:00-0400"
        const dt = new Date(e.date);
        if (isNaN(dt)) continue;
        dateKey = dt.toISOString().slice(0, 10);
        const h = dt.getUTCHours(), m = dt.getUTCMinutes();
        timeStr = (h === 0 && m === 0) ? null
                : fmtHour(h) + ':' + String(m).padStart(2,'0') + ' UTC';
      } else if (typeof e.date === 'string') {
        // Plain date "YYYY-MM-DD" + separate time field
        dateKey = e.date.slice(0, 10);
        if (!dateKey || !/^\d{4}-\d{2}-\d{2}$/.test(dateKey)) continue;

        const t = (e.time || '').trim();
        if (!t || t.toLowerCase().includes('day') || t.toLowerCase() === 'tentative') {
          timeStr = null;
        } else {
          // Formats: "2:00am", "10:30am", "14:00"
          const m12 = t.match(/^(\d{1,2}):(\d{2})\s*(am|pm)$/i);
          const m24 = t.match(/^(\d{1,2}):(\d{2})$/);
          if (m12) {
            let h = parseInt(m12[1]), min = parseInt(m12[2]);
            if (m12[3].toLowerCase() === 'pm' && h < 12) h += 12;
            if (m12[3].toLowerCase() === 'am' && h === 12) h = 0;
            const utcH = (h + etOffset) % 24;
            // local hour for display
            const localH = (utcH - new Date().getTimezoneOffset() / 60 + 24 + 24) % 24;
            timeStr = `${fmtHour(localH)} (${fmtHour(utcH)} UTC)`;
          } else if (m24) {
            let h = parseInt(m24[1]), min = parseInt(m24[2]);
            const utcH = (h + etOffset) % 24;
            const localH = (utcH - new Date().getTimezoneOffset() / 60 + 24 + 24) % 24;
            timeStr = `${fmtHour(localH)} (${fmtHour(utcH)} UTC)`;
          } else {
            timeStr = t;
          }
        }
      } else continue;

      if (!dateKey) continue;

      if (!byDay[dateKey]) byDay[dateKey] = [];
      byDay[dateKey].push({
        time:     timeStr || 'Horario pend.',
        country:  e.country || '—',
        title:    e.title   || '—',
        forecast: e.forecast || '',
        previous: e.previous || '',
        isToday:  dateKey === todayStr,
        isPast:   dateKey < todayStr,
        stars:    e.impact === 'High' ? 3 : 2,
      });
    }

    // Sort events within day by time
    for (const evs of Object.values(byDay)) {
      evs.sort((a, b) => a.time.localeCompare(b.time));
    }

    return Object.entries(byDay)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, events]) => {
        const parts = date.split('-').map(Number);
        const d = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2]));
        const lbl = date === todayStr
          ? 'HOY — ' + d.toLocaleDateString('es', { weekday:'long', day:'numeric', month:'long', timeZone:'UTC' })
          : DAY_NAMES[d.getUTCDay()] + ' ' + d.getUTCDate() + ' ' +
            d.toLocaleDateString('es', { month:'short', timeZone:'UTC' });
        return { date, label: lbl, isToday: date === todayStr, isPast: date < todayStr, events };
      });
  } catch (_) { return []; }
}

async function fetchKlinesYahoo(symbol, interval = '15m', limit = 200) {
  const ivMap = { '1m':'1m','5m':'5m','15m':'15m','30m':'30m','1h':'60m','2h':'60m','4h':'1h','1d':'1d','1w':'1wk' };
  const iv    = ivMap[interval] || '15m';
  const range = limit <= 80 ? '5d' : limit <= 200 ? '1mo' : '3mo';
  const r = await fetch(`${YAHOO}/${encodeURIComponent(symbol)}?interval=${iv}&range=${range}&includePrePost=false`);
  if (!r.ok) throw new Error(`Yahoo Finance ${r.status}`);
  const j      = await r.json();
  const result = j.chart?.result?.[0];
  if (!result) throw new Error('Sin datos Yahoo Finance');
  const times  = result.timestamp || [];
  const q      = result.indicators?.quote?.[0] || {};
  const candles = times.map((t, i) => ({
    time:   t * 1000,
    open:   q.open?.[i]   ?? 0,
    high:   q.high?.[i]   ?? 0,
    low:    q.low?.[i]    ?? 0,
    close:  q.close?.[i]  ?? 0,
    volume: q.volume?.[i] ?? 0,
  })).filter(c => c.close > 0);
  if (!candles.length) throw new Error(`Sin velas Yahoo Finance para ${symbol}`);
  return candles;
}

async function fetchKlinesUniversal(symbol, interval, limit) {
  try {
    return await fetchKlines(symbol, interval, limit);
  } catch (_) {
    return fetchKlinesYahoo(symbol, interval, limit);
  }
}

// ═══════════════════════════════════════════════════════════════════
// TECHNICAL INDICATORS
// ═══════════════════════════════════════════════════════════════════

function calcSMA(data, period) {
  return data.map((_, i) => {
    if (i < period - 1) return null;
    return data.slice(i - period + 1, i + 1).reduce((s, v) => s + v, 0) / period;
  });
}

function calcEMA(data, period) {
  const k = 2 / (period + 1);
  const out = [];
  let ema = null;
  for (const v of data) {
    if (ema === null) { ema = v; }
    else              { ema = v * k + ema * (1 - k); }
    out.push(ema);
  }
  return out;
}

function calcRSI(closes, period = 14) {
  const gains = [], losses = [];
  for (let i = 1; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    gains.push(d > 0 ? d : 0);
    losses.push(d < 0 ? -d : 0);
  }
  const rsi = [null];
  let avgG = gains.slice(0, period).reduce((s, v) => s + v, 0) / period;
  let avgL = losses.slice(0, period).reduce((s, v) => s + v, 0) / period;
  rsi.push(avgL === 0 ? 100 : 100 - 100 / (1 + avgG / avgL));
  for (let i = period; i < gains.length; i++) {
    avgG = (avgG * (period - 1) + gains[i]) / period;
    avgL = (avgL * (period - 1) + losses[i]) / period;
    rsi.push(avgL === 0 ? 100 : 100 - 100 / (1 + avgG / avgL));
  }
  // Pad to match closes length
  while (rsi.length < closes.length) rsi.unshift(null);
  return rsi;
}

function calcMACD(closes, fast = 12, slow = 26, sig = 9) {
  const emaFast = calcEMA(closes, fast);
  const emaSlow = calcEMA(closes, slow);
  const macd    = emaFast.map((v, i) => v - emaSlow[i]);
  const signal  = calcEMA(macd, sig);
  const hist    = macd.map((v, i) => v - signal[i]);
  return { macd, signal, hist };
}

function calcBB(closes, period = 20, mult = 2) {
  const sma = calcSMA(closes, period);
  return closes.map((_, i) => {
    if (i < period - 1) return { mid: null, upper: null, lower: null, width: null };
    const slice = closes.slice(i - period + 1, i + 1);
    const mean  = sma[i];
    const std   = Math.sqrt(slice.reduce((s, v) => s + (v - mean) ** 2, 0) / period);
    return { mid: mean, upper: mean + mult * std, lower: mean - mult * std, width: (2 * mult * std) / mean };
  });
}

function calcATR(candles, period = 14) {
  const tr = candles.map((c, i) => {
    if (i === 0) return c.high - c.low;
    const prev = candles[i - 1];
    return Math.max(c.high - c.low, Math.abs(c.high - prev.close), Math.abs(c.low - prev.close));
  });
  const atr = calcSMA(tr, period);
  return atr;
}

function calcStoch(candles, kPeriod = 14, dPeriod = 3) {
  const k = candles.map((_, i) => {
    if (i < kPeriod - 1) return null;
    const slice = candles.slice(i - kPeriod + 1, i + 1);
    const hh = Math.max(...slice.map(c => c.high));
    const ll = Math.min(...slice.map(c => c.low));
    return hh === ll ? 50 : ((candles[i].close - ll) / (hh - ll)) * 100;
  });
  const d = calcSMA(k.filter(v => v !== null), dPeriod);
  return { k, d };
}

// ═══════════════════════════════════════════════════════════════════
// SWING DETECTION
// ═══════════════════════════════════════════════════════════════════

function findSwings(candles, lookback = 5) {
  const highs = [], lows = [];
  for (let i = lookback; i < candles.length - lookback; i++) {
    const slice = candles.slice(i - lookback, i + lookback + 1);
    const maxH  = Math.max(...slice.map(c => c.high));
    const minL  = Math.min(...slice.map(c => c.low));
    if (candles[i].high === maxH) highs.push({ i, price: candles[i].high });
    if (candles[i].low  === minL) lows.push({ i,  price: candles[i].low });
  }
  return { highs, lows };
}

function supportResistance(candles, n = 5) {
  const { highs, lows } = findSwings(candles, n);
  const levels = [];
  // Cluster nearby levels
  const allLevels = [
    ...highs.map(h => ({ price: h.price, type: 'resistance' })),
    ...lows.map(l  => ({ price: l.price, type: 'support' })),
  ].sort((a, b) => a.price - b.price);

  let cluster = null;
  for (const lvl of allLevels) {
    if (!cluster) { cluster = { ...lvl, count: 1 }; continue; }
    if (Math.abs(lvl.price - cluster.price) / cluster.price < 0.005) {
      cluster.count++;
      cluster.price = (cluster.price + lvl.price) / 2;
    } else {
      levels.push(cluster);
      cluster = { ...lvl, count: 1 };
    }
  }
  if (cluster) levels.push(cluster);
  return levels.sort((a, b) => b.count - a.count).slice(0, 8);
}

// ═══════════════════════════════════════════════════════════════════
// HARMONIC PATTERNS (Gartley, Butterfly, Bat, Crab, Shark)
// ═══════════════════════════════════════════════════════════════════

function findAlternatingPivots(candles, lb = 5) {
  const raw = [];
  for (let i = lb; i < candles.length - lb; i++) {
    const hi = candles[i].high;
    const lo = candles[i].low;
    let isH = true, isL = true;
    for (let k = 1; k <= lb; k++) {
      if (candles[i-k].high >= hi || candles[i+k].high >= hi) isH = false;
      if (candles[i-k].low  <= lo || candles[i+k].low  <= lo) isL = false;
    }
    if (isH) raw.push({ i, type: 'H', price: hi });
    else if (isL) raw.push({ i, type: 'L', price: lo });
  }
  // Keep only alternating pivots (pick more extreme when consecutive same type)
  const clean = [];
  for (const p of raw) {
    const last = clean[clean.length - 1];
    if (!last || last.type !== p.type) { clean.push(p); continue; }
    if ((p.type === 'H' && p.price > last.price) || (p.type === 'L' && p.price < last.price))
      clean[clean.length - 1] = p;
  }
  return clean;
}

function detectHarmonicPatterns(candles) {
  const price = candles[candles.length - 1].close;
  // Intentar con distintos lookbacks para adaptarse a distintos timeframes
  let pivots = findAlternatingPivots(candles, 5);
  if (pivots.length < 4) pivots = findAlternatingPivots(candles, 3);
  if (pivots.length < 4) return [];

  const FIB_TOL = 0.09;
  function inRange(v, lo, hi) {
    return v >= lo * (1 - FIB_TOL) && v <= hi * (1 + FIB_TOL);
  }
  function near(v, t) { return Math.abs(v - t) / t <= FIB_TOL; }

  // Pattern definitions: { name, abXa_min, abXa_max, bcAb_min, bcAb_max, dXa_min, dXa_max, acc }
  const DEFS = [
    { name: 'Gartley',   abMin:0.58, abMax:0.66, bcMin:0.38, bcMax:0.89, dMin:0.74, dMax:0.83, acc:78 },
    { name: 'Bat',       abMin:0.36, abMax:0.52, bcMin:0.38, bcMax:0.89, dMin:0.84, dMax:0.92, acc:82 },
    { name: 'Butterfly', abMin:0.74, abMax:0.84, bcMin:0.38, bcMax:0.89, dMin:1.22, dMax:1.70, acc:73 },
    { name: 'Crab',      abMin:0.36, abMax:0.62, bcMin:0.38, bcMax:0.89, dMin:1.55, dMax:1.68, acc:71 },
    { name: 'Shark',     abMin:0.44, abMax:0.56, bcMin:1.00, bcMax:1.68, dMin:0.85, dMax:0.92, acc:68 },
  ];

  const results = [];
  const recent = pivots.slice(-8);

  for (let i = 0; i <= recent.length - 4; i++) {
    const [X, A, B, C] = recent.slice(i, i + 4);
    if (X.type === A.type || A.type === B.type || B.type === C.type) continue;

    const XA = Math.abs(A.price - X.price);
    const AB = Math.abs(B.price - A.price);
    const BC = Math.abs(C.price - B.price);
    if (XA < 1e-12 || AB < 1e-12) continue;

    const abXa = AB / XA;
    const bcAb = BC / AB;
    const isBull = A.type === 'L'; // A=low → D is bullish reversal zone

    for (const def of DEFS) {
      if (!inRange(abXa, def.abMin, def.abMax)) continue;
      if (!inRange(bcAb, def.bcMin, def.bcMax)) continue;

      // D completion zone (PRZ)
      const dir = A.price > X.price ? 1 : -1; // XA direction
      const dMid = X.price - dir * XA * ((def.dMin + def.dMax) / 2);
      const dLow  = X.price - dir * XA * def.dMax;
      const dHigh = X.price - dir * XA * def.dMin;

      const distPct = Math.abs(price - dMid) / Math.max(price, dMid) * 100;
      let stage, prob;
      if (distPct < 1.5)  { stage = 'COMPLETANDO'; prob = def.acc; }
      else if (distPct < 5) { stage = 'APPROACHANDO PRZ'; prob = Math.round(def.acc * 0.8); }
      else { stage = 'FORMANDO'; prob = Math.round(def.acc * 0.5); }

      const t1 = isBull ? B.price : B.price;
      const t2 = isBull ? A.price : A.price;
      const fib618 = dMid + (isBull ? 1 : -1) * Math.abs(C.price - dMid) * 0.618;
      const fib100 = isBull ? A.price : A.price;

      results.push({
        name: def.name,
        type: isBull ? 'bullish' : 'bearish',
        stage, prob,
        dZone: { low: Math.min(dLow, dHigh), high: Math.max(dLow, dHigh), mid: dMid },
        sl:   isBull ? Math.min(dLow, dHigh) * 0.993 : Math.max(dLow, dHigh) * 1.007,
        tp1: fib618,
        tp2: fib100,
        distPct: distPct.toFixed(1),
        ratios: { 'AB/XA': abXa.toFixed(3), 'BC/AB': bcAb.toFixed(3) },
      });
      break; // one pattern per XABC combination
    }
  }

  const seen = new Set();
  return results
    .sort((a, b) => b.prob - a.prob)
    .filter(p => { if (seen.has(p.name)) return false; seen.add(p.name); return true; })
    .slice(0, 4);
}

// ═══════════════════════════════════════════════════════════════════
// ELLIOTT WAVE DETECTION (simplified 5-wave impulse / 3-wave ABC)
// ═══════════════════════════════════════════════════════════════════

function detectElliottWave(candles) {
  const price = candles[candles.length - 1].close;
  let pivots = findAlternatingPivots(candles, 4);
  if (pivots.length < 5) pivots = findAlternatingPivots(candles, 3);
  if (pivots.length < 5) pivots = findAlternatingPivots(candles, 2);
  if (pivots.length < 5) return null;

  const recent = pivots.slice(-7);

  // Try to find 5-wave impulse in the last 5+ pivots
  for (let s = 0; s <= recent.length - 5; s++) {
    const pts = recent.slice(s, s + 5);
    const [p0, p1, p2, p3, p4] = pts;

    const isBullImpulse = p0.type === 'L' && p1.type === 'H' && p2.type === 'L' && p3.type === 'H' && p4.type === 'L';
    const isBearImpulse = p0.type === 'H' && p1.type === 'L' && p2.type === 'H' && p3.type === 'L' && p4.type === 'H';
    if (!isBullImpulse && !isBearImpulse) continue;

    const w1 = Math.abs(p1.price - p0.price);
    const w2 = Math.abs(p2.price - p1.price);
    const w3 = Math.abs(p3.price - p2.price);
    const w4 = Math.abs(p4.price - p3.price);
    if (!w1 || !w3) continue;

    // Elliott rules — reglas DURAS (ambas deben cumplirse)
    const hard1 = w2 / w1 < 1.0;   // W2 nunca retrocede 100%+ de W1 (fundamental)
    const hard2 = w4 / w3 < 1.0;   // W4 nunca retrocede 100%+ de W3
    if (!hard1 || !hard2) continue; // si falla cualquiera, no es impulso válido

    // Reglas BLANDAS — necesita al menos 1 de 2
    const soft1 = w3 >= Math.min(w1, w4);   // W3 no es la ola más corta
    const soft2 = isBullImpulse ? p4.price > p1.price : p4.price < p1.price; // W4 no solapa W1
    if (!soft1 && !soft2) continue;

    const score = [hard1, hard2, soft1, soft2].filter(Boolean).length; // 2–4

    const dir = isBullImpulse ? 'bullish' : 'bearish';
    const confidence = 45 + score * 13;

    // Determine current wave from current price
    let currentWave, nextMove, nextTarget;
    if (isBullImpulse) {
      if (price > p3.price) {
        currentWave = 'W5 ▲'; nextMove = 'Impulso final alcista — buscar techo';
        nextTarget = p3.price + w3 * 0.618;
      } else if (price > p2.price) {
        currentWave = 'W4 ↘ (correctivo)'; nextMove = 'Corrección — soporte en ~' + fmtPrice(p2.price) + ', luego W5';
        nextTarget = p3.price + w1 * 0.618;
      } else {
        currentWave = 'W3 ▲ (más fuerte)'; nextMove = 'Impulso W3 en marcha — la más poderosa';
        nextTarget = p1.price + w1 * 1.618;
      }
    } else {
      if (price < p3.price) {
        currentWave = 'W5 ▼'; nextMove = 'Impulso bajista final — buscar suelo';
        nextTarget = p3.price - w3 * 0.618;
      } else if (price < p2.price) {
        currentWave = 'W4 ↗ (correctivo)'; nextMove = 'Corrección — resistencia en ~' + fmtPrice(p2.price) + ', luego W5 bajista';
        nextTarget = p3.price - w1 * 0.618;
      } else {
        currentWave = 'W3 ▼ (más fuerte)'; nextMove = 'Caída W3 en marcha';
        nextTarget = p1.price - w1 * 1.618;
      }
    }

    return { type: 'impulso', dir, currentWave, nextMove, nextTarget, confidence: Math.min(88, confidence),
      w2Ret: (w2/w1*100).toFixed(0), w3vsW1: (w3/w1).toFixed(2), w4Ret: (w4/w3*100).toFixed(0),
      pts: { p0, p1, p2, p3, p4 } };
  }

  // Try ABC correction (3 alternating pivots)
  if (recent.length >= 3) {
    const [pA, pB, pC] = recent.slice(-3);
    if (pA.type !== pB.type && pB.type !== pC.type) {
      const wA = Math.abs(pB.price - pA.price);
      const wB = Math.abs(pC.price - pB.price);
      const bcRet = wB / wA;
      if (bcRet < 1.0) {
        const dir = pA.type === 'H' ? 'bearish' : 'bullish';
        const cTarget = pA.type === 'H'
          ? pB.price - wA * 0.618
          : pB.price + wA * 0.618;
        return {
          type: 'corrección', dir, currentWave: 'Wave C', confidence: 52,
          nextMove: `Wave C ${dir === 'bearish' ? 'bajista' : 'alcista'} — objetivo ~$${fmtPrice(cTarget)}`,
          nextTarget: cTarget, w2Ret: null, w3vsW1: null, w4Ret: null,
          pts: { p0: pA, p1: pB, p2: pC }
        };
      }
    }
  }
  return null;
}

// ═══════════════════════════════════════════════════════════════════
// MULTI-TIMEFRAME ANALYSIS
// ═══════════════════════════════════════════════════════════════════

const MTF_MAP = {
  '1m':  { primary: '1h',  secondary: '15m' },
  '5m':  { primary: '4h',  secondary: '1h'  },
  '15m': { primary: '4h',  secondary: '1h'  },
  '30m': { primary: '1d',  secondary: '4h'  },
  '1h':  { primary: '1d',  secondary: '4h'  },
  '4h':  { primary: '1w',  secondary: '1d'  },
  '1d':  { primary: '1w',  secondary: '4h'  },
  '1w':  { primary: '1w',  secondary: '1d'  },
};

async function fetchMTFanalysis(symbol, currentTf) {
  const tfConf = MTF_MAP[currentTf] || { primary: '1d', secondary: '4h' };

  // ── S/R rápido por swing highs/lows ──────────────────────────────
  const calcSR = (cArr) => {
    if (!cArr || cArr.length < 20) return { res: null, sup: null };
    const price = cArr[cArr.length - 1].close;
    const lb = 4;
    let res = null, sup = null;
    for (let i = lb; i < cArr.length - lb; i++) {
      const sl = cArr.slice(i - lb, i + lb + 1);
      if (cArr[i].high >= Math.max(...sl.map(c => c.high)) && cArr[i].high > price) {
        if (!res || cArr[i].high < res) res = cArr[i].high;
      }
      if (cArr[i].low <= Math.min(...sl.map(c => c.low)) && cArr[i].low < price) {
        if (!sup || cArr[i].low > sup) sup = cArr[i].low;
      }
    }
    return { res, sup };
  };

  // ── Tendencia + RSI + S/R por TF ─────────────────────────────────
  const calcTrend = (closes, cArr) => {
    if (!closes?.length) return null;
    const n = closes.length;
    const ema = (arr, per) => {
      const k = 2 / (per + 1);
      return arr.reduce((prev, v, i) => i === 0 ? v : prev * (1 - k) + v * k);
    };
    const ema20 = ema(closes.slice(-20), 20);
    const ema50 = ema(closes.slice(-Math.min(50, n)), 50);
    const last = closes[n - 1];
    const rsiData = closes.slice(-15);
    const gains = [], losses = [];
    for (let i = 1; i < rsiData.length; i++) {
      const d = rsiData[i] - rsiData[i-1];
      gains.push(d > 0 ? d : 0); losses.push(d < 0 ? -d : 0);
    }
    const ag = gains.reduce((a,b)=>a+b,0)/gains.length;
    const al = losses.reduce((a,b)=>a+b,0)/losses.length;
    const rsi = al === 0 ? 99 : Math.round(100 - 100/(1+ag/al));
    let dir;
    if (last > ema20 && ema20 > ema50)      dir = 'bullish';
    else if (last < ema20 && ema20 < ema50) dir = 'bearish';
    else if (last > ema20)                   dir = 'neutral_up';
    else                                     dir = 'neutral_down';
    const sr = calcSR(cArr);
    return { dir, rsi, ema20: ema20.toFixed(2), last: last.toFixed(2), res: sr.res, sup: sr.sup };
  };

  try {
    const [primCandles, secCandles] = await Promise.all([
      fetchKlinesUniversal(symbol, tfConf.primary,   80).catch(() => null),
      fetchKlinesUniversal(symbol, tfConf.secondary, 80).catch(() => null),
    ]);
    const prim = calcTrend(primCandles?.map(c=>c.close), primCandles);
    const sec  = calcTrend(secCandles?.map(c=>c.close),  secCandles);
    if (!prim && !sec) return null;

    const dirs = [prim?.dir, sec?.dir].filter(Boolean);
    const bullN = dirs.filter(d => d.includes('bullish') || d === 'neutral_up').length;
    const bearN = dirs.filter(d => d.includes('bearish') || d === 'neutral_down').length;
    let align, alignScore;
    if      (bullN === 2)    { align = 'ALCISTA ↑↑';  alignScore = 88; }
    else if (bearN === 2)    { align = 'BAJISTA ↓↓';  alignScore = 88; }
    else if (bullN > bearN)  { align = 'ALCISTA ↑~';  alignScore = 62; }
    else if (bearN > bullN)  { align = 'BAJISTA ↓~';  alignScore = 62; }
    else                     { align = 'DIVERGENTE ↕'; alignScore = 35; }

    return { primaryTf: tfConf.primary, secondaryTf: tfConf.secondary,
      currentTf, prim, sec, align, alignScore };
  } catch { return null; }
}

// ═══════════════════════════════════════════════════════════════════
// ADVANCED PROBABILITY ENGINE
// ═══════════════════════════════════════════════════════════════════

function computeAdvancedProb({ score, mtf, harmonics, elliott, div, volume, patterns, rsi, hist, mlResult }) {
  let bull, bear, conf;
  const factors = [];

  if (mlResult) {
    // Cuando hay modelo ML: la base es el ML (60%) + score técnico (40%)
    const mlBull = mlResult.bullPct;
    const mlBear = mlResult.bearPct;
    bull = mlBull * 0.6 + score * 0.4;
    bear = mlBear * 0.6 + (100 - score) * 0.4;
    conf = 50 + Math.round((mlResult.auc - 0.5) * 80); // AUC 0.70 → conf 66
    factors.push(`🤖 ML ${mlBull}% alcista · AUC ${mlResult.auc}`);
  } else {
    bull = score;
    bear = 100 - score;
    conf = 48;
    factors.push('Reglas técnicas (sin modelo ML)');
  }

  // MTF alignment (high weight)
  if (mtf) {
    if (mtf.align === 'ALCISTA ↑↑')  { bull += 18; conf += 18; factors.push('MTF ↑↑ alineado alcista'); }
    else if (mtf.align === 'BAJISTA ↓↓') { bear += 18; conf += 18; factors.push('MTF ↓↓ alineado bajista'); }
    else if (mtf.align?.includes('ALCISTA')) { bull += 9; conf += 6; factors.push('MTF parcialmente alcista'); }
    else if (mtf.align?.includes('BAJISTA')) { bear += 9; conf += 6; factors.push('MTF parcialmente bajista'); }
    else { conf -= 12; factors.push('MTF divergente — señal reducida'); }
  }

  // Harmonic patterns (very high weight when near PRZ)
  const bestH = harmonics?.[0];
  if (bestH?.stage === 'COMPLETANDO' || bestH?.stage === 'APPROACHANDO PRZ') {
    const w = bestH.stage === 'COMPLETANDO' ? 22 : 14;
    if (bestH.type === 'bullish') { bull += w; conf += 14; }
    else { bear += w; conf += 14; }
    factors.push(`Armónico ${bestH.name} ${bestH.stage} (${bestH.prob}%)`);
  }

  // Elliott wave
  if (elliott) {
    const w = elliott.currentWave;
    if (w?.includes('W3') && elliott.dir === 'bullish')  { bull += 16; conf += 10; factors.push(`Elliott ${w} — ola más fuerte`); }
    else if (w?.includes('W3') && elliott.dir === 'bearish') { bear += 16; conf += 10; factors.push(`Elliott ${w}`); }
    else if (w?.includes('W5') && elliott.dir === 'bullish') { bull += 8; bear += 4; factors.push(`Elliott ${w} — ola final (reversión pronto)`); }
    else if (w?.includes('W5') && elliott.dir === 'bearish') { bear += 8; bull += 4; factors.push(`Elliott ${w} — ola final`); }
    else if (w?.includes('W4')) { conf -= 8; factors.push(`Elliott ${w} — corrección activa`); }
    conf += Math.round(elliott.confidence * 0.12);
  }

  // Classic signals
  if (div?.type === 'bullish') { bull += 14; conf += 8; factors.push('Divergencia alcista RSI'); }
  if (div?.type === 'bearish') { bear += 14; conf += 8; factors.push('Divergencia bajista RSI'); }
  if (volume?.spike) {
    if (volume.bias === 'buyPressure') { bull += 8; factors.push(`Spike volumen ${volume.ratio?.toFixed(1)}x comprador`); }
    else { bear += 8; factors.push(`Spike volumen ${volume.ratio?.toFixed(1)}x vendedor`); }
  }
  const topPat = patterns?.find(p => p.probability >= 60);
  if (topPat) {
    if (topPat.type === 'bullish') { bull += 10; factors.push(`Patrón ${topPat.name} (${topPat.probability}%)`); }
    else { bear += 10; factors.push(`Patrón ${topPat.name} (${topPat.probability}%)`); }
  }

  const total = bull + bear || 1;
  const bullPct = Math.min(97, Math.max(3, Math.round(bull / total * 100)));
  const bearPct = 100 - bullPct;
  conf = Math.min(96, Math.max(28, Math.round(conf)));
  return { bullPct, bearPct, conf, factors };
}

// ═══════════════════════════════════════════════════════════════════
// RSI DIVERGENCE
// ═══════════════════════════════════════════════════════════════════

function detectRSIDivergence(candles, rsi) {
  const n = candles.length;
  const lookback = 30;
  const start    = Math.max(n - lookback, 5);
  const recent   = candles.slice(start);
  const recentRsi = rsi.slice(start);

  // Find last two swing lows and highs in price + RSI
  const priceHighs = [], priceLows = [];
  for (let i = 2; i < recent.length - 2; i++) {
    if (recent[i].high > recent[i-1].high && recent[i].high > recent[i+1].high &&
        recent[i].high > recent[i-2].high && recent[i].high > recent[i+2].high)
      priceHighs.push({ i, price: recent[i].high, rsi: recentRsi[i] });
    if (recent[i].low < recent[i-1].low && recent[i].low < recent[i+1].low &&
        recent[i].low < recent[i-2].low && recent[i].low < recent[i+2].low)
      priceLows.push({ i, price: recent[i].low, rsi: recentRsi[i] });
  }

  const result = { type: null, strength: 0, label: '' };

  // Bearish divergence: price HH but RSI LH
  if (priceHighs.length >= 2) {
    const [prev, curr] = priceHighs.slice(-2);
    if (curr.price > prev.price && curr.rsi < prev.rsi && curr.rsi !== null && prev.rsi !== null) {
      const priceDiff = ((curr.price - prev.price) / prev.price) * 100;
      const rsiDiff   = curr.rsi - prev.rsi;
      result.type     = 'bearish';
      result.strength = Math.min(100, Math.abs(rsiDiff) * 3);
      result.label    = `Divergencia BAJISTA: Precio +${priceDiff.toFixed(1)}% | RSI ${rsiDiff.toFixed(1)} pts`;
    }
  }
  // Bullish divergence: price LL but RSI HL
  if (priceLows.length >= 2) {
    const [prev, curr] = priceLows.slice(-2);
    if (curr.price < prev.price && curr.rsi > prev.rsi && curr.rsi !== null && prev.rsi !== null) {
      const priceDiff = ((prev.price - curr.price) / prev.price) * 100;
      const rsiDiff   = curr.rsi - prev.rsi;
      if (Math.abs(rsiDiff) > Math.abs(result.strength / 3 || 0)) {
        result.type     = 'bullish';
        result.strength = Math.min(100, Math.abs(rsiDiff) * 3);
        result.label    = `Divergencia ALCISTA: Precio −${priceDiff.toFixed(1)}% | RSI +${rsiDiff.toFixed(1)} pts`;
      }
    }
  }
  return result;
}

// ═══════════════════════════════════════════════════════════════════
// CHART PATTERN DETECTION
// ═══════════════════════════════════════════════════════════════════

function detectChartPatterns(candles) {
  const patterns = [];
  const n = candles.length;
  const last = candles.slice(-50);
  const { highs, lows } = findSwings(last, 3);

  const currentPrice = last[last.length - 1].close;

  // ── Double Top ─────────────────────────────────────────────────
  if (highs.length >= 2) {
    const [h1, h2] = highs.slice(-2);
    const diff = Math.abs(h1.price - h2.price) / h1.price;
    if (diff < 0.015 && h2.i > h1.i + 3) {
      const neckline = Math.min(...last.slice(h1.i, h2.i + 1).map(c => c.low));
      const target   = neckline - (h1.price - neckline);
      const prob     = Math.round(60 + (1 - diff / 0.015) * 20);
      patterns.push({
        name: 'Doble Techo',
        type: 'bearish',
        probability: prob,
        target,
        description: `Dos máximos en ~$${fmtPrice(h1.price)}. Objetivo bajista: $${fmtPrice(target)}`,
        neckline,
      });
    }
  }

  // ── Double Bottom ──────────────────────────────────────────────
  if (lows.length >= 2) {
    const [l1, l2] = lows.slice(-2);
    const diff = Math.abs(l1.price - l2.price) / l1.price;
    if (diff < 0.015 && l2.i > l1.i + 3) {
      const neckline = Math.max(...last.slice(l1.i, l2.i + 1).map(c => c.high));
      const target   = neckline + (neckline - l1.price);
      const prob     = Math.round(60 + (1 - diff / 0.015) * 20);
      patterns.push({
        name: 'Doble Suelo',
        type: 'bullish',
        probability: prob,
        target,
        description: `Dos mínimos en ~$${fmtPrice(l1.price)}. Objetivo alcista: $${fmtPrice(target)}`,
        neckline,
      });
    }
  }

  // ── Head & Shoulders ───────────────────────────────────────────
  if (highs.length >= 3) {
    const [left, head, right] = highs.slice(-3);
    const shoulderDiff = Math.abs(left.price - right.price) / left.price;
    if (head.price > left.price * 1.01 && head.price > right.price * 1.01 && shoulderDiff < 0.02) {
      const neckline = Math.min(
        Math.min(...last.slice(left.i, head.i).map(c => c.low)),
        Math.min(...last.slice(head.i, right.i + 1).map(c => c.low))
      );
      const target = neckline - (head.price - neckline);
      patterns.push({
        name: 'Cabeza y Hombros',
        type: 'bearish',
        probability: 72,
        target,
        description: `Patrón bajista clásico. Objetivo: $${fmtPrice(target)}`,
        neckline,
      });
    }
  }

  // ── Inverse Head & Shoulders ───────────────────────────────────
  if (lows.length >= 3) {
    const [left, head, right] = lows.slice(-3);
    const shoulderDiff = Math.abs(left.price - right.price) / left.price;
    if (head.price < left.price * 0.99 && head.price < right.price * 0.99 && shoulderDiff < 0.02) {
      const neckline = Math.max(
        Math.max(...last.slice(left.i, head.i).map(c => c.high)),
        Math.max(...last.slice(head.i, right.i + 1).map(c => c.high))
      );
      const target = neckline + (neckline - head.price);
      patterns.push({
        name: 'HCH Invertido',
        type: 'bullish',
        probability: 72,
        target,
        description: `Reversión alcista. Objetivo: $${fmtPrice(target)}`,
        neckline,
      });
    }
  }

  // ── Triangle (Ascending / Descending / Symmetrical) ────────────
  if (highs.length >= 3 && lows.length >= 3) {
    const recentHighs = highs.slice(-3);
    const recentLows  = lows.slice(-3);
    const highSlope = (recentHighs[2].price - recentHighs[0].price) / (recentHighs[2].i - recentHighs[0].i);
    const lowSlope  = (recentLows[2].price  - recentLows[0].price)  / (recentLows[2].i  - recentLows[0].i);
    const isAscending   = highSlope < 0.1 && lowSlope > 0.5;
    const isDescending  = highSlope < -0.5 && lowSlope > -0.1;
    const isSymmetrical = highSlope < -0.3 && lowSlope > 0.3;
    if (isAscending) {
      patterns.push({ name: 'Triángulo Ascendente', type: 'bullish', probability: 65, description: 'Compresión con techo plano — ruptura alcista esperada.' });
    } else if (isDescending) {
      patterns.push({ name: 'Triángulo Descendente', type: 'bearish', probability: 65, description: 'Compresión con suelo plano — ruptura bajista esperada.' });
    } else if (isSymmetrical) {
      patterns.push({ name: 'Triángulo Simétrico', type: 'neutral', probability: 55, description: 'Compresión — ruptura en la dirección de la tendencia.' });
    }
  }

  // ── Bull/Bear Flag ─────────────────────────────────────────────
  const flagLookback = last.slice(-20);
  const poleStart    = flagLookback[0].close;
  const poleEnd      = flagLookback[10]?.close;
  if (poleEnd && Math.abs(poleEnd - poleStart) / poleStart > 0.03) {
    const flagBody = flagLookback.slice(10);
    const flagHH   = Math.max(...flagBody.map(c => c.high));
    const flagLL   = Math.min(...flagBody.map(c => c.low));
    const flagRange = (flagHH - flagLL) / flagLookback[10].close;
    if (flagRange < 0.02) {
      if (poleEnd > poleStart) {
        patterns.push({ name: 'Bull Flag', type: 'bullish', probability: 68, description: `Consolidación tras subida de ${((poleEnd-poleStart)/poleStart*100).toFixed(1)}%. Continuación alcista esperada.` });
      } else {
        patterns.push({ name: 'Bear Flag', type: 'bearish', probability: 68, description: `Consolidación tras caída de ${((poleStart-poleEnd)/poleStart*100).toFixed(1)}%. Continuación bajista esperada.` });
      }
    }
  }

  return patterns;
}

// ═══════════════════════════════════════════════════════════════════
// SMART MONEY CONCEPTS (SMC)
// ═══════════════════════════════════════════════════════════════════

function detectSMC(candles) {
  const n = candles.length;
  const recent = candles.slice(-80);

  // ── Fair Value Gaps (FVG) ──────────────────────────────────────
  const fvgs = [];
  for (let i = 2; i < recent.length - 1; i++) {
    const c1 = recent[i - 2], c2 = recent[i - 1], c3 = recent[i];
    // Bullish FVG: c3.low > c1.high (gap between c1 top and c3 bottom)
    if (c3.low > c1.high) {
      fvgs.push({ type: 'bullish', top: c3.low, bottom: c1.high, i, filled: false,
        label: `FVG Alcista $${fmtPrice(c1.high)}–$${fmtPrice(c3.low)}` });
    }
    // Bearish FVG: c3.high < c1.low
    if (c3.high < c1.low) {
      fvgs.push({ type: 'bearish', top: c1.low, bottom: c3.high, i, filled: false,
        label: `FVG Bajista $${fmtPrice(c3.high)}–$${fmtPrice(c1.low)}` });
    }
  }
  // Mark filled FVGs
  const price = recent[recent.length - 1].close;
  const activeFvgs = fvgs.slice(-6).filter(g => {
    if (g.type === 'bullish') return price > g.bottom; // still valid if price above
    return price < g.top;
  });

  // ── Order Blocks ───────────────────────────────────────────────
  const orderBlocks = [];
  for (let i = 5; i < recent.length - 5; i++) {
    const c = recent[i];
    const nextMove = recent.slice(i + 1, i + 6);
    const maxNext  = Math.max(...nextMove.map(x => x.close));
    const minNext  = Math.min(...nextMove.map(x => x.close));
    // Bullish OB: a bearish candle followed by strong up move
    if (c.close < c.open && maxNext > c.high * 1.01) {
      orderBlocks.push({ type: 'bullish', top: c.open, bottom: c.close, i,
        label: `OB Alcista ~$${fmtPrice((c.open + c.close) / 2)}` });
    }
    // Bearish OB: a bullish candle followed by strong down move
    if (c.close > c.open && minNext < c.low * 0.99) {
      orderBlocks.push({ type: 'bearish', top: c.close, bottom: c.open, i,
        label: `OB Bajista ~$${fmtPrice((c.open + c.close) / 2)}` });
    }
  }
  const activeOBs = orderBlocks.slice(-4).filter(ob => {
    if (ob.type === 'bullish') return price >= ob.bottom * 0.995 && price <= ob.top * 1.05;
    return price <= ob.top * 1.005 && price >= ob.bottom * 0.95;
  });

  // ── Break of Structure (BOS) / Change of Character (ChoCH) ────
  const { highs, lows } = findSwings(recent, 4);
  let bos = null;
  if (highs.length >= 2) {
    const [prev, curr] = highs.slice(-2);
    if (curr.price > prev.price) bos = { type: 'bullish_bos', label: `BOS Alcista — nuevo máximo $${fmtPrice(curr.price)}`, price: curr.price };
    else                         bos = { type: 'choch',        label: `ChoCH Bajista — máximo más bajo $${fmtPrice(curr.price)}`, price: curr.price };
  }
  if (lows.length >= 2) {
    const [prev, curr] = lows.slice(-2);
    const bullBos = curr.price > prev.price;
    if (!bos || (bullBos && bos?.type !== 'bullish_bos')) {
      bos = bullBos
        ? { type: 'choch',       label: `ChoCH Alcista — mínimo más alto $${fmtPrice(curr.price)}`, price: curr.price }
        : { type: 'bearish_bos', label: `BOS Bajista — nuevo mínimo $${fmtPrice(curr.price)}`,       price: curr.price };
    }
  }

  // ── Premium / Discount Zones ───────────────────────────────────
  const rangeHigh = Math.max(...recent.map(c => c.high));
  const rangeLow  = Math.min(...recent.map(c => c.low));
  const rangeMid  = (rangeHigh + rangeLow) / 2;
  const zone = price > rangeMid
    ? { label: 'ZONA PREMIUM (sobrecompra)', type: 'bearish', pct: ((price - rangeMid) / (rangeHigh - rangeMid) * 100).toFixed(0) }
    : { label: 'ZONA DISCOUNT (descuento)',  type: 'bullish', pct: ((rangeMid - price) / (rangeMid - rangeLow) * 100).toFixed(0) };

  // ── Liquidity ─────────────────────────────────────────────────
  const swingHighs = highs.slice(-3).map(h => h.price);
  const swingLows  = lows.slice(-3).map(l => l.price);

  return { fvgs: activeFvgs, orderBlocks: activeOBs, bos, zone, rangeHigh, rangeLow, swingHighs, swingLows };
}

// ═══════════════════════════════════════════════════════════════════
// MARKET STRUCTURE
// ═══════════════════════════════════════════════════════════════════

function detectTrend(candles, ema20, ema50, ema200) {
  const n    = candles.length - 1;
  const { highs, lows } = findSwings(candles.slice(-40), 4);

  let hhhl = 0; // higher highs & higher lows = uptrend
  if (highs.length >= 2) {
    const [prev, curr] = highs.slice(-2);
    if (curr.price > prev.price) hhhl++;
    else                         hhhl--;
  }
  if (lows.length >= 2) {
    const [prev, curr] = lows.slice(-2);
    if (curr.price > prev.price) hhhl++;
    else                         hhhl--;
  }

  const e20  = ema20[n], e50 = ema50[n], e200 = ema200[n];
  const price = candles[n].close;

  let score = 0;
  if (price > e20)   score++;
  if (e20 > e50)     score++;
  if (e50 > e200)    score++;
  if (hhhl >= 1)     score++;
  if (price > e200)  score += 2;

  let trend, strength;
  if      (score >= 5) { trend = 'UPTREND FUERTE';  strength = 95; }
  else if (score >= 4) { trend = 'UPTREND';          strength = 75; }
  else if (score >= 3) { trend = 'NEUTRAL / LATERAL'; strength = 50; }
  else if (score >= 2) { trend = 'DOWNTREND';         strength = 35; }
  else                 { trend = 'DOWNTREND FUERTE';  strength = 10; }

  return {
    trend, strength, score,
    ema20: e20, ema50: e50, ema200: e200,
    priceVsEma200: e200 ? ((price - e200) / e200 * 100).toFixed(2) : null,
    goldCross: e20 > e50 && ema20[n-1] <= ema50[n-1],
    deathCross: e20 < e50 && ema20[n-1] >= ema50[n-1],
  };
}

// ═══════════════════════════════════════════════════════════════════
// VOLUME ANALYSIS
// ═══════════════════════════════════════════════════════════════════

function analyzeVolume(candles) {
  const n     = candles.length;
  const last5 = candles.slice(-5);
  const avg20 = calcSMA(candles.map(c => c.volume), 20)[n - 1];
  const curr  = candles[n - 1];
  const volRatio = avg20 ? curr.volume / avg20 : 1;
  const buyVol = last5.filter(c => c.close > c.open).reduce((s, c) => s + c.volume, 0);
  const sellVol = last5.filter(c => c.close < c.open).reduce((s, c) => s + c.volume, 0);
  const cvd = buyVol - sellVol; // cumulative volume delta approximation

  return {
    current:  curr.volume,
    avg20,
    ratio:    volRatio,
    spike:    volRatio > 2,
    buyVol, sellVol, cvd,
    label:    volRatio > 2   ? 'SPIKE — alta actividad'
            : volRatio > 1.5 ? 'Por encima del promedio'
            : volRatio < 0.5 ? 'Bajo — poca actividad'
            : 'Normal',
    bias:     cvd > 0 ? 'buyPressure' : 'sellPressure',
  };
}

// ═══════════════════════════════════════════════════════════════════
// VERDICT & PROJECTION
// ═══════════════════════════════════════════════════════════════════

function computeVerdict({ score, trend, patterns, smc, div, volume, indicators, price }) {
  const bullReasons = [];
  const bearReasons = [];

  // EMA alignment
  if (trend.score >= 5) bullReasons.push('EMAs 20/50/200 alineadas alcistas — tendencia confirmada');
  else if (trend.score <= 2) bearReasons.push('EMAs 20/50/200 alineadas bajistas — tendencia confirmada');
  if (trend.goldCross)  bullReasons.push('Golden Cross activo (EMA20 cruzó EMA50 al alza)');
  if (trend.deathCross) bearReasons.push('Death Cross activo (EMA20 cruzó EMA50 a la baja)');

  // RSI
  const rsi = indicators.rsi;
  if (rsi !== null) {
    if (rsi < 30)       bullReasons.push(`RSI sobrevendido (${rsi.toFixed(0)}) — posible rebote inminente`);
    else if (rsi > 70)  bearReasons.push(`RSI sobrecomprado (${rsi.toFixed(0)}) — zona de rechazo potencial`);
    else if (rsi > 55)  bullReasons.push(`RSI en zona alcista (${rsi.toFixed(0)}) — momentum positivo`);
    else if (rsi < 45)  bearReasons.push(`RSI en zona bajista (${rsi.toFixed(0)}) — momentum negativo`);
  }

  // RSI divergence
  if (div?.type === 'bullish') bullReasons.push(`Divergencia alcista RSI — precio baja pero RSI sube`);
  if (div?.type === 'bearish') bearReasons.push(`Divergencia bajista RSI — precio sube pero RSI baja`);

  // MACD histogram
  const hist = indicators.hist;
  if (hist > 0)  bullReasons.push('Histograma MACD positivo — compradores en control');
  if (hist < 0)  bearReasons.push('Histograma MACD negativo — vendedores en control');

  // Chart patterns
  for (const p of patterns.slice(0, 3)) {
    if (p.type === 'bullish') bullReasons.push(`Patrón: ${p.name} (confianza ${p.probability}%)`);
    if (p.type === 'bearish') bearReasons.push(`Patrón: ${p.name} (confianza ${p.probability}%)`);
  }

  // SMC zone
  if (smc.zone.type === 'bullish') bullReasons.push('Precio en zona de descuento SMC — posición favorable compradores');
  if (smc.zone.type === 'bearish') bearReasons.push('Precio en zona premium SMC — posición favorable vendedores');

  // BOS / ChoCH
  if (smc.bos) {
    const bLabel = smc.bos.label || '';
    if (smc.bos.type === 'bullish_bos')   bullReasons.push(`BOS alcista — estructura rota al alza`);
    if (smc.bos.type === 'bearish_bos')   bearReasons.push(`BOS bajista — estructura rota a la baja`);
    if (smc.bos.type === 'choch' && bLabel.includes('Alcista')) bullReasons.push('ChoCH alcista — posible cambio de tendencia');
    if (smc.bos.type === 'choch' && bLabel.includes('Bajista')) bearReasons.push('ChoCH bajista — posible cambio de tendencia');
  }

  // Order Blocks
  for (const ob of smc.orderBlocks.slice(0, 2)) {
    const obMid = ((ob.top || ob.high || 0) + (ob.bottom || ob.low || 0)) / 2;
    if (ob.type === 'bullish' && obMid > 0) bullReasons.push(`Order Block comprador ~$${fmtPrice(obMid)}`);
    if (ob.type === 'bearish' && obMid > 0) bearReasons.push(`Order Block vendedor ~$${fmtPrice(obMid)}`);
  }

  // Volume
  if (volume.bias === 'buyPressure')  bullReasons.push('CVD positivo — presión compradora domina');
  if (volume.bias === 'sellPressure') bearReasons.push('CVD negativo — presión vendedora domina');
  if (volume.spike && score >= 50)    bullReasons.push('Spike de volumen acompañando movimiento alcista');
  if (volume.spike && score < 50)     bearReasons.push('Spike de volumen acompañando movimiento bajista');

  // Final verdict
  const bullPct = score;
  const bearPct = 100 - score;
  const confidence = Math.min(100, Math.round(Math.abs(score - 50) * 2));

  let label, cls;
  if      (score >= 72) { label = 'ALCISTA FUERTE';    cls = 'verd-strong-bull'; }
  else if (score >= 58) { label = 'ALCISTA';            cls = 'verd-bull'; }
  else if (score >= 48) { label = 'NEUTRAL';            cls = 'verd-neutral'; }
  else if (score >= 35) { label = 'BAJISTA';            cls = 'verd-bear'; }
  else                  { label = 'BAJISTA FUERTE';     cls = 'verd-strong-bear'; }

  return { bullPct, bearPct, confidence, label, cls, bullReasons, bearReasons };
}

// ═══════════════════════════════════════════════════════════════════
// TRADING BRAIN — synthesizes all data like a pro trader
// ═══════════════════════════════════════════════════════════════════

function computeTradingBrain({ score, trend, patterns, smc, div, volume, signals,
  indicators, price, symbol, tf, support, resistance, suggestedSL, suggestedTP }) {

  const { rsi, hist, bb, atr } = indicators;
  const supP   = support?.price;
  const resP   = resistance?.price;
  const bbSqueeze = bb?.width != null && bb.width < 0.025;
  const atrPct    = atr && price ? (atr / price) * 100 : 1;

  const topPat = patterns.find(p => p.probability >= 60);

  // ── Key signals with prices ───────────────────────────────────────
  const keySignals = [];

  // Patterns — most important, show exact prices
  for (const p of patterns.slice(0, 3)) {
    const hasPrice = p.neckline || p.target;
    keySignals.push({
      dir:   p.type,
      label: `${p.name} (${p.probability}%)`,
      sub:   [
        p.neckline ? `Neckline: $${fmtPrice(p.neckline)}` : '',
        p.target   ? `Objetivo: $${fmtPrice(p.target)}`    : '',
        p.description?.replace(/[^$0-9.,A-Za-záéíóú~ ]/g,'').trim() || '',
      ].filter(Boolean).join(' · '),
    });
  }

  // RSI
  if (rsi !== null) {
    if      (rsi < 30)  keySignals.push({ dir:'bullish', label:`RSI sobrevendido (${rsi.toFixed(0)})`, sub:`Zona de rebote — posible giro alcista` });
    else if (rsi > 70)  keySignals.push({ dir:'bearish', label:`RSI sobrecomprado (${rsi.toFixed(0)})`, sub:`Zona de rechazo — posible corrección` });
    else if (rsi > 55)  keySignals.push({ dir:'bullish', label:`RSI alcista (${rsi.toFixed(0)})`, sub:`Momentum positivo` });
    else if (rsi < 45)  keySignals.push({ dir:'bearish', label:`RSI bajista (${rsi.toFixed(0)})`, sub:`Momentum negativo` });
  }

  // RSI divergence
  if (div?.type === 'bullish') keySignals.push({ dir:'bullish', label:'Divergencia alcista RSI', sub:'Precio hace mínimos, RSI sube — señal de giro' });
  if (div?.type === 'bearish') keySignals.push({ dir:'bearish', label:'Divergencia bajista RSI', sub:'Precio sube, RSI baja — agotamiento del impulso' });

  // MACD
  if (hist > 0)  keySignals.push({ dir:'bullish', label:'MACD histograma positivo', sub:`Momentum comprador activo` });
  if (hist < 0)  keySignals.push({ dir:'bearish', label:'MACD histograma negativo', sub:`Momentum vendedor activo` });

  // BB squeeze
  if (bbSqueeze) keySignals.push({ dir:'neutral', label:'BB Squeeze', sub:`Ancho ${(bb.width*100).toFixed(2)}% — ruptura explosiva inminente, dirección sin confirmar` });

  // Support/Resistance levels
  if (resP) keySignals.push({ dir:'bearish', label:`Resistencia $${fmtPrice(resP)}`, sub:`${fmtPct((resP-price)/price*100)} arriba — techo clave` });
  if (supP) keySignals.push({ dir:'bullish', label:`Soporte $${fmtPrice(supP)}`, sub:`${fmtPct((supP-price)/price*100)} abajo — piso clave` });

  // SMC
  if (smc.bos) keySignals.push({ dir: smc.bos.type.includes('bull') ? 'bullish':'bearish', label: smc.bos.label, sub:'Estructura de mercado' });
  if (smc.zone.type === 'bullish') keySignals.push({ dir:'bullish', label:'Zona SMC Descuento', sub:'Precio bajo punto medio del rango — favorable compradores' });
  if (smc.zone.type === 'bearish') keySignals.push({ dir:'bearish', label:'Zona SMC Premium', sub:'Precio sobre punto medio del rango — favorable vendedores' });

  // Volume
  if (volume.spike) keySignals.push({ dir: volume.bias==='buyPressure'?'bullish':'bearish',
    label:`Spike de volumen ${volume.ratio.toFixed(1)}x promedio`,
    sub: volume.bias==='buyPressure' ? 'Acumulación institucional potencial' : 'Distribución / venta institucional' });

  // ── Final direction verdict ───────────────────────────────────────
  const bullSignals = keySignals.filter(s => s.dir === 'bullish').length;
  const bearSignals = keySignals.filter(s => s.dir === 'bearish').length;
  const totalSig    = bullSignals + bearSignals || 1;
  const bullWeight  = score; // 0-100 from full analysis
  const bearWeight  = 100 - score;

  // Clear verdict when pattern or strong divergence present
  let finalDir, finalConf, finalReason;
  if (topPat?.type === 'bearish' && topPat.probability >= 70 && bearWeight >= 45) {
    finalDir  = 'SHORT';
    finalConf = Math.round((topPat.probability + bearWeight) / 2);
    finalReason = `${topPat.name} (${topPat.probability}%)${topPat.neckline ? ` — romper $${fmtPrice(topPat.neckline)} activa el patrón` : ''}`;
  } else if (topPat?.type === 'bullish' && topPat.probability >= 70 && bullWeight >= 45) {
    finalDir  = 'LONG';
    finalConf = Math.round((topPat.probability + bullWeight) / 2);
    finalReason = `${topPat.name} (${topPat.probability}%)${topPat.neckline ? ` — romper $${fmtPrice(topPat.neckline)} activa el patrón` : ''}`;
  } else if (div?.type === 'bearish' && score < 50) {
    finalDir  = 'SHORT'; finalConf = Math.round(65 + (50-score)); finalReason = 'Divergencia bajista RSI + sesgo bajista';
  } else if (div?.type === 'bullish' && score > 50) {
    finalDir  = 'LONG';  finalConf = Math.round(65 + (score-50)); finalReason = 'Divergencia alcista RSI + sesgo alcista';
  } else if (score >= 65) {
    finalDir  = 'LONG';  finalConf = score; finalReason = 'Score técnico alto — tendencia alcista dominante';
  } else if (score <= 35) {
    finalDir  = 'SHORT'; finalConf = 100-score; finalReason = 'Score técnico bajo — tendencia bajista dominante';
  } else if (bbSqueeze) {
    finalDir  = 'ESPERAR'; finalConf = null; finalReason = 'BB Squeeze — sin dirección hasta la ruptura';
  } else {
    finalDir  = bearSignals > bullSignals ? 'SHORT' : bullSignals > bearSignals ? 'LONG' : 'ESPERAR';
    finalConf = finalDir === 'ESPERAR' ? null : Math.round(Math.max(bearWeight,bullWeight) * 0.8);
    finalReason = 'Mayoría de señales ' + (finalDir === 'SHORT' ? 'bajistas' : finalDir === 'LONG' ? 'alcistas' : 'neutrales');
  }

  // ── Narrative (concise summary) ──────────────────────────────────
  const narrative = [
    score >= 65 ? 'Momentum alcista dominante.' : score <= 35 ? 'Presión bajista dominante.' : 'Mercado en zona neutral.',
    topPat ? `${topPat.name}${topPat.neckline ? ` — neckline $${fmtPrice(topPat.neckline)}` : ''}${topPat.target ? `, objetivo $${fmtPrice(topPat.target)}` : ''}.` : '',
    bbSqueeze ? `BB Squeeze (${(bb.width*100).toFixed(2)}%) — esperar ruptura con volumen.` : '',
    div?.type ? `Divergencia ${div.type === 'bullish' ? 'alcista' : 'bajista'} RSI activa.` : '',
  ].filter(Boolean).slice(0,3).join(' ');

  // ── Trading mode — reactive to current price ─────────────────────
  let mode, modeCol, modeIcon, breakoutAlert = null;
  const nearRes = resP && Math.abs(price - resP) / price < 0.004;
  const nearSup = supP && Math.abs(price - supP) / price < 0.004;

  // Check real-time breakouts (highest priority)
  const bearNeckBroken = topPat?.type === 'bearish' && topPat.neckline && price < topPat.neckline * 0.9995;
  const bullNeckBroken = topPat?.type === 'bullish' && topPat.neckline && price > topPat.neckline * 1.0005;
  const bbUpperBreak   = indicators.bb?.upper && price > indicators.bb.upper && !bbSqueeze;
  const bbLowerBreak   = indicators.bb?.lower && price < indicators.bb.lower && !bbSqueeze;
  const supBroken      = supP && price < supP * 0.998;
  const resBroken      = resP && price > resP * 1.002;

  if (bearNeckBroken) {
    mode = `▼ SHORT ACTIVADO — Neckline $${fmtPrice(topPat.neckline)} roto`;
    modeCol = '#f85149'; modeIcon = '🔴';
    breakoutAlert = { type:'bear', msg:`Neckline ${topPat.name} roto en $${fmtPrice(topPat.neckline)} — TARGET $${fmtPrice(topPat.target)}`, urgent: true };
  } else if (bullNeckBroken) {
    mode = `▲ LONG ACTIVADO — Neckline $${fmtPrice(topPat.neckline)} roto`;
    modeCol = '#00ff41'; modeIcon = '🟢';
    breakoutAlert = { type:'bull', msg:`Neckline ${topPat.name} roto en $${fmtPrice(topPat.neckline)} — TARGET $${fmtPrice(topPat.target)}`, urgent: true };
  } else if (supBroken && score < 50) {
    mode = `▼ SOPORTE PERDIDO — $${fmtPrice(supP)} roto`;
    modeCol = '#f85149'; modeIcon = '🔴';
    breakoutAlert = { type:'bear', msg:`Soporte $${fmtPrice(supP)} perdido — presión bajista aumenta`, urgent: false };
  } else if (resBroken && score > 50) {
    mode = `▲ RESISTENCIA ROTA — $${fmtPrice(resP)} superado`;
    modeCol = '#00ff41'; modeIcon = '🟢';
    breakoutAlert = { type:'bull', msg:`Resistencia $${fmtPrice(resP)} rota — momentum alcista confirmado`, urgent: false };
  } else if (bbUpperBreak) {
    mode = 'RUPTURA ALCISTA BB — MOMENTUM FUERTE'; modeCol = '#00ff41'; modeIcon = '▲';
    breakoutAlert = { type:'bull', msg:`Precio rompe Banda Superior BB — sesgo alcista activado`, urgent: false };
  } else if (bbLowerBreak) {
    mode = 'RUPTURA BAJISTA BB — POSIBLE CONTINUACIÓN'; modeCol = '#f85149'; modeIcon = '▼';
    breakoutAlert = { type:'bear', msg:`Precio rompe Banda Inferior BB — sesgo bajista activado`, urgent: false };
  } else if (bbSqueeze) {
    mode = 'ESPERAR RUPTURA BB'; modeCol = '#e3b341'; modeIcon = '⏳';
  } else if (nearRes && score < 60) {
    mode = `EN RESISTENCIA $${fmtPrice(resP)} — ESPERAR RECHAZO O RUPTURA`; modeCol = '#f85149'; modeIcon = '🚫';
  } else if (nearSup && score > 40) {
    mode = `EN SOPORTE $${fmtPrice(supP)} — BUSCAR REBOTE`; modeCol = '#00ff41'; modeIcon = '👁';
  } else if (score >= 65) {
    mode = 'BUSCAR ENTRADA LONG EN RETROCESO'; modeCol = '#00ff41'; modeIcon = '▲';
  } else if (score <= 35) {
    mode = 'BUSCAR ENTRADA SHORT EN REBOTE'; modeCol = '#f85149'; modeIcon = '▼';
  } else {
    mode = 'MODO OBSERVACIÓN — SIN EDGE CLARO'; modeCol = '#e3b341'; modeIcon = '👁';
  }

  // Fire browser notification on breakout (once per event)
  if (breakoutAlert) {
    const alertKey = `brain_alert_${symbol}_${breakoutAlert.msg.slice(0,30)}`;
    if (!window[alertKey]) {
      window[alertKey] = true;
      setTimeout(() => delete window[alertKey], 120000); // reset after 2min
      showNotif(`🔔 ${breakoutAlert.msg}`, breakoutAlert.type === 'bull' ? 'success' : 'error');
    }
  }

  // ── Plan A: Bull scenario ─────────────────────────────────────────
  const bullPct  = score;
  const atrVal   = Math.max(atr || 0, price * 0.001); // mínimo 0.1% para evitar ATR=0

  // Validar S/R: solo usar supP si está DEBAJO del precio (es soporte real)
  //               solo usar resP si está ENCIMA del precio (es resistencia real)
  const validSup = supP && supP < price * 0.9998;  // soporte ≥ 0.02% bajo precio
  const validRes = resP && resP > price * 1.0002;  // resistencia ≥ 0.02% sobre precio

  // LONG — entrada cerca del precio, SL siempre por debajo, TP siempre por encima
  // Usamos ATR consistente para evitar inversiones en cualquier timeframe
  const bEntry = validSup ? +(Math.min(supP * 1.001, price - atrVal * 0.3)).toFixed(8)
                           : +(price - atrVal * 0.3).toFixed(8);
  const bTP1   = validRes ? +(resP * 0.998).toFixed(8)
                           : +(price + atrVal * 3).toFixed(8);
  const bTP2   = validRes ? +(resP * 1.015).toFixed(8)
                           : +(price + atrVal * 5).toFixed(8);
  const bSL    = validSup ? +(supP * 0.994).toFixed(8)
                           : +(price - atrVal * 2.5).toFixed(8);
  // Garantía: bEntry siempre entre bSL y bTP1
  const bRR    = bEntry > bSL && bTP1 > bEntry
                   ? +((bTP1 - bEntry) / (bEntry - bSL)).toFixed(1)
                   : 0;

  const bullConditions = [];
  if (validSup) bullConditions.push(`Precio mantiene soporte $${fmtPrice(supP)} como piso`);
  if (div?.type === 'bullish') bullConditions.push('Divergencia alcista RSI confirma reversión');
  if (topPat?.type === 'bullish') bullConditions.push(`Ruptura del neckline $${fmtPrice(topPat.neckline || (validRes ? resP : price))} con volumen`);
  if (smc.zone.type === 'bullish') bullConditions.push('Precio en zona de descuento SMC');
  if (bullConditions.length === 0) bullConditions.push(`Cierre de vela por encima de $${fmtPrice(price + atrVal)} con volumen`);

  // ── Plan B: Bear scenario ─────────────────────────────────────────
  const bearPct  = 100 - score;
  // SHORT — entrada cerca del precio, SL siempre por encima, TP siempre por debajo
  const sEntry = validRes ? +(Math.max(resP * 0.999, price + atrVal * 0.3)).toFixed(8)
                           : +(price + atrVal * 0.3).toFixed(8);
  const sTP1   = validSup ? +(supP * 1.001).toFixed(8)
                           : +(price - atrVal * 3).toFixed(8);
  const sTP2   = validSup ? +(supP * 0.985).toFixed(8)
                           : +(price - atrVal * 5).toFixed(8);
  const sSL    = validRes ? +(resP * 1.006).toFixed(8)
                           : +(price + atrVal * 2.5).toFixed(8);
  // Garantía: sEntry siempre entre sTP1 y sSL
  const sRR    = sSL > sEntry && sEntry > sTP1
                   ? +((sEntry - sTP1) / (sSL - sEntry)).toFixed(1)
                   : 0;

  const bearConditions = [];
  if (resP)  bearConditions.push(`Precio rechazado en resistencia $${fmtPrice(resP)}`);
  if (div?.type === 'bearish') bearConditions.push('Divergencia bajista RSI confirma agotamiento');
  if (topPat?.type === 'bearish') bearConditions.push(`Ruptura del neckline $${fmtPrice(topPat.neckline || supP || price)} a la baja`);
  if (smc.zone.type === 'bearish') bearConditions.push('Precio en zona premium SMC — distribución');
  if (bearConditions.length === 0) bearConditions.push(`Cierre de vela por debajo de $${fmtPrice(price * 0.997)} con volumen`);

  // ── Wait-for conditions (with prices) ────────────────────────────
  const waitFor = [];
  if (topPat?.type === 'bearish' && topPat.neckline) {
    waitFor.push(`Cierre de vela bajo neckline $${fmtPrice(topPat.neckline)} con volumen → SHORT activado. Sin romper ese nivel, patrón inválido.`);
  } else if (topPat?.type === 'bullish' && topPat.neckline) {
    waitFor.push(`Cierre de vela sobre neckline $${fmtPrice(topPat.neckline)} con volumen → LONG activado.`);
  }
  if (bbSqueeze) waitFor.push(`Ruptura BB (ancho ${(bb.width*100).toFixed(2)}%) — no entrar antes de saber la dirección. Monitorear cierre fuera de las bandas.`);
  if (div)       waitFor.push(`Divergencia ${div.type==='bullish'?'alcista':'bajista'} RSI — confirmar que el precio sigue la señal del RSI.`);
  if (resP && price < resP * 0.998) waitFor.push(`Resistencia clave en $${fmtPrice(resP)} — solo LONG si cierra con fuerza por encima.`);
  if (supP && price > supP * 1.002) waitFor.push(`Soporte clave en $${fmtPrice(supP)} — SHORT si pierde ese nivel con volumen.`);
  if (!waitFor.length) {
    if (score >= 60)      waitFor.push(`Retroceso a EMA20 (~$${fmtPrice(trend.ema20)}) sin perder estructura alcista — zona de entrada LONG`);
    else if (score <= 40) waitFor.push(`Rebote a EMA20 (~$${fmtPrice(trend.ema20)}) para entrada SHORT con SL ajustado sobre el rebote`);
    else                  waitFor.push(`Ruptura del rango con volumen 2x el promedio para definir dirección`);
  }

  // ── Pro summary (with prices) ─────────────────────────────────────
  let proTake;
  if (finalDir === 'SHORT' && topPat?.type === 'bearish') {
    proTake = `${topPat.name} confirma sesgo bajista. ${topPat.neckline ? `Esperaría cierre bajo $${fmtPrice(topPat.neckline)} (neckline) para entrar SHORT` : `Buscaría rebote a resistencia $${fmtPrice(resP||sEntry)} para entrar SHORT`}, SL sobre $${fmtPrice(sSL)}, TP en $${fmtPrice(sTP1)}${topPat.target ? ` (objetivo patrón $${fmtPrice(topPat.target)})` : ''}. R:R ${sRR}:1.`;
  } else if (finalDir === 'LONG' && topPat?.type === 'bullish') {
    proTake = `${topPat.name} confirma sesgo alcista. ${topPat.neckline ? `Esperaría cierre sobre $${fmtPrice(topPat.neckline)} (neckline) para entrar LONG` : `Buscaría pullback a soporte $${fmtPrice(supP||bEntry)}`}, SL bajo $${fmtPrice(bSL)}, TP en $${fmtPrice(bTP1)}${topPat.target ? ` (objetivo $${fmtPrice(topPat.target)})` : ''}. R:R ${bRR}:1.`;
  } else if (finalDir === 'ESPERAR') {
    proTake = `Sin setup claro todavía. Esperaría: ${waitFor[0]?.toLowerCase() || 'señal de confirmación'}. Flat (sin posición) es una posición válida.`;
  } else if (finalDir === 'LONG') {
    proTake = `Sesgo alcista. Retroceso a $${fmtPrice(bEntry)} con SL en $${fmtPrice(bSL)} y TP en $${fmtPrice(bTP1)}. R:R ${bRR}:1. ${atrPct.toFixed(1)}% de movimiento promedio (ATR).`;
  } else {
    proTake = `Sesgo bajista. Rebote a $${fmtPrice(sEntry)} con SL en $${fmtPrice(sSL)} y TP en $${fmtPrice(sTP1)}. R:R ${sRR}:1.`;
  }

  // ── Pine Script pipeline ──────────────────────────────────────────
  const pine = buildBrainPine({ symbol, tf, price, resP, supP, bEntry, bTP1, bTP2, bSL, sEntry, sTP1, sTP2, sSL, bullPct, bearPct });

  return {
    narrative, mode, modeCol, modeIcon,
    finalDir, finalConf, finalReason,
    keySignals, breakoutAlert,
    bullPct, bEntry, bTP1, bTP2, bSL, bRR, bullConditions,
    bearPct, sEntry, sTP1, sTP2, sSL, sRR, bearConditions,
    waitFor, proTake, pine,
  };
}

function buildBrainPine({ symbol, tf, price, resP, supP, bEntry, bTP1, bSL, sEntry, sTP1, sSL, bullPct, bearPct }) {
  const p = v => v ? v.toFixed(8) : 'na';
  return `//@version=5
// TVA Brain — ${symbol} ${tf}  |  Alcista ${bullPct}%  Bajista ${bearPct}%
indicator("TVA Brain — ${symbol}", overlay=true, max_lines_count=10)

// ── Niveles ──
var float res    = ${p(resP)}
var float sup    = ${p(supP)}
var float bEntry = ${p(bEntry)}
var float bTP    = ${p(bTP1)}
var float bSL    = ${p(bSL)}
var float sEntry = ${p(sEntry)}
var float sTP    = ${p(sTP1)}
var float sSL    = ${p(sSL)}

// ── Visualización ──
plot(res,    "Resistencia",    color.new(color.orange, 20), 2)
plot(sup,    "Soporte",        color.new(color.aqua,   20), 2)
plot(bEntry, "Long Entry",     color.new(color.green,  40), 1, plot.style_circles)
plot(bTP,    "Long TP",        color.new(color.green,  20), 1, plot.style_cross)
plot(bSL,    "Long SL",        color.new(color.red,    20), 1, plot.style_cross)
plot(sEntry, "Short Entry",    color.new(color.red,    40), 1, plot.style_circles)
plot(sTP,    "Short TP",       color.new(color.aqua,   20), 1, plot.style_cross)
plot(sSL,    "Short SL",       color.new(color.orange, 20), 1, plot.style_cross)

// ── Alertas automáticas ──
alertcondition(ta.crossunder(close, sup),  "⚠️ Soporte roto",     "Soporte $${supP ? supP.toFixed(2) : '?'} roto — evaluar SHORT")
alertcondition(ta.crossover(close,  res),  "⚠️ Resistencia rota", "Resistencia $${resP ? resP.toFixed(2) : '?'} rota — evaluar LONG")
alertcondition(close <= bEntry and close > bSL, "✅ Zona LONG",   "Precio en zona de entrada LONG — confirmar")
alertcondition(close >= sEntry and close < sSL, "✅ Zona SHORT",  "Precio en zona de entrada SHORT — confirmar")
`;
}

// ═══════════════════════════════════════════════════════════════════
// FULL ANALYSIS ENGINE
// ═══════════════════════════════════════════════════════════════════

function runAnalysis(candles, symbol, tf) {
  if (candles.length < 50) throw new Error('Datos insuficientes');

  const closes  = candles.map(c => c.close);
  const n       = candles.length - 1;

  // Indicators
  const ema20   = calcEMA(closes, 20);
  const ema50   = calcEMA(closes, 50);
  const ema200  = calcEMA(closes, 200);
  const rsi     = calcRSI(closes, 14);
  const macd    = calcMACD(closes);
  const bb      = calcBB(closes, 20, 2);
  const atr     = calcATR(candles, 14);
  const stoch   = calcStoch(candles);

  const price   = closes[n];
  const rsiNow  = rsi[n];
  const macdNow = macd.macd[n];
  const sigNow  = macd.signal[n];
  const histNow = macd.hist[n];
  const bbNow   = bb[n];
  const atrNow  = atr[n];
  const stochK  = (stoch.k.filter(v => v !== null)).slice(-1)[0] || 50;

  // Analysis modules
  const trend    = detectTrend(candles, ema20, ema50, ema200);
  const patterns = detectChartPatterns(candles);
  const smc      = detectSMC(candles);
  const div      = detectRSIDivergence(candles, rsi);
  const volume   = analyzeVolume(candles);
  const levels   = supportResistance(candles);

  // ── Scoring ────────────────────────────────────────────────────
  let score = 50; // neutral start
  const signals = [];

  // Trend (±20)
  const trendBull = trend.score >= 4;
  const trendBear = trend.score <= 2;
  if (trendBull)  { score += 15; signals.push({ label: trend.trend, type: 'bullish', pts: 15 }); }
  if (trendBear)  { score -= 15; signals.push({ label: trend.trend, type: 'bearish', pts: -15 }); }

  // EMA cross (±10)
  if (trend.goldCross)  { score += 10; signals.push({ label: 'Golden Cross EMA20/50', type: 'bullish', pts: 10 }); }
  if (trend.deathCross) { score -= 10; signals.push({ label: 'Death Cross EMA20/50',  type: 'bearish', pts: -10 }); }

  // RSI (±12)
  if (rsiNow !== null) {
    if      (rsiNow > 70) { score -= 10; signals.push({ label: `RSI Sobrecomprado ${rsiNow.toFixed(0)}`, type: 'bearish', pts: -10 }); }
    else if (rsiNow < 30) { score += 10; signals.push({ label: `RSI Sobrevendido ${rsiNow.toFixed(0)}`,  type: 'bullish', pts: 10 }); }
    else if (rsiNow > 55) { score += 4; }
    else if (rsiNow < 45) { score -= 4; }
  }

  // RSI Divergence (±15)
  if (div.type === 'bullish') { score += 12; signals.push({ label: div.label, type: 'bullish', pts: 12 }); }
  if (div.type === 'bearish') { score -= 12; signals.push({ label: div.label, type: 'bearish', pts: -12 }); }

  // MACD (±10)
  if (macdNow !== null && sigNow !== null) {
    if (macdNow > sigNow && macd.hist[n] > macd.hist[n-1]) { score += 8;  signals.push({ label: 'MACD Cruce Alcista', type: 'bullish', pts: 8 }); }
    if (macdNow < sigNow && macd.hist[n] < macd.hist[n-1]) { score -= 8;  signals.push({ label: 'MACD Cruce Bajista', type: 'bearish', pts: -8 }); }
  }

  // Bollinger Bands (±8)
  if (bbNow.lower && bbNow.upper) {
    if (price < bbNow.lower) { score += 8;  signals.push({ label: 'Precio bajo Banda Inferior BB', type: 'bullish', pts: 8 }); }
    if (price > bbNow.upper) { score -= 8;  signals.push({ label: 'Precio sobre Banda Superior BB', type: 'bearish', pts: -8 }); }
    if (bbNow.width < 0.03)  { score += 3;  signals.push({ label: 'BB Squeeze — ruptura inminente', type: 'neutral', pts: 3 }); }
  }

  // Volume (±5)
  if (volume.spike && volume.bias === 'buyPressure')   { score += 5; signals.push({ label: 'Spike volumen comprador', type: 'bullish', pts: 5 }); }
  if (volume.spike && volume.bias === 'sellPressure')  { score -= 5; signals.push({ label: 'Spike volumen vendedor',  type: 'bearish', pts: -5 }); }

  // SMC (±10)
  if (smc.zone.type === 'bullish') { score += 5; signals.push({ label: smc.zone.label, type: 'bullish', pts: 5 }); }
  if (smc.zone.type === 'bearish') { score -= 5; signals.push({ label: smc.zone.label, type: 'bearish', pts: -5 }); }
  if (smc.bos?.type === 'bullish_bos') { score += 7; signals.push({ label: smc.bos.label, type: 'bullish', pts: 7 }); }
  if (smc.bos?.type === 'bearish_bos') { score -= 7; signals.push({ label: smc.bos.label, type: 'bearish', pts: -7 }); }
  if (smc.bos?.type === 'choch')       { score += smc.bos.label.includes('Alcista') ? 4 : -4; }

  // Patterns (±8 for top pattern)
  const bestPat = patterns[0];
  if (bestPat) {
    const pts = bestPat.type === 'bullish' ? 8 : bestPat.type === 'bearish' ? -8 : 0;
    score += pts;
    signals.push({ label: `${bestPat.name} (${bestPat.probability}%)`, type: bestPat.type, pts });
  }

  score = Math.max(0, Math.min(100, Math.round(score)));

  // Signal label
  let signal, signalCls;
  if      (score >= 75) { signal = 'COMPRA FUERTE';  signalCls = 'sig-strong-buy'; }
  else if (score >= 60) { signal = 'COMPRAR';         signalCls = 'sig-buy'; }
  else if (score >= 45) { signal = 'NEUTRAL';          signalCls = 'sig-neutral'; }
  else if (score >= 30) { signal = 'VENDER';           signalCls = 'sig-sell'; }
  else                  { signal = 'VENTA FUERTE';    signalCls = 'sig-strong-sell'; }

  // Key levels closest to price
  const support    = levels.filter(l => l.price < price && l.type === 'support').slice(-1)[0];
  const resistance = levels.filter(l => l.price > price && l.type === 'resistance').slice(0, 1)[0];

  // Risk/Reward suggestion
  const atrVal  = atrNow || price * 0.01;
  const sl      = score >= 50 ? price - atrVal * 1.5 : price + atrVal * 1.5;
  const tp      = score >= 50 ? price + atrVal * 3   : price - atrVal * 3;
  const rr      = Math.abs(tp - price) / Math.abs(sl - price);

  // ── Verdict & projection ─────────────────────────────────────────
  const verdict = computeVerdict({
    score, trend, patterns, smc, div, volume,
    indicators: { rsi: rsiNow, hist: histNow },
    price,
  });

  const brain = computeTradingBrain({
    score, trend, patterns, smc, div, volume, signals,
    indicators: { rsi: rsiNow, macd: macdNow, hist: histNow, bb: bbNow, atr: atrNow },
    price, symbol, tf, support, resistance,
    suggestedEntry: price, suggestedSL: sl, suggestedTP: tp,
  });

  return {
    symbol, tf, price, score, signal, signalCls,
    trend, patterns, smc, div, volume, levels,
    indicators: {
      rsi: rsiNow, macd: macdNow, signal_macd: sigNow, hist: histNow,
      ema20: ema20[n], ema50: ema50[n], ema200: ema200[n],
      bb: bbNow, atr: atrNow, stochK,
    },
    signals: signals.sort((a, b) => Math.abs(b.pts) - Math.abs(a.pts)),
    support, resistance,
    suggestedEntry: price,
    suggestedSL:    sl,
    suggestedTP:    tp,
    riskReward:     rr,
    verdict, brain,
    ts: Date.now(),
  };
}

// ═══════════════════════════════════════════════════════════════════
// FORMAT HELPERS
// ═══════════════════════════════════════════════════════════════════

function fmtPrice(n) {
  if (!n || isNaN(n)) return '—';
  if (n >= 10000) return n.toLocaleString('en', { maximumFractionDigits: 0 });
  if (n >= 100)   return n.toFixed(2);
  if (n >= 1)     return n.toFixed(4);
  return n.toFixed(6);
}

function fmtPct(n, digits = 2) {
  if (n == null || isNaN(n)) return '—';
  return (n >= 0 ? '+' : '') + n.toFixed(digits) + '%';
}

function fmtVol(n) {
  if (!n) return '—';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toFixed(0);
}

function scoreColor(s) {
  if (s >= 65) return '#00ff41';
  if (s >= 50) return '#e3b341';
  return '#f85149';
}

function typeColor(t) {
  if (t === 'bullish') return '#00ff41';
  if (t === 'bearish') return '#f85149';
  return '#e3b341';
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = String(s || '');
  return d.innerHTML;
}

// ═══════════════════════════════════════════════════════════════════
// RENDER ANALYSIS
// ═══════════════════════════════════════════════════════════════════

function renderAnalysis(a) {
  if (!a) return;
  lastAnalysis = a;

  // Header
  document.getElementById('aSymbol').textContent  = a.symbol;
  document.getElementById('aTf').textContent       = a.tf;
  document.getElementById('aPrice').textContent    = '$' + fmtPrice(a.price);
  document.getElementById('aLastUpdate').textContent = new Date(a.ts).toLocaleTimeString('es', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  // Score gauge
  const scoreEl = document.getElementById('aScore');
  const barEl   = document.getElementById('aScoreBar');
  const sigEl   = document.getElementById('aSignal');
  scoreEl.textContent = a.score;
  scoreEl.style.color = scoreColor(a.score);
  barEl.style.width   = a.score + '%';
  barEl.style.background = scoreColor(a.score);
  sigEl.textContent   = a.signal;
  sigEl.className     = 'a-signal ' + a.signalCls;

  // Verdict
  const verdEl = document.getElementById('aVerdict');
  if (verdEl && a.verdict) {
    const v = a.verdict;
    const mainCol  = v.cls.includes('bull') ? '#00ff41' : v.cls.includes('bear') ? '#f85149' : '#e3b341';
    const bullW    = v.bullPct;
    const bearW    = v.bearPct;
    verdEl.innerHTML = `
      <div class="verd-header">
        <div class="verd-label ${v.cls}">${v.label}</div>
        <div class="verd-conf">Confianza <span style="color:${mainCol};font-weight:900">${v.confidence}%</span></div>
      </div>
      <div class="verd-prob-bar">
        <div class="verd-pb-bull" style="width:${bullW}%"></div>
        <div class="verd-pb-bear" style="width:${bearW}%"></div>
      </div>
      <div class="verd-prob-labels">
        <span style="color:#00ff41">▲ Alcista ${bullW}%</span>
        <span style="color:#f85149">▼ Bajista ${bearW}%</span>
      </div>
      <div class="verd-reasons">
        ${v.bullReasons.slice(0,4).map(r => `<div class="verd-reason verd-r-bull">▲ ${escHtml(r)}</div>`).join('')}
        ${v.bearReasons.slice(0,4).map(r => `<div class="verd-reason verd-r-bear">▼ ${escHtml(r)}</div>`).join('')}
      </div>
    `;
  }

  // Brain
  const brainEl = document.getElementById('aBrain');
  if (brainEl && a.brain) {
    const adv = window._lastAdvanced;
    if (adv) {
      // Preservar datos avanzados (MTF/harmónico/Elliott/ML) ya calculados
      // El setInterval 1s actualiza precio pero NO debe resetear el análisis avanzado
      renderBrain(brainEl, { ...a.brain, ...adv });
    } else {
      // Primera carga: mostrar placeholder hasta que runAdvancedBrain termine
      renderBrain(brainEl, { ...a.brain, _loading: true });
    }
  }

  // Trend
  const trendEl = document.getElementById('aTrend');
  if (trendEl) {
    const t = a.trend;
    const col = t.score >= 4 ? '#00ff41' : t.score <= 2 ? '#f85149' : '#e3b341';
    trendEl.innerHTML = `
      <div class="tv-row"><span class="tv-k">Estructura</span><span class="tv-v" style="color:${col}">${t.trend}</span></div>
      <div class="tv-row"><span class="tv-k">EMA 20</span><span class="tv-v">$${fmtPrice(t.ema20)}</span></div>
      <div class="tv-row"><span class="tv-k">EMA 50</span><span class="tv-v">$${fmtPrice(t.ema50)}</span></div>
      <div class="tv-row"><span class="tv-k">EMA 200</span><span class="tv-v">$${fmtPrice(t.ema200)}</span></div>
      ${t.priceVsEma200 ? `<div class="tv-row"><span class="tv-k">vs EMA200</span><span class="tv-v" style="color:${parseFloat(t.priceVsEma200)>=0?'#00ff41':'#f85149'}">${fmtPct(parseFloat(t.priceVsEma200))}</span></div>` : ''}
      ${t.goldCross  ? '<div class="tv-badge tv-bull">● GOLDEN CROSS</div>' : ''}
      ${t.deathCross ? '<div class="tv-badge tv-bear">● DEATH CROSS</div>' : ''}
    `;
  }

  // Indicators
  const indEl = document.getElementById('aIndicators');
  if (indEl) {
    const ind = a.indicators;
    const rsiCol = ind.rsi > 70 ? '#f85149' : ind.rsi < 30 ? '#00ff41' : '#e3b341';
    const bbPos  = ind.bb?.lower && ind.bb?.upper
      ? ((a.price - ind.bb.lower) / (ind.bb.upper - ind.bb.lower) * 100).toFixed(0) : null;
    indEl.innerHTML = `
      <div class="tv-ind-grid">
        <div class="tv-ind-card">
          <div class="tv-ind-lbl">RSI (14)</div>
          <div class="tv-ind-val" style="color:${rsiCol}">${ind.rsi != null ? ind.rsi.toFixed(1) : '—'}</div>
          <div class="tv-ind-sub">${ind.rsi > 70 ? 'Sobrecomprado' : ind.rsi < 30 ? 'Sobrevendido' : 'Neutral'}</div>
        </div>
        <div class="tv-ind-card">
          <div class="tv-ind-lbl">MACD</div>
          <div class="tv-ind-val" style="color:${ind.hist > 0 ? '#00ff41' : '#f85149'}">${ind.hist != null ? ind.hist.toFixed(4) : '—'}</div>
          <div class="tv-ind-sub">${ind.hist > 0 ? 'Histograma ▲' : 'Histograma ▼'}</div>
        </div>
        <div class="tv-ind-card">
          <div class="tv-ind-lbl">Stoch K</div>
          <div class="tv-ind-val" style="color:${ind.stochK > 80 ? '#f85149' : ind.stochK < 20 ? '#00ff41' : '#e3b341'}">${ind.stochK ? ind.stochK.toFixed(0) : '—'}</div>
          <div class="tv-ind-sub">${ind.stochK > 80 ? 'Sobrecomprado' : ind.stochK < 20 ? 'Sobrevendido' : 'Neutral'}</div>
        </div>
        <div class="tv-ind-card">
          <div class="tv-ind-lbl">BB Width</div>
          <div class="tv-ind-val" style="color:${ind.bb?.width < 0.03 ? '#00d4ff' : '#aaa'}">${ind.bb?.width != null ? (ind.bb.width * 100).toFixed(2) + '%' : '—'}</div>
          <div class="tv-ind-sub">${ind.bb?.width < 0.03 ? '⚡ Squeeze' : 'Normal'}</div>
        </div>
        <div class="tv-ind-card">
          <div class="tv-ind-lbl">ATR (14)</div>
          <div class="tv-ind-val">$${fmtPrice(ind.atr)}</div>
          <div class="tv-ind-sub">${ind.atr ? (ind.atr / a.price * 100).toFixed(2) + '% del precio' : '—'}</div>
        </div>
        <div class="tv-ind-card">
          <div class="tv-ind-lbl">BB Posición</div>
          <div class="tv-ind-val" style="color:${bbPos > 80 ? '#f85149' : bbPos < 20 ? '#00ff41' : '#aaa'}">${bbPos != null ? bbPos + '%' : '—'}</div>
          <div class="tv-ind-sub">${bbPos > 80 ? 'Banda superior' : bbPos < 20 ? 'Banda inferior' : 'Zona media'}</div>
        </div>
      </div>
      ${a.div.type ? `<div class="tv-div-alert tv-div-${a.div.type}">⚡ ${escHtml(a.div.label)}<div class="tv-div-strength">Fuerza: ${a.div.strength.toFixed(0)}%</div></div>` : ''}
    `;
  }

  // Chart Patterns
  const patEl = document.getElementById('aPatterns');
  if (patEl) {
    if (!a.patterns.length) {
      patEl.innerHTML = '<div class="tv-empty">Sin patrones claros detectados en este timeframe</div>';
    } else {
      patEl.innerHTML = a.patterns.map(p => `
        <div class="tv-pattern tv-pat-${p.type}">
          <div class="tv-pat-top">
            <span class="tv-pat-name">${escHtml(p.name)}</span>
            <span class="tv-pat-prob" style="color:${typeColor(p.type)}">${p.probability}%</span>
            <span class="tv-pat-badge tv-pat-${p.type}">${p.type === 'bullish' ? '▲ ALCISTA' : p.type === 'bearish' ? '▼ BAJISTA' : '◆ NEUTRAL'}</span>
          </div>
          <div class="tv-pat-desc">${escHtml(p.description)}</div>
          ${p.target ? `<div class="tv-pat-target">Objetivo: <b>$${fmtPrice(p.target)}</b> · Neckline: $${fmtPrice(p.neckline)}</div>` : ''}
          <div class="tv-pat-prob-bar"><div class="tv-pat-prob-fill" style="width:${p.probability}%;background:${typeColor(p.type)}"></div></div>
        </div>
      `).join('');
    }
  }

  // SMC
  const smcEl = document.getElementById('aSMC');
  if (smcEl) {
    const s = a.smc;
    smcEl.innerHTML = `
      <div class="tv-smc-zone tv-smc-${s.zone.type}">
        <span class="tv-smc-zone-icon">${s.zone.type === 'bullish' ? '▼ DESCUENTO' : '▲ PREMIUM'}</span>
        <span class="tv-smc-zone-pct">${s.zone.pct}% del rango</span>
      </div>
      ${s.bos ? `<div class="tv-bos tv-bos-${s.bos.type.includes('bull') ? 'bull' : s.bos.type.includes('bear') ? 'bear' : 'choch'}">
        <span class="tv-bos-lbl">${escHtml(s.bos.label)}</span>
      </div>` : ''}
      ${s.fvgs.length ? `<div class="tv-smc-section-title">Fair Value Gaps</div>
        ${s.fvgs.map(g => `<div class="tv-fvg tv-fvg-${g.type}">
          <span>${escHtml(g.label)}</span>
        </div>`).join('')}` : ''}
      ${s.orderBlocks.length ? `<div class="tv-smc-section-title">Order Blocks activos</div>
        ${s.orderBlocks.map(ob => `<div class="tv-ob tv-ob-${ob.type}">
          <span>${escHtml(ob.label)}</span>
        </div>`).join('')}` : ''}
      <div class="tv-smc-section-title">Liquidez</div>
      <div class="tv-liq-row">
        ${s.swingHighs.map(h => `<span class="tv-liq tv-liq-high">⚡ $${fmtPrice(h)}</span>`).join('')}
        ${s.swingLows.map(l  => `<span class="tv-liq tv-liq-low">⚡ $${fmtPrice(l)}</span>`).join('')}
      </div>
    `;
  }

  // Signals breakdown
  const sigListEl = document.getElementById('aSignals');
  if (sigListEl) {
    sigListEl.innerHTML = a.signals.slice(0, 8).map(s => {
      const col = s.type === 'bullish' ? '#00ff41' : s.type === 'bearish' ? '#f85149' : '#e3b341';
      const pts = s.pts > 0 ? '+' + s.pts : s.pts;
      return `<div class="tv-sig-row">
        <span class="tv-sig-dot" style="background:${col}"></span>
        <span class="tv-sig-label">${escHtml(s.label)}</span>
        <span class="tv-sig-pts" style="color:${col}">${pts} pts</span>
      </div>`;
    }).join('');
  }

  // Key levels + R/R
  const levelsEl = document.getElementById('aLevels');
  if (levelsEl) {
    levelsEl.innerHTML = `
      <div class="tv-rr-row">
        <div class="tv-rr-card tv-rr-entry"><div class="tv-rr-lbl">ENTRADA</div><div class="tv-rr-val">$${fmtPrice(a.suggestedEntry)}</div></div>
        <div class="tv-rr-card tv-rr-sl"><div class="tv-rr-lbl">STOP LOSS</div><div class="tv-rr-val">$${fmtPrice(a.suggestedSL)}</div></div>
        <div class="tv-rr-card tv-rr-tp"><div class="tv-rr-lbl">TAKE PROFIT</div><div class="tv-rr-val">$${fmtPrice(a.suggestedTP)}</div></div>
        <div class="tv-rr-card tv-rr-rr"><div class="tv-rr-lbl">R:R</div><div class="tv-rr-val" style="color:${a.riskReward>=2?'#00ff41':'#e3b341'}">${a.riskReward.toFixed(1)}</div></div>
      </div>
      <div class="tv-levels-list">
        ${a.resistance ? `<div class="tv-level tv-res">
          <span class="tv-level-type">RESISTENCIA</span>
          <span class="tv-level-price">$${fmtPrice(a.resistance.price)}</span>
          <span class="tv-level-dist">${fmtPct((a.resistance.price - a.price)/a.price*100)}</span>
          <button class="btn-tv-line" data-price="${a.resistance.price}" data-label="Resistencia" data-color="#f5a623">→TV</button>
        </div>` : ''}
        <div class="tv-level tv-cur">
          <span class="tv-level-type">PRECIO</span>
          <span class="tv-level-price">$${fmtPrice(a.price)}</span>
          <span class="tv-level-dist">—</span>
          <button class="btn-tv-line" data-price="${a.price}" data-label="Precio" data-color="#ffffff">→TV</button>
        </div>
        ${a.support ? `<div class="tv-level tv-sup">
          <span class="tv-level-type">SOPORTE</span>
          <span class="tv-level-price">$${fmtPrice(a.support.price)}</span>
          <span class="tv-level-dist">${fmtPct((a.support.price - a.price)/a.price*100)}</span>
          <button class="btn-tv-line" data-price="${a.support.price}" data-label="Soporte" data-color="#00d4ff">→TV</button>
        </div>` : ''}
        <div class="tv-level tv-rr-row2">
          <button class="btn-tv-line btn-tv-tp" data-price="${a.suggestedTP}" data-label="TP" data-color="#00ff41">TP →TV</button>
          <button class="btn-tv-line btn-tv-sl" data-price="${a.suggestedSL}" data-label="SL" data-color="#f85149">SL →TV</button>
          <button class="btn-tv-all" id="drawLevelsBtn2">Todos →TV</button>
        </div>
      </div>
    `;
  }

  // Volume
  const volEl = document.getElementById('aVolume');
  if (volEl) {
    const v = a.volume;
    const col = v.bias === 'buyPressure' ? '#00ff41' : '#f85149';
    volEl.innerHTML = `
      <div class="tv-row"><span class="tv-k">Vol actual</span><span class="tv-v">${fmtVol(v.current)}</span></div>
      <div class="tv-row"><span class="tv-k">Vol promedio 20</span><span class="tv-v">${fmtVol(v.avg20)}</span></div>
      <div class="tv-row"><span class="tv-k">Ratio</span><span class="tv-v" style="color:${v.spike?'#e3b341':'#aaa'}">${v.ratio.toFixed(2)}x ${v.spike?'⚡':''}</span></div>
      <div class="tv-row"><span class="tv-k">CVD</span><span class="tv-v" style="color:${col}">${v.cvd > 0 ? '▲ Compradores' : '▼ Vendedores'}</span></div>
      <div class="tv-row"><span class="tv-k">Estado</span><span class="tv-v">${escHtml(v.label)}</span></div>
    `;
  }
}

// ═══════════════════════════════════════════════════════════════════
// DRAW LEVELS ON TRADINGVIEW CHART
// ═══════════════════════════════════════════════════════════════════

function sendToTV(levels) {
  if (!tvTabId) { showNotif('Abre TradingView en la pestaña activa', 'error'); return; }
  if (!candles.length) { showNotif('Analiza primero', 'error'); return; }

  // Pass a visible price range so the overlay can calibrate Y positions
  const slice    = candles.slice(-80);
  const visHigh  = Math.max(...slice.map(c => c.high));
  const visLow   = Math.min(...slice.map(c => c.low));

  chrome.tabs.sendMessage(tvTabId, {
    type: 'DRAW_LEVELS',
    levels,
    currentPrice,
    visibleHigh: visHigh,
    visibleLow:  visLow,
  }, () => {
    void chrome.runtime.lastError;
    showNotif(`${levels.length === 1 ? levels[0].label : 'Niveles'} dibujado en TV ✓`);
  });
}

function drawLevels() {
  if (!lastAnalysis) { showNotif('Analiza primero', 'error'); return; }
  const a = lastAnalysis;
  const levels = [];
  if (a.price)       levels.push({ price: +a.price.toFixed(8),            label: 'Precio',      color: '#ffffff' });
  if (a.suggestedTP) levels.push({ price: +a.suggestedTP.toFixed(8),      label: 'TP',          color: '#00ff41' });
  if (a.suggestedSL) levels.push({ price: +a.suggestedSL.toFixed(8),      label: 'SL',          color: '#f85149' });
  if (a.resistance)  levels.push({ price: +a.resistance.price.toFixed(8), label: 'Resistencia', color: '#f5a623' });
  if (a.support)     levels.push({ price: +a.support.price.toFixed(8),    label: 'Soporte',     color: '#00d4ff' });
  sendToTV(levels);
}

// ═══════════════════════════════════════════════════════════════════
// RENDER BRAIN
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// ML ANALYST — interpreta todos los datos y genera texto en lenguaje natural
// ═══════════════════════════════════════════════════════════════════

function generateMLAnalysis(b) {
  const ml   = b.mlResult;
  const adv  = b.advProb;
  const mtf  = b.mtf;
  const harm = b.harmonics?.[0];
  const ew   = b.elliott;

  if (!ml && !adv) return null;

  // ── Dirección y probabilidad combinada ──────────────────────────────
  const bullPct = adv?.bullPct ?? ml?.bullPct ?? 50;
  const bearPct = adv?.bearPct ?? ml?.bearPct ?? 50;
  const conf    = adv?.conf ?? 50;
  const direction = bullPct >= 58 ? 'ALCISTA' : bearPct >= 58 ? 'BAJISTA' : 'NEUTRAL';
  const dirCol    = direction === 'ALCISTA' ? '#00ff41' : direction === 'BAJISTA' ? '#f85149' : '#e3b341';
  const mainPct   = Math.max(bullPct, bearPct);

  // ── Interpretación del ML ───────────────────────────────────────────
  const mlLines = [];
  if (ml) {
    const mlDir = ml.bullPct >= 58 ? 'alcista' : ml.bearPct >= 58 ? 'bajista' : 'neutral';
    const mlStr = ml.bullPct >= 58 ? 'favorece el LONG' : ml.bearPct >= 58 ? 'favorece el SHORT' : 'no tiene sesgo claro (neutral)';
    mlLines.push(`El modelo ML (${ml.bullPct}% alcista) ${mlStr}.`);

    if (ml.importance) {
      const top = Object.entries(ml.importance)
        .sort((a, c) => Math.abs(c[1]) - Math.abs(a[1]))
        .slice(0, 3);
      const featDesc = {
        rsi_norm:    v => v < 0 ? 'RSI débil — vendedores dominan' : 'RSI fuerte — compradores activos',
        bb_pos:      v => v < 0 ? 'precio en zona baja BB' : 'precio en zona alta BB',
        bull_candle: v => v < 0 ? 'vela bajista — presión vendedora' : 'vela alcista — presión compradora',
        ema_bear:    v => v > 0 ? 'EMAs alineadas bajistas' : 'EMAs no bajistas',
        ema_bull:    v => v > 0 ? 'EMAs alineadas alcistas' : 'EMAs no alcistas',
        rsi_slope:   v => v < 0 ? 'RSI cayendo — momentum bajista' : 'RSI subiendo — momentum alcista',
        vol_ratio:   v => v > 0 ? 'volumen elevado — interés institucional' : 'volumen bajo',
        macd_bull:   v => v > 0 ? 'MACD positivo — impulso alcista' : 'MACD negativo',
        above_ema200:v => v > 0 ? 'precio sobre EMA200 — tendencia alcista de fondo' : 'precio bajo EMA200',
        stoch_os:    v => v > 0 ? 'Stoch sobrevendido — rebote posible' : '',
        stoch_ob:    v => v > 0 ? 'Stoch sobrecomprado — cuidado con corrección' : '',
      };
      const descs = top.map(([k, v]) => featDesc[k] ? featDesc[k](v) : k.replace(/_/g,' ')).filter(Boolean);
      if (descs.length) mlLines.push(`Drivers del modelo: ${descs.join('; ')}.`);
    }
  }

  // ── MTF ─────────────────────────────────────────────────────────────
  const mtfLines = [];
  if (mtf) {
    const mtfMap = {
      'ALCISTA ↑↑':  'Todos los marcos temporales alineados alcistas — mayor probabilidad de subida sostenida.',
      'BAJISTA ↓↓':  'Todos los marcos temporales alineados bajistas — presión vendedora en todas las escalas.',
      'ALCISTA ↑~':  'Marcos superiores alcistas pero con divergencia — tendencia alcista sin confirmación plena.',
      'BAJISTA ↓~':  'Marcos superiores bajistas con algo de ruido — sesgo bajista moderado.',
      'DIVERGENTE ↕': 'Marcos temporales en conflicto — señal reducida, esperar alineación antes de entrar.',
    };
    mtfLines.push(mtfMap[mtf.align] || `MTF: ${mtf.align}`);
    if (mtf.prim?.rsi) {
      const rsiTxt = mtf.prim.rsi > 60 ? 'sobrecomprado' : mtf.prim.rsi < 40 ? 'sobrevendido' : 'neutral';
      mtfLines.push(`TF primario (${mtf.primaryTf}): RSI ${mtf.prim.rsi} — ${rsiTxt}.`);
    }
  }

  // ── Harmonic ────────────────────────────────────────────────────────
  const harmLines = [];
  if (harm) {
    const hDir = harm.type === 'bullish' ? 'alcista (señal de compra)' : 'bajista (señal de venta)';
    if (harm.stage === 'COMPLETANDO') {
      harmLines.push(`⚠ Patrón armónico ${harm.name} ${hDir} COMPLETÁNDOSE — zona PRZ $${fmtPrice(harm.dZone?.low)}–$${fmtPrice(harm.dZone?.high)}. Alta probabilidad de reversión inmediata.`);
    } else if (harm.stage === 'APPROACHANDO PRZ') {
      harmLines.push(`Patrón ${harm.name} ${hDir} acercándose al PRZ (${harm.distPct}% de distancia). Si el precio llega a la zona, esperar señal de inversión.`);
    } else {
      harmLines.push(`${harm.name} ${hDir} en formación — todavía lejos del PRZ.`);
    }
  }

  // ── Elliott ─────────────────────────────────────────────────────────
  const ewLines = [];
  if (ew && ew.currentWave) {
    const ewMap = {
      'W3': w => `Onda ${w} activa — ola más poderosa del ciclo Elliott. ${ew.dir === 'bullish' ? 'Impulso alcista fuerte, no ir en contra.' : 'Caída acelerada, zona de alto riesgo para longs.'}`,
      'W5': w => `Onda ${w} — ola final del impulso. Atención: tras W5 viene una corrección ABC significativa.`,
      'W4': w => `Onda ${w} — corrección en marcha. Esperar fin de corrección antes de entrar en la dirección del impulso.`,
      'W2': w => `Onda ${w} — primer retroceso después del impulso inicial. Zona de entrada con buen R:R si se mantiene por encima de W1.`,
    };
    const wNum = ew.currentWave.match(/W(\d)/)?.[1];
    const desc = wNum && ewMap[`W${wNum}`] ? ewMap[`W${wNum}`](ew.currentWave) : `${ew.currentWave} detectada.`;
    ewLines.push(desc);
    if (ew.nextTarget) ewLines.push(`Objetivo Elliott: $${fmtPrice(ew.nextTarget)} con conf. ${ew.confidence}%.`);
  }

  // ── Escenarios ──────────────────────────────────────────────────────
  const mainCond = bullPct > bearPct
    ? (b.bullConditions?.[0] || 'mantiene estructura alcista')
    : (b.bearConditions?.[0] || 'rompe soporte clave');
  const altPct = Math.min(bullPct, bearPct);
  const mainLabel = bullPct > bearPct ? `LONG ${bullPct}%` : `SHORT ${bearPct}%`;
  const altLabel  = bullPct > bearPct ? `SHORT ${bearPct}%` : `LONG ${bullPct}%`;
  const altCond   = bullPct > bearPct
    ? 'MTF gira bajista o ML cae bajo 40%'
    : 'MTF gira alcista o precio recupera soporte';

  // ── Recomendación de acción ──────────────────────────────────────────
  let action;
  if (conf >= 75 && direction !== 'NEUTRAL') {
    action = `Alta confianza — el setup está ${direction === 'ALCISTA' ? 'favorable para LONG' : 'favorable para SHORT'} pero confirmar con ruptura/volumen.`;
  } else if (direction === 'NEUTRAL' || conf < 50) {
    action = 'Zona de indecisión — mantener posición flat hasta que algún factor confirme dirección.';
  } else {
    action = `Sesgo ${direction.toLowerCase()} con confianza moderada (${conf}%). Usar R:R mínimo 2:1 y stop ajustado.`;
  }

  return { direction, dirCol, mainPct, conf, mlLines, mtfLines, harmLines, ewLines, mainLabel, altLabel, mainCond, altCond, altPct, action };
}

function renderBrain(el, b) {
  // ── Fix race condition: calcular ML aquí mismo si el modelo ya está disponible ──
  // Esto elimina por completo la race condition entre mlLoadModel y runAdvancedBrain
  let mlResult = b.mlResult;
  if (!mlResult && _mlModel && candles.length >= 200) {
    try { mlResult = mlInfer(candles); } catch {}
  }
  b = { ...b, mlResult };

  // Si tenemos ML pero el advProb fue calculado sin él, recalcular
  if (mlResult && b.advProb && !b.advProb.factors?.some(f => f.includes('ML'))) {
    try {
      b.advProb = computeAdvancedProb({
        score: b.bullPct, mlResult,
        mtf: b.mtf, harmonics: b.harmonics, elliott: b.elliott,
        div: null, volume: null, patterns: null,
      });
    } catch {}
  }

  // ── MTF panel ────────────────────────────────────────────────────
  const mtfHtml = b.mtf ? (() => {
    const m = b.mtf;
    const dirIcon = d => d?.includes('bullish')||d==='neutral_up' ? '▲' : d?.includes('bearish')||d==='neutral_down' ? '▼' : '→';
    const dirCol  = d => d?.includes('bullish')||d==='neutral_up' ? '#00ff41' : d?.includes('bearish')||d==='neutral_down' ? '#f85149' : '#e3b341';
    const alignCol = a => a?.includes('ALCISTA') ? '#00ff41' : a?.includes('BAJISTA') ? '#f85149' : '#e3b341';
    return `<div class="brain-mtf">
      <div class="brain-mtf-title">
        📊 MULTI-TIMEFRAME · <span style="color:${alignCol(m.align)}">${m.align}</span>
        ${m._localOnly ? '<span class="brain-mtf-local">(estimado · sin fetch externo)</span>' : ''}
      </div>
      <div class="brain-mtf-row">
        <div class="brain-mtf-cell">
          <span class="brain-mtf-tf">${m.primaryTf} (primaria)</span>
          <span class="brain-mtf-dir" style="color:${dirCol(m.prim?.dir)}">${dirIcon(m.prim?.dir)} ${(m.prim?.dir||'—').replace('_',' ')}</span>
          <span class="brain-mtf-rsi">RSI ${m.prim?.rsi||'—'}</span>
          ${m.prim?.res ? `<span class="brain-mtf-res">R $${fmtPrice(m.prim.res)}</span>` : ''}
          ${m.prim?.sup ? `<span class="brain-mtf-sup">S $${fmtPrice(m.prim.sup)}</span>` : ''}
        </div>
        <div class="brain-mtf-cell">
          <span class="brain-mtf-tf">${m.secondaryTf} (secundaria)</span>
          <span class="brain-mtf-dir" style="color:${dirCol(m.sec?.dir)}">${dirIcon(m.sec?.dir)} ${(m.sec?.dir||'—').replace('_',' ')}</span>
          <span class="brain-mtf-rsi">RSI ${m.sec?.rsi||'—'}</span>
          ${m.sec?.res ? `<span class="brain-mtf-res">R $${fmtPrice(m.sec.res)}</span>` : ''}
          ${m.sec?.sup ? `<span class="brain-mtf-sup">S $${fmtPrice(m.sec.sup)}</span>` : ''}
        </div>
        <div class="brain-mtf-cell brain-mtf-current">
          <span class="brain-mtf-tf">${m.currentTf} (actual)</span>
          <span class="brain-mtf-dir" style="color:${b.modeCol}">${b.modeIcon}</span>
          <span class="brain-mtf-rsi">Score ${b.finalConf||'—'}%</span>
          ${lastAnalysis?.resistance?.price ? `<span class="brain-mtf-res">R $${fmtPrice(lastAnalysis.resistance.price)}</span>` : ''}
          ${lastAnalysis?.support?.price ? `<span class="brain-mtf-sup">S $${fmtPrice(lastAnalysis.support.price)}</span>` : ''}
        </div>
      </div>
    </div>`;
  })() : '';

  // ── Harmonic patterns panel ───────────────────────────────────────
  const harmonicHtml = b.harmonics?.length ? `
    <div class="brain-harmonic">
      <div class="brain-harmonic-title">🔺 PATRONES ARMÓNICOS</div>
      ${b.harmonics.map(h => {
        const col = h.type === 'bullish' ? '#00ff41' : '#f85149';
        const stageCol = h.stage === 'COMPLETANDO' ? '#ff4500' : h.stage === 'APPROACHANDO PRZ' ? '#e3b341' : '#888';
        return `<div class="brain-harm-row">
          <div class="brain-harm-top">
            <span class="brain-harm-name" style="color:${col}">${h.name}</span>
            <span class="brain-harm-type">${h.type === 'bullish' ? '▲ ALCISTA' : '▼ BAJISTA'}</span>
            <span class="brain-harm-stage" style="color:${stageCol}">${h.stage}</span>
            <span class="brain-harm-prob" style="color:${col}">${h.prob}%</span>
          </div>
          <div class="brain-harm-details">
            <span class="brain-harm-d">PRZ: $${fmtPrice(h.dZone.low)}–$${fmtPrice(h.dZone.high)}</span>
            <span class="brain-harm-sl">SL: $${fmtPrice(h.sl)}</span>
            <span class="brain-harm-tp">TP1: $${fmtPrice(h.tp1)}</span>
            <span class="brain-harm-tp">TP2: $${fmtPrice(h.tp2)}</span>
            <span class="brain-harm-dist">${h.distPct}% del PRZ</span>
          </div>
          <div class="brain-harm-ratios">AB/XA ${h.ratios['AB/XA']} · BC/AB ${h.ratios['BC/AB']}</div>
        </div>`;
      }).join('')}
    </div>` : '';

  // ── Elliott Wave panel ────────────────────────────────────────────
  const elliottHtml = b.elliott ? (() => {
    const e = b.elliott;
    const col = e.dir === 'bullish' ? '#00ff41' : '#f85149';
    const waveNums = ['1','2','3','4','5'];
    const curWave = e.currentWave?.includes('W') ? e.currentWave.match(/W(\d)/)?.[1] : null;
    const waveDots = waveNums.map(n => {
      const active = curWave === n;
      return `<span class="brain-wave-dot ${active ? 'active' : ''}" style="${active ? `color:${col};font-weight:900` : ''}">${n}</span>`;
    }).join('<span class="brain-wave-sep">–</span>');
    return `<div class="brain-elliott">
      <div class="brain-elliott-title">🌊 ONDAS DE ELLIOTT · <span style="color:${col}">${e.type.toUpperCase()} ${e.dir === 'bullish' ? '▲' : '▼'}</span></div>
      <div class="brain-wave-track">${waveDots}<span class="brain-wave-label" style="color:${col}"> ${e.currentWave}</span></div>
      <div class="brain-elliott-next">${escHtml(e.nextMove)}</div>
      ${e.nextTarget ? `<div class="brain-elliott-target">Objetivo: <strong style="color:${col}">$${fmtPrice(e.nextTarget)}</strong></div>` : ''}
      <div class="brain-elliott-ratios">
        ${e.w2Ret ? `W2 ret. ${e.w2Ret}% · ` : ''}
        ${e.w3vsW1 ? `W3/W1 ${e.w3vsW1}x · ` : ''}
        ${e.w4Ret && parseFloat(e.w4Ret) < 100 ? `W4 ret. ${e.w4Ret}%` : ''}
        <span class="brain-elliott-conf">Conf. ${e.confidence}%</span>
      </div>
    </div>`;
  })() : `<div class="brain-elliott brain-elliott-none">
    <div class="brain-elliott-title" style="opacity:.5">🌊 ONDAS DE ELLIOTT</div>
    <div style="font-size:8.5px;color:var(--text2);padding:3px 0">Sin estructura de ondas clara en este TF/momento — el mercado puede estar en consolidación lateral</div>
  </div>`;

  // ── Advanced probability ──────────────────────────────────────────
  // ── ML Score block ────────────────────────────────────────────────
  mlUpdateHeaderBadge();
  const mlHtml = b.mlResult ? (() => {
    const ml = b.mlResult;
    const col = ml.bullPct >= 55 ? '#00ff41' : ml.bearPct >= 55 ? '#f85149' : '#e3b341';
    const topImp = b.mlResult.importance
      ? Object.entries(b.mlResult.importance)
          .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
          .slice(0, 5)
          .map(([k, v]) => `<span class="ml-feat ${v >= 0 ? 'ml-bull' : 'ml-bear'}">${v >= 0 ? '+' : ''}${v.toFixed(2)} ${k.replace(/_/g,' ')}</span>`)
          .join('')
      : '';
    return `<div class="brain-ml-block">
      <div class="brain-ml-title">🤖 ML Score <span class="brain-ml-auc">AUC ${ml.auc} · ${ml.samples?.toLocaleString() || '?'} muestras</span></div>
      <div class="brain-ml-bar-row">
        <span style="color:#00ff41;font-weight:900">▲ ${ml.bullPct}%</span>
        <div class="brain-prob-bar">
          <div class="brain-pb-bull" style="width:${ml.bullPct}%"></div>
          <div class="brain-pb-bear" style="width:${ml.bearPct}%"></div>
        </div>
        <span style="color:#f85149;font-weight:900">${ml.bearPct}% ▼</span>
        <span class="brain-ml-signal" style="color:${col}">
          ${ml.bullPct >= 60 ? '▲ LONG' : ml.bearPct >= 60 ? '▼ SHORT' : '→ NEUTRAL'}
        </span>
      </div>
      <div class="brain-ml-features">${topImp}</div>
    </div>`;
  })() : `<div class="brain-ml-block brain-ml-empty">
      <span class="brain-ml-icon">🤖</span>
      <span class="brain-ml-msg">Sin modelo ML — pulsa <strong>🤖 Sin ML</strong> en el header para importar model_weights.json</span>
    </div>`;

  // ── Probabilidad combinada ────────────────────────────────────────
  const advProbHtml = b.advProb ? (() => {
    const ap = b.advProb;
    const confCol = ap.conf >= 75 ? '#00ff41' : ap.conf >= 55 ? '#e3b341' : '#888';
    return `<div class="brain-adv-prob">
      <div class="brain-adv-prob-row">
        <span style="color:#00ff41;font-weight:900">▲ ${ap.bullPct}%</span>
        <div class="brain-prob-bar">
          <div class="brain-pb-bull" style="width:${ap.bullPct}%"></div>
          <div class="brain-pb-bear" style="width:${ap.bearPct}%"></div>
        </div>
        <span style="color:#f85149;font-weight:900">${ap.bearPct}% ▼</span>
        <span class="brain-adv-conf" style="color:${confCol}">Conf. ${ap.conf}%</span>
      </div>
      <div class="brain-adv-factors">
        ${ap.factors.map(f => `<span class="brain-factor">${escHtml(f)}</span>`).join('')}
      </div>
    </div>`;
  })() : `<div class="brain-prob-row">
      <span style="color:#00ff41;font-weight:900">▲ ${b.bullPct}%</span>
      <div class="brain-prob-bar">
        <div class="brain-pb-bull" style="width:${b.bullPct}%"></div>
        <div class="brain-pb-bear" style="width:${b.bearPct}%"></div>
      </div>
      <span style="color:#f85149;font-weight:900">${b.bearPct}% ▼</span>
    </div>`;

  // ── ML Analyst — interpretación en lenguaje natural ──────────────────
  const ana = !b._loading ? (() => { try { return generateMLAnalysis(b); } catch(e) { console.warn('[Analyst]', e); return null; } })() : null;
  const analystHtml = ana ? (() => {
    const lines = [...ana.mlLines, ...ana.mtfLines, ...ana.harmLines, ...ana.ewLines];
    return `<div class="brain-analyst-block">
      <div class="brain-analyst-hdr">
        <span class="brain-analyst-title">🧠 ML ANALYST</span>
        <span class="brain-analyst-verdict" style="color:${ana.dirCol}">${ana.direction} ${ana.mainPct}%</span>
        <span class="brain-analyst-conf">Conf. ${ana.conf}%</span>
      </div>
      <div class="brain-analyst-lines">
        ${lines.map(l => `<div class="ba-line">${escHtml(l)}</div>`).join('')}
      </div>
      <div class="brain-analyst-scenarios">
        <div class="ba-scenario ba-sce-main">▶ Principal (${ana.mainLabel}): ${escHtml(ana.mainCond)}</div>
        <div class="ba-scenario ba-sce-alt">◀ Alternativo (${ana.altLabel}): ${escHtml(ana.altCond)}</div>
      </div>
      <div class="brain-analyst-action">${escHtml(ana.action)}</div>
    </div>`;
  })() : '';

  // ── Loading placeholder: solo durante el primer render (antes del async) ────
  // b._loading indica que aún NO ha corrido runAdvancedBrain
  const loadingPlaceholder = b._loading
    ? `<div class="brain-loading-adv">⏳ Calculando patrones y análisis avanzado…</div>`
    : '';

  // ── Alert + Mode + Narrative (unchanged) ─────────────────────────
  const alertHtml = b.breakoutAlert ? `
    <div class="brain-alert brain-alert-${b.breakoutAlert.type}">
      <span class="brain-alert-icon">${b.breakoutAlert.type === 'bull' ? '🟢' : '🔴'}</span>
      <span>${escHtml(b.breakoutAlert.msg)}</span>
    </div>` : '';

  el.innerHTML = `
    ${alertHtml}

    <!-- Mode badge -->
    <div class="brain-mode" style="color:${b.modeCol};border-color:${b.modeCol}40;background:${b.modeCol}12">
      <span class="brain-mode-icon">${b.modeIcon}</span>
      <span>${b.mode}</span>
    </div>

    <!-- Narrative -->
    <div class="brain-narrative">${escHtml(b.narrative)}</div>

    <!-- ML Score -->
    ${mlHtml}

    <!-- Advanced probability (combinada ML + reglas) -->
    ${advProbHtml}

    <!-- ML Analyst — interpretación en lenguaje natural -->
    ${analystHtml}

    <!-- Loading placeholder (desaparece cuando MTF/armónicos cargan) -->
    ${loadingPlaceholder}

    <!-- Multi-timeframe -->
    ${mtfHtml}

    <!-- Harmonic patterns -->
    ${harmonicHtml}

    <!-- Elliott Wave -->
    ${elliottHtml}

    <!-- Plans -->
    <div class="brain-plans">
      <div class="brain-plan brain-plan-bull">
        <div class="brain-plan-hdr">
          <span class="brain-plan-tag bull">▲ LONG ${b.advProb?.bullPct ?? b.bullPct}%</span>
          <span class="brain-plan-rr" style="color:${b.bRR>=2?'#00ff41':b.bRR>0?'#e3b341':'#555'}">${b.bRR > 0 ? `R:R ${b.bRR}:1` : 'R:R —'}</span>
        </div>
        <div class="brain-plan-grid">
          <div class="brain-pg-item"><span class="brain-pg-lbl">ENTRADA</span><span class="brain-pg-val">$${fmtPrice(b.bEntry)}</span></div>
          <div class="brain-pg-item"><span class="brain-pg-lbl">TP1</span><span class="brain-pg-val" style="color:#00ff41">$${fmtPrice(b.bTP1)}</span></div>
          <div class="brain-pg-item"><span class="brain-pg-lbl">TP2</span><span class="brain-pg-val" style="color:#00ff41">$${fmtPrice(b.bTP2)}</span></div>
          <div class="brain-pg-item"><span class="brain-pg-lbl">SL</span><span class="brain-pg-val" style="color:#f85149">$${fmtPrice(b.bSL)}</span></div>
        </div>
        <div class="brain-conditions">
          ${b.bullConditions.map(c => `<div class="brain-cond brain-cond-bull">✓ ${escHtml(c)}</div>`).join('')}
        </div>
      </div>
      <div class="brain-plan brain-plan-bear">
        <div class="brain-plan-hdr">
          <span class="brain-plan-tag bear">▼ SHORT ${b.advProb?.bearPct ?? b.bearPct}%</span>
          <span class="brain-plan-rr" style="color:${b.sRR>=2?'#00ff41':b.sRR>0?'#e3b341':'#555'}">${b.sRR > 0 ? `R:R ${b.sRR}:1` : 'R:R —'}</span>
        </div>
        <div class="brain-plan-grid">
          <div class="brain-pg-item"><span class="brain-pg-lbl">ENTRADA</span><span class="brain-pg-val">$${fmtPrice(b.sEntry)}</span></div>
          <div class="brain-pg-item"><span class="brain-pg-lbl">TP1</span><span class="brain-pg-val" style="color:#00ff41">$${fmtPrice(b.sTP1)}</span></div>
          <div class="brain-pg-item"><span class="brain-pg-lbl">TP2</span><span class="brain-pg-val" style="color:#00ff41">$${fmtPrice(b.sTP2)}</span></div>
          <div class="brain-pg-item"><span class="brain-pg-lbl">SL</span><span class="brain-pg-val" style="color:#f85149">$${fmtPrice(b.sSL)}</span></div>
        </div>
        <div class="brain-conditions">
          ${b.bearConditions.map(c => `<div class="brain-cond brain-cond-bear">✓ ${escHtml(c)}</div>`).join('')}
        </div>
      </div>
    </div>

    <!-- Wait for -->
    <div class="brain-wait">
      <div class="brain-wait-hdr">⏳ ESPERAR ANTES DE ENTRAR</div>
      ${b.waitFor.map(w => `<div class="brain-wait-item">→ ${escHtml(w)}</div>`).join('')}
    </div>

    <!-- Pro take -->
    <div class="brain-pro">
      <span class="brain-pro-lbl">🎯 Trader Pro diría:</span>
      <span class="brain-pro-txt">${escHtml(b.proTake)}</span>
    </div>

    <!-- Pipeline button -->
    <button class="btn-brain-pine" id="brainPineBtn">📋 Copiar Pipeline Pine Script → TV</button>
  `;

  // Wire pine copy button
  document.getElementById('brainPineBtn')?.addEventListener('click', () => {
    if (tvTabId && lastAnalysis?.brain?.pine) {
      chrome.tabs.sendMessage(tvTabId, {
        type: 'DRAW_LEVELS',
        levels: [],
        _pine: lastAnalysis.brain.pine,
      }, () => void chrome.runtime.lastError);
    }
    navigator.clipboard?.writeText(b.pine).catch(() => {});
    const btn = document.getElementById('brainPineBtn');
    if (btn) { btn.textContent = '✓ Copiado — pega en Pine Editor TV'; setTimeout(() => { btn.textContent = '📋 Copiar Pipeline Pine Script → TV'; }, 3000); }
    showNotif('Pine Script copiado. Pega en el Editor Pine de TradingView');
  });
}

// ═══════════════════════════════════════════════════════════════════
// RENDER SESSIONS & NEWS
// ═══════════════════════════════════════════════════════════════════

// Live session status bar — updates every second, no API calls
function fmtCountdown(ms) {
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  // Always show seconds so the display ticks every second
  if (h > 0) return `${h}h ${String(m).padStart(2,'0')}m ${String(s).padStart(2,'0')}s`;
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

function renderSessionStatusBar() {
  const el = document.getElementById('snSbSessions');
  if (!el) return;

  const now   = Date.now();
  const dayMs = (() => { const d = new Date(); d.setUTCHours(0,0,0,0); return d.getTime(); })();

  const chips = SESSIONS.map(s => {
    const startMs = dayMs + s.start * 3600000;
    const endMs   = dayMs + s.end   * 3600000;
    const isOpen  = now >= startMs && now < endMs;
    const isPend  = now < startMs;

    if (!isOpen && !isPend) {
      // Closed — show only main sessions (not sub) very dim
      if (s.sub) return '';
      return `<div class="sn-chip sn-chip-closed" style="border-color:rgba(255,255,255,.08)">
        <span style="color:#333;font-size:7.5px;font-weight:700">${s.code}</span>
        <span style="color:#2a2a2a;font-size:7px">✕</span>
      </div>`;
    }

    if (isOpen) {
      const remMs = endMs - now;
      // Color urgency: < 5min = red, < 30min = yellow, else = session color
      const col     = remMs < 5*60000 ? '#f85149' : remMs < 30*60000 ? '#e3b341' : s.color;
      const pulse   = remMs < 5*60000 ? ' sn-pulse' : '';
      const remStr  = fmtCountdown(remMs);
      const label   = remMs < 30*60000 ? `${s.code} cierra` : `${s.code}`;
      return `<div class="sn-chip sn-chip-open${pulse}" style="border-color:${col};background:${col}14">
        <span class="sn-chip-name" style="color:${col}">${label}</span>
        <span class="sn-chip-timer" style="color:${col};font-family:'Orbitron',monospace">${remStr}</span>
      </div>`;
    }

    // Pending
    const waitMs   = startMs - now;
    const localOpen = fmtHourLocal(s.start);
    // Color urgency: < 5min = red pulsing, < 30min = yellow, else = dim
    let col, pulse = '', borderCol;
    if (waitMs < 5*60000) {
      col = '#f85149'; pulse = ' sn-pulse'; borderCol = '#f85149';
    } else if (waitMs < 30*60000) {
      col = '#e3b341'; borderCol = '#e3b341';
    } else {
      col = '#555'; borderCol = 'rgba(255,255,255,.12)';
    }
    const waitStr = fmtCountdown(waitMs);
    const prefix  = waitMs < 30*60000 ? `${s.code} abre` : s.code;
    return `<div class="sn-chip sn-chip-pend${pulse}" style="border-color:${borderCol}">
      <span class="sn-chip-name" style="color:${col}">${prefix}</span>
      <span class="sn-chip-wait" style="color:${col}">${localOpen} <b style="font-family:'Orbitron',monospace">${waitStr}</b></span>
    </div>`;
  }).filter(Boolean).join('');

  // ── Also update detail card remaining times in the dropdown ──────
  SESSIONS.forEach(s => {
    const cardEl = document.getElementById(`sn-card-rem-${s.key}`);
    if (!cardEl) return;
    const startMs = dayMs + s.start * 3600000;
    const endMs   = dayMs + s.end   * 3600000;
    const isOpen  = now >= startMs && now < endMs;
    const isPend  = now < startMs;
    if (isOpen) {
      const remMs = endMs - now;
      const col   = remMs < 5*60000 ? '#f85149' : remMs < 30*60000 ? '#e3b341' : s.color;
      cardEl.textContent = `Cierra en ${fmtCountdown(remMs)}`;
      cardEl.style.color = col;
    } else if (isPend) {
      const waitMs = startMs - now;
      const col    = waitMs < 5*60000 ? '#f85149' : waitMs < 30*60000 ? '#e3b341' : '';
      cardEl.textContent = `Abre en ${fmtCountdown(waitMs)}`;
      cardEl.style.color = col;
    } else {
      cardEl.textContent = '';
    }
  });

  // Next event today
  const nextEv = (window._lastCalDays || [])
    .filter(d => d.isToday).flatMap(d => d.events).find(e => !e.isPast);
  const evChip = nextEv
    ? `<div class="sn-chip sn-chip-event" title="${escHtml(nextEv.title)}">
        <span style="color:${nextEv.stars===3?'#f85149':'#e3b341'}">${nextEv.stars===3?'★★★':'★★☆'}</span>
        <span style="color:var(--text2);font-size:7.5px">${nextEv.country} ${escHtml(nextEv.title.slice(0,15))}…</span>
        <span style="color:var(--accent);font-size:7.5px">${nextEv.time}</span>
      </div>`
    : '';

  el.innerHTML = chips + evChip;
}

function renderSessionsNews(sessions, calDays) {
  const el = document.getElementById('aSessionsNews');
  if (!el) return;

  const nowH = new Date().getUTCHours() + new Date().getUTCMinutes() / 60;

  // ── Visual 24h timeline bar ───────────────────────────────────────
  // Sessions: Asia 0-9, London 7-16, NY 13-22. Overlaps are intentional.
  const W = 100; // % width = 24h
  const pct = h => (h / 24 * 100).toFixed(2);
  const timelineBlocks = sessions.map(s =>
    `<div class="sn-tl-block" style="left:${pct(s.start)}%;width:${pct(s.end - s.start)}%;background:${s.color}22;border-top:2px solid ${s.color}" title="${s.name}: ${s.start}:00–${s.end}:00 UTC"></div>`
  ).join('');
  // Overlap zones
  const overlapBlocks = `
    <div class="sn-tl-overlap" style="left:${pct(7)}%;width:${pct(2)}%" title="Overlap Asia-Londres 07-09 UTC"></div>
    <div class="sn-tl-overlap" style="left:${pct(13)}%;width:${pct(3)}%" title="Overlap Londres-NY 13-16 UTC"></div>`;
  // Current time marker
  const nowPct = (nowH / 24 * 100).toFixed(2);
  const timeMarker = `<div class="sn-tl-now" style="left:${nowPct}%"></div>`;
  // Hour labels
  const hourLabels = [0,4,8,12,16,20].map(h =>
    `<span class="sn-tl-lbl" style="left:${pct(h)}%">${String(h).padStart(2,'0')}</span>`
  ).join('');

  const timelineHtml = `
    <div class="sn-timeline-wrap">
      <div class="sn-timeline">
        ${timelineBlocks}${overlapBlocks}${timeMarker}
      </div>
      <div class="sn-tl-labels">${hourLabels}</div>
    </div>
    <div class="sn-tl-legend">
      ${sessions.map(s=>`<span style="color:${s.color}">▬ ${s.name} ${fmtHourLocal(s.start)}–${fmtHourLocal(s.end)}</span>`).join('')}
      <span class="sn-tl-overlap-lbl">▓ Overlap</span>
    </div>
    <div style="font-size:7px;color:var(--text2);margin-bottom:6px">Horario local (España)</div>`;

  // ── Session cards ─────────────────────────────────────────────────
  const sessHtml = sessions.map(s => {
    const statusCol = s.status==='open' ? '#00ff41' : s.status==='closed' ? '#444' : '#e3b341';
    const statusTxt = s.status==='open' ? '● ABIERTA' : s.status==='closed' ? '✕ CERRADA' : '○ PENDIENTE';
    const rangeInfo = s.high
      ? `<span class="sn-range-hi">H $${fmtPrice(s.high)}</span><span class="sn-range-lo">L $${fmtPrice(s.low)}</span><span class="sn-range-pct">${s.rangePct}%</span>`
      : s.status === 'closed'  ? `<span class="sn-range-na">Sin datos registrados</span>`
      : s.sub                  ? `<span class="sn-range-na">Extended hours</span>`
      :                          `<span class="sn-range-na">Esperando apertura…</span>`;
    const opacity = s.status==='closed' ? ';opacity:.45' : s.sub ? ';opacity:.75' : '';
    return `<div class="sn-session" style="border-left-color:${s.color}${opacity}">
      <div class="sn-sess-hdr">
        <span class="sn-cc-badge" style="border-color:${s.color};color:${s.color}">${s.code}</span>
        <span class="sn-sess-name">${s.name}</span>
        <span class="sn-sess-utc">${fmtHourLocal(s.start)}–${fmtHourLocal(s.end)} <span style="opacity:.5">/ ${fmtHour(s.start)}–${fmtHour(s.end)} UTC</span></span>
        <span class="sn-status" style="color:${statusCol}">${statusTxt}</span>
        <span class="sn-remaining" id="sn-card-rem-${s.key}"></span>
      </div>
      <div class="sn-range">${rangeInfo}</div>
    </div>`;
  }).join('');

  // Overlap active banner
  const asiaLondon = nowH >= 7 && nowH < 9;
  const londonNY   = nowH >= 13 && nowH < 16;
  const overlapHtml = (asiaLondon || londonNY)
    ? `<div class="sn-overlap">⚡ OVERLAP ACTIVO: ${asiaLondon ? 'Asia–Londres (07–09 UTC)' : 'Londres–NY (13–16 UTC)'} — máxima liquidez ahora</div>` : '';

  // ── Full-week economic calendar ──
  let calHtml = '';
  if (calDays.length) {
    calHtml = calDays.map(day => {
      const dayHdr = day.isToday
        ? `<div class="sn-day-hdr sn-day-today">📅 HOY — ${new Date(day.date+'T12:00:00Z').toLocaleDateString('es',{weekday:'long',day:'numeric',month:'long'})}</div>`
        : `<div class="sn-day-hdr${day.isPast?' sn-day-past':''}">${day.label} · ${new Date(day.date+'T12:00:00Z').toLocaleDateString('es',{day:'numeric',month:'short'})}</div>`;

      const evRows = day.events.map(e => {
        const past   = day.isPast || (day.isToday && isPastEvent(e.time));
        const stars  = e.stars === 3 ? '★★★' : '★★☆';
        const sCol   = e.stars === 3 ? '#f85149' : '#e3b341';
        return `<div class="sn-event${past?' sn-ev-past':''}${day.isToday&&!past?' sn-ev-today':''}">
          <span class="sn-ev-time">${e.time}</span>
          <span class="sn-ev-stars" style="color:${sCol}">${stars}</span>
          <span class="sn-ev-country" style="background:${sCol}22;color:${sCol}">${e.country}</span>
          <span class="sn-ev-title">${escHtml(e.title)}</span>
          ${e.forecast ? `<span class="sn-ev-fore">Est: ${escHtml(e.forecast)}</span>` : ''}
          ${e.previous ? `<span class="sn-ev-prev">Ant: ${escHtml(e.previous)}</span>` : ''}
        </div>`;
      }).join('');

      return dayHdr + evRows;
    }).join('');
  } else {
    calHtml = '<div class="sn-empty">Sin eventos de alto impacto esta semana</div>';
  }

  el.innerHTML = `
    <div class="sn-block-title">SESIONES HOY (UTC)</div>
    ${timelineHtml}
    ${overlapHtml}
    <div class="sn-sessions">${sessHtml}</div>
    <div class="sn-block-title" style="margin-top:10px">CALENDARIO ECONOMICO SEMANA — 3 ESTRELLAS</div>
    <div class="sn-cal">${calHtml}</div>
  `;
}

function isPastEvent(timeStr) {
  if (!timeStr || timeStr === 'Todo el día') return false;
  const [h, m] = timeStr.split(':').map(Number);
  const evUtc = h * 60 + (m || 0); // assume time is already UTC (Forex Factory uses ET, adjust later)
  const nowUtc = new Date().getUTCHours() * 60 + new Date().getUTCMinutes();
  return nowUtc > evUtc + 30; // 30min buffer
}

// ═══════════════════════════════════════════════════════════════════
// PAPER TRADING
// ═══════════════════════════════════════════════════════════════════

function ptSave() {
  chrome.storage.local.set({ tvPtBalance: ptBalance, tvPtPositions: ptPositions.slice(0, 200), tvPtStart: ptStart, tvPtPeak: ptPeakEq });
}

function ptLoad(cb) {
  chrome.storage.local.get(['tvPtBalance','tvPtPositions','tvPtStart','tvPtPeak'], r => {
    if (r.tvPtBalance != null)   ptBalance   = r.tvPtBalance;
    if (r.tvPtPositions?.length) ptPositions = r.tvPtPositions;
    if (r.tvPtStart != null)     ptStart     = r.tvPtStart;
    if (r.tvPtPeak  != null)     ptPeakEq    = r.tvPtPeak;
    if (cb) cb();
  });
}

function ptUpdateHeaderBal() {
  const el = document.getElementById('hdrPtBal');
  if (el) {
    el.textContent = '$' + ptBalance.toFixed(2);
    const pct = ptBalance / ptStart;
    el.style.color = pct < 0.9 ? '#f85149' : pct < 0.95 ? '#e3b341' : '';
  }
}

// ── Futures constants (Binance Futures rates) ─────────────────────────────
const FUTURES_TAKER_FEE = 0.0004;   // 0.04% on notional
const FUTURES_MAKER_FEE = 0.0002;   // 0.02% on notional
const MAINT_MARGIN_RATE = 0.004;    // 0.4% maintenance margin (tier 1)
const FUNDING_RATE      = 0.0001;   // 0.01% every 8h (simulated)

function calcLiqPrice(entryPrice, leverage, direction) {
  // Isolated margin liquidation formula (simplified)
  // LONG: liq = entry * (1 - 1/lev + MMR)
  // SHORT: liq = entry * (1 + 1/lev - MMR)
  const mmr = MAINT_MARGIN_RATE;
  return direction === 'long'
    ? entryPrice * (1 - 1 / leverage + mmr)
    : entryPrice * (1 + 1 / leverage - mmr);
}

function ptBuyDemo(symbol, price, tp, sl, margin, direction = 'long', leverage = 10) {
  if (!price || price <= 0) { showNotif('Precio no disponible — analiza primero', 'error'); return; }
  if (margin > ptBalance)   { showNotif(`Margen insuficiente ($${ptBalance.toFixed(2)} disponible)`, 'error'); return; }
  if (ptPositions.find(p => p.status === 'open' && p.symbol === symbol && p.direction === direction)) {
    showNotif(`Ya tienes un ${direction.toUpperCase()} abierto en ${symbol}`, 'error'); return;
  }

  const notional  = margin * leverage;                       // tamaño real de posición
  const openFee   = notional * FUTURES_TAKER_FEE;           // fee apertura
  const qty       = notional / price;                        // cantidad de contratos
  const liqPrice  = calcLiqPrice(price, leverage, direction);

  // TP/SL based on PRICE movement (not margin-amplified)
  const tpPrice   = direction === 'long' ? price * (1 + tp / 100) : price * (1 - tp / 100);
  const slPrice   = direction === 'long' ? price * (1 - sl / 100) : price * (1 + sl / 100);

  ptBalance -= margin + openFee;  // deduct margin + opening fee

  const pos = {
    id:           `pt_${Date.now()}`,
    symbol, price, direction, leverage,
    margin,         // collateral locked
    notional,       // position size USD
    qty,            // token quantity
    openFee,
    feesTotal:    openFee,
    tpPct: tp, slPct: sl, tpPrice, slPrice, liqPrice,
    currentPrice: price,
    entryTime:    Date.now(),
    status:       'open',
    fundingPaid:  0,
    pnlUsd: null, pnlPct: null,
    exitPrice: null, exitTime: null, exitReason: null,
  };
  ptPositions.unshift(pos);
  ptSave();
  renderDemo();
  wsApply(); // add new symbol to live feed
  const icon = direction === 'long' ? '🟢' : '🔴';
  showNotif(`${icon} ${direction.toUpperCase()} ${symbol} ${leverage}x · Margen $${margin.toFixed(0)} · Nocional $${notional.toFixed(0)} · Liq $${fmtPrice(liqPrice)} · Fee $${openFee.toFixed(2)}`);
}

function ptSellDemo(posId, pct = 100, reason = 'manual') {
  const pos = ptPositions.find(p => p.id === posId && p.status === 'open');
  if (!pos) return;

  const frac       = pct / 100;
  const closeQty   = pos.qty * frac;
  const closeNot   = closeQty * pos.currentPrice;   // closing notional
  const closeFee   = closeNot * FUTURES_TAKER_FEE;

  // Gross P&L on position (futures: based on price move × qty)
  const priceMove  = pos.direction === 'long'
    ? pos.currentPrice - pos.price
    : pos.price - pos.currentPrice;
  const grossPnl   = priceMove * closeQty;

  // Funding cost accumulated (rough simulation)
  const holdHours  = (Date.now() - pos.entryTime) / 3600000;
  const fundPaid   = pos.notional * frac * FUNDING_RATE * Math.floor(holdHours / 8);

  const pnlUsd     = grossPnl - closeFee - fundPaid;
  const pnlPct     = pos.margin > 0 ? (pnlUsd / (pos.margin * frac)) * 100 : 0;
  const net        = (pos.margin * frac) + pnlUsd;

  ptBalance += Math.max(0, net); // margin + PnL back (can't go negative)

  if (pct >= 100) {
    Object.assign(pos, {
      status: 'closed', exitPrice: pos.currentPrice, exitTime: Date.now(),
      pnlUsd, pnlPct, exitReason: reason,
      holdMin: (Date.now() - pos.entryTime) / 60000,
      closeFee, fundPaid,
      feesTotal: (pos.openFee || 0) + closeFee,
    });
  } else {
    const partial = {
      ...pos, id: `pt_${Date.now()}_p`, qty: closeQty,
      margin: pos.margin * frac, notional: closeNot,
      status: 'closed', exitPrice: pos.currentPrice,
      exitTime: Date.now(), pnlUsd, pnlPct, exitReason: reason,
      holdMin: (Date.now() - pos.entryTime) / 60000,
      closeFee, fundPaid, _partial: true,
    };
    pos.qty     -= closeQty;
    pos.margin  -= pos.margin * frac;
    pos.notional = pos.qty * pos.price;
    ptPositions.splice(1, 0, partial);
  }

  const eq = ptBalance + ptPositions.filter(p => p.status === 'open')
    .reduce((s, p) => s + p.margin + (p.qty * p.currentPrice - p.notional), 0);
  if (eq > ptPeakEq) ptPeakEq = eq;
  ptSave();
  renderDemo();
  wsApply(); // remove closed symbol from feed if no longer needed
  const icon = reason === 'tp' ? '🎯' : reason === 'sl' ? '🛑' : (pnlUsd >= 0 ? '✓' : '✗');
  showNotif(`${icon} ${pos.symbol} · ${pnlUsd >= 0 ? '+' : ''}${pnlUsd.toFixed(2)}$ (${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(1)}%)`
    , pnlUsd >= 0 ? 'success' : 'error');
}

function ptUpdatePrices() {
  // Update currentPrice from Binance for open positions
  const open = ptPositions.filter(p => p.status === 'open');
  if (!open.length) return;
  Promise.allSettled(open.map(p => fetchTicker(p.symbol))).then(results => {
    results.forEach((r, i) => {
      if (r.status === 'fulfilled' && r.value) {
        const price = parseFloat(r.value.lastPrice);
        if (price > 0) {
          open[i].currentPrice = price;
          const isLong = open[i].direction !== 'short';
          const tpHit  = isLong ? price >= open[i].tpPrice  : price <= open[i].tpPrice;
          const slHit  = isLong ? price <= open[i].slPrice  : price >= open[i].slPrice;
          const liqHit = open[i].liqPrice && (isLong ? price <= open[i].liqPrice : price >= open[i].liqPrice);
          if (liqHit)                         ptSellDemo(open[i].id, 100, 'liquidated');
          else if (open[i].tpPrice && tpHit)  ptSellDemo(open[i].id, 100, 'tp');
          else if (open[i].slPrice && slHit)  ptSellDemo(open[i].id, 100, 'sl');
        }
      }
    });
    ptSave();
    renderDemo();
    ptUpdateHeaderBal();
  });
}

function renderDemo() {
  const open    = ptPositions.filter(p => p.status === 'open');
  const closed  = ptPositions.filter(p => p.status === 'closed');
  const wins    = closed.filter(p => (p.pnlUsd || 0) > 0).length;
  const wr      = closed.length ? Math.round(wins / closed.length * 100) : 0;
  const realPnl = closed.reduce((s, p) => s + (p.pnlUsd || 0), 0);
  // Futures unrealized P&L = qty × (currentPrice - entryPrice) × direction
  const unrealPnl = open.reduce((s, p) => {
    const move = p.direction === 'long'
      ? (p.currentPrice - p.price) * p.qty
      : (p.price - p.currentPrice) * p.qty;
    return s + move;
  }, 0);
  const lockedMargin = open.reduce((s, p) => s + (p.margin || p.totalCost || 0), 0);
  const equity   = ptBalance + lockedMargin + unrealPnl;
  const totalPnl = equity - ptStart;
  const totalPct = ptStart > 0 ? (totalPnl / ptStart) * 100 : 0;
  if (equity > ptPeakEq) ptPeakEq = equity;
  const maxDD = ptPeakEq > 0 ? ((ptPeakEq - equity) / ptPeakEq * 100) : 0;

  const balEl = document.getElementById('dBalMain');
  if (balEl) balEl.textContent = '$' + ptBalance.toFixed(2);
  const eqEl = document.getElementById('dEquity');
  if (eqEl) { eqEl.textContent = '$' + equity.toFixed(2); eqEl.style.color = equity >= ptStart ? '#00ff41' : '#f85149'; }
  const pnlEl = document.getElementById('dTotalPnl');
  if (pnlEl) {
    const col = totalPnl >= 0 ? '#00ff41' : '#f85149';
    pnlEl.innerHTML = '<span style="color:' + col + '">' + (totalPnl >= 0 ? '+' : '') + '$' + totalPnl.toFixed(2) + ' (' + (totalPct >= 0 ? '+' : '') + totalPct.toFixed(2) + '%)</span>';
  }

  const statsEl = document.getElementById('dStats');
  if (statsEl) {
    const wrCol = wr >= 50 ? '#00ff41' : '#f85149';
    statsEl.innerHTML =
      '<div class="d-stat"><span class="d-stat-n">' + closed.length + '</span><span class="d-stat-l">Trades</span></div>' +
      '<div class="d-stat"><span class="d-stat-n" style="color:' + wrCol + '">' + wr + '%</span><span class="d-stat-l">Win Rate</span></div>' +
      '<div class="d-stat"><span class="d-stat-n" style="color:' + (realPnl>=0?'#00ff41':'#f85149') + '">' + (realPnl>=0?'+':'') + '$' + realPnl.toFixed(2) + '</span><span class="d-stat-l">P&L Cerrado</span></div>' +
      '<div class="d-stat"><span class="d-stat-n" style="color:' + (unrealPnl>=0?'#00ff41':'#f85149') + '">' + (unrealPnl>=0?'+':'') + '$' + unrealPnl.toFixed(2) + '</span><span class="d-stat-l">P&L Abierto</span></div>' +
      '<div class="d-stat"><span class="d-stat-n" style="color:' + (maxDD>10?'#f85149':'#e3b341') + '">' + maxDD.toFixed(1) + '%</span><span class="d-stat-l">Max DD</span></div>' +
      '<div class="d-stat"><span class="d-stat-n" style="color:#00d4ff">' + open.length + '</span><span class="d-stat-l">Abiertas</span></div>';
  }

  const tpSymEl   = document.getElementById('dTpSymbol');
  const tpPriceEl = document.getElementById('dTpPrice');
  const tpInfoEl  = document.getElementById('dTpInfo');
  if (tpSymEl)   tpSymEl.textContent   = currentSymbol;
  if (tpPriceEl) tpPriceEl.textContent = currentPrice > 0 ? '$' + fmtPrice(currentPrice) : '—';
  if (tpInfoEl && currentPrice > 0) {
    const margin  = parseFloat(document.getElementById('dTpAmount')?.value) || 100;
    const tp      = parseFloat(document.getElementById('dTpTp')?.value)     || 30;
    const sl      = parseFloat(document.getElementById('dTpSl')?.value)     || 15;
    const lev     = parseInt(document.getElementById('dTpLev')?.value)      || 10;
    const notional = margin * lev;
    const openFee  = notional * FUTURES_TAKER_FEE;
    const closeFeeEst = notional * FUTURES_TAKER_FEE * (1 + tp / 100); // at TP price
    const tpUsd    = margin * (tp / 100) * lev - openFee - closeFeeEst;
    const slUsd    = margin * (sl / 100) * lev + openFee;
    const liqLong  = calcLiqPrice(currentPrice, lev, 'long');
    const liqShort = calcLiqPrice(currentPrice, lev, 'short');
    tpInfoEl.innerHTML =
      '<div class="d-tp-sum-row">' +
        '<span class="d-ti-lbl">Nocional</span><span class="d-ti-val">$' + notional.toFixed(0) + '</span>' +
        '<span class="d-ti-lbl">Fee apertura</span><span class="d-ti-val d-ti-fee">$' + openFee.toFixed(2) + '</span>' +
        '<span class="d-ti-lbl">R:R</span><span class="d-ti-val">' + (tp/sl).toFixed(1) + '</span>' +
      '</div>' +
      '<div class="d-tp-sum-row">' +
        '<span class="d-ti-lbl">TP est.</span><span class="d-ti-val d-ti-g">+$' + tpUsd.toFixed(2) + ' (+' + (tp*lev).toFixed(0) + '% margen)</span>' +
        '<span class="d-ti-lbl">SL est.</span><span class="d-ti-val d-ti-r">-$' + slUsd.toFixed(2) + ' (-' + (sl*lev).toFixed(0) + '% margen)</span>' +
      '</div>' +
      '<div class="d-tp-sum-row">' +
        '<span class="d-ti-lbl">Liq. LONG</span><span class="d-ti-val d-ti-r">$' + fmtPrice(liqLong) + '</span>' +
        '<span class="d-ti-lbl">Liq. SHORT</span><span class="d-ti-val d-ti-r">$' + fmtPrice(liqShort) + '</span>' +
      '</div>';
  }

  const ocEl = document.getElementById('dOpenCount');
  if (ocEl) ocEl.textContent = open.length ? '(' + open.length + ')' : '';

  const openEl = document.getElementById('dOpenList');
  if (openEl) {
    if (!open.length) {
      openEl.innerHTML = '<div class="d-empty">Sin posiciones — abre un LONG o SHORT</div>';
    } else {
      openEl.innerHTML = open.map(function(pos) {
        const isLong   = pos.direction !== 'short';
        const priceChg = pos.price > 0 ? (pos.currentPrice - pos.price) / pos.price : 0;
        const pricePct = (isLong ? priceChg : -priceChg) * 100;
        const pnlUsd   = (pos.notional || pos.usdAmount || 0) * (isLong ? priceChg : -priceChg);
        const col      = pnlUsd >= 0 ? '#00ff41' : '#f85149';
        const dirCol   = isLong ? '#00ff41' : '#f85149';
        const tp  = Math.min(500, Math.max(1, pos.tpPct || 30));
        const sl  = Math.min(80,  Math.max(1, pos.slPct || 15));
        const cur = Math.max(3, Math.min(97, ((pricePct + sl) / (tp + sl)) * 100));
        const heldMs  = Date.now() - pos.entryTime;
        const heldMin = Math.floor(heldMs / 60000);
        const heldSec = Math.floor((heldMs % 60000) / 1000);
        const heldStr = heldMin > 0 ? (heldMin + 'm ' + heldSec + 's') : (heldSec + 's');
        const arrowDir = pnlUsd >= 0 ? (isLong ? '▲' : '▼') : (isLong ? '▼' : '▲');
        const lev       = pos.leverage || 1;
        const notional  = pos.notional || pos.qty * pos.price;
        const liqPrice  = pos.liqPrice || calcLiqPrice(pos.price, lev, pos.direction);
        const liqPct    = pos.price > 0 ? Math.abs((liqPrice - pos.price) / pos.price * 100) : 0;
        const fundHours = Math.floor((Date.now() - pos.entryTime) / 28800000); // every 8h
        const fundEst   = notional * FUNDING_RATE * fundHours;
        const marginPnlPct = pos.margin > 0 ? (pnlUsd / pos.margin) * 100 : pricePct;
        return '<div class="d-pos-card">' +
          '<div class="d-pos-header">' +
            '<span class="d-pos-dir-badge" style="background:' + dirCol + '20;color:' + dirCol + ';border:1px solid ' + dirCol + '50">' + (isLong ? '▲ LONG' : '▼ SHORT') + '</span>' +
            '<span class="d-lev-badge">' + lev + 'x</span>' +
            '<span class="d-pos-sym">' + escHtml(pos.symbol) + '</span>' +
            '<span class="d-pos-age">' + heldStr + '</span>' +
          '</div>' +

          '<div class="d-pos-pnl-block">' +
            '<div class="d-pos-pnl-pct" style="color:' + col + ';text-shadow:0 0 14px ' + col + '50">' + (pnlUsd >= 0 ? '+' : '') + pnlUsd.toFixed(2) + ' USD</div>' +
            '<div class="d-pos-pnl-row">' +
              '<span class="d-pos-pnl-margin" style="color:' + col + '">' + (marginPnlPct >= 0 ? '+' : '') + marginPnlPct.toFixed(2) + '% margen</span>' +
              '<span class="d-pos-pnl-move" style="color:' + col + '">' + (pricePct >= 0 ? '+' : '') + pricePct.toFixed(3) + '% precio</span>' +
            '</div>' +
          '</div>' +

          '<div class="d-pos-prices">' +
            '<div class="d-pos-price-card"><div class="d-pp-lbl">ENTRADA</div><div class="d-pp-val">$' + fmtPrice(pos.price) + '</div></div>' +
            '<div class="d-pos-arrow" style="color:' + col + '">' + arrowDir + '</div>' +
            '<div class="d-pos-price-card" style="border-color:' + col + '40"><div class="d-pp-lbl">AHORA</div><div class="d-pp-val" style="color:' + col + '">$' + fmtPrice(pos.currentPrice) + '</div></div>' +
          '</div>' +

          '<div class="d-pos-meta-row">' +
            '<span class="d-meta-item"><span class="d-meta-lbl">Nocional</span> $' + notional.toFixed(0) + '</span>' +
            '<span class="d-meta-item"><span class="d-meta-lbl">Margen</span> $' + (pos.margin || 0).toFixed(2) + '</span>' +
            '<span class="d-meta-item d-meta-liq"><span class="d-meta-lbl">Liquidación</span> $' + fmtPrice(liqPrice) + ' (' + liqPct.toFixed(1) + '%)</span>' +
            (fundEst > 0 ? '<span class="d-meta-item d-meta-fund"><span class="d-meta-lbl">Funding</span> -$' + fundEst.toFixed(3) + '</span>' : '') +
          '</div>' +

          '<div class="d-pos-tpsl-row">' +
            '<span class="d-tpsl-sl-lbl">SL $' + fmtPrice(pos.slPrice) + '</span>' +
            '<div class="d-tpsl-track"><div class="d-tpsl-cur" style="left:' + cur + '%;background:' + col + '"></div></div>' +
            '<span class="d-tpsl-tp-lbl">TP $' + fmtPrice(pos.tpPrice) + '</span>' +
          '</div>' +
          '<div class="d-pos-actions">' +
            '<button class="btn-d-half" data-id="' + escHtml(pos.id) + '">− 50%</button>' +
            '<button class="btn-d-close" data-id="' + escHtml(pos.id) + '">✕ Cerrar todo</button>' +
          '</div></div>';
      }).join('');
      openEl.querySelectorAll('.btn-d-half').forEach(function(b)  { b.addEventListener('click', function() { ptSellDemo(b.dataset.id, 50); }); });
      openEl.querySelectorAll('.btn-d-close').forEach(function(b) { b.addEventListener('click', function() { ptSellDemo(b.dataset.id, 100); }); });
    }
  }

  const hcEl = document.getElementById('dHistCount');
  if (hcEl) hcEl.textContent = closed.length ? '(' + closed.length + ')' : '';

  const histEl = document.getElementById('dHistList');
  if (histEl) {
    if (!closed.length) {
      histEl.innerHTML = '<div class="d-empty">Sin trades cerrados aun</div>';
    } else {
      histEl.innerHTML = closed.slice(0, 50).map(function(t) {
        const col    = (t.pnlUsd || 0) >= 0 ? '#00ff41' : '#f85149';
        const isLong = t.direction !== 'short';
        const dirCol = isLong ? '#00ff41' : '#f85149';
        const eIcon  = t.exitReason === 'tp' ? 'TP' : t.exitReason === 'sl' ? 'SL' : (t._partial ? '1/2' : 'MAN');
        const hm = t.holdMin != null ? (t.holdMin < 60 ? Math.round(t.holdMin) + 'm' : (t.holdMin/60).toFixed(1) + 'h') : '--';
        return '<div class="d-hist-row">' +
          '<span style="color:' + dirCol + ';font-weight:900">' + (isLong ? '▲' : '▼') + '</span>' +
          '<span class="d-hr-sym">' + escHtml(t.symbol) + '</span>' +
          '<span class="d-hr-badge">' + eIcon + '</span>' +
          '<span class="d-hr-entry">$' + fmtPrice(t.price) + '</span>' +
          '<span style="color:#555">→</span>' +
          '<span class="d-hr-ep">$' + fmtPrice(t.exitPrice) + '</span>' +
          '<span class="d-hr-pnl" style="color:' + col + '">' + ((t.pnlPct||0)>=0?'+':'') + (t.pnlPct||0).toFixed(1) + '%</span>' +
          '<span class="d-hr-usd" style="color:' + col + '">' + ((t.pnlUsd||0)>=0?'+':'') + '$' + Math.abs(t.pnlUsd||0).toFixed(2) + '</span>' +
          '<span class="d-hr-hold">' + hm + '</span>' +
        '</div>';
      }).join('');
    }
  }
  ptUpdateHeaderBal();
}
// ═══════════════════════════════════════════════════════════════════
// WATCHLIST
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// HARMONIC PATTERNS — Biblioteca visual + Modal
// ═══════════════════════════════════════════════════════════════════

const HARMONIC_LIBRARY = {
  Gartley: {
    emoji: '🎯', color: '#00d4ff', strength: 4, frequency: 'Alta',
    ratios: { 'AB/XA': '0.618', 'BC/AB': '0.382-0.886', 'CD/BC': '1.13-1.618', 'AD/XA': '0.786' },
    description: 'El patron mas clasico, identificado por Harold Gartley en 1935. Muy preciso: la reversion en D ocurre al 78.6% de la onda XA.',
    signal: 'Entrada en D con SL justo debajo de X (bullish) o encima de X (bearish). R:R tipico 2:1 a 3:1.',
    tip: 'Si AB/XA esta entre 0.60 y 0.64, la fiabilidad aumenta considerablemente.',
    pts_bull: {X:[15,115], A:[60,12], B:[95,60], C:[125,38], D:[172,72]},
    pts_bear: {X:[15,25],  A:[60,125], B:[95,78], C:[125,100], D:[172,64]},
  },
  Bat: {
    emoji: '🦇', color: '#a78bfa', strength: 4, frequency: 'Media-alta',
    ratios: { 'AB/XA': '0.382-0.5', 'BC/AB': '0.382-0.886', 'CD/BC': '1.618-2.618', 'AD/XA': '0.886' },
    description: 'Variante del Gartley con retroceso AB mas corto (38-50%). El punto D llega al 88.6% de XA, mas profundo que el Gartley.',
    signal: 'Stop muy ajustado bajo X. Alta precision en el PRZ. Ideal para traders con disciplina de SL.',
    tip: 'El Bat requiere que AB NO supere el 50% de XA. Si lo supera, puede ser un Gartley.',
    pts_bull: {X:[15,115], A:[60,12], B:[95,68], C:[125,42], D:[172,82]},
    pts_bear: {X:[15,25],  A:[60,125], B:[95,70], C:[125,95], D:[172,55]},
  },
  Butterfly: {
    emoji: '🦋', color: '#f9a8d4', strength: 3, frequency: 'Media',
    ratios: { 'AB/XA': '0.786', 'BC/AB': '0.382-0.886', 'CD/BC': '1.618-2.24', 'AD/XA': '1.27-1.618' },
    description: 'Patron de extension: el punto D supera el nivel X original. Captura extremos del mercado donde otros patrones no llegan.',
    signal: 'D puede extenderse bastante mas alla de X. Confirmar con vela de reversion y volumen antes de entrar.',
    tip: 'La extension de D mas alla de X es su caracteristica clave. Si D no supera X, no es Butterfly.',
    pts_bull: {X:[15,80], A:[60,12], B:[95,58], C:[125,35], D:[172,128]},
    pts_bear: {X:[15,90], A:[60,130], B:[95,70], C:[125,100], D:[172,28]},
  },
  Crab: {
    emoji: '🦀', color: '#f97316', strength: 3, frequency: 'Baja',
    ratios: { 'AB/XA': '0.382-0.618', 'BC/AB': '0.382-0.886', 'CD/BC': '2.618-3.618', 'AD/XA': '1.618' },
    description: 'El mas extremo de los armonicos. D llega al 161.8% de XA marcando minimos/maximos muy claros. Raro pero muy poderoso.',
    signal: 'Las reversiones son profundas y precisas cuando se cumple. Stop amplio: usar posicion pequena.',
    tip: 'La onda CD es muy larga (2.618-3.618x de BC). Esta extension extrema define al Crab.',
    pts_bull: {X:[15,75], A:[60,12], B:[95,52], C:[125,28], D:[172,138]},
    pts_bear: {X:[15,95], A:[60,132], B:[95,80], C:[125,108], D:[172,18]},
  },
  Shark: {
    emoji: '🦈', color: '#00ff41', strength: 4, frequency: 'Media-alta en cripto',
    ratios: { 'AB/XA': '0.446-0.618', 'BC/AB': '1.13-1.618', 'CD/BC': '0.5-0.886', 'AD/XA': '0.886-1.13' },
    description: 'Patron agresivo y moderno. BC SUPERA el punto A: esa es su firma. Detecta reversiones tras impulsos fuertes. Muy comun en cripto.',
    signal: 'Entrada en D cuando BC retrocede 50-88.6% de BC. Confirmar con volumen decreciente en CD.',
    tip: 'La clave del Shark: el punto B debe superar a A. Si B no cruza A, busca otro patron.',
    pts_bull: {X:[15,105], A:[60,25], B:[108,5],  C:[138,40], D:[172,68]},
    pts_bear: {X:[15,55],  A:[60,138], B:[108,128], C:[138,110], D:[172,88]},
  },
  Cypher: {
    emoji: '🔷', color: '#38bdf8', strength: 4, frequency: 'Media en cripto',
    ratios: { 'AB/XA': '0.382-0.618', 'BC/XA': '1.272-1.414', 'CD/XC': '0.786', 'AD/XA': '0.786' },
    description: 'Patron moderno descubierto por Darren Oglesbee. El punto C supera el nivel A (extension 1.272-1.414 de XA). El punto D retrocede 78.6% de XC. Muy frecuente en cripto.',
    signal: 'Entrada en D al 78.6% de XC. SL detras de X. TP1 en C, TP2 en A. Requiere que C supere A para ser valido.',
    tip: 'La firma del Cypher es C por encima de A (bullish) o por debajo de A (bearish). Sin esa condicion no es Cypher.',
    pts_bull: {X:[15,115], A:[60,22], B:[100,82], C:[148,8], D:[182,92]},
    pts_bear: {X:[15,22],  A:[60,115], B:[100,55], C:[148,132], D:[182,48]},
  },
  'AB=CD': {
    emoji: '🔄', color: '#fb923c', strength: 3, frequency: 'Muy alta',
    ratios: { 'BC/AB': '0.618-0.786', 'CD/AB': '1.0 (iguales)', 'Tiempo AB': '≈ Tiempo CD' },
    description: 'El patron armonico mas simple y frecuente. Las piernas AB y CD son iguales en precio y tiempo. Es la base de todos los demas patrones armonicos.',
    signal: 'Entrada en D cuando CD = AB. SL detras del punto inicial. La igualdad AB=CD es la confirmacion. Buscar confluencia con Fibonacci.',
    tip: 'Si CD < AB el patron es debil. Si CD > AB (extension 1.27 o 1.618) la reversion en D es mas potente pero menos frecuente.',
    pts_bull: {X:[15,28], A:[15,28], B:[78,110], C:[118,48], D:[175,118]},
    pts_bear: {X:[15,118], A:[15,118], B:[78,36], C:[118,98], D:[175,28]},
  },
  'Deep Crab': {
    emoji: '🦀', color: '#ef4444', strength: 3, frequency: 'Muy baja',
    ratios: { 'AB/XA': '0.382-0.618', 'BC/AB': '0.382-0.886', 'CD/BC': '2.618-3.618', 'AD/XA': '1.618+' },
    description: 'Version extrema del Crab donde D supera el 161.8% de XA. Marca extremos absolutos de mercado. Muy raro pero con reversiones explosivas cuando se cumple.',
    signal: 'Posicion muy pequena por el riesgo extremo. Stop amplio detras del nivel D+margin. Solo operar con confirmacion de vela en D.',
    tip: 'El Deep Crab aparece en crashes y blow-off tops. En cripto lo veremos en maximos de panico vendedor o comprador extremo.',
    pts_bull: {X:[15,78], A:[55,12], B:[90,55], C:[122,28], D:[175,142]},
    pts_bear: {X:[15,92], A:[55,130], B:[90,87], C:[122,112], D:[175,8]},
  },
  'Alt Bat': {
    emoji: '🦇', color: '#c084fc', strength: 3, frequency: 'Media-baja',
    ratios: { 'AB/XA': '0.382', 'BC/AB': '0.382-0.886', 'CD/BC': '2.0-3.618', 'AD/XA': '1.13' },
    description: 'Variante del Bat donde AB retrocede exactamente 38.2% de XA y D llega al 113% de XA. Menos comun que el Bat estandar pero igual de preciso.',
    signal: 'Entrada en D al 113% de XA. SL ajustado debajo de D. AB debe ser exactamente 0.382 para confirmar Alt Bat vs Bat.',
    tip: 'La diferencia clave con el Bat: AB es 0.382 (no 0.382-0.5) y D va al 1.13 en vez del 0.886. AB muy corto es la primera senal.',
    pts_bull: {X:[15,110], A:[55,28], B:[92,88], C:[128,52], D:[178,124]},
    pts_bear: {X:[15,28],  A:[55,110], B:[92,50], C:[128,86], D:[178,16]},
  },
};

const CHART_PATTERN_LIBRARY = {
  'Doble Suelo': {
    emoji: '🟢', color: '#00ff41', strength: 4, frequency: 'Alta',
    signal_direction: 'bullish',
    keyLevels: ['Linea de cuello: resistencia entre los dos minimos', 'Target: altura del patron proyectada desde el breakout', 'Stop Loss: debajo del segundo minimo'],
    description: 'Patron de reversion alcista con forma de "W". Dos minimos al mismo nivel indican que el mercado rechazo el mismo soporte dos veces. Los vendedores se agotan.',
    signal: 'Entrada en el breakout confirmado sobre la linea de cuello con cierre de vela. Target = altura del patron desde el breakout. No anticipar antes del breakout.',
    tip: 'El segundo minimo suele estar ligeramente por encima del primero. Volumen creciente en el breakout de la neckline confirma el patron.',
  },
  'Doble Techo': {
    emoji: '🔴', color: '#f85149', strength: 4, frequency: 'Alta',
    signal_direction: 'bearish',
    keyLevels: ['Linea de cuello: soporte entre los dos maximos', 'Target: altura del patron proyectada hacia abajo', 'Stop Loss: encima del segundo maximo'],
    description: 'Patron de reversion bajista con forma de "M". Dos maximos al mismo nivel indican que el mercado rechazo la misma resistencia dos veces. Los compradores se agotan.',
    signal: 'Entrada en el breakdown bajo la linea de cuello. Target = altura del patron. Evitar entradas anticipadas en el segundo techo antes de confirmar el breakdown.',
    tip: 'El volumen suele disminuir en el segundo techo respecto al primero. Confirmar con vela bajista de cierre bajo la neckline.',
  },
  'Cabeza y Hombros': {
    emoji: '💀', color: '#f85149', strength: 5, frequency: 'Alta',
    signal_direction: 'bearish',
    keyLevels: ['Linea de cuello: conecta los valles entre hombros y cabeza', 'Target: distancia cabeza-cuello proyectada hacia abajo', 'Stop Loss: encima del hombro derecho'],
    description: 'Uno de los patrones de reversion mas poderosos. Tres picos: el central (cabeza) mas alto que los laterales (hombros). Indica agotamiento del impulso alcista y cambio de tendencia.',
    signal: 'Entrada al romper y cerrar bajo la neckline. El pull-back a la neckline despues del breakout ofrece segunda entrada con mejor R:R.',
    tip: 'El hombro derecho DEBE ser mas bajo que la cabeza. Si el hombro derecho sube mas que la cabeza el patron se invalida.',
  },
  'Cab. y Hombros Inv.': {
    emoji: '🚀', color: '#00ff41', strength: 5, frequency: 'Alta',
    signal_direction: 'bullish',
    keyLevels: ['Linea de cuello: conecta los picos entre hombros y cabeza invertida', 'Target: distancia cuello-cabeza hacia arriba', 'Stop Loss: debajo del hombro derecho invertido'],
    description: 'Version invertida del Cabeza y Hombros. Tres valles: el central mas profundo. Senal de reversion alcista muy confiable. Muy comun en crypto al final de mercados bajistas.',
    signal: 'Entrada al superar la neckline con cierre por encima. Volumen creciente en el breakout es clave. El pull-back a la neckline es segunda oportunidad de entrada.',
    tip: 'El IH&S al final de una tendencia bajista prolongada es uno de los setups mas poderosos. Busca divergencia alcista en RSI en la cabeza.',
  },
  'Triangulo Ascendente': {
    emoji: '📈', color: '#00d4ff', strength: 4, frequency: 'Media-alta',
    signal_direction: 'bullish',
    keyLevels: ['Resistencia horizontal plana: techos al mismo nivel', 'Soporte creciente: minimos cada vez mas altos', 'Target: altura del triangulo desde la resistencia'],
    description: 'Triangulo con resistencia horizontal y soporte ascendente. Los compradores son mas agresivos en cada retroceso. La presion acumulada se libera en un breakout alcista.',
    signal: 'Entrada en el breakout por encima de la resistencia plana. Target = altura del triangulo. Stop debajo del ultimo minimo dentro del triangulo.',
    tip: 'Cuanto mas tiempo cotiza el precio dentro del triangulo mas explosivo suele ser el breakout. Volumen creciente en el breakout confirma.',
  },
  'Triangulo Descendente': {
    emoji: '📉', color: '#f85149', strength: 4, frequency: 'Media-alta',
    signal_direction: 'bearish',
    keyLevels: ['Soporte horizontal plano: minimos al mismo nivel', 'Resistencia decreciente: maximos cada vez mas bajos', 'Target: altura del triangulo hacia abajo'],
    description: 'Triangulo con soporte horizontal y resistencia descendente. Los vendedores son mas agresivos en cada rebote. Patron bajista con alta probabilidad de ruptura a la baja.',
    signal: 'Entrada en el breakdown bajo el soporte plano. Target = altura del triangulo hacia abajo. Stop encima del ultimo maximo.',
    tip: 'Si el precio rompe al ALZA un triangulo descendente es una senal contraria muy fuerte. Siempre confirmar el breakout con cierre de vela.',
  },
  'Triangulo Simetrico': {
    emoji: '🔺', color: '#00d4ff', strength: 3, frequency: 'Alta',
    signal_direction: 'neutral',
    keyLevels: ['Linea superior descendente', 'Linea inferior ascendente', 'Vertice: punto de convergencia donde ocurre el breakout'],
    description: 'Dos lineas convergentes con pendientes opuestas. El precio se comprime hasta el vertice. Puede romper en cualquier direccion: operar en la direccion del breakout, nunca anticipar.',
    signal: 'Esperar el breakout y operar en esa direccion. Target = anchura inicial del triangulo desde el punto de ruptura. Falsos breakouts son comunes: esperar cierre de vela.',
    tip: 'En tendencias alcistas el triangulo simetrico suele ser continuacion (rompe al alza). En tendencias bajistas ruptura a la baja. Volumen confirma.',
  },
  'Bandera Alcista': {
    emoji: '🚩', color: '#00ff41', strength: 4, frequency: 'Alta',
    signal_direction: 'bullish',
    keyLevels: ['Asta (pole): el impulso fuerte previo', 'Canal de consolidacion en leve correccion', 'Target: longitud del asta desde el breakout'],
    description: 'Consolidacion rectangular en leve correccion tras un movimiento alcista fuerte. Los vendedores no tienen fuerza suficiente. El breakout continua el impulso original.',
    signal: 'Entrada en el breakout por encima del canal de la bandera. Target = longitud del asta proyectada. Stop debajo del canal de consolidacion.',
    tip: 'El asta debe ser un movimiento fuerte y rapido. Si la correccion supera el 50% del asta no es una bandera sino una correccion mas compleja.',
  },
  'Bandera Bajista': {
    emoji: '🚩', color: '#f85149', strength: 4, frequency: 'Alta',
    signal_direction: 'bearish',
    keyLevels: ['Asta bajista: el impulso fuerte a la baja', 'Canal de correccion al alza', 'Target: longitud del asta hacia abajo'],
    description: 'Consolidacion en leve rebote tras una caida fuerte. Los compradores no tienen fuerza para revertir. El breakdown continua la tendencia bajista original.',
    signal: 'Entrada en el breakdown bajo el canal de la bandera. Target = longitud del asta bajista. Stop encima del canal de correccion.',
    tip: 'Las banderas bajistas son muy efectivas en crypto durante mercados bajistas. El rebote dentro de la bandera atrapa compradores que quedan atrapados.',
  },
  'Cuna Alcista': {
    emoji: '📐', color: '#f85149', strength: 4, frequency: 'Media',
    signal_direction: 'bearish',
    keyLevels: ['Dos lineas de tendencia alcistas convergentes', 'Breakout a la baja (contra-intuitivo)', 'Target: inicio del patron (base de la cuna)'],
    description: 'A pesar de subir es un patron BAJISTA. Dos lineas alcistas convergentes con momentum decreciente. Cada maximo y minimo es mas alto pero con menos fuerza.',
    signal: 'Vender en el breakdown bajo la linea inferior. Target = base de la cuna. Frecuente en correcciones dentro de tendencias bajistas.',
    tip: 'La cuna alcista es enganosa: parece subida fuerte pero es trampa. Busca divergencia bajista en RSI y volumen decreciente en los maximos.',
  },
  'Cuna Bajista': {
    emoji: '📐', color: '#00ff41', strength: 4, frequency: 'Media',
    signal_direction: 'bullish',
    keyLevels: ['Dos lineas de tendencia bajistas convergentes', 'Breakout al alza (patron de acumulacion)', 'Target: inicio del patron (parte superior de la cuna)'],
    description: 'A pesar de bajar es un patron ALCISTA. Dos lineas bajistas convergentes. Los vendedores pierden fuerza: cada minimo es mas bajo pero con menos momentum.',
    signal: 'Comprar en el breakout por encima de la linea superior. Target = inicio de la cuna. Comun al final de correcciones dentro de tendencias alcistas.',
    tip: 'La cuna bajista es un patron de acumulacion institucional. Volumen decreciente en la formacion y alto en el breakout confirman el patron.',
  },
  'Copa y Asa': {
    emoji: '☕', color: '#00ff41', strength: 4, frequency: 'Media-baja',
    signal_direction: 'bullish',
    keyLevels: ['Borde de la copa: nivel de resistencia clave', 'Fondo de la copa: soporte principal', 'Asa: pequena correccion antes del breakout final'],
    description: 'Formacion en forma de U seguida de una pequena correccion (asa). Popularizado por William O\'Neil. Indica acumulacion prolongada antes de un movimiento alcista importante.',
    signal: 'Entrada en el breakout por encima del borde de la copa. El asa es la ultima correccion que limpia el papel flojo antes del movimiento real.',
    tip: 'La copa debe ser suave y redondeada, no en V. El asa no debe corregir mas del 33% de la copa. Muy confiable en activos con tendencia alcista de largo plazo.',
  },
};

const STUDY_CONCEPTS = [
  { icon:'📐', title:'Que son los Patrones Armonicos',
    body:'Son estructuras XABCD basadas en proporciones de Fibonacci. Harold Gartley los popularizo en 1935, Scott Carney los sistematizo en los 2000s. La idea: el mercado se mueve en proporciones matematicas predecibles que se repiten.' },
  { icon:'📊', title:'La PRZ — Zona de Reversion Potencial',
    body:'La PRZ (Potential Reversal Zone) es el area donde convergen multiples niveles Fibonacci alrededor del punto D. Cuantos mas niveles convergen en la misma zona, mas fuerte es la senal de reversion.' },
  { icon:'🔢', title:'Fibonacci y los patrones',
    body:'Ratios clave: 0.382, 0.5, 0.618 (retrocesos basicos), 0.786 (raiz de 0.618), 0.886 (raiz de 0.786), 1.27 y 1.618 (extensiones, el numero aureo). Cada patron armonico usa combinaciones especificas de estos ratios.' },
  { icon:'⚡', title:'Como operar un patron armonico',
    body:'1. Identificar XABCD con los ratios correctos. 2. Esperar que el precio llegue al PRZ. 3. Buscar confirmacion: vela de reversion, volumen bajo en D, divergencia RSI. 4. Entrada en D con SL detras de X. 5. TP1 al 38.2% de CD, TP2 al 61.8%, TP3 al nivel A.' },
  { icon:'⚠️', title:'Errores comunes',
    body:'Entrar antes de que el precio llegue al PRZ. No esperar confirmacion de vela. Stop demasiado ajustado (el precio puede probar brevemente mas alla de X). Operar patrones con ratios fuera del rango valido: son invalidos.' },
  { icon:'🎯', title:'STAGE: COMPLETANDO vs APPROACHANDO vs FORMANDO',
    body:'COMPLETANDO: precio dentro del 1.5% del PRZ — senal inminente, maxima alerta, preparar la orden. APPROACHANDO PRZ: a 1.5-5% — prepararse y poner alerta en el PRZ. FORMANDO: aun lejos — patron en desarrollo, no actuar todavia.' },
  { icon:'💡', title:'Gartley vs Bat vs Butterfly — las diferencias',
    body:'GARTLEY: AB retrocede 61.8% de XA, D en 78.6%. BAT: AB retrocede solo 38-50%, D llega al 88.6%. BUTTERFLY: AB en 78.6%, D SUPERA X (127-161.8%). CRAB: D llega al 161.8%, el mas extremo. SHARK: B supera A, el mas agresivo.' },
];

function createChartPatternSVG(name) {
  const W = 200, H = 145;
  const wrap = inner => `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:145px;display:block">${inner}</svg>`;
  const bg = `<rect width="${W}" height="${H}" fill="rgba(0,0,0,0.3)" rx="6"/>`;
  const pl = (pts, col, w=1.8) => `<polyline points="${pts.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${col}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"/>`;
  const ln = (x1,y1,x2,y2,col,w=1.2,da='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="${w}"${da?` stroke-dasharray="${da}"`:''}/>`;
  const tx = (x,y,t,col,sz=8,a='middle') => `<text x="${x}" y="${y}" text-anchor="${a}" font-size="${sz}" fill="${col}" font-family="monospace" font-weight="700">${t}</text>`;
  const dot = (x,y,col) => `<circle cx="${x}" cy="${y}" r="3.5" fill="${col}" stroke="#0d1117" stroke-width="1"/>`;
  const G='#00ff41', R='#f85149', B='#00d4ff', Y='#e3b341', DIM='rgba(255,255,255,0.18)';

  const defs = {
    'Doble Suelo': () => bg +
      ln(48,54,188,54,G,1.2,'5,3') + tx(118,50,'CUELLO',G,7) +
      pl([[10,42],[48,110],[88,54],[128,112],[188,24]],G) +
      dot(48,110,G)+dot(128,112,G) +
      tx(48,125,'1',G,9)+tx(128,127,'2',G,9) +
      tx(100,13,'DOBLE SUELO ▲',G,9),
    'Doble Techo': () => bg +
      ln(48,90,188,90,R,1.2,'5,3') + tx(118,102,'CUELLO',R,7) +
      pl([[10,108],[48,36],[88,90],[128,34],[188,120]],R) +
      dot(48,36,R)+dot(128,34,R) +
      tx(48,28,'1',R,9)+tx(128,26,'2',R,9) +
      tx(100,13,'DOBLE TECHO ▼',R,9),
    'Cabeza y Hombros': () => bg +
      ln(52,84,178,84,R,1.2,'5,3') + tx(115,97,'CUELLO',R,7) +
      pl([[10,100],[32,62],[52,84],[82,28],[112,84],[136,62],[178,118]],R) +
      dot(82,28,R) + tx(82,19,'CABEZA',R,7) +
      tx(32,54,'H.Izq',R,7)+tx(136,54,'H.Der',R,7) +
      tx(100,13,'CAB. Y HOMBROS ▼',R,9),
    'Cab. y Hombros Inv.': () => bg +
      ln(52,62,178,62,G,1.2,'5,3') + tx(115,55,'CUELLO',G,7) +
      pl([[10,46],[32,84],[52,62],[82,118],[112,62],[136,84],[178,28]],G) +
      dot(82,118,G) + tx(82,135,'CABEZA',G,7) +
      tx(32,100,'H.Izq',G,7)+tx(136,100,'H.Der',G,7) +
      tx(100,13,'C.H. INVERTIDO ▲',G,9),
    'Triangulo Ascendente': () => bg +
      ln(10,44,182,44,Y,1.2,'4,3') +
      ln(10,116,155,44,DIM,1,'4,3') +
      pl([[10,116],[40,44],[62,88],[92,44],[112,74],[142,44],[158,38],[190,16]],G) +
      tx(100,13,'T. ASCENDENTE ▲',G,9),
    'Triangulo Descendente': () => bg +
      ln(10,110,182,110,Y,1.2,'4,3') +
      ln(10,32,155,110,DIM,1,'4,3') +
      pl([[10,32],[40,110],[62,70],[92,110],[112,80],[142,110],[158,115],[190,132]],R) +
      tx(100,13,'T. DESCENDENTE ▼',R,9),
    'Triangulo Simetrico': () => bg +
      ln(10,32,162,72,DIM,1,'4,3') +
      ln(10,112,162,72,DIM,1,'4,3') +
      pl([[10,32],[40,104],[68,50],[98,88],[128,64],[148,72],[185,38]],B) +
      tx(172,64,'▶',G,11) +
      tx(100,13,'T. SIMETRICO',B,9),
    'Bandera Alcista': () => bg +
      pl([[10,120],[62,32]],'rgba(0,255,65,0.4)',3) +
      ln(62,32,142,36,G,1,'3,2') + ln(62,50,142,54,G,1,'3,2') +
      pl([[62,32],[80,46],[100,36],[122,50],[142,40]],G) +
      pl([[142,40],[188,6]],G,2.4) +
      tx(100,13,'BANDERA ALCISTA ▲',G,9),
    'Bandera Bajista': () => bg +
      pl([[10,16],[62,104]],'rgba(248,81,73,0.4)',3) +
      ln(62,90,142,86,R,1,'3,2') + ln(62,108,142,100,R,1,'3,2') +
      pl([[62,104],[80,90],[100,100],[122,86],[142,96]],R) +
      pl([[142,96],[188,132]],R,2.4) +
      tx(100,13,'BANDERA BAJISTA ▼',R,9),
    'Cuna Alcista': () => bg +
      ln(10,78,160,30,R,1,'3,2') + ln(10,108,160,60,R,1,'3,2') +
      pl([[10,78],[28,108],[56,72],[84,98],[110,68],[136,88],[158,62],[188,116]],R) +
      tx(172,112,'↘',R,13) +
      tx(100,13,'CUNA ALCISTA ▼',R,9),
    'Cuna Bajista': () => bg +
      ln(10,36,160,78,G,1,'3,2') + ln(10,66,160,108,G,1,'3,2') +
      pl([[10,36],[28,66],[56,50],[84,72],[110,58],[136,80],[158,70],[188,30]],G) +
      tx(172,30,'↗',G,13) +
      tx(100,13,'CUNA BAJISTA ▲',G,9),
    'Copa y Asa': () => bg +
      ln(10,38,168,38,G,1.2,'4,3') +
      pl([[10,38],[26,68],[46,96],[70,116],[96,124],[120,116],[144,96],[162,68],[168,38]],G) +
      pl([[168,38],[176,54],[184,44]],G) +
      pl([[184,44],[196,18]],G,2.4) +
      tx(88,135,'Copa',G,8) + tx(174,40,'Asa',G,7) +
      tx(100,13,'COPA Y ASA ▲',G,9),
  };
  const fn = defs[name];
  return fn ? wrap(fn()) : '';
}

function createPatternSVG(patName, isBullish) {
  const lib = HARMONIC_LIBRARY[patName];
  if (!lib) return '';
  const W = 200, H = 145;
  const pts = isBullish ? lib.pts_bull : lib.pts_bear;
  const { X: px, A: pa, B: pb, C: pc, D: pd } = pts;
  const col = isBullish ? '#00ff41' : '#f85149';

  const seg = (p1, p2, c) =>
    `<line x1="${p1[0]}" y1="${p1[1]}" x2="${p2[0]}" y2="${p2[1]}" stroke="${c}" stroke-width="1.8" stroke-linecap="round"/>`;
  const node = (p, lbl, c) => {
    const above = p[1] > H / 2;
    const dy = above ? -8 : 13;
    const anchor = p[0] < 30 ? 'start' : p[0] > 160 ? 'end' : 'middle';
    return `<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="${c}" stroke="#0d1117" stroke-width="1.5"/>` +
      `<text x="${p[0]}" y="${p[1] + dy}" text-anchor="${anchor}" font-size="10" font-weight="900" fill="${c}" font-family="monospace">${lbl}</text>`;
  };
  const przY = Math.min(pd[1] - 10, H - 20);
  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:145px;display:block">` +
    `<rect x="0" y="0" width="${W}" height="${H}" fill="rgba(0,0,0,0.3)" rx="6"/>` +
    `<rect x="${pd[0] - 22}" y="${przY}" width="44" height="20" rx="4" fill="${col}" opacity="0.15"/>` +
    `<text x="${pd[0]}" y="${przY - 3}" text-anchor="middle" font-size="7.5" fill="${col}" opacity="0.8" font-family="monospace">PRZ</text>` +
    seg(px, pa, 'rgba(255,255,255,0.25)') + seg(pa, pb, 'rgba(255,255,255,0.25)') +
    seg(pb, pc, 'rgba(255,255,255,0.25)') + seg(pc, pd, col) +
    node(px, 'X', '#8b949e') + node(pa, 'A', lib.color) + node(pb, 'B', lib.color) +
    node(pc, 'C', lib.color) + node(pd, 'D', col) +
    `</svg>`;
}

function renderPatternsTab() {
  const detEl = document.getElementById('ptDetectedList');
  const subEl = document.getElementById('ptDetectedSub');
  const harms = window._lastAdvanced?.harmonics || [];
  window._ptHarms = harms;

  if (detEl) {
    if (!harms.length) {
      detEl.innerHTML = '<div class="pt-empty">Sin patrones armonicos detectados en el analisis actual</div>';
      if (subEl) subEl.textContent = 'ninguno detectado';
    } else {
      if (subEl) subEl.textContent = `${harms.length} detectado${harms.length > 1 ? 's' : ''}`;
      detEl.innerHTML = harms.map((h, idx) => {
        const lib = HARMONIC_LIBRARY[h.name] || {};
        const col = h.type === 'bullish' ? '#00ff41' : '#f85149';
        const stageCol = h.stage === 'COMPLETANDO' ? '#ff4500' : h.stage.includes('APPROACH') ? '#e3b341' : '#888';
        return `<div class="pt-detected-card ${h.type}" data-idx="${idx}">` +
          `<div class="ptd-top">` +
          `<span class="ptd-emoji">${lib.emoji || '🔺'}</span>` +
          `<span class="ptd-name" style="color:${lib.color || col}">${escHtml(h.name)}</span>` +
          `<span class="ptd-dir" style="color:${col}">${h.type === 'bullish' ? '▲ ALCISTA' : '▼ BAJISTA'}</span>` +
          `<span class="ptd-stage" style="color:${stageCol}">${escHtml(h.stage)}</span>` +
          `<span class="ptd-prob">${h.prob}%</span>` +
          `</div>` +
          `<div class="ptd-prz">PRZ ${fmtPrice(h.dZone?.low)} – ${fmtPrice(h.dZone?.high)} &nbsp;·&nbsp; ${h.distPct}% de distancia</div>` +
          `<div class="ptd-cta">Toca para ver diagrama →</div>` +
          `</div>`;
      }).join('');
      detEl.querySelectorAll('.pt-detected-card').forEach(card => {
        card.addEventListener('click', () => {
          const h = window._ptHarms?.[parseInt(card.dataset.idx)];
          if (h) showPatternModal(h.name, h, 'harmonic');
        });
      });
    }
  }

  const libEl = document.getElementById('ptLibrary');
  if (libEl && !libEl.children.length) {
    const mkCard = (name, p, libtype, svgFn) => {
      const stars = '★'.repeat(p.strength) + '☆'.repeat(5 - p.strength);
      const sub = libtype === 'harmonic' ? p.frequency
        : (p.signal_direction === 'bullish' ? '▲ Alcista' : p.signal_direction === 'bearish' ? '▼ Bajista' : '◆ Neutro');
      const subCol = libtype === 'classic'
        ? (p.signal_direction === 'bullish' ? '#00ff41' : p.signal_direction === 'bearish' ? '#f85149' : '#00d4ff')
        : 'rgba(255,255,255,0.5)';
      return `<div class="pt-lib-card" data-name="${name}" data-libtype="${libtype}">` +
        `<div class="pt-lib-hdr">` +
        `<span class="pt-lib-emoji">${p.emoji}</span>` +
        `<div class="pt-lib-title-wrap">` +
        `<span class="pt-lib-name" style="color:${p.color}">${name}</span>` +
        `<span class="pt-lib-freq" style="color:${subCol}">${sub}</span>` +
        `</div><span class="pt-lib-stars" style="color:${p.color}">${stars}</span>` +
        `</div>` +
        `<div class="pt-lib-svg">${svgFn()}</div>` +
        `<div class="pt-lib-desc">${escHtml(p.description.substring(0,90))}…</div>` +
        `</div>`;
    };

    const harmKeys = Object.keys(HARMONIC_LIBRARY);
    const classicKeys = Object.keys(CHART_PATTERN_LIBRARY);

    libEl.innerHTML =
      `<div class="pt-subsec-hdr">📐 Patrones Armónicos <span class="pt-subsec-count">${harmKeys.length}</span></div>` +
      `<div class="pt-lib-grid">` +
      harmKeys.map(n => mkCard(n, HARMONIC_LIBRARY[n], 'harmonic', () => createPatternSVG(n, true))).join('') +
      `</div>` +
      `<div class="pt-subsec-hdr" style="margin-top:8px">📊 Patrones Clásicos <span class="pt-subsec-count">${classicKeys.length}</span></div>` +
      `<div class="pt-lib-grid">` +
      classicKeys.map(n => mkCard(n, CHART_PATTERN_LIBRARY[n], 'classic', () => createChartPatternSVG(n))).join('') +
      `</div>`;

    libEl.querySelectorAll('.pt-lib-card').forEach(card => {
      card.addEventListener('click', () => showPatternModal(card.dataset.name, null, card.dataset.libtype));
    });
  }

  const conEl = document.getElementById('ptConcepts');
  if (conEl && !conEl.children.length) {
    conEl.innerHTML = STUDY_CONCEPTS.map(c =>
      `<div class="pt-concept-card">` +
      `<div class="pt-concept-hdr"><span>${c.icon}</span><span class="pt-concept-title">${escHtml(c.title)}</span></div>` +
      `<div class="pt-concept-body">${escHtml(c.body)}</div>` +
      `</div>`
    ).join('');
  }
}

function showPatternModal(name, detected, libtype) {
  const isClassic = libtype === 'classic';
  const lib = isClassic ? CHART_PATTERN_LIBRARY[name] : HARMONIC_LIBRARY[name];
  if (!lib) return;

  if (!isClassic && typeof detected === 'string') {
    try { detected = JSON.parse(detected.replace(/'/g,'"')); } catch { detected = null; }
  }
  const isBull = isClassic ? lib.signal_direction !== 'bearish' : (detected ? detected.type === 'bullish' : true);

  const svgHtml = isClassic ? createChartPatternSVG(name) : createPatternSVG(name, isBull);

  const detHtml = (!isClassic && detected) ? `<div class="pm-detected-info">` +
    `<div class="pm-di-row"><span>Estado:</span><span style="font-weight:700;color:${detected.stage === 'COMPLETANDO' ? '#ff4500' : '#e3b341'}">${escHtml(detected.stage)}</span></div>` +
    `<div class="pm-di-row"><span>PRZ:</span><span>${fmtPrice(detected.dZone?.low)} – ${fmtPrice(detected.dZone?.high)}</span></div>` +
    `<div class="pm-di-row"><span>SL:</span><span style="color:#f85149">${fmtPrice(detected.sl)}</span></div>` +
    `<div class="pm-di-row"><span>TP1:</span><span style="color:#00ff41">${fmtPrice(detected.tp1)}</span></div>` +
    `<div class="pm-di-row"><span>TP2:</span><span style="color:#00ff41">${fmtPrice(detected.tp2)}</span></div>` +
    `<div class="pm-di-row"><span>Dist. PRZ:</span><span>${detected.distPct}%</span></div>` +
    `</div>` : '';

  const levelsHtml = isClassic
    ? lib.keyLevels.map(l => `<div class="pm-ratio"><span class="pm-ratio-v" style="color:var(--text1)">${l}</span></div>`).join('')
    : Object.entries(lib.ratios).map(([k,v]) => `<div class="pm-ratio"><span class="pm-ratio-k">${k}</span><span class="pm-ratio-v">${v}</span></div>`).join('');

  const dirTag = isClassic
    ? (lib.signal_direction === 'bullish' ? ' <span style="color:#00ff41">▲ ALCISTA</span>' : lib.signal_direction === 'bearish' ? ' <span style="color:#f85149">▼ BAJISTA</span>' : ' <span style="color:#00d4ff">◆ NEUTRO</span>')
    : (detected ? (isBull ? ' <span style="color:#00ff41">▲ ALCISTA</span>' : ' <span style="color:#f85149">▼ BAJISTA</span>') : '');

  document.getElementById('pmContent').innerHTML =
    `<div class="pm-header" style="border-color:${lib.color}40">` +
    `<span class="pm-emoji">${lib.emoji}</span>` +
    `<div><div class="pm-name" style="color:${lib.color}">${name}${dirTag}</div>` +
    `<div class="pm-freq">${lib.frequency} &nbsp;·&nbsp; ${'★'.repeat(lib.strength)}${'☆'.repeat(5 - lib.strength)}</div></div>` +
    `</div>` +
    `<div class="pm-svg-wrap">${svgHtml}</div>` +
    detHtml +
    `<div class="pm-section-title">${isClassic ? 'Niveles Clave' : 'Ratios de Fibonacci'}</div><div class="pm-ratios">${levelsHtml}</div>` +
    `<div class="pm-section-title">Que es el ${name}?</div><div class="pm-text">${escHtml(lib.description)}</div>` +
    `<div class="pm-section-title">Como operar?</div><div class="pm-text">${escHtml(lib.signal)}</div>` +
    `<div class="pm-tip">💡 ${escHtml(lib.tip)}</div>`;

  document.getElementById('pmOverlay').style.display = 'block';
  document.getElementById('pmModal').style.display = 'flex';
  document.getElementById('pmModal').scrollTop = 0;
}

function closePatternModal() {
  document.getElementById('pmOverlay').style.display = 'none';
  document.getElementById('pmModal').style.display = 'none';
}

async function refreshWatchlist() {
  const tickers = await fetchMultiTicker(watchlist);
  watchData = tickers;
  renderWatchlist();
}

function renderWatchlist() {
  const el = document.getElementById('watchList');
  if (!el) return;
  el.innerHTML = watchlist.map(sym => {
    const t   = watchData[sym];
    if (!t) return `<div class="wl-row wl-loading"><span class="wl-sym">${escHtml(sym)}</span><span class="wl-price">…</span></div>`;
    const chg  = parseFloat(t.priceChangePercent);
    const col  = chg >= 0 ? '#00ff41' : '#f85149';
    const price = parseFloat(t.lastPrice);
    return `<div class="wl-row" data-sym="${escHtml(sym)}">
      <span class="wl-sym">${escHtml(sym.replace('USDT',''))}</span>
      <span class="wl-price">$${fmtPrice(price)}</span>
      <span class="wl-chg" style="color:${col}">${fmtPct(chg)}</span>
      <span class="wl-vol">${fmtVol(parseFloat(t.quoteVolume))}</span>
      <button class="btn-wl-analyze" data-sym="${escHtml(sym)}">▶</button>
    </div>`;
  }).join('');
  el.querySelectorAll('.btn-wl-analyze').forEach(btn => {
    btn.addEventListener('click', () => {
      currentSymbol = btn.dataset.sym;
      document.getElementById('symInput').value = currentSymbol;
      switchTab('analysis');
      loadAndAnalyze();
    });
  });
  el.querySelectorAll('.wl-row[data-sym]').forEach(row => {
    row.addEventListener('click', (e) => {
      if (e.target.classList.contains('btn-wl-analyze')) return;
      currentSymbol = row.dataset.sym;
      document.getElementById('symInput').value = currentSymbol;
      switchTab('analysis');
      loadAndAnalyze();
    });
  });
}

// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
// ADVANCED BRAIN UPDATE (ML + MTF + Harmonic + Elliott)
// Función centralizada — se llama tras loadAndAnalyze Y tras cargar el modelo ML
// ═══════════════════════════════════════════════════════════════════

async function runAdvancedBrain(analysis, candleData, symbol, tf) {
  const brainEl = document.getElementById('aBrain');

  // ── Paso 1: Cálculos síncronos — siempre rápidos ─────────────────
  let harmonics = [], elliott = null, mlResult = null;
  try { harmonics = detectHarmonicPatterns(candleData) || []; } catch {}
  try { elliott   = detectElliottWave(candleData); }              catch {}
  try { mlResult  = mlInfer(candleData); }                        catch {}

  // ── Paso 2: Primer re-render (elimina placeholder "_loading") ─────
  // Este render siempre ocurre, incluso si todo es null
  const mkProb = (mtf) => {
    try {
      return computeAdvancedProb({
        score: analysis.score, mtf, harmonics, elliott,
        div: analysis.div, volume: analysis.volume,
        patterns: analysis.patterns,
        rsi: analysis.indicators?.rsi, hist: analysis.indicators?.hist,
        mlResult,
      });
    } catch { return null; }
  };

  const adv1 = mkProb(null);
  window._lastAdvanced = { mtf: null, harmonics, elliott, advProb: adv1, mlResult };
  if (brainEl && analysis.brain) {
    try {
      renderBrain(brainEl, { ...analysis.brain, mtf: null, harmonics, elliott, advProb: adv1, mlResult });
    } catch (e) { console.warn('[AdvBrain] render1 error:', e); }
  }

  // ── Paso 3: Fetch MTF (puede tardar, no bloquea el render anterior) ─
  let mtf = null;
  try {
    mtf = await fetchMTFanalysis(symbol, tf);
  } catch {}

  // Fallback local si MTF externo falla
  if (!mtf && candleData.length >= 50) {
    try {
      const closes = candleData.map(c => c.close);
      const k20 = 2 / 21, k50 = 2 / 51;
      let e20 = closes[0], e50 = closes[0];
      for (const v of closes) { e20 = e20 * (1-k20) + v * k20; e50 = e50 * (1-k50) + v * k50; }
      const last = closes[closes.length - 1];
      const dir = last > e20 && e20 > e50 ? 'bullish'
                : last < e20 && e20 < e50 ? 'bearish'
                : last > e20 ? 'neutral_up' : 'neutral_down';
      mtf = {
        primaryTf: '—', secondaryTf: '—', currentTf: tf,
        prim: { dir, rsi: null, ema20: e20.toFixed(0), last: last.toFixed(0) },
        sec: null, align: dir === 'bullish' ? 'ALCISTA ↑~' : dir === 'bearish' ? 'BAJISTA ↓~' : 'NEUTRAL →',
        alignScore: 50, _localOnly: true,
      };
    } catch {}
  }

  // ── Paso 4: Re-render final con MTF ──────────────────────────────
  const adv2 = mkProb(mtf);
  window._lastAdvanced = { mtf, harmonics, elliott, advProb: adv2, mlResult };
  if (brainEl && analysis.brain) {
    try {
      renderBrain(brainEl, { ...analysis.brain, mtf, harmonics, elliott, advProb: adv2, mlResult });
    } catch (e) { console.warn('[AdvBrain] render2 error:', e); }
  }
}

// ═══════════════════════════════════════════════════════════════════
// LOAD & ANALYZE
// ═══════════════════════════════════════════════════════════════════

async function loadAndAnalyze(silent = false) {
  const loadEl  = document.getElementById('aLoading');
  const contEl  = document.getElementById('aContent');
  const loadTxt = document.getElementById('aLoadingText');
  const errEl   = document.getElementById('aError');

  // Al analizar un símbolo/TF nuevo o manualmente: limpiar datos avanzados
  // para que el placeholder aparezca y runAdvancedBrain los recalcule
  if (!silent) window._lastAdvanced = null;

  // Only show the loading spinner on first load or manual symbol change.
  // Silent refreshes (auto every 30s) update in place — no flicker.
  const isFirstLoad = !lastAnalysis;
  if (!silent && isFirstLoad) {
    if (loadEl)  { loadEl.style.display = 'flex'; }
    if (contEl)  { contEl.style.display = 'none'; }
    if (loadTxt) { loadTxt.textContent  = `Descargando ${currentSymbol} ${currentTf}…`; }
    if (errEl)   { errEl.style.display = 'none'; }
  }

  try {
    candles = await fetchKlinesUniversal(currentSymbol, currentTf, 200);
    if (!candles?.length) throw new Error(`Sin datos para ${currentSymbol}`);
    currentPrice = candles[candles.length - 1].close;
    const analysis = runAnalysis(candles, currentSymbol, currentTf);
    if (loadEl) loadEl.style.display = 'none';
    if (contEl) contEl.style.display = 'block';
    if (errEl)  errEl.style.display  = 'none';
    renderAnalysis(analysis);
    wsApply();

    // ── Advanced Brain: ML + harmonic + MTF + Elliott (no bloquea UI) ──────────
    runAdvancedBrain(analysis, candles, currentSymbol, currentTf);

    // Fetch sessions, news & economic calendar in background (non-blocking)
    Promise.all([
      fetchSessionRanges(currentSymbol),
      fetchEcoCalendar(),
    ]).then(([sessions, calDays]) => {
      window._lastCalDays  = calDays;
      window._lastSessions = sessions;
      renderSessionsNews(sessions, calDays);
      renderSessionStatusBar(); // initial render
    }).catch(() => {});

    // Re-render sessions every minute (session status changes over time)
    clearInterval(window._sessionTimer);
    window._sessionTimer = setInterval(async () => {
      const sessions = await fetchSessionRanges(currentSymbol).catch(() => []);
      const el = document.getElementById('aSessionsNews');
      if (el && sessions.length) renderSessionsNews(sessions, window._lastCalDays || []);
    }, 60000);
  } catch (e) {
    // On silent refresh, swallow errors — don't break the UI mid-session
    if (!silent) {
      if (loadEl) loadEl.style.display = 'none';
      if (contEl) { contEl.style.display = 'block'; }
      if (errEl)  { errEl.style.display = 'block'; errEl.textContent = '❌ ' + (e.message || String(e)); }
      showNotif((e.message || String(e)), 'error');
    }
    console.error('[TVAnalyzer]', e);
  }
}

// ═══════════════════════════════════════════════════════════════════
// NOTIFICATION
// ═══════════════════════════════════════════════════════════════════

function showNotif(msg, type = 'success') {
  const el = document.getElementById('notification');
  if (!el) return;
  el.textContent = msg;
  el.className   = 'notification show ' + type;
  setTimeout(() => el.classList.remove('show'), 3000);
}

// ═══════════════════════════════════════════════════════════════════
// TABS
// ═══════════════════════════════════════════════════════════════════

function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  const tabEl = document.getElementById('tab-' + name);
  const btnEl = document.querySelector(`.tab[data-tab="${name}"]`);
  if (tabEl) tabEl.classList.add('active');
  if (btnEl) btnEl.classList.add('active');
}

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════

function mlUpdateHeaderBadge() {
  const btn    = document.getElementById('mlHdrBtn');
  const status = document.getElementById('mlHdrStatus');
  if (!btn || !status) return;
  if (_mlModel) {
    btn.classList.add('ml-active');
    status.textContent = `ML ${_mlModel.cv_auc}`;
    btn.title = `Modelo ML activo — AUC ${_mlModel.cv_auc} · ${_mlModel.n_samples?.toLocaleString()} muestras\nClick para reemplazar`;
  } else {
    btn.classList.remove('ml-active');
    status.textContent = 'Sin ML';
    btn.title = 'Importar model_weights.json (entrena con Python en tv-analyzer/train/)';
  }
}

// ── ML file import handler ────────────────────────────────────────────────────
document.addEventListener('change', async (e) => {
  if (e.target.id !== 'mlFileInput') return;
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const text  = await file.text();
    const model = JSON.parse(text);
    if (!model.coefficients || !model.scaler_mean) throw new Error('JSON invalido');
    _mlModel = model;
    chrome.storage.local.set({ mlModel: model });
    mlUpdateHeaderBadge();
    showNotif(`Modelo ML cargado - AUC ${model.cv_auc} - ${model.n_samples?.toLocaleString()} muestras`, 'success');
    // Re-renderiza el Brain con el modelo recién cargado (sin descargar datos de nuevo)
    if (lastAnalysis && candles.length) {
      runAdvancedBrain(lastAnalysis, candles, currentSymbol, currentTf);
    } else if (candles.length) {
      loadAndAnalyze(true);
    }
  } catch (err) {
    showNotif('Error al leer el JSON: ' + err.message, 'error');
  }
});

document.addEventListener('DOMContentLoaded', () => {
  // Cargar modelo ML al iniciar — si llega tarde, re-renderiza el Brain
  mlLoadModel().then(loaded => {
    if (!loaded) return;
    mlUpdateHeaderBadge();
    // Si el análisis ya estaba renderizado sin el modelo, lo actualizamos
    if (lastAnalysis && candles.length) {
      runAdvancedBrain(lastAnalysis, candles, currentSymbol, currentTf);
    }
  });

  // Tab navigation
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      switchTab(btn.dataset.tab);
      if (btn.dataset.tab === 'watchlist') refreshWatchlist();
      if (btn.dataset.tab === 'patrones') renderPatternsTab();
    });
  });

  document.getElementById('pmOverlay')?.addEventListener('click', closePatternModal);
  document.getElementById('pmClose')?.addEventListener('click', closePatternModal);

  // Symbol input
  const symInput = document.getElementById('symInput');
  const tfSelect = document.getElementById('tfSelect');

  symInput?.addEventListener('keydown', e => {
    if (e.key === 'Enter') { loadAndAnalyze(); if (autoRefresh) startAutoRefresh(); }
  });
  document.getElementById('analyzeBtn')?.addEventListener('click', () => {
    loadAndAnalyze();
    if (autoRefresh) startAutoRefresh(); // reset countdown
  });

  tfSelect?.addEventListener('change', () => {
    currentTf = tfSelect.value;
    loadAndAnalyze();
    if (autoRefresh) startAutoRefresh();
  });

  // Auto-refresh — starts automatically, button pauses/resumes
  function startAutoRefresh() {
    if (autoRefresh) clearInterval(autoRefresh);
    autoRefresh = setInterval(() => loadAndAnalyze(true), 60000); // reload candle history silently every 60s
    const btn = document.getElementById('autoRefreshBtn');
    if (btn) { btn.textContent = '⟳ 30s'; btn.classList.add('active'); btn.style.color = '#00ff41'; }
  }
  function stopAutoRefresh() {
    clearInterval(autoRefresh);
    autoRefresh = null;
    const btn = document.getElementById('autoRefreshBtn');
    if (btn) { btn.textContent = '⟳ Pausado'; btn.classList.remove('active'); btn.style.color = ''; }
  }

  document.getElementById('autoRefreshBtn')?.addEventListener('click', () => {
    if (autoRefresh) stopAutoRefresh(); else startAutoRefresh();
  });

  // ── Sync status UI ───────────────────────────────────────────────────────
  function setSyncStatus(state, text) {
    // state: 'ok' | 'warn' | 'err' | 'idle'
    const dot = document.getElementById('syncDot');
    const lbl = document.getElementById('syncLabel');
    if (dot) { dot.className = 'sync-dot ' + (state === 'idle' ? '' : state); }
    if (lbl) lbl.textContent = text;
  }

  // ── Core sync function — reads current symbol from the TV tab ─────────────
  function syncFromTV(andAnalyze = true) {
    const btn = document.getElementById('syncBtn');
    if (btn) btn.classList.add('syncing');
    setSyncStatus('warn', 'Buscando pestaña TradingView…');

    chrome.tabs.query({ url: '*://*.tradingview.com/*' }, tabs => {
      // Prefer the currently active tab; fallback to any TV tab
      chrome.tabs.query({ active: true, currentWindow: true }, activeTabs => {
        const activeTab = activeTabs[0];
        const tvTab = (activeTab?.url?.includes('tradingview.com') ? activeTab : null)
                   || tabs.find(t => t.url?.includes('tradingview.com/chart'));

        if (btn) btn.classList.remove('syncing');

        if (!tvTab) {
          setSyncStatus('err', 'No hay pestaña de TradingView abierta');
          showNotif('Abre TradingView en el navegador primero', 'error');
          return;
        }

        tvTabId = tvTab.id;

        chrome.tabs.sendMessage(tvTab.id, { type: 'GET_TV_DATA' }, resp => {
          void chrome.runtime.lastError;

          if (!resp?.symbol) {
            // Reintentar inyectando el content script manualmente
            chrome.scripting.executeScript(
              { target: { tabId: tvTab.id }, files: ['content.js'] },
              () => {
                void chrome.runtime.lastError;
                setTimeout(() => {
                  chrome.tabs.sendMessage(tvTab.id, { type: 'GET_TV_DATA' }, resp2 => {
                    void chrome.runtime.lastError;
                    if (!resp2?.symbol) {
                      setSyncStatus('err', `TV detectada · sin datos · tab ${tvTab.id}`);
                      showNotif('Recarga la pestaña de TradingView (F5) e inténtalo de nuevo', 'error');
                    }
                  });
                }, 800);
              }
            );
            return;
          }

          const sym = resp.symbol.replace(/[^A-Z0-9.^-]/g, '').toUpperCase() || currentSymbol;
          const tf  = resp.timeframe || currentTf;
          const changed = sym !== currentSymbol;

          currentSymbol = sym;
          currentTf     = tf in TF_MAP ? tf : currentTf;
          if (symInput) symInput.value = sym;
          if (tfSelect) tfSelect.value = currentTf;

          setSyncStatus('ok', `Sincronizado · ${sym} · ${currentTf} · tab ${tvTab.id}`);

          if (andAnalyze && changed) {
            loadAndAnalyze();
            if (autoRefresh) startAutoRefresh();
          } else if (andAnalyze && !changed) {
            loadAndAnalyze(true);   // silent refresh with same symbol
          }
        });
      });
    });
  }

  // ── Listen for auto symbol changes from content script ───────────────────
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'TV_SYMBOL_CHANGE' && msg.symbol) {
      const sym = msg.symbol.replace(/[^A-Z0-9.^-]/g, '').toUpperCase() || currentSymbol;
      setSyncStatus('ok', `Auto-sync · ${sym}`);
      if (sym !== currentSymbol) {
        currentSymbol = sym;
        if (symInput) symInput.value = sym;
        loadAndAnalyze();
        if (autoRefresh) startAutoRefresh();
      }
    }
  });

  // ── Sync button ───────────────────────────────────────────────────────────
  document.getElementById('syncBtn')?.addEventListener('click', () => syncFromTV(true));

  function firstLoad() {
    loadAndAnalyze();
    startAutoRefresh();
  }

  // ── Initial load: try to sync with open TV tab ────────────────────────────
  const fallbackTimer = setTimeout(() => {
    if (!candles.length) { setSyncStatus('warn', 'Sin pestaña TV — modo manual'); firstLoad(); }
  }, 4000);

  chrome.tabs.query({ url: '*://*.tradingview.com/*' }, tabs => {
    chrome.tabs.query({ active: true, currentWindow: true }, activeTabs => {
      const activeTab = activeTabs[0];
      const tvTab = (activeTab?.url?.includes('tradingview.com') ? activeTab : null)
                 || tabs.find(t => t.url?.includes('tradingview.com/chart'));

      if (!tvTab) {
        setSyncStatus('warn', 'Sin pestaña TV — escribe símbolo manual');
        clearTimeout(fallbackTimer);
        firstLoad();
        return;
      }

      tvTabId = tvTab.id;
      chrome.tabs.sendMessage(tvTab.id, { type: 'GET_TV_DATA' }, resp => {
        void chrome.runtime.lastError;
        clearTimeout(fallbackTimer);

        if (resp?.symbol) {
          currentSymbol = resp.symbol.replace(/[^A-Z0-9.^-]/g, '').toUpperCase() || 'BTCUSDT';
          currentTf     = resp.timeframe || '15m';
          if (symInput) symInput.value = currentSymbol;
          if (tfSelect) tfSelect.value = currentTf in TF_MAP ? currentTf : '15m';
          setSyncStatus('ok', `Sincronizado · ${currentSymbol} · tab ${tvTab.id}`);
        } else {
          setSyncStatus('warn', `TV detectada · sin datos · tab ${tvTab.id}`);
        }
        firstLoad();
      });
    });

  });

  // Sessions & Calendar toggle
  document.getElementById('snToggleBtn')?.addEventListener('click', () => {
    const d = document.getElementById('snDropdown');
    if (d) {
      const open = d.style.display === 'none';
      d.style.display = open ? 'block' : 'none';
      document.getElementById('snToggleBtn').classList.toggle('sn-open', open);
    }
  });

  // Draw levels buttons — top button + individual →TV buttons (event delegation)
  document.getElementById('drawLevelsBtn')?.addEventListener('click', drawLevels);
  document.getElementById('tab-analysis')?.addEventListener('click', e => {
    const btn = e.target.closest('.btn-tv-line, .btn-tv-all');
    if (!btn) return;
    if (btn.classList.contains('btn-tv-all') || btn.id === 'drawLevelsBtn2') {
      drawLevels(); return;
    }
    const price = parseFloat(btn.dataset.price);
    const label = btn.dataset.label || 'Nivel';
    const color = btn.dataset.color || '#00d4ff';
    if (price > 0) sendToTV([{ price, label, color }]);
  });

  // ── Demo buttons (Analysis tab strip) ────────────────────────────────────
  const getTradeParams = (amtId, tpId, slId, levId) => ({
    usd: parseFloat(document.getElementById(amtId)?.value) || 100,
    tp:  parseFloat(document.getElementById(tpId)?.value)  || 30,
    sl:  parseFloat(document.getElementById(slId)?.value)  || 15,
    lev: parseInt(document.getElementById(levId || 'dTpLev')?.value) || 10,
  });

  // (Analysis tab trade strip removed — trading is now in Demo tab only)

  // ── Demo tab panel buttons ────────────────────────────────────────────────
  document.getElementById('btnDemoLong')?.addEventListener('click', () => {
    const p = getTradeParams('dTpAmount','dTpTp','dTpSl','dTpLev');
    if (!currentPrice) { showNotif('Analiza una moneda primero', 'error'); return; }
    ptBuyDemo(currentSymbol, currentPrice, p.tp, p.sl, p.usd, 'long', p.lev);
  });
  document.getElementById('btnDemoShort')?.addEventListener('click', () => {
    const p = getTradeParams('dTpAmount','dTpTp','dTpSl','dTpLev');
    if (!currentPrice) { showNotif('Analiza una moneda primero', 'error'); return; }
    ptBuyDemo(currentSymbol, currentPrice, p.tp, p.sl, p.usd, 'short', p.lev);
  });

  // Update trade panel info when inputs change
  ['dTpAmount','dTpTp','dTpSl','dTpLev'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener(el.tagName === 'SELECT' ? 'change' : 'input', renderDemo);
  });

  // 1s — re-render Demo timers + analysis + session status bar
  setInterval(() => {
    // Session status bar — always update (uses SESSIONS constant, no API needed)
    renderSessionStatusBar();

    // Update positions display
    if (ptPositions.some(p => p.status === 'open')) renderDemo();

    // Re-run analysis using current WebSocket price on last candle
    if (candles.length && currentPrice > 0) {
      candles[candles.length - 1].close = currentPrice;
      try {
        const a = runAnalysis(candles, currentSymbol, currentTf);
        renderAnalysis(a);
      } catch (_) {}
    }
  }, 1000);

  // Demo reset
  document.getElementById('dResetBtn')?.addEventListener('click', () => {
    const chosen = parseFloat(document.getElementById('dStartBal')?.value) || 10000;
    if (!confirm(`¿Nueva demo con $${chosen}?`)) return;
    ptStart = chosen; ptBalance = chosen; ptPositions = []; ptPeakEq = chosen;
    ptSave(); renderDemo();
    showNotif(`Demo reseteada a $${chosen}`);
  });

  document.getElementById('dExportBtn')?.addEventListener('click', () => {
    const data = { start: ptStart, balance: ptBalance, positions: ptPositions, exportedAt: new Date().toISOString() };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = `tv-demo-${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // Watchlist add
  document.getElementById('wlAddBtn')?.addEventListener('click', () => {
    const val = (document.getElementById('wlInput')?.value || '').trim().toUpperCase();
    if (!val) return;
    const sym = val.includes('USDT') ? val : val + 'USDT';
    if (!watchlist.includes(sym)) {
      watchlist.push(sym);
      chrome.storage.local.set({ tvWatchlist: watchlist });
      refreshWatchlist();
    }
  });

  // ── STARK-OS: Neural Lab init ──────────────────────────────────
  try { initNeuralLab(); } catch(e) { console.warn('[STARK-OS] Neural Lab init error:', e); }
  try { initWhaleTracker(); } catch(e) { console.warn('[STARK-OS] Whale Tracker init error:', e); }

  // 30s REST fallback — reconnects WS if it silently dropped
  setInterval(() => {
    if (!liveWS || liveWS.readyState === WebSocket.CLOSED || liveWS.readyState === WebSocket.CLOSING)
      wsApply();
  }, 30000);

  setInterval(() => { if (document.getElementById('tab-watchlist')?.classList.contains('active')) refreshWatchlist(); }, 30000);

  // Load persisted data
  chrome.storage.local.get(['tvWatchlist'], r => {
    if (r.tvWatchlist?.length) watchlist = r.tvWatchlist;
  });
  ptLoad(() => { renderDemo(); ptUpdateHeaderBal(); wsApply(); });
});
