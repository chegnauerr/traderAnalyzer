// ═══════════════════════════════════════════════════════════════════
// STARK-OS · INFRASTRUCTURE LAYER — Indicators Adapter
// Adaptador compatible con TechnicalIndicators (anandanand84/technicalindicators)
// Implementa IIndicatorsPort con toda la lógica de cálculo extraída de popup.js.
// Capa: infrastructure/indicators.adapter.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

// ── SMA ────────────────────────────────────────────────────────────────────────

function calcSMA(data, period) {
  return data.map((_, i) => {
    if (i < period - 1) return null;
    return data.slice(i - period + 1, i + 1).reduce((s, v) => s + v, 0) / period;
  });
}

// ── EMA ────────────────────────────────────────────────────────────────────────

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

// ── RSI ────────────────────────────────────────────────────────────────────────

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
  while (rsi.length < closes.length) rsi.unshift(null);
  return rsi;
}

// ── MACD ───────────────────────────────────────────────────────────────────────

function calcMACD(closes, fast = 12, slow = 26, sig = 9) {
  const emaFast = calcEMA(closes, fast);
  const emaSlow = calcEMA(closes, slow);
  const macd    = emaFast.map((v, i) => v - emaSlow[i]);
  const signal  = calcEMA(macd, sig);
  const hist    = macd.map((v, i) => v - signal[i]);
  return { macd, signal, hist };
}

// ── Bollinger Bands ────────────────────────────────────────────────────────────

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

// ── ATR ────────────────────────────────────────────────────────────────────────

function calcATR(candles, period = 14) {
  const tr = candles.map((c, i) => {
    if (i === 0) return c.high - c.low;
    const prev = candles[i - 1];
    return Math.max(c.high - c.low, Math.abs(c.high - prev.close), Math.abs(c.low - prev.close));
  });
  return calcSMA(tr, period);
}

// ── Stochastic ─────────────────────────────────────────────────────────────────

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

// ── Swing Detection ────────────────────────────────────────────────────────────

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

// ── Support / Resistance (clustered) ──────────────────────────────────────────

function supportResistance(candles, n = 5) {
  const { highs, lows } = findSwings(candles, n);
  const levels = [];
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

// ── RSI Divergence ─────────────────────────────────────────────────────────────

function detectRSIDivergence(candles, rsi) {
  const n = candles.length;
  const lookback = 30;
  const start    = Math.max(n - lookback, 5);
  const recent   = candles.slice(start);
  const recentRsi = rsi.slice(start);

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

// ─────────────────────────────────────────────────────────────────────────────
// SINGLETON EXPORT — IndicatorsAdapter (implementa IIndicatorsPort)
// ─────────────────────────────────────────────────────────────────────────────

export const IndicatorsAdapter = Object.freeze({
  // Primitivos
  sma:    calcSMA,
  ema:    calcEMA,
  rsi:    calcRSI,
  macd:   calcMACD,
  bb:     calcBB,
  atr:    calcATR,
  stoch:  calcStoch,

  // Análisis estructural
  findSwings,
  supportResistance,
  detectRSIDivergence,
});

export default IndicatorsAdapter;
