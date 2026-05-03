// ═══════════════════════════════════════════════════════════════════
// STARK-OS · APPLICATION LAYER — Hunter Agent
// Agente orquestador que detecta ballenas de BTC (1-10 BTC) y
// genera señales de trading con win-rate estimado > 70%.
//
// Algoritmo:
//  1. Fetch 100 velas 1m (BinanceAdapter)
//  2. Fetch aggTrades recientes → detectar ballenas 1-10 BTC (WhaleAdapter)
//  3. Calcular indicadores locales: RSI, S/R, EMA (IndicatorsAdapter)
//  4. Puntuar confluencia de señales → win-rate estimado
//  5. Emitir HunterSignal si winRate > HUNTER_WIN_RATE_MIN
//  6. Persistir en StorageAdapter
//
// Capa: application/hunter.agent.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

import { BinanceAdapter }               from '../infrastructure/binance.adapter.js';
import { WhaleAdapter, analyzeWhalePressure } from '../infrastructure/whale.adapter.js';
import { IndicatorsAdapter }            from '../infrastructure/indicators.adapter.js';
import { StorageAdapter }               from '../infrastructure/storage.adapter.js';
import { makeHunterSignal, STARK_CONSTRAINTS } from '../domain/entities.js';

const { WHALE_BTC_MIN, WHALE_BTC_MAX, HUNTER_WIN_RATE_MIN } = STARK_CONSTRAINTS;

// ── Helpers ────────────────────────────────────────────────────────────────────

function fmtPrice(n) {
  if (!n || isNaN(n)) return '—';
  if (n >= 10000) return n.toLocaleString('en', { maximumFractionDigits: 0 });
  if (n >= 100)   return n.toFixed(2);
  if (n >= 1)     return n.toFixed(4);
  return n.toFixed(6);
}

// ── Scoring Engine ─────────────────────────────────────────────────────────────

/**
 * Calcula el win-rate estimado basado en confluencia de señales.
 * Base matemática: cada factor independiente eleva la probabilidad.
 *
 * @param {object} params
 * @returns {{ winRate: number, direction: 'LONG'|'SHORT'|'WAIT', signals: string[], confidence: number }}
 */
function scoreConfluence({ rsiNow, ema20, ema50, price, whalePressure, srLevels, atrNow }) {
  let bullScore = 0;
  let bearScore = 0;
  const signals = [];

  // ── RSI (peso: 25 puntos) ──────────────────────────────────────────
  if (rsiNow !== null) {
    if (rsiNow < 30) {
      bullScore += 25;
      signals.push(`🟢 RSI sobrevendido (${rsiNow.toFixed(0)}) — rebote esperado`);
    } else if (rsiNow < 40) {
      bullScore += 12;
      signals.push(`🟡 RSI zona baja (${rsiNow.toFixed(0)}) — sesgo alcista moderado`);
    } else if (rsiNow > 70) {
      bearScore += 25;
      signals.push(`🔴 RSI sobrecomprado (${rsiNow.toFixed(0)}) — rechazo esperado`);
    } else if (rsiNow > 60) {
      bearScore += 12;
      signals.push(`🟠 RSI zona alta (${rsiNow.toFixed(0)}) — sesgo bajista moderado`);
    }
  }

  // ── EMA Trend (peso: 20 puntos) ────────────────────────────────────
  if (ema20 && ema50) {
    if (price > ema20 && ema20 > ema50) {
      bullScore += 20;
      signals.push('🟢 Precio sobre EMA20 > EMA50 — tendencia alcista');
    } else if (price < ema20 && ema20 < ema50) {
      bearScore += 20;
      signals.push('🔴 Precio bajo EMA20 < EMA50 — tendencia bajista');
    } else if (price > ema20) {
      bullScore += 8;
      signals.push('🟡 Precio sobre EMA20 — sesgo alcista débil');
    } else {
      bearScore += 8;
      signals.push('🟠 Precio bajo EMA20 — sesgo bajista débil');
    }
  }

  // ── Whale Pressure (peso: 30 puntos — señal más potente) ──────────
  if (whalePressure.buyBtc > 0 || whalePressure.sellBtc > 0) {
    const total = whalePressure.buyBtc + whalePressure.sellBtc;
    if (whalePressure.bias === 'buy') {
      const pct = Math.round(whalePressure.buyBtc / total * 100);
      bullScore += 30;
      signals.push(`🐋 Acumulación de ballenas: ${whalePressure.buyBtc.toFixed(1)} BTC compra (${pct}%) · ${whalePressure.buyCount} trades`);
    } else if (whalePressure.bias === 'sell') {
      const pct = Math.round(whalePressure.sellBtc / total * 100);
      bearScore += 30;
      signals.push(`🐋 Distribución de ballenas: ${whalePressure.sellBtc.toFixed(1)} BTC venta (${pct}%) · ${whalePressure.sellCount} trades`);
    } else {
      signals.push(`⚖️ Ballenas neutrales: ${whalePressure.buyBtc.toFixed(1)}B / ${whalePressure.sellBtc.toFixed(1)}S BTC`);
    }
  } else {
    signals.push('⚪ Sin actividad de ballenas detectada en este intervalo');
  }

  // ── Support/Resistance Proximity (peso: 15 puntos) ─────────────────
  const supports    = srLevels.filter(l => l.price < price && l.type === 'support');
  const resistances = srLevels.filter(l => l.price > price && l.type === 'resistance');
  const nearSup = supports.length > 0
    ? supports.reduce((a, b) => Math.abs(a.price - price) < Math.abs(b.price - price) ? a : b)
    : null;
  const nearRes = resistances.length > 0
    ? resistances.reduce((a, b) => Math.abs(a.price - price) < Math.abs(b.price - price) ? a : b)
    : null;

  if (nearSup) {
    const distPct = Math.abs(price - nearSup.price) / price * 100;
    if (distPct < 0.5) {
      bullScore += 15;
      signals.push(`🟢 Precio en soporte clave $${fmtPrice(nearSup.price)} (${distPct.toFixed(2)}% distancia)`);
    } else if (distPct < 1.5) {
      bullScore += 7;
      signals.push(`🟡 Soporte próximo $${fmtPrice(nearSup.price)} (${distPct.toFixed(2)}%)`);
    }
  }
  if (nearRes) {
    const distPct = Math.abs(nearRes.price - price) / price * 100;
    if (distPct < 0.5) {
      bearScore += 15;
      signals.push(`🔴 Precio en resistencia clave $${fmtPrice(nearRes.price)} (${distPct.toFixed(2)}% distancia)`);
    } else if (distPct < 1.5) {
      bearScore += 7;
      signals.push(`🟠 Resistencia próxima $${fmtPrice(nearRes.price)} (${distPct.toFixed(2)}%)`);
    }
  }

  // ── ATR volatility filter (peso: 10 puntos) ────────────────────────
  if (atrNow && price) {
    const atrPct = atrNow / price * 100;
    if (atrPct > 0.3 && atrPct < 2.0) {
      // Volatilidad saludable — ATR razonable para tener buen R:R
      const dominant = bullScore >= bearScore ? bullScore : bearScore;
      if (bullScore >= bearScore) bullScore += 10;
      else                        bearScore += 10;
      signals.push(`⚡ Volatilidad ATR ${atrPct.toFixed(2)}% — favorable para el trade`);
    } else if (atrPct >= 2.0) {
      signals.push(`⚠️ Alta volatilidad ATR ${atrPct.toFixed(2)}% — riesgo elevado`);
    }
  }

  // ── Calcular win-rate ──────────────────────────────────────────────
  const total       = bullScore + bearScore || 1;
  const dominant    = Math.max(bullScore, bearScore);
  const rawWinRate  = dominant / total; // 0.5 – 1.0
  // Escala: 50% raw → 50% WR, 80% raw → 80% WR
  const winRate     = Math.min(0.95, Math.max(0.50, rawWinRate));

  const direction = bullScore > bearScore
    ? 'LONG'
    : bearScore > bullScore
      ? 'SHORT'
      : 'WAIT';

  const confidence = Math.min(96, Math.round(winRate * 100));

  return { winRate, direction, signals, confidence, bullScore, bearScore };
}

// ── Entry / SL / TP ────────────────────────────────────────────────────────────

function computeSetup({ price, direction, atrNow, srLevels }) {
  const atrVal = atrNow || price * 0.005;
  const supports    = srLevels.filter(l => l.price < price * 0.9998 && l.type === 'support');
  const resistances = srLevels.filter(l => l.price > price * 1.0002 && l.type === 'resistance');
  const nearSup = supports.length > 0
    ? supports.reduce((a, b) => Math.abs(a.price - price) < Math.abs(b.price - price) ? a : b)
    : null;
  const nearRes = resistances.length > 0
    ? resistances.reduce((a, b) => Math.abs(a.price - price) < Math.abs(b.price - price) ? a : b)
    : null;

  let entry, sl, tp;
  if (direction === 'LONG') {
    entry = nearSup ? Math.min(nearSup.price * 1.001, price) : price;
    sl    = nearSup ? nearSup.price * 0.994 : price - atrVal * 2;
    tp    = nearRes ? nearRes.price * 0.998 : price + atrVal * 4;
  } else if (direction === 'SHORT') {
    entry = nearRes ? Math.max(nearRes.price * 0.999, price) : price;
    sl    = nearRes ? nearRes.price * 1.006 : price + atrVal * 2;
    tp    = nearSup ? nearSup.price * 1.001 : price - atrVal * 4;
  } else {
    entry = price; sl = price; tp = price;
  }

  const riskReward = sl !== entry
    ? Math.abs(tp - entry) / Math.abs(sl - entry)
    : 0;

  return { entryPrice: entry, sl, tp, riskReward: parseFloat(riskReward.toFixed(2)) };
}

// ── Main Hunter Scan ───────────────────────────────────────────────────────────

/**
 * Escanea ballenas de BTC y genera señales de trading.
 *
 * @param {string} [symbol='BTCUSDT']
 * @returns {Promise<{ signals: HunterSignal[], whales: WhaleEvent[], scanTime: number }>}
 */
async function scan(symbol = 'BTCUSDT') {
  const scanTime = Date.now();

  // 1. Datos de mercado en paralelo
  const [candles, whales] = await Promise.all([
    BinanceAdapter.fetchKlines(symbol, '1m', 100),
    WhaleAdapter.getRecentWhales(symbol, WHALE_BTC_MIN, WHALE_BTC_MAX, 1000),
  ]);

  if (!candles || candles.length < 50) {
    return { signals: [], whales: [], scanTime };
  }

  // 2. Indicadores técnicos
  const closes  = candles.map(c => c.close);
  const n       = closes.length - 1;
  const price   = closes[n];

  const rsiArr  = IndicatorsAdapter.rsi(closes, 14);
  const ema20   = IndicatorsAdapter.ema(closes, 20);
  const ema50   = IndicatorsAdapter.ema(closes, 50);
  const atrArr  = IndicatorsAdapter.atr(candles, 14);
  const srLevels = IndicatorsAdapter.supportResistance(candles, 5);

  const rsiNow  = rsiArr[n] ?? null;
  const ema20Now = ema20[n];
  const ema50Now = ema50[n];
  const atrNow  = atrArr[n] ?? price * 0.005;

  // 3. Análisis de presión de ballenas
  const whalePressure = WhaleAdapter.analyzeWhalePressure(whales);

  // 4. Scoring de confluencia
  const scoring = scoreConfluence({
    rsiNow, ema20: ema20Now, ema50: ema50Now,
    price, whalePressure, srLevels, atrNow,
  });

  // 5. Filtro de win-rate mínimo
  const passesThreshold = scoring.winRate >= HUNTER_WIN_RATE_MIN
    && scoring.direction !== 'WAIT';

  if (!passesThreshold) {
    return { signals: [], whales, scanTime, scoring };
  }

  // 6. Calcular setup (Entry / SL / TP)
  const setup = computeSetup({
    price, direction: scoring.direction, atrNow, srLevels,
  });

  // 7. Crear señal
  const signal = makeHunterSignal({
    symbol,
    direction:  scoring.direction,
    winRate:    parseFloat((scoring.winRate * 100).toFixed(1)),
    entryPrice: setup.entryPrice,
    tp:         setup.tp,
    sl:         setup.sl,
    riskReward: setup.riskReward,
    confidence: scoring.confidence,
    source:     'hunter-whale-rsi',
    signals:    scoring.signals,
    timestamp:  scanTime,
  });

  // 8. Persistir en storage
  try {
    await StorageAdapter.appendHunterSignal(signal);
  } catch (e) {
    console.warn('[HunterAgent] storage error:', e);
  }

  return { signals: [signal], whales, scanTime, scoring };
}

/**
 * Recupera el historial persistido de señales.
 * @returns {Promise<HunterSignal[]>}
 */
async function getHistory() {
  return StorageAdapter.loadHunterHistory();
}

/**
 * Limpia el historial del Hunter.
 * @returns {Promise<void>}
 */
async function clearHistory() {
  return StorageAdapter.clearHunterHistory();
}

// ─────────────────────────────────────────────────────────────────────────────
// SINGLETON EXPORT — HunterAgent
// ─────────────────────────────────────────────────────────────────────────────

export const HunterAgent = Object.freeze({
  scan,
  getHistory,
  clearHistory,
  analyzeWhalePressure: WhaleAdapter.analyzeWhalePressure,
});

export default HunterAgent;
