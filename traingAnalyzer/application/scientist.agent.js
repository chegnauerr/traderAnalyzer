// ═══════════════════════════════════════════════════════════════════
// STARK-OS · APPLICATION LAYER — Scientist Agent
// Backtesting forense: busca en el historial de klines clusters de
// volumen similares al trade de ballena, calcula drawdown/TP,
// genera un mini-chart SVG y guarda NeuralPattern si WR > 65%.
//
// Algoritmo:
//  1. Fetch últimas 500 velas 1m (Binance)
//  2. Calcular RSI / EMA20 / Vol-ratio para toda la serie
//  3. Buscar clusters de volumen similares (±30%) en condiciones
//     técnicas parecidas (RSI ±10, misma zona EMA)
//  4. Para cada match: medir precio a +5m, +15m, +30m
//  5. Determinar si fue WIN (precio en dirección del whale >+0.3%)
//     o LOSS (precio en contra >-0.3%) o NEUTRAL
//  6. Calcular drawdown máximo (peor momento antes del TP)
//  7. Generar SVG mini-chart con los 20 casos
//  8. Si WR ≥ 65%: hacer makeNeuralPattern y persistir en storage
//
// Capa: application/scientist.agent.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

import { BinanceAdapter }    from '../infrastructure/binance.adapter.js';
import { IndicatorsAdapter } from '../infrastructure/indicators.adapter.js';
import { StorageAdapter }    from '../infrastructure/storage.adapter.js';
import {
  makeBacktestResult,
  makeNeuralPattern,
  MO_TYPES,
  STARK_CONSTRAINTS,
} from '../domain/entities.js';

const { SCIENTIST_MATCHES, SCIENTIST_WR_SAVE } = STARK_CONSTRAINTS;

// ── Helpers ────────────────────────────────────────────────────────────────────

const fmtP = n => {
  if (!n || isNaN(n)) return '—';
  if (n >= 10000) return n.toLocaleString('en', { maximumFractionDigits: 0 });
  if (n >= 100)   return n.toFixed(2);
  return n.toFixed(4);
};

/**
 * Busca clusters de volumen similares en la serie de klines.
 * Un "cluster similar" es una vela donde:
 *   - El volumen en BTC ≈ targetBtc (±30%)
 *   - El RSI ≈ targetRsi (±12 puntos)
 *   - Misma posición EMA (alcista o bajista)
 *
 * @param {Candle[]} candles
 * @param {number[]} rsiArr
 * @param {number[]} ema20Arr
 * @param {number} targetBtc - Cantidad en BTC del trade de ballena
 * @param {number} btcPrice  - Precio actual de BTC para convertir USD→BTC
 * @param {number} targetRsi - RSI en el momento del trade original
 * @param {string} targetEmaPos - 'above_ema20'|'below_ema20'|etc.
 * @param {number} maxMatches
 * @returns {number[]} - Índices de velas que hacen match
 */
function findSimilarClusters(candles, rsiArr, ema20Arr, ema50Arr, targetBtc, btcPrice, targetRsi, targetEmaPos, maxMatches = 20) {
  const matches = [];
  // Dejar las últimas 30 velas libres para medir el outcome
  const scanEnd = candles.length - 31;

  for (let i = 14; i < scanEnd; i++) {
    const c      = candles[i];
    const volBtc = btcPrice > 0 ? (c.volume * c.close) / btcPrice : c.volume;
    const rsi    = rsiArr[i] ?? 50;
    const ema20  = ema20Arr[i];
    const ema50  = ema50Arr[i];

    // Filtro de volumen BTC (±30%)
    if (targetBtc > 0) {
      const ratio = volBtc / targetBtc;
      if (ratio < 0.7 || ratio > 1.3) continue;
    }

    // Filtro de RSI (±12 puntos)
    if (targetRsi !== null && Math.abs(rsi - targetRsi) > 12) continue;

    // Filtro de posición EMA (mismo régimen: alcista / bajista)
    const isAbove = c.close > ema20;
    const targetAbove = targetEmaPos?.startsWith('above') ?? true;
    if (isAbove !== targetAbove) continue;

    matches.push(i);
    if (matches.length >= maxMatches) break;
  }
  return matches;
}

/**
 * Mide el outcome de un trade dado el índice de entrada.
 * WIN = precio subió/bajó >+0.3% en la dirección esperada dentro de 30 velas
 * LOSS = precio se movió >-0.3% en contra sin recuperarse
 *
 * @param {Candle[]} candles
 * @param {number} entryIdx
 * @param {string} direction - 'buy'|'sell' (dirección del whale)
 * @returns {{ outcome: 'win'|'loss'|'neutral', maxDrawdown: number, timeToTpMin: number, exitPrice: number }}
 */
function measureOutcome(candles, entryIdx, direction) {
  const entry    = candles[entryIdx].close;
  const horizon  = Math.min(30, candles.length - entryIdx - 1);
  const isBuy    = direction === 'buy';
  const TP_PCT   = 0.003; // +0.3%
  const SL_PCT   = 0.003; // -0.3%

  let maxDrawdown  = 0;
  let timeToTpMin  = horizon;
  let outcome      = 'neutral';
  let exitPrice    = candles[entryIdx + horizon]?.close ?? entry;

  for (let k = 1; k <= horizon; k++) {
    if (entryIdx + k >= candles.length) break;
    const c = candles[entryIdx + k];
    // Ganancia en dirección del whale
    const gain = isBuy
      ? (c.high - entry) / entry
      : (entry - c.low)  / entry;
    // Drawdown contra la dirección del whale
    const dd = isBuy
      ? (entry - c.low)  / entry
      : (c.high - entry) / entry;

    if (dd > maxDrawdown) maxDrawdown = dd;

    if (gain >= TP_PCT && outcome !== 'win') {
      outcome     = 'win';
      timeToTpMin = k;
      exitPrice   = isBuy ? c.high : c.low;
      break;
    }
    if (dd >= SL_PCT && outcome === 'neutral') {
      outcome   = 'loss';
      exitPrice = isBuy ? c.low : c.high;
    }
  }

  return {
    outcome,
    maxDrawdown: parseFloat((maxDrawdown * 100).toFixed(3)),
    timeToTpMin,
    exitPrice,
    entryPrice: entry,
  };
}

// ── SVG Mini-Chart Generator ───────────────────────────────────────────────────

/**
 * Genera un SVG compacto (240×80px) con los 20 casos de backtesting.
 * Cada caso es una pequeña línea de precio con un punto de entrada (⬤)
 * y un punto de salida (▲WIN/▼LOSS).
 *
 * @param {Array<{entryPrice,exitPrice,outcome,candles}>} matches
 * @returns {string} - SVG HTML string
 */
function generateMiniChartSVG(matches) {
  const W  = 240;
  const H  = 80;
  const M  = 6;   // margin
  const CH = H - M * 2;
  const CW = W - M * 2;

  if (!matches.length) return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><text x="${W/2}" y="${H/2}" text-anchor="middle" fill="#555" font-size="8">Sin datos</text></svg>`;

  // Normalizar todos los precios en rango 0-1
  const allPrices = matches.flatMap(m => [m.entryPrice, m.exitPrice]);
  const minP = Math.min(...allPrices);
  const maxP = Math.max(...allPrices) || minP + 1;
  const norm = p => 1 - (p - minP) / (maxP - minP); // invertido (Y crece hacia abajo)

  const colWidth  = CW / matches.length;
  const winCol    = '#00ff41';
  const lossCol   = '#f85149';
  const neutralCl = '#e3b341';

  const lines  = [];
  const dots   = [];

  matches.forEach((m, i) => {
    const x1 = M + i * colWidth + colWidth * 0.2;
    const x2 = M + i * colWidth + colWidth * 0.8;
    const y1 = M + norm(m.entryPrice) * CH;
    const y2 = M + norm(m.exitPrice)  * CH;
    const col = m.outcome === 'win' ? winCol : m.outcome === 'loss' ? lossCol : neutralCl;

    lines.push(`<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${col}" stroke-width="1.2" stroke-opacity="0.7"/>`);
    // Entry dot
    dots.push(`<circle cx="${x1.toFixed(1)}" cy="${y1.toFixed(1)}" r="1.8" fill="${col}" opacity="0.9"/>`);
    // Exit marker
    const exitSym = m.outcome === 'win' ? '▲' : m.outcome === 'loss' ? '▼' : '●';
    dots.push(`<text x="${x2.toFixed(1)}" y="${(y2 + 1).toFixed(1)}" text-anchor="middle" font-size="5" fill="${col}" opacity="0.9">${exitSym}</text>`);
  });

  // Eje horizontal central
  const midY = (M + (M + CH)) / 2;
  const grid = `<line x1="${M}" y1="${midY}" x2="${W-M}" y2="${midY}" stroke="rgba(255,255,255,0.06)" stroke-width="0.5" stroke-dasharray="2,2"/>`;

  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" class="nl-mini-chart-svg">
  <rect width="${W}" height="${H}" fill="rgba(0,0,0,0)" rx="4"/>
  ${grid}
  ${lines.join('\n  ')}
  ${dots.join('\n  ')}
  <text x="${M}" y="${H-1}" font-size="6" fill="rgba(255,255,255,0.25)" font-family="monospace">
    ${matches.length} casos · ▲WIN  ▼LOSS  ●NEUTRAL
  </text>
</svg>`;
}

// ── Main Backtest Function ─────────────────────────────────────────────────────

/**
 * Realiza un backtesting forense buscando trades similares al whale en
 * el historial de klines y calcula estadísticas de WR / Drawdown / TP.
 *
 * @param {import('../domain/entities.js').WhaleEvent} whaleEvent
 * @param {import('../domain/entities.js').ForensicReport} forensicReport - Salida del ForensicAgent
 * @param {string} [symbol='BTCUSDT']
 * @returns {Promise<import('../domain/entities.js').BacktestResult>}
 */
export async function backTestWalletCluster(whaleEvent, forensicReport, symbol = 'BTCUSDT') {
  // 1. Klines + BTC price
  const [candles, btcPrice] = await Promise.all([
    BinanceAdapter.fetchKlines(symbol, '1m', 500),
    BinanceAdapter.getBtcPrice(),
  ]);

  if (!candles || candles.length < 50) {
    return makeBacktestResult({ symbol, matchCount: 0 });
  }

  const closes  = candles.map(c => c.close);
  const rsiArr  = IndicatorsAdapter.rsi(closes, 14);
  const ema20Arr= IndicatorsAdapter.ema(closes, 20);
  const ema50Arr= IndicatorsAdapter.ema(closes, 50);

  // 2. Parámetros del whale target
  const targetBtc  = whaleEvent.amount;
  const targetRsi  = forensicReport.rsiAtTrade;
  const targetEmaPos = forensicReport.emaPos;
  const whaleSide  = whaleEvent.transactionType; // 'buy'|'sell'

  // 3. Encontrar clusters similares
  const matchIdxs = findSimilarClusters(
    candles, rsiArr, ema20Arr, ema50Arr,
    targetBtc, btcPrice, targetRsi, targetEmaPos,
    SCIENTIST_MATCHES,
  );

  if (!matchIdxs.length) {
    return makeBacktestResult({ symbol, matchCount: 0 });
  }

  // 4. Medir outcome de cada match
  const matchResults = matchIdxs.map(idx => {
    const outcome = measureOutcome(candles, idx, whaleSide);
    return {
      ...outcome,
      candleIdx: idx,
      candleTime: candles[idx].time,
      // Para el SVG chart guardamos las velas del contexto (±5)
      ctxCandles: candles.slice(Math.max(0, idx - 3), idx + 8),
    };
  });

  // 5. Estadísticas
  const wins   = matchResults.filter(r => r.outcome === 'win');
  const losses = matchResults.filter(r => r.outcome === 'loss');
  const winRate = matchResults.length
    ? parseFloat((wins.length / matchResults.length * 100).toFixed(1))
    : 0;

  const avgDrawdown = matchResults.length
    ? parseFloat((matchResults.reduce((s, r) => s + r.maxDrawdown, 0) / matchResults.length).toFixed(3))
    : 0;

  const avgTimeToTp = wins.length
    ? parseFloat((wins.reduce((s, r) => s + r.timeToTpMin, 0) / wins.length).toFixed(1))
    : 0;

  // R:R estimado: TP = 0.3%, SL = 0.3% pero ajustado por drawdown real
  const avgRR = avgDrawdown > 0
    ? parseFloat((0.3 / avgDrawdown).toFixed(2))
    : 0;

  // 6. Preparar datos para el mini-chart SVG
  const chartPoints = matchResults.map(r => ({
    entryPrice: r.entryPrice,
    exitPrice:  r.exitPrice,
    outcome:    r.outcome,
  }));
  const svgChart = generateMiniChartSVG(chartPoints);

  // 7. Guardar NeuralPattern si WR es suficiente
  let savedPattern = null;
  if (winRate >= SCIENTIST_WR_SAVE * 100 && matchResults.length >= 5) {
    savedPattern = await saveNeuralPattern({
      symbol, whaleSide, forensicReport, winRate, matchResults, avgRR,
    });
  }

  return makeBacktestResult({
    symbol,
    matchCount:     matchResults.length,
    winCount:       wins.length,
    lossCount:      losses.length,
    winRate,
    avgDrawdown,
    avgTimeToTpMin: avgTimeToTp,
    avgRR,
    matches:        matchResults.map(({ entryPrice, exitPrice, outcome, candleTime, maxDrawdown, timeToTpMin }) =>
      ({ entryPrice, exitPrice, outcome, candleTime, maxDrawdown, timeToTpMin })
    ),
    chartPoints,
    svgChart,          // Campo extra (no en el factory, pero se pasa por spread)
    savedPattern,      // null o el NeuralPattern guardado
    generatedAt: Date.now(),
  });
}

// ── NeuralPattern persistence ──────────────────────────────────────────────────

async function saveNeuralPattern({ symbol, whaleSide, forensicReport, winRate, matchResults, avgRR }) {
  const pattern = makeNeuralPattern({
    symbol,
    name:      `${forensicReport.moLabel} · ${whaleSide.toUpperCase()} ${symbol}`,
    mo:        forensicReport.mo,
    direction: whaleSide === 'buy' ? 'LONG' : 'SHORT',
    winRate,
    matchCount: matchResults.length,
    avgRR,
    conditions: {
      rsiRange:     forensicReport.rsiAtTrade !== null
        ? [Math.max(0, forensicReport.rsiAtTrade - 10), Math.min(100, forensicReport.rsiAtTrade + 10)]
        : [0, 100],
      minVolRatio:  forensicReport.volRatio > 0 ? Math.max(1.0, forensicReport.volRatio * 0.8) : 1.5,
      emaPos:       forensicReport.emaPos,
      priceVsLevel: forensicReport.priceVsLevel,
      hasDivergence:forensicReport.hasDivergence,
    },
  });

  try {
    await StorageAdapter.saveNeuralPattern(pattern);
    console.log(`[ScientistAgent] NeuralPattern guardado: "${pattern.name}" WR=${winRate}%`);
  } catch (e) {
    console.warn('[ScientistAgent] Error guardando NeuralPattern:', e);
  }
  return pattern;
}

// ─────────────────────────────────────────────────────────────────────────────
// SINGLETON EXPORT — ScientistAgent
// ─────────────────────────────────────────────────────────────────────────────

export const ScientistAgent = Object.freeze({
  backTestWalletCluster,
  generateMiniChartSVG,
  measureOutcome,
  findSimilarClusters,
});

export default ScientistAgent;
