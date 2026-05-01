// ═══════════════════════════════════════════════════════════════════
// STARK-OS · DOMAIN LAYER — Entities
// Entidades puras de trading. Sin lógica de negocio ni dependencias externas.
// Capa: domain/entities.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

/**
 * @typedef {{ time: number, open: number, high: number, low: number, close: number, volume: number }} Candle
 * @typedef {{ id: string, symbol: string, blockchain: string, transactionType: string,
 *             from: string, to: string, amount: number, amountUsd: number,
 *             timestamp: number, hash: string }} WhaleEvent
 * @typedef {{ symbol: string, direction: 'LONG'|'SHORT'|'WAIT', winRate: number,
 *             entryPrice: number, tp: number, sl: number, riskReward: number,
 *             confidence: number, source: string, signals: string[],
 *             timestamp: number }} HunterSignal
 * @typedef {{ time: number, open: number, high: number, low: number, close: number,
 *             volume: number, price: number, score: number, signal: string,
 *             signalCls: string, trend: object, patterns: object[], smc: object,
 *             div: object, volume: object, levels: object[], indicators: object,
 *             signals: object[], support: object, resistance: object,
 *             suggestedEntry: number, suggestedSL: number, suggestedTP: number,
 *             riskReward: number, verdict: object, brain: object }} AnalysisResult
 */

/**
 * Factory: crea una entidad Candle normalizada.
 * @param {object} raw
 * @returns {Candle}
 */
export function makeCandle(raw) {
  return {
    time:   Number(raw.time)   || 0,
    open:   Number(raw.open)   || 0,
    high:   Number(raw.high)   || 0,
    low:    Number(raw.low)    || 0,
    close:  Number(raw.close)  || 0,
    volume: Number(raw.volume) || 0,
  };
}

/**
 * Factory: crea un evento de ballena (WhaleEvent).
 * @param {object} raw
 * @returns {WhaleEvent}
 */
export function makeWhaleEvent(raw) {
  return {
    id:              String(raw.id              || `whale_${Date.now()}_${Math.random()}`),
    symbol:          String(raw.symbol          || 'BTCUSDT'),
    blockchain:      String(raw.blockchain      || 'BITCOIN'),
    transactionType: String(raw.transactionType || 'transfer'),
    from:            String(raw.from            || 'unknown'),
    to:              String(raw.to              || 'unknown'),
    amount:          Number(raw.amount          || 0),
    amountUsd:       Number(raw.amountUsd       || 0),
    timestamp:       Number(raw.timestamp       || Date.now()),
    hash:            String(raw.hash            || ''),
  };
}

/**
 * Factory: crea una señal del Agente Hunter.
 * @param {object} raw
 * @returns {HunterSignal}
 */
export function makeHunterSignal(raw) {
  return {
    symbol:      String(raw.symbol      || 'BTCUSDT'),
    direction:   String(raw.direction   || 'WAIT'),
    winRate:     Number(raw.winRate     || 0),
    entryPrice:  Number(raw.entryPrice  || 0),
    tp:          Number(raw.tp          || 0),
    sl:          Number(raw.sl          || 0),
    riskReward:  Number(raw.riskReward  || 0),
    confidence:  Number(raw.confidence  || 0),
    source:      String(raw.source      || 'hunter'),
    signals:     Array.isArray(raw.signals) ? raw.signals : [],
    timestamp:   Number(raw.timestamp   || Date.now()),
  };
}

/**
 * Constantes del sistema STARK-OS.
 */
export const STARK_CONSTRAINTS = {
  BUDGET_BASE:          100,
  RISK_PER_TRADE_MAX:   0.10,
  DEFAULT_TIMEFRAME:    '1m',
  WHALE_BTC_MIN:        1,
  WHALE_BTC_MAX:        10,
  HUNTER_WIN_RATE_MIN:  0.70,
  FORENSIC_LOOKBACK_M:  30,   // minutos hacia atrás para análisis forense
  SCIENTIST_MATCHES:    20,   // número de trades similares a buscar
  SCIENTIST_WR_SAVE:    0.65, // win-rate mínimo para guardar NeuralPattern
};

/**
 * Factory: crea un informe forense de una ballena (ForensicReport).
 * @param {object} raw
 * @returns {ForensicReport}
 */
export function makeForensicReport(raw) {
  return {
    whaleId:          String(raw.whaleId          || ''),
    whaleFingerprint: String(raw.whaleFingerprint  || ''),  // ID sintético estable
    symbol:           String(raw.symbol            || 'BTCUSDT'),
    timestamp:        Number(raw.timestamp         || Date.now()),
    // Indicadores en el momento del trade
    rsiAtTrade:     raw.rsiAtTrade   != null ? Number(raw.rsiAtTrade)  : null,
    emaPos:         String(raw.emaPos          || 'unknown'),
    volumeDelta:    Number(raw.volumeDelta     || 0),
    volRatio:       Number(raw.volRatio        || 1),
    hasDivergence:  Boolean(raw.hasDivergence  || false),
    divergenceType: raw.divergenceType || null,
    priceVsLevel:   String(raw.priceVsLevel    || 'mid-range'),
    // Análisis de modus operandi
    mo:             String(raw.mo              || 'unknown'),
    moLabel:        String(raw.moLabel         || ''),
    moDescription:  String(raw.moDescription   || ''),
    moIcon:         String(raw.moIcon          || '🔍'),
    confidence:     Number(raw.confidence      || 0),
    signals:        Array.isArray(raw.signals) ? raw.signals : [],
    // Predicción a 15m
    predictedDir:      raw.predictedDir || null,
    predictedStrength: Number(raw.predictedStrength || 0),
  };
}

/** Modus Operandi conocidos del Hunter Forensic */
export const MO_TYPES = {
  SMART_MONEY_BUY:   'smart_money_buy',
  SMART_MONEY_SELL:  'smart_money_sell',
  DISTRIBUTION:      'distribution',
  ACCUMULATION:      'accumulation',
  PANIC_SELL:        'panic_sell',
  LIQUIDITY_GRAB:    'liquidity_grab',
  STOP_HUNT:         'stop_hunt',
  UNKNOWN:           'unknown',
};

/**
 * Factory: crea un WalletProfile — historial acumulado por fingerprint.
 * Se actualiza cada vez que el ForensicAgent analiza el mismo cluster.
 * @param {object} raw
 * @returns {WalletProfile}
 */
export function makeWalletProfile(raw) {
  return {
    fingerprint:   String(raw.fingerprint   || ''),
    symbol:        String(raw.symbol        || 'BTCUSDT'),
    label:         String(raw.label         || 'Ballena Anónima'),
    firstSeen:     Number(raw.firstSeen     || Date.now()),
    lastSeen:      Number(raw.lastSeen      || Date.now()),
    tradeCount:    Number(raw.tradeCount    || 1),
    totalBtc:      Number(raw.totalBtc      || 0),
    winCount:      Number(raw.winCount      || 0),
    lossCount:     Number(raw.lossCount     || 0),
    winRate:       Number(raw.winRate       || 0),    // 0-100
    // Patrón dominante de entrada
    dominantMO:    String(raw.dominantMO    || 'unknown'),
    dominantSide:  String(raw.dominantSide  || 'unknown'), // 'buy'|'sell'|'mixed'
    avgRsiEntry:   raw.avgRsiEntry  != null ? Number(raw.avgRsiEntry) : null,
    avgVolRatio:   Number(raw.avgVolRatio   || 1),
    // Historial de MOs observados
    moHistory:     Array.isArray(raw.moHistory) ? raw.moHistory : [],
    // Historial de precios de entrada
    priceHistory:  Array.isArray(raw.priceHistory) ? raw.priceHistory : [],
  };
}

/**
 * Factory: crea un resultado de backtesting del Scientist Agent.
 * @param {object} raw
 * @returns {BacktestResult}
 */
export function makeBacktestResult(raw) {
  return {
    symbol:         String(raw.symbol          || 'BTCUSDT'),
    matchCount:     Number(raw.matchCount       || 0),
    winCount:       Number(raw.winCount         || 0),
    lossCount:      Number(raw.lossCount        || 0),
    winRate:        Number(raw.winRate          || 0),   // 0-100
    avgDrawdown:    Number(raw.avgDrawdown      || 0),   // % máximo en contra
    avgTimeToTpMin: Number(raw.avgTimeToTpMin   || 0),   // minutos promedio al TP
    avgRR:          Number(raw.avgRR            || 0),
    matches:        Array.isArray(raw.matches) ? raw.matches : [],
    // Para el mini-chart SVG
    chartPoints:    Array.isArray(raw.chartPoints) ? raw.chartPoints : [],
    generatedAt:    Number(raw.generatedAt      || Date.now()),
  };
}

/**
 * Factory: crea un NeuralPattern (patrón guardado automáticamente).
 * @param {object} raw
 * @returns {NeuralPattern}
 */
export function makeNeuralPattern(raw) {
  return {
    id:             String(raw.id              || `np_${Date.now()}`),
    symbol:         String(raw.symbol          || 'BTCUSDT'),
    name:           String(raw.name            || 'Unknown Pattern'),
    mo:             String(raw.mo              || MO_TYPES.UNKNOWN),
    direction:      String(raw.direction       || 'WAIT'),
    winRate:        Number(raw.winRate         || 0),
    matchCount:     Number(raw.matchCount      || 0),
    avgRR:          Number(raw.avgRR           || 0),
    // Condiciones del patrón para detección automática
    conditions: {
      rsiRange:     raw.conditions?.rsiRange   || [0, 100],
      minVolRatio:  raw.conditions?.minVolRatio|| 1.5,
      emaPos:       raw.conditions?.emaPos     || null,
      priceVsLevel: raw.conditions?.priceVsLevel || null,
      hasDivergence:raw.conditions?.hasDivergence || false,
    },
    createdAt:      Number(raw.createdAt       || Date.now()),
    lastSeenAt:     Number(raw.lastSeenAt      || Date.now()),
    triggerCount:   Number(raw.triggerCount    || 0),
  };
}

