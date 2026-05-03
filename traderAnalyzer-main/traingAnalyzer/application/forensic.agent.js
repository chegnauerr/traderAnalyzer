// ═══════════════════════════════════════════════════════════════════
// STARK-OS · APPLICATION LAYER — Forensic Agent
// Realiza ingeniería inversa de un trade de ballena para identificar
// su Modus Operandi: Smart Money, Distribución, Panic Sell, etc.
//
// Algoritmo:
//  1. Obtiene las velas 1m alrededor del timestamp del trade
//  2. Localiza la vela exacta del trade por timestamp
//  3. Calcula indicadores en ese instante: RSI, EMA20/50, S/R proximity
//  4. Analiza el delta de volumen de esa vela (buyVol vs sellVol estimado)
//  5. Detecta divergencia RSI en ventana ±5 velas
//  6. Clasifica el Modus Operandi con un sistema de puntuación
//  7. Retorna un ForensicReport completo
//
// Capa: application/forensic.agent.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

import { BinanceAdapter }    from '../infrastructure/binance.adapter.js';
import { IndicatorsAdapter } from '../infrastructure/indicators.adapter.js';
import { StorageAdapter }    from '../infrastructure/storage.adapter.js';
import {
  makeForensicReport,
  makeWalletProfile,
  MO_TYPES,
  STARK_CONSTRAINTS,
} from '../domain/entities.js';

const { FORENSIC_LOOKBACK_M } = STARK_CONSTRAINTS;

// ── Helpers ────────────────────────────────────────────────────────────────────

/**
 * Genera un Whale Fingerprint determinísta.
 * Binance no expone wallets, pero podemos crear un ID estable basado en
 * la firma del trade: volumen-bucket + zona de precio + dirección.
 * Clusters con el mismo perfil tendrán el mismo fingerprint.
 *
 * Formato: WF-XXXX-YYYY  (WF = Whale Fingerprint)
 * @param {object} whale - WhaleEvent
 * @param {number} price - Precio actual
 * @returns {string}
 */
function generateWhaleFingerprint(whale, price) {
  // Vol bucket: rounds to nearest 0.5 BTC (1.0, 1.5, 2.0 ... 10.0)
  const volBucket = (Math.round(whale.amount * 2) / 2).toFixed(1);
  // Price zone: nearest $500 for BTC
  const priceZone = Math.round(price / 500) * 500;
  // Direction
  const side = whale.transactionType === 'buy' ? 'B' : 'S';
  // Compact hash: sum of char codes → hex, take last 4
  const raw  = `${volBucket}|${priceZone}|${side}`;
  let hash   = 0;
  for (let i = 0; i < raw.length; i++) hash = ((hash << 5) - hash + raw.charCodeAt(i)) | 0;
  const hex  = Math.abs(hash).toString(16).toUpperCase().padStart(8, '0');
  return `WF-${hex.slice(0, 4)}-${hex.slice(4, 8)}`;
}

/**
 * Carga o crea el WalletProfile para un fingerprint dado.
 * Actualiza el perfil con los datos del último ForensicReport.
 * @param {string} fingerprint
 * @param {object} report - Datos del trade actual
 * @returns {Promise<object>} WalletProfile actualizado
 */
async function upsertWalletProfile(fingerprint, { symbol, mo, side, amount, price, rsiAtTrade, volRatio }) {
  const existing = await StorageAdapter.loadWalletProfile(fingerprint);

  if (!existing) {
    // Primer avistamiento
    const profile = makeWalletProfile({
      fingerprint,
      symbol,
      label: `Ballena ${fingerprint}`,
      firstSeen:    Date.now(),
      lastSeen:     Date.now(),
      tradeCount:   1,
      totalBtc:     amount,
      winCount:     0,
      lossCount:    0,
      winRate:      0,
      dominantMO:   mo,
      dominantSide: side,
      avgRsiEntry:  rsiAtTrade,
      avgVolRatio:  volRatio,
      moHistory:    [{ mo, ts: Date.now() }],
      priceHistory: [{ price, ts: Date.now() }],
    });
    await StorageAdapter.saveWalletProfile(profile);
    return profile;
  }

  // Avistamiento repetido — actualizar acumulados
  const n            = existing.tradeCount;
  const newMoHistory = [...(existing.moHistory || []), { mo, ts: Date.now() }].slice(-20);
  const newPriceHistory = [...(existing.priceHistory || []), { price, ts: Date.now() }].slice(-20);

  // MO dominante = el más frecuente en el historial
  const moFreq = {};
  newMoHistory.forEach(m => { moFreq[m.mo] = (moFreq[m.mo] || 0) + 1; });
  const dominantMO = Object.keys(moFreq).reduce((a, b) => moFreq[a] >= moFreq[b] ? a : b, mo);

  const updated = makeWalletProfile({
    ...existing,
    lastSeen:     Date.now(),
    tradeCount:   n + 1,
    totalBtc:     (existing.totalBtc || 0) + amount,
    avgRsiEntry:  rsiAtTrade != null
      ? ((existing.avgRsiEntry || rsiAtTrade) * n + rsiAtTrade) / (n + 1)
      : existing.avgRsiEntry,
    avgVolRatio:  ((existing.avgVolRatio || 1) * n + volRatio) / (n + 1),
    dominantMO,
    moHistory:    newMoHistory,
    priceHistory: newPriceHistory,
  });
  await StorageAdapter.saveWalletProfile(updated);
  return updated;
}

/**
 * Estima el volumen comprador/vendedor de una vela usando la heurística
 * de Bull Power: si la vela cierra en la mitad superior del rango, la
 * mayor parte del volumen es comprador.
 * @param {Candle} candle
 * @returns {{ buyVol: number, sellVol: number, delta: number }}
 */
function estimateVolumeDelta(candle) {
  const range = candle.high - candle.low || 1;
  const bullRatio = (candle.close - candle.low) / range; // 0–1
  const buyVol  = candle.volume * bullRatio;
  const sellVol = candle.volume * (1 - bullRatio);
  return { buyVol, sellVol, delta: buyVol - sellVol };
}

/**
 * Determina la posición del precio respecto a EMAs clave.
 * @param {number} price
 * @param {number} ema20
 * @param {number} ema50
 * @param {number} ema200
 * @returns {string}
 */
function classifyEmaPosition(price, ema20, ema50, ema200) {
  const within = (p, e, pct) => Math.abs(p - e) / e < pct;
  if (within(price, ema200, 0.003)) return 'at_ema200';
  if (within(price, ema50, 0.002))  return 'at_ema50';
  if (within(price, ema20, 0.002))  return 'at_ema20';
  if (price > ema20 && ema20 > ema50) return 'above_ema20';
  if (price < ema20 && ema20 < ema50) return 'below_ema20';
  if (price > ema50)  return 'above_ema50';
  return 'below_ema50';
}

/**
 * Clasifica la posición del precio respecto a soportes/resistencias.
 * @param {number} price
 * @param {object[]} srLevels
 * @returns {string}
 */
function classifyPriceVsLevel(price, srLevels) {
  for (const lvl of srLevels) {
    const dist = Math.abs(price - lvl.price) / price;
    if (dist < 0.004) {
      return lvl.type === 'support' ? 'at_support' : 'at_resistance';
    }
  }
  return 'mid-range';
}

// ── Modus Operandi Classification ─────────────────────────────────────────────

const MO_DEFINITIONS = [
  {
    id:    MO_TYPES.LIQUIDITY_GRAB,
    label: 'Liquidity Grab',
    icon:  '⚡',
    description: 'Barrido de stops de minoristas — precio perforó un nivel clave brevemente y regresó. Señal clásica de Smart Money cazando liquidez.',
    match: ({ emaPos, priceVsLevel, rsiAtTrade, volRatio, whaleSide }) =>
      priceVsLevel !== 'mid-range' && volRatio > 2.0 && rsiAtTrade !== null &&
      (rsiAtTrade < 28 || rsiAtTrade > 72),
    weight: 90,
  },
  {
    id:    MO_TYPES.SMART_MONEY_BUY,
    label: 'Smart Money Acumulación',
    icon:  '🟢',
    description: 'Compra institucional en zona de descuento. Precio en soporte + RSI bajo + volumen comprador superior al promedio.',
    match: ({ emaPos, priceVsLevel, rsiAtTrade, volumeDelta, volRatio, whaleSide }) =>
      whaleSide === 'buy' && priceVsLevel === 'at_support' &&
      rsiAtTrade !== null && rsiAtTrade < 45 && volumeDelta > 0 && volRatio > 1.3,
    weight: 85,
  },
  {
    id:    MO_TYPES.SMART_MONEY_SELL,
    label: 'Smart Money Distribución Agresiva',
    icon:  '🔴',
    description: 'Venta institucional en zona de premium. Precio en resistencia + RSI alto + volumen vendedor dominante.',
    match: ({ priceVsLevel, rsiAtTrade, volumeDelta, volRatio, whaleSide }) =>
      whaleSide === 'sell' && priceVsLevel === 'at_resistance' &&
      rsiAtTrade !== null && rsiAtTrade > 55 && volumeDelta < 0 && volRatio > 1.3,
    weight: 85,
  },
  {
    id:    MO_TYPES.DISTRIBUTION,
    label: 'Distribución Silenciosa',
    icon:  '📦',
    description: 'Venta gradual en zonas de alta liquidez para no hundir el precio. Volumen moderado, precio en rango medio-alto.',
    match: ({ whaleSide, volRatio, rsiAtTrade, emaPos }) =>
      whaleSide === 'sell' && volRatio >= 1.0 && volRatio < 2.5 &&
      rsiAtTrade !== null && rsiAtTrade > 50 &&
      (emaPos === 'above_ema20' || emaPos === 'above_ema50'),
    weight: 70,
  },
  {
    id:    MO_TYPES.ACCUMULATION,
    label: 'Acumulación Silenciosa',
    icon:  '🏗️',
    description: 'Compra gradual en zonas de precio bajo para construir posición sin mover el mercado.',
    match: ({ whaleSide, volRatio, rsiAtTrade, emaPos }) =>
      whaleSide === 'buy' && volRatio >= 1.0 && volRatio < 2.5 &&
      rsiAtTrade !== null && rsiAtTrade < 50 &&
      (emaPos === 'below_ema20' || emaPos === 'below_ema50'),
    weight: 70,
  },
  {
    id:    MO_TYPES.PANIC_SELL,
    label: 'Panic Sell',
    icon:  '🚨',
    description: 'Venta de emergencia. Volumen extremo, RSI sobrevendido, vela bajista. Posiblemente margin call o stop loss masivo.',
    match: ({ whaleSide, volRatio, rsiAtTrade, volumeDelta }) =>
      whaleSide === 'sell' && volRatio > 3.0 &&
      rsiAtTrade !== null && rsiAtTrade < 35 && volumeDelta < 0,
    weight: 80,
  },
  {
    id:    MO_TYPES.STOP_HUNT,
    label: 'Stop Hunt',
    icon:  '🎯',
    description: 'Manipulación para activar stops por encima/debajo de un nivel técnico visible. El movimiento suele revertir en <15 minutos.',
    match: ({ priceVsLevel, hasDivergence, volRatio }) =>
      priceVsLevel !== 'mid-range' && hasDivergence && volRatio > 1.8,
    weight: 75,
  },
];

/**
 * Clasifica el Modus Operandi de la ballena.
 * @param {object} ctx - Contexto con todos los indicadores
 * @returns {{ mo, moLabel, moIcon, moDescription, confidence, predictedDir, predictedStrength }}
 */
function classifyMO(ctx) {
  const ranked = MO_DEFINITIONS
    .filter(def => def.match(ctx))
    .sort((a, b) => b.weight - a.weight);

  if (!ranked.length) {
    return {
      mo: MO_TYPES.UNKNOWN, moLabel: 'Patrón no identificado',
      moIcon: '🔍', moDescription: 'Insuficientes señales técnicas para clasificar este trade.',
      confidence: 20, predictedDir: 'neutral', predictedStrength: 20,
    };
  }

  const top = ranked[0];
  const confidence = Math.min(95, top.weight + (ranked.length > 1 ? 5 : 0));

  // Predicción de dirección a 15m basada en MO
  const bullMOs = [MO_TYPES.SMART_MONEY_BUY, MO_TYPES.ACCUMULATION, MO_TYPES.LIQUIDITY_GRAB];
  const bearMOs = [MO_TYPES.SMART_MONEY_SELL, MO_TYPES.DISTRIBUTION, MO_TYPES.PANIC_SELL];
  const predictedDir = bullMOs.includes(top.id) ? 'bull'
    : bearMOs.includes(top.id) ? 'bear'
    : 'neutral';

  // Stop hunt es neutro inicialmente pero con reversión esperada
  const predictedStrength = top.id === MO_TYPES.STOP_HUNT ? 65
    : top.id === MO_TYPES.PANIC_SELL ? 55
    : confidence;

  return {
    mo: top.id, moLabel: top.label, moIcon: top.icon,
    moDescription: top.description, confidence, predictedDir, predictedStrength,
  };
}

// ── Main Forensic Function ─────────────────────────────────────────────────────

/**
 * Analiza el Modus Operandi de una ballena cruzando su timestamp
 * con los indicadores técnicos activos en ese momento.
 *
 * @param {import('../domain/entities.js').WhaleEvent} whaleEvent
 * @param {string} [symbol='BTCUSDT']
 * @returns {Promise<import('../domain/entities.js').ForensicReport>}
 */
export async function decodeWhaleStrategy(whaleEvent, symbol = 'BTCUSDT') {
  const tradeTs = whaleEvent.timestamp;

  // 1. Obtener velas 1m — necesitamos suficiente historia para indicadores
  const candles = await BinanceAdapter.fetchKlines(symbol, '1m', 100);
  if (!candles || candles.length < 20) {
    return makeForensicReport({ whaleId: whaleEvent.id, symbol, timestamp: tradeTs });
  }

  // 2. Calcular todos los indicadores sobre la serie completa
  const closes  = candles.map(c => c.close);
  const n       = closes.length - 1;

  const rsiArr   = IndicatorsAdapter.rsi(closes, 14);
  const ema20Arr = IndicatorsAdapter.ema(closes, 20);
  const ema50Arr = IndicatorsAdapter.ema(closes, 50);
  const ema200Arr= IndicatorsAdapter.ema(closes, Math.min(200, closes.length));
  const atrArr   = IndicatorsAdapter.atr(candles, 14);
  const srLevels = IndicatorsAdapter.supportResistance(candles, 5);
  const divResult= IndicatorsAdapter.detectRSIDivergence(candles, rsiArr);

  // 3. Localizar la vela del trade por timestamp (o la más cercana)
  let tradeIdx = n; // fallback: última vela
  let minDiff  = Infinity;
  for (let i = 0; i < candles.length; i++) {
    const diff = Math.abs(candles[i].time - tradeTs);
    if (diff < minDiff) { minDiff = diff; tradeIdx = i; }
  }

  const tradeCandle = candles[tradeIdx];
  const price       = tradeCandle.close;

  // 4. Indicadores en el momento del trade
  const rsiAtTrade  = rsiArr[tradeIdx] ?? null;
  const ema20       = ema20Arr[tradeIdx];
  const ema50       = ema50Arr[tradeIdx];
  const ema200      = ema200Arr[tradeIdx];
  const atr         = atrArr[tradeIdx] ?? price * 0.005;

  // 5. Volumen delta de esa vela
  const { buyVol, sellVol, delta: volumeDelta } = estimateVolumeDelta(tradeCandle);
  const vol20 = candles.slice(Math.max(0, tradeIdx - 20), tradeIdx)
    .reduce((s, c) => s + c.volume, 0) / 20 || tradeCandle.volume;
  const volRatio = tradeCandle.volume / vol20;

  // 6. Posición respecto a EMAs y S/R
  const emaPos      = classifyEmaPosition(price, ema20, ema50, ema200);
  const priceVsLevel= classifyPriceVsLevel(price, srLevels);

  // 7. Divergencia en ventana ±5 velas del trade
  const startDiv  = Math.max(0, tradeIdx - 5);
  const endDiv    = Math.min(n, tradeIdx + 5);
  const localRsi  = rsiArr.slice(startDiv, endDiv + 1);
  const localCand = candles.slice(startDiv, endDiv + 1);
  const localDiv  = IndicatorsAdapter.detectRSIDivergence(localCand, localRsi);
  const hasDivergence  = Boolean(localDiv.type);
  const divergenceType = localDiv.type;

  // 8. Modus Operandi
  const whaleSide = whaleEvent.transactionType; // 'buy' | 'sell'
  const moCtx = {
    rsiAtTrade, emaPos, priceVsLevel, volumeDelta, volRatio,
    hasDivergence, whaleSide,
  };
  const moResult = classifyMO(moCtx);

  // 9. Whale Fingerprint + Wallet Profile
  const whaleFingerprint = generateWhaleFingerprint(whaleEvent, price);
  let walletProfile = null;
  try {
    walletProfile = await upsertWalletProfile(whaleFingerprint, {
      symbol,
      mo:        moResult.mo,
      side:      whaleSide,
      amount:    whaleEvent.amount,
      price,
      rsiAtTrade,
      volRatio,
    });
  } catch (e) {
    console.warn('[ForensicAgent] walletProfile error:', e);
  }

  // 10. Señales en lenguaje natural
  const signals = [];
  if (rsiAtTrade !== null) {
    const rsiState = rsiAtTrade < 30 ? '⚡ Sobrevendido' : rsiAtTrade > 70 ? '⚡ Sobrecomprado' : 'Neutral';
    signals.push(`RSI ${rsiAtTrade.toFixed(1)} — ${rsiState}`);
  }
  signals.push(`EMA: ${emaPos.replace(/_/g, ' ')}`);
  signals.push(`Precio vs nivel: ${priceVsLevel.replace(/_/g, ' ')}`);
  signals.push(`Vol ratio: ${volRatio.toFixed(2)}x del promedio 20`);
  signals.push(`Delta vol: ${volumeDelta > 0 ? '▲ Compradores' : '▼ Vendedores'} (${Math.abs(volumeDelta / tradeCandle.volume * 100).toFixed(0)}%)`);
  if (hasDivergence) signals.push(`⚠️ Divergencia RSI ${divergenceType} detectada en ventana del trade`);
  if (walletProfile && walletProfile.tradeCount > 1) {
    signals.push(`📖 Billetera vista ${walletProfile.tradeCount} veces · Vol total ${walletProfile.totalBtc.toFixed(2)} BTC`);
  }

  return makeForensicReport({
    whaleId:          whaleEvent.id,
    whaleFingerprint,
    walletProfile,     // campo extra fuera del factory, pasado por spread
    symbol,
    timestamp:        tradeTs,
    rsiAtTrade,
    emaPos,
    volumeDelta,
    volRatio,
    hasDivergence,
    divergenceType,
    priceVsLevel,
    signals,
    ...moResult,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// SINGLETON EXPORT — ForensicAgent
// ─────────────────────────────────────────────────────────────────────────────

export const ForensicAgent = Object.freeze({
  decodeWhaleStrategy,
  classifyMO,
  estimateVolumeDelta,
  generateWhaleFingerprint,
  MO_TYPES,
});

export default ForensicAgent;
