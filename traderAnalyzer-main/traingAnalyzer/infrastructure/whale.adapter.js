// ═══════════════════════════════════════════════════════════════════
// STARK-OS · INFRASTRUCTURE LAYER — Whale Adapter
// Adaptador compatible con whale-alert-js que escanea trades grandes
// de BTC usando los endpoints públicos de Binance (sin API key).
// Implementa IWhalePort.
// Capa: infrastructure/whale.adapter.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

import { BinanceAdapter } from './binance.adapter.js';
import { makeWhaleEvent } from '../domain/entities.js';

/**
 * Obtiene el precio actual de BTC en USD.
 * @returns {Promise<number>}
 */
async function getBtcPrice() {
  return BinanceAdapter.getBtcPrice();
}

/**
 * Analiza los aggTrades recientes de Binance para detectar
 * transacciones en el rango [minBtc, maxBtc].
 *
 * Implementa IWhalePort.getRecentWhales()
 *
 * @param {string} symbol - Par de trading (ej: 'BTCUSDT')
 * @param {number} minBtc - Cantidad mínima en BTC (ej: 1)
 * @param {number} maxBtc - Cantidad máxima en BTC (ej: 10)
 * @param {number} limit  - Número de aggTrades a escanear (max 1000)
 * @returns {Promise<import('../domain/entities.js').WhaleEvent[]>}
 */
async function getRecentWhales(symbol = 'BTCUSDT', minBtc = 1, maxBtc = 10, limit = 1000) {
  const [btcPrice, aggTrades] = await Promise.all([
    getBtcPrice(),
    BinanceAdapter.fetchAggTrades(symbol, Math.min(limit, 1000)),
  ]);

  if (!btcPrice || !aggTrades?.length) return [];

  /** Agrupa trades consecutivos del mismo lado en ventanas de 500ms */
  const clusters = clusterTrades(aggTrades);

  return clusters
    .filter(cluster => {
      const btcAmount = cluster.qty; // qty ya está en BTC para BTCUSDT
      return btcAmount >= minBtc && btcAmount <= maxBtc;
    })
    .map(cluster => makeWhaleEvent({
      id:              `whale_${cluster.aggId}`,
      symbol,
      blockchain:      'BITCOIN',
      transactionType: cluster.isBuyerMaker ? 'sell' : 'buy',
      from:            cluster.isBuyerMaker ? 'whale_seller' : 'exchange',
      to:              cluster.isBuyerMaker ? 'exchange' : 'whale_buyer',
      amount:          cluster.qty,
      amountUsd:       cluster.qty * btcPrice,
      timestamp:       cluster.time,
      hash:            `agg_${cluster.aggId}`,
    }))
    .sort((a, b) => b.amount - a.amount); // mayor a menor BTC
}

/**
 * Agrupa aggTrades consecutivos del mismo lado (compra/venta) en
 * clusters de hasta 500ms. Permite detectar órdenes institucionales
 * que se ejecutan en múltiples trades pequeños.
 *
 * @param {object[]} trades - Array de aggTrades de Binance
 * @returns {object[]} - Array de clusters con { aggId, qty, time, isBuyerMaker }
 */
function clusterTrades(trades) {
  if (!trades.length) return [];

  const clusters = [];
  let current = {
    aggId:        trades[0].a,
    qty:          parseFloat(trades[0].q),
    time:         trades[0].T,
    isBuyerMaker: trades[0].m,
  };

  for (let i = 1; i < trades.length; i++) {
    const t = trades[i];
    const qty = parseFloat(t.q);
    const sameDir = t.m === current.isBuyerMaker;
    const within500ms = (t.T - current.time) < 500;

    if (sameDir && within500ms) {
      current.qty += qty;
    } else {
      clusters.push({ ...current });
      current = { aggId: t.a, qty, time: t.T, isBuyerMaker: t.m };
    }
  }
  clusters.push({ ...current });

  return clusters;
}

/**
 * Obtiene estadísticas de presión compradora/vendedora de ballenas.
 * Útil para calcular el sesgo de dirección del Hunter Agent.
 *
 * @param {import('../domain/entities.js').WhaleEvent[]} whales
 * @returns {{ buyCount: number, sellCount: number, buyBtc: number, sellBtc: number, bias: 'buy'|'sell'|'neutral' }}
 */
export function analyzeWhalePressure(whales) {
  const buys  = whales.filter(w => w.transactionType === 'buy');
  const sells = whales.filter(w => w.transactionType === 'sell');
  const buyBtc  = buys.reduce((s, w) => s + w.amount, 0);
  const sellBtc = sells.reduce((s, w) => s + w.amount, 0);
  const total   = buyBtc + sellBtc || 1;
  const bias    = buyBtc / total > 0.55 ? 'buy' : sellBtc / total > 0.55 ? 'sell' : 'neutral';
  return { buyCount: buys.length, sellCount: sells.length, buyBtc, sellBtc, bias };
}

// ─────────────────────────────────────────────────────────────────────────────
// SINGLETON EXPORT — WhaleAdapter (implementa IWhalePort)
// ─────────────────────────────────────────────────────────────────────────────

export const WhaleAdapter = Object.freeze({
  getRecentWhales,
  getBtcPrice,
  analyzeWhalePressure,
});

export default WhaleAdapter;
