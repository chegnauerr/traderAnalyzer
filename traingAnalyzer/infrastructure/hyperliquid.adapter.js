// ═══════════════════════════════════════════════════════════════════
// STARK-OS · INFRASTRUCTURE LAYER — Hyperliquid Adapter
// Accede a la API pública de Hyperliquid (DEX on-chain) para obtener
// datos reales de wallets: historial de fills, posiciones, PnL, stats.
// ═══════════════════════════════════════════════════════════════════

'use strict';

const HL_API = 'https://api.hyperliquid.xyz/info';

// ── Helper ─────────────────────────────────────────────────────────

async function hlPost(body) {
  try {
    const r = await fetch(HL_API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      let msg = `Hyperliquid API ${r.status}`;
      try {
        const txt = await r.text();
        msg += `: ${txt}`;
      } catch (_) {}
      throw new Error(msg);
    }
    return r.json();
  } catch (e) {
    console.error('[HL Adapter] Request failed:', body, e);
    throw e;
  }
}

// ── Wallet: historial de fills ─────────────────────────────────────

async function getUserFills(address) {
  const data = await hlPost({ type: 'userFills', user: address });
  return Array.isArray(data) ? data : [];
}

async function getUserFillsByTime(address, startTime) {
  const body = { type: 'userFillsByTime', user: address, startTime: startTime || 0 };
  const data = await hlPost(body);
  return Array.isArray(data) ? data : [];
}

// ── Wallet: estado actual ──────────────────────────────────────────

async function getClearinghouseState(address) {
  return hlPost({ type: 'clearinghouseState', user: address });
}

async function getOpenOrders(address) {
  const data = await hlPost({ type: 'openOrders', user: address });
  return Array.isArray(data) ? data : [];
}

// ── Vaults: ballenas públicas y estrategias ────────────────────────

/**
 * Obtiene todas las Vaults (bóvedas) públicas de Hyperliquid.
 * Las vaults son perfiles de ballenas/estrategias que podemos analizar.
 * @returns {Promise<object[]>}
 */
async function getVaults() {
  return await hlPost({ type: "vaults" });
}

// ── Recientes trades ───────────────────────────────────────────────

async function getRecentTrades(coin) {
  const data = await hlPost({ type: 'recentTrades', coin });
  return Array.isArray(data) ? data : [];
}

// ── Análisis: Whale DNA ────────────────────────────────────────────

async function analyzeWhaleDNA(address) {
  const [fills, state] = await Promise.all([
    getUserFills(address),
    getClearinghouseState(address).catch(() => null),
  ]);

  if (!fills.length) {
    return { address, error: 'Sin historial de trades', fills: [] };
  }

  const closingFills = fills.filter(f => f.closedPnl != null && parseFloat(f.closedPnl) !== 0);
  const wins   = closingFills.filter(f => parseFloat(f.closedPnl) > 0);
  const losses = closingFills.filter(f => parseFloat(f.closedPnl) < 0);
  const winRate = closingFills.length
    ? parseFloat((wins.length / closingFills.length * 100).toFixed(1))
    : 0;

  const totalPnl = closingFills.reduce((s, f) => s + parseFloat(f.closedPnl || 0), 0);
  const totalFees = fills.reduce((s, f) => s + parseFloat(f.fee || 0), 0);

  const coinCount = {};
  for (const f of fills) {
    coinCount[f.coin] = (coinCount[f.coin] || 0) + 1;
  }
  const favoriteCoins = Object.entries(coinCount)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([coin, count]) => ({ coin, count }));

  const sessionCount = { asia: 0, europe: 0, us: 0, late_us: 0 };
  for (const f of fills) {
    const h = new Date(f.time).getUTCHours();
    if (h >= 0  && h < 8)  sessionCount.asia++;
    else if (h >= 8 && h < 14) sessionCount.europe++;
    else if (h >= 14 && h < 20) sessionCount.us++;
    else sessionCount.late_us++;
  }
  const dominantSession = Object.keys(sessionCount)
    .reduce((a, b) => sessionCount[a] >= sessionCount[b] ? a : b);

  const avgPositionSz = fills.length
    ? fills.reduce((s, f) => s + parseFloat(f.sz || 0), 0) / fills.length
    : 0;

  const sideCount  = { A: 0, B: 0 };
  const liquidityCount = { maker: 0, taker: 0 };
  for (const f of fills) {
    if (f.side) sideCount[f.side] = (sideCount[f.side] || 0) + 1;
    const feeRate = Math.abs(parseFloat(f.fee || 0)) / (parseFloat(f.px || 1) * parseFloat(f.sz || 1));
    if (feeRate < 0.0003) liquidityCount.maker++;
    else liquidityCount.taker++;
  }
  const entryStyle = liquidityCount.maker > liquidityCount.taker ? 'Limit (Paciente)' : 'Market (Agresivo)';
  const dominantSide = (sideCount.B || 0) >= (sideCount.A || 0) ? 'LONG' : 'SHORT';

  let currentPositions = [];
  if (state?.assetPositions) {
    currentPositions = state.assetPositions
      .filter(p => parseFloat(p.position?.szi || 0) !== 0)
      .map(p => ({
        coin:      p.position.coin,
        size:      parseFloat(p.position.szi),
        entryPx:   parseFloat(p.position.entryPx),
        leverage:  parseFloat(p.position.leverage?.value || p.position.leverage || 1),
        unrealPnl: parseFloat(p.position.unrealizedPnl || 0),
        marginUsed:parseFloat(p.position.marginUsed || 0),
      }));
  }

  const avgLeverage = currentPositions.length
    ? currentPositions.reduce((s, p) => s + p.leverage, 0) / currentPositions.length
    : null;

  let strategy = 'Desconocida';
  if (winRate > 70 && totalPnl > 0) strategy = '🧠 Smart Money — Alta Precisión';
  else if (avgLeverage && avgLeverage > 20) strategy = '⚡ Scalper Agresivo — Alto Apalancamiento';
  else if (liquidityCount.maker > liquidityCount.taker * 2) strategy = '🏗️ Market Maker / Proveedor de Liquidez';
  else if (favoriteCoins.length === 1) strategy = '🎯 Especialista en ' + favoriteCoins[0].coin;
  else if (winRate > 60) strategy = '📈 Swing Trader — Tendencia';
  else strategy = '🔄 Trader Diversificado';

  return {
    address,
    totalFills:  fills.length,
    winRate,
    wins:        wins.length,
    losses:      losses.length,
    totalPnl:    parseFloat(totalPnl.toFixed(2)),
    totalFees:   parseFloat(totalFees.toFixed(2)),
    netPnl:      parseFloat((totalPnl - totalFees).toFixed(2)),
    avgPositionSz: parseFloat(avgPositionSz.toFixed(4)),
    avgLeverage,
    favoriteCoins,
    dominantSession,
    sessionCount,
    entryStyle,
    dominantSide,
    strategy,
    currentPositions,
    liquidityCount,
    recentFills: fills.slice(0, 20),
  };
}

export const HyperliquidAdapter = Object.freeze({
  getUserFills,
  getUserFillsByTime,
  getClearinghouseState,
  getOpenOrders,
  getVaults,
  getRecentTrades,
  analyzeWhaleDNA,
});

export default HyperliquidAdapter;
