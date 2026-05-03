// ═══════════════════════════════════════════════════════════════════
// STARK-OS · INFRASTRUCTURE LAYER — Hyperliquid Adapter
// Accede a la API pública de Hyperliquid (DEX on-chain) para obtener
// datos reales de wallets: historial de fills, posiciones, PnL, stats.
// ═══════════════════════════════════════════════════════════════════

'use strict';

const HL_API = 'https://api.hyperliquid.xyz/info';
const REQUEST_TIMEOUT = 15000; // 15 segundos timeout

// ── Rate limiter global — max 1 request/s, backoff en 429 ──────────
let _hlLastRequest = 0;
let _hlBackoffUntil = 0;
const HL_MIN_INTERVAL = 1200; // ms entre requests

async function _hlRateLimit() {
  // Si estamos en backoff por 429, esperar
  const now = Date.now();
  if (now < _hlBackoffUntil) {
    const wait = _hlBackoffUntil - now;
    console.warn(`[HLAdapter] Rate limit backoff, esperando ${Math.ceil(wait/1000)}s`);
    await new Promise(r => setTimeout(r, wait));
  }
  // Espaciar requests
  const elapsed = Date.now() - _hlLastRequest;
  if (elapsed < HL_MIN_INTERVAL) {
    await new Promise(r => setTimeout(r, HL_MIN_INTERVAL - elapsed));
  }
  _hlLastRequest = Date.now();
}

// ── Helper ─────────────────────────────────────────────────────────

// Wrapper con timeout para fetch
async function fetchWithTimeout(url, options, timeout = REQUEST_TIMEOUT) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (e) {
    clearTimeout(timeoutId);
    if (e.name === 'AbortError') {
      throw new Error('Timeout: La API de Hyperliquid no respondió en 15 segundos');
    }
    throw e;
  }
}

async function hlPost(body) {
  const requestType = body.type || 'unknown';

  // Aplicar rate limiting
  await _hlRateLimit();

  try {
    const r = await fetchWithTimeout(HL_API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    
    if (r.status === 429) {
      // Backoff 30 segundos en rate limit
      _hlBackoffUntil = Date.now() + 30000;
      throw new Error('Hyperliquid API 429: Rate limit alcanzado');
    }
    
    if (!r.ok) {
      let msg = `Hyperliquid API ${r.status}`;
      try {
        const txt = await r.text();
        msg += `: ${txt}`;
      } catch (_) {}
      throw new Error(msg);
    }
    
    const data = await r.json();
    return data;
  } catch (e) {
    if (!e.message?.includes('429')) {
      console.error('[HL Adapter] Request failed:', body.type, e.message);
    }
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
  console.log('[HLAdapter] → Solicitando fills y state para:', address);
  
  const startFills = Date.now();
  const fillsPromise = getUserFills(address);
  
  const startState = Date.now();
  const statePromise = getClearinghouseState(address).catch(() => null);
  
  const [fills, state] = await Promise.all([fillsPromise, statePromise]);
  
  console.log(`[HLAdapter] ✓ Fills obtenidos: ${fills?.length || 0} en ${Date.now() - startFills}ms`);
  console.log(`[HLAdapter] ✓ State obtenido en ${Date.now() - startState}ms`);

  if (!fills.length) {
    console.warn('[HLAdapter] ⚠ Wallet sin historial de trades');
    return { address, error: 'Sin historial de trades', fills: [] };
  }

  console.log('[HLAdapter] → Procesando estadísticas...');
  const closingFills = fills.filter(f => f.closedPnl != null && parseFloat(f.closedPnl) !== 0);
  const wins   = closingFills.filter(f => parseFloat(f.closedPnl) > 0);
  const losses = closingFills.filter(f => parseFloat(f.closedPnl) < 0);
  const winRate = closingFills.length
    ? parseFloat((wins.length / closingFills.length * 100).toFixed(1))
    : 0;

  const totalPnl = closingFills.reduce((s, f) => s + parseFloat(f.closedPnl || 0), 0);
  const totalFees = fills.reduce((s, f) => s + parseFloat(f.fee || 0), 0);

  console.log('[HLAdapter] → Calculando monedas favoritas...');
  const coinCount = {};
  for (const f of fills) {
    coinCount[f.coin] = (coinCount[f.coin] || 0) + 1;
  }
  const favoriteCoins = Object.entries(coinCount)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([coin, count]) => ({ coin, count }));

  console.log('[HLAdapter] → Detectando sesiones de trading...');
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
