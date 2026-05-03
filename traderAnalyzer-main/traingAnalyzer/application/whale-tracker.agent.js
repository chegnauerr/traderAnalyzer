// ═══════════════════════════════════════════════════════════════════
// STARK-OS · APPLICATION LAYER — Whale Tracker Agent
// Rastrea wallets reales de Hyperliquid: obtiene su ADN completo,
// calcula su estrategia, gestión de riesgo y patrones de entrada.
//
// Flujo:
//  1. Recibe una dirección 0x (de la UI o del leaderboard)
//  2. Llama a HyperliquidAdapter.analyzeWhaleDNA()
//  3. Enriquece con indicadores de timing y patrones de precio
//  4. Persiste el perfil en storage para seguimiento continuo
//  5. Emite alertas si la wallet abre/cierra posición grande
//
// Capa: application/whale-tracker.agent.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

import { HyperliquidAdapter } from '../infrastructure/hyperliquid.adapter.js';
import { StorageAdapter }     from '../infrastructure/storage.adapter.js';

// ── Tracking Storage Key ───────────────────────────────────────────

const TRACKED_KEY = 'stark_tracked_wallets';

async function loadTrackedWallets() {
  const d = await StorageAdapter.get([TRACKED_KEY]);
  return Array.isArray(d[TRACKED_KEY]) ? d[TRACKED_KEY] : [];
}
async function saveTrackedWallets(list) {
  await StorageAdapter.set({ [TRACKED_KEY]: list.slice(0, 20) });
}

// ── Main: analizar wallet ──────────────────────────────────────────

/**
 * Analiza el ADN completo de una wallet de Hyperliquid.
 * @param {string} address - Dirección 0x
 * @returns {Promise<object>} WhaleDNA enriquecido
 */
export async function analyzeWallet(address) {
  console.log('[WhaleAgent] → Iniciando análisis de:', address);
  
  const clean = address.trim();
  if (!clean.startsWith('0x') || clean.length !== 42) {
    throw new Error('Dirección inválida. Debe ser 0x... (42 caracteres).');
  }
  
  console.log('[WhaleAgent] → Consultando Hyperliquid API...');
  const startTime = Date.now();
  const dna = await HyperliquidAdapter.analyzeWhaleDNA(clean);
  const elapsed = Date.now() - startTime;
  console.log(`[WhaleAgent] ✓ API respondió en ${elapsed}ms`);
  
  if (dna.error) {
    console.error('[WhaleAgent] ✗ Error en respuesta:', dna.error);
    throw new Error(dna.error);
  }

  console.log('[WhaleAgent] → Guardando en tracked wallets...');
  // Guardar en historial de wallets rastreadas
  const tracked = await loadTrackedWallets();
  const lowerClean = clean.toLowerCase();
  const existing = tracked.findIndex(w => w.address.toLowerCase() === lowerClean);
  const entry = {
    address: clean,
    label:   dna.strategy || 'Wallet',
    winRate: dna.winRate,
    lastSeen: Date.now(),
    totalFills: dna.totalFills,
  };
  if (existing >= 0) tracked[existing] = entry;
  else tracked.unshift(entry);
  await saveTrackedWallets(tracked);

  return dna;
}

/**
 * Obtiene el leaderboard de Hyperliquid (vaults profesionales).
 * @param {number} [limit=30]
 * @returns {Promise<object[]>}
 */
export async function getTopWhales(limit = 30) {
  try {
    const vaults = await HyperliquidAdapter.getVaults();
    if (!Array.isArray(vaults)) return [];

    console.log('[WhaleTracker] Vaults recibidas:', vaults.length);

    // Mapear vaults con datos reales
    return vaults
      .filter(v => v.vaultAddress && v.tvl > 0) // Solo vaults activas con TVL
      .sort((a, b) => {
        // Ordenar por score combinado: TVL + APR
        const scoreA = (parseFloat(a.tvl || 0) * 0.7) + (parseFloat(a.apr || 0) * 1000000);
        const scoreB = (parseFloat(b.tvl || 0) * 0.7) + (parseFloat(b.apr || 0) * 1000000);
        return scoreB - scoreA;
      })
      .slice(0, limit)
      .map((v, i) => ({
        rank:         i + 1,
        address:      v.vaultAddress,
        displayName:  v.name || `Vault #${i + 1}`,
        apr:          parseFloat(v.apr || 0),      // APR real (ej: 0.45 = 45%)
        tvl:          parseFloat(v.tvl || 0),      // Total Value Locked en USD
        pnl:          parseFloat(v.pnl || 0),      // PnL 7 días
        followers:    v.followerCount || 0,
        leader:       v.leader || v.vaultAddress,  // Dirección del creador/manager
        isVault:      true,
      }));
  } catch (e) {
    console.error('[WhaleTrackerAgent] Error fetching vaults:', e);
    return [];
  }
}

/**
 * Lista las wallets rastreadas guardadas.
 * @returns {Promise<object[]>}
 */
export async function getTrackedWallets() {
  return loadTrackedWallets();
}

/**
 * Elimina una wallet del seguimiento.
 * @param {string} address
 */
export async function removeTrackedWallet(address) {
  const tracked = await loadTrackedWallets();
  await saveTrackedWallets(tracked.filter(w => w.address !== address.toLowerCase()));
}

/**
 * Obtiene las reglas de trading de las wallets seguidas para aplicar al análisis.
 * @param {string} symbol - Símbolo actual (ej: BTCUSDT -> BTC)
 * @param {number} currentPrice - Precio actual
 * @param {object} marketData - Datos de mercado (trend, RSI, etc.)
 * @returns {Promise<object[]>}
 */
export async function getSmartMoneyRules(symbol, currentPrice, marketData = {}) {
  const tracked = await loadTrackedWallets();
  if (!tracked.length) return [];

  const coin = symbol.replace('USDT', '').replace('USD', '');
  const rules = [];

  for (const wallet of tracked.slice(0, 10)) {
    try {
      // Obtener DNA completo de la wallet
      const dna = await HyperliquidAdapter.analyzeWhaleDNA(wallet.address);
      if (dna.error) continue;

      // Verificar si esta wallet opera este coin
      const tradesCoin = (dna.topCoins || []).find(c => c.coin === coin);
      if (!tradesCoin) continue; // Esta wallet no opera este par

      // Analizar patrones recientes en este coin
      const recentFills = (dna.recentFills || [])
        .filter(f => f.coin === coin)
        .slice(0, 20);

      if (recentFills.length < 3) continue;

      // Detectar sesión horaria actual
      const currentHour = new Date().getUTCHours();
      const currentSession = 
        currentHour >= 0 && currentHour < 8 ? 'asia' :
        currentHour >= 8 && currentHour < 14 ? 'europe' :
        currentHour >= 14 && currentHour < 20 ? 'us' : 'late_us';

      const sessionMatch = dna.dominantSession === currentSession;

      // Calcular señal
      const longCount = recentFills.filter(f => f.dir.includes('Long')).length;
      const shortCount = recentFills.filter(f => f.dir.includes('Short')).length;
      const dominantSide = longCount > shortCount ? 'LONG' : 'SHORT';
      const sideConfidence = Math.max(longCount, shortCount) / recentFills.length;

      // Calcular niveles de entrada sugeridos
      const recentPrices = recentFills.map(f => f.px).sort((a, b) => a - b);
      const avgEntry = recentPrices.reduce((s, p) => s + p, 0) / recentPrices.length;
      const entryLow = Math.min(...recentPrices);
      const entryHigh = Math.max(...recentPrices);

      // Score basado en múltiples factores
      let score = 50;
      score += (dna.winRate - 50) * 0.8; // WR > 50% suma puntos
      score += sessionMatch ? 15 : -10; // Sesión correcta importante
      score += sideConfidence * 20; // Consistencia en dirección
      score += (tradesCoin.count / dna.totalFills) * 30; // % de trades en este coin
      score = Math.max(0, Math.min(100, score));

      rules.push({
        walletAddr: wallet.address,
        walletLabel: dna.strategy || wallet.label,
        winRate: dna.winRate,
        score: Math.round(score),
        signal: dominantSide,
        confidence: Math.round(sideConfidence * 100),
        sessionMatch,
        currentSession,
        preferredSession: dna.dominantSession,
        entryZone: {
          low: entryLow,
          avg: avgEntry,
          high: entryHigh,
        },
        leverage: dna.avgLeverage || 1,
        totalTrades: tradesCoin.count,
        recentActivity: recentFills.slice(0, 5),
      });
    } catch (e) {
      console.error(`[SmartMoneyRules] Error analyzing ${wallet.address}:`, e);
    }
  }

  // Ordenar por score descendente
  return rules.sort((a, b) => b.score - a.score);
}

// ── Singleton ──────────────────────────────────────────────────────

export const WhaleTrackerAgent = Object.freeze({
  analyzeWallet,
  getTopWhales,
  getTrackedWallets,
  removeTrackedWallet,
  getSmartMoneyRules,
});

export default WhaleTrackerAgent;
