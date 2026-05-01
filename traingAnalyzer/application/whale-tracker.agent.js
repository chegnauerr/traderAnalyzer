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
  const clean = address.trim();
  if (!clean.startsWith('0x') || clean.length !== 42) {
    throw new Error('Dirección inválida. Debe ser 0x... (42 caracteres).');
  }
  const dna = await HyperliquidAdapter.analyzeWhaleDNA(clean);
  if (dna.error) throw new Error(dna.error);

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
 * Obtiene el leaderboard de Hyperliquid (wallets ballena públicas).
 * @param {number} [limit=20]
 * @returns {Promise<object[]>}
 */
export async function getTopWhales(limit = 20) {
  try {
    const vaults = await HyperliquidAdapter.getVaults();
    if (!Array.isArray(vaults)) return [];

    // Mapear vaults como si fueran "ballenas top"
    return vaults
      .sort((a, b) => parseFloat(b.pnl || 0) - parseFloat(a.pnl || 0))
      .slice(0, limit)
      .map((v, i) => ({
        rank:        i + 1,
        address:     v.vaultAddress,
        displayName: v.name,
        pnl:         parseFloat(v.pnl || 0),
        winRate:     parseFloat(v.apr || 0) * 100, // Usamos APR como proxy de rendimiento
        volume:      parseFloat(v.tvl || 0),        // Usamos TVL como proxy de volumen/tamaño
        isVault:     true,
        followerCount: v.followerCount || 0,
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

// ── Singleton ──────────────────────────────────────────────────────

export const WhaleTrackerAgent = Object.freeze({
  analyzeWallet,
  getTopWhales,
  getTrackedWallets,
  removeTrackedWallet,
});

export default WhaleTrackerAgent;
