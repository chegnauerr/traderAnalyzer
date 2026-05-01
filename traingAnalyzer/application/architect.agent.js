// ═══════════════════════════════════════════════════════════════════
// STARK-OS · APPLICATION LAYER — Architect Agent
// Monitoriza los NeuralPatterns guardados por el Scientist.
// Cada vez que el Hunter detecta una ballena nueva, el Architect
// compara sus indicadores contra los 50 patrones en storage.
//
// Flujo:
//  1. Recibe el contexto del trade actual (ForensicReport)
//  2. Carga los NeuralPatterns guardados
//  3. Compara condiciones: rsiRange, minVolRatio, emaPos, priceVsLevel
//  4. Si similaridad > MATCH_THRESHOLD (85%) → dispara alerta
//  5. Si el trade termina en LOSS → reduce confianza del patrón
//  6. Lee neural-config.json para prioridades de MO
//
// Capa: application/architect.agent.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

import { StorageAdapter } from '../infrastructure/storage.adapter.js';

// ── Constantes ─────────────────────────────────────────────────────
const MATCH_THRESHOLD  = 0.85;   // Similitud mínima para disparar alerta
const CONFIDENCE_DECAY = 0.05;   // Reducción de WR cuando el patrón falla
const CONFIDENCE_BOOST = 0.02;   // Aumento cuando el patrón tiene éxito
const MIN_CONFIDENCE   = 0.35;   // Por debajo de esto el patrón se marca inactivo

// Cache de la config
let _neuralConfig = null;

// ── Cargar neural-config.json ──────────────────────────────────────

async function loadNeuralConfig() {
  if (_neuralConfig) return _neuralConfig;
  try {
    const url  = chrome.runtime.getURL('neural-config.json');
    const resp = await fetch(url);
    if (resp.ok) _neuralConfig = await resp.json();
  } catch (_) {}
  return _neuralConfig || {
    learning_loop: { activation_threshold: 0.65 },
    modus_operandi_matrix: {},
  };
}

// ── Similitud entre ForensicReport y NeuralPattern ─────────────────

/**
 * Calcula la similitud (0–1) entre el contexto actual y un patrón guardado.
 * @param {import('../domain/entities.js').ForensicReport} report
 * @param {import('../domain/entities.js').NeuralPattern} pattern
 * @returns {number} 0-1
 */
function computeSimilarity(report, pattern) {
  const cond = pattern.conditions;
  let score  = 0;
  let checks = 0;

  // RSI en rango ±12 del patrón
  if (cond.rsiRange && report.rsiAtTrade != null) {
    checks++;
    const [lo, hi] = cond.rsiRange;
    if (report.rsiAtTrade >= lo && report.rsiAtTrade <= hi) score++;
    else {
      // Similitud parcial: qué tan lejos está
      const mid  = (lo + hi) / 2;
      const dist = Math.abs(report.rsiAtTrade - mid);
      score += Math.max(0, 1 - dist / 20);
    }
  }

  // Vol ratio >= umbral mínimo del patrón
  if (cond.minVolRatio) {
    checks++;
    score += report.volRatio >= cond.minVolRatio ? 1
      : Math.max(0, report.volRatio / cond.minVolRatio);
  }

  // Misma posición EMA
  if (cond.emaPos) {
    checks++;
    score += cond.emaPos === report.emaPos ? 1
      : cond.emaPos.startsWith('above') === (report.emaPos || '').startsWith('above') ? 0.5 : 0;
  }

  // Misma zona de precio vs nivel
  if (cond.priceVsLevel) {
    checks++;
    score += cond.priceVsLevel === report.priceVsLevel ? 1 : 0;
  }

  // Divergencia RSI coincide
  if (cond.hasDivergence !== undefined) {
    checks++;
    score += cond.hasDivergence === report.hasDivergence ? 1 : 0;
  }

  // Mismo Modus Operandi
  if (pattern.mo && report.mo) {
    checks++;
    score += pattern.mo === report.mo ? 1 : 0;
  }

  return checks > 0 ? score / checks : 0;
}

// ── Prioridad por MO desde neural-config ──────────────────────────

const MO_PRIORITY_LABELS = {
  'High_Priority_Reversal': '🔄 Reversión de Alta Prioridad',
  'Volatility_Warning':     '⚡ Alerta de Volatilidad',
  'Long_Trend_Entry':       '📈 Entrada Tendencia Alcista',
  'Short_Trend_Entry':      '📉 Entrada Tendencia Bajista',
};

function getMOPriority(mo, config) {
  const matrix = config?.modus_operandi_matrix || {};
  // Mapear mo id a key del JSON
  const moMap = {
    'liquidity_grab':    'Liquidity_Grab',
    'stop_hunt':         'Stop_Hunt',
    'smart_money_buy':   'Smart_Money_Accumulation',
    'smart_money_sell':  'Distribution_Whale',
    'distribution':      'Distribution_Whale',
    'accumulation':      'Smart_Money_Accumulation',
  };
  const key   = moMap[mo] || '';
  const prio  = matrix[key] || null;
  return prio ? MO_PRIORITY_LABELS[prio] || prio : null;
}

// ── Match Result ───────────────────────────────────────────────────

/**
 * @typedef {object} ArchitectMatch
 * @property {import('../domain/entities.js').NeuralPattern} pattern
 * @property {number} similarity   0-1
 * @property {string} alertMessage Mensaje de notificación
 * @property {string|null} priority Prioridad estratégica del MO
 */

// ── Main: Comparar contra patrones guardados ───────────────────────

/**
 * Compara el contexto forense actual contra los NeuralPatterns en storage.
 * Retorna las coincidencias > MATCH_THRESHOLD ordenadas por similitud.
 *
 * @param {import('../domain/entities.js').ForensicReport} forensicReport
 * @returns {Promise<ArchitectMatch[]>}
 */
export async function matchPatterns(forensicReport) {
  const [patterns, config] = await Promise.all([
    StorageAdapter.loadNeuralPatterns(),
    loadNeuralConfig(),
  ]);

  if (!patterns.length) return [];

  const matches = [];
  for (const pattern of patterns) {
    const sim = computeSimilarity(forensicReport, pattern);
    if (sim >= MATCH_THRESHOLD) {
      const pct      = Math.round(sim * 100);
      const priority = getMOPriority(pattern.mo, config);
      const alertMessage = [
        `🎯 PATRÓN NEURAL DETECTADO: ${pattern.name}`,
        `Confianza Histórica: ${pattern.winRate.toFixed(1)}%`,
        `Similitud: ${pct}%`,
        priority ? `Estrategia: ${priority}` : null,
        `Activaciones: ${pattern.triggerCount}`,
      ].filter(Boolean).join(' · ');

      matches.push({ pattern, similarity: sim, alertMessage, priority });

      // Actualizar triggerCount en storage (fire-and-forget)
      StorageAdapter.incrementPatternTrigger(pattern.id).catch(() => {});
    }
  }

  return matches.sort((a, b) => b.similarity - a.similarity);
}

// ── Reinforcement: actualizar confianza en storage ─────────────────

/**
 * Refuerzo positivo: el patrón se activó y el trade fue WIN.
 * Sube el WR ligeramente (máx 99%).
 * @param {string} patternId
 */
export async function reinforcePositive(patternId) {
  const patterns = await StorageAdapter.loadNeuralPatterns();
  const p = patterns.find(x => x.id === patternId);
  if (!p) return;
  p.winRate    = Math.min(99, p.winRate + CONFIDENCE_BOOST * 100);
  p.lastSeenAt = Date.now();
  await StorageAdapter.set({ [StorageAdapter.KEYS.NEURAL_PATTERNS]: patterns });
  console.log(`[ArchitectAgent] ✅ Refuerzo positivo → ${p.name} WR=${p.winRate.toFixed(1)}%`);
}

/**
 * Refuerzo negativo: el patrón se activó pero el trade fue LOSS.
 * Baja el WR. Si baja de MIN_CONFIDENCE, marca el patrón como degradado.
 * @param {string} patternId
 */
export async function reinforceNegative(patternId) {
  const patterns = await StorageAdapter.loadNeuralPatterns();
  const p = patterns.find(x => x.id === patternId);
  if (!p) return;
  p.winRate    = Math.max(0, p.winRate - CONFIDENCE_DECAY * 100);
  p.lastSeenAt = Date.now();
  if (p.winRate / 100 < MIN_CONFIDENCE) {
    p.degraded = true;
    console.warn(`[ArchitectAgent] ⚠️ Patrón degradado: ${p.name} WR=${p.winRate.toFixed(1)}%`);
  }
  await StorageAdapter.set({ [StorageAdapter.KEYS.NEURAL_PATTERNS]: patterns });
}

/**
 * Elimina patrones que estén por debajo del umbral de efectividad mínimo (50%).
 * @returns {Promise<number>} Número de patrones eliminados
 */
export async function pruneWeakPatterns() {
  const patterns  = await StorageAdapter.loadNeuralPatterns();
  const valid     = patterns.filter(p => p.winRate >= 50);
  const removed   = patterns.length - valid.length;
  if (removed > 0) {
    await StorageAdapter.set({ [StorageAdapter.KEYS.NEURAL_PATTERNS]: valid });
    console.log(`[ArchitectAgent] 🗑️ ${removed} patrón(es) débiles eliminados`);
  }
  return removed;
}

// ── Singleton export ───────────────────────────────────────────────

export const ArchitectAgent = Object.freeze({
  matchPatterns,
  reinforcePositive,
  reinforceNegative,
  pruneWeakPatterns,
  loadNeuralConfig,
  MATCH_THRESHOLD,
});

export default ArchitectAgent;
