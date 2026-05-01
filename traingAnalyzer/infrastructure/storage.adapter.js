// ═══════════════════════════════════════════════════════════════════
// STARK-OS · INFRASTRUCTURE LAYER — Storage Adapter
// Wrapper sobre chrome.storage.local que implementa IStoragePort.
// Añade persistencia del historial de ballenas detectadas por Hunter.
// Capa: infrastructure/storage.adapter.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

const STORAGE_KEYS = {
  HUNTER_HISTORY:   'stark_hunter_history',
  ML_MODEL:         'mlModel',
  PT_BALANCE:       'tvPtBalance',
  PT_POSITIONS:     'tvPtPositions',
  PT_START:         'tvPtStart',
  PT_PEAK:          'tvPtPeak',
  WATCHLIST:        'tvWatchlist',
  NEURAL_PATTERNS:  'stark_neural_patterns',
  WALLET_PROFILES:  'stark_wallet_profiles',
};

/**
 * Obtiene valores de chrome.storage.local.
 * @param {string[]} keys
 * @returns {Promise<Record<string, any>>}
 */
async function get(keys) {
  return new Promise(resolve =>
    chrome.storage.local.get(keys, resolve)
  );
}

/**
 * Guarda valores en chrome.storage.local.
 * @param {Record<string, any>} data
 * @returns {Promise<void>}
 */
async function set(data) {
  return new Promise(resolve =>
    chrome.storage.local.set(data, resolve)
  );
}

/**
 * Elimina claves de chrome.storage.local.
 * @param {string[]} keys
 * @returns {Promise<void>}
 */
async function remove(keys) {
  return new Promise(resolve =>
    chrome.storage.local.remove(keys, resolve)
  );
}

// ── Hunter History ─────────────────────────────────────────────────────────────

/**
 * Carga el historial de señales del Hunter desde el storage.
 * @returns {Promise<import('../domain/entities.js').HunterSignal[]>}
 */
async function loadHunterHistory() {
  const data = await get([STORAGE_KEYS.HUNTER_HISTORY]);
  return Array.isArray(data[STORAGE_KEYS.HUNTER_HISTORY])
    ? data[STORAGE_KEYS.HUNTER_HISTORY]
    : [];
}

/**
 * Persiste una nueva señal del Hunter en el historial.
 * Mantiene solo las últimas 200 señales para no saturar el storage.
 * @param {import('../domain/entities.js').HunterSignal} signal
 * @returns {Promise<void>}
 */
async function appendHunterSignal(signal) {
  const history = await loadHunterHistory();
  const updated = [signal, ...history].slice(0, 200);
  await set({ [STORAGE_KEYS.HUNTER_HISTORY]: updated });
}

/**
 * Persiste el array completo de señales del Hunter.
 * @param {import('../domain/entities.js').HunterSignal[]} signals
 * @returns {Promise<void>}
 */
async function saveHunterHistory(signals) {
  await set({ [STORAGE_KEYS.HUNTER_HISTORY]: signals.slice(0, 200) });
}

/**
 * Limpia el historial del Hunter.
 * @returns {Promise<void>}
 */
async function clearHunterHistory() {
  await remove([STORAGE_KEYS.HUNTER_HISTORY]);
}

// ── Paper Trading ──────────────────────────────────────────────────────────────

async function savePaperTrading({ balance, positions, start, peak }) {
  await set({
    [STORAGE_KEYS.PT_BALANCE]:   balance,
    [STORAGE_KEYS.PT_POSITIONS]: (positions || []).slice(0, 200),
    [STORAGE_KEYS.PT_START]:     start,
    [STORAGE_KEYS.PT_PEAK]:      peak,
  });
}

async function loadPaperTrading() {
  const data = await get([
    STORAGE_KEYS.PT_BALANCE,
    STORAGE_KEYS.PT_POSITIONS,
    STORAGE_KEYS.PT_START,
    STORAGE_KEYS.PT_PEAK,
  ]);
  return {
    balance:   data[STORAGE_KEYS.PT_BALANCE]   ?? null,
    positions: data[STORAGE_KEYS.PT_POSITIONS] ?? [],
    start:     data[STORAGE_KEYS.PT_START]     ?? null,
    peak:      data[STORAGE_KEYS.PT_PEAK]      ?? null,
  };
}

// ── ML Model ───────────────────────────────────────────────────────────────────

async function saveMLModel(model) {
  await set({ [STORAGE_KEYS.ML_MODEL]: model });
}

async function loadMLModel() {
  const data = await get([STORAGE_KEYS.ML_MODEL]);
  return data[STORAGE_KEYS.ML_MODEL] ?? null;
}

// ── Watchlist ──────────────────────────────────────────────────────────────────

async function saveWatchlist(watchlist) {
  await set({ [STORAGE_KEYS.WATCHLIST]: watchlist });
}

async function loadWatchlist() {
  const data = await get([STORAGE_KEYS.WATCHLIST]);
  return data[STORAGE_KEYS.WATCHLIST] ?? null;
}

// ── Neural Patterns ────────────────────────────────────────────────────────────

/**
 * Carga todos los NeuralPatterns guardados.
 * @returns {Promise<import('../domain/entities.js').NeuralPattern[]>}
 */
async function loadNeuralPatterns() {
  const data = await get([STORAGE_KEYS.NEURAL_PATTERNS]);
  return Array.isArray(data[STORAGE_KEYS.NEURAL_PATTERNS])
    ? data[STORAGE_KEYS.NEURAL_PATTERNS]
    : [];
}

/**
 * Guarda (upsert) un NeuralPattern. Si ya existe uno con el mismo id lo reemplaza.
 * Mantiene un máximo de 50 patrones.
 * @param {import('../domain/entities.js').NeuralPattern} pattern
 * @returns {Promise<void>}
 */
async function saveNeuralPattern(pattern) {
  const existing = await loadNeuralPatterns();
  const idx = existing.findIndex(p => p.id === pattern.id);
  if (idx >= 0) {
    existing[idx] = { ...pattern, lastSeenAt: Date.now() };
  } else {
    existing.unshift(pattern);
  }
  await set({ [STORAGE_KEYS.NEURAL_PATTERNS]: existing.slice(0, 50) });
}

/**
 * Incrementa el contador de triggers de un NeuralPattern cuando se detecta de nuevo.
 * @param {string} patternId
 * @returns {Promise<void>}
 */
async function incrementPatternTrigger(patternId) {
  const patterns = await loadNeuralPatterns();
  const p = patterns.find(x => x.id === patternId);
  if (p) {
    p.triggerCount = (p.triggerCount || 0) + 1;
    p.lastSeenAt   = Date.now();
    await set({ [STORAGE_KEYS.NEURAL_PATTERNS]: patterns });
  }
}

/**
 * Elimina un NeuralPattern por id.
 * @param {string} patternId
 * @returns {Promise<void>}
 */
async function removeNeuralPattern(patternId) {
  const patterns = await loadNeuralPatterns();
  await set({
    [STORAGE_KEYS.NEURAL_PATTERNS]: patterns.filter(p => p.id !== patternId),
  });
}

/**
 * Elimina todos los NeuralPatterns.
 * @returns {Promise<void>}
 */
async function clearNeuralPatterns() {
  await remove([STORAGE_KEYS.NEURAL_PATTERNS]);
}

// ── Wallet Profiles (ForensicAgent fingerprint persistence) ────────────────────

/**
 * Carga todos los WalletProfiles guardados.
 * @returns {Promise<object[]>}
 */
async function loadAllWalletProfiles() {
  const data = await get([STORAGE_KEYS.WALLET_PROFILES]);
  return Array.isArray(data[STORAGE_KEYS.WALLET_PROFILES])
    ? data[STORAGE_KEYS.WALLET_PROFILES]
    : [];
}

/**
 * Carga el WalletProfile de un fingerprint específico.
 * @param {string} fingerprint
 * @returns {Promise<object|null>}
 */
async function loadWalletProfile(fingerprint) {
  const all = await loadAllWalletProfiles();
  return all.find(p => p.fingerprint === fingerprint) || null;
}

/**
 * Guarda (upsert) un WalletProfile. Máximo 200 perfiles.
 * @param {object} profile
 * @returns {Promise<void>}
 */
async function saveWalletProfile(profile) {
  const all = await loadAllWalletProfiles();
  const idx = all.findIndex(p => p.fingerprint === profile.fingerprint);
  if (idx >= 0) {
    all[idx] = profile;
  } else {
    all.unshift(profile);
  }
  await set({ [STORAGE_KEYS.WALLET_PROFILES]: all.slice(0, 200) });
}

/**
 * Elimina todos los WalletProfiles.
 * @returns {Promise<void>}
 */
async function clearWalletProfiles() {
  await remove([STORAGE_KEYS.WALLET_PROFILES]);
}

// ─────────────────────────────────────────────────────────────────────────────
// SINGLETON EXPORT — StorageAdapter (implementa IStoragePort)
// ─────────────────────────────────────────────────────────────────────────────

export const StorageAdapter = Object.freeze({
  // IStoragePort primitivos
  get,
  set,
  remove,

  // Hunter history
  loadHunterHistory,
  appendHunterSignal,
  saveHunterHistory,
  clearHunterHistory,

  // Paper trading
  savePaperTrading,
  loadPaperTrading,

  // ML Model
  saveMLModel,
  loadMLModel,

  // Watchlist
  saveWatchlist,
  loadWatchlist,

  // Neural Patterns (Scientist Agent)
  loadNeuralPatterns,
  saveNeuralPattern,
  incrementPatternTrigger,
  removeNeuralPattern,
  clearNeuralPatterns,

  // Wallet Profiles (Forensic Agent fingerprint)
  loadWalletProfile,
  saveWalletProfile,
  loadAllWalletProfiles,
  clearWalletProfiles,

  // Keys
  KEYS: STORAGE_KEYS,
});

export default StorageAdapter;
