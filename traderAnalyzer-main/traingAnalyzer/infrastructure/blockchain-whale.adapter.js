// ═══════════════════════════════════════════════════════════════════
// INFRASTRUCTURE LAYER — Blockchain Whale Adapter
// Fuente alternativa de ballenas: transacciones on-chain grandes
// de BTC usando APIs públicas (sin API key).
//
// Fuentes:
//  1. Blockchain.info — últimos bloques/tx grandes de BTC
//  2. Blockchair API  — top transacciones del día
//  3. Mempool.space   — transacciones no confirmadas grandes
//
// Capa: infrastructure/blockchain-whale.adapter.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

// ── APIs públicas ──────────────────────────────────────────────────

const MEMPOOL_API = 'https://mempool.space/api';

// ── Mempool.space: transacciones recientes grandes ─────────────────

/**
 * Obtiene transacciones recientes del mempool (no confirmadas) con
 * valor alto. Mempool.space es gratuito sin API key.
 * @param {number} minBtc - Mínimo BTC para considerar "ballena"
 * @returns {Promise<object[]>}
 */
async function getMempoolWhales(minBtc = 5) {
  try {
    const r = await fetch(`${MEMPOOL_API}/mempool/recent`);
    if (!r.ok) throw new Error(`Mempool API ${r.status}`);
    const txs = await r.json();
    if (!Array.isArray(txs)) return [];

    const satoshiMin = minBtc * 1e8;
    return txs
      .filter(tx => tx.value >= satoshiMin)
      .slice(0, 50)
      .map(tx => ({
        hash: tx.txid,
        valueBtc: tx.value / 1e8,
        fee: tx.fee / 1e8,
        size: tx.vsize || tx.size,
        timestamp: Date.now(),
        source: 'mempool',
        status: 'unconfirmed',
      }));
  } catch (e) {
    console.warn('[BlockchainWhale] Mempool error:', e.message);
    return [];
  }
}

/**
 * Obtiene detalles de las últimas transacciones confirmadas grandes.
 * Usa el endpoint de bloques recientes de mempool.space
 * @param {number} minBtc
 * @returns {Promise<object[]>}
 */
async function getRecentBlockWhales(minBtc = 10) {
  try {
    // Obtener los últimos 3 bloques
    const tipR = await fetch(`${MEMPOOL_API}/blocks/tip/height`);
    if (!tipR.ok) throw new Error('Mempool tip height failed');
    const tipHeight = parseInt(await tipR.text());

    const blocksR = await fetch(`${MEMPOOL_API}/v1/blocks/${tipHeight}`);
    if (!blocksR.ok) throw new Error('Mempool blocks failed');
    const blocks = await blocksR.json();
    if (!Array.isArray(blocks) || !blocks.length) return [];

    // Para cada bloque reciente, obtener las transacciones grandes
    const whales = [];
    const recentBlocks = blocks.slice(0, 3);

    for (const block of recentBlocks) {
      try {
        const txsR = await fetch(`${MEMPOOL_API}/block/${block.id}/txs`);
        if (!txsR.ok) continue;
        const txs = await txsR.json();
        if (!Array.isArray(txs)) continue;

        const satoshiMin = minBtc * 1e8;
        for (const tx of txs) {
          // Calcular valor total de outputs
          const totalOut = (tx.vout || []).reduce((s, o) => s + (o.value || 0), 0);
          if (totalOut >= satoshiMin) {
            // Clasificar: exchange deposit, withdrawal, o transfer
            const outputCount = (tx.vout || []).length;
            const inputCount  = (tx.vin || []).length;
            const type = classifyTx(inputCount, outputCount, totalOut);

            // Extraer direcciones principales
            const { sender, receiver } = extractAddresses(tx);

            whales.push({
              hash: tx.txid,
              valueBtc: totalOut / 1e8,
              fee: tx.fee / 1e8,
              timestamp: (block.timestamp || Math.floor(Date.now() / 1000)) * 1000,
              blockHeight: block.height,
              source: 'blockchain',
              status: 'confirmed',
              type,
              inputCount,
              outputCount,
              sender,
              receiver,
            });
          }
        }
      } catch (_) { continue; }
    }

    return whales
      .sort((a, b) => b.valueBtc - a.valueBtc)
      .slice(0, 30);
  } catch (e) {
    console.warn('[BlockchainWhale] Block whales error:', e.message);
    return [];
  }
}

/**
 * Obtiene transacciones grandes del último bloque usando mempool.space.
 * Sin límite de requests, totalmente gratuito.
 * @param {number} minBtc
 * @returns {Promise<object[]>}
 */
async function getLatestBlockWhales(minBtc = 5) {
  try {
    // Obtener el hash del último bloque
    const hashR = await fetch(`${MEMPOOL_API}/blocks/tip/hash`);
    if (!hashR.ok) throw new Error(`Mempool tip hash ${hashR.status}`);
    const tipHash = await hashR.text();

    // Obtener las txs del bloque (primera página = 25 más grandes por defecto)
    const txsR = await fetch(`${MEMPOOL_API}/block/${tipHash}/txs`);
    if (!txsR.ok) throw new Error(`Mempool block txs ${txsR.status}`);
    const txs = await txsR.json();
    if (!Array.isArray(txs)) return [];

    const satoshiMin = minBtc * 1e8;
    const whales = [];

    for (const tx of txs) {
      const totalOut = (tx.vout || []).reduce((s, o) => s + (o.value || 0), 0);
      if (totalOut >= satoshiMin) {
        const inputCount = (tx.vin || []).length;
        const outputCount = (tx.vout || []).length;

        // Extraer direcciones principales
        const { sender, receiver } = extractAddresses(tx);

        whales.push({
          hash: tx.txid,
          valueBtc: totalOut / 1e8,
          fee: (tx.fee || 0) / 1e8,
          timestamp: (tx.status?.block_time || Math.floor(Date.now() / 1000)) * 1000,
          blockHeight: tx.status?.block_height,
          source: 'mempool-block',
          status: 'confirmed',
          type: classifyTx(inputCount, outputCount, totalOut),
          inputCount,
          outputCount,
          sender,
          receiver,
        });
      }
    }

    return whales.sort((a, b) => b.valueBtc - a.valueBtc).slice(0, 30);
  } catch (e) {
    console.warn('[BlockchainWhale] Latest block whales error:', e.message);
    return [];
  }
}

// ── Clasificación de transacciones ─────────────────────────────────

/**
 * Extrae la dirección principal de envío y recepción de una tx.
 * Usa la dirección con mayor valor en inputs/outputs.
 */
function extractAddresses(tx) {
  let sender = null, receiver = null;

  // Sender = dirección con mayor input (prevout)
  const inputs = (tx.vin || [])
    .filter(v => v.prevout?.scriptpubkey_address)
    .map(v => ({ addr: v.prevout.scriptpubkey_address, value: v.prevout.value || 0 }));
  if (inputs.length) {
    inputs.sort((a, b) => b.value - a.value);
    sender = inputs[0].addr;
  }

  // Receiver = dirección con mayor output (excluyendo change al sender)
  const outputs = (tx.vout || [])
    .filter(v => v.scriptpubkey_address && v.scriptpubkey_address !== sender)
    .map(v => ({ addr: v.scriptpubkey_address, value: v.value || 0 }));
  if (outputs.length) {
    outputs.sort((a, b) => b.value - a.value);
    receiver = outputs[0].addr;
  } else {
    // Si todos los outputs van al mismo sender (auto-transfer)
    const allOutputs = (tx.vout || []).filter(v => v.scriptpubkey_address);
    if (allOutputs.length) receiver = allOutputs[0].scriptpubkey_address;
  }

  return { sender, receiver };
}

function classifyTx(inputCount, outputCount, totalSatoshi) {
  // Heurísticas para clasificar el tipo de movimiento
  if (inputCount === 1 && outputCount >= 5) return 'distribution'; // Distribución (posible exchange → muchas wallets)
  if (inputCount >= 5 && outputCount <= 2) return 'consolidation'; // Consolidación (muchas wallets → 1, posible acumulación)
  if (inputCount === 1 && outputCount <= 2) return 'transfer'; // Transfer simple (wallet → wallet)
  if (inputCount >= 2 && outputCount >= 5) return 'mixing'; // Posible mixing/coinjoin
  return 'unknown';
}

// ── Análisis de patrones de wallet (usando mempool.space) ──────────

/**
 * Analiza una dirección BTC: balance, transacciones, patrones.
 * @param {string} address - Dirección BTC (bc1..., 1..., 3...)
 * @returns {Promise<object>}
 */
async function analyzeAddress(address) {
  try {
    const [addrR, txsR] = await Promise.all([
      fetch(`${MEMPOOL_API}/address/${address}`),
      fetch(`${MEMPOOL_API}/address/${address}/txs`),
    ]);

    if (!addrR.ok) throw new Error(`Address API ${addrR.status}`);
    const addrData = await addrR.json();
    const txs = txsR.ok ? await txsR.json() : [];

    // Balance
    const funded = addrData.chain_stats?.funded_txo_sum || 0;
    const spent  = addrData.chain_stats?.spent_txo_sum || 0;
    const balanceSat = funded - spent;
    const balanceBtc = balanceSat / 1e8;

    // Estadísticas de transacciones
    const txCount = addrData.chain_stats?.tx_count || 0;
    const mempoolTx = addrData.mempool_stats?.tx_count || 0;

    // Analizar patrones temporales
    const txTimes = txs
      .filter(tx => tx.status?.block_time)
      .map(tx => tx.status.block_time * 1000);

    const sessionCount = { asia: 0, europe: 0, us: 0, late_us: 0 };
    for (const t of txTimes) {
      const h = new Date(t).getUTCHours();
      if (h >= 0 && h < 8) sessionCount.asia++;
      else if (h >= 8 && h < 14) sessionCount.europe++;
      else if (h >= 14 && h < 20) sessionCount.us++;
      else sessionCount.late_us++;
    }
    const dominantSession = Object.keys(sessionCount)
      .reduce((a, b) => sessionCount[a] >= sessionCount[b] ? a : b);

    // Analizar montos
    const amounts = [];
    let totalIn = 0, totalOut = 0;
    for (const tx of txs.slice(0, 50)) {
      const myInputs = (tx.vin || []).filter(v =>
        v.prevout?.scriptpubkey_address === address
      );
      const myOutputs = (tx.vout || []).filter(v =>
        v.scriptpubkey_address === address
      );
      const inputSum  = myInputs.reduce((s, v) => s + (v.prevout?.value || 0), 0);
      const outputSum = myOutputs.reduce((s, v) => s + (v.value || 0), 0);

      if (inputSum > 0) {
        totalOut += inputSum;
        amounts.push({ type: 'send', btc: inputSum / 1e8, time: tx.status?.block_time * 1000 || 0 });
      }
      if (outputSum > 0 && inputSum === 0) {
        totalIn += outputSum;
        amounts.push({ type: 'receive', btc: outputSum / 1e8, time: tx.status?.block_time * 1000 || 0 });
      }
    }

    // Frecuencia de operaciones
    let avgDaysBetweenTx = null;
    if (txTimes.length >= 2) {
      const sorted = [...txTimes].sort((a, b) => a - b);
      const diffs = sorted.slice(1).map((t, i) => t - sorted[i]);
      avgDaysBetweenTx = diffs.reduce((s, d) => s + d, 0) / diffs.length / 86400000;
    }

    // Clasificar tipo de wallet
    let walletType = 'unknown';
    if (txCount > 1000) walletType = 'exchange_or_service';
    else if (balanceBtc > 100 && txCount < 50) walletType = 'cold_storage';
    else if (avgDaysBetweenTx && avgDaysBetweenTx < 1) walletType = 'active_trader';
    else if (avgDaysBetweenTx && avgDaysBetweenTx < 7) walletType = 'swing_trader';
    else walletType = 'holder';

    return {
      address,
      balanceBtc,
      txCount,
      mempoolTx,
      totalIn: totalIn / 1e8,
      totalOut: totalOut / 1e8,
      sessionCount,
      dominantSession,
      avgDaysBetweenTx: avgDaysBetweenTx ? parseFloat(avgDaysBetweenTx.toFixed(2)) : null,
      walletType,
      recentTxs: amounts.slice(0, 20),
      lastActive: txTimes.length ? Math.max(...txTimes) : null,
    };
  } catch (e) {
    console.warn('[BlockchainWhale] Address analysis error:', e.message);
    throw new Error(`No se pudo analizar la dirección: ${e.message}`);
  }
}

/**
 * Obtiene ballenas del día combinando fuentes de mempool.space.
 * 100% gratuito, sin API key.
 * @param {number} minBtc
 * @returns {Promise<object[]>}
 */
async function getDailyWhales(minBtc = 5) {
  // Usar todas las fuentes de mempool.space en paralelo
  const [mempool, mempoolBlocks, latestBlock] = await Promise.allSettled([
    getMempoolWhales(minBtc),
    getRecentBlockWhales(minBtc),
    getLatestBlockWhales(minBtc),
  ]);

  const results = [];
  const seen = new Set();

  // Prioridad: último bloque → bloques recientes → mempool pendientes
  for (const source of [latestBlock, mempoolBlocks, mempool]) {
    if (source.status === 'fulfilled' && source.value.length) {
      for (const tx of source.value) {
        if (!seen.has(tx.hash)) {
          seen.add(tx.hash);
          results.push(tx);
        }
      }
    }
  }

  return results
    .sort((a, b) => b.valueBtc - a.valueBtc)
    .slice(0, 50);
}

// ── Export ──────────────────────────────────────────────────────────

export const BlockchainWhaleAdapter = Object.freeze({
  getMempoolWhales,
  getRecentBlockWhales,
  getLatestBlockWhales,
  getDailyWhales,
  analyzeAddress,
});

export default BlockchainWhaleAdapter;
