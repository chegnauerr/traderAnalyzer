// ═══════════════════════════════════════════════════════════════════
// STARK-OS · UI LAYER — Whale Tracker Controller
// Controlador de interfaz para el tab 🐋 Whale Tracker.
// Conecta la UI (popup.html) con WhaleTrackerAgent (application layer).
// ═══════════════════════════════════════════════════════════════════

'use strict';

import { WhaleTrackerAgent } from './application/whale-tracker.agent.js';
import { BlockchainWhaleAdapter } from './infrastructure/blockchain-whale.adapter.js';

// ── DOM helpers ────────────────────────────────────────────────────

const el     = id  => document.getElementById(id);
const hide   = id  => { const e = el(id); if (e) e.style.display = 'none'; };
const show   = id  => { const e = el(id); if (e) e.style.display = ''; };
const setTxt = (id, t) => { const e = el(id); if (e) e.textContent = t; };

// ── Format helpers ─────────────────────────────────────────────────

const fmtP   = n => {
  if (!n || isNaN(n)) return '—';
  if (n >= 10000) return n.toLocaleString('en', { maximumFractionDigits: 0 });
  if (n >= 100)   return n.toFixed(2);
  if (n >= 1)     return n.toFixed(4);
  return n.toFixed(6);
};

const fmtUsd = n => {
  if (!n && n !== 0) return '—';
  const abs = Math.abs(n);
  const fmt = abs >= 1e6 ? '$' + (abs / 1e6).toFixed(2) + 'M'
            : abs >= 1e3 ? '$' + (abs / 1e3).toFixed(1) + 'K'
            : '$' + abs.toFixed(0);
  return n < 0 ? '-' + fmt : fmt;
};

const fmtAddr = a => a ? a.slice(0, 6) + '…' + a.slice(-4) : '—';

const fmtTime = ts => new Date(ts).toLocaleTimeString('es', {
  hour: '2-digit', minute: '2-digit', second: '2-digit',
});

const pnlColor = pnl => pnl > 0 ? '#00ff41' : pnl < 0 ? '#f85149' : '#888';

const escHtml = s => {
  if (!s) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
};

// ── Error / Loading helpers ────────────────────────────────────────

function showError(msg) {
  const errEl = el('wtError');
  if (errEl) { errEl.textContent = '❌ ' + msg; errEl.style.display = 'block'; }
  hide('wtLoading');
}

function clearError() {
  const errEl = el('wtError');
  if (errEl) errEl.style.display = 'none';
}

function setLoading(on) {
  const loadEl = el('wtLoading');
  if (loadEl) loadEl.style.display = on ? 'flex' : 'none';
}

// ── Render: Wallet DNA ─────────────────────────────────────────────

function renderDna(dna) {
  // Guardar wallet actual para análisis de estrategia
  currentWalletForAnalysis = dna;
  
  const addr = fmtAddr(dna.address);
  setTxt('wtAddressBadge', dna.address);

  // Strategy banner
  const bannerEl = el('wtStrategyBanner');
  if (bannerEl) bannerEl.innerHTML = `<span class="wt-strategy-label">${dna.strategy || 'Desconocida'}</span>`;

  // DNA grid cards
  const pnlColor_ = pnlColor(dna.netPnl);
  const wrColor   = dna.winRate >= 70 ? '#00ff41' : dna.winRate >= 50 ? '#e3b341' : '#f85149';
  const gridEl    = el('wtDnaGrid');
  if (gridEl) {
    gridEl.innerHTML = `
      <div class="wt-dna-cell">
        <div class="wt-dna-lbl">Win Rate</div>
        <div class="wt-dna-val" style="color:${wrColor}">${dna.winRate.toFixed(1)}%</div>
      </div>
      <div class="wt-dna-cell">
        <div class="wt-dna-lbl">Net PnL</div>
        <div class="wt-dna-val" style="color:${pnlColor_}">${fmtUsd(dna.netPnl)}</div>
      </div>
      <div class="wt-dna-cell">
        <div class="wt-dna-lbl">Trades</div>
        <div class="wt-dna-val">${dna.totalFills}</div>
      </div>
      <div class="wt-dna-cell">
        <div class="wt-dna-lbl">Wins / Losses</div>
        <div class="wt-dna-val"><span style="color:#00ff41">${dna.wins}</span> / <span style="color:#f85149">${dna.losses}</span></div>
      </div>
      <div class="wt-dna-cell">
        <div class="wt-dna-lbl">Estilo entrada</div>
        <div class="wt-dna-val" style="font-size:9px">${dna.entryStyle || '—'}</div>
      </div>
      <div class="wt-dna-cell">
        <div class="wt-dna-lbl">Lado dominante</div>
        <div class="wt-dna-val" style="color:${dna.dominantSide === 'LONG' ? '#00ff41' : '#f85149'}">${dna.dominantSide || '—'}</div>
      </div>
      ${dna.avgLeverage ? `
      <div class="wt-dna-cell">
        <div class="wt-dna-lbl">Apalancamiento</div>
        <div class="wt-dna-val">${dna.avgLeverage.toFixed(1)}x</div>
      </div>` : ''}
      <div class="wt-dna-cell">
        <div class="wt-dna-lbl">Fees pagados</div>
        <div class="wt-dna-val" style="color:#888">${fmtUsd(dna.totalFees)}</div>
      </div>
      ${dna.favoriteCoins?.length ? `
      <div class="wt-dna-cell wt-dna-cell-wide">
        <div class="wt-dna-lbl">Monedas favoritas</div>
        <div class="wt-dna-val" style="font-size:9px">${dna.favoriteCoins.map(c => `${c.coin} (${c.count})`).join(' · ')}</div>
      </div>` : ''}
    `;
  }

  // Sessions heatmap
  renderSessions(dna.sessionCount, dna.dominantSession);

  // Current positions
  if (dna.currentPositions?.length) {
    show('wtPositionsSection');
    const tbody = el('wtPositionsTbody');
    if (tbody) {
      tbody.innerHTML = dna.currentPositions.map(p => {
        const side   = p.size > 0 ? '▲ LONG' : '▼ SHORT';
        const sideCls = p.size > 0 ? 'wt-side-long' : 'wt-side-short';
        return `
          <tr>
            <td style="font-weight:700;color:var(--accent)">${p.coin}</td>
            <td><span style="color:${p.size > 0 ? '#00ff41' : '#f85149'};font-weight:900">${side}</span></td>
            <td>${Math.abs(p.size).toFixed(4)}</td>
            <td>$${fmtP(p.entryPx)}</td>
            <td>${p.leverage}x</td>
            <td style="color:${pnlColor(p.unrealPnl)};font-weight:700">${fmtUsd(p.unrealPnl)}</td>
          </tr>
        `;
      }).join('');
    }
  } else {
    hide('wtPositionsSection');
  }

  // Recent fills
  const fillsSub = el('wtFillsSub');
  if (fillsSub) fillsSub.textContent = `${dna.recentFills?.length || 0} últimas operaciones`;

  const fillsTbody = el('wtFillsTbody');
  if (fillsTbody && dna.recentFills?.length) {
    fillsTbody.innerHTML = dna.recentFills.map(f => {
      const side    = f.side === 'B' ? '▲ LONG' : '▼ SHORT';
      const sideColor = f.side === 'B' ? '#00ff41' : '#f85149';
      const pnl     = parseFloat(f.closedPnl || 0);
      return `
        <tr>
          <td style="font-weight:700;color:var(--accent)">${f.coin || '—'}</td>
          <td><span style="color:${sideColor};font-weight:900;font-size:8px">${side}</span></td>
          <td>$${fmtP(parseFloat(f.px || 0))}</td>
          <td>${parseFloat(f.sz || 0).toFixed(4)}</td>
          <td style="color:${pnlColor(pnl)};font-weight:${pnl !== 0 ? '700' : '400'}">${pnl !== 0 ? fmtUsd(pnl) : '—'}</td>
          <td style="color:var(--text2);font-size:9px">${fmtTime(f.time)}</td>
        </tr>
      `;
    }).join('');
  }

  show('wtDnaSection');
  hide('wtLeaderboardSection');
}

// ── Render: Session heatmap ────────────────────────────────────────

const SESSION_LABELS = {
  asia:    'Asia',
  europe:  'Europa',
  us:      'EE.UU.',
  late_us: 'After-Hrs',
};

function renderSessions(sessionCount, dominant) {
  const sessEl = el('wtSessions');
  if (!sessEl || !sessionCount) return;

  const total = Object.values(sessionCount).reduce((s, v) => s + v, 0) || 1;
  sessEl.innerHTML = Object.entries(SESSION_LABELS).map(([key, label]) => {
    const count = sessionCount[key] || 0;
    const pct   = Math.round(count / total * 100);
    const active = key === dominant;
    return `
      <div class="wt-session-bar-wrap">
        <div class="wt-session-lbl">${active ? '⭐ ' : ''}${label}</div>
        <div class="wt-session-bar-bg">
          <div class="wt-session-bar-fill" style="width:${pct}%;${active ? 'background:var(--accent);' : ''}"></div>
        </div>
        <div class="wt-session-count">${pct}%</div>
      </div>
    `;
  }).join('');
}

// ── Render: Leaderboard ────────────────────────────────────────────

function renderLeaderboard(whales) {
  hide('wtDnaSection');
  show('wtLeaderboardSection');

  const tbody = el('wtLeaderTbody');
  if (!tbody) return;

  if (!whales.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text2);padding:16px">⚠️ No se pudieron cargar las vaults. Intenta de nuevo.</td></tr>';
    return;
  }

  console.log('[WT] Renderizando', whales.length, 'vaults');

  tbody.innerHTML = whales.map(w => {
    const pnl     = w.pnl || 0;
    const apr     = (w.apr || 0) * 100; // Convertir a porcentaje
    const aprColor = apr >= 50 ? '#00ff41' : apr >= 20 ? '#e3b341' : apr >= 0 ? '#e3b341' : '#f85149';
    const tvl     = w.tvl || 0;
    return `
      <tr class="wt-leader-row" data-address="${w.address}">
        <td style="color:var(--text2);font-weight:700">#${w.rank}</td>
        <td>
          <div style="font-weight:700;font-size:8.5px;color:var(--accent)">${escHtml(w.displayName)}</div>
          <div style="font-size:7px;color:var(--text2);font-family:monospace;margin-top:1px;">${fmtAddr(w.address)}</div>
          ${w.followers > 0 ? `<div style="font-size:6.5px;color:var(--text2);margin-top:2px;">👥 ${w.followers} followers</div>` : ''}
        </td>
        <td style="color:${aprColor};font-weight:900;font-size:9px;">${apr >= 0 ? '+' : ''}${apr.toFixed(1)}%</td>
        <td style="color:var(--text1);font-weight:700;font-size:8px;">${fmtUsd(tvl)}</td>
        <td style="color:${pnlColor(pnl)};font-weight:700;font-size:8px;">${fmtUsd(pnl)}</td>
        <td>
          <button class="btn-wt-analyze-row" data-address="${w.address}" title="Analizar ADN de esta vault">🔬</button>
        </td>
        <td>
          <button class="btn-wt-follow-row" data-address="${w.address}" data-name="${escHtml(w.displayName)}" title="Seguir esta vault">➕</button>
        </td>
      </tr>
    `;
  }).join('');

  // Botón analizar → llenar dirección y analizar
  tbody.querySelectorAll('.btn-wt-analyze-row').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const addr = btn.dataset.address;
      if (!addr) return;
      const input = el('wtAddressInput');
      if (input) input.value = addr;
      analyzeAddress(addr);
    });
  });

  // Botón seguir → agregar directamente a tracked wallets
  tbody.querySelectorAll('.btn-wt-follow-row').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const addr = btn.dataset.address;
      const name = btn.dataset.name;
      if (!addr) return;
      
      btn.disabled = true;
      btn.textContent = '⏳';
      
      try {
        // Primero analizar para obtener DNA
        const dna = await WhaleTrackerAgent.analyzeWallet(addr);
        // Agregar a seguimiento
        await WhaleTrackerAgent.trackWallet(addr);
        btn.textContent = '✓';
        btn.style.background = 'rgba(0,255,65,.2)';
        btn.style.color = '#00ff41';
        await refreshTracked();
        showNotification(`✓ ${name} agregada al seguimiento`);
      } catch (err) {
        console.error('[WT] Error al seguir vault:', err);
        btn.textContent = '➕';
        btn.disabled = false;
        showError('Error al agregar vault: ' + err.message);
      }
    });
  });
}

// ── Render: Tracked wallets ────────────────────────────────────────

async function refreshTracked() {
  try {
    const tracked = await WhaleTrackerAgent.getTrackedWallets();
    const countEl = el('wtTrackedCount');
    if (countEl) countEl.textContent = tracked.length;

    const section = el('wtTrackedSection');
    const empty = el('wtTrackedEmpty');
    const table = el('wtTrackedTable');
    const tbody = el('wtTrackedTbody');

    if (!tracked.length) {
      if (section) section.style.display = 'none';
      if (empty) empty.style.display = '';
      if (table) table.style.display = 'none';
      return;
    }
    if (section) section.style.display = '';
    if (empty) empty.style.display = 'none';
    if (table) table.style.display = '';

    if (tbody) {
      tbody.innerHTML = tracked.map(w => {
        const wrColor = (w.winRate || 0) >= 60 ? '#00ff41' : (w.winRate || 0) >= 40 ? '#e3b341' : '#f85149';
        return `
          <tr>
            <td style="font-size:9px;color:var(--accent)">${fmtAddr(w.address)}</td>
            <td style="font-size:9px;max-width:90px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${w.label || '—'}</td>
            <td style="color:${wrColor};font-weight:700">${(w.winRate || 0).toFixed(1)}%</td>
            <td>${w.totalFills || '—'}</td>
            <td>
              <button class="btn-wt-analyze-tracked" data-address="${w.address}" title="Cargar wallet">🔬</button>
              <button class="btn-wt-del" data-address="${w.address}" title="Eliminar seguimiento">🗑</button>
            </td>
          </tr>
        `;
      }).join('');

      tbody.querySelectorAll('.btn-wt-analyze-tracked').forEach(btn => {
        btn.addEventListener('click', () => {
          const input = el('wtAddressInput');
          if (input) input.value = btn.dataset.address;
          analyzeAddress(btn.dataset.address);
        });
      });
      tbody.querySelectorAll('.btn-wt-del').forEach(btn => {
        btn.addEventListener('click', async () => {
          await WhaleTrackerAgent.removeTrackedWallet(btn.dataset.address);
          refreshTracked();
        });
      });
    }
  } catch (e) {
    console.warn('[WhaleTracker] refreshTracked error:', e);
  }
}

// ── Core: analyze ─────────────────────────────────────────────────

async function analyzeAddress(address) {
  if (!address || !address.trim()) {
    showError('Ingresa una dirección de wallet válida');
    return;
  }

  address = address.trim();
  console.log('[WT] ═══════════════════════════════════════════');
  console.log('[WT] Iniciando análisis de wallet:', address);
  
  // Feedback visual INMEDIATO en el botón
  const btn = el('wtAnalyzeBtn');
  const originalText = btn ? btn.textContent : '';
  if (btn) {
    btn.disabled = true;
    btn.textContent = '⏳ Analizando...';
    btn.style.opacity = '0.6';
  }

  clearError();
  setLoading(true);
  hide('wtDnaSection');
  hide('wtLeaderboardSection');
  hide('wtDailySection');
  hide('wtBtcDnaSection');
  hide('wtStrategySection');
  hide('wtMirrorSection');

  // Mostrar mensaje de progreso
  const loadingMsg = el('wtLoading');
  if (loadingMsg) {
    loadingMsg.innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px;padding:30px;">
        <div style="font-size:24px;animation:spin 1s linear infinite;">⚙️</div>
        <div style="font-size:9px;color:var(--accent);font-weight:700;">Analizando wallet...</div>
        <div style="font-size:7px;color:var(--text2);font-family:monospace;">${address.slice(0,10)}...${address.slice(-8)}</div>
        <div style="font-size:7px;color:var(--text2);">Descargando historial de operaciones</div>
      </div>
      <style>
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      </style>
    `;
  }

  // Detectar si es dirección BTC (bc1, 1, 3) o Hyperliquid (0x)
  const isBtc = /^(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,62}$/.test(address);
  console.log('[WT] Tipo de wallet detectado:', isBtc ? 'Bitcoin' : 'Hyperliquid');

  try {
    if (isBtc) {
      console.log('[WT] Consultando mempool.space API...');
      const profile = await BlockchainWhaleAdapter.analyzeAddress(address);
      console.log('[WT] ✓ Perfil BTC obtenido:', profile);
      renderBtcProfile(profile);
    } else {
      console.log('[WT] Consultando Hyperliquid API...');
      const startTime = Date.now();
      const dna = await WhaleTrackerAgent.analyzeWallet(address);
      const elapsed = Date.now() - startTime;
      console.log(`[WT] ✓ ADN obtenido en ${elapsed}ms:`, dna);
      console.log(`[WT]   - Trades totales: ${dna.totalFills}`);
      console.log(`[WT]   - Win Rate: ${dna.winRate}%`);
      console.log(`[WT]   - Estrategia: ${dna.strategy}`);
      renderDna(dna);
      // Mostrar secciones de análisis para wallets Hyperliquid
      show('wtStrategySection');
      show('wtMirrorSection');
    }
    await refreshTracked();
    console.log('[WT] ✓ Análisis completado exitosamente');
  } catch (e) {
    console.error('[WT] ✗ Error durante análisis:', e);
    console.error('[WT] Stack trace:', e.stack);
    showError(e.message || String(e));
  } finally {
    setLoading(false);
    // Restaurar botón
    if (btn) {
      btn.disabled = false;
      btn.textContent = originalText;
      btn.style.opacity = '1';
    }
    console.log('[WT] ═══════════════════════════════════════════');
  }
}

async function loadLeaderboard() {
  clearError();
  setLoading(true);
  hide('wtDnaSection');
  hide('wtLeaderboardSection');
  hide('wtDailySection');
  hide('wtBtcDnaSection');

  try {
    const whales = await WhaleTrackerAgent.getTopWhales(20);
    renderLeaderboard(whales);
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    setLoading(false);
  }
}

// ── Daily On-Chain Whales ─────────────────────────────────────────

async function loadDailyWhales() {
  clearError();
  setLoading(true);
  hide('wtDnaSection');
  hide('wtLeaderboardSection');
  hide('wtBtcDnaSection');

  try {
    const whales = await BlockchainWhaleAdapter.getDailyWhales(10);
    renderDailyWhales(whales);
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    setLoading(false);
  }
}

function renderDailyWhales(whales) {
  show('wtDailySection');
  const tbody = el('wtDailyTbody');
  if (!tbody) return;

  if (!whales.length) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text2);padding:16px">Sin transacciones grandes detectadas</td></tr>';
    return;
  }

  const typeLabels = {
    distribution: '📦 Distribución',
    consolidation: '🏗️ Consolidación',
    transfer: '↔️ Transfer',
    mixing: '🌀 Mixing',
    unknown: '❓ Desconocido',
  };
  const typeColors = {
    distribution: '#f85149',
    consolidation: '#00ff41',
    transfer: '#e3b341',
    mixing: '#a78bfa',
    unknown: '#888',
  };

  tbody.innerHTML = whales.map((w, i) => {
    const typeLbl = typeLabels[w.type] || typeLabels.unknown;
    const typeCol = typeColors[w.type] || '#888';
    const time = w.timestamp ? new Date(w.timestamp).toLocaleTimeString('es', { hour:'2-digit', minute:'2-digit' }) : '—';
    const statusBadge = w.status === 'unconfirmed'
      ? '<span style="color:#e3b341;font-size:7px">⏳ MEMPOOL</span>'
      : '<span style="color:#00ff41;font-size:7px">✓ CONF.</span>';
    const senderShort = w.sender ? w.sender.slice(0, 8) + '…' + w.sender.slice(-4) : '—';
    const receiverShort = w.receiver ? w.receiver.slice(0, 8) + '…' + w.receiver.slice(-4) : '—';
    return `
      <tr class="wt-daily-row" data-idx="${i}" style="cursor:pointer" title="Clic para ver detalles">
        <td style="font-weight:900;color:var(--accent);font-family:'Orbitron',monospace">${w.valueBtc.toFixed(2)} BTC</td>
        <td style="color:${typeCol};font-weight:700;font-size:8px">${typeLbl}</td>
        <td style="font-size:8px">${statusBadge}</td>
        <td style="color:var(--text2);font-size:8px">${time}</td>
        <td style="font-family:monospace;font-size:7px;color:var(--text2);max-width:80px;overflow:hidden;text-overflow:ellipsis" title="${w.hash}">${w.hash ? w.hash.slice(0, 10) + '…' : '—'}</td>
      </tr>
      <tr class="wt-daily-detail" id="wt-dd-${i}" style="display:none">
        <td colspan="5">
          <div class="wt-detail-panel">
            <div class="wt-detail-row">
              <span class="wt-detail-lbl">📤 Emisor:</span>
              <span class="wt-detail-addr" title="${w.sender || ''}">${senderShort}</span>
              ${w.sender ? `<button class="btn-wt-detail-analyze" data-address="${w.sender}">🔬 Analizar</button>` : ''}
            </div>
            <div class="wt-detail-row">
              <span class="wt-detail-lbl">📥 Receptor:</span>
              <span class="wt-detail-addr" title="${w.receiver || ''}">${receiverShort}</span>
              ${w.receiver ? `<button class="btn-wt-detail-analyze" data-address="${w.receiver}">🔬 Analizar</button>` : ''}
            </div>
            <div class="wt-detail-row">
              <span class="wt-detail-lbl">⛽ Fee:</span>
              <span style="color:var(--text2)">${w.fee ? w.fee.toFixed(6) + ' BTC' : '—'}</span>
            </div>
            <div class="wt-detail-row">
              <span class="wt-detail-lbl">📊 Inputs/Outputs:</span>
              <span style="color:var(--text2)">${w.inputCount || '?'} → ${w.outputCount || '?'}</span>
            </div>
            <div class="wt-detail-row">
              <span class="wt-detail-lbl">🔗 Hash:</span>
              <a class="wt-detail-hash" href="https://mempool.space/tx/${w.hash}" target="_blank" rel="noopener">${w.hash || '—'}</a>
            </div>
          </div>
        </td>
      </tr>
    `;
  }).join('');

  // Toggle detail rows on click
  tbody.querySelectorAll('.wt-daily-row').forEach(row => {
    row.addEventListener('click', () => {
      const detail = el(`wt-dd-${row.dataset.idx}`);
      if (detail) detail.style.display = detail.style.display === 'none' ? '' : 'none';
    });
  });

  // Analyze buttons inside detail panels
  tbody.querySelectorAll('.btn-wt-detail-analyze').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const addr = btn.dataset.address;
      if (addr) {
        const input = el('wtAddressInput');
        if (input) input.value = addr;
        analyzeAddress(addr);
      }
    });
  });
}

// ── BTC Wallet Profile ────────────────────────────────────────────

function renderBtcProfile(profile) {
  show('wtBtcDnaSection');
  const section = el('wtBtcDnaSection');
  if (section) section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  const grid = el('wtBtcDnaGrid');
  if (!grid) return;

  const walletLabels = {
    exchange_or_service: '🏦 Exchange / Servicio',
    cold_storage: '🧊 Cold Storage',
    active_trader: '⚡ Trader Activo',
    swing_trader: '📈 Swing Trader',
    holder: '💎 Holder',
    unknown: '❓ Desconocido',
  };

  const balColor = profile.balanceBtc >= 10 ? '#00ff41' : profile.balanceBtc >= 1 ? '#e3b341' : '#888';
  const typeLabel = walletLabels[profile.walletType] || walletLabels.unknown;

  setTxt('wtBtcAddressBadge', profile.address);

  grid.innerHTML = `
    <div class="wt-dna-cell" style="grid-column:span 2">
      <div class="wt-dna-lbl">Tipo de Wallet</div>
      <div class="wt-dna-val" style="font-size:10px">${typeLabel}</div>
    </div>
    <div class="wt-dna-cell">
      <div class="wt-dna-lbl">Balance</div>
      <div class="wt-dna-val" style="color:${balColor}">${profile.balanceBtc.toFixed(4)} BTC</div>
    </div>
    <div class="wt-dna-cell">
      <div class="wt-dna-lbl">Total Txs</div>
      <div class="wt-dna-val">${profile.txCount}</div>
    </div>
    <div class="wt-dna-cell">
      <div class="wt-dna-lbl">Recibido</div>
      <div class="wt-dna-val" style="color:#00ff41">${profile.totalIn.toFixed(3)} BTC</div>
    </div>
    <div class="wt-dna-cell">
      <div class="wt-dna-lbl">Enviado</div>
      <div class="wt-dna-val" style="color:#f85149">${profile.totalOut.toFixed(3)} BTC</div>
    </div>
    ${profile.avgDaysBetweenTx != null ? `
    <div class="wt-dna-cell">
      <div class="wt-dna-lbl">Frecuencia</div>
      <div class="wt-dna-val">${profile.avgDaysBetweenTx < 1 ? 'Varias/día' : profile.avgDaysBetweenTx.toFixed(1) + ' días'}</div>
    </div>` : ''}
    <div class="wt-dna-cell">
      <div class="wt-dna-lbl">Sesión dominante</div>
      <div class="wt-dna-val">${SESSION_LABELS[profile.dominantSession] || '—'}</div>
    </div>
    ${profile.mempoolTx > 0 ? `
    <div class="wt-dna-cell">
      <div class="wt-dna-lbl">En Mempool</div>
      <div class="wt-dna-val" style="color:#e3b341">${profile.mempoolTx} tx pendientes</div>
    </div>` : ''}
    ${profile.lastActive ? `
    <div class="wt-dna-cell" style="grid-column:span 3">
      <div class="wt-dna-lbl">Última actividad</div>
      <div class="wt-dna-val" style="font-size:8px">${new Date(profile.lastActive).toLocaleString('es')}</div>
    </div>` : ''}
  `;

  // Render recent txs
  const txList = el('wtBtcTxsTbody');
  if (txList && profile.recentTxs?.length) {
    show('wtBtcTxsSection');
    txList.innerHTML = profile.recentTxs.map(tx => {
      const col = tx.type === 'receive' ? '#00ff41' : '#f85149';
      const icon = tx.type === 'receive' ? '▼ Recibido' : '▲ Enviado';
      const time = tx.time ? new Date(tx.time).toLocaleString('es', { month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit' }) : '—';
      return `
        <tr>
          <td style="color:${col};font-weight:700;font-size:8px">${icon}</td>
          <td style="font-weight:700;color:var(--accent)">${tx.btc.toFixed(4)} BTC</td>
          <td style="color:var(--text2);font-size:8px">${time}</td>
        </tr>
      `;
    }).join('');
  } else {
    hide('wtBtcTxsSection');
  }

  // Sessions heatmap
  if (profile.sessionCount) {
    const sessEl = el('wtBtcSessions');
    if (sessEl) {
      const total = Object.values(profile.sessionCount).reduce((s, v) => s + v, 0) || 1;
      sessEl.innerHTML = Object.entries(SESSION_LABELS).map(([key, label]) => {
        const count = profile.sessionCount[key] || 0;
        const pct   = Math.round(count / total * 100);
        const active = key === profile.dominantSession;
        return `
          <div class="wt-session-bar-wrap">
            <div class="wt-session-lbl">${active ? '⭐ ' : ''}${label}</div>
            <div class="wt-session-bar-bg">
              <div class="wt-session-bar-fill" style="width:${pct}%;${active ? 'background:var(--accent);' : ''}"></div>
            </div>
            <div class="wt-session-count">${pct}%</div>
          </div>
        `;
      }).join('');
    }
    show('wtBtcSessionsWrap');
  }
}

// ── Export: initWhaleTracker ───────────────────────────────────────

// ── Strategy Analysis ──────────────────────────────────────────────

let currentWalletForAnalysis = null;
let mirrorTradeInterval = null;

async function analyzeStrategy() {
  if (!currentWalletForAnalysis) {
    console.warn('[WT] No hay wallet seleccionada para analizar');
    showError('Primero analiza una wallet con el botón 🔬');
    return;
  }
  const addr = currentWalletForAnalysis.address;
  
  // Feedback visual INMEDIATO
  const btn = el('wtAnalyzeStrategyBtn');
  const originalText = btn ? btn.textContent : '';
  if (btn) {
    btn.disabled = true;
    btn.textContent = '⏳ Analizando...';
    btn.style.opacity = '0.6';
  }
  
  // Mostrar sección con mensaje de carga
  show('wtStrategySection');
  const contentEl = el('wtStrategyContent');
  if (contentEl) {
    contentEl.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:center;gap:8px;padding:20px;flex-direction:column;">
        <div style="font-size:20px;animation:spin 1s linear infinite;">⚙️</div>
        <div style="font-size:8px;color:var(--accent);font-weight:700;">Calculando patrones de trading...</div>
        <div style="font-size:7px;color:var(--text2);">Analizando ${currentWalletForAnalysis.totalFills || '...'} operaciones</div>
      </div>
      <style>
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      </style>
    `;
  }
  
  console.log('[WT] Analizando estrategia de:', addr);
  setLoading(true);
  
  try {
    // Obtener todos los fills recientes
    const dna = await WhaleTrackerAgent.analyzeWallet(addr);
    const fills = dna.recentFills || [];
    
    console.log('[WT] Fills obtenidos:', fills.length);
    
    if (fills.length < 3) {
      if (contentEl) {
        contentEl.innerHTML = `
          <div style="font-size:8px;color:var(--text2);padding:10px;text-align:center;">
            ⚠️ Insuficientes trades para análisis (mínimo 3)<br>
            <span style="font-size:7px;margin-top:4px;display:block;">Trades encontrados: ${fills.length}</span>
          </div>
        `;
      }
      return;
    }

    // Clasificar cada trade por patrón
    const patterns = detectTradePatterns(fills, dna);
    console.log('[WT] Patrones detectados:', patterns);
    renderStrategyAnalysis(patterns, dna);
    show('wtMirrorSection');
    
  } catch (e) {
    console.error('[WT] Error al analizar estrategia:', e);
    showError('Error al analizar estrategia: ' + e.message);
    if (contentEl) {
      contentEl.innerHTML = `
        <div style="font-size:8px;color:#f85149;padding:10px;text-align:center;">
          ❌ Error al analizar estrategia<br>
          <span style="font-size:7px;margin-top:4px;display:block;">${e.message}</span>
        </div>
      `;
    }
  } finally {
    setLoading(false);
    // Restaurar botón
    if (btn) {
      btn.disabled = false;
      btn.textContent = originalText;
      btn.style.opacity = '1';
    }
  }
}

function detectTradePatterns(fills, dna) {
  const patterns = {
    breakout: 0,
    pullback: 0,
    momentum: 0,
    reversal: 0,
    scalp: 0,
  };
  
  const rules = {
    preferredSession: dna.dominantSession || 'unknown',
    avgLeverage: dna.avgLeverage || 1,
    preferredSide: dna.dominantSide || 'neutral',
    entryStyle: dna.entryStyle || 'unknown',
    topCoins: dna.favoriteCoins ? dna.favoriteCoins.map(c => c.coin).slice(0, 5) : [],
  };

  // Analizar fills
  fills.slice(0, 50).forEach((fill, i) => {
    // Detectar scalping (muchas operaciones rápidas en poco tiempo)
    if (i > 0 && fills[i - 1]) {
      const timeDiff = Math.abs(fill.time - fills[i - 1].time);
      if (timeDiff < 300000 && fill.coin === fills[i - 1].coin) {
        patterns.scalp++;
      }
    }
    
    // Breakout: entrada agresiva (taker) en dirección de momento
    if (fill.dir === 'Open Long' || fill.dir === 'Open Short') {
      patterns.breakout++;
    }
    
    // Pullback: entrada límite paciente
    if (dna.entryStyle?.includes('Limit')) {
      patterns.pullback++;
    }
    
    // Momentum: muchas operaciones en el mismo lado
    if ((fill.dir.includes('Long') && dna.dominantSide === 'LONG') ||
        (fill.dir.includes('Short') && dna.dominantSide === 'SHORT')) {
      patterns.momentum++;
    }
    
    // Reversal: cambia de lado frecuentemente
    if (i > 0 && fills[i - 1]) {
      const prevLong = fills[i - 1].dir.includes('Long');
      const currLong = fill.dir.includes('Long');
      if (prevLong !== currLong) {
        patterns.reversal++;
      }
    }
  });

  // Determinar patrón dominante
  const total = Object.values(patterns).reduce((a, b) => a + b, 0);
  const dominant = Object.keys(patterns).reduce((a, b) => 
    patterns[a] > patterns[b] ? a : b
  );

  return {
    patterns,
    total,
    dominant,
    rules,
    confidence: total > 0 ? Math.round((patterns[dominant] / total) * 100) : 0,
  };
}

function renderStrategyAnalysis(analysis, dna) {
  const { patterns, dominant, rules, confidence } = analysis;
  
  const patternLabels = {
    breakout: '🚀 Breakout (momentum agresivo)',
    pullback: '⏳ Pullback (espera retrocesos)',
    momentum: '📈 Momentum (sigue tendencia)',
    reversal: '🔄 Reversión (cambios de dirección)',
    scalp: '⚡ Scalping (entradas/salidas rápidas)',
  };

  const patternDesc = {
    breakout: 'Entra cuando el precio rompe niveles clave con volumen',
    pullback: 'Espera retrocesos a soporte/resistencia antes de entrar',
    momentum: 'Sigue la tendencia dominante, evita contra-tendencia',
    reversal: 'Identifica cambios de tendencia y pivotea rápido',
    scalp: 'Operaciones muy rápidas, captura movimientos mínimos',
  };

  const content = el('wtStrategyContent');
  if (!content) return;

  content.innerHTML = `
    <div style="background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.2);border-radius:4px;padding:10px;margin-bottom:10px;">
      <div style="font-size:9px;font-weight:900;color:var(--accent);margin-bottom:5px;">
        PATRÓN DOMINANTE: ${patternLabels[dominant]}
      </div>
      <div style="font-size:7.5px;color:var(--text2);margin-bottom:8px;">
        ${patternDesc[dominant]}
      </div>
      <div style="font-size:7px;color:var(--text2);">
        Confianza: <span style="color:${confidence >= 70 ? '#00ff41' : confidence >= 50 ? '#e3b341' : '#f85149'}">${confidence}%</span>
      </div>
    </div>

    <div style="font-size:8px;font-weight:700;margin-bottom:6px;color:var(--text1);">📊 Distribución de Patrones</div>
    ${Object.keys(patterns).map(key => {
      const pct = analysis.total > 0 ? Math.round((patterns[key] / analysis.total) * 100) : 0;
      return `
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
          <div style="font-size:7.5px;color:var(--text2);min-width:120px;">${patternLabels[key]}</div>
          <div style="flex:1;background:rgba(255,255,255,.05);border-radius:2px;height:8px;overflow:hidden;">
            <div style="background:var(--accent);height:100%;width:${pct}%;transition:width .3s;"></div>
          </div>
          <div style="font-size:7px;color:var(--accent);min-width:30px;text-align:right;">${pct}%</div>
        </div>
      `;
    }).join('')}

    <div style="font-size:8px;font-weight:700;margin:12px 0 6px;color:var(--text1);">🎯 Reglas Detectadas (Copiar a tu trading)</div>
    <div style="background:rgba(0,255,65,.06);border:1px solid rgba(0,255,65,.2);border-radius:4px;padding:8px;margin-bottom:8px;">
      <div style="font-size:7.5px;color:var(--text2);line-height:1.8;">
        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
          <span>⏰ <strong style="color:var(--accent)">Sesión óptima:</strong></span>
          <span style="color:#00ff41;font-weight:900;">${rules.preferredSession}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
          <span>📊 <strong style="color:var(--accent)">Lado a operar:</strong></span>
          <span style="color:${rules.preferredSide === 'LONG' ? '#00ff41' : rules.preferredSide === 'SHORT' ? '#f85149' : '#e3b341'};font-weight:900;">
            ${rules.preferredSide === 'LONG' ? '▲ LONG' : rules.preferredSide === 'SHORT' ? '▼ SHORT' : '↔ NEUTRAL'}
          </span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
          <span>💎 <strong style="color:var(--accent)">Apalancamiento:</strong></span>
          <span style="color:#e3b341;font-weight:900;">${rules.avgLeverage.toFixed(1)}x</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
          <span>🎯 <strong style="color:var(--accent)">Tipo de entrada:</strong></span>
          <span style="color:var(--text1);">${rules.entryStyle}</span>
        </div>
        <div style="margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,.08);">
          <div style="font-size:7px;color:var(--accent);font-weight:700;margin-bottom:3px;">💰 MONEDAS CON MEJOR RENDIMIENTO:</div>
          <div style="font-size:7px;color:var(--text1);">${rules.topCoins.slice(0,5).join(' · ') || 'N/A'}</div>
        </div>
      </div>
    </div>

    <div style="background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.3);border-radius:4px;padding:8px;">
      <div style="font-size:7.5px;font-weight:700;color:var(--accent);margin-bottom:4px;">💡 CÓMO COPIAR ESTAS REGLAS:</div>
      <div style="font-size:7px;color:var(--text2);line-height:1.7;">
        1️⃣ Esta wallet está en tu lista de seguimiento<br>
        2️⃣ Ve al tab <strong style="color:var(--accent)">Análisis</strong><br>
        3️⃣ Selecciona una de estas monedas: <strong style="color:var(--accent)">${rules.topCoins.slice(0,3).join(', ')}</strong><br>
        4️⃣ Haz clic en <strong style="color:var(--accent)">▶ Analizar</strong><br>
        5️⃣ Verás la sección <strong style="color:#00ff41">🧠 Smart Money Rules</strong><br>
        6️⃣ Te mostrará automáticamente señales LONG/SHORT, zona de entrada sugerida, y probabilidades basadas en el patrón de esta wallet<br>
        <div style="margin-top:6px;padding:6px;background:rgba(0,255,65,.1);border-radius:3px;border-left:2px solid #00ff41;">
          <strong style="color:#00ff41;">✨ TIP:</strong> El sistema calcula automáticamente cuándo tomar SHORT o LONG según el historial de esta wallet en ese par específico.
        </div>
      </div>
    </div>
  `;
}

// ── Mirror Trading System ──────────────────────────────────────────

let mirrorActive = false;
let lastSeenFills = [];

function activateMirrorTrading() {
  if (mirrorActive) return;
  if (!currentWalletForAnalysis) {
    showError('Primero analiza una wallet');
    return;
  }

  mirrorActive = true;
  const statusBadge = el('wtMirrorStatus');
  if (statusBadge) {
    statusBadge.textContent = '● Activo';
    statusBadge.style.background = 'rgba(0,255,65,.08)';
    statusBadge.style.color = '#00ff41';
  }

  show('wtActivateMirrorBtn', false);
  el('wtActivateMirrorBtn').style.display = 'none';
  el('wtStopMirrorBtn').style.display = '';

  logMirror('🚀 Mirror Trading activado para ' + fmtAddr(currentWalletForAnalysis.address));
  logMirror('⏳ Monitoreando nuevas operaciones cada 10 segundos...');

  // Inicializar con fills actuales
  lastSeenFills = currentWalletForAnalysis.recentFills || [];

  // Poll cada 10 segundos - DESACTIVADO (no hacer llamadas a Hyperliquid)
  // mirrorTradeInterval = setInterval(checkForNewTrades, 10000);
}

function stopMirrorTrading() {
  if (!mirrorActive) return;
  
  mirrorActive = false;
  if (mirrorTradeInterval) {
    clearInterval(mirrorTradeInterval);
    mirrorTradeInterval = null;
  }

  const statusBadge = el('wtMirrorStatus');
  if (statusBadge) {
    statusBadge.textContent = '● Inactivo';
    statusBadge.style.background = 'rgba(255,255,255,.05)';
    statusBadge.style.color = 'var(--text2)';
  }

  el('wtActivateMirrorBtn').style.display = '';
  el('wtStopMirrorBtn').style.display = 'none';

  logMirror('⏹ Mirror Trading detenido');
}

async function checkForNewTrades() {
  if (!mirrorActive || !currentWalletForAnalysis) return;

  // DESACTIVADO - No hacer llamadas a Hyperliquid
  return;

  /*
  try {
    const dna = await WhaleTrackerAgent.analyzeWallet(currentWalletForAnalysis.address);
    const newFills = dna.recentFills || [];

    // Detectar fills nuevos
    const newTrades = newFills.filter(nf => 
      !lastSeenFills.some(lf => lf.time === nf.time && lf.coin === nf.coin && lf.px === nf.px)
    );

    if (newTrades.length > 0) {
      newTrades.forEach(trade => {
        const isLong = trade.dir.includes('Long');
        const isOpen = trade.dir.includes('Open');
        
        if (isOpen) {
          logMirror(`🎯 NUEVA ENTRADA DETECTADA: ${isLong ? '▲ LONG' : '▼ SHORT'} ${trade.coin} @ $${fmtP(trade.px)} (${trade.sz})`);
          logMirror(`💡 Replicando en demo...`);
          
          // Aquí ejecutarías la lógica de demo trading
          // Por ahora solo mostramos la alerta
          showNotification(`🔔 Mirror Trade: ${isLong ? 'LONG' : 'SHORT'} ${trade.coin} @ $${fmtP(trade.px)}`);
        }
      });
    }

    lastSeenFills = newFills;
  } catch (e) {
    console.error('[Mirror] Error checking trades:', e);
  }
  */
}

function logMirror(msg) {
  const log = el('wtMirrorLog');
  if (!log) return;
  
  const time = new Date().toLocaleTimeString('es', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const entry = document.createElement('div');
  entry.style.cssText = 'font-size:7.5px;color:var(--text2);margin-bottom:3px;padding:3px 6px;background:rgba(255,255,255,.02);border-radius:2px;';
  entry.textContent = `[${time}] ${msg}`;
  
  log.insertBefore(entry, log.firstChild);
  
  // Keep only last 20 entries
  while (log.children.length > 20) {
    log.removeChild(log.lastChild);
  }
}

function showNotification(msg) {
  const notif = el('notification');
  if (!notif) return;
  
  notif.textContent = msg;
  notif.style.display = 'block';
  notif.style.animation = 'slideIn .3s ease-out';
  
  setTimeout(() => {
    notif.style.animation = 'slideOut .3s ease-out';
    setTimeout(() => notif.style.display = 'none', 300);
  }, 4000);
}

// ── Init ───────────────────────────────────────────────────────────

export function initWhaleTracker() {
  // Analyze button — soporta 0x (Hyperliquid) y bc1/1/3 (BTC on-chain)
  el('wtAnalyzeBtn')?.addEventListener('click', () => {
    const addr = (el('wtAddressInput')?.value || '').trim();
    if (!addr) { showError('Introduce una dirección (0x... para Hyperliquid, bc1.../1.../3... para BTC)'); return; }
    analyzeAddress(addr);
  });

  // Enter key on address input
  el('wtAddressInput')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      const addr = (el('wtAddressInput')?.value || '').trim();
      if (addr) analyzeAddress(addr);
    }
  });

  // Leaderboard button (Hyperliquid vaults)
  el('wtLeaderboardBtn')?.addEventListener('click', () => loadLeaderboard());

  // Daily whales button (on-chain BTC)
  el('wtDailyBtn')?.addEventListener('click', () => loadDailyWhales());

  // Strategy Analysis button
  el('wtAnalyzeStrategyBtn')?.addEventListener('click', () => analyzeStrategy());

  // Mirror Trading buttons
  el('wtActivateMirrorBtn')?.addEventListener('click', () => activateMirrorTrading());
  el('wtStopMirrorBtn')?.addEventListener('click', () => stopMirrorTrading());

  // Load tracked wallets on init
  refreshTracked();
}
