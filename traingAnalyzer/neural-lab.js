// ═══════════════════════════════════════════════════════════════════
// STARK-OS · UI LAYER — Neural Lab Controller v2
// Forensic Agent panel + Scientist popup integrados.
// Capa: neural-lab.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

import { HunterAgent }    from './application/hunter.agent.js';
import { ForensicAgent }  from './application/forensic.agent.js';
import { ScientistAgent } from './application/scientist.agent.js';
import { ArchitectAgent } from './application/architect.agent.js';
import { StorageAdapter } from './infrastructure/storage.adapter.js';

// ── Format helpers ─────────────────────────────────────────────────

const fmtP = n => {
  if (!n || isNaN(n)) return '—';
  if (n >= 10000) return n.toLocaleString('en', { maximumFractionDigits: 0 });
  if (n >= 100)   return n.toFixed(2);
  if (n >= 1)     return n.toFixed(4);
  return n.toFixed(6);
};
const fmtBtc  = n  => n != null ? n.toFixed(3) : '—';
const fmtUsd  = n  => {
  if (!n) return '—';
  if (n >= 1e6) return '$' + (n/1e6).toFixed(2) + 'M';
  if (n >= 1e3) return '$' + (n/1e3).toFixed(1) + 'K';
  return '$' + n.toFixed(0);
};
const fmtTime = ts => new Date(ts).toLocaleTimeString('es', { hour:'2-digit', minute:'2-digit', second:'2-digit' });
const fmtDT   = ts => new Date(ts).toLocaleString('es', { month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit' });

// ── DOM helpers ────────────────────────────────────────────────────

const el  = id => document.getElementById(id);
const show = id => { const e = el(id); if (e) e.style.display = ''; };
const hide = id => { const e = el(id); if (e) e.style.display = 'none'; };
const setTxt = (id, t) => { const e = el(id); if (e) e.textContent = t; };

// ── Badge helpers ──────────────────────────────────────────────────

const wrBadge = wr => {
  const cls = wr >= 80 ? 'nl-wr-high' : 'nl-wr-mid';
  return `<span class="nl-wr-badge ${cls}">${wr.toFixed(1)}%</span>`;
};
const dirBadge = dir => {
  const cls = dir === 'LONG' ? 'nl-dir-long' : dir === 'SHORT' ? 'nl-dir-short' : 'nl-dir-wait';
  const icon = dir === 'LONG' ? '▲' : dir === 'SHORT' ? '▼' : '→';
  return `<span class="${cls}">${icon} ${dir}</span>`;
};
const moBadge = (mo, moLabel, moIcon) =>
  `<span class="nl-mo-badge nl-mo-${mo}">${moIcon} ${moLabel}</span>`;

// ── Pattern Library ────────────────────────────────────────────────

const MO_ICONS = {
  smart_money_buy:  '🟢', smart_money_sell: '🔴', distribution:  '📦',
  accumulation:     '🏗️', panic_sell:       '🚨', liquidity_grab: '⚡',
  stop_hunt:        '🎯', unknown:          '🔍',
};

async function refreshPatternLibrary() {
  try {
    const patterns = await StorageAdapter.loadNeuralPatterns();
    const countEl  = el('nlPatternCount');
    if (countEl) countEl.textContent = `${patterns.length} patrones`;

    const empty   = el('nlPatternEmpty');
    const table   = el('nlPatternTable');
    const tbody   = el('nlPatternTbody');

    if (!patterns.length) {
      if (empty) empty.style.display = '';
      if (table) table.style.display = 'none';
      return;
    }
    if (empty) empty.style.display = 'none';
    if (table) table.style.display = '';

    tbody.innerHTML = patterns.map(p => {
      const wrCol   = p.winRate >= 70 ? '#00ff41' : p.winRate >= 55 ? '#e3b341' : '#f85149';
      const icon    = MO_ICONS[p.mo] || '🔍';
      const dirCls  = p.direction === 'LONG' ? 'nl-dir-long' : p.direction === 'SHORT' ? 'nl-dir-short' : 'nl-dir-wait';
      const degraded= p.degraded ? ' nl-pattern-degraded' : '';
      return `
        <tr class="nl-pattern-row${degraded}" data-id="${p.id}">
          <td style="font-size:8px;color:var(--text1);font-weight:700;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${p.name}">${p.name}</td>
          <td><span style="font-size:11px">${icon}</span></td>
          <td><span class="${dirCls}">${p.direction}</span></td>
          <td>
            <div class="nl-wr-mini-bar">
              <div class="nl-wr-mini-fill" style="width:${p.winRate}%;background:${wrCol}"></div>
            </div>
            <span style="font-family:'Orbitron',monospace;font-size:9px;font-weight:900;color:${wrCol}">${p.winRate.toFixed(0)}%</span>
          </td>
          <td style="color:#e3b341;font-weight:700">${p.avgRR.toFixed(1)}:1</td>
          <td>
            <span class="nl-trigger-count">${p.triggerCount}</span>
            ${p.triggerCount > 0 ? '<span style="font-size:8px;color:var(--accent)"> act.</span>' : ''}
          </td>
          <td>
            <button class="btn-nl-del-pattern" data-id="${p.id}" title="Eliminar patrón">🗑</button>
          </td>
        </tr>
      `;
    }).join('');

    // Wire delete buttons
    tbody.querySelectorAll('.btn-nl-del-pattern').forEach(btn => {
      btn.addEventListener('click', async () => {
        await StorageAdapter.removeNeuralPattern(btn.dataset.id);
        refreshPatternLibrary();
      });
    });
  } catch (e) {
    console.warn('[NeuralLab] Pattern library error:', e);
  }
}

// ── Architect Alert ────────────────────────────────────────────────

function showArchitectAlert(matches) {
  const alertEl = el('nlArchitectAlert');
  const msgEl   = el('nlArchitectMsg');
  if (!alertEl || !msgEl || !matches.length) return;

  const top = matches[0];
  msgEl.textContent = top.alertMessage;
  alertEl.style.display = 'flex';
  // Auto-hide after 12s
  setTimeout(() => { alertEl.style.display = 'none'; }, 12000);
  el('nlArchitectClose')?.addEventListener('click', () => { alertEl.style.display = 'none'; }, { once: true });

  // Parpadeo dorado en el badge de status
  const badge = el('nlStatusBadge');
  if (badge) {
    badge.classList.add('nl-badge-architect');
    setTimeout(() => badge.classList.remove('nl-badge-architect'), 6000);
  }
}


function renderSignals(signals) {
  const tbody = el('nlSignalsTbody');
  const table = el('nlSignalsTable');
  if (!signals?.length) {
    hide('nlSignalsTable'); show('nlEmptyState');
    const m = el('nlEmptyState')?.querySelector('.nl-empty-msg');
    if (m) m.textContent = 'Sin confluencias ≥ 70% — mercado sin sesgo claro';
    return;
  }
  hide('nlEmptyState'); show('nlSignalsTable');
  tbody.innerHTML = signals.map(s => `
    <tr class="nl-sig-row" data-id="${s.timestamp}" style="cursor:pointer">
      <td class="nl-price-val">${s.symbol}</td>
      <td>${dirBadge(s.direction)}</td>
      <td>${wrBadge(s.winRate)}</td>
      <td class="nl-price-val">$${fmtP(s.entryPrice)}</td>
      <td class="nl-tp-val">$${fmtP(s.tp)}</td>
      <td class="nl-sl-val">$${fmtP(s.sl)}</td>
      <td class="nl-rr-val">${s.riskReward}:1</td>
      <td><span class="nl-wr-badge ${s.confidence>=80?'nl-wr-high':'nl-wr-mid'}">${s.confidence}%</span></td>
      <td>${fmtTime(s.timestamp)}</td>
    </tr>
    <tr class="nl-detail-row" id="nl-detail-${s.timestamp}" style="display:none">
      <td colspan="9"><div class="nl-signal-detail">
        ${s.signals.map(sg => `<div>${sg}</div>`).join('')}
      </div></td>
    </tr>
  `).join('');
  tbody.querySelectorAll('.nl-sig-row').forEach(row => {
    row.addEventListener('click', () => {
      const d = el(`nl-detail-${row.dataset.id}`);
      if (d) d.style.display = d.style.display === 'none' ? '' : 'none';
    });
  });
}

// ── Render: Whale table with Forensic/Scientist buttons ────────────

let _lastWhales  = [];
let _lastSymbol  = 'BTCUSDT';

function renderWhales(whales, pressure, symbol) {
  _lastWhales = whales || [];
  _lastSymbol = symbol || 'BTCUSDT';
  const sec = el('nlWhaleSection');
  if (!whales?.length) { if (sec) sec.style.display = 'none'; return; }
  sec.style.display = '';

  const total = pressure.buyBtc + pressure.sellBtc || 1;
  const buyPct = Math.round(pressure.buyBtc / total * 100);
  const selPct = 100 - buyPct;
  const biasCol = pressure.bias === 'buy' ? 'nl-ws-buy' : pressure.bias === 'sell' ? 'nl-ws-sell' : 'nl-ws-neutral';

  el('nlWhaleStats').innerHTML = `
    <div class="nl-ws-card"><div class="nl-ws-lbl">Ballenas</div><div class="nl-ws-val" style="color:var(--accent)">${whales.length}</div></div>
    <div class="nl-ws-card"><div class="nl-ws-lbl">Compra</div><div class="nl-ws-val nl-ws-buy">${fmtBtc(pressure.buyBtc)} BTC</div></div>
    <div class="nl-ws-card"><div class="nl-ws-lbl">Venta</div><div class="nl-ws-val nl-ws-sell">${fmtBtc(pressure.sellBtc)} BTC</div></div>
    <div class="nl-ws-card"><div class="nl-ws-lbl">Sesgo</div><div class="nl-ws-val ${biasCol}">${pressure.bias.toUpperCase()}</div></div>
  `;

  // Bias bar (remove old if exists)
  el('nlBiasBar')?.remove();
  el('nlWhaleStats').insertAdjacentHTML('afterend', `
    <div class="nl-bias-bar-wrap" id="nlBiasBar">
      <span style="color:var(--green);font-size:8px;font-weight:700">${buyPct}%▲</span>
      <div class="nl-bias-bar">
        <div class="nl-bias-buy" style="width:${buyPct}%"></div>
        <div class="nl-bias-sell" style="width:${selPct}%"></div>
      </div>
      <span style="color:var(--red);font-size:8px;font-weight:700">${selPct}%▼</span>
    </div>
  `);

  setTxt('nlWhaleSub', `${whales.length} trades · top ${Math.min(whales.length,10)} mostrados`);

  const tbody = el('nlWhaleTbody');
  tbody.innerHTML = whales.slice(0,10).map((w, i) => {
    const icon      = w.transactionType === 'buy' ? '🐋' : '🦈';
    const sideClass = w.transactionType === 'buy'  ? 'nl-ws-buy'  : 'nl-ws-sell';
    const sideLabel = w.transactionType === 'buy'  ? '▲ COMPRA'   : '▼ VENTA';
    return `
      <tr>
        <td class="nl-whale-type">${icon}</td>
        <td style="font-weight:700;color:var(--text1)">${fmtBtc(w.amount)}</td>
        <td>${fmtUsd(w.amountUsd)}</td>
        <td class="${sideClass}" style="font-weight:900;font-size:8px">${sideLabel}</td>
        <td>${fmtTime(w.timestamp)}</td>
        <td>
          <button class="btn-nl-forensic" data-idx="${i}" title="Decodificar MO">🔬 Decode</button>
          <button class="btn-nl-scientist" data-idx="${i}" title="Backtest cluster">📊 BT</button>
        </td>
      </tr>
      <tr class="nl-forensic-row" id="nl-fr-${i}" style="display:none">
        <td colspan="6"><div class="nl-forensic-panel" id="nl-fp-${i}">
          <div class="nl-fp-loading"><span class="nl-spin">⟳</span> Analizando…</div>
        </div></td>
      </tr>
    `;
  }).join('');

  // Wire buttons
  tbody.querySelectorAll('.btn-nl-forensic').forEach(btn => {
    btn.addEventListener('click', () => runForensic(parseInt(btn.dataset.idx)));
  });
  tbody.querySelectorAll('.btn-nl-scientist').forEach(btn => {
    btn.addEventListener('click', () => runScientist(parseInt(btn.dataset.idx)));
  });
}

// ── Forensic Runner ────────────────────────────────────────────────

async function runForensic(idx) {
  const whale     = _lastWhales[idx];
  const rowId     = `nl-fr-${idx}`;
  const panelId   = `nl-fp-${idx}`;
  const row       = el(rowId);
  const panel     = el(panelId);
  if (!row || !panel || !whale) return;

  if (row.style.display !== 'none' && !panel.querySelector('.nl-fp-loading')) {
    row.style.display = 'none'; return;
  }

  row.style.display = '';
  panel.innerHTML = `<div class="nl-fp-loading"><span class="nl-spin">⟳</span> Cruzando timestamp con indicadores…</div>`;

  try {
    const report = await ForensicAgent.decodeWhaleStrategy(whale, _lastSymbol);
    panel.innerHTML = renderForensicPanel(report, idx);

    // ── Architect: comparar contra patrones guardados ──
    const matches = await ArchitectAgent.matchPatterns(report);
    if (matches.length) showArchitectAlert(matches);

    const btBtn = panel.querySelector('.btn-fp-scientist');
    if (btBtn) {
      btBtn.addEventListener('click', () => runScientistWithReport(idx, report));
    }
  } catch (e) {
    panel.innerHTML = `<div class="nl-fp-error">⚠ Error forensic: ${e.message}</div>`;
  }
}

function renderForensicPanel(r, idx) {
  const dirCol    = r.predictedDir === 'bull' ? '#00ff41' : r.predictedDir === 'bear' ? '#f85149' : '#e3b341';
  const confWidth = r.confidence || 0;
  const rsiColor  = r.rsiAtTrade < 30 ? '#00ff41' : r.rsiAtTrade > 70 ? '#f85149' : '#e3b341';
  const wp        = r.walletProfile || null;

  // ── Wallet Profile card ──
  const wpHtml = wp ? `
  <div class="nl-wallet-profile">
    <div class="nl-wp-id">
      <span class="nl-wp-id-label">🆔 Wallet ID</span>
      <span class="nl-wp-fingerprint">${r.whaleFingerprint || '—'}</span>
      <span class="nl-wp-badge ${wp.tradeCount > 3 ? 'nl-wp-known' : 'nl-wp-new'}">
        ${wp.tradeCount > 1 ? `📡 CONOCIDA · ${wp.tradeCount} avistamientos` : '🆕 PRIMER AVISTAMIENTO'}
      </span>
    </div>
    <div class="nl-wp-grid">
      <div class="nl-wp-cell">
        <div class="nl-wp-lbl">Vol. total detectado</div>
        <div class="nl-wp-val" style="color:var(--accent)">${wp.totalBtc.toFixed(2)} BTC</div>
      </div>
      <div class="nl-wp-cell">
        <div class="nl-wp-lbl">Trades observados</div>
        <div class="nl-wp-val">${wp.tradeCount}</div>
      </div>
      <div class="nl-wp-cell">
        <div class="nl-wp-lbl">MO dominante</div>
        <div class="nl-wp-val" style="font-size:8px">${(wp.dominantMO||'').replace(/_/g,' ')}</div>
      </div>
      <div class="nl-wp-cell">
        <div class="nl-wp-lbl">RSI promedio entrada</div>
        <div class="nl-wp-val" style="color:${wp.avgRsiEntry<40?'#00ff41':wp.avgRsiEntry>60?'#f85149':'#e3b341'}">
          ${wp.avgRsiEntry != null ? wp.avgRsiEntry.toFixed(1) : '—'}
        </div>
      </div>
      <div class="nl-wp-cell">
        <div class="nl-wp-lbl">Vol ratio promedio</div>
        <div class="nl-wp-val">${wp.avgVolRatio.toFixed(2)}x</div>
      </div>
      <div class="nl-wp-cell">
        <div class="nl-wp-lbl">Primer avistamiento</div>
        <div class="nl-wp-val" style="font-size:7.5px">${new Date(wp.firstSeen).toLocaleDateString('es',{month:'2-digit',day:'2-digit'})}</div>
      </div>
    </div>
    ${wp.priceHistory && wp.priceHistory.length > 1 ? `
    <div class="nl-wp-price-history">
      <div class="nl-wp-lbl" style="margin-bottom:4px">Zonas de precio de entrada históricas:</div>
      <div class="nl-wp-price-chips">
        ${wp.priceHistory.slice(-8).map(p =>
          `<span class="nl-wp-price-chip">$${p.price >= 10000 ? p.price.toLocaleString('en',{maximumFractionDigits:0}) : p.price.toFixed(2)}</span>`
        ).join('')}
      </div>
    </div>` : ''}
  </div>` : `
  <div class="nl-wallet-profile nl-wp-new-wrap">
    <div class="nl-wp-id">
      <span class="nl-wp-id-label">🆔 Wallet ID</span>
      <span class="nl-wp-fingerprint">${r.whaleFingerprint || '—'}</span>
      <span class="nl-wp-badge nl-wp-new">🆕 PRIMER AVISTAMIENTO</span>
    </div>
  </div>`;

  return `
  ${wpHtml}
  <div class="nl-fp-header">
    <span class="nl-fp-mo-badge nl-mo-${r.mo}">${r.moIcon} ${r.moLabel}</span>
    <span class="nl-fp-conf">Confianza ${r.confidence}%</span>
    <span class="nl-fp-pred" style="color:${dirCol}">
      ${r.predictedDir === 'bull' ? '▲' : r.predictedDir === 'bear' ? '▼' : '→'} ${r.predictedStrength}% a 15m
    </span>
  </div>
  <div class="nl-fp-desc">${r.moDescription}</div>
  <div class="nl-fp-conf-bar"><div class="nl-fp-conf-fill" style="width:${confWidth}%;background:${dirCol}"></div></div>
  <div class="nl-fp-grid">
    <div class="nl-fp-cell">
      <div class="nl-fp-lbl">RSI al trade</div>
      <div class="nl-fp-val" style="color:${rsiColor}">${r.rsiAtTrade != null ? r.rsiAtTrade.toFixed(1) : '—'}</div>
    </div>
    <div class="nl-fp-cell">
      <div class="nl-fp-lbl">Pos. EMA</div>
      <div class="nl-fp-val">${(r.emaPos || '').replace(/_/g,' ')}</div>
    </div>
    <div class="nl-fp-cell">
      <div class="nl-fp-lbl">Vol ratio</div>
      <div class="nl-fp-val" style="color:${r.volRatio>2?'#e3b341':''}">
        ${r.volRatio?.toFixed(2) ?? '—'}x
      </div>
    </div>
    <div class="nl-fp-cell">
      <div class="nl-fp-lbl">Precio vs nivel</div>
      <div class="nl-fp-val">${(r.priceVsLevel || '').replace(/_/g,' ')}</div>
    </div>
    <div class="nl-fp-cell">
      <div class="nl-fp-lbl">Vol delta</div>
      <div class="nl-fp-val" style="color:${r.volumeDelta>=0?'#00ff41':'#f85149'}">
        ${r.volumeDelta >= 0 ? '▲ Compra' : '▼ Venta'}
      </div>
    </div>
    <div class="nl-fp-cell">
      <div class="nl-fp-lbl">Divergencia RSI</div>
      <div class="nl-fp-val" style="color:${r.hasDivergence?'#e3b341':'#555'}">
        ${r.hasDivergence ? `⚠ ${r.divergenceType}` : 'No'}
      </div>
    </div>
  </div>
  <div class="nl-fp-signals">
    ${(r.signals || []).map(s => `<div class="nl-fp-sig">${s}</div>`).join('')}
  </div>
  <button class="btn-fp-scientist" data-idx="${idx}">📊 Backtest de este cluster →</button>
  `;
}


// ── Scientist Runner ───────────────────────────────────────────────

async function runScientist(idx) {
  const whale = _lastWhales[idx];
  if (!whale) return;
  // Run forensic first to get the report context
  let report;
  try {
    showScientistLoading();
    report = await ForensicAgent.decodeWhaleStrategy(whale, _lastSymbol);
  } catch {
    report = { rsiAtTrade: null, emaPos: 'above_ema20', volRatio: 1, hasDivergence: false, priceVsLevel: 'mid-range' };
  }
  await runScientistWithReport(idx, report);
}

async function runScientistWithReport(idx, forensicReport) {
  const whale = _lastWhales[idx];
  if (!whale) return;
  showScientistLoading();

  try {
    const result = await ScientistAgent.backTestWalletCluster(whale, forensicReport, _lastSymbol);
    showScientistResult(result, whale, forensicReport);
  } catch (e) {
    closeScientistPopup();
    const errEl = el('nlError');
    if (errEl) { errEl.textContent = `Scientist error: ${e.message}`; errEl.style.display = ''; }
  }
}

// ── Scientist Popup ────────────────────────────────────────────────

function showScientistLoading() {
  let popup = el('nlScientistPopup');
  if (!popup) {
    popup = document.createElement('div');
    popup.id = 'nlScientistPopup';
    popup.className = 'nl-scientist-popup';
    document.body.appendChild(popup);
  }
  popup.style.display = 'flex';
  popup.innerHTML = `
    <div class="nl-sci-modal">
      <div class="nl-sci-header">
        <span class="nl-sci-title">📊 Scientist Agent — Backtesting</span>
        <button class="nl-sci-close" id="nlSciClose">✕</button>
      </div>
      <div class="nl-sci-loading">
        <div class="nl-spin" style="font-size:24px">⟳</div>
        <span>Buscando clusters similares en 500 velas…</span>
      </div>
    </div>
  `;
  el('nlSciClose')?.addEventListener('click', closeScientistPopup);
  popup.addEventListener('click', e => { if (e.target === popup) closeScientistPopup(); });
}

function showScientistResult(result, whale, report) {
  const popup = el('nlScientistPopup');
  if (!popup) return;

  const wrCol  = result.winRate >= 70 ? '#00ff41' : result.winRate >= 55 ? '#e3b341' : '#f85149';
  const savedBadge = result.savedPattern
    ? `<span class="nl-sci-saved-badge">💾 NeuralPattern guardado: "${result.savedPattern.name}"</span>`
    : '';

  popup.innerHTML = `
    <div class="nl-sci-modal">
      <div class="nl-sci-header">
        <span class="nl-sci-title">📊 Scientist — ${whale.amount.toFixed(2)} BTC ${whale.transactionType.toUpperCase()}</span>
        <button class="nl-sci-close" id="nlSciClose">✕</button>
      </div>

      ${savedBadge ? `<div class="nl-sci-saved">${savedBadge}</div>` : ''}

      <div class="nl-sci-stats">
        <div class="nl-sci-stat">
          <div class="nl-sci-stat-lbl">Casos encontrados</div>
          <div class="nl-sci-stat-val" style="color:var(--accent)">${result.matchCount}</div>
        </div>
        <div class="nl-sci-stat">
          <div class="nl-sci-stat-lbl">Win-Rate</div>
          <div class="nl-sci-stat-val" style="color:${wrCol}">${result.winRate}%</div>
        </div>
        <div class="nl-sci-stat">
          <div class="nl-sci-stat-lbl">Wins / Losses</div>
          <div class="nl-sci-stat-val"><span style="color:#00ff41">${result.winCount}▲</span> / <span style="color:#f85149">${result.lossCount}▼</span></div>
        </div>
        <div class="nl-sci-stat">
          <div class="nl-sci-stat-lbl">Drawdown avg</div>
          <div class="nl-sci-stat-val" style="color:#e3b341">${result.avgDrawdown}%</div>
        </div>
        <div class="nl-sci-stat">
          <div class="nl-sci-stat-lbl">Tiempo al TP</div>
          <div class="nl-sci-stat-val">${result.avgTimeToTpMin} min</div>
        </div>
        <div class="nl-sci-stat">
          <div class="nl-sci-stat-lbl">R:R estimado</div>
          <div class="nl-sci-stat-val" style="color:${result.avgRR>=2?'#00ff41':result.avgRR>=1?'#e3b341':'#f85149'}">${result.avgRR}:1</div>
        </div>
      </div>

      <div class="nl-sci-chart-wrap">
        <div class="nl-sci-chart-title">Snapshot — ${result.matchCount} casos históricos (▲WIN ▼LOSS ●NEUTRAL)</div>
        <div class="nl-sci-chart" id="nlSciChart">
          ${result.svgChart || `<div class="nl-sci-no-chart">Sin suficientes datos para el chart</div>`}
        </div>
      </div>

      <div class="nl-sci-context">
        <div class="nl-sci-ctx-title">Contexto del trade original</div>
        <div class="nl-sci-ctx-grid">
          <span>RSI: ${report.rsiAtTrade?.toFixed(1) ?? '—'}</span>
          <span>EMA: ${(report.emaPos||'').replace(/_/g,' ')}</span>
          <span>Vol ratio: ${report.volRatio?.toFixed(2) ?? '—'}x</span>
          <span>MO: ${report.moIcon || '🔍'} ${report.moLabel || '—'}</span>
        </div>
      </div>

      ${result.matchCount === 0 ? `
        <div class="nl-sci-empty">
          Sin clusters similares encontrados en las últimas 500 velas.
          Prueba ajustando el rango BTC o el símbolo.
        </div>` : ''}
    </div>
  `;
  el('nlSciClose')?.addEventListener('click', closeScientistPopup);
  popup.addEventListener('click', e => { if (e.target === popup) closeScientistPopup(); });
}

function closeScientistPopup() {
  const popup = el('nlScientistPopup');
  if (popup) popup.style.display = 'none';
}

// ── Render: History ────────────────────────────────────────────────

function renderHistory(history) {
  const histEmpty = el('nlHistEmpty');
  const histTable = el('nlHistTable');
  const tbody     = el('nlHistTbody');
  if (!history?.length) {
    if (histEmpty) histEmpty.style.display = '';
    if (histTable) histTable.style.display = 'none';
    return;
  }
  if (histEmpty) histEmpty.style.display = 'none';
  if (histTable) histTable.style.display = '';
  tbody.innerHTML = history.slice(0, 50).map(s => `
    <tr>
      <td style="color:var(--text1);font-weight:700">${s.symbol}</td>
      <td>${dirBadge(s.direction)}</td>
      <td>${wrBadge(s.winRate)}</td>
      <td class="nl-price-val">$${fmtP(s.entryPrice)}</td>
      <td class="nl-rr-val">${s.riskReward}:1</td>
      <td class="nl-time-cell">${fmtDT(s.timestamp)}</td>
    </tr>
  `).join('');
}

async function refreshHistory() {
  try { renderHistory(await StorageAdapter.loadHunterHistory()); } catch {}
}

// ── Main scan ──────────────────────────────────────────────────────

let _scanning = false;

async function doScan() {
  if (_scanning) return;
  _scanning = true;
  const symbol = (el('nlSymbolInput')?.value || 'BTCUSDT').trim().toUpperCase();
  const badge  = el('nlStatusBadge');
  const scanBtn = el('nlScanBtn');

  if (badge) { badge.textContent = '⟳ SCANNING'; badge.className = 'nl-badge nl-badge-live scanning'; }
  if (scanBtn) scanBtn.disabled = true;
  hide('nlError'); show('nlLoading'); hide('nlSignalsTable');
  setTxt('nlScanSub', `Escaneando ${symbol}…`);
  el('nlBiasBar')?.remove();

  try {
    const { signals, whales, scanTime, scoring } = await HunterAgent.scan(symbol);
    hide('nlLoading');

    if (badge) {
      badge.className = 'nl-badge nl-badge-live';
      badge.style.color = signals.length ? 'var(--green)' : 'var(--yellow)';
      badge.style.borderColor = signals.length ? 'rgba(0,255,65,.4)' : 'rgba(227,179,65,.35)';
      badge.textContent = signals.length
        ? `● ${signals.length} SEÑAL${signals.length>1?'ES':''}`
        : '● SIN CONFLUENCIA';
    }
    const elapsed = ((Date.now() - scanTime) / 1000).toFixed(1);
    setTxt('nlScanSub', `${new Date(scanTime).toLocaleTimeString('es')} · ${elapsed}s · ${whales.length} ballenas`);

    renderSignals(signals);
    const pressure = HunterAgent.analyzeWhalePressure(whales);
    renderWhales(whales, pressure, symbol);

    if (!signals.length && scoring) {
      setTxt('nlScanSub', `Bull ${scoring.bullScore} / Bear ${scoring.bearScore} — WR ${(scoring.winRate*100).toFixed(0)}% (min 70%)`);
    }
  } catch (err) {
    hide('nlLoading');
    const errEl = el('nlError');
    if (errEl) { errEl.textContent = `Error: ${err.message}`; errEl.style.display = ''; }
    if (badge) { badge.textContent = '● ERROR'; badge.style.color = 'var(--red)'; }
    console.error('[NeuralLab]', err);
  } finally {
    _scanning = false;
    if (scanBtn) scanBtn.disabled = false;
    refreshHistory();
  }
}

// ── Init ───────────────────────────────────────────────────────────

export function initNeuralLab() {
  el('nlScanBtn')?.addEventListener('click', doScan);
  el('nlSymbolInput')?.addEventListener('keydown', e => { if (e.key === 'Enter') doScan(); });

  el('nlClearHistBtn')?.addEventListener('click', () => {
    if (confirm('¿Borrar historial de señales?')) {
      StorageAdapter.clearHunterHistory().then(() => renderHistory([]));
    }
  });

  // Podar patrones débiles
  el('nlPruneBtn')?.addEventListener('click', async () => {
    const removed = await ArchitectAgent.pruneWeakPatterns();
    refreshPatternLibrary();
    if (removed > 0) {
      const badge = el('nlStatusBadge');
      if (badge) { badge.textContent = `✂ ${removed} patrón(es) podados`; }
      setTimeout(() => { if (badge) badge.textContent = '● IDLE'; }, 3000);
    }
  });

  // Cerrar alerta Architect
  el('nlArchitectClose')?.addEventListener('click', () => {
    const a = el('nlArchitectAlert');
    if (a) a.style.display = 'none';
  });

  refreshHistory();
  refreshPatternLibrary();
  console.log('[STARK-OS] ✓ Neural Lab v2 + Architect inicializado');
}

export default { initNeuralLab };
