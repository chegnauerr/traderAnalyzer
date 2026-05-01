'use strict';

// ── Extract symbol / exchange / tf / price ────────────────────────────────────

function extractSymbol() {
  const m = window.location.href.match(/[?&]symbol=([^&#]+)/i);
  if (m) { const raw = decodeURIComponent(m[1]); return raw.includes(':') ? raw.split(':')[1] : raw; }
  const tm = document.title.match(/([A-Z]{1,10})[\/\-]?(USDT|USDC|BTC|ETH|BNB)?/i);
  return tm ? (tm[1] + (tm[2] || '')).toUpperCase() : 'BTCUSDT';
}
function extractExchange() {
  const m = window.location.href.match(/symbol=([A-Z]+):/i);
  return m ? m[1].toUpperCase() : 'BINANCE';
}
function extractTimeframe() {
  for (const sel of ['[data-active-chart-time-unit="true"]','[class*="isActive"][class*="timeframe"]','[class*="timeframes"] [class*="isActive"]']) {
    const el = document.querySelector(sel);
    if (el?.textContent?.trim()) return el.textContent.trim();
  }
  const m = window.location.href.match(/interval=(\d+[mhDWM]?)/i);
  return m ? m[1] : '15';
}
function extractPrice() {
  for (const sel of ['[data-field="last_price"]','[class*="lastPrice"]','[class*="priceValue"]','.js-symbol-last']) {
    const el = document.querySelector(sel);
    if (!el) continue;
    const n = parseFloat(el.textContent.replace(/[^0-9.]/g, ''));
    if (n > 0) return n;
  }
  return null;
}
function getData() {
  return { symbol: extractSymbol(), exchange: extractExchange(),
           timeframe: extractTimeframe(), price: extractPrice(),
           url: window.location.href, ts: Date.now() };
}

// ── Messages ──────────────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg.type === 'GET_TV_DATA') { reply(getData()); return true; }
  if (msg.type === 'DRAW_LEVELS') { handleDraw(msg).then(r => reply(r)); return true; }
  return true;
});

// ── Symbol change watch ───────────────────────────────────────────────────────

let lastUrl = location.href, lastSym = extractSymbol(), changeTmr = null;
function notifyChange() {
  clearTimeout(changeTmr);
  changeTmr = setTimeout(() => {
    const d = getData();
    if (d.symbol !== lastSym) { lastSym = d.symbol; chrome.runtime.sendMessage({ type: 'TV_SYMBOL_CHANGE', ...d }); }
  }, 600);
}
new MutationObserver(() => { if (location.href !== lastUrl) { lastUrl = location.href; notifyChange(); } })
  .observe(document.body, { subtree: true, childList: true });
new MutationObserver(notifyChange)
  .observe(document.querySelector('title') || document.head, { subtree: true, characterData: true, childList: true });
setInterval(() => { if (extractSymbol() !== lastSym) notifyChange(); }, 2000);

// ═══════════════════════════════════════════════════════════════════════════════
// DRAW LEVELS
// ═══════════════════════════════════════════════════════════════════════════════

async function handleDraw(msg) {
  const levels = msg.levels || [];

  // ── Tier 1: TradingView internal chart API (works if tvWidget is global) ──
  const apiOk = await tryTvApi(levels);
  if (apiOk) {
    showToast('✓ Líneas dibujadas nativamente en TradingView', '#00ff41');
    return { ok: true, method: 'api' };
  }

  // ── Tier 2: Pine Script — genera código y lo pega en el editor abierto ────
  const pine = buildPineScript(levels, msg.symbol || extractSymbol());
  const pasted = await tryPastePine(pine);
  if (pasted) {
    showToast('✓ Pine Script aplicado al gráfico', '#00ff41');
    return { ok: true, method: 'pine-paste' };
  }

  // ── Tier 3: Copia Pine Script al portapapeles y abre el editor ────────────
  await navigator.clipboard.writeText(pine).catch(() => {});
  showPinePanel(pine);
  return { ok: true, method: 'pine-clipboard' };
}

// ── Tier 1: TV internal API ───────────────────────────────────────────────────

function tryTvApi(levels) {
  return new Promise(resolve => {
    const key = '__tva_ok_' + Date.now();
    const s = document.createElement('script');
    s.textContent = `(function(){
      var ok=false, lvls=${JSON.stringify(levels)};
      var fns=[
        function(){ return window.tvWidget; },
        function(){ return window._tvWidget; },
        function(){ return Object.values(window).find(function(v){ return v&&typeof v.activeChart==='function'; }); }
      ];
      for(var i=0;i<fns.length&&!ok;i++){
        try{
          var w=fns[i](); if(!w||typeof w.activeChart!=='function') continue;
          var c=w.activeChart(), t=c.getVisibleRange().to;
          lvls.forEach(function(l){
            c.createShape({time:t,price:l.price},{shape:'horizontal_line',text:l.label,
              overrides:{linecolor:l.color,linewidth:2,linestyle:2,showLabel:true,textcolor:l.color,fontsize:11}});
          });
          ok=true;
        }catch(e){}
      }
      window['${key}']=ok;
    })();`;
    document.head.appendChild(s); s.remove();
    setTimeout(() => { const r = !!window[key]; delete window[key]; resolve(r); }, 300);
  });
}

// ── Tier 2: Try to paste Pine Script directly into an open Pine editor ────────

async function tryPastePine(pine) {
  // Look for an open Pine Script editor textarea/codemirror
  const editors = [
    document.querySelector('.pine-editor textarea'),
    document.querySelector('[class*="pine-editor"] textarea'),
    document.querySelector('[class*="PineEditor"] textarea'),
    document.querySelector('textarea[spellcheck="false"]'),
  ];
  const editor = editors.find(Boolean);
  if (!editor) return false;

  // Focus and replace content
  editor.focus();
  const nativeInput = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
  if (nativeInput) {
    nativeInput.call(editor, pine);
    editor.dispatchEvent(new Event('input', { bubbles: true }));
  }
  return true;
}

// ── Pine Script builder ───────────────────────────────────────────────────────

function buildPineScript(levels, symbol) {
  const colorMap = {
    '#00ff41': 'color.green',
    '#f85149': 'color.red',
    '#00d4ff': 'color.aqua',
    '#f5a623': 'color.orange',
    '#ffffff': 'color.white',
    '#e3b341': 'color.yellow',
  };
  const lines = levels.map(l => {
    const col   = colorMap[l.color] || 'color.blue';
    const style = (l.label === 'Precio') ? 'hline.style_solid' : 'hline.style_dashed';
    const width = (l.label === 'Precio') ? 1 : 2;
    return `hline(${l.price}, "${l.label}", color=${col}, linewidth=${width}, linestyle=${style})`;
  }).join('\n');

  return `//@version=5
indicator("TVA Niveles — ${symbol}", overlay=true, max_lines_count=10)
${lines}
`;
}

// ── Pine Script panel (clipboard fallback) ────────────────────────────────────

function showPinePanel(pine) {
  document.getElementById('tva-pine-panel')?.remove();

  const panel = document.createElement('div');
  panel.id = 'tva-pine-panel';
  panel.style.cssText = `
    position:fixed; bottom:20px; right:20px; width:340px;
    background:#0d1117; border:1px solid #00d4ff;
    border-radius:8px; padding:14px 16px; z-index:999999;
    font-family:monospace; box-shadow:0 4px 24px rgba(0,212,255,.2);
  `;

  panel.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
      <span style="color:#00d4ff;font-size:11px;font-weight:900">◈ TV ANALYZER — PINE SCRIPT</span>
      <button id="tva-pine-close" style="background:none;border:none;color:#666;font-size:14px;cursor:pointer;padding:0">✕</button>
    </div>
    <div style="color:#8b949e;font-size:9px;margin-bottom:8px;line-height:1.5">
      Pine Script copiado al portapapeles.<br>
      <b style="color:#e3b341">▶ Pega en el editor Pine de TradingView:</b>
    </div>
    <ol style="color:#8b949e;font-size:8.5px;margin:0 0 10px 14px;line-height:1.8">
      <li>Haz clic en <b style="color:#fff">Pine Editor</b> (barra inferior)</li>
      <li>Selecciona todo (<b style="color:#fff">Ctrl+A</b>) y pega (<b style="color:#fff">Ctrl+V</b>)</li>
      <li>Haz clic en <b style="color:#00ff41">Añadir al gráfico</b></li>
    </ol>
    <pre style="background:#161b22;border:1px solid #30363d;border-radius:4px;padding:8px;font-size:8px;color:#e6edf3;overflow-x:auto;max-height:120px;margin:0 0 10px">${pine.replace(/</g,'&lt;')}</pre>
    <div style="display:flex;gap:8px">
      <button id="tva-pine-copy" style="flex:1;background:rgba(0,212,255,.12);border:1px solid rgba(0,212,255,.4);color:#00d4ff;border-radius:4px;padding:6px;font:700 9px monospace;cursor:pointer">
        ⧉ Copiar código
      </button>
      <button id="tva-pine-open" style="flex:1;background:rgba(0,255,65,.08);border:1px solid rgba(0,255,65,.3);color:#00ff41;border-radius:4px;padding:6px;font:700 9px monospace;cursor:pointer">
        ▶ Abrir Pine Editor
      </button>
    </div>
  `;

  document.body.appendChild(panel);

  document.getElementById('tva-pine-close').onclick = () => panel.remove();
  document.getElementById('tva-pine-copy').onclick  = async () => {
    await navigator.clipboard.writeText(pine);
    document.getElementById('tva-pine-copy').textContent = '✓ Copiado';
  };
  document.getElementById('tva-pine-open').onclick = () => {
    // Try to click the Pine Editor tab in TradingView
    const pineBtn = [...document.querySelectorAll('button,[class*="tab"]')]
      .find(el => el.textContent.includes('Pine') || el.textContent.includes('pine'));
    if (pineBtn) pineBtn.click();
    // Also scroll panel into view
    panel.scrollIntoView();
  };
}

// ── Toast ─────────────────────────────────────────────────────────────────────

function showToast(msg, color) {
  document.getElementById('tva-toast')?.remove();
  const d = document.createElement('div');
  d.id = 'tva-toast';
  d.textContent = msg;
  d.style.cssText = `position:fixed;bottom:20px;left:50%;transform:translateX(-50%);`
    + `background:rgba(13,17,23,.96);border:1px solid ${color};color:${color};`
    + `padding:8px 18px;border-radius:6px;font:700 11px monospace;z-index:999999;`
    + `pointer-events:none;`;
  document.body.appendChild(d);
  setTimeout(() => { d.style.transition = 'opacity .5s'; d.style.opacity = '0'; setTimeout(() => d.remove(), 500); }, 3500);
}
