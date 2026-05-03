'use strict';

// ═══════════════════════════════════════════════════════════════════
// STARK-OS IMPORTS — Hexagonal Architecture Modules
// Los módulos de la nueva arquitectura se importan aquí.
// popup.js actúa como orquestador fino (Thin Shell).
// ═══════════════════════════════════════════════════════════════════
import { initNeuralLab } from './neural-lab.js';
import { initWhaleTracker } from './whale-tracker.js';
import { WhaleTrackerAgent } from './application/whale-tracker.agent.js';

// ═══════════════════════════════════════════════════════════════════
// ML MODEL — Weighted Scoring Engine (Logistic Regression)
// Entrenado con Python en tv-analyzer/train/
// El JSON exportado se carga automáticamente si existe en la extensión
// o manualmente desde el botón de importar en el Brain.
// ═══════════════════════════════════════════════════════════════════

let _mlModel = null; // { coefficients, intercept, scaler_mean, scaler_std, feature_names, cv_auc, importance }

async function mlLoadModel() {
  // 1. Intentar cargar model_weights.json desde la carpeta de la extensión
  try {
    const url  = chrome.runtime.getURL('model_weights.json');
    const resp = await fetch(url);
    if (resp.ok) {
      _mlModel = await resp.json();
      console.log('[ML] Modelo cargado desde extensión — AUC:', _mlModel.cv_auc);
      return true;
    }
  } catch {}
  // 2. Intentar desde chrome.storage (importado manualmente)
  try {
    const data = await new Promise(res => chrome.storage.local.get('mlModel', res));
    if (data.mlModel?.coefficients) {
      _mlModel = data.mlModel;
      console.log('[ML] Modelo cargado desde storage — AUC:', _mlModel.cv_auc);
      return true;
    }
  } catch {}
  return false;
}

function mlSigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, x)))); }

function mlExtractFeatures(candles, indicators) {
  const n = candles.length;
  if (n < 200) return null;
  const closes  = candles.map(c => c.close);
  const highs   = candles.map(c => c.high);
  const lows    = candles.map(c => c.low);
  const opens   = candles.map(c => c.open);
  const volumes = candles.map(c => c.volume || 0);

  // Recalcular series completas (misma lógica que build_features.py)
  const rsiArr  = calcRSI(closes, 14);
  const { hist: histArr } = calcMACD(closes);
  const e20Arr  = calcEMA(closes, 20);
  const e50Arr  = calcEMA(closes, 50);
  const e200Arr = calcEMA(closes, 200);
  const bbArr   = calcBB(closes, 20, 2);
  const atrArr  = calcATR(candles, 14);
  const stochArr = calcStoch(candles, 14);

  const rsi   = rsiArr[n-1]  ?? 50;
  const rsiP3 = rsiArr[n-4]  ?? rsi;
  const hist  = histArr[n-1] ?? 0;
  const histP = histArr[n-2] ?? 0;
  const e20   = e20Arr[n-1];
  const e50   = e50Arr[n-1];
  const e200  = e200Arr[n-1];
  const e20P5 = e20Arr[n-6] ?? e20;
  const e50P5 = e50Arr[n-6] ?? e50;
  const bb    = bbArr[n-1];
  const atr   = atrArr[n-1] ?? (closes[n-1] * 0.005);
  const stk   = stochArr.k[n-1] ?? 50;
  const price = closes[n-1];
  const open  = opens[n-1];
  const high  = highs[n-1];
  const low   = lows[n-1];

  const avgVol20 = volumes.slice(-20).reduce((a,b)=>a+b,0) / 20 || 1;
  const volR     = volumes[n-1] / avgVol20;
  const body     = Math.abs(price - open);
  const rng      = (high - low) || 1e-10;
  const upperW   = high - Math.max(price, open);
  const lowerW   = Math.min(price, open) - low;

  const hour = candles[n-1].time
    ? new Date(candles[n-1].time).getUTCHours()
    : 12;

  const bbP = bb?.lower != null ? (price - bb.lower) / (bb.upper - bb.lower + 1e-10) : 0.5;
  const bbW = bb?.width ?? 0.05;

  // DEBE coincidir con FEATURE_NAMES en train_model.py (mismo orden)
  return [
    (rsi - 50) / 50,                                   // rsi_norm
    (rsi - rsiP3) / 3 / 50,                            // rsi_slope
    rsi > 70 ? 1 : 0,                                  // rsi_ob
    rsi < 30 ? 1 : 0,                                  // rsi_os
    rsi > 50 ? 1 : 0,                                  // rsi_bull
    hist / (atr || 1),                                 // macd_hist_norm
    hist > 0 ? 1 : 0,                                  // macd_bull
    (hist > 0 && histP <= 0) ? 1 : 0,                  // macd_cross_up
    (hist < 0 && histP >= 0) ? 1 : 0,                  // macd_cross_dn
    e20 ? price / e20 - 1 : 0,                         // price_ema20
    e50 ? price / e50 - 1 : 0,                         // price_ema50
    e200 ? price / e200 - 1 : 0,                       // price_ema200
    e20P5 ? e20 / e20P5 - 1 : 0,                       // ema20_slope
    e50P5 ? e50 / e50P5 - 1 : 0,                       // ema50_slope
    (e20 > e50 && e50 > e200) ? 1 : 0,                 // ema_bull
    (e20 < e50 && e50 < e200) ? 1 : 0,                 // ema_bear
    price > e200 ? 1 : 0,                              // above_ema200
    Math.min(1, Math.max(0, bbP)),                     // bb_pos
    bbW,                                               // bb_width
    bbW < 0.025 ? 1 : 0,                               // bb_squeeze
    atr / price,                                       // atr_norm
    stk / 100,                                         // stoch_norm
    stk < 20 ? 1 : 0,                                  // stoch_os
    stk > 80 ? 1 : 0,                                  // stoch_ob
    Math.min(3, Math.log1p(volR)),                     // vol_ratio
    volR > 2 ? 1 : 0,                                  // vol_spike
    price > open ? 1 : 0,                              // bull_candle
    Math.min(1, body / rng),                           // body_ratio
    Math.min(1, upperW / rng),                         // upper_wick
    Math.min(1, lowerW / rng),                         // lower_wick
    Math.sin(2 * Math.PI * hour / 24),                 // hour_sin
    Math.cos(2 * Math.PI * hour / 24),                 // hour_cos
  ];
}

function mlInfer(candles) {
  if (!_mlModel) return null;
  const feat = mlExtractFeatures(candles);
  if (!feat) return null;
  // Normalizar con los parámetros del scaler entrenado en Python
  const norm = feat.map((v, i) =>
    (_mlModel.scaler_std[i] > 0)
      ? (v - _mlModel.scaler_mean[i]) / _mlModel.scaler_std[i]
      : 0
  );
  // Producto punto + bias → sigmoide
  const z = norm.reduce((s, v, i) => s + v * _mlModel.coefficients[i], 0) + _mlModel.intercept;
  const prob = mlSigmoid(z);
  return {
    bullPct:    Math.round(prob * 100),
    bearPct:    Math.round((1 - prob) * 100),
    raw:        prob,
    auc:        _mlModel.cv_auc,
    samples:    _mlModel.n_samples,
    importance: _mlModel.importance,
    trainedAt:  _mlModel.trained_at,
  };
}

// ═══════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════

let currentSymbol = 'BTCUSDT';
let currentTf     = '15m';
let currentPrice  = 0;
let candles       = [];       // OHLCV array
let lastAnalysis  = null;
let autoRefresh   = null;

// Paper trading
let ptBalance   = 10000;   // USD
let ptStart     = 10000;
let ptPositions = [];
let ptPeakEq    = 10000;
let ptSolPrice  = 1;       // dummy, we use USD directly

// Watchlist
const WATCHLIST_DEFAULT = ['BTCUSDT','ETHUSDT','XRPUSDT','SOLUSDT','BNBUSDT','DOGEUSDT'];
let watchlist = [...WATCHLIST_DEFAULT];
let watchData = {};

// WebSocket live feed
let liveWS  = null;
let wsTimer = null;

// TradingView tab tracking
let tvTabId = null;

// Bot Trades — auto-registro de señales del Brain
let botTrades = [];
let btMargin = 100;     // Margen dinámico (editable por usuario)
let btLeverage = 10;    // Leverage dinámico (editable por usuario)

// Precios en vivo de múltiples símbolos (para Bot Trades)
let livePrices = {}; // { 'BTCUSDT': 77900, 'ETHUSDT': 2287, ... }

// Throttle para renderizado de Bot Trades
let lastBotTradesRender = 0;

// ═══════════════════════════════════════════════════════════════════
// BOT TRADES — Auto-registro de señales y tracking
// ═══════════════════════════════════════════════════════════════════

async function loadBotTrades() {
  const data = await new Promise(res => chrome.storage.local.get(['botTrades', 'btMargin', 'btLeverage'], res));
  botTrades = data.botTrades || [];
  btMargin = data.btMargin || 100;
  btLeverage = data.btLeverage || 10;
  
  // Reconectar WebSocket para incluir símbolos con trades activos
  if (botTrades.some(t => t.status === 'pending' || t.status === 'running')) {
    wsApply();
  }
}

function saveBotTrades() {
  // Guardar solo últimos 200 trades
  chrome.storage.local.set({ botTrades: botTrades.slice(-200) });
}

function saveBotSettings() {
  chrome.storage.local.set({ btMargin, btLeverage });
}

function deleteBotTrade(tradeId) {
  const index = botTrades.findIndex(t => t.id === tradeId);
  if (index === -1) return;
  
  botTrades.splice(index, 1);
  saveBotTrades();
  renderBotTradesTab();
  showNotif('Trade eliminado', 'success');
}

function createBotTrade({ symbol, direction, entry, sl, tp1, tp2, rr, confidence, reason }) {
  // Cálculos de inversión con valores dinámicos (editables por usuario)
  const margin = btMargin;
  const leverage = btLeverage;
  const notional = margin * leverage;
  const openFee = notional * 0.0004; // Taker fee 0.04%
  
  // Calcular distancias en %
  const isLong = direction === 'LONG';
  const tpPct1 = isLong ? ((tp1 - entry) / entry * 100) : ((entry - tp1) / entry * 100);
  const tpPct2 = isLong ? ((tp2 - entry) / entry * 100) : ((entry - tp2) / entry * 100);
  const slPct = isLong ? ((entry - sl) / entry * 100) : ((sl - entry) / entry * 100);
  
  // Calcular ganancias/pérdidas en USD (sobre margen con leverage)
  const tpUsd1 = margin * (tpPct1 / 100) * leverage - openFee - (notional * 0.0004 * (1 + tpPct1 / 100));
  const tpUsd2 = margin * (tpPct2 / 100) * leverage - openFee - (notional * 0.0004 * (1 + tpPct2 / 100));
  const slUsd = -(margin * (slPct / 100) * leverage + openFee);
  
  const trade = {
    id: Date.now(),
    symbol,
    direction,    // 'LONG' | 'SHORT'
    entry,
    sl,
    tp1,
    tp2,
    rr,
    confidence,
    reason,
    // Inversión
    margin,
    leverage,
    notional,
    openFee,
    tpPct1,
    tpPct2,
    slPct,
    tpUsd1,
    tpUsd2,
    slUsd,
    timestamp: Date.now(),
    status: 'pending', // 'pending' | 'running' | 'win_tp1' | 'win_tp2' | 'loss' | 'cancelled'
    hitPrice: null,
    hitTime: null,
    pnlPct: null,
  };
  
  botTrades.push(trade);
  saveBotTrades();
  console.log('[BotTrade] Señal registrada:', trade);
  return trade;
}

function updateBotTrade(tradeId, updates) {
  const trade = botTrades.find(t => t.id === tradeId);
  if (!trade) return;
  
  Object.assign(trade, updates);
  saveBotTrades();
  console.log('[BotTrade] Actualizado:', trade);
}

function trackBotTrades(symbol, currentPrice) {
  // Get execution mode (default to limit for auto-execution)
  const executionMode = document.getElementById('btExecutionMode')?.value || 'limit';
  
  // ── AUTO-CONVERSIÓN POR TIEMPO: 1m timeframe alertas → pending después 1-2 min ──
  const now = Date.now();
  const timeBasedAlerts = botTrades.filter(t => 
    t.symbol === symbol && 
    t.status === 'alerta' &&
    t._alertTf === '1m' &&
    t._alertCreatedAt
  );
  
  for (const alert of timeBasedAlerts) {
    const elapsed = (now - alert._alertCreatedAt) / 1000; // segundos
    const timeout = 90; // 1.5 minutos = 90 segundos (balance entre 1-2 min)
    
    if (elapsed >= timeout) {
      console.log(`[AUTO-CONFIRM 1m] ${alert.direction} ${symbol}: Alerta creada hace ${Math.round(elapsed)}s → PENDING`);
      updateBotTrade(alert.id, {
        status: 'pending',
        entry: currentPrice,
        reason: alert.reason.replace('[🔔 Pre-alerta', '[⏰ AUTO-CONFIRMADA 1m').replace('espera señal', 'ORDEN ACTIVA'),
      });
      saveBotTrades();
      continue;
    }
  }
  
  // ── MODO LÍMITE: Auto-ejecutar pre-alertas cuando precio toca entrada ──
  // SIEMPRE revisar alertas, incluso si modo no está definido
  const alertTrades = botTrades.filter(t => 
    t.symbol === symbol && 
    t.status === 'alerta'
  );
  
  if (alertTrades.length > 0 && executionMode === 'limit') {
    for (const alert of alertTrades) {
      const isLong = alert.direction === 'LONG';

      // ── LÍMITE: máximo 1 trade activo (pending/running) por timeframe ──
      const alertTf = tradeTf(alert);
      if (alertTf) {
        const tfHasActive = botTrades.some(t =>
          t.id !== alert.id &&
          (t.status === 'pending' || t.status === 'running') &&
          tradeTf(t) === alertTf
        );
        if (tfHasActive) {
          console.log(`[AUTO-EXEC] TF ${alertTf} ya tiene trade activo — cancelando alerta ${alert.id}`);
          updateBotTrade(alert.id, { status: 'expired', hitTime: Date.now(), reason: alert.reason + ' | ⛔ Cancelada: TF ya ocupado' });
          saveBotTrades();
          continue;
        }
      }
      
      // TOLERANCIA AMPLIA: ±1% para asegurar ejecución
      const priceDistance = Math.abs(currentPrice - alert.entry) / alert.entry;
      const withinRange = priceDistance <= 0.01; // 1% tolerance
      
      // Para SHORT: si precio ya está por debajo de entrada, ejecutar inmediatamente
      // Para LONG: si precio ya está por encima de entrada, ejecutar inmediatamente
      const shouldExecute = withinRange || 
        (isLong && currentPrice >= alert.entry * 0.99) ||
        (!isLong && currentPrice <= alert.entry * 1.01);
      
      if (shouldExecute) {
        console.log(`[AUTO-EXEC] ${alert.direction} ${symbol}: Precio ${currentPrice} toca entrada ${alert.entry}`);
        
        // Upgrade alerta → pending with actual entry price
        updateBotTrade(alert.id, {
          status: 'pending',
          entry: currentPrice,
          _tf: alertTf,
          reason: alert.reason.replace('[🔔 Pre-alerta', '[💰 AUTO-EJECUTADA').replace('espera señal', 'ORDEN ACTIVA'),
        });
        
        saveBotTrades();
        renderBotTradesTab();
        showNotif(`💰 ORDEN EJECUTADA: ${alert.direction} ${symbol} @ $${fmtPrice(currentPrice)}`, 'success');
      }
    }
  }
  
  // Trackear trades activos del símbolo actual
  const activeTrades = botTrades.filter(t => 
    t.symbol === symbol && 
    (t.status === 'pending' || t.status === 'running')
  );
  
  for (const trade of activeTrades) {
    const isLong = trade.direction === 'LONG';
    
    // ── Verificar si trade pending ya expiró (oportunidad perdida) ──
    if (trade.status === 'pending') {
      // Para SHORT: si precio bajó mucho sin alcanzar entrada, ya pasó la oportunidad
      // Para LONG: si precio subió mucho sin alcanzar entrada, ya pasó la oportunidad
      const opportunityLost = isLong
        ? currentPrice > trade.entry * 1.015  // Subió +1.5% sin entrar → tarde
        : currentPrice < trade.entry * 0.985; // Bajó -1.5% sin entrar → tarde
      
      if (opportunityLost) {
        updateBotTrade(trade.id, {
          status: 'expired',
          hitPrice: currentPrice,
          hitTime: Date.now(),
        });
        
        const direction = isLong ? 'LONG' : 'SHORT';
        const reason = isLong 
          ? `Precio subió a $${fmtPrice(currentPrice)} sin alcanzar entrada $${fmtPrice(trade.entry)}`
          : `Precio bajó a $${fmtPrice(currentPrice)} sin alcanzar entrada $${fmtPrice(trade.entry)}`;
        
        console.log(`[BotTrade] EXPIRADO: ${direction} ${trade.symbol} - ${reason}`);
        continue;
      }
    }
    
    // Activar trade si precio alcanza entrada (±0.2%)
    if (trade.status === 'pending') {
      const entryHit = isLong 
        ? (currentPrice <= trade.entry * 1.002 && currentPrice >= trade.entry * 0.998)
        : (currentPrice >= trade.entry * 0.998 && currentPrice <= trade.entry * 1.002);
      
      if (entryHit) {
        updateBotTrade(trade.id, {
          status: 'running',
          hitPrice: currentPrice,
          hitTime: Date.now(),
        });
        
        showNotif(`✅ Bot Trade ACTIVO: ${trade.direction} ${trade.symbol} @ $${fmtPrice(currentPrice)}`, 'success');
      }
    }
    
    // Verificar SL/TP para trades running
    if (trade.status === 'running') {
      // Usar también el low/high de la vela actual para detectar wicks perdidos
      const lastCandle = window._lastCandles?.[window._lastCandles.length - 1];
      const wickLow  = lastCandle ? Math.min(lastCandle.low,  currentPrice) : currentPrice;
      const wickHigh = lastCandle ? Math.max(lastCandle.high, currentPrice) : currentPrice;

      // Check SL (usando wick)
      const slHit = isLong 
        ? wickLow  <= trade.sl 
        : wickHigh >= trade.sl;
      
      if (slHit) {
        const closePrice = isLong ? Math.min(currentPrice, trade.sl) : Math.max(currentPrice, trade.sl);
        const pnl = isLong
          ? ((closePrice - trade.entry) / trade.entry * 100)
          : ((trade.entry - closePrice) / trade.entry * 100);
        
        // Si TP1 ya fue asegurado → cierre como breakeven (SL estaba en entrada)
        if (trade._tp1Secured) {
          const margin = btMargin; const leverage = btLeverage;
          const pnlUsd = margin * (pnl / 100) * leverage;
          updateBotTrade(trade.id, {
            status: 'win_tp1',
            hitPrice: closePrice,
            hitTime: Date.now(),
            pnlPct: pnl,
            reason: trade.reason + ' | 🔄 TP1 asegurado + SL BE alcanzado',
          });
          saveBotTrades();
          showNotif(`⚖ BE alcanzado tras TP1: ${trade.direction} ${trade.symbol} (${pnl >= 0 ? '+' : ''}${(pnl * btLeverage).toFixed(2)}% ROI)`, pnl >= 0 ? 'success' : 'info');
        } else {
          updateBotTrade(trade.id, {
            status: 'loss',
            hitPrice: closePrice,
            hitTime: Date.now(),
            pnlPct: pnl,
          });
          saveBotTrades();
          showNotif(`❌ Bot Trade LOSS: ${trade.direction} ${trade.symbol} ${(pnl * btLeverage).toFixed(2)}% ROI`, 'error');
        }
        continue;
      }
      
      // Check TP2 (primero porque es más lejano)
      const tp2Hit = isLong 
        ? wickHigh >= trade.tp2 
        : wickLow  <= trade.tp2;
      
      if (tp2Hit) {
        const closePrice = isLong ? Math.max(currentPrice, trade.tp2) : Math.min(currentPrice, trade.tp2);
        const pnl = isLong
          ? ((closePrice - trade.entry) / trade.entry * 100)
          : ((trade.entry - closePrice) / trade.entry * 100);
        
        updateBotTrade(trade.id, {
          status: 'win_tp2',
          hitPrice: closePrice,
          hitTime: Date.now(),
          pnlPct: pnl,
        });
        saveBotTrades();
        showNotif(`🏆 Bot Trade WIN TP2: ${trade.direction} ${trade.symbol} +${(pnl * btLeverage).toFixed(2)}% ROI`, 'success');
        continue;
      }
      
      // Check TP1 — gestión parcial: mover SL a entrada y continuar para TP2
      const tp1Hit = isLong 
        ? wickHigh >= trade.tp1 
        : wickLow  <= trade.tp1;
      
      if (tp1Hit && !trade._tp1Secured) {
        const tp1Price = isLong ? Math.max(currentPrice, trade.tp1) : Math.min(currentPrice, trade.tp1);
        const pnl1 = isLong
          ? ((tp1Price - trade.entry) / trade.entry * 100)
          : ((trade.entry - tp1Price) / trade.entry * 100);
        const margin = btMargin; const leverage = btLeverage;
        const notional = margin * leverage;
        const pnlUsd1 = margin * (pnl1 / 100) * leverage - (notional * 0.0004);
        
        // Parcial: asegurar TP1, mover SL a entrada (breakeven)
        updateBotTrade(trade.id, {
          sl:          trade.entry,      // SL → entrada (breakeven)
          slPct:       0,
          _tp1Secured: true,
          _tp1Price:   tp1Price,
          _tp1Pnl:     pnl1,
          reason:      trade.reason + ` | ✅ TP1 asegurado @ $${fmtPrice(tp1Price)} → SL movido a BE`,
        });
        saveBotTrades();
        showNotif(`✅ TP1 alcanzado: ${trade.direction} ${trade.symbol} +${(pnl1 * btLeverage).toFixed(2)}% ROI ($+${pnlUsd1.toFixed(2)}) — SL → BE, esperando TP2`, 'success');
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
// WEBSOCKET LIVE FEED  (Binance miniTicker — ~500ms push)
// ═══════════════════════════════════════════════════════════════════

const isCrypto = s => /USDT$|USDC$|BTC$|ETH$|BNB$|BUSD$|SOL$/i.test(s);

function wsApply() {
  const syms = [...new Set([
    currentSymbol,
    'BTCUSDT',   // always track for header price
    'XRPUSDT',   // always track for header price
    ...ptPositions.filter(p => p.status === 'open').map(p => p.symbol),
    ...botTrades.filter(t => t.status === 'pending' || t.status === 'running').map(t => t.symbol),
  ])].filter(s => s && isCrypto(s)); // WebSocket only for Binance crypto pairs

  // Kill old connection
  if (liveWS) {
    liveWS.onclose = null;
    liveWS.onerror = null;
    if (liveWS.readyState !== WebSocket.CLOSED) liveWS.close();
    liveWS = null;
  }
  if (wsTimer) { clearTimeout(wsTimer); wsTimer = null; }
  if (!syms.length) return;

  const streams = syms.map(s => s.toLowerCase() + '@miniTicker').join('/');
  liveWS = new WebSocket(`wss://stream.binance.com:9443/stream?streams=${streams}`);

  liveWS.onmessage = (e) => {
    try {
      const msg   = JSON.parse(e.data);
      const data  = msg.data;
      if (!data?.c) return;
      const sym   = data.s;
      const price = parseFloat(data.c);
      if (!sym || !(price > 0)) return;

      // Guardar precio en livePrices para todos los símbolos
      livePrices[sym] = price;

      // ── Header live prices (BTC + XRP) ──────────────────────────────────
      if (sym === 'BTCUSDT') {
        const el = document.getElementById('hdrBtcPrice');
        if (el) el.textContent = '$' + fmtPrice(price);
      }
      if (sym === 'XRPUSDT') {
        const el = document.getElementById('hdrXrpPrice');
        if (el) el.textContent = '$' + price.toFixed(4);
      }

      // Price display update
      if (sym === currentSymbol) {
        currentPrice = price;
        const pe = document.getElementById('aPrice');
        if (pe && document.getElementById('aContent')?.style.display !== 'none')
          pe.textContent = '$' + fmtPrice(price);
        const dp = document.getElementById('dTpPrice');
        if (dp) dp.textContent = '$' + fmtPrice(price);
        const ts = document.getElementById('aLastUpdate');
        if (ts) ts.textContent = new Date().toLocaleTimeString('es',
          { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      }
      
      // Track bot trades para TODOS los símbolos
      trackBotTrades(sym, price);
      
      // Update portfolio progress bars live for current symbol
      if (sym === currentSymbol && window._lastCandles && window._lastIndicators && window._portStatus) {
        updatePortfolioProgressLive(price);
      }
      
      // Actualizar renderizado si estamos en Bot Trades tab (throttle: 1s)
      const now = Date.now();
      if (now - lastBotTradesRender > 1000) {
        lastBotTradesRender = now;
        
        const activeTab = document.querySelector('.tab.active')?.dataset.tab;
        if (activeTab === 'bot-trades') {
          renderBotTradesTab();
        } else if (activeTab === 'analysis') {
          renderBotTrades(); // Vista compacta en Analysis
        }
      }

      // Open positions — update price & check TP/SL/liq
      for (const pos of ptPositions.filter(p => p.status === 'open' && p.symbol === sym)) {
        pos.currentPrice = price;
        const isLong = pos.direction !== 'short';
        if (pos.liqPrice && (isLong ? price <= pos.liqPrice : price >= pos.liqPrice))
          { ptSellDemo(pos.id, 100, 'liquidated'); continue; }
        if (pos.tpPrice  && (isLong ? price >= pos.tpPrice  : price <= pos.tpPrice))
          { ptSellDemo(pos.id, 100, 'tp');         continue; }
        if (pos.slPrice  && (isLong ? price <= pos.slPrice  : price >= pos.slPrice))
          { ptSellDemo(pos.id, 100, 'sl'); }
      }
    } catch (_) {}
  };

  liveWS.onclose = () => { wsTimer = setTimeout(wsApply, 3000); };
  liveWS.onerror = () => {};  // onclose fires after onerror
}

// ═══════════════════════════════════════════════════════════════════
// BINANCE API
// ═══════════════════════════════════════════════════════════════════

const BINANCE = 'https://api.binance.com/api/v3';
const TF_MAP  = { '1m':'1m','3m':'3m','5m':'5m','15m':'15m','30m':'30m',
                  '1h':'1h','2h':'2h','4h':'4h','1d':'1d','1w':'1w' };

async function fetchKlines(symbol, interval = '15m', limit = 200) {
  const iv = TF_MAP[interval] || '15m';
  
  // Timeout de 10 segundos para evitar que se quede colgado
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000);
  
  try {
    const r = await fetch(`${BINANCE}/klines?symbol=${symbol}&interval=${iv}&limit=${limit}`, {
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    
    if (!r.ok) throw new Error(`Binance ${r.status}`);
    const raw = await r.json();
    
    if (!raw || !Array.isArray(raw) || raw.length === 0) {
      throw new Error(`Sin datos de Binance para ${symbol}`);
    }
    
    return raw.map(k => ({
      time:   k[0],
      open:   parseFloat(k[1]),
      high:   parseFloat(k[2]),
      low:    parseFloat(k[3]),
      close:  parseFloat(k[4]),
      volume: parseFloat(k[5]),
    }));
  } catch (e) {
    clearTimeout(timeoutId);
    if (e.name === 'AbortError') {
      throw new Error(`Timeout: Binance no responde para ${symbol}`);
    }
    throw e;
  }
}

async function fetchTicker(symbol) {
  const r = await fetch(`${BINANCE}/ticker/24hr?symbol=${symbol}`);
  if (!r.ok) return null;
  return r.json();
}

async function fetchMultiTicker(symbols) {
  const results = await Promise.allSettled(symbols.map(s => fetchTicker(s)));
  const out = {};
  symbols.forEach((s, i) => {
    if (results[i].status === 'fulfilled' && results[i].value)
      out[s] = results[i].value;
  });
  return out;
}

// ── Yahoo Finance (stocks, forex, indices) ────────────────────────────────────
const YAHOO       = 'https://query1.finance.yahoo.com/v8/finance/chart';
const CRYPTOCOMP  = 'https://min-api.cryptocompare.com/data/v2';
const FF_CAL      = 'https://nfs.faireconomy.media/ff_calendar_thisweek.json';

// ── Session ranges ────────────────────────────────────────────────────────────

// All times UTC (summer, EDT = UTC-4 / CEST = UTC+2)
//   Asia:          00:00–09:00 UTC  = 02:00–11:00 España
//   Londres:       07:00–16:00 UTC  = 09:00–18:00 España
//   NY Pre-market: 13:00–13:30 UTC  = 15:00–15:30 España  (30 min antes apertura)
//   NY Regular:    13:30–20:00 UTC  = 15:30–22:00 España
//   NY Post-market:20:00–24:00 UTC  = 22:00–02:00 España
const SESSIONS = [
  { key:'asia',    name:'Asia (Tokio)',   code:'JP', start:0,    end:9,    color:'#e3b341' },
  { key:'london',  name:'Londres',        code:'EU', start:7,    end:16,   color:'#00d4ff' },
  { key:'ny_pre',  name:'NY Pre-mercado', code:'US', start:13,   end:13.5, color:'#a78bfa', sub:true },
  { key:'ny',      name:'Nueva York',     code:'US', start:13.5, end:20,   color:'#00ff41' },
  { key:'ny_post', name:'NY Post-mercado',code:'US', start:20,   end:24,   color:'#6b7280', sub:true },
];

function fmtHour(h) {
  const hh = Math.floor(h), mm = Math.round((h - hh) * 60);
  return `${String(hh).padStart(2,'0')}:${String(mm).padStart(2,'0')}`;
}

// Convert UTC decimal hour to local time string using browser offset
function fmtHourLocal(h) {
  const offsetH = -new Date().getTimezoneOffset() / 60; // e.g. +2 for CEST
  const local = (h + offsetH + 24) % 24;
  return fmtHour(local);
}

async function fetchSessionRanges(symbol) {
  try {
    // Always fetch 1h candles for today's session ranges regardless of selected TF
    const h1 = await fetchKlinesUniversal(symbol, '1h', 30);
    return computeSessionRanges(h1);
  } catch (_) { return computeSessionRanges([]); }
}

function computeSessionRanges(candles1h) {
  const now      = Date.now();
  const dayStart = new Date(); dayStart.setUTCHours(0,0,0,0);
  const dayMs    = dayStart.getTime();

  return SESSIONS.map(s => {
    const startMs = dayMs + s.start * 3600000;
    const endMs   = dayMs + s.end   * 3600000;
    const sc      = candles1h.filter(c => c.time >= startMs && c.time < Math.min(endMs, now));

    let status, remaining = '';
    if (now < startMs) {
      status = 'pending';
      const mins = Math.round((startMs - now) / 60000);
      remaining = mins < 60 ? `Abre en ${mins}m` : `Abre en ${Math.floor(mins/60)}h${mins%60?` ${mins%60}m`:''}`;
    } else if (now >= endMs) {
      status = 'closed';
    } else {
      status = 'open';
      const mins = Math.round((endMs - now) / 60000);
      remaining = mins < 60 ? `Cierra en ${mins}m` : `Cierra en ${Math.floor(mins/60)}h${mins%60?` ${mins%60}m`:''}`;
    }

    if (!sc.length) return { ...s, status, remaining, high:null, low:null, rangePct:null };
    const high     = Math.max(...sc.map(c => c.high));
    const low      = Math.min(...sc.map(c => c.low));
    const rangePct = ((high - low) / low * 100).toFixed(2);
    return { ...s, status, remaining, high, low, rangePct };
  });
}

// ── Economic calendar — full week, high impact only (Forex Factory) ──────────

const DAY_NAMES = ['Dom','Lun','Mar','Mié','Jue','Vie','Sáb'];
const COUNTRY_FLAG = {
  USD:'🇺🇸', EUR:'🇪🇺', GBP:'🇬🇧', JPY:'🇯🇵', AUD:'🇦🇺',
  CAD:'🇨🇦', CHF:'🇨🇭', CNY:'🇨🇳', NZD:'🇳🇿',
};

async function fetchEcoCalendar() {
  try {
    // Try primary source (Forex Factory via faireconomy.media)
    // Also try with cache-busting to avoid stale weekly data
    const r = await fetch(FF_CAL + '?_=' + new Date().toISOString().slice(0,10));
    if (!r.ok) throw new Error('cal fail');
    const all = await r.json();
    if (!Array.isArray(all) || all.length === 0) throw new Error('empty');
    const todayStr = new Date().toISOString().slice(0,10);

    const high = all.filter(e => e.country === 'USD' && (e.impact === 'High' || e.impact === 'Medium'));
    const byDay = {};
    // isDST: detect if browser is in summer time (affects ET→UTC offset)
    const browserDST = new Date().getTimezoneOffset() < new Date(new Date().getFullYear(), 0, 1).getTimezoneOffset();
    const etOffset   = browserDST ? 4 : 5; // EDT=+4h to UTC, EST=+5h

    for (const e of high) {
      let dateKey, timeStr;

      if (typeof e.date === 'string' && e.date.includes('T')) {
        // Full ISO: "2026-04-27T13:00:00-0400"
        const dt = new Date(e.date);
        if (isNaN(dt)) continue;
        dateKey = dt.toISOString().slice(0, 10);
        const h = dt.getUTCHours(), m = dt.getUTCMinutes();
        timeStr = (h === 0 && m === 0) ? null
                : fmtHour(h) + ':' + String(m).padStart(2,'0') + ' UTC';
      } else if (typeof e.date === 'string') {
        // Plain date "YYYY-MM-DD" + separate time field
        dateKey = e.date.slice(0, 10);
        if (!dateKey || !/^\d{4}-\d{2}-\d{2}$/.test(dateKey)) continue;

        const t = (e.time || '').trim();
        if (!t || t.toLowerCase().includes('day') || t.toLowerCase() === 'tentative') {
          timeStr = null;
        } else {
          // Formats: "2:00am", "10:30am", "14:00"
          const m12 = t.match(/^(\d{1,2}):(\d{2})\s*(am|pm)$/i);
          const m24 = t.match(/^(\d{1,2}):(\d{2})$/);
          if (m12) {
            let h = parseInt(m12[1]), min = parseInt(m12[2]);
            if (m12[3].toLowerCase() === 'pm' && h < 12) h += 12;
            if (m12[3].toLowerCase() === 'am' && h === 12) h = 0;
            const utcH = (h + etOffset) % 24;
            // local hour for display
            const localH = (utcH - new Date().getTimezoneOffset() / 60 + 24 + 24) % 24;
            timeStr = `${fmtHour(localH)} (${fmtHour(utcH)} UTC)`;
          } else if (m24) {
            let h = parseInt(m24[1]), min = parseInt(m24[2]);
            const utcH = (h + etOffset) % 24;
            const localH = (utcH - new Date().getTimezoneOffset() / 60 + 24 + 24) % 24;
            timeStr = `${fmtHour(localH)} (${fmtHour(utcH)} UTC)`;
          } else {
            timeStr = t;
          }
        }
      } else continue;

      if (!dateKey) continue;

      if (!byDay[dateKey]) byDay[dateKey] = [];
      byDay[dateKey].push({
        time:     timeStr || 'Horario pend.',
        country:  e.country || '—',
        title:    e.title   || '—',
        forecast: e.forecast || '',
        previous: e.previous || '',
        isToday:  dateKey === todayStr,
        isPast:   dateKey < todayStr,
        stars:    e.impact === 'High' ? 3 : 2,
      });
    }

    // Sort events within day by time
    for (const evs of Object.values(byDay)) {
      evs.sort((a, b) => a.time.localeCompare(b.time));
    }

    return Object.entries(byDay)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, events]) => {
        const parts = date.split('-').map(Number);
        const d = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2]));
        const lbl = date === todayStr
          ? 'HOY — ' + d.toLocaleDateString('es', { weekday:'long', day:'numeric', month:'long', timeZone:'UTC' })
          : DAY_NAMES[d.getUTCDay()] + ' ' + d.getUTCDate() + ' ' +
            d.toLocaleDateString('es', { month:'short', timeZone:'UTC' });
        return { date, label: lbl, isToday: date === todayStr, isPast: date < todayStr, events };
      });
  } catch (_) { return []; }
}

async function fetchKlinesYahoo(symbol, interval = '15m', limit = 200) {
  const ivMap = { '1m':'1m','5m':'5m','15m':'15m','30m':'30m','1h':'60m','2h':'60m','4h':'1h','1d':'1d','1w':'1wk' };
  const iv    = ivMap[interval] || '15m';
  const range = limit <= 80 ? '5d' : limit <= 200 ? '1mo' : '3mo';
  const r = await fetch(`${YAHOO}/${encodeURIComponent(symbol)}?interval=${iv}&range=${range}&includePrePost=false`);
  if (!r.ok) throw new Error(`Yahoo Finance ${r.status}`);
  const j      = await r.json();
  const result = j.chart?.result?.[0];
  if (!result) throw new Error('Sin datos Yahoo Finance');
  const times  = result.timestamp || [];
  const q      = result.indicators?.quote?.[0] || {};
  const candles = times.map((t, i) => ({
    time:   t * 1000,
    open:   q.open?.[i]   ?? 0,
    high:   q.high?.[i]   ?? 0,
    low:    q.low?.[i]    ?? 0,
    close:  q.close?.[i]  ?? 0,
    volume: q.volume?.[i] ?? 0,
  })).filter(c => c.close > 0);
  if (!candles.length) throw new Error(`Sin velas Yahoo Finance para ${symbol}`);
  return candles;
}

async function fetchKlinesUniversal(symbol, interval, limit) {
  try {
    return await fetchKlines(symbol, interval, limit);
  } catch (_) {
    return fetchKlinesYahoo(symbol, interval, limit);
  }
}

// ═══════════════════════════════════════════════════════════════════
// TECHNICAL INDICATORS
// ═══════════════════════════════════════════════════════════════════

function calcSMA(data, period) {
  return data.map((_, i) => {
    if (i < period - 1) return null;
    return data.slice(i - period + 1, i + 1).reduce((s, v) => s + v, 0) / period;
  });
}

function calcEMA(data, period) {
  const k = 2 / (period + 1);
  const out = [];
  let ema = null;
  for (const v of data) {
    if (ema === null) { ema = v; }
    else              { ema = v * k + ema * (1 - k); }
    out.push(ema);
  }
  return out;
}

function calcRSI(closes, period = 14) {
  const gains = [], losses = [];
  for (let i = 1; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    gains.push(d > 0 ? d : 0);
    losses.push(d < 0 ? -d : 0);
  }
  const rsi = [null];
  let avgG = gains.slice(0, period).reduce((s, v) => s + v, 0) / period;
  let avgL = losses.slice(0, period).reduce((s, v) => s + v, 0) / period;
  rsi.push(avgL === 0 ? 100 : 100 - 100 / (1 + avgG / avgL));
  for (let i = period; i < gains.length; i++) {
    avgG = (avgG * (period - 1) + gains[i]) / period;
    avgL = (avgL * (period - 1) + losses[i]) / period;
    rsi.push(avgL === 0 ? 100 : 100 - 100 / (1 + avgG / avgL));
  }
  // Pad to match closes length
  while (rsi.length < closes.length) rsi.unshift(null);
  return rsi;
}

function calcMACD(closes, fast = 12, slow = 26, sig = 9) {
  const emaFast = calcEMA(closes, fast);
  const emaSlow = calcEMA(closes, slow);
  const macd    = emaFast.map((v, i) => v - emaSlow[i]);
  const signal  = calcEMA(macd, sig);
  const hist    = macd.map((v, i) => v - signal[i]);
  return { macd, signal, hist };
}

function calcBB(closes, period = 20, mult = 2) {
  const sma = calcSMA(closes, period);
  return closes.map((_, i) => {
    if (i < period - 1) return { mid: null, upper: null, lower: null, width: null };
    const slice = closes.slice(i - period + 1, i + 1);
    const mean  = sma[i];
    const std   = Math.sqrt(slice.reduce((s, v) => s + (v - mean) ** 2, 0) / period);
    return { mid: mean, upper: mean + mult * std, lower: mean - mult * std, width: (2 * mult * std) / mean };
  });
}

function calcATR(candles, period = 14) {
  const tr = candles.map((c, i) => {
    if (i === 0) return c.high - c.low;
    const prev = candles[i - 1];
    return Math.max(c.high - c.low, Math.abs(c.high - prev.close), Math.abs(c.low - prev.close));
  });
  const atr = calcSMA(tr, period);
  return atr;
}

function calcStoch(candles, kPeriod = 14, dPeriod = 3) {
  const k = candles.map((_, i) => {
    if (i < kPeriod - 1) return null;
    const slice = candles.slice(i - kPeriod + 1, i + 1);
    const hh = Math.max(...slice.map(c => c.high));
    const ll = Math.min(...slice.map(c => c.low));
    return hh === ll ? 50 : ((candles[i].close - ll) / (hh - ll)) * 100;
  });
  const d = calcSMA(k.filter(v => v !== null), dPeriod);
  return { k, d };
}

// ═══════════════════════════════════════════════════════════════════
// SWING DETECTION
// ═══════════════════════════════════════════════════════════════════

function findSwings(candles, lookback = 5) {
  const highs = [], lows = [];
  for (let i = lookback; i < candles.length - lookback; i++) {
    const slice = candles.slice(i - lookback, i + lookback + 1);
    const maxH  = Math.max(...slice.map(c => c.high));
    const minL  = Math.min(...slice.map(c => c.low));
    if (candles[i].high === maxH) highs.push({ i, price: candles[i].high });
    if (candles[i].low  === minL) lows.push({ i,  price: candles[i].low });
  }
  return { highs, lows };
}

function supportResistance(candles, n = 5) {
  const { highs, lows } = findSwings(candles, n);
  const levels = [];
  // Cluster nearby levels
  const allLevels = [
    ...highs.map(h => ({ price: h.price, type: 'resistance' })),
    ...lows.map(l  => ({ price: l.price, type: 'support' })),
  ].sort((a, b) => a.price - b.price);

  let cluster = null;
  for (const lvl of allLevels) {
    if (!cluster) { cluster = { ...lvl, count: 1 }; continue; }
    if (Math.abs(lvl.price - cluster.price) / cluster.price < 0.005) {
      cluster.count++;
      cluster.price = (cluster.price + lvl.price) / 2;
    } else {
      levels.push(cluster);
      cluster = { ...lvl, count: 1 };
    }
  }
  if (cluster) levels.push(cluster);
  return levels.sort((a, b) => b.count - a.count).slice(0, 8);
}

// ═══════════════════════════════════════════════════════════════════
// HARMONIC PATTERNS (Gartley, Butterfly, Bat, Crab, Shark)
// ═══════════════════════════════════════════════════════════════════

function findAlternatingPivots(candles, lb = 5) {
  const raw = [];
  for (let i = lb; i < candles.length - lb; i++) {
    const hi = candles[i].high;
    const lo = candles[i].low;
    let isH = true, isL = true;
    for (let k = 1; k <= lb; k++) {
      if (candles[i-k].high >= hi || candles[i+k].high >= hi) isH = false;
      if (candles[i-k].low  <= lo || candles[i+k].low  <= lo) isL = false;
    }
    if (isH) raw.push({ i, type: 'H', price: hi });
    else if (isL) raw.push({ i, type: 'L', price: lo });
  }
  // Keep only alternating pivots (pick more extreme when consecutive same type)
  const clean = [];
  for (const p of raw) {
    const last = clean[clean.length - 1];
    if (!last || last.type !== p.type) { clean.push(p); continue; }
    if ((p.type === 'H' && p.price > last.price) || (p.type === 'L' && p.price < last.price))
      clean[clean.length - 1] = p;
  }
  return clean;
}

function detectHarmonicPatterns(candles) {
  const price = candles[candles.length - 1].close;
  // Intentar con distintos lookbacks para adaptarse a distintos timeframes
  let pivots = findAlternatingPivots(candles, 5);
  if (pivots.length < 4) pivots = findAlternatingPivots(candles, 3);
  if (pivots.length < 4) return [];

  const FIB_TOL = 0.09;
  function inRange(v, lo, hi) {
    return v >= lo * (1 - FIB_TOL) && v <= hi * (1 + FIB_TOL);
  }
  function near(v, t) { return Math.abs(v - t) / t <= FIB_TOL; }

  // Pattern definitions: { name, abXa_min, abXa_max, bcAb_min, bcAb_max, dXa_min, dXa_max, acc }
  const DEFS = [
    { name: 'Gartley',   abMin:0.58, abMax:0.66, bcMin:0.38, bcMax:0.89, dMin:0.74, dMax:0.83, acc:78 },
    { name: 'Bat',       abMin:0.36, abMax:0.52, bcMin:0.38, bcMax:0.89, dMin:0.84, dMax:0.92, acc:82 },
    { name: 'Butterfly', abMin:0.74, abMax:0.84, bcMin:0.38, bcMax:0.89, dMin:1.22, dMax:1.70, acc:73 },
    { name: 'Crab',      abMin:0.36, abMax:0.62, bcMin:0.38, bcMax:0.89, dMin:1.55, dMax:1.68, acc:71 },
    { name: 'Shark',     abMin:0.44, abMax:0.56, bcMin:1.00, bcMax:1.68, dMin:0.85, dMax:0.92, acc:68 },
  ];

  const results = [];
  const recent = pivots.slice(-8);

  for (let i = 0; i <= recent.length - 4; i++) {
    const [X, A, B, C] = recent.slice(i, i + 4);
    if (X.type === A.type || A.type === B.type || B.type === C.type) continue;

    const XA = Math.abs(A.price - X.price);
    const AB = Math.abs(B.price - A.price);
    const BC = Math.abs(C.price - B.price);
    if (XA < 1e-12 || AB < 1e-12) continue;

    const abXa = AB / XA;
    const bcAb = BC / AB;
    const isBull = A.type === 'L'; // A=low → D is bullish reversal zone

    for (const def of DEFS) {
      if (!inRange(abXa, def.abMin, def.abMax)) continue;
      if (!inRange(bcAb, def.bcMin, def.bcMax)) continue;

      // D completion zone (PRZ)
      const dir = A.price > X.price ? 1 : -1; // XA direction
      const dMid = X.price - dir * XA * ((def.dMin + def.dMax) / 2);
      const dLow  = X.price - dir * XA * def.dMax;
      const dHigh = X.price - dir * XA * def.dMin;

      const distPct = Math.abs(price - dMid) / Math.max(price, dMid) * 100;
      let stage, prob;
      if (distPct < 1.5)  { stage = 'COMPLETANDO'; prob = def.acc; }
      else if (distPct < 5) { stage = 'APPROACHANDO PRZ'; prob = Math.round(def.acc * 0.8); }
      else { stage = 'FORMANDO'; prob = Math.round(def.acc * 0.5); }

      const t1 = isBull ? B.price : B.price;
      const t2 = isBull ? A.price : A.price;
      const fib618 = dMid + (isBull ? 1 : -1) * Math.abs(C.price - dMid) * 0.618;
      const fib100 = isBull ? A.price : A.price;

      results.push({
        name: def.name,
        type: isBull ? 'bullish' : 'bearish',
        stage, prob,
        dZone: { low: Math.min(dLow, dHigh), high: Math.max(dLow, dHigh), mid: dMid },
        sl:   isBull ? Math.min(dLow, dHigh) * 0.993 : Math.max(dLow, dHigh) * 1.007,
        tp1: fib618,
        tp2: fib100,
        distPct: distPct.toFixed(1),
        ratios: { 'AB/XA': abXa.toFixed(3), 'BC/AB': bcAb.toFixed(3) },
      });
      break; // one pattern per XABC combination
    }
  }

  const seen = new Set();
  return results
    .sort((a, b) => b.prob - a.prob)
    .filter(p => { if (seen.has(p.name)) return false; seen.add(p.name); return true; })
    .slice(0, 4);
}

// ═══════════════════════════════════════════════════════════════════
// ELLIOTT WAVE DETECTION (simplified 5-wave impulse / 3-wave ABC)
// ═══════════════════════════════════════════════════════════════════

function detectElliottWave(candles) {
  const price = candles[candles.length - 1].close;
  let pivots = findAlternatingPivots(candles, 4);
  if (pivots.length < 5) pivots = findAlternatingPivots(candles, 3);
  if (pivots.length < 5) pivots = findAlternatingPivots(candles, 2);
  if (pivots.length < 5) return null;

  const recent = pivots.slice(-7);

  // Try to find 5-wave impulse in the last 5+ pivots
  for (let s = 0; s <= recent.length - 5; s++) {
    const pts = recent.slice(s, s + 5);
    const [p0, p1, p2, p3, p4] = pts;

    const isBullImpulse = p0.type === 'L' && p1.type === 'H' && p2.type === 'L' && p3.type === 'H' && p4.type === 'L';
    const isBearImpulse = p0.type === 'H' && p1.type === 'L' && p2.type === 'H' && p3.type === 'L' && p4.type === 'H';
    if (!isBullImpulse && !isBearImpulse) continue;

    const w1 = Math.abs(p1.price - p0.price);
    const w2 = Math.abs(p2.price - p1.price);
    const w3 = Math.abs(p3.price - p2.price);
    const w4 = Math.abs(p4.price - p3.price);
    if (!w1 || !w3) continue;

    // Elliott rules — reglas DURAS (ambas deben cumplirse)
    const hard1 = w2 / w1 < 1.0;   // W2 nunca retrocede 100%+ de W1 (fundamental)
    const hard2 = w4 / w3 < 1.0;   // W4 nunca retrocede 100%+ de W3
    if (!hard1 || !hard2) continue; // si falla cualquiera, no es impulso válido

    // Reglas BLANDAS — necesita al menos 1 de 2
    const soft1 = w3 >= Math.min(w1, w4);   // W3 no es la ola más corta
    const soft2 = isBullImpulse ? p4.price > p1.price : p4.price < p1.price; // W4 no solapa W1
    if (!soft1 && !soft2) continue;

    const score = [hard1, hard2, soft1, soft2].filter(Boolean).length; // 2–4

    const dir = isBullImpulse ? 'bullish' : 'bearish';
    const confidence = 45 + score * 13;

    // Determine current wave from current price
    let currentWave, nextMove, nextTarget;
    if (isBullImpulse) {
      if (price > p3.price) {
        currentWave = 'W5 ▲'; nextMove = 'Impulso final alcista — buscar techo';
        nextTarget = p3.price + w3 * 0.618;
      } else if (price > p2.price) {
        currentWave = 'W4 ↘ (correctivo)'; nextMove = 'Corrección — soporte en ~' + fmtPrice(p2.price) + ', luego W5';
        nextTarget = p3.price + w1 * 0.618;
      } else {
        currentWave = 'W3 ▲ (más fuerte)'; nextMove = 'Impulso W3 en marcha — la más poderosa';
        nextTarget = p1.price + w1 * 1.618;
      }
    } else {
      if (price < p3.price) {
        currentWave = 'W5 ▼'; nextMove = 'Impulso bajista final — buscar suelo';
        nextTarget = p3.price - w3 * 0.618;
      } else if (price < p2.price) {
        currentWave = 'W4 ↗ (correctivo)'; nextMove = 'Corrección — resistencia en ~' + fmtPrice(p2.price) + ', luego W5 bajista';
        nextTarget = p3.price - w1 * 0.618;
      } else {
        currentWave = 'W3 ▼ (más fuerte)'; nextMove = 'Caída W3 en marcha';
        nextTarget = p1.price - w1 * 1.618;
      }
    }

    return { type: 'impulso', dir, currentWave, nextMove, nextTarget, confidence: Math.min(88, confidence),
      w2Ret: (w2/w1*100).toFixed(0), w3vsW1: (w3/w1).toFixed(2), w4Ret: (w4/w3*100).toFixed(0),
      pts: { p0, p1, p2, p3, p4 } };
  }

  // Try ABC correction (3 alternating pivots)
  if (recent.length >= 3) {
    const [pA, pB, pC] = recent.slice(-3);
    if (pA.type !== pB.type && pB.type !== pC.type) {
      const wA = Math.abs(pB.price - pA.price);
      const wB = Math.abs(pC.price - pB.price);
      const bcRet = wB / wA;
      if (bcRet < 1.0) {
        const dir = pA.type === 'H' ? 'bearish' : 'bullish';
        const cTarget = pA.type === 'H'
          ? pB.price - wA * 0.618
          : pB.price + wA * 0.618;
        return {
          type: 'corrección', dir, currentWave: 'Wave C', confidence: 52,
          nextMove: `Wave C ${dir === 'bearish' ? 'bajista' : 'alcista'} — objetivo ~$${fmtPrice(cTarget)}`,
          nextTarget: cTarget, w2Ret: null, w3vsW1: null, w4Ret: null,
          pts: { p0: pA, p1: pB, p2: pC }
        };
      }
    }
  }
  return null;
}

// ═══════════════════════════════════════════════════════════════════
// MULTI-TIMEFRAME ANALYSIS
// ═══════════════════════════════════════════════════════════════════

const MTF_MAP = {
  '1m':  { primary: '1h',  secondary: '15m' },
  '5m':  { primary: '4h',  secondary: '1h'  },
  '15m': { primary: '4h',  secondary: '1h'  },
  '30m': { primary: '1d',  secondary: '4h'  },
  '1h':  { primary: '1d',  secondary: '4h'  },
  '4h':  { primary: '1w',  secondary: '1d'  },
  '1d':  { primary: '1w',  secondary: '4h'  },
  '1w':  { primary: '1w',  secondary: '1d'  },
};

async function fetchMTFanalysis(symbol, currentTf) {
  const tfConf = MTF_MAP[currentTf] || { primary: '1d', secondary: '4h' };

  // ── S/R rápido por swing highs/lows ──────────────────────────────
  const calcSR = (cArr) => {
    if (!cArr || cArr.length < 20) return { res: null, sup: null };
    const price = cArr[cArr.length - 1].close;
    const lb = 4;
    let res = null, sup = null;
    for (let i = lb; i < cArr.length - lb; i++) {
      const sl = cArr.slice(i - lb, i + lb + 1);
      if (cArr[i].high >= Math.max(...sl.map(c => c.high)) && cArr[i].high > price) {
        if (!res || cArr[i].high < res) res = cArr[i].high;
      }
      if (cArr[i].low <= Math.min(...sl.map(c => c.low)) && cArr[i].low < price) {
        if (!sup || cArr[i].low > sup) sup = cArr[i].low;
      }
    }
    return { res, sup };
  };

  // ── Tendencia + RSI + S/R por TF ─────────────────────────────────
  const calcTrend = (closes, cArr) => {
    if (!closes?.length) return null;
    const n = closes.length;
    const ema = (arr, per) => {
      const k = 2 / (per + 1);
      return arr.reduce((prev, v, i) => i === 0 ? v : prev * (1 - k) + v * k);
    };
    const ema20 = ema(closes.slice(-20), 20);
    const ema50 = ema(closes.slice(-Math.min(50, n)), 50);
    const last = closes[n - 1];
    const rsiData = closes.slice(-15);
    const gains = [], losses = [];
    for (let i = 1; i < rsiData.length; i++) {
      const d = rsiData[i] - rsiData[i-1];
      gains.push(d > 0 ? d : 0); losses.push(d < 0 ? -d : 0);
    }
    const ag = gains.reduce((a,b)=>a+b,0)/gains.length;
    const al = losses.reduce((a,b)=>a+b,0)/losses.length;
    const rsi = al === 0 ? 99 : Math.round(100 - 100/(1+ag/al));
    let dir;
    if (last > ema20 && ema20 > ema50)      dir = 'bullish';
    else if (last < ema20 && ema20 < ema50) dir = 'bearish';
    else if (last > ema20)                   dir = 'neutral_up';
    else                                     dir = 'neutral_down';
    const sr = calcSR(cArr);
    return { dir, rsi, ema20: ema20.toFixed(2), last: last.toFixed(2), res: sr.res, sup: sr.sup };
  };

  try {
    const [primCandles, secCandles] = await Promise.all([
      fetchKlinesUniversal(symbol, tfConf.primary,   80).catch(() => null),
      fetchKlinesUniversal(symbol, tfConf.secondary, 80).catch(() => null),
    ]);
    const prim = calcTrend(primCandles?.map(c=>c.close), primCandles);
    const sec  = calcTrend(secCandles?.map(c=>c.close),  secCandles);
    if (!prim && !sec) return null;

    const dirs = [prim?.dir, sec?.dir].filter(Boolean);
    const bullN = dirs.filter(d => d.includes('bullish') || d === 'neutral_up').length;
    const bearN = dirs.filter(d => d.includes('bearish') || d === 'neutral_down').length;
    let align, alignScore;
    if      (bullN === 2)    { align = 'ALCISTA ↑↑';  alignScore = 88; }
    else if (bearN === 2)    { align = 'BAJISTA ↓↓';  alignScore = 88; }
    else if (bullN > bearN)  { align = 'ALCISTA ↑~';  alignScore = 62; }
    else if (bearN > bullN)  { align = 'BAJISTA ↓~';  alignScore = 62; }
    else                     { align = 'DIVERGENTE ↕'; alignScore = 35; }

    return { primaryTf: tfConf.primary, secondaryTf: tfConf.secondary,
      currentTf, prim, sec, align, alignScore };
  } catch { return null; }
}

// ═══════════════════════════════════════════════════════════════════
// ADVANCED PROBABILITY ENGINE
// ═══════════════════════════════════════════════════════════════════

function computeAdvancedProb({ score, mtf, harmonics, elliott, div, volume, patterns, rsi, hist, mlResult }) {
  let bull, bear, conf;
  const factors = [];

  if (mlResult) {
    // Cuando hay modelo ML: la base es el ML (60%) + score técnico (40%)
    const mlBull = mlResult.bullPct;
    const mlBear = mlResult.bearPct;
    bull = mlBull * 0.6 + score * 0.4;
    bear = mlBear * 0.6 + (100 - score) * 0.4;
    conf = 50 + Math.round((mlResult.auc - 0.5) * 80); // AUC 0.70 → conf 66
    factors.push(`🤖 ML ${mlBull}% alcista · AUC ${mlResult.auc}`);
  } else {
    bull = score;
    bear = 100 - score;
    conf = 48;
    factors.push('Reglas técnicas (sin modelo ML)');
  }

  // MTF alignment (high weight)
  if (mtf) {
    if (mtf.align === 'ALCISTA ↑↑')  { bull += 18; conf += 18; factors.push('MTF ↑↑ alineado alcista'); }
    else if (mtf.align === 'BAJISTA ↓↓') { bear += 18; conf += 18; factors.push('MTF ↓↓ alineado bajista'); }
    else if (mtf.align?.includes('ALCISTA')) { bull += 9; conf += 6; factors.push('MTF parcialmente alcista'); }
    else if (mtf.align?.includes('BAJISTA')) { bear += 9; conf += 6; factors.push('MTF parcialmente bajista'); }
    else { conf -= 12; factors.push('MTF divergente — señal reducida'); }
  }

  // Harmonic patterns (very high weight when near PRZ)
  const bestH = harmonics?.[0];
  if (bestH?.stage === 'COMPLETANDO' || bestH?.stage === 'APPROACHANDO PRZ') {
    const w = bestH.stage === 'COMPLETANDO' ? 22 : 14;
    if (bestH.type === 'bullish') { bull += w; conf += 14; }
    else { bear += w; conf += 14; }
    factors.push(`Armónico ${bestH.name} ${bestH.stage} (${bestH.prob}%)`);
  }

  // Elliott wave
  if (elliott) {
    const w = elliott.currentWave;
    if (w?.includes('W3') && elliott.dir === 'bullish')  { bull += 16; conf += 10; factors.push(`Elliott ${w} — ola más fuerte`); }
    else if (w?.includes('W3') && elliott.dir === 'bearish') { bear += 16; conf += 10; factors.push(`Elliott ${w}`); }
    else if (w?.includes('W5') && elliott.dir === 'bullish') { bull += 8; bear += 4; factors.push(`Elliott ${w} — ola final (reversión pronto)`); }
    else if (w?.includes('W5') && elliott.dir === 'bearish') { bear += 8; bull += 4; factors.push(`Elliott ${w} — ola final`); }
    else if (w?.includes('W4')) { conf -= 8; factors.push(`Elliott ${w} — corrección activa`); }
    conf += Math.round(elliott.confidence * 0.12);
  }

  // Classic signals
  if (div?.type === 'bullish') { bull += 14; conf += 8; factors.push('Divergencia alcista RSI'); }
  if (div?.type === 'bearish') { bear += 14; conf += 8; factors.push('Divergencia bajista RSI'); }
  if (volume?.spike) {
    if (volume.bias === 'buyPressure') { bull += 8; factors.push(`Spike volumen ${volume.ratio?.toFixed(1)}x comprador`); }
    else { bear += 8; factors.push(`Spike volumen ${volume.ratio?.toFixed(1)}x vendedor`); }
  }
  const topPat = patterns?.find(p => p.probability >= 60);
  if (topPat) {
    if (topPat.type === 'bullish') { bull += 10; factors.push(`Patrón ${topPat.name} (${topPat.probability}%)`); }
    else { bear += 10; factors.push(`Patrón ${topPat.name} (${topPat.probability}%)`); }
  }

  const total = bull + bear || 1;
  const bullPct = Math.min(97, Math.max(3, Math.round(bull / total * 100)));
  const bearPct = 100 - bullPct;
  conf = Math.min(96, Math.max(28, Math.round(conf)));
  return { bullPct, bearPct, conf, factors };
}

// ═══════════════════════════════════════════════════════════════════
// RSI DIVERGENCE
// ═══════════════════════════════════════════════════════════════════

function detectRSIDivergence(candles, rsi) {
  const n = candles.length;
  const lookback = 30;
  const start    = Math.max(n - lookback, 5);
  const recent   = candles.slice(start);
  const recentRsi = rsi.slice(start);

  // Find last two swing lows and highs in price + RSI
  const priceHighs = [], priceLows = [];
  for (let i = 2; i < recent.length - 2; i++) {
    if (recent[i].high > recent[i-1].high && recent[i].high > recent[i+1].high &&
        recent[i].high > recent[i-2].high && recent[i].high > recent[i+2].high)
      priceHighs.push({ i, price: recent[i].high, rsi: recentRsi[i] });
    if (recent[i].low < recent[i-1].low && recent[i].low < recent[i+1].low &&
        recent[i].low < recent[i-2].low && recent[i].low < recent[i+2].low)
      priceLows.push({ i, price: recent[i].low, rsi: recentRsi[i] });
  }

  const result = { type: null, strength: 0, label: '' };

  // Bearish divergence: price HH but RSI LH
  if (priceHighs.length >= 2) {
    const [prev, curr] = priceHighs.slice(-2);
    if (curr.price > prev.price && curr.rsi < prev.rsi && curr.rsi !== null && prev.rsi !== null) {
      const priceDiff = ((curr.price - prev.price) / prev.price) * 100;
      const rsiDiff   = curr.rsi - prev.rsi;
      result.type     = 'bearish';
      result.strength = Math.min(100, Math.abs(rsiDiff) * 3);
      result.label    = `Divergencia BAJISTA: Precio +${priceDiff.toFixed(1)}% | RSI ${rsiDiff.toFixed(1)} pts`;
    }
  }
  // Bullish divergence: price LL but RSI HL
  if (priceLows.length >= 2) {
    const [prev, curr] = priceLows.slice(-2);
    if (curr.price < prev.price && curr.rsi > prev.rsi && curr.rsi !== null && prev.rsi !== null) {
      const priceDiff = ((prev.price - curr.price) / prev.price) * 100;
      const rsiDiff   = curr.rsi - prev.rsi;
      if (Math.abs(rsiDiff) > Math.abs(result.strength / 3 || 0)) {
        result.type     = 'bullish';
        result.strength = Math.min(100, Math.abs(rsiDiff) * 3);
        result.label    = `Divergencia ALCISTA: Precio −${priceDiff.toFixed(1)}% | RSI +${rsiDiff.toFixed(1)} pts`;
      }
    }
  }
  return result;
}

// ═══════════════════════════════════════════════════════════════════
// CHART PATTERN DETECTION
// ═══════════════════════════════════════════════════════════════════

function detectChartPatterns(candles) {
  const patterns = [];
  const n = candles.length;
  const last = candles.slice(-50);
  const { highs, lows } = findSwings(last, 3);

  const currentPrice = last[last.length - 1].close;

  // ── Double Top ─────────────────────────────────────────────────
  if (highs.length >= 2) {
    const [h1, h2] = highs.slice(-2);
    const diff = Math.abs(h1.price - h2.price) / h1.price;
    if (diff < 0.015 && h2.i > h1.i + 3) {
      const neckline = Math.min(...last.slice(h1.i, h2.i + 1).map(c => c.low));
      const target   = neckline - (h1.price - neckline);
      const prob     = Math.round(60 + (1 - diff / 0.015) * 20);
      patterns.push({
        name: 'Doble Techo',
        type: 'bearish',
        probability: prob,
        target,
        description: `Dos máximos en ~$${fmtPrice(h1.price)}. Objetivo bajista: $${fmtPrice(target)}`,
        neckline,
      });
    }
  }

  // ── Double Bottom ──────────────────────────────────────────────
  if (lows.length >= 2) {
    const [l1, l2] = lows.slice(-2);
    const diff = Math.abs(l1.price - l2.price) / l1.price;
    if (diff < 0.015 && l2.i > l1.i + 3) {
      const neckline = Math.max(...last.slice(l1.i, l2.i + 1).map(c => c.high));
      const target   = neckline + (neckline - l1.price);
      const prob     = Math.round(60 + (1 - diff / 0.015) * 20);
      patterns.push({
        name: 'Doble Suelo',
        type: 'bullish',
        probability: prob,
        target,
        description: `Dos mínimos en ~$${fmtPrice(l1.price)}. Objetivo alcista: $${fmtPrice(target)}`,
        neckline,
      });
    }
  }

  // ── Head & Shoulders ───────────────────────────────────────────
  if (highs.length >= 3) {
    const [left, head, right] = highs.slice(-3);
    const shoulderDiff = Math.abs(left.price - right.price) / left.price;
    if (head.price > left.price * 1.01 && head.price > right.price * 1.01 && shoulderDiff < 0.02) {
      const neckline = Math.min(
        Math.min(...last.slice(left.i, head.i).map(c => c.low)),
        Math.min(...last.slice(head.i, right.i + 1).map(c => c.low))
      );
      const target = neckline - (head.price - neckline);
      patterns.push({
        name: 'Cabeza y Hombros',
        type: 'bearish',
        probability: 72,
        target,
        description: `Patrón bajista clásico. Objetivo: $${fmtPrice(target)}`,
        neckline,
      });
    }
  }

  // ── Inverse Head & Shoulders ───────────────────────────────────
  if (lows.length >= 3) {
    const [left, head, right] = lows.slice(-3);
    const shoulderDiff = Math.abs(left.price - right.price) / left.price;
    if (head.price < left.price * 0.99 && head.price < right.price * 0.99 && shoulderDiff < 0.02) {
      const neckline = Math.max(
        Math.max(...last.slice(left.i, head.i).map(c => c.high)),
        Math.max(...last.slice(head.i, right.i + 1).map(c => c.high))
      );
      const target = neckline + (neckline - head.price);
      patterns.push({
        name: 'HCH Invertido',
        type: 'bullish',
        probability: 72,
        target,
        description: `Reversión alcista. Objetivo: $${fmtPrice(target)}`,
        neckline,
      });
    }
  }

  // ── Triangle (Ascending / Descending / Symmetrical) ────────────
  if (highs.length >= 3 && lows.length >= 3) {
    const recentHighs = highs.slice(-3);
    const recentLows  = lows.slice(-3);
    const highSlope = (recentHighs[2].price - recentHighs[0].price) / (recentHighs[2].i - recentHighs[0].i);
    const lowSlope  = (recentLows[2].price  - recentLows[0].price)  / (recentLows[2].i  - recentLows[0].i);
    const isAscending   = highSlope < 0.1 && lowSlope > 0.5;
    const isDescending  = highSlope < -0.5 && lowSlope > -0.1;
    const isSymmetrical = highSlope < -0.3 && lowSlope > 0.3;
    if (isAscending) {
      patterns.push({ name: 'Triángulo Ascendente', type: 'bullish', probability: 65, description: 'Compresión con techo plano — ruptura alcista esperada.' });
    } else if (isDescending) {
      patterns.push({ name: 'Triángulo Descendente', type: 'bearish', probability: 65, description: 'Compresión con suelo plano — ruptura bajista esperada.' });
    } else if (isSymmetrical) {
      patterns.push({ name: 'Triángulo Simétrico', type: 'neutral', probability: 55, description: 'Compresión — ruptura en la dirección de la tendencia.' });
    }
  }

  // ── Bull/Bear Flag ─────────────────────────────────────────────
  const flagLookback = last.slice(-20);
  const poleStart    = flagLookback[0].close;
  const poleEnd      = flagLookback[10]?.close;
  if (poleEnd && Math.abs(poleEnd - poleStart) / poleStart > 0.03) {
    const flagBody = flagLookback.slice(10);
    const flagHH   = Math.max(...flagBody.map(c => c.high));
    const flagLL   = Math.min(...flagBody.map(c => c.low));
    const flagRange = (flagHH - flagLL) / flagLookback[10].close;
    if (flagRange < 0.02) {
      if (poleEnd > poleStart) {
        patterns.push({ name: 'Bull Flag', type: 'bullish', probability: 68, description: `Consolidación tras subida de ${((poleEnd-poleStart)/poleStart*100).toFixed(1)}%. Continuación alcista esperada.` });
      } else {
        patterns.push({ name: 'Bear Flag', type: 'bearish', probability: 68, description: `Consolidación tras caída de ${((poleStart-poleEnd)/poleStart*100).toFixed(1)}%. Continuación bajista esperada.` });
      }
    }
  }

  return patterns;
}

// ═══════════════════════════════════════════════════════════════════
// SMART MONEY CONCEPTS (SMC)
// ═══════════════════════════════════════════════════════════════════

function detectSMC(candles) {
  const n = candles.length;
  const recent = candles.slice(-80);

  // ── Fair Value Gaps (FVG) ──────────────────────────────────────
  const fvgs = [];
  for (let i = 2; i < recent.length - 1; i++) {
    const c1 = recent[i - 2], c2 = recent[i - 1], c3 = recent[i];
    // Bullish FVG: c3.low > c1.high (gap between c1 top and c3 bottom)
    if (c3.low > c1.high) {
      fvgs.push({ type: 'bullish', top: c3.low, bottom: c1.high, i, filled: false,
        label: `FVG Alcista $${fmtPrice(c1.high)}–$${fmtPrice(c3.low)}` });
    }
    // Bearish FVG: c3.high < c1.low
    if (c3.high < c1.low) {
      fvgs.push({ type: 'bearish', top: c1.low, bottom: c3.high, i, filled: false,
        label: `FVG Bajista $${fmtPrice(c3.high)}–$${fmtPrice(c1.low)}` });
    }
  }
  // Mark filled FVGs
  const price = recent[recent.length - 1].close;
  const activeFvgs = fvgs.slice(-6).filter(g => {
    if (g.type === 'bullish') return price > g.bottom; // still valid if price above
    return price < g.top;
  });

  // ── Order Blocks ───────────────────────────────────────────────
  const orderBlocks = [];
  for (let i = 5; i < recent.length - 5; i++) {
    const c = recent[i];
    const nextMove = recent.slice(i + 1, i + 6);
    const maxNext  = Math.max(...nextMove.map(x => x.close));
    const minNext  = Math.min(...nextMove.map(x => x.close));
    // Bullish OB: a bearish candle followed by strong up move
    if (c.close < c.open && maxNext > c.high * 1.01) {
      orderBlocks.push({ type: 'bullish', top: c.open, bottom: c.close, i,
        label: `OB Alcista ~$${fmtPrice((c.open + c.close) / 2)}` });
    }
    // Bearish OB: a bullish candle followed by strong down move
    if (c.close > c.open && minNext < c.low * 0.99) {
      orderBlocks.push({ type: 'bearish', top: c.close, bottom: c.open, i,
        label: `OB Bajista ~$${fmtPrice((c.open + c.close) / 2)}` });
    }
  }
  const activeOBs = orderBlocks.slice(-4).filter(ob => {
    if (ob.type === 'bullish') return price >= ob.bottom * 0.995 && price <= ob.top * 1.05;
    return price <= ob.top * 1.005 && price >= ob.bottom * 0.95;
  });

  // ── Break of Structure (BOS) / Change of Character (ChoCH) ────
  const { highs, lows } = findSwings(recent, 4);
  let bos = null;
  if (highs.length >= 2) {
    const [prev, curr] = highs.slice(-2);
    if (curr.price > prev.price) bos = { type: 'bullish_bos', label: `BOS Alcista — nuevo máximo $${fmtPrice(curr.price)}`, price: curr.price };
    else                         bos = { type: 'choch',        label: `ChoCH Bajista — máximo más bajo $${fmtPrice(curr.price)}`, price: curr.price };
  }
  if (lows.length >= 2) {
    const [prev, curr] = lows.slice(-2);
    const bullBos = curr.price > prev.price;
    if (!bos || (bullBos && bos?.type !== 'bullish_bos')) {
      bos = bullBos
        ? { type: 'choch',       label: `ChoCH Alcista — mínimo más alto $${fmtPrice(curr.price)}`, price: curr.price }
        : { type: 'bearish_bos', label: `BOS Bajista — nuevo mínimo $${fmtPrice(curr.price)}`,       price: curr.price };
    }
  }

  // ── Premium / Discount Zones ───────────────────────────────────
  const rangeHigh = Math.max(...recent.map(c => c.high));
  const rangeLow  = Math.min(...recent.map(c => c.low));
  const rangeMid  = (rangeHigh + rangeLow) / 2;
  const zone = price > rangeMid
    ? { label: 'ZONA PREMIUM (sobrecompra)', type: 'bearish', pct: ((price - rangeMid) / (rangeHigh - rangeMid) * 100).toFixed(0) }
    : { label: 'ZONA DISCOUNT (descuento)',  type: 'bullish', pct: ((rangeMid - price) / (rangeMid - rangeLow) * 100).toFixed(0) };

  // ── Liquidity ─────────────────────────────────────────────────
  const swingHighs = highs.slice(-3).map(h => h.price);
  const swingLows  = lows.slice(-3).map(l => l.price);

  return { fvgs: activeFvgs, orderBlocks: activeOBs, bos, zone, rangeHigh, rangeLow, swingHighs, swingLows };
}

// ═══════════════════════════════════════════════════════════════════
// MARKET STRUCTURE
// ═══════════════════════════════════════════════════════════════════

function detectTrend(candles, ema20, ema50, ema200) {
  const n    = candles.length - 1;
  const { highs, lows } = findSwings(candles.slice(-40), 4);

  let hhhl = 0; // higher highs & higher lows = uptrend
  if (highs.length >= 2) {
    const [prev, curr] = highs.slice(-2);
    if (curr.price > prev.price) hhhl++;
    else                         hhhl--;
  }
  if (lows.length >= 2) {
    const [prev, curr] = lows.slice(-2);
    if (curr.price > prev.price) hhhl++;
    else                         hhhl--;
  }

  const e20  = ema20[n], e50 = ema50[n], e200 = ema200[n];
  const price = candles[n].close;

  let score = 0;
  if (price > e20)   score++;
  if (e20 > e50)     score++;
  if (e50 > e200)    score++;
  if (hhhl >= 1)     score++;
  if (price > e200)  score += 2;

  let trend, strength;
  if      (score >= 5) { trend = 'UPTREND FUERTE';  strength = 95; }
  else if (score >= 4) { trend = 'UPTREND';          strength = 75; }
  else if (score >= 3) { trend = 'NEUTRAL / LATERAL'; strength = 50; }
  else if (score >= 2) { trend = 'DOWNTREND';         strength = 35; }
  else                 { trend = 'DOWNTREND FUERTE';  strength = 10; }

  return {
    trend, strength, score,
    ema20: e20, ema50: e50, ema200: e200,
    priceVsEma200: e200 ? ((price - e200) / e200 * 100).toFixed(2) : null,
    goldCross: e20 > e50 && ema20[n-1] <= ema50[n-1],
    deathCross: e20 < e50 && ema20[n-1] >= ema50[n-1],
  };
}

// ═══════════════════════════════════════════════════════════════════
// VOLUME ANALYSIS
// ═══════════════════════════════════════════════════════════════════

function analyzeVolume(candles) {
  const n     = candles.length;
  const last5 = candles.slice(-5);
  const avg20 = calcSMA(candles.map(c => c.volume), 20)[n - 1];
  const curr  = candles[n - 1];
  const volRatio = avg20 ? curr.volume / avg20 : 1;
  const buyVol = last5.filter(c => c.close > c.open).reduce((s, c) => s + c.volume, 0);
  const sellVol = last5.filter(c => c.close < c.open).reduce((s, c) => s + c.volume, 0);
  const cvd = buyVol - sellVol; // cumulative volume delta approximation

  return {
    current:  curr.volume,
    avg20,
    ratio:    volRatio,
    spike:    volRatio > 2,
    buyVol, sellVol, cvd,
    label:    volRatio > 2   ? 'SPIKE — alta actividad'
            : volRatio > 1.5 ? 'Por encima del promedio'
            : volRatio < 0.5 ? 'Bajo — poca actividad'
            : 'Normal',
    bias:     cvd > 0 ? 'buyPressure' : 'sellPressure',
  };
}

// ═══════════════════════════════════════════════════════════════════
// VERDICT & PROJECTION
// ═══════════════════════════════════════════════════════════════════

function computeVerdict({ score, trend, patterns, smc, div, volume, indicators, price }) {
  const bullReasons = [];
  const bearReasons = [];

  // EMA alignment
  if (trend.score >= 5) bullReasons.push('EMAs 20/50/200 alineadas alcistas — tendencia confirmada');
  else if (trend.score <= 2) bearReasons.push('EMAs 20/50/200 alineadas bajistas — tendencia confirmada');
  if (trend.goldCross)  bullReasons.push('Golden Cross activo (EMA20 cruzó EMA50 al alza)');
  if (trend.deathCross) bearReasons.push('Death Cross activo (EMA20 cruzó EMA50 a la baja)');

  // RSI
  const rsi = indicators.rsi;
  if (rsi !== null) {
    if (rsi < 30)       bullReasons.push(`RSI sobrevendido (${rsi.toFixed(0)}) — posible rebote inminente`);
    else if (rsi > 70)  bearReasons.push(`RSI sobrecomprado (${rsi.toFixed(0)}) — zona de rechazo potencial`);
    else if (rsi > 55)  bullReasons.push(`RSI en zona alcista (${rsi.toFixed(0)}) — momentum positivo`);
    else if (rsi < 45)  bearReasons.push(`RSI en zona bajista (${rsi.toFixed(0)}) — momentum negativo`);
  }

  // RSI divergence
  if (div?.type === 'bullish') bullReasons.push(`Divergencia alcista RSI — precio baja pero RSI sube`);
  if (div?.type === 'bearish') bearReasons.push(`Divergencia bajista RSI — precio sube pero RSI baja`);

  // MACD histogram
  const hist = indicators.hist;
  if (hist > 0)  bullReasons.push('Histograma MACD positivo — compradores en control');
  if (hist < 0)  bearReasons.push('Histograma MACD negativo — vendedores en control');

  // Chart patterns
  for (const p of patterns.slice(0, 3)) {
    if (p.type === 'bullish') bullReasons.push(`Patrón: ${p.name} (confianza ${p.probability}%)`);
    if (p.type === 'bearish') bearReasons.push(`Patrón: ${p.name} (confianza ${p.probability}%)`);
  }

  // SMC zone
  if (smc.zone.type === 'bullish') bullReasons.push('Precio en zona de descuento SMC — posición favorable compradores');
  if (smc.zone.type === 'bearish') bearReasons.push('Precio en zona premium SMC — posición favorable vendedores');

  // BOS / ChoCH
  if (smc.bos) {
    const bLabel = smc.bos.label || '';
    if (smc.bos.type === 'bullish_bos')   bullReasons.push(`BOS alcista — estructura rota al alza`);
    if (smc.bos.type === 'bearish_bos')   bearReasons.push(`BOS bajista — estructura rota a la baja`);
    if (smc.bos.type === 'choch' && bLabel.includes('Alcista')) bullReasons.push('ChoCH alcista — posible cambio de tendencia');
    if (smc.bos.type === 'choch' && bLabel.includes('Bajista')) bearReasons.push('ChoCH bajista — posible cambio de tendencia');
  }

  // Order Blocks
  for (const ob of smc.orderBlocks.slice(0, 2)) {
    const obMid = ((ob.top || ob.high || 0) + (ob.bottom || ob.low || 0)) / 2;
    if (ob.type === 'bullish' && obMid > 0) bullReasons.push(`Order Block comprador ~$${fmtPrice(obMid)}`);
    if (ob.type === 'bearish' && obMid > 0) bearReasons.push(`Order Block vendedor ~$${fmtPrice(obMid)}`);
  }

  // Volume
  if (volume.bias === 'buyPressure')  bullReasons.push('CVD positivo — presión compradora domina');
  if (volume.bias === 'sellPressure') bearReasons.push('CVD negativo — presión vendedora domina');
  if (volume.spike && score >= 50)    bullReasons.push('Spike de volumen acompañando movimiento alcista');
  if (volume.spike && score < 50)     bearReasons.push('Spike de volumen acompañando movimiento bajista');

  // Final verdict
  const bullPct = score;
  const bearPct = 100 - score;
  const confidence = Math.min(100, Math.round(Math.abs(score - 50) * 2));

  let label, cls;
  if      (score >= 72) { label = 'ALCISTA FUERTE';    cls = 'verd-strong-bull'; }
  else if (score >= 58) { label = 'ALCISTA';            cls = 'verd-bull'; }
  else if (score >= 48) { label = 'NEUTRAL';            cls = 'verd-neutral'; }
  else if (score >= 35) { label = 'BAJISTA';            cls = 'verd-bear'; }
  else                  { label = 'BAJISTA FUERTE';     cls = 'verd-strong-bear'; }

  return { bullPct, bearPct, confidence, label, cls, bullReasons, bearReasons };
}

// ═══════════════════════════════════════════════════════════════════
// TRADING BRAIN — synthesizes all data like a pro trader
// ═══════════════════════════════════════════════════════════════════

function computeTradingBrain({ score, trend, patterns, smc, div, volume, signals,
  indicators, price, symbol, tf, support, resistance, suggestedSL, suggestedTP }) {

  const { rsi, hist, bb, atr } = indicators;
  const supP   = support?.price;
  const resP   = resistance?.price;
  const bbSqueeze = bb?.width != null && bb.width < 0.025;
  const atrPct    = atr && price ? (atr / price) * 100 : 1;

  const topPat = patterns.find(p => p.probability >= 60);

  // ── Key signals with prices ───────────────────────────────────────
  const keySignals = [];

  // Patterns — most important, show exact prices
  for (const p of patterns.slice(0, 3)) {
    const hasPrice = p.neckline || p.target;
    keySignals.push({
      dir:   p.type,
      label: `${p.name} (${p.probability}%)`,
      sub:   [
        p.neckline ? `Neckline: $${fmtPrice(p.neckline)}` : '',
        p.target   ? `Objetivo: $${fmtPrice(p.target)}`    : '',
        p.description?.replace(/[^$0-9.,A-Za-záéíóú~ ]/g,'').trim() || '',
      ].filter(Boolean).join(' · '),
    });
  }

  // RSI
  if (rsi !== null) {
    if      (rsi < 30)  keySignals.push({ dir:'bullish', label:`RSI sobrevendido (${rsi.toFixed(0)})`, sub:`Zona de rebote — posible giro alcista` });
    else if (rsi > 70)  keySignals.push({ dir:'bearish', label:`RSI sobrecomprado (${rsi.toFixed(0)})`, sub:`Zona de rechazo — posible corrección` });
    else if (rsi > 55)  keySignals.push({ dir:'bullish', label:`RSI alcista (${rsi.toFixed(0)})`, sub:`Momentum positivo` });
    else if (rsi < 45)  keySignals.push({ dir:'bearish', label:`RSI bajista (${rsi.toFixed(0)})`, sub:`Momentum negativo` });
  }

  // RSI divergence
  if (div?.type === 'bullish') keySignals.push({ dir:'bullish', label:'Divergencia alcista RSI', sub:'Precio hace mínimos, RSI sube — señal de giro' });
  if (div?.type === 'bearish') keySignals.push({ dir:'bearish', label:'Divergencia bajista RSI', sub:'Precio sube, RSI baja — agotamiento del impulso' });

  // MACD
  if (hist > 0)  keySignals.push({ dir:'bullish', label:'MACD histograma positivo', sub:`Momentum comprador activo` });
  if (hist < 0)  keySignals.push({ dir:'bearish', label:'MACD histograma negativo', sub:`Momentum vendedor activo` });

  // BB squeeze
  if (bbSqueeze) keySignals.push({ dir:'neutral', label:'BB Squeeze', sub:`Ancho ${(bb.width*100).toFixed(2)}% — ruptura explosiva inminente, dirección sin confirmar` });

  // Support/Resistance levels
  if (resP) keySignals.push({ dir:'bearish', label:`Resistencia $${fmtPrice(resP)}`, sub:`${fmtPct((resP-price)/price*100)} arriba — techo clave` });
  if (supP) keySignals.push({ dir:'bullish', label:`Soporte $${fmtPrice(supP)}`, sub:`${fmtPct((supP-price)/price*100)} abajo — piso clave` });

  // SMC
  if (smc.bos) keySignals.push({ dir: smc.bos.type.includes('bull') ? 'bullish':'bearish', label: smc.bos.label, sub:'Estructura de mercado' });
  if (smc.zone.type === 'bullish') keySignals.push({ dir:'bullish', label:'Zona SMC Descuento', sub:'Precio bajo punto medio del rango — favorable compradores' });
  if (smc.zone.type === 'bearish') keySignals.push({ dir:'bearish', label:'Zona SMC Premium', sub:'Precio sobre punto medio del rango — favorable vendedores' });

  // Volume
  if (volume.spike) keySignals.push({ dir: volume.bias==='buyPressure'?'bullish':'bearish',
    label:`Spike de volumen ${volume.ratio.toFixed(1)}x promedio`,
    sub: volume.bias==='buyPressure' ? 'Acumulación institucional potencial' : 'Distribución / venta institucional' });

  // ── Final direction verdict ───────────────────────────────────────
  const bullSignals = keySignals.filter(s => s.dir === 'bullish').length;
  const bearSignals = keySignals.filter(s => s.dir === 'bearish').length;
  const totalSig    = bullSignals + bearSignals || 1;
  const bullWeight  = score; // 0-100 from full analysis
  const bearWeight  = 100 - score;

  // Clear verdict when pattern or strong divergence present
  let finalDir, finalConf, finalReason;
  if (topPat?.type === 'bearish' && topPat.probability >= 70 && bearWeight >= 45) {
    finalDir  = 'SHORT';
    finalConf = Math.round((topPat.probability + bearWeight) / 2);
    finalReason = `${topPat.name} (${topPat.probability}%)${topPat.neckline ? ` — romper $${fmtPrice(topPat.neckline)} activa el patrón` : ''}`;
  } else if (topPat?.type === 'bullish' && topPat.probability >= 70 && bullWeight >= 45) {
    finalDir  = 'LONG';
    finalConf = Math.round((topPat.probability + bullWeight) / 2);
    finalReason = `${topPat.name} (${topPat.probability}%)${topPat.neckline ? ` — romper $${fmtPrice(topPat.neckline)} activa el patrón` : ''}`;
  } else if (div?.type === 'bearish' && score < 50) {
    finalDir  = 'SHORT'; finalConf = Math.round(65 + (50-score)); finalReason = 'Divergencia bajista RSI + sesgo bajista';
  } else if (div?.type === 'bullish' && score > 50) {
    finalDir  = 'LONG';  finalConf = Math.round(65 + (score-50)); finalReason = 'Divergencia alcista RSI + sesgo alcista';
  } else if (score >= 65) {
    finalDir  = 'LONG';  finalConf = score; finalReason = 'Score técnico alto — tendencia alcista dominante';
  } else if (score <= 35) {
    finalDir  = 'SHORT'; finalConf = 100-score; finalReason = 'Score técnico bajo — tendencia bajista dominante';
  } else if (bbSqueeze) {
    finalDir  = 'ESPERAR'; finalConf = null; finalReason = 'BB Squeeze — sin dirección hasta la ruptura';
  } else {
    finalDir  = bearSignals > bullSignals ? 'SHORT' : bullSignals > bearSignals ? 'LONG' : 'ESPERAR';
    finalConf = finalDir === 'ESPERAR' ? null : Math.round(Math.max(bearWeight,bullWeight) * 0.8);
    finalReason = 'Mayoría de señales ' + (finalDir === 'SHORT' ? 'bajistas' : finalDir === 'LONG' ? 'alcistas' : 'neutrales');
  }

  // ── Narrative (concise summary) ──────────────────────────────────
  const narrative = [
    score >= 65 ? 'Momentum alcista dominante.' : score <= 35 ? 'Presión bajista dominante.' : 'Mercado en zona neutral.',
    topPat ? `${topPat.name}${topPat.neckline ? ` — neckline $${fmtPrice(topPat.neckline)}` : ''}${topPat.target ? `, objetivo $${fmtPrice(topPat.target)}` : ''}.` : '',
    bbSqueeze ? `BB Squeeze (${(bb.width*100).toFixed(2)}%) — esperar ruptura con volumen.` : '',
    div?.type ? `Divergencia ${div.type === 'bullish' ? 'alcista' : 'bajista'} RSI activa.` : '',
  ].filter(Boolean).slice(0,3).join(' ');

  // ── Trading mode — reactive to current price ─────────────────────
  let mode, modeCol, modeIcon, breakoutAlert = null;
  const nearRes = resP && Math.abs(price - resP) / price < 0.004;
  const nearSup = supP && Math.abs(price - supP) / price < 0.004;

  // Check real-time breakouts (highest priority)
  const bearNeckBroken = topPat?.type === 'bearish' && topPat.neckline && price < topPat.neckline * 0.9995;
  const bullNeckBroken = topPat?.type === 'bullish' && topPat.neckline && price > topPat.neckline * 1.0005;
  const bbUpperBreak   = indicators.bb?.upper && price > indicators.bb.upper && !bbSqueeze;
  const bbLowerBreak   = indicators.bb?.lower && price < indicators.bb.lower && !bbSqueeze;
  const supBroken      = supP && price < supP * 0.998;
  const resBroken      = resP && price > resP * 1.002;

  if (bearNeckBroken) {
    mode = `▼ SHORT ACTIVADO — Neckline $${fmtPrice(topPat.neckline)} roto`;
    modeCol = '#f85149'; modeIcon = '🔴';
    breakoutAlert = { type:'bear', msg:`Neckline ${topPat.name} roto en $${fmtPrice(topPat.neckline)} — TARGET $${fmtPrice(topPat.target)}`, urgent: true };
  } else if (bullNeckBroken) {
    mode = `▲ LONG ACTIVADO — Neckline $${fmtPrice(topPat.neckline)} roto`;
    modeCol = '#00ff41'; modeIcon = '🟢';
    breakoutAlert = { type:'bull', msg:`Neckline ${topPat.name} roto en $${fmtPrice(topPat.neckline)} — TARGET $${fmtPrice(topPat.target)}`, urgent: true };
  } else if (supBroken && score < 50) {
    mode = `▼ SOPORTE PERDIDO — $${fmtPrice(supP)} roto`;
    modeCol = '#f85149'; modeIcon = '🔴';
    breakoutAlert = { type:'bear', msg:`Soporte $${fmtPrice(supP)} perdido — presión bajista aumenta`, urgent: false };
  } else if (resBroken && score > 50) {
    mode = `▲ RESISTENCIA ROTA — $${fmtPrice(resP)} superado`;
    modeCol = '#00ff41'; modeIcon = '🟢';
    breakoutAlert = { type:'bull', msg:`Resistencia $${fmtPrice(resP)} rota — momentum alcista confirmado`, urgent: false };
  } else if (bbUpperBreak) {
    mode = 'RUPTURA ALCISTA BB — MOMENTUM FUERTE'; modeCol = '#00ff41'; modeIcon = '▲';
    breakoutAlert = { type:'bull', msg:`Precio rompe Banda Superior BB — sesgo alcista activado`, urgent: false };
  } else if (bbLowerBreak) {
    mode = 'RUPTURA BAJISTA BB — POSIBLE CONTINUACIÓN'; modeCol = '#f85149'; modeIcon = '▼';
    breakoutAlert = { type:'bear', msg:`Precio rompe Banda Inferior BB — sesgo bajista activado`, urgent: false };
  } else if (bbSqueeze) {
    mode = 'ESPERAR RUPTURA BB'; modeCol = '#e3b341'; modeIcon = '⏳';
  } else if (nearRes && score < 60) {
    mode = `EN RESISTENCIA $${fmtPrice(resP)} — ESPERAR RECHAZO O RUPTURA`; modeCol = '#f85149'; modeIcon = '🚫';
  } else if (nearSup && score > 40) {
    mode = `EN SOPORTE $${fmtPrice(supP)} — BUSCAR REBOTE`; modeCol = '#00ff41'; modeIcon = '👁';
  } else if (score >= 65) {
    mode = 'BUSCAR ENTRADA LONG EN RETROCESO'; modeCol = '#00ff41'; modeIcon = '▲';
  } else if (score <= 35) {
    mode = 'BUSCAR ENTRADA SHORT EN REBOTE'; modeCol = '#f85149'; modeIcon = '▼';
  } else {
    mode = 'MODO OBSERVACIÓN — SIN EDGE CLARO'; modeCol = '#e3b341'; modeIcon = '👁';
  }

  // Fire browser notification on breakout (once per event)
  if (breakoutAlert) {
    const alertKey = `brain_alert_${symbol}_${breakoutAlert.msg.slice(0,30)}`;
    if (!window[alertKey]) {
      window[alertKey] = true;
      setTimeout(() => delete window[alertKey], 120000); // reset after 2min
      showNotif(`🔔 ${breakoutAlert.msg}`, breakoutAlert.type === 'bull' ? 'success' : 'error');
    }
  }

  // ── Plan A: Bull scenario ─────────────────────────────────────────
  const bullPct  = score;
  const atrVal   = Math.max(atr || 0, price * 0.001); // mínimo 0.1% para evitar ATR=0

  // Validar S/R: solo usar supP si está DEBAJO del precio (es soporte real)
  //               solo usar resP si está ENCIMA del precio (es resistencia real)
  const validSup = supP && supP < price * 0.9998;  // soporte ≥ 0.02% bajo precio
  const validRes = resP && resP > price * 1.0002;  // resistencia ≥ 0.02% sobre precio

  // LONG — entrada cerca del precio, SL siempre por debajo, TP siempre por encima
  // Usamos ATR consistente para evitar inversiones en cualquier timeframe
  const bEntry = validSup ? +(Math.min(supP * 1.001, price - atrVal * 0.3)).toFixed(8)
                           : +(price - atrVal * 0.3).toFixed(8);
  const bTP1   = validRes ? +(resP * 0.998).toFixed(8)
                           : +(price + atrVal * 3).toFixed(8);
  const bTP2   = validRes ? +(resP * 1.015).toFixed(8)
                           : +(price + atrVal * 5).toFixed(8);
  const bSL    = validSup ? +(supP * 0.994).toFixed(8)
                           : +(price - atrVal * 2.5).toFixed(8);
  // Garantía: bEntry siempre entre bSL y bTP1
  const bRR    = bEntry > bSL && bTP1 > bEntry
                   ? +((bTP1 - bEntry) / (bEntry - bSL)).toFixed(1)
                   : 0;

  const bullConditions = [];
  if (validSup) bullConditions.push(`Precio mantiene soporte $${fmtPrice(supP)} como piso`);
  if (div?.type === 'bullish') bullConditions.push('Divergencia alcista RSI confirma reversión');
  if (topPat?.type === 'bullish') bullConditions.push(`Ruptura del neckline $${fmtPrice(topPat.neckline || (validRes ? resP : price))} con volumen`);
  if (smc.zone.type === 'bullish') bullConditions.push('Precio en zona de descuento SMC');
  if (bullConditions.length === 0) bullConditions.push(`Cierre de vela por encima de $${fmtPrice(price + atrVal)} con volumen`);

  // ── Plan B: Bear scenario ─────────────────────────────────────────
  const bearPct  = 100 - score;
  // SHORT — entrada cerca del precio, SL siempre por encima, TP siempre por debajo
  const sEntry = validRes ? +(Math.max(resP * 0.999, price + atrVal * 0.3)).toFixed(8)
                           : +(price + atrVal * 0.3).toFixed(8);
  const sTP1   = validSup ? +(supP * 1.001).toFixed(8)
                           : +(price - atrVal * 3).toFixed(8);
  const sTP2   = validSup ? +(supP * 0.985).toFixed(8)
                           : +(price - atrVal * 5).toFixed(8);
  const sSL    = validRes ? +(resP * 1.006).toFixed(8)
                           : +(price + atrVal * 2.5).toFixed(8);
  // Garantía: sEntry siempre entre sTP1 y sSL
  const sRR    = sSL > sEntry && sEntry > sTP1
                   ? +((sEntry - sTP1) / (sSL - sEntry)).toFixed(1)
                   : 0;

  const bearConditions = [];
  if (resP)  bearConditions.push(`Precio rechazado en resistencia $${fmtPrice(resP)}`);
  if (div?.type === 'bearish') bearConditions.push('Divergencia bajista RSI confirma agotamiento');
  if (topPat?.type === 'bearish') bearConditions.push(`Ruptura del neckline $${fmtPrice(topPat.neckline || supP || price)} a la baja`);
  if (smc.zone.type === 'bearish') bearConditions.push('Precio en zona premium SMC — distribución');
  if (bearConditions.length === 0) bearConditions.push(`Cierre de vela por debajo de $${fmtPrice(price * 0.997)} con volumen`);

  // ── Wait-for conditions (with prices) ────────────────────────────
  const waitFor = [];
  if (topPat?.type === 'bearish' && topPat.neckline) {
    waitFor.push(`Cierre de vela bajo neckline $${fmtPrice(topPat.neckline)} con volumen → SHORT activado. Sin romper ese nivel, patrón inválido.`);
  } else if (topPat?.type === 'bullish' && topPat.neckline) {
    waitFor.push(`Cierre de vela sobre neckline $${fmtPrice(topPat.neckline)} con volumen → LONG activado.`);
  }
  if (bbSqueeze) waitFor.push(`Ruptura BB (ancho ${(bb.width*100).toFixed(2)}%) — no entrar antes de saber la dirección. Monitorear cierre fuera de las bandas.`);
  if (div)       waitFor.push(`Divergencia ${div.type==='bullish'?'alcista':'bajista'} RSI — confirmar que el precio sigue la señal del RSI.`);
  if (resP && price < resP * 0.998) waitFor.push(`Resistencia clave en $${fmtPrice(resP)} — solo LONG si cierra con fuerza por encima.`);
  if (supP && price > supP * 1.002) waitFor.push(`Soporte clave en $${fmtPrice(supP)} — SHORT si pierde ese nivel con volumen.`);
  if (!waitFor.length) {
    if (score >= 60)      waitFor.push(`Retroceso a EMA20 (~$${fmtPrice(trend.ema20)}) sin perder estructura alcista — zona de entrada LONG`);
    else if (score <= 40) waitFor.push(`Rebote a EMA20 (~$${fmtPrice(trend.ema20)}) para entrada SHORT con SL ajustado sobre el rebote`);
    else                  waitFor.push(`Ruptura del rango con volumen 2x el promedio para definir dirección`);
  }

  // ── Pro summary (with prices) ─────────────────────────────────────
  let proTake;
  if (finalDir === 'SHORT' && topPat?.type === 'bearish') {
    proTake = `${topPat.name} confirma sesgo bajista. ${topPat.neckline ? `Esperaría cierre bajo $${fmtPrice(topPat.neckline)} (neckline) para entrar SHORT` : `Buscaría rebote a resistencia $${fmtPrice(resP||sEntry)} para entrar SHORT`}, SL sobre $${fmtPrice(sSL)}, TP en $${fmtPrice(sTP1)}${topPat.target ? ` (objetivo patrón $${fmtPrice(topPat.target)})` : ''}. R:R ${sRR}:1.`;
  } else if (finalDir === 'LONG' && topPat?.type === 'bullish') {
    proTake = `${topPat.name} confirma sesgo alcista. ${topPat.neckline ? `Esperaría cierre sobre $${fmtPrice(topPat.neckline)} (neckline) para entrar LONG` : `Buscaría pullback a soporte $${fmtPrice(supP||bEntry)}`}, SL bajo $${fmtPrice(bSL)}, TP en $${fmtPrice(bTP1)}${topPat.target ? ` (objetivo $${fmtPrice(topPat.target)})` : ''}. R:R ${bRR}:1.`;
  } else if (finalDir === 'ESPERAR') {
    proTake = `Sin setup claro todavía. Esperaría: ${waitFor[0]?.toLowerCase() || 'señal de confirmación'}. Flat (sin posición) es una posición válida.`;
  } else if (finalDir === 'LONG') {
    proTake = `Sesgo alcista. Retroceso a $${fmtPrice(bEntry)} con SL en $${fmtPrice(bSL)} y TP en $${fmtPrice(bTP1)}. R:R ${bRR}:1. ${atrPct.toFixed(1)}% de movimiento promedio (ATR).`;
  } else {
    proTake = `Sesgo bajista. Rebote a $${fmtPrice(sEntry)} con SL en $${fmtPrice(sSL)} y TP en $${fmtPrice(sTP1)}. R:R ${sRR}:1.`;
  }

  // ── Pine Script pipeline ──────────────────────────────────────────
  const pine = buildBrainPine({ symbol, tf, price, resP, supP, bEntry, bTP1, bTP2, bSL, sEntry, sTP1, sTP2, sSL, bullPct, bearPct });

  return {
    narrative, mode, modeCol, modeIcon,
    finalDir, finalConf, finalReason,
    keySignals, breakoutAlert,
    bullPct, bEntry, bTP1, bTP2, bSL, bRR, bullConditions,
    bearPct, sEntry, sTP1, sTP2, sSL, sRR, bearConditions,
    waitFor, proTake, pine,
  };
}

function buildBrainPine({ symbol, tf, price, resP, supP, bEntry, bTP1, bSL, sEntry, sTP1, sSL, bullPct, bearPct }) {
  const p = v => v ? v.toFixed(8) : 'na';
  return `//@version=5
// TVA Brain — ${symbol} ${tf}  |  Alcista ${bullPct}%  Bajista ${bearPct}%
indicator("TVA Brain — ${symbol}", overlay=true, max_lines_count=10)

// ── Niveles ──
var float res    = ${p(resP)}
var float sup    = ${p(supP)}
var float bEntry = ${p(bEntry)}
var float bTP    = ${p(bTP1)}
var float bSL    = ${p(bSL)}
var float sEntry = ${p(sEntry)}
var float sTP    = ${p(sTP1)}
var float sSL    = ${p(sSL)}

// ── Visualización ──
plot(res,    "Resistencia",    color.new(color.orange, 20), 2)
plot(sup,    "Soporte",        color.new(color.aqua,   20), 2)
plot(bEntry, "Long Entry",     color.new(color.green,  40), 1, plot.style_circles)
plot(bTP,    "Long TP",        color.new(color.green,  20), 1, plot.style_cross)
plot(bSL,    "Long SL",        color.new(color.red,    20), 1, plot.style_cross)
plot(sEntry, "Short Entry",    color.new(color.red,    40), 1, plot.style_circles)
plot(sTP,    "Short TP",       color.new(color.aqua,   20), 1, plot.style_cross)
plot(sSL,    "Short SL",       color.new(color.orange, 20), 1, plot.style_cross)

// ── Alertas automáticas ──
alertcondition(ta.crossunder(close, sup),  "⚠️ Soporte roto",     "Soporte $${supP ? supP.toFixed(2) : '?'} roto — evaluar SHORT")
alertcondition(ta.crossover(close,  res),  "⚠️ Resistencia rota", "Resistencia $${resP ? resP.toFixed(2) : '?'} rota — evaluar LONG")
alertcondition(close <= bEntry and close > bSL, "✅ Zona LONG",   "Precio en zona de entrada LONG — confirmar")
alertcondition(close >= sEntry and close < sSL, "✅ Zona SHORT",  "Precio en zona de entrada SHORT — confirmar")
`;
}

// ═══════════════════════════════════════════════════════════════════
// FULL ANALYSIS ENGINE
// ═══════════════════════════════════════════════════════════════════

function runAnalysis(candles, symbol, tf) {
  if (candles.length < 50) throw new Error('Datos insuficientes');

  const closes  = candles.map(c => c.close);
  const n       = candles.length - 1;

  // Indicators
  const ema20   = calcEMA(closes, 20);
  const ema50   = calcEMA(closes, 50);
  const ema200  = calcEMA(closes, 200);
  const rsi     = calcRSI(closes, 14);
  const macd    = calcMACD(closes);
  const bb      = calcBB(closes, 20, 2);
  const atr     = calcATR(candles, 14);
  const stoch   = calcStoch(candles);

  const price   = closes[n];
  const rsiNow  = rsi[n];
  const macdNow = macd.macd[n];
  const sigNow  = macd.signal[n];
  const histNow = macd.hist[n];
  const bbNow   = bb[n];
  const atrNow  = atr[n];
  const stochK  = (stoch.k.filter(v => v !== null)).slice(-1)[0] || 50;

  // Analysis modules
  const trend    = detectTrend(candles, ema20, ema50, ema200);
  const patterns = detectChartPatterns(candles);
  const smc      = detectSMC(candles);
  const div      = detectRSIDivergence(candles, rsi);
  const volume   = analyzeVolume(candles);
  const levels   = supportResistance(candles);

  // ── Scoring ────────────────────────────────────────────────────
  let score = 50; // neutral start
  const signals = [];

  // Trend (±20)
  const trendBull = trend.score >= 4;
  const trendBear = trend.score <= 2;
  if (trendBull)  { score += 15; signals.push({ label: trend.trend, type: 'bullish', pts: 15 }); }
  if (trendBear)  { score -= 15; signals.push({ label: trend.trend, type: 'bearish', pts: -15 }); }

  // EMA cross (±10)
  if (trend.goldCross)  { score += 10; signals.push({ label: 'Golden Cross EMA20/50', type: 'bullish', pts: 10 }); }
  if (trend.deathCross) { score -= 10; signals.push({ label: 'Death Cross EMA20/50',  type: 'bearish', pts: -10 }); }

  // RSI (±12)
  if (rsiNow !== null) {
    if      (rsiNow > 70) { score -= 10; signals.push({ label: `RSI Sobrecomprado ${rsiNow.toFixed(0)}`, type: 'bearish', pts: -10 }); }
    else if (rsiNow < 30) { score += 10; signals.push({ label: `RSI Sobrevendido ${rsiNow.toFixed(0)}`,  type: 'bullish', pts: 10 }); }
    else if (rsiNow > 55) { score += 4; }
    else if (rsiNow < 45) { score -= 4; }
  }

  // RSI Divergence (±15)
  if (div.type === 'bullish') { score += 12; signals.push({ label: div.label, type: 'bullish', pts: 12 }); }
  if (div.type === 'bearish') { score -= 12; signals.push({ label: div.label, type: 'bearish', pts: -12 }); }

  // MACD (±10)
  if (macdNow !== null && sigNow !== null) {
    if (macdNow > sigNow && macd.hist[n] > macd.hist[n-1]) { score += 8;  signals.push({ label: 'MACD Cruce Alcista', type: 'bullish', pts: 8 }); }
    if (macdNow < sigNow && macd.hist[n] < macd.hist[n-1]) { score -= 8;  signals.push({ label: 'MACD Cruce Bajista', type: 'bearish', pts: -8 }); }
  }

  // Bollinger Bands (±8)
  if (bbNow.lower && bbNow.upper) {
    if (price < bbNow.lower) { score += 8;  signals.push({ label: 'Precio bajo Banda Inferior BB', type: 'bullish', pts: 8 }); }
    if (price > bbNow.upper) { score -= 8;  signals.push({ label: 'Precio sobre Banda Superior BB', type: 'bearish', pts: -8 }); }
    if (bbNow.width < 0.03)  { score += 3;  signals.push({ label: 'BB Squeeze — ruptura inminente', type: 'neutral', pts: 3 }); }
  }

  // Volume (±5)
  if (volume.spike && volume.bias === 'buyPressure')   { score += 5; signals.push({ label: 'Spike volumen comprador', type: 'bullish', pts: 5 }); }
  if (volume.spike && volume.bias === 'sellPressure')  { score -= 5; signals.push({ label: 'Spike volumen vendedor',  type: 'bearish', pts: -5 }); }

  // SMC (±10)
  if (smc.zone.type === 'bullish') { score += 5; signals.push({ label: smc.zone.label, type: 'bullish', pts: 5 }); }
  if (smc.zone.type === 'bearish') { score -= 5; signals.push({ label: smc.zone.label, type: 'bearish', pts: -5 }); }
  if (smc.bos?.type === 'bullish_bos') { score += 7; signals.push({ label: smc.bos.label, type: 'bullish', pts: 7 }); }
  if (smc.bos?.type === 'bearish_bos') { score -= 7; signals.push({ label: smc.bos.label, type: 'bearish', pts: -7 }); }
  if (smc.bos?.type === 'choch')       { score += smc.bos.label.includes('Alcista') ? 4 : -4; }

  // Patterns (±8 for top pattern)
  const bestPat = patterns[0];
  if (bestPat) {
    const pts = bestPat.type === 'bullish' ? 8 : bestPat.type === 'bearish' ? -8 : 0;
    score += pts;
    signals.push({ label: `${bestPat.name} (${bestPat.probability}%)`, type: bestPat.type, pts });
  }

  score = Math.max(0, Math.min(100, Math.round(score)));

  // Signal label
  let signal, signalCls;
  if      (score >= 75) { signal = 'COMPRA FUERTE';  signalCls = 'sig-strong-buy'; }
  else if (score >= 60) { signal = 'COMPRAR';         signalCls = 'sig-buy'; }
  else if (score >= 45) { signal = 'NEUTRAL';          signalCls = 'sig-neutral'; }
  else if (score >= 30) { signal = 'VENDER';           signalCls = 'sig-sell'; }
  else                  { signal = 'VENTA FUERTE';    signalCls = 'sig-strong-sell'; }

  // Key levels closest to price
  const support    = levels.filter(l => l.price < price && l.type === 'support').slice(-1)[0];
  const resistance = levels.filter(l => l.price > price && l.type === 'resistance').slice(0, 1)[0];

  // Risk/Reward suggestion
  const atrVal  = atrNow || price * 0.01;
  const sl      = score >= 50 ? price - atrVal * 1.5 : price + atrVal * 1.5;
  const tp      = score >= 50 ? price + atrVal * 3   : price - atrVal * 3;
  const rr      = Math.abs(tp - price) / Math.abs(sl - price);

  // ── Verdict & projection ─────────────────────────────────────────
  const verdict = computeVerdict({
    score, trend, patterns, smc, div, volume,
    indicators: { rsi: rsiNow, hist: histNow },
    price,
  });

  const brain = computeTradingBrain({
    score, trend, patterns, smc, div, volume, signals,
    indicators: { rsi: rsiNow, macd: macdNow, hist: histNow, bb: bbNow, atr: atrNow },
    price, symbol, tf, support, resistance,
    suggestedEntry: price, suggestedSL: sl, suggestedTP: tp,
  });

  return {
    symbol, tf, price, score, signal, signalCls,
    trend, patterns, smc, div, volume, levels,
    indicators: {
      rsi: rsiNow, macd: macdNow, signal_macd: sigNow, hist: histNow,
      ema20: ema20[n], ema50: ema50[n], ema200: ema200[n],
      bb: bbNow, atr: atrNow, stochK,
    },
    signals: signals.sort((a, b) => Math.abs(b.pts) - Math.abs(a.pts)),
    support, resistance,
    suggestedEntry: price,
    suggestedSL:    sl,
    suggestedTP:    tp,
    riskReward:     rr,
    verdict, brain,
    ts: Date.now(),
  };
}

// ═══════════════════════════════════════════════════════════════════
// FORMAT HELPERS
// ═══════════════════════════════════════════════════════════════════

function fmtPrice(n) {
  if (!n || isNaN(n)) return '—';
  if (n >= 10000) return n.toLocaleString('en', { maximumFractionDigits: 0 });
  if (n >= 100)   return n.toFixed(2);
  if (n >= 1)     return n.toFixed(4);
  return n.toFixed(6);
}

function fmtPct(n, digits = 2) {
  if (n == null || isNaN(n)) return '—';
  return (n >= 0 ? '+' : '') + n.toFixed(digits) + '%';
}

function fmtVol(n) {
  if (!n) return '—';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toFixed(0);
}

function scoreColor(s) {
  if (s >= 65) return '#00ff41';
  if (s >= 50) return '#e3b341';
  return '#f85149';
}

function typeColor(t) {
  if (t === 'bullish') return '#00ff41';
  if (t === 'bearish') return '#f85149';
  return '#e3b341';
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = String(s || '');
  return d.innerHTML;
}

// ═══════════════════════════════════════════════════════════════════
// SMART MONEY RULES — Apply tracked wallets' strategies
// ═══════════════════════════════════════════════════════════════════

async function renderSmartMoneyRules(symbol, currentPrice, marketData) {
  const section = document.getElementById('aSmartMoneySection');
  const content = document.getElementById('aSmartMoneyRules');
  if (!section || !content) return;

  // Guardar en cache para evitar spam a Hyperliquid
  if (!window._smcCache) window._smcCache = {};

  try {
    const rules = await WhaleTrackerAgent.getSmartMoneyRules(symbol, currentPrice, marketData);

    // Actualizar cache
    window._smcCache[`smc_${symbol}`] = { ts: Date.now() };
    
    if (!rules || rules.length === 0) {
      section.style.display = 'none';
      return;
    }

    section.style.display = '';
    
    // Renderizar cada regla
    content.innerHTML = rules.map((rule, i) => {
      const scoreColor = rule.score >= 70 ? '#00ff41' : rule.score >= 50 ? '#e3b341' : '#f85149';
      const signalColor = rule.signal === 'LONG' ? '#00ff41' : '#f85149';
      const signalIcon = rule.signal === 'LONG' ? '▲' : '▼';
      const sessionIcon = rule.sessionMatch ? '✓' : '⚠';
      const sessionColor = rule.sessionMatch ? '#00ff41' : '#e3b341';
      
      return `
        <div class="sm-rule-card" style="margin-bottom: 10px; background: rgba(255,255,255,.02); border: 1px solid rgba(255,255,255,.06); border-radius: 4px; padding: 10px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div>
              <div style="font-size: 8px; font-weight: 900; color: var(--text1);">${escHtml(rule.walletLabel)}</div>
              <div style="font-size: 7px; color: var(--text2); font-family: monospace; margin-top: 2px;">${rule.walletAddr.slice(0,8)}...${rule.walletAddr.slice(-6)}</div>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 11px; font-weight: 900; color: ${scoreColor};">${rule.score}</div>
              <div style="font-size: 7px; color: var(--text2);">Score</div>
            </div>
          </div>
          
          <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 8px;">
            <div style="background: rgba(0,212,255,.05); border: 1px solid rgba(0,212,255,.15); border-radius: 3px; padding: 6px;">
              <div style="font-size: 7px; color: var(--text2); margin-bottom: 3px;">SEÑAL</div>
              <div style="font-size: 10px; font-weight: 900; color: ${signalColor};">${signalIcon} ${rule.signal}</div>
              <div style="font-size: 7px; color: var(--text2); margin-top: 2px;">Confianza: ${rule.confidence}%</div>
            </div>
            
            <div style="background: rgba(0,212,255,.05); border: 1px solid rgba(0,212,255,.15); border-radius: 3px; padding: 6px;">
              <div style="font-size: 7px; color: var(--text2); margin-bottom: 3px;">SESIÓN</div>
              <div style="font-size: 8px; font-weight: 700; color: ${sessionColor};">${sessionIcon} ${rule.currentSession.toUpperCase()}</div>
              <div style="font-size: 7px; color: var(--text2); margin-top: 2px;">Prefer: ${rule.preferredSession}</div>
            </div>
          </div>

          <div style="background: rgba(0,212,255,.05); border: 1px solid rgba(0,212,255,.15); border-radius: 3px; padding: 6px; margin-bottom: 8px;">
            <div style="font-size: 7px; font-weight: 700; color: var(--text1); margin-bottom: 4px;">📍 ZONA DE ENTRADA SUGERIDA</div>
            <div style="display: flex; justify-content: space-between; font-size: 7.5px;">
              <span style="color: #f85149;">Bajo: $${fmtPrice(rule.entryZone.low)}</span>
              <span style="color: #e3b341;">Prom: $${fmtPrice(rule.entryZone.avg)}</span>
              <span style="color: #00ff41;">Alto: $${fmtPrice(rule.entryZone.high)}</span>
            </div>
            <div style="font-size: 7px; color: var(--text2); margin-top: 3px;">
              💎 Leverage: ${rule.leverage.toFixed(1)}x · Trades en ${symbol.replace('USDT','')}: ${rule.totalTrades}
            </div>
          </div>

          <div style="font-size: 7px; color: var(--text2);">
            <div style="margin-bottom: 2px;"><strong style="color: var(--accent);">Win Rate histórico:</strong> ${rule.winRate.toFixed(1)}%</div>
            ${rule.recentActivity && rule.recentActivity.length > 0 ? `
              <div style="margin-top: 4px; padding-top: 4px; border-top: 1px solid rgba(255,255,255,.05);">
                <strong style="color: var(--accent);">Últimas 3 operaciones:</strong><br>
                ${rule.recentActivity.slice(0,3).map(f => {
                  const dir = f.dir.includes('Long') ? '▲' : '▼';
                  const col = f.dir.includes('Long') ? '#00ff41' : '#f85149';
                  return `<span style="color: ${col};">${dir} $${fmtPrice(f.px)}</span>`;
                }).join(' · ')}
              </div>
            ` : ''}
          </div>
        </div>
      `;
    }).join('');

    // Actualizar contador en subtitle
    const sub = document.getElementById('aSmartMoneySub');
    if (sub) sub.textContent = `${rules.length} wallet${rules.length !== 1 ? 's' : ''} con señales activas`;

  } catch (e) {
    console.error('[SmartMoneyRules] Error:', e);
    section.style.display = 'none';
  }
}

// ═══════════════════════════════════════════════════════════════════
// RENDER ANALYSIS
// ═══════════════════════════════════════════════════════════════════

function renderAnalysis(a) {
  if (!a) return;
  lastAnalysis = a;

  // Header
  document.getElementById('aSymbol').textContent  = a.symbol;
  document.getElementById('aTf').textContent       = a.tf;
  document.getElementById('aPrice').textContent    = '$' + fmtPrice(a.price);
  document.getElementById('aLastUpdate').textContent = new Date(a.ts).toLocaleTimeString('es', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  // Score gauge
  const scoreEl = document.getElementById('aScore');
  const barEl   = document.getElementById('aScoreBar');
  const sigEl   = document.getElementById('aSignal');
  scoreEl.textContent = a.score;
  scoreEl.style.color = scoreColor(a.score);
  barEl.style.width   = a.score + '%';
  barEl.style.background = scoreColor(a.score);
  sigEl.textContent   = a.signal;
  sigEl.className     = 'a-signal ' + a.signalCls;

  // Smart Money Rules (async) — solo si hay wallets trackeadas, con cache de 5 min
  const smCacheKey = `smc_${a.symbol}`;
  const smCache = window._smcCache?.[smCacheKey];
  const smAge = smCache ? (Date.now() - smCache.ts) : Infinity;
  if (smAge > 300000) { // 5 minutos de cache
    renderSmartMoneyRules(a.symbol, a.price, a);
  }

  // Verdict
  const verdEl = document.getElementById('aVerdict');
  if (verdEl && a.verdict) {
    const v = a.verdict;
    const mainCol  = v.cls.includes('bull') ? '#00ff41' : v.cls.includes('bear') ? '#f85149' : '#e3b341';
    const bullW    = v.bullPct;
    const bearW    = v.bearPct;
    verdEl.innerHTML = `
      <div class="verd-header">
        <div class="verd-label ${v.cls}">${v.label}</div>
        <div class="verd-conf">Confianza <span style="color:${mainCol};font-weight:900">${v.confidence}%</span></div>
      </div>
      <div class="verd-prob-bar">
        <div class="verd-pb-bull" style="width:${bullW}%"></div>
        <div class="verd-pb-bear" style="width:${bearW}%"></div>
      </div>
      <div class="verd-prob-labels">
        <span style="color:#00ff41">▲ Alcista ${bullW}%</span>
        <span style="color:#f85149">▼ Bajista ${bearW}%</span>
      </div>
      <div class="verd-reasons">
        ${v.bullReasons.slice(0,4).map(r => `<div class="verd-reason verd-r-bull">▲ ${escHtml(r)}</div>`).join('')}
        ${v.bearReasons.slice(0,4).map(r => `<div class="verd-reason verd-r-bear">▼ ${escHtml(r)}</div>`).join('')}
      </div>
    `;
  }

  // Brain
  const brainEl = document.getElementById('aBrain');
  if (brainEl && a.brain) {
    const adv = window._lastAdvanced;
    if (adv) {
      // Preservar datos avanzados (MTF/harmónico/Elliott/ML) ya calculados
      // El setInterval 1s actualiza precio pero NO debe resetear el análisis avanzado
      renderBrain(brainEl, { ...a.brain, ...adv });
    } else {
      // Primera carga: mostrar placeholder hasta que runAdvancedBrain termine
      renderBrain(brainEl, { ...a.brain, _loading: true });
    }
  }
  
  // Bot Trades historial
  renderBotTrades();

  // Trend
  const trendEl = document.getElementById('aTrend');
  if (trendEl) {
    const t = a.trend;
    const col = t.score >= 4 ? '#00ff41' : t.score <= 2 ? '#f85149' : '#e3b341';
    trendEl.innerHTML = `
      <div class="tv-row"><span class="tv-k">Estructura</span><span class="tv-v" style="color:${col}">${t.trend}</span></div>
      <div class="tv-row"><span class="tv-k">EMA 20</span><span class="tv-v">$${fmtPrice(t.ema20)}</span></div>
      <div class="tv-row"><span class="tv-k">EMA 50</span><span class="tv-v">$${fmtPrice(t.ema50)}</span></div>
      <div class="tv-row"><span class="tv-k">EMA 200</span><span class="tv-v">$${fmtPrice(t.ema200)}</span></div>
      ${t.priceVsEma200 ? `<div class="tv-row"><span class="tv-k">vs EMA200</span><span class="tv-v" style="color:${parseFloat(t.priceVsEma200)>=0?'#00ff41':'#f85149'}">${fmtPct(parseFloat(t.priceVsEma200))}</span></div>` : ''}
      ${t.goldCross  ? '<div class="tv-badge tv-bull">● GOLDEN CROSS</div>' : ''}
      ${t.deathCross ? '<div class="tv-badge tv-bear">● DEATH CROSS</div>' : ''}
    `;
  }

  // Indicators
  const indEl = document.getElementById('aIndicators');
  if (indEl) {
    const ind = a.indicators;
    const rsiCol = ind.rsi > 70 ? '#f85149' : ind.rsi < 30 ? '#00ff41' : '#e3b341';
    const bbPos  = ind.bb?.lower && ind.bb?.upper
      ? ((a.price - ind.bb.lower) / (ind.bb.upper - ind.bb.lower) * 100).toFixed(0) : null;
    indEl.innerHTML = `
      <div class="tv-ind-grid">
        <div class="tv-ind-card">
          <div class="tv-ind-lbl">RSI (14)</div>
          <div class="tv-ind-val" style="color:${rsiCol}">${ind.rsi != null ? ind.rsi.toFixed(1) : '—'}</div>
          <div class="tv-ind-sub">${ind.rsi > 70 ? 'Sobrecomprado' : ind.rsi < 30 ? 'Sobrevendido' : 'Neutral'}</div>
        </div>
        <div class="tv-ind-card">
          <div class="tv-ind-lbl">MACD</div>
          <div class="tv-ind-val" style="color:${ind.hist > 0 ? '#00ff41' : '#f85149'}">${ind.hist != null ? ind.hist.toFixed(4) : '—'}</div>
          <div class="tv-ind-sub">${ind.hist > 0 ? 'Histograma ▲' : 'Histograma ▼'}</div>
        </div>
        <div class="tv-ind-card">
          <div class="tv-ind-lbl">Stoch K</div>
          <div class="tv-ind-val" style="color:${ind.stochK > 80 ? '#f85149' : ind.stochK < 20 ? '#00ff41' : '#e3b341'}">${ind.stochK ? ind.stochK.toFixed(0) : '—'}</div>
          <div class="tv-ind-sub">${ind.stochK > 80 ? 'Sobrecomprado' : ind.stochK < 20 ? 'Sobrevendido' : 'Neutral'}</div>
        </div>
        <div class="tv-ind-card">
          <div class="tv-ind-lbl">BB Width</div>
          <div class="tv-ind-val" style="color:${ind.bb?.width < 0.03 ? '#00d4ff' : '#aaa'}">${ind.bb?.width != null ? (ind.bb.width * 100).toFixed(2) + '%' : '—'}</div>
          <div class="tv-ind-sub">${ind.bb?.width < 0.03 ? '⚡ Squeeze' : 'Normal'}</div>
        </div>
        <div class="tv-ind-card">
          <div class="tv-ind-lbl">ATR (14)</div>
          <div class="tv-ind-val">$${fmtPrice(ind.atr)}</div>
          <div class="tv-ind-sub">${ind.atr ? (ind.atr / a.price * 100).toFixed(2) + '% del precio' : '—'}</div>
        </div>
        <div class="tv-ind-card">
          <div class="tv-ind-lbl">BB Posición</div>
          <div class="tv-ind-val" style="color:${bbPos > 80 ? '#f85149' : bbPos < 20 ? '#00ff41' : '#aaa'}">${bbPos != null ? bbPos + '%' : '—'}</div>
          <div class="tv-ind-sub">${bbPos > 80 ? 'Banda superior' : bbPos < 20 ? 'Banda inferior' : 'Zona media'}</div>
        </div>
      </div>
      ${a.div.type ? `<div class="tv-div-alert tv-div-${a.div.type}">⚡ ${escHtml(a.div.label)}<div class="tv-div-strength">Fuerza: ${a.div.strength.toFixed(0)}%</div></div>` : ''}
    `;
  }

  // Chart Patterns
  const patEl = document.getElementById('aPatterns');
  if (patEl) {
    if (!a.patterns.length) {
      patEl.innerHTML = '<div class="tv-empty">Sin patrones claros detectados en este timeframe</div>';
    } else {
      patEl.innerHTML = a.patterns.map(p => `
        <div class="tv-pattern tv-pat-${p.type}">
          <div class="tv-pat-top">
            <span class="tv-pat-name">${escHtml(p.name)}</span>
            <span class="tv-pat-prob" style="color:${typeColor(p.type)}">${p.probability}%</span>
            <span class="tv-pat-badge tv-pat-${p.type}">${p.type === 'bullish' ? '▲ ALCISTA' : p.type === 'bearish' ? '▼ BAJISTA' : '◆ NEUTRAL'}</span>
          </div>
          <div class="tv-pat-desc">${escHtml(p.description)}</div>
          ${p.target ? `<div class="tv-pat-target">Objetivo: <b>$${fmtPrice(p.target)}</b> · Neckline: $${fmtPrice(p.neckline)}</div>` : ''}
          <div class="tv-pat-prob-bar"><div class="tv-pat-prob-fill" style="width:${p.probability}%;background:${typeColor(p.type)}"></div></div>
        </div>
      `).join('');
    }
  }

  // SMC
  const smcEl = document.getElementById('aSMC');
  if (smcEl) {
    const s = a.smc;
    smcEl.innerHTML = `
      <div class="tv-smc-zone tv-smc-${s.zone.type}">
        <span class="tv-smc-zone-icon">${s.zone.type === 'bullish' ? '▼ DESCUENTO' : '▲ PREMIUM'}</span>
        <span class="tv-smc-zone-pct">${s.zone.pct}% del rango</span>
      </div>
      ${s.bos ? `<div class="tv-bos tv-bos-${s.bos.type.includes('bull') ? 'bull' : s.bos.type.includes('bear') ? 'bear' : 'choch'}">
        <span class="tv-bos-lbl">${escHtml(s.bos.label)}</span>
      </div>` : ''}
      ${s.fvgs.length ? `<div class="tv-smc-section-title">Fair Value Gaps</div>
        ${s.fvgs.map(g => `<div class="tv-fvg tv-fvg-${g.type}">
          <span>${escHtml(g.label)}</span>
        </div>`).join('')}` : ''}
      ${s.orderBlocks.length ? `<div class="tv-smc-section-title">Order Blocks activos</div>
        ${s.orderBlocks.map(ob => `<div class="tv-ob tv-ob-${ob.type}">
          <span>${escHtml(ob.label)}</span>
        </div>`).join('')}` : ''}
      <div class="tv-smc-section-title">Liquidez</div>
      <div class="tv-liq-row">
        ${s.swingHighs.map(h => `<span class="tv-liq tv-liq-high">⚡ $${fmtPrice(h)}</span>`).join('')}
        ${s.swingLows.map(l  => `<span class="tv-liq tv-liq-low">⚡ $${fmtPrice(l)}</span>`).join('')}
      </div>
    `;
  }

  // Signals breakdown
  const sigListEl = document.getElementById('aSignals');
  if (sigListEl) {
    sigListEl.innerHTML = a.signals.slice(0, 8).map(s => {
      const col = s.type === 'bullish' ? '#00ff41' : s.type === 'bearish' ? '#f85149' : '#e3b341';
      const pts = s.pts > 0 ? '+' + s.pts : s.pts;
      return `<div class="tv-sig-row">
        <span class="tv-sig-dot" style="background:${col}"></span>
        <span class="tv-sig-label">${escHtml(s.label)}</span>
        <span class="tv-sig-pts" style="color:${col}">${pts} pts</span>
      </div>`;
    }).join('');
  }

  // Key levels + R/R
  const levelsEl = document.getElementById('aLevels');
  if (levelsEl) {
    levelsEl.innerHTML = `
      <div class="tv-rr-row">
        <div class="tv-rr-card tv-rr-entry"><div class="tv-rr-lbl">ENTRADA</div><div class="tv-rr-val">$${fmtPrice(a.suggestedEntry)}</div></div>
        <div class="tv-rr-card tv-rr-sl"><div class="tv-rr-lbl">STOP LOSS</div><div class="tv-rr-val">$${fmtPrice(a.suggestedSL)}</div></div>
        <div class="tv-rr-card tv-rr-tp"><div class="tv-rr-lbl">TAKE PROFIT</div><div class="tv-rr-val">$${fmtPrice(a.suggestedTP)}</div></div>
        <div class="tv-rr-card tv-rr-rr"><div class="tv-rr-lbl">R:R</div><div class="tv-rr-val" style="color:${a.riskReward>=2?'#00ff41':'#e3b341'}">${a.riskReward.toFixed(1)}</div></div>
      </div>
      <div class="tv-levels-list">
        ${a.resistance ? `<div class="tv-level tv-res">
          <span class="tv-level-type">RESISTENCIA</span>
          <span class="tv-level-price">$${fmtPrice(a.resistance.price)}</span>
          <span class="tv-level-dist">${fmtPct((a.resistance.price - a.price)/a.price*100)}</span>
          <button class="btn-tv-line" data-price="${a.resistance.price}" data-label="Resistencia" data-color="#f5a623">→TV</button>
        </div>` : ''}
        <div class="tv-level tv-cur">
          <span class="tv-level-type">PRECIO</span>
          <span class="tv-level-price">$${fmtPrice(a.price)}</span>
          <span class="tv-level-dist">—</span>
          <button class="btn-tv-line" data-price="${a.price}" data-label="Precio" data-color="#ffffff">→TV</button>
        </div>
        ${a.support ? `<div class="tv-level tv-sup">
          <span class="tv-level-type">SOPORTE</span>
          <span class="tv-level-price">$${fmtPrice(a.support.price)}</span>
          <span class="tv-level-dist">${fmtPct((a.support.price - a.price)/a.price*100)}</span>
          <button class="btn-tv-line" data-price="${a.support.price}" data-label="Soporte" data-color="#00d4ff">→TV</button>
        </div>` : ''}
        <div class="tv-level tv-rr-row2">
          <button class="btn-tv-line btn-tv-tp" data-price="${a.suggestedTP}" data-label="TP" data-color="#00ff41">TP →TV</button>
          <button class="btn-tv-line btn-tv-sl" data-price="${a.suggestedSL}" data-label="SL" data-color="#f85149">SL →TV</button>
          <button class="btn-tv-all" id="drawLevelsBtn2">Todos →TV</button>
        </div>
      </div>
    `;
  }

  // Volume
  const volEl = document.getElementById('aVolume');
  if (volEl) {
    const v = a.volume;
    const col = v.bias === 'buyPressure' ? '#00ff41' : '#f85149';
    volEl.innerHTML = `
      <div class="tv-row"><span class="tv-k">Vol actual</span><span class="tv-v">${fmtVol(v.current)}</span></div>
      <div class="tv-row"><span class="tv-k">Vol promedio 20</span><span class="tv-v">${fmtVol(v.avg20)}</span></div>
      <div class="tv-row"><span class="tv-k">Ratio</span><span class="tv-v" style="color:${v.spike?'#e3b341':'#aaa'}">${v.ratio.toFixed(2)}x ${v.spike?'⚡':''}</span></div>
      <div class="tv-row"><span class="tv-k">CVD</span><span class="tv-v" style="color:${col}">${v.cvd > 0 ? '▲ Compradores' : '▼ Vendedores'}</span></div>
      <div class="tv-row"><span class="tv-k">Estado</span><span class="tv-v">${escHtml(v.label)}</span></div>
    `;
  }
}

// ═══════════════════════════════════════════════════════════════════
// DRAW LEVELS ON TRADINGVIEW CHART
// ═══════════════════════════════════════════════════════════════════

function sendToTV(levels) {
  if (!tvTabId) { showNotif('Abre TradingView en la pestaña activa', 'error'); return; }
  if (!candles.length) { showNotif('Analiza primero', 'error'); return; }

  // Pass a visible price range so the overlay can calibrate Y positions
  const slice    = candles.slice(-80);
  const visHigh  = Math.max(...slice.map(c => c.high));
  const visLow   = Math.min(...slice.map(c => c.low));

  chrome.tabs.sendMessage(tvTabId, {
    type: 'DRAW_LEVELS',
    levels,
    currentPrice,
    visibleHigh: visHigh,
    visibleLow:  visLow,
  }, () => {
    void chrome.runtime.lastError;
    showNotif(`${levels.length === 1 ? levels[0].label : 'Niveles'} dibujado en TV ✓`);
  });
}

function drawLevels() {
  if (!lastAnalysis) { showNotif('Analiza primero', 'error'); return; }
  const a = lastAnalysis;
  const levels = [];
  if (a.price)       levels.push({ price: +a.price.toFixed(8),            label: 'Precio',      color: '#ffffff' });
  if (a.suggestedTP) levels.push({ price: +a.suggestedTP.toFixed(8),      label: 'TP',          color: '#00ff41' });
  if (a.suggestedSL) levels.push({ price: +a.suggestedSL.toFixed(8),      label: 'SL',          color: '#f85149' });
  if (a.resistance)  levels.push({ price: +a.resistance.price.toFixed(8), label: 'Resistencia', color: '#f5a623' });
  if (a.support)     levels.push({ price: +a.support.price.toFixed(8),    label: 'Soporte',     color: '#00d4ff' });
  sendToTV(levels);
}

// ═══════════════════════════════════════════════════════════════════
// RENDER BRAIN
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// ML ANALYST — interpreta todos los datos y genera texto en lenguaje natural
// ═══════════════════════════════════════════════════════════════════

function generateMLAnalysis(b) {
  const ml   = b.mlResult;
  const adv  = b.advProb;
  const mtf  = b.mtf;
  const harm = b.harmonics?.[0];
  const ew   = b.elliott;

  if (!ml && !adv) return null;

  // ── Dirección y probabilidad combinada ──────────────────────────────
  const bullPct = adv?.bullPct ?? ml?.bullPct ?? 50;
  const bearPct = adv?.bearPct ?? ml?.bearPct ?? 50;
  const conf    = adv?.conf ?? 50;
  const direction = bullPct >= 58 ? 'ALCISTA' : bearPct >= 58 ? 'BAJISTA' : 'NEUTRAL';
  const dirCol    = direction === 'ALCISTA' ? '#00ff41' : direction === 'BAJISTA' ? '#f85149' : '#e3b341';
  const mainPct   = Math.max(bullPct, bearPct);

  // ── Interpretación del ML ───────────────────────────────────────────
  const mlLines = [];
  if (ml) {
    const mlDir = ml.bullPct >= 58 ? 'alcista' : ml.bearPct >= 58 ? 'bajista' : 'neutral';
    const mlStr = ml.bullPct >= 58 ? 'favorece el LONG' : ml.bearPct >= 58 ? 'favorece el SHORT' : 'no tiene sesgo claro (neutral)';
    mlLines.push(`El modelo ML (${ml.bullPct}% alcista) ${mlStr}.`);

    if (ml.importance) {
      const top = Object.entries(ml.importance)
        .sort((a, c) => Math.abs(c[1]) - Math.abs(a[1]))
        .slice(0, 3);
      const featDesc = {
        rsi_norm:    v => v < 0 ? 'RSI débil — vendedores dominan' : 'RSI fuerte — compradores activos',
        bb_pos:      v => v < 0 ? 'precio en zona baja BB' : 'precio en zona alta BB',
        bull_candle: v => v < 0 ? 'vela bajista — presión vendedora' : 'vela alcista — presión compradora',
        ema_bear:    v => v > 0 ? 'EMAs alineadas bajistas' : 'EMAs no bajistas',
        ema_bull:    v => v > 0 ? 'EMAs alineadas alcistas' : 'EMAs no alcistas',
        rsi_slope:   v => v < 0 ? 'RSI cayendo — momentum bajista' : 'RSI subiendo — momentum alcista',
        vol_ratio:   v => v > 0 ? 'volumen elevado — interés institucional' : 'volumen bajo',
        macd_bull:   v => v > 0 ? 'MACD positivo — impulso alcista' : 'MACD negativo',
        above_ema200:v => v > 0 ? 'precio sobre EMA200 — tendencia alcista de fondo' : 'precio bajo EMA200',
        stoch_os:    v => v > 0 ? 'Stoch sobrevendido — rebote posible' : '',
        stoch_ob:    v => v > 0 ? 'Stoch sobrecomprado — cuidado con corrección' : '',
      };
      const descs = top.map(([k, v]) => featDesc[k] ? featDesc[k](v) : k.replace(/_/g,' ')).filter(Boolean);
      if (descs.length) mlLines.push(`Drivers del modelo: ${descs.join('; ')}.`);
    }
  }

  // ── MTF ─────────────────────────────────────────────────────────────
  const mtfLines = [];
  if (mtf) {
    const mtfMap = {
      'ALCISTA ↑↑':  'Todos los marcos temporales alineados alcistas — mayor probabilidad de subida sostenida.',
      'BAJISTA ↓↓':  'Todos los marcos temporales alineados bajistas — presión vendedora en todas las escalas.',
      'ALCISTA ↑~':  'Marcos superiores alcistas pero con divergencia — tendencia alcista sin confirmación plena.',
      'BAJISTA ↓~':  'Marcos superiores bajistas con algo de ruido — sesgo bajista moderado.',
      'DIVERGENTE ↕': 'Marcos temporales en conflicto — señal reducida, esperar alineación antes de entrar.',
    };
    mtfLines.push(mtfMap[mtf.align] || `MTF: ${mtf.align}`);
    if (mtf.prim?.rsi) {
      const rsiTxt = mtf.prim.rsi > 60 ? 'sobrecomprado' : mtf.prim.rsi < 40 ? 'sobrevendido' : 'neutral';
      mtfLines.push(`TF primario (${mtf.primaryTf}): RSI ${mtf.prim.rsi} — ${rsiTxt}.`);
    }
  }

  // ── Harmonic ────────────────────────────────────────────────────────
  const harmLines = [];
  if (harm) {
    const hDir = harm.type === 'bullish' ? 'alcista (señal de compra)' : 'bajista (señal de venta)';
    if (harm.stage === 'COMPLETANDO') {
      harmLines.push(`⚠ Patrón armónico ${harm.name} ${hDir} COMPLETÁNDOSE — zona PRZ $${fmtPrice(harm.dZone?.low)}–$${fmtPrice(harm.dZone?.high)}. Alta probabilidad de reversión inmediata.`);
    } else if (harm.stage === 'APPROACHANDO PRZ') {
      harmLines.push(`Patrón ${harm.name} ${hDir} acercándose al PRZ (${harm.distPct}% de distancia). Si el precio llega a la zona, esperar señal de inversión.`);
    } else {
      harmLines.push(`${harm.name} ${hDir} en formación — todavía lejos del PRZ.`);
    }
  }

  // ── Elliott ─────────────────────────────────────────────────────────
  const ewLines = [];
  if (ew && ew.currentWave) {
    const ewMap = {
      'W3': w => `Onda ${w} activa — ola más poderosa del ciclo Elliott. ${ew.dir === 'bullish' ? 'Impulso alcista fuerte, no ir en contra.' : 'Caída acelerada, zona de alto riesgo para longs.'}`,
      'W5': w => `Onda ${w} — ola final del impulso. Atención: tras W5 viene una corrección ABC significativa.`,
      'W4': w => `Onda ${w} — corrección en marcha. Esperar fin de corrección antes de entrar en la dirección del impulso.`,
      'W2': w => `Onda ${w} — primer retroceso después del impulso inicial. Zona de entrada con buen R:R si se mantiene por encima de W1.`,
    };
    const wNum = ew.currentWave.match(/W(\d)/)?.[1];
    const desc = wNum && ewMap[`W${wNum}`] ? ewMap[`W${wNum}`](ew.currentWave) : `${ew.currentWave} detectada.`;
    ewLines.push(desc);
    if (ew.nextTarget) ewLines.push(`Objetivo Elliott: $${fmtPrice(ew.nextTarget)} con conf. ${ew.confidence}%.`);
  }

  // ── Escenarios ──────────────────────────────────────────────────────
  const mainCond = bullPct > bearPct
    ? (b.bullConditions?.[0] || 'mantiene estructura alcista')
    : (b.bearConditions?.[0] || 'rompe soporte clave');
  const altPct = Math.min(bullPct, bearPct);
  const mainLabel = bullPct > bearPct ? `LONG ${bullPct}%` : `SHORT ${bearPct}%`;
  const altLabel  = bullPct > bearPct ? `SHORT ${bearPct}%` : `LONG ${bullPct}%`;
  const altCond   = bullPct > bearPct
    ? 'MTF gira bajista o ML cae bajo 40%'
    : 'MTF gira alcista o precio recupera soporte';

  // ── Recomendación de acción ──────────────────────────────────────────
  let action;
  if (conf >= 75 && direction !== 'NEUTRAL') {
    action = `Alta confianza — el setup está ${direction === 'ALCISTA' ? 'favorable para LONG' : 'favorable para SHORT'} pero confirmar con ruptura/volumen.`;
  } else if (direction === 'NEUTRAL' || conf < 50) {
    action = 'Zona de indecisión — mantener posición flat hasta que algún factor confirme dirección.';
  } else {
    action = `Sesgo ${direction.toLowerCase()} con confianza moderada (${conf}%). Usar R:R mínimo 2:1 y stop ajustado.`;
  }

  return { direction, dirCol, mainPct, conf, mlLines, mtfLines, harmLines, ewLines, mainLabel, altLabel, mainCond, altCond, altPct, action };
}

function renderBrain(el, b) {
  // ── Fix race condition: calcular ML aquí mismo si el modelo ya está disponible ──
  // Esto elimina por completo la race condition entre mlLoadModel y runAdvancedBrain
  let mlResult = b.mlResult;
  if (!mlResult && _mlModel && candles.length >= 200) {
    try { mlResult = mlInfer(candles); } catch {}
  }
  b = { ...b, mlResult };

  // Si tenemos ML pero el advProb fue calculado sin él, recalcular
  if (mlResult && b.advProb && !b.advProb.factors?.some(f => f.includes('ML'))) {
    try {
      b.advProb = computeAdvancedProb({
        score: b.bullPct, mlResult,
        mtf: b.mtf, harmonics: b.harmonics, elliott: b.elliott,
        div: null, volume: null, patterns: null,
      });
    } catch {}
  }

  // ── MTF panel ────────────────────────────────────────────────────
  const mtfHtml = b.mtf ? (() => {
    const m = b.mtf;
    const dirIcon = d => d?.includes('bullish')||d==='neutral_up' ? '▲' : d?.includes('bearish')||d==='neutral_down' ? '▼' : '→';
    const dirCol  = d => d?.includes('bullish')||d==='neutral_up' ? '#00ff41' : d?.includes('bearish')||d==='neutral_down' ? '#f85149' : '#e3b341';
    const alignCol = a => a?.includes('ALCISTA') ? '#00ff41' : a?.includes('BAJISTA') ? '#f85149' : '#e3b341';
    return `<div class="brain-mtf">
      <div class="brain-mtf-title">
        📊 MULTI-TIMEFRAME · <span style="color:${alignCol(m.align)}">${m.align}</span>
        ${m._localOnly ? '<span class="brain-mtf-local">(estimado · sin fetch externo)</span>' : ''}
      </div>
      <div class="brain-mtf-row">
        <div class="brain-mtf-cell">
          <span class="brain-mtf-tf">${m.primaryTf} (primaria)</span>
          <span class="brain-mtf-dir" style="color:${dirCol(m.prim?.dir)}">${dirIcon(m.prim?.dir)} ${(m.prim?.dir||'—').replace('_',' ')}</span>
          <span class="brain-mtf-rsi">RSI ${m.prim?.rsi||'—'}</span>
          ${m.prim?.res ? `<span class="brain-mtf-res">R $${fmtPrice(m.prim.res)}</span>` : ''}
          ${m.prim?.sup ? `<span class="brain-mtf-sup">S $${fmtPrice(m.prim.sup)}</span>` : ''}
        </div>
        <div class="brain-mtf-cell">
          <span class="brain-mtf-tf">${m.secondaryTf} (secundaria)</span>
          <span class="brain-mtf-dir" style="color:${dirCol(m.sec?.dir)}">${dirIcon(m.sec?.dir)} ${(m.sec?.dir||'—').replace('_',' ')}</span>
          <span class="brain-mtf-rsi">RSI ${m.sec?.rsi||'—'}</span>
          ${m.sec?.res ? `<span class="brain-mtf-res">R $${fmtPrice(m.sec.res)}</span>` : ''}
          ${m.sec?.sup ? `<span class="brain-mtf-sup">S $${fmtPrice(m.sec.sup)}</span>` : ''}
        </div>
        <div class="brain-mtf-cell brain-mtf-current">
          <span class="brain-mtf-tf">${m.currentTf} (actual)</span>
          <span class="brain-mtf-dir" style="color:${b.modeCol}">${b.modeIcon}</span>
          <span class="brain-mtf-rsi">Score ${b.finalConf||'—'}%</span>
          ${lastAnalysis?.resistance?.price ? `<span class="brain-mtf-res">R $${fmtPrice(lastAnalysis.resistance.price)}</span>` : ''}
          ${lastAnalysis?.support?.price ? `<span class="brain-mtf-sup">S $${fmtPrice(lastAnalysis.support.price)}</span>` : ''}
        </div>
      </div>
    </div>`;
  })() : '';

  // ── Harmonic patterns panel ───────────────────────────────────────
  const harmonicHtml = b.harmonics?.length ? `
    <div class="brain-harmonic">
      <div class="brain-harmonic-title">🔺 PATRONES ARMÓNICOS</div>
      ${b.harmonics.map(h => {
        const col = h.type === 'bullish' ? '#00ff41' : '#f85149';
        const stageCol = h.stage === 'COMPLETANDO' ? '#ff4500' : h.stage === 'APPROACHANDO PRZ' ? '#e3b341' : '#888';
        return `<div class="brain-harm-row">
          <div class="brain-harm-top">
            <span class="brain-harm-name" style="color:${col}">${h.name}</span>
            <span class="brain-harm-type">${h.type === 'bullish' ? '▲ ALCISTA' : '▼ BAJISTA'}</span>
            <span class="brain-harm-stage" style="color:${stageCol}">${h.stage}</span>
            <span class="brain-harm-prob" style="color:${col}">${h.prob}%</span>
          </div>
          <div class="brain-harm-details">
            <span class="brain-harm-d">PRZ: $${fmtPrice(h.dZone.low)}–$${fmtPrice(h.dZone.high)}</span>
            <span class="brain-harm-sl">SL: $${fmtPrice(h.sl)}</span>
            <span class="brain-harm-tp">TP1: $${fmtPrice(h.tp1)}</span>
            <span class="brain-harm-tp">TP2: $${fmtPrice(h.tp2)}</span>
            <span class="brain-harm-dist">${h.distPct}% del PRZ</span>
          </div>
          <div class="brain-harm-ratios">AB/XA ${h.ratios['AB/XA']} · BC/AB ${h.ratios['BC/AB']}</div>
        </div>`;
      }).join('')}
    </div>` : '';

  // ── Elliott Wave panel ────────────────────────────────────────────
  const elliottHtml = b.elliott ? (() => {
    const e = b.elliott;
    const col = e.dir === 'bullish' ? '#00ff41' : '#f85149';
    const waveNums = ['1','2','3','4','5'];
    const curWave = e.currentWave?.includes('W') ? e.currentWave.match(/W(\d)/)?.[1] : null;
    const waveDots = waveNums.map(n => {
      const active = curWave === n;
      return `<span class="brain-wave-dot ${active ? 'active' : ''}" style="${active ? `color:${col};font-weight:900` : ''}">${n}</span>`;
    }).join('<span class="brain-wave-sep">–</span>');
    return `<div class="brain-elliott">
      <div class="brain-elliott-title">🌊 ONDAS DE ELLIOTT · <span style="color:${col}">${e.type.toUpperCase()} ${e.dir === 'bullish' ? '▲' : '▼'}</span></div>
      <div class="brain-wave-track">${waveDots}<span class="brain-wave-label" style="color:${col}"> ${e.currentWave}</span></div>
      <div class="brain-elliott-next">${escHtml(e.nextMove)}</div>
      ${e.nextTarget ? `<div class="brain-elliott-target">Objetivo: <strong style="color:${col}">$${fmtPrice(e.nextTarget)}</strong></div>` : ''}
      <div class="brain-elliott-ratios">
        ${e.w2Ret ? `W2 ret. ${e.w2Ret}% · ` : ''}
        ${e.w3vsW1 ? `W3/W1 ${e.w3vsW1}x · ` : ''}
        ${e.w4Ret && parseFloat(e.w4Ret) < 100 ? `W4 ret. ${e.w4Ret}%` : ''}
        <span class="brain-elliott-conf">Conf. ${e.confidence}%</span>
      </div>
    </div>`;
  })() : `<div class="brain-elliott brain-elliott-none">
    <div class="brain-elliott-title" style="opacity:.5">🌊 ONDAS DE ELLIOTT</div>
    <div style="font-size:8.5px;color:var(--text2);padding:3px 0">Sin estructura de ondas clara en este TF/momento — el mercado puede estar en consolidación lateral</div>
  </div>`;

  // ── Advanced probability ──────────────────────────────────────────
  // ── ML Score block ────────────────────────────────────────────────
  mlUpdateHeaderBadge();
  const mlHtml = b.mlResult ? (() => {
    const ml = b.mlResult;
    const col = ml.bullPct >= 55 ? '#00ff41' : ml.bearPct >= 55 ? '#f85149' : '#e3b341';
    const topImp = b.mlResult.importance
      ? Object.entries(b.mlResult.importance)
          .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
          .slice(0, 5)
          .map(([k, v]) => `<span class="ml-feat ${v >= 0 ? 'ml-bull' : 'ml-bear'}">${v >= 0 ? '+' : ''}${v.toFixed(2)} ${k.replace(/_/g,' ')}</span>`)
          .join('')
      : '';
    return `<div class="brain-ml-block">
      <div class="brain-ml-title">🤖 ML Score <span class="brain-ml-auc">AUC ${ml.auc} · ${ml.samples?.toLocaleString() || '?'} muestras</span></div>
      <div class="brain-ml-bar-row">
        <span style="color:#00ff41;font-weight:900">▲ ${ml.bullPct}%</span>
        <div class="brain-prob-bar">
          <div class="brain-pb-bull" style="width:${ml.bullPct}%"></div>
          <div class="brain-pb-bear" style="width:${ml.bearPct}%"></div>
        </div>
        <span style="color:#f85149;font-weight:900">${ml.bearPct}% ▼</span>
        <span class="brain-ml-signal" style="color:${col}">
          ${ml.bullPct >= 60 ? '▲ LONG' : ml.bearPct >= 60 ? '▼ SHORT' : '→ NEUTRAL'}
        </span>
      </div>
      <div class="brain-ml-features">${topImp}</div>
    </div>`;
  })() : `<div class="brain-ml-block brain-ml-empty">
      <span class="brain-ml-icon">🤖</span>
      <span class="brain-ml-msg">Sin modelo ML — pulsa <strong>🤖 Sin ML</strong> en el header para importar model_weights.json</span>
    </div>`;

  // ── Probabilidad combinada ────────────────────────────────────────
  const advProbHtml = b.advProb ? (() => {
    const ap = b.advProb;
    const confCol = ap.conf >= 75 ? '#00ff41' : ap.conf >= 55 ? '#e3b341' : '#888';
    return `<div class="brain-adv-prob">
      <div class="brain-adv-prob-row">
        <span style="color:#00ff41;font-weight:900">▲ ${ap.bullPct}%</span>
        <div class="brain-prob-bar">
          <div class="brain-pb-bull" style="width:${ap.bullPct}%"></div>
          <div class="brain-pb-bear" style="width:${ap.bearPct}%"></div>
        </div>
        <span style="color:#f85149;font-weight:900">${ap.bearPct}% ▼</span>
        <span class="brain-adv-conf" style="color:${confCol}">Conf. ${ap.conf}%</span>
      </div>
      <div class="brain-adv-factors">
        ${ap.factors.map(f => `<span class="brain-factor">${escHtml(f)}</span>`).join('')}
      </div>
    </div>`;
  })() : `<div class="brain-prob-row">
      <span style="color:#00ff41;font-weight:900">▲ ${b.bullPct}%</span>
      <div class="brain-prob-bar">
        <div class="brain-pb-bull" style="width:${b.bullPct}%"></div>
        <div class="brain-pb-bear" style="width:${b.bearPct}%"></div>
      </div>
      <span style="color:#f85149;font-weight:900">${b.bearPct}% ▼</span>
    </div>`;

  // ── ML Analyst — interpretación en lenguaje natural ──────────────────
  const ana = !b._loading ? (() => { try { return generateMLAnalysis(b); } catch(e) { console.warn('[Analyst]', e); return null; } })() : null;
  const analystHtml = ana ? (() => {
    const lines = [...ana.mlLines, ...ana.mtfLines, ...ana.harmLines, ...ana.ewLines];
    return `<div class="brain-analyst-block">
      <div class="brain-analyst-hdr">
        <span class="brain-analyst-title">🧠 ML ANALYST</span>
        <span class="brain-analyst-verdict" style="color:${ana.dirCol}">${ana.direction} ${ana.mainPct}%</span>
        <span class="brain-analyst-conf">Conf. ${ana.conf}%</span>
      </div>
      <div class="brain-analyst-lines">
        ${lines.map(l => `<div class="ba-line">${escHtml(l)}</div>`).join('')}
      </div>
      <div class="brain-analyst-scenarios">
        <div class="ba-scenario ba-sce-main">▶ Principal (${ana.mainLabel}): ${escHtml(ana.mainCond)}</div>
        <div class="ba-scenario ba-sce-alt">◀ Alternativo (${ana.altLabel}): ${escHtml(ana.altCond)}</div>
      </div>
      <div class="brain-analyst-action">${escHtml(ana.action)}</div>
    </div>`;
  })() : '';

  // ── Loading placeholder: solo durante el primer render (antes del async) ────
  // b._loading indica que aún NO ha corrido runAdvancedBrain
  const loadingPlaceholder = b._loading
    ? `<div class="brain-loading-adv">⏳ Calculando patrones y análisis avanzado…</div>`
    : '';

  // ── Alert + Mode + Narrative (unchanged) ─────────────────────────
  const alertHtml = b.breakoutAlert ? `
    <div class="brain-alert brain-alert-${b.breakoutAlert.type}">
      <span class="brain-alert-icon">${b.breakoutAlert.type === 'bull' ? '🟢' : '🔴'}</span>
      <span>${escHtml(b.breakoutAlert.msg)}</span>
    </div>` : '';

  // ── GATE: Trading Brain signal only fires when a portfolio strategy is active ──
  {
    const ps = window._portStatus;
    const hasPf = ps && ps.tf === currentTf && ps.total > 0;
    if (!hasPf) {
      b = { ...b, finalDir: 'ESPERAR', finalConf: null,
        finalReason: 'Sin estrategias en portafolio \u2014 ve a \ud83d\udcca Backtest \u2192 \ud83e\uddec F\u00e1brica+ y guarda una con \ud83d\udccc',
        waitFor: ['Guarda al menos una estrategia en el portafolio para activar las se\u00f1ales del Trading Brain'] };
    } else if (!ps.active.length) {
      const names = ps.waiting.map(s => (_BT_SIG_LBL[s.sig]||s.sig)).join(', ');
      b = { ...b, finalDir: 'ESPERAR', finalConf: null,
        finalReason: `Portafolio activo (${ps.total} estrategia${ps.total>1?'s':''}) \u2014 ninguna disparando ahora`,
        waitFor: [`Esperando se\u00f1al de: ${names}`] };
    } else {
      const longs  = ps.active.filter(s => s.dir === 'LONG');
      const shorts = ps.active.filter(s => s.dir === 'SHORT');
      const portDir = longs.length >= shorts.length ? 'LONG' : 'SHORT';
      const portSig = (portDir === 'LONG' ? longs : shorts)[0];
      b = { ...b, finalDir: portDir,
        finalConf: Math.min(85, 60 + (portSig.filt !== 'none' ? 6 : 0)),
        finalReason: `[Portafolio] ${_BT_SIG_LBL[portSig.sig]||portSig.sig} ${_BT_FILT_LBL[portSig.filt]||portSig.filt} (WR:${portSig.wr}% \u00b7 ${portSig.trades}t)` };
    }
  }

  // \u2500\u2500 SE\u00d1AL DE TRADING: ENTRAR O ESPERAR \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  const tradingSignalHtml = (() => {
    const isLong = b.finalDir === 'LONG';
    const isShort = b.finalDir === 'SHORT';
    const isWait = b.finalDir === 'ESPERAR';
    
    if (isWait || !b.finalConf) {
      return `<div class="brain-signal brain-signal-wait">
        <div class="brain-signal-verdict">⏸ NO ENTRAR TODAVÍA — ESPERAR</div>
        <div class="brain-signal-reason">${escHtml(b.finalReason || 'Mercado sin edge claro')}</div>
        <div class="brain-signal-waitfor">
          <div class="brain-signal-waitfor-title">💡 Qué esperar:</div>
          ${b.waitFor.slice(0,2).map(w => `<div class="brain-signal-waitfor-item">→ ${escHtml(w)}</div>`).join('')}
        </div>
      </div>`;
    }

    // LONG or SHORT signal
    const col = isLong ? '#00ff41' : '#f85149';
    const icon = isLong ? '🟢 ▲' : '🔴 ▼';
    const label = isLong ? 'LONG' : 'SHORT';
    const entry = isLong ? b.bEntry : b.sEntry;
    const sl = isLong ? b.bSL : b.sSL;
    const tp1 = isLong ? b.bTP1 : b.sTP1;
    const tp2 = isLong ? b.bTP2 : b.sTP2;
    const rr = isLong ? b.bRR : b.sRR;
    
    // Determinar si es AHORA o ESPERAR ENTRADA
    const currentPrice = lastAnalysis?.price || 0;
    const isNowEntry = isLong 
      ? (currentPrice <= entry * 1.002) // Precio cerca de entrada LONG
      : (currentPrice >= entry * 0.998); // Precio cerca de entrada SHORT
    
    const verdictText = isNowEntry
      ? `✅ ENTRAR ${label} AHORA`
      : `⏳ ESPERAR ${label} en $${fmtPrice(entry)}`;
    
    const confCol = b.finalConf >= 75 ? col : b.finalConf >= 60 ? '#e3b341' : '#888';
    
    // ── AUTO-GUARDAR BOT TRADE ────────────────────────────────────────
    // Solo guardar si hay señal de portafolio activa (no solo análisis técnico)
    const _ps = window._portStatus;
    const _portFired = _ps && _ps.tf === currentTf && _ps.active.length > 0;
    if (b.finalConf >= 55 && _portFired) {
      const fiveMinAgo = Date.now() - 5 * 60 * 1000;
      const existingTrade = botTrades.find(t => 
        t.symbol === currentSymbol &&
        t.direction === label &&
        t.timestamp >= fiveMinAgo &&
        Math.abs(t.entry - entry) / entry < 0.01 // ±1% de diferencia en precio
      );
      
      if (!existingTrade) {
        createBotTrade({
          symbol: currentSymbol,
          direction: label,
          entry,
          sl,
          tp1,
          tp2,
          rr,
          confidence: b.finalConf,
          reason: b.finalReason,
        });
      }
    }
    
    return `<div class="brain-signal brain-signal-${isLong ? 'long' : 'short'}" style="border-color:${col}60;background:${col}0a">
      <div class="brain-signal-verdict" style="color:${col}">
        <span class="brain-signal-icon">${icon}</span>
        <span class="brain-signal-text">${verdictText}</span>
        <span class="brain-signal-conf" style="color:${confCol}">Confianza ${b.finalConf}%</span>
      </div>
      <div class="brain-signal-reason">${escHtml(b.finalReason)}</div>
      <div class="brain-signal-prices">
        <div class="brain-signal-price-row">
          <span class="brain-sp-label">ENTRADA:</span>
          <span class="brain-sp-val" style="color:${col}">$${fmtPrice(entry)}</span>
        </div>
        <div class="brain-signal-price-row">
          <span class="brain-sp-label">SL:</span>
          <span class="brain-sp-val" style="color:#f85149">$${fmtPrice(sl)}</span>
          <span class="brain-sp-risk">${isLong ? `-${((1 - sl/entry) * 100).toFixed(2)}%` : `-${((sl/entry - 1) * 100).toFixed(2)}%`}</span>
        </div>
        <div class="brain-signal-price-row">
          <span class="brain-sp-label">TP1:</span>
          <span class="brain-sp-val" style="color:#00ff41">$${fmtPrice(tp1)}</span>
          <span class="brain-sp-profit">${isLong ? `+${((tp1/entry - 1) * 100).toFixed(2)}%` : `+${((1 - tp1/entry) * 100).toFixed(2)}%`}</span>
        </div>
        <div class="brain-signal-price-row">
          <span class="brain-sp-label">TP2:</span>
          <span class="brain-sp-val" style="color:#00ff41">$${fmtPrice(tp2)}</span>
          <span class="brain-sp-profit">${isLong ? `+${((tp2/entry - 1) * 100).toFixed(2)}%` : `+${((1 - tp2/entry) * 100).toFixed(2)}%`}</span>
        </div>
        <div class="brain-signal-price-row brain-signal-rr">
          <span class="brain-sp-label">Risk:Reward</span>
          <span class="brain-sp-val" style="color:${rr >= 2 ? '#00ff41' : rr > 0 ? '#e3b341' : '#888'}">${rr > 0 ? `${rr}:1` : '—'}</span>
        </div>
      </div>
    </div>`;
  })();

  el.innerHTML = `
    ${alertHtml}

    <!-- SEÑAL DE TRADING: DECISIÓN CLARA -->
    ${tradingSignalHtml}

    <!-- Portfolio detail: conditions + projected levels for each strategy -->
    ${(() => {
      const ps = window._portStatus;
      if (!ps || !ps.total) return '';
      const fp = v => fmtPrice(v);
      const renderLevels = (lv, dir) => {
        const isL = dir === 'LONG';
        const col = isL ? '#00ff41' : '#f85149';
        const slPct  = isL ? `-${((1 - lv.sl/lv.entry)*100).toFixed(2)}%` : `-${((lv.sl/lv.entry-1)*100).toFixed(2)}%`;
        const tp1Pct = isL ? `+${((lv.tp1/lv.entry-1)*100).toFixed(2)}%` : `+${((1-lv.tp1/lv.entry)*100).toFixed(2)}%`;
        const tp2Pct = isL ? `+${((lv.tp2/lv.entry-1)*100).toFixed(2)}%` : `+${((1-lv.tp2/lv.entry)*100).toFixed(2)}%`;
        return `<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:2px;margin-top:4px;background:${col}10;border-radius:4px;padding:4px">
          <div style="text-align:center"><div style="font-size:8px;color:var(--text3)">ENTRADA</div><div style="font-size:10px;color:${col};font-weight:700">$${fp(lv.entry)}</div></div>
          <div style="text-align:center"><div style="font-size:8px;color:var(--text3)">SL</div><div style="font-size:10px;color:#f85149;font-weight:700">$${fp(lv.sl)}</div><div style="font-size:8px;color:#f85149">${slPct}</div></div>
          <div style="text-align:center"><div style="font-size:8px;color:var(--text3)">TP1</div><div style="font-size:10px;color:#00ff41;font-weight:700">$${fp(lv.tp1)}</div><div style="font-size:8px;color:#00ff41">${tp1Pct}</div></div>
          <div style="text-align:center"><div style="font-size:8px;color:var(--text3)">TP2</div><div style="font-size:10px;color:#00ff41;font-weight:700">$${fp(lv.tp2)}</div><div style="font-size:8px;color:#00ff41">${tp2Pct}</div></div>
        </div>
        <div style="font-size:8px;color:var(--text3);margin-top:2px;text-align:right">R:R ${lv.rr}:1</div>`;
      };

      const allStrats = [...ps.active, ...ps.waiting];
      const rows = allStrats.map(s => {
        const isActive = s.active;
        const sigCond  = _BT_SIG_COND[s.sig] || '—';
        const filtCond = _BT_FILT_COND[s.filt] || null;
        const condText = filtCond ? `${sigCond} · ${filtCond}` : sigCond;
        const lv = s._levels;

        if (isActive) {
          return `<div style="border:1px solid rgba(0,255,65,.3);border-radius:6px;padding:6px 8px;margin-bottom:6px;background:rgba(0,255,65,.04)">
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">
              <span style="color:${s.dir==='LONG'?'#00ff41':'#f85149'};font-weight:700;font-size:11px">${s.dir==='LONG'?'▲ LONG':'▼ SHORT'}</span>
              <span style="color:var(--text1);font-size:10px;font-weight:600">${_BT_SIG_LBL[s.sig]||s.sig}</span>
              <span style="color:var(--text3);font-size:9px">${_BT_FILT_LBL[s.filt]||s.filt}</span>
              <span style="margin-left:auto;color:#e3b341;font-size:9px">WR:${s.wr}% · ${s.trades}t</span>
              <span style="color:#00ff41;font-size:9px;font-weight:700">🟢 SEÑAL ACTIVA</span>
              <button class="strat-info-btn" data-sig="${s.sig}" data-filt="${s.filt}" data-wr="${s.wr}" data-trades="${s.trades}" style="background:rgba(0,212,255,.15);border:1px solid rgba(0,212,255,.4);border-radius:3px;color:var(--accent);cursor:pointer;font-size:9px;padding:1px 5px">ℹ️</button>
            </div>
            ${renderLevels(lv, s.dir)}
          </div>`;
        } else {
          return `<div style="border:1px solid var(--border);border-radius:6px;padding:6px 8px;margin-bottom:6px;opacity:.9">
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">
              <span style="color:var(--text2);font-size:10px;font-weight:600">${_BT_SIG_LBL[s.sig]||s.sig}</span>
              <span style="color:var(--text3);font-size:9px">${_BT_FILT_LBL[s.filt]||s.filt}</span>
              <span style="margin-left:auto;color:var(--text3);font-size:9px">WR:${s.wr}% · ${s.trades}t</span>
              <span style="color:#e3b341;font-size:9px">⏳ EN ESPERA</span>
              <button class="strat-info-btn" data-sig="${s.sig}" data-filt="${s.filt}" data-wr="${s.wr}" data-trades="${s.trades}" style="background:rgba(0,212,255,.15);border:1px solid rgba(0,212,255,.4);border-radius:3px;color:var(--accent);cursor:pointer;font-size:9px;padding:1px 5px">ℹ️</button>
            </div>
            <div style="font-size:9px;color:var(--text3);margin-bottom:4px;line-height:1.4">
              <span style="color:#e3b341">⚡ Condición:</span> ${escHtml(condText)}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:5px" data-strat-sig="${s.sig}" data-strat-filt="${s.filt}">
              ${[{dir:'LONG',col:'#00ff41',pct:(s._dist?.longPct??0),txt:(s._dist?.long??'—')},{dir:'SHORT',col:'#f85149',pct:(s._dist?.shortPct??0),txt:(s._dist?.short??'—')}].map(({dir,col,pct,txt})=>{const bc=pct>=80?col:pct>=50?'#f0883e':'#e3b341';return `<div style="background:${col}0d;border-radius:4px;padding:4px 6px" data-dir="${dir}"><div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:2px"><span style="font-size:8px;color:${col};font-weight:700">${dir==='LONG'?'▲ LONG':'▼ SHORT'}</span><span class="strat-pct-val" style="font-size:14px;font-weight:900;color:${bc};line-height:1">${pct}%</span></div><div style="height:5px;background:rgba(255,255,255,.12);border-radius:3px;overflow:hidden;margin-bottom:3px"><div class="strat-pct-bar" style="height:100%;width:${pct}%;background:${bc};border-radius:3px;transition:width .4s"></div></div><div class="strat-dist-txt" style="font-size:8px;color:${txt.startsWith('✅')?col:'var(--text3)'};line-height:1.4">${escHtml(txt)}</div></div>`;}).join('')}
            </div>
            <div style="border-top:1px solid rgba(255,255,255,.07);margin-top:4px;padding-top:4px">
              <div style="font-size:8px;color:var(--text3);margin-bottom:3px">📍 Niveles estimados si dispara cerca del precio actual:</div>
              ${renderLevels(lv.long, 'LONG')}
              ${renderLevels(lv.short, 'SHORT')}
            </div>
          </div>`;
        }
      });
      return rows.join('');
    })()}
  `;

  // Wire pine copy button
  document.getElementById('brainPineBtn')?.addEventListener('click', () => {
    if (tvTabId && lastAnalysis?.brain?.pine) {
      chrome.tabs.sendMessage(tvTabId, {
        type: 'DRAW_LEVELS',
        levels: [],
        _pine: lastAnalysis.brain.pine,
      }, () => void chrome.runtime.lastError);
    }
    navigator.clipboard?.writeText(b.pine).catch(() => {});
    const btn = document.getElementById('brainPineBtn');
    if (btn) { btn.textContent = '✓ Copiado — pega en Pine Editor TV'; setTimeout(() => { btn.textContent = '📋 Copiar Pipeline Pine Script → TV'; }, 3000); }
    showNotif('Pine Script copiado. Pega en el Editor Pine de TradingView');
  });
}

// ═══════════════════════════════════════════════════════════════════
// RENDER SESSIONS & NEWS
// ═══════════════════════════════════════════════════════════════════

// Live session status bar — updates every second, no API calls
function fmtCountdown(ms) {
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  // Always show seconds so the display ticks every second
  if (h > 0) return `${h}h ${String(m).padStart(2,'0')}m ${String(s).padStart(2,'0')}s`;
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

function renderSessionStatusBar() {
  const el = document.getElementById('snSbSessions');
  if (!el) return;

  const now   = Date.now();
  const dayMs = (() => { const d = new Date(); d.setUTCHours(0,0,0,0); return d.getTime(); })();

  const chips = SESSIONS.map(s => {
    const startMs = dayMs + s.start * 3600000;
    const endMs   = dayMs + s.end   * 3600000;
    const isOpen  = now >= startMs && now < endMs;
    const isPend  = now < startMs;

    if (!isOpen && !isPend) {
      // Closed — show only main sessions (not sub) very dim
      if (s.sub) return '';
      return `<div class="sn-chip sn-chip-closed" style="border-color:rgba(255,255,255,.08)">
        <span style="color:#333;font-size:7.5px;font-weight:700">${s.code}</span>
        <span style="color:#2a2a2a;font-size:7px">✕</span>
      </div>`;
    }

    if (isOpen) {
      const remMs = endMs - now;
      // Color urgency: < 5min = red, < 30min = yellow, else = session color
      const col     = remMs < 5*60000 ? '#f85149' : remMs < 30*60000 ? '#e3b341' : s.color;
      const pulse   = remMs < 5*60000 ? ' sn-pulse' : '';
      const remStr  = fmtCountdown(remMs);
      const label   = remMs < 30*60000 ? `${s.code} cierra` : `${s.code}`;
      return `<div class="sn-chip sn-chip-open${pulse}" style="border-color:${col};background:${col}14">
        <span class="sn-chip-name" style="color:${col}">${label}</span>
        <span class="sn-chip-timer" style="color:${col};font-family:'Orbitron',monospace">${remStr}</span>
      </div>`;
    }

    // Pending
    const waitMs   = startMs - now;
    const localOpen = fmtHourLocal(s.start);
    // Color urgency: < 5min = red pulsing, < 30min = yellow, else = dim
    let col, pulse = '', borderCol;
    if (waitMs < 5*60000) {
      col = '#f85149'; pulse = ' sn-pulse'; borderCol = '#f85149';
    } else if (waitMs < 30*60000) {
      col = '#e3b341'; borderCol = '#e3b341';
    } else {
      col = '#555'; borderCol = 'rgba(255,255,255,.12)';
    }
    const waitStr = fmtCountdown(waitMs);
    const prefix  = waitMs < 30*60000 ? `${s.code} abre` : s.code;
    return `<div class="sn-chip sn-chip-pend${pulse}" style="border-color:${borderCol}">
      <span class="sn-chip-name" style="color:${col}">${prefix}</span>
      <span class="sn-chip-wait" style="color:${col}">${localOpen} <b style="font-family:'Orbitron',monospace">${waitStr}</b></span>
    </div>`;
  }).filter(Boolean).join('');

  // ── Also update detail card remaining times in the dropdown ──────
  SESSIONS.forEach(s => {
    const cardEl = document.getElementById(`sn-card-rem-${s.key}`);
    if (!cardEl) return;
    const startMs = dayMs + s.start * 3600000;
    const endMs   = dayMs + s.end   * 3600000;
    const isOpen  = now >= startMs && now < endMs;
    const isPend  = now < startMs;
    if (isOpen) {
      const remMs = endMs - now;
      const col   = remMs < 5*60000 ? '#f85149' : remMs < 30*60000 ? '#e3b341' : s.color;
      cardEl.textContent = `Cierra en ${fmtCountdown(remMs)}`;
      cardEl.style.color = col;
    } else if (isPend) {
      const waitMs = startMs - now;
      const col    = waitMs < 5*60000 ? '#f85149' : waitMs < 30*60000 ? '#e3b341' : '';
      cardEl.textContent = `Abre en ${fmtCountdown(waitMs)}`;
      cardEl.style.color = col;
    } else {
      cardEl.textContent = '';
    }
  });

  // Next event today
  const nextEv = (window._lastCalDays || [])
    .filter(d => d.isToday).flatMap(d => d.events).find(e => !e.isPast);
  const evChip = nextEv
    ? `<div class="sn-chip sn-chip-event" title="${escHtml(nextEv.title)}">
        <span style="color:${nextEv.stars===3?'#f85149':'#e3b341'}">${nextEv.stars===3?'★★★':'★★☆'}</span>
        <span style="color:var(--text2);font-size:7.5px">${nextEv.country} ${escHtml(nextEv.title.slice(0,15))}…</span>
        <span style="color:var(--accent);font-size:7.5px">${nextEv.time}</span>
      </div>`
    : '';

  el.innerHTML = chips + evChip;
}

function renderSessionsNews(sessions, calDays) {
  const el = document.getElementById('aSessionsNews');
  if (!el) return;

  const nowH = new Date().getUTCHours() + new Date().getUTCMinutes() / 60;

  // ── Visual 24h timeline bar ───────────────────────────────────────
  // Sessions: Asia 0-9, London 7-16, NY 13-22. Overlaps are intentional.
  const W = 100; // % width = 24h
  const pct = h => (h / 24 * 100).toFixed(2);
  const timelineBlocks = sessions.map(s =>
    `<div class="sn-tl-block" style="left:${pct(s.start)}%;width:${pct(s.end - s.start)}%;background:${s.color}22;border-top:2px solid ${s.color}" title="${s.name}: ${s.start}:00–${s.end}:00 UTC"></div>`
  ).join('');
  // Overlap zones
  const overlapBlocks = `
    <div class="sn-tl-overlap" style="left:${pct(7)}%;width:${pct(2)}%" title="Overlap Asia-Londres 07-09 UTC"></div>
    <div class="sn-tl-overlap" style="left:${pct(13)}%;width:${pct(3)}%" title="Overlap Londres-NY 13-16 UTC"></div>`;
  // Current time marker
  const nowPct = (nowH / 24 * 100).toFixed(2);
  const timeMarker = `<div class="sn-tl-now" style="left:${nowPct}%"></div>`;
  // Hour labels
  const hourLabels = [0,4,8,12,16,20].map(h =>
    `<span class="sn-tl-lbl" style="left:${pct(h)}%">${String(h).padStart(2,'0')}</span>`
  ).join('');

  const timelineHtml = `
    <div class="sn-timeline-wrap">
      <div class="sn-timeline">
        ${timelineBlocks}${overlapBlocks}${timeMarker}
      </div>
      <div class="sn-tl-labels">${hourLabels}</div>
    </div>
    <div class="sn-tl-legend">
      ${sessions.map(s=>`<span style="color:${s.color}">▬ ${s.name} ${fmtHourLocal(s.start)}–${fmtHourLocal(s.end)}</span>`).join('')}
      <span class="sn-tl-overlap-lbl">▓ Overlap</span>
    </div>
    <div style="font-size:7px;color:var(--text2);margin-bottom:6px">Horario local (España)</div>`;

  // ── Session cards ─────────────────────────────────────────────────
  const sessHtml = sessions.map(s => {
    const statusCol = s.status==='open' ? '#00ff41' : s.status==='closed' ? '#444' : '#e3b341';
    const statusTxt = s.status==='open' ? '● ABIERTA' : s.status==='closed' ? '✕ CERRADA' : '○ PENDIENTE';
    const rangeInfo = s.high
      ? `<span class="sn-range-hi">H $${fmtPrice(s.high)}</span><span class="sn-range-lo">L $${fmtPrice(s.low)}</span><span class="sn-range-pct">${s.rangePct}%</span>`
      : s.status === 'closed'  ? `<span class="sn-range-na">Sin datos registrados</span>`
      : s.sub                  ? `<span class="sn-range-na">Extended hours</span>`
      :                          `<span class="sn-range-na">Esperando apertura…</span>`;
    const opacity = s.status==='closed' ? ';opacity:.45' : s.sub ? ';opacity:.75' : '';
    return `<div class="sn-session" style="border-left-color:${s.color}${opacity}">
      <div class="sn-sess-hdr">
        <span class="sn-cc-badge" style="border-color:${s.color};color:${s.color}">${s.code}</span>
        <span class="sn-sess-name">${s.name}</span>
        <span class="sn-sess-utc">${fmtHourLocal(s.start)}–${fmtHourLocal(s.end)} <span style="opacity:.5">/ ${fmtHour(s.start)}–${fmtHour(s.end)} UTC</span></span>
        <span class="sn-status" style="color:${statusCol}">${statusTxt}</span>
        <span class="sn-remaining" id="sn-card-rem-${s.key}"></span>
      </div>
      <div class="sn-range">${rangeInfo}</div>
    </div>`;
  }).join('');

  // Overlap active banner
  const asiaLondon = nowH >= 7 && nowH < 9;
  const londonNY   = nowH >= 13 && nowH < 16;
  const overlapHtml = (asiaLondon || londonNY)
    ? `<div class="sn-overlap">⚡ OVERLAP ACTIVO: ${asiaLondon ? 'Asia–Londres (07–09 UTC)' : 'Londres–NY (13–16 UTC)'} — máxima liquidez ahora</div>` : '';

  // ── Full-week economic calendar ──
  let calHtml = '';
  if (calDays.length) {
    calHtml = calDays.map(day => {
      const dayHdr = day.isToday
        ? `<div class="sn-day-hdr sn-day-today">📅 HOY — ${new Date(day.date+'T12:00:00Z').toLocaleDateString('es',{weekday:'long',day:'numeric',month:'long'})}</div>`
        : `<div class="sn-day-hdr${day.isPast?' sn-day-past':''}">${day.label} · ${new Date(day.date+'T12:00:00Z').toLocaleDateString('es',{day:'numeric',month:'short'})}</div>`;

      const evRows = day.events.map(e => {
        const past   = day.isPast || (day.isToday && isPastEvent(e.time));
        const stars  = e.stars === 3 ? '★★★' : '★★☆';
        const sCol   = e.stars === 3 ? '#f85149' : '#e3b341';
        return `<div class="sn-event${past?' sn-ev-past':''}${day.isToday&&!past?' sn-ev-today':''}">
          <span class="sn-ev-time">${e.time}</span>
          <span class="sn-ev-stars" style="color:${sCol}">${stars}</span>
          <span class="sn-ev-country" style="background:${sCol}22;color:${sCol}">${e.country}</span>
          <span class="sn-ev-title">${escHtml(e.title)}</span>
          ${e.forecast ? `<span class="sn-ev-fore">Est: ${escHtml(e.forecast)}</span>` : ''}
          ${e.previous ? `<span class="sn-ev-prev">Ant: ${escHtml(e.previous)}</span>` : ''}
        </div>`;
      }).join('');

      return dayHdr + evRows;
    }).join('');
  } else {
    calHtml = '<div class="sn-empty">Sin eventos de alto impacto esta semana</div>';
  }

  el.innerHTML = `
    <div class="sn-block-title">SESIONES HOY (UTC)</div>
    ${timelineHtml}
    ${overlapHtml}
    <div class="sn-sessions">${sessHtml}</div>
    <div class="sn-block-title" style="margin-top:10px">CALENDARIO ECONOMICO SEMANA — 3 ESTRELLAS</div>
    <div class="sn-cal">${calHtml}</div>
  `;
}

function isPastEvent(timeStr) {
  if (!timeStr || timeStr === 'Todo el día') return false;
  const [h, m] = timeStr.split(':').map(Number);
  const evUtc = h * 60 + (m || 0); // assume time is already UTC (Forex Factory uses ET, adjust later)
  const nowUtc = new Date().getUTCHours() * 60 + new Date().getUTCMinutes();
  return nowUtc > evUtc + 30; // 30min buffer
}

// ═══════════════════════════════════════════════════════════════════
// BOT TRADES — Render historial
// ═══════════════════════════════════════════════════════════════════

function renderBotTrades() {
  const el = document.getElementById('aBotTrades');
  const section = document.getElementById('aBotTradesSection');
  if (!el || !section) return;
  // Bot Trades section is always hidden in the analysis tab —
  // portfolio signals are shown in aBrainPortfolio (inside Trading Brain)
  section.style.display = 'none';
  return;
  
  // Mostrar solo trades del símbolo actual
  const symbolTrades = botTrades
    .filter(t => t.symbol === currentSymbol)
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, 20); // Últimos 20
  
  if (!symbolTrades.length) {
    section.style.display = 'none';
    return;
  }
  
  section.style.display = '';
  
  // Stats
  const completed = symbolTrades.filter(t => t.status.startsWith('win') || t.status === 'loss');
  const wins = symbolTrades.filter(t => t.status.startsWith('win')).length;
  const losses = symbolTrades.filter(t => t.status === 'loss').length;
  const winRate = completed.length > 0 ? (wins / completed.length * 100).toFixed(1) : '—';
  const avgPnl = completed.length > 0 
    ? (completed.reduce((sum, t) => sum + (t.pnlPct || 0), 0) / completed.length).toFixed(2)
    : '—';
  
  const statsHtml = `<div class="bot-stats">
    <div class="bot-stat">
      <span class="bot-stat-lbl">Win Rate</span>
      <span class="bot-stat-val" style="color:${parseFloat(winRate) >= 60 ? '#00ff41' : parseFloat(winRate) >= 45 ? '#e3b341' : '#f85149'}">${winRate}%</span>
    </div>
    <div class="bot-stat">
      <span class="bot-stat-lbl">Trades</span>
      <span class="bot-stat-val">${wins}W / ${losses}L</span>
    </div>
    <div class="bot-stat">
      <span class="bot-stat-lbl">Avg PnL</span>
      <span class="bot-stat-val" style="color:${parseFloat(avgPnl) > 0 ? '#00ff41' : '#f85149'}">${avgPnl}%</span>
    </div>
  </div>`;
  
  const tradesHtml = symbolTrades.map(t => {
    const isLong = t.direction === 'LONG';
    const col = isLong ? '#00ff41' : '#f85149';
    const icon = isLong ? '▲' : '▼';
    
    const statusBadge = (() => {
      switch(t.status) {
        case 'pending':   return '<span class="bot-badge bot-pending">⏳ Esperando entrada</span>';
        case 'alerta':    return '<span class="bot-badge" style="background:rgba(255,165,0,.2);color:#ffa500;border:1px solid rgba(255,165,0,.5)">🔔 PRE-ALERTA ≥95%</span>';
        case 'running':   return '<span class="bot-badge bot-running">🔄 Activo</span>';
        case 'win_tp1':   return '<span class="bot-badge bot-win">✅ WIN TP1</span>';
        case 'win_tp2':   return '<span class="bot-badge bot-win">✅ WIN TP2</span>';
        case 'loss':      return '<span class="bot-badge bot-loss">❌ LOSS</span>';
        case 'cancelled': return '<span class="bot-badge bot-cancelled">⊘ Cancelado</span>';
        case 'expired':   return '<span class="bot-badge bot-expired">⏰ Oportunidad perdida</span>';
        default:          return '<span class="bot-badge">' + t.status + '</span>';
      }
    })();
    
    // Calcular PnL actual para trades running
    let currentPnl = null;
    let livePrice = livePrices[t.symbol] || (t.symbol === currentSymbol ? currentPrice : 0);
    
    if (t.status === 'running' && livePrice > 0) {
      const isLong = t.direction === 'LONG';
      currentPnl = isLong
        ? ((livePrice - t.entry) / t.entry * 100)
        : ((t.entry - livePrice) / t.entry * 100);
    }
    
    const pnlBadge = t.pnlPct != null
      ? (() => { const roi = (t.pnlPct * (t.leverage || btLeverage)); return `<span class="bot-pnl" style="color:${roi > 0 ? '#00ff41' : '#f85149'}">${roi > 0 ? '+' : ''}${roi.toFixed(2)}%</span>`; })()
      : currentPnl != null
      ? `<span class="bot-pnl bot-pnl-live" style="color:${currentPnl > 0 ? '#00ff41' : '#f85149'}">${currentPnl > 0 ? '+' : ''}${(currentPnl * btLeverage).toFixed(2)}% 📊</span>`
      : '';
    
    // Mostrar precio actual para trades running
    const currentPriceDisplay = (t.status === 'running' && livePrice > 0)
      ? `<span class="bot-current-price">@ $${fmtPrice(livePrice)}</span>`
      : '';
    
    const time = new Date(t.timestamp).toLocaleString('es', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
    
    return `<div class="bot-trade" data-trade-id="${t.id}">
      <div class="bot-trade-hdr">
        <span class="bot-trade-dir" style="color:${col}">${icon} ${t.direction}</span>
        ${currentPriceDisplay}
        <span class="bot-trade-time">${time}</span>
        <span class="bot-trade-conf">Conf. ${t.confidence}%</span>
        ${statusBadge}
        ${pnlBadge}
        <button class="bot-delete-btn-small" data-trade-id="${t.id}" title="Eliminar">🗑</button>
      </div>
      <div class="bot-trade-reason">${escHtml(t.reason)}</div>
      <div class="bot-trade-prices">
        <span class="bot-tp-item">E: $${fmtPrice(t.entry)}</span>
        <span class="bot-tp-item">SL: $${fmtPrice(t.sl)}</span>
        <span class="bot-tp-item" style="color:#00ff41">TP1: $${fmtPrice(t.tp1)}</span>
        <span class="bot-tp-item" style="color:#00ff41">TP2: $${fmtPrice(t.tp2)}</span>
        <span class="bot-tp-item">R:R ${t.rr}:1</span>
      </div>
      ${t.margin ? `<div class="bot-investment-grid">
        <div class="bot-inv-box"><span class="bot-inv-lbl">💰 Margen</span><span class="bot-inv-val">$${t.margin.toFixed(0)}</span></div>
        <div class="bot-inv-box"><span class="bot-inv-lbl">📈 Leverage</span><span class="bot-inv-val">${t.leverage}x</span></div>
        <div class="bot-inv-box"><span class="bot-inv-lbl">💎 Nocional</span><span class="bot-inv-val">$${t.notional.toFixed(0)}</span></div>
        <div class="bot-inv-box"><span class="bot-inv-lbl">💸 Fee apertura</span><span class="bot-inv-val bot-inv-fee">$${t.openFee.toFixed(2)}</span></div>
        <div class="bot-inv-box bot-inv-tp"><span class="bot-inv-lbl">✅ TP1 est.</span><span class="bot-inv-val">+$${t.tpUsd1.toFixed(2)}</span></div>
        <div class="bot-inv-box bot-inv-tp"><span class="bot-inv-lbl">✅ TP2 est.</span><span class="bot-inv-val">+$${t.tpUsd2.toFixed(2)}</span></div>
        <div class="bot-inv-box bot-inv-sl"><span class="bot-inv-lbl">❌ SL est.</span><span class="bot-inv-val">${t.slUsd.toFixed(2)}</span></div>
      </div>` : ''}
    </div>`;
  }).join('');
  
  el.innerHTML = statsHtml + tradesHtml;
  
  // Event listener para botones de eliminar (vista compacta)
  el.querySelectorAll('.bot-delete-btn-small').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const tradeId = parseInt(btn.dataset.tradeId);
      if (confirm('¿Eliminar este trade?')) {
        deleteBotTrade(tradeId);
        renderBotTrades(); // Re-renderizar vista compacta
      }
    });
  });
}

// ═══════════════════════════════════════════════════════════════════
// BOT TRADES TAB — Historial completo con filtros
// ═══════════════════════════════════════════════════════════════════

function renderBotTradesTab() {
  // Helper: ROI real sobre margen = pnlPct × leverage del trade
  const pnlRoi = (t) => (t.pnlPct || 0) * (t.leverage || btLeverage);

  // Stats globales
  const completed = botTrades.filter(t => t.status.startsWith('win') || t.status === 'loss');
  const wins = botTrades.filter(t => t.status.startsWith('win'));
  const losses = botTrades.filter(t => t.status === 'loss');
  const winRate = completed.length > 0 ? (wins.length / completed.length * 100).toFixed(1) : 0;
  const avgPnl = completed.length > 0 
    ? (completed.reduce((sum, t) => sum + pnlRoi(t), 0) / completed.length).toFixed(2)
    : 0;
  const bestTrade = completed.length > 0
    ? Math.max(...completed.map(t => pnlRoi(t))).toFixed(2)
    : 0;
  const worstTrade = completed.length > 0
    ? Math.min(...completed.map(t => pnlRoi(t))).toFixed(2)
    : 0;
  
  const statTotal = document.getElementById('btStatTotal');
  const statWinRate = document.getElementById('btStatWinRate');
  const statAvgPnl = document.getElementById('btStatAvgPnl');
  const statBest = document.getElementById('btStatBest');
  const statWorst = document.getElementById('btStatWorst');
  
  if (statTotal) statTotal.textContent = botTrades.length;
  if (statWinRate) {
    statWinRate.textContent = `${winRate}%`;
    statWinRate.style.color = parseFloat(winRate) >= 60 ? '#00ff41' : parseFloat(winRate) >= 45 ? '#e3b341' : '#f85149';
  }
  if (statAvgPnl) {
    statAvgPnl.textContent = `${avgPnl > 0 ? '+' : ''}${avgPnl}%`;
    statAvgPnl.style.color = parseFloat(avgPnl) > 0 ? '#00ff41' : '#f85149';
  }
  if (statBest) {
    statBest.textContent = `+${bestTrade}%`;
    statBest.style.color = '#00ff41';
  }
  if (statWorst) {
    statWorst.textContent = `${worstTrade}%`;
    statWorst.style.color = '#f85149';
  }
  
  // Poblar filtro de símbolos
  const symbolFilter = document.getElementById('btFilterSymbol');
  if (symbolFilter && symbolFilter.options.length === 1) {
    const symbols = [...new Set(botTrades.map(t => t.symbol))].sort();
    symbols.forEach(sym => {
      const opt = document.createElement('option');
      opt.value = sym;
      opt.textContent = sym;
      symbolFilter.appendChild(opt);
    });
  }
  
  // Filtrar trades
  const symbolFilterVal = document.getElementById('btFilterSymbol')?.value || '';
  const statusFilterVal = document.getElementById('btFilterStatus')?.value || '';
  
  let filtered = [...botTrades];
  
  if (symbolFilterVal) {
    filtered = filtered.filter(t => t.symbol === symbolFilterVal);
  }
  
  if (statusFilterVal) {
    if (statusFilterVal === 'win') {
      filtered = filtered.filter(t => t.status.startsWith('win'));
    } else {
      filtered = filtered.filter(t => t.status === statusFilterVal);
    }
  }
  
  filtered.sort((a, b) => b.timestamp - a.timestamp);
  
  // Renderizar lista
  const listEl = document.getElementById('btTradesList');
  if (!listEl) return;
  
  if (!filtered.length) {
    listEl.innerHTML = '<div class="bt-empty">No hay trades registrados</div>';
    return;
  }
  
  const tradesHtml = filtered.map(t => {
    const isLong = t.direction === 'LONG';
    const col = isLong ? '#00ff41' : '#f85149';
    const icon = isLong ? '▲' : '▼';
    
    // ── RECALCULAR valores USD con margen/leverage ACTUAL (dinámico) ──
    const margin = btMargin;
    const leverage = btLeverage;
    const notional = margin * leverage;
    const openFee = notional * 0.0004;
    
    // Usar % guardados (no cambian), recalcular USD
    const tpPct1 = t.tpPct1;
    const tpPct2 = t.tpPct2;
    const slPct = t.slPct;
    
    const tpUsd1 = margin * (tpPct1 / 100) * leverage - openFee - (notional * 0.0004);
    const tpUsd2 = margin * (tpPct2 / 100) * leverage - openFee - (notional * 0.0004);
    const slUsd = -(margin * (slPct / 100) * leverage + openFee);
    
    // Check execution mode for alert badges
    const executionMode = document.getElementById('btExecutionMode')?.value || 'signal';
    
    const statusBadge = (() => {
      switch(t.status) {
        case 'pending':   return '<span class="bot-badge bot-pending">⏳ Esperando entrada</span>';
        case 'alerta':    
          if (executionMode === 'limit') {
            return '<span class="bot-badge" style="background:rgba(0,255,65,.2);color:#00ff41;border:1px solid rgba(0,255,65,.5);font-weight:700">💰 ORDEN LÍMITE — Auto-ejecuta</span>';
          }
          return '<span class="bot-badge" style="background:rgba(255,165,0,.2);color:#ffa500;border:1px solid rgba(255,165,0,.5)">🔔 PRE-ALERTA ≥95%</span>';
        case 'running':
          if (t._tp1Secured) return '<span class="bot-badge" style="background:rgba(0,255,65,.25);color:#00ff41;border:1px solid #00ff41;font-weight:700;animation:pulse 1.5s infinite">✅ TP1 ✓ → Esperando TP2</span>';
          return '<span class="bot-badge bot-running">🔄 Activo</span>';
        case 'win_tp1':   return '<span class="bot-badge bot-win">✅ WIN TP1</span>';
        case 'win_tp2':   return '<span class="bot-badge bot-win">🏆 WIN TP2</span>';
        case 'loss':      return '<span class="bot-badge bot-loss">❌ LOSS</span>';
        case 'cancelled': return '<span class="bot-badge bot-cancelled">⊘ Cancelado</span>';
        case 'expired':   return '<span class="bot-badge bot-expired">⏰ Oportunidad perdida</span>';
        default:          return '<span class="bot-badge">' + t.status + '</span>';
      }
    })();
    
    // Calcular PnL actual para trades running o pending
    let currentPnl = null;
    let currentPnlUsd = null;
    let livePrice = livePrices[t.symbol] || (t.symbol === currentSymbol ? currentPrice : 0);
    
    if ((t.status === 'running' || t.status === 'pending') && livePrice > 0) {
      const isLong = t.direction === 'LONG';
      const priceDiff = isLong ? (livePrice - t.entry) : (t.entry - livePrice);
      currentPnl = (priceDiff / t.entry) * 100;
      // PnL en USD con leverage DINÁMICO
      currentPnlUsd = margin * (currentPnl / 100) * leverage;
      
      // Si está en ganancia, restar fee de cierre estimado
      if (currentPnlUsd > 0) {
        currentPnlUsd -= (notional * 0.0004); // Taker fee al cerrar
      }
    }
    
    const pnlBadge = t.pnlPct != null
      ? (() => { const roi = (t.pnlPct * (t.leverage || btLeverage)); return `<span class="bot-pnl-big" style="color:${roi > 0 ? '#00ff41' : '#f85149'}">${roi > 0 ? '+' : ''}${roi.toFixed(2)}% <span style="font-size:9px;opacity:.6">(ROI margen)</span></span>`; })()
      : currentPnl != null
      ? `<span class="bot-pnl-big bot-pnl-live" style="color:${currentPnl > 0 ? '#00ff41' : '#f85149'}">${currentPnl > 0 ? '+' : ''}${(currentPnl * btLeverage).toFixed(2)}% 
         <span style="font-size:10px;opacity:0.8">(${currentPnlUsd > 0 ? '+' : ''}$${currentPnlUsd.toFixed(2)})</span></span>`
      : '';
    
    // Mostrar precio actual y distancias para trades activos
    let currentPriceDisplay = '';
    if ((t.status === 'running' || t.status === 'pending') && livePrice > 0) {
      const isLong = t.direction === 'LONG';
      
      // Distancias a TP1, TP2, SL (en % desde precio actual)
      // Para LONG: TP arriba, SL abajo
      // Para SHORT: TP abajo, SL arriba
      const distToTp1Pct = Math.abs((t.tp1 - livePrice) / livePrice * 100);
      const distToTp2Pct = Math.abs((t.tp2 - livePrice) / livePrice * 100);
      const distToSlPct = Math.abs((t.sl - livePrice) / livePrice * 100);
      
      const tp1Hit = isLong ? livePrice >= t.tp1 : livePrice <= t.tp1;
      const tp2Hit = isLong ? livePrice >= t.tp2 : livePrice <= t.tp2;
      const slHit = isLong ? livePrice <= t.sl : livePrice >= t.sl;
      
      const statusIcon = tp1Hit ? '✅ TP1 alcanzado!' : slHit ? '❌ SL alcanzado!' : '🎯';
      
      const tp1Arrow = isLong ? '↑' : '↓';
      const tp2Arrow = isLong ? '↑' : '↓';
      const slArrow = isLong ? '↓' : '↑';
      
      currentPriceDisplay = `<div class="bot-live-status">
        <span class="bot-current-price-big">${statusIcon} Precio: $${fmtPrice(livePrice)}</span>
        <div class="bot-distances">
          <span style="color:#00ff41;font-size:9px">TP1: ${tp1Hit ? `✓ ALCANZADO` : `${tp1Arrow}${distToTp1Pct.toFixed(2)}%`}</span>
          <span style="color:#00ff41;font-size:9px">TP2: ${tp2Hit ? `✓ ALCANZADO` : `${tp2Arrow}${distToTp2Pct.toFixed(2)}%`}</span>
          <span style="color:#f85149;font-size:9px">SL: ${slHit ? `✓ TOCADO` : `${slArrow}${distToSlPct.toFixed(2)}%`}</span>
        </div>
      </div>`;
    }

    const time = new Date(t.timestamp).toLocaleString('es', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
    const hitTime = t.hitTime ? new Date(t.hitTime).toLocaleString('es', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    }) : null;

    const tradeTfLabel = tradeTf(t) || '';
    return `<div class="bt-trade-full">
      <div class="bt-trade-hdr-full">
        <div class="bt-trade-left">
          <span class="bt-trade-symbol">${t.symbol}</span>
          ${tradeTfLabel ? `<span style="background:rgba(0,212,255,.18);border:1px solid rgba(0,212,255,.4);border-radius:3px;color:#00d4ff;font-size:8px;font-weight:700;padding:1px 4px;font-family:'Orbitron',monospace">${tradeTfLabel}</span>` : ''}
          <span class="bt-trade-dir-big" style="color:${col}">${icon} ${t.direction}</span>
          ${pnlBadge}
        </div>
        <div class="bt-trade-right">
          ${statusBadge}
          <span class="bt-trade-conf-big">Conf. ${t.confidence}%</span>
          ${(t.status === 'running' || t.status === 'pending') ? `
          <button class="bt-be-btn" data-trade-id="${t.id}" title="Mover SL a breakeven (entrada)" style="background:rgba(227,179,65,.18);border:1px solid rgba(227,179,65,.6);color:#e3b341;border-radius:4px;font-size:9px;padding:2px 5px;cursor:pointer;font-weight:700">⚖ BE</button>
          <button class="bt-close-btn" data-trade-id="${t.id}" title="Cerrar trade al precio actual" style="background:rgba(248,81,73,.18);border:1px solid rgba(248,81,73,.6);color:#f85149;border-radius:4px;font-size:9px;padding:2px 5px;cursor:pointer;font-weight:700">✕ Cerrar</button>
          ` : ''}
          <button class="bt-delete-btn" data-trade-id="${t.id}" title="Eliminar trade">🗑</button>
        </div>
      </div>
      <div class="bt-trade-time-full">
        <span>📅 Señal: ${time}</span>
        ${hitTime ? `<span>⚡ ${t.status === 'loss' ? 'SL' : 'TP'}: ${hitTime}</span>` : ''}
      </div>
      <div class="bt-trade-reason-full">${escHtml(t.reason)}</div>
      ${currentPriceDisplay}
      <div class="bt-trade-grid">
        <div class="bt-grid-item">
          <span class="bt-grid-lbl">ENTRADA</span>
          <span class="bt-grid-val">$${fmtPrice(t.entry)}</span>
        </div>
        <div class="bt-grid-item">
          <span class="bt-grid-lbl">SL</span>
          <span class="bt-grid-val" style="color:#f85149">$${fmtPrice(t.sl)}</span>
        </div>
        <div class="bt-grid-item">
          <span class="bt-grid-lbl">TP1</span>
          <span class="bt-grid-val" style="color:#00ff41">$${fmtPrice(t.tp1)}</span>
        </div>
        <div class="bt-grid-item">
          <span class="bt-grid-lbl">TP2</span>
          <span class="bt-grid-val" style="color:#00ff41">$${fmtPrice(t.tp2)}</span>
        </div>
        <div class="bt-grid-item">
          <span class="bt-grid-lbl">R:R</span>
          <span class="bt-grid-val">${t.rr}:1</span>
        </div>
        ${t.hitPrice ? `<div class="bt-grid-item">
          <span class="bt-grid-lbl">${t.status === 'loss' ? 'HIT SL' : 'HIT TP'}</span>
          <span class="bt-grid-val">$${fmtPrice(t.hitPrice)}</span>
        </div>` : ''}
      </div>
      ${t.margin ? `<div class="bot-investment-grid">
        <div class="bot-inv-summary" style="grid-column:1/-1;background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.3);border-radius:5px;padding:6px 8px;margin-bottom:6px">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:10px;color:var(--text3)">💰 Inversión: <b style="color:var(--accent)">$${margin}</b> × <b style="color:var(--accent)">${leverage}x</b> = <b style="color:#00ff41">$${notional}</b> nocional</span>
            <span style="font-size:9px;color:var(--text3)">Fee apertura: <span style="color:#f85149">-$${openFee.toFixed(2)}</span></span>
          </div>
        </div>
        <div class="bot-inv-box bot-inv-tp" style="background:rgba(0,255,65,.12);border:1px solid rgba(0,255,65,.3)">
          <span class="bot-inv-lbl">✅ TP1</span>
          <div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px">
            <span class="bot-inv-val" style="font-size:12px;font-weight:700;color:#00ff41">+$${tpUsd1.toFixed(2)}</span>
            <span style="font-size:8px;color:var(--text3)">${tpPct1 > 0 ? '+' : ''}${tpPct1.toFixed(2)}% · ROI: ${((tpUsd1/margin)*100).toFixed(1)}%</span>
          </div>
        </div>
        <div class="bot-inv-box bot-inv-tp" style="background:rgba(0,255,65,.12);border:1px solid rgba(0,255,65,.3)">
          <span class="bot-inv-lbl">✅ TP2</span>
          <div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px">
            <span class="bot-inv-val" style="font-size:12px;font-weight:700;color:#00ff41">+$${tpUsd2.toFixed(2)}</span>
            <span style="font-size:8px;color:var(--text3)">${tpPct2 > 0 ? '+' : ''}${tpPct2.toFixed(2)}% · ROI: ${((tpUsd2/margin)*100).toFixed(1)}%</span>
          </div>
        </div>
        <div class="bot-inv-box bot-inv-sl" style="background:rgba(248,81,73,.12);border:1px solid rgba(248,81,73,.3)">
          <span class="bot-inv-lbl">❌ SL</span>
          <div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px">
            <span class="bot-inv-val" style="font-size:12px;font-weight:700;color:#f85149">${slUsd.toFixed(2)}</span>
            <span style="font-size:8px;color:var(--text3)">-${slPct.toFixed(2)}% · ROI: ${((slUsd/margin)*100).toFixed(1)}%</span>
          </div>
        </div>
        ${currentPnlUsd != null ? `<div class="bot-inv-box" style="grid-column:1/-1;background:${currentPnlUsd >= 0 ? 'rgba(0,255,65,.15)' : 'rgba(248,81,73,.15)'};border:2px solid ${currentPnlUsd >= 0 ? '#00ff41' : '#f85149'}">
          <span class="bot-inv-lbl" style="font-size:10px">📊 PnL Actual (en vivo)</span>
          <div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px">
            <span class="bot-inv-val" style="font-size:14px;font-weight:700;color:${currentPnlUsd >= 0 ? '#00ff41' : '#f85149'}">${currentPnlUsd > 0 ? '+' : ''}$${currentPnlUsd.toFixed(2)}</span>
            <span style="font-size:9px;color:var(--text2)">${currentPnl > 0 ? '+' : ''}${currentPnl.toFixed(3)}% precio · ROI: ${((currentPnlUsd/margin)*100).toFixed(1)}% sobre margen</span>
          </div>
        </div>` : ''}
      </div>` : ''}
    </div>`;
  }).join('');
  
  listEl.innerHTML = tradesHtml;
}

// ═══════════════════════════════════════════════════════════════════
// PAPER TRADING
// ═══════════════════════════════════════════════════════════════════

function ptSave() {
  chrome.storage.local.set({ tvPtBalance: ptBalance, tvPtPositions: ptPositions.slice(0, 200), tvPtStart: ptStart, tvPtPeak: ptPeakEq });
}

function ptLoad(cb) {
  chrome.storage.local.get(['tvPtBalance','tvPtPositions','tvPtStart','tvPtPeak'], r => {
    if (r.tvPtBalance != null)   ptBalance   = r.tvPtBalance;
    if (r.tvPtPositions?.length) ptPositions = r.tvPtPositions;
    if (r.tvPtStart != null)     ptStart     = r.tvPtStart;
    if (r.tvPtPeak  != null)     ptPeakEq    = r.tvPtPeak;
    if (cb) cb();
  });
}

function ptUpdateHeaderBal() {
  const el = document.getElementById('hdrPtBal');
  if (el) {
    el.textContent = '$' + ptBalance.toFixed(2);
    const pct = ptBalance / ptStart;
    el.style.color = pct < 0.9 ? '#f85149' : pct < 0.95 ? '#e3b341' : '';
  }
}

// ── Futures constants (Binance Futures rates) ─────────────────────────────
const FUTURES_TAKER_FEE = 0.0004;   // 0.04% on notional
const FUTURES_MAKER_FEE = 0.0002;   // 0.02% on notional
const MAINT_MARGIN_RATE = 0.004;    // 0.4% maintenance margin (tier 1)
const FUNDING_RATE      = 0.0001;   // 0.01% every 8h (simulated)

function calcLiqPrice(entryPrice, leverage, direction) {
  // Isolated margin liquidation formula (simplified)
  // LONG: liq = entry * (1 - 1/lev + MMR)
  // SHORT: liq = entry * (1 + 1/lev - MMR)
  const mmr = MAINT_MARGIN_RATE;
  return direction === 'long'
    ? entryPrice * (1 - 1 / leverage + mmr)
    : entryPrice * (1 + 1 / leverage - mmr);
}

function ptBuyDemo(symbol, price, tp, sl, margin, direction = 'long', leverage = 10) {
  if (!price || price <= 0) { showNotif('Precio no disponible — analiza primero', 'error'); return; }
  if (margin > ptBalance)   { showNotif(`Margen insuficiente ($${ptBalance.toFixed(2)} disponible)`, 'error'); return; }
  if (ptPositions.find(p => p.status === 'open' && p.symbol === symbol && p.direction === direction)) {
    showNotif(`Ya tienes un ${direction.toUpperCase()} abierto en ${symbol}`, 'error'); return;
  }

  const notional  = margin * leverage;                       // tamaño real de posición
  const openFee   = notional * FUTURES_TAKER_FEE;           // fee apertura
  const qty       = notional / price;                        // cantidad de contratos
  const liqPrice  = calcLiqPrice(price, leverage, direction);

  // TP/SL based on PRICE movement (not margin-amplified)
  const tpPrice   = direction === 'long' ? price * (1 + tp / 100) : price * (1 - tp / 100);
  const slPrice   = direction === 'long' ? price * (1 - sl / 100) : price * (1 + sl / 100);

  ptBalance -= margin + openFee;  // deduct margin + opening fee

  const pos = {
    id:           `pt_${Date.now()}`,
    symbol, price, direction, leverage,
    margin,         // collateral locked
    notional,       // position size USD
    qty,            // token quantity
    openFee,
    feesTotal:    openFee,
    tpPct: tp, slPct: sl, tpPrice, slPrice, liqPrice,
    currentPrice: price,
    entryTime:    Date.now(),
    status:       'open',
    fundingPaid:  0,
    pnlUsd: null, pnlPct: null,
    exitPrice: null, exitTime: null, exitReason: null,
  };
  ptPositions.unshift(pos);
  ptSave();
  renderDemo();
  wsApply(); // add new symbol to live feed
  const icon = direction === 'long' ? '🟢' : '🔴';
  showNotif(`${icon} ${direction.toUpperCase()} ${symbol} ${leverage}x · Margen $${margin.toFixed(0)} · Nocional $${notional.toFixed(0)} · Liq $${fmtPrice(liqPrice)} · Fee $${openFee.toFixed(2)}`);
}

function ptSellDemo(posId, pct = 100, reason = 'manual') {
  const pos = ptPositions.find(p => p.id === posId && p.status === 'open');
  if (!pos) return;

  const frac       = pct / 100;
  const closeQty   = pos.qty * frac;
  const closeNot   = closeQty * pos.currentPrice;   // closing notional
  const closeFee   = closeNot * FUTURES_TAKER_FEE;

  // Gross P&L on position (futures: based on price move × qty)
  const priceMove  = pos.direction === 'long'
    ? pos.currentPrice - pos.price
    : pos.price - pos.currentPrice;
  const grossPnl   = priceMove * closeQty;

  // Funding cost accumulated (rough simulation)
  const holdHours  = (Date.now() - pos.entryTime) / 3600000;
  const fundPaid   = pos.notional * frac * FUNDING_RATE * Math.floor(holdHours / 8);

  const pnlUsd     = grossPnl - closeFee - fundPaid;
  const pnlPct     = pos.margin > 0 ? (pnlUsd / (pos.margin * frac)) * 100 : 0;
  const net        = (pos.margin * frac) + pnlUsd;

  ptBalance += Math.max(0, net); // margin + PnL back (can't go negative)

  if (pct >= 100) {
    Object.assign(pos, {
      status: 'closed', exitPrice: pos.currentPrice, exitTime: Date.now(),
      pnlUsd, pnlPct, exitReason: reason,
      holdMin: (Date.now() - pos.entryTime) / 60000,
      closeFee, fundPaid,
      feesTotal: (pos.openFee || 0) + closeFee,
    });
  } else {
    const partial = {
      ...pos, id: `pt_${Date.now()}_p`, qty: closeQty,
      margin: pos.margin * frac, notional: closeNot,
      status: 'closed', exitPrice: pos.currentPrice,
      exitTime: Date.now(), pnlUsd, pnlPct, exitReason: reason,
      holdMin: (Date.now() - pos.entryTime) / 60000,
      closeFee, fundPaid, _partial: true,
    };
    pos.qty     -= closeQty;
    pos.margin  -= pos.margin * frac;
    pos.notional = pos.qty * pos.price;
    ptPositions.splice(1, 0, partial);
  }

  const eq = ptBalance + ptPositions.filter(p => p.status === 'open')
    .reduce((s, p) => s + p.margin + (p.qty * p.currentPrice - p.notional), 0);
  if (eq > ptPeakEq) ptPeakEq = eq;
  ptSave();
  renderDemo();
  wsApply(); // remove closed symbol from feed if no longer needed
  const icon = reason === 'tp' ? '🎯' : reason === 'sl' ? '🛑' : (pnlUsd >= 0 ? '✓' : '✗');
  showNotif(`${icon} ${pos.symbol} · ${pnlUsd >= 0 ? '+' : ''}${pnlUsd.toFixed(2)}$ (${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(1)}%)`
    , pnlUsd >= 0 ? 'success' : 'error');
}

function ptUpdatePrices() {
  // Update currentPrice from Binance for open positions
  const open = ptPositions.filter(p => p.status === 'open');
  if (!open.length) return;
  Promise.allSettled(open.map(p => fetchTicker(p.symbol))).then(results => {
    results.forEach((r, i) => {
      if (r.status === 'fulfilled' && r.value) {
        const price = parseFloat(r.value.lastPrice);
        if (price > 0) {
          open[i].currentPrice = price;
          const isLong = open[i].direction !== 'short';
          const tpHit  = isLong ? price >= open[i].tpPrice  : price <= open[i].tpPrice;
          const slHit  = isLong ? price <= open[i].slPrice  : price >= open[i].slPrice;
          const liqHit = open[i].liqPrice && (isLong ? price <= open[i].liqPrice : price >= open[i].liqPrice);
          if (liqHit)                         ptSellDemo(open[i].id, 100, 'liquidated');
          else if (open[i].tpPrice && tpHit)  ptSellDemo(open[i].id, 100, 'tp');
          else if (open[i].slPrice && slHit)  ptSellDemo(open[i].id, 100, 'sl');
        }
      }
    });
    ptSave();
    renderDemo();
    ptUpdateHeaderBal();
  });
}

function renderDemo() {
  const open    = ptPositions.filter(p => p.status === 'open');
  const closed  = ptPositions.filter(p => p.status === 'closed');
  const wins    = closed.filter(p => (p.pnlUsd || 0) > 0).length;
  const wr      = closed.length ? Math.round(wins / closed.length * 100) : 0;
  const realPnl = closed.reduce((s, p) => s + (p.pnlUsd || 0), 0);
  // Futures unrealized P&L = qty × (currentPrice - entryPrice) × direction
  const unrealPnl = open.reduce((s, p) => {
    const move = p.direction === 'long'
      ? (p.currentPrice - p.price) * p.qty
      : (p.price - p.currentPrice) * p.qty;
    return s + move;
  }, 0);
  const lockedMargin = open.reduce((s, p) => s + (p.margin || p.totalCost || 0), 0);
  const equity   = ptBalance + lockedMargin + unrealPnl;
  const totalPnl = equity - ptStart;
  const totalPct = ptStart > 0 ? (totalPnl / ptStart) * 100 : 0;
  if (equity > ptPeakEq) ptPeakEq = equity;
  const maxDD = ptPeakEq > 0 ? ((ptPeakEq - equity) / ptPeakEq * 100) : 0;

  const balEl = document.getElementById('dBalMain');
  if (balEl) balEl.textContent = '$' + ptBalance.toFixed(2);
  const eqEl = document.getElementById('dEquity');
  if (eqEl) { eqEl.textContent = '$' + equity.toFixed(2); eqEl.style.color = equity >= ptStart ? '#00ff41' : '#f85149'; }
  const pnlEl = document.getElementById('dTotalPnl');
  if (pnlEl) {
    const col = totalPnl >= 0 ? '#00ff41' : '#f85149';
    pnlEl.innerHTML = '<span style="color:' + col + '">' + (totalPnl >= 0 ? '+' : '') + '$' + totalPnl.toFixed(2) + ' (' + (totalPct >= 0 ? '+' : '') + totalPct.toFixed(2) + '%)</span>';
  }

  const statsEl = document.getElementById('dStats');
  if (statsEl) {
    const wrCol = wr >= 50 ? '#00ff41' : '#f85149';
    statsEl.innerHTML =
      '<div class="d-stat"><span class="d-stat-n">' + closed.length + '</span><span class="d-stat-l">Trades</span></div>' +
      '<div class="d-stat"><span class="d-stat-n" style="color:' + wrCol + '">' + wr + '%</span><span class="d-stat-l">Win Rate</span></div>' +
      '<div class="d-stat"><span class="d-stat-n" style="color:' + (realPnl>=0?'#00ff41':'#f85149') + '">' + (realPnl>=0?'+':'') + '$' + realPnl.toFixed(2) + '</span><span class="d-stat-l">P&L Cerrado</span></div>' +
      '<div class="d-stat"><span class="d-stat-n" style="color:' + (unrealPnl>=0?'#00ff41':'#f85149') + '">' + (unrealPnl>=0?'+':'') + '$' + unrealPnl.toFixed(2) + '</span><span class="d-stat-l">P&L Abierto</span></div>' +
      '<div class="d-stat"><span class="d-stat-n" style="color:' + (maxDD>10?'#f85149':'#e3b341') + '">' + maxDD.toFixed(1) + '%</span><span class="d-stat-l">Max DD</span></div>' +
      '<div class="d-stat"><span class="d-stat-n" style="color:#00d4ff">' + open.length + '</span><span class="d-stat-l">Abiertas</span></div>';
  }

  const tpSymEl   = document.getElementById('dTpSymbol');
  const tpPriceEl = document.getElementById('dTpPrice');
  const tpInfoEl  = document.getElementById('dTpInfo');
  if (tpSymEl)   tpSymEl.textContent   = currentSymbol;
  if (tpPriceEl) tpPriceEl.textContent = currentPrice > 0 ? '$' + fmtPrice(currentPrice) : '—';
  if (tpInfoEl && currentPrice > 0) {
    const margin  = parseFloat(document.getElementById('dTpAmount')?.value) || 100;
    const tp      = parseFloat(document.getElementById('dTpTp')?.value)     || 30;
    const sl      = parseFloat(document.getElementById('dTpSl')?.value)     || 15;
    const lev     = parseInt(document.getElementById('dTpLev')?.value)      || 10;
    const notional = margin * lev;
    const openFee  = notional * FUTURES_TAKER_FEE;
    const closeFeeEst = notional * FUTURES_TAKER_FEE * (1 + tp / 100); // at TP price
    const tpUsd    = margin * (tp / 100) * lev - openFee - closeFeeEst;
    const slUsd    = margin * (sl / 100) * lev + openFee;
    const liqLong  = calcLiqPrice(currentPrice, lev, 'long');
    const liqShort = calcLiqPrice(currentPrice, lev, 'short');
    tpInfoEl.innerHTML =
      '<div class="d-tp-sum-row">' +
        '<span class="d-ti-lbl">Nocional</span><span class="d-ti-val">$' + notional.toFixed(0) + '</span>' +
        '<span class="d-ti-lbl">Fee apertura</span><span class="d-ti-val d-ti-fee">$' + openFee.toFixed(2) + '</span>' +
        '<span class="d-ti-lbl">R:R</span><span class="d-ti-val">' + (tp/sl).toFixed(1) + '</span>' +
      '</div>' +
      '<div class="d-tp-sum-row">' +
        '<span class="d-ti-lbl">TP est.</span><span class="d-ti-val d-ti-g">+$' + tpUsd.toFixed(2) + ' (+' + (tp*lev).toFixed(0) + '% margen)</span>' +
        '<span class="d-ti-lbl">SL est.</span><span class="d-ti-val d-ti-r">-$' + slUsd.toFixed(2) + ' (-' + (sl*lev).toFixed(0) + '% margen)</span>' +
      '</div>' +
      '<div class="d-tp-sum-row">' +
        '<span class="d-ti-lbl">Liq. LONG</span><span class="d-ti-val d-ti-r">$' + fmtPrice(liqLong) + '</span>' +
        '<span class="d-ti-lbl">Liq. SHORT</span><span class="d-ti-val d-ti-r">$' + fmtPrice(liqShort) + '</span>' +
      '</div>';
  }

  const ocEl = document.getElementById('dOpenCount');
  if (ocEl) ocEl.textContent = open.length ? '(' + open.length + ')' : '';

  const openEl = document.getElementById('dOpenList');
  if (openEl) {
    if (!open.length) {
      openEl.innerHTML = '<div class="d-empty">Sin posiciones — abre un LONG o SHORT</div>';
    } else {
      openEl.innerHTML = open.map(function(pos) {
        const isLong   = pos.direction !== 'short';
        const priceChg = pos.price > 0 ? (pos.currentPrice - pos.price) / pos.price : 0;
        const pricePct = (isLong ? priceChg : -priceChg) * 100;
        const pnlUsd   = (pos.notional || pos.usdAmount || 0) * (isLong ? priceChg : -priceChg);
        const col      = pnlUsd >= 0 ? '#00ff41' : '#f85149';
        const dirCol   = isLong ? '#00ff41' : '#f85149';
        const tp  = Math.min(500, Math.max(1, pos.tpPct || 30));
        const sl  = Math.min(80,  Math.max(1, pos.slPct || 15));
        const cur = Math.max(3, Math.min(97, ((pricePct + sl) / (tp + sl)) * 100));
        const heldMs  = Date.now() - pos.entryTime;
        const heldMin = Math.floor(heldMs / 60000);
        const heldSec = Math.floor((heldMs % 60000) / 1000);
        const heldStr = heldMin > 0 ? (heldMin + 'm ' + heldSec + 's') : (heldSec + 's');
        const arrowDir = pnlUsd >= 0 ? (isLong ? '▲' : '▼') : (isLong ? '▼' : '▲');
        const lev       = pos.leverage || 1;
        const notional  = pos.notional || pos.qty * pos.price;
        const liqPrice  = pos.liqPrice || calcLiqPrice(pos.price, lev, pos.direction);
        const liqPct    = pos.price > 0 ? Math.abs((liqPrice - pos.price) / pos.price * 100) : 0;
        const fundHours = Math.floor((Date.now() - pos.entryTime) / 28800000); // every 8h
        const fundEst   = notional * FUNDING_RATE * fundHours;
        const marginPnlPct = pos.margin > 0 ? (pnlUsd / pos.margin) * 100 : pricePct;
        return '<div class="d-pos-card">' +
          '<div class="d-pos-header">' +
            '<span class="d-pos-dir-badge" style="background:' + dirCol + '20;color:' + dirCol + ';border:1px solid ' + dirCol + '50">' + (isLong ? '▲ LONG' : '▼ SHORT') + '</span>' +
            '<span class="d-lev-badge">' + lev + 'x</span>' +
            '<span class="d-pos-sym">' + escHtml(pos.symbol) + '</span>' +
            '<span class="d-pos-age">' + heldStr + '</span>' +
          '</div>' +

          '<div class="d-pos-pnl-block">' +
            '<div class="d-pos-pnl-pct" style="color:' + col + ';text-shadow:0 0 14px ' + col + '50">' + (pnlUsd >= 0 ? '+' : '') + pnlUsd.toFixed(2) + ' USD</div>' +
            '<div class="d-pos-pnl-row">' +
              '<span class="d-pos-pnl-margin" style="color:' + col + '">' + (marginPnlPct >= 0 ? '+' : '') + marginPnlPct.toFixed(2) + '% margen</span>' +
              '<span class="d-pos-pnl-move" style="color:' + col + '">' + (pricePct >= 0 ? '+' : '') + pricePct.toFixed(3) + '% precio</span>' +
            '</div>' +
          '</div>' +

          '<div class="d-pos-prices">' +
            '<div class="d-pos-price-card"><div class="d-pp-lbl">ENTRADA</div><div class="d-pp-val">$' + fmtPrice(pos.price) + '</div></div>' +
            '<div class="d-pos-arrow" style="color:' + col + '">' + arrowDir + '</div>' +
            '<div class="d-pos-price-card" style="border-color:' + col + '40"><div class="d-pp-lbl">AHORA</div><div class="d-pp-val" style="color:' + col + '">$' + fmtPrice(pos.currentPrice) + '</div></div>' +
          '</div>' +

          '<div class="d-pos-meta-row">' +
            '<span class="d-meta-item"><span class="d-meta-lbl">Nocional</span> $' + notional.toFixed(0) + '</span>' +
            '<span class="d-meta-item"><span class="d-meta-lbl">Margen</span> $' + (pos.margin || 0).toFixed(2) + '</span>' +
            '<span class="d-meta-item d-meta-liq"><span class="d-meta-lbl">Liquidación</span> $' + fmtPrice(liqPrice) + ' (' + liqPct.toFixed(1) + '%)</span>' +
            (fundEst > 0 ? '<span class="d-meta-item d-meta-fund"><span class="d-meta-lbl">Funding</span> -$' + fundEst.toFixed(3) + '</span>' : '') +
          '</div>' +

          '<div class="d-pos-tpsl-row">' +
            '<span class="d-tpsl-sl-lbl">SL $' + fmtPrice(pos.slPrice) + '</span>' +
            '<div class="d-tpsl-track"><div class="d-tpsl-cur" style="left:' + cur + '%;background:' + col + '"></div></div>' +
            '<span class="d-tpsl-tp-lbl">TP $' + fmtPrice(pos.tpPrice) + '</span>' +
          '</div>' +
          '<div class="d-pos-actions">' +
            '<button class="btn-d-half" data-id="' + escHtml(pos.id) + '">− 50%</button>' +
            '<button class="btn-d-close" data-id="' + escHtml(pos.id) + '">✕ Cerrar todo</button>' +
          '</div></div>';
      }).join('');
      openEl.querySelectorAll('.btn-d-half').forEach(function(b)  { b.addEventListener('click', function() { ptSellDemo(b.dataset.id, 50); }); });
      openEl.querySelectorAll('.btn-d-close').forEach(function(b) { b.addEventListener('click', function() { ptSellDemo(b.dataset.id, 100); }); });
    }
  }

  const hcEl = document.getElementById('dHistCount');
  if (hcEl) hcEl.textContent = closed.length ? '(' + closed.length + ')' : '';

  const histEl = document.getElementById('dHistList');
  if (histEl) {
    if (!closed.length) {
      histEl.innerHTML = '<div class="d-empty">Sin trades cerrados aun</div>';
    } else {
      histEl.innerHTML = closed.slice(0, 50).map(function(t) {
        const col    = (t.pnlUsd || 0) >= 0 ? '#00ff41' : '#f85149';
        const isLong = t.direction !== 'short';
        const dirCol = isLong ? '#00ff41' : '#f85149';
        const eIcon  = t.exitReason === 'tp' ? 'TP' : t.exitReason === 'sl' ? 'SL' : (t._partial ? '1/2' : 'MAN');
        const hm = t.holdMin != null ? (t.holdMin < 60 ? Math.round(t.holdMin) + 'm' : (t.holdMin/60).toFixed(1) + 'h') : '--';
        return '<div class="d-hist-row">' +
          '<span style="color:' + dirCol + ';font-weight:900">' + (isLong ? '▲' : '▼') + '</span>' +
          '<span class="d-hr-sym">' + escHtml(t.symbol) + '</span>' +
          '<span class="d-hr-badge">' + eIcon + '</span>' +
          '<span class="d-hr-entry">$' + fmtPrice(t.price) + '</span>' +
          '<span style="color:#555">→</span>' +
          '<span class="d-hr-ep">$' + fmtPrice(t.exitPrice) + '</span>' +
          '<span class="d-hr-pnl" style="color:' + col + '">' + ((t.pnlPct||0)>=0?'+':'') + (t.pnlPct||0).toFixed(1) + '%</span>' +
          '<span class="d-hr-usd" style="color:' + col + '">' + ((t.pnlUsd||0)>=0?'+':'') + '$' + Math.abs(t.pnlUsd||0).toFixed(2) + '</span>' +
          '<span class="d-hr-hold">' + hm + '</span>' +
        '</div>';
      }).join('');
    }
  }
  ptUpdateHeaderBal();
}
// ═══════════════════════════════════════════════════════════════════
// WATCHLIST
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// HARMONIC PATTERNS — Biblioteca visual + Modal
// ═══════════════════════════════════════════════════════════════════

const HARMONIC_LIBRARY = {
  Gartley: {
    emoji: '🎯', color: '#00d4ff', strength: 4, frequency: 'Alta',
    ratios: { 'AB/XA': '0.618', 'BC/AB': '0.382-0.886', 'CD/BC': '1.13-1.618', 'AD/XA': '0.786' },
    description: 'El patron mas clasico, identificado por Harold Gartley en 1935. Muy preciso: la reversion en D ocurre al 78.6% de la onda XA.',
    signal: 'Entrada en D con SL justo debajo de X (bullish) o encima de X (bearish). R:R tipico 2:1 a 3:1.',
    tip: 'Si AB/XA esta entre 0.60 y 0.64, la fiabilidad aumenta considerablemente.',
    pts_bull: {X:[15,115], A:[60,12], B:[95,60], C:[125,38], D:[172,72]},
    pts_bear: {X:[15,25],  A:[60,125], B:[95,78], C:[125,100], D:[172,64]},
  },
  Bat: {
    emoji: '🦇', color: '#a78bfa', strength: 4, frequency: 'Media-alta',
    ratios: { 'AB/XA': '0.382-0.5', 'BC/AB': '0.382-0.886', 'CD/BC': '1.618-2.618', 'AD/XA': '0.886' },
    description: 'Variante del Gartley con retroceso AB mas corto (38-50%). El punto D llega al 88.6% de XA, mas profundo que el Gartley.',
    signal: 'Stop muy ajustado bajo X. Alta precision en el PRZ. Ideal para traders con disciplina de SL.',
    tip: 'El Bat requiere que AB NO supere el 50% de XA. Si lo supera, puede ser un Gartley.',
    pts_bull: {X:[15,115], A:[60,12], B:[95,68], C:[125,42], D:[172,82]},
    pts_bear: {X:[15,25],  A:[60,125], B:[95,70], C:[125,95], D:[172,55]},
  },
  Butterfly: {
    emoji: '🦋', color: '#f9a8d4', strength: 3, frequency: 'Media',
    ratios: { 'AB/XA': '0.786', 'BC/AB': '0.382-0.886', 'CD/BC': '1.618-2.24', 'AD/XA': '1.27-1.618' },
    description: 'Patron de extension: el punto D supera el nivel X original. Captura extremos del mercado donde otros patrones no llegan.',
    signal: 'D puede extenderse bastante mas alla de X. Confirmar con vela de reversion y volumen antes de entrar.',
    tip: 'La extension de D mas alla de X es su caracteristica clave. Si D no supera X, no es Butterfly.',
    pts_bull: {X:[15,80], A:[60,12], B:[95,58], C:[125,35], D:[172,128]},
    pts_bear: {X:[15,90], A:[60,130], B:[95,70], C:[125,100], D:[172,28]},
  },
  Crab: {
    emoji: '🦀', color: '#f97316', strength: 3, frequency: 'Baja',
    ratios: { 'AB/XA': '0.382-0.618', 'BC/AB': '0.382-0.886', 'CD/BC': '2.618-3.618', 'AD/XA': '1.618' },
    description: 'El mas extremo de los armonicos. D llega al 161.8% de XA marcando minimos/maximos muy claros. Raro pero muy poderoso.',
    signal: 'Las reversiones son profundas y precisas cuando se cumple. Stop amplio: usar posicion pequena.',
    tip: 'La onda CD es muy larga (2.618-3.618x de BC). Esta extension extrema define al Crab.',
    pts_bull: {X:[15,75], A:[60,12], B:[95,52], C:[125,28], D:[172,138]},
    pts_bear: {X:[15,95], A:[60,132], B:[95,80], C:[125,108], D:[172,18]},
  },
  Shark: {
    emoji: '🦈', color: '#00ff41', strength: 4, frequency: 'Media-alta en cripto',
    ratios: { 'AB/XA': '0.446-0.618', 'BC/AB': '1.13-1.618', 'CD/BC': '0.5-0.886', 'AD/XA': '0.886-1.13' },
    description: 'Patron agresivo y moderno. BC SUPERA el punto A: esa es su firma. Detecta reversiones tras impulsos fuertes. Muy comun en cripto.',
    signal: 'Entrada en D cuando BC retrocede 50-88.6% de BC. Confirmar con volumen decreciente en CD.',
    tip: 'La clave del Shark: el punto B debe superar a A. Si B no cruza A, busca otro patron.',
    pts_bull: {X:[15,105], A:[60,25], B:[108,5],  C:[138,40], D:[172,68]},
    pts_bear: {X:[15,55],  A:[60,138], B:[108,128], C:[138,110], D:[172,88]},
  },
  Cypher: {
    emoji: '🔷', color: '#38bdf8', strength: 4, frequency: 'Media en cripto',
    ratios: { 'AB/XA': '0.382-0.618', 'BC/XA': '1.272-1.414', 'CD/XC': '0.786', 'AD/XA': '0.786' },
    description: 'Patron moderno descubierto por Darren Oglesbee. El punto C supera el nivel A (extension 1.272-1.414 de XA). El punto D retrocede 78.6% de XC. Muy frecuente en cripto.',
    signal: 'Entrada en D al 78.6% de XC. SL detras de X. TP1 en C, TP2 en A. Requiere que C supere A para ser valido.',
    tip: 'La firma del Cypher es C por encima de A (bullish) o por debajo de A (bearish). Sin esa condicion no es Cypher.',
    pts_bull: {X:[15,115], A:[60,22], B:[100,82], C:[148,8], D:[182,92]},
    pts_bear: {X:[15,22],  A:[60,115], B:[100,55], C:[148,132], D:[182,48]},
  },
  'AB=CD': {
    emoji: '🔄', color: '#fb923c', strength: 3, frequency: 'Muy alta',
    ratios: { 'BC/AB': '0.618-0.786', 'CD/AB': '1.0 (iguales)', 'Tiempo AB': '≈ Tiempo CD' },
    description: 'El patron armonico mas simple y frecuente. Las piernas AB y CD son iguales en precio y tiempo. Es la base de todos los demas patrones armonicos.',
    signal: 'Entrada en D cuando CD = AB. SL detras del punto inicial. La igualdad AB=CD es la confirmacion. Buscar confluencia con Fibonacci.',
    tip: 'Si CD < AB el patron es debil. Si CD > AB (extension 1.27 o 1.618) la reversion en D es mas potente pero menos frecuente.',
    pts_bull: {X:[15,28], A:[15,28], B:[78,110], C:[118,48], D:[175,118]},
    pts_bear: {X:[15,118], A:[15,118], B:[78,36], C:[118,98], D:[175,28]},
  },
  'Deep Crab': {
    emoji: '🦀', color: '#ef4444', strength: 3, frequency: 'Muy baja',
    ratios: { 'AB/XA': '0.382-0.618', 'BC/AB': '0.382-0.886', 'CD/BC': '2.618-3.618', 'AD/XA': '1.618+' },
    description: 'Version extrema del Crab donde D supera el 161.8% de XA. Marca extremos absolutos de mercado. Muy raro pero con reversiones explosivas cuando se cumple.',
    signal: 'Posicion muy pequena por el riesgo extremo. Stop amplio detras del nivel D+margin. Solo operar con confirmacion de vela en D.',
    tip: 'El Deep Crab aparece en crashes y blow-off tops. En cripto lo veremos en maximos de panico vendedor o comprador extremo.',
    pts_bull: {X:[15,78], A:[55,12], B:[90,55], C:[122,28], D:[175,142]},
    pts_bear: {X:[15,92], A:[55,130], B:[90,87], C:[122,112], D:[175,8]},
  },
  'Alt Bat': {
    emoji: '🦇', color: '#c084fc', strength: 3, frequency: 'Media-baja',
    ratios: { 'AB/XA': '0.382', 'BC/AB': '0.382-0.886', 'CD/BC': '2.0-3.618', 'AD/XA': '1.13' },
    description: 'Variante del Bat donde AB retrocede exactamente 38.2% de XA y D llega al 113% de XA. Menos comun que el Bat estandar pero igual de preciso.',
    signal: 'Entrada en D al 113% de XA. SL ajustado debajo de D. AB debe ser exactamente 0.382 para confirmar Alt Bat vs Bat.',
    tip: 'La diferencia clave con el Bat: AB es 0.382 (no 0.382-0.5) y D va al 1.13 en vez del 0.886. AB muy corto es la primera senal.',
    pts_bull: {X:[15,110], A:[55,28], B:[92,88], C:[128,52], D:[178,124]},
    pts_bear: {X:[15,28],  A:[55,110], B:[92,50], C:[128,86], D:[178,16]},
  },
};

const CHART_PATTERN_LIBRARY = {
  'Doble Suelo': {
    emoji: '🟢', color: '#00ff41', strength: 4, frequency: 'Alta',
    signal_direction: 'bullish',
    keyLevels: ['Linea de cuello: resistencia entre los dos minimos', 'Target: altura del patron proyectada desde el breakout', 'Stop Loss: debajo del segundo minimo'],
    description: 'Patron de reversion alcista con forma de "W". Dos minimos al mismo nivel indican que el mercado rechazo el mismo soporte dos veces. Los vendedores se agotan.',
    signal: 'Entrada en el breakout confirmado sobre la linea de cuello con cierre de vela. Target = altura del patron desde el breakout. No anticipar antes del breakout.',
    tip: 'El segundo minimo suele estar ligeramente por encima del primero. Volumen creciente en el breakout de la neckline confirma el patron.',
  },
  'Doble Techo': {
    emoji: '🔴', color: '#f85149', strength: 4, frequency: 'Alta',
    signal_direction: 'bearish',
    keyLevels: ['Linea de cuello: soporte entre los dos maximos', 'Target: altura del patron proyectada hacia abajo', 'Stop Loss: encima del segundo maximo'],
    description: 'Patron de reversion bajista con forma de "M". Dos maximos al mismo nivel indican que el mercado rechazo la misma resistencia dos veces. Los compradores se agotan.',
    signal: 'Entrada en el breakdown bajo la linea de cuello. Target = altura del patron. Evitar entradas anticipadas en el segundo techo antes de confirmar el breakdown.',
    tip: 'El volumen suele disminuir en el segundo techo respecto al primero. Confirmar con vela bajista de cierre bajo la neckline.',
  },
  'Cabeza y Hombros': {
    emoji: '💀', color: '#f85149', strength: 5, frequency: 'Alta',
    signal_direction: 'bearish',
    keyLevels: ['Linea de cuello: conecta los valles entre hombros y cabeza', 'Target: distancia cabeza-cuello proyectada hacia abajo', 'Stop Loss: encima del hombro derecho'],
    description: 'Uno de los patrones de reversion mas poderosos. Tres picos: el central (cabeza) mas alto que los laterales (hombros). Indica agotamiento del impulso alcista y cambio de tendencia.',
    signal: 'Entrada al romper y cerrar bajo la neckline. El pull-back a la neckline despues del breakout ofrece segunda entrada con mejor R:R.',
    tip: 'El hombro derecho DEBE ser mas bajo que la cabeza. Si el hombro derecho sube mas que la cabeza el patron se invalida.',
  },
  'Cab. y Hombros Inv.': {
    emoji: '🚀', color: '#00ff41', strength: 5, frequency: 'Alta',
    signal_direction: 'bullish',
    keyLevels: ['Linea de cuello: conecta los picos entre hombros y cabeza invertida', 'Target: distancia cuello-cabeza hacia arriba', 'Stop Loss: debajo del hombro derecho invertido'],
    description: 'Version invertida del Cabeza y Hombros. Tres valles: el central mas profundo. Senal de reversion alcista muy confiable. Muy comun en crypto al final de mercados bajistas.',
    signal: 'Entrada al superar la neckline con cierre por encima. Volumen creciente en el breakout es clave. El pull-back a la neckline es segunda oportunidad de entrada.',
    tip: 'El IH&S al final de una tendencia bajista prolongada es uno de los setups mas poderosos. Busca divergencia alcista en RSI en la cabeza.',
  },
  'Triangulo Ascendente': {
    emoji: '📈', color: '#00d4ff', strength: 4, frequency: 'Media-alta',
    signal_direction: 'bullish',
    keyLevels: ['Resistencia horizontal plana: techos al mismo nivel', 'Soporte creciente: minimos cada vez mas altos', 'Target: altura del triangulo desde la resistencia'],
    description: 'Triangulo con resistencia horizontal y soporte ascendente. Los compradores son mas agresivos en cada retroceso. La presion acumulada se libera en un breakout alcista.',
    signal: 'Entrada en el breakout por encima de la resistencia plana. Target = altura del triangulo. Stop debajo del ultimo minimo dentro del triangulo.',
    tip: 'Cuanto mas tiempo cotiza el precio dentro del triangulo mas explosivo suele ser el breakout. Volumen creciente en el breakout confirma.',
  },
  'Triangulo Descendente': {
    emoji: '📉', color: '#f85149', strength: 4, frequency: 'Media-alta',
    signal_direction: 'bearish',
    keyLevels: ['Soporte horizontal plano: minimos al mismo nivel', 'Resistencia decreciente: maximos cada vez mas bajos', 'Target: altura del triangulo hacia abajo'],
    description: 'Triangulo con soporte horizontal y resistencia descendente. Los vendedores son mas agresivos en cada rebote. Patron bajista con alta probabilidad de ruptura a la baja.',
    signal: 'Entrada en el breakdown bajo el soporte plano. Target = altura del triangulo hacia abajo. Stop encima del ultimo maximo.',
    tip: 'Si el precio rompe al ALZA un triangulo descendente es una senal contraria muy fuerte. Siempre confirmar el breakout con cierre de vela.',
  },
  'Triangulo Simetrico': {
    emoji: '🔺', color: '#00d4ff', strength: 3, frequency: 'Alta',
    signal_direction: 'neutral',
    keyLevels: ['Linea superior descendente', 'Linea inferior ascendente', 'Vertice: punto de convergencia donde ocurre el breakout'],
    description: 'Dos lineas convergentes con pendientes opuestas. El precio se comprime hasta el vertice. Puede romper en cualquier direccion: operar en la direccion del breakout, nunca anticipar.',
    signal: 'Esperar el breakout y operar en esa direccion. Target = anchura inicial del triangulo desde el punto de ruptura. Falsos breakouts son comunes: esperar cierre de vela.',
    tip: 'En tendencias alcistas el triangulo simetrico suele ser continuacion (rompe al alza). En tendencias bajistas ruptura a la baja. Volumen confirma.',
  },
  'Bandera Alcista': {
    emoji: '🚩', color: '#00ff41', strength: 4, frequency: 'Alta',
    signal_direction: 'bullish',
    keyLevels: ['Asta (pole): el impulso fuerte previo', 'Canal de consolidacion en leve correccion', 'Target: longitud del asta desde el breakout'],
    description: 'Consolidacion rectangular en leve correccion tras un movimiento alcista fuerte. Los vendedores no tienen fuerza suficiente. El breakout continua el impulso original.',
    signal: 'Entrada en el breakout por encima del canal de la bandera. Target = longitud del asta proyectada. Stop debajo del canal de consolidacion.',
    tip: 'El asta debe ser un movimiento fuerte y rapido. Si la correccion supera el 50% del asta no es una bandera sino una correccion mas compleja.',
  },
  'Bandera Bajista': {
    emoji: '🚩', color: '#f85149', strength: 4, frequency: 'Alta',
    signal_direction: 'bearish',
    keyLevels: ['Asta bajista: el impulso fuerte a la baja', 'Canal de correccion al alza', 'Target: longitud del asta hacia abajo'],
    description: 'Consolidacion en leve rebote tras una caida fuerte. Los compradores no tienen fuerza para revertir. El breakdown continua la tendencia bajista original.',
    signal: 'Entrada en el breakdown bajo el canal de la bandera. Target = longitud del asta bajista. Stop encima del canal de correccion.',
    tip: 'Las banderas bajistas son muy efectivas en crypto durante mercados bajistas. El rebote dentro de la bandera atrapa compradores que quedan atrapados.',
  },
  'Cuna Alcista': {
    emoji: '📐', color: '#f85149', strength: 4, frequency: 'Media',
    signal_direction: 'bearish',
    keyLevels: ['Dos lineas de tendencia alcistas convergentes', 'Breakout a la baja (contra-intuitivo)', 'Target: inicio del patron (base de la cuna)'],
    description: 'A pesar de subir es un patron BAJISTA. Dos lineas alcistas convergentes con momentum decreciente. Cada maximo y minimo es mas alto pero con menos fuerza.',
    signal: 'Vender en el breakdown bajo la linea inferior. Target = base de la cuna. Frecuente en correcciones dentro de tendencias bajistas.',
    tip: 'La cuna alcista es enganosa: parece subida fuerte pero es trampa. Busca divergencia bajista en RSI y volumen decreciente en los maximos.',
  },
  'Cuna Bajista': {
    emoji: '📐', color: '#00ff41', strength: 4, frequency: 'Media',
    signal_direction: 'bullish',
    keyLevels: ['Dos lineas de tendencia bajistas convergentes', 'Breakout al alza (patron de acumulacion)', 'Target: inicio del patron (parte superior de la cuna)'],
    description: 'A pesar de bajar es un patron ALCISTA. Dos lineas bajistas convergentes. Los vendedores pierden fuerza: cada minimo es mas bajo pero con menos momentum.',
    signal: 'Comprar en el breakout por encima de la linea superior. Target = inicio de la cuna. Comun al final de correcciones dentro de tendencias alcistas.',
    tip: 'La cuna bajista es un patron de acumulacion institucional. Volumen decreciente en la formacion y alto en el breakout confirman el patron.',
  },
  'Copa y Asa': {
    emoji: '☕', color: '#00ff41', strength: 4, frequency: 'Media-baja',
    signal_direction: 'bullish',
    keyLevels: ['Borde de la copa: nivel de resistencia clave', 'Fondo de la copa: soporte principal', 'Asa: pequena correccion antes del breakout final'],
    description: 'Formacion en forma de U seguida de una pequena correccion (asa). Popularizado por William O\'Neil. Indica acumulacion prolongada antes de un movimiento alcista importante.',
    signal: 'Entrada en el breakout por encima del borde de la copa. El asa es la ultima correccion que limpia el papel flojo antes del movimiento real.',
    tip: 'La copa debe ser suave y redondeada, no en V. El asa no debe corregir mas del 33% de la copa. Muy confiable en activos con tendencia alcista de largo plazo.',
  },
};

const STUDY_CONCEPTS = [
  { icon:'📐', title:'Que son los Patrones Armonicos',
    body:'Son estructuras XABCD basadas en proporciones de Fibonacci. Harold Gartley los popularizo en 1935, Scott Carney los sistematizo en los 2000s. La idea: el mercado se mueve en proporciones matematicas predecibles que se repiten.' },
  { icon:'📊', title:'La PRZ — Zona de Reversion Potencial',
    body:'La PRZ (Potential Reversal Zone) es el area donde convergen multiples niveles Fibonacci alrededor del punto D. Cuantos mas niveles convergen en la misma zona, mas fuerte es la senal de reversion.' },
  { icon:'🔢', title:'Fibonacci y los patrones',
    body:'Ratios clave: 0.382, 0.5, 0.618 (retrocesos basicos), 0.786 (raiz de 0.618), 0.886 (raiz de 0.786), 1.27 y 1.618 (extensiones, el numero aureo). Cada patron armonico usa combinaciones especificas de estos ratios.' },
  { icon:'⚡', title:'Como operar un patron armonico',
    body:'1. Identificar XABCD con los ratios correctos. 2. Esperar que el precio llegue al PRZ. 3. Buscar confirmacion: vela de reversion, volumen bajo en D, divergencia RSI. 4. Entrada en D con SL detras de X. 5. TP1 al 38.2% de CD, TP2 al 61.8%, TP3 al nivel A.' },
  { icon:'⚠️', title:'Errores comunes',
    body:'Entrar antes de que el precio llegue al PRZ. No esperar confirmacion de vela. Stop demasiado ajustado (el precio puede probar brevemente mas alla de X). Operar patrones con ratios fuera del rango valido: son invalidos.' },
  { icon:'🎯', title:'STAGE: COMPLETANDO vs APPROACHANDO vs FORMANDO',
    body:'COMPLETANDO: precio dentro del 1.5% del PRZ — senal inminente, maxima alerta, preparar la orden. APPROACHANDO PRZ: a 1.5-5% — prepararse y poner alerta en el PRZ. FORMANDO: aun lejos — patron en desarrollo, no actuar todavia.' },
  { icon:'💡', title:'Gartley vs Bat vs Butterfly — las diferencias',
    body:'GARTLEY: AB retrocede 61.8% de XA, D en 78.6%. BAT: AB retrocede solo 38-50%, D llega al 88.6%. BUTTERFLY: AB en 78.6%, D SUPERA X (127-161.8%). CRAB: D llega al 161.8%, el mas extremo. SHARK: B supera A, el mas agresivo.' },
];

function createChartPatternSVG(name) {
  const W = 200, H = 145;
  const wrap = inner => `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:145px;display:block">${inner}</svg>`;
  const bg = `<rect width="${W}" height="${H}" fill="rgba(0,0,0,0.3)" rx="6"/>`;
  const pl = (pts, col, w=1.8) => `<polyline points="${pts.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${col}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"/>`;
  const ln = (x1,y1,x2,y2,col,w=1.2,da='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="${w}"${da?` stroke-dasharray="${da}"`:''}/>`;
  const tx = (x,y,t,col,sz=8,a='middle') => `<text x="${x}" y="${y}" text-anchor="${a}" font-size="${sz}" fill="${col}" font-family="monospace" font-weight="700">${t}</text>`;
  const dot = (x,y,col) => `<circle cx="${x}" cy="${y}" r="3.5" fill="${col}" stroke="#0d1117" stroke-width="1"/>`;
  const G='#00ff41', R='#f85149', B='#00d4ff', Y='#e3b341', DIM='rgba(255,255,255,0.18)';

  const defs = {
    'Doble Suelo': () => bg +
      ln(48,54,188,54,G,1.2,'5,3') + tx(118,50,'CUELLO',G,7) +
      pl([[10,42],[48,110],[88,54],[128,112],[188,24]],G) +
      dot(48,110,G)+dot(128,112,G) +
      tx(48,125,'1',G,9)+tx(128,127,'2',G,9) +
      tx(100,13,'DOBLE SUELO ▲',G,9),
    'Doble Techo': () => bg +
      ln(48,90,188,90,R,1.2,'5,3') + tx(118,102,'CUELLO',R,7) +
      pl([[10,108],[48,36],[88,90],[128,34],[188,120]],R) +
      dot(48,36,R)+dot(128,34,R) +
      tx(48,28,'1',R,9)+tx(128,26,'2',R,9) +
      tx(100,13,'DOBLE TECHO ▼',R,9),
    'Cabeza y Hombros': () => bg +
      ln(52,84,178,84,R,1.2,'5,3') + tx(115,97,'CUELLO',R,7) +
      pl([[10,100],[32,62],[52,84],[82,28],[112,84],[136,62],[178,118]],R) +
      dot(82,28,R) + tx(82,19,'CABEZA',R,7) +
      tx(32,54,'H.Izq',R,7)+tx(136,54,'H.Der',R,7) +
      tx(100,13,'CAB. Y HOMBROS ▼',R,9),
    'Cab. y Hombros Inv.': () => bg +
      ln(52,62,178,62,G,1.2,'5,3') + tx(115,55,'CUELLO',G,7) +
      pl([[10,46],[32,84],[52,62],[82,118],[112,62],[136,84],[178,28]],G) +
      dot(82,118,G) + tx(82,135,'CABEZA',G,7) +
      tx(32,100,'H.Izq',G,7)+tx(136,100,'H.Der',G,7) +
      tx(100,13,'C.H. INVERTIDO ▲',G,9),
    'Triangulo Ascendente': () => bg +
      ln(10,44,182,44,Y,1.2,'4,3') +
      ln(10,116,155,44,DIM,1,'4,3') +
      pl([[10,116],[40,44],[62,88],[92,44],[112,74],[142,44],[158,38],[190,16]],G) +
      tx(100,13,'T. ASCENDENTE ▲',G,9),
    'Triangulo Descendente': () => bg +
      ln(10,110,182,110,Y,1.2,'4,3') +
      ln(10,32,155,110,DIM,1,'4,3') +
      pl([[10,32],[40,110],[62,70],[92,110],[112,80],[142,110],[158,115],[190,132]],R) +
      tx(100,13,'T. DESCENDENTE ▼',R,9),
    'Triangulo Simetrico': () => bg +
      ln(10,32,162,72,DIM,1,'4,3') +
      ln(10,112,162,72,DIM,1,'4,3') +
      pl([[10,32],[40,104],[68,50],[98,88],[128,64],[148,72],[185,38]],B) +
      tx(172,64,'▶',G,11) +
      tx(100,13,'T. SIMETRICO',B,9),
    'Bandera Alcista': () => bg +
      pl([[10,120],[62,32]],'rgba(0,255,65,0.4)',3) +
      ln(62,32,142,36,G,1,'3,2') + ln(62,50,142,54,G,1,'3,2') +
      pl([[62,32],[80,46],[100,36],[122,50],[142,40]],G) +
      pl([[142,40],[188,6]],G,2.4) +
      tx(100,13,'BANDERA ALCISTA ▲',G,9),
    'Bandera Bajista': () => bg +
      pl([[10,16],[62,104]],'rgba(248,81,73,0.4)',3) +
      ln(62,90,142,86,R,1,'3,2') + ln(62,108,142,100,R,1,'3,2') +
      pl([[62,104],[80,90],[100,100],[122,86],[142,96]],R) +
      pl([[142,96],[188,132]],R,2.4) +
      tx(100,13,'BANDERA BAJISTA ▼',R,9),
    'Cuna Alcista': () => bg +
      ln(10,78,160,30,R,1,'3,2') + ln(10,108,160,60,R,1,'3,2') +
      pl([[10,78],[28,108],[56,72],[84,98],[110,68],[136,88],[158,62],[188,116]],R) +
      tx(172,112,'↘',R,13) +
      tx(100,13,'CUNA ALCISTA ▼',R,9),
    'Cuna Bajista': () => bg +
      ln(10,36,160,78,G,1,'3,2') + ln(10,66,160,108,G,1,'3,2') +
      pl([[10,36],[28,66],[56,50],[84,72],[110,58],[136,80],[158,70],[188,30]],G) +
      tx(172,30,'↗',G,13) +
      tx(100,13,'CUNA BAJISTA ▲',G,9),
    'Copa y Asa': () => bg +
      ln(10,38,168,38,G,1.2,'4,3') +
      pl([[10,38],[26,68],[46,96],[70,116],[96,124],[120,116],[144,96],[162,68],[168,38]],G) +
      pl([[168,38],[176,54],[184,44]],G) +
      pl([[184,44],[196,18]],G,2.4) +
      tx(88,135,'Copa',G,8) + tx(174,40,'Asa',G,7) +
      tx(100,13,'COPA Y ASA ▲',G,9),
  };
  const fn = defs[name];
  return fn ? wrap(fn()) : '';
}

function createPatternSVG(patName, isBullish) {
  const lib = HARMONIC_LIBRARY[patName];
  if (!lib) return '';
  const W = 200, H = 145;
  const pts = isBullish ? lib.pts_bull : lib.pts_bear;
  const { X: px, A: pa, B: pb, C: pc, D: pd } = pts;
  const col = isBullish ? '#00ff41' : '#f85149';

  const seg = (p1, p2, c) =>
    `<line x1="${p1[0]}" y1="${p1[1]}" x2="${p2[0]}" y2="${p2[1]}" stroke="${c}" stroke-width="1.8" stroke-linecap="round"/>`;
  const node = (p, lbl, c) => {
    const above = p[1] > H / 2;
    const dy = above ? -8 : 13;
    const anchor = p[0] < 30 ? 'start' : p[0] > 160 ? 'end' : 'middle';
    return `<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="${c}" stroke="#0d1117" stroke-width="1.5"/>` +
      `<text x="${p[0]}" y="${p[1] + dy}" text-anchor="${anchor}" font-size="10" font-weight="900" fill="${c}" font-family="monospace">${lbl}</text>`;
  };
  const przY = Math.min(pd[1] - 10, H - 20);
  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:145px;display:block">` +
    `<rect x="0" y="0" width="${W}" height="${H}" fill="rgba(0,0,0,0.3)" rx="6"/>` +
    `<rect x="${pd[0] - 22}" y="${przY}" width="44" height="20" rx="4" fill="${col}" opacity="0.15"/>` +
    `<text x="${pd[0]}" y="${przY - 3}" text-anchor="middle" font-size="7.5" fill="${col}" opacity="0.8" font-family="monospace">PRZ</text>` +
    seg(px, pa, 'rgba(255,255,255,0.25)') + seg(pa, pb, 'rgba(255,255,255,0.25)') +
    seg(pb, pc, 'rgba(255,255,255,0.25)') + seg(pc, pd, col) +
    node(px, 'X', '#8b949e') + node(pa, 'A', lib.color) + node(pb, 'B', lib.color) +
    node(pc, 'C', lib.color) + node(pd, 'D', col) +
    `</svg>`;
}

function renderPatternsTab() {
  const detEl = document.getElementById('ptDetectedList');
  const subEl = document.getElementById('ptDetectedSub');
  const harms = window._lastAdvanced?.harmonics || [];
  window._ptHarms = harms;

  if (detEl) {
    if (!harms.length) {
      detEl.innerHTML = '<div class="pt-empty">Sin patrones armonicos detectados en el analisis actual</div>';
      if (subEl) subEl.textContent = 'ninguno detectado';
    } else {
      if (subEl) subEl.textContent = `${harms.length} detectado${harms.length > 1 ? 's' : ''}`;
      detEl.innerHTML = harms.map((h, idx) => {
        const lib = HARMONIC_LIBRARY[h.name] || {};
        const col = h.type === 'bullish' ? '#00ff41' : '#f85149';
        const stageCol = h.stage === 'COMPLETANDO' ? '#ff4500' : h.stage.includes('APPROACH') ? '#e3b341' : '#888';
        return `<div class="pt-detected-card ${h.type}" data-idx="${idx}">` +
          `<div class="ptd-top">` +
          `<span class="ptd-emoji">${lib.emoji || '🔺'}</span>` +
          `<span class="ptd-name" style="color:${lib.color || col}">${escHtml(h.name)}</span>` +
          `<span class="ptd-dir" style="color:${col}">${h.type === 'bullish' ? '▲ ALCISTA' : '▼ BAJISTA'}</span>` +
          `<span class="ptd-stage" style="color:${stageCol}">${escHtml(h.stage)}</span>` +
          `<span class="ptd-prob">${h.prob}%</span>` +
          `</div>` +
          `<div class="ptd-prz">PRZ ${fmtPrice(h.dZone?.low)} – ${fmtPrice(h.dZone?.high)} &nbsp;·&nbsp; ${h.distPct}% de distancia</div>` +
          `<div class="ptd-cta">Toca para ver diagrama →</div>` +
          `</div>`;
      }).join('');
      detEl.querySelectorAll('.pt-detected-card').forEach(card => {
        card.addEventListener('click', () => {
          const h = window._ptHarms?.[parseInt(card.dataset.idx)];
          if (h) showPatternModal(h.name, h, 'harmonic');
        });
      });
    }
  }

  const libEl = document.getElementById('ptLibrary');
  if (libEl && !libEl.children.length) {
    const mkCard = (name, p, libtype, svgFn) => {
      const stars = '★'.repeat(p.strength) + '☆'.repeat(5 - p.strength);
      const sub = libtype === 'harmonic' ? p.frequency
        : (p.signal_direction === 'bullish' ? '▲ Alcista' : p.signal_direction === 'bearish' ? '▼ Bajista' : '◆ Neutro');
      const subCol = libtype === 'classic'
        ? (p.signal_direction === 'bullish' ? '#00ff41' : p.signal_direction === 'bearish' ? '#f85149' : '#00d4ff')
        : 'rgba(255,255,255,0.5)';
      return `<div class="pt-lib-card" data-name="${name}" data-libtype="${libtype}">` +
        `<div class="pt-lib-hdr">` +
        `<span class="pt-lib-emoji">${p.emoji}</span>` +
        `<div class="pt-lib-title-wrap">` +
        `<span class="pt-lib-name" style="color:${p.color}">${name}</span>` +
        `<span class="pt-lib-freq" style="color:${subCol}">${sub}</span>` +
        `</div><span class="pt-lib-stars" style="color:${p.color}">${stars}</span>` +
        `</div>` +
        `<div class="pt-lib-svg">${svgFn()}</div>` +
        `<div class="pt-lib-desc">${escHtml(p.description.substring(0,90))}…</div>` +
        `</div>`;
    };

    const harmKeys = Object.keys(HARMONIC_LIBRARY);
    const classicKeys = Object.keys(CHART_PATTERN_LIBRARY);

    libEl.innerHTML =
      `<div class="pt-subsec-hdr">📐 Patrones Armónicos <span class="pt-subsec-count">${harmKeys.length}</span></div>` +
      `<div class="pt-lib-grid">` +
      harmKeys.map(n => mkCard(n, HARMONIC_LIBRARY[n], 'harmonic', () => createPatternSVG(n, true))).join('') +
      `</div>` +
      `<div class="pt-subsec-hdr" style="margin-top:8px">📊 Patrones Clásicos <span class="pt-subsec-count">${classicKeys.length}</span></div>` +
      `<div class="pt-lib-grid">` +
      classicKeys.map(n => mkCard(n, CHART_PATTERN_LIBRARY[n], 'classic', () => createChartPatternSVG(n))).join('') +
      `</div>`;

    libEl.querySelectorAll('.pt-lib-card').forEach(card => {
      card.addEventListener('click', () => showPatternModal(card.dataset.name, null, card.dataset.libtype));
    });
  }

  const conEl = document.getElementById('ptConcepts');
  if (conEl && !conEl.children.length) {
    conEl.innerHTML = STUDY_CONCEPTS.map(c =>
      `<div class="pt-concept-card">` +
      `<div class="pt-concept-hdr"><span>${c.icon}</span><span class="pt-concept-title">${escHtml(c.title)}</span></div>` +
      `<div class="pt-concept-body">${escHtml(c.body)}</div>` +
      `</div>`
    ).join('');
  }
}

function showPatternModal(name, detected, libtype) {
  const isClassic = libtype === 'classic';
  const lib = isClassic ? CHART_PATTERN_LIBRARY[name] : HARMONIC_LIBRARY[name];
  if (!lib) return;

  if (!isClassic && typeof detected === 'string') {
    try { detected = JSON.parse(detected.replace(/'/g,'"')); } catch { detected = null; }
  }
  const isBull = isClassic ? lib.signal_direction !== 'bearish' : (detected ? detected.type === 'bullish' : true);

  const svgHtml = isClassic ? createChartPatternSVG(name) : createPatternSVG(name, isBull);

  const detHtml = (!isClassic && detected) ? `<div class="pm-detected-info">` +
    `<div class="pm-di-row"><span>Estado:</span><span style="font-weight:700;color:${detected.stage === 'COMPLETANDO' ? '#ff4500' : '#e3b341'}">${escHtml(detected.stage)}</span></div>` +
    `<div class="pm-di-row"><span>PRZ:</span><span>${fmtPrice(detected.dZone?.low)} – ${fmtPrice(detected.dZone?.high)}</span></div>` +
    `<div class="pm-di-row"><span>SL:</span><span style="color:#f85149">${fmtPrice(detected.sl)}</span></div>` +
    `<div class="pm-di-row"><span>TP1:</span><span style="color:#00ff41">${fmtPrice(detected.tp1)}</span></div>` +
    `<div class="pm-di-row"><span>TP2:</span><span style="color:#00ff41">${fmtPrice(detected.tp2)}</span></div>` +
    `<div class="pm-di-row"><span>Dist. PRZ:</span><span>${detected.distPct}%</span></div>` +
    `</div>` : '';

  const levelsHtml = isClassic
    ? lib.keyLevels.map(l => `<div class="pm-ratio"><span class="pm-ratio-v" style="color:var(--text1)">${l}</span></div>`).join('')
    : Object.entries(lib.ratios).map(([k,v]) => `<div class="pm-ratio"><span class="pm-ratio-k">${k}</span><span class="pm-ratio-v">${v}</span></div>`).join('');

  const dirTag = isClassic
    ? (lib.signal_direction === 'bullish' ? ' <span style="color:#00ff41">▲ ALCISTA</span>' : lib.signal_direction === 'bearish' ? ' <span style="color:#f85149">▼ BAJISTA</span>' : ' <span style="color:#00d4ff">◆ NEUTRO</span>')
    : (detected ? (isBull ? ' <span style="color:#00ff41">▲ ALCISTA</span>' : ' <span style="color:#f85149">▼ BAJISTA</span>') : '');

  document.getElementById('pmContent').innerHTML =
    `<div class="pm-header" style="border-color:${lib.color}40">` +
    `<span class="pm-emoji">${lib.emoji}</span>` +
    `<div><div class="pm-name" style="color:${lib.color}">${name}${dirTag}</div>` +
    `<div class="pm-freq">${lib.frequency} &nbsp;·&nbsp; ${'★'.repeat(lib.strength)}${'☆'.repeat(5 - lib.strength)}</div></div>` +
    `</div>` +
    `<div class="pm-svg-wrap">${svgHtml}</div>` +
    detHtml +
    `<div class="pm-section-title">${isClassic ? 'Niveles Clave' : 'Ratios de Fibonacci'}</div><div class="pm-ratios">${levelsHtml}</div>` +
    `<div class="pm-section-title">Que es el ${name}?</div><div class="pm-text">${escHtml(lib.description)}</div>` +
    `<div class="pm-section-title">Como operar?</div><div class="pm-text">${escHtml(lib.signal)}</div>` +
    `<div class="pm-tip">💡 ${escHtml(lib.tip)}</div>`;

  document.getElementById('pmOverlay').style.display = 'block';
  document.getElementById('pmModal').style.display = 'flex';
  document.getElementById('pmModal').scrollTop = 0;
}

function closePatternModal() {
  document.getElementById('pmOverlay').style.display = 'none';
  document.getElementById('pmModal').style.display = 'none';
}

// ── Checklist de condiciones actuales (signal + filtro) ──────────────────────
function btBuildConditionChecklist(ind, candles, i, sig, filt, dir) {
  if (!ind || !candles || i < 1) return null;
  const items = [];
  const n2 = (v) => v == null || isNaN(v);
  const c = candles[i].close;
  const pc = candles[i-1].close;
  
  // Signal conditions
  if (sig === 'elder_imp') {
    const e13 = ind.ema13[i], e13p = ind.ema13[i-1];
    const macd = ind.macdHist[i], macdp = ind.macdHist[i-1];
    const eSlope = !n2(e13) && !n2(e13p) ? ((e13 - e13p) / e13p * 100).toFixed(3) : null;
    const mSlope = !n2(macd) && !n2(macdp) ? (macd - macdp).toFixed(4) : null;
    if (dir === 'LONG') {
      items.push({ ok: eSlope > 0, label: 'EMA13 subiendo', detail: eSlope ? `slope: ${eSlope > 0 ? '+' : ''}${eSlope}%` : 'n/a' });
      items.push({ ok: macd > 0, label: 'MACD histogram > 0', detail: macd != null ? `actual: ${macd.toFixed(4)}` : 'n/a' });
    } else {
      items.push({ ok: eSlope < 0, label: 'EMA13 bajando', detail: eSlope ? `slope: ${eSlope}%` : 'n/a' });
      items.push({ ok: macd < 0, label: 'MACD histogram < 0', detail: macd != null ? `actual: ${macd.toFixed(4)}` : 'n/a' });
    }
  } else if (sig === 'rsi_os30' || sig === 'rsi_os35' || sig === 'rsi_os40') {
    const rsi = ind.rsi[i];
    const thr = sig === 'rsi_os30' ? 30 : sig === 'rsi_os35' ? 35 : 40;
    if (dir === 'LONG') {
      const gap = rsi != null ? (thr - rsi).toFixed(1) : '?';
      items.push({ ok: rsi < thr, label: `RSI < ${thr} (zona sobreventa)`, detail: rsi != null ? `actual: ${rsi.toFixed(1)}${rsi < thr ? '' : `, necesita bajar ${gap} pts`}` : 'n/a' });
    } else {
      const thr2 = 100 - thr;
      const gap = rsi != null ? (rsi - thr2).toFixed(1) : '?';
      items.push({ ok: rsi > thr2, label: `RSI > ${thr2} (zona sobrecompra)`, detail: rsi != null ? `actual: ${rsi.toFixed(1)}${rsi > thr2 ? '' : `, necesita subir ${gap} pts`}` : 'n/a' });
    }
  } else if (sig === 'macd_flip') {
    const macd = ind.macdHist[i], macdp = ind.macdHist[i-1];
    if (dir === 'LONG') {
      items.push({ ok: macdp < 0 && macd > 0, label: 'MACD flip: - → +', detail: macd != null && macdp != null ? `prev: ${macdp.toFixed(4)}, ahora: ${macd.toFixed(4)}` : 'n/a' });
    } else {
      items.push({ ok: macdp > 0 && macd < 0, label: 'MACD flip: + → -', detail: macd != null && macdp != null ? `prev: ${macdp.toFixed(4)}, ahora: ${macd.toFixed(4)}` : 'n/a' });
    }
  } else if (sig === 'bb_bounce') {
    const bb = { up: ind.bbUpper[i], mid: ind.bbMid[i], low: ind.bbLower[i] };
    if (dir === 'LONG') {
      const gap = bb.low != null ? (c - bb.low).toFixed(2) : '?';
      items.push({ ok: c <= bb.low, label: 'Precio tocó BB inferior', detail: bb.low ? `BB Low: $${fmtPrice(bb.low)}, precio: $${fmtPrice(c)}${c <= bb.low ? '' : `, falta: $${gap}`}` : 'n/a' });
    } else {
      const gap = bb.up != null ? (bb.up - c).toFixed(2) : '?';
      items.push({ ok: c >= bb.up, label: 'Precio tocó BB superior', detail: bb.up ? `BB Up: $${fmtPrice(bb.up)}, precio: $${fmtPrice(c)}${c >= bb.up ? '' : `, falta: $${gap}`}` : 'n/a' });
    }
  } else if (sig === 'price_ema20') {
    const e20 = ind.ema20[i];
    if (dir === 'LONG') {
      const gap = e20 != null ? ((c - e20) / e20 * 100).toFixed(2) : '?';
      items.push({ ok: c < e20, label: 'Precio < EMA20 (preparando compra)', detail: e20 ? `EMA20: $${fmtPrice(e20)}, precio: $${fmtPrice(c)}, dist: ${gap}%` : 'n/a' });
    } else {
      const gap = e20 != null ? ((e20 - c) / e20 * 100).toFixed(2) : '?';
      items.push({ ok: c > e20, label: 'Precio > EMA20 (preparando venta)', detail: e20 ? `EMA20: $${fmtPrice(e20)}, precio: $${fmtPrice(c)}, dist: +${gap}%` : 'n/a' });
    }
  } else if (sig === 'ema_8_21') {
    const e8 = ind.ema8[i], e21 = ind.ema21[i];
    if (dir === 'LONG') {
      const gap = e8 != null && e21 != null ? ((e21 - e8) / e8 * 100).toFixed(2) : '?';
      items.push({ ok: e8 < e21, label: 'EMA8 < EMA21 (preparando cruce alcista)', detail: e8 && e21 ? `EMA8: $${fmtPrice(e8)}, EMA21: $${fmtPrice(e21)}, gap: ${gap}%` : 'n/a' });
    } else {
      const gap = e8 != null && e21 != null ? ((e8 - e21) / e8 * 100).toFixed(2) : '?';
      items.push({ ok: e8 > e21, label: 'EMA8 > EMA21 (preparando cruce bajista)', detail: e8 && e21 ? `EMA8: $${fmtPrice(e8)}, EMA21: $${fmtPrice(e21)}, gap: +${gap}%` : 'n/a' });
    }
  } else if (sig === 'cci_cross') {
    const cci = ind.cci[i];
    if (dir === 'LONG') {
      const gap = cci != null && cci < -100 ? (cci - (-100)).toFixed(1) : cci != null ? (-100 - cci).toFixed(1) : '?';
      items.push({ ok: cci <= -100, label: 'CCI en zona ≤-100', detail: cci != null ? `actual: ${cci.toFixed(1)}${cci <= -100 ? ' — espera cruce al alza' : `, faltan ${gap} pts`}` : 'n/a' });
    } else {
      const gap = cci != null && cci > 100 ? (cci - 100).toFixed(1) : cci != null ? (100 - cci).toFixed(1) : '?';
      items.push({ ok: cci >= 100, label: 'CCI en zona ≥100', detail: cci != null ? `actual: ${cci.toFixed(1)}${cci >= 100 ? ' — espera cruce a la baja' : `, faltan ${gap} pts`}` : 'n/a' });
    }
  } else if (sig === 'williams_r') {
    const wr = ind.williamsR[i];
    if (dir === 'LONG') {
      const gap = wr != null && wr < -80 ? (wr - (-80)).toFixed(1) : wr != null ? (-80 - wr).toFixed(1) : '?';
      items.push({ ok: wr <= -80, label: 'Williams %R en zona ≤-80', detail: wr != null ? `actual: ${wr.toFixed(1)}${wr <= -80 ? ' — espera cruce al alza' : `, faltan ${gap} pts`}` : 'n/a' });
    } else {
      const gap = wr != null && wr > -20 ? (wr - (-20)).toFixed(1) : wr != null ? (-20 - wr).toFixed(1) : '?';
      items.push({ ok: wr >= -20, label: 'Williams %R en zona ≥-20', detail: wr != null ? `actual: ${wr.toFixed(1)}${wr >= -20 ? ' — espera cruce a la baja' : `, faltan ${gap} pts`}` : 'n/a' });
    }
  } else if (sig === 'stoch_cross') {
    const k = ind.stochK[i], d = ind.stochD[i];
    if (dir === 'LONG') {
      const inZone = k <= 20;
      const gap = k != null && k > 20 ? (k - 20).toFixed(1) : 0;
      items.push({ ok: inZone, label: 'Stoch K en zona ≤20', detail: k != null ? `K: ${k.toFixed(1)}${inZone ? ', espera cruce K>D' : `, faltan bajar ${gap} pts`}` : 'n/a' });
    } else {
      const inZone = k >= 80;
      const gap = k != null && k < 80 ? (80 - k).toFixed(1) : 0;
      items.push({ ok: inZone, label: 'Stoch K en zona ≥80', detail: k != null ? `K: ${k.toFixed(1)}${inZone ? ', espera cruce K<D' : `, faltan subir ${gap} pts`}` : 'n/a' });
    }
  } else if (sig === 'trix_cross') {
    const trix = ind.trix[i], sig = ind.trixSignal[i];
    if (dir === 'LONG') {
      const gap = trix != null && sig != null ? ((sig - trix) * 1000).toFixed(2) : '?';
      items.push({ ok: trix < sig, label: 'TRIX < Signal (preparando cruce alcista)', detail: trix && sig ? `TRIX: ${(trix*100).toFixed(3)}%, Signal: ${(sig*100).toFixed(3)}%, gap: ${gap}` : 'n/a' });
    } else {
      const gap = trix != null && sig != null ? ((trix - sig) * 1000).toFixed(2) : '?';
      items.push({ ok: trix > sig, label: 'TRIX > Signal (preparando cruce bajista)', detail: trix && sig ? `TRIX: ${(trix*100).toFixed(3)}%, Signal: ${(sig*100).toFixed(3)}%, gap: ${gap}` : 'n/a' });
    }
  } else if (sig === 'ema_20_50') {
    const e20 = ind.ema20[i], e50 = ind.ema50[i];
    if (dir === 'LONG') {
      const gap = e20 != null && e50 != null ? ((e50 - e20) / e20 * 100).toFixed(2) : '?';
      items.push({ ok: e20 < e50, label: 'EMA20 < EMA50 (preparando golden cross)', detail: e20 && e50 ? `EMA20: $${fmtPrice(e20)}, EMA50: $${fmtPrice(e50)}, gap: ${gap}%` : 'n/a' });
    } else {
      const gap = e20 != null && e50 != null ? ((e20 - e50) / e20 * 100).toFixed(2) : '?';
      items.push({ ok: e20 > e50, label: 'EMA20 > EMA50 (preparando death cross)', detail: e20 && e50 ? `EMA20: $${fmtPrice(e20)}, EMA50: $${fmtPrice(e50)}, gap: +${gap}%` : 'n/a' });
    }
  } else if (sig === 'rsi3_pull') {
    const rsi3 = ind.rsi3[i], e50 = ind.ema50[i], e50p = ind.ema50[i-1];
    const uptrend = e50 && e50p && e50 > e50p;
    if (dir === 'LONG') {
      items.push({ ok: uptrend, label: 'En uptrend (EMA50 subiendo)', detail: e50 && e50p ? `EMA50: $${fmtPrice(e50)} ${uptrend ? '↑' : '↓'}` : 'n/a' });
      const gap = rsi3 != null && rsi3 > 20 ? (rsi3 - 20).toFixed(1) : 0;
      items.push({ ok: rsi3 < 20, label: 'RSI(3) < 20 (pullback)', detail: rsi3 != null ? `actual: ${rsi3.toFixed(1)}${rsi3 < 20 ? ', espera recuperación >20' : `, falta bajar ${gap} pts`}` : 'n/a' });
    }
  } else if (sig === 'connors_rsi') {
    const crsi = ind.connorsRSI[i];
    if (dir === 'LONG') {
      const gap = crsi != null && crsi > 10 ? (crsi - 10).toFixed(1) : 0;
      items.push({ ok: crsi < 10, label: 'ConnorsRSI < 10 (sobreventa extrema)', detail: crsi != null ? `actual: ${crsi.toFixed(1)}${crsi < 10 ? '' : `, falta bajar ${gap} pts`}` : 'n/a' });
    } else {
      const gap = crsi != null && crsi < 90 ? (90 - crsi).toFixed(1) : 0;
      items.push({ ok: crsi > 90, label: 'ConnorsRSI > 90 (sobrecompra extrema)', detail: crsi != null ? `actual: ${crsi.toFixed(1)}${crsi > 90 ? '' : `, falta subir ${gap} pts`}` : 'n/a' });
    }
  } else if (sig === 'five_ribbon') {
    const e5=ind.ema5[i], e8=ind.ema8[i], e13=ind.ema13[i], e21=ind.ema21[i], e34=ind.ema34[i];
    if (dir === 'LONG') {
      items.push({ ok: e5>e8, label: 'EMA5 > EMA8', detail: e5&&e8 ? `${fmtPrice(e5)} vs ${fmtPrice(e8)}` : 'n/a' });
      items.push({ ok: e8>e13, label: 'EMA8 > EMA13', detail: e8&&e13 ? `${fmtPrice(e8)} vs ${fmtPrice(e13)}` : 'n/a' });
      items.push({ ok: e13>e21, label: 'EMA13 > EMA21', detail: e13&&e21 ? `${fmtPrice(e13)} vs ${fmtPrice(e21)}` : 'n/a' });
      items.push({ ok: e21>e34, label: 'EMA21 > EMA34', detail: e21&&e34 ? `${fmtPrice(e21)} vs ${fmtPrice(e34)}` : 'n/a' });
    } else {
      items.push({ ok: e5<e8, label: 'EMA5 < EMA8', detail: e5&&e8 ? `${fmtPrice(e5)} vs ${fmtPrice(e8)}` : 'n/a' });
      items.push({ ok: e8<e13, label: 'EMA8 < EMA13', detail: e8&&e13 ? `${fmtPrice(e8)} vs ${fmtPrice(e13)}` : 'n/a' });
      items.push({ ok: e13<e21, label: 'EMA13 < EMA21', detail: e13&&e21 ? `${fmtPrice(e13)} vs ${fmtPrice(e21)}` : 'n/a' });
      items.push({ ok: e21<e34, label: 'EMA21 < EMA34', detail: e21&&e34 ? `${fmtPrice(e21)} vs ${fmtPrice(e34)}` : 'n/a' });
    }
  } else if (sig === 'zscore_rev') {
    const z = ind.zscore[i];
    if (dir === 'LONG') {
      const gap = z != null && z > -2 ? (z - (-2)).toFixed(2) : 0;
      items.push({ ok: z < -2, label: 'Z-score < -2 (precio muy bajo)', detail: z != null ? `actual: ${z.toFixed(2)}σ${z < -2 ? ', espera reversión' : `, falta bajar ${gap}σ`}` : 'n/a' });
    } else {
      const gap = z != null && z < 2 ? (2 - z).toFixed(2) : 0;
      items.push({ ok: z > 2, label: 'Z-score > 2 (precio muy alto)', detail: z != null ? `actual: ${z.toFixed(2)}σ${z > 2 ? ', espera reversión' : `, falta subir ${gap}σ`}` : 'n/a' });
    }
  } else if (sig === 'adx_di') {
    const adx = ind.adx[i], diPlus = ind.diPlus[i], diMinus = ind.diMinus[i];
    const strong = adx >= 25;
    if (dir === 'LONG') {
      items.push({ ok: strong, label: 'ADX ≥ 25 (tendencia fuerte)', detail: adx != null ? `ADX: ${adx.toFixed(1)}` : 'n/a' });
      items.push({ ok: diPlus > diMinus, label: '+DI > -DI', detail: diPlus && diMinus ? `+DI: ${diPlus.toFixed(1)}, -DI: ${diMinus.toFixed(1)}` : 'n/a' });
    } else {
      items.push({ ok: strong, label: 'ADX ≥ 25 (tendencia fuerte)', detail: adx != null ? `ADX: ${adx.toFixed(1)}` : 'n/a' });
      items.push({ ok: diMinus > diPlus, label: '-DI > +DI', detail: diPlus && diMinus ? `+DI: ${diPlus.toFixed(1)}, -DI: ${diMinus.toFixed(1)}` : 'n/a' });
    }
  } else if (sig === 'donchian_10') {
    const hi = ind.donchianHi[i], lo = ind.donchianLo[i];
    if (dir === 'LONG') {
      const gap = lo != null ? (c - lo).toFixed(2) : '?';
      items.push({ ok: c <= lo, label: 'Precio tocó Donchian Low (10)', detail: lo ? `Low: $${fmtPrice(lo)}, precio: $${fmtPrice(c)}${c <= lo ? '' : `, falta: $${gap}`}` : 'n/a' });
    } else {
      const gap = hi != null ? (hi - c).toFixed(2) : '?';
      items.push({ ok: c >= hi, label: 'Precio tocó Donchian High (10)', detail: hi ? `High: $${fmtPrice(hi)}, precio: $${fmtPrice(c)}${c >= hi ? '' : `, falta: $${gap}`}` : 'n/a' });
    }
  } else if (sig === 'keltner') {
    const mid = ind.keltnerMid[i], up = ind.keltnerUpper[i], lo = ind.keltnerLower[i];
    if (dir === 'LONG') {
      const gap = lo != null ? (c - lo).toFixed(2) : '?';
      items.push({ ok: c <= lo, label: 'Precio tocó Keltner Lower', detail: lo ? `Lower: $${fmtPrice(lo)}, precio: $${fmtPrice(c)}${c <= lo ? '' : `, falta: $${gap}`}` : 'n/a' });
    } else {
      const gap = up != null ? (up - c).toFixed(2) : '?';
      items.push({ ok: c >= up, label: 'Precio tocó Keltner Upper', detail: up ? `Upper: $${fmtPrice(up)}, precio: $${fmtPrice(c)}${c >= up ? '' : `, falta: $${gap}`}` : 'n/a' });
    }
  } else if (sig === 'consec_5') {
    let count = 0;
    if (dir === 'LONG') {
      for (let j = i; j >= Math.max(0, i-6); j--) {
        if (candles[j].close < candles[j].open) count++; else break;
      }
      items.push({ ok: count >= 5, label: '5+ velas bajistas consecutivas', detail: `actual: ${count} velas bajistas${count >= 5 ? ', espera reversión' : ''}` });
    } else {
      for (let j = i; j >= Math.max(0, i-6); j--) {
        if (candles[j].close > candles[j].open) count++; else break;
      }
      items.push({ ok: count >= 5, label: '5+ velas alcistas consecutivas', detail: `actual: ${count} velas alcistas${count >= 5 ? ', espera reversión' : ''}` });
    }
  }
  
  // Filter conditions
  if (filt === 'rsi_zone') {
    const rsi = ind.rsi[i];
    if (dir === 'LONG') {
      const inZone = rsi >= 40 && rsi <= 68;
      const gap = rsi < 40 ? (40 - rsi).toFixed(1) : rsi > 68 ? (rsi - 68).toFixed(1) : 0;
      items.push({ ok: inZone, label: 'RSI en zona LONG (40-68)', detail: rsi != null ? `actual: ${rsi.toFixed(1)}${inZone ? '' : rsi < 40 ? `, faltan ${gap} pts para entrar` : `, sobran ${gap} pts (demasiado alto)`}` : 'n/a' });
    } else {
      const inZone = rsi >= 32 && rsi <= 60;
      const gap = rsi < 32 ? (32 - rsi).toFixed(1) : rsi > 60 ? (rsi - 60).toFixed(1) : 0;
      items.push({ ok: inZone, label: 'RSI en zona SHORT (32-60)', detail: rsi != null ? `actual: ${rsi.toFixed(1)}${inZone ? '' : rsi < 32 ? `, faltan ${gap} pts (demasiado bajo)` : `, sobran ${gap} pts para entrar`}` : 'n/a' });
    }
  } else if (filt === 'ema50_align') {
    const e50 = ind.ema50[i];
    if (dir === 'LONG') {
      const gap = e50 != null ? ((c - e50) / e50 * 100).toFixed(2) : '?';
      items.push({ ok: c > e50, label: 'Precio > EMA50 (tendencia alcista)', detail: e50 ? `EMA50: $${fmtPrice(e50)}, precio: $${fmtPrice(c)}, dist: ${gap}%` : 'n/a' });
    } else {
      const gap = e50 != null ? ((e50 - c) / e50 * 100).toFixed(2) : '?';
      items.push({ ok: c < e50, label: 'Precio < EMA50 (tendencia bajista)', detail: e50 ? `EMA50: $${fmtPrice(e50)}, precio: $${fmtPrice(c)}, dist: ${gap}%` : 'n/a' });
    }
  } else if (filt === 'ema200_align') {
    const e200 = ind.ema200[i];
    if (dir === 'LONG') {
      const gap = e200 != null ? ((c - e200) / e200 * 100).toFixed(2) : '?';
      items.push({ ok: c > e200, label: 'Precio > EMA200 (tendencia alcista LT)', detail: e200 ? `EMA200: $${fmtPrice(e200)}, precio: $${fmtPrice(c)}, dist: ${gap}%` : 'n/a' });
    } else {
      const gap = e200 != null ? ((e200 - c) / e200 * 100).toFixed(2) : '?';
      items.push({ ok: c < e200, label: 'Precio < EMA200 (tendencia bajista LT)', detail: e200 ? `EMA200: $${fmtPrice(e200)}, precio: $${fmtPrice(c)}, dist: ${gap}%` : 'n/a' });
    }
  } else if (filt === 'adx_trend') {
    const adx = ind.adx[i];
    const gap = adx != null && adx < 20 ? (20 - adx).toFixed(1) : 0;
    items.push({ ok: adx >= 20, label: 'ADX ≥ 20 (tendencia activa)', detail: adx != null ? `actual: ${adx.toFixed(1)}${adx >= 20 ? '' : `, faltan ${gap} pts`}` : 'n/a' });
  } else if (filt === 'vol_spike') {
    const vol = candles[i].volume;
    const avgVol = i >= 20 ? candles.slice(i-20, i).reduce((s, c) => s + c.volume, 0) / 20 : vol;
    const ratio = avgVol > 0 ? (vol / avgVol).toFixed(2) : '?';
    items.push({ ok: vol > avgVol * 1.3, label: 'Volumen > 1.3× promedio', detail: `actual: ${fmtVol(vol)}, prom: ${fmtVol(avgVol)}, ratio: ${ratio}×` });
  } else if (filt === 'ema20_slope') {
    const e20 = ind.ema20[i], e20p = ind.ema20[i-1];
    const slope = e20 && e20p ? ((e20 - e20p) / e20p * 100).toFixed(3) : null;
    if (dir === 'LONG') {
      items.push({ ok: slope > 0, label: 'EMA20 con pendiente positiva', detail: slope ? `slope: ${slope > 0 ? '+' : ''}${slope}%` : 'n/a' });
    } else {
      items.push({ ok: slope < 0, label: 'EMA20 con pendiente negativa', detail: slope ? `slope: ${slope}%` : 'n/a' });
    }
  } else if (filt === 'macd_dir') {
    const macd = ind.macdHist[i];
    if (dir === 'LONG') {
      items.push({ ok: macd > 0, label: 'MACD histogram > 0', detail: macd != null ? `actual: ${macd.toFixed(4)}` : 'n/a' });
    } else {
      items.push({ ok: macd < 0, label: 'MACD histogram < 0', detail: macd != null ? `actual: ${macd.toFixed(4)}` : 'n/a' });
    }
  } else if (filt === 'psar_confirm') {
    const psar = ind.psar[i];
    if (dir === 'LONG') {
      const gap = psar != null ? ((c - psar) / psar * 100).toFixed(2) : '?';
      items.push({ ok: c > psar, label: 'Precio > PSAR', detail: psar ? `PSAR: $${fmtPrice(psar)}, precio: $${fmtPrice(c)}, dist: ${gap}%` : 'n/a' });
    } else {
      const gap = psar != null ? ((psar - c) / psar * 100).toFixed(2) : '?';
      items.push({ ok: c < psar, label: 'Precio < PSAR', detail: psar ? `PSAR: $${fmtPrice(psar)}, precio: $${fmtPrice(c)}, dist: ${gap}%` : 'n/a' });
    }
  }
  
  return items.length ? items : null;
}

// ── Strategy detail modal (portfolio) ─────────────────────────────────────────
function showStrategyModal(sig, filt, wr, trades) {
  // ── STEP 1: Confirm function was called ──
  showNotif(`📋 Abriendo detalle: ${sig}`, 'info');
  console.log('[showStrategyModal] STEP1 called sig=', sig, 'filt=', filt);

  // ── STEP 2: Remove old dynamic modal if exists, create fresh one ──
  document.getElementById('_stratModalOverlay')?.remove();
  document.getElementById('_stratModal')?.remove();

  const overlay = document.createElement('div');
  overlay.id = '_stratModalOverlay';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:99998;';
  overlay.addEventListener('click', () => { overlay.remove(); modal.remove(); });

  const modal = document.createElement('div');
  modal.id = '_stratModal';
  modal.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);width:300px;max-height:85vh;overflow-y:auto;background:#1a1a2e;border:1px solid rgba(255,255,255,.15);border-radius:10px;z-index:99999;padding:14px;box-shadow:0 8px 32px rgba(0,0,0,.8);color:#e6edf3;font-family:Orbitron,sans-serif;';
  
  const closeBtn = document.createElement('div');
  closeBtn.style.cssText = 'position:absolute;top:8px;right:10px;font-size:14px;cursor:pointer;color:#8b949e;';
  closeBtn.textContent = '✕';
  closeBtn.addEventListener('click', () => { overlay.remove(); modal.remove(); });
  modal.appendChild(closeBtn);

  const content = document.createElement('div');
  modal.appendChild(content);

  document.body.appendChild(overlay);
  document.body.appendChild(modal);

  // ── Set loading text IMMEDIATELY so modal is never blank ──
  content.style.cssText = 'padding-top:8px;font-size:12px;color:#00d4ff;';
  content.textContent = '⏳ Cargando detalle...';

  console.log('[showStrategyModal] STEP2 modal appended to body');

  // ── STEP 3: Build all content inside try/catch ──
  try {

  // Lookup levels from global port status
  let levels = null;
  const ps = window._portStatus;
  if (ps) {
    const all = [...(ps.active||[]), ...(ps.waiting||[])];
    const match = all.find(s => s.sig === sig && s.filt === filt);
    if (match) levels = match._levels;
  }
  
  const sigLabel  = _BT_SIG_LBL[sig]  || sig;
  const filtLabel = _BT_FILT_LBL[filt] || filt;
  const sigCond   = _BT_SIG_COND[sig]  || '—';
  const filtCond  = _BT_FILT_COND[filt] || null;

  // Signal descriptions and operation guide
  const sigGuides = {
    rsi_os30:     { how: 'Esperar que RSI(14) baje de 30 y luego cruce al alza. Entrar en la vela que cierre por encima de 30. SL bajo el mínimo de la vela de entrada.', tip: 'Más efectivo en mercados laterales o después de caídas fuertes. Evitar en tendencia bajista fuerte.' },
    rsi_os35:     { how: 'RSI(14) cruza al alza desde debajo de 35. Confirmar con cierre de vela verde. SL bajo soporte más cercano.', tip: 'Zona 30-35 da señales de mayor calidad que zonas menos extremas.' },
    rsi_os40:     { how: 'RSI(14) sale de zona baja (<40). Señal más frecuente, usar con filtro adicional. SL 1 ATR bajo entrada.', tip: 'Filtrar solo en contexto alcista (EMA200 por debajo del precio).' },
    macd_flip:    { how: 'Histograma MACD cambia de signo (de - a + para LONG, de + a - para SHORT). Entrar en el cierre de esa vela.', tip: 'Reducir ruido usando el cruce en gráficos de mayor TF.' },
    macd_grow:    { how: 'Histograma MACD creciendo 2 velas seguidas en la misma dirección. Entrar en cierre de la segunda vela.', tip: 'Buena señal de momentum. Combinar con EMA slope para confirmar tendencia.' },
    bb_bounce:    { how: 'Precio toca o cierra fuera de la banda BB. Esperar la vela de reversión (cierre de vuelta dentro de la banda). Entrar en esa vela.', tip: 'No operar en tendencia fuerte — el precio puede "caminar" por la banda sin rebotar.' },
    bb_mid:       { how: 'Precio cruza la media de BB (EMA20) desde abajo (LONG) o desde arriba (SHORT). Entrar en el cierre de la vela de cruce.', tip: 'Señal de retorno a la media. Funciona bien en mercados laterales.' },
    ema_8_21:     { how: 'EMA8 cruza EMA21. LONG cuando EMA8 sube sobre EMA21. Entrar en el cierre de la vela del cruce.', tip: 'Cruce rápido — genera señales frecuentes. Filtrar con tendencia macro (EMA200).' },
    ema_20_50:    { how: 'EMA20 cruza EMA50. LONG cuando EMA20 sube sobre EMA50. Señal de cambio de tendencia intermedia.', tip: 'Señal más fiable que EMA8×21. Tolera retraso en entrada.' },
    price_ema20:  { how: 'Precio cruza la EMA20. LONG cuando precio cierra sobre EMA20 tras estar debajo. Entrar en ese cierre.', tip: 'Señal sencilla y muy usada. Funciona mejor en TF 15m o superior.' },
    stoch_cross:  { how: 'Stocástico K cruza D. LONG: cruce en zona <20 hacia arriba. SHORT: cruce en zona >80 hacia abajo. Entrar en el cierre de la vela del cruce.', tip: 'Solo operar cruces en zonas extremas (< 20 o > 80) para mayor precisión.' },
    donchian_10:  { how: 'Precio cierra por encima del máximo de los últimos 10 períodos (LONG) o por debajo del mínimo (SHORT). Entrar en ese cierre.', tip: 'Señal de breakout puro. Evitar en rangos sin volumen.' },
    vol_body:     { how: 'Vela con volumen > 1.5× promedio y cuerpo grande (>50% del rango). Dirección = dirección de la vela. Entrar en el cierre.', tip: 'Señal de momentum institucional. Muy confiable cuando aparece.' },
    williams_r:   { how: 'Williams %R sale de zona extrema. LONG: WR cruza al alza -80. SHORT: WR cruza a la baja -20. Entrar en ese cruce.', tip: 'Oscilador rápido. Combinar con tendencia de mayor TF para filtrar.' },
    cci_cross:    { how: 'CCI(20) cruza cero desde zona extrema. LONG: CCI sube sobre 0 desde -100. SHORT: CCI baja de 0 desde +100.', tip: 'El extremo (-100/+100) da contexto. El cruce de cero es la entrada.' },
    keltner:      { how: 'Precio cierra fuera del canal Keltner (EMA20 ± 2×ATR). Entrada en dirección de la ruptura.', tip: 'Muy útil para confirmar breakouts de BB squeeze.' },
    psar_flip:    { how: 'Parabolic SAR cambia de lado. LONG: SAR pasa de estar sobre el precio a debajo. Entrar en la vela del flip.', tip: 'Señal tardía pero confiable. SL en el nivel anterior del SAR.' },
    adx_di:       { how: 'ADX > 25 y DI+ cruza DI- hacia arriba (LONG) o DI- cruza DI+ hacia arriba (SHORT). Entrar en ese cruce con ADX confirmado.', tip: 'ADX confirma que hay tendencia. Sin ADX > 20, la señal DI es ruido.' },
    trix_cross:   { how: 'TRIX(14) cruza su línea de señal (EMA de TRIX). LONG: TRIX sube sobre la señal. Entrar en el cierre de esa vela.', tip: 'TRIX filtra el ruido por ser triple suavizado. Señales menos frecuentes pero de calidad.' },
    three_bar_rev:{ how: '3 velas consecutivas en una dirección + 4ª vela que cierra en la dirección opuesta con cuerpo sólido. Entrar al cierre de la 4ª vela.', tip: 'Buscar en zonas de soporte/resistencia para mayor efectividad.' },
    rsi_50:       { how: 'RSI(14) cruza el nivel 50. LONG: sube sobre 50. SHORT: baja de 50. Confirmar tendencia con EMA.', tip: 'El nivel 50 del RSI separa mercados alcistas de bajistas.' },
    engulf_vol:   { how: 'Vela envolvente (cuerpo de la vela actual cubre todo el cuerpo de la anterior) con volumen >1.5× promedio. Entrar al cierre.', tip: 'La combinación envolvente + volumen es la señal más confiable de reversión de vela individual.' },
    heikin_ashi:  { how: 'Cambio de color Heikin-Ashi (de rojo a verde para LONG, de verde a rojo para SHORT). La vela de cambio debe tener cuerpo sólido.', tip: 'Heikin-Ashi suaviza el precio. Retrasa la entrada pero reduce falsas señales.' },
    rsi2_ext:     { how: 'RSI(2) < 5 con precio sobre EMA200 = LONG. RSI(2) > 95 con precio bajo EMA200 = SHORT. Entrar al cierre de esa vela.', tip: 'Estrategia de Larry Connors. Muy efectiva en tendencias fuertes. Salir rápido (RSI2 > 70).' },
    connors_rsi:  { how: 'ConnorsRSI (promedio de RSI3 + StreakRSI + Percentil100) < 10 = LONG. > 90 = SHORT. Entrar al cierre.', tip: 'Más sofisticado que RSI simple. Menor tasa de falsos positivos en datos históricos.' },
    five_ribbon:  { how: 'Todas las 5 EMAs (5,8,13,21,34) perfectamente alineadas: 5>8>13>21>34 para LONG. Entrar cuando se alinean.', tip: 'Señal de tendencia fuerte. No apta para rangos. Gestiona con trailing stop.' },
    zscore_rev:   { how: 'Z-score del precio (cuántas σ está del promedio 20) > 2 = SHORT. < -2 = LONG. Entrar al cierre de la vela extrema.', tip: 'Reversión estadística. Combinar con soporte/resistencia para confirmar zona de giro.' },
    elder_imp:    { how: 'Elder Impulse System: barra azul (EMA13 sube + MACD histograma sube) = LONG. Barra roja = SHORT. Entrar al cierre de la barra de señal.', tip: 'Alexander Elder diseñó esto para evitar entrar contra el impulso. Esperar barra neutra antes de entrar en contra.' },
    consec_5:     { how: '5 velas bajistas consecutivas (cada cierre < apertura) → LONG al cierre de la 5ª. 5 velas alcistas → SHORT. Señal de agotamiento.', tip: 'Cuantas más velas consecutivas, mayor el agotamiento potencial. 7+ velas = señal de alta calidad.' },
    rsi3_pull:    { how: 'En uptrend (EMA50 subiendo): RSI(3) cae bajo 20 (pullback) y luego sube sobre 20. Entrar al cierre de esa recuperación.', tip: 'Solo operar LONG en uptrend confirmado. La caída del RSI3 es la oportunidad de entrada en la tendencia.' },
    rsi_div:      { how: 'Divergencia alcista: precio hace mínimo más bajo pero RSI(14) hace mínimo más alto → LONG. Inverso → SHORT. Confirmar con vela de reversión.', tip: 'Las divergencias predicen reversiones. A mayor timeframe, mayor fiabilidad.' },
    ultra_conf:   { how: 'Confluencia de 5 señales: EMA trend + MACD cruce + BB rebote + Stoch cruce extremo + RSI zona. Solo entra si las 5 coinciden.', tip: 'La señal más selectiva de la biblioteca. Pocas ocurrencias, alta tasa de éxito histórico.' },
  };

  const filtGuides = {
    ema50_align:  'Solo entrar si el precio está por encima de EMA50 para LONG (o por debajo para SHORT). Asegura que la tendencia de mediano plazo acompaña la entrada.',
    ema200_align: 'Solo entrar en la dirección de la tendencia macro (EMA200). Filtra entradas contra-tendencia de largo plazo.',
    ema20_slope:  'EMA20 debe tener pendiente positiva (LONG) o negativa (SHORT). Confirma que la tendencia de corto plazo está activa.',
    rsi_zone:     'RSI debe estar en zona compatible: 40–70 para LONG, 30–60 para SHORT. Evita entrar en extremos opuestos.',
    macd_dir:     'El histograma MACD debe estar en la misma dirección que la señal. Confirma que el momentum acompaña la entrada.',
    adx_trend:    'ADX > 20 indica que el mercado tiene tendencia suficiente. Filtra entradas en mercados laterales.',
    vol_spike:    'Volumen actual > 1.3× promedio de 20 velas. Confirma que hay interés institucional en el movimiento.',
    psar_confirm: 'Parabolic SAR debe estar del lado correcto: precio > SAR para LONG, precio < SAR para SHORT.',
  };

  const guide  = sigGuides[sig]  || { how: 'Esperar la condición de entrada descrita y gestionar con SL/TP del backtest.', tip: 'Respetar siempre el SL definido por el backtest.' };
  const fGuide = filtGuides[filt] || null;

  // Render levels — use hardcoded colors, no CSS vars
  const fp = v => { try { return fmtPrice(v); } catch(e) { return String(v); } };
  const renderLevRow = (label, val, col, pct) =>
    `<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid #30363d">
      <span style="font-size:9px;color:#8b949e">${label}</span>
      <span style="font-size:11px;font-weight:700;color:${col}">$${fp(val)}${pct ? ` <span style="font-size:9px;font-weight:400">${pct}</span>` : ''}</span>
    </div>`;

  let levelsHtml = '';
  if (levels) {
    const hasDir = !levels.long; // active — single direction
    if (hasDir) {
      const lv = levels; const isL = lv.entry && lv.tp1 > lv.entry;
      const col = isL ? '#00ff41' : '#f85149';
      levelsHtml = `<div style="margin-top:10px">
        <div style="font-size:10px;color:#00d4ff;font-weight:600;margin-bottom:4px">📍 Niveles actuales</div>
        ${renderLevRow('Entrada', lv.entry, col, '')}
        ${renderLevRow('Stop Loss', lv.sl, '#f85149', isL ? `-${((1-lv.sl/lv.entry)*100).toFixed(2)}%` : `-${((lv.sl/lv.entry-1)*100).toFixed(2)}%`)}
        ${renderLevRow('TP1', lv.tp1, '#00ff41', isL ? `+${((lv.tp1/lv.entry-1)*100).toFixed(2)}%` : `+${((1-lv.tp1/lv.entry)*100).toFixed(2)}%`)}
        ${renderLevRow('TP2', lv.tp2, '#00ff41', isL ? `+${((lv.tp2/lv.entry-1)*100).toFixed(2)}%` : `+${((1-lv.tp2/lv.entry)*100).toFixed(2)}%`)}
        <div style="font-size:9px;color:#8b949e;margin-top:4px;text-align:right">R:R ${lv.rr}:1</div>
      </div>`;
    } else {
      const mk = (dir, lv) => {
        const isL = dir==='LONG'; const col = isL ? '#00ff41':'#f85149';
        return `<div style="flex:1;background:${col}08;border:1px solid ${col}30;border-radius:5px;padding:5px">
          <div style="font-size:9px;color:${col};font-weight:700;margin-bottom:4px">${isL?'▲ LONG':'▼ SHORT'}</div>
          ${renderLevRow('Entrada', lv.entry, col, '')}
          ${renderLevRow('SL', lv.sl, '#f85149', isL?`-${((1-lv.sl/lv.entry)*100).toFixed(2)}%`:`-${((lv.sl/lv.entry-1)*100).toFixed(2)}%`)}
          ${renderLevRow('TP1', lv.tp1, '#00ff41', isL?`+${((lv.tp1/lv.entry-1)*100).toFixed(2)}%`:`+${((1-lv.tp1/lv.entry)*100).toFixed(2)}%`)}
          ${renderLevRow('TP2', lv.tp2, '#00ff41', isL?`+${((lv.tp2/lv.entry-1)*100).toFixed(2)}%`:`+${((1-lv.tp2/lv.entry)*100).toFixed(2)}%`)}
          <div style="font-size:8px;color:#8b949e;margin-top:3px;text-align:right">R:R ${lv.rr}:1</div>
        </div>`;
      };
      levelsHtml = `<div style="margin-top:10px">
        <div style="font-size:10px;color:#00d4ff;font-weight:600;margin-bottom:4px">📍 Niveles si dispara ahora</div>
        <div style="display:flex;gap:5px">${mk('LONG',levels.long)}${mk('SHORT',levels.short)}</div>
      </div>`;
    }
  }

  const wrCol = wr >= 60 ? '#00ff41' : wr >= 50 ? '#e3b341' : '#f85149';

  // Build condition checklist if we have current data
  let checklistHtml = '';
  try {
  const ind = window._lastIndicators;
  const candles = window._lastCandles;
  if (ind && candles && candles.length > 1) {
    const i = candles.length - 1;
    const longItems = btBuildConditionChecklist(ind, candles, i, sig, filt, 'LONG');
    const shortItems = btBuildConditionChecklist(ind, candles, i, sig, filt, 'SHORT');
    
    if (longItems || shortItems) {
      checklistHtml = `<div style="margin-bottom:10px">
        <div style="font-size:10px;color:#f0883e;font-weight:600;margin-bottom:4px">📋 Estado actual</div>
        <div style="display:flex;gap:5px">`;
      
      if (longItems) {
        checklistHtml += `<div style="flex:1;background:#00ff4108;border:1px solid #00ff4130;border-radius:5px;padding:5px">
          <div style="font-size:9px;color:#00ff41;font-weight:700;margin-bottom:4px">▲ LONG</div>
          ${longItems.map(it => `<div style="display:flex;align-items:start;gap:4px;margin-bottom:3px">
            <span style="font-size:10px">${it.ok ? '✅' : '❌'}</span>
            <div style="flex:1">
              <div style="font-size:9px;color:${it.ok ? '#00ff41' : 'var(--text2)'};font-weight:${it.ok ? '600' : '400'}">${escHtml(it.label)}</div>
              <div style="font-size:8px;color:var(--text3);margin-top:1px">${escHtml(it.detail)}</div>
            </div>
          </div>`).join('')}
        </div>`;
      }
      
      if (shortItems) {
        checklistHtml += `<div style="flex:1;background:#f8514908;border:1px solid #f8514930;border-radius:5px;padding:5px">
          <div style="font-size:9px;color:#f85149;font-weight:700;margin-bottom:4px">▼ SHORT</div>
          ${shortItems.map(it => `<div style="display:flex;align-items:start;gap:4px;margin-bottom:3px">
            <span style="font-size:10px">${it.ok ? '✅' : '❌'}</span>
            <div style="flex:1">
              <div style="font-size:9px;color:${it.ok ? '#f85149' : 'var(--text2)'};font-weight:${it.ok ? '600' : '400'}">${escHtml(it.label)}</div>
              <div style="font-size:8px;color:var(--text3);margin-top:1px">${escHtml(it.detail)}</div>
            </div>
          </div>`).join('')}
        </div>`;
      }
      
      checklistHtml += `</div></div>`;
    }
  }
  } catch(chkErr) {
    console.warn('[showStrategyModal] Checklist error (ignored):', chkErr);
    checklistHtml = '';
  }

  // ── Build final HTML ──
  const wilson = (() => { try { return Math.round(btWilsonLower(Math.round(wr/100*trades), trades)*1000)/10; } catch(e) { return '?'; } })();
  const _sigLabel = (_BT_SIG_LBL && _BT_SIG_LBL[sig]) || sig;
  const _filtLabel = (_BT_FILT_LBL && _BT_FILT_LBL[filt]) || filt;
  const _sigCond2 = (_BT_SIG_COND && _BT_SIG_COND[sig]) || '—';
  const _filtCond2 = (_BT_FILT_COND && _BT_FILT_COND[filt]) || null;
  const _filtCondTxt = _filtCond2 ? `<div style="font-size:9px;color:#8b949e;margin-top:4px;padding:4px 8px;background:rgba(255,255,255,.04);border-radius:4px">+ Filtro: ${_filtCond2}</div>` : '';

  content.style.cssText = '';
  content.innerHTML = `
    <div style="border-bottom:1px solid #30363d;padding-bottom:10px;margin-bottom:10px">
      <div style="font-size:14px;font-weight:700;color:#00d4ff;margin-bottom:3px">📋 ${_sigLabel} ${_filtLabel}</div>
      <div style="display:flex;gap:10px;font-size:11px">
        <span style="color:#e6edf3">WR: <b style="color:${wrCol}">${wr}%</b></span>
        <span style="color:#e6edf3">Trades: <b style="color:#e6edf3">${trades}</b></span>
        <span style="color:#e6edf3">Wilson: <b style="color:#e3b341">${wilson}%</b></span>
      </div>
    </div>

    <div style="margin-bottom:10px">
      <div style="font-size:10px;color:#00d4ff;font-weight:600;margin-bottom:4px">⚡ Condición de entrada</div>
      <div style="font-size:10px;color:#e6edf3;line-height:1.5;background:rgba(0,212,255,.06);border-radius:4px;padding:6px 8px">${_sigCond2}</div>
      ${_filtCondTxt}
    </div>

    ${checklistHtml}

    <div style="margin-bottom:10px">
      <div style="font-size:10px;color:#00d4ff;font-weight:600;margin-bottom:4px">🎯 Cómo operar</div>
      <div style="font-size:10px;color:#e6edf3;line-height:1.5">${guide.how}</div>
      ${fGuide ? `<div style="font-size:9px;color:#8b949e;margin-top:5px;line-height:1.4;border-left:2px solid #00d4ff;padding-left:6px">${fGuide}</div>` : ''}
    </div>

    <div style="margin-bottom:10px">
      <div style="font-size:10px;color:#e3b341;font-weight:600;margin-bottom:4px">💡 Pro tip</div>
      <div style="font-size:9px;color:#8b949e;line-height:1.5">${guide.tip}</div>
    </div>

    <div style="margin-bottom:10px">
      <div style="font-size:10px;color:#00d4ff;font-weight:600;margin-bottom:4px">📊 Probabilidades</div>
      <div style="display:flex;align-items:center;gap:6px">
        <div style="flex:1;height:8px;background:#30363d;border-radius:4px;overflow:hidden">
          <div style="height:100%;width:${wr}%;background:${wrCol};border-radius:4px"></div>
        </div>
        <span style="font-size:11px;font-weight:700;color:${wrCol}">${wr}%</span>
      </div>
      <div style="font-size:9px;color:#8b949e;margin-top:3px">${trades} operaciones · Wilson: ${wilson}%</div>
    </div>

    ${levelsHtml}
  `;
  } catch(err) {
    console.error('[showStrategyModal] Error:', err);
    content.style.cssText = '';
    content.innerHTML = `<div style="padding:12px;font-size:13px;color:#f85149">⚠️ Error: ${String(err)}</div>`;
  }
}

async function refreshWatchlist() {
  const tickers = await fetchMultiTicker(watchlist);
  watchData = tickers;
  renderWatchlist();
}

function renderWatchlist() {
  const el = document.getElementById('watchList');
  if (!el) return;
  el.innerHTML = watchlist.map(sym => {
    const t   = watchData[sym];
    if (!t) return `<div class="wl-row wl-loading"><span class="wl-sym">${escHtml(sym)}</span><span class="wl-price">…</span></div>`;
    const chg  = parseFloat(t.priceChangePercent);
    const col  = chg >= 0 ? '#00ff41' : '#f85149';
    const price = parseFloat(t.lastPrice);
    return `<div class="wl-row" data-sym="${escHtml(sym)}">
      <span class="wl-sym">${escHtml(sym.replace('USDT',''))}</span>
      <span class="wl-price">$${fmtPrice(price)}</span>
      <span class="wl-chg" style="color:${col}">${fmtPct(chg)}</span>
      <span class="wl-vol">${fmtVol(parseFloat(t.quoteVolume))}</span>
      <button class="btn-wl-analyze" data-sym="${escHtml(sym)}">▶</button>
    </div>`;
  }).join('');
  el.querySelectorAll('.btn-wl-analyze').forEach(btn => {
    btn.addEventListener('click', () => {
      currentSymbol = btn.dataset.sym;
      document.getElementById('symInput').value = currentSymbol;
      switchTab('analysis');
      loadAndAnalyze();
    });
  });
  el.querySelectorAll('.wl-row[data-sym]').forEach(row => {
    row.addEventListener('click', (e) => {
      if (e.target.classList.contains('btn-wl-analyze')) return;
      currentSymbol = row.dataset.sym;
      document.getElementById('symInput').value = currentSymbol;
      switchTab('analysis');
      loadAndAnalyze();
    });
  });
}

// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
// ADVANCED BRAIN UPDATE (ML + MTF + Harmonic + Elliott)
// Función centralizada — se llama tras loadAndAnalyze Y tras cargar el modelo ML
// ═══════════════════════════════════════════════════════════════════

async function runAdvancedBrain(analysis, candleData, symbol, tf) {
  const brainEl = document.getElementById('aBrain');

  // ── Paso 1: Cálculos síncronos — siempre rápidos ─────────────────
  let harmonics = [], elliott = null, mlResult = null;
  try { harmonics = detectHarmonicPatterns(candleData) || []; } catch {}
  try { elliott   = detectElliottWave(candleData); }              catch {}
  try { mlResult  = mlInfer(candleData); }                        catch {}

  // ── Paso 2: Primer re-render (elimina placeholder "_loading") ─────
  // Este render siempre ocurre, incluso si todo es null
  const mkProb = (mtf) => {
    try {
      return computeAdvancedProb({
        score: analysis.score, mtf, harmonics, elliott,
        div: analysis.div, volume: analysis.volume,
        patterns: analysis.patterns,
        rsi: analysis.indicators?.rsi, hist: analysis.indicators?.hist,
        mlResult,
      });
    } catch { return null; }
  };

  const adv1 = mkProb(null);
  window._lastAdvanced = { mtf: null, harmonics, elliott, advProb: adv1, mlResult };
  if (brainEl && analysis.brain) {
    try {
      renderBrain(brainEl, { ...analysis.brain, mtf: null, harmonics, elliott, advProb: adv1, mlResult });
    } catch (e) { console.warn('[AdvBrain] render1 error:', e); }
  }

  // ── Paso 3: Fetch MTF (puede tardar, no bloquea el render anterior) ─
  let mtf = null;
  try {
    mtf = await fetchMTFanalysis(symbol, tf);
  } catch {}

  // Fallback local si MTF externo falla
  if (!mtf && candleData.length >= 50) {
    try {
      const closes = candleData.map(c => c.close);
      const k20 = 2 / 21, k50 = 2 / 51;
      let e20 = closes[0], e50 = closes[0];
      for (const v of closes) { e20 = e20 * (1-k20) + v * k20; e50 = e50 * (1-k50) + v * k50; }
      const last = closes[closes.length - 1];
      const dir = last > e20 && e20 > e50 ? 'bullish'
                : last < e20 && e20 < e50 ? 'bearish'
                : last > e20 ? 'neutral_up' : 'neutral_down';
      mtf = {
        primaryTf: '—', secondaryTf: '—', currentTf: tf,
        prim: { dir, rsi: null, ema20: e20.toFixed(0), last: last.toFixed(0) },
        sec: null, align: dir === 'bullish' ? 'ALCISTA ↑~' : dir === 'bearish' ? 'BAJISTA ↓~' : 'NEUTRAL →',
        alignScore: 50, _localOnly: true,
      };
    } catch {}
  }

  // ── Paso 4: Re-render final con MTF ──────────────────────────────
  const adv2 = mkProb(mtf);
  window._lastAdvanced = { mtf, harmonics, elliott, advProb: adv2, mlResult };
  if (brainEl && analysis.brain) {
    try {
      renderBrain(brainEl, { ...analysis.brain, mtf, harmonics, elliott, advProb: adv2, mlResult });
    } catch (e) { console.warn('[AdvBrain] render2 error:', e); }
  }
}

// ═══════════════════════════════════════════════════════════════════
// BACKTESTING ENGINE
// ═══════════════════════════════════════════════════════════════════

const BT_CANDLES_PER_DAY = { '1m': 1440, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '4h': 6, '1d': 1 };

// ── Strategy signal functions ────────────────────────────────────
// Each returns { dir: 'LONG'|'SHORT', conf: 0-100, reason: string, atr: number } or null

function btSigEMATrend(slice, minConf) {
  if (slice.length < 210) return null;
  const closes = slice.map(c => c.close);
  const n = closes.length - 1;
  const ema20  = calcEMA(closes, 20)[n];
  const ema50  = calcEMA(closes, 50)[n];
  const ema200 = calcEMA(closes, 200)[n];
  const rsi    = calcRSI(closes, 14)[n];
  const atr    = calcATR(slice, 14)[n];
  const price  = closes[n];
  if (!ema20 || !ema50 || !ema200 || rsi === null) return null;
  // Full alignment required
  const bull = price > ema20 && ema20 > ema50 && ema50 > ema200;
  const bear = price < ema20 && ema20 < ema50 && ema50 < ema200;
  if (!bull && !bear) return null;
  // RSI not at extreme (avoid chasing)
  if (rsi > 75 || rsi < 25) return null;
  // Confidence from EMA spread
  const spread = Math.abs(ema20 - ema200) / ema200 * 100;
  const conf = Math.min(80, 52 + spread * 8);
  if (conf < minConf) return null;
  return bull
    ? { dir: 'LONG',  conf, reason: `EMA 20>${ema50.toFixed(0)}>200, RSI ${rsi.toFixed(0)}`, atr }
    : { dir: 'SHORT', conf, reason: `EMA 20<${ema50.toFixed(0)}<200, RSI ${rsi.toFixed(0)}`, atr };
}

function btSigMACDRSI(slice, minConf) {
  if (slice.length < 60) return null;
  const closes = slice.map(c => c.close);
  const n = closes.length - 1;
  if (n < 2) return null;
  const macd = calcMACD(closes);
  const rsi  = calcRSI(closes, 14);
  const atr  = calcATR(slice, 14)[n];
  const histNow  = macd.hist[n];
  const histPrev = macd.hist[n - 1];
  const rsiNow   = rsi[n];
  if (histNow === undefined || histPrev === undefined || rsiNow === null) return null;
  // MACD histogram flip
  const bullFlip = histPrev < 0 && histNow > 0;
  const bearFlip = histPrev > 0 && histNow < 0;
  if (!bullFlip && !bearFlip) return null;
  // RSI confirmation (not at extreme opposite side)
  if (bullFlip && rsiNow > 72) return null;
  if (bearFlip && rsiNow < 28) return null;
  // Confidence from magnitude of flip and RSI distance from 50
  const magn = Math.abs(histNow) / Math.abs(closes[n]) * 100000;
  const conf = Math.min(82, 50 + magn * 2 + Math.abs(rsiNow - 50) * 0.4);
  if (conf < minConf) return null;
  return bullFlip
    ? { dir: 'LONG',  conf, reason: `MACD hist flip +, RSI ${rsiNow.toFixed(0)}`, atr }
    : { dir: 'SHORT', conf, reason: `MACD hist flip -, RSI ${rsiNow.toFixed(0)}`, atr };
}

function btSigBBBounce(slice, minConf) {
  if (slice.length < 30) return null;
  const closes = slice.map(c => c.close);
  const n = closes.length - 1;
  if (n < 2) return null;
  const bb  = calcBB(closes, 20, 2);
  const rsi = calcRSI(closes, 14);
  const atr = calcATR(slice, 14)[n];
  const bbN  = bb[n],  bbP  = bb[n - 1];
  const rsiN = rsi[n];
  if (!bbN?.lower || !bbN?.upper || !bbP?.lower || !bbP?.upper || rsiN === null) return null;
  // Bounce: candle crossed back inside from below lower band
  const lowerBounce = closes[n - 1] <= bbP.lower && closes[n] > bbN.lower && rsiN < 48;
  // Rejection: candle crossed back inside from above upper band
  const upperReject = closes[n - 1] >= bbP.upper && closes[n] < bbN.upper && rsiN > 52;
  if (!lowerBounce && !upperReject) return null;
  const dist = lowerBounce ? (50 - rsiN) : (rsiN - 50);
  const conf = Math.min(82, 52 + dist * 0.7);
  if (conf < minConf) return null;
  return lowerBounce
    ? { dir: 'LONG',  conf, reason: `BB rebote inferior, RSI ${rsiN.toFixed(0)}`, atr }
    : { dir: 'SHORT', conf, reason: `BB rechazo superior, RSI ${rsiN.toFixed(0)}`, atr };
}

function btSigSupertrend(slice, minConf) {
  if (slice.length < 20) return null;
  const closes  = slice.map(c => c.close);
  const atrArr  = calcATR(slice, 14);
  const n = slice.length - 1;
  const mult = 3;
  let upBand = 0, dnBand = 0;
  const trends = [];
  for (let i = 1; i <= n; i++) {
    const hl2  = (slice[i].high + slice[i].low) / 2;
    const atrI = atrArr[i] || (slice[i].high - slice[i].low);
    const rawUp = hl2 + mult * atrI;
    const rawDn = hl2 - mult * atrI;
    upBand = (rawUp < upBand || closes[i - 1] > upBand) ? rawUp : upBand;
    dnBand = (rawDn > dnBand || closes[i - 1] < dnBand) ? rawDn : dnBand;
    const prevT = trends.length > 0 ? trends[trends.length - 1] : 1;
    const t = prevT === 1 ? (closes[i] < dnBand ? -1 : 1) : (closes[i] > upBand ? 1 : -1);
    trends.push(t);
  }
  if (trends.length < 2) return null;
  const cur  = trends[trends.length - 1];
  const prev = trends[trends.length - 2];
  if (cur === prev) return null; // No flip → no signal
  const conf = 62;
  if (conf < minConf) return null;
  const atr = atrArr[n];
  return cur === 1
    ? { dir: 'LONG',  conf, reason: `Supertrend flip alcista (ATR×${mult})`, atr }
    : { dir: 'SHORT', conf, reason: `Supertrend flip bajista (ATR×${mult})`, atr };
}

// RSI Mean Reversion
function btSigRSIReversal(slice, minConf) {
  if (slice.length < 30) return null;
  const closes = slice.map(c => c.close);
  const n = closes.length - 1;
  if (n < 2) return null;
  const rsi = calcRSI(closes, 14);
  const atr = calcATR(slice, 14)[n];
  const rsiNow = rsi[n], rsiPrev = rsi[n - 1];
  if (rsiNow === null || rsiPrev === null) return null;
  const bullRev = rsiPrev < 32 && rsiNow > rsiPrev;
  const bearRev = rsiPrev > 68 && rsiNow < rsiPrev;
  if (!bullRev && !bearRev) return null;
  const lastC = slice[n];
  if (bullRev && lastC.close <= lastC.open) return null;
  if (bearRev && lastC.close >= lastC.open) return null;
  const intensity = bullRev ? (32 - rsiPrev) : (rsiPrev - 68);
  const conf = Math.min(85, 52 + intensity * 1.5);
  if (conf < minConf) return null;
  return bullRev
    ? { dir: 'LONG',  conf, reason: `RSI rebote ${rsiPrev.toFixed(0)}→${rsiNow.toFixed(0)} (oversold)`, atr }
    : { dir: 'SHORT', conf, reason: `RSI rechazo ${rsiPrev.toFixed(0)}→${rsiNow.toFixed(0)} (overbought)`, atr };
}

// EMA Pullback to dynamic support
function btSigEMAPullback(slice, minConf) {
  if (slice.length < 55) return null;
  const closes = slice.map(c => c.close);
  const n = closes.length - 1;
  const ema20  = calcEMA(closes, 20);
  const ema50  = calcEMA(closes, 50);
  const ema200 = calcEMA(closes, Math.min(200, closes.length - 1));
  const rsi    = calcRSI(closes, 14);
  const atr    = calcATR(slice, 14)[n];
  const e20 = ema20[n], e50 = ema50[n], e200 = ema200[n];
  const e20p = ema20[n - 1];
  const rsiN = rsi[n];
  const price = closes[n], pricePrev = closes[n - 1];
  if (!e20 || !e50 || !e200 || rsiN === null) return null;
  const uptrend   = e50 > e200 * 1.001;
  const downtrend = e50 < e200 * 0.999;
  if (!uptrend && !downtrend) return null;
  if (uptrend) {
    const touched = pricePrev <= e20p * 1.003 && price > e20;
    if (!touched || rsiN > 65) return null;
    const conf = Math.min(82, 55 + Math.abs(e50 - e200) / e200 * 500);
    if (conf < minConf) return null;
    return { dir: 'LONG', conf, reason: `Pullback EMA20 en uptrend (E50>${e200.toFixed(0)})`, atr };
  } else {
    const touched = pricePrev >= e20p * 0.997 && price < e20;
    if (!touched || rsiN < 35) return null;
    const conf = Math.min(82, 55 + Math.abs(e200 - e50) / e200 * 500);
    if (conf < minConf) return null;
    return { dir: 'SHORT', conf, reason: `Rechazo EMA20 en downtrend (E50<${e200.toFixed(0)})`, atr };
  }
}

// Stochastic Cross in extreme zones
function btSigStochCross(slice, minConf) {
  if (slice.length < 30) return null;
  const n = slice.length - 1;
  if (n < 2) return null;
  const stoch = calcStoch(slice, 14, 3);
  const kValid = stoch.k.filter(v => v !== null);
  const dFull  = stoch.d;
  if (kValid.length < 4 || dFull.length < 2) return null;
  const kN = kValid[kValid.length - 1], kP = kValid[kValid.length - 2];
  const dN = dFull[dFull.length - 1],   dP = dFull[dFull.length - 2];
  const atr = calcATR(slice, 14)[n];
  const bullCross = kP < dP && kN > dN && kN < 30;
  const bearCross = kP > dP && kN < dN && kN > 70;
  if (!bullCross && !bearCross) return null;
  const depth = bullCross ? (30 - kN) : (kN - 70);
  const conf = Math.min(82, 52 + depth * 0.9);
  if (conf < minConf) return null;
  return bullCross
    ? { dir: 'LONG',  conf, reason: `Stoch K↑ D oversold (K=${kN.toFixed(0)})`, atr }
    : { dir: 'SHORT', conf, reason: `Stoch K↓ D overbought (K=${kN.toFixed(0)})`, atr };
}

// Engulfing candle pattern
function btSigEngulfing(slice, minConf) {
  if (slice.length < 10) return null;
  const n = slice.length - 1;
  const curr = slice[n], prev = slice[n - 1];
  const atr    = calcATR(slice, 14)[n];
  const closes = slice.map(c => c.close);
  const ema50  = calcEMA(closes, 50)[n];
  const currBody = Math.abs(curr.close - curr.open);
  const prevBody = Math.abs(prev.close - prev.open);
  if (prevBody < (atr || 1) * 0.25 || currBody < (atr || 1) * 0.3) return null;
  const bullEngulf = prev.close < prev.open && curr.close > curr.open &&
                     curr.open <= prev.close && curr.close >= prev.open && currBody > prevBody * 1.1;
  const bearEngulf = prev.close > prev.open && curr.close < curr.open &&
                     curr.open >= prev.close && curr.close <= prev.open && currBody > prevBody * 1.1;
  if (!bullEngulf && !bearEngulf) return null;
  if (ema50) {
    if (bullEngulf && curr.close < ema50 * 0.993) return null;
    if (bearEngulf && curr.close > ema50 * 1.007) return null;
  }
  const ratio = currBody / Math.max(prevBody, 0.001);
  const conf = Math.min(82, 52 + (ratio - 1) * 18);
  if (conf < minConf) return null;
  return bullEngulf
    ? { dir: 'LONG',  conf, reason: `Engulfing alcista (×${ratio.toFixed(1)} cuerpo)`, atr }
    : { dir: 'SHORT', conf, reason: `Engulfing bajista (×${ratio.toFixed(1)} cuerpo)`, atr };
}

// Triple EMA 8/21/55 cross
function btSigTripleEMA(slice, minConf) {
  if (slice.length < 60) return null;
  const closes = slice.map(c => c.close);
  const n = closes.length - 1;
  const ema8  = calcEMA(closes, 8);
  const ema21 = calcEMA(closes, 21);
  const ema55 = calcEMA(closes, 55);
  const rsi   = calcRSI(closes, 14);
  const atr   = calcATR(slice, 14)[n];
  const e8 = ema8[n], e21 = ema21[n], e55 = ema55[n];
  const e8p = ema8[n - 1], e21p = ema21[n - 1];
  const rsiN = rsi[n];
  if (!e8 || !e21 || !e55 || rsiN === null) return null;
  const bullCross = e8p <= e21p && e8 > e21;
  const bearCross = e8p >= e21p && e8 < e21;
  if (!bullCross && !bearCross) return null;
  if (bullCross && e21 < e55 * 0.997) return null;
  if (bearCross && e21 > e55 * 1.003) return null;
  if (rsiN > 72 || rsiN < 28) return null;
  const spread = Math.abs(e8 - e55) / e55 * 100;
  const conf = Math.min(82, 54 + spread * 7);
  if (conf < minConf) return null;
  return bullCross
    ? { dir: 'LONG',  conf, reason: `EMA8↑EMA21 sobre EMA55, RSI ${rsiN.toFixed(0)}`, atr }
    : { dir: 'SHORT', conf, reason: `EMA8↓EMA21 bajo EMA55, RSI ${rsiN.toFixed(0)}`, atr };
}

// Donchian channel breakout
function btSigDonchian(slice, minConf) {
  if (slice.length < 25) return null;
  const n = slice.length - 1;
  if (n < 22) return null;
  const win = slice.slice(n - 20, n);
  const hh = Math.max(...win.map(c => c.high));
  const ll  = Math.min(...win.map(c => c.low));
  const curr = slice[n], prev = slice[n - 1];
  const atr  = calcATR(slice, 14)[n];
  const closes = slice.map(c => c.close);
  const rsi = calcRSI(closes, 14)[n];
  if (!hh || !ll || rsi === null) return null;
  const bullBreak = prev.close <= hh && curr.close > hh && curr.volume > prev.volume;
  const bearBreak = prev.close >= ll && curr.close < ll && curr.volume > prev.volume;
  if (!bullBreak && !bearBreak) return null;
  if (bullBreak && rsi > 78) return null;
  if (bearBreak && rsi < 22) return null;
  const breakPct = bullBreak ? (curr.close - hh) / (atr || 1) : (ll - curr.close) / (atr || 1);
  const conf = Math.min(84, 54 + breakPct * 15);
  if (conf < minConf) return null;
  return bullBreak
    ? { dir: 'LONG',  conf, reason: `Donchian 20 break ↑ (RSI ${rsi.toFixed(0)})`, atr }
    : { dir: 'SHORT', conf, reason: `Donchian 20 break ↓ (RSI ${rsi.toFixed(0)})`, atr };
}

// Momentum candle: big body + volume spike in trend direction
function btSigMomentum(slice, minConf) {
  if (slice.length < 25) return null;
  const n = slice.length - 1;
  const curr    = slice[n];
  const closes  = slice.map(c => c.close);
  const vols    = slice.map(c => c.volume);
  const atr     = calcATR(slice, 14)[n];
  const ema20   = calcEMA(closes, 20)[n];
  const rsi     = calcRSI(closes, 14)[n];
  if (!atr || !ema20 || rsi === null) return null;
  const body    = Math.abs(curr.close - curr.open);
  const avgVol  = vols.slice(Math.max(0, n - 20), n).reduce((s, v) => s + v, 0) / 20;
  if (avgVol === 0) return null;
  const volRatio = curr.volume / avgVol;
  if (volRatio < 1.5 || body < atr * 1.2) return null;
  const isBull = curr.close > curr.open;
  const isBear = curr.close < curr.open;
  if (isBull && curr.close < ema20) return null;
  if (isBear && curr.close > ema20) return null;
  if (rsi > 78 || rsi < 22) return null;
  const conf = Math.min(85, 52 + volRatio * 3 + (body / atr) * 3);
  if (conf < minConf) return null;
  return isBull
    ? { dir: 'LONG',  conf, reason: `Momento ↑ vol×${volRatio.toFixed(1)} cuerpo=${( body/atr).toFixed(1)}ATR`, atr, slMult: 1.5, tp1Mult: 2.5, tp2Mult: 4 }
    : { dir: 'SHORT', conf, reason: `Momento ↓ vol×${volRatio.toFixed(1)} cuerpo=${(body/atr).toFixed(1)}ATR`, atr, slMult: 1.5, tp1Mult: 2.5, tp2Mult: 4 };
}

// BB Squeeze release
function btSigBBSqueeze(slice, minConf) {
  if (slice.length < 35) return null;
  const closes = slice.map(c => c.close);
  const n = closes.length - 1;
  const bb   = calcBB(closes, 20, 2);
  const rsi  = calcRSI(closes, 14);
  const atr  = calcATR(slice, 14)[n];
  const widths = bb.slice(Math.max(0, n - 22), n + 1).map(b => b?.width).filter(w => w != null && w > 0);
  if (widths.length < 8) return null;
  const currW = widths[widths.length - 1];
  const prevW = widths[widths.length - 2];
  const minW  = Math.min(...widths.slice(0, widths.length - 2));
  if (prevW > minW * 1.15) return null; // was not squeezing
  if (currW < prevW * 1.12) return null; // not yet expanding
  const bbNow = bb[n];
  const rsiN  = rsi[n];
  const price = closes[n];
  if (!bbNow?.mid || rsiN === null) return null;
  const bullBreak = price > bbNow.mid && rsiN > 50;
  const bearBreak = price < bbNow.mid && rsiN < 50;
  if (!bullBreak && !bearBreak) return null;
  const expansion = currW / Math.max(prevW, 0.00001);
  const conf = Math.min(84, 54 + expansion * 8 + (bullBreak ? rsiN - 50 : 50 - rsiN) * 0.4);
  if (conf < minConf) return null;
  return bullBreak
    ? { dir: 'LONG',  conf, reason: `BB Squeeze ×${expansion.toFixed(1)} RSI ${rsiN.toFixed(0)}`, atr }
    : { dir: 'SHORT', conf, reason: `BB Squeeze ×${expansion.toFixed(1)} RSI ${rsiN.toFixed(0)}`, atr };
}

// Triple confluence: EMA direction + RSI zone + MACD histogram
function btSigMultiConf(slice, minConf) {
  if (slice.length < 60) return null;
  const closes = slice.map(c => c.close);
  const n = closes.length - 1;
  const ema20 = calcEMA(closes, 20)[n];
  const ema50 = calcEMA(closes, 50)[n];
  const macd  = calcMACD(closes);
  const rsi   = calcRSI(closes, 14);
  const atr   = calcATR(slice, 14)[n];
  const rsiN  = rsi[n];
  const histN = macd.hist[n], histP = macd.hist[n - 1];
  if (!ema20 || !ema50 || rsiN === null || histN === undefined) return null;
  const price = closes[n];
  const bullConf = price > ema20 && ema20 > ema50 && rsiN > 50 && rsiN < 68 && histN > 0 && histN > histP;
  const bearConf = price < ema20 && ema20 < ema50 && rsiN < 50 && rsiN > 32 && histN < 0 && histN < histP;
  if (!bullConf && !bearConf) return null;
  const spread = Math.abs(ema20 - ema50) / ema50 * 100;
  const conf = Math.min(87, 57 + spread * 7 + (bullConf ? rsiN - 50 : 50 - rsiN) * 0.5);
  if (conf < minConf) return null;
  return bullConf
    ? { dir: 'LONG',  conf, reason: `3×Conf EMA↑+RSI ${rsiN.toFixed(0)}+MACD↑`, atr }
    : { dir: 'SHORT', conf, reason: `3×Conf EMA↓+RSI ${rsiN.toFixed(0)}+MACD↓`, atr };
}

// Hammer / Shooting star candlestick
function btSigHammer(slice, minConf) {
  if (slice.length < 25) return null;
  const n    = slice.length - 1;
  const curr = slice[n];
  const closes = slice.map(c => c.close);
  const atr   = calcATR(slice, 14)[n];
  const ema50 = calcEMA(closes, 50)[n];
  const rsi   = calcRSI(closes, 14)[n];
  if (!atr || rsi === null) return null;
  const body      = Math.abs(curr.close - curr.open);
  const upperWick = curr.high - Math.max(curr.close, curr.open);
  const lowerWick = Math.min(curr.close, curr.open) - curr.low;
  const range     = curr.high - curr.low;
  if (range < atr * 0.4 || body < 0.00001) return null;
  const isHammer = lowerWick > body * 2 && lowerWick > range * 0.5 && upperWick < lowerWick * 0.4;
  const isStar   = upperWick > body * 2 && upperWick > range * 0.5 && lowerWick < upperWick * 0.4;
  if (!isHammer && !isStar) return null;
  if (isHammer && ema50 && curr.close > ema50 * 1.015) return null;
  if (isStar   && ema50 && curr.close < ema50 * 0.985) return null;
  if (isHammer && rsi > 55) return null;
  if (isStar   && rsi < 45) return null;
  const wickRatio = isHammer ? lowerWick / range : upperWick / range;
  const conf = Math.min(82, 52 + wickRatio * 36 + Math.abs(50 - rsi) * 0.5);
  if (conf < minConf) return null;
  const wickMult = isHammer ? (lowerWick / body).toFixed(1) : (upperWick / body).toFixed(1);
  return isHammer
    ? { dir: 'LONG',  conf, reason: `Martillo mecha×${wickMult} RSI ${rsi.toFixed(0)}`, atr, slMult: 1.5, tp1Mult: 2.5, tp2Mult: 4 }
    : { dir: 'SHORT', conf, reason: `Estrella fugaz mecha×${wickMult} RSI ${rsi.toFixed(0)}`, atr, slMult: 1.5, tp1Mult: 2.5, tp2Mult: 4 };
}

// Inside bar breakout
function btSigInsideBar(slice, minConf) {
  if (slice.length < 25) return null;
  const n = slice.length - 1;
  if (n < 2) return null;
  const curr = slice[n], ib = slice[n - 1], mother = slice[n - 2];
  const closes = slice.map(c => c.close);
  const atr    = calcATR(slice, 14)[n];
  const ema50  = calcEMA(closes, 50)[n];
  const rsi    = calcRSI(closes, 14)[n];
  if (!atr) return null;
  const isIB = ib.high <= mother.high && ib.low >= mother.low;
  if (!isIB) return null;
  const bullBreak = curr.close > mother.high;
  const bearBreak = curr.close < mother.low;
  if (!bullBreak && !bearBreak) return null;
  if (bullBreak && ema50 && curr.close < ema50 * 0.994) return null;
  if (bearBreak && ema50 && curr.close > ema50 * 1.006) return null;
  if (rsi !== null && (rsi > 76 || rsi < 24)) return null;
  const ibSize = (ib.high - ib.low) / (atr || 1);
  const conf = Math.min(83, 56 + Math.max(0, 1 - ibSize) * 16 + (rsi !== null ? Math.abs(50 - rsi) * 0.3 : 0));
  if (conf < minConf) return null;
  return bullBreak
    ? { dir: 'LONG',  conf, reason: `Inside Bar break ↑ IB=${ibSize.toFixed(2)}ATR`, atr, slMult: 1.5, tp1Mult: 2, tp2Mult: 3.5 }
    : { dir: 'SHORT', conf, reason: `Inside Bar break ↓ IB=${ibSize.toFixed(2)}ATR`, atr, slMult: 1.5, tp1Mult: 2, tp2Mult: 3.5 };
}

// Adaptive trend: EMA slope + volume surge
function btSigAdaptiveTrend(slice, minConf) {
  if (slice.length < 35) return null;
  const closes = slice.map(c => c.close);
  const vols   = slice.map(c => c.volume);
  const n = closes.length - 1;
  const ema20 = calcEMA(closes, 20);
  const rsi   = calcRSI(closes, 14);
  const atr   = calcATR(slice, 14)[n];
  const rsiN  = rsi[n];
  if (!ema20[n] || !ema20[n - 10] || rsiN === null) return null;
  const slope5  = (ema20[n] - ema20[n - 5])  / (ema20[n - 5] || 1);
  const slope10 = (ema20[n] - ema20[n - 10]) / (ema20[n - 10] || 1);
  const vol5  = vols.slice(n - 5, n + 1).reduce((s, v) => s + v, 0) / 5;
  const vol20 = vols.slice(n - 20, n).reduce((s, v) => s + v, 0) / 20;
  if (vol20 === 0) return null;
  if (vol5 < vol20 * 1.2) return null;
  const bullSlope = slope5 > 0.0008 && slope10 > 0.0008 && rsiN > 52 && rsiN < 70;
  const bearSlope = slope5 < -0.0008 && slope10 < -0.0008 && rsiN < 48 && rsiN > 30;
  if (!bullSlope && !bearSlope) return null;
  const slopeStr = Math.abs(slope5) * 1000;
  const conf = Math.min(83, 54 + slopeStr * 5 + (bullSlope ? rsiN - 50 : 50 - rsiN) * 0.5);
  if (conf < minConf) return null;
  return bullSlope
    ? { dir: 'LONG',  conf, reason: `Tendencia↑ slope ${(slope5*1000).toFixed(1)}‰ vol↑${(vol5/vol20).toFixed(1)}x`, atr }
    : { dir: 'SHORT', conf, reason: `Tendencia↓ slope ${(slope5*1000).toFixed(1)}‰ vol↑${(vol5/vol20).toFixed(1)}x`, atr };
}

// ── Ultra-Confluence Mean Reversion (5 condiciones simultáneas extremas) ──────
// Señal rara (~2-4 veces por semana en 15m) pero de máxima calidad
// Lógica: RSI(2)<5 + precio ≤ BB inferior + Williams R < -90 + Stoch < 10 + sobre EMA200
// WR empírico documentado en setups similares (Connors 4-6 filter): ~72-82%
function btSigUltraConf(slice, minConf) {
  const closes = slice.map(c => c.close);
  const n    = slice.length - 1;
  if (n < 5) return null;
  const atr  = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  const r2   = calcRSI(closes, 2);
  const bb   = calcBB(closes, 20, 2);
  const wr   = calcWilliamsR(slice, 14);
  const st   = calcStochFull(slice, 14, 3);
  const e200 = calcEMA(closes, Math.min(200, closes.length - 1));
  const e50  = calcEMA(closes, 50);
  const rv2  = r2[n], bbN = bb[n], wrN = wr[n], skN = st.k[n], e2 = e200[n];
  if (rv2 == null || !bbN?.lower || wrN == null || skN == null) return null;

  // Count how many conditions are met
  let longScore = 0, shortScore = 0;
  if (rv2 < 5)                          longScore++;
  if (closes[n] <= bbN.lower)           longScore++;
  if (wrN < -90)                        longScore++;
  if (skN < 10)                         longScore++;
  if (!e2 || closes[n] > e2 * 0.99)    longScore++;   // EMA200 trend filter (optional)
  if (e50[n] && closes[n] > e50[n] * 0.97) longScore++; // near EMA50 = pullback not crash

  if (rv2 > 95)                         shortScore++;
  if (closes[n] >= bbN.upper)           shortScore++;
  if (wrN > -10)                        shortScore++;
  if (skN > 90)                         shortScore++;
  if (!e2 || closes[n] < e2 * 1.01)    shortScore++;
  if (e50[n] && closes[n] < e50[n] * 1.03) shortScore++;

  let dir = null, conf = 0;
  if (longScore >= 5)  { dir = 'LONG';  conf = 55 + longScore * 5; }
  if (shortScore >= 5) { dir = 'SHORT'; conf = 55 + shortScore * 5; }
  if (!dir || conf < minConf) return null;

  const scoreStr = dir === 'LONG' ? longScore : shortScore;
  return {
    dir, conf: Math.min(92, conf),
    reason: `Ultra-Confluencia ${scoreStr}/6 condiciones: RSI2=${rv2.toFixed(1)} WR=${wrN.toFixed(1)} Stoch=${skN.toFixed(1)}`,
    atr, slMult: 1.5, tp1Mult: 2, tp2Mult: 3.5,
  };
}

// ── Walk-Forward Adaptativo ────────────────────────────────────────────────────
// Mini-optimizer sobre las últimas 300 velas → elige el mejor signal
// para las próximas velas (out-of-sample real, no curve-fitting)
const _wfCache = { ts: 0, bestSig: null, bestFilt: null, window: 300 };
function btSigWalkForward(slice, minConf) {
  const n = slice.length - 1;
  if (n < 450) return null; // necesita suficiente historia

  // Recalculate best signal every 50 candles or at start
  if (!_wfCache.bestSig || (n % 50 === 0)) {
    const trainEnd   = n - 50;  // últimas 50 velas = out-of-sample
    const trainStart = trainEnd - _wfCache.window;
    if (trainStart < 200) return null;
    const trainSlice = slice.slice(trainStart, trainEnd);
    const indTrain   = btPrecompute(trainSlice);
    const sigs = ['rsi2_ext','connors_rsi','ultra_conf','zscore_rev','rsi3_pull','rsi_div',
                  'engulf_vol','psar_flip','three_bar_rev','consec_5'];
    const filts = ['none','ema200_align','rsi_zone','macd_dir','adx_trend'];
    let bestWR = 0, bestSig = 'rsi2_ext', bestFilt = 'none';

    for (const sig of sigs) {
      for (const filt of filts) {
        let wins = 0, total = 0;
        for (let ti = 215; ti < trainSlice.length - 1; ti++) {
          const dir = btFSig(indTrain, trainSlice, ti, sig);
          if (!dir || !btFFilter(indTrain, ti, filt, dir)) continue;
          total++;
          const entry = trainSlice[ti].close;
          const atr   = indTrain.atr14[ti] || entry * 0.005;
          const isL   = dir === 'LONG';
          const tp    = isL ? entry + atr * 2 : entry - atr * 2;
          const sl    = isL ? entry - atr * 2 : entry + atr * 2;
          let won = false;
          for (let fi = ti + 1; fi < Math.min(ti + 24, trainSlice.length); fi++) {
            const c = trainSlice[fi];
            if (isL ? c.high >= tp : c.low <= tp)  { won = true; break; }
            if (isL ? c.low  <= sl : c.high >= sl) break;
          }
          if (won) wins++;
        }
        if (total >= 5) {
          const wr = wins / total;
          if (wr > bestWR) { bestWR = wr; bestSig = sig; bestFilt = filt; }
        }
      }
    }
    _wfCache.bestSig  = bestSig;
    _wfCache.bestFilt = bestFilt;
    _wfCache.bestWR   = Math.round(bestWR * 1000) / 10;
  }

  // Apply best signal to current candle (out-of-sample)
  const ind = btPrecompute(slice);
  const dir = btFSig(ind, slice, n, _wfCache.bestSig);
  if (!dir) return null;
  if (!btFFilter(ind, n, _wfCache.bestFilt, dir)) return null;
  const atr = ind.atr14[n] || slice[n].close * 0.005;
  const conf = Math.min(90, 60 + (_wfCache.bestWR || 0) * 0.3);
  if (conf < minConf) return null;
  return {
    dir, conf: Math.round(conf),
    reason: `WalkFwd [${_BT_SIG_LBL[_wfCache.bestSig]||_wfCache.bestSig}+${_BT_FILT_LBL[_wfCache.bestFilt]||_wfCache.bestFilt}] WR:${_wfCache.bestWR}% (train 300c)`,
    atr, slMult: 1.5, tp1Mult: 2, tp2Mult: 3.5,
  };
}

// ── 8 HIGH WIN-RATE named strategies (Connors, Elder, quant papers) ──────────// RSI(2) Extreme Reversion — Connors/Alvarez: ~65-72% WR documented
function btSigRSI2Reversion(slice, minConf) {
  const closes = slice.map(c => c.close);
  const n   = slice.length - 1;
  const r2  = calcRSI(closes, 2);
  const e50 = calcEMA(closes, 50);
  const e200= calcEMA(closes, Math.min(200, closes.length - 1));
  const atr = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  const r = r2[n], rp = r2[n-1];
  if (r == null || rp == null) return null;
  let dir = null, conf = 0;
  if (rp < 10 && r > rp) { dir = 'LONG';  conf = 62 + Math.round((10 - rp) * 1.2); }
  if (rp > 90 && r < rp) { dir = 'SHORT'; conf = 62 + Math.round((rp - 90) * 1.2); }
  if (!dir) return null;
  // Bonus: in-trend confirmation (Connors: only trade with SMA trend)
  const e2 = e200[n], e5 = e50[n];
  if (dir === 'LONG'  && e2 && slice[n].close > e2) conf = Math.min(88, conf + 8);
  if (dir === 'SHORT' && e2 && slice[n].close < e2) conf = Math.min(88, conf + 8);
  if (conf < minConf) return null;
  return { dir, conf, reason: `RSI(2):${r.toFixed(1)} reversión extrema${e5?' +EMA50':''}`, atr, slMult: 1.5, tp1Mult: 2, tp2Mult: 3 };
}
// Connors RSI (composite: RSI3 + StreakRSI + PercentRank100)
function btSigConnorsRSI(slice, minConf) {
  const closes = slice.map(c => c.close);
  const n    = slice.length - 1;
  const crsi = calcConnorsRSI(closes);
  const e200 = calcEMA(closes, Math.min(200, closes.length - 1));
  const atr  = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  const c = crsi[n], cp = crsi[n-1];
  if (c == null || cp == null) return null;
  let dir = null, conf = 0;
  if (cp < 20 && c > cp) { dir = 'LONG';  conf = 62 + Math.round((20 - cp) * 0.8); }
  if (cp > 80 && c < cp) { dir = 'SHORT'; conf = 62 + Math.round((cp - 80) * 0.8); }
  if (!dir) return null;
  const e2 = e200[n];
  if (dir === 'LONG'  && e2 && slice[n].close > e2) conf = Math.min(90, conf + 6);
  if (dir === 'SHORT' && e2 && slice[n].close < e2) conf = Math.min(90, conf + 6);
  if (conf < minConf) return null;
  return { dir, conf, reason: `ConnorsRSI:${c.toFixed(1)} zona ${dir==='LONG'?'<20':'>80'}`, atr, slMult: 1.5, tp1Mult: 2, tp2Mult: 3.5 };
}
// Elder Impulse System — Elder: EMA13 + MACD hist flip together
function btSigElderImpulse(slice, minConf) {
  const closes = slice.map(c => c.close);
  const n    = slice.length - 1;
  const e13  = calcEMA(closes, 13);
  const macd = calcMACD(closes);
  const atr  = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  if (n < 3) return null;
  const eSlope  = e13[n]   > e13[n-1];
  const eSlopeP = e13[n-1] > (e13[n-2] || e13[n-1]);
  const mSlope  = macd.hist[n]   > macd.hist[n-1];
  const mSlopeP = macd.hist[n-1] > (macd.hist[n-2] || macd.hist[n-1]);
  let dir = null;
  if (!eSlopeP && !mSlopeP && eSlope && mSlope)  dir = 'LONG';
  if (eSlopeP  && mSlopeP  && !eSlope && !mSlope) dir = 'SHORT';
  if (!dir) return null;
  let conf = 63;
  const r = calcRSI(closes, 14)[n];
  if (dir === 'LONG'  && r != null && r < 60) conf += 7;
  if (dir === 'SHORT' && r != null && r > 40) conf += 7;
  if (conf < minConf) return null;
  return { dir, conf: Math.min(88, conf), reason: `Elder Impulse: EMA13+MACD ambos ${dir==='LONG'?'↑':'↓'}`, atr };
}
// 5-EMA Fibonacci Ribbon — all 5 aligned = strong trend quality
function btSigFiveRibbon(slice, minConf) {
  const closes = slice.map(c => c.close);
  const n   = slice.length - 1;
  const e5  = calcEMA(closes, 5);
  const e8  = calcEMA(closes, 8);
  const e13 = calcEMA(closes, 13);
  const e21 = calcEMA(closes, 21);
  const e34 = calcEMA(closes, 34);
  const atr = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  const [a,b,c,d,e] = [e5[n],e8[n],e13[n],e21[n],e34[n]];
  const [ap,bp,cp,dp,ep_] = [e5[n-1],e8[n-1],e13[n-1],e21[n-1],e34[n-1]];
  if (!a||!b||!c||!d||!e||!ap||!bp||!cp||!dp||!ep_) return null;
  let dir = null;
  // Requires alignment just formed (not already aligned last bar)
  if (a>b&&b>c&&c>d&&d>e && !(ap>bp&&bp>cp&&cp>dp&&dp>ep_)) dir = 'LONG';
  if (a<b&&b<c&&c<d&&d<e && !(ap<bp&&bp<cp&&cp<dp&&dp<ep_)) dir = 'SHORT';
  if (!dir) return null;
  if (minConf > 72) return null; // rare signal, max natural conf is 72
  return { dir, conf: 72, reason: `5-EMA Ribbon alineado ${dir==='LONG'?'alcista':'bajista'} (Fibonacci)`, atr };
}
// Z-Score Mean Reversion — statistical edge when >2σ from mean
function btSigZScore(slice, minConf) {
  const closes = slice.map(c => c.close);
  const n   = slice.length - 1;
  const bb  = calcBB(closes, 20, 2);
  const atr = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  const b = bb[n]; if (!b?.mid || !b?.upper) return null;
  const std = (b.upper - b.mid) / 2;
  if (std === 0) return null;
  const z = (closes[n] - b.mid) / std;
  let dir = null, conf = 0;
  if (z < -2 && closes[n] > closes[n-1]) { dir = 'LONG';  conf = 60 + Math.min(20, Math.abs(z + 2) * 10); }
  if (z >  2 && closes[n] < closes[n-1]) { dir = 'SHORT'; conf = 60 + Math.min(20, Math.abs(z - 2) * 10); }
  if (!dir || conf < minConf) return null;
  return { dir, conf: Math.min(88, Math.round(conf)), reason: `Z-Score:${z.toFixed(2)} reversión ${dir==='LONG'?'-2σ':'+2σ'}`, atr, slMult: 1.5, tp1Mult: 2.5, tp2Mult: 4 };
}
// 5 Consecutive Bars Reversal — documented ~63-68% WR in mean-reversion markets
function btSigConsecutive5(slice, minConf) {
  const n = slice.length - 1;
  if (n < 6) return null;
  const cs   = slice.slice(n - 5, n);
  const curr = slice[n];
  const atr  = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  const allDown = cs.every(c => c.close < c.open);
  const allUp   = cs.every(c => c.close > c.open);
  let dir = null, conf = 63;
  if (allDown && curr.close > curr.open && curr.close > cs[4].open)  dir = 'LONG';
  if (allUp   && curr.close < curr.open && curr.close < cs[4].open)  dir = 'SHORT';
  if (!dir || conf < minConf) return null;
  const bodyRatio = Math.abs(curr.close - curr.open) / (atr || 1);
  if (bodyRatio > 1.2) conf = Math.min(80, conf + 8);
  return { dir, conf, reason: `5 velas consecutivas ${dir==='LONG'?'bajistas':'alcistas'} + reversión`, atr, slMult: 1.5, tp1Mult: 2.5, tp2Mult: 4 };
}
// RSI(3) Pullback in Trend — Connors TPS (Trading Pullbacks in Stocks/Crypto)
function btSigRSI3Pullback(slice, minConf) {
  const closes = slice.map(c => c.close);
  const n    = slice.length - 1;
  const r3   = calcRSI(closes, 3);
  const s100 = calcSMA(closes, Math.min(100, closes.length - 1));
  const atr  = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  const r = r3[n], rp = r3[n-1], s = s100[n];
  if (r == null || rp == null || s == null) return null;
  let dir = null, conf = 0;
  if (s && closes[n] > s && rp < 20 && r > rp) { dir = 'LONG';  conf = 65 + Math.round((20 - rp) * 0.8); }
  if (s && closes[n] < s && rp > 80 && r < rp) { dir = 'SHORT'; conf = 65 + Math.round((rp - 80) * 0.8); }
  if (!dir || conf < minConf) return null;
  return { dir, conf: Math.min(88, conf), reason: `RSI3:${r.toFixed(1)} pullback en tendencia (SMA100)`, atr, slMult: 1.5, tp1Mult: 2, tp2Mult: 3.5 };
}
// RSI Divergence — price makes new extreme but RSI doesn't confirm
function btSigRSIDivergence(slice, minConf) {
  const closes = slice.map(c => c.close);
  const n    = slice.length - 1;
  const lb   = 10;
  if (n < lb + 2) return null;
  const rsi  = calcRSI(closes, 14);
  const atr  = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  const recentCloses = closes.slice(n - lb, n);
  const recentRSI    = rsi.slice(n - lb, n).filter(v => v != null);
  if (recentRSI.length < lb - 2) return null;
  const minC = Math.min(...recentCloses), maxC = Math.max(...recentCloses);
  const minR = Math.min(...recentRSI),    maxR = Math.max(...recentRSI);
  const r = rsi[n];
  if (r == null) return null;
  let dir = null, conf = 0;
  // Bullish divergence: price at/below recent low but RSI higher than its low
  if (closes[n] <= minC * 1.002 && r > minR + 4) { dir = 'LONG';  conf = 65 + Math.min(15, (r - minR) / 2); }
  // Bearish divergence: price at/above recent high but RSI lower than its high
  if (closes[n] >= maxC * 0.998 && r < maxR - 4) { dir = 'SHORT'; conf = 65 + Math.min(15, (maxR - r) / 2); }
  if (!dir || conf < minConf) return null;
  return { dir, conf: Math.min(88, Math.round(conf)), reason: `RSI Divergencia ${dir==='LONG'?'alcista':'bajista'} (${lb}b)`, atr };
}

// ── 5 additional named strategies ─────────────────────────────────────────────
function btSigWilliamsR(slice, minConf) {
  const n = slice.length - 1;
  const wr  = calcWilliamsR(slice, 14);
  const e20 = calcEMA(slice.map(c => c.close), 20);
  const atr = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  const w = wr[n], wp = wr[n-1];
  if (w == null || wp == null) return null;
  let dir = null, conf = 0;
  if (wp < -80 && w > -80) { dir = 'LONG';  conf = 55 + Math.min(15, Math.abs(wp + 80)); }
  if (wp > -20 && w < -20) { dir = 'SHORT'; conf = 55 + Math.min(15, Math.abs(wp + 20)); }
  if (!dir || conf < minConf) return null;
  const e = e20[n];
  if (e && dir === 'LONG'  && slice[n].close > e * 1.005) return null; // not oversold relative to EMA
  if (e && dir === 'SHORT' && slice[n].close < e * 0.995) return null;
  return { dir, conf: Math.min(88, Math.round(conf)), reason: `WR%:${w.toFixed(1)} rebote desde ${dir==='LONG'?'<-80':'>-20'}`, atr };
}
function btSigCCIMomentum(slice, minConf) {
  const n   = slice.length - 1;
  const cci = calcCCI(slice, 20);
  const rsi = calcRSI(slice.map(c => c.close), 14);
  const atr = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  const c = cci[n], cp = cci[n-1];
  if (c == null || cp == null) return null;
  let dir = null, conf = 0;
  if (cp < -100 && c > -100) { dir = 'LONG';  conf = 55 + Math.min(20, (c - cp) / 2); }
  if (cp > 100  && c < 100)  { dir = 'SHORT'; conf = 55 + Math.min(20, (cp - c) / 2); }
  if (!dir || conf < minConf) return null;
  const r = rsi[n];
  if (dir === 'LONG'  && r != null && r > 65) return null;
  if (dir === 'SHORT' && r != null && r < 35) return null;
  return { dir, conf: Math.min(88, Math.round(conf)), reason: `CCI:${c.toFixed(0)} cruzó ${dir==='LONG'?'-100':'+100'}`, atr };
}
function btSigADXTrend(slice, minConf) {
  const n    = slice.length - 1;
  const adxD = calcADX(slice, 14);
  const atr  = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  const adv = adxD.adx[n], pd = adxD.pdi[n], md = adxD.mdi[n];
  const pdp = adxD.pdi[n-1], mdp = adxD.mdi[n-1];
  if (adv == null || pd == null || md == null || pdp == null || mdp == null) return null;
  if (adv < 20) return null;
  let dir = null, conf = 55 + Math.min(25, adv - 20);
  if (pdp <= mdp && pd > md) dir = 'LONG';
  if (mdp <= pdp && md > pd) dir = 'SHORT';
  if (!dir || conf < minConf) return null;
  return { dir, conf: Math.min(90, Math.round(conf)), reason: `ADX:${adv.toFixed(1)} DI+${pd.toFixed(1)} DI-${md.toFixed(1)}`, atr };
}
function btSigTrix(slice, minConf) {
  const n     = slice.length - 1;
  const closes = slice.map(c => c.close);
  const trix  = calcTRIX(closes, 14);
  const e50   = calcEMA(closes, 50);
  const atr   = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  const t = trix[n], tp = trix[n-1];
  if (t == null || tp == null) return null;
  let dir = null, conf = 0;
  if (tp < 0 && t > 0) { dir = 'LONG';  conf = 55 + Math.min(20, Math.abs(t) * 500); }
  if (tp > 0 && t < 0) { dir = 'SHORT'; conf = 55 + Math.min(20, Math.abs(t) * 500); }
  if (!dir || conf < minConf) return null;
  const e = e50[n];
  if (dir === 'LONG'  && e && slice[n].close < e) conf = Math.max(minConf, conf - 8);
  if (dir === 'SHORT' && e && slice[n].close > e) conf = Math.max(minConf, conf - 8);
  return { dir, conf: Math.min(85, Math.round(conf)), reason: `TRIX:${t.toFixed(2)} cruzó cero`, atr };
}
function btSigHeikinAshi(slice, minConf) {
  const n   = slice.length - 1;
  if (n < 3) return null;
  const ha  = calcHA(slice);
  const rsi = calcRSI(slice.map(c => c.close), 14);
  const atr = calcATR(slice, 14)[n] || slice[n].close * 0.005;
  const bN = ha.close[n]   > ha.open[n];
  const b1 = ha.close[n-1] > ha.open[n-1];
  const b2 = ha.close[n-2] > ha.open[n-2];
  let dir = null, conf = 58;
  if (!b2 && !b1 && bN) dir = 'LONG';
  if (b2  && b1  && !bN) dir = 'SHORT';
  if (!dir || conf < minConf) return null;
  const r = rsi[n];
  if (r != null) {
    if (dir === 'LONG'  && r < 45) conf += 7;
    if (dir === 'SHORT' && r > 55) conf += 7;
  }
  return { dir, conf: Math.min(85, conf), reason: `Heikin-Ashi patrón ${dir==='LONG'?'alcista':'bajista'} 3-bar`, atr };
}

// ── Strategy Portfolio — persisted per timeframe ──────────────────────────────
const BT_PORTFOLIO_KEY = 'btStratPortfolio_v1';

function portfolioLoad() {
  try { return JSON.parse(localStorage.getItem(BT_PORTFOLIO_KEY) || '{}'); } catch { return {}; }
}
function portfolioSave(data) {
  try { localStorage.setItem(BT_PORTFOLIO_KEY, JSON.stringify(data)); } catch {}
}
function portfolioAdd(tf, entry) {
  const data = portfolioLoad();
  if (!data[tf]) data[tf] = [];
  // Avoid exact duplicates
  const dup = data[tf].some(e => e.sig === entry.sig && e.filt === entry.filt && e.sl === entry.sl);
  if (!dup) data[tf].push(entry);
  portfolioSave(data);
}
function portfolioRemove(tf, idx) {
  const data = portfolioLoad();
  if (data[tf]) { data[tf].splice(idx, 1); if (!data[tf].length) delete data[tf]; }
  portfolioSave(data);
}

// ── Helper global: extraer TF de un trade por cualquier propiedad ─────────
function tradeTf(t) {
  return t._tf || t._alertTf ||
    (t._portKey?.match(/^(?:pre_)?port_([^_]+)_/)?.[1]) || null;
}
function portfolioGetForTf(tf) {
  const data = portfolioLoad();
  return data[tf] || [];
}

function renderPortfolioPanel() {
  const panelEl = document.getElementById('btPortfolioPanel');
  if (!panelEl) return;
  const data = portfolioLoad();
  const tfs = Object.keys(data);
  if (!tfs.length) {
    panelEl.innerHTML = '<div style="color:var(--text3);font-size:9px;padding:4px 0">Sin estrategias guardadas. Pulsa 📌 en la Fábrica.</div>';
    return;
  }
  panelEl.innerHTML = tfs.map(tf => {
    const entries = data[tf];
    return `<div style="margin-bottom:6px">
      <div style="font-size:9px;color:var(--accent);font-weight:600;margin-bottom:2px">${tf} — ${entries.length} estrategia${entries.length > 1 ? 's' : ''}</div>
      ${entries.map((e, idx) => `
        <div class="bt-port-row" style="display:flex;align-items:center;gap:4px;font-size:9px;padding:2px 0;border-bottom:1px solid var(--border)">
          <span style="flex:1;color:var(--text1)">${_BT_SIG_LBL[e.sig]||e.sig} <span style="color:var(--text3)">${_BT_FILT_LBL[e.filt]||e.filt}</span></span>
          <span style="color:${e.wr >= 60 ? '#00ff41' : e.wr >= 50 ? '#e3b341' : '#f85149'};font-weight:700">${e.wr}%WR</span>
          <span style="color:var(--text3)">${e.trades}t</span>
          <button class="port-remove-btn" data-tf="${tf}" data-idx="${idx}" style="background:none;border:none;color:#f85149;cursor:pointer;font-size:9px;padding:0 2px">✕</button>
        </div>
      `).join('')}
    </div>`;
  }).join('');
}

function btSigPortfolio(slice, minConf, interval) {
  const strategies = portfolioGetForTf(interval);
  if (!strategies.length) return null;
  const ind = btPrecompute(slice);
  const n   = slice.length - 1;
  let best  = null;
  for (const s of strategies) {
    const dir = btFSig(ind, slice, n, s.sig);
    if (!dir || !btFFilter(ind, n, s.filt, dir)) continue;
    const atr  = ind.atr14[n] || slice[n].close * 0.005;
    // Dynamic conf
    let conf = 60;
    if (s.filt !== 'none') conf += 6;
    const e2 = ind.ema200[n];
    if (e2 && dir === 'LONG'  && slice[n].close > e2) conf += 5;
    if (e2 && dir === 'SHORT' && slice[n].close < e2) conf += 5;
    conf = Math.min(85, conf);
    if (conf < minConf) continue;
    if (!best || conf > best.conf) {
      best = { dir, conf, reason: `[Portfolio ${interval}] ${_BT_SIG_LBL[s.sig]||s.sig} ${_BT_FILT_LBL[s.filt]||s.filt} (WR:${s.wr}%)`, atr, slMult: s.sl, tp1Mult: s.tp1, tp2Mult: s.tp2 };
    }
  }
  return best;
}

function checkPortfolioAlerts(candles, tf) {
  const el = document.getElementById('aPortfolioAlerts');
  if (!el) return;
  const strategies = portfolioGetForTf(tf);
  if (!strategies.length) {
    // No portfolio — clear portStatus so Trading Brain shows "sin estrategias"
    window._portStatus = { tf, total: 0, active: [], waiting: [] };
    el.style.display = '';
    el.innerHTML = `
      <div style="background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:6px;padding:6px 8px;margin:4px 0">
        <div style="font-size:9px;color:var(--text3);text-align:center">
          📋 Sin estrategias en portafolio para <b style="color:var(--accent)">${tf}</b>
          — ve a <b>📊 Backtest → 🧬 Fábrica+</b> y pulsa 📌 para guardar
        </div>
      </div>`;
    return;
  }
  const ind = btPrecompute(candles);
  const n   = candles.length - 1;

  // Check which are currently firing
  const statuses = strategies.map((s, idx) => {
    const dir = btFSig(ind, candles, n, s.sig);
    const ok  = dir && btFFilter(ind, n, s.filt, dir);
    return { ...s, _idx: idx, dir: ok ? dir : null, active: !!ok };
  });

  const active  = statuses.filter(s => s.active);
  const waiting = statuses.filter(s => !s.active);

  // ── Store portfolio status globally so Trading Brain can gate its signal ───
  // Compute projected levels for every strategy (active and waiting)
  const price0 = candles[n].close;
  const atr0   = ind.atr14[n] || price0 * 0.005;
  const statusesWithLevels = statuses.map(s => {
    // We don't know yet if it will be LONG or SHORT for waiting ones — show both
    const projDir = s.active ? s.dir : null;
    const offsets = s.active ? { longOff: 0, shortOff: 0 } : btFSigEntryOffset(ind, candles, n, s.sig);
    const calcLevels = (dir) => {
      const isL = dir === 'LONG';
      const entryOff = isL ? offsets.longOff : offsets.shortOff;
      const entry = price0 + entryOff;
      const sl   = isL ? entry - atr0 * s.sl  : entry + atr0 * s.sl;
      const tp1  = isL ? entry + atr0 * s.tp1 : entry - atr0 * s.tp1;
      const tp2  = isL ? entry + atr0 * s.tp2 : entry - atr0 * s.tp2;
      const rr   = Math.abs(tp1 - entry) / Math.abs(sl - entry);
      return { entry: +entry.toFixed(2), sl: +sl.toFixed(2), tp1: +tp1.toFixed(2), tp2: +tp2.toFixed(2), rr: +rr.toFixed(1) };
    };
    const dist = btFSigDistance(ind, candles, n, s.sig);
    const dpct = btFSigPct(ind, candles, n, s.sig);
    return { ...s, _levels: projDir ? calcLevels(projDir) : { long: calcLevels('LONG'), short: calcLevels('SHORT') }, _dist: { ...dist, ...dpct } };
  });
  window._portStatus = { tf, total: strategies.length,
    active:  statusesWithLevels.filter(s => s.active),
    waiting: statusesWithLevels.filter(s => !s.active) };
  window._lastIndicators = ind;  // Cache for live updates

  // ── Auto-create BotTrades for active signals not yet registered ───────────
  for (const s of active) {
    const dedupeKey = `port_${tf}_${s.sig}_${s.filt}_${s.dir}`;

    // ── LÍMITE: máximo 1 trade activo por timeframe ──────────────────────────
    // Busca trades activos en este TF usando todas las fuentes posibles de TF
    const activeTfTrades = botTrades.filter(t =>
      tradeTf(t) === tf &&
      (t.status === 'alerta' || t.status === 'pending' || t.status === 'running')
    );

    // Si ya hay un trade activo en este TF, verificar si es señal contraria
    if (activeTfTrades.length > 0) {
      const oppositeDir = s.dir === 'LONG' ? 'SHORT' : 'LONG';
      const contraryTrade = activeTfTrades.find(t =>
        t.direction === oppositeDir && (t.status === 'pending' || t.status === 'running')
      );

      if (contraryTrade) {
        // SEÑAL CONTRARIA → cerrar el trade actual y abrir el nuevo (flip)
        const currentPrice = candles[n].close;
        const pnl = contraryTrade.direction === 'LONG'
          ? ((currentPrice - contraryTrade.entry) / contraryTrade.entry * 100)
          : ((contraryTrade.entry - currentPrice) / contraryTrade.entry * 100);

        updateBotTrade(contraryTrade.id, {
          status: pnl >= 0 ? 'win_tp1' : 'loss',
          hitPrice: currentPrice,
          hitTime: Date.now(),
          pnlPct: pnl,
          reason: contraryTrade.reason + ` | 🔄 CERRADO por señal contraria ${s.dir}`,
        });
        saveBotTrades();
        showNotif(`🔄 Flip ${tf}: ${contraryTrade.direction} cerrado (${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%) → abriendo ${s.dir}`, pnl >= 0 ? 'success' : 'error');
        // Cancela también alertas sobrantes del mismo TF
        activeTfTrades.filter(t => t.status === 'alerta' && t.id !== contraryTrade.id).forEach(t => {
          updateBotTrade(t.id, { status: 'expired', hitTime: Date.now(), reason: t.reason + ' | ⛔ Cancelada por flip' });
        });
        saveBotTrades();
        // Continúa para crear el nuevo trade abajo
      } else {
        // TF ya tiene trade/alerta activo en la misma dirección o simplemente está ocupado → skip
        console.log(`[BotTrade] TF ${tf} ya ocupado (${activeTfTrades.map(t=>t.status).join(',')}) — señal ignorada (doble confirmación)`);
        continue;
      }
    }

    const alreadyOpen = botTrades.some(t =>
      (t.status === 'pending' || t.status === 'running') &&
      t._portKey === dedupeKey
    );
    if (alreadyOpen) continue;
    // Upgrade alerta → pending if pre-alert exists
    const alertaTrade = botTrades.find(t => t.status === 'alerta' && t._portKey === `pre_${dedupeKey}`);
    if (alertaTrade) {
      // Upgrade: update levels to exact fire price and promote to pending
      const price = candles[n].close;
      const atr   = ind.atr14[n] || price * 0.005;
      const isL   = s.dir === 'LONG';
      const slP   = isL ? price - atr * s.sl  : price + atr * s.sl;
      const tp1P  = isL ? price + atr * s.tp1 : price - atr * s.tp1;
      const tp2P  = isL ? price + atr * s.tp2 : price - atr * s.tp2;
      updateBotTrade(alertaTrade.id, {
        status: 'pending', entry: price, _tf: tf,
        sl: Math.round(slP*100)/100, tp1: Math.round(tp1P*100)/100, tp2: Math.round(tp2P*100)/100,
        _portKey: dedupeKey,
        reason: `[Portafolio ${tf}] ${_BT_SIG_LBL[s.sig]||s.sig} ${_BT_FILT_LBL[s.filt]||s.filt} (WR:${s.wr}% · ${s.trades}t) — ✅ Señal disparada`,
      });
      saveBotTrades(); renderBotTradesTab();
      showNotif(`📊 Pre-alerta convertida en trade real: ${s.dir} ${currentSymbol}`, 'success');
      continue;
    }

    const price = candles[n].close;
    const atr   = ind.atr14[n] || price * 0.005;
    const isL   = s.dir === 'LONG';
    const slP   = isL ? price - atr * s.sl   : price + atr * s.sl;
    const tp1P  = isL ? price + atr * s.tp1  : price - atr * s.tp1;
    const tp2P  = isL ? price + atr * s.tp2  : price - atr * s.tp2;
    const rr    = Math.abs(tp1P - price) / Math.abs(slP - price);

    const trade = createBotTrade({
      symbol:     currentSymbol,
      direction:  s.dir,
      entry:      price,
      sl:         Math.round(slP * 100) / 100,
      tp1:        Math.round(tp1P * 100) / 100,
      tp2:        Math.round(tp2P * 100) / 100,
      rr:         Math.round(rr * 10) / 10,
      confidence: Math.min(85, 60 + (s.filt !== 'none' ? 6 : 0) + (ind.ema200[n] && ((isL && price > ind.ema200[n]) || (!isL && price < ind.ema200[n])) ? 5 : 0)),
      reason:     `[Portafolio ${tf}] ${_BT_SIG_LBL[s.sig]||s.sig} ${_BT_FILT_LBL[s.filt]||s.filt} (WR:${s.wr}% · ${s.trades}t)`,
    });
    trade._portKey = dedupeKey;
    trade._tf = tf;  // store timeframe for max-1-per-TF enforcement
    saveBotTrades();
    renderBotTradesTab();
    showNotif(`📋 Bot Trade abierto: ${s.dir} ${currentSymbol} por ${_BT_SIG_LBL[s.sig]||s.sig}`, 'info');
  }

  // ── Pre-alertas: crear entrada 'alerta' para estrategias ≥95% sin alerta previa ──
  // Verificar si ya hay un trade activo/alerta en este TF antes de crear alertas
  for (const sw of statusesWithLevels.filter(sw => !sw.active)) {
    const dpct = sw._dist;
    const longPct  = dpct?.longPct  ?? 0;
    const shortPct = dpct?.shortPct ?? 0;
    const bestPct  = Math.max(longPct, shortPct);
    if (bestPct < 95) continue;
    const alertDir = longPct >= shortPct ? 'LONG' : 'SHORT';
    const alertKey = `pre_port_${tf}_${sw.sig}_${sw.filt}_${alertDir}`;
    const alreadyAlert = botTrades.some(t => t.status === 'alerta' && t._portKey === alertKey);
    if (alreadyAlert) continue;
    // ── LÍMITE: máximo 1 alerta/trade activo por timeframe ────────────────
    const tfAlreadyOccupied = botTrades.some(t =>
      (t.status === 'alerta' || t.status === 'pending' || t.status === 'running') &&
      tradeTf(t) === tf
    );
    if (tfAlreadyOccupied) { console.log(`[PreAlerta] TF ${tf} ya ocupado — máx 1`); continue; }
    const priceA = candles[n].close;
    const atrA   = ind.atr14[n] || priceA * 0.005;
    const isLA   = alertDir === 'LONG';
    const lv     = isLA ? (sw._levels?.long || sw._levels) : (sw._levels?.short || sw._levels);
    const entryEst = lv?.entry || priceA;
    const slPA   = isLA ? entryEst - atrA * sw.sl  : entryEst + atrA * sw.sl;
    const tp1PA  = isLA ? entryEst + atrA * sw.tp1 : entryEst - atrA * sw.tp1;
    const tp2PA  = isLA ? entryEst + atrA * sw.tp2 : entryEst - atrA * sw.tp2;
    const rrA    = Math.abs(tp1PA - entryEst) / Math.abs(slPA - entryEst);
    const alertTrade = createBotTrade({
      symbol:     currentSymbol,
      direction:  alertDir,
      entry:      +entryEst.toFixed(2),
      sl:         Math.round(slPA*100)/100,
      tp1:        Math.round(tp1PA*100)/100,
      tp2:        Math.round(tp2PA*100)/100,
      rr:         Math.round(rrA*10)/10,
      confidence: Math.round(bestPct),
      reason:     `[🔔 Pre-alerta ${bestPct}%] ${_BT_SIG_LBL[sw.sig]||sw.sig} ${_BT_FILT_LBL[sw.filt]||sw.filt} (WR:${sw.wr}% · ${sw.trades}t) — espera señal`,
    });
    alertTrade.status = 'alerta';
    alertTrade._portKey = alertKey;
    alertTrade._alertCreatedAt = Date.now();
    alertTrade._alertTf = tf;
    saveBotTrades(); renderBotTradesTab();
    showNotif(`🔔 Pre-alerta ${bestPct}%: ${alertDir} ${currentSymbol} — ${_BT_SIG_LBL[sw.sig]||sw.sig} cerca de disparar`, 'info');
  }

  // ── Update aBrainPortfolio inside Trading Brain ──────────────────────────
  const brainPortEl = document.getElementById('aBrainPortfolio');
  if (brainPortEl) {
    if (!strategies.length) {
      brainPortEl.innerHTML = '';
    } else {
      brainPortEl.innerHTML = `
        <div style="background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.3);border-radius:6px;padding:6px 8px;margin-bottom:8px">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
            <span style="font-size:10px;color:var(--accent);font-weight:700">📋 PORTAFOLIO ${tf} — ${strategies.length} estrategia${strategies.length>1?'s':''}</span>
            <button class="port-refresh-btn" style="background:none;border:none;color:var(--text3);cursor:pointer;font-size:10px">⟳</button>
          </div>
          ${active.length ? `
            <div style="font-size:9px;color:#00ff41;font-weight:600;margin-bottom:3px">🟢 SEÑAL ACTIVA (${active.length})</div>
            ${active.map(s => `
              <div style="display:flex;align-items:center;gap:5px;font-size:10px;padding:4px 0;border-bottom:1px solid rgba(0,255,65,.15)">
                <span style="color:${s.dir==='LONG'?'#00ff41':'#f85149'};font-weight:700;min-width:42px">${s.dir==='LONG'?'▲ LONG':'▼ SHORT'}</span>
                <span style="color:var(--text1);flex:1;font-weight:600">${_BT_SIG_LBL[s.sig]||s.sig}</span>
                <span style="color:var(--text3);font-size:9px">${_BT_FILT_LBL[s.filt]||s.filt}</span>
                <span style="color:#e3b341;font-size:9px">WR:${s.wr}% · ${s.trades}t</span>
                <button class="strat-info-btn" data-sig="${s.sig}" data-filt="${s.filt}" data-wr="${s.wr}" data-trades="${s.trades}" style="background:rgba(0,212,255,.15);border:1px solid rgba(0,212,255,.4);border-radius:3px;color:var(--accent);cursor:pointer;font-size:9px;padding:1px 5px">ℹ️</button>
                <button class="port-remove-btn" data-tf="${tf}" data-idx="${s._idx}" style="background:none;border:none;color:#f85149;cursor:pointer;font-size:11px;padding:0 3px">✕</button>
              </div>
            `).join('')}
          ` : ''}
          ${waiting.length ? `
            <div style="font-size:9px;color:var(--text3);font-weight:600;margin:${active.length?'5px':0} 0 2px">⏳ EN ESPERA — sin señal ahora (${waiting.length})</div>
            ${waiting.map(s => `
              <div style="display:flex;align-items:center;gap:5px;font-size:9px;padding:2px 0;opacity:.6">
                <span style="color:var(--text3);min-width:42px">—</span>
                <span style="color:var(--text2);flex:1">${_BT_SIG_LBL[s.sig]||s.sig}</span>
                <span style="color:var(--text3);font-size:8px">${_BT_FILT_LBL[s.filt]||s.filt}</span>
                <span style="color:var(--text3);font-size:8px">WR:${s.wr}% · ${s.trades}t</span>
                <button class="strat-info-btn" data-sig="${s.sig}" data-filt="${s.filt}" data-wr="${s.wr}" data-trades="${s.trades}" style="background:rgba(0,212,255,.15);border:1px solid rgba(0,212,255,.4);border-radius:3px;color:var(--accent);cursor:pointer;font-size:8px;padding:1px 4px;opacity:1">ℹ️</button>
                <button class="port-remove-btn" data-tf="${tf}" data-idx="${s._idx}" style="background:none;border:none;color:#f85149;cursor:pointer;font-size:11px;padding:0 3px">✕</button>
              </div>
            `).join('')}
          ` : ''}
        </div>`;
    }
  }

  el.style.display = '';
  el.innerHTML = `
    <div style="background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.25);border-radius:6px;padding:6px 8px;margin:4px 0">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px">
        <span style="font-size:9px;color:var(--accent);font-weight:700">📋 PORTAFOLIO ${tf} — ${strategies.length} estrategia${strategies.length>1?'s':''}</span>
        <button class="port-refresh-btn" style="background:none;border:none;color:var(--text3);cursor:pointer;font-size:9px">⟳</button>      </div>

      ${active.length ? `
        <div style="font-size:8px;color:#00ff41;font-weight:600;margin-bottom:2px">🟢 ACTIVAS — Bot Trade creado (${active.length})</div>
        ${active.map(s => `
          <div style="display:flex;align-items:center;gap:4px;font-size:9px;padding:3px 0;border-bottom:1px solid rgba(255,255,255,.06)">
            <span style="color:${s.dir==='LONG'?'#00ff41':'#f85149'};font-weight:700;min-width:36px">${s.dir==='LONG'?'▲ LONG':'▼ SHORT'}</span>
            <span style="color:var(--text1);flex:1">${_BT_SIG_LBL[s.sig]||s.sig}</span>
            <span style="color:var(--text3);font-size:8px">${_BT_FILT_LBL[s.filt]||s.filt}</span>
            <span style="color:#e3b341;font-size:8px">WR:${s.wr}%</span>
            <button title="Eliminar estrategia del portafolio" class="port-remove-btn" data-tf="${tf}" data-idx="${s._idx}" style="background:none;border:none;color:#f85149;cursor:pointer;font-size:11px;padding:0 3px;line-height:1">✕</button>
          </div>
        `).join('')}
      ` : ''}

      ${waiting.length ? `
        <div style="font-size:8px;color:var(--text3);font-weight:600;margin:${active.length?'5px':0} 0 2px">⏳ EN ESPERA — sin señal ahora (${waiting.length})</div>
        ${waiting.map(s => `
          <div style="display:flex;align-items:center;gap:4px;font-size:9px;padding:2px 0;opacity:.65">
            <span style="color:var(--text3);min-width:36px">—</span>
            <span style="color:var(--text2);flex:1">${_BT_SIG_LBL[s.sig]||s.sig}</span>
            <span style="color:var(--text3);font-size:8px">${_BT_FILT_LBL[s.filt]||s.filt}</span>
            <span style="color:var(--text3);font-size:8px">WR:${s.wr}% · ${s.trades}t</span>
            <button title="Eliminar estrategia del portafolio" class="port-remove-btn" data-tf="${tf}" data-idx="${s._idx}" style="background:none;border:none;color:#f85149;cursor:pointer;font-size:11px;padding:0 3px;line-height:1">✕</button>
          </div>
        `).join('')}
      ` : ''}
    </div>
  `;
}

// Update portfolio progress bars with live price (called from WebSocket every ~1s)
function updatePortfolioProgressLive(livePrice) {
  const status = window._portStatus;
  const ind = window._lastIndicators;
  const candles = window._lastCandles;
  if (!status?.waiting?.length || !ind || !candles?.length) return;
  
  const i = candles.length - 1;
  const lastCandle = candles[i];
  // Price-dependent signals that can update intra-candle
  const PRICE_SIGS = ['price_ema20', 'bb_mid', 'bb_bounce', 'ema_8_21', 'ema_20_50', 'keltner', 'donchian_10', 'zscore_rev'];
  
  for (const s of status.waiting) {
    if (!PRICE_SIGS.includes(s.sig)) continue; // Skip signals that don't depend on live price
    
    // Create synthetic candle with live price (keep indicators from last closed candle)
    const liveCandle = { ...lastCandle, close: livePrice, high: Math.max(lastCandle.high, livePrice), low: Math.min(lastCandle.low, livePrice) };
    const liveCandles = [...candles.slice(0, -1), liveCandle];
    
    // Recalculate pct with live price
    const livePct = btFSigPct({ ...ind, closes: [...ind.closes.slice(0, -1), livePrice] }, liveCandles, i, s.sig);
    const liveDistTxt = btFSigDistance(ind, liveCandles, i, s.sig);
    
    // Update DOM for this strategy's bars
    const container = document.querySelector(`[data-strat-sig="${s.sig}"][data-strat-filt="${s.filt}"]`);
    if (!container) continue;
    
    const longDiv = container.querySelector('[data-dir="LONG"]');
    const shortDiv = container.querySelector('[data-dir="SHORT"]');
    
    if (longDiv) {
      const longPct = livePct.longPct ?? 0;
      const bc = longPct >= 80 ? '#00ff41' : longPct >= 50 ? '#f0883e' : '#e3b341';
      const pctEl = longDiv.querySelector('.strat-pct-val');
      const barEl = longDiv.querySelector('.strat-pct-bar');
      const txtEl = longDiv.querySelector('.strat-dist-txt');
      if (pctEl) { pctEl.textContent = `${longPct}%`; pctEl.style.color = bc; }
      if (barEl) { barEl.style.width = `${longPct}%`; barEl.style.background = bc; }
      if (txtEl && liveDistTxt.long) { txtEl.textContent = liveDistTxt.long; txtEl.style.color = liveDistTxt.long.startsWith('✅') ? '#00ff41' : 'var(--text3)'; }
    }
    
    if (shortDiv) {
      const shortPct = livePct.shortPct ?? 0;
      const bc = shortPct >= 80 ? '#f85149' : shortPct >= 50 ? '#f0883e' : '#e3b341';
      const pctEl = shortDiv.querySelector('.strat-pct-val');
      const barEl = shortDiv.querySelector('.strat-pct-bar');
      const txtEl = shortDiv.querySelector('.strat-dist-txt');
      if (pctEl) { pctEl.textContent = `${shortPct}%`; pctEl.style.color = bc; }
      if (barEl) { barEl.style.width = `${shortPct}%`; barEl.style.background = bc; }
      if (txtEl && liveDistTxt.short) { txtEl.textContent = liveDistTxt.short; txtEl.style.color = liveDistTxt.short.startsWith('✅') ? '#f85149' : 'var(--text3)'; }
    }
  }
}

function btDispatchSignal(slice, strategy, minConf, symbol, interval) {
  switch (strategy) {
    case 'ema_trend':      return btSigEMATrend(slice, minConf);
    case 'rsi_macd':       return btSigMACDRSI(slice, minConf);
    case 'bb_bounce':      return btSigBBBounce(slice, minConf);
    case 'supertrend':     return btSigSupertrend(slice, minConf);
    case 'rsi_reversal':   return btSigRSIReversal(slice, minConf);
    case 'ema_pullback':   return btSigEMAPullback(slice, minConf);
    case 'stoch_cross':    return btSigStochCross(slice, minConf);
    case 'engulfing':      return btSigEngulfing(slice, minConf);
    case 'triple_ema':     return btSigTripleEMA(slice, minConf);
    case 'donchian':       return btSigDonchian(slice, minConf);
    case 'momentum':       return btSigMomentum(slice, minConf);
    case 'bb_squeeze':     return btSigBBSqueeze(slice, minConf);
    case 'multi_conf':     return btSigMultiConf(slice, minConf);
    case 'hammer':         return btSigHammer(slice, minConf);
    case 'inside_bar':     return btSigInsideBar(slice, minConf);
    case 'adaptive_trend': return btSigAdaptiveTrend(slice, minConf);
    case 'williams_r':     return btSigWilliamsR(slice, minConf);
    case 'cci':            return btSigCCIMomentum(slice, minConf);
    case 'adx_trend':      return btSigADXTrend(slice, minConf);
    case 'trix':           return btSigTrix(slice, minConf);
    case 'heikin_ashi':    return btSigHeikinAshi(slice, minConf);
    case 'rsi2_rev':       return btSigRSI2Reversion(slice, minConf);
    case 'connors_rsi':    return btSigConnorsRSI(slice, minConf);
    case 'elder_impulse':  return btSigElderImpulse(slice, minConf);
    case 'five_ribbon':    return btSigFiveRibbon(slice, minConf);
    case 'zscore':         return btSigZScore(slice, minConf);
    case 'consecutive5':   return btSigConsecutive5(slice, minConf);
    case 'rsi3_pullback':  return btSigRSI3Pullback(slice, minConf);
    case 'rsi_divergence': return btSigRSIDivergence(slice, minConf);
    case 'ultra_conf':     return btSigUltraConf(slice, minConf);
    case 'walk_forward':   return btSigWalkForward(slice, minConf);
    case 'portfolio':      return btSigPortfolio(slice, minConf, interval);
    case 'factory_combo': {
      const fc = window._btFactorySelected;
      if (!fc) return null;
      const ind2 = btPrecompute(slice);
      const n2   = slice.length - 1;
      const dir  = btFSig(ind2, slice, n2, fc.sig);
      if (!dir || !btFFilter(ind2, n2, fc.filt, dir)) return null;
      const atr2 = ind2.atr14[n2] || slice[n2].close * 0.005;
      // Conf dinámica: base 60 + bonus por filtro activo + bonus por tendencia EMA200
      let conf = 60;
      if (fc.filt !== 'none') conf += 6;
      const e2 = ind2.ema200[n2];
      if (e2 && dir === 'LONG'  && slice[n2].close > e2) conf += 5;
      if (e2 && dir === 'SHORT' && slice[n2].close < e2) conf += 5;
      conf = Math.min(85, conf);
      if (conf < minConf) return null;
      return { dir, conf, reason: `[Fábrica] ${_BT_SIG_LBL[fc.sig]||fc.sig} ${_BT_FILT_LBL[fc.filt]||fc.filt}`, atr: atr2, slMult: fc.sl, tp1Mult: fc.tp1, tp2Mult: fc.tp2 };
    }
    case 'analysis': default: {
      let a;
      try { a = runAnalysis(slice, symbol, interval); } catch { return null; }
      const b = a?.brain;
      if (!b || b.finalDir === 'ESPERAR' || !b.finalConf || b.finalConf < minConf) return null;
      const atr = a.indicators?.atr || slice[slice.length - 1].close * 0.005;
      return { dir: b.finalDir, conf: b.finalConf, reason: (b.finalReason || '').slice(0, 80), atr };
    }
  }
}

const _btCandleCache = {};

async function fetchKlinesBacktest(symbol, interval, days, onProgress) {
  const cacheKey = `${symbol}-${interval}-${days}`;
  const cached = _btCandleCache[cacheKey];
  if (cached && (Date.now() - cached.ts) < 300000) { // 5 min cache
    if (onProgress) onProgress(50, `📦 Caché (${cached.data.length} velas)`);
    return cached.data;
  }
  const perDay = BT_CANDLES_PER_DAY[interval] || 96;
  const totalNeeded = perDay * days + 210; // +210 warm-up para indicadores
  const iv = TF_MAP[interval] || '15m';
  const allCandles = [];
  let endTime = Date.now();

  if (onProgress) onProgress(0, `Descargando datos históricos (${totalNeeded} velas)...`);

  while (allCandles.length < totalNeeded) {
    const remaining = totalNeeded - allCandles.length;
    const batchSize = Math.min(1000, remaining);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    let raw;
    try {
      const r = await fetch(
        `${BINANCE}/klines?symbol=${symbol}&interval=${iv}&limit=${batchSize}&endTime=${endTime}`,
        { signal: controller.signal }
      );
      clearTimeout(timer);
      if (!r.ok) throw new Error(`Binance ${r.status}`);
      raw = await r.json();
    } catch (e) {
      clearTimeout(timer);
      throw e;
    }

    if (!raw || !raw.length) break;

    const batch = raw.map(k => ({
      time: k[0], open: parseFloat(k[1]), high: parseFloat(k[2]),
      low: parseFloat(k[3]), close: parseFloat(k[4]), volume: parseFloat(k[5]),
    }));
    allCandles.unshift(...batch);
    endTime = raw[0][0] - 1;
    if (raw.length < batchSize) break;

    const pct = Math.min(50, Math.round(allCandles.length / totalNeeded * 50));
    if (onProgress) onProgress(pct, `Descargando... ${allCandles.length}/${totalNeeded} velas`);
  }

  _btCandleCache[cacheKey] = { data: allCandles, ts: Date.now() };
  return allCandles;
}

// Core simulation engine — reusable without fetch (for auto-optimizer)
function runBacktestOnCandles(candles, cfg) {
  const { symbol, interval, minConf, margin, leverage, strategy = 'analysis' } = cfg;
  const step = strategy === 'analysis'
    ? (interval === '1m' ? 15 : interval === '5m' ? 12 : interval === '15m' ? 4 : 1)
    : 1;
  const trades = [];
  let openTrade = null;
  const warmup = 210;
  for (let i = warmup; i < candles.length; i++) {
    if (openTrade) {
      const c = candles[i];
      const isLong = openTrade.direction === 'LONG';
      const slHit   = isLong ? c.low  <= openTrade.sl  : c.high >= openTrade.sl;
      const tp2Hit  = isLong ? c.high >= openTrade.tp2 : c.low  <= openTrade.tp2;
      const tp1Hit  = isLong ? c.high >= openTrade.tp1 : c.low  <= openTrade.tp1;
      const maxHold = (i - openTrade.signalIdx) > (BT_CANDLES_PER_DAY[interval] || 96) * 2;
      let closed = false;
      if (slHit && !tp1Hit) {
        const p = isLong ? (openTrade.sl - openTrade.entry) / openTrade.entry * 100 : (openTrade.entry - openTrade.sl) / openTrade.entry * 100;
        trades.push({ ...openTrade, status: 'loss',    exitPrice: openTrade.sl,  exitTime: c.time, exitIdx: i, pnlPct: p, pnlUsd: margin * (p / 100) * leverage });
        closed = true;
      } else if (tp2Hit) {
        const p = isLong ? (openTrade.tp2 - openTrade.entry) / openTrade.entry * 100 : (openTrade.entry - openTrade.tp2) / openTrade.entry * 100;
        trades.push({ ...openTrade, status: 'win_tp2', exitPrice: openTrade.tp2, exitTime: c.time, exitIdx: i, pnlPct: p, pnlUsd: margin * (p / 100) * leverage });
        closed = true;
      } else if (tp1Hit) {
        const p = isLong ? (openTrade.tp1 - openTrade.entry) / openTrade.entry * 100 : (openTrade.entry - openTrade.tp1) / openTrade.entry * 100;
        trades.push({ ...openTrade, status: 'win_tp1', exitPrice: openTrade.tp1, exitTime: c.time, exitIdx: i, pnlPct: p, pnlUsd: margin * (p / 100) * leverage });
        closed = true;
      } else if (maxHold) {
        const p = isLong ? (c.close - openTrade.entry) / openTrade.entry * 100 : (openTrade.entry - c.close) / openTrade.entry * 100;
        trades.push({ ...openTrade, status: 'timeout', exitPrice: c.close,       exitTime: c.time, exitIdx: i, pnlPct: p, pnlUsd: margin * (p / 100) * leverage });
        closed = true;
      }
      if (closed) openTrade = null;
      if (!closed) continue;
    }
    if ((i - warmup) % step !== 0) continue;
    const slice = candles.slice(i - warmup, i);
    const sig = btDispatchSignal(slice, strategy, minConf, symbol, interval);
    if (!sig) continue;
    const isLong    = sig.dir === 'LONG';
    const sigCandle = candles[i - 1];
    const entry     = sigCandle.close;
    const atr       = sig.atr || entry * 0.005;
    const slM  = sig.slMult  || 2;
    const tp1M = sig.tp1Mult || 3;
    const tp2M = sig.tp2Mult || 5;
    openTrade = {
      direction: sig.dir, confidence: sig.conf, reason: sig.reason,
      entry,
      sl:  isLong ? entry - atr * slM  : entry + atr * slM,
      tp1: isLong ? entry + atr * tp1M : entry - atr * tp1M,
      tp2: isLong ? entry + atr * tp2M : entry - atr * tp2M,
      rr: tp1M / slM, signalIdx: i, signalTime: sigCandle.time,
    };
  }
  return trades;
}

async function runBacktest(cfg, onProgress) {
  const { symbol, interval, days, strategy = 'analysis' } = cfg;
  const BT_ALL_LABELS = { analysis: 'Análisis', ema_trend: 'EMA Trend', rsi_macd: 'MACD+RSI', bb_bounce: 'BB Bounce', supertrend: 'Supertrend', rsi_reversal: 'RSI Reversal', ema_pullback: 'EMA Pullback', stoch_cross: 'Stoch Cross', engulfing: 'Engulfing', triple_ema: 'Triple EMA', donchian: 'Donchian', momentum: 'Momentum', bb_squeeze: 'BB Squeeze', multi_conf: 'Multi Conf', hammer: 'Hammer', inside_bar: 'Inside Bar', adaptive_trend: 'Adapt.Trend' };
  const stratLabel = BT_ALL_LABELS[strategy] || strategy;
  const candles = await fetchKlinesBacktest(symbol, interval, days, onProgress);
  if (candles.length < 250) throw new Error(`Datos insuficientes: ${candles.length} velas`);
  if (onProgress) onProgress(55, `[${stratLabel}] Simulando ${candles.length} velas...`);
  const trades = runBacktestOnCandles(candles, cfg);
  if (onProgress) onProgress(100, 'Backtest completado');
  return trades;
}

function renderBacktestResults(trades, cfg) {
  const { margin, leverage, strategy = 'analysis', symbol, interval, days } = cfg;
  const stratNames = { analysis: '🔬 Análisis completo', ema_trend: '📈 EMA Trend', rsi_macd: '⚡ MACD+RSI', bb_bounce: '🎯 BB Bounce', supertrend: '🌀 Supertrend', rsi_reversal: '↩️ RSI Reversal', ema_pullback: '📉 EMA Pullback', stoch_cross: '📊 Stoch Cross', engulfing: '🕯️ Engulfing', triple_ema: '3️⃣ Triple EMA', donchian: '🔳 Donchian', momentum: '🚀 Momentum', bb_squeeze: '💥 BB Squeeze', multi_conf: '🔗 Multi Conf', hammer: '🔨 Hammer/Star', inside_bar: '📦 Inside Bar', adaptive_trend: '🌟 Adapt.Trend' };
  const titleEl = document.getElementById('bt2ResultTitle');
  if (titleEl) titleEl.textContent = `${stratNames[strategy] || strategy} — ${symbol} ${interval} ${days}d`;
  if (titleEl) titleEl.style.display = '';

  // ── Stats ─────────────────────────────────────────────────────
  const wins    = trades.filter(t => t.status.startsWith('win'));
  const losses  = trades.filter(t => t.status === 'loss');
  const timeout = trades.filter(t => t.status === 'timeout');
  const longs   = trades.filter(t => t.direction === 'LONG');
  const shorts  = trades.filter(t => t.direction === 'SHORT');
  const wr      = trades.length ? Math.round(wins.length / trades.length * 100) : 0;
  const totalPnlUsd = trades.reduce((s, t) => s + (t.pnlUsd || 0), 0);
  const avgPnlPct   = trades.length ? trades.reduce((s, t) => s + (t.pnlPct || 0), 0) / trades.length : 0;
  const bestPnl     = trades.length ? Math.max(...trades.map(t => t.pnlPct)) : 0;
  const worstPnl    = trades.length ? Math.min(...trades.map(t => t.pnlPct)) : 0;

  const wrColor  = wr >= 55 ? '#00ff41' : wr >= 45 ? '#e3b341' : '#f85149';
  const pnlColor = totalPnlUsd >= 0 ? '#00ff41' : '#f85149';

  // Stats cards
  document.getElementById('bt2Stats').style.display = 'grid';
  document.getElementById('bt2TotalTrades').textContent = trades.length;
  document.getElementById('bt2WinRate').textContent = wr + '%';
  document.getElementById('bt2WinRate').style.color = wrColor;
  document.getElementById('bt2TotalPnl').textContent = (totalPnlUsd >= 0 ? '+' : '') + '$' + totalPnlUsd.toFixed(2);
  document.getElementById('bt2TotalPnl').style.color = pnlColor;
  document.getElementById('bt2AvgPnl').textContent = (avgPnlPct >= 0 ? '+' : '') + avgPnlPct.toFixed(2) + '%';
  document.getElementById('bt2AvgPnl').style.color = avgPnlPct >= 0 ? '#00ff41' : '#f85149';
  document.getElementById('bt2BestTrade').textContent = '+' + bestPnl.toFixed(1) + '%';
  document.getElementById('bt2BestTrade').style.color = '#00ff41';
  document.getElementById('bt2WorstTrade').textContent = worstPnl.toFixed(1) + '%';
  document.getElementById('bt2WorstTrade').style.color = '#f85149';
  document.getElementById('bt2Longs').textContent = longs.length + (longs.length ? ` (${Math.round(longs.filter(t=>t.status.startsWith('win')).length/longs.length*100)}%WR)` : '');
  document.getElementById('bt2Shorts').textContent = shorts.length + (shorts.length ? ` (${Math.round(shorts.filter(t=>t.status.startsWith('win')).length/Math.max(1,shorts.length)*100)}%WR)` : '');

  // ── Equity mini-chart ─────────────────────────────────────────
  const eqEl = document.getElementById('bt2Equity');
  const eqRow = document.getElementById('bt2EquityRow');
  if (trades.length > 0) {
    eqEl.style.display = '';
    let equity = 0;
    const maxPnl = Math.max(Math.abs(Math.max(...trades.map(t => t.pnlUsd))), 0.01);
    eqRow.innerHTML = trades.map(t => {
      const h = Math.round(Math.abs(t.pnlUsd) / maxPnl * 28) + 2;
      const col = t.status.startsWith('win') ? '#00ff41' : t.status === 'timeout' ? '#e3b341' : '#f85149';
      equity += t.pnlUsd;
      const dt = new Date(t.signalTime).toLocaleDateString('es', { month: 'short', day: 'numeric' });
      return `<div class="bt2-eq-bar" style="height:${h}px;background:${col};opacity:.8" title="${dt}: ${(t.pnlPct >= 0 ? '+' : '')}${t.pnlPct.toFixed(2)}% ($${t.pnlUsd.toFixed(2)})\nEquity: $${equity.toFixed(2)}"></div>`;
    }).join('');
  }

  // ── Trade list ─────────────────────────────────────────────────
  const listEl = document.getElementById('bt2TradeList');
  const container = document.getElementById('bt2TradesContainer');
  const countEl = document.getElementById('bt2ListCount');
  listEl.style.display = '';
  if (countEl) countEl.textContent = `${trades.length} trades`;

  if (!trades.length) {
    container.innerHTML = '<div style="text-align:center;padding:20px;color:var(--text2);font-size:9px;">No se encontraron señales con los parámetros elegidos.</div>';
    return;
  }

  // Header row
  container.innerHTML = `
    <div class="bt2-trade-row" style="margin-bottom:6px">
      <span class="bt2-th">Fecha</span>
      <span class="bt2-th">Dir.</span>
      <span class="bt2-th">Entrada</span>
      <span class="bt2-th">Salida</span>
      <span class="bt2-th">Conf.</span>
      <span class="bt2-th">PnL %</span>
      <span class="bt2-th">PnL USD</span>
      <span class="bt2-th">Estado</span>
    </div>
    ${trades.map(t => {
      const col = t.status.startsWith('win') ? '#00ff41' : t.status === 'timeout' ? '#e3b341' : '#f85149';
      const dirCol = t.direction === 'LONG' ? '#00ff41' : '#f85149';
      const icon = t.direction === 'LONG' ? '▲' : '▼';
      const statusLabel = t.status === 'win_tp1' ? '✅ TP1' : t.status === 'win_tp2' ? '✅ TP2' : t.status === 'loss' ? '❌ SL' : '⏱ Exp.';
      const dt = new Date(t.signalTime).toLocaleDateString('es', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      const confColor = t.confidence >= 70 ? '#00ff41' : t.confidence >= 60 ? '#e3b341' : '#aaa';
      return `<div class="bt2-trade-row" title="${escHtml(t.reason)}">
        <span class="bt2-tr-date">${dt}</span>
        <span class="bt2-tr-dir" style="color:${dirCol}">${icon} ${t.direction}</span>
        <span class="bt2-tr-entry">$${fmtPrice(t.entry)}</span>
        <span class="bt2-tr-exit">$${fmtPrice(t.exitPrice)}</span>
        <span class="bt2-tr-pnl" style="color:${confColor};font-weight:600">${Math.round(t.confidence)}%</span>
        <span class="bt2-tr-pnl" style="color:${col}">${(t.pnlPct >= 0 ? '+' : '')}${t.pnlPct.toFixed(2)}%</span>
        <span class="bt2-tr-usd" style="color:${col}">${(t.pnlUsd >= 0 ? '+' : '')}$${Math.abs(t.pnlUsd).toFixed(2)}</span>
        <span class="bt2-tr-status" style="color:${col}">${statusLabel}</span>
      </div>`;
    }).join('')}
  `;
}

async function startBacktest() {
  const btn = document.getElementById('bt2RunBtn');
  const progEl = document.getElementById('bt2Progress');
  const progFill = document.getElementById('bt2ProgressFill');
  const progLbl = document.getElementById('bt2ProgressLbl');
  const statsEl = document.getElementById('bt2Stats');
  const eqEl = document.getElementById('bt2Equity');
  const listEl = document.getElementById('bt2TradeList');

  const cfg = {
    symbol:   (document.getElementById('bt2Symbol')?.value || 'BTCUSDT').trim().toUpperCase(),
    interval: document.getElementById('bt2Tf')?.value || '15m',
    days:     parseInt(document.getElementById('bt2Days')?.value || '30'),
    minConf:  Math.max(10, Math.min(99, parseInt(document.getElementById('bt2MinConf')?.value || '55'))),
    margin:   parseFloat(document.getElementById('bt2Margin')?.value || '100'),
    leverage: parseInt(document.getElementById('bt2Lev')?.value || '10'),
    strategy: document.getElementById('bt2Strategy')?.value || 'analysis',
  };

  btn.disabled = true;
  btn.textContent = '⏳ Ejecutando...';
  progEl.style.display = '';
  statsEl.style.display = 'none';
  if (eqEl) eqEl.style.display = 'none';
  if (listEl) listEl.style.display = 'none';

  const onProgress = (pct, msg) => {
    if (progFill) progFill.style.width = pct + '%';
    if (progLbl) progLbl.textContent = msg;
  };

  try {
    const trades = await runBacktest(cfg, onProgress);
    renderBacktestResults(trades, cfg);
  } catch (e) {
    if (progLbl) progLbl.textContent = '❌ Error: ' + e.message;
    showNotif('Backtest error: ' + e.message, 'error');
    console.error('[Backtest]', e);
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Ejecutar Backtest';
  }
}

// ═══════════════════════════════════════════════════════════════════
// AUTO-OPTIMIZER
// ═══════════════════════════════════════════════════════════════════

const BT_OPT_LABELS = {
  ema_trend: '📈 EMA Trend', rsi_macd: '⚡ MACD+RSI', bb_bounce: '🎯 BB Bounce',
  supertrend: '🌀 Supertrend', rsi_reversal: '↩️ RSI Reversal', ema_pullback: '📉 EMA Pullback',
  stoch_cross: '📊 Stoch Cross', engulfing: '🕯️ Engulfing', triple_ema: '3️⃣ Triple EMA', donchian: '🔳 Donchian',
  momentum: '🚀 Momentum', bb_squeeze: '💥 BB Squeeze', multi_conf: '🔗 Multi Conf',
  hammer: '🔨 Hammer/Star', inside_bar: '📦 Inside Bar', adaptive_trend: '🌟 Tendencia Adaptativa',
};

async function runAutoOptimize(symbol, interval, days, margin, leverage, onProgress) {
  const candles = await fetchKlinesBacktest(symbol, interval, days,
    p => onProgress(Math.round(p * 0.4), `Descargando ${symbol} ${interval} ${days}d...`));
  if (candles.length < 250) throw new Error('Datos insuficientes');

  const strategies = Object.keys(BT_OPT_LABELS);
  const confLevels = [45, 50, 55, 60, 65, 70];
  const results = [];
  const total = strategies.length * confLevels.length;
  let done = 0;

  for (const strat of strategies) {
    for (const conf of confLevels) {
      done++;
      if (done % 5 === 0) await new Promise(r => setTimeout(r, 0)); // yield to UI
      onProgress(40 + Math.round(done / total * 58),
        `[${done}/${total}] ${BT_OPT_LABELS[strat]} conf:${conf}%`);

      const trades = runBacktestOnCandles(candles, { symbol, interval, minConf: conf, margin, leverage, strategy: strat });
      if (trades.length < 5) continue;

      const wins   = trades.filter(t => t.status.startsWith('win'));
      const losses = trades.filter(t => t.status === 'loss');
      const wr = wins.length / trades.length * 100;
      const totalPnl = trades.reduce((s, t) => s + t.pnlUsd, 0);
      const grossWin  = wins.reduce((s, t) => s + Math.abs(t.pnlUsd), 0);
      const grossLoss = losses.reduce((s, t) => s + Math.abs(t.pnlUsd), 0);
      const pf = grossLoss > 0 ? grossWin / grossLoss : (grossWin > 0 ? 99 : 0);

      results.push({
        strategy: strat, stratLabel: BT_OPT_LABELS[strat], conf,
        trades: trades.length, wr: Math.round(wr * 10) / 10,
        totalPnl: Math.round(totalPnl * 100) / 100,
        pf: Math.round(pf * 100) / 100,
      });
    }
  }

  onProgress(100, `Completado — ${results.length}/${total} configs válidas`);
  return results.sort((a, b) => b.wr !== a.wr ? b.wr - a.wr : b.totalPnl - a.totalPnl);
}

function renderOptimizerResults(results, cfg) {
  const container = document.getElementById('bt2OptResults');
  const table     = document.getElementById('bt2OptTable');
  const subtitle  = document.getElementById('bt2OptSubtitle');
  const countEl   = document.getElementById('bt2OptCount');
  if (!container || !table) return;

  container.style.display = '';
  if (countEl)   countEl.textContent  = `${results.length} válidas`;
  if (subtitle)  subtitle.textContent = `${cfg.symbol} ${cfg.interval} ${cfg.days}d · $${cfg.margin}×${cfg.leverage}x · orden: WR → PnL`;

  const medals = ['🥇', '🥈', '🥉'];
  const top20  = results.slice(0, 20);

  table.innerHTML = `
    <div class="bt2-opt-row bt2-opt-header">
      <span>#</span><span>Estrategia</span><span>Conf</span><span>N</span>
      <span>WR%</span><span>PnL$</span><span>PF</span><span></span>
    </div>
    ${top20.map((r, idx) => {
      const wrCol  = r.wr >= 58 ? '#00ff41' : r.wr >= 48 ? '#e3b341' : '#f85149';
      const pnlCol = r.totalPnl >= 0 ? '#00ff41' : '#f85149';
      return `<div class="bt2-opt-row" data-strat="${r.strategy}" data-conf="${r.conf}">
        <span>${medals[idx] || (idx + 1)}</span>
        <span class="bt2-opt-name">${r.stratLabel}</span>
        <span>${r.conf}%</span>
        <span>${r.trades}</span>
        <span style="color:${wrCol};font-weight:700">${r.wr}%</span>
        <span style="color:${pnlCol}">${r.totalPnl >= 0 ? '+' : ''}$${r.totalPnl}</span>
        <span>${r.pf}x</span>
        <button class="bt2-opt-use">Usar</button>
      </div>`;
    }).join('')}
  `;
}

async function startAutoOptimize() {
  const btn    = document.getElementById('bt2OptBtn');
  const runBtn = document.getElementById('bt2RunBtn');
  const progEl = document.getElementById('bt2Progress');
  const progFill = document.getElementById('bt2ProgressFill');
  const progLbl  = document.getElementById('bt2ProgressLbl');
  const statsEl  = document.getElementById('bt2Stats');
  const eqEl     = document.getElementById('bt2Equity');
  const listEl   = document.getElementById('bt2TradeList');
  const optEl    = document.getElementById('bt2OptResults');

  const cfg = {
    symbol:   (document.getElementById('bt2Symbol')?.value || 'BTCUSDT').trim().toUpperCase(),
    interval: document.getElementById('bt2Tf')?.value || '15m',
    days:     parseInt(document.getElementById('bt2Days')?.value || '30'),
    margin:   parseFloat(document.getElementById('bt2Margin')?.value || '100'),
    leverage: parseInt(document.getElementById('bt2Lev')?.value || '10'),
  };

  btn.disabled = true;
  if (runBtn) runBtn.disabled = true;
  btn.textContent = '⏳ Optimizando...';
  progEl.style.display = '';
  if (statsEl) statsEl.style.display = 'none';
  if (eqEl) eqEl.style.display = 'none';
  if (listEl) listEl.style.display = 'none';
  if (optEl) optEl.style.display = 'none';

  const onProgress = (pct, msg) => {
    if (progFill) progFill.style.width = pct + '%';
    if (progLbl)  progLbl.textContent  = msg;
  };

  try {
    const results = await runAutoOptimize(cfg.symbol, cfg.interval, cfg.days, cfg.margin, cfg.leverage, onProgress);
    renderOptimizerResults(results, cfg);
    if (results.length > 0) {
      const best = results[0];
      showNotif(`🏆 Mejor: ${best.stratLabel} conf:${best.conf}% → WR:${best.wr}%`, 'info');
    }
  } catch (e) {
    if (progLbl) progLbl.textContent = '❌ Error: ' + e.message;
    showNotif('Optimizer error: ' + e.message, 'error');
    console.error('[Optimizer]', e);
  } finally {
    btn.disabled = false;
    if (runBtn) runBtn.disabled = false;
    btn.textContent = '🔍 Auto-Optimizar';
  }
}

// ═══════════════════════════════════════════════════════════════════
// STRATEGY FACTORY — 234 parametric combos
// Pre-computes ALL indicators once → fast O(N) per combo
// ═══════════════════════════════════════════════════════════════════

function calcStochFull(candles, kP = 14, dP = 3) {
  const n = candles.length;
  const k = new Array(n).fill(null);
  for (let i = kP - 1; i < n; i++) {
    const sl = candles.slice(i - kP + 1, i + 1);
    const hh = Math.max(...sl.map(c => c.high));
    const ll = Math.min(...sl.map(c => c.low));
    k[i] = hh === ll ? 50 : (candles[i].close - ll) / (hh - ll) * 100;
  }
  const d = new Array(n).fill(null);
  for (let i = kP + dP - 2; i < n; i++) {
    const sl = k.slice(i - dP + 1, i + 1).filter(v => v !== null);
    d[i] = sl.length === dP ? sl.reduce((s, v) => s + v, 0) / dP : null;
  }
  return { k, d };
}

// ── Extra indicator calc functions for expanded factory ──────────────────────
function calcWilliamsR(candles, period = 14) {
  return candles.map((_, i) => {
    if (i < period - 1) return null;
    const sl = candles.slice(i - period + 1, i + 1);
    const hh = Math.max(...sl.map(c => c.high));
    const ll = Math.min(...sl.map(c => c.low));
    return hh === ll ? -50 : ((hh - candles[i].close) / (hh - ll)) * -100;
  });
}
function calcCCI(candles, period = 20) {
  return candles.map((_, i) => {
    if (i < period - 1) return null;
    const sl = candles.slice(i - period + 1, i + 1);
    const tp = sl.map(c => (c.high + c.low + c.close) / 3);
    const sma = tp.reduce((s, v) => s + v, 0) / period;
    const md  = tp.reduce((s, v) => s + Math.abs(v - sma), 0) / period;
    return md === 0 ? 0 : (tp[period - 1] - sma) / (0.015 * md);
  });
}
function calcPSAR(candles, step = 0.02, maxAF = 0.2) {
  const n = candles.length;
  const psarArr = new Array(n).fill(null);
  const bullArr = new Array(n).fill(false);
  if (n < 3) return { psar: psarArr, bull: bullArr };
  let isBull = candles[1].close > candles[0].close;
  let sar = isBull ? Math.min(candles[0].low, candles[1].low) : Math.max(candles[0].high, candles[1].high);
  let ep = isBull ? candles[1].high : candles[1].low;
  let af = step;
  for (let i = 2; i < n; i++) {
    const c = candles[i];
    if (isBull) {
      sar = Math.min(sar + af * (ep - sar), candles[i-1].low, candles[i-2].low);
      if (c.low < sar) { isBull = false; sar = ep; ep = c.low; af = step; }
      else if (c.high > ep) { ep = c.high; af = Math.min(af + step, maxAF); }
    } else {
      sar = Math.max(sar + af * (ep - sar), candles[i-1].high, candles[i-2].high);
      if (c.high > sar) { isBull = true; sar = ep; ep = c.high; af = step; }
      else if (c.low < ep) { ep = c.low; af = Math.min(af + step, maxAF); }
    }
    psarArr[i] = sar;
    bullArr[i] = isBull;
  }
  return { psar: psarArr, bull: bullArr };
}
function calcADX(candles, period = 14) {
  const n = candles.length;
  const adxOut = new Array(n).fill(null);
  const pdiOut = new Array(n).fill(null);
  const mdiOut = new Array(n).fill(null);
  if (n < period * 2 + 2) return { adx: adxOut, pdi: pdiOut, mdi: mdiOut };
  const tr = [], pdm = [], ndm = [];
  for (let i = 1; i < n; i++) {
    const h = candles[i].high, l = candles[i].low, pc = candles[i-1].close;
    tr.push(Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc)));
    const up = h - candles[i-1].high, dn = candles[i-1].low - l;
    pdm.push(up > dn && up > 0 ? up : 0);
    ndm.push(dn > up && dn > 0 ? dn : 0);
  }
  let smTR = tr.slice(0, period).reduce((s, v) => s + v, 0);
  let smPDM = pdm.slice(0, period).reduce((s, v) => s + v, 0);
  let smNDM = ndm.slice(0, period).reduce((s, v) => s + v, 0);
  const diStart = period;
  { const p_=smTR>0?100*smPDM/smTR:0, m_=smTR>0?100*smNDM/smTR:0; pdiOut[diStart]=p_; mdiOut[diStart]=m_; }
  const dxArr = [pdiOut[diStart]+mdiOut[diStart]>0 ? 100*Math.abs(pdiOut[diStart]-mdiOut[diStart])/(pdiOut[diStart]+mdiOut[diStart]) : 0];
  for (let j = period; j < tr.length; j++) {
    smTR=smTR-smTR/period+tr[j]; smPDM=smPDM-smPDM/period+pdm[j]; smNDM=smNDM-smNDM/period+ndm[j];
    const ci=j+1, p_=smTR>0?100*smPDM/smTR:0, m_=smTR>0?100*smNDM/smTR:0;
    pdiOut[ci]=p_; mdiOut[ci]=m_;
    dxArr.push(p_+m_>0?100*Math.abs(p_-m_)/(p_+m_):0);
  }
  if (dxArr.length < period) return { adx: adxOut, pdi: pdiOut, mdi: mdiOut };
  let adxVal = dxArr.slice(0, period).reduce((s, v) => s + v, 0) / period;
  adxOut[diStart + period - 1] = adxVal;
  for (let k = period; k < dxArr.length; k++) { adxVal=(adxVal*(period-1)+dxArr[k])/period; adxOut[diStart+k]=adxVal; }
  return { adx: adxOut, pdi: pdiOut, mdi: mdiOut };
}
function calcTRIX(closes, period = 14) {
  const ema3 = calcEMA(calcEMA(calcEMA(closes, period), period), period);
  return ema3.map((v, i) => i === 0 ? null : ((v - ema3[i-1]) / ema3[i-1]) * 10000);
}
function calcHA(candles) {
  const haOpen = [], haClose = [];
  for (let i = 0; i < candles.length; i++) {
    const hc = (candles[i].open + candles[i].high + candles[i].low + candles[i].close) / 4;
    const ho = i === 0 ? (candles[0].open + candles[0].close) / 2 : (haOpen[i-1] + haClose[i-1]) / 2;
    haOpen.push(ho); haClose.push(hc);
  }
  return { open: haOpen, close: haClose };
}

// Connors RSI = avg(RSI3, StreakRSI, PercentRank100)
function calcConnorsRSI(closes) {
  const rsi3 = calcRSI(closes, 3);
  // Streak: consecutive up/down days
  const streak = closes.map((_, i) => {
    if (i === 0) return 0;
    let s = closes[i] > closes[i-1] ? 1 : closes[i] < closes[i-1] ? -1 : 0;
    let j = i - 1;
    while (j > 0 && ((s > 0 && closes[j] > closes[j-1]) || (s < 0 && closes[j] < closes[j-1]))) { s += s > 0 ? 1 : -1; j--; }
    return s;
  });
  const streakRSI = calcRSI(streak, 2);
  // PercentRank: % of last 100 1-day returns that are less than current
  const returns = closes.map((v, i) => i === 0 ? 0 : (v - closes[i-1]) / closes[i-1]);
  const prank = returns.map((r, i) => {
    if (i < 100) return null;
    const window = returns.slice(i - 100, i);
    return (window.filter(x => x < r).length / 100) * 100;
  });
  return closes.map((_, i) => {
    const a = rsi3[i], b = streakRSI[i], c = prank[i];
    return (a == null || b == null || c == null) ? null : (a + b + c) / 3;
  });
}

function btPrecompute(candles) {
  const closes = candles.map(c => c.close);
  const st = calcStochFull(candles, 14, 3);
  return {
    closes,
    vols:   candles.map(c => c.volume),
    ema5:   calcEMA(closes, 5),
    ema8:   calcEMA(closes, 8),
    ema13:  calcEMA(closes, 13),
    ema20:  calcEMA(closes, 20),
    ema21:  calcEMA(closes, 21),
    ema34:  calcEMA(closes, 34),
    ema50:  calcEMA(closes, 50),
    ema200: calcEMA(closes, Math.min(200, closes.length - 1)),
    rsi2:   calcRSI(closes, 2),
    rsi3:   calcRSI(closes, 3),
    rsi14:  calcRSI(closes, 14),
    macd:   calcMACD(closes),
    bb20:   calcBB(closes, 20, 2),
    atr14:  calcATR(candles, 14),
    stochK: st.k,
    stochD: st.d,
    wr14:   calcWilliamsR(candles, 14),
    cci20:  calcCCI(candles, 20),
    psar:   calcPSAR(candles),
    adx14:  calcADX(candles, 14),
    trix14: calcTRIX(closes, 14),
    ha:     calcHA(candles),
    crsi:   calcConnorsRSI(closes),
    sma100: calcSMA(closes, 100),
  };
}

// Raw signal using pre-computed indicators — returns 'LONG'|'SHORT'|null
function btFSig(ind, candles, i, sigKey) {
  if (i < 2) return null;
  const { closes, vols, ema5, ema8, ema13, ema20, ema21, ema34, ema50, ema200,
          rsi2, rsi3, rsi14, macd, bb20, atr14, stochK, stochD,
          wr14, cci20, psar, adx14, trix14, ha, crsi, sma100 } = ind;
  switch (sigKey) {
    case 'rsi_os30':    { const r=rsi14[i],rp=rsi14[i-1]; if(r==null||rp==null)return null; return rp<30&&r>rp?'LONG':rp>70&&r<rp?'SHORT':null; }
    case 'rsi_os35':    { const r=rsi14[i],rp=rsi14[i-1]; if(r==null||rp==null)return null; return rp<35&&r>rp?'LONG':rp>65&&r<rp?'SHORT':null; }
    case 'rsi_os40':    { const r=rsi14[i],rp=rsi14[i-1]; if(r==null||rp==null)return null; return rp<40&&r>rp&&r<50?'LONG':rp>60&&r<rp&&r>50?'SHORT':null; }
    case 'macd_flip':   { const h=macd.hist[i],hp=macd.hist[i-1]; if(h==null||hp==null)return null; return hp<0&&h>0?'LONG':hp>0&&h<0?'SHORT':null; }
    case 'macd_grow':   { const h=macd.hist[i],hp=macd.hist[i-1],hpp=macd.hist[i-2]; if(h==null||hp==null||hpp==null)return null; return h>0&&h>hp&&hp>hpp?'LONG':h<0&&h<hp&&hp<hpp?'SHORT':null; }
    case 'bb_bounce':   { const b=bb20[i],bp=bb20[i-1]; if(!b?.lower||!bp?.lower)return null; return closes[i-1]<=bp.lower&&closes[i]>b.lower?'LONG':closes[i-1]>=bp.upper&&closes[i]<b.upper?'SHORT':null; }
    case 'bb_mid':      { const b=bb20[i],bp=bb20[i-1]; if(!b?.mid||!bp?.mid)return null; return closes[i-1]<bp.mid&&closes[i]>b.mid?'LONG':closes[i-1]>bp.mid&&closes[i]<b.mid?'SHORT':null; }
    case 'ema_8_21':    { const e8=ema8[i],e8p=ema8[i-1],e2=ema20[i],e2p=ema20[i-1]; if(!e8||!e8p||!e2||!e2p)return null; return e8p<=e2p&&e8>e2?'LONG':e8p>=e2p&&e8<e2?'SHORT':null; }
    case 'ema_20_50':   { const e2=ema20[i],e2p=ema20[i-1],e5=ema50[i],e5p=ema50[i-1]; if(!e2||!e2p||!e5||!e5p)return null; return e2p<=e5p&&e2>e5?'LONG':e2p>=e5p&&e2<e5?'SHORT':null; }
    case 'price_ema20': { const e=ema20[i],ep=ema20[i-1]; if(!e||!ep)return null; return closes[i-1]<ep&&closes[i]>e?'LONG':closes[i-1]>ep&&closes[i]<e?'SHORT':null; }
    case 'stoch_cross': { const kN=stochK[i],kP2=stochK[i-1],dN=stochD[i],dP2=stochD[i-1]; if(kN==null||kP2==null||dN==null||dP2==null)return null; return kP2<dP2&&kN>dN&&kN<30?'LONG':kP2>dP2&&kN<dN&&kN>70?'SHORT':null; }
    case 'donchian_10': { if(i<12)return null; const w=candles.slice(i-10,i); const hh=Math.max(...w.map(c=>c.high)),ll=Math.min(...w.map(c=>c.low)); return candles[i-1].close<=hh&&candles[i].close>hh?'LONG':candles[i-1].close>=ll&&candles[i].close<ll?'SHORT':null; }
    case 'vol_body':    { const c=candles[i]; const body=Math.abs(c.close-c.open); const atr=atr14[i]||c.close*0.005; const avgV=vols.slice(Math.max(0,i-20),i).reduce((s,v)=>s+v,0)/20; return (body>=atr&&avgV>0&&c.volume>=avgV*1.5)?(c.close>c.open?'LONG':'SHORT'):null; }
    // ── New signals ──────────────────────────────────────────────────────────
    case 'williams_r':  { const w=wr14[i],wp=wr14[i-1]; if(w==null||wp==null)return null; return wp<-80&&w>-80?'LONG':wp>-20&&w<-20?'SHORT':null; }
    case 'cci_cross':   { const c=cci20[i],cp=cci20[i-1]; if(c==null||cp==null)return null; return cp<-100&&c>-100?'LONG':cp>100&&c<100?'SHORT':null; }
    case 'keltner':     { const e=ema20[i],ep=ema20[i-1],a=atr14[i],ap=atr14[i-1]; if(!e||!ep||!a||!ap)return null; return closes[i-1]<ep-1.5*ap&&closes[i]>e-1.5*a?'LONG':closes[i-1]>ep+1.5*ap&&closes[i]<e+1.5*a?'SHORT':null; }
    case 'psar_flip':   { if(!psar||psar.bull[i]==null||psar.bull[i-1]==null)return null; return !psar.bull[i-1]&&psar.bull[i]?'LONG':psar.bull[i-1]&&!psar.bull[i]?'SHORT':null; }
    case 'adx_di':      { const adv=adx14.adx[i],pd=adx14.pdi[i],md=adx14.mdi[i],pdp=adx14.pdi[i-1],mdp=adx14.mdi[i-1]; if(adv==null||pd==null||md==null||pdp==null||mdp==null)return null; if(adv<18)return null; return pdp<=mdp&&pd>md?'LONG':mdp<=pdp&&md>pd?'SHORT':null; }
    case 'trix_cross':  { const t=trix14[i],tp=trix14[i-1]; if(t==null||tp==null)return null; return tp<0&&t>0?'LONG':tp>0&&t<0?'SHORT':null; }
    case 'three_bar_rev':{ if(i<3)return null; const [c1,c2,c3,c4]=[candles[i-3],candles[i-2],candles[i-1],candles[i]]; return c1.close<c1.open&&c2.close<c2.open&&c3.close<c3.open&&c4.close>c4.open&&c4.close>c3.open?'LONG':c1.close>c1.open&&c2.close>c2.open&&c3.close>c3.open&&c4.close<c4.open&&c4.close<c3.open?'SHORT':null; }
    case 'rsi_50':      { const r=rsi14[i],rp=rsi14[i-1]; if(r==null||rp==null)return null; return rp<50&&r>50&&r<68?'LONG':rp>50&&r<50&&r>32?'SHORT':null; }
    case 'engulf_vol':  { if(i<1)return null; const prev=candles[i-1],curr=candles[i]; const avgV=vols.slice(Math.max(0,i-20),i).reduce((s,v)=>s+v,0)/20; if(curr.volume<avgV*1.5)return null; return prev.close<prev.open&&curr.close>curr.open&&curr.close>prev.open&&curr.open<prev.close?'LONG':prev.close>prev.open&&curr.close<curr.open&&curr.close<prev.open&&curr.open>prev.close?'SHORT':null; }
    case 'heikin_ashi': { if(!ha||i<2)return null; const bN=ha.close[i]>ha.open[i],b1=ha.close[i-1]>ha.open[i-1],b2=ha.close[i-2]>ha.open[i-2]; return !b2&&!b1&&bN?'LONG':b2&&b1&&!bN?'SHORT':null; }
    // ── High Win-Rate strategies ──────────────────────────────────────────────
    // RSI(2) extreme reversion (Connors/Alvarez ~65-72% WR documented)
    case 'rsi2_ext':    { const r=rsi2[i],rp=rsi2[i-1]; if(r==null||rp==null)return null; return rp<10&&r>rp?'LONG':rp>90&&r<rp?'SHORT':null; }
    // Connors RSI composite (<20 = buy, >80 = sell)
    case 'connors_rsi': { const c=crsi[i],cp=crsi[i-1]; if(c==null||cp==null)return null; return cp<20&&c>cp?'LONG':cp>80&&c<cp?'SHORT':null; }
    // 5-EMA Fibonacci ribbon all aligned — trend quality filter
    case 'five_ribbon': { const [a,b,c,d,e]=[ema5[i],ema8[i],ema13[i],ema21[i],ema34[i]]; if(!a||!b||!c||!d||!e)return null; return a>b&&b>c&&c>d&&d>e?'LONG':a<b&&b<c&&c<d&&d<e?'SHORT':null; }
    // Z-Score statistical reversion (>2σ from SMA20 → mean revert)
    case 'zscore_rev':  { const bb=bb20[i]; if(!bb?.mid||!bb?.upper)return null; const zscore=(closes[i]-bb.mid)/(bb.width*bb.mid/4||1); return zscore<-2&&closes[i]>closes[i-1]?'LONG':zscore>2&&closes[i]<closes[i-1]?'SHORT':null; }
    // Elder Impulse — EMA13 slope + MACD hist same direction
    case 'elder_impulse':
    case 'elder_imp':   { const e=ema13[i],ep=ema13[i-1]; if(!e||!ep)return null; const eSlope=e>ep,mSlope=macd.hist[i]>macd.hist[i-1]; const eSlopeP=ep>(ema13[i-2]||ep),mSlopeP=macd.hist[i-1]>macd.hist[i-2]; return !eSlopeP&&!mSlopeP&&eSlope&&mSlope?'LONG':eSlopeP&&mSlopeP&&!eSlope&&!mSlope?'SHORT':null; }
    // 5 consecutive candles reversal (documented ~63-68% WR)
    case 'consec_5':    { if(i<6)return null; const cs=candles.slice(i-5,i); const allDown=cs.every(c=>c.close<c.open); const allUp=cs.every(c=>c.close>c.open); const curr=candles[i]; return allDown&&curr.close>curr.open&&curr.close>cs[4].open?'LONG':allUp&&curr.close<curr.open&&curr.close<cs[4].open?'SHORT':null; }
    // RSI(3) pullback in trend above SMA100 (Connors TPS strategy)
    case 'rsi3_pull':   { const r=rsi3[i],rp=rsi3[i-1]; if(r==null||rp==null)return null; const s=sma100[i]; return s&&closes[i]>s&&rp<20&&r>rp?'LONG':s&&closes[i]<s&&rp>80&&r<rp?'SHORT':null; }
    // RSI Divergence — price new low but RSI higher low
    case 'rsi_div':     { if(i<10)return null; const p=closes[i],pp=Math.min(...closes.slice(i-10,i)); const r=rsi14[i],rr=Math.min(...rsi14.slice(i-10,i).filter(v=>v!=null)); return p<=pp&&r>rr+3?'LONG':p>=Math.max(...closes.slice(i-10,i))&&r<Math.max(...rsi14.slice(i-10,i).filter(v=>v!=null))-3?'SHORT':null; }
    // Ultra-Confluence: RSI2<5 + BB<lower + WR<-90 + Stoch<10 + above EMA200 (all 5 conditions)
    case 'ultra_conf':  { const r2=rsi2[i],bb=bb20[i],wr=wr14[i],sk=stochK[i],e2=ema200[i]; if(r2==null||!bb?.lower||wr==null||sk==null||!e2)return null; const longOk=r2<5&&closes[i]<=bb.lower&&wr<-90&&sk<10&&closes[i]>e2*0.995; const shortOk=r2>95&&closes[i]>=bb.upper&&wr>-10&&sk>90&&closes[i]<e2*1.005; return longOk?'LONG':shortOk?'SHORT':null; }
  }
  return null;
}

// Returns { long: string, short: string } — how far current value is from triggering
// "—" if no meaningful distance can be computed
function btFSigDistance(ind, candles, i, sigKey) {
  if (i < 2) return { long: '—', short: '—' };
  const { closes, ema5, ema8, ema13, ema20, ema21, ema34, ema50, ema200,
          rsi2, rsi3, rsi14, macd, bb20, atr14, stochK, stochD,
          wr14, cci20, psar, adx14, trix14, ha, crsi, sma100 } = ind;
  const n2 = (v) => v == null || isNaN(v);   // null/NaN guard
  const fp2 = (v) => v == null ? '?' : Number(v).toFixed(1);
  const fp4 = (v) => v == null ? '?' : Number(v).toFixed(4);
  const fpp = (v) => v == null ? '?' : Number(v).toFixed(2) + '%';

  switch (sigKey) {
    // ── RSI family ────────────────────────────────────────────────────────
    case 'rsi_os30': { const r=rsi14[i]; if(n2(r))return{long:'—',short:'—'}; return{long:r<=30?`✅ RSI ${fp2(r)} ≤30 — en zona`:`RSI ${fp2(r)} → necesita ≤30 (faltan ${(r-30).toFixed(1)} pts)`,short:r>=70?`✅ RSI ${fp2(r)} ≥70 — en zona`:`RSI ${fp2(r)} → necesita ≥70 (faltan ${(70-r).toFixed(1)} pts)`}; }
    case 'rsi_os35': { const r=rsi14[i]; if(n2(r))return{long:'—',short:'—'}; return{long:r<=35?`✅ RSI ${fp2(r)} ≤35 — en zona`:`RSI ${fp2(r)} → necesita ≤35 (faltan ${(r-35).toFixed(1)} pts)`,short:r>=65?`✅ RSI ${fp2(r)} ≥65 — en zona`:`RSI ${fp2(r)} → necesita ≥65 (faltan ${(65-r).toFixed(1)} pts)`}; }
    case 'rsi_os40': { const r=rsi14[i]; if(n2(r))return{long:'—',short:'—'}; return{long:r<=40?`✅ RSI ${fp2(r)} ≤40 — en zona`:`RSI ${fp2(r)} → necesita ≤40 (faltan ${(r-40).toFixed(1)} pts)`,short:r>=60?`✅ RSI ${fp2(r)} ≥60 — en zona`:`RSI ${fp2(r)} → necesita ≥60 (faltan ${(60-r).toFixed(1)} pts)`}; }
    case 'rsi_50':   { const r=rsi14[i]; if(n2(r))return{long:'—',short:'—'}; return{long:r<50?`RSI ${fp2(r)} ↑ — espera cruce de 50`:`✅ RSI ${fp2(r)} sobre 50 — listo para LONG`,short:r>50?`RSI ${fp2(r)} ↓ — espera cruce bajo 50`:`✅ RSI ${fp2(r)} bajo 50 — listo para SHORT`}; }
    case 'rsi2_ext': { const r=rsi2[i];  if(n2(r))return{long:'—',short:'—'}; return{long:r<=10?`✅ RSI2 ${fp2(r)} ≤10 extremo`:`RSI2 ${fp2(r)} → necesita ≤10 (faltan ${(r-10).toFixed(1)} pts)`,short:r>=90?`✅ RSI2 ${fp2(r)} ≥90 extremo`:`RSI2 ${fp2(r)} → necesita ≥90 (faltan ${(90-r).toFixed(1)} pts)`}; }
    case 'rsi3_pull':{ const r=rsi3[i];  if(n2(r))return{long:'—',short:'—'}; const s100=sma100[i]; const tr=s100?(closes[i]>s100?'↑trend':'↓trend'):''; return{long:r<=20?`✅ RSI3 ${fp2(r)} pullback en ${tr}`:`RSI3 ${fp2(r)} → necesita ≤20 (faltan ${(r-20).toFixed(1)} pts) ${tr}`,short:r>=80?`✅ RSI3 ${fp2(r)} pullback en ${tr}`:`RSI3 ${fp2(r)} → necesita ≥80 (faltan ${(80-r).toFixed(1)} pts) ${tr}`}; }
    case 'rsi_div':  { return{long:'Busca precio en nuevo mínimo con RSI más alto que mín anterior',short:'Busca precio en nuevo máximo con RSI más bajo que máx anterior'}; }
    // ── MACD ────────────────────────────────────────────────────────────
    case 'macd_flip':{ const h=macd.hist[i]; if(n2(h))return{long:'—',short:'—'}; return{long:h<0?`✅ hist ${fp4(h)} — espera cruce a >0`:`hist +${fp4(h)} > 0, necesita caer bajo 0`,short:h>0?`✅ hist +${fp4(h)} — espera cruce a <0`:`hist ${fp4(h)} < 0, necesita subir sobre 0`}; }
    case 'macd_grow':{ const h=macd.hist[i],hp2=macd.hist[i-1]; if(n2(h)||n2(hp2))return{long:'—',short:'—'}; return{long:h>0&&h>hp2?`✅ hist ${fp4(h)} subiendo`:`hist ${fp4(h)} (necesita positivo y creciendo, prev ${fp4(hp2)})`,short:h<0&&h<hp2?`✅ hist ${fp4(h)} cayendo`:`hist ${fp4(h)} (necesita negativo y cayendo, prev ${fp4(hp2)})`}; }
    // ── Bollinger Bands ──────────────────────────────────────────────────
    case 'bb_bounce':{ const b=bb20[i]; if(!b?.lower||!b?.upper)return{long:'—',short:'—'}; const dL=((closes[i]-b.lower)/closes[i]*100),dS=((b.upper-closes[i])/closes[i]*100); return{long:dL<=0?`✅ precio en/bajo banda inf ($${b.lower.toFixed(0)})`:`precio $${closes[i].toFixed(0)} — banda inf $${b.lower.toFixed(0)} (falta ${dL.toFixed(2)}% caída)`,short:dS<=0?`✅ precio en/sobre banda sup ($${b.upper.toFixed(0)})`:`precio $${closes[i].toFixed(0)} — banda sup $${b.upper.toFixed(0)} (falta ${dS.toFixed(2)}% subida)`}; }
    case 'bb_mid':   { const b=bb20[i]; if(!b?.mid)return{long:'—',short:'—'}; return{long:closes[i]<b.mid?`precio $${closes[i].toFixed(0)} — espera cruce sobre media $${b.mid.toFixed(0)}`:`✅ precio $${closes[i].toFixed(0)} sobre media $${b.mid.toFixed(0)}`,short:closes[i]>b.mid?`precio $${closes[i].toFixed(0)} — espera cruce bajo media $${b.mid.toFixed(0)}`:`✅ precio $${closes[i].toFixed(0)} bajo media $${b.mid.toFixed(0)}`}; }
    case 'zscore_rev':{ const b=bb20[i]; if(!b?.mid)return{long:'—',short:'—'}; const z=(closes[i]-b.mid)/((b.upper-b.lower)/4||1); return{long:z<=-2?`✅ Z-score ${z.toFixed(2)} ≤-2`:`Z-score ${z.toFixed(2)} → necesita ≤-2 (faltan ${(-2-z<0?0:-2-z).toFixed(2)})`,short:z>=2?`✅ Z-score ${z.toFixed(2)} ≥2`:`Z-score ${z.toFixed(2)} → necesita ≥2 (faltan ${(2-z).toFixed(2)})`}; }
    // ── EMA crosses ─────────────────────────────────────────────────────
    case 'ema_8_21': { const e8v=ema8[i],e21v=ema21[i]; if(n2(e8v)||n2(e21v))return{long:'—',short:'—'}; const d=((e8v-e21v)/e21v*100); return{long:e8v>e21v?`EMA8 $${e8v.toFixed(0)} sobre EMA21 $${e21v.toFixed(0)} — espera cruce bajista primero`:`EMA8 $${e8v.toFixed(0)} bajo EMA21 $${e21v.toFixed(0)} — falta ${Math.abs(d).toFixed(3)}% para cruce alcista`,short:e8v<e21v?`EMA8 $${e8v.toFixed(0)} bajo EMA21 $${e21v.toFixed(0)} — espera cruce alcista primero`:`EMA8 $${e8v.toFixed(0)} sobre EMA21 $${e21v.toFixed(0)} — falta ${Math.abs(d).toFixed(3)}% para cruce bajista`}; }
    case 'ema_20_50':{ const e20=ema20[i],e50=ema50[i]; if(n2(e20)||n2(e50))return{long:'—',short:'—'}; const d=((e20-e50)/e50*100); return{long:e20>e50?`EMA20 $${e20.toFixed(0)} sobre EMA50 — espera cruce bajista primero`:`EMA20 $${e20.toFixed(0)} bajo EMA50 $${e50.toFixed(0)} — falta ${Math.abs(d).toFixed(3)}%`,short:e20<e50?`EMA20 $${e20.toFixed(0)} bajo EMA50 — espera cruce alcista primero`:`EMA20 $${e20.toFixed(0)} sobre EMA50 $${e50.toFixed(0)} — falta ${Math.abs(d).toFixed(3)}%`}; }
    case 'price_ema20':{ const e=ema20[i]; if(n2(e))return{long:'—',short:'—'}; const d=((closes[i]-e)/e*100); return{long:closes[i]<e?`precio $${closes[i].toFixed(0)} bajo EMA20 $${e.toFixed(0)} — espera cruce alcista`:`precio $${closes[i].toFixed(0)} sobre EMA20 $${e.toFixed(0)} (+${d.toFixed(3)}%) — espera pullback`,short:closes[i]>e?`precio $${closes[i].toFixed(0)} sobre EMA20 $${e.toFixed(0)} — espera cruce bajista`:`precio $${closes[i].toFixed(0)} bajo EMA20 $${e.toFixed(0)} (${d.toFixed(3)}%) — espera rebote`}; }
    case 'five_ribbon':{ const [a,b,c,d,e]=[ema5[i],ema8[i],ema13[i],ema21[i],ema34[i]]; if(n2(a)||n2(b)||n2(c)||n2(d)||n2(e))return{long:'—',short:'—'}; const bchk=[a>b,b>c,c>d,d>e],brchk=[a<b,b<c,c<d,d<e]; const bullC=bchk.filter(Boolean).length,bearC=brchk.filter(Boolean).length; const lbls=['5-8','8-13','13-21','21-34']; const bull5=bchk.map((ok,j)=>`${lbls[j]}:${ok?'✅':'✗'}`).join(' '),bear5=brchk.map((ok,j)=>`${lbls[j]}:${ok?'✅':'✗'}`).join(' '); return{long:bullC===4?`✅ 5 EMAs alineadas alcistas`:`${bullC}/4: ${bull5}`,short:bearC===4?`✅ 5 EMAs alineadas bajistas`:`${bearC}/4: ${bear5}`}; }
    // ── Oscillators ──────────────────────────────────────────────────────
    case 'stoch_cross':{ const k=stochK[i],dv=stochD[i]; if(n2(k)||n2(dv))return{long:'—',short:'—'}; return{long:k<30?(k>dv?`✅ K${k.toFixed(0)}>D${dv.toFixed(0)} en zona <30`:`K:${k.toFixed(0)} D:${dv.toFixed(0)} en zona — espera K>D`):`K:${k.toFixed(0)} — necesita <30 (faltan ${(k-30).toFixed(0)} pts)`,short:k>70?(k<dv?`✅ K${k.toFixed(0)}<D${dv.toFixed(0)} en zona >70`:`K:${k.toFixed(0)} D:${dv.toFixed(0)} en zona — espera K<D`):`K:${k.toFixed(0)} — necesita >70 (faltan ${(70-k).toFixed(0)} pts)`}; }
    case 'williams_r': { const w=wr14[i]; if(n2(w))return{long:'—',short:'—'}; return{long:w<=-80?`✅ WR ${w.toFixed(0)} en zona sobreventa — espera cruce sobre -80 (faltan ${Math.abs(w+80).toFixed(0)} pts)`:`WR ${w.toFixed(0)} → necesita ≤-80 (faltan ${(w+80).toFixed(0)} pts)`,short:w>=-20?`✅ WR ${w.toFixed(0)} en zona sobrecompra — espera cruce bajo -20 (faltan ${Math.abs(w+20).toFixed(0)} pts)`:`WR ${w.toFixed(0)} → necesita ≥-20 (faltan ${(-20-w).toFixed(0)} pts)`}; }
    case 'cci_cross':  { const c=cci20[i]; if(n2(c))return{long:'—',short:'—'}; return{long:c<=-100?`✅ CCI ${c.toFixed(0)} en zona ≤-100 — espera cruce sobre -100 (faltan ${Math.abs(c+100).toFixed(0)} pts)`:`CCI ${c.toFixed(0)} → necesita entrar a ≤-100 (faltan ${(c+100).toFixed(0)} pts)`,short:c>=100?`✅ CCI ${c.toFixed(0)} en zona ≥100 — espera cruce bajo 100 (faltan ${(c-100).toFixed(0)} pts)`:`CCI ${c.toFixed(0)} → necesita entrar a ≥100 (faltan ${(100-c).toFixed(0)} pts)`}; }
    case 'connors_rsi':{ const c=crsi[i]; if(n2(c))return{long:'—',short:'—'}; return{long:c<=20?`✅ CRSI ${fp2(c)} ≤20`:`CRSI ${fp2(c)} → necesita ≤20 (faltan ${(c-20).toFixed(1)} pts)`,short:c>=80?`✅ CRSI ${fp2(c)} ≥80`:`CRSI ${fp2(c)} → necesita ≥80 (faltan ${(80-c).toFixed(1)} pts)`}; }
    case 'adx_di':     { const adv=adx14.adx[i],pd=adx14.pdi[i],md=adx14.mdi[i]; if(n2(adv))return{long:'—',short:'—'}; const adxTxt=adv>=18?`ADX ${adv.toFixed(0)} ✅`:`ADX ${adv.toFixed(0)} (necesita ≥18, faltan ${(18-adv).toFixed(0)} pts)`; return{long:`${adxTxt} · DI+ ${pd?.toFixed(0)} DI- ${md?.toFixed(0)} → espera DI+ cruce sobre DI-`,short:`${adxTxt} · DI+ ${pd?.toFixed(0)} DI- ${md?.toFixed(0)} → espera DI- cruce sobre DI+`}; }
    case 'ultra_conf': { const r2v=rsi2[i],b=bb20[i],wrv=wr14[i],sk=stochK[i]; if(n2(r2v)||!b)return{long:'—',short:'—'}; return{long:[`RSI2:${r2v.toFixed(1)}${r2v<=5?'✅':'→≤5'}`,`WR:${wrv?.toFixed(0)}${wrv<=-90?'✅':'→≤-90'}`,`K:${sk?.toFixed(0)}${sk<=10?'✅':'→≤10'}`].join(' · '),short:'Señal ultra solo aplica LONG'}; }
    // ── Elder Impulse ────────────────────────────────────────────────────
    case 'elder_impulse':
    case 'elder_imp': {
      if (i < 2) return { long: '—', short: '—' };
      const e13c=ema13[i], e13p=ema13[i-1], e13pp=ema13[i-2]||e13p;
      const hc=macd?.hist?.[i], hp=macd?.hist?.[i-1], hpp=macd?.hist?.[i-2]??hp;
      if (e13c == null || isNaN(e13c) || e13p == null || isNaN(e13p) || hc == null || hp == null) return { long: '—', short: '—' };
      const eSlope=(e13c-e13p), eSlopePrev=(e13p-e13pp);
      const hSlope=(hc-hp), hSlopePrev=(hp-hpp);
      const eTxt=`EMA13 ${eSlope>0?'↑':'↓'}Δ${Math.abs(eSlope).toFixed(1)}`;
      const mTxt=`MACD hist ${hSlope>0?'↑':'↓'}Δ${Math.abs(hSlope).toFixed(3)}`;
      // LONG: prev ↓↓, curr ↑↑
      const lPrev = (eSlopePrev<0?'✅':'✗')+'EMA '+(hSlopePrev<0?'✅':'✗')+'MACD';
      const lCurr = (eSlope>0?'✅':'✗')+'EMA '+(hSlope>0?'✅':'✗')+'MACD';
      // SHORT: prev ↑↑, curr ↓↓
      const sPrev = (eSlopePrev>0?'✅':'✗')+'EMA '+(hSlopePrev>0?'✅':'✗')+'MACD';
      const sCurr = (eSlope<0?'✅':'✗')+'EMA '+(hSlope<0?'✅':'✗')+'MACD';
      return {
        long:  `${eTxt} · ${mTxt} | prev↓: ${lPrev} | curr↑: ${lCurr}`,
        short: `${eTxt} · ${mTxt} | prev↑: ${sPrev} | curr↓: ${sCurr}`,
      };
    }
    // ── Consecutive candles ──────────────────────────────────────────────
    case 'consec_5': {
      if (i < 6) return { long: '—', short: '—' };
      const cs=candles.slice(i-5,i), dn=cs.filter(c=>c.close<c.open).length, up=cs.filter(c=>c.close>c.open).length;
      return{long:dn>=5?`✅ 5/5 velas bajistas — espera vela alcista de reversal`:`${dn}/5 velas bajistas (faltan ${5-dn} velas bajistas más)`,short:up>=5?`✅ 5/5 velas alcistas — espera vela bajista de reversal`:`${up}/5 velas alcistas (faltan ${5-up} velas alcistas más)`};
    }
    // ── Structural & other ───────────────────────────────────────────────
    case 'three_bar_rev':{ if(i<3)return{long:'—',short:'—'}; const [c1,c2,c3]=[candles[i-3],candles[i-2],candles[i-1]]; const bear3=c1.close<c1.open&&c2.close<c2.open&&c3.close<c3.open; const bull3=c1.close>c1.open&&c2.close>c2.open&&c3.close>c3.open; return{long:bear3?`✅ 3 velas bajistas — espera reversal alcista`:`Solo ${[c1,c2,c3].filter(c=>c.close<c.open).length}/3 bajistas acumuladas`,short:bull3?`✅ 3 velas alcistas — espera reversal bajista`:`Solo ${[c1,c2,c3].filter(c=>c.close>c.open).length}/3 alcistas acumuladas`}; }
    case 'engulf_vol':{ const avgV=ind.vols.slice(Math.max(0,i-20),i).reduce((s,v)=>s+v,0)/20; const volRatio=(candles[i].volume/avgV); return{long:`Vol ${volRatio.toFixed(1)}x promedio (necesita ≥1.5x) · espera vela envolvente alcista`,short:`Vol ${volRatio.toFixed(1)}x promedio (necesita ≥1.5x) · espera vela envolvente bajista`}; }
    case 'heikin_ashi':{ if(!ha||i<2)return{long:'—',short:'—'}; const [h0,h1,h2]=[ha.close[i]>ha.open[i],ha.close[i-1]>ha.open[i-1],ha.close[i-2]>ha.open[i-2]]; const bullC2=[h2,h1,h0].filter(Boolean).length, bearC2=[!h2,!h1,!h0].filter(Boolean).length; return{long:bullC2===3?`✅ 3/3 HA alcistas`:`${3-bullC2} HA bajistas antes del giro LONG (${bullC2}/3 alcistas)`,short:bearC2===3?`✅ 3/3 HA bajistas`:`${3-bearC2} HA alcistas antes del giro SHORT (${bearC2}/3 bajistas)`}; }
    case 'vol_body':{ const c=candles[i]; const body2=Math.abs(c.close-c.open),atrV=atr14[i]||c.close*0.005; const avgV=ind.vols.slice(Math.max(0,i-20),i).reduce((s,v)=>s+v,0)/20; return{long:`Cuerpo $${body2.toFixed(0)} vs ATR $${atrV.toFixed(0)} · Vol ${(c.volume/avgV).toFixed(1)}x (necesita ≥1.5x) · espera vela alcista`,short:`Cuerpo $${body2.toFixed(0)} vs ATR $${atrV.toFixed(0)} · Vol ${(c.volume/avgV).toFixed(1)}x (necesita ≥1.5x) · espera vela bajista`}; }
    case 'donchian_10':{ if(i<12)return{long:'—',short:'—'}; const w=candles.slice(i-10,i); const hh=Math.max(...w.map(c=>c.high)),ll=Math.min(...w.map(c=>c.low)); return{long:`precio $${closes[i].toFixed(0)} — breakout sobre máx canal $${hh.toFixed(0)} (falta $${(hh-closes[i]).toFixed(0)})`,short:`precio $${closes[i].toFixed(0)} — breakout bajo mín canal $${ll.toFixed(0)} (falta $${(closes[i]-ll).toFixed(0)})`}; }
    case 'keltner':{ const e=ema20[i],a=atr14[i]; if(n2(e)||n2(a))return{long:'—',short:'—'}; const lower=e-1.5*a,upper=e+1.5*a; return{long:closes[i]<=lower?`✅ precio $${closes[i].toFixed(0)} bajo banda Keltner $${lower.toFixed(0)}`:`precio $${closes[i].toFixed(0)} — banda inf $${lower.toFixed(0)} (falta $${(closes[i]-lower).toFixed(0)} caída)`,short:closes[i]>=upper?`✅ precio $${closes[i].toFixed(0)} sobre banda Keltner $${upper.toFixed(0)}`:`precio $${closes[i].toFixed(0)} — banda sup $${upper.toFixed(0)} (falta $${(upper-closes[i]).toFixed(0)} subida)`}; }
    case 'psar_flip': { if(!psar)return{long:'—',short:'—'}; return{long:psar.bull[i]?`✅ SAR bullish ($${psar.value?.[i]?.toFixed(0)||'?'})`:`SAR bearish ($${psar.value?.[i]?.toFixed(0)||'?'}) — espera flip alcista`,short:!psar.bull[i]?`✅ SAR bearish ($${psar.value?.[i]?.toFixed(0)||'?'})`:`SAR bullish ($${psar.value?.[i]?.toFixed(0)||'?'}) — espera flip bajista`}; }
    case 'trix_cross': { const t=trix14[i]; if(n2(t))return{long:'—',short:'—'}; return{long:t>0?`TRIX +${t.toFixed(4)} positivo — espera cruce a <0 primero`:`✅ TRIX ${t.toFixed(4)} <0 — espera cruce a >0`,short:t<0?`TRIX ${t.toFixed(4)} negativo — espera cruce a >0 primero`:`✅ TRIX +${t.toFixed(4)} >0 — espera cruce a <0`}; }
    default:
      return { long: '—', short: '—' };
  }
}

// Returns { longPct, shortPct } 0-100 — progress toward triggering each direction
// Derives pct from btFSigEntryOffset so the bar is always consistent with the entry price shown
function btFSigPct(ind, candles, i, sigKey) {
  if (i < 2) return { longPct: 0, shortPct: 0 };
  const atr  = ind.atr14[i] || candles[i].close * 0.005;
  const body = atr * 0.55;
  const price = candles[i].close;
  // Max possible offset (cold-start distance) per signal — when pct=0
  const maxOff = {
    consec_5:     5  * body,
    elder_impulse:3  * body,  elder_imp: 3 * body,
    rsi_os30:     40 / 200   * price,
    rsi_os35:     30 / 200   * price,
    rsi_os40:     20 / 200   * price,
    rsi_50:       20 / 300   * price,
    rsi2_ext:     80 / 40    * atr * 4,
    rsi3_pull:    60 / 30    * atr * 3,
    rsi_div:      2  * body,
    macd_flip:    1.5 * body,
    macd_grow:    1.5 * body,
    bb_bounce:    atr * 2,
    bb_mid:       atr * 1.5,
    zscore_rev:   atr * 2,
    ema_8_21:     atr * 2,
    ema_20_50:    atr * 3,
    price_ema20:  atr * 2,
    five_ribbon:  4  * body,
    stoch_cross:  atr * 3,
    williams_r:   atr * 3,
    cci_cross:    atr * 4,
    connors_rsi:  atr * 3,
    adx_di:       atr * 3,
    trix_cross:   body,
    ultra_conf:   atr * 5,
    three_bar_rev:2 * body,
    engulf_vol:   body,
    heikin_ashi:  2 * body,
    vol_body:     body,
    donchian_10:  atr * 2,
    keltner:      atr * 2,
    psar_flip:    body,
  }[sigKey] || atr * 2;
  const off = btFSigEntryOffset(ind, candles, i, sigKey);
  const clamp = v => Math.max(0, Math.min(99, Math.round(v)));
  // ── Crossover signals: direction-aware proximity, penalty for wrong side ──
  const xoProx = (dist, atrV, wrongSide) => {
    const p = clamp((1 - dist / atrV) * 100);
    return wrongSide ? Math.max(5, Math.round(p * 0.15)) : p;
  };
  const atrV = ind.atr14[i] || candles[i].close * 0.005;
  switch (sigKey) {
    case 'elder_impulse':
    case 'elder_imp': {
      // Elder needs BOTH conditions to flip direction (EMA13 slope + MACD hist slope)
      // For LONG: need both ↓ first (prev bar), then both ↑ (current bar)
      // Calculate smooth proximity for each sub-condition
      const e13c=ind.ema13[i], e13p=ind.ema13[i-1], e13pp=ind.ema13[i-2]||e13p;
      const hc=ind.macd.hist[i], hp=ind.macd.hist[i-1], hpp=ind.macd.hist[i-2]??hp;
      if (!e13c||!e13p||hc==null||hp==null) break;
      const emaSlope = e13c - e13p;      // >0 = rising
      const emaSlopePrev = e13p - e13pp;  // prev bar slope
      const histSlope = hc - hp;          // >0 = rising
      const histSlopePrev = hp - hpp;     // prev bar slope
      // Reference magnitudes for normalization
      const emaRef = Math.max(Math.abs(emaSlope), Math.abs(emaSlopePrev), atrV * 0.01) || 1;
      const histRef = Math.max(Math.abs(histSlope), Math.abs(histSlopePrev), atrV * 0.001) || 1;
      // LONG: prev bar both ↓, current bar both ↑ — score how close each is
      // Prev bar ↓ readiness (0-50 per condition)
      const prevEmaDownPct = emaSlopePrev < 0 ? 25 : clamp(25 * (1 - emaSlopePrev / emaRef));
      const prevHistDownPct = histSlopePrev < 0 ? 25 : clamp(25 * (1 - histSlopePrev / histRef));
      // Current bar ↑ readiness
      const currEmaUpPct = emaSlope > 0 ? 25 : clamp(25 * (1 + emaSlope / emaRef));
      const currHistUpPct = histSlope > 0 ? 25 : clamp(25 * (1 + histSlope / histRef));
      const longPct = clamp(prevEmaDownPct + prevHistDownPct + currEmaUpPct + currHistUpPct);
      // SHORT: prev bar both ↑, current bar both ↓
      const prevEmaUpPct = emaSlopePrev > 0 ? 25 : clamp(25 * (1 + emaSlopePrev / emaRef));
      const prevHistUpPct = histSlopePrev > 0 ? 25 : clamp(25 * (1 + histSlopePrev / histRef));
      const currEmaDownPct = emaSlope < 0 ? 25 : clamp(25 * (1 - emaSlope / emaRef));
      const currHistDownPct = histSlope < 0 ? 25 : clamp(25 * (1 - histSlope / histRef));
      const shortPct = clamp(prevEmaUpPct + prevHistUpPct + currEmaDownPct + currHistDownPct);
      return { longPct, shortPct };
    }
    case 'price_ema20': {
      const e=ind.ema20[i]; if(!e) break;
      const dist=Math.abs(ind.closes[i]-e);
      return { longPct: xoProx(dist,atrV,ind.closes[i]>e), shortPct: xoProx(dist,atrV,ind.closes[i]<e) };
    }
    case 'bb_mid': {
      const b=ind.bb20[i]; if(!b?.mid) break;
      const dist=Math.abs(ind.closes[i]-b.mid);
      return { longPct: xoProx(dist,atrV,ind.closes[i]>b.mid), shortPct: xoProx(dist,atrV,ind.closes[i]<b.mid) };
    }
    case 'ema_8_21': {
      const e8=ind.ema8[i],e21=ind.ema21[i]; if(!e8||!e21) break;
      const gap=Math.abs(e8-e21);
      return { longPct: xoProx(gap,atrV,e8>e21), shortPct: xoProx(gap,atrV,e8<e21) };
    }
    case 'ema_20_50': {
      const e20=ind.ema20[i],e50=ind.ema50[i]; if(!e20||!e50) break;
      const gap=Math.abs(e20-e50);
      return { longPct: xoProx(gap,atrV*1.5,e20>e50), shortPct: xoProx(gap,atrV*1.5,e20<e50) };
    }
    case 'macd_flip': {
      const h=ind.macd.hist[i]; if(h==null) break;
      const absH=Math.abs(h), maxH=Math.abs(ind.macd.hist[i-3]||atrV*0.1)||1;
      return { longPct: xoProx(absH,maxH,h>0), shortPct: xoProx(absH,maxH,h<0) };
    }
    case 'trix_cross': {
      const t=ind.trix14[i]; if(t==null) break;
      const absT=Math.abs(t), maxT=Math.abs(ind.trix14[i-3]||0.001)||0.001;
      return { longPct: xoProx(absT,maxT,t>0), shortPct: xoProx(absT,maxT,t<0) };
    }
    case 'cci_cross': {
      const c=ind.cci20[i]; if(c==null) break;
      // CCI cross: signal fires when CCI LEAVES extreme zone (>100→<100 for SHORT, <-100→>-100 for LONG)
      // If in zone: distance = how far from threshold (closer = higher pct but NOT 99)
      // If not in zone: far away
      if (c >= 100) {
        // SHORT precondition met, needs to drop below 100. Distance = c - 100
        const distS = (c - 100) / 200; // normalize: 0=at threshold, 1=very far
        return { longPct: clamp(Math.max(5, 15 - c/20)), shortPct: clamp((1 - distS) * 95 + 5) };
      } else if (c <= -100) {
        // LONG precondition met, needs to rise above -100. Distance = |c+100|
        const distL = Math.abs(c + 100) / 200;
        return { longPct: clamp((1 - distL) * 95 + 5), shortPct: clamp(Math.max(5, 15 + c/20)) };
      } else {
        // Not in extreme zone — far from both
        const distToNeg100 = Math.abs(c + 100) / 200;
        const distTo100 = Math.abs(100 - c) / 200;
        return { longPct: clamp((1 - distToNeg100) * 50), shortPct: clamp((1 - distTo100) * 50) };
      }
    }
    case 'williams_r': {
      const w=ind.wr14[i]; if(w==null) break;
      // WR cross: fires when WR LEAVES extreme (<-80→>-80 for LONG, >-20→<-20 for SHORT)
      if (w <= -80) {
        const distL = Math.abs(w + 80) / 60;
        return { longPct: clamp((1 - distL) * 95 + 5), shortPct: clamp(Math.max(5, 10)) };
      } else if (w >= -20) {
        const distS = Math.abs(w + 20) / 60;
        return { longPct: clamp(Math.max(5, 10)), shortPct: clamp((1 - distS) * 95 + 5) };
      } else {
        const distToNeg80 = Math.abs(w + 80) / 60;
        const distToNeg20 = Math.abs(w + 20) / 60;
        return { longPct: clamp((1 - distToNeg80) * 50), shortPct: clamp((1 - distToNeg20) * 50) };
      }
    }
    case 'stoch_cross': {
      const k=ind.stochK[i],d=ind.stochD[i]; if(k==null||d==null) break;
      // Stoch cross: K crosses D in extreme zone (<30 for LONG, >70 for SHORT)
      if (k < 30) {
        const gap = Math.abs(k - d);
        const ready = k > d; // already crossed
        return { longPct: ready ? 99 : clamp(90 - gap * 2), shortPct: clamp(Math.max(5, 10)) };
      } else if (k > 70) {
        const gap = Math.abs(k - d);
        const ready = k < d;
        return { longPct: clamp(Math.max(5, 10)), shortPct: ready ? 99 : clamp(90 - gap * 2) };
      } else {
        const distTo30 = Math.abs(k - 30) / 40;
        const distTo70 = Math.abs(70 - k) / 40;
        return { longPct: clamp((1 - distTo30) * 50), shortPct: clamp((1 - distTo70) * 50) };
      }
    }
  }
  // ── Remaining signals: compute dynamic pct from actual indicator values ──
  switch (sigKey) {
    // RSI family — distance from threshold is inherently dynamic
    case 'rsi_os30': { const r=ind.rsi14[i]; if(r==null) break; return { longPct: clamp(r<=30?90+(30-r):Math.max(5,(1-(r-30)/40)*80)), shortPct: clamp(r>=70?90+(r-70):Math.max(5,(1-(70-r)/40)*80)) }; }
    case 'rsi_os35': { const r=ind.rsi14[i]; if(r==null) break; return { longPct: clamp(r<=35?90+(35-r):Math.max(5,(1-(r-35)/30)*80)), shortPct: clamp(r>=65?90+(r-65):Math.max(5,(1-(65-r)/30)*80)) }; }
    case 'rsi_os40': { const r=ind.rsi14[i]; if(r==null) break; return { longPct: clamp(r<=40?90:Math.max(5,(1-(r-40)/20)*80)), shortPct: clamp(r>=60?90:Math.max(5,(1-(60-r)/20)*80)) }; }
    case 'rsi_50':   { const r=ind.rsi14[i]; if(r==null) break; return { longPct: clamp(r<50?Math.max(10,(1-(50-r)/20)*85):clamp(15-((r-50)/5))), shortPct: clamp(r>50?Math.max(10,(1-(r-50)/20)*85):clamp(15-((50-r)/5))) }; }
    case 'rsi2_ext': { const r=ind.rsi2[i]; if(r==null) break; return { longPct: clamp(r<=10?90+(10-r)*2:Math.max(5,(1-(r-10)/80)*70)), shortPct: clamp(r>=90?90+(r-90)*2:Math.max(5,(1-(90-r)/80)*70)) }; }
    case 'rsi3_pull':{ const r=ind.rsi3[i],s=ind.sma100[i]; if(r==null) break; const trend=s?(ind.closes[i]>s?25:0):12; return { longPct: clamp(r<=20?70+trend:Math.max(5,(1-(r-20)/60)*50+trend)), shortPct: clamp(r>=80?70+(25-trend):Math.max(5,(1-(80-r)/60)*50+(25-trend))) }; }
    case 'connors_rsi':{ const c=ind.crsi[i]; if(c==null) break; return { longPct: clamp(c<=20?90+(20-c):Math.max(5,(1-(c-20)/60)*70)), shortPct: clamp(c>=80?90+(c-80):Math.max(5,(1-(80-c)/60)*70)) }; }
    // MACD grow — needs 3 consecutive bars growing
    case 'macd_grow': {
      const h=ind.macd.hist[i],hp=ind.macd.hist[i-1],hpp=ind.macd.hist[i-2];
      if(h==null||hp==null||hpp==null) break;
      const bullOk = [h>0,h>hp,hp>hpp].filter(Boolean).length;
      const bearOk = [h<0,h<hp,hp<hpp].filter(Boolean).length;
      return { longPct: clamp(bullOk/3*99), shortPct: clamp(bearOk/3*99) };
    }
    // BB bounce — how close to lower/upper band
    case 'bb_bounce': {
      const b=ind.bb20[i]; if(!b?.lower||!b?.upper) break;
      const p=ind.closes[i], range=b.upper-b.lower||1;
      const dL=(p-b.lower)/range, dU=(b.upper-p)/range;
      return { longPct: clamp(dL<=0?95:(1-dL)*90), shortPct: clamp(dU<=0?95:(1-dU)*90) };
    }
    // Z-score reversion
    case 'zscore_rev': {
      const b=ind.bb20[i]; if(!b?.mid) break;
      const z=(ind.closes[i]-b.mid)/((b.upper-b.lower)/4||1);
      return { longPct: clamp(z<=-2?95:z<0?((-z)/2)*90:Math.max(5,10-z*5)), shortPct: clamp(z>=2?95:z>0?(z/2)*90:Math.max(5,10+z*5)) };
    }
    // ADX+DI crossover
    case 'adx_di': {
      const adv=ind.adx14.adx[i],pd=ind.adx14.pdi[i],md=ind.adx14.mdi[i];
      if(adv==null||pd==null||md==null) break;
      const adxReady = adv >= 18 ? 30 : (adv/18)*30;
      const diGapL = pd > md ? 0 : (md-pd);
      const diGapS = md > pd ? 0 : (pd-md);
      const maxGap = Math.max(diGapL, diGapS, 10);
      return { longPct: clamp(adxReady + (1-diGapL/maxGap)*69), shortPct: clamp(adxReady + (1-diGapS/maxGap)*69) };
    }
    // Five ribbon — count aligned EMAs
    case 'five_ribbon': {
      const [a,b2,c2,d,e]=[ind.ema5[i],ind.ema8[i],ind.ema13[i],ind.ema21[i],ind.ema34[i]];
      if(!a||!b2||!c2||!d||!e) break;
      const bull=[a>b2,b2>c2,c2>d,d>e].filter(Boolean).length;
      const bear=[a<b2,b2<c2,c2<d,d<e].filter(Boolean).length;
      // Add fractional proximity for each pair
      const gaps = [[a,b2],[b2,c2],[c2,d],[d,e]];
      const bullFrac = gaps.reduce((s,[x,y]) => s + (x>y ? 1 : 1-Math.min(1,Math.abs(y-x)/(atrV*0.3))), 0) / 4;
      const bearFrac = gaps.reduce((s,[x,y]) => s + (x<y ? 1 : 1-Math.min(1,Math.abs(x-y)/(atrV*0.3))), 0) / 4;
      return { longPct: clamp(bullFrac * 99), shortPct: clamp(bearFrac * 99) };
    }
    // Consecutive 5 candles — count progress
    case 'consec_5': {
      if(i<6) break;
      const cs=candles.slice(i-5,i);
      const dn=cs.filter(c2=>c2.close<c2.open).length, up=cs.filter(c2=>c2.close>c2.open).length;
      // Also check if current candle is reversal
      const curr=candles[i], bodyR=Math.abs(curr.close-curr.open)/(atrV||1);
      return { longPct: clamp(dn>=5?(curr.close>curr.open?95:85):dn/5*80), shortPct: clamp(up>=5?(curr.close<curr.open?95:85):up/5*80) };
    }
    // Three bar reversal — count bearish/bullish bars in last 3
    case 'three_bar_rev': {
      if(i<3) break;
      const [c1,c2,c3]=[candles[i-3],candles[i-2],candles[i-1]], curr=candles[i];
      const bear3=[c1,c2,c3].filter(c2=>c2.close<c2.open).length;
      const bull3=[c1,c2,c3].filter(c2=>c2.close>c2.open).length;
      const revL = curr.close>curr.open ? 20 : 0;
      const revS = curr.close<curr.open ? 20 : 0;
      return { longPct: clamp(bear3/3*75 + revL), shortPct: clamp(bull3/3*75 + revS) };
    }
    // Heikin Ashi — 2 consecutive + flip
    case 'heikin_ashi': {
      if(!ind.ha||i<2) break;
      const [h0,h1,h2]=[ind.ha.close[i]>ind.ha.open[i], ind.ha.close[i-1]>ind.ha.open[i-1], ind.ha.close[i-2]>ind.ha.open[i-2]];
      // LONG: need 2 bearish then 1 bullish
      const bearHA = (!h2?1:0)+(!h1?1:0);
      const bullHA = (h2?1:0)+(h1?1:0);
      return { longPct: clamp(bearHA/2*60 + (h0?35:0)), shortPct: clamp(bullHA/2*60 + (!h0?35:0)) };
    }
    // Engulfing + volume
    case 'engulf_vol': {
      if(i<1) break;
      const prev=candles[i-1], curr=candles[i];
      const avgV=ind.vols.slice(Math.max(0,i-20),i).reduce((s,v)=>s+v,0)/20||1;
      const volOk = curr.volume >= avgV*1.5 ? 40 : (curr.volume/avgV/1.5)*40;
      const bullEngulf = prev.close<prev.open && curr.close>curr.open && curr.close>prev.open && curr.open<prev.close ? 55 : (curr.close>curr.open?20:0);
      const bearEngulf = prev.close>prev.open && curr.close<curr.open && curr.close<prev.open && curr.open>prev.close ? 55 : (curr.close<curr.open?20:0);
      return { longPct: clamp(volOk + bullEngulf), shortPct: clamp(volOk + bearEngulf) };
    }
    // Volume body
    case 'vol_body': {
      const c2=candles[i], bodyV=Math.abs(c2.close-c2.open), atrRef=atrV||c2.close*0.005;
      const avgV=ind.vols.slice(Math.max(0,i-20),i).reduce((s,v)=>s+v,0)/20||1;
      const bodyPct = Math.min(1, bodyV/atrRef) * 45;
      const volPct = Math.min(1, c2.volume/(avgV*1.5)) * 45;
      const dirL = c2.close>c2.open ? 9 : 0;
      const dirS = c2.close<c2.open ? 9 : 0;
      return { longPct: clamp(bodyPct + volPct + dirL), shortPct: clamp(bodyPct + volPct + dirS) };
    }
    // Donchian breakout
    case 'donchian_10': {
      if(i<12) break;
      const w=candles.slice(i-10,i);
      const hh=Math.max(...w.map(c2=>c2.high)), ll=Math.min(...w.map(c2=>c2.low));
      const range=hh-ll||1;
      return { longPct: clamp(((ind.closes[i]-ll)/range)*95), shortPct: clamp(((hh-ind.closes[i])/range)*95) };
    }
    // Keltner channel
    case 'keltner': {
      const e=ind.ema20[i],a=atrV; if(!e) break;
      const lower=e-1.5*a, upper=e+1.5*a, range=upper-lower||1;
      const p=ind.closes[i];
      return { longPct: clamp(p<=lower?95:((lower-p+range)/range)*85), shortPct: clamp(p>=upper?95:((p-upper+range)/range)*85) };
    }
    // PSAR flip
    case 'psar_flip': {
      if(!ind.psar||ind.psar.bull[i]==null) break;
      const bull=ind.psar.bull[i], val=ind.psar.value?.[i];
      const p=ind.closes[i];
      if(val) {
        const dist=Math.abs(p-val)/atrV;
        // Closer to PSAR = more likely to flip
        return { longPct: clamp(!bull?(1-Math.min(1,dist))*90+5:Math.max(5,15-dist*10)), shortPct: clamp(bull?(1-Math.min(1,dist))*90+5:Math.max(5,15-dist*10)) };
      }
      return { longPct: bull?10:70, shortPct: bull?70:10 };
    }
    // RSI divergence — structural, hard to compute smoothly
    case 'rsi_div': {
      if(i<10) break;
      const r=ind.rsi14[i],p=ind.closes[i];
      const pLow=Math.min(...ind.closes.slice(i-10,i)), pHigh=Math.max(...ind.closes.slice(i-10,i));
      const rLow=Math.min(...ind.rsi14.slice(i-10,i).filter(v=>v!=null));
      const rHigh=Math.max(...ind.rsi14.slice(i-10,i).filter(v=>v!=null));
      // Bullish div: price near low but RSI higher than its low
      const pNearLow = 1-Math.min(1,(p-pLow)/((pHigh-pLow)||1));
      const rHigherLow = r!=null&&rLow!=null ? Math.min(1,(r-rLow)/20) : 0;
      const bullDiv = clamp((pNearLow*50 + rHigherLow*49));
      // Bearish div: price near high but RSI lower than its high
      const pNearHigh = Math.min(1,(p-pLow)/((pHigh-pLow)||1));
      const rLowerHigh = r!=null&&rHigh!=null ? Math.min(1,(rHigh-r)/20) : 0;
      const bearDiv = clamp((pNearHigh*50 + rLowerHigh*49));
      return { longPct: bullDiv, shortPct: bearDiv };
    }
    // Ultra confluence — count how many conditions met
    case 'ultra_conf': {
      const r2=ind.rsi2[i],b=ind.bb20[i],wr=ind.wr14[i],sk=ind.stochK[i],e2=ind.ema200[i];
      if(r2==null||!b) break;
      const p=ind.closes[i];
      const checks = [r2<5, b.lower&&p<=b.lower, wr!=null&&wr<-90, sk!=null&&sk<10, e2&&p>e2*0.995];
      const met = checks.filter(Boolean).length;
      return { longPct: clamp(met/5*99), shortPct: clamp(Math.max(5, 10)) }; // ultra is mainly LONG
    }
  }
  return {
    longPct:  clamp((1 - Math.abs(off.longOff)  / maxOff) * 100),
    shortPct: clamp((1 - Math.abs(off.shortOff) / maxOff) * 100),
  };
}

// Returns estimated price delta from NOW to when signal fires
// longOff < 0 means price expected to drop before LONG triggers
// shortOff > 0 means price expected to rise before SHORT triggers
function btFSigEntryOffset(ind, candles, i, sigKey) {
  const atr  = ind.atr14[i] || candles[i].close * 0.005;
  const body = atr * 0.55;
  const price = candles[i].close;
  const rsiOff = (rsi, target, scale=200) => -(Math.max(0, rsi - target) / scale) * price;
  const rsiOffS = (rsi, target, scale=200) =>  (Math.max(0, target - rsi) / scale) * price;
  switch (sigKey) {
    // ── Consecutive candles ─────────────────────────────────────────────────
    case 'consec_5': {
      if (i < 5) return { longOff: 0, shortOff: 0 };
      const cs = candles.slice(i - 5, i);
      const remL = Math.max(0, 5 - cs.filter(c => c.close < c.open).length);
      const remS = Math.max(0, 5 - cs.filter(c => c.close > c.open).length);
      return { longOff: -remL * body, shortOff: remS * body };
    }
    // ── Elder Impulse ───────────────────────────────────────────────────────
    case 'elder_impulse':
    case 'elder_imp': {
      const e13c=ind.ema13[i],e13p=ind.ema13[i-1],hc=ind.macd.hist[i],hp=ind.macd.hist[i-1];
      if (!e13c||!e13p||hc==null||hp==null) return { longOff: 0, shortOff: 0 };
      const lN=(e13c<e13p?1:0)+(hc<hp?1:0), sN=(e13c>e13p?1:0)+(hc>hp?1:0);
      return { longOff: -(lN===2?0.5:lN===1?1.5:3)*body, shortOff: (sN===2?0.5:sN===1?1.5:3)*body };
    }
    // ── RSI family ─────────────────────────────────────────────────────────
    case 'rsi_os30':  { const r=ind.rsi14[i]; return r==null?{longOff:0,shortOff:0}:{longOff:rsiOff(r,30),shortOff:rsiOffS(r,70)}; }
    case 'rsi_os35':  { const r=ind.rsi14[i]; return r==null?{longOff:0,shortOff:0}:{longOff:rsiOff(r,35),shortOff:rsiOffS(r,65)}; }
    case 'rsi_os40':  { const r=ind.rsi14[i]; return r==null?{longOff:0,shortOff:0}:{longOff:rsiOff(r,40),shortOff:rsiOffS(r,60)}; }
    case 'rsi_50':    { const r=ind.rsi14[i]; return r==null?{longOff:0,shortOff:0}:{longOff:r<50?rsiOff(r,50,300):0,shortOff:r>50?rsiOffS(r,50,300):0}; }
    case 'rsi2_ext':  { const r=ind.rsi2[i];  return r==null?{longOff:0,shortOff:0}:{longOff:-(Math.max(0,r-10)/40)*atr*4,shortOff:(Math.max(0,90-r)/40)*atr*4}; }
    case 'rsi3_pull': { const r=ind.rsi3[i];  return r==null?{longOff:0,shortOff:0}:{longOff:-(Math.max(0,r-20)/30)*atr*3,shortOff:(Math.max(0,80-r)/30)*atr*3}; }
    case 'rsi_div':   { return { longOff: -2*body, shortOff: 2*body }; } // divergences need a few candles
    // ── MACD family ─────────────────────────────────────────────────────────
    case 'macd_flip': {
      const h=ind.macd.hist[i]; if (h==null) return {longOff:0,shortOff:0};
      return { longOff: h<0 ? 0 : -1.5*body, shortOff: h>0 ? 0 : 1.5*body };
    }
    case 'macd_grow': {
      const h=ind.macd.hist[i],hp2=ind.macd.hist[i-1]; if(h==null||hp2==null) return {longOff:0,shortOff:0};
      return { longOff: h>0&&h>hp2?0:-1.5*body, shortOff: h<0&&h<hp2?0:1.5*body };
    }
    // ── Bollinger Bands ─────────────────────────────────────────────────────
    case 'bb_bounce': {
      const b=ind.bb20[i]; if(!b?.lower||!b?.upper) return {longOff:0,shortOff:0};
      return { longOff: -(price-b.lower), shortOff: b.upper-price };
    }
    case 'bb_mid': {
      const b=ind.bb20[i]; if(!b?.mid) return {longOff:0,shortOff:0};
      return { longOff: price<b.mid?0:-(price-b.mid), shortOff: price>b.mid?0:b.mid-price };
    }
    case 'zscore_rev': {
      const b=ind.bb20[i]; if(!b?.upper||!b?.lower) return {longOff:0,shortOff:0};
      return { longOff: -(price-b.lower)*0.7, shortOff: (b.upper-price)*0.7 };
    }
    // ── EMA crosses — entry always AT the threshold (crossing level) ─────────
    case 'ema_8_21': {
      const e8=ind.ema8[i],e21=ind.ema21[i]; if(!e8||!e21) return {longOff:0,shortOff:0};
      // When already on correct side, entry ≈ current price; wrong side = large penalty
      return { longOff: e8<e21?-(e21-e8)*0.3:-(e8-e21)*2.5, shortOff: e8>e21?(e8-e21)*0.3:(e21-e8)*2.5 };
    }
    case 'ema_20_50': {
      const e20=ind.ema20[i],e50=ind.ema50[i]; if(!e20||!e50) return {longOff:0,shortOff:0};
      return { longOff: e20<e50?-(e50-e20)*0.3:-(e20-e50)*2.5, shortOff: e20>e50?(e20-e50)*0.3:(e50-e20)*2.5 };
    }
    case 'price_ema20': {
      const e=ind.ema20[i]; if(!e) return {longOff:0,shortOff:0};
      // Entry = EMA20 level (the crossover point), same for both directions
      return { longOff: e-price, shortOff: e-price };
    }
    case 'five_ribbon': {
      const [a,b,c,d,e]=[ind.ema5[i],ind.ema8[i],ind.ema13[i],ind.ema21[i],ind.ema34[i]];
      if(!a||!b||!c||!d||!e) return {longOff:0,shortOff:0};
      const bull=[a>b,b>c,c>d,d>e].filter(Boolean).length;
      const bear=[a<b,b<c,c<d,d<e].filter(Boolean).length;
      return { longOff: -(4-bull)*body, shortOff: (4-bear)*body };
    }
    // ── Oscillators ─────────────────────────────────────────────────────────
    case 'stoch_cross': { const k=ind.stochK[i]; return k==null?{longOff:0,shortOff:0}:{longOff:-(Math.max(0,k-30)/100)*atr*3,shortOff:(Math.max(0,70-k)/100)*atr*3}; }
    case 'williams_r':  { const w=ind.wr14[i];   return w==null?{longOff:0,shortOff:0}:{longOff:-(Math.max(0,w-(-80))/60)*atr*3,shortOff:(Math.max(0,(-20)-w)/60)*atr*3}; }
    case 'cci_cross':   { const c=ind.cci20[i];  if(c==null) return {longOff:0,shortOff:0}; return { longOff: c<=-100 ? -(c-(-100))/200*atr*4 : -(Math.max(0,c-(-100))/200)*atr*4, shortOff: c>=100 ? (c-100)/200*atr*4 : (Math.max(0,100-c)/200)*atr*4 }; }
    case 'connors_rsi': { const c=ind.crsi[i];   return c==null?{longOff:0,shortOff:0}:{longOff:-(Math.max(0,c-20)/30)*atr*3,shortOff:(Math.max(0,80-c)/30)*atr*3}; }
    case 'adx_di': {
      const pd=ind.adx14.pdi[i],md=ind.adx14.mdi[i]; if(pd==null||md==null) return {longOff:0,shortOff:0};
      return { longOff: pd<md?-(md-pd)*0.3:0, shortOff: md<pd?(pd-md)*0.3:0 };
    }
    case 'trix_cross': {
      const t=ind.trix14[i]; if(t==null) return {longOff:0,shortOff:0};
      return { longOff: t<0?0:-1*body, shortOff: t>0?0:1*body };
    }
    case 'ultra_conf': {
      const r2=ind.rsi2[i]; if(r2==null) return {longOff:0,shortOff:0};
      return { longOff: -(Math.max(0,r2-5)/45)*atr*5, shortOff: (Math.max(0,95-r2)/45)*atr*5 };
    }
    // ── Price action / candle patterns ──────────────────────────────────────
    case 'three_bar_rev': { return { longOff: -2*body, shortOff: 2*body }; }
    case 'engulf_vol':    { return { longOff: -1*body, shortOff: 1*body }; }
    case 'heikin_ashi':   { return { longOff: -2*body, shortOff: 2*body }; }
    case 'vol_body':      { return { longOff: -1*body, shortOff: 1*body }; }
    // ── Structural breakouts ─────────────────────────────────────────────────
    case 'donchian_10': {
      if (i < 12) return { longOff: 0, shortOff: 0 };
      const w=candles.slice(i-10,i);
      const hh=Math.max(...w.map(c=>c.high)), ll=Math.min(...w.map(c=>c.low));
      return { longOff: Math.min(0, price-hh), shortOff: Math.max(0, ll-price) };
    }
    case 'keltner': {
      const e=ind.ema20[i]; if(!e) return {longOff:0,shortOff:0};
      const lower=e-1.5*atr, upper=e+1.5*atr;
      return { longOff: price>lower?-(price-lower):0, shortOff: price<upper?upper-price:0 };
    }
    case 'psar_flip': {
      const ps=ind.psar; if(!ps) return {longOff:0,shortOff:0};
      return { longOff: ps.bull[i]?0:-1*body, shortOff: ps.bull[i]?1*body:0 };
    }
    default: return { longOff: 0, shortOff: 0 };
  }
}

function btFFilter(ind, i, filtKey, dir) {
  if (i < 5) return false;
  const { closes, ema20, ema50, ema200, rsi14, macd, vols, adx14, psar } = ind;
  const isL = dir === 'LONG';
  switch (filtKey) {
    case 'none':         return true;
    case 'ema50_align':  { const e=ema50[i]; return !e||(isL?closes[i]>e:closes[i]<e); }
    case 'ema200_align': { const e=ema200[i]; return !e||(isL?closes[i]>e:closes[i]<e); }
    case 'ema20_slope':  { if(!ema20[i]||!ema20[i-5])return true; const sl=(ema20[i]-ema20[i-5])/ema20[i-5]; return isL?sl>0:sl<0; }
    case 'rsi_zone':     { const r=rsi14[i]; return r==null||(isL?(r>40&&r<68):(r<60&&r>32)); }
    case 'macd_dir':     { const h=macd.hist[i]; return h==null||(isL?h>0:h<0); }
    case 'adx_trend':    { const adv=adx14?.adx[i]; return adv==null||adv>18; }
    case 'vol_spike':    { const avgV=vols.slice(Math.max(0,i-20),i).reduce((s,v)=>s+v,0)/20; return vols[i]>=avgV*1.2; }
    case 'psar_confirm': { return !psar||psar.bull[i]==null||(isL?psar.bull[i]:!psar.bull[i]); }
  }
  return true;
}

const _BT_F_SIGS  = [
  'rsi_os30','rsi_os35','rsi_os40','macd_flip','macd_grow','bb_bounce','bb_mid',
  'ema_8_21','ema_20_50','price_ema20','stoch_cross','donchian_10','vol_body',
  'williams_r','cci_cross','keltner','psar_flip','adx_di','trix_cross',
  'three_bar_rev','rsi_50','engulf_vol','heikin_ashi',
  'rsi2_ext','connors_rsi','five_ribbon','zscore_rev','elder_imp','consec_5','rsi3_pull','rsi_div',
  'ultra_conf',
];
const _BT_F_FILTS = ['none','ema50_align','ema200_align','ema20_slope','rsi_zone','macd_dir','adx_trend','vol_spike','psar_confirm'];
const _BT_F_SLTP  = [{ sl:1.5, tp1:2.5, tp2:4 }, { sl:2, tp1:3, tp2:5 }, { sl:2.5, tp1:4, tp2:7 }];

let _BT_FAC_CACHE = null;
function getBtFactoryCombos() {
  if (_BT_FAC_CACHE) return _BT_FAC_CACHE;
  _BT_FAC_CACHE = [];
  for (const sig of _BT_F_SIGS)
    for (const filt of _BT_F_FILTS)
      for (const st of _BT_F_SLTP)
        _BT_FAC_CACHE.push({ sig, filt, ...st }); // 32×9×3 = 864
  return _BT_FAC_CACHE;
}

function runFactoryOnCandles(candles, ind, combo, margin, leverage, interval = '15m') {
  const { sig, filt, sl: slM, tp1: tp1M, tp2: tp2M } = combo;
  const maxHold = (BT_CANDLES_PER_DAY[interval] || 96) * 2;
  const px  = (exit, entry, isL) => isL ? (exit - entry) / entry * 100 : (entry - exit) / entry * 100;
  const trades = [];
  let ot = null;
  for (let i = 215; i < candles.length; i++) {
    if (ot) {
      const c = candles[i];
      const isL = ot.dir === 'LONG';
      const slHit  = isL ? c.low  <= ot.sl  : c.high >= ot.sl;
      const tp2Hit = isL ? c.high >= ot.tp2 : c.low  <= ot.tp2;
      const tp1Hit = isL ? c.high >= ot.tp1 : c.low  <= ot.tp1;
      const mxH    = (i - ot.si) > maxHold;
      let closed = false;
      if      (slHit && !tp1Hit) { trades.push({ status:'loss',    pnlPct:px(ot.sl,ot.entry,isL),    pnlUsd:margin*(px(ot.sl,ot.entry,isL)/100)*leverage });    closed=true; }
      else if (tp2Hit)           { trades.push({ status:'win_tp2', pnlPct:px(ot.tp2,ot.entry,isL),   pnlUsd:margin*(px(ot.tp2,ot.entry,isL)/100)*leverage });  closed=true; }
      else if (tp1Hit)           { trades.push({ status:'win_tp1', pnlPct:px(ot.tp1,ot.entry,isL),   pnlUsd:margin*(px(ot.tp1,ot.entry,isL)/100)*leverage });  closed=true; }
      else if (mxH)              { trades.push({ status:'timeout', pnlPct:px(c.close,ot.entry,isL),  pnlUsd:margin*(px(c.close,ot.entry,isL)/100)*leverage }); closed=true; }
      if (closed) ot = null;
      if (!closed) continue;
    }
    const dir = btFSig(ind, candles, i, sig);
    if (!dir || !btFFilter(ind, i, filt, dir)) continue;
    const entry = candles[i].close;
    const atr   = ind.atr14[i] || entry * 0.005;
    const isL   = dir === 'LONG';
    ot = { dir, si: i, entry,
      sl:  isL ? entry - atr * slM  : entry + atr * slM,
      tp1: isL ? entry + atr * tp1M : entry - atr * tp1M,
      tp2: isL ? entry + atr * tp2M : entry - atr * tp2M,
    };
  }
  return trades;
}

function btWilsonLower(wins, n) {
  if (n === 0) return 0;
  const z = 1.96, z2 = z * z, p = wins / n;
  return (p + z2 / (2 * n) - z * Math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / (1 + z2 / n);
}

async function runFactoryOptimize(symbol, interval, days, margin, leverage, minTrades, onProgress) {
  const candles = await fetchKlinesBacktest(symbol, interval, days,
    p => onProgress(Math.round(p * 0.35), `Descargando ${symbol} ${interval} ${days}d...`));
  if (candles.length < 250) throw new Error('Datos insuficientes');
  onProgress(36, 'Pre-calculando indicadores (1 vez)...');
  const ind    = btPrecompute(candles);
  const combos = getBtFactoryCombos();
  const results = [];
  for (let ci = 0; ci < combos.length; ci++) {
    if (ci % 25 === 0) {
      await new Promise(r => setTimeout(r, 0));
      onProgress(37 + Math.round(ci / combos.length * 60),
        `[${ci + 1}/${combos.length}] ${combos[ci].sig} + ${combos[ci].filt}`);
    }
    const trades = runFactoryOnCandles(candles, ind, combos[ci], margin, leverage, interval);
    if (trades.length < 5) continue;
    const wins   = trades.filter(t => t.status.startsWith('win'));
    const losses = trades.filter(t => t.status === 'loss');
    const wr     = wins.length / trades.length * 100;
    const totPnl = trades.reduce((s, t) => s + t.pnlUsd, 0);
    const gw     = wins.reduce((s, t) => s + Math.abs(t.pnlUsd), 0);
    const gl     = losses.reduce((s, t) => s + Math.abs(t.pnlUsd), 0);
    results.push({
      sig: combos[ci].sig, filt: combos[ci].filt,
      sl: combos[ci].sl, tp1: combos[ci].tp1, tp2: combos[ci].tp2,
      trades: trades.length,
      wins: wins.length,
      wr: Math.round(wr * 10) / 10,
      totalPnl: Math.round(totPnl * 100) / 100,
      pf: Math.round((gl > 0 ? gw / gl : gw > 0 ? 99 : 0) * 100) / 100,
    });
  }
  onProgress(100, `🧬 Fábrica: ${results.length}/${combos.length} combos válidos`);
  const filtered = results.filter(r => r.trades >= minTrades);
  filtered.sort((a, b) => {
    const wa = btWilsonLower(a.wins, a.trades);
    const wb = btWilsonLower(b.wins, b.trades);
    return wb !== wa ? wb - wa : b.totalPnl - a.totalPnl;
  });
  return filtered;
}

const _BT_SIG_LBL  = {
  rsi_os30:'RSI<30', rsi_os35:'RSI<35', rsi_os40:'RSI<40',
  macd_flip:'MACD↕', macd_grow:'MACD↗',
  bb_bounce:'BB±Band', bb_mid:'BB Mid',
  ema_8_21:'EMA8×21', ema_20_50:'EMA20×50', price_ema20:'P×EMA20',
  stoch_cross:'Stoch×', donchian_10:'Don10', vol_body:'Vol+Body',
  williams_r:'WR%', cci_cross:'CCI×', keltner:'Keltner',
  psar_flip:'PSAR↕', adx_di:'ADX+DI', trix_cross:'TRIX×',
  three_bar_rev:'3Bar↩', rsi_50:'RSI50×', engulf_vol:'Engulf+V', heikin_ashi:'Heikin-Ashi',
  rsi2_ext:'RSI2<10', connors_rsi:'ConnorsRSI', five_ribbon:'5EMA🎀',
  zscore_rev:'Z-Score', elder_imp:'Elder⚡', consec_5:'5Consec↩',
  rsi3_pull:'RSI3 Pull', rsi_div:'RSI Div',
  ultra_conf:'🎯 Ultra5',
};
const _BT_FILT_LBL = {
  none:'—', ema50_align:'+EMA50', ema200_align:'+EMA200', ema20_slope:'+Slope',
  rsi_zone:'+RSI✓', macd_dir:'+MACD✓', adx_trend:'+ADX↑', vol_spike:'+Vol↑', psar_confirm:'+PSAR✓',
};

// Human-readable condition required for each signal to fire
const _BT_SIG_COND = {
  rsi_os30:     'RSI(14) cruza por debajo de 30 (sobreventa extrema)',
  rsi_os35:     'RSI(14) cruza por debajo de 35 (sobreventa)',
  rsi_os40:     'RSI(14) cruza por debajo de 40 (zona baja)',
  macd_flip:    'MACD cruza su línea de señal (cruce histograma)',
  macd_grow:    'Histograma MACD creciendo en la misma dirección',
  bb_bounce:    'Precio toca o cierra fuera de la banda BB (rebote)',
  bb_mid:       'Precio cruza la media de las Bandas de Bollinger',
  ema_8_21:     'EMA8 cruza EMA21 (cruce rápido)',
  ema_20_50:    'EMA20 cruza EMA50 (cruce de tendencia)',
  price_ema20:  'Precio cruza la EMA20',
  stoch_cross:  'Stocástico K cruza D (por debajo/encima de 20/80)',
  donchian_10:  'Precio rompe el canal Donchian de 10 períodos',
  vol_body:     'Vela con volumen alto y cuerpo grande (momentum)',
  williams_r:   'Williams %R sale de zona de sobreventa (<-80) o sobrecompra (>-20)',
  cci_cross:    'CCI cruza cero desde zona extrema (±100)',
  keltner:      'Precio rompe el canal Keltner',
  psar_flip:    'Parabolic SAR cambia de lado (flip de tendencia)',
  adx_di:       'ADX > 25 y DI+ cruza DI- (o viceversa)',
  trix_cross:   'TRIX(14) cruza la línea de señal',
  three_bar_rev:'3 velas en reversa + última cierra en dirección opuesta',
  rsi_50:       'RSI(14) cruza el nivel 50',
  engulf_vol:   'Vela envolvente con volumen 1.5× promedio',
  heikin_ashi:  'Cambio de color Heikin-Ashi (patrón de reversión)',
  rsi2_ext:     'RSI(2) extremo (<5 LONG o >95 SHORT) + EMA200 alineada',
  connors_rsi:  'ConnorsRSI <10 (LONG) o >90 (SHORT)',
  five_ribbon:  '5 EMAs alineadas: EMA5 > EMA8 > EMA13 > EMA21 > EMA34',
  zscore_rev:   'Z-score de precio > 2σ en zona extrema (reversión)',
  elder_imp:    'Elder Impulse System: EMA13 y MACD alineados (barra azul/roja)',
  consec_5:     '5 velas consecutivas bajistas (SHORT) o alcistas (LONG)',
  rsi3_pull:    'RSI(3) < 20 en uptrend (pullback + EMA50 alcista)',
  rsi_div:      'Divergencia RSI: precio hace nuevo extremo pero RSI no',
  ultra_conf:   '5 de 5: EMA trend + MACD cruce + BB rebote + Stoch cruce + RSI zona',
};
const _BT_FILT_COND = {
  none:         null,
  ema50_align:  'EMA50 por encima/debajo del precio (alineación)',
  ema200_align: 'EMA200 confirma la dirección (tendencia macro)',
  ema20_slope:  'EMA20 con pendiente positiva/negativa',
  rsi_zone:     'RSI en zona válida: LONG requiere RSI 40–68 · SHORT requiere RSI 32–60',
  macd_dir:     'Histograma MACD en la misma dirección que la señal',
  adx_trend:    'ADX > 20 (tendencia con fuerza suficiente)',
  vol_spike:    'Volumen actual > 1.3× promedio de 20 velas',
  psar_confirm: 'Parabolic SAR del lado correcto (LONG: precio > SAR)',
};

function renderFactoryResults(results, cfg) {
  const divEl = document.getElementById('bt2FactoryResultsDiv');
  const tblEl = document.getElementById('bt2FactoryTable');
  const cntEl = document.getElementById('bt2FactoryCount');
  const subEl = document.getElementById('bt2FactorySubtitle');
  if (!divEl || !tblEl) return;
  divEl.style.display = '';
  if (cntEl) cntEl.textContent = `${results.length} válidas`;
  if (subEl) subEl.textContent = `${cfg.symbol} ${cfg.interval} ${cfg.days}d · ${getBtFactoryCombos().length} combos · mín.${cfg.minTrades || 30}t · orden: Wilson Score`;
  const medals = ['🥇','🥈','🥉'];
  const curTf = cfg.interval || '15m';
  tblEl.innerHTML = `
    <div class="bt2-fac-row bt2-opt-header">
      <span>#</span><span>Señal</span><span>Filtro</span><span>N</span><span>WR%</span><span>PnL$</span><span>★Score</span><span></span><span></span>
    </div>
    ${results.slice(0, 30).map((r, idx) => {
      const wc = r.wr >= 58 ? '#00ff41' : r.wr >= 48 ? '#e3b341' : '#f85149';
      const pc = r.totalPnl >= 0 ? '#00ff41' : '#f85149';
      return `<div class="bt2-fac-row" title="SL:${r.sl}×ATR | TP1:${r.tp1}× | TP2:${r.tp2}×" data-sig="${r.sig}" data-filt="${r.filt}" data-sl="${r.sl}" data-tp1="${r.tp1}" data-tp2="${r.tp2}" data-wr="${r.wr}" data-trades="${r.trades}" data-tf="${curTf}">
        <span>${medals[idx] || (idx + 1)}</span>
        <span class="bt2-fac-sig">${_BT_SIG_LBL[r.sig] || r.sig}</span>
        <span class="bt2-fac-filt">${_BT_FILT_LBL[r.filt] || r.filt}</span>
        <span>${r.trades}</span>
        <span style="color:${wc};font-weight:700">${r.wr}%</span>
        <span style="color:${pc}">${r.totalPnl >= 0 ? '+' : ''}$${r.totalPnl}</span>
        <span title="Wilson lower bound 95%CI">${Math.round(btWilsonLower(r.wins, r.trades) * 1000) / 10}%</span>
        <button class="bt2-opt-use bt2-fac-use">Usar</button>
        <button class="bt2-fac-save" title="Guardar en portafolio ${curTf}" style="background:none;border:1px solid rgba(160,100,255,.4);border-radius:3px;color:#cba6f7;cursor:pointer;font-size:9px;padding:1px 4px">📌</button>
      </div>`;
    }).join('')}
  `;
}

async function startFactoryOptimize() {
  const btn      = document.getElementById('bt2FactoryBtn');
  const progEl   = document.getElementById('bt2Progress');
  const progFill = document.getElementById('bt2ProgressFill');
  const progLbl  = document.getElementById('bt2ProgressLbl');
  const statsEl  = document.getElementById('bt2Stats');
  const eqEl     = document.getElementById('bt2Equity');
  const listEl   = document.getElementById('bt2TradeList');
  const optEl    = document.getElementById('bt2OptResults');
  const facEl    = document.getElementById('bt2FactoryResultsDiv');
  const cfg = {
    symbol:    (document.getElementById('bt2Symbol')?.value || 'BTCUSDT').trim().toUpperCase(),
    interval:  document.getElementById('bt2Tf')?.value || '15m',
    days:      parseInt(document.getElementById('bt2Days')?.value || '30'),
    margin:    parseFloat(document.getElementById('bt2Margin')?.value || '100'),
    leverage:  parseInt(document.getElementById('bt2Lev')?.value || '10'),
    minTrades: parseInt(document.getElementById('bt2FacMinTrades')?.value || '30'),
  };
  btn.disabled = true;
  btn.textContent = '⏳ 864 combos...';
  _BT_FAC_CACHE = null; // reset cache so new signals are included
  if (progEl)  progEl.style.display  = '';
  if (statsEl) statsEl.style.display = 'none';
  if (eqEl)    eqEl.style.display    = 'none';
  if (listEl)  listEl.style.display  = 'none';
  if (optEl)   optEl.style.display   = 'none';
  if (facEl)   facEl.style.display   = 'none';
  const onProg = (pct, msg) => {
    if (progFill) progFill.style.width = pct + '%';
    if (progLbl)  progLbl.textContent  = msg;
  };
  try {
    const results = await runFactoryOptimize(cfg.symbol, cfg.interval, cfg.days, cfg.margin, cfg.leverage, cfg.minTrades, onProg);
    renderFactoryResults(results, cfg);
    if (results.length > 0) {
      const b = results[0];
      showNotif(`🧬 Mejor combo: ${_BT_SIG_LBL[b.sig]||b.sig} ${_BT_FILT_LBL[b.filt]||b.filt} WR:${b.wr}% (${b.trades}t)`, 'info');
    }
  } catch (e) {
    if (progLbl) progLbl.textContent = '❌ ' + e.message;
    console.error('[Factory]', e);
  } finally {
    btn.disabled = false;
    btn.textContent = '🧬 Fábrica+';
  }
}

// ═══════════════════════════════════════════════════════════════════
// LOAD & ANALYZE
// ═══════════════════════════════════════════════════════════════════

async function loadAndAnalyze(silent = false) {
  const loadEl  = document.getElementById('aLoading');
  const contEl  = document.getElementById('aContent');
  const loadTxt = document.getElementById('aLoadingText');
  const errEl   = document.getElementById('aError');

  // Al analizar un símbolo/TF nuevo o manualmente: limpiar datos avanzados
  // para que el placeholder aparezca y runAdvancedBrain los recalcule
  if (!silent) window._lastAdvanced = null;

  // Only show the loading spinner on first load or manual symbol change.
  // Silent refreshes (auto every 30s) update in place — no flicker.
  const isFirstLoad = !lastAnalysis;
  if (!silent && isFirstLoad) {
    if (loadEl)  { loadEl.style.display = 'flex'; }
    if (contEl)  { contEl.style.display = 'none'; }
    if (loadTxt) { loadTxt.textContent  = `Descargando ${currentSymbol} ${currentTf}…`; }
    if (errEl)   { errEl.style.display = 'none'; }
  }

  // Timeout global de 15 segundos para toda la operación
  const globalTimeout = setTimeout(() => {
    console.error('[TVAnalyzer] Timeout global alcanzado');
    if (!silent || isFirstLoad) {
      if (loadEl) loadEl.style.display = 'none';
      if (contEl) contEl.style.display = 'block';
      if (errEl) {
        errEl.style.display = 'block';
        errEl.innerHTML = `
          <div style="text-align:center; padding:20px;">
            <div style="font-size:32px; margin-bottom:10px;">⏱️</div>
            <div style="font-size:14px; color:#f85149; margin-bottom:10px;">Timeout: La operación tardó demasiado</div>
            <div style="font-size:11px; color:var(--text2); margin-bottom:15px;">
              Puede ser un problema de conexión o la API está lenta.
            </div>
            <button onclick="loadAndAnalyze()" style="
              padding: 8px 16px;
              background: rgba(0,212,255,.2);
              border: 1px solid var(--accent);
              border-radius: 5px;
              color: var(--accent);
              cursor: pointer;
              font-size: 12px;
            ">🔄 Reintentar</button>
          </div>
        `;
      }
      showNotif('Timeout: La carga tardó demasiado', 'error');
    }
  }, 15000);

  try {
    console.log('[TVAnalyzer] Iniciando carga:', currentSymbol, currentTf);
    candles = await fetchKlinesUniversal(currentSymbol, currentTf, 200);
    console.log('[TVAnalyzer] Velas recibidas:', candles?.length);
    
    if (!candles?.length) throw new Error(`Sin datos para ${currentSymbol}`);
    currentPrice = candles[candles.length - 1].close;
    
    console.log('[TVAnalyzer] Ejecutando análisis...');
    const analysis = runAnalysis(candles, currentSymbol, currentTf);
    
    clearTimeout(globalTimeout); // Éxito, cancelar timeout
    
    if (loadEl) loadEl.style.display = 'none';
    if (contEl) contEl.style.display = 'block';
    if (errEl)  errEl.style.display  = 'none';
    
    renderAnalysis(analysis);
    wsApply();

    // Check portfolio strategies for current TF and show alerts
    window._lastCandles = candles;
    checkPortfolioAlerts(candles, currentTf);

    // ── Advanced Brain: ML + harmonic + MTF + Elliott (no bloquea UI) ──────────
    runAdvancedBrain(analysis, candles, currentSymbol, currentTf);

    // Fetch sessions, news & economic calendar in background (non-blocking)
    Promise.all([
      fetchSessionRanges(currentSymbol),
      fetchEcoCalendar(),
    ]).then(([sessions, calDays]) => {
      window._lastCalDays  = calDays;
      window._lastSessions = sessions;
      renderSessionsNews(sessions, calDays);
      renderSessionStatusBar(); // initial render
    }).catch(() => {});

    // Re-render sessions every minute (session status changes over time)
    clearInterval(window._sessionTimer);
    window._sessionTimer = setInterval(async () => {
      const sessions = await fetchSessionRanges(currentSymbol).catch(() => []);
      const el = document.getElementById('aSessionsNews');
      if (el && sessions.length) renderSessionsNews(sessions, window._lastCalDays || []);
    }, 60000);
  } catch (e) {
    clearTimeout(globalTimeout); // Limpiar timeout en caso de error
    console.error('[TVAnalyzer] Error en carga:', e);
    
    // Siempre mostrar error en primera carga, solo ocultar en refreshes silenciosos
    const shouldShowError = !silent || isFirstLoad;
    
    if (shouldShowError) {
      if (loadEl) loadEl.style.display = 'none';
      if (contEl) contEl.style.display = 'block';
      if (errEl) {
        errEl.style.display = 'block';
        errEl.innerHTML = `
          <div style="text-align:center; padding:20px;">
            <div style="font-size:32px; margin-bottom:10px;">❌</div>
            <div style="font-size:14px; color:#f85149; margin-bottom:10px;">${escHtml(e.message || String(e))}</div>
            <div style="font-size:11px; color:var(--text2); margin-bottom:15px;">
              ${e.message?.includes('Timeout') ? 'La API de Binance no responde. Intenta de nuevo.' : ''}
              ${e.message?.includes('Sin datos') ? 'Verifica que el símbolo sea correcto (ej: BTCUSDT)' : ''}
            </div>
            <button onclick="loadAndAnalyze()" style="
              padding: 8px 16px;
              background: rgba(0,212,255,.2);
              border: 1px solid var(--accent);
              border-radius: 5px;
              color: var(--accent);
              cursor: pointer;
              font-size: 12px;
            ">🔄 Reintentar</button>
          </div>
        `;
      }
      showNotif((e.message || String(e)), 'error');
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
// NOTIFICATION
// ═══════════════════════════════════════════════════════════════════

function showNotif(msg, type = 'success') {
  const el = document.getElementById('notification');
  if (!el) return;
  el.textContent = msg;
  el.className   = 'notification show ' + type;
  setTimeout(() => el.classList.remove('show'), 3000);
}

// ═══════════════════════════════════════════════════════════════════
// TABS
// ═══════════════════════════════════════════════════════════════════

function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  const tabEl = document.getElementById('tab-' + name);
  const btnEl = document.querySelector(`.tab[data-tab="${name}"]`);
  if (tabEl) tabEl.classList.add('active');
  if (btnEl) btnEl.classList.add('active');
  
  // Renderizar tab Bot Trades cuando se activa
  if (name === 'bot-trades') {
    renderBotTradesTab();
  }
}

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════

function mlUpdateHeaderBadge() {
  const btn    = document.getElementById('mlHdrBtn');
  const status = document.getElementById('mlHdrStatus');
  if (!btn || !status) return;
  if (_mlModel) {
    btn.classList.add('ml-active');
    status.textContent = `ML ${_mlModel.cv_auc}`;
    btn.title = `Modelo ML activo — AUC ${_mlModel.cv_auc} · ${_mlModel.n_samples?.toLocaleString()} muestras\nClick para reemplazar`;
  } else {
    btn.classList.remove('ml-active');
    status.textContent = 'Sin ML';
    btn.title = 'Importar model_weights.json (entrena con Python en tv-analyzer/train/)';
  }
}

// ── ML file import handler ────────────────────────────────────────────────────
document.addEventListener('change', async (e) => {
  if (e.target.id !== 'mlFileInput') return;
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const text  = await file.text();
    const model = JSON.parse(text);
    if (!model.coefficients || !model.scaler_mean) throw new Error('JSON invalido');
    _mlModel = model;
    chrome.storage.local.set({ mlModel: model });
    mlUpdateHeaderBadge();
    showNotif(`Modelo ML cargado - AUC ${model.cv_auc} - ${model.n_samples?.toLocaleString()} muestras`, 'success');
    // Re-renderiza el Brain con el modelo recién cargado (sin descargar datos de nuevo)
    if (lastAnalysis && candles.length) {
      runAdvancedBrain(lastAnalysis, candles, currentSymbol, currentTf);
    } else if (candles.length) {
      loadAndAnalyze(true);
    }
  } catch (err) {
    showNotif('Error al leer el JSON: ' + err.message, 'error');
  }
});

document.addEventListener('DOMContentLoaded', () => {
  // Cargar bot trades y modelo ML al iniciar
  loadBotTrades();
  
  mlLoadModel().then(loaded => {
    if (!loaded) return;
    mlUpdateHeaderBadge();
    // Si el análisis ya estaba renderizado sin el modelo, lo actualizamos
    if (lastAnalysis && candles.length) {
      runAdvancedBrain(lastAnalysis, candles, currentSymbol, currentTf);
    }
  });

  // Tab navigation
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      switchTab(btn.dataset.tab);
      if (btn.dataset.tab === 'watchlist') refreshWatchlist();
      if (btn.dataset.tab === 'patrones') renderPatternsTab();
      if (btn.dataset.tab === 'backtest') {
        // Pre-fill symbol from current analysis
        const symEl = document.getElementById('bt2Symbol');
        if (symEl && currentSymbol) symEl.value = currentSymbol;
        const tfEl = document.getElementById('bt2Tf');
        if (tfEl && currentTf && tfEl.querySelector(`option[value="${currentTf}"]`)) tfEl.value = currentTf;
      }
    });
  });

  // Backtest run button
  document.getElementById('bt2RunBtn')?.addEventListener('click', () => startBacktest());

  // Auto-optimizer button
  document.getElementById('bt2OptBtn')?.addEventListener('click', () => startAutoOptimize());

  // Strategy Factory button
  document.getElementById('bt2FactoryBtn')?.addEventListener('click', () => startFactoryOptimize());

  // "Usar esta config" button via event delegation on optimizer table
  document.getElementById('bt2OptTable')?.addEventListener('click', e => {
    const btn = e.target.closest('.bt2-opt-use');
    if (!btn) return;
    const row = btn.closest('[data-strat]');
    if (!row) return;
    const stratEl = document.getElementById('bt2Strategy');
    const confEl  = document.getElementById('bt2MinConf');
    if (stratEl) stratEl.value = row.dataset.strat;
    if (confEl)  confEl.value  = row.dataset.conf;
    showNotif(`✅ Config: ${BT_OPT_LABELS[row.dataset.strat] || row.dataset.strat} conf:${row.dataset.conf}%`, 'info');
  });

  // Factory table "Usar" + "📌 Guardar" buttons via event delegation
  document.getElementById('bt2FactoryTable')?.addEventListener('click', e => {
    // ── Usar ──
    const btnUsar = e.target.closest('.bt2-fac-use');
    if (btnUsar) {
      const row = btnUsar.closest('[data-sig]');
      if (!row) return;
      window._btFactorySelected = {
        sig:  row.dataset.sig,
        filt: row.dataset.filt,
        sl:   parseFloat(row.dataset.sl),
        tp1:  parseFloat(row.dataset.tp1),
        tp2:  parseFloat(row.dataset.tp2),
      };
      const stratEl = document.getElementById('bt2Strategy');
      if (stratEl) {
        let opt = stratEl.querySelector('option[value="factory_combo"]');
        if (!opt) { opt = document.createElement('option'); opt.value = 'factory_combo'; stratEl.appendChild(opt); }
        const sig  = window._btFactorySelected.sig;
        const filt = window._btFactorySelected.filt;
        opt.textContent = `🧬 ${_BT_SIG_LBL[sig]||sig} + ${_BT_FILT_LBL[filt]||filt} (SL:${window._btFactorySelected.sl}×)`;
        stratEl.value = 'factory_combo';
      }
      showNotif(`🧬 Combo lista: ${_BT_SIG_LBL[row.dataset.sig]||row.dataset.sig} + ${_BT_FILT_LBL[row.dataset.filt]||row.dataset.filt} — pulsa ▶ Ejecutar`, 'info');
      return;
    }
    // ── Guardar en portafolio ──
    const btnSave = e.target.closest('.bt2-fac-save');
    if (btnSave) {
      const row = btnSave.closest('[data-sig]');
      if (!row) return;
      const tf = row.dataset.tf || document.getElementById('bt2Tf')?.value || '15m';
      portfolioAdd(tf, {
        sig:    row.dataset.sig,
        filt:   row.dataset.filt,
        sl:     parseFloat(row.dataset.sl),
        tp1:    parseFloat(row.dataset.tp1),
        tp2:    parseFloat(row.dataset.tp2),
        wr:     parseFloat(row.dataset.wr),
        trades: parseInt(row.dataset.trades),
        addedAt: Date.now(),
      });
      renderPortfolioPanel();
      showNotif(`📌 Guardado en portafolio ${tf}: ${_BT_SIG_LBL[row.dataset.sig]||row.dataset.sig} ${_BT_FILT_LBL[row.dataset.filt]||row.dataset.filt}`, 'info');
    }
  });

  // Init portfolio panel on load
  renderPortfolioPanel();

  // Expose portfolio functions to window (needed for onclick in module scripts)
  window.portfolioRemove       = portfolioRemove;
  window.renderPortfolioPanel  = renderPortfolioPanel;
  window.checkPortfolioAlerts  = checkPortfolioAlerts;
  window.showStrategyModal     = showStrategyModal;

  // Event delegation for portfolio ✕ buttons in analysis tab
  document.getElementById('aPortfolioAlerts')?.addEventListener('click', e => {
    if (e.target.closest('.port-refresh-btn')) {
      checkPortfolioAlerts(window._lastCandles || [], currentTf);
      return;
    }
    const btn = e.target.closest('.port-remove-btn');
    if (!btn) return;
    const tf  = btn.dataset.tf;
    const idx = parseInt(btn.dataset.idx);
    portfolioRemove(tf, idx);
    renderPortfolioPanel();
    checkPortfolioAlerts(window._lastCandles || [], tf);
  });

  // Event delegation for portfolio ✕ / ⟳ in backtest portfolio panel
  document.getElementById('btPortfolioPanel')?.addEventListener('click', e => {
    const btn = e.target.closest('.port-remove-btn');
    if (!btn) return;
    const tf  = btn.dataset.tf;
    const idx = parseInt(btn.dataset.idx);
    portfolioRemove(tf, idx);
    renderPortfolioPanel();
    if (window._lastCandles?.length) checkPortfolioAlerts(window._lastCandles, tf);
  });

  // ── Exportar portafolio de estrategias como JSON ──────────────────────────
  document.getElementById('btPortExportBtn')?.addEventListener('click', () => {
    const data = portfolioLoad();
    const totalStrats = Object.values(data).reduce((s, arr) => s + arr.length, 0);
    if (!totalStrats) { showNotif('No hay estrategias guardadas para exportar', 'error'); return; }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `traderAnalyzer_portfolio_${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showNotif(`📤 Portafolio exportado: ${totalStrats} estrategia(s)`, 'success');
  });

  // ── Importar portafolio de estrategias desde JSON ─────────────────────────
  document.getElementById('btPortImportBtn')?.addEventListener('click', () => {
    document.getElementById('btPortImportFile').click();
  });
  document.getElementById('btPortImportFile')?.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const imported = JSON.parse(ev.target.result);
        // Validate: must be object with tf keys containing arrays
        const validTfs = ['1m','3m','5m','15m','30m','1h','4h','1d'];
        const keys = Object.keys(imported);
        if (!keys.length) throw new Error('Archivo vacío');
        const invalid = keys.find(k => !validTfs.includes(k) || !Array.isArray(imported[k]));
        if (invalid) throw new Error(`Clave inválida: ${invalid}`);
        // Merge with existing (no duplicates)
        const existing = portfolioLoad();
        let added = 0;
        for (const tf of keys) {
          if (!existing[tf]) existing[tf] = [];
          for (const entry of imported[tf]) {
            const dup = existing[tf].some(e => e.sig === entry.sig && e.filt === entry.filt && e.sl === entry.sl);
            if (!dup) { existing[tf].push(entry); added++; }
          }
        }
        portfolioSave(existing);
        renderPortfolioPanel();
        showNotif(`📥 Importadas ${added} estrategia(s) nueva(s)`, 'success');
      } catch (err) {
        showNotif(`❌ Error al importar: ${err.message}`, 'error');
      }
      e.target.value = ''; // Reset file input
    };
    reader.readAsText(file);
  });

  // Event delegation for portfolio ✕ / ⟳ inside Trading Brain (aBrainPortfolio)
  // Also handles ℹ️ strategy detail buttons
  document.getElementById('aContent')?.addEventListener('click', e => {
    const refreshBtn = e.target.closest('#aBrainPortfolio .port-refresh-btn');
    if (refreshBtn) {
      checkPortfolioAlerts(window._lastCandles || [], currentTf);
      return;
    }
    const removeBtn = e.target.closest('#aBrainPortfolio .port-remove-btn');
    if (removeBtn) {
      const tf  = removeBtn.dataset.tf;
      const idx = parseInt(removeBtn.dataset.idx);
      portfolioRemove(tf, idx);
      renderPortfolioPanel();
      checkPortfolioAlerts(window._lastCandles || [], tf);
      return;
    }
  });

  // GLOBAL event delegation for ℹ️ strategy info buttons — CAPTURE PHASE (bypasses stopPropagation)
  document.body.addEventListener('click', e => {
    const infoBtn = e.target.closest('.strat-info-btn');
    if (!infoBtn) return;
    e.stopPropagation();
    e.preventDefault();
    console.log('[StratModal] CLICK detected on strat-info-btn:', infoBtn.dataset);
    showStrategyModal(
      infoBtn.dataset.sig,
      infoBtn.dataset.filt,
      parseFloat(infoBtn.dataset.wr),
      parseInt(infoBtn.dataset.trades)
    );
  }, true); // <<< CAPTURE PHASE: catches click before any child handler can stop it

  // MutationObserver: attach direct listeners to strat-info-btn whenever they appear in DOM
  new MutationObserver(() => {
    document.querySelectorAll('.strat-info-btn:not([data-wired])').forEach(btn => {
      btn.setAttribute('data-wired', '1');
      btn.addEventListener('click', e => {
        e.stopPropagation();
        e.preventDefault();
        console.log('[StratModal] DIRECT click on strat-info-btn:', btn.dataset);
        showStrategyModal(
          btn.dataset.sig,
          btn.dataset.filt,
          parseFloat(btn.dataset.wr),
          parseInt(btn.dataset.trades)
        );
      });
    });
  }).observe(document.body, { childList: true, subtree: true });

  document.getElementById('pmOverlay')?.addEventListener('click', closePatternModal);
  document.getElementById('pmClose')?.addEventListener('click', closePatternModal);

  // Symbol input
  const symInput = document.getElementById('symInput');
  const tfSelect = document.getElementById('tfSelect');

  symInput?.addEventListener('keydown', e => {
    if (e.key === 'Enter') { loadAndAnalyze(); if (autoRefresh) startAutoRefresh(); }
  });
  document.getElementById('analyzeBtn')?.addEventListener('click', () => {
    loadAndAnalyze();
    if (autoRefresh) startAutoRefresh(); // reset countdown
  });

  tfSelect?.addEventListener('change', () => {
    currentTf = tfSelect.value;
    loadAndAnalyze();
    if (autoRefresh) startAutoRefresh();
  });

  // Auto-refresh — starts automatically, button pauses/resumes
  function startAutoRefresh() {
    if (autoRefresh) clearInterval(autoRefresh);
    autoRefresh = setInterval(() => loadAndAnalyze(true), 60000); // reload candle history silently every 60s
    const btn = document.getElementById('autoRefreshBtn');
    if (btn) { btn.textContent = '⟳ 30s'; btn.classList.add('active'); btn.style.color = '#00ff41'; }
  }
  function stopAutoRefresh() {
    clearInterval(autoRefresh);
    autoRefresh = null;
    const btn = document.getElementById('autoRefreshBtn');
    if (btn) { btn.textContent = '⟳ Pausado'; btn.classList.remove('active'); btn.style.color = ''; }
  }

  document.getElementById('autoRefreshBtn')?.addEventListener('click', () => {
    if (autoRefresh) stopAutoRefresh(); else startAutoRefresh();
  });

  // ── Sync status UI ───────────────────────────────────────────────────────
  function setSyncStatus(state, text) {
    // state: 'ok' | 'warn' | 'err' | 'idle'
    const dot = document.getElementById('syncDot');
    const lbl = document.getElementById('syncLabel');
    if (dot) { dot.className = 'sync-dot ' + (state === 'idle' ? '' : state); }
    if (lbl) lbl.textContent = text;
  }

  // ── Core sync function — reads current symbol from the TV tab ─────────────
  function syncFromTV(andAnalyze = true) {
    const btn = document.getElementById('syncBtn');
    if (btn) btn.classList.add('syncing');
    setSyncStatus('warn', 'Buscando pestaña TradingView…');

    chrome.tabs.query({ url: '*://*.tradingview.com/*' }, tabs => {
      // Prefer the currently active tab; fallback to any TV tab
      chrome.tabs.query({ active: true, currentWindow: true }, activeTabs => {
        const activeTab = activeTabs[0];
        const tvTab = (activeTab?.url?.includes('tradingview.com') ? activeTab : null)
                   || tabs.find(t => t.url?.includes('tradingview.com/chart'));

        if (btn) btn.classList.remove('syncing');

        if (!tvTab) {
          setSyncStatus('err', 'No hay pestaña de TradingView abierta');
          showNotif('Abre TradingView en el navegador primero', 'error');
          return;
        }

        tvTabId = tvTab.id;

        chrome.tabs.sendMessage(tvTab.id, { type: 'GET_TV_DATA' }, resp => {
          void chrome.runtime.lastError;

          if (!resp?.symbol) {
            // Reintentar inyectando el content script manualmente
            chrome.scripting.executeScript(
              { target: { tabId: tvTab.id }, files: ['content.js'] },
              () => {
                void chrome.runtime.lastError;
                setTimeout(() => {
                  chrome.tabs.sendMessage(tvTab.id, { type: 'GET_TV_DATA' }, resp2 => {
                    void chrome.runtime.lastError;
                    if (!resp2?.symbol) {
                      setSyncStatus('err', `TV detectada · sin datos · tab ${tvTab.id}`);
                      showNotif('Recarga la pestaña de TradingView (F5) e inténtalo de nuevo', 'error');
                    }
                  });
                }, 800);
              }
            );
            return;
          }

          const sym = resp.symbol.replace(/[^A-Z0-9.^-]/g, '').toUpperCase() || currentSymbol;
          const tf  = resp.timeframe || currentTf;
          const changed = sym !== currentSymbol;

          currentSymbol = sym;
          currentTf     = tf in TF_MAP ? tf : currentTf;
          if (symInput) symInput.value = sym;
          if (tfSelect) tfSelect.value = currentTf;

          setSyncStatus('ok', `Sincronizado · ${sym} · ${currentTf} · tab ${tvTab.id}`);

          if (andAnalyze && changed) {
            loadAndAnalyze();
            if (autoRefresh) startAutoRefresh();
          } else if (andAnalyze && !changed) {
            loadAndAnalyze(true);   // silent refresh with same symbol
          }
        });
      });
    });
  }

  // ── Listen for auto symbol changes from content script ───────────────────
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'TV_SYMBOL_CHANGE' && msg.symbol) {
      const sym = msg.symbol.replace(/[^A-Z0-9.^-]/g, '').toUpperCase() || currentSymbol;
      setSyncStatus('ok', `Auto-sync · ${sym}`);
      if (sym !== currentSymbol) {
        currentSymbol = sym;
        if (symInput) symInput.value = sym;
        loadAndAnalyze();
        if (autoRefresh) startAutoRefresh();
      }
    }
  });

  // ── Sync button ───────────────────────────────────────────────────────────
  document.getElementById('syncBtn')?.addEventListener('click', () => syncFromTV(true));

  function firstLoad() {
    loadAndAnalyze();
    startAutoRefresh();
  }

  // ── Initial load: try to sync with open TV tab ────────────────────────────
  const fallbackTimer = setTimeout(() => {
    if (!candles.length) { setSyncStatus('warn', 'Sin pestaña TV — modo manual'); firstLoad(); }
  }, 4000);

  chrome.tabs.query({ url: '*://*.tradingview.com/*' }, tabs => {
    chrome.tabs.query({ active: true, currentWindow: true }, activeTabs => {
      const activeTab = activeTabs[0];
      const tvTab = (activeTab?.url?.includes('tradingview.com') ? activeTab : null)
                 || tabs.find(t => t.url?.includes('tradingview.com/chart'));

      if (!tvTab) {
        setSyncStatus('warn', 'Sin pestaña TV — escribe símbolo manual');
        clearTimeout(fallbackTimer);
        firstLoad();
        return;
      }

      tvTabId = tvTab.id;
      chrome.tabs.sendMessage(tvTab.id, { type: 'GET_TV_DATA' }, resp => {
        void chrome.runtime.lastError;
        clearTimeout(fallbackTimer);

        if (resp?.symbol) {
          currentSymbol = resp.symbol.replace(/[^A-Z0-9.^-]/g, '').toUpperCase() || 'BTCUSDT';
          currentTf     = resp.timeframe || '15m';
          if (symInput) symInput.value = currentSymbol;
          if (tfSelect) tfSelect.value = currentTf in TF_MAP ? currentTf : '15m';
          setSyncStatus('ok', `Sincronizado · ${currentSymbol} · tab ${tvTab.id}`);
        } else {
          setSyncStatus('warn', `TV detectada · sin datos · tab ${tvTab.id}`);
        }
        firstLoad();
      });
    });

  });

  // Sessions & Calendar toggle
  document.getElementById('snToggleBtn')?.addEventListener('click', () => {
    const d = document.getElementById('snDropdown');
    if (d) {
      const open = d.style.display === 'none';
      d.style.display = open ? 'block' : 'none';
      document.getElementById('snToggleBtn').classList.toggle('sn-open', open);
    }
  });

  // Draw levels buttons — top button + individual →TV buttons (event delegation)
  document.getElementById('drawLevelsBtn')?.addEventListener('click', drawLevels);
  document.getElementById('tab-analysis')?.addEventListener('click', e => {
    const btn = e.target.closest('.btn-tv-line, .btn-tv-all');
    if (!btn) return;
    if (btn.classList.contains('btn-tv-all') || btn.id === 'drawLevelsBtn2') {
      drawLevels(); return;
    }
    const price = parseFloat(btn.dataset.price);
    const label = btn.dataset.label || 'Nivel';
    const color = btn.dataset.color || '#00d4ff';
    if (price > 0) sendToTV([{ price, label, color }]);
  });

  // ── Demo buttons (Analysis tab strip) ────────────────────────────────────
  const getTradeParams = (amtId, tpId, slId, levId) => ({
    usd: parseFloat(document.getElementById(amtId)?.value) || 100,
    tp:  parseFloat(document.getElementById(tpId)?.value)  || 30,
    sl:  parseFloat(document.getElementById(slId)?.value)  || 15,
    lev: parseInt(document.getElementById(levId || 'dTpLev')?.value) || 10,
  });

  // (Analysis tab trade strip removed — trading is now in Demo tab only)

  // ── Demo tab panel buttons ────────────────────────────────────────────────
  document.getElementById('btnDemoLong')?.addEventListener('click', () => {
    const p = getTradeParams('dTpAmount','dTpTp','dTpSl','dTpLev');
    if (!currentPrice) { showNotif('Analiza una moneda primero', 'error'); return; }
    ptBuyDemo(currentSymbol, currentPrice, p.tp, p.sl, p.usd, 'long', p.lev);
  });
  document.getElementById('btnDemoShort')?.addEventListener('click', () => {
    const p = getTradeParams('dTpAmount','dTpTp','dTpSl','dTpLev');
    if (!currentPrice) { showNotif('Analiza una moneda primero', 'error'); return; }
    ptBuyDemo(currentSymbol, currentPrice, p.tp, p.sl, p.usd, 'short', p.lev);
  });

  // Update trade panel info when inputs change
  ['dTpAmount','dTpTp','dTpSl','dTpLev'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener(el.tagName === 'SELECT' ? 'change' : 'input', renderDemo);
  });
  
  // ── Bot Trades tab filtros ───────────────────────────────────────────────
  document.getElementById('btFilterSymbol')?.addEventListener('change', renderBotTradesTab);
  document.getElementById('btFilterStatus')?.addEventListener('change', renderBotTradesTab);
  
  // Investment controls
  const btMarginInput = document.getElementById('btMarginInput');
  const btLeverageInput = document.getElementById('btLeverageInput');
  const btNocionalDisplay = document.getElementById('btNocionalDisplay');
  const btFeeDisplay = document.getElementById('btFeeDisplay');
  
  function updateBotInvestmentDisplays() {
    const nocional = btMargin * btLeverage;
    const fee = nocional * 0.0004;
    if (btNocionalDisplay) btNocionalDisplay.textContent = '$' + nocional.toFixed(0);
    if (btFeeDisplay) btFeeDisplay.textContent = '$' + fee.toFixed(2);
  }
  
  if (btMarginInput) {
    btMarginInput.value = btMargin;
    btMarginInput.addEventListener('input', () => {
      btMargin = parseFloat(btMarginInput.value) || 100;
      saveBotSettings();
      updateBotInvestmentDisplays();
      renderBotTradesTab(); // Re-render to recalculate all USD values
    });
  }
  
  if (btLeverageInput) {
    btLeverageInput.value = btLeverage;
    btLeverageInput.addEventListener('input', () => {
      btLeverage = parseFloat(btLeverageInput.value) || 10;
      saveBotSettings();
      updateBotInvestmentDisplays();
      renderBotTradesTab(); // Re-render to recalculate all USD values
    });
  }
  
  updateBotInvestmentDisplays();
  
  // Execution mode selector
  const btModeSelect = document.getElementById('btExecutionMode');
  if (btModeSelect) {
    // Load saved mode (default to LIMIT for auto-execution)
    chrome.storage.local.get(['btExecutionMode'], (res) => {
      const savedMode = res.btExecutionMode || 'limit';
      btModeSelect.value = savedMode;
      
      // Force initial check if in limit mode
      if (savedMode === 'limit' && currentSymbol && currentPrice > 0) {
        trackBotTrades(currentSymbol, currentPrice);
      }
    });
    // Save on change and re-render
    btModeSelect.addEventListener('change', () => {
      const newMode = btModeSelect.value;
      chrome.storage.local.set({ btExecutionMode: newMode });
      const mode = newMode === 'limit' ? 'Límite (auto-ejecuta al precio)' : 'Señal (espera confirmación)';
      showNotif(`✅ Modo cambiado: ${mode}`, 'info');
      renderBotTradesTab(); // Re-render to update badges
      
      // If switching to limit mode, immediately check all alerts
      if (newMode === 'limit' && currentSymbol && currentPrice > 0) {
        console.log('[MODO LÍMITE] Verificando alertas existentes...');
        trackBotTrades(currentSymbol, currentPrice);
      }
    });
  }
  
  document.getElementById('btClearBtn')?.addEventListener('click', () => {
    if (confirm('¿Eliminar TODO el historial de Bot Trades?')) {
      botTrades = [];
      saveBotTrades();
      renderBotTradesTab();
      showNotif('Historial de Bot Trades eliminado', 'success');
    }
  });
  
  // Event delegation para botones de eliminar individual
  document.getElementById('btTradesList')?.addEventListener('click', (e) => {
    const deleteBtn = e.target.closest('.bt-delete-btn');
    if (deleteBtn) {
      const tradeId = parseInt(deleteBtn.dataset.tradeId);
      if (confirm('¿Eliminar este trade?')) {
        deleteBotTrade(tradeId);
      }
      return;
    }

    // ── Cerrar trade al precio actual ──────────────────────────────────────
    const closeBtn = e.target.closest('.bt-close-btn');
    if (closeBtn) {
      const tradeId = parseInt(closeBtn.dataset.tradeId);
      const trade = botTrades.find(t => t.id === tradeId);
      if (!trade) return;
      const liveP = livePrices[trade.symbol] || currentPrice;
      if (!liveP) { showNotif('No hay precio en vivo disponible', 'error'); return; }
      const isL = trade.direction === 'LONG';
      const pnl = isL
        ? ((liveP - trade.entry) / trade.entry * 100)
        : ((trade.entry - liveP) / trade.entry * 100);
      const margin   = btMargin;
      const leverage = btLeverage;
      const notional = margin * leverage;
      const pnlUsd = margin * (pnl / 100) * leverage - (notional * 0.0004);
      const newStatus = pnl >= 0 ? 'win_tp1' : 'loss';
      updateBotTrade(tradeId, {
        status: newStatus,
        hitPrice: liveP,
        hitTime: Date.now(),
        pnlPct: pnl,
        reason: trade.reason + ` | 🖐 CERRADO MANUAL @ $${fmtPrice(liveP)}`,
      });
      saveBotTrades();
      renderBotTradesTab();
      showNotif(`🖐 Trade cerrado: ${pnl >= 0 ? '+' : ''}${(pnl * btLeverage).toFixed(2)}% ROI (${pnlUsd >= 0 ? '+' : ''}$${pnlUsd.toFixed(2)})`, pnl >= 0 ? 'success' : 'error');
      return;
    }

    // ── Mover SL a breakeven (entrada) ─────────────────────────────────────
    const beBtn = e.target.closest('.bt-be-btn');
    if (beBtn) {
      const tradeId = parseInt(beBtn.dataset.tradeId);
      const trade = botTrades.find(t => t.id === tradeId);
      if (!trade) return;
      if (trade.sl === trade.entry) {
        showNotif('SL ya está en breakeven', 'info');
        return;
      }
      const notional = btMargin * btLeverage;
      const feeEstimate = notional * 0.0004;
      updateBotTrade(tradeId, {
        sl:    trade.entry,
        slPct: 0,
        slUsd: -feeEstimate, // solo fees, riesgo = 0
        reason: trade.reason + ` | ⚖ SL → BE @ $${fmtPrice(trade.entry)}`,
      });
      saveBotTrades();
      renderBotTradesTab();
      showNotif(`⚖ Breakeven activado: SL movido a entrada $${fmtPrice(trade.entry)}`, 'success');
      return;
    }
  });

  // 1s — re-render Demo timers + analysis + session status bar
  setInterval(() => {
    // Session status bar — always update (uses SESSIONS constant, no API needed)
    renderSessionStatusBar();

    // Backup tick: verificar SL/TP de trades activos (por si WebSocket perdió un wick)
    if (currentPrice > 0) {
      trackBotTrades(currentSymbol, currentPrice);
    }

    // Update positions display
    if (ptPositions.some(p => p.status === 'open')) renderDemo();

    // Re-run analysis using current WebSocket price on last candle
    if (candles.length && currentPrice > 0) {
      candles[candles.length - 1].close = currentPrice;
      try {
        const a = runAnalysis(candles, currentSymbol, currentTf);
        renderAnalysis(a);
      } catch (_) {}
    }
  }, 1000);

  // Demo reset
  document.getElementById('dResetBtn')?.addEventListener('click', () => {
    const chosen = parseFloat(document.getElementById('dStartBal')?.value) || 10000;
    if (!confirm(`¿Nueva demo con $${chosen}?`)) return;
    ptStart = chosen; ptBalance = chosen; ptPositions = []; ptPeakEq = chosen;
    ptSave(); renderDemo();
    showNotif(`Demo reseteada a $${chosen}`);
  });

  document.getElementById('dExportBtn')?.addEventListener('click', () => {
    const data = { start: ptStart, balance: ptBalance, positions: ptPositions, exportedAt: new Date().toISOString() };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = `tv-demo-${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // Watchlist add
  document.getElementById('wlAddBtn')?.addEventListener('click', () => {
    const val = (document.getElementById('wlInput')?.value || '').trim().toUpperCase();
    if (!val) return;
    const sym = val.includes('USDT') ? val : val + 'USDT';
    if (!watchlist.includes(sym)) {
      watchlist.push(sym);
      chrome.storage.local.set({ tvWatchlist: watchlist });
      refreshWatchlist();
    }
  });

  // ── STARK-OS: Neural Lab init ──────────────────────────────────
  try { initNeuralLab(); } catch(e) { console.warn('[STARK-OS] Neural Lab init error:', e); }
  try { initWhaleTracker(); } catch(e) { console.warn('[STARK-OS] Whale Tracker init error:', e); }

  // 30s REST fallback — reconnects WS if it silently dropped
  setInterval(() => {
    if (!liveWS || liveWS.readyState === WebSocket.CLOSED || liveWS.readyState === WebSocket.CLOSING)
      wsApply();
  }, 30000);

  setInterval(() => { if (document.getElementById('tab-watchlist')?.classList.contains('active')) refreshWatchlist(); }, 30000);

  // Load persisted data
  chrome.storage.local.get(['tvWatchlist'], r => {
    if (r.tvWatchlist?.length) watchlist = r.tvWatchlist;
  });
  ptLoad(() => { renderDemo(); ptUpdateHeaderBal(); wsApply(); });
});
