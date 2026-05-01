// ═══════════════════════════════════════════════════════════════════
// STARK-OS · INFRASTRUCTURE LAYER — Binance Adapter
// Adaptador compatible con CCXT (ccxt/ccxt) que envuelve las llamadas
// a la API pública de Binance + Yahoo Finance como fallback.
// Implementa IMarketDataPort.
// Capa: infrastructure/binance.adapter.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

// ── Constantes de API ──────────────────────────────────────────────────────────

const BINANCE_REST = 'https://api.binance.com/api/v3';
const YAHOO_REST   = 'https://query1.finance.yahoo.com/v8/finance/chart';

/** Mapeo de timeframes a intervalos de Binance */
const TF_MAP = {
  '1m':'1m', '3m':'3m', '5m':'5m', '15m':'15m', '30m':'30m',
  '1h':'1h', '2h':'2h', '4h':'4h', '1d':'1d',   '1w':'1w',
};

/** Mapeo de timeframes a intervalos de Yahoo Finance */
const YAHOO_IV_MAP = {
  '1m':'1m', '5m':'5m', '15m':'15m', '30m':'30m',
  '1h':'60m', '2h':'60m', '4h':'1h', '1d':'1d', '1w':'1wk',
};

// ── Binance: klines ────────────────────────────────────────────────────────────

async function fetchKlines(symbol, interval = '15m', limit = 200) {
  const iv = TF_MAP[interval] || '15m';
  const r  = await fetch(`${BINANCE_REST}/klines?symbol=${symbol}&interval=${iv}&limit=${limit}`);
  if (!r.ok) throw new Error(`Binance ${r.status} — ${symbol}`);
  const raw = await r.json();
  return raw.map(k => ({
    time:   k[0],
    open:   parseFloat(k[1]),
    high:   parseFloat(k[2]),
    low:    parseFloat(k[3]),
    close:  parseFloat(k[4]),
    volume: parseFloat(k[5]),
  }));
}

// ── Binance: ticker 24h ────────────────────────────────────────────────────────

async function fetchTicker(symbol) {
  const r = await fetch(`${BINANCE_REST}/ticker/24hr?symbol=${symbol}`);
  if (!r.ok) return null;
  return r.json();
}

// ── Binance: múltiples tickers en paralelo ─────────────────────────────────────

async function fetchMultiTicker(symbols) {
  const results = await Promise.allSettled(symbols.map(s => fetchTicker(s)));
  const out = {};
  symbols.forEach((s, i) => {
    if (results[i].status === 'fulfilled' && results[i].value)
      out[s] = results[i].value;
  });
  return out;
}

// ── Binance: aggTrades (últimos N trades agregados) ───────────────────────────

async function fetchAggTrades(symbol, limit = 500) {
  const r = await fetch(`${BINANCE_REST}/aggTrades?symbol=${symbol}&limit=${limit}`);
  if (!r.ok) throw new Error(`Binance aggTrades ${r.status}`);
  return r.json();
  // Cada trade: { a (aggId), p (price), q (qty), f (firstTradeId), l (lastId), T (time), m (isBuyerMaker) }
}

// ── Yahoo Finance: klines fallback ─────────────────────────────────────────────

async function fetchKlinesYahoo(symbol, interval = '15m', limit = 200) {
  const iv    = YAHOO_IV_MAP[interval] || '15m';
  const range = limit <= 80 ? '5d' : limit <= 200 ? '1mo' : '3mo';
  const r = await fetch(`${YAHOO_REST}/${encodeURIComponent(symbol)}?interval=${iv}&range=${range}&includePrePost=false`);
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
  if (!candles.length) throw new Error(`Sin velas Yahoo para ${symbol}`);
  return candles;
}

// ── Universal: Binance → Yahoo fallback ───────────────────────────────────────

async function fetchKlinesUniversal(symbol, interval, limit) {
  try {
    return await fetchKlines(symbol, interval, limit);
  } catch (_) {
    return fetchKlinesYahoo(symbol, interval, limit);
  }
}

// ── Precio actual de BTC en USD ───────────────────────────────────────────────

async function getBtcPrice() {
  try {
    const t = await fetchTicker('BTCUSDT');
    return t ? parseFloat(t.lastPrice) : 0;
  } catch (_) { return 0; }
}

// ─────────────────────────────────────────────────────────────────────────────
// SINGLETON EXPORT — BinanceAdapter (implementa IMarketDataPort)
// ─────────────────────────────────────────────────────────────────────────────

export const BinanceAdapter = Object.freeze({
  fetchKlines,
  fetchTicker,
  fetchMultiTicker,
  fetchKlinesUniversal,
  fetchKlinesYahoo,
  fetchAggTrades,
  getBtcPrice,
  TF_MAP,
});

export default BinanceAdapter;
