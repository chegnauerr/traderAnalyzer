// ═══════════════════════════════════════════════════════════════════
// STARK-OS · DOMAIN LAYER — Port Interfaces (Contracts)
// Documentación de los contratos que deben implementar los adaptadores
// de infraestructura. En JS puro usamos JSDoc; en TS serían interfaces.
// Capa: domain/ports.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

/**
 * @interface IMarketDataPort
 * Contrato para obtener datos de mercado (OHLCV + ticker).
 * Los adaptadores de infraestructura (ej: BinanceAdapter) deben
 * implementar todos estos métodos.
 *
 * @method fetchKlines(symbol: string, interval: string, limit: number): Promise<Candle[]>
 * @method fetchTicker(symbol: string): Promise<object|null>
 * @method fetchMultiTicker(symbols: string[]): Promise<Record<string, object>>
 * @method fetchKlinesUniversal(symbol: string, interval: string, limit: number): Promise<Candle[]>
 *   Intenta Binance primero, luego Yahoo Finance como fallback.
 */
export const IMarketDataPort = Object.freeze({
  METHODS: ['fetchKlines', 'fetchTicker', 'fetchMultiTicker', 'fetchKlinesUniversal'],
});

/**
 * @interface IWhalePort
 * Contrato para detectar transacciones de ballenas en Bitcoin.
 *
 * @method getRecentWhales(symbol: string, minBtc: number, maxBtc: number, limit: number): Promise<WhaleEvent[]>
 *   Retorna transacciones grandes de BTC en el rango [minBtc, maxBtc] de los últimos trades.
 *
 * @method getBtcPrice(): Promise<number>
 *   Precio actual de BTC en USD necesario para convertir amounts a BTC.
 */
export const IWhalePort = Object.freeze({
  METHODS: ['getRecentWhales', 'getBtcPrice'],
});

/**
 * @interface IIndicatorsPort
 * Contrato para el cálculo de indicadores técnicos.
 *
 * @method rsi(closes: number[], period: number): number[]
 * @method ema(closes: number[], period: number): number[]
 * @method sma(closes: number[], period: number): (number|null)[]
 * @method macd(closes: number[], fast?: number, slow?: number, signal?: number): { macd: number[], signal: number[], hist: number[] }
 * @method bb(closes: number[], period?: number, mult?: number): { mid: number|null, upper: number|null, lower: number|null, width: number|null }[]
 * @method atr(candles: Candle[], period?: number): (number|null)[]
 * @method stoch(candles: Candle[], kPeriod?: number, dPeriod?: number): { k: (number|null)[], d: (number|null)[] }
 * @method findSwings(candles: Candle[], lookback?: number): { highs: object[], lows: object[] }
 * @method supportResistance(candles: Candle[], n?: number): object[]
 * @method detectRSIDivergence(candles: Candle[], rsi: number[]): object
 */
export const IIndicatorsPort = Object.freeze({
  METHODS: ['rsi', 'ema', 'sma', 'macd', 'bb', 'atr', 'stoch', 'findSwings', 'supportResistance', 'detectRSIDivergence'],
});

/**
 * @interface IStoragePort
 * Contrato para la persistencia de datos (chrome.storage.local).
 *
 * @method get(keys: string[]): Promise<Record<string, any>>
 * @method set(data: Record<string, any>): Promise<void>
 * @method remove(keys: string[]): Promise<void>
 */
export const IStoragePort = Object.freeze({
  METHODS: ['get', 'set', 'remove'],
});

/**
 * Valida que un adaptador implementa todos los métodos de un port.
 * @param {object} adapter - El adaptador a validar.
 * @param {object} port - El port (IMarketDataPort, IWhalePort, etc.)
 * @returns {boolean}
 */
export function validateAdapter(adapter, port) {
  if (!adapter || typeof adapter !== 'object') return false;
  return port.METHODS.every(method => typeof adapter[method] === 'function');
}
