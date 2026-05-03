# -*- coding: utf-8 -*-
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Calcula features tecnicos identicos a los del JS de la extension.
El orden de columnas DEBE coincidir exactamente con FEATURE_NAMES en train_model.py.

Label:
  1  = precio sube > umbral en las siguientes LABEL_PERIODS velas  (señal LONG)
  0  = precio baja > umbral en las siguientes LABEL_PERIODS velas  (señal SHORT)
  NaN = movimiento plano → excluido del entrenamiento
"""
import numpy as np
import pandas as pd
import glob
import os

LABEL_PERIODS = 4       # velas hacia adelante para el label
ATR_MULT      = 0.50    # label si retorno > 0.5 × ATR/precio


# ── Indicadores (misma lógica que el JS) ─────────────────────────────────────

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = ag / al.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)

def macd(close, fast=12, slow=26, sig=9):
    m    = ema(close, fast) - ema(close, slow)
    s    = ema(m, sig)
    return m - s                     # histogram

def bb(close, period=20, mult=2):
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower) / mid.replace(0, 1e-10)
    pos   = (close - lower) / (upper - lower + 1e-10)
    return upper, lower, width, pos

def atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def stoch(high, low, close, k=14):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    return 100 * (close - ll) / (hh - ll + 1e-10)


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df['close'];  h = df['high'];  l = df['low']
    o = df['open'];   v = df['volume']

    rsi_s  = rsi(c)
    hist_s = macd(c)
    e20    = ema(c, 20);   e50 = ema(c, 50);  e200 = ema(c, 200)
    bb_u, bb_l, bb_w, bb_p = bb(c)
    atr_s  = atr(h, l, c)
    stk    = stoch(h, l, c)
    vol_a  = v.rolling(20).mean()
    vol_r  = v / vol_a.replace(0, 1e-10)

    body       = (c - o).abs()
    rng        = (h - l).replace(0, 1e-10)
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l

    hour = df['open_time'].dt.hour

    feat = pd.DataFrame({
        # RSI
        'rsi_norm':         (rsi_s - 50) / 50,
        'rsi_slope':        rsi_s.diff(3) / 3 / 50,
        'rsi_ob':           (rsi_s > 70).astype(float),
        'rsi_os':           (rsi_s < 30).astype(float),
        'rsi_bull':         (rsi_s > 50).astype(float),
        # MACD histogram
        'macd_hist_norm':   hist_s / atr_s.replace(0, 1e-10),
        'macd_bull':        (hist_s > 0).astype(float),
        'macd_cross_up':    ((hist_s > 0) & (hist_s.shift() <= 0)).astype(float),
        'macd_cross_dn':    ((hist_s < 0) & (hist_s.shift() >= 0)).astype(float),
        # EMAs
        'price_ema20':      c / e20 - 1,
        'price_ema50':      c / e50 - 1,
        'price_ema200':     c / e200 - 1,
        'ema20_slope':      e20.pct_change(5),
        'ema50_slope':      e50.pct_change(5),
        'ema_bull':         ((e20 > e50) & (e50 > e200)).astype(float),
        'ema_bear':         ((e20 < e50) & (e50 < e200)).astype(float),
        'above_ema200':     (c > e200).astype(float),
        # Bollinger
        'bb_pos':           bb_p.clip(0, 1),
        'bb_width':         bb_w,
        'bb_squeeze':       (bb_w < 0.025).astype(float),
        # ATR
        'atr_norm':         atr_s / c,
        # Stochastic
        'stoch_norm':       stk / 100,
        'stoch_os':         (stk < 20).astype(float),
        'stoch_ob':         (stk > 80).astype(float),
        # Volume
        'vol_ratio':        np.log1p(vol_r).clip(0, 3),
        'vol_spike':        (vol_r > 2).astype(float),
        # Candlestick
        'bull_candle':      (c > o).astype(float),
        'body_ratio':       (body / rng).clip(0, 1),
        'upper_wick':       (upper_wick / rng).clip(0, 1),
        'lower_wick':       (lower_wick / rng).clip(0, 1),
        # Tiempo (encoding cíclico)
        'hour_sin':         np.sin(2 * np.pi * hour / 24),
        'hour_cos':         np.cos(2 * np.pi * hour / 24),
    }, index=df.index)

    # ── Label ─────────────────────────────────────────────────────────────────
    future_ret = c.shift(-LABEL_PERIODS) / c - 1
    threshold  = (atr_s / c).rolling(20).mean() * ATR_MULT
    feat['label'] = np.where(future_ret >  threshold, 1.0,
                    np.where(future_ret < -threshold, 0.0, np.nan))

    return feat.dropna(subset=feat.columns.difference(['label']))


def main():
    os.makedirs('data', exist_ok=True)
    for csv in glob.glob('data/*USDT_*.csv'):
        if 'features' in csv:
            continue
        try:
            df  = pd.read_csv(csv, parse_dates=['open_time'])
            feat = build_features(df)
            out  = csv.replace('.csv', '_features.csv')
            feat.to_csv(out, index=False)
            total   = len(feat)
            labeled = feat['label'].notna().sum()
            bull    = (feat['label'] == 1).sum()
            bear    = (feat['label'] == 0).sum()
            print(f'{csv}: {total} filas → {labeled} etiquetadas  '
                  f'(LONG {bull} | SHORT {bear})')
        except Exception as e:
            print(f'ERROR {csv}: {e}')


if __name__ == '__main__':
    main()
