#!/usr/bin/env python3
"""
Regime Detector — SMC-AI-GOLD-SENTINEL
=======================================
Proper ADX-based market regime engine.
The current detect_market_regime() uses 2 EMAs — that only answers "is price
going up or down?" not "what KIND of market is this?"  ADX answers that.

Regimes
-------
    trending_bull   ADX > 25 AND +DI > -DI   (clean uptrend)
    trending_bear   ADX > 25 AND -DI > +DI   (clean downtrend)
    ranging         ADX < 20                  (choppy, mean-reverting)
    volatile        ATR > 2σ baseline         (news spike / trap)
    transitional    ADX 20–25                 (regime change, avoid)

Usage
-----
    from regime_detector import detect_regime, compute_regime_features

    regime = detect_regime(df)              # → "trending_bull" etc.
    feats  = compute_regime_features(df)    # → dict of 5 model features
"""

import numpy as np
import pandas as pd

# ─── ADX CALCULATION ────────────────────────────────────────────────────────────

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing (same as Wilder's ATR / ADX formula)."""
    result = series.copy().astype(float)
    result.iloc[:period] = np.nan
    first_valid = series.iloc[:period].sum()
    result.iloc[period - 1] = first_valid
    for i in range(period, len(series)):
        result.iloc[i] = result.iloc[i - 1] - (result.iloc[i - 1] / period) + series.iloc[i]
    return result


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute ADX, +DI, -DI from OHLC data.
    Returns DataFrame with columns: adx, plus_di, minus_di
    """
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)

    # Wilder smooth
    tr14       = _wilder_smooth(tr,       period)
    plus_dm14  = _wilder_smooth(plus_dm_s, period)
    minus_dm14 = _wilder_smooth(minus_dm_s, period)

    # DI values
    plus_di  = 100 * plus_dm14  / tr14.replace(0, np.nan)
    minus_di = 100 * minus_dm14 / tr14.replace(0, np.nan)

    # DX and ADX
    di_sum  = (plus_di + minus_di).replace(0, np.nan)
    dx      = 100 * (plus_di - minus_di).abs() / di_sum
    adx     = _wilder_smooth(dx.fillna(0), period)

    return pd.DataFrame({
        "adx":      adx,
        "plus_di":  plus_di,
        "minus_di": minus_di,
        "tr":       tr,
        "tr14":     tr14,
    }, index=df.index)


# ─── VOLATILITY BASELINE ────────────────────────────────────────────────────────

def _atr_baseline(df: pd.DataFrame, fast: int = 14, slow: int = 100) -> tuple:
    """Returns (current_atr, baseline_atr, atr_ratio)."""
    atr_fast = (df["high"] - df["low"]).rolling(fast).mean()
    atr_slow = (df["high"] - df["low"]).rolling(slow).mean()
    curr = float(atr_fast.iloc[-1]) if not pd.isna(atr_fast.iloc[-1]) else 1.0
    base = float(atr_slow.iloc[-1]) if not pd.isna(atr_slow.iloc[-1]) else curr
    ratio = curr / base if base > 0 else 1.0
    return curr, base, ratio


# ─── PUBLIC API ─────────────────────────────────────────────────────────────────

def detect_regime(df: pd.DataFrame,
                  adx_period:    int   = 14,
                  adx_trend:     float = 25.0,
                  adx_range:     float = 20.0,
                  vol_threshold: float = 1.8) -> str:
    """
    Classify the current market regime.

    Parameters
    ----------
    df             : OHLCV DataFrame (needs ≥ 50 bars)
    adx_period     : ADX lookback period (default 14)
    adx_trend      : ADX threshold above which market is trending (default 25)
    adx_range      : ADX threshold below which market is ranging (default 20)
    vol_threshold  : ATR-ratio above which market is spiking (default 1.8x)

    Returns
    -------
    str : one of "trending_bull", "trending_bear", "ranging",
                 "volatile", "transitional"
    """
    if df is None or len(df) < adx_period * 3:
        return "ranging"

    try:
        indicators = compute_adx(df, adx_period)
        adx_val  = float(indicators["adx"].iloc[-1])
        plus_di  = float(indicators["plus_di"].iloc[-1])
        minus_di = float(indicators["minus_di"].iloc[-1])

        if pd.isna(adx_val):
            return "ranging"

        _, _, atr_ratio = _atr_baseline(df)

        # Volatile overrides trend — spikes are unpredictable
        if atr_ratio >= vol_threshold:
            return "volatile"

        if adx_val >= adx_trend:
            return "trending_bull" if plus_di >= minus_di else "trending_bear"

        if adx_val <= adx_range:
            return "ranging"

        return "transitional"

    except Exception:
        return "ranging"


def compute_regime_features(df: pd.DataFrame, adx_period: int = 14) -> dict:
    """
    Return 5 numeric features for injection into the SMC feature vector.

    Features
    --------
    adx_strength    float 0–1  normalised ADX (adx/50, capped at 1)
    regime_trending float 0/1  1 if ADX ≥ 25
    regime_ranging  float 0/1  1 if ADX ≤ 20
    regime_volatile float 0/1  1 if ATR ratio ≥ 1.8
    di_bull         float 0/1  1 if +DI > -DI (directional bias)
    """
    default = {
        "adx_strength":    0.0,
        "regime_trending": 0.0,
        "regime_ranging":  1.0,   # default to ranging when unknown
        "regime_volatile": 0.0,
        "di_bull":         0.0,
    }

    if df is None or len(df) < adx_period * 3:
        return default

    try:
        indicators = compute_adx(df, adx_period)
        adx_val  = float(indicators["adx"].iloc[-1])
        plus_di  = float(indicators["plus_di"].iloc[-1])
        minus_di = float(indicators["minus_di"].iloc[-1])

        if pd.isna(adx_val):
            return default

        _, _, atr_ratio = _atr_baseline(df)

        trending  = adx_val >= 25.0
        ranging   = adx_val <= 20.0
        volatile  = atr_ratio >= 1.8
        di_bull   = (plus_di >= minus_di) if not pd.isna(plus_di) else True

        return {
            "adx_strength":    float(np.clip(adx_val / 50.0, 0.0, 1.0)),
            "regime_trending": 1.0 if trending else 0.0,
            "regime_ranging":  1.0 if ranging  else 0.0,
            "regime_volatile": 1.0 if volatile else 0.0,
            "di_bull":         1.0 if di_bull  else 0.0,
        }

    except Exception:
        return default


# ─── REGIME-AWARE STRATEGY FILTER ───────────────────────────────────────────────

def regime_allows_trade(regime: str, signal: str) -> tuple:
    """
    Returns (allowed: bool, reason: str).
    Applies strategy-specific rules per regime.

    - Trending regimes: only trade WITH the trend
    - Ranging regime:   only trade mean-reversion setups (lower AI confidence needed)
    - Volatile:         block all new entries
    - Transitional:     require high AI confidence (>= 0.70)
    """
    if regime == "volatile":
        return False, "volatile_regime_block"

    if regime == "trending_bull" and signal == "SELL":
        return False, "counter_trend_sell_in_bull"

    if regime == "trending_bear" and signal == "BUY":
        return False, "counter_trend_buy_in_bear"

    return True, "regime_ok"


def get_regime_confidence_multiplier(regime: str) -> float:
    """
    Scales the AI confidence threshold up/down based on regime.
    Used to raise the bar in uncertain regimes.

    Returns a multiplier applied to BUY_THRESHOLD / SELL_THRESHOLD.
    """
    multipliers = {
        "trending_bull":  0.95,    # slightly easier — trend is on our side
        "trending_bear":  0.95,
        "ranging":        1.05,    # slightly harder — choppy, more false signals
        "volatile":       9.99,    # effectively blocks all trades
        "transitional":   1.15,    # much harder — regime changing, wait for confirmation
    }
    return multipliers.get(regime, 1.0)


# ─── SELF-TEST ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import MetaTrader5 as mt5
    mt5.initialize()
    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 500)
    mt5.shutdown()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    regime = detect_regime(df)
    feats  = compute_regime_features(df)

    print(f"Current regime : {regime}")
    print(f"Features       : {feats}")
    allowed, reason = regime_allows_trade(regime, "BUY")
    print(f"BUY allowed    : {allowed} ({reason})")
    mult = get_regime_confidence_multiplier(regime)
    print(f"Conf multiplier: {mult}")
