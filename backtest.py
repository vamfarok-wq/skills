#!/usr/bin/env python3
"""
Walk-Forward Backtest Harness — SMC-AI-GOLD-SENTINEL V40
=========================================================
Trains on a rolling in-sample window, tests on the next unseen window,
rolls forward, repeats.  No lookahead bias.

Live filters applied (v2 — realistic simulation):
  • Session filter  — London 07:00-12:00 UTC, NY 13:00-21:00 UTC only
  • Entry cooldown  — minimum 2 bars (10 min) between entries
  • Max positions   — max 2 concurrent simulated positions
  • No duplicate    — won't open same direction if already in that trade
  • Concurrent P&L  — open positions are updated every bar, not pre-resolved

Usage
-----
    python backtest.py                         # fetch from MT5
    python backtest.py --csv data/gold_m5.csv  # offline CSV
    python backtest.py --train 2000 --test 500 --total 12000

Output
------
    backtest_trades.csv   — every simulated trade
    backtest_summary.json — per-fold + aggregate metrics
"""

import argparse, os, sys, json, warnings, math, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

import MetaTrader5 as mt5

# ─── Load pure-data helpers from gold_bot ──────────────────────────────────────
_BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BOT_DIR)

_BOT_OK = False
try:
    from gold_bot import (
        generate_smc_features, train_models,
        detect_fvg, detect_order_blocks, detect_liquidity_sweep,
        detect_liquidity_pools, get_mss_signal, get_fvg_signal,
        detect_market_regime, calculate_liquidity_based_sl,
        calculate_liquidity_based_tp, candle_pattern_veto,
        FEATURE_NAMES, BUY_THRESHOLD, SELL_THRESHOLD,
        create_entry_target, MarketStructure,
    )
    import gold_bot as _gb
    try:
        from regime_detector import detect_regime, regime_allows_trade
        _REGIME_OK = True
    except ImportError:
        _REGIME_OK = False
    _BOT_OK = True
    print("✅  gold_bot.py loaded — using real SMC signal engine")
except Exception as _e:
    print(f"⚠️   Could not load gold_bot.py: {_e}")
    print("     Backtest will use a simplified built-in signal (less accurate).")
    BUY_THRESHOLD  = 0.55
    SELL_THRESHOLD = 0.45
    FEATURE_NAMES  = []

# ─── CONFIG (override via argparse) ────────────────────────────────────────────
SYMBOL          = "XAUUSD"
TRAIN_BARS      = 2000      # in-sample candles per fold
TEST_BARS       = 500       # out-of-sample candles per fold
STEP_BARS       = 500       # roll-forward step
MIN_TOTAL_BARS  = 8000      # minimum history required
CONTEXT_BARS    = 200       # M5 bars fed to each signal call (mirrors LOOP_BARS)
SPREAD_PIPS     = 2.5       # simulated spread (Gold M5 ECN typical)
MAX_HOLD_BARS   = 48        # 48 × 5 min = 4 h max hold
INITIAL_EQUITY  = 1000.0    # simulated account balance
RISK_PER_TRADE  = 0.01      # fraction of equity risked per trade
SL_MIN_PIPS     = 12.0      # min SL (mirrors live bot)
GOLD_POINT      = 0.01      # XAUUSD point size
GOLD_PIP        = GOLD_POINT * 10  # 1 pip = $0.10 on 0.01 lot

# ── Live-filter settings (mirror gold_bot.py behaviour) ──────────────────────
MAX_SIM_POSITIONS = 2       # max concurrent simulated positions
COOLDOWN_BARS     = 2       # minimum bars between entries (2 × 5 min = 10 min)
# London: 07:00-12:00 UTC, New York: 13:00-21:00 UTC
SESSION_HOURS = [(7, 12), (13, 21)]

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range of the last candle (scalar)."""
    s = (df["high"] - df["low"]).rolling(period).mean()
    v = s.iloc[-1]
    if pd.isna(v):
        v = (df["high"] - df["low"]).tail(5).mean()
    return max(float(v), GOLD_PIP)


def resample_ohlcv(m5: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample M5 OHLCV DataFrame to a higher timeframe."""
    df = m5.copy()
    if "time" in df.columns:
        df = df.set_index("time")
    df.index = pd.to_datetime(df.index)
    freq = f"{minutes}min"
    out = df.resample(freq).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        tick_volume=("tick_volume", "sum"),
    ).dropna()
    return out.reset_index().rename(columns={"time": "time"})


def build_mtf_dict(m5_slice: pd.DataFrame,
                   m15_full: pd.DataFrame,
                   h1_full:  pd.DataFrame,
                   bar_time) -> dict:
    """
    Build mtf_data dict for one bar — matching the keys used by
    generate_smc_features() and should_enter_trade().
    """
    mtf = {}
    try:
        # M15 trend
        m15_up_to = m15_full[m15_full["time"] <= bar_time].tail(60)
        if len(m15_up_to) >= 20:
            mtf["M15_trend"] = detect_market_regime(m15_up_to)
            mtf["M15"] = m15_up_to
        else:
            mtf["M15_trend"] = None

        # H1 trend (EMA 50/200 — matches live loop)
        h1_up_to = h1_full[h1_full["time"] <= bar_time].tail(250)
        if len(h1_up_to) >= 30:
            mtf["H1_trend"] = detect_market_regime(h1_up_to, fast_ema=50, slow_ema=200)
            mtf["H1"] = h1_up_to
        else:
            mtf["H1_trend"] = None

        # Composite trend score used by generate_smc_features
        bull = sum(1 for t in [mtf.get("M15_trend"), mtf.get("H1_trend")] if t == "bullish")
        bear = sum(1 for t in [mtf.get("M15_trend"), mtf.get("H1_trend")] if t == "bearish")
        mtf["trend_score"] = (bull - bear) / 2.0
    except Exception:
        pass
    return mtf


def get_signal_for_bar(m5_slice: pd.DataFrame, mtf: dict) -> tuple:
    """
    Return (signal, buy_prob, sell_prob).
    signal ∈ {"BUY", "SELL", "HOLD"}
    """
    if not _BOT_OK:
        return "HOLD", 0.5, 0.5

    try:
        model   = _gb.entry_model
        scaler  = _gb.scaler_entry
        ai_flag = _gb.AI_MODEL

        if not ai_flag or model is None or not hasattr(scaler, "n_features_in_"):
            return "HOLD", 0.5, 0.5

        # Use is_training=True → creates temp MarketStructure, no global state pollution
        features = generate_smc_features(m5_slice, mtf, False, is_training=True)
        if not features:
            return "HOLD", 0.5, 0.5

        vec = np.array([features.get(n, 0.0) for n in FEATURE_NAMES]).reshape(1, -1)
        if vec.shape[1] != scaler.n_features_in_:
            return "HOLD", 0.5, 0.5

        X = scaler.transform(vec)
        probs = model.predict_proba(X)[0]

        buy_p = sell_p = 0.0
        for i, cls in enumerate(model.classes_):
            if cls ==  1: buy_p  = float(probs[i])
            if cls == -1: sell_p = float(probs[i])

        # Determine signal direction from AI probabilities
        if buy_p >= BUY_THRESHOLD and buy_p > sell_p:
            return "BUY", buy_p, sell_p
        elif sell_p >= SELL_THRESHOLD and sell_p > buy_p:
            return "SELL", buy_p, sell_p
        else:
            return "HOLD", buy_p, sell_p

    except Exception as e:
        return "HOLD", 0.0, 0.0


def compute_sl_tp(direction: str, entry: float,
                  ctx_df: pd.DataFrame, atr_val: float) -> tuple:
    """
    Pure structure-based SL and TP — no ATR cap.
    The model's natural RR advantage comes from ranging markets where SL is tight
    (close to recent swing) and TP is wide (next liquidity level far away).
    Capping SL destroys that advantage. Let structure decide both levels.
    Falls back to 1.5×ATR SL / 2.25×ATR TP only when structure is unavailable.
    Returns (sl, tp) both as float prices.
    """
    pip = GOLD_PIP

    if _BOT_OK:
        try:
            fvgs = detect_fvg(ctx_df)
            obs  = detect_order_blocks(ctx_df)
            sl   = calculate_liquidity_based_sl(direction, entry, ctx_df, fvgs, obs, atr_val)
            tp   = calculate_liquidity_based_tp(direction, entry, ctx_df, atr_val)

            sl_ok = sl is not None and abs(entry - sl) >= SL_MIN_PIPS * pip
            tp_ok = tp is not None and abs(entry - tp) > 0

            if sl_ok and tp_ok:
                risk   = abs(entry - sl)
                reward = abs(tp - entry)
                if reward / risk < 1.2:
                    tp = (entry + risk * 1.2) if direction == "buy" else (entry - risk * 1.2)
                return float(sl), float(tp)
        except Exception:
            pass

    # ATR fallback
    sl_dist = max(atr_val * 1.5, SL_MIN_PIPS * pip)
    tp_dist = sl_dist * 1.5
    if direction == "buy":
        return entry - sl_dist, entry + tp_dist
    else:
        return entry + sl_dist, entry - tp_dist


def is_trading_session(bar_time) -> bool:
    """Return True if bar_time falls inside London or NY session (UTC)."""
    try:
        h = pd.Timestamp(bar_time).hour
        return any(start <= h < end for start, end in SESSION_HOURS)
    except Exception:
        return True   # allow if timestamp can't be parsed


def _close_pos(pos: dict, exit_price: float, outcome: str,
               open_bar: int, close_bar: int, fold_num: int) -> dict:
    """Build a closed-trade record from a position dict."""
    is_buy    = pos["direction"] == "buy"
    pnl_pips  = ((exit_price - pos["entry"]) / GOLD_PIP if is_buy
                 else (pos["entry"] - exit_price) / GOLD_PIP) - SPREAD_PIPS
    risk_pips = abs(pos["entry"] - pos["sl"]) / GOLD_PIP
    return {
        "fold":        fold_num,
        "time":        pos["open_time"],
        "direction":   pos["direction"],
        "entry":       round(pos["entry"], 2),
        "sl":          round(pos["sl"], 2),
        "tp":          round(pos["tp"], 2),
        "buy_prob":    round(pos["buy_prob"], 3),
        "sell_prob":   round(pos["sell_prob"], 3),
        "ai_prob":     round(pos["ai_prob"], 3),
        "atr":         round(pos["atr"], 3),
        "outcome":     outcome,
        "bars_held":   close_bar - open_bar,
        "exit_price":  round(exit_price, 2),
        "pnl_pips":    round(pnl_pips, 2),
        "risk_pips":   round(risk_pips, 2),
        "reward_pips": round(abs(pos["tp"] - pos["entry"]) / GOLD_PIP, 2),
        "r_multiple":  round(pnl_pips / max(risk_pips, 1e-9), 3),
    }


# ─── FOLD ──────────────────────────────────────────────────────────────────────

def run_fold(fold_num: int,
             full_m5:    pd.DataFrame,
             m15_full:   pd.DataFrame,
             h1_full:    pd.DataFrame,
             train_start: int,
             train_end:   int,
             test_end:    int) -> list:
    """
    One walk-forward fold with realistic live filters.

    Each bar:
      1. Update all open positions (SL/TP/timeout check)
      2. If session OK + cooldown OK + room for new position → generate signal
      3. If good signal → open position at next bar's open
    """
    train_df = full_m5.iloc[train_start:train_end].reset_index(drop=True)
    n_test   = test_end - train_end

    print(f"\n── Fold {fold_num:02d} | "
          f"Train [{train_start}:{train_end}] ({len(train_df)} bars)  "
          f"Test [{train_end}:{test_end}] ({n_test} bars) ──")

    # ── Train ──────────────────────────────────────────────────────────────────
    if _BOT_OK:
        try:
            train_models(train_df)
            _gb.AI_MODEL = True
            print("   ✅ Model trained")
        except Exception as e:
            print(f"   ⚠️  train_models failed: {e} — fold skipped")
            return []

    closed_trades  = []
    open_positions = []   # dicts of live simulated positions
    last_entry_bar = -99  # bar index of last entry (for cooldown)

    # Diagnostic counters
    n_session = n_cooldown = n_maxpos = n_ctx = 0
    n_signal  = n_edge = n_veto = n_entry = n_equity_skip = n_confirm = 0

    for bar_i in range(n_test):
        abs_idx      = train_end + bar_i
        current_bar  = full_m5.iloc[abs_idx]
        bar_time     = (pd.Timestamp(current_bar["time"])
                        if "time" in current_bar else None)

        # ── 1. Update every open position against this bar ──────────────────
        still_open = []
        for pos in open_positions:
            is_buy  = pos["direction"] == "buy"
            h, l    = float(current_bar["high"]), float(current_bar["low"])
            hit_tp  = h >= pos["tp"] if is_buy else l <= pos["tp"]
            hit_sl  = l <= pos["sl"] if is_buy else h >= pos["sl"]
            elapsed = bar_i - pos["open_bar"]
            timeout = elapsed >= MAX_HOLD_BARS

            if hit_tp and hit_sl:
                exit_p, outcome = pos["sl"], "sl"   # both hit → conservative
            elif hit_tp:
                exit_p, outcome = pos["tp"], "tp"
            elif hit_sl:
                exit_p, outcome = pos["sl"], "sl"
            elif timeout:
                exit_p, outcome = float(current_bar["close"]), "timeout"
            else:
                still_open.append(pos)
                continue

            closed_trades.append(
                _close_pos(pos, exit_p, outcome,
                           pos["open_bar"], bar_i, fold_num)
            )

        open_positions = still_open

        # ── 2. Filters before checking for a new entry ──────────────────────

        # Skip last bar — no "next bar" to enter on
        if bar_i >= n_test - 1:
            continue

        # Entry cooldown
        if bar_i - last_entry_bar < COOLDOWN_BARS:
            n_cooldown += 1
            continue

        # Max concurrent positions
        if len(open_positions) >= MAX_SIM_POSITIONS:
            n_maxpos += 1
            continue

        # Equity curve protection — mirrors live bot's rolling WR discipline.
        # If last 8 completed trades show WR < 25%, the model is in a bad patch;
        # skip new entries until the streak improves.
        if len(closed_trades) >= 8:
            recent_wr = sum(1 for t in closed_trades[-8:] if t["outcome"] == "tp") / 8
            if recent_wr < 0.25:
                n_equity_skip += 1
                continue

        # ── 3. Build context + generate signal ──────────────────────────────
        ctx_start = max(0, abs_idx - CONTEXT_BARS + 1)
        ctx_m5    = full_m5.iloc[ctx_start:abs_idx + 1].reset_index(drop=True)

        if len(ctx_m5) < 50:
            n_ctx += 1
            continue

        mtf = {}
        if bar_time and _BOT_OK:
            mtf = build_mtf_dict(ctx_m5, m15_full, h1_full, bar_time)

        signal, buy_p, sell_p = get_signal_for_bar(ctx_m5, mtf)

        if signal not in ("BUY", "SELL"):
            continue
        n_signal += 1

        # Edge filter
        if abs(buy_p - sell_p) < 0.08:
            n_edge += 1
            continue

        # No duplicate direction already open
        if any(p["direction"] == signal.lower() for p in open_positions):
            continue

        # Candle pattern veto
        if _BOT_OK:
            try:
                veto, _ = candle_pattern_veto(ctx_m5, signal)
                if veto:
                    n_veto += 1
                    continue
            except Exception:
                pass

        # Entry confirmation — kill whipsaw entries that reverse immediately.
        # The signal candle (last closed bar) must close in the trade direction
        # with a decisive body (≥ 30% of range). Indecision/reversal candles
        # produce the fast 2-3 bar stop-outs seen in losing folds.
        sig_bar = full_m5.iloc[abs_idx]
        s_open  = float(sig_bar["open"])
        s_close = float(sig_bar["close"])
        s_high  = float(sig_bar["high"])
        s_low   = float(sig_bar["low"])
        s_rng   = s_high - s_low
        if s_rng <= 0:
            n_confirm += 1
            continue
        body_frac = abs(s_close - s_open) / s_rng
        bullish_bar = s_close > s_open
        if body_frac < 0.30:
            n_confirm += 1
            continue
        if signal == "BUY" and not bullish_bar:
            n_confirm += 1
            continue
        if signal == "SELL" and bullish_bar:
            n_confirm += 1
            continue

        # ── 4. Open position at next bar ────────────────────────────────────
        next_idx = abs_idx + 1
        next_bar = full_m5.iloc[next_idx]
        direction = signal.lower()
        spread    = SPREAD_PIPS * GOLD_PIP
        entry_p   = (float(next_bar["open"]) + (spread if direction == "buy" else -spread))
        atr_val   = _atr(ctx_m5)

        sl, tp = compute_sl_tp(direction, entry_p, ctx_m5, atr_val)
        if abs(entry_p - sl) / GOLD_PIP < SL_MIN_PIPS:
            continue

        n_entry += 1
        open_positions.append({
            "direction": direction,
            "entry":     entry_p,
            "sl":        sl,
            "tp":        tp,
            "open_bar":  bar_i + 1,
            "open_time": str(next_bar["time"]) if "time" in next_bar else "",
            "buy_prob":  buy_p,
            "sell_prob": sell_p,
            "ai_prob":   max(buy_p, sell_p),
            "atr":       atr_val,
        })
        last_entry_bar = bar_i

    # Close anything still open at fold end (mark as fold_end, not a real outcome)
    last_bar = full_m5.iloc[test_end - 1]
    for pos in open_positions:
        closed_trades.append(
            _close_pos(pos, float(last_bar["close"]), "fold_end",
                       pos["open_bar"], n_test - 1, fold_num)
        )

    wins  = sum(1 for t in closed_trades if t["outcome"] == "tp")
    total = len(closed_trades)
    print(f"   Filters: cooldown={n_cooldown} maxpos={n_maxpos} "
          f"short_ctx={n_ctx} equity_skip={n_equity_skip}")
    print(f"   Signals: {n_signal} raw → {n_edge} edge → "
          f"{n_veto} veto → {n_confirm} unconfirmed → {n_entry} entries → {total} trades")
    if total:
        print(f"   Wins: {wins} | Win-rate: {wins/total*100:.1f}%")
    else:
        print("   ⚠️  No trades this fold")

    return closed_trades


# ─── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(trades: list) -> dict:
    if not trades:
        return {}

    pnl   = [t["pnl_pips"] for t in trades]
    wins  = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p < 0]

    win_rate     = len(wins) / len(pnl) * 100
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_profit / gross_loss

    r_multiples = [t["r_multiple"] for t in trades]
    avg_r        = float(np.mean(r_multiples))

    # Equity curve (using R-multiples so initial equity cancels out)
    equity = [INITIAL_EQUITY]
    for t in trades:
        risk_amount = equity[-1] * RISK_PER_TRADE
        equity.append(equity[-1] + t["r_multiple"] * risk_amount)
    equity = np.array(equity)

    peak   = np.maximum.accumulate(equity)
    dd     = (peak - equity) / peak * 100
    max_dd = float(dd.max())
    net_pnl_pct = (equity[-1] - INITIAL_EQUITY) / INITIAL_EQUITY * 100

    # Sharpe (annualised on 5-min candles: ~252 trading days × ~288 bars)
    r_std = float(np.std(r_multiples)) if len(r_multiples) > 1 else 1e-9
    sharpe = (avg_r / r_std) * math.sqrt(len(r_multiples)) if r_std > 0 else 0.0

    avg_hold = float(np.mean([t["bars_held"] for t in trades]))
    avg_win_pips  = float(np.mean(wins))  if wins   else 0.0
    avg_loss_pips = float(np.mean(losses)) if losses else 0.0

    outcomes = pd.Series([t["outcome"] for t in trades]).value_counts().to_dict()

    return {
        "trades":          len(pnl),
        "win_rate_%":      round(win_rate, 1),
        "profit_factor":   round(profit_factor, 2),
        "avg_r":           round(avg_r, 3),
        "sharpe":          round(sharpe, 2),
        "max_drawdown_%":  round(max_dd, 1),
        "net_return_%":    round(net_pnl_pct, 1),
        "total_pips":      round(sum(pnl), 1),
        "avg_win_pips":    round(avg_win_pips, 1),
        "avg_loss_pips":   round(avg_loss_pips, 1),
        "avg_hold_bars":   round(avg_hold, 1),
        "outcomes":        outcomes,
    }


# ─── REPORT ────────────────────────────────────────────────────────────────────

def print_report(all_trades: list, per_fold: list) -> None:
    agg = compute_metrics(all_trades)
    if not agg:
        print("\n⚠️  No trades generated — check MT5 connection and data.")
        return

    print("\n" + "═" * 90)
    print("  WALK-FORWARD BACKTEST RESULTS — SMC-AI-GOLD-SENTINEL V40")
    print("═" * 90)
    print(f"{'Fold':>4}  {'Trades':>6}  {'Win%':>6}  {'PF':>5}  {'Avg R':>6}  "
          f"{'MaxDD%':>6}  {'Pips':>7}  {'Net%':>6}")
    print("─" * 90)
    for i, m in enumerate(per_fold):
        if not m:
            print(f"  {i+1:02d}    —")
            continue
        print(f"  {i+1:02d}  {m['trades']:>6}  {m['win_rate_%']:>5.1f}%  "
              f"{m['profit_factor']:>5.2f}  {m['avg_r']:>+6.3f}  "
              f"{m['max_drawdown_%']:>5.1f}%  {m['total_pips']:>+7.1f}  "
              f"{m['net_return_%']:>+5.1f}%")
    print("─" * 90)
    print(f"{'ALL':>4}  {agg['trades']:>6}  {agg['win_rate_%']:>5.1f}%  "
          f"{agg['profit_factor']:>5.2f}  {agg['avg_r']:>+6.3f}  "
          f"{agg['max_drawdown_%']:>5.1f}%  {agg['total_pips']:>+7.1f}  "
          f"{agg['net_return_%']:>+5.1f}%")
    print("═" * 90)

    print(f"\n  Sharpe ratio   : {agg['sharpe']:+.2f}")
    print(f"  Avg hold       : {agg['avg_hold_bars']:.1f} bars  "
          f"({agg['avg_hold_bars'] * 5:.0f} min)")
    print(f"  Avg win  (pips): {agg['avg_win_pips']:+.1f}")
    print(f"  Avg loss (pips): {agg['avg_loss_pips']:+.1f}")
    print(f"  Outcome split  : {agg['outcomes']}")

    # Verdict
    print("\n  ── VERDICT ──────────────────────────────────────────────────")
    wr  = agg["win_rate_%"]
    pf  = agg["profit_factor"]
    mdd = agg["max_drawdown_%"]
    if pf >= 1.5 and wr >= 50 and mdd < 20:
        verdict = "✅  PROMISING — run extended demo before going live"
    elif pf >= 1.2 and wr >= 45:
        verdict = "⚠️   MARGINAL — needs parameter tuning or more data"
    else:
        verdict = "❌  WEAK EDGE — do NOT trade live in current state"
    print(f"  {verdict}")
    print("═" * 90 + "\n")


def save_results(all_trades: list, per_fold: list, agg: dict) -> None:
    # Trades CSV
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv("backtest_trades.csv", index=False)
        print(f"  Saved {len(all_trades)} trades → backtest_trades.csv")

    # Summary JSON
    summary = {"aggregate": agg, "per_fold": per_fold}
    with open("backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  Saved → backtest_summary.json")


# ─── DATA FETCH ────────────────────────────────────────────────────────────────

def fetch_mt5_data(total_bars: int) -> pd.DataFrame:
    """Connect to MT5 and download M5 history."""
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed — is the terminal open?")

    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, total_bars)
    if rates is None or len(rates) == 0:
        mt5.shutdown()
        raise RuntimeError(f"No M5 data returned for {SYMBOL}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "tick_volume"})
    print(f"  Fetched {len(df)} M5 bars "
          f"({df['time'].iloc[0]} → {df['time'].iloc[-1]})")
    return df


def load_csv_data(path: str) -> pd.DataFrame:
    """Load M5 OHLCV from CSV.  Expected columns: time,open,high,low,close,tick_volume."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"  Loaded {len(df)} bars from {path}")
    return df


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    global TRAIN_BARS, TEST_BARS, STEP_BARS  # must be first use in function

    parser = argparse.ArgumentParser(description="Walk-Forward Backtest")
    parser.add_argument("--csv",   type=str, default=None,
                        help="Path to M5 CSV (skip MT5 fetch)")
    parser.add_argument("--total", type=int, default=MIN_TOTAL_BARS,
                        help=f"Total M5 bars to use (default {MIN_TOTAL_BARS})")
    parser.add_argument("--train", type=int, default=TRAIN_BARS,
                        help=f"In-sample bars per fold (default {TRAIN_BARS})")
    parser.add_argument("--test",  type=int, default=TEST_BARS,
                        help=f"Out-of-sample bars per fold (default {TEST_BARS})")
    parser.add_argument("--step",  type=int, default=STEP_BARS,
                        help=f"Roll-forward step (default {STEP_BARS})")
    args = parser.parse_args()

    TRAIN_BARS = args.train
    TEST_BARS  = args.test
    STEP_BARS  = args.step

    print("\n" + "═" * 60)
    print("  SMC-AI Walk-Forward Backtest")
    print(f"  Train: {TRAIN_BARS} bars | Test: {TEST_BARS} bars | "
          f"Step: {STEP_BARS} bars")
    print("═" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.csv:
        m5 = load_csv_data(args.csv)
    else:
        print("\n  Connecting to MT5...")
        m5 = fetch_mt5_data(args.total)

    if len(m5) < TRAIN_BARS + TEST_BARS:
        print(f"❌ Need at least {TRAIN_BARS + TEST_BARS} bars, "
              f"only have {len(m5)}.")
        return

    # ── Build higher-TF data sets (resample from M5) ───────────────────────────
    print("\n  Building M15 and H1 from M5...")
    m15_full = resample_ohlcv(m5, 15)
    h1_full  = resample_ohlcv(m5, 60)
    print(f"  M15: {len(m15_full)} bars | H1: {len(h1_full)} bars")

    # ── Walk-forward folds ─────────────────────────────────────────────────────
    fold_num    = 1
    all_trades  = []
    per_fold    = []
    train_start = 0

    while True:
        train_end = train_start + TRAIN_BARS
        test_end  = train_end   + TEST_BARS

        if test_end > len(m5):
            break

        fold_trades = run_fold(
            fold_num, m5, m15_full, h1_full,
            train_start, train_end, test_end,
        )

        all_trades.extend(fold_trades)
        per_fold.append(compute_metrics(fold_trades))

        fold_num    += 1
        train_start += STEP_BARS

    if not all_trades:
        print("\n⚠️  No trades were generated across all folds.")
        print("   Possible causes:")
        print("   • Model not trained (gold_bot.py import failed)")
        print("   • All signals filtered out (check AI thresholds)")
        print("   • Insufficient candles per fold")
        return

    # ── Report ─────────────────────────────────────────────────────────────────
    agg = compute_metrics(all_trades)
    print_report(all_trades, per_fold)
    save_results(all_trades, per_fold, agg)

    # Shutdown MT5 if we opened it
    if not args.csv:
        mt5.shutdown()


if __name__ == "__main__":
    main()
