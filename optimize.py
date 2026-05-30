#!/usr/bin/env python3
"""
Parameter Optimizer — SMC-AI-GOLD-SENTINEL
===========================================
Walk-forward grid search over key thresholds.
Evaluates every combination on out-of-sample data using the same
backtest engine as backtest.py — no in-sample overfitting.

Usage
-----
    python optimize.py                      # default grid, 10k bars
    python optimize.py --total 20000        # more history
    python optimize.py --csv gold_m5.csv    # offline (no MT5)
    python optimize.py --quick              # small grid, fast run

Output
------
    optimize_results.csv    — every combo tested with metrics
    config_optimized.json   — best parameters ready to paste into gold_bot.py
"""

import argparse, os, sys, json, warnings, itertools, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

_BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BOT_DIR)

# ─── PARAMETER GRID ────────────────────────────────────────────────────────────
# Each key maps to a list of candidate values.
# QUICK mode uses the shorter inner lists.

FULL_GRID = {
    "BUY_THRESHOLD":   [0.50, 0.55, 0.60, 0.65],
    "SELL_THRESHOLD":  [0.40, 0.45, 0.50, 0.55],
    "TP_ATR_MULT":     [1.5, 2.0, 2.5, 3.0],
    "SL_ATR_MULT":     [1.0, 1.5, 2.0],
    "LOOKAHEAD_BARS":  [8, 12, 16],
}

QUICK_GRID = {
    "BUY_THRESHOLD":   [0.50, 0.55, 0.60],
    "SELL_THRESHOLD":  [0.40, 0.45, 0.50],
    "TP_ATR_MULT":     [1.5, 2.0, 2.5],
    "SL_ATR_MULT":     [1.0, 1.5],
    "LOOKAHEAD_BARS":  [8, 12],
}

# Walk-forward windows (matching backtest.py defaults)
TRAIN_BARS  = 2000
TEST_BARS   = 500
STEP_BARS   = 500
CONTEXT_BARS = 200
GOLD_PIP    = 0.10
SPREAD_PIPS = 2.5
SL_MIN_PIPS = 12.0
MAX_HOLD    = 48
SYMBOL      = "XAUUSD"

# ─── IMPORTS ───────────────────────────────────────────────────────────────────

try:
    from gold_bot import (
        generate_smc_features, train_models,
        detect_fvg, detect_order_blocks, detect_liquidity_sweep,
        detect_liquidity_pools, get_mss_signal, candle_pattern_veto,
        FEATURE_NAMES, create_entry_target, MarketStructure,
    )
    import gold_bot as _gb
    _BOT_OK = True
except Exception as e:
    print(f"❌ Cannot import gold_bot.py: {e}")
    sys.exit(1)

# ─── HELPERS (shared with backtest.py) ─────────────────────────────────────────

def _atr(df, period=14):
    s = (df["high"] - df["low"]).rolling(period).mean()
    v = s.iloc[-1]
    return max(float(v) if not pd.isna(v) else (df["high"] - df["low"]).tail(5).mean(), GOLD_PIP)


def resample_ohlcv(m5, minutes):
    df = m5.copy()
    if "time" in df.columns:
        df = df.set_index("time")
    df.index = pd.to_datetime(df.index)
    out = df.resample(f"{minutes}min").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"),
        tick_volume=("tick_volume","sum"),
    ).dropna().reset_index()
    return out.rename(columns={"index":"time"}) if "index" in out.columns else out


def build_mtf_dict(ctx_m5, m15_full, h1_full, bar_time):
    from gold_bot import detect_market_regime
    mtf = {}
    try:
        m15u = m15_full[m15_full["time"] <= bar_time].tail(60)
        if len(m15u) >= 20:
            mtf["M15_trend"] = detect_market_regime(m15u)
            mtf["M15"] = m15u
        h1u = h1_full[h1_full["time"] <= bar_time].tail(250)
        if len(h1u) >= 30:
            mtf["H1_trend"] = detect_market_regime(h1u, fast_ema=50, slow_ema=200)
            mtf["H1"] = h1u
        bull = sum(1 for t in [mtf.get("M15_trend"), mtf.get("H1_trend")] if t == "bullish")
        bear = sum(1 for t in [mtf.get("M15_trend"), mtf.get("H1_trend")] if t == "bearish")
        mtf["trend_score"] = (bull - bear) / 2.0
    except Exception:
        pass
    return mtf


def simulate_trade(direction, entry, sl, tp, future_bars):
    for i, (_, bar) in enumerate(future_bars.iterrows()):
        hit_tp = bar["high"] >= tp if direction == "buy" else bar["low"]  <= tp
        hit_sl = bar["low"]  <= sl if direction == "buy" else bar["high"] >= sl
        if hit_tp and hit_sl:
            exit_p, outcome = sl, "sl"
        elif hit_tp:
            exit_p, outcome = tp, "tp"
        elif hit_sl:
            exit_p, outcome = sl, "sl"
        else:
            if i == len(future_bars) - 1:
                exit_p, outcome = bar["close"], "timeout"
            else:
                continue
        risk_pips = abs(entry - sl) / GOLD_PIP
        pnl_pips  = ((exit_p - entry) / GOLD_PIP if direction == "buy"
                     else (entry - exit_p) / GOLD_PIP) - SPREAD_PIPS
        r_mult    = pnl_pips / max(risk_pips, 1e-9)
        return {"outcome": outcome, "pnl_pips": pnl_pips, "r_multiple": r_mult,
                "bars_held": i + 1}
    return {"outcome": "timeout", "pnl_pips": 0.0, "r_multiple": 0.0, "bars_held": 0}


def compute_sl_tp_atr(direction, price, atr_val, sl_mult, tp_mult):
    """Simple ATR-based SL/TP — consistent across all parameter combos."""
    sl_dist = max(atr_val * sl_mult, SL_MIN_PIPS * GOLD_PIP)
    tp_dist = sl_dist * (tp_mult / sl_mult)
    if direction == "buy":
        return price - sl_dist, price + tp_dist
    return price + sl_dist, price - tp_dist


def profit_factor(trades):
    wins   = sum(t["pnl_pips"] for t in trades if t["pnl_pips"] > 0)
    losses = abs(sum(t["pnl_pips"] for t in trades if t["pnl_pips"] < 0))
    return wins / losses if losses > 0 else (wins if wins > 0 else 0.0)


# ─── ONE FOLD UNDER GIVEN PARAMS ────────────────────────────────────────────────

def run_fold_with_params(full_m5, m15_full, h1_full,
                         train_start, train_end, test_end,
                         params):
    train_df = full_m5.iloc[train_start:train_end].reset_index(drop=True)

    # Retrain with modified lookahead
    try:
        # Temporarily monkey-patch create_entry_target to use the param lookahead
        orig_fn = _gb.create_entry_target
        def _patched(df, lookahead=params["LOOKAHEAD_BARS"],
                     tp_atr=params["TP_ATR_MULT"],
                     sl_atr=params["SL_ATR_MULT"]):
            return orig_fn(df, lookahead=lookahead, tp_atr=tp_atr, sl_atr=sl_atr)
        _gb.create_entry_target = _patched
        train_models(train_df)
        _gb.AI_MODEL = True
    except Exception as e:
        return []
    finally:
        _gb.create_entry_target = orig_fn

    buy_thresh  = params["BUY_THRESHOLD"]
    sell_thresh = params["SELL_THRESHOLD"]
    sl_mult     = params["SL_ATR_MULT"]
    tp_mult     = params["TP_ATR_MULT"]

    trades = []
    n_test = test_end - train_end

    for bar_i in range(n_test - 1):
        ctx_start = max(0, train_end - CONTEXT_BARS + bar_i)
        ctx_m5    = full_m5.iloc[ctx_start:train_end + bar_i + 1].reset_index(drop=True)
        if len(ctx_m5) < 50:
            continue

        bar_time = ctx_m5["time"].iloc[-1] if "time" in ctx_m5.columns else None
        mtf = build_mtf_dict(ctx_m5, m15_full, h1_full, bar_time) if bar_time else {}

        try:
            model  = _gb.entry_model
            scaler = _gb.scaler_entry
            if model is None or not hasattr(scaler, "n_features_in_"):
                continue
            feats = generate_smc_features(ctx_m5, mtf, False, is_training=True)
            if not feats:
                continue
            vec   = np.array([feats.get(n, 0.0) for n in FEATURE_NAMES]).reshape(1, -1)
            if vec.shape[1] != scaler.n_features_in_:
                continue
            probs = model.predict_proba(scaler.transform(vec))[0]
            buy_p = sell_p = 0.0
            for i, cls in enumerate(model.classes_):
                if cls ==  1: buy_p  = float(probs[i])
                if cls == -1: sell_p = float(probs[i])
        except Exception:
            continue

        if buy_p >= buy_thresh and buy_p > sell_p:
            signal = "buy"
        elif sell_p >= sell_thresh and sell_p > buy_p:
            signal = "sell"
        else:
            continue

        if abs(buy_p - sell_p) < 0.08:
            continue

        next_bar = full_m5.iloc[train_end + bar_i + 1]
        spread   = SPREAD_PIPS * GOLD_PIP
        entry    = next_bar["open"] + (spread if signal == "buy" else -spread)
        atr_val  = _atr(ctx_m5)

        sl, tp = compute_sl_tp_atr(signal, entry, atr_val, sl_mult, tp_mult)
        if abs(entry - sl) / GOLD_PIP < SL_MIN_PIPS:
            continue

        f_start = train_end + bar_i + 1
        f_end   = min(f_start + MAX_HOLD, len(full_m5))
        future  = full_m5.iloc[f_start:f_end].reset_index(drop=True)
        if len(future) == 0:
            continue

        result = simulate_trade(signal, entry, sl, tp, future)
        trades.append(result)

    return trades


# ─── FULL GRID SEARCH ──────────────────────────────────────────────────────────

def run_optimization(full_m5, m15_full, h1_full, grid):
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total  = len(combos)
    print(f"\n  Grid: {total} combinations × "
          f"{(len(full_m5) - TRAIN_BARS) // STEP_BARS} folds each\n")

    results = []

    for ci, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        label  = " | ".join(f"{k}={v}" for k, v in params.items())
        print(f"  [{ci+1:3d}/{total}] {label}", end="  ", flush=True)

        all_trades = []
        train_start = 0
        while True:
            train_end = train_start + TRAIN_BARS
            test_end  = train_end   + TEST_BARS
            if test_end > len(full_m5):
                break
            fold_trades = run_fold_with_params(
                full_m5, m15_full, h1_full,
                train_start, train_end, test_end, params,
            )
            all_trades.extend(fold_trades)
            train_start += STEP_BARS

        if not all_trades:
            print("→ no trades")
            results.append({**params, "trades": 0, "win_rate": 0,
                             "profit_factor": 0, "avg_r": 0, "total_pips": 0})
            continue

        pnl   = [t["pnl_pips"]  for t in all_trades]
        rs    = [t["r_multiple"] for t in all_trades]
        wins  = sum(1 for p in pnl if p > 0)
        pf    = profit_factor(all_trades)
        wr    = wins / len(pnl) * 100
        avg_r = float(np.mean(rs))

        print(f"→ {len(pnl):3d} trades | WR {wr:4.1f}% | PF {pf:.2f} | AvgR {avg_r:+.3f}")
        results.append({
            **params,
            "trades":        len(pnl),
            "win_rate":      round(wr, 1),
            "profit_factor": round(pf, 2),
            "avg_r":         round(avg_r, 3),
            "total_pips":    round(sum(pnl), 1),
        })

    return results


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default=None)
    parser.add_argument("--total", type=int, default=10000)
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller grid for fast testing")
    args = parser.parse_args()

    grid = QUICK_GRID if args.quick else FULL_GRID

    print("\n" + "═" * 60)
    print("  SMC-AI Parameter Optimizer")
    print(f"  Grid size: {sum(len(v) for v in grid.values())} values across "
          f"{len(grid)} parameters")
    print("═" * 60)

    if args.csv:
        m5 = pd.read_csv(args.csv)
        m5["time"] = pd.to_datetime(m5["time"])
        m5 = m5.sort_values("time").reset_index(drop=True)
    else:
        print("\n  Connecting to MT5...")
        if not mt5.initialize():
            print("❌ MT5 not available"); return
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, args.total)
        mt5.shutdown()
        if rates is None:
            print("❌ No data returned"); return
        m5 = pd.DataFrame(rates)
        m5["time"] = pd.to_datetime(m5["time"], unit="s")

    m15_full = resample_ohlcv(m5, 15)
    h1_full  = resample_ohlcv(m5, 60)
    print(f"  Data: {len(m5)} M5 bars | {len(m15_full)} M15 | {len(h1_full)} H1")

    t0      = time.time()
    results = run_optimization(m5, m15_full, h1_full, grid)
    elapsed = time.time() - t0

    # ── Save full results ──────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    df_results.to_csv("optimize_results.csv", index=False)
    print(f"\n  Full results saved → optimize_results.csv  ({elapsed:.0f}s)")

    # ── Pick best combo ────────────────────────────────────────────────────────
    # Rank by: profit_factor first, then avg_r, then win_rate.
    # Require at least 20 trades (otherwise cherry-picking a lucky 3-trade combo).
    valid = df_results[df_results["trades"] >= 20].copy()
    if valid.empty:
        valid = df_results.copy()

    valid["score"] = (valid["profit_factor"] * 0.5 +
                      valid["avg_r"]         * 0.3 +
                      valid["win_rate"] / 100 * 0.2)

    best_row = valid.sort_values("score", ascending=False).iloc[0]

    print("\n" + "═" * 60)
    print("  BEST PARAMETERS FOUND")
    print("═" * 60)
    for k in grid.keys():
        print(f"  {k:22s}: {best_row[k]}")
    print(f"\n  Profit Factor : {best_row['profit_factor']:.2f}")
    print(f"  Win Rate      : {best_row['win_rate']:.1f}%")
    print(f"  Avg R         : {best_row['avg_r']:+.3f}")
    print(f"  Trades        : {int(best_row['trades'])}")
    print("═" * 60)

    # ── Save optimized config ──────────────────────────────────────────────────
    config = {k: float(best_row[k]) for k in grid.keys()}
    config["_note"] = (
        "Generated by optimize.py — paste these values into gold_bot.py "
        "constants section. Re-run optimize.py monthly to refresh."
    )
    config["_generated"] = pd.Timestamp.now().isoformat()
    config["_profit_factor"] = float(best_row["profit_factor"])
    config["_win_rate"]      = float(best_row["win_rate"])
    config["_trades_tested"] = int(best_row["trades"])

    with open("config_optimized.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  Best config saved → config_optimized.json")


if __name__ == "__main__":
    main()
