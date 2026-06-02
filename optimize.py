#!/usr/bin/env python3
"""
Parameter Optimizer — SMC-AI-GOLD-SENTINEL  (TURBO edition)
============================================================
Trains ONCE per fold (not per combo), then sweeps all parameters for free.

How it works:
  Phase 1  Pre-compute features for all test bars          (~1.5 h, runs once)
  Phase 2  Train + cache probabilities — 1 config × N folds (~2 h)
  Phase 3  Sweep ALL 5 parameters on cached data            (~2 min)

Total: ~3-4 hours for 576 combinations.

The key insight: the model is a DIRECTION predictor. BUY/SELL thresholds and
execution-level SL/TP multipliers can be varied without retraining.

Usage
-----
    python optimize.py                      # default, ~3-4 h
    python optimize.py --quick              # fewer combos, ~2-3 h
    python optimize.py --total 8000         # less data = fewer folds = faster
    python optimize.py --csv gold_m5.csv    # offline mode

Output
------
    optimize_results.csv    — all combos ranked
    config_optimized.json   — best parameters for gold_bot.py
"""

import argparse, os, sys, json, warnings, itertools, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BOT_DIR)

# ─── PARAMETER GRID ────────────────────────────────────────────────────────────
# ALL of these are swept in Phase 3 (near-instant).
# Only LOOKAHEAD_BARS affects training — fixed at 12 for training.

FULL_GRID = {
    "BUY_THRESHOLD":   [0.50, 0.55, 0.60, 0.65],
    "SELL_THRESHOLD":  [0.40, 0.45, 0.50, 0.55],
    "TP_ATR_MULT":     [1.5, 2.0, 2.5, 3.0],
    "SL_ATR_MULT":     [1.0, 1.5, 2.0],
    "LOOKAHEAD_BARS":  [12],   # fixed for training; only affects labeling
}

QUICK_GRID = {
    "BUY_THRESHOLD":   [0.50, 0.55, 0.60],
    "SELL_THRESHOLD":  [0.40, 0.45, 0.50],
    "TP_ATR_MULT":     [1.5, 2.0, 2.5, 3.0],
    "SL_ATR_MULT":     [1.0, 1.5, 2.0],
    "LOOKAHEAD_BARS":  [12],
}

# Fixed training parameters (what the model learns from)
TRAIN_LOOKAHEAD = 12
TRAIN_TP_ATR    = 1.5
TRAIN_SL_ATR    = 1.0

TRAIN_BARS   = 2000
TEST_BARS    = 500
STEP_BARS    = 500
CONTEXT_BARS = 200
GOLD_PIP     = 0.10
SPREAD_PIPS  = 2.5
SL_MIN_PIPS  = 12.0
MAX_HOLD     = 48
SYMBOL       = "XAUUSD"

# ─── IMPORTS ───────────────────────────────────────────────────────────────────

try:
    import MetaTrader5 as mt5
    _MT5_OK = True
except ImportError:
    _MT5_OK = False

try:
    from gold_bot import (
        generate_smc_features, train_models,
        FEATURE_NAMES, create_entry_target,
    )
    import gold_bot as _gb
    _BOT_OK = True
except Exception as e:
    print(f"  Cannot import gold_bot.py: {e}")
    sys.exit(1)

try:
    from gold_bot import detect_market_regime
    _REGIME_OK = True
except ImportError:
    _REGIME_OK = False

# ─── HELPERS ───────────────────────────────────────────────────────────────────

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
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),    close=("close", "last"),
        tick_volume=("tick_volume", "sum"),
    ).dropna().reset_index()
    return out.rename(columns={"index": "time"}) if "index" in out.columns else out


def build_mtf_dict(ctx_m5, m15_full, h1_full, bar_time):
    mtf = {}
    if not _REGIME_OK:
        return mtf
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


def compute_sl_tp_atr(direction, price, atr_val, sl_mult, tp_mult):
    sl_dist = max(atr_val * sl_mult, SL_MIN_PIPS * GOLD_PIP)
    tp_dist = sl_dist * (tp_mult / sl_mult)
    if direction == "buy":
        return price - sl_dist, price + tp_dist
    return price + sl_dist, price - tp_dist


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


def profit_factor(trades):
    wins   = sum(t["pnl_pips"] for t in trades if t["pnl_pips"] > 0)
    losses = abs(sum(t["pnl_pips"] for t in trades if t["pnl_pips"] < 0))
    return wins / losses if losses > 0 else (wins if wins > 0 else 0.0)


def fmt_time(seconds):
    if seconds < 60:   return f"{seconds:.0f}s"
    if seconds < 3600: return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


# ─── PHASE 1+2 COMBINED: Train per fold + score test bars ──────────────────────
# Features are computed per-bar inside the fold loop (not pre-cached) because
# the training step already generates features internally. We just need to
# score test bars after training — that part is fast.

def train_and_score_all_folds(full_m5, m15_full, h1_full, fold_ranges):
    """
    For each fold:
      1. Train the model on the training window (with fixed TRAIN params)
      2. Score every test bar → cache (buy_p, sell_p, atr, open_next, bar_abs)

    Returns: list of lists — one per fold, each containing scored bar tuples.
    """
    n_folds = len(fold_ranges)
    all_fold_scores = []
    t0 = time.time()

    # Monkey-patch create_entry_target for training
    orig_fn = _gb.create_entry_target
    def _patched(df, lookahead=TRAIN_LOOKAHEAD,
                 tp_atr=TRAIN_TP_ATR, sl_atr=TRAIN_SL_ATR):
        return orig_fn(df, lookahead=lookahead, tp_atr=tp_atr, sl_atr=sl_atr)
    _gb.create_entry_target = _patched

    try:
        for fi, (train_start, train_end, test_end) in enumerate(fold_ranges):
            t_fold = time.time()
            elapsed_so_far = t_fold - t0
            if fi > 0:
                eta = elapsed_so_far / fi * (n_folds - fi)
                print(f"\n  Fold {fi+1}/{n_folds}  "
                      f"(elapsed {fmt_time(elapsed_so_far)} | ETA {fmt_time(eta)})")
            else:
                print(f"\n  Fold {fi+1}/{n_folds}")

            # ── Train ──
            train_df = full_m5.iloc[train_start:train_end].reset_index(drop=True)
            try:
                train_models(train_df)
                _gb.AI_MODEL = True
            except Exception as e:
                print(f"    Train failed: {e}")
                all_fold_scores.append([])
                continue

            model  = _gb.entry_model
            scaler = _gb.scaler_entry
            if model is None or not hasattr(scaler, "n_features_in_"):
                all_fold_scores.append([])
                continue

            n_feats = scaler.n_features_in_

            # ── Score test bars ──
            fold_scores = []
            n_bars = test_end - train_end - 1
            for bar_i, bar_abs in enumerate(range(train_end, test_end - 1)):
                if bar_abs + 1 >= len(full_m5):
                    continue

                ctx_start = max(0, bar_abs - CONTEXT_BARS + 1)
                ctx_m5    = full_m5.iloc[ctx_start:bar_abs + 1].reset_index(drop=True)
                if len(ctx_m5) < 50:
                    continue

                bar_time = ctx_m5["time"].iloc[-1] if "time" in ctx_m5.columns else None
                mtf = build_mtf_dict(ctx_m5, m15_full, h1_full, bar_time) if bar_time else {}

                try:
                    feats = generate_smc_features(ctx_m5, mtf, False, is_training=True)
                    if not feats:
                        continue
                    vec = np.array(
                        [feats.get(n, 0.0) for n in FEATURE_NAMES]
                    ).reshape(1, -1)
                    if vec.shape[1] != n_feats:
                        continue
                    probs  = model.predict_proba(scaler.transform(vec))[0]
                    buy_p  = 0.0
                    sell_p = 0.0
                    for k, cls in enumerate(model.classes_):
                        if cls ==  1: buy_p  = float(probs[k])
                        if cls == -1: sell_p = float(probs[k])

                    atr_val   = _atr(ctx_m5)
                    open_next = float(full_m5.iloc[bar_abs + 1]["open"])
                    fold_scores.append((bar_abs, buy_p, sell_p, atr_val, open_next))
                except Exception:
                    continue

                if bar_i > 0 and bar_i % 100 == 0:
                    pct = bar_i / n_bars * 100
                    print(f"\r    Scoring: {bar_i}/{n_bars} ({pct:.0f}%)  "
                          f"{len(fold_scores)} scored", end="", flush=True)

            fold_time = time.time() - t_fold
            print(f"\r    {len(fold_scores)} bars scored in {fmt_time(fold_time)}        ")
            all_fold_scores.append(fold_scores)

    finally:
        _gb.create_entry_target = orig_fn

    total = time.time() - t0
    total_scored = sum(len(s) for s in all_fold_scores)
    print(f"\n  Training + scoring done: {total_scored} total bar scores "
          f"in {fmt_time(total)}")
    return all_fold_scores


# ─── PHASE 3: Lightning-fast parameter sweep ──────────────────────────────────

def sweep_parameters(full_m5, fold_ranges, all_fold_scores, grid):
    """
    Sweep ALL parameter combinations using cached probabilities.
    Training already happened — this only applies thresholds and SL/TP sizing.
    """
    keys   = ["BUY_THRESHOLD", "SELL_THRESHOLD", "TP_ATR_MULT", "SL_ATR_MULT", "LOOKAHEAD_BARS"]
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total  = len(combos)

    print(f"\n  Phase 3 — Sweeping {total} parameter combinations (fast)...")
    t0 = time.time()
    results = []

    for ci, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        buy_thresh  = params["BUY_THRESHOLD"]
        sell_thresh = params["SELL_THRESHOLD"]
        tp_mult     = params["TP_ATR_MULT"]
        sl_mult     = params["SL_ATR_MULT"]

        all_trades = []

        for fi, (train_start, train_end, test_end) in enumerate(fold_ranges):
            fold_scores = all_fold_scores[fi]

            for (bar_abs, buy_p, sell_p, atr_val, open_next) in fold_scores:
                # Threshold + edge filter
                if buy_p >= buy_thresh and buy_p > sell_p and abs(buy_p - sell_p) >= 0.08:
                    signal = "buy"
                elif sell_p >= sell_thresh and sell_p > buy_p and abs(buy_p - sell_p) >= 0.08:
                    signal = "sell"
                else:
                    continue

                spread = SPREAD_PIPS * GOLD_PIP
                entry  = open_next + (spread if signal == "buy" else -spread)
                sl, tp = compute_sl_tp_atr(signal, entry, atr_val, sl_mult, tp_mult)
                if abs(entry - sl) / GOLD_PIP < SL_MIN_PIPS:
                    continue

                f_start = bar_abs + 1
                f_end   = min(f_start + MAX_HOLD, len(full_m5))
                future  = full_m5.iloc[f_start:f_end].reset_index(drop=True)
                if len(future) == 0:
                    continue

                result = simulate_trade(signal, entry, sl, tp, future)
                all_trades.append(result)

        if not all_trades:
            results.append({**params, "trades": 0, "win_rate": 0,
                            "profit_factor": 0, "avg_r": 0, "total_pips": 0})
            continue

        pnl  = [t["pnl_pips"]  for t in all_trades]
        rs   = [t["r_multiple"] for t in all_trades]
        wins = sum(1 for p in pnl if p > 0)
        pf   = profit_factor(all_trades)
        wr   = wins / len(pnl) * 100
        results.append({
            **params,
            "trades":        len(pnl),
            "win_rate":      round(wr, 1),
            "profit_factor": round(pf, 2),
            "avg_r":         round(float(np.mean(rs)), 3),
            "total_pips":    round(sum(pnl), 1),
        })

        if (ci + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (ci + 1) * (total - ci - 1)
            print(f"\r    {ci+1}/{total} combos  |  {fmt_time(elapsed)}  "
                  f"|  ETA {fmt_time(eta)}   ", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\r    {total}/{total} combos done in {fmt_time(elapsed)}          ")
    return results


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SMC-AI Parameter Optimizer (TURBO)")
    parser.add_argument("--csv",   default=None,
                        help="Path to OHLCV CSV file (offline mode)")
    parser.add_argument("--total", type=int, default=10000,
                        help="Bars to fetch from MT5 (default 10000)")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller grid, faster run")
    parser.add_argument("--step",  type=int, default=None,
                        help="Fold step size (default 500; use 1000 for 2x speed)")
    args = parser.parse_args()

    global STEP_BARS
    if args.step:
        STEP_BARS = args.step

    grid = QUICK_GRID if args.quick else FULL_GRID

    total_combos = 1
    for v in grid.values():
        total_combos *= len(v)

    print("\n" + "=" * 65)
    print("  SMC-AI Parameter Optimizer  (TURBO)")
    print("=" * 65)
    print(f"  Total combinations   : {total_combos}")
    print(f"  Training configs     : 1  (train once, sweep for free)")
    print(f"  Grid                 : {'QUICK' if args.quick else 'FULL'}")
    print(f"  Fold step            : {STEP_BARS} bars")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.csv:
        m5 = pd.read_csv(args.csv)
        m5["time"] = pd.to_datetime(m5["time"])
        m5 = m5.sort_values("time").reset_index(drop=True)
    else:
        if not _MT5_OK:
            print("  MetaTrader5 not installed. Use --csv to provide data.")
            return
        print("\n  Connecting to MT5...")
        import MetaTrader5 as mt5_mod
        if not mt5_mod.initialize():
            print("  MT5 not available"); return
        rates = mt5_mod.copy_rates_from_pos(SYMBOL, mt5_mod.TIMEFRAME_M5, 0, args.total)
        mt5_mod.shutdown()
        if rates is None:
            print("  No data returned"); return
        m5 = pd.DataFrame(rates)
        m5["time"] = pd.to_datetime(m5["time"], unit="s")

    m15_full = resample_ohlcv(m5, 15)
    h1_full  = resample_ohlcv(m5, 60)
    print(f"\n  Data: {len(m5)} M5 bars | {len(m15_full)} M15 | {len(h1_full)} H1")

    # ── Build fold ranges ──────────────────────────────────────────────────────
    fold_ranges = []
    train_start = 0
    while True:
        train_end = train_start + TRAIN_BARS
        test_end  = train_end   + TEST_BARS
        if test_end > len(m5):
            break
        fold_ranges.append((train_start, train_end, test_end))
        train_start += STEP_BARS

    n_folds = len(fold_ranges)
    est_hours = n_folds * 5 / 60   # ~5 min per fold
    print(f"  Folds: {n_folds}  |  Estimated training time: ~{est_hours:.1f} hours")

    # ── Phase 1+2: Train + Score (runs once) ──────────────────────────────────
    t_total = time.time()
    print("\n" + "-" * 65)
    print("  PHASE 1+2 — Training 1 model per fold + scoring all test bars")
    print("  (This is the slow part. Runs ONCE, then sweep is instant.)")
    print("-" * 65)

    all_fold_scores = train_and_score_all_folds(
        m5, m15_full, h1_full, fold_ranges
    )

    # ── Phase 3: Parameter sweep ───────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  PHASE 3 — Parameter sweep (near-instant)")
    print("-" * 65)

    results = sweep_parameters(m5, fold_ranges, all_fold_scores, grid)

    elapsed_total = time.time() - t_total
    print(f"\n  Total optimizer runtime: {fmt_time(elapsed_total)}")

    # ── Save results ───────────────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values("profit_factor", ascending=False)
    df.to_csv("optimize_results.csv", index=False)
    print(f"  Full results saved -> optimize_results.csv")

    # ── Show top 10 ───────────────────────────────────────────────────────────
    valid = df[df["trades"] >= 20].copy()
    if valid.empty:
        valid = df.copy()

    valid["score"] = (valid["profit_factor"] * 0.5 +
                      valid["avg_r"]          * 0.3 +
                      valid["win_rate"] / 100  * 0.2)
    valid = valid.sort_values("score", ascending=False)

    print("\n" + "=" * 65)
    print("  TOP 10 PARAMETER COMBINATIONS")
    print("=" * 65)
    for rank, (_, row) in enumerate(valid.head(10).iterrows(), 1):
        print(f"\n  #{rank}")
        print(f"    BUY={row['BUY_THRESHOLD']}  SELL={row['SELL_THRESHOLD']}  "
              f"TP_ATR={row['TP_ATR_MULT']}  SL_ATR={row['SL_ATR_MULT']}")
        print(f"    PF {row['profit_factor']:.2f}  |  WR {row['win_rate']:.1f}%  "
              f"|  AvgR {row['avg_r']:+.3f}  |  {int(row['trades'])} trades  "
              f"|  {row['total_pips']:.0f} pips")

    best = valid.iloc[0]

    print("\n" + "=" * 65)
    print("  BEST PARAMETERS")
    print("=" * 65)
    for k in grid.keys():
        print(f"  {k:22s}: {best[k]}")
    print(f"\n  Profit Factor : {best['profit_factor']:.2f}")
    print(f"  Win Rate      : {best['win_rate']:.1f}%")
    print(f"  Avg R         : {best['avg_r']:+.3f}")
    print(f"  Trades        : {int(best['trades'])}")
    print(f"  Total Pips    : {best['total_pips']:.0f}")
    print("=" * 65)

    # ── Save best config ───────────────────────────────────────────────────────
    config = {k: float(best[k]) for k in grid.keys()}
    config["_note"] = (
        "Generated by optimize.py TURBO. "
        "Paste BUY_THRESHOLD / SELL_THRESHOLD into gold_bot.py constants. "
        "Re-run monthly to refresh."
    )
    config["_generated"]     = pd.Timestamp.now().isoformat()
    config["_profit_factor"] = float(best["profit_factor"])
    config["_win_rate"]      = float(best["win_rate"])
    config["_trades_tested"] = int(best["trades"])
    config["_total_pips"]    = float(best["total_pips"])

    with open("config_optimized.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  Best config saved -> config_optimized.json\n")


if __name__ == "__main__":
    main()
