#!/usr/bin/env python3
"""
Parameter Optimizer — SMC-AI-GOLD-SENTINEL  (FAST edition)
===========================================================
50-100× faster than the naive approach by separating work into three phases:

  Phase 1  Pre-compute features for every test bar ONCE (shared across all combos)
  Phase 2  Train + score ONCE per unique (LOOKAHEAD, TP_ATR, SL_ATR) config (12 configs)
  Phase 3  Sweep BUY/SELL thresholds on cached probabilities (near-instant)

Original: 108 combos × 26 folds × feature_computation  ≈ 324 hours
Optimised:  12 unique train configs × 26 folds + threshold sweep  ≈  4-8 hours

Usage
-----
    python optimize.py                      # default grid (~8 h)
    python optimize.py --total 8000         # faster (~4 h, fewer folds)
    python optimize.py --csv gold_m5.csv    # offline — no MT5 needed
    python optimize.py --quick              # smaller grid

Output
------
    optimize_results.csv    — every combo ranked by PF
    config_optimized.json   — best parameters, ready to paste into gold_bot.py
"""

import argparse, os, sys, json, warnings, itertools, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BOT_DIR)

# ─── PARAMETER GRID ────────────────────────────────────────────────────────────
# Split into TRAIN params (affect model) and FILTER params (threshold only).
# Only TRAIN combos require retraining; FILTER combos are swept for free.

FULL_GRID = {
    "LOOKAHEAD_BARS":  [8, 12, 16],       # ─┐ affect training labels
    "TP_ATR_MULT":     [1.5, 2.0, 2.5, 3.0], # │ → unique train configs
    "SL_ATR_MULT":     [1.0, 1.5, 2.0],   # ─┘   = 3×4×3 = 36 configs
    "BUY_THRESHOLD":   [0.50, 0.55, 0.60, 0.65],  # ─┐ sweep on cached probs
    "SELL_THRESHOLD":  [0.40, 0.45, 0.50, 0.55],  # ─┘ = 4×4 = 16 combos each
}

QUICK_GRID = {
    "LOOKAHEAD_BARS":  [8, 12],
    "TP_ATR_MULT":     [1.5, 2.0, 2.5],
    "SL_ATR_MULT":     [1.0, 1.5],
    "BUY_THRESHOLD":   [0.50, 0.55, 0.60],
    "SELL_THRESHOLD":  [0.40, 0.45, 0.50],
}

TRAIN_KEYS  = ["LOOKAHEAD_BARS", "TP_ATR_MULT", "SL_ATR_MULT"]
FILTER_KEYS = ["BUY_THRESHOLD", "SELL_THRESHOLD"]

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
    print(f"❌  Cannot import gold_bot.py: {e}")
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
    sl_dist = max(atr_val * sl_mult, SL_MIN_PIPS * GOLD_PIP)
    tp_dist = sl_dist * (tp_mult / sl_mult)
    if direction == "buy":
        return price - sl_dist, price + tp_dist
    return price + sl_dist, price - tp_dist


def profit_factor(trades):
    wins   = sum(t["pnl_pips"] for t in trades if t["pnl_pips"] > 0)
    losses = abs(sum(t["pnl_pips"] for t in trades if t["pnl_pips"] < 0))
    return wins / losses if losses > 0 else (wins if wins > 0 else 0.0)


def fmt_time(seconds):
    if seconds < 60:   return f"{seconds:.0f}s"
    if seconds < 3600: return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


# ─── PHASE 1: Pre-compute features for ALL test bars (once) ────────────────────

def precompute_features(full_m5, m15_full, h1_full, fold_ranges):
    """
    Compute SMC features for every bar that appears in any test window.
    This is the expensive step but runs only ONCE regardless of combo count.
    Returns dict: bar_abs_idx → {"feats": dict, "atr": float, "open_next": float} | None
    """
    test_bars = sorted({b for (_, te, ts_end) in fold_ranges
                        for b in range(te, ts_end - 1)})
    n = len(test_bars)
    print(f"\n  Phase 1 — Pre-computing features for {n} bars "
          f"(runs once, shared across all combos)...")

    t0      = time.time()
    cache   = {}
    n_fail  = 0

    for i, bar_abs in enumerate(test_bars):
        # Progress + ETA
        if i > 0 and i % 50 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / i * (n - i)
            print(f"\r    {i}/{n} bars  |  {fmt_time(elapsed)} elapsed  "
                  f"|  ETA {fmt_time(eta)}   ", end="", flush=True)

        if bar_abs + 1 >= len(full_m5):
            cache[bar_abs] = None
            continue

        ctx_start = max(0, bar_abs - CONTEXT_BARS + 1)
        ctx_m5    = full_m5.iloc[ctx_start:bar_abs + 1].reset_index(drop=True)
        if len(ctx_m5) < 50:
            cache[bar_abs] = None
            continue

        bar_time = ctx_m5["time"].iloc[-1] if "time" in ctx_m5.columns else None
        mtf = build_mtf_dict(ctx_m5, m15_full, h1_full, bar_time) if bar_time else {}

        try:
            feats = generate_smc_features(ctx_m5, mtf, False, is_training=True)
            if not feats:
                cache[bar_abs] = None
                n_fail += 1
                continue
            atr_val   = _atr(ctx_m5)
            open_next = float(full_m5.iloc[bar_abs + 1]["open"])
            cache[bar_abs] = {"feats": feats, "atr": atr_val, "open_next": open_next}
        except Exception:
            cache[bar_abs] = None
            n_fail += 1

    elapsed = time.time() - t0
    ok = sum(1 for v in cache.values() if v is not None)
    print(f"\r    {n}/{n} bars  |  done in {fmt_time(elapsed)}  "
          f"|  {ok} ok  {n_fail} failed          ")
    return cache


# ─── PHASE 2: Train + score for each unique training config ────────────────────

def build_prob_cache(full_m5, fold_ranges, feature_cache, unique_configs):
    """
    For each unique (LOOKAHEAD, TP_ATR, SL_ATR) config:
      • Retrain models on each fold's training window
      • Score every test bar using pre-computed features → (buy_p, sell_p)
    Returns: {(config_key, fold_idx): [(bar_abs, buy_p, sell_p, atr, open_next), ...]}
    """
    n_configs = len(unique_configs)
    n_folds   = len(fold_ranges)
    prob_cache = {}
    t0 = time.time()

    print(f"\n  Phase 2 — Training + scoring "
          f"{n_configs} unique configs × {n_folds} folds...")

    orig_fn = _gb.create_entry_target  # save before patching

    for ci, cfg in enumerate(unique_configs):
        lookahead, tp_atr, sl_atr = cfg
        label = f"lookahead={lookahead} tp_atr={tp_atr} sl_atr={sl_atr}"
        print(f"\n  Config {ci+1}/{n_configs}: {label}")

        # Monkey-patch create_entry_target with this config's params
        def _patched(df, _la=lookahead, _tp=tp_atr, _sl=sl_atr,
                     _orig=orig_fn, **kw):
            return _orig(df, lookahead=_la, tp_atr=_tp, sl_atr=_sl)

        _gb.create_entry_target = _patched

        try:
            for fi, (train_start, train_end, test_end) in enumerate(fold_ranges):
                t_fold = time.time()

                # Train
                train_df = full_m5.iloc[train_start:train_end].reset_index(drop=True)
                try:
                    train_models(train_df)
                    _gb.AI_MODEL = True
                except Exception as e:
                    prob_cache[(cfg, fi)] = []
                    print(f"    fold {fi+1}: train failed ({e})")
                    continue

                model  = _gb.entry_model
                scaler = _gb.scaler_entry
                if model is None or not hasattr(scaler, "n_features_in_"):
                    prob_cache[(cfg, fi)] = []
                    continue

                n_feats = scaler.n_features_in_

                # Score test bars from pre-computed feature cache
                fold_scores = []
                for bar_abs in range(train_end, test_end - 1):
                    item = feature_cache.get(bar_abs)
                    if item is None:
                        continue
                    feats = item.get("feats")
                    if not feats:
                        continue
                    try:
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
                        fold_scores.append((
                            bar_abs, buy_p, sell_p,
                            item["atr"], item["open_next"],
                        ))
                    except Exception:
                        continue

                prob_cache[(cfg, fi)] = fold_scores
                elapsed_fold = time.time() - t_fold
                print(f"    fold {fi+1}/{n_folds}: {len(fold_scores)} bars scored "
                      f"in {fmt_time(elapsed_fold)}", flush=True)

        finally:
            _gb.create_entry_target = orig_fn  # always restore

    total = time.time() - t0
    print(f"\n  Phase 2 done in {fmt_time(total)}")
    return prob_cache


# ─── PHASE 3: Fast threshold sweep on cached probabilities ─────────────────────

def sweep_all_combos(full_m5, fold_ranges, prob_cache, grid):
    """
    Iterate every BUY_THRESHOLD × SELL_THRESHOLD combination against each
    unique training config's cached probabilities.  No feature computation here.
    """
    train_combos  = list(itertools.product(*[grid[k] for k in TRAIN_KEYS]))
    filter_combos = list(itertools.product(*[grid[k] for k in FILTER_KEYS]))

    all_combos   = list(itertools.product(train_combos, filter_combos))
    total_combos = len(all_combos)

    print(f"\n  Phase 3 — Sweeping {len(train_combos)} × {len(filter_combos)} = "
          f"{total_combos} parameter combinations (fast)...")

    results = []
    t0 = time.time()

    for ci, (train_cfg, filter_cfg) in enumerate(all_combos):
        lookahead, tp_atr, sl_atr = train_cfg
        buy_thresh, sell_thresh   = filter_cfg

        cfg_key = train_cfg
        all_trades = []

        for fi, (train_start, train_end, test_end) in enumerate(fold_ranges):
            fold_scores = prob_cache.get((cfg_key, fi), [])

            for (bar_abs, buy_p, sell_p, atr_val, open_next) in fold_scores:
                # Apply threshold + edge filter
                if buy_p >= buy_thresh and buy_p > sell_p and abs(buy_p - sell_p) >= 0.08:
                    signal = "buy"
                elif sell_p >= sell_thresh and sell_p > buy_p and abs(buy_p - sell_p) >= 0.08:
                    signal = "sell"
                else:
                    continue

                spread = SPREAD_PIPS * GOLD_PIP
                entry  = open_next + (spread if signal == "buy" else -spread)
                sl, tp = compute_sl_tp_atr(signal, entry, atr_val, sl_atr, tp_atr)
                if abs(entry - sl) / GOLD_PIP < SL_MIN_PIPS:
                    continue

                f_start = bar_abs + 1
                f_end   = min(f_start + MAX_HOLD, len(full_m5))
                future  = full_m5.iloc[f_start:f_end].reset_index(drop=True)
                if len(future) == 0:
                    continue

                result = simulate_trade(signal, entry, sl, tp, future)
                all_trades.append(result)

        if ci % 20 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (ci + 1) * (total_combos - ci - 1) if ci > 0 else 0
            print(f"\r    {ci+1}/{total_combos}  |  ETA {fmt_time(eta)}   ",
                  end="", flush=True)

        if not all_trades:
            results.append({
                "LOOKAHEAD_BARS": lookahead, "TP_ATR_MULT": tp_atr,
                "SL_ATR_MULT": sl_atr, "BUY_THRESHOLD": buy_thresh,
                "SELL_THRESHOLD": sell_thresh,
                "trades": 0, "win_rate": 0, "profit_factor": 0,
                "avg_r": 0, "total_pips": 0,
            })
            continue

        pnl  = [t["pnl_pips"]  for t in all_trades]
        rs   = [t["r_multiple"] for t in all_trades]
        wins = sum(1 for p in pnl if p > 0)
        pf   = profit_factor(all_trades)
        wr   = wins / len(pnl) * 100
        results.append({
            "LOOKAHEAD_BARS": lookahead, "TP_ATR_MULT": tp_atr,
            "SL_ATR_MULT": sl_atr, "BUY_THRESHOLD": buy_thresh,
            "SELL_THRESHOLD": sell_thresh,
            "trades":        len(pnl),
            "win_rate":      round(wr, 1),
            "profit_factor": round(pf, 2),
            "avg_r":         round(float(np.mean(rs)), 3),
            "total_pips":    round(sum(pnl), 1),
        })

    elapsed = time.time() - t0
    print(f"\r    {total_combos}/{total_combos}  |  done in {fmt_time(elapsed)}          ")
    return results


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default=None,  help="Path to OHLCV CSV (offline mode)")
    parser.add_argument("--total", type=int, default=10000,
                        help="Bars to fetch from MT5 (default 10000)")
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller grid for faster testing")
    args = parser.parse_args()

    grid = QUICK_GRID if args.quick else FULL_GRID

    train_combos  = list(itertools.product(*[grid[k] for k in TRAIN_KEYS]))
    filter_combos = list(itertools.product(*[grid[k] for k in FILTER_KEYS]))
    total_combos  = len(train_combos) * len(filter_combos)

    print("\n" + "═" * 65)
    print("  SMC-AI Parameter Optimizer  (FAST edition)")
    print("═" * 65)
    print(f"  Unique train configs : {len(train_combos)}")
    print(f"  Threshold combos     : {len(filter_combos)}")
    print(f"  Total combinations   : {total_combos}")
    print(f"  Grid                 : {'QUICK' if args.quick else 'FULL'}")
    print("═" * 65)

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.csv:
        m5 = pd.read_csv(args.csv)
        m5["time"] = pd.to_datetime(m5["time"])
        m5 = m5.sort_values("time").reset_index(drop=True)
    else:
        if not _MT5_OK:
            print("❌  MetaTrader5 not installed. Use --csv to provide data.")
            return
        print("\n  Connecting to MT5...")
        import MetaTrader5 as mt5_mod
        if not mt5_mod.initialize():
            print("❌  MT5 not available"); return
        rates = mt5_mod.copy_rates_from_pos(SYMBOL, mt5_mod.TIMEFRAME_M5, 0, args.total)
        mt5_mod.shutdown()
        if rates is None:
            print("❌  No data returned"); return
        m5 = pd.DataFrame(rates)
        m5["time"] = pd.to_datetime(m5["time"], unit="s")

    m15_full = resample_ohlcv(m5, 15)
    h1_full  = resample_ohlcv(m5, 60)
    print(f"\n  Data loaded: {len(m5)} M5 bars | {len(m15_full)} M15 | {len(h1_full)} H1")

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
    print(f"  Folds: {n_folds}  |  {TRAIN_BARS} train + {TEST_BARS} test bars each")

    # ── Phase 1: Pre-compute features (once) ──────────────────────────────────
    t_total = time.time()
    feature_cache = precompute_features(m5, m15_full, h1_full, fold_ranges)

    # ── Phase 2: Train + score for each unique training config ─────────────────
    unique_train_configs = list(dict.fromkeys(
        (lh, tp, sl) for (lh, tp, sl) in train_combos
    ))
    prob_cache = build_prob_cache(m5, fold_ranges, feature_cache, unique_train_configs)

    # Free the feature cache — no longer needed
    del feature_cache

    # ── Phase 3: Threshold sweep ───────────────────────────────────────────────
    results = sweep_all_combos(m5, fold_ranges, prob_cache, grid)

    elapsed_total = time.time() - t_total
    print(f"\n  Total runtime: {fmt_time(elapsed_total)}")

    # ── Save full results ──────────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values("profit_factor", ascending=False)
    df.to_csv("optimize_results.csv", index=False)
    print(f"  Full results → optimize_results.csv")

    # ── Show top 5 ────────────────────────────────────────────────────────────
    valid = df[df["trades"] >= 20].copy()
    if valid.empty:
        valid = df.copy()

    valid["score"] = (valid["profit_factor"] * 0.5 +
                      valid["avg_r"]          * 0.3 +
                      valid["win_rate"] / 100  * 0.2)
    valid = valid.sort_values("score", ascending=False)

    print("\n" + "═" * 65)
    print("  TOP 5 PARAMETER COMBINATIONS")
    print("═" * 65)
    for _, row in valid.head(5).iterrows():
        params = "  ".join(f"{k}={row[k]}" for k in list(grid.keys()))
        print(f"  {params}")
        print(f"    → PF {row['profit_factor']:.2f}  WR {row['win_rate']:.1f}%  "
              f"AvgR {row['avg_r']:+.3f}  Trades {int(row['trades'])}")
        print()

    best = valid.iloc[0]

    print("═" * 65)
    print("  BEST PARAMETERS")
    print("═" * 65)
    for k in grid.keys():
        print(f"  {k:22s}: {best[k]}")
    print(f"\n  Profit Factor : {best['profit_factor']:.2f}")
    print(f"  Win Rate      : {best['win_rate']:.1f}%")
    print(f"  Avg R         : {best['avg_r']:+.3f}")
    print(f"  Trades        : {int(best['trades'])}")
    print("═" * 65)

    # ── Save best config ───────────────────────────────────────────────────────
    config = {k: float(best[k]) for k in grid.keys()}
    config["_note"] = (
        "Generated by optimize.py — paste BUY_THRESHOLD / SELL_THRESHOLD into "
        "gold_bot.py constants. Re-run monthly to refresh."
    )
    config["_generated"]     = pd.Timestamp.now().isoformat()
    config["_profit_factor"] = float(best["profit_factor"])
    config["_win_rate"]      = float(best["win_rate"])
    config["_trades_tested"] = int(best["trades"])

    with open("config_optimized.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  Best config → config_optimized.json")


if __name__ == "__main__":
    main()
