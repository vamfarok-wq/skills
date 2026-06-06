#!/usr/bin/env python3
"""Full V55 chain (velocity + SMC) using FarooqV55_fixed.py, 20k training, May 29–June 5."""
import sys, types, warnings; warnings.filterwarnings("ignore")

for m in ['MetaTrader5','telegram','telegram.ext','schedule','plyer','plyer.notification']:
    sys.modules[m] = types.ModuleType(m)
mt5 = sys.modules['MetaTrader5']
for a in ['TIMEFRAME_M5','TIMEFRAME_M15','TIMEFRAME_H1','TIMEFRAME_H4',
          'ORDER_TYPE_BUY','ORDER_TYPE_SELL','TRADE_RETCODE_DONE']:
    setattr(mt5, a, 0)
for fn in ['initialize','shutdown','copy_rates_from_pos','positions_get',
           'orders_get','account_info','symbol_info','symbol_info_tick',
           'order_send','last_error']:
    setattr(mt5, fn, lambda *a,**k: None)

import pandas as pd, numpy as np
sys.path.insert(0, '.')
import importlib.util

spec = importlib.util.spec_from_file_location("v55", '/home/user/skills/FarooqV55_fixed.py')
v55 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v55)

CSV        = '/root/.claude/uploads/de940fb2-f208-4dc2-9771-2522f01335e4/eb94ebd4-xauusd_m5_history.csv'
TRAIN_BARS = 20000
CONTEXT    = 200
SPREAD     = 0.25
RISK_PCT   = 0.01
EQUITY0    = 1000.0
MAX_HOLD   = 72
SL_MIN_PIP = 12.0
SL_MAX_PIP = 120.0
GOLD_PIP   = 0.1
SM_VEL     = 2.5
SM_BIAS    = 1.5

df = pd.read_csv(CSV)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

train_end  = df[df['time'] < '2026-05-29'].index[-1] + 1
test_start = train_end

print(f"Total bars before May 29 : {train_end}")
print(f"Training on last {TRAIN_BARS} bars  [{train_end-TRAIN_BARS} → {train_end}]")
print(f"Test window : {df.iloc[test_start]['time'].date()} → {df.iloc[-1]['time'].date()}  ({len(df)-test_start} bars)\n")

train_df = df.iloc[train_end-TRAIN_BARS : train_end].copy().reset_index(drop=True)
v55.train_models(train_df)
print(f"\nAI ready: {v55.entry_model is not None}  features: {len(v55.FEATURE_NAMES)}\n")

# ── MTF proxy (EMA-based, same as V55) ───────────────────────────────────────
def build_mtf(ctx):
    c   = ctx['close']
    e9  = c.ewm(span=9 ).mean().iloc[-1]
    e21 = c.ewm(span=21).mean().iloc[-1]
    e50 = c.ewm(span=50).mean().iloc[-1]
    e99 = c.ewm(span=99).mean().iloc[-1]
    m15 = "bullish" if e21>e50 else "bearish" if e21<e50 else "range"
    h1  = "bullish" if e50>e99 else "bearish" if e50<e99 else "range"
    return {"M15_trend": m15, "H1_trend": h1, "trend_score": 0.5}

def get_ai_probs(ctx, mtf):
    buy_p = sell_p = 0.5
    try:
        feats = v55.generate_smc_features(ctx, mtf, False, is_training=True)
        if feats:
            vec = np.array([feats.get(n, 0.0) for n in v55.FEATURE_NAMES]).reshape(1,-1)
            if vec.shape[1] == v55.scaler_entry.n_features_in_:
                Xs    = v55.scaler_entry.transform(vec)
                probs = v55.entry_model.predict_proba(Xs)[0]
                for i, cl in enumerate(v55.entry_model.classes_):
                    if cl ==  1: buy_p  = float(probs[i])
                    if cl == -1: sell_p = float(probs[i])
    except: pass
    return buy_p, sell_p

def atr14(ctx):
    return float((ctx['high'] - ctx['low']).rolling(14).mean().iloc[-1])

# ── SMC chain decision (mirrors should_enter_trade logic) ─────────────────────
def chain_decision(ctx, mtf, buy_p, sell_p):
    """Returns (direction, signal_type) or (None, reason)."""
    try:
        feats = v55.generate_smc_features(ctx, mtf, False, is_training=True)
        if not feats:
            return None, 'no_feats'
    except:
        return None, 'feat_err'

    ai_prob = max(buy_p, sell_p)

    # ── hard edge filter ──
    if abs(buy_p - sell_p) < 0.08:
        return None, 'no_edge'

    # ── weak AI filter ──
    if buy_p > sell_p and buy_p < 0.55:
        return None, 'weak_ai'
    if sell_p > buy_p and sell_p < 0.45:
        return None, 'weak_ai'

    m5t  = v55.detect_market_regime(ctx)
    m15t = mtf['M15_trend']
    h1t  = mtf['H1_trend']

    # ── BUY SIDE ─────────────────────────────────────────────────────────────
    if buy_p > sell_p:
        if m5t not in ['bullish', 'range', None]:
            return None, 'trend_mismatch'
        if m15t == 'bearish':
            if not (ai_prob >= 0.78 and h1t == 'bullish'):
                return None, 'm15_conflict'

        if feats.get('bullish_reversal_setup', 0) == 1:
            return 'BUY', 'smc_reversal'

        min_conf = 1.5 if ai_prob > 0.72 else 2.5
        if feats.get('bullish_confluence', 0) >= min_conf and ai_prob > 0.70:
            return 'BUY', 'confluence'

        if feats.get('choch_bull', 0) == 1.0 and ai_prob > 0.65:
            return 'BUY', 'choch'

        min_bias = 0.8 if ai_prob > 0.72 else 1.5
        if feats.get('net_bias', 0) > min_bias and ai_prob > 0.60:
            return 'BUY', 'momentum_bias'

        return None, 'no_conf_buy'

    # ── SELL SIDE ────────────────────────────────────────────────────────────
    elif sell_p > buy_p:
        if m5t not in ['bearish', 'range', None]:
            return None, 'trend_mismatch'
        if m15t == 'bullish':
            if not (ai_prob >= 0.78 and h1t == 'bearish'):
                return None, 'm15_conflict'

        if feats.get('bearish_reversal_setup', 0) == 1:
            return 'SELL', 'smc_reversal'

        min_conf = 1.5 if ai_prob > 0.72 else 2.5
        if feats.get('bearish_confluence', 0) >= min_conf and ai_prob > 0.70:
            return 'SELL', 'confluence'

        if feats.get('choch_bear', 0) == 1.0 and ai_prob > 0.65:
            return 'SELL', 'choch'

        min_bias = 0.8 if ai_prob > 0.72 else 1.5
        if feats.get('net_bias', 0) < -min_bias and ai_prob > 0.60:
            return 'SELL', 'momentum_bias'

        return None, 'no_conf_sell'

    return None, 'no_signal'

# ── Backtest ──────────────────────────────────────────────────────────────────
print("="*65)
print("  BACKTEST  May 29 – June 5, 2026  (full V55 chain — fixed)")
print("="*65)

trades = []; equity = EQUITY0; open_pos = []; last_entry = -999
skips  = {}

for idx in range(test_start + CONTEXT, len(df) - 1):
    bar = df.iloc[idx]; ctx = df.iloc[idx-CONTEXT:idx].copy(); bt = bar['time']

    # ── Close open positions ──────────────────────────────────────────────────
    for p in list(open_pos):
        hi = float(bar['high']); lo = float(bar['low'])
        if p['dir'] == 'buy':
            if   lo <= p['sl']: pnl = (p['sl'] - p['entry']) * p['size']; res = 'sl'
            elif hi >= p['tp']: pnl = (p['tp'] - p['entry']) * p['size']; res = 'tp'
            else:
                p['bars'] += 1
                if p['bars'] < MAX_HOLD: continue
                pnl = (float(bar['close']) - p['entry']) * p['size']; res = 'timeout'
        else:
            if   hi >= p['sl']: pnl = (p['entry'] - p['sl']) * p['size']; res = 'sl'
            elif lo <= p['tp']: pnl = (p['entry'] - p['tp']) * p['size']; res = 'tp'
            else:
                p['bars'] += 1
                if p['bars'] < MAX_HOLD: continue
                pnl = (p['entry'] - float(bar['close'])) * p['size']; res = 'timeout'
        equity += pnl
        trades.append({**p, 'pnl': round(pnl,2), 'result': res,
                       'close_time': bt, 'equity': round(equity,2)})
        open_pos.remove(p)

    if idx - last_entry < 2:       continue
    if len(open_pos) >= 2:         continue
    h = bt.hour
    if not((7<=h<12)or(13<=h<21)):
        skips['session'] = skips.get('session', 0) + 1; continue

    mtf    = build_mtf(ctx)
    m5t    = v55.detect_market_regime(ctx)
    buy_p, sell_p = get_ai_probs(ctx, mtf)

    # ── Path 1: strong_momentum (velocity) ───────────────────────────────────
    direction = None; signal_type = None
    atr15 = float((ctx['high'] - ctx['low']).tail(15).mean())
    if atr15 > 0:
        vel3   = (float(ctx['close'].iloc[-1]) - float(ctx['close'].iloc[-4]))  / atr15
        bias12 = (float(ctx['close'].iloc[-1]) - float(ctx['close'].iloc[-13])) / atr15
        m15t   = mtf['M15_trend']; h1t = mtf['H1_trend']
        all_bull = (m5t=='bullish' and m15t=='bullish' and h1t=='bullish')
        all_bear = (m5t=='bearish' and m15t=='bearish' and h1t=='bearish')
        if all_bull and vel3 > SM_VEL and bias12 > SM_BIAS and buy_p > sell_p:
            direction = 'BUY';  signal_type = 'strong_momentum'
        elif all_bear and vel3 < -SM_VEL and bias12 < -SM_BIAS and sell_p > buy_p:
            direction = 'SELL'; signal_type = 'strong_momentum'
    else:
        vel3 = bias12 = 0.0

    # ── Path 2: full SMC chain ────────────────────────────────────────────────
    if direction is None:
        direction, signal_type = chain_decision(ctx, mtf, buy_p, sell_p)
        if direction is None:
            skips[signal_type] = skips.get(signal_type, 0) + 1
            continue

    if any(p['dir'] == direction.lower() for p in open_pos): continue

    # ── SL / TP ───────────────────────────────────────────────────────────────
    _atr = atr14(ctx)
    nb   = df.iloc[idx+1]; d = direction.lower()
    entry_p = float(nb['open']) + (SPREAD if d=='buy' else -SPREAD)
    sl = entry_p - _atr*0.6 if d=='buy' else entry_p + _atr*0.6
    tp = entry_p + _atr*1.5 if d=='buy' else entry_p - _atr*1.5

    sl_pips = abs(entry_p - sl) / GOLD_PIP
    if sl_pips < SL_MIN_PIP or sl_pips > SL_MAX_PIP:
        skips['sl_range'] = skips.get('sl_range', 0) + 1; continue

    size = (equity * RISK_PCT) / max(abs(entry_p - sl), 1e-9)
    open_pos.append({'dir': d, 'entry': entry_p, 'sl': sl, 'tp': tp, 'size': size,
                     'bars': 0, 'open_time': bt, 'signal': signal_type,
                     'buy_p': round(buy_p,3), 'sell_p': round(sell_p,3),
                     'vel3': round(vel3,2), 'bias12': round(bias12,2)})
    last_entry = idx

# close remaining
for p in open_pos:
    lp  = float(df['close'].iloc[-1])
    pnl = (lp - p['entry'])*p['size'] if p['dir']=='buy' else (p['entry']-lp)*p['size']
    trades.append({**p, 'pnl': round(pnl,2), 'result': 'end',
                   'close_time': df['time'].iloc[-1], 'equity': round(equity+pnl,2)})
    equity += pnl

# ── Results ───────────────────────────────────────────────────────────────────
print()
if not trades:
    print("  No trades fired.")
    print(f"  Skips: {skips}")
else:
    tdf   = pd.DataFrame(trades)
    wins  = tdf[tdf.pnl > 0]; losses = tdf[tdf.pnl <= 0]
    buys  = tdf[tdf.dir=='buy'];  sells = tdf[tdf.dir=='sell']
    gw    = wins.pnl.sum() if len(wins) else 0
    gl    = abs(losses.pnl.sum()) if len(losses) else 1e-9

    print(f"  Trades      :  {len(tdf)}")
    print(f"  Wins        :  {len(wins)}")
    print(f"  Losses      :  {len(losses)}")
    print(f"  Win rate    :  {100*len(wins)/len(tdf):.1f}%")
    print(f"  Profit fct  :  {gw/gl:.2f}")
    print(f"  Net P&L     :  ${tdf.pnl.sum():.2f}")
    print(f"  $1000 → ${equity:.2f}")
    print(f"  BUY / SELL  :  {len(buys)} / {len(sells)}")
    print()

    # signal type breakdown
    for st in tdf.signal.unique():
        sub = tdf[tdf.signal==st]
        w   = (sub.pnl > 0).sum()
        print(f"  [{st:<18}]  {len(sub):>3} trades  {w}/{len(sub)} wins  "
              f"P&L ${sub.pnl.sum():>+7.2f}")
    print()

    print(f"  {'Time':<22} {'Dir':<5} {'Signal':<18} {'Entry':>8} {'SL':>8} {'TP':>8} "
          f"{'buy_p':>6} {'sell_p':>7} {'Result':<8} {'P&L':>8}")
    print("  " + "-"*105)
    for _, r in tdf.iterrows():
        print(f"  {str(r.open_time):<22} {r.dir.upper():<5} {r.signal:<18} "
              f"{r.entry:>8.2f} {r.sl:>8.2f} {r.tp:>8.2f} "
              f"{r.buy_p:>6.3f} {r.sell_p:>7.3f} {r.result:<8} {r.pnl:>+8.2f}")
    print()
    print(f"  Skips: {skips}")
