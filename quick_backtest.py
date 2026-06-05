#!/usr/bin/env python3
"""
Quick walk-forward style backtest on the uploaded M5 CSV.
Trains once on first TRAIN_BARS, tests on the rest.
Reports: win rate, profit factor, total trades, BUY/SELL split,
         label distribution (old vs new), sweep accuracy.
"""
import sys, types, warnings
warnings.filterwarnings("ignore")

# ── Stub MetaTrader5 and optional deps ────────────────────────────────────────
for _mod in ['MetaTrader5','telegram','telegram.ext','schedule',
             'plyer','plyer.notification']:
    sys.modules[_mod] = types.ModuleType(_mod)
_mt5 = sys.modules['MetaTrader5']
for _a in ['TIMEFRAME_M5','TIMEFRAME_M15','TIMEFRAME_H1','TIMEFRAME_H4',
           'ORDER_TYPE_BUY','ORDER_TYPE_SELL','TRADE_RETCODE_DONE']:
    setattr(_mt5, _a, 0)
for _fn in ['initialize','shutdown','copy_rates_from_pos','positions_get',
            'orders_get','account_info','symbol_info','symbol_info_tick',
            'order_send','last_error']:
    setattr(_mt5, _fn, lambda *a,**k: None)

import pandas as pd, numpy as np
sys.path.insert(0, '.')
import gold_bot as _gb
from gold_bot import (
    generate_smc_features, train_models, detect_liquidity_sweep,
    detect_liquidity_pools, detect_fvg, detect_order_blocks,
    candle_pattern_veto, calculate_liquidity_based_sl,
    calculate_liquidity_based_tp, FEATURE_NAMES,
    create_entry_target, BUY_THRESHOLD, SELL_THRESHOLD
)
from tqdm import tqdm

CSV        = '/root/.claude/uploads/de940fb2-f208-4dc2-9771-2522f01335e4/eb94ebd4-xauusd_m5_history.csv'
TRAIN_BARS = 2000
CONTEXT    = 200     # bars fed to signal function per bar
SPREAD     = 0.25    # $0.25 per unit (2.5 pips for gold)
RISK_PCT   = 0.01
EQUITY0    = 1000.0
MAX_HOLD   = 72
SL_MIN_PIPS = 12.0
SL_MAX_PIPS = 120.0
GOLD_PIP    = 0.1

# ─────────────────────────────────────────────────────────────────────────────
print("Loading CSV …")
df = pd.read_csv(CSV)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
print(f"  {len(df)} bars  {df.time.iloc[0].date()} → {df.time.iloc[-1].date()}")
print(f"  Price: {df.close.min():.2f} → {df.close.max():.2f}")

# ─── Label distribution: old vs new ──────────────────────────────────────────
print("\n=== LABEL BIAS: OLD vs NEW ===")
sample5k = df.iloc[:TRAIN_BARS].copy()

old_lbl, _ = create_entry_target(sample5k, lookahead=24, tp_atr=2.0, sl_atr=1.0, detrend_span=999999)
ov = old_lbl.value_counts()
bo, so = ov.get(1,0), ov.get(-1,0)
print(f"  OLD  tp=2.0 sl=1.0 raw:   BUY={bo:5d}  SELL={so:5d}  ratio={bo/(so+1e-9):.2f}x  ← BUY {'over' if bo>so else 'UNDER'}weighted")

new_lbl, _ = create_entry_target(sample5k, lookahead=24, tp_atr=1.5, sl_atr=1.5, detrend_span=100)
nv = new_lbl.value_counts()
bn, sn = nv.get(1,0), nv.get(-1,0)
print(f"  NEW  tp=1.5 sl=1.5 detr:  BUY={bn:5d}  SELL={sn:5d}  ratio={bn/(sn+1e-9):.2f}x  ← {'BALANCED ✓' if 0.7<bn/(sn+1e-9)<1.4 else 'still skewed'}")

# ─── Sweep accuracy ───────────────────────────────────────────────────────────
print("\n=== SWEEP ACCURACY (next-5-bar) ===")
pb=ps=tb=ts=0
for i in range(100, min(len(df)-5, 10000), 5):
    ctx = df.iloc[i-100:i].copy()
    sw  = detect_liquidity_sweep(ctx)
    fh  = df['high'].iloc[i:i+5].max()
    fl  = df['low'].iloc[i:i+5].min()
    cp  = float(df['close'].iloc[i-1])
    if sw=='buy_sweep':
        tb+=1
        if fh>cp: pb+=1
    elif sw=='sell_sweep':
        ts+=1
        if fl<cp: ps+=1
ba = 100*pb/tb if tb>0 else 0
sa = 100*ps/ts if ts>0 else 0
print(f"  buy_sweep  → price UP   in 5 bars: {ba:.1f}%  ({pb}/{tb})")
print(f"  sell_sweep → price DOWN in 5 bars: {sa:.1f}%  ({ps}/{ts})")
verdict = "KEEP ✓" if (ba>50 and sa>50) else "REMOVE ✗"
print(f"  Verdict: {verdict}")

# ─── Build M15 / H1 proxies ───────────────────────────────────────────────────
def build_mtf(m5_slice):
    c = m5_slice['close']
    e20 = c.ewm(span=20).mean().iloc[-1]
    e50 = c.ewm(span=50).mean().iloc[-1]
    e99 = c.ewm(span=99).mean().iloc[-1]
    m15 = "bullish" if e20>e50 else "bearish" if e20<e50 else "range"
    h1  = "bullish" if e50>e99 else "bearish" if e50<e99 else "range"
    return {"M15_trend": m15, "H1_trend": h1, "trend_score": 0.5}

# ─── ATR helper ───────────────────────────────────────────────────────────────
def atr(df_s): return float((df_s['high']-df_s['low']).rolling(14).mean().iloc[-1])

# ─── Train on first TRAIN_BARS ────────────────────────────────────────────────
print(f"\n=== TRAINING on first {TRAIN_BARS} bars ===")
train_df = df.iloc[:TRAIN_BARS].copy().reset_index(drop=True)
if 'time' not in train_df.columns:
    train_df['time'] = pd.date_range('2020-01-01', periods=len(train_df), freq='5min')
train_models(train_df)

ai_ok = (_gb.entry_model is not None)
print(f"  AI model ready: {ai_ok}")

# ─── Walk-forward test on remaining bars ─────────────────────────────────────
print(f"\n=== BACKTEST: bars {TRAIN_BARS}–{len(df)} ===")
start = TRAIN_BARS + CONTEXT
trades = []
equity = EQUITY0
open_pos = []    # list of dicts
last_entry_idx = -999

stats_sweep   = {'none':0,'buy_sweep':0,'sell_sweep':0}
stats_skip    = {}

for idx in tqdm(range(start, len(df)-1), desc="Backtesting"):

    bar = df.iloc[idx]
    ctx = df.iloc[idx-CONTEXT:idx].copy()
    bar_time = bar['time']

    # ── Update open positions ────────────────────────────────────────────────
    for p in list(open_pos):
        hi = float(bar['high']); lo = float(bar['low'])
        if p['dir'] == 'buy':
            if lo <= p['sl']:
                pnl = (p['sl'] - p['entry']) * p['size']; p['result']='sl'
            elif hi >= p['tp']:
                pnl = (p['tp'] - p['entry']) * p['size']; p['result']='tp'
            else:
                p['bars'] += 1
                if p['bars'] >= MAX_HOLD:
                    pnl = (float(bar['close']) - p['entry']) * p['size']; p['result']='timeout'
                else:
                    continue
        else:  # sell
            if hi >= p['sl']:
                pnl = (p['entry'] - p['sl']) * p['size']; p['result']='sl'
            elif lo <= p['tp']:
                pnl = (p['entry'] - p['tp']) * p['size']; p['result']='tp'
            else:
                p['bars'] += 1
                if p['bars'] >= MAX_HOLD:
                    pnl = (p['entry'] - float(bar['close'])) * p['size']; p['result']='timeout'
                else:
                    continue
        equity += pnl
        trades.append({**p, 'pnl': round(pnl,2), 'close_idx': idx,
                       'close_time': bar_time, 'equity': round(equity,2)})
        open_pos.remove(p)

    # ── Cooldown ─────────────────────────────────────────────────────────────
    if idx - last_entry_idx < 2:
        continue
    if len(open_pos) >= 2:
        continue

    # ── Session filter (London / NY UTC) ─────────────────────────────────────
    h = bar_time.hour
    if not ((7 <= h < 12) or (13 <= h < 21)):
        continue

    # ── Sweep decides direction ───────────────────────────────────────────────
    sw = detect_liquidity_sweep(ctx)
    stats_sweep[sw if sw else 'none'] += 1
    if sw is None:
        continue
    sweep_dir = 'BUY' if sw=='buy_sweep' else 'SELL'

    # ── AI signal (quality check only, not direction) ─────────────────────────
    buy_p = sell_p = 0.5
    if ai_ok and _gb.entry_model is not None:
        try:
            mtf = build_mtf(ctx)
            feats = generate_smc_features(ctx, mtf, False, is_training=True)
            if feats:
                vec = np.array([feats.get(n,0.0) for n in FEATURE_NAMES]).reshape(1,-1)
                if vec.shape[1] == _gb.scaler_entry.n_features_in_:
                    Xs = _gb.scaler_entry.transform(vec)
                    probs = _gb.entry_model.predict_proba(Xs)[0]
                    for i,c in enumerate(_gb.entry_model.classes_):
                        if c== 1: buy_p  = float(probs[i])
                        if c==-1: sell_p = float(probs[i])
        except Exception:
            pass

    # AI quality gate: direction's probability ≥ 0.25
    if sweep_dir=='BUY'  and buy_p  < 0.25: stats_skip['ai_weak']=stats_skip.get('ai_weak',0)+1; continue
    if sweep_dir=='SELL' and sell_p < 0.25: stats_skip['ai_weak']=stats_skip.get('ai_weak',0)+1; continue

    # Edge: skip coin-flip
    if abs(buy_p - sell_p) < 0.08: stats_skip['no_edge']=stats_skip.get('no_edge',0)+1; continue

    # ── MTF trend must agree with sweep ──────────────────────────────────────
    mtf = build_mtf(ctx)
    m15 = mtf.get('M15_trend'); h1 = mtf.get('H1_trend')
    if sweep_dir=='BUY'  and (m15=='bearish' or h1=='bearish'): stats_skip['mtf_mismatch']=stats_skip.get('mtf_mismatch',0)+1; continue
    if sweep_dir=='SELL' and (m15=='bullish' or h1=='bullish'): stats_skip['mtf_mismatch']=stats_skip.get('mtf_mismatch',0)+1; continue

    # ── Duplicate direction check ─────────────────────────────────────────────
    if any(p['dir']==sweep_dir.lower() for p in open_pos): continue

    # ── Momentum gate ─────────────────────────────────────────────────────────
    _atr = atr(ctx)
    net5 = float(ctx['close'].iloc[-1] - ctx['open'].iloc[-5])
    if sweep_dir=='BUY'  and net5 < -_atr*1.5: stats_skip['momentum']=stats_skip.get('momentum',0)+1; continue
    if sweep_dir=='SELL' and net5 >  _atr*1.5: stats_skip['momentum']=stats_skip.get('momentum',0)+1; continue

    # ── Compute SL/TP ─────────────────────────────────────────────────────────
    next_bar = df.iloc[idx+1]
    direction = sweep_dir.lower()
    entry_p = float(next_bar['open']) + (SPREAD if direction=='buy' else -SPREAD)

    try:
        sl = calculate_liquidity_based_sl(direction, entry_p, ctx, _atr)
        tp = calculate_liquidity_based_tp(direction, entry_p, ctx, _atr, sl)
    except Exception:
        sl = entry_p - _atr*1.5 if direction=='buy' else entry_p + _atr*1.5
        tp = entry_p + _atr*2.0 if direction=='buy' else entry_p - _atr*2.0

    sl_dist = abs(entry_p - sl)
    sl_pips = sl_dist / GOLD_PIP
    if sl_pips < SL_MIN_PIPS or sl_pips > SL_MAX_PIPS: stats_skip['sl_range']=stats_skip.get('sl_range',0)+1; continue
    if sl_dist > _atr*3.0: stats_skip['sl_wide']=stats_skip.get('sl_wide',0)+1; continue

    # ── Position sizing ───────────────────────────────────────────────────────
    risk_amt = equity * RISK_PCT
    size = risk_amt / max(sl_dist, 1e-9)

    open_pos.append({
        'dir': direction, 'entry': entry_p, 'sl': sl, 'tp': tp,
        'size': size, 'bars': 0, 'open_idx': idx, 'open_time': bar_time,
        'buy_p': round(buy_p,3), 'sell_p': round(sell_p,3)
    })
    last_entry_idx = idx

# ── Close any still-open at end ───────────────────────────────────────────────
last_close = float(df['close'].iloc[-1])
for p in open_pos:
    pnl = (last_close - p['entry'])*p['size'] if p['dir']=='buy' else (p['entry']-last_close)*p['size']
    trades.append({**p,'pnl':round(pnl,2),'result':'end','close_idx':len(df)-1,
                   'close_time':df['time'].iloc[-1],'equity':round(equity+pnl,2)})
    equity += pnl

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  FINAL BACKTEST RESULTS")
print("="*60)
if not trades:
    print("  ⚠ No trades generated")
else:
    tdf = pd.DataFrame(trades)
    wins  = tdf[tdf.pnl>0]; losses = tdf[tdf.pnl<=0]
    buys  = tdf[tdf.dir=='buy'];  sells = tdf[tdf.dir=='sell']

    gross_win  = wins.pnl.sum()  if len(wins)  else 0
    gross_loss = abs(losses.pnl.sum()) if len(losses) else 1e-9
    pf = gross_win / gross_loss

    print(f"  Trades total : {len(tdf)}")
    print(f"  Win rate     : {100*len(wins)/len(tdf):.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Net P&L      : ${tdf.pnl.sum():.2f}  (start ${EQUITY0:.0f} → end ${equity:.2f})")
    print(f"  BUY  trades  : {len(buys)}  ({100*len(buys)/len(tdf):.1f}%)")
    print(f"  SELL trades  : {len(sells)}  ({100*len(sells)/len(tdf):.1f}%)")
    print()
    print(f"  Exit breakdown:")
    for r, cnt in tdf.result.value_counts().items():
        pnl_r = tdf[tdf.result==r].pnl.sum()
        print(f"    {r:10s}: {cnt:4d}  P&L ${pnl_r:.2f}")
    print()
    print(f"  Sweep stats : {stats_sweep}")
    print(f"  Skip reasons: {stats_skip}")
    print()
    tdf.to_csv('backtest_trades_csv.csv', index=False)
    print("  Saved: backtest_trades_csv.csv")
