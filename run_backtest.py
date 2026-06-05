#!/usr/bin/env python3
"""Thin wrapper: stubs MetaTrader5 so backtest.py runs without MT5 installed."""
import sys, types, warnings
warnings.filterwarnings("ignore")

# ── Stub MetaTrader5 and optional deps ────────────────────────────────────────
for _mod in ['MetaTrader5', 'telegram', 'telegram.ext',
             'schedule', 'plyer', 'plyer.notification']:
    _m = types.ModuleType(_mod)
    sys.modules[_mod] = _m

_mt5 = sys.modules['MetaTrader5']
for _a in ['TIMEFRAME_M5', 'TIMEFRAME_M15', 'TIMEFRAME_H1', 'TIMEFRAME_H4',
           'ORDER_TYPE_BUY', 'ORDER_TYPE_SELL', 'TRADE_RETCODE_DONE']:
    setattr(_mt5, _a, 0)
for _fn in ['initialize', 'shutdown', 'copy_rates_from_pos', 'positions_get',
            'orders_get', 'account_info', 'symbol_info', 'symbol_info_tick',
            'order_send', 'last_error']:
    setattr(_mt5, _fn, lambda *a, **k: None)

# ── Now run backtest with remaining argv ──────────────────────────────────────
import runpy, os
sys.path.insert(0, os.path.dirname(__file__))
runpy.run_path('backtest.py', run_name='__main__')
