#!/usr/bin/env python3
# ================= SMC-AI-GOLD-SENTINEL V40 =================
# Enhanced M5 Gold Trading Bot with Smart Money Concepts (SMC)
# Features: Fair Value Gaps, Order Blocks, Liquidity Pools, Multi-Timeframe Analysis
# AI-Enhanced Decision Making with 80-90% Win Rate Target
# ================= VERSION 4.0 - ULTIMATE SMC EDITION =================

import os
import sys
import json
import pickle
import threading
import time
import math
import requests
import joblib
import warnings
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import csv
import xml.etree.ElementTree as ET
import pytz
import xgboost as xgb


warnings.filterwarnings("ignore")

sys.stdout.reconfigure(line_buffering=True)

from datetime import datetime, timezone, timedelta
from collections import deque
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
try:
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder as _LabelEncoder

    class _XGBWrapper:
        """XGBClassifier that accepts arbitrary class labels (-1, 0, 1).

        XGBoost 3.x requires labels in [0, N-1]. This wrapper remaps them
        transparently so the rest of the codebase keeps using -1/0/1 labels
        and the existing classes_-based probability reading stays unchanged.
        """
        def __init__(self, **kwargs):
            self._clf     = xgb.XGBClassifier(**kwargs)
            self._le      = _LabelEncoder()
            self.classes_ = None

        def fit(self, X, y, sample_weight=None):
            y_enc = self._le.fit_transform(y)          # [-1,0,1] → [0,1,2]
            self._clf.fit(X, y_enc, sample_weight=sample_weight)
            self.classes_ = self._le.classes_           # restore original labels
            return self

        def predict_proba(self, X):
            # Ordering matches self.classes_ (LabelEncoder preserves sort order)
            return self._clf.predict_proba(X)

        def predict(self, X):
            return self._le.inverse_transform(self._clf.predict(X))

    _XGB_OK = True
except ImportError:
    _XGB_OK = False
from dotenv import load_dotenv
load_dotenv()

# Regime detection (Tier-1 upgrade)
try:
    from regime_detector import (
        compute_regime_features,
        detect_regime,
        regime_allows_trade,
        get_regime_confidence_multiplier,
    )
    REGIME_DETECTOR_OK = True
except ImportError:
    REGIME_DETECTOR_OK = False
    def compute_regime_features(df): return {}
    def detect_regime(df): return "ranging"
    def regime_allows_trade(r, s): return True, "no_regime_detector"
    def get_regime_confidence_multiplier(r): return 1.0
# ================= CONFIGURATION =================

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
TIMEFRAME_M15 = mt5.TIMEFRAME_M15
TIMEFRAME_H1 = mt5.TIMEFRAME_H1
TIMEFRAME_H4 = mt5.TIMEFRAME_H4

MT5_RECONNECT_ATTEMPTS = 5
MT5_RECONNECT_DELAY = 3
MT5_INITIALIZED = False
start_balance = 0.0

market_cache = None
market_cache_time = 0

MODEL_META = "model_meta.pkl"
RETRAIN_HOURS = 4
training_lock = False

BARS = 19000   # full history for AI training
LOOP_BARS = 200  # fast fetch for live loop — 200 M5 candles = ~16h, covers all SMC functions
BARS_M5  = 5000
BARS_M15 = 3000
BARS_H1 = 1500
BARS_H4 = 800

last_entry_time = 0
MIN_ENTRY_INTERVAL = 60

# Risk Management
RISK_PER_TRADE = 0.005   # 0.5% per trade — safe for M5 gold

MAX_SPREAD = 80  # Maximum spread in points (150 = 15 pips for XAUUSD)
MIN_SPREAD = 5

AI_ENTRY_THRESHOLD = 0.55  # must match BUY_THRESHOLD — both gate on 55% confidence
AI_EXIT_THRESHOLD  = 0.45
BUY_THRESHOLD  = 0.55   # AI needs >55% confidence for BUY
SELL_THRESHOLD = 0.45   # AI needs >55% confidence for SELL (1 - 0.55)

MAX_POSITIONS = 2
MIN_RR = 1.5

MAGIC_NUMBER = 20250317

PAUSE_REASON = "" 

MAX_DRAWDOWN_PERCENT = 30.0        # Stop/pause duration as a fraction of balance
MAX_CONSECUTIVE_LOSSES = 5         # Pause after 5 consecutive losses
DAILY_LOSS_LIMIT = 0.10            # Stop trading if daily losses exceed 10% of balance

# NY Session End Protection
# Block new entries in last X minutes of NY session (22:00 GMT close)
# Prevents trades from sitting through the 22:00-00:00 GMT liquidity vacuum
NY_END_BLOCK_MINUTES = 45
COOLDOWN_MINUTES = 30              # Cooldown period after max losses

# Candle Pattern Filter
# Vetoes entries when recent candle anatomy contradicts trade direction.
# Independent of AI — pure price action filter.
# Catches shooting star / hammer / engulfing / tweezer / morning-evening star /
# three soldiers / three crows patterns that signal reversal against entry.
CANDLE_PATTERN_FILTER_ENABLED = True
# COOLDOWN removed — COOLDOWN_MINUTES is used throughout
balance_peak = 0

equity_peak = 0
consecutive_losses = 0
last_loss_time = 0

max_trade_minutes = 360

last_position_close_time = 0
cooldown_duration = 0
COOLDOWN_AFTER_TRADE = 480   # seconds
COOLDOWN_AFTER_LOSS = 660   # seconds

last_volatility_alert = 0
last_news_print = 0
last_smc_update = 0

# SL/TP Settings
SL_FVG_BUFFER = 2.0  # pips
SL_OB_BUFFER = 3.0   # pips
TP_LIQUIDITY_MULTIPLIER = 1.2

# Trailing Stop
TRAILING_ENABLED = True            # set False to disable trailing stop entirely
TRAIL_ACTIVATION_PIPS = 8.0
TRAIL_DISTANCE_PIPS = 4.0

# Partial Profit Taking
PARTIAL_TP_ENABLED = True          # set False to disable partial closes
PARTIAL_TP_LEVELS = [
    {"pips": 6.0, "percentage": 0.30},  # 30% at +6 pips
    {"pips": 10.0, "percentage": 0.30}, # 30% at +10 pips
    {"pips": 15.0, "percentage": 0.20}, # 20% at +15 pips
]

# Move SL to breakeven
BREAKEVEN_ENABLED = True           # set False to disable auto-breakeven
MOVE_TO_BREAKEVEN_AFTER_PIPS = 5.0

# Position Scaling
SCALE_IN_ENABLED = False
SCALE_IN_LEVELS = [
    {"pips_from_entry": -5.0, "volume_multiplier": 0.5},
    {"pips_from_entry": -10.0, "volume_multiplier": 0.75},
]

# News Filter
NEWS_FILTER_MINUTES_BEFORE = 45
NEWS_FILTER_MINUTES_AFTER = 30
NEWS_API_KEY = ""  # Optional: Add your ForexFactory API key
NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

NEWS_CACHE = []
LAST_NEWS_LOG = 0
NEWS_LOG_COOLDOWN = 300

# Volume Profile
VOLUME_PROFILE_BINS = 50
VOLUME_PROFILE_LOOKBACK = 100

# Multi-Timeframe Analysis
MTF_CONFIDENCE_WEIGHT = {
    "M5": 0.35,
    "M15": 0.25,
    "H1": 0.25,
    "H4": 0.15
}

# Data Persistence
DATA_DIR = "bot_data_smc"
os.makedirs(DATA_DIR, exist_ok=True)

TRADE_HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.json")
MARKET_HISTORY_FILE = os.path.join(DATA_DIR, "market_history.csv")
MODEL_STATE_FILE = os.path.join(DATA_DIR, "model_state.pkl")
HEATMAP_FILE = os.path.join(DATA_DIR, "liquidity_heatmap.json")
VOLUME_PROFILE_FILE = os.path.join(DATA_DIR, "volume_profile.json")
NEWS_CACHE_FILE = os.path.join(DATA_DIR, "news_cache.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "bot_settings.json")
SESSION_FILE = os.path.join(DATA_DIR, "session_data.json")
DAILY_STATS_FILE = os.path.join(DATA_DIR, "daily_stats.json")

# ================= GLOBAL STATE =================

regime_model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
# XGBoost outperforms GradientBoosting on tabular data — better regularisation,
# column subsampling, and native handling of class imbalance.
if _XGB_OK:
    entry_model = _XGBWrapper(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42, n_jobs=1,
    )
else:
    entry_model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
exit_model   = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)

scaler_regime = StandardScaler()
scaler_entry = StandardScaler()
scaler_exit = StandardScaler()

last_train = 0
last_train_candles = 0   # tracks candle count at last training (for retrain guard)
model_lock = threading.Lock()
AI_MODEL = False  # Set True once models are trained/loaded

# ── Feature names — defined here (top-level) so load_model_state can validate ──
# Must match generate_smc_features output exactly. 38 features total.
FEATURE_NAMES = [
    # original 26
    'relative_range', 'body_ratio', 'momentum_bull', 'momentum_bear',
    'trend_bullish', 'trend_bearish', 'bos_bull', 'bos_bear',
    'in_bullish_fvg', 'in_bearish_fvg', 'in_bullish_ob', 'in_bearish_ob',
    'dist_to_upper_liq', 'dist_to_lower_liq', 'liq_sweep_low', 'liq_sweep_high',
    'mtf_bullish', 'mtf_bearish', 'mtf_score', 'news_active', 'spread',
    'bullish_confluence', 'bearish_confluence', 'bullish_reversal_setup',
    'bearish_reversal_setup', 'net_bias',
    # 10 precision features
    'rsi_14', 'choch_bull', 'choch_bear',
    'htf_bull_align', 'htf_bear_align',
    'fvg_strength_bull', 'fvg_strength_bear',
    'vol_ratio', 'ema_cross', 'bos_strength',
    'dist_to_bull_ob', 'dist_to_bear_ob',
    # 5 regime features (Tier-1 upgrade)
    'adx_strength', 'regime_trending', 'regime_ranging', 'regime_volatile', 'di_bull',
    # 5 temporal + volatility features (Tier-2 upgrade)
    # Model previously had no sense of time — these let it learn intraday/weekly patterns.
    'hour_sin', 'hour_cos',      # time-of-day (cyclical, avoids midnight discontinuity)
    'dow_sin', 'dow_cos',        # day-of-week (cyclical Mon=0 … Fri=4)
    'vol_percentile',            # current ATR vs last-100-bar ATR distribution (0–1)
]

liquidity_heatmap = deque(maxlen=500)
volume_profile_data = deque(maxlen=200)
daily_trades = deque(maxlen=100)

ob_cache = {"bullish": [], "bearish": []}
liquidity_pools = {"high": [], "low": []}
fvg_cache = {"bullish": [], "bearish": []}

trade_journal = deque(maxlen=10000)
ai_decision_history = deque(maxlen=1000)

exit_tracker = {}
position_profit_tracker = {}
known_tickets = set()  # Tracks open position tickets to detect TP/SL hits between loops
scaled_positions = {}   # tracks scale-in levels per ticket

# Post-trade re-entry discipline
last_trade_was_loss    = False   # True if last closed trade was a loss
post_loss_recheck_done = False   # True once market has been re-validated after a loss

news_cache = []

# Risk control (defined once here, also at config above for initialization)
trading_paused = False

emergency_stop = False
daily_loss_triggered = False

# ================= DATA PERSISTENCE FUNCTIONS =================

def get_settings():
    """Load or create bot settings"""

    default_settings = {
        "symbol": SYMBOL,
        "risk_per_trade": RISK_PER_TRADE,
        "max_drawdown_percent": MAX_DRAWDOWN_PERCENT,
        "max_spread": MAX_SPREAD,
        "ai_entry_threshold": AI_ENTRY_THRESHOLD,
        "cooldown": COOLDOWN_MINUTES,
        "first_run": True,
        "total_runtime_hours": 0,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "version": "4.0-SMC"
    }

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)

            # Ensure new keys are added if version updates
            for key, value in default_settings.items():
                if key not in settings:
                    settings[key] = value

            return settings

        except Exception as e:
            print(f"Error loading settings: {e}")

    # Create new settings if file doesn't exist or failed
    save_settings(default_settings)
    return default_settings

def save_settings(settings):
    """Save bot settings safely"""
    try:
        temp_file = SETTINGS_FILE + ".tmp"

        with open(temp_file, 'w') as f:
            json.dump(settings, f, indent=2)

        os.replace(temp_file, SETTINGS_FILE)

    except Exception as e:
        print(f"Error saving settings: {e}")

TRADE_FIELDS = [
    "time",
    "ticket",
    "symbol",
    "signal",
    "entry",
    "sl",
    "tp",
    "lot",
    "ai_prob",
    "buy_prob",
    "sell_prob",
    "trend",
    "spread",
    "sweep",
    "fvg",
    "reason",
    "result"
]

def log_trade(data):
    try:
        file_path = "trades.csv"
        file_exists = os.path.isfile(file_path)

        # Ensure time is always present
        if "time" not in data or data["time"] is None:
            data["time"] = datetime.now().isoformat()

        # Normalize row (prevents missing keys breaking CSV)
        row = {key: data.get(key, None) for key in TRADE_FIELDS}

        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)

            # Write header only once
            if not file_exists or os.stat(file_path).st_size == 0:
                writer.writeheader()

            writer.writerow(row)

    except Exception as e:
        print(f"❌ Error logging trade: {e}")

def load_trade_history():
    history = []

    if os.path.exists(TRADE_HISTORY_FILE):
        try:
            with open(TRADE_HISTORY_FILE, 'r') as f:
                for line in f:
                    history.append(json.loads(line))

            print(f"Loaded {len(history)} trades from history")
        except Exception as e:
            print(f"Error loading trade history: {e}")

    return history

def add_trade_to_history(trade_data):
    """Append trade to history file"""
    try:
        with open(TRADE_HISTORY_FILE, 'a') as f:
            json.dump(trade_data, f)
            f.write("\n")
    except Exception as e:
        print(f"Error appending trade history: {e}")

def load_market_history():
    global market_cache, market_cache_time

    if not os.path.exists(MARKET_HISTORY_FILE):
        return None

    if market_cache is not None and time.time() - market_cache_time < 60:
        return market_cache.copy()

    try:
        df = pd.read_csv(MARKET_HISTORY_FILE)

        # =============================
        # COLUMN VALIDATION
        # =============================
        required_cols = ['time', 'open', 'high', 'low', 'close']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"❌ Missing columns: {missing}")
            return None

        # =============================
        # TIME PARSING
        # =============================
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])

        # =============================
        # CLEANING
        # =============================
        df = df.drop_duplicates(subset=['time'], keep='last')
        df = df.sort_values('time')
        df = df.set_index('time')

        # Ensure unique index
        df = df[~df.index.duplicated(keep='last')]

        # =============================
        # OPTIONAL: ENFORCE M5 STRUCTURE
        # =============================
        # df = df.asfreq('5min').dropna()

        # =============================
        # LIMIT SIZE
        # =============================
        MAX_LOAD = 50000
        if len(df) > MAX_LOAD:
            df = df.tail(MAX_LOAD)

        print(f"Loaded {len(df)} candles from market history")

        market_cache = df
        market_cache_time = time.time()

        return df.copy()

    except Exception as e:
        print(f"Error loading market history: {e}")
        return None

def save_market_history(df):

    global market_cache

    try:
        # =============================
        # 1. Ensure datetime
        # =============================
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], errors='coerce')

        df = df.dropna(subset=['time'])

        # =============================
        # 2. Load cache (if needed)
        # =============================
        if market_cache is None:
            market_cache = load_market_history()

        if market_cache is not None and len(market_cache) > 0:

            # IMPORTANT: reset index to merge properly
            cache_df = market_cache.reset_index()

            combined = pd.concat([cache_df, df], ignore_index=True)

            # =============================
            # 3. Remove duplicates
            # =============================
            combined = combined.drop_duplicates(
                subset=['time', 'open', 'high', 'low', 'close'],
                keep='last'
            )

            combined = combined.sort_values('time')

            # =============================
            # 4. Limit size
            # =============================
            if len(combined) > 300000:
                combined = combined.tail(300000)

            # =============================
            # 5. SAVE (column format)
            # =============================
            combined.to_csv(MARKET_HISTORY_FILE, index=False)

            # =============================
            # 6. UPDATE CACHE (CRITICAL FIX)
            # =============================
            combined = combined.set_index('time')
            market_cache = combined

        else:
            df = df.sort_values('time')
            df.to_csv(MARKET_HISTORY_FILE, index=False)

            # FIX: ensure cache matches training format
            market_cache = df.set_index('time')

    except Exception as e:
        print(f"Error saving market history: {e}")

def load_model_state():
    global regime_model, entry_model, exit_model
    global scaler_regime, scaler_entry, scaler_exit, last_train

    if not os.path.exists(MODEL_STATE_FILE):
        print("ℹ️ No model state file found. Starting fresh.")
        return False

    try:
        state = joblib.load(MODEL_STATE_FILE)

        # ✅ Version check — bump to 4.1 for 38-feature model
        if state.get('version') not in ('4.0', '4.1'):
            print("⚠️ Model version mismatch → retraining")
            return False

        # Load components
        regime_model = state.get('regime_model')
        entry_model  = state.get('entry_model')
        exit_model   = state.get('exit_model')
        scaler_regime = state.get('scaler_regime')
        scaler_entry  = state.get('scaler_entry')
        scaler_exit   = state.get('scaler_exit')
        last_train    = state.get('last_train', 0)

        # ✅ Feature count validation — reject model if trained on wrong number of features
        expected = len(FEATURE_NAMES)
        if hasattr(scaler_entry, 'n_features_in_') and scaler_entry.n_features_in_ != expected:
            print(f"⚠️ Model trained on {scaler_entry.n_features_in_} features, current is {expected} → retraining")
            return False

        # ✅ Full validation
        if not all([entry_model, exit_model, scaler_entry, scaler_exit]):
            print("⚠️ Model state incomplete → retraining")
            return False

        # ✅ Timestamp validation
        if not last_train or last_train == 0:
            print("⚠️ Invalid last_train → retraining")
            return False

        print(f"📊 Loaded model → Entry: {type(entry_model)} | Exit: {type(exit_model)}")
        print(f"✅ AI Model loaded - Last training: {datetime.fromtimestamp(last_train)}")

        return True

    except Exception as e:
        print(f"❌ Critical Error loading model state: {e}")
        return False

def save_model_state():
    try:
        # ✅ Full validation
        if entry_model is None or exit_model is None or scaler_entry is None or scaler_exit is None:
            print("⚠️ Skipping save: Model state incomplete.")
            return

        # ✅ Ensure training timestamp exists
        if not last_train or last_train == 0:
            print("⚠️ Skipping save: last_train not set.")
            return

        state = {
            'regime_model':  regime_model,
            'entry_model':   entry_model,
            'exit_model':    exit_model,
            'scaler_regime': scaler_regime,
            'scaler_entry':  scaler_entry,
            'scaler_exit':   scaler_exit,
            'last_train':    last_train,
            'version':       '4.1',
            'feature_count': len(FEATURE_NAMES),
            'saved_at':      datetime.now().isoformat()
        }

        print(
            f"📊 Saving model → "
            f"Entry: {entry_model.__class__.__name__} | "
            f"Exit: {exit_model.__class__.__name__}"
        )

        temp_file = MODEL_STATE_FILE + ".tmp"
        joblib.dump(state, temp_file)

        if os.path.exists(MODEL_STATE_FILE):
            os.remove(MODEL_STATE_FILE)

        os.rename(temp_file, MODEL_STATE_FILE)

        print("💾 AI Model state saved successfully")

    except Exception as e:
        print(f"❌ Error saving model state: {e}")

def load_liquidity_heatmap():
    """Load liquidity heatmap data"""
    global liquidity_heatmap
    if os.path.exists(HEATMAP_FILE):
        try:
            with open(HEATMAP_FILE, 'r') as f:
                data = json.load(f)
                liquidity_heatmap = deque(maxlen=500)
                liquidity_heatmap.extend(data)
                print(f"Loaded {len(liquidity_heatmap)} heatmap records")
        except Exception as e:
            print(f"Error loading heatmap: {e}")
    return liquidity_heatmap

def save_liquidity_heatmap():
    """Save liquidity heatmap data"""
    try:
        with open(HEATMAP_FILE, 'w') as f:
            json.dump(list(liquidity_heatmap), f)
    except Exception as e:
        print(f"Error saving heatmap: {e}")

def load_daily_stats():
    """Load daily trading statistics"""
    if os.path.exists(DAILY_STATS_FILE):
        try:
            with open(DAILY_STATS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "trades": 0,
        "profit": 0.0,
        "loss": 0.0,
        "net_profit": 0.0,
        "closed_today": False
    }

def save_daily_stats(stats):
    """Save daily trading statistics"""
    try:
        with open(DAILY_STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"Error saving daily stats: {e}")

def initialize_data_system():
    """Initialize all data systems"""
    global liquidity_heatmap

    print("\n" + "="*60)
    print("💾 SMC-AI-GOLD-SENTINEL V4.0 - LOADING DATA...")
    print("="*60)

    settings = get_settings()
    if settings.get("first_run", True):
        print("First run - initializing fresh SMC system")
        settings["first_run"] = False
        settings["version"] = "4.0-SMC"
        save_settings(settings)
    else:
        print(f"Welcome back! Version: {settings.get('version', '4.0-SMC')}")
        print(f"Previous runtime: {settings.get('total_runtime_hours', 0):.1f} hours")

    # ✅ Safe model load
    with model_lock:
        model_loaded = load_model_state()

    if not model_loaded:
        print("⚠️ No previous AI model found - will train from scratch")

    # ✅ Safe liquidity load
    lh = load_liquidity_heatmap()
    if lh is not None:
        liquidity_heatmap = lh
    else:
        print("⚠️ Using fresh liquidity heatmap")

    trade_history = load_trade_history()
    if trade_history:
        print(f"📜 Loaded {len(trade_history)} trades from history")

    market_history = load_market_history()
    if market_history is not None and len(market_history) > 0:
        print(f"📊 Loaded {len(market_history)} candles from market history")
    else:
        print("⚠️ No market history found")

    print("="*60)
    print("✅ ALL SMC DATA SYSTEMS INITIALIZED!")
    print("="*60 + "\n")

    return market_history if market_history is not None else pd.DataFrame()

def save_all_data():
    """Save all data"""
    print("\n💾 Saving all SMC data...")

    if last_train > 0:
        save_model_state()

    save_liquidity_heatmap()

    session_data = {"last_save": datetime.now().isoformat()}
    with open(SESSION_FILE, 'w') as f:
        json.dump(session_data, f, indent=2)

    settings = get_settings()
    if "start_time" in session_data:
        try:
            start = datetime.fromisoformat(session_data["start_time"])
            runtime = (datetime.now() - start).total_seconds() / 3600
            settings["total_runtime_hours"] = settings.get("total_runtime_hours", 0) + runtime
        except Exception:
            pass
    save_settings(settings)

    print("✅ All data saved!")

def is_trading_session():
    """Allow trading only during London and New York sessions"""
    now = datetime.utcnow()

    hour = now.hour

    # London session: 07:00 - 16:00 UTC
    london = 7 <= hour < 16

    # New York session: 12:00 - 21:00 UTC
    newyork = 12 <= hour < 21

    return london or newyork


def is_near_ny_close():
    """
    Returns True if within NY_END_BLOCK_MINUTES of NY session close (22:00 GMT).
    Used to block NEW entries — does NOT affect open position management.
    Prevents trades from sitting through the 22:00-00:00 GMT liquidity vacuum.
    """
    now_gmt = datetime.utcnow()
    hour = now_gmt.hour
    minute = now_gmt.minute

    # NY closes at 22:00 GMT — block window starts at 22:00 - NY_END_BLOCK_MINUTES
    # For 45 min: block from 21:15 GMT to 22:00 GMT
    if hour == 21:
        block_start_minute = 60 - NY_END_BLOCK_MINUTES
        if minute >= block_start_minute:
            return True

    return False


# ════════════════════════════════════════════════════════════════════════════
#                       CANDLE PATTERN DETECTION
# ════════════════════════════════════════════════════════════════════════════
# Classifies the last 1-3 candles into named reversal / continuation patterns.
# Output is used by `candle_pattern_veto()` as ENTRY FILTER, not AI feature.
# No retrain required, no FEATURE_NAMES change, no model file change.
# ════════════════════════════════════════════════════════════════════════════

def detect_candle_patterns(df, atr_ref=None):
    """
    Detect 12 reliable candlestick patterns on the most recent candles.

    Single-candle (on c3):  shooting_star, hammer, bullish_marubozu, bearish_marubozu
    Two-candle (c2+c3):     bullish_engulfing, bearish_engulfing, tweezer_top, tweezer_bottom
    Three-candle (c1+c2+c3): morning_star, evening_star, three_white_soldiers, three_black_crows

    Returns dict {pattern_name: 0|1}. Never raises — failures return all-zero dict.
    """
    patterns = {
        # Single-candle
        "shooting_star": 0, "hammer": 0,
        "bullish_marubozu": 0, "bearish_marubozu": 0,
        # Two-candle
        "bullish_engulfing": 0, "bearish_engulfing": 0,
        "tweezer_top": 0, "tweezer_bottom": 0,
        # Three-candle
        "morning_star": 0, "evening_star": 0,
        "three_white_soldiers": 0, "three_black_crows": 0,
    }

    try:
        if df is None or len(df) < 5:
            return patterns

        # Reference ATR for size normalization
        if atr_ref is None or atr_ref <= 0:
            atr_series = (df['high'] - df['low']).rolling(14).mean()
            atr_ref = float(atr_series.iloc[-1])
            if not np.isfinite(atr_ref) or atr_ref <= 0:
                return patterns

        # Last 3 candles
        c3 = df.iloc[-1]   # current (most recent)
        c2 = df.iloc[-2]   # one before
        c1 = df.iloc[-3]   # two before

        def anatomy(c):
            """Compute candle anatomy with safe-divide guards."""
            o  = float(c['open'])
            h  = float(c['high'])
            l  = float(c['low'])
            cl = float(c['close'])
            rng = h - l
            if rng <= 0:
                rng = 1e-9  # safe divide
            body       = abs(cl - o)
            upper_wick = h - max(o, cl)
            lower_wick = min(o, cl) - l
            return {
                'o': o, 'h': h, 'l': l, 'c': cl, 'rng': rng,
                'body': body, 'upper_wick': upper_wick, 'lower_wick': lower_wick,
                'is_bull': cl > o,
                'is_bear': cl < o,
                'body_ratio':  body       / rng,
                'upper_ratio': upper_wick / rng,
                'lower_ratio': lower_wick / rng,
            }

        a3 = anatomy(c3)
        a2 = anatomy(c2)
        a1 = anatomy(c1)

        # ───── SINGLE-CANDLE PATTERNS ─────

        # SHOOTING STAR: long upper wick + small body + minimal lower wick
        # Bearish reversal pattern when seen at top of uptrend.
        # Wick limit is RANGE-relative (industry standard) — body-relative
        # becomes impossibly strict when body is tiny (which is the point).
        if (a3['upper_wick'] >= 2 * a3['body']
            and a3['lower_ratio'] <= 0.10
            and a3['body_ratio'] <= 0.35
            and a3['rng'] >= atr_ref * 0.5):
            patterns['shooting_star'] = 1

        # HAMMER: mirror of shooting star — long lower wick + small body
        # Bullish reversal pattern when seen at bottom of downtrend.
        if (a3['lower_wick'] >= 2 * a3['body']
            and a3['upper_ratio'] <= 0.10
            and a3['body_ratio'] <= 0.35
            and a3['rng'] >= atr_ref * 0.5):
            patterns['hammer'] = 1

        # BULLISH MARUBOZU: full bullish body, minimal wicks
        # Strong continuation — buyers in total control.
        if (a3['is_bull']
            and a3['body_ratio'] >= 0.80
            and a3['upper_ratio'] <= 0.10
            and a3['lower_ratio'] <= 0.10
            and a3['rng'] >= atr_ref * 0.8):
            patterns['bullish_marubozu'] = 1

        # BEARISH MARUBOZU: mirror — strong bearish continuation
        if (a3['is_bear']
            and a3['body_ratio'] >= 0.80
            and a3['upper_ratio'] <= 0.10
            and a3['lower_ratio'] <= 0.10
            and a3['rng'] >= atr_ref * 0.8):
            patterns['bearish_marubozu'] = 1

        # ───── TWO-CANDLE PATTERNS ─────

        # BULLISH ENGULFING: bearish c2 followed by larger bullish c3 covering c2 body
        if (a2['is_bear'] and a3['is_bull']
            and a3['o'] <= a2['c']
            and a3['c'] >= a2['o']
            and a3['body'] >= 1.5 * max(a2['body'], 1e-9)):
            patterns['bullish_engulfing'] = 1

        # BEARISH ENGULFING: bullish c2 followed by larger bearish c3 covering c2 body
        if (a2['is_bull'] and a3['is_bear']
            and a3['o'] >= a2['c']
            and a3['c'] <= a2['o']
            and a3['body'] >= 1.5 * max(a2['body'], 1e-9)):
            patterns['bearish_engulfing'] = 1

        # TWEEZER TOP: matching highs after uptrend candle, then rejection candle
        # Bearish reversal — sellers defending the same price level twice.
        high_diff = abs(a2['h'] - a3['h'])
        if (a2['is_bull'] and a3['is_bear']
            and high_diff <= atr_ref * 0.15
            and a2['upper_wick'] >= 0.15 * a2['rng']
            and a3['upper_wick'] >= 0.15 * a3['rng']):
            patterns['tweezer_top'] = 1

        # TWEEZER BOTTOM: matching lows after downtrend candle, then recovery
        # Bullish reversal — buyers defending same level.
        low_diff = abs(a2['l'] - a3['l'])
        if (a2['is_bear'] and a3['is_bull']
            and low_diff <= atr_ref * 0.15
            and a2['lower_wick'] >= 0.15 * a2['rng']
            and a3['lower_wick'] >= 0.15 * a3['rng']):
            patterns['tweezer_bottom'] = 1

        # ───── THREE-CANDLE PATTERNS ─────

        # EVENING STAR: bullish + small star + bearish closing below c1 midpoint
        # Classic top-reversal pattern.
        c1_mid = (a1['o'] + a1['c']) / 2
        if (a1['is_bull'] and a1['body_ratio'] >= 0.5 and a1['rng'] >= atr_ref * 0.7
            and a2['body'] <= 0.3 * max(a1['body'], 1e-9)
            and a3['is_bear'] and a3['body_ratio'] >= 0.4
            and a3['c'] <= c1_mid):
            patterns['evening_star'] = 1

        # MORNING STAR: bearish + small star + bullish closing above c1 midpoint
        # Classic bottom-reversal pattern.
        if (a1['is_bear'] and a1['body_ratio'] >= 0.5 and a1['rng'] >= atr_ref * 0.7
            and a2['body'] <= 0.3 * max(a1['body'], 1e-9)
            and a3['is_bull'] and a3['body_ratio'] >= 0.4
            and a3['c'] >= c1_mid):
            patterns['morning_star'] = 1

        # THREE WHITE SOLDIERS: 3 strong consecutive bullish candles
        # Each opens within previous body, closes higher than previous close.
        if (a1['is_bull'] and a2['is_bull'] and a3['is_bull']
            and a1['body_ratio'] >= 0.5 and a2['body_ratio'] >= 0.5 and a3['body_ratio'] >= 0.5
            and a1['o'] < a2['o'] < a1['c']
            and a2['o'] < a3['o'] < a2['c']
            and a1['c'] < a2['c'] < a3['c']):
            patterns['three_white_soldiers'] = 1

        # THREE BLACK CROWS: 3 strong consecutive bearish candles (mirror)
        if (a1['is_bear'] and a2['is_bear'] and a3['is_bear']
            and a1['body_ratio'] >= 0.5 and a2['body_ratio'] >= 0.5 and a3['body_ratio'] >= 0.5
            and a1['c'] < a2['o'] < a1['o']
            and a2['c'] < a3['o'] < a2['o']
            and a1['c'] > a2['c'] > a3['c']):
            patterns['three_black_crows'] = 1

        return patterns

    except Exception as e:
        # Never crash — fail open
        print(f"⚠️ Candle pattern detection error: {e}")
        return patterns


# Patterns that veto BUY entries (bearish reversal signals)
BEARISH_VETO_PATTERNS = (
    "shooting_star",
    "bearish_engulfing",
    "tweezer_top",
    "evening_star",
    "three_black_crows",
)

# Patterns that veto SELL entries (bullish reversal signals)
BULLISH_VETO_PATTERNS = (
    "hammer",
    "bullish_engulfing",
    "tweezer_bottom",
    "morning_star",
    "three_white_soldiers",
)

# Note: marubozu patterns are CONTINUATION, not reversal — they don't veto
# same-direction trades. A bullish_marubozu doesn't block BUY.


def candle_pattern_veto(df, direction, atr_ref=None):
    """
    Returns (veto: bool, pattern_name: str | None).
    Vetoes the entry if a contradicting reversal pattern is detected.
    Fail-safe: returns (False, None) on any error or when filter disabled.
    """
    try:
        if not CANDLE_PATTERN_FILTER_ENABLED:
            return False, None

        if df is None or len(df) < 5 or direction is None:
            return False, None

        patterns = detect_candle_patterns(df, atr_ref)

        direction = direction.upper()

        if direction == "BUY":
            for p in BEARISH_VETO_PATTERNS:
                if patterns.get(p, 0) == 1:
                    return True, p

        elif direction == "SELL":
            for p in BULLISH_VETO_PATTERNS:
                if patterns.get(p, 0) == 1:
                    return True, p

        return False, None

    except Exception as e:
        print(f"⚠️ Candle pattern veto error: {e}")
        return False, None


# ════════════════════════════════════════════════════════════════════════════
# ================= MT5 CONNECTION DATAS =================

# ---------------- Symbol Finder ----------------
def find_symbol(base="XAUUSD"):
    symbols = mt5.symbols_get()

    # ✅ First try exact match
    for s in symbols:
        if s.name == base:
            print(f"✅ Found exact symbol: {s.name}")
            return s.name

    # ⚠️ Fallback to partial match
    for s in symbols:
        if base in s.name:
            print(f"⚠️ Using fallback symbol: {s.name}")
            return s.name

    print("❌ No matching symbol found for", base)
    return None

# ---------------- MT5 Initialization ----------------
def initialize_mt5():
    """Initialize MetaTrader 5 with reconnection logic and robust symbol detection"""
    global start_balance, MT5_INITIALIZED, SYMBOL

    for attempt in range(MT5_RECONNECT_ATTEMPTS):
        try:
            # Already initialized?
            if MT5_INITIALIZED:
                account = mt5.account_info()
                if account is not None:
                    print("MT5 connection verified")
                    return True

            # Initialize MT5
            if not mt5.initialize():
                print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            else:
                print("✅ MetaTrader 5 initialized successfully")
                time.sleep(1)  # small delay to refresh symbols

                # Auto-detect symbol if not set or invalid
                if SYMBOL is None or mt5.symbol_info(SYMBOL) is None:
                    SYMBOL = find_symbol("XAUUSD")

                if SYMBOL is None:
                    print("❌ Could not find XAUUSD symbol on broker")
                    mt5.shutdown()
                    return False

                # Ensure symbol is visible in Market Watch
                if not mt5.symbol_select(SYMBOL, True):
                    print(f"❌ Failed to enable symbol {SYMBOL}")
                    mt5.shutdown()
                    return False

                # Account info check
                account = mt5.account_info()
                if account is None:
                    print("❌ Failed to retrieve account info")
                    mt5.shutdown()
                    return False

                start_balance = account.balance
                print(f"💰 Account balance: ${start_balance:.2f}")
                print(f"📈 Equity: ${account.equity:.2f}")
                print(f"📊 Profit: ${account.profit:.2f}")

                MT5_INITIALIZED = True
                return True

        except Exception as e:
            print(f"❌ MT5 initialization error ({attempt+1}/{MT5_RECONNECT_ATTEMPTS}): {e}")

        if attempt < MT5_RECONNECT_ATTEMPTS - 1:
            print(f"⏳ Retrying in {MT5_RECONNECT_DELAY} seconds...")
            time.sleep(MT5_RECONNECT_DELAY)

    print("❌ Failed to initialize MT5 after all attempts")
    return False

# ---------------- MT5 Reconnection ----------------
def reconnect_mt5():
    """Reconnect to MT5 after connection loss"""
    global MT5_INITIALIZED, SYMBOL

    print("🔄 Attempting to reconnect to MT5...")

    try:
        # Shutdown if needed
        try:
            mt5.shutdown()
        except Exception:
            pass

        time.sleep(2)

        for attempt in range(MT5_RECONNECT_ATTEMPTS):
            if mt5.initialize():
                account = mt5.account_info()
                if account is None:
                    print("❌ Reconnected but account info unavailable")
                    return False

                # Ensure symbol is enabled after reconnection
                if SYMBOL is None or mt5.symbol_info(SYMBOL) is None:
                    SYMBOL = find_symbol("XAUUSD")
                if SYMBOL and not mt5.symbol_select(SYMBOL, True):
                    print(f"❌ Failed to enable symbol {SYMBOL}")
                    return False

                print("✅ MT5 reconnected successfully")
                print(f"💰 Account balance: ${account.balance:.2f}")
                MT5_INITIALIZED = True
                return True

            print(f"⏳ Reconnection attempt {attempt+1} failed, retrying...")
            time.sleep(MT5_RECONNECT_DELAY)

        print("❌ Failed to reconnect to MT5")
        return False

    except Exception as e:
        print(f"❌ MT5 reconnection error: {e}")
        return False

# ---------------- MT5 Connection Check ----------------
def check_mt5_connection():
    """Check if MT5 connection is alive"""
    try:
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        if terminal is None or account is None:
            return False
        return terminal.connected
    except Exception:
        return False

# ================= DATA FETCHING =================

def get_data(timeframe=mt5.TIMEFRAME_M5, bars=BARS, retries=3):
    """Fetch market data from MT5 (ultra-safe version)"""

    for attempt in range(retries):
        try:
            # ✅ Ensure MT5 initialized
            if not check_mt5_connection():
                print("⚠️ MT5 disconnected during data fetch")
                return None

            # ✅ Ensure symbol exists
            info = mt5.symbol_info(SYMBOL)
            if info is None:
                raise Exception(f"Symbol {SYMBOL} not found")

            # ✅ Ensure symbol is visible
            if not info.visible:
                print(f"👁️ Enabling {SYMBOL}...")
    
            if not mt5.symbol_select(SYMBOL, True):
                raise Exception(f"Failed to select {SYMBOL}")

            # ✅ Fetch data
            rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, bars)

            if rates is None or not isinstance(rates, np.ndarray) or rates.size == 0:
                raise Exception("Invalid or empty rates")

            df = pd.DataFrame(rates)

            required_cols = {"time", "open", "high", "low", "close"}
            if not required_cols.issubset(df.columns):
                raise Exception("Missing required columns")

            df["time"] = pd.to_datetime(df["time"], unit="s")

            return df

        except Exception as e:
            print(f"⚠️ Data fetch error (attempt {attempt+1}): {e}")

            # ✅ CRITICAL: re-select symbol after reset
            mt5.symbol_select(SYMBOL, True)

            time.sleep(1)

    print("❌ MT5 data fetch FAILED after retries")
    return None

def get_multi_timeframe_data():
    """Fetch multiple timeframe data (optimized)"""

    if not check_mt5_connection():
        print("⚠️ MT5 disconnected in MTF fetch")
        return None

    data = {}

    tf_map = {
        "M5": (TIMEFRAME, BARS),
        "M15": (TIMEFRAME_M15, BARS_M15),
        "H1": (TIMEFRAME_H1, BARS_H1),
        "H4": (TIMEFRAME_H4, BARS_H4),
    }

    for tf_name, (tf, bars) in tf_map.items():
        df = get_data(tf, bars)
        if df is None:
            return None
        data[tf_name] = df

    return data

# ================= SPREAD & RISK FUNCTIONS =================

def get_spread():
    """Get current spread in raw points (float for precision)"""
    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        symbol_info = mt5.symbol_info(SYMBOL)

        if tick is None or symbol_info is None:
            return 999.0

        # Use float instead of int to catch fractional spread spikes
        return round((tick.ask - tick.bid) / symbol_info.point, 1)

    except Exception:
        return 999.0

def spread_ok():
    """Check if spread is acceptable"""
    spread = get_spread()

    if spread > MAX_SPREAD:
        print(f"Spread too high ({spread}), skipping trade")
        return False

    if spread < MIN_SPREAD:
        return False

    return True


def spread_too_high():
    """Quick check for spread limit"""
    return get_spread() > MAX_SPREAD
# ================= Liquidity Sweep Traps =================
def detect_sweep_trap(df):
    """
    Detects Stop-Hunt Liquidity Sweeps (The 'Spring' or 'Upthrust').
    Essential for catching XAUUSD reversals.
    """
    try:
        if len(df) < 20:
            return None

        # 1. Define 'Recent' levels EXCLUDING the current candle
        # 50-candle lookback (~4 hours) — catches institutional stop levels
        # 15 candles was too short for meaningful SMC sweeps on gold
        lookback_df = df.iloc[-51:-1]
        recent_high = lookback_df['high'].max()
        recent_low = lookback_df['low'].min()

        current = df.iloc[-1]
        
        # Calculate candle metrics for 'Rejection' confirmation
        candle_range = current['high'] - current['low']
        body_size = abs(current['close'] - current['open'])
        
        if candle_range == 0: return None # Avoid division by zero
        
        # A strong sweep usually has a wick that is at least 50% of the candle
        wick_percentage = (candle_range - body_size) / candle_range

        # 2. BUY TRAP (Sell-Side Liquidity Sweep)
        # Price dipped below recent support but closed back above it
        if current['low'] < recent_low and current['close'] > recent_low:
            # Confirm with rejection strength
            if wick_percentage > 0.4: 
                return "buy_trap"

        # 3. SELL TRAP (Buy-Side Liquidity Sweep)
        # Price poked above recent resistance but closed back below it
        if current['high'] > recent_high and current['close'] < recent_high:
            # Confirm with rejection strength
            if wick_percentage > 0.4:
                return "sell_trap"

        return None

    except Exception as e:
        print(f"❌ Sweep detection error: {e}")
        return None

# ================= SMC ENGINE: FAIR VALUE GAPS =================

def detect_fvg(df):
    """
    Detects Fair Value Gaps with Displacement & Mitigation checks.
    Optimized for XAUUSD volatility.
    """
    fvgs = {"bullish": [], "bearish": []}

    try:
        if len(df) < 50:
            return fvgs

        df = df.copy().reset_index(drop=True)
        high_low_range = df['high'] - df['low']
        atr = high_low_range.rolling(14).mean().bfill()

        # We only scan the last 100 candles to keep the execution fast
        start_idx = max(2, len(df) - 100)

        for i in range(start_idx, len(df)):
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            
            # 1. DISPLACEMENT CHECK (Candle 2 must be an expansion candle)
            body_size = abs(c2['close'] - c2['open'])
            if body_size < (atr.iloc[i] * 1.2): # 1.2x ATR ensures "Big Paper" move
                continue

            # 2. BULLISH FVG (Gap between C1 High and C3 Low)
            if c1['high'] < c3['low']:
                gap_size = c3['low'] - c1['high']
                if gap_size > (atr.iloc[i] * 0.30): # Significant gap filter
                    # Mitigation: Check if any candle AFTER i has touched this gap
                    future_candles = df.iloc[i+1:]
                    if not any(future_candles['low'] <= c1['high']):
                        fvgs["bullish"].append({
                            "top": c3['low'],
                            "bottom": c1['high'],
                            "mid": (c3['low'] + c1['high']) / 2,
                            "strength": gap_size / atr.iloc[i], # Normalized strength
                            "index": i
                        })

            # 3. BEARISH FVG (Gap between C1 Low and C3 High)
            elif c1['low'] > c3['high']:
                gap_size = c1['low'] - c3['high']
                if gap_size > (atr.iloc[i] * 0.30):
                    future_candles = df.iloc[i+1:]
                    if not any(future_candles['high'] >= c1['low']):
                        fvgs["bearish"].append({
                            "top": c1['low'],
                            "bottom": c3['high'],
                            "mid": (c1['low'] + c3['high']) / 2,
                            "strength": gap_size / atr.iloc[i],
                            "index": i
                        })

        return {"bullish": fvgs["bullish"][-5:], "bearish": fvgs["bearish"][-5:]}

    except Exception as e:
        print(f"❌ FVG detection error: {e}")
        return {"bullish": [], "bearish": []}

def get_nearest_fvg(direction, current_price, fvgs, atr=None):
    """
    Finds the closest unmitigated FVG. 
    Added: ATR filter to ignore 'Ghost Gaps' that are too far away to matter.
    """
    try:
        # 1. Select target pool
        pool = "bullish" if direction == "buy" else "bearish"
        target_fvgs = fvgs.get(pool, [])

        if not target_fvgs:
            return None

        nearest = None
        min_distance = float('inf')
        
        # 2. Set a 'Relevance Horizon' (e.g., ignore gaps further than 3 ATRs)
        # If ATR isn't provided, we skip the filter
        max_dist = (atr * 3.0) if atr else float('inf')

        for fvg in target_fvgs:
            # BULLISH (Support below us)
            if direction == "buy":
                level = fvg["top"]
                if level < current_price:
                    dist = current_price - level
                    if dist < min_distance and dist < max_dist:
                        min_distance = dist
                        nearest = fvg

            # BEARISH (Resistance above us)
            else:
                level = fvg["bottom"]
                if level > current_price:
                    dist = level - current_price
                    if dist < min_distance and dist < max_dist:
                        min_distance = dist
                        nearest = fvg

        return nearest

    except Exception as e:
        print(f"❌ Error in get_nearest_fvg: {e}")
        return None

def get_fvg_signal(fvg_data, current_price, df):
    if not fvg_data or df is None or len(df) < 20:
        return None

    # 🔥 Safe ATR calculation
    atr_series = (df['high'] - df['low']).rolling(14).mean()
    atr = atr_series.iloc[-1]

    if atr is None or atr == 0 or str(atr) == "nan":
        return None

    # 🔥 Clamp max distance (avoid extreme volatility issues)
    max_distance = min(atr * 2.5, atr * 5)

    nearest_bearish = None
    nearest_bullish = None
    min_bear_dist = float('inf')
    min_bull_dist = float('inf')

    # --- Bearish FVG ---
    for fvg in fvg_data.get("bearish", []):
        if fvg["mid"] > current_price:
            dist = fvg["mid"] - current_price

            if dist > max_distance:
                continue

            # 🔥 Stronger weighting
            strength = fvg.get("strength", 0)
            strength_mult = 0.75 if strength > 15 else 1.0

            adj_dist = dist * strength_mult

            if adj_dist < min_bear_dist:
                min_bear_dist = adj_dist
                nearest_bearish = fvg

    # --- Bullish FVG ---
    for fvg in fvg_data.get("bullish", []):
        if fvg["mid"] < current_price:
            dist = current_price - fvg["mid"]

            if dist > max_distance:
                continue

            strength = fvg.get("strength", 0)
            strength_mult = 0.75 if strength > 15 else 1.0

            adj_dist = dist * strength_mult

            if adj_dist < min_bull_dist:
                min_bull_dist = adj_dist
                nearest_bullish = fvg

    # --- Decision ---
    if nearest_bearish and not nearest_bullish:
        return "bearish_fvg"

    if nearest_bullish and not nearest_bearish:
        return "bullish_fvg"

    if nearest_bearish and nearest_bullish:
        return "bearish_fvg" if min_bear_dist < min_bull_dist else "bullish_fvg"

    return None

# ================= SMC ENGINE: ORDER BLOCKS =================

def detect_order_blocks(df, lookback=20, is_training=False):
    """
    Detects unmitigated institutional Order Blocks.
    Optimization: Added Mitigation Check and Dynamic Volume Thresholding.
    """
    global ob_cache
    obs = {"bullish": [], "bearish": []}

    try:
        if len(df) < 50: return obs
        df = df.copy().reset_index(drop=True)
        
        # Scan the last 100 candles to find zones that might still be active
        start_idx = max(50, len(df) - 100)
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        vol_mean = df['tick_volume'].rolling(20).mean()

        for i in range(start_idx, len(df) - 1):
            curr, prev = df.iloc[i], df.iloc[i-1]
            
            # 1. Detect Potential OB Creation (BOS + Volume)
            recent_high = df['high'].iloc[i-lookback:i].max()
            recent_low = df['low'].iloc[i-lookback:i].min()
            
            is_ob = False
            ob_type = None

            # Bullish OB logic
            if curr['close'] > recent_high and prev['close'] < prev['open']:
                # UPGRADE: Check Volume AND ensure the expansion candle had a massive impulsive body (> 50% ATR)
                candle_body = abs(curr['close'] - curr['open'])
                if curr['tick_volume'] > vol_mean.iloc[i] * 1.5 and candle_body > (atr * 0.5):
                    is_ob = True
                    ob_type = "bullish"

            # Bearish OB logic
            elif curr['close'] < recent_low and prev['close'] > prev['open']:
                candle_body = abs(curr['close'] - curr['open'])
                if curr['tick_volume'] > vol_mean.iloc[i] * 1.5 and candle_body > (atr * 0.5):
                    is_ob = True
                    ob_type = "bearish"

            # 2. Mitigation Check (Is the zone still 'Fresh'?)
            if is_ob:
                # Look at all candles from the OB creation (i) to the current moment
                future_candles = df.iloc[i+1:]
                
                if ob_type == "bullish":
                    # A Bullish OB is mitigated if price drops back into/below the bottom
                    mitigated = (future_candles['low'] <= prev['low']).any()
                    if not mitigated:
                        obs["bullish"].append({
                            "top": prev['high'],
                            "bottom": prev['low'],
                            "mid": (prev['high'] + prev['low']) / 2,
                            "strength": (curr['close'] - prev['low']) / atr,
                            "index": i-1,
                            "timestamp": time.time()
                        })
                
                else: # Bearish
                    # A Bearish OB is mitigated if price rises back into/above the top
                    mitigated = (future_candles['high'] >= prev['high']).any()
                    if not mitigated:
                        obs["bearish"].append({
                            "top": prev['high'],
                            "bottom": prev['low'],
                            "mid": (prev['high'] + prev['low']) / 2,
                            "strength": (prev['high'] - curr['close']) / atr,
                            "index": i-1,
                            "timestamp": time.time()
                        })

        # --- OUTSIDE THE FOR LOOP ---
        if not is_training:
            ob_cache["bullish"] = obs["bullish"][-15:]
            ob_cache["bearish"] = obs["bearish"][-15:]
        
        return obs

    except Exception as e:
        print(f"❌ OB detection error: {e}")
        return obs

def get_nearest_ob(direction, current_price, atr=None):
    """
    Finds the most relevant unmitigated Order Block.
    Logic: Prioritizes Strength and Freshness over pure distance.
    """
    global ob_cache
    
    try:
        # 1. Select the correct pool
        # Normalize direction (handles 'buy'/'bullish' and 'sell'/'bearish')
        target_pool = "bullish" if direction.lower() in ["buy", "bullish"] else "bearish"
        target_obs = ob_cache.get(target_pool, [])
        
        if not target_obs:
            return None

        nearest = None
        min_weighted_dist = float('inf')
        now = time.time()

        # Relevance Horizon: Ignore OBs further than 4 ATRs away
        max_dist = (atr * 4.0) if atr else float('inf')

        for ob in target_obs:
            # --- Bullish OB (Look for Support BELOW price) ---
            if target_pool == "bullish":
                level = ob["top"]
                if level < current_price:
                    raw_dist = current_price - level
                else: continue # Skip if price is already below the OB

            # --- Bearish OB (Look for Resistance ABOVE price) ---
            else:
                level = ob["bottom"]
                if level > current_price:
                    raw_dist = level - current_price
                else: continue # Skip if price is already above the OB

            # 2. VALIDATION & WEIGHTING
            if raw_dist > max_dist:
                continue

            # Strength Multiplier: A high-strength OB (massive breakout) 
            # effectively 'pulls' the bot's attention more.
            strength = ob.get("strength", 1.0)
            strength_bonus = 0.7 if strength > 1.5 else 1.0
            
            # Freshness Multiplier: Give a slight edge to OBs formed in the last 2 hours
            age_minutes = (now - ob.get("timestamp", now)) / 60
            age_bonus = 0.9 if age_minutes < 120 else 1.0

            weighted_dist = raw_dist * strength_bonus * age_bonus

            # 3. SELECTION
            if weighted_dist < min_weighted_dist:
                min_weighted_dist = weighted_dist
                nearest = ob

        return nearest

    except Exception as e:
        print(f"❌ Error in get_nearest_ob: {e}")
        return None

# ================= SMC ENGINE: LIQUIDITY POOLS & SWEEPS =================

def detect_liquidity_pools(df, lookback=60, is_training=False):
    """
    Identifies Equal Highs (EQH) and Equal Lows (EQL).
    V4.0 Fix: Added 'timestamp' for freshness and 'strength' for weighting.
    """
    global liquidity_pools
    pools = {"high": [], "low": []}
    now = time.time()

    try:
        if len(df) < lookback: return pools
        
        df = df.copy().reset_index(drop=True)
        # ATR-based threshold for "equality" (0.15x ATR is optimal for Gold)
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        threshold = atr * 0.15 

        df_slice = df.tail(lookback).copy()
        
        # 1. Pivot Detection (Fractals)
        highs = df_slice['high'].values
        lows = df_slice['low'].values
        
        # We need an array of booleans the same size as highs
        is_pivot_h = np.zeros(len(highs), dtype=bool)
        is_pivot_l = np.zeros(len(lows), dtype=bool)
        
        # Fast array iteration
        PIVOT_BARS = 3   # require N candles each side to confirm pivot
                         # Filters noise — real swing highs/lows resist multiple candles

        for k in range(PIVOT_BARS, len(highs) - PIVOT_BARS):
            # Pivot high: must be highest of ALL surrounding 2*PIVOT_BARS+1 candles
            if all(highs[k] > highs[k-j] for j in range(1, PIVOT_BARS+1)) and \
               all(highs[k] > highs[k+j] for j in range(1, PIVOT_BARS+1)):
                is_pivot_h[k] = True
            # Pivot low: must be lowest of ALL surrounding candles
            if all(lows[k] < lows[k-j] for j in range(1, PIVOT_BARS+1)) and \
               all(lows[k] < lows[k+j] for j in range(1, PIVOT_BARS+1)):
                is_pivot_l[k] = True
                
        pivots_h = df_slice[is_pivot_h]
        pivots_l = df_slice[is_pivot_l]

        # 2. EQH Detection (Buy-Side Liquidity)
        for i in range(len(pivots_h)):
            for j in range(i + 1, len(pivots_h)):
                p1, p2 = pivots_h.iloc[i], pivots_h.iloc[j]
                
                # Check if prices are "equal" within the ATR threshold
                if abs(p1['high'] - p2['high']) <= threshold:
                    # Logic: Ensure the level hasn't been swept (broken) yet
                    if df['high'].iloc[int(p1.name):].max() <= max(p1['high'], p2['high']) + threshold:
                        pools["high"].append({
                            "price": max(p1['high'], p2['high']), 
                            "index": int(p1.name),
                            "timestamp": now,        # <--- FIX: Added for Freshness
                            "strength": 2            # <--- FIX: EQH is high strength
                        })
                        break

        # 3. EQL Detection (Sell-Side Liquidity)
        for i in range(len(pivots_l)):
            for j in range(i + 1, len(pivots_l)):
                p1, p2 = pivots_l.iloc[i], pivots_l.iloc[j]
                
                if abs(p1['low'] - p2['low']) <= threshold:
                    # Ensure it hasn't been swept yet
                    if df['low'].iloc[int(p1.name):].min() >= min(p1['low'], p2['low']) - threshold:
                        pools["low"].append({
                            "price": min(p1['low'], p2['low']), 
                            "index": int(p1.name),
                            "timestamp": now,        # <--- FIX: Added for Freshness
                            "strength": 2            # <--- FIX: EQL is high strength
                        })
                        break

        # Update global cache (Keeping only the 5 most recent zones per side)
        if not is_training:
            liquidity_pools["high"] = pools["high"][-5:]
            liquidity_pools["low"] = pools["low"][-5:]
        
        return pools

    except Exception as e:
        print(f"❌ Liquidity detection error: {e}")
        return pools

def detect_liquidity_sweep(df, lookback=20, min_sweep=0.3):
    """
    Detects if the last 5 bars swept a recent liquidity level and rejected.

    A valid sweep requires:
      - Price broke beyond the liquidity wall (recent high/low) by at least
        min_sweep × ATR — filters noise wicks that barely graze the level
      - The CLOSE of the last bar is back inside the wall — confirms rejection
        and a true reversal intent, not a breakout continuation

    Returns "buy_sweep", "sell_sweep", or None.
    """
    try:
        if len(df) < 30:
            return None
        h, l, c = df["high"].values, df["low"].values, df["close"].values

        # Liquidity wall: extremes from the lookback window (excluding last 5 bars)
        recent_high = h[-(lookback + 5):-5].max()
        recent_low  = l[-(lookback + 5):-5].min()

        # Sweep wick: extremes of the last 5 bars
        max_high = h[-5:].max()
        min_low  = l[-5:].min()

        atr      = float((df['high'] - df['low']).rolling(14).mean().iloc[-1])
        min_dist = atr * min_sweep   # was hardcoded 0.2, now uses parameter

        # Bullish sweep: wick broke below sell-side liquidity, close recovered above
        if (recent_low - min_low) > min_dist and c[-1] > recent_low:
            return "buy_sweep"

        # Bearish sweep: wick broke above buy-side liquidity, close retreated below
        if (max_high - recent_high) > min_dist and c[-1] < recent_high:
            return "sell_sweep"

        return None
    except Exception:
        return None

def get_nearest_liquidity_level(direction, current_price, atr=None, max_age_minutes=180):
    """
    Finds the most relevant Liquidity Pool to act as a TP target.
    Optimized for XAUUSD to prioritize Freshness, Strength, and Proximity.
    """
    global liquidity_pools
    direction = direction.lower()

    try:
        # 1. SELECT POOL TYPE
        pool_key = "high" if direction == "buy" else "low"
        all_pools = liquidity_pools.get(pool_key, [])

        if not all_pools:
            return None

        nearest = None
        min_weighted_dist = float("inf")
        now = time.time()

        # 2. SYMBOL INFO SAFETY
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            return None
        
        # 3. SET DYNAMIC MIN DISTANCE (Use ATR if available, else 50 points)
        # We don't want a TP that is basically on top of our entry.
        min_valid_dist = (atr * 0.15) if atr else (symbol_info.point * 50)

        for pool in all_pools:
            level_price = pool["price"]

            # --- Freshness filter ---
            # Fallback to 'now' if timestamp is missing to avoid errors
            pool_time = pool.get("timestamp", now)
            age_minutes = (now - pool_time) / 60
            if age_minutes > max_age_minutes:
                continue

            # --- Proximity Check (Directional) ---
            if direction == "buy":
                if level_price <= current_price: continue
                raw_dist = level_price - current_price
            else: # sell
                if level_price >= current_price: continue
                raw_dist = current_price - level_price

            # --- Final Distance Validation ---
            if raw_dist < min_valid_dist:
                continue

            # --- Strength Weighting ---
            # A 'strength' of 2 (Equal Highs) makes the level 20% 'closer' in priority
            strength = pool.get("strength", 1)
            strength_bonus = 0.8 if strength >= 2 else 1.0
            
            weighted_dist = raw_dist * strength_bonus

            # --- Select Nearest ---
            if weighted_dist < min_weighted_dist:
                min_weighted_dist = weighted_dist
                nearest = pool

        return nearest

    except Exception as e:
        print(f"❌ Error in get_nearest_liquidity_level: {e}")
        return None

# ================= VOLUME PROFILE =================

def calculate_volume_profile(df, bins=VOLUME_PROFILE_BINS):
    """Calculate Volume Profile - shows where most trading volume occurred"""
    global volume_profile_data

    if len(df) < 20 or 'tick_volume' not in df.columns:
        return {}

    try:
        price_range = df['high'].max() - df['low'].min()
        if price_range <= 0 or bins <= 0:   # div-by-zero guard
            return {}
        bin_size = price_range / bins

        profile = {}
        low_min = df['low'].min()

        for i in range(len(df)):
            price  = (df['high'].iloc[i] + df['low'].iloc[i]) / 2
            volume = df['tick_volume'].iloc[i]
            bin_index = int((price - low_min) / bin_size)
            profile[bin_index] = profile.get(bin_index, 0) + volume

        if profile:
            total_vol  = sum(profile.values())
            avg_volume = total_vol / len(profile) if len(profile) > 0 else 0
            if avg_volume == 0:
                return profile

            hvn_levels = [k for k, v in profile.items() if v > avg_volume * 1.3]
            lvn_levels = [k for k, v in profile.items() if v < avg_volume * 0.5]

            volume_profile_data.append({
                "hvn": hvn_levels,
                "lvn": lvn_levels,
                "profile": profile,
                "time": datetime.now()
            })

        return profile

    except Exception as e:
        print(f"Volume profile error: {e}")
        return {}

# ================= LIQUIDITY HEATMAP =================

MAX_HEATMAP = 200

def update_liquidity_heatmap(df):
    """Update liquidity heatmap"""
    global liquidity_heatmap

    try:
        if len(df) < 20:
            return

        heatmap_entry = {
            "high": float(df['high'].tail(10).max()),
            "low": float(df['low'].tail(10).min()),
            "close": float(df['close'].iloc[-1]),
            "volume": float(df['tick_volume'].tail(10).sum()) if 'tick_volume' in df.columns else 0,
            "time": str(df['time'].iloc[-1])
        }

        liquidity_heatmap.append(heatmap_entry)

    except Exception as e:
        print(f"Heatmap update error: {e}")

def get_liquidity_density(price):
    """Calculate liquidity density"""

    global liquidity_heatmap

    if not liquidity_heatmap:
        return 0.0

    try:
        density = 0

        for level in liquidity_heatmap:

            if level["low"] <= price <= level["high"]:
                density += level["volume"]

        return density / len(liquidity_heatmap)

    except Exception:
        return 0.0


# ================= NEWS FILTER CONFIG =================
# (NEWS_URL, NEWS_CACHE, NEWS_FILTER_MINUTES_BEFORE/AFTER, NEWS_LOG_COOLDOWN defined in config above)
LAST_NEWS_UPDATE = 0
NEWS_UPDATE_INTERVAL = 600   # 10 minutes

def fetch_news_calendar():
    global NEWS_CACHE, LAST_NEWS_UPDATE

    now_ts = time.time()

    # Return cache if it's still fresh
    if NEWS_CACHE and (now_ts - LAST_NEWS_UPDATE < NEWS_UPDATE_INTERVAL):
        return NEWS_CACHE

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = None
        
        # Retry logic
        for _ in range(3):
            try:
                response = requests.get(NEWS_URL, headers=headers, timeout=10)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                time.sleep(2)

        if response is None or response.status_code != 200:
            return NEWS_CACHE

        root = ET.fromstring(response.content)
        events = []

        ny_tz = pytz.timezone("America/New_York")

        for event in root.findall("event"):
            try:
                currency = event.find("currency").text
                impact = event.find("impact").text
                title = event.find("title").text
                date_str = event.find("date").text
                time_str = event.find("time").text

                if not all([currency, impact, title, date_str, time_str]):
                    continue

                if time_str in ["All Day", "Tentative"]:
                    continue

                # Parse NY time from XML
                naive_time = datetime.strptime(
                    f"{date_str} {time_str}", 
                    "%m-%d-%Y %I:%M%p"
                )
                
                # Localize to NY and convert to UTC
                event_time_utc = ny_tz.localize(naive_time).astimezone(pytz.utc)

                events.append({
                    "currency": currency.strip(),
                    "impact": impact.strip(),
                    "event": title.strip(),
                    "time": event_time_utc
                })
            except (AttributeError, ValueError):
                continue

        NEWS_CACHE = events
        LAST_NEWS_UPDATE = now_ts
        return events

    except Exception as e:
        print(f"⚠️ News fetch error: {e}")
        # If fetch fails, don't clear cache, but allow retry sooner
        if not NEWS_CACHE:
            LAST_NEWS_UPDATE = now_ts - NEWS_UPDATE_INTERVAL
        return NEWS_CACHE

def is_news_time(min_impact="Medium"):
    """Check if current UTC time is inside a news event window"""
    
    # Use UTC-aware "now" to match event_time
    now_utc = datetime.now(timezone.utc)
    
    impact_priority = {"High": 3, "Medium": 2, "Low": 1}
    min_level = impact_priority.get(min_impact.capitalize(), 2)

    for news in fetch_news_calendar():
        # Only USD news affects gold/major pairs usually
        if news["currency"] != "USD":
            continue

        level = impact_priority.get(news["impact"], 1)
        if level < min_level:
            continue

        event_time = news["time"]
        before_window = event_time - timedelta(minutes=NEWS_FILTER_MINUTES_BEFORE)
        after_window = event_time + timedelta(minutes=NEWS_FILTER_MINUTES_AFTER)

        if before_window <= now_utc <= after_window:
            return True, news["event"]

    return False, None

def news_filter_active(min_impact="Medium"):
    """Returns True if trading should be blocked"""
    global LAST_NEWS_LOG

    active, event = is_news_time(min_impact)

    if active:
        if time.time() - LAST_NEWS_LOG > NEWS_LOG_COOLDOWN:
            print(f"🛑 NEWS FILTER ACTIVE: {event}")
            LAST_NEWS_LOG = time.time()
        return True

    return False

def get_next_high_impact_news():
    """Return hours until next high impact USD news"""
    now_utc = datetime.now(timezone.utc)

    min_hours = float("inf")
    next_event = None

    for news in fetch_news_calendar():
        if news["currency"] != "USD" or news["impact"] != "High":
            continue

        event_time = news["time"]
        
        if event_time < now_utc:
            continue

        hours_until = (event_time - now_utc).total_seconds() / 3600

        if hours_until < min_hours:
            min_hours = hours_until
            next_event = news["event"]

    return (min_hours, next_event) if next_event else (None, None)

# ================= LIVE NEWS API INTEGRATION =================
NEWS_CACHE_DURATION = 300
NEWS_API_TIMEOUT = 10

# (NEWS_FILTER_MINUTES_BEFORE/AFTER already defined in main config above)

live_news_cache = {
    "data": [],
    "timestamp": 0,
    "last_fetch": None
}

def fetch_live_news_api():
    global live_news_cache
    now_ts = time.time()

    # 1. CACHE CHECK
    if live_news_cache["data"] and (now_ts - live_news_cache["timestamp"]) < NEWS_CACHE_DURATION:
        return live_news_cache["data"]

    try:
        # fetch_news_calendar() was defined in the previous step
        calendar_events = fetch_news_calendar() 

        news_events = []
        for event in calendar_events:
            # We store the time as an ISO string to keep the cache serializable
            event_time = event.get("time")
            news_events.append({
                "title": event.get("event", ""),
                "pubDate": event_time.isoformat() if event_time else None,
                "impact": event.get("impact", "LOW").upper(),
                "currency": event.get("currency", ""),
                "source": "forexfactory_xml"
            })

        # 3. UPDATE CACHE ONLY ON SUCCESS
        live_news_cache["data"] = news_events
        live_news_cache["timestamp"] = now_ts
        live_news_cache["last_fetch"] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        # 4. SILENT FAIL: Keep old data so the bot stays safe
        print(f"📡 News Server Busy: {e}. Keeping last known calendar.")
        return live_news_cache["data"]

    return live_news_cache["data"]

def parse_news_event_time(date_string):
    if not date_string:
        return None
    try:
        # fromisoformat handles the 'Z' or '+00:00' if present
        dt = datetime.fromisoformat(date_string)
        # Ensure it is treated as UTC if it's naive
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def get_live_news_filter_status():
    """Combined live + calendar news filter"""
    live_news = fetch_live_news_api()

    result = {
        "is_active": False,
        "current_event": None,
        "upcoming_events": [],
        "source": "forexfactory_xml",
        "last_updated": live_news_cache.get("last_fetch")
    }

    # CRITICAL: Use timezone-aware 'now' to match the UTC event times
    now = datetime.now(timezone.utc)

    for event in live_news:
        if event.get("currency") != "USD":
            continue

        if event.get("impact") != "HIGH":
            continue

        event_time = parse_news_event_time(event.get("pubDate"))
        if not event_time:
            continue

        before_window = event_time - timedelta(minutes=NEWS_FILTER_MINUTES_BEFORE)
        after_window = event_time + timedelta(minutes=NEWS_FILTER_MINUTES_AFTER)

        # Active news check
        if before_window <= now <= after_window:
            result["is_active"] = True
            result["current_event"] = event.get("title")
            # Don't break yet if you want to collect upcoming events too, 
            # but for 'is_active', we found what we need.

        # Upcoming events (next 24 hours)
        if event_time > now:
            hours_until = (event_time - now).total_seconds() / 3600
            if hours_until < 24:
                result["upcoming_events"].append({
                    "title": event.get("title"),
                    "time": event_time.isoformat(),
                    "impact": event.get("impact"),
                    "hours_until": hours_until
                })

    # Sort by nearest event first
    result["upcoming_events"] = sorted(
        result["upcoming_events"],
        key=lambda x: x["hours_until"]
    )[:5]

    return result

def get_enhanced_news_signal():
    """Final news filter used by trading engine"""
    live_status = get_live_news_filter_status()

    # 1. Block if news is currently active
    if live_status["is_active"]:
        return True, f"🚫 News Active: {live_status['current_event']}", 0.95

    # 2. Block if high impact news is coming in less than 1 hour
    for event in live_status.get("upcoming_events", []):
        if event.get("hours_until", 999) < 1:
            return True, f"⚠️ Upcoming News: {event.get('title')}", 0.80

    return False, "✅ News Clear", 0.0

# ================= MARKET REGIME & TREND =================

def detect_market_regime(df, fast_ema=9, slow_ema=21): # Changed from 20/50
    """Detect market regime using EMAs + Structure Confirmation."""
    try:
        # Faster EMAs eliminate the 30 minute M5 lag
        ema_fast = df["close"].ewm(span=fast_ema, adjust=False).mean().iloc[-1]
        ema_slow = df["close"].ewm(span=slow_ema, adjust=False).mean().iloc[-1]

        recent_high = df['high'].iloc[-10:-1].max() # Look closer
        recent_low  = df['low'].iloc[-10:-1].min()
        current_close = df['close'].iloc[-1]

        if ema_fast > ema_slow and current_close > recent_low:
            return "bullish"
        elif ema_fast < ema_slow and current_close < recent_high:
            return "bearish"
        else:
            return "range"

    except Exception as e:
        print(f"Market regime detection error: {e}")
        return "range"

def detect_mtf_trend(mtf_data):
    """
    Detect trend across MTF with Conflict Detection.
    Optimized for XAUUSD: H4/H1 sets the Narrative, M15/M5 sets the Entry.
    """
    trends = {}
    # Weighted Importance: Narrative (H4/H1) vs. Execution (M15/M5)
    weights = {"H4": 3, "H1": 2, "M15": 1, "M5": 1}
    score = 0
    total_weight = 0

    try:
        for tf, df in mtf_data.items():
            # Only process raw DataFrame timeframes — skip enriched keys like
            # "M5_trend", "overall_trend", "trend_score" which are strings/floats
            if not isinstance(df, pd.DataFrame) or len(df) < 20:
                continue
                
            regime = detect_market_regime(df)
            trends[tf] = regime
            
            weight = weights.get(tf, 1)
            total_weight += weight
            
            if regime == "bullish":
                score += weight
            elif regime == "bearish":
                score -= weight

        if total_weight == 0:
            return "neutral", 0.0

        # Calculate Raw Strength
        strength = abs(score) / total_weight

        # ======================================================
        # SMC ALIGNMENT CHECK (The "Secret Sauce")
        # ======================================================
        h1_trend = trends.get("H1")
        m15_trend = trends.get("M15")
        h4_trend = trends.get("H4")

        # 1. High Probability Setup: H4 + H1 + M15 all agree
        if h4_trend == h1_trend == m15_trend and h4_trend is not None:
            strength += 0.2  # "Conviction Bonus"
        
        # 2. Conflict Check: If HTF (H4) opposes LTF (M15)
        # This usually means a deep retracement is happening.
        if h4_trend and m15_trend and h4_trend != m15_trend:
            strength -= 0.15 # "Conflict Penalty"

        # Final Clamp for strength (0.0 to 1.0)
        strength = max(0.0, min(1.0, strength))

        # ======================================================
        # FINAL OUTPUT
        # ======================================================
        if score > 0.5: # Slightly higher threshold for Bullish
            return "bullish", strength
        elif score < -0.5:
            return "bearish", strength
        else:
            return "range", strength

    except Exception as e:
        print(f"❌ MTF Trend Error: {e}")
        return "neutral", 0.0

# ================= AI FEATURES =================

def generate_smc_features(df, mtf_data=None, news_active=False, is_training=False):
    """
    SMC Feature Engineering — 38 features (26 original + 10 precision + 2 OB distance features).
    """
    try:
        if df is None or len(df) < 50:
            return None

        df = df.copy().reset_index(drop=True)
        last = len(df) - 1

        close = df['close']
        high  = df['high']
        low   = df['low']
        open_ = df['open']
        curr_p = close.iloc[last]

        atr_series = (high - low).rolling(14).mean()
        curr_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 1e-5
        
        # Enforce minimum bound safely. 
        # XAUUSD minimum typical M5 ATR is ~0.20 ($0.20c).
        curr_atr = max(curr_atr, 1e-5)

        features = {}

        features['relative_range'] = (high.iloc[last] - low.iloc[last]) / curr_atr
        features['body_ratio']     = abs(close.iloc[last] - open_.iloc[last]) / curr_atr

        momentum = (close.iloc[last] - close.iloc[last-10]) / curr_atr

        features['momentum_bull']  = max(momentum, 0)
        features['momentum_bear']  = abs(min(momentum, 0))

        sma50 = close.rolling(50).mean().iloc[last]
        features['trend_bullish'] = 1.0 if curr_p > sma50 else 0.0
        features['trend_bearish'] = 1.0 if curr_p < sma50 else 0.0

        prev_high = high.iloc[last-21:last].max()
        prev_low  = low.iloc[last-21:last].min()
        features['bos_bull'] = 1.0 if close.iloc[last] > prev_high else 0.0
        features['bos_bear'] = 1.0 if close.iloc[last] < prev_low  else 0.0

        fvgs = detect_fvg(df) or {}
        # 🔥 FIX: Pass the training flag safely
        obs  = detect_order_blocks(df, is_training=is_training) or {}

        def is_in_zone(zones, price):
            if not zones: return 0.0
            return 1.0 if any(z["bottom"] <= price <= z["top"] for z in zones) else 0.0

        features['in_bullish_fvg'] = is_in_zone(fvgs.get("bullish", []), curr_p)
        features['in_bearish_fvg'] = is_in_zone(fvgs.get("bearish", []), curr_p)
        features['in_bullish_ob']  = is_in_zone(obs.get("bullish", []), curr_p)
        features['in_bearish_ob']  = is_in_zone(obs.get("bearish", []), curr_p)

        # 🔥 FIX: Pass the training flag safely
        liquidity  = detect_liquidity_pools(df, is_training=is_training) or {}
        
        high_pools = [h['price'] for h in liquidity.get("high", [])]
        low_pools  = [l['price'] for l in liquidity.get("low",  [])]
        features['dist_to_upper_liq'] = (min(high_pools, key=lambda x: abs(x - curr_p)) - curr_p) / curr_atr if high_pools else 0.0
        features['dist_to_lower_liq'] = (curr_p - min(low_pools, key=lambda x: abs(x - curr_p))) / curr_atr if low_pools else 0.0

        sweep = detect_liquidity_sweep(df)
        features['liq_sweep_low']  = 1.0 if sweep == "buy_sweep"  else 0.0
        features['liq_sweep_high'] = 1.0 if sweep == "sell_sweep" else 0.0

        if mtf_data:
            features['mtf_bullish'] = 1.0 if mtf_data.get("M15_trend") == "bullish" else 0.0
            features['mtf_bearish'] = 1.0 if mtf_data.get("M15_trend") == "bearish" else 0.0
            features['mtf_score']   = float(mtf_data.get("trend_score", 0))
        else:
            features.update({'mtf_bullish': 0.0, 'mtf_bearish': 0.0, 'mtf_score': 0.0})

        features['news_active'] = 1.0 if news_active else 0.0

        try:
            raw_spread = get_spread()
            features['spread'] = float(np.clip(raw_spread / MAX_SPREAD, 0.0, 1.0)) if raw_spread is not None else 0.5
        except Exception:
            features['spread'] = 0.5

        # 1. INITIALIZE CONFLUENCE FIRST
        features['bullish_confluence'] = (
            0.5 * features['trend_bullish'] +
            2.0 * features['in_bullish_fvg'] +   # heavily reward buying in discount logic
            2.0 * features['in_bullish_ob'] +    # heavily reward buying in OB
            1.5 * features['liq_sweep_low']
        )
        
        features['bearish_confluence'] = (
            0.5 * features['trend_bearish'] +
            2.0 * features['in_bearish_fvg'] +   # heavily reward shorting in premium logic
            2.0 * features['in_bearish_ob'] +
            1.5 * features['liq_sweep_high']
        )

        # 2. THEN APPLY THE STRUCTURAL PENALTY FOR BREAKOUTS WITHOUT ZONES
        if (features['bos_bull'] == 1.0) and (features['in_bullish_fvg'] == 0.0) and (features['in_bullish_ob'] == 0.0):
             features['bullish_confluence'] -= 2.0
             
        if (features['bos_bear'] == 1.0) and (features['in_bearish_fvg'] == 0.0) and (features['in_bearish_ob'] == 0.0):
             features['bearish_confluence'] -= 2.0

        # 3. CALCULATE REVERSAL AND BIAS
        features['bullish_reversal_setup'] = features['liq_sweep_low']  * features['in_bullish_fvg']
        features['bearish_reversal_setup'] = features['liq_sweep_high'] * features['in_bearish_fvg']
        features['net_bias'] = features['bullish_confluence'] - features['bearish_confluence']

        # ── 10 NEW precision features ─────────────────────────────────────────

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean().iloc[last]
        loss  = (-delta.clip(upper=0)).rolling(14).mean().iloc[last]
        
        # Ensure loss is not NaN or zero
        if pd.isna(loss) or loss < 1e-9:
             features['rsi_14'] = 1.0 # Max RSI (100 normalized to 1.0)
        else:
             rs = gain / loss
             features['rsi_14'] = float(np.clip(100 - (100 / (1 + rs)), 0, 100) / 100.0)

        # 🔥 FIX: Isolate Market Structure so background training doesn't break live trend
        if is_training:
            temp_ms = MarketStructure()
            ms_analysis = temp_ms.analyze(df)
        else:
            ms_analysis = market_structure.analyze(df)
            
        choch = ms_analysis.get("choch")
        features['choch_bull'] = 1.0 if choch == "bullish" else 0.0
        features['choch_bear'] = 1.0 if choch == "bearish" else 0.0

        if mtf_data:
            h1_trend = mtf_data.get("H1_trend", "range")
            h4_trend = mtf_data.get("H4_trend", "range")
            m5_trend = ms_analysis.get("trend", "range")
            bull_align = sum([h1_trend == "bullish", h4_trend == "bullish", m5_trend == "bullish"])
            bear_align = sum([h1_trend == "bearish", h4_trend == "bearish", m5_trend == "bearish"])
            features['htf_bull_align'] = bull_align / 3.0
            features['htf_bear_align'] = bear_align / 3.0
        else:
            features['htf_bull_align'] = 0.0
            features['htf_bear_align'] = 0.0

        bull_fvgs = fvgs.get("bullish", [])
        bear_fvgs = fvgs.get("bearish", [])
        features['fvg_strength_bull'] = max((f.get("strength", 0) for f in bull_fvgs), default=0.0)
        features['fvg_strength_bear'] = max((f.get("strength", 0) for f in bear_fvgs), default=0.0)

        if 'tick_volume' in df.columns:
            vol_mean = df['tick_volume'].rolling(20).mean().iloc[last]
            vol_curr = df['tick_volume'].iloc[last]
            features['vol_ratio'] = float(np.clip(vol_curr / vol_mean if vol_mean > 0 else 1.0, 0, 5) / 5.0)
        else:
            features['vol_ratio'] = 0.5

        ema20 = close.ewm(span=20).mean().iloc[last]
        ema50 = close.ewm(span=50).mean().iloc[last]
        features['ema_cross'] = 1.0 if ema20 > ema50 else 0.0

        if features['bos_bull'] == 1.0:
            features['bos_strength'] = float(np.clip((close.iloc[last] - prev_high) / curr_atr, 0, 3) / 3.0)
        elif features['bos_bear'] == 1.0:
            features['bos_strength'] = float(np.clip((prev_low - close.iloc[last]) / curr_atr, 0, 3) / 3.0)
        else:
            features['bos_strength'] = 0.0

        nearest_bull_ob = get_nearest_ob("bullish", curr_p)
        nearest_bear_ob = get_nearest_ob("bearish", curr_p)
        features['dist_to_bull_ob'] = float(np.clip(abs(curr_p - nearest_bull_ob["mid"]) / curr_atr if nearest_bull_ob else 5.0, 0, 5) / 5.0)
        features['dist_to_bear_ob'] = float(np.clip(abs(curr_p - nearest_bear_ob["mid"]) / curr_atr if nearest_bear_ob else 5.0, 0, 5) / 5.0)

        # ── Regime features (ADX-based) ───────────────────────────────────────
        regime_feats = compute_regime_features(df)
        features.update(regime_feats)

        # ── Temporal + volatility features ───────────────────────────────────
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                ts = df.index[-1]
            elif 'time' in df.columns:
                ts = pd.Timestamp(df['time'].iloc[-1])
            else:
                ts = None

            if ts is not None:
                hour = ts.hour + ts.minute / 60.0
                dow  = float(ts.dayofweek)   # 0=Mon … 4=Fri (6=Sun rarely, gold 24/5)
                features['hour_sin'] = float(np.sin(2 * np.pi * hour / 24))
                features['hour_cos'] = float(np.cos(2 * np.pi * hour / 24))
                features['dow_sin']  = float(np.sin(2 * np.pi * dow  / 5))
                features['dow_cos']  = float(np.cos(2 * np.pi * dow  / 5))
            else:
                features['hour_sin'] = features['hour_cos'] = 0.0
                features['dow_sin']  = features['dow_cos']  = 0.0
        except Exception:
            features['hour_sin'] = features['hour_cos'] = 0.0
            features['dow_sin']  = features['dow_cos']  = 0.0

        try:
            atr_hist = (df['high'] - df['low']).rolling(14).mean().dropna()
            if len(atr_hist) >= 10:
                features['vol_percentile'] = float(
                    np.mean(atr_hist.iloc[-100:].values <= curr_atr)
                )
            else:
                features['vol_percentile'] = 0.5
        except Exception:
            features['vol_percentile'] = 0.5

        return features

    except Exception as e:
        print(f"❌ Feature Error: {e}")
        return None

# ================= SENTIMENT ANALYSIS =================
# Analyzes market sentiment from news and social media

# Sentiment configuration
SENTIMENT_API_KEY = ""  # Optional: Add API key for sentiment APIs
SENTIMENT_CACHE_DURATION = 300  # 5 minutes
SENTIMENT_LOOKBACK = 20  # Bars to analyze sentiment impact

# Sentiment cache
sentiment_cache = {
    "score": 0.0,  # -1 (bearish) to +1 (bullish)
    "timestamp": 0,
    "confidence": 0.0
}

def analyze_sentiment_from_news():
    """
    Analyze market sentiment from recent news headlines

    Returns:
        dict: {"sentiment": float, "confidence": float, "sources": list}
    """
    global sentiment_cache

    current_time = time.time()

    # Check cache
    if (current_time - sentiment_cache["timestamp"]) < SENTIMENT_CACHE_DURATION:
        return sentiment_cache

    try:
        # Get live news
        live_news = fetch_live_news_api()

        if not live_news:
            # Use existing news signal
            live_status = get_live_news_filter_status()
            if live_status.get("upcoming_events"):
                # Use upcoming events to gauge sentiment
                positive_keywords = ["bullish", "rise", "gain", "positive", "growth", "optimistic"]
                negative_keywords = ["bearish", "fall", "drop", "negative", "recession", "concern"]

                score = 0
                count = 0

                for event in live_status.get("upcoming_events", [])[:10]:
                    title = event.get("title", "").lower()
                    if any(kw in title for kw in positive_keywords):
                        score += 1
                    elif any(kw in title for kw in negative_keywords):
                        score -= 1
                    count += 1

                if count > 0:
                    sentiment_score = score / count
                    confidence = min(abs(sentiment_score), 1.0)

                    sentiment_cache = {
                        "score": sentiment_score,
                        "timestamp": current_time,
                        "confidence": confidence
                    }

        # If no sentiment data, return neutral
        if sentiment_cache["confidence"] == 0:
            sentiment_cache = {
                "score": 0.0,
                "timestamp": current_time,
                "confidence": 0.0
            }

    except Exception as e:
        print(f"Sentiment analysis error: {e}")

    return sentiment_cache

def get_sentiment_signal():
    """
    Get trading signal based on market sentiment

    Returns:
        tuple: (direction: str or None, confidence: float)
    """
    sentiment = analyze_sentiment_from_news()

    score = sentiment.get("score", 0)
    confidence = sentiment.get("confidence", 0)

    # Only generate signal if confidence is high enough
    if confidence > 0.6:
        if score > 0.3:
            return "buy", confidence
        elif score < -0.3:
            return "sell", confidence

    return None, confidence

# ==========================================================
# MARKET STRUCTURE (BOS/CHoCH) MODULE
# ==========================================================
RECENT_WINDOW = 30  # tuned for M5

class MarketStructure:
    def find_swing_points(self, df):
        if len(df) < 5:
            return [], []

        highs = df['high'].values
        lows = df['low'].values
        
        swing_highs = []
        swing_lows = []

        # 5-Bar Fractal
        for i in range(2, len(df) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append({"price": highs[i], "index": i})
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append({"price": lows[i], "index": i})

        return swing_highs, swing_lows

    def analyze(self, df):
        # Stateless analysis!
        res = {
            "trend":      "range",
            "tf":         "M5",
            "bos":        None,
            "choch":      None,
            "signal":     "WAIT",
            "support":    None,
            "resistance": None,
            "confidence": 0.5
        }

        try:
            sh, sl = self.find_swing_points(df)
            if len(sh) < 2 or len(sl) < 2: return res

            # Use the last two established swing points to determine structure
            sh1, sh2 = sh[-2]['price'], sh[-1]['price']
            sl1, sl2 = sl[-2]['price'], sl[-1]['price']
            
            res["resistance"] = sh2
            res["support"] = sl2

            curr_close = df['close'].iloc[-1]

            # True SMC Trend Definition - Higher Highs + Higher Lows
            if sh2 > sh1 and sl2 > sl1:
                res["trend"] = "bullish"
                if curr_close < sl2: # Structural failure
                    res["choch"] = "bearish"
                    res["trend"] = "bearish"
                elif curr_close > sh2:
                    res["bos"] = "bullish"
            
            # Lower Highs + Lower Lows
            elif sh2 < sh1 and sl2 < sl1:
                res["trend"] = "bearish"
                if curr_close > sh2: # Structural failure
                    res["choch"] = "bullish"
                    res["trend"] = "bullish"
                elif curr_close < sl2:
                    res["bos"] = "bearish"

            if res["trend"] == "bullish":
                res["signal"] = "BUY"
                res["confidence"] = 0.80
            elif res["trend"] == "bearish":
                res["signal"] = "SELL"
                res["confidence"] = 0.80

            return res

        except Exception as e:
            print(f"❌ MS Error: {e}")
            return res

# Initialize the global instance
market_structure = MarketStructure()

def get_mss_signal(df):
    """Wrapper used by the Main Loop and AI logic"""
    try:
        if df is None or len(df) < 50:
            return "WAIT", 0.0, {}
        analysis = market_structure.analyze(df)
        return analysis["signal"], analysis["confidence"], analysis
    except Exception as e:
        print(f"⚠️ MSS signal error: {e}")
        return "WAIT", 0.0, {}

# ================= AI MODELS =================

def initialize_ai(live_df=None):
    global AI_MODEL, last_train

    print("🧠 Initializing SMC AI models...")
    now = datetime.now()

    # =============================
    # 1. READ STATE SAFELY
    # =============================
    # We only lock the thread while reading the file, then we unlock it immediately!
    with model_lock:
        model_loaded = load_model_state()
        last_trained = None
        if model_loaded and last_train:
            last_trained = datetime.fromtimestamp(last_train)

    # =============================
    # 2. DECISION LOGIC & TRAINING
    # =============================
    if model_loaded and last_trained:
        if now - last_trained < timedelta(hours=RETRAIN_HOURS):
            print("✅ AI model is fresh → loading")
            AI_MODEL = True
            return
        else:
            print("♻️ Model outdated → retraining")
    else:
        print("🧠 No model → training from scratch")

    df = live_df if (live_df is not None and len(live_df) > 2000) else load_market_history()

    if df is not None and len(df) > 2000:
        train_models(df)
        AI_MODEL = True
        print("✅ AI training completed")
    else:
        print(f"❌ Not enough data to train AI (have {len(df) if df is not None else 0} candles, need 2000+)")
        print("   Bot will retrain automatically once market history accumulates.")

def ai_decision(df, mtf_data, news_active):
    try:
        # Guard: don't predict if model is not ready
        if not AI_MODEL or not hasattr(scaler_entry, "n_features_in_"):
            return "HOLD", 0.5, 0.5

        features = generate_smc_features(df, mtf_data, news_active)

        if not features or sum(features.values()) == 0:
            print("⚠️ Invalid features → skipping prediction")
            return "HOLD", 0.5, 0.5

        feature_vector = np.array([features.get(name, 0.0) for name in FEATURE_NAMES]).reshape(1, -1)

        with model_lock:
            if feature_vector.shape[1] != scaler_entry.n_features_in_:
                print(f"⚠️ Feature shape mismatch")
                return "HOLD", 0.0, 0.0

            X = scaler_entry.transform(feature_vector)
            probs = entry_model.predict_proba(X)[0]

            # 🔥 THE BUG FIX: Dynamically find probabilities by class label
            buy_prob = 0.0
            sell_prob = 0.0
            
            for i, class_label in enumerate(entry_model.classes_):
                if class_label == 1:       # BUY LABEL
                    buy_prob = float(probs[i])
                elif class_label == -1:    # SELL LABEL
                    sell_prob = float(probs[i])

        if buy_prob > BUY_THRESHOLD:
            return "BUY", buy_prob, sell_prob
        elif sell_prob > SELL_THRESHOLD:
            return "SELL", buy_prob, sell_prob
        else:
            return "HOLD", buy_prob, sell_prob

    except Exception as e:
        print(f"❌ AI Error: {e}")
        return "HOLD", 0.0, 0.0

def ai_exit_confidence(df, pos, current_price, profit):
    global last_train

    try:
        if exit_model is None or scaler_exit is None or not hasattr(scaler_exit, "n_features_in_"):
            return 0.5

        symbol = mt5.symbol_info(SYMBOL)
        if symbol is None:
            return 0.5

        features_dict = generate_smc_features(df, None, False)

        if features_dict is None:
            return 0.5

        feature_vector = [float(features_dict.get(name, 0.0)) for name in FEATURE_NAMES]

        # Exit-specific features — MUST match train_models exit_extras exactly.
        # ATR for normalization (same formula as training)
        _atr_s   = (df['high'] - df['low']).rolling(14).mean()
        exit_atr = float(_atr_s.iloc[-1]) if not pd.isna(_atr_s.iloc[-1]) else float((df['high'] - df['low']).tail(5).mean() or 1e-5)

        # Convert dollar P&L → price-unit move → normalize by ATR
        contract_size = float(symbol.trade_contract_size) if symbol.trade_contract_size else 100.0
        price_diff    = (profit / (pos.volume * contract_size)) if pos.volume > 0 else 0.0
        if pos.type != mt5.ORDER_TYPE_BUY:
            price_diff = -price_diff  # positive = in-profit direction

        mom_norm  = float(np.clip(price_diff / (exit_atr + 1e-9), -4.0, 4.0))
        dist_norm = float(np.clip(abs(current_price - pos.price_open) / (exit_atr + 1e-9), 0.0, 4.0))
        time_mins = float(np.clip((time.time() - pos.time) / 60.0, 0.0, 360.0))
        side      = 1.0 if pos.type == mt5.ORDER_TYPE_BUY else 0.0

        feature_vector.extend([mom_norm, dist_norm, time_mins, side])

        # Validate shape + predict under lock — prevents race with background retrain
        with model_lock:
            if len(feature_vector) != scaler_exit.n_features_in_:
                return 0.5

            X = scaler_exit.transform(np.array(feature_vector).reshape(1, -1))
            prob = exit_model.predict_proba(X)[0][1]

        return float(np.clip(prob, 0.0, 1.0))

    except Exception as e:
        print(f"AI exit error: {e}")
        return 0.5

def create_entry_target(df: pd.DataFrame,
                        lookahead: int = 12,
                        tp_atr: float = 1.5,
                        sl_atr: float = 1.0) -> tuple:
    """
    Triple-barrier labeling (replaces the old FixedForwardWindow approach).

    For each bar i, three barriers are set:
        Upper barrier  = close[i] + tp_atr  × ATR[i]   → label  1 (buy winner)
        Lower barrier  = close[i] - sl_atr  × ATR[i]   → label -1 (sell winner)
        Time barrier   = bar i + lookahead               → label  0 (unclear)

    Walk bar-by-bar (no lookahead bias) to find WHICH barrier hits first.
    If both barriers hit on the same candle → label 0 (trap / chop).

    Returns
    -------
    target  : pd.Series of int  {-1, 0, 1}
    weights : pd.Series of float — faster hits weighted higher (more decisive)
    """
    close  = df['close'].values
    high   = df['high'].values
    low    = df['low'].values
    n      = len(df)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr_vals = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )

    # Wilder-smooth ATR (pandas rolling mean is fine here)
    atr = pd.Series(tr_vals).rolling(14).mean().values

    labels  = np.zeros(n, dtype=int)
    weights = np.full(n, 0.5)

    for i in range(n - lookahead):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        upper = close[i] + tp_atr * atr[i]
        lower = close[i] - sl_atr * atr[i]
        hit   = 0
        bars  = lookahead

        for j in range(i + 1, min(i + lookahead + 1, n)):
            hit_up = high[j] >= upper
            hit_dn = low[j]  <= lower
            if hit_up and hit_dn:
                hit  = 0    # both hit same candle — trap / indecision
                bars = j - i
                break
            if hit_up:
                hit  = 1
                bars = j - i
                break
            if hit_dn:
                hit  = -1
                bars = j - i
                break

        labels[i]  = hit
        # Faster decisive hits → higher weight (model chases quality)
        if hit != 0:
            weights[i] = float(np.clip(lookahead / bars, 1.0, 4.0))
        else:
            weights[i] = 0.5

    # Zero out last lookahead bars (no future data available)
    labels[-lookahead:]  = 0
    weights[-lookahead:] = 0.5

    return pd.Series(labels, index=df.index), pd.Series(weights, index=df.index)

# ================= AI TRAINING ENGINE =================

def train_models(df):
    global scaler_entry, entry_model, scaler_exit, exit_model, last_train, last_train_candles

    try:
        print(f"🧠 [AI] Training SMC models... (Raw Input: {len(df)})")

        df = df.copy()
        if 'time' in df.columns:
            df = df.set_index('time')

        if len(df) < 2000:
            print("❌ Not enough candles for M5 training (need at least 2000)")
            return

        current_candle_count = len(df)
        print(f"📊 Using {len(df)} candles for training")

        # ── Pre-calculate indicators ──────────────────────────────────────────
        # Use 20-bar lookahead for labels — matches production's avg hold of ~8 bars
        # but gives price time to reach structural targets (avg_win ~215 pips).
        LABEL_LOOKAHEAD = 20
        atr         = (df['high'] - df['low']).rolling(14).mean()
        future_move = df.close.shift(-LABEL_LOOKAHEAD) - df.close

        rows     = []
        indexes  = []
        rr_weights = []  # for sample_weight

        # ── Feature generation ────────────────────────────────────────────────
        for i in tqdm(range(100, len(df) - LABEL_LOOKAHEAD), desc="Generating Features"):

            if pd.isna(future_move.iloc[i]) or pd.isna(atr.iloc[i]):
                continue
            if atr.iloc[i] == 0:
                continue

            window   = df.iloc[i-99:i+1]
            try:
                close_w   = window['close']
                ema20_w   = close_w.ewm(span=20, adjust=False).mean().iloc[-1]
                ema50_w   = close_w.ewm(span=50, adjust=False).mean().iloc[-1]
                m15_proxy = "bullish" if ema20_w > ema50_w else "bearish" if ema20_w < ema50_w else "range"
                ema99_w   = close_w.ewm(span=99, adjust=False).mean().iloc[-1]
                h1_proxy  = "bullish" if ema50_w > ema99_w else "bearish" if ema50_w < ema99_w else "range"
                train_mtf = {
                    "M15_trend": m15_proxy,
                    "H1_trend":  h1_proxy,
                    "trend_score": 0.5
                }
            except Exception:
                train_mtf = None
            features = generate_smc_features(window, mtf_data=train_mtf, news_active=False, is_training=True)

            if not features:
                continue

            vector = [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
            rows.append(vector)
            indexes.append(df.index[i])

            # RR weight: bigger moves get more weight so model chases quality setups
            rr = abs(future_move.iloc[i]) / atr.iloc[i]
            rr_weights.append(float(np.clip(rr, 0.5, 4.0)))

        if len(rows) < 200:
            print("❌ Insufficient valid feature rows generated.")
            return

        X_df = pd.DataFrame(rows, columns=FEATURE_NAMES, index=indexes)
        weights_series = pd.Series(rr_weights, index=indexes)

        # ── Label generation (triple-barrier) ───────────────────────────────
        # Labels aligned with live execution: require 2:1 RR within 24 bars (2h).
        # Previous tp_atr=1.5/lookahead=12 taught the model to fire on 60-min
        # momentum setups, but execution was extended to 72-bar holds with 2:1 floor.
        # This alignment means model fires only when structure supports a 2×ATR move.
        y_entry_all, tb_weights_all = create_entry_target(df, lookahead=24,
                                                           tp_atr=2.0, sl_atr=1.0)

        # Exit: shift(-10) = looks 50 minutes ahead
        future_move_exit = df.close.shift(-10) - df.close
        y_exit_all = np.where(future_move_exit > atr * 0.15, 1,
                      np.where(future_move_exit < -atr * 0.15, 0, np.nan))
        y_exit_all = pd.Series(y_exit_all, index=df.index)

        y_entry    = y_entry_all.loc[X_df.index]
        tb_weights = tb_weights_all.loc[X_df.index]
        y_exit     = y_exit_all.loc[X_df.index]

        if not X_df.index.equals(y_entry.index) or not X_df.index.equals(y_exit.index):
            print("❌ Index misalignment — skipping training")
            return

        valid_mask = (~y_entry.isna()) & (~y_exit.isna())
        X_df      = X_df[valid_mask]
        y_entry   = y_entry[valid_mask].astype(int)
        y_exit    = y_exit[valid_mask].astype(int)
        tb_weights = tb_weights[valid_mask]

        # Training label distribution — confirms if SELL samples are scarce
        unique, counts = np.unique(y_entry.values, return_counts=True)
        print(f"📊 Training label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

        # Combine triple-barrier weights with RR weights — decisiveness × size
        w_entry = (weights_series[valid_mask].values * tb_weights.values).clip(0.5, 4.0)

        if len(X_df) < 200:
            print("❌ Not enough clean samples after filtering")
            return

        # ── Scaling + training ────────────────────────────────────────────────
        X = X_df.values

        atr_vals     = atr.loc[X_df.index].values

        # Exit features — MUST match ai_exit_confidence() exactly.
        # Training uses a 10-bar lagged price change / ATR as a proxy for
        # "current position P&L / ATR". No future data used here.
        close_now    = df['close'].loc[X_df.index].values
        close_lag10  = df['close'].shift(10).loc[X_df.index].values
        close_lag10  = np.where(np.isnan(close_lag10), close_now, close_lag10)

        momentum     = close_now - close_lag10                                          # price move over last 10 bars
        mom_norm     = np.clip(momentum / (atr_vals + 1e-9), -4.0, 4.0)               # ATR-normalized (matches inference)
        dist_norm    = np.clip(np.abs(momentum) / (atr_vals + 1e-9), 0.0, 4.0)       # abs ATR-normalized distance
        time_mins    = np.clip(np.arange(len(X_df), dtype=float) * 5.0, 0.0, 360.0)  # proxy: bar_index × 5 min/bar
        side_vals    = (y_entry.values == 1).astype(float)

        exit_extras  = np.column_stack([mom_norm, dist_norm, time_mins, side_vals])
        X_exit       = np.hstack([X, exit_extras])

        # 🔥 THE LOCK MUST GO HERE!
        from sklearn.utils.class_weight import compute_sample_weight
        
        with model_lock:
            # 1. NEW: Calculate class imbalance penalties to cure 'weak_ai'
            bal_weights_entry = compute_sample_weight(class_weight='balanced', y=y_entry.values)
            bal_weights_exit = compute_sample_weight(class_weight='balanced', y=y_exit.values)
            
            # 2. Combine your custom RR weights with the strong penalties
            final_entry_weights = w_entry * bal_weights_entry
            
            # 3. Update Entry Model
            X_scaled = scaler_entry.fit_transform(X)
            entry_model.fit(X_scaled, y_entry.values, sample_weight=final_entry_weights)

            # 4. Update Exit Model
            X_scaled_exit = scaler_exit.fit_transform(X_exit)
            exit_model.fit(X_scaled_exit, y_exit.values, sample_weight=bal_weights_exit)

            # 5. Save state while safely locked
            last_train         = time.time()
            last_train_candles = current_candle_count
            save_model_state()

        print(f"✅ AI Training Successful | Samples: {len(X_df)} | Features: {len(FEATURE_NAMES)}")
        print(f"📅 Done at: {datetime.now().strftime('%H:%M:%S')}")

    except Exception as e:
        print(f"❌ Training error: {e}")

# ================= DRAWDOWN PROTECTION =================

def update_equity_protection():
    global equity_peak, trading_paused, PAUSE_REASON

    account = mt5.account_info()
    if not account:
        return False

    if trading_paused and PAUSE_REASON == "MAX_DRAWDOWN":
        return False

    drawdown = get_current_drawdown() # Use the safe function we just updated

    if 10 <= drawdown < MAX_DRAWDOWN_PERCENT:
        print(f"⚠️ Warning: Drawdown at {drawdown:.2f}%")

    if drawdown >= MAX_DRAWDOWN_PERCENT:
        trading_paused = True
        PAUSE_REASON = "MAX_DRAWDOWN"

        print(f"\n🚨 FATAL: Drawdown {drawdown:.2f}% → BOT PAUSED for {COOLDOWN_MINUTES} minutes")

        # Send Telegram alert (non-blocking)
        try:
            send_telegram_msg(
                f"🚨 **MAX DRAWDOWN HIT**\n"
                f"Drawdown: `{drawdown:.2f}%` (limit: `{MAX_DRAWDOWN_PERCENT}%`)\n"
                f"Bot paused for `{COOLDOWN_MINUTES}` minutes. Restart manually if needed."
            )
        except Exception:
            pass

        # Non-blocking cooldown — bot sleeps then auto-resumes
        time.sleep(COOLDOWN_MINUTES * 60)

        # ✅ RESET STATE
        trading_paused = False
        PAUSE_REASON = ""

        account = mt5.account_info()
        if account:
            equity_peak = account.equity  # Reset peak after cooldown

        print("✅ Drawdown cooldown finished — trading resumed safely\n")

        return True

def reset_daily_loss():
    global daily_loss_triggered

    daily_stats = load_daily_stats()
    today = datetime.now().strftime("%Y-%m-%d")

    if daily_stats.get("date") != today:
        account = mt5.account_info()

        daily_stats = {
            "date": today,
            "start_balance": account.balance if account else 0,
            "trades": 0,
            "profit": 0.0,
            "loss": 0.0,
            "net_profit": 0.0,
            "daily_loss_triggered": False
        }

        save_daily_stats(daily_stats)
        daily_loss_triggered = False

def check_daily_loss():
    global daily_loss_triggered

    daily_stats = load_daily_stats()

    # ✅ restore flag after restart
    daily_loss_triggered = daily_stats.get("daily_loss_triggered", False)

    if daily_loss_triggered:
        print("⛔ Daily loss limit active — paused until midnight reset")
        return True

    account = mt5.account_info()
    if not account:
        return False

    balance = account.balance
    start_balance = daily_stats.get("start_balance", balance)

    if start_balance <= 0:
        return False

    daily_loss = (start_balance - balance) / start_balance

    if daily_loss >= DAILY_LOSS_LIMIT:
        print(f"🛑 Daily loss hit: {daily_loss:.2%}")

        daily_loss_triggered = True
        daily_stats["daily_loss_triggered"] = True
        save_daily_stats(daily_stats)

        return True

    return False

def check_loss_streak():
    """Single owner of loss streak pause logic — called every loop iteration."""
    global last_loss_time, trading_paused, PAUSE_REASON, consecutive_losses

    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        if last_loss_time:
            elapsed = (datetime.now() - last_loss_time).total_seconds() / 60

            if elapsed < COOLDOWN_MINUTES:
                # Set pause once, then just report remaining time
                if not trading_paused:
                    trading_paused = True
                    PAUSE_REASON   = f"loss_streak_{consecutive_losses}"
                    msg = (f"⛔ *Loss Streak Triggered*\n"
                           f"{consecutive_losses} consecutive losses\n"
                           f"Pausing for {COOLDOWN_MINUTES} minutes.")
                    threading.Thread(target=send_telegram_msg, args=(msg,), daemon=True).start()
                    print(f"⛔ TRADING PAUSED — {consecutive_losses} losses in a row. "
                          f"Cooldown: {COOLDOWN_MINUTES} min")
                else:
                    mins_left = int(COOLDOWN_MINUTES - elapsed)
                    print(f"⏸️  Loss cooldown active — {mins_left} min remaining")
                return True
            else:
                # Cooldown expired — resume and reset
                trading_paused     = False
                PAUSE_REASON       = ""
                consecutive_losses = 0
                print("✅ Loss streak cooldown finished → trading resumed")
                threading.Thread(target=send_telegram_msg,
                                 args=("✅ Loss streak cooldown finished. Trading resumed.",),
                                 daemon=True).start()

    return False

def update_trade_result(profit):
    global consecutive_losses, last_loss_time, trading_paused, PAUSE_REASON

    daily_stats = load_daily_stats()

    # =============================
    # UPDATE STATS
    # =============================
    daily_stats["trades"] += 1
    daily_stats["net_profit"] += profit

    if profit < 0:
        daily_stats["loss"] += abs(profit)
        consecutive_losses += 1
        last_loss_time = datetime.now()

        print(f"📉 Loss recorded. Streak: {consecutive_losses}")

    else:
        daily_stats["profit"] += profit
        consecutive_losses = 0
        # Clear streak pause on a win (check_loss_streak owns the pause/resume logic)
        if PAUSE_REASON.startswith("loss_streak"):
            PAUSE_REASON   = ""
            trading_paused = False

    save_daily_stats(daily_stats)

# ================= DYNAMIC SL/TP CALCULATION =================

def calculate_liquidity_based_sl(direction, entry_price, df, fvgs, obs, atr=None):
    """
    SL placed just beyond the sweep extreme that triggered the entry.

    After a liquidity sweep the invalidation point is unambiguous: the
    wick tip that grabbed the stops.  If price sweeps that exact level
    again the setup is wrong — SL goes just past it.

    BUY  (sell-side swept): SL = min(last 5 bars low)  - ATR*0.15
    SELL (buy-side swept):  SL = max(last 5 bars high) + ATR*0.15

    Minimum distance clamp: SL is never tighter than 0.8×ATR so it
    can't get picked off by spread alone.
    """
    try:
        if atr is None:
            _s = (df['high'] - df['low']).rolling(14).mean()
            atr = float(_s.iloc[-1]) if not pd.isna(_s.iloc[-1]) \
                  else float((df['high'] - df['low']).tail(5).mean() or 1.0)

        buf = atr * 0.15          # small buffer past the wick
        min_dist = atr * 0.6      # clamp: prevent micro-stops without overriding clean structure

        h = df['high'].values
        l = df['low'].values

        if direction.lower() in ["buy", "bullish"]:
            sweep_extreme = float(l[-5:].min())
            sl = sweep_extreme - buf
            if entry_price - sl < min_dist:
                sl = entry_price - min_dist
        else:
            sweep_extreme = float(h[-5:].max())
            sl = sweep_extreme + buf
            if sl - entry_price < min_dist:
                sl = entry_price + min_dist

        try:
            info = mt5.symbol_info(SYMBOL)
            if info:
                return round(sl, info.digits)
        except Exception:
            pass
        return round(sl, 2)

    except Exception as e:
        print(f"❌ SL calculation error: {e}")
        return None


def calculate_liquidity_based_tp(direction, entry_price, df, atr=None):
    """
    TP placed just before the nearest opposite liquidity level.

    After the entry, price is heading toward the next liquidity pool
    (swing highs for BUY, swing lows for SELL).  We exit just before
    that level — 'a bit below the sweep' — so we capture the bulk of
    the move without being caught in the next reversal when institutions
    defend/raid that level.

    Algorithm:
      1. Find swing highs/lows in the last 100 bars via pivot detection.
      2. Pick the NEAREST one beyond entry_price in the trade direction.
      3. TP = level - ATR*0.3 (BUY) / level + ATR*0.3 (SELL).
      4. Fallback: 50-bar extreme with the same buffer.
      5. Minimum TP distance: 1.5×ATR (avoids trivially close targets).
    """
    try:
        if atr is None:
            _s = (df['high'] - df['low']).rolling(14).mean()
            atr = float(_s.iloc[-1]) if not pd.isna(_s.iloc[-1]) \
                  else float((df['high'] - df['low']).tail(5).mean() or 1.0)

        buf      = atr * 0.3    # exit this far before the liquidity level
        min_dist = atr * 1.5    # TP must be meaningful

        h = df['high'].values
        l = df['low'].values
        N = len(h)
        W = 3                   # pivot window: N candles each side

        tp = None

        if direction.lower() in ["buy", "bullish"]:
            # Nearest pivot HIGH above entry
            best = float("inf")
            for k in range(W, N - W):
                if h[k] > entry_price:
                    if all(h[k] >= h[k-j] for j in range(1, W+1)) and \
                       all(h[k] >= h[k+j] for j in range(1, W+1)):
                        if h[k] < best:
                            best = h[k]
            if best < float("inf"):
                tp = best - buf
            else:
                tp = float(h[-50:].max()) - buf

            if tp - entry_price < min_dist:
                tp = entry_price + min_dist

        else:
            # Nearest pivot LOW below entry
            best = float("-inf")
            for k in range(W, N - W):
                if l[k] < entry_price:
                    if all(l[k] <= l[k-j] for j in range(1, W+1)) and \
                       all(l[k] <= l[k+j] for j in range(1, W+1)):
                        if l[k] > best:
                            best = l[k]
            if best > float("-inf"):
                tp = best + buf
            else:
                tp = float(l[-50:].min()) + buf

            if entry_price - tp < min_dist:
                tp = entry_price - min_dist

        try:
            info = mt5.symbol_info(SYMBOL)
            if info:
                return round(tp, info.digits)
        except Exception:
            pass
        return round(tp, 2)

    except Exception as e:
        print(f"❌ TP calculation error: {e}")
        return None

# ================= POSITION MANAGEMENT =================

def calculate_position_profit(pos, current_price):
    """
    Calculate current floating profit in account currency.
    Essential for Drawdown and Scale-in management.
    """
    try:
        symbol = mt5.symbol_info(pos.symbol)
        if symbol is None:
            return 0.0

        # Gold contract size is typically 100 oz per lot
        contract_size = symbol.trade_contract_size or 100

        if pos.type == mt5.ORDER_TYPE_BUY:
            price_diff = current_price - pos.price_open
        elif pos.type == mt5.ORDER_TYPE_SELL:
            price_diff = pos.price_open - current_price
        else:
            return 0.0

        # Profit = Price Change * Lot Size * Contract Size
        total_profit = price_diff * pos.volume * contract_size
        
        # Include swap and commission for 'Real' net profit if needed
        # net_profit = total_profit + pos.swap + pos.commission
        
        return float(total_profit)

    except Exception as e:
        print(f"❌ Profit calculation error: {e}")
        return 0.0

def get_profit_pips(pos, current_price):
    """
    Get profit in Standard Pips (10 points = 1 Pip).
    Standardizes Gold moves: $1.00 move = 10 Pips.
    """
    try:
        symbol = mt5.symbol_info(pos.symbol)
        if symbol is None:
            return 0.0

        # In MT5, 'point' for Gold is usually 0.01
        # To get standard pips, we use (point * 10)
        pip_size = symbol.point * 10 

        if pos.type == mt5.ORDER_TYPE_BUY:
            pips = (current_price - pos.price_open) / pip_size
        elif pos.type == mt5.ORDER_TYPE_SELL:
            pips = (pos.price_open - current_price) / pip_size
        else:
            return 0.0

        return float(pips)

    except Exception as e:
        print(f"❌ Pip calculation error: {e}")
        return 0.0

def should_close_position_ai(df, pos, current_price, profit, mtf_data=None):
    try:
        symbol = mt5.symbol_info(SYMBOL)
        if symbol is None:
            return False, "symbol_error"

        pip = symbol.point * 10
        is_buy = pos.type == mt5.ORDER_TYPE_BUY

        ai_exit_prob = ai_exit_confidence(df, pos, current_price, profit)
        profit_pips = get_profit_pips(pos, current_price)

        # =============================
        # 1. HARD EXITS
        # =============================
        sweeps = detect_liquidity_sweep(df)

        if is_buy and sweeps == "sell_sweep":
            return True, "BSL_sweep_exit"
        if not is_buy and sweeps == "buy_sweep":
            return True, "SSL_sweep_exit"

        # AI reversal
        if is_buy and ai_exit_prob < 0.25:
            return True, "strong_ai_reversal"
        if not is_buy and ai_exit_prob > 0.75:
            return True, "strong_ai_reversal"

        # =============================
        # 2. PROFIT RATIO
        # =============================
        if pos.tp > 0:
            tp_dist = abs(pos.tp - pos.price_open) / pip
            profit_ratio = profit_pips / max(tp_dist, 1)
        else:
            profit_ratio = profit_pips / 50

        # =============================
        # TP DISTANCE PROTECTION (Option 3)
        # =============================
        # If trade is meaningfully winning (>50% to TP) but not yet near
        # TP (>85%), block all soft close signals. Only HARD exits (sweep
        # against, extreme AI reversal) can still close — those already
        # ran above (lines ~3017-3031) and didn't trigger.
        if 0.50 < profit_ratio < 0.85:
            return False, f"protected_winner_{int(profit_ratio*100)}%"

        # =============================
        # 3. SIGNALS
        # =============================
        hold_signals = []
        close_signals = []

        last_candle = df.iloc[-1]
        body = abs(last_candle['close'] - last_candle['open'])
        rng = last_candle['high'] - last_candle['low']

        is_exhaustion = (rng > 0 and body < rng * 0.3 and profit_pips > 20)

        if is_buy:
            if profit_pips > 0 and not is_exhaustion:
                hold_signals.append("momentum")

            if current_price < df['high'].tail(5).max():
                hold_signals.append("room_to_run")

            if is_exhaustion:
                close_signals.append("exhaustion")

            # Option 2A — only STRONG bearish AI signals count for BUY exit
            if ai_exit_prob < 0.30:
                close_signals.append("ai_weak")

        else:
            if profit_pips > 0 and not is_exhaustion:
                hold_signals.append("momentum")

            if current_price > df['low'].tail(5).min():
                hold_signals.append("room_to_run")

            if is_exhaustion:
                close_signals.append("exhaustion")

            # Option 2A — only STRONG bullish AI signals count for SELL exit
            if ai_exit_prob > 0.70:
                close_signals.append("ai_weak")

        # =============================
        # 4. PROFIT TRACKING FIX (IMPORTANT)
        # =============================
        ticket = pos.ticket

        if ticket not in position_profit_tracker:
            position_profit_tracker[ticket] = {"peak_profit": profit}

        else:
            peak = position_profit_tracker[ticket]["peak_profit"]
            if profit > peak:
                position_profit_tracker[ticket]["peak_profit"] = profit

            peak = position_profit_tracker[ticket]["peak_profit"]

            if peak > 50 and profit < peak * 0.65:
                close_signals.append("giveback")

        # =============================
        # 5. DECISION
        # =============================
        # Option 2C — hold signals weighted higher (was 0.4)
        score = len(close_signals) - (len(hold_signals) * 0.7)

        if profit_ratio > 0.85 and len(close_signals) >= 1:
            return True, f"target_with_{close_signals[0]}"

        # Option 2B — require 2+ net close signals (was 1.2)
        if score >= 2.0:
            return True, f"weakness: {', '.join(close_signals)}"

        if "giveback" in close_signals and profit_pips > 10:
            return True, "protect_profit"

        # STRONG HOLD override
        if is_buy and ai_exit_prob > 0.70:
            return False, "strong_hold"
        if not is_buy and ai_exit_prob < 0.30:
            return False, "strong_hold"

        return False, "hold"

    except Exception as e:
        print(f"❌ AI exit decision error: {e}")
        return False, "error"

# ================= PARTIAL PROFIT TAKING =================

def check_partial_tp(pos, current_price, tracker):
    """
    Sentry Logic: Monitors floating profit and triggers partial closures
    based on the PARTIAL_TP_LEVELS configuration.
    tracker is the position tracker dict containing 'closed_levels' list.
    """
    symbol = mt5.symbol_info(SYMBOL)
    if symbol is None:
        return False, 0, 0

    try:
        profit_pips = get_profit_pips(pos, current_price)

        # closed_levels is a list of pip levels already taken on this ticket
        closed_levels = tracker.get("closed_levels", [])

        # Pre-check: position must be big enough so 30% × volume rounds
        # cleanly to broker minimum (volume_min, usually 0.01).
        # Without this, the loop signals "ready to close" forever but
        # execute_partial_tp silently rejects below-minimum volumes.
        volume_min = symbol.volume_min if symbol.volume_min > 0 else 0.01
        max_pct    = max((p["percentage"] for p in PARTIAL_TP_LEVELS), default=0.3)
        if pos.volume * max_pct < volume_min:
            return False, 0, 0

        for level in PARTIAL_TP_LEVELS:
            target_pips = level["pips"]
            percentage  = level["percentage"]

            if target_pips in closed_levels:
                continue

            if profit_pips >= target_pips:
                spread_pips = (symbol.ask - symbol.bid) / (symbol.point * 10)
                if spread_pips > (target_pips * 0.25):
                    # Throttle this spam to once every 30s
                    now = time.time()
                    last_warn = tracker.get("last_spread_warn", 0)
                    if now - last_warn > 30:
                        print(f"⏳ Partial TP at {target_pips} delayed: spread too wide ({spread_pips:.1f} pips)")
                        tracker["last_spread_warn"] = now
                    continue

                print(f"🎯 PARTIAL TP SIGNAL: {target_pips} pips reached! Closing {percentage*100:.0f}%")
                return True, percentage, target_pips

        return False, 0, 0

    except Exception as e:
        print(f"Partial TP check error: {e}")
        return False, 0, 0

def execute_partial_tp(pos, percentage):
    """
    Execute partial take profit: Closes a portion of the trade 
    while keeping the rest running for the 'runner' target.
    """
    try:
        symbol = mt5.symbol_info(SYMBOL)
        tick = mt5.symbol_info_tick(SYMBOL)

        if symbol is None or tick is None:
            return False

        # 1. VOLUME CALCULATION (Broker Rules)
        volume_step = symbol.volume_step
        volume_min = symbol.volume_min

        # Target volume to close
        raw_volume = pos.volume * percentage
        
        # Round to broker's allowed step (0.01 for most Gold accounts)
        volume_to_close = round(raw_volume / volume_step) * volume_step
        
        # Safety: Final rounding to 2 decimal places to avoid MT5 floating point errors
        volume_to_close = float(f"{volume_to_close:.2f}")

        # 2. SAFETY CHECKS
        # Don't try to close more than we have or less than allowed
        if volume_to_close >= pos.volume:
            volume_to_close = pos.volume - volume_step
            
        if volume_to_close < volume_min:
            print("⚠️ Volume to close is below broker minimum — skipping")
            return False

        # 3. POSITION DIRECTION & PRICE
        # BUY position is closed with a SELL order at BID price
        # SELL position is closed with a BUY order at ASK price
        is_buy = pos.type == mt5.ORDER_TYPE_BUY
        order_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        price = tick.bid if is_buy else tick.ask

        # 4. ORDER REQUEST
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": volume_to_close,
            "type": order_type,
            "position": pos.ticket,  # CRITICAL: Links the close to the specific trade
            "price": round(price, symbol.digits),
            "deviation": 50,         # Gold-safe deviation (matches execute_trade)
            "magic": MAGIC_NUMBER,   # Unified with your main system
            "comment": f"SMC Partial {percentage*100:.0f}%",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # 5. EXECUTION WITH FILLING MODE FALLBACK
        result = mt5.order_send(request)
        
        # Fallback for brokers requiring 'Fill or Kill' (FOK)
        if result and result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
            request["type_filling"] = mt5.ORDER_FILLING_FOK
            result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Partial TP Success: {volume_to_close} lots closed at {price}")
            return True
        else:
            err = result.comment if result else "MT5 Timeout"
            print(f"❌ Partial TP Failed: {err}")
            return False

    except Exception as e:
        print(f"Partial TP Logic Error: {e}")
        return False

# ================= TRAILING STOP =================

def update_smart_trailing_stop(pos, current_price, profit, df):
    """SMC Trailing Stop: Protects profit by hiding behind structural swings with ATR buffer"""

    symbol = mt5.symbol_info(SYMBOL)
    if symbol is None or len(df) < 20:
        return None

    point = symbol.point
    pip = point * 10
    is_buy = pos.type == mt5.ORDER_TYPE_BUY
    
    # 1. Activation Check (Must be in profit)
    profit_pips = get_profit_pips(pos, current_price)
    if profit_pips < TRAIL_ACTIVATION_PIPS:
        return None

    try:
        # Calculate ATR for a dynamic buffer (prevents wicking out)
        # Using a 14-period ATR
        high, low, close = df['high'], df['low'], df['close']
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Buffer: Use 0.5x ATR or at least 2 pips
        buffer = max(atr * 0.5, 2 * pip)

        # 2. Identify Structure (Look back 15 candles)
        if is_buy:
            # Swing Low + ATR Buffer
            swing_low = df['low'].tail(15).min()
            new_sl = swing_low - buffer
            
            # Safety: Initial move should at least lock in Breakeven + small profit
            if new_sl < pos.price_open:
                new_sl = pos.price_open + (1 * pip)
                
            # Rule: Only move UP (Protects against logic reversing the SL)
            if pos.sl > 0 and new_sl <= pos.sl:
                return None
            
            # Rule: Distance check (Don't set SL too close to current price)
            # Gold needs at least 15-20 pips of 'breathing room' on M5
            if current_price - new_sl < (8 * pip):
                return None

        else: # SELL
            # Swing High + ATR Buffer
            swing_high = df['high'].tail(15).max()
            new_sl = swing_high + buffer
            
            if new_sl > pos.price_open:
                new_sl = pos.price_open - (1 * pip)
                
            # Rule: Only move DOWN
            if pos.sl > 0 and new_sl >= pos.sl:
                return None
                
            if new_sl - current_price < (8 * pip):
                return None

        # 3. Final validation: Ensure the move is worth the execution cost (min 2 pip change)
        if pos.sl > 0 and abs(new_sl - pos.sl) < (2 * pip):
            return None

        return round(new_sl, symbol.digits)

    except Exception as e:
        print(f"Trailing stop error: {e}")
        return None

def modify_position_sl(new_sl, position):
    """Modify SL with Request-Spam protection and Directional Safety"""
    try:
        # 0. GLOBAL MARKET-CLOSED COOLDOWN
        # When broker rejects with "Market closed" we set a 5-minute cooldown
        # Some brokers report trade_mode=FULL while still rejecting orders.
        # This catches that case and stops the spam.
        if hasattr(modify_position_sl, "_market_closed_until"):
            if time.time() < modify_position_sl._market_closed_until:
                return False

        # 1. Fetch Fresh Data
        symbol = mt5.symbol_info(SYMBOL)
        tick = mt5.symbol_info_tick(SYMBOL)
        if not symbol or not tick: return False

        # MARKET-OPEN CHECK — first defense (some brokers don't report this honestly)
        if symbol.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
            return False

        # 2. Normalize and Redundancy Check
        new_sl = round(new_sl, symbol.digits)
        
        # FIX: If the SL hasn't changed by at least 0.5 pips, don't spam the broker
        pip = symbol.point * 10
        if abs(new_sl - position.sl) < (0.5 * pip):
            return False

        # 3. Directional Safety Check
        is_buy = position.type == 0
        if is_buy:
            # For Buys, SL must ONLY go up
            if position.sl > 0 and new_sl <= position.sl:
                return False
        else:
            # For Sells, SL must ONLY go down
            if position.sl > 0 and new_sl >= position.sl:
                return False

        # 4. Stop Level Protection (with 1-pip safety buffer)
        stop_level = (symbol.trade_stops_level * symbol.point) + (1.0 * pip)
        
        if is_buy:
            if tick.bid - new_sl < stop_level:
                # Too close to current price, broker will reject
                return False
        else:
            if new_sl - tick.ask < stop_level:
                # Too close to current price
                return False

        # 5. Send Request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": SYMBOL,
            "position": position.ticket,
            "sl": new_sl,
            "tp": position.tp,
            "deviation": 50, # Gold-safe (matches execute_trade)
        }

        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"🛡️ SL Protected: {position.ticket} moved to {new_sl}")
            return True
        else:
            # Logging failures helps debug "Off-quotes" or "Invalid Stops"
            reason = result.comment if result else "No Result"

            # If broker rejects with any "broker-unavailable" condition,
            # set 15-min cooldown to stop spam. Catches: market closed,
            # no prices, off quotes, requote storms during low liquidity.
            broker_unavailable = (
                "Market closed" in reason
                or "market closed" in reason.lower()
                or "No prices" in reason
                or "no price" in reason.lower()
                or "Off quotes" in reason
                or "off quotes" in reason.lower()
            )
            if broker_unavailable:
                modify_position_sl._market_closed_until = time.time() + 900
                print(f"⚠️ Broker unavailable ({reason}) — pausing SL mods for 15 minutes (ticket {position.ticket})")
                return False

            # Show ALL SL modification failures (no silent filtering).
            # We need to see "Invalid stops" rejections to diagnose BE/trail failures.
            print(f"⚠️ SL Mod Failed: {position.ticket} | {reason}")
            return False

    except Exception as e:
        print(f"Major SL Modify Error: {e}")
        return False

# ================= POSITION SCALING =================

def check_scale_in(pos, current_price, df, ai_confidence):
    """
    Check if we should scale into a position (Averaging Down).
    Only triggers if AI confidence remains high.
    """
    global scaled_positions

    if not SCALE_IN_ENABLED:
        return None, 0

    try:
        ticket = pos.ticket
        # Initialize tracking for this ticket if new
        if ticket not in scaled_positions:
            scaled_positions[ticket] = []

        # Calculate profit in pips (Negative if losing)
        # Assuming get_profit_pips returns -30 for a 30 pip drawdown
        profit_pips = get_profit_pips(pos, current_price)

        for level in SCALE_IN_LEVELS:
            target_drawdown = level["pips_from_entry"] # e.g., -30
            multiplier = level["volume_multiplier"]

            # 1. Check if we have hit the drawdown target
            # 2. Check if this specific level hasn't been fired yet
            if profit_pips <= target_drawdown and target_drawdown not in scaled_positions[ticket]:
                
                # --- AI GUARDRAIL ---
                # Don't add to a losing trade if AI confidence has collapsed
                if ai_confidence < 0.50:
                    print(f"⚠️ Scale-in skipped: AI Confidence too low ({ai_confidence:.2f})")
                    continue

                # --- VOLATILITY GUARDRAIL ---
                # Don't scale in if the candle range is too massive (News spike)
                last_candle_range = (df['high'].iloc[-1] - df['low'].iloc[-1]) / (mt5.symbol_info(SYMBOL).point * 10)
                if last_candle_range > 50: # 50 pip candle
                    print("⚠️ Scale-in skipped: Volatility too high")
                    continue

                # Mark level as executed
                scaled_positions[ticket].append(target_drawdown)
                
                print(f"🔥 SCALE-IN TRIGGERED: Level {target_drawdown} pips | Mult: {multiplier}")
                return level, multiplier

        return None, 0

    except Exception as e:
        print(f"Scale-in check error: {e}")
        return None, 0

# ================= CLOSE POSITION =================

def close_position(pos, reason="manual"):
    """Close an existing position with Slippage and Requote protection"""
    try:
        # 1. Refresh Symbol Info
        symbol = mt5.symbol_info(SYMBOL)
        if symbol is None: return False

        # 2. Get FRESH Tick immediately before the request
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None: return False

        is_buy = pos.type == mt5.ORDER_TYPE_BUY
        order_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        
        # FIX: Use the most aggressive price to ensure fill
        price = tick.bid if is_buy else tick.ask
        profit = getattr(pos, 'profit', 0.0)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "position": pos.ticket,
            "volume": pos.volume,
            "type": order_type,
            "price": round(price, symbol.digits),
            "deviation": 50, # Increased deviation for Gold volatility
            "magic": MAGIC_NUMBER,
            "comment": f"SMC Exit: {reason[:15]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK 
        }

        # 3. Execution
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
            request["type_filling"] = mt5.ORDER_FILLING_IOC
            result = mt5.order_send(request)

        # 4. Error Handling
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err_msg = result.comment if result else "Unknown MT5 Error"
            print(f"❌ Close Failed: {pos.ticket} | Code: {result.retcode if result else 'N/A'} | {err_msg}")
            return False

        # 5. Success Logic (Cooldowns)
        print(f"✅ Position Closed: {pos.ticket} | Profit: ${profit:.2f} | Reason: {reason}")

        # ======================================================
        # 🚀 TELEGRAM CLOSE ALERT
        # ======================================================
        status_icon = "🟢" if profit >= 0 else "🔴"
        result_text = "PROFIT" if profit >= 0 else "LOSS"

        # Calculate trade duration directly from position open time
        pos_age_sec  = time.time() - int(pos.time)
        duration_str = f"{int(pos_age_sec / 60)}m {int(pos_age_sec % 60)}s"

        exit_msg = (
            f"{status_icon} **TRADE CLOSED: {SYMBOL}**\n"
            f"----------------------------\n"
            f"📊 Result: `{result_text}`\n"
            f"💰 P/L:    `{'+' if profit >= 0 else ''}{profit:.2f} USD`\n"
            f"📜 Reason:  _{reason}_\n"
            f"⏱ Time:    `{duration_str}`\n"
            f"----------------------------\n"
            f"🎟 Ticket:  #{pos.ticket}"
        )

        threading.Thread(target=send_telegram_msg, args=(exit_msg,), daemon=True).start()
        # ======================================================

        update_trade_result(profit)
        # Update result in trades.csv
        update_trade_result_in_csv(pos.ticket, profit, entry_price=pos.price_open)
        global last_position_close_time, cooldown_duration
        last_position_close_time = time.time()

        # Dynamic Cooldown + post-trade re-entry discipline
        global last_trade_was_loss, post_loss_recheck_done
        if profit > 0:
            cooldown_duration      = COOLDOWN_AFTER_TRADE   # config: default 60s
            last_trade_was_loss    = False
            post_loss_recheck_done = True
        else:
            cooldown_duration      = COOLDOWN_AFTER_LOSS    # config: default 300s
            last_trade_was_loss    = True
            post_loss_recheck_done = False    # force market re-check before next entry
            print("🛑 Loss detected — bot will re-validate market structure before next entry.")

        # Update Daily Stats & History (Modular cleanup)
        try:
            daily_stats = load_daily_stats()
            daily_stats["trades"] += 1
            if profit > 0:
                daily_stats["profit"] += profit
                daily_stats["net_profit"] += profit
            else:
                daily_stats["loss"] += abs(profit)
                daily_stats["net_profit"] -= abs(profit)
            save_daily_stats(daily_stats)

            # Record in History
            trade_record = {
                "ticket": pos.ticket,
                "direction": "buy" if is_buy else "sell",
                "profit": profit,
                "exit_reason": reason,
                "time": datetime.now().isoformat()
            }
            add_trade_to_history(trade_record)
        except Exception as e:
            print(f"Stat update error: {e}")

        # Cleanup tracker
        if pos.ticket in position_profit_tracker:
            del position_profit_tracker[pos.ticket]

        return True

    except Exception as e:
        print(f"Major Close Error: {e}")
        return False

# ================= MANAGE POSITIONS =================

def manage_open_positions(df, mtf_data, ai_prob):            
    """Manage all open positions with optimized AI and MT5 calls"""
    global position_profit_tracker

    try:
        # =============================
        # PENDING ORDER WATCHDOG
        # Cancel any of our limit orders older than 15 minutes.
        # Prevents stale orders from filling hours later on wrong signals.
        # =============================
        PENDING_EXPIRY_MINUTES = 15
        pending_orders = mt5.orders_get(symbol=SYMBOL)
        if pending_orders:
            now = time.time()
            for order in pending_orders:
                if order.magic != MAGIC_NUMBER:
                    continue
                order_age_min = (now - int(order.time_setup)) / 60
                if order_age_min > PENDING_EXPIRY_MINUTES:
                    cancel_req = {
                        "action": mt5.TRADE_ACTION_REMOVE,
                        "order":  order.ticket,
                    }
                    result = mt5.order_send(cancel_req)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"🗑️ Cancelled stale limit order #{order.ticket} "
                              f"(age: {order_age_min:.0f} min)")
                    else:
                        print(f"⚠️ Could not cancel order #{order.ticket}: "
                              f"{result.comment if result else 'no result'}")

        # Only manage positions opened by THIS bot (filter by magic number)
        all_positions = mt5.positions_get(symbol=SYMBOL)
        positions = [p for p in all_positions if p.magic == MAGIC_NUMBER] if all_positions else []

        # ============================================================
        # FIX: DETECT CLOSED POSITIONS (TP/SL hits, manual closes)
        # close_position() only runs for AI/timeout exits. Broker-side
        # TP/SL hits never call it, so trade_history.json and trades.csv
        # never record those outcomes. Detect them here by comparing
        # last loop's tickets to current tickets.
        # ============================================================
        global known_tickets
        current_tickets = {p.ticket for p in positions}
        closed_tickets  = known_tickets - current_tickets

        for closed_ticket in closed_tickets:
            try:
                # Query MT5 history for this ticket's closing deal
                from_date = datetime.now() - timedelta(days=2)
                deals = mt5.history_deals_get(from_date, datetime.now())
                if not deals:
                    continue

                # Find the CLOSING deal (entry=1) for this position
                close_profit = 0.0
                close_reason = "broker_close"
                position_direction = "unknown"
                for deal in deals:
                    if deal.magic != MAGIC_NUMBER:
                        continue
                    if deal.position_id != closed_ticket:
                        continue
                    if deal.entry == 1:  # closing deal
                        close_profit += float(deal.profit)
                        # deal.type: 0=BUY 1=SELL — closing deal type is OPPOSITE of original
                        position_direction = "sell" if deal.type == 0 else "buy"
                        # Try to infer TP/SL from deal comment
                        comment = (deal.comment or "").lower()
                        if "tp" in comment:
                            close_reason = "tp_hit"
                        elif "sl" in comment:
                            close_reason = "sl_hit"

                if position_direction != "unknown":
                    # Record to trade_history.json
                    trade_record = {
                        "ticket": int(closed_ticket),
                        "direction": position_direction,
                        "profit": round(close_profit, 2),
                        "exit_reason": close_reason,
                        "time": datetime.now().isoformat()
                    }
                    add_trade_to_history(trade_record)

                    # Update trades.csv result field
                    update_trade_result_in_csv(closed_ticket, close_profit, entry_price=None)

                    # Update stats counters
                    update_trade_result(close_profit)

                    icon = "✅" if close_profit > 0 else ("❌" if close_profit < 0 else "➖")
                    print(f"{icon} Position {closed_ticket} closed via broker ({close_reason}): ${close_profit:.2f}")

            except Exception as e:
                print(f"⚠️ Could not record closed ticket {closed_ticket}: {e}")

        # Update known tickets for next loop
        known_tickets = current_tickets

        if not positions:
            return

        # Fetch MT5 data ONCE (efficiency)
        tick = mt5.symbol_info_tick(SYMBOL)
        symbol = mt5.symbol_info(SYMBOL)
        if not tick or not symbol:
            return

        pip = symbol.point * 10
        active_tickets = {p.ticket for p in positions}

        # =============================
        # CLEANUP TRACKER
        # =============================
        for t in list(position_profit_tracker.keys()):
            if t not in active_tickets:
                del position_profit_tracker[t]

        for pos in positions:
            try:
                ticket = pos.ticket
                profit = pos.profit
                is_buy = pos.type == mt5.ORDER_TYPE_BUY
                current_price = tick.bid if is_buy else tick.ask

                # =============================
                # INIT TRACKER
                # =============================
                if ticket not in position_profit_tracker:
                    position_profit_tracker[ticket] = {
                        "peak_profit": profit,
                        "entry_price": pos.price_open,
                        "closed_levels": [],
                        "last_ai_check": 0,
                        "be_active": False,
                        "closing": False
                    }

                tracker = position_profit_tracker[ticket]

                # Update peak profit
                if profit > tracker["peak_profit"]:
                    tracker["peak_profit"] = profit

                # =============================
                # POSITION AGE CHECK
                # =============================
                pos_age = time.time() - int(pos.time)

                if pos_age < 60:  # give gold breathing room
                    continue

                # =============================
                # MAX TRADE TIME EXIT
                # =============================
                if (pos_age / 60) > max_trade_minutes:
                    close_position(pos, "max_trade_time")
                    continue

                # =============================
                # HARD LOSS PROTECTION
                # =============================
                if profit < -100:
                    print(f"🚨 HARD SL: Ticket {ticket}")
                    close_position(pos, "hard_stop_loss")
                    continue

                # =============================
                # AI EXIT CHECK (CONTROLLED)
                # =============================
                last_ai = tracker["last_ai_check"]

                # Only check if enough time passed OR we hit a danger zone (-$50)
                if (time.time() - last_ai > 3) or (profit < -50):
    
                    # 1. IMMEDIATELY update the timer so the next loop cycle waits 3 seconds
                    tracker["last_ai_check"] = time.time() 

                    # 2. Only run the AI logic if we aren't already trying to close this ticket
                    if not tracker["closing"]:
                        should_close, reason = should_close_position_ai(
                            df, pos, current_price, profit, mtf_data
                        )

                        if should_close:
                            tracker["closing"] = True # Set flag to prevent double-calls
                            print(f"🤖 AI EXIT TRIGGERED: {reason} | Ticket {ticket}")
            
                            success = close_position(pos, reason)
            
                            if success:
                                continue # Move to next position in the loop
                            else:
                                # If close failed (requote/slippage), reset flag to try again next 3s cycle
                                tracker["closing"] = False

                # =============================
                # PARTIAL TP (if enabled)
                # =============================
                if PARTIAL_TP_ENABLED:
                    should_partial, percentage, level_pips = check_partial_tp(
                        pos, current_price, tracker
                    )

                    if should_partial:
                        # safety: avoid micro lot errors
                        if pos.volume >= 0.04:
                            if execute_partial_tp(pos, percentage):
                                tracker["closed_levels"].append(level_pips)

                # =============================
                # BREAKEVEN (if enabled)
                # =============================
                profit_pips = get_profit_pips(pos, current_price)

                if BREAKEVEN_ENABLED and profit_pips >= MOVE_TO_BREAKEVEN_AFTER_PIPS and not tracker["be_active"]:

                    new_sl = pos.price_open + (3 * pip if is_buy else -3 * pip)

                    if (is_buy and (pos.sl == 0 or new_sl > pos.sl)) or \
                       (not is_buy and (pos.sl == 0 or new_sl < pos.sl)):

                        if modify_position_sl(new_sl, pos):
                            tracker["be_active"] = True
                            print(f"🛡️ BE SET: Ticket {ticket}")

                # =============================
                # SMART TRAILING (if enabled)
                # =============================
                if TRAILING_ENABLED:
                    trail_sl = update_smart_trailing_stop(pos, current_price, profit, df)

                    if trail_sl:
                        if (is_buy and (pos.sl == 0 or trail_sl > pos.sl)) or \
                           (not is_buy and (pos.sl == 0 or trail_sl < pos.sl)):

                            # avoid too tight trailing on gold
                            if abs(current_price - trail_sl) > (5 * pip):
                                modify_position_sl(trail_sl, pos)

                # =============================
                # SCALE-IN CHECK
                # =============================
                # NOTE: profit > 0 with negative SCALE_IN_LEVELS = scale-in never fires.
                # Effectively a safety no-op. Avoids martingale (profit < 0) behavior.
                if SCALE_IN_ENABLED and profit > 0:
                    scale_level, scale_mult = check_scale_in(pos, current_price, df, ai_prob)
                    if scale_level and scale_mult > 0:
                        scale_vol = round(pos.volume * scale_mult, 2)
                        scale_vol = max(mt5.symbol_info(SYMBOL).volume_min, scale_vol)
                        scale_request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": SYMBOL,
                            "volume": scale_vol,
                            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
                            "price": current_price,
                            "sl": pos.sl,
                            "tp": pos.tp,
                            "deviation": 50,
                            "magic": MAGIC_NUMBER,
                            "comment": "SMC Scale-in",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(scale_request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"✅ Scale-in executed: +{scale_vol} lots")

            except Exception as e:
                print(f"⚠️ Error in position {pos.ticket}: {e}")
                continue

    except Exception as e:
        print(f"🚨 Manage positions loop error: {e}")

def check_equity_discipline() -> tuple:
    """
    Pauses bot if live performance diverges from expected.

    Checks
    ------
    1. Rolling 20-trade win rate < 30%  → model is degrading in current regime
    2. Current drawdown > 10% from peak → excessive loss streak

    Returns (should_pause: bool, reason: str)
    """
    try:
        # ── Rolling win-rate check ────────────────────────────────────────────
        recent = [t for t in list(trade_journal)[-20:]
                  if t.get("profit") is not None]
        if len(recent) >= 10:
            wins     = sum(1 for t in recent if float(t.get("profit", 0)) > 0)
            win_rate = wins / len(recent)
            if win_rate < 0.30:
                return (True,
                        f"rolling_win_rate_{win_rate:.0%}_on_last_{len(recent)}_trades_–_model_degraded")

        # ── Drawdown check ────────────────────────────────────────────────────
        dd = get_current_drawdown()
        if dd > 10.0:
            return (True,
                    f"drawdown_{dd:.1f}%_exceeds_10%_limit_–_pausing_2h")

    except Exception:
        pass

    return False, ""


def volatility_kill_switch(df):
    global last_volatility_alert

    try:
        if len(df) < 20:
            return False

        last_candle_range = df.high.iloc[-1] - df.low.iloc[-1]
        avg_range = (df.high - df.low).tail(14).mean()

        if avg_range == 0:
            return False

        spike_ratio = last_candle_range / avg_range

        body_size = abs(df.close.iloc[-1] - df.open.iloc[-1])
        avg_body = abs(df.close - df.open).tail(14).mean()
        body_spike = body_size / avg_body if avg_body > 0 else 0

        spike = max(spike_ratio, body_spike)

        if spike > 3.5:

            if time.time() - last_volatility_alert > 30:
                print(f"⚠️ VOLATILITY SPIKE: {spike:.2f}x - SAFETY ON")
                last_volatility_alert = time.time()

            return True

        return False

    except Exception:
        return False

# ================= ENTRY DECISION =================

def should_enter_trade(df, mtf_data, news_active):
    """FINAL clean AI + SMC decision engine using ai_decision()"""

    try:
        global last_position_close_time, cooldown_duration

        # =============================
        # EQUITY DISCIPLINE (NEW)
        # =============================
        paused, pause_reason = check_equity_discipline()
        if paused:
            return "WAIT", pause_reason, 0.5, None, 0.0, 0.0

        # =============================
        # SAFETY FILTERS (KEEP)
        # =============================
        if time.time() - last_position_close_time < cooldown_duration:
            return "WAIT", "cooldown", 0.5, None, 0.0, 0.0

        if not spread_ok() or trading_paused:
            return "WAIT", "safety_filter", 0.5, None, 0.0, 0.0

        if volatility_kill_switch(df):
            return "WAIT", "volatility", 0.5, None, 0.0, 0.0

        # =============================
        # POST-LOSS RE-ENTRY GATE
        # M5+M15 are the real gates. H1 is advisory only. H4 removed.
        # =============================
        global last_trade_was_loss, post_loss_recheck_done
        if last_trade_was_loss and not post_loss_recheck_done:
            ms_check, ms_conf_check, _ = get_mss_signal(df)
            h1_check  = mtf_data.get("H1_trend")  if mtf_data else None
            m15_check = mtf_data.get("M15_trend") if mtf_data else None

            # M5 must have clear BOS/CHoCH with confidence >= 0.70
            m5_ok = ms_check in ["BUY", "SELL"] and ms_conf_check >= 0.70

            # M15 must not directly oppose M5
            m15_ok = not (
                (ms_check == "BUY"  and m15_check == "bearish") or
                (ms_check == "SELL" and m15_check == "bullish")
            )

            recheck_passed = m5_ok and m15_ok

            if recheck_passed:
                post_loss_recheck_done = True
                h1_note = "aligned" if (
                    (ms_check == "BUY"  and h1_check != "bearish") or
                    (ms_check == "SELL" and h1_check != "bullish") or
                    h1_check is None
                ) else "opposing"
                print(f"✅ Post-loss re-validated → {ms_check} | "
                      f"M15:{m15_check} H1:{h1_check}({h1_note}) — re-entry unlocked.")
            else:
                return "WAIT", "post_loss_recheck", 0.5, None, 0.0, 0.0

        # =============================
        # MARKET STRUCTURE & FEATURES (🔥 FIX)
        # =============================
        ms_signal, ms_conf, ms_data = get_mss_signal(df)

        features = generate_smc_features(df, mtf_data, news_active)

        if features is None:
            return "WAIT", "no_features", 0.5, ms_data, 0.0, 0.0

        # =============================
        # AI DECISION
        # =============================
        signal, buy_prob, sell_prob = ai_decision(df, mtf_data, news_active)
        ai_prob = max(buy_prob, sell_prob)

        # =============================
        # REGIME FILTER (NEW — Tier 1)
        # =============================
        current_regime = detect_regime(df)
        regime_ok, regime_block_reason = regime_allows_trade(current_regime, signal)
        if not regime_ok:
            return "WAIT", regime_block_reason, ai_prob, ms_data, buy_prob, sell_prob

        # Raise AI confidence threshold in uncertain regimes
        regime_mult = get_regime_confidence_multiplier(current_regime)
        effective_buy_threshold  = BUY_THRESHOLD  * regime_mult
        effective_sell_threshold = SELL_THRESHOLD * regime_mult

        # =============================
        # AI EDGE FILTER (asymmetric — ternary class compression)
        # Data analysis from 750 decisions: buy_prob ≥ 0.55 = 18.1%,
        # sell_prob ≥ 0.55 = 0.0%. SELL needs 0.45 to match model's range.
        # Thresholds now scaled by regime_mult (harder in choppy/transitional markets).
        # =============================
        if buy_prob > sell_prob and buy_prob < effective_buy_threshold:
            return "WAIT", "weak_ai", ai_prob, ms_data, buy_prob, sell_prob
        elif sell_prob > buy_prob and sell_prob < effective_sell_threshold:
            return "WAIT", "weak_ai", ai_prob, ms_data, buy_prob, sell_prob

        # =============================
        # MTF FILTER — M5 and M15 primary gates, H1 for M15 bypass only
        # =============================
        h1_trend  = mtf_data.get("H1_trend")  if mtf_data else None
        m15_trend = mtf_data.get("M15_trend") if mtf_data else None      
        m5_trend  = ms_data.get("trend")      if ms_data else None

        # =============================
        # SMC SIGNALS
        # =============================
        sweep = detect_liquidity_sweep(df)
        fvg_data = detect_fvg(df)
        current_price = df["close"].iloc[-1]
        fvg_signal = get_fvg_signal(fvg_data, current_price, df)

        # =========================
        # FINAL ENTRY DECISION LOGIC
        # =========================

        # --- HARD FILTER: NO EDGE ---
        if abs(buy_prob - sell_prob) < 0.08:
            return "WAIT", "no_clear_edge", ai_prob, ms_data, buy_prob, sell_prob

        # --- DIAGNOSTIC: feature dump for borderline AI moments ---
        if 0.40 <= buy_prob <= 0.65 and 0.25 <= sell_prob <= 0.50:
            print(f"📐 FEATURES: trend_bull={features.get('trend_bullish',0):.0f} trend_bear={features.get('trend_bearish',0):.0f} "
                  f"bull_fvg={features.get('in_bullish_fvg',0):.0f} bear_fvg={features.get('in_bearish_fvg',0):.0f} "
                  f"sweep_low={features.get('liq_sweep_low',0):.0f} sweep_high={features.get('liq_sweep_high',0):.0f} "
                  f"bos_bull={features.get('bos_bull',0):.0f} bos_bear={features.get('bos_bear',0):.0f} "
                  f"bull_conf={features.get('bullish_confluence',0):.1f} bear_conf={features.get('bearish_confluence',0):.1f} "
                  f"bias={features.get('net_bias',0):.1f}")

        # ── Step 1: Sweep sets direction (AI cannot decide direction) ───────────
        # The AI has a structural BUY bias trained on a gold bull market.
        # It can see the live candle but it pattern-matches it to past dips-
        # that-bounced, so it always leans BUY. The liquidity sweep is the
        # real structural trigger — price swept a level and is reversing.
        # Direction comes from the sweep; AI only confirms quality.
        sweep_direction = detect_liquidity_sweep(df)  # "buy_sweep" / "sell_sweep" / None
        if sweep_direction is None:
            return "WAIT", "no_sweep", 0.0, ms_data, buy_prob, sell_prob

        # ── Step 2: Live momentum gate ───────────────────────────────────────
        # If price is in active freefall/surge (net > 1.5×ATR in 5 bars)
        # against the sweep direction, the sweep is a continuation move,
        # not a reversal — skip.
        _atr_live = float((df["high"] - df["low"]).rolling(14).mean().iloc[-1])
        _net5_live = float(df["close"].values[-1] - df["open"].values[-5])
        if sweep_direction == "buy_sweep"  and _net5_live < -_atr_live * 1.5:
            return "WAIT", "momentum_crash", buy_prob, ms_data, buy_prob, sell_prob
        if sweep_direction == "sell_sweep" and _net5_live >  _atr_live * 1.5:
            return "WAIT", "momentum_surge", sell_prob, ms_data, buy_prob, sell_prob

        # ── Step 3: AI must not strongly oppose the sweep direction ──────────
        # AI no longer decides direction, but if its probability for the sweep
        # direction is < 0.25 the setup is structurally very weak — skip.
        if sweep_direction == "buy_sweep"  and buy_prob  < 0.25:
            return "WAIT", "ai_weak_buy", buy_prob, ms_data, buy_prob, sell_prob
        if sweep_direction == "sell_sweep" and sell_prob < 0.25:
            return "WAIT", "ai_weak_sell", sell_prob, ms_data, buy_prob, sell_prob

        # ── Map sweep → signal (sweep owns direction, not AI) ────────────────
        signal = "BUY" if sweep_direction == "buy_sweep" else "SELL"

        # =========================
        # BUY SIDE (sweep = buy_sweep)
        # =========================
        if signal == "BUY":

            # ── M5 PRIMARY GATE ────────────────────────────────────────
            if m5_trend not in ["bullish", "range", None]:
                return "WAIT", "trend_mismatch", buy_prob, ms_data, buy_prob, sell_prob

            # ── M15 CONFIRMATION ───────────────────────────────────────
            if m15_trend == "bearish":
                if not (buy_prob >= 0.78 and h1_trend == "bullish"):
                    return "WAIT", "m15_conflict", buy_prob, ms_data, buy_prob, sell_prob

            # ── SMC REVERSAL — fires if M5+M15 passed ─────────────────
            if features.get("bullish_reversal_setup", 0) == 1:
                return "BUY", "smc_reversal", buy_prob, ms_data, buy_prob, sell_prob

            # ── CONFLUENCE ENTRY ───────────────────────────────────────
            min_conf = 1.5 if buy_prob > 0.50 else 2.5
            if features.get("bullish_confluence", 0) >= min_conf:
                return "BUY", "confluence_entry", buy_prob, ms_data, buy_prob, sell_prob

            # ── CHoCH ENTRY ────────────────────────────────────────────
            if features.get("choch_bull", 0) == 1.0:
                return "BUY", "choch_reversal", buy_prob, ms_data, buy_prob, sell_prob

            # ── MOMENTUM ENTRY ─────────────────────────────────────────
            min_bias = 0.8 if buy_prob > 0.50 else 1.5
            if features.get("net_bias", 0) > min_bias:
                return "BUY", "momentum_bias", buy_prob, ms_data, buy_prob, sell_prob

            # ── MTF ALIGNMENT ──────────────────────────────────────────
            if m5_trend == "bullish" and m15_trend == "bullish":
                return "BUY", "mtf_alignment", buy_prob, ms_data, buy_prob, sell_prob

        # =========================
        # SELL SIDE (sweep = sell_sweep)
        # =========================
        elif signal == "SELL":

            # ── M5 PRIMARY GATE ────────────────────────────────────────
            if m5_trend not in ["bearish", "range", None]:
                return "WAIT", "trend_mismatch", sell_prob, ms_data, buy_prob, sell_prob

            # ── M15 CONFIRMATION ───────────────────────────────────────
            if m15_trend == "bullish":
                if not (sell_prob >= 0.78 and h1_trend == "bearish"):
                    return "WAIT", "m15_conflict", sell_prob, ms_data, buy_prob, sell_prob

            # ── SMC REVERSAL ───────────────────────────────────────────
            if features.get("bearish_reversal_setup", 0) == 1:
                return "SELL", "smc_reversal", sell_prob, ms_data, buy_prob, sell_prob

            # ── CONFLUENCE ENTRY ───────────────────────────────────────
            min_conf = 1.5 if sell_prob > 0.50 else 2.5
            if features.get("bearish_confluence", 0) >= min_conf:
                return "SELL", "confluence_entry", sell_prob, ms_data, buy_prob, sell_prob

            # ── CHoCH ENTRY ────────────────────────────────────────────
            if features.get("choch_bear", 0) == 1.0:
                return "SELL", "choch_reversal", sell_prob, ms_data, buy_prob, sell_prob

            # ── MOMENTUM ENTRY ─────────────────────────────────────────
            min_bias = 0.8 if sell_prob > 0.50 else 1.5
            if features.get("net_bias", 0) < -min_bias:
                return "SELL", "momentum_bias", sell_prob, ms_data, buy_prob, sell_prob

            # ── MTF ALIGNMENT ──────────────────────────────────────────
            if m5_trend == "bearish" and m15_trend == "bearish":
                return "SELL", "mtf_alignment", sell_prob, ms_data, buy_prob, sell_prob

        # =========================
        # DEFAULT
        # =========================
        return "WAIT", "no_confluence", 0.0, ms_data, buy_prob, sell_prob


    except Exception as e:
        print(f"Entry decision error: {e}")
        return "WAIT", "error", 0.5, None, 0.0, 0.0

# ================= EXECUTE TRADE =================

def execute_trade(direction, df, ai_prob, ms_data=None):
    direction = direction.lower()

    try:
        account = mt5.account_info()
        symbol = mt5.symbol_info(SYMBOL)
        tick = mt5.symbol_info_tick(SYMBOL)

        if not account or not symbol or not tick:
            print("❌ MT5 Data Fetch Failed")
            return False

        # Prevent duplicate trade same candle
        current_candle_time = df['time'].iloc[-1]
        if hasattr(execute_trade, "last_candle_time"):
            if execute_trade.last_candle_time == current_candle_time:
                return False
        execute_trade.last_candle_time = current_candle_time

        # Candle confirmation
        last_candle = df.iloc[-2]
        body = abs(last_candle["close"] - last_candle["open"])
        rng = last_candle["high"] - last_candle["low"]

        if rng == 0 or body < rng * 0.25:
            print("🚫 Weak candle — skip")
            return False

        price = tick.ask if direction == "buy" else tick.bid
        pip   = symbol.point * 10

        # ================= ATR =================
        _atr_series = (df['high'] - df['low']).rolling(14).mean()
        atr = float(_atr_series.iloc[-1]) if not pd.isna(_atr_series.iloc[-1]) else float((df['high'] - df['low']).tail(5).mean() or 1.0)

        # ================= SL LOGIC =================
        if ms_data and direction == "buy" and ms_data.get("support"):
            sl = ms_data["support"] - (5 * pip)
        elif ms_data and direction == "sell" and ms_data.get("resistance"):
            sl = ms_data["resistance"] + (5 * pip)
        else:
            fvgs = detect_fvg(df)
            obs  = detect_order_blocks(df)
            sl   = calculate_liquidity_based_sl(direction, price, df, fvgs, obs, atr)

        if sl is None or abs(price - sl) < (10 * pip):
            sl = price - (80 * pip) if direction == "buy" else price + (80 * pip)

        # ================= BROKER STOPS_LEVEL CHECK =================
        # Must happen BEFORE lot calc so lot reflects final SL distance.
        # Some brokers require SL/TP to be at least stops_level points away from price.
        min_stop_dist = symbol.trade_stops_level * symbol.point
        if min_stop_dist > 0:
            if abs(price - sl) < min_stop_dist:
                sl = price - min_stop_dist * 1.2 if direction == "buy" else price + min_stop_dist * 1.2
                print(f"⚠️ SL adjusted to broker min stops_level: {round(sl, symbol.digits)}")

        # ================= TP LOGIC =================
        tp = calculate_liquidity_based_tp(direction, price, df)

        # Apply broker stops_level to TP as well
        if min_stop_dist > 0 and tp is not None:
            if abs(price - tp) < min_stop_dist:
                tp = price + min_stop_dist * 1.2 if direction == "buy" else price - min_stop_dist * 1.2

        # ================= LOT SIZE =================
        sl_dist = abs(price - sl)
        if sl_dist <= 0:
            return False

        stop_pips = sl_dist / (symbol.point * 10)
        lot = calculate_dynamic_lot(stop_pips, ai_prob, df)

        if lot <= 0:
            return False

        # ================= MARGIN CHECK =================
        # Don't submit if margin insufficient — prevents rejected orders
        required_margin = mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
            SYMBOL, lot, price
        )
        if required_margin is None:
            print("⚠️ Margin calc failed — skipping trade")
            return False
        if account.margin_free < required_margin * 1.2:  # 20% safety buffer
            print(f"⛔ Insufficient margin — need ${required_margin:.2f}, have ${account.margin_free:.2f}")
            return False

        # ================= RR CHECK =================
        risk   = abs(price - sl)
        reward = abs(tp - price) if tp else 0

        # Remove arbitrary TP clamp and enforce a strict RR floor
        if tp and risk > 0 and reward / risk < 1.2:
            _atr_now_s = (df['high'] - df['low']).rolling(14).mean()
            atr_now = float(_atr_now_s.iloc[-1]) if not pd.isna(_atr_now_s.iloc[-1]) else atr
            # Forcematically push the TP to a 1.2 RR if the structure target was too small
            tp = price + (risk * 1.2) if direction == "buy" else price - (risk * 1.2)
            reward = abs(tp - price)

        # Final strict guard: Never send a negative math trade to the broker
        if risk > 0 and (reward / risk) < 1.0:
            print(f"🚫 Trade rejected: Bad Risk/Reward ratio ({(reward / risk):.2f})")
            return False

        # ================= ORDER (MARKET) =================
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       SYMBOL,
            "volume":       lot,
            "type":         mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
            "price":        round(price, symbol.digits),
            "sl":           round(sl, symbol.digits),
            "tp":           round(tp, symbol.digits) if tp else 0.0,
            "deviation":    50,
            "magic":        MAGIC_NUMBER,
            "comment":      f"V4 SMC {ai_prob:.2f}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        print(f"🚀 {direction.upper()} | Price:{round(price,2)} | "
              f"Lot:{lot} | SL:{round(sl,2)} | TP:{round(tp,2) if tp else 0}")

        result = mt5.order_send(request)

        # Fallback fill mode: IOC → FOK → RETURN
        if result and result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
            request["type_filling"] = mt5.ORDER_FILLING_FOK
            result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
            request["type_filling"] = mt5.ORDER_FILLING_RETURN
            result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            rr = (reward / risk) if risk > 0 else 0
            print(f"✅ Trade Opened | RR: {rr:.2f}")
            return {
                "entry":  price,
                "sl":     round(sl, symbol.digits),
                "tp":     round(tp, symbol.digits) if tp else 0.0,
                "lot":    lot,
                "ticket": result.order
            }
        else:
            print(f"❌ Failed: {result.comment if result else 'No result'}")
            return None

    except Exception as e:
        print(f"💥 EXECUTION ERROR: {e}")
        return None

def get_current_drawdown():
    """Returns current drawdown % based on the highest point between Balance and previous Equity Peak."""
    global equity_peak

    account = mt5.account_info()
    if not account:
        return 0.0
        
    # The true peak is at LEAST the starting balance.
    # If the bot restarts during a loss, this prevents the bot from 
    # resetting the peak to the currently damaged equity level.
    true_peak = max(equity_peak, account.balance)

    if account.equity > true_peak:
         true_peak = account.equity
         equity_peak = true_peak # Update global

    if true_peak == 0: return 0.0

    return ((true_peak - account.equity) / true_peak) * 100

def calculate_dynamic_lot(stop_pips, ai_confidence, df):
    """
    SAFE PRO VERSION (M5 XAUUSD):
    Equity Risk → AI Scaling → Volatility → Drawdown → Hard Caps → Broker Normalize
    """

    try:
        account = mt5.account_info()
        symbol = mt5.symbol_info(SYMBOL)

        if account is None or symbol is None:
            return 0.01

        equity = account.equity

        # =============================
        # 1. USE GLOBAL RISK_PER_TRADE
        # =============================
        # Uses the top-level RISK_PER_TRADE constant so user can tune it in one place
        risk_amount = equity * RISK_PER_TRADE

        # =============================
        # 2. FIXED GOLD PIP VALUE 🔥
        # =============================
        # Gold standard: 0.01 lot ≈ $0.1 per pip
        pip_value = 1.0

        # =============================
        # 3. SAFE STOP LOSS (CRITICAL)
        # =============================
        # M5 realistic SL
        stop_pips = max(stop_pips, 12.0)

        # =============================
        # 4. BASE LOT (CONTROLLED)
        # =============================
        base_lot = risk_amount / (stop_pips * pip_value)

        dynamic_max_base = (equity / 1000.0) * 0.03
        dynamic_max_base = max(0.01, dynamic_max_base)
        base_lot = min(base_lot, dynamic_max_base)

        # =============================
        # 5. AI CONFIDENCE
        # =============================
        ai_confidence = max(0.0, min(ai_confidence, 1.0))

        if ai_confidence < 0.55:
            ai_multiplier = 0.7
        else:
            strength = (ai_confidence - 0.55) / 0.45
            ai_multiplier = 0.7 + (strength * 0.5)  # reduced (0.7 → 1.2)

        # =============================
        # 6. VOLATILITY (ATR)
        # =============================
        high, low, close = df['high'], df['low'], df['close']

        tr = np.maximum(
            high - low,
            np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
        )

        atr = tr.rolling(14).mean().iloc[-1]
        volatility = atr / close.iloc[-1]

        if volatility > 0.015:
            vol_multiplier = 0.7
        elif volatility < 0.005:
            vol_multiplier = 1.1
        else:
            vol_multiplier = 1.0

        # =============================
        # 7. DRAWDOWN PROTECTION
        # =============================
        drawdown = get_current_drawdown()

        dd_multiplier = 1.0

        # =============================
        # 8. FINAL LOT
        # =============================
        final_lot = base_lot * ai_multiplier * vol_multiplier * dd_multiplier

        # =============================
        # 9. HARD ACCOUNT-SAFE CAPS 🔥
        # =============================
        if equity <= 300:
            max_lot = 0.01
        elif equity <= 1000:
            max_lot = 0.02
        elif equity <= 3000:
            max_lot = 0.03
        elif equity <= 5000:
            max_lot = 0.05
        else:
            max_lot = 0.08  # max for 10k+

        final_lot = min(final_lot, max_lot)

        # =============================
        # 10. BROKER NORMALIZATION
        # =============================
        step = symbol.volume_step
        final_lot = max(symbol.volume_min, min(final_lot, symbol.volume_max))
        final_lot = round(final_lot / step) * step

        final_lot = float(f"{final_lot:.2f}")

        # =============================
        # DEBUG
        # =============================
        print(
            f"📦 LOT FINAL: {final_lot} | "
            f"Base:{base_lot:.3f} | "
            f"SL:{stop_pips} | "
            f"AI:{ai_multiplier:.2f}x | "
            f"Vol:{vol_multiplier:.2f}x | "
            f"DD:{dd_multiplier:.2f}x ({drawdown:.2f}%) | "
            f"Equity:{equity:.2f}"
        )

        return final_lot

    except Exception as e:
        print(f"❌ Lot calculation failed: {e}")
        return 0.01



def _read_trades_csv():
    """Read trades.csv safely — handles old 16-field rows mixed with new 17-field rows."""
    if not os.path.exists("trades.csv"):
        return None
    try:
        return pd.read_csv("trades.csv", on_bad_lines="skip")
    except TypeError:
        # pandas < 1.3 fallback
        return pd.read_csv("trades.csv", error_bad_lines=False)
    except Exception:
        return pd.read_csv("trades.csv")


# ================= PERFORMANCE TRACKER =================

def get_performance_stats():
    """
    Reads trades.csv and returns full performance statistics.
    Call this any time to see how the bot is doing.
    """
    try:
        df = _read_trades_csv()
        if df is None or df.empty or "result" not in df.columns:
            return None

        # Only look at closed trades (result is not None/NaN)
        closed = df[df["result"].notna() & (df["result"] != "None")].copy()
        if closed.empty:
            return None

        closed["result"] = pd.to_numeric(closed["result"], errors="coerce")
        closed = closed.dropna(subset=["result"])
        if closed.empty:
            return None

        total       = len(closed)
        wins        = closed[closed["result"] > 0]
        losses      = closed[closed["result"] <= 0]
        win_count   = len(wins)
        loss_count  = len(losses)
        win_rate    = (win_count / total * 100) if total > 0 else 0

        gross_profit = wins["result"].sum()
        gross_loss   = abs(losses["result"].sum())
        net_profit   = gross_profit - gross_loss
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        avg_win  = wins["result"].mean()  if win_count  > 0 else 0
        avg_loss = abs(losses["result"].mean()) if loss_count > 0 else 0
        rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

        # Max consecutive losses
        max_consec_loss = 0
        curr_streak     = 0
        for r in closed["result"]:
            if r <= 0:
                curr_streak += 1
                max_consec_loss = max(max_consec_loss, curr_streak)
            else:
                curr_streak = 0

        # Best / worst trade
        best_trade  = closed["result"].max()
        worst_trade = closed["result"].min()

        # Today's P&L
        if "time" in closed.columns:
            try:
                closed["time"] = pd.to_datetime(closed["time"], errors="coerce")
                today = datetime.now().strftime("%Y-%m-%d")
                today_trades = closed[closed["time"].dt.strftime("%Y-%m-%d") == today]
                daily_pnl  = today_trades["result"].sum()
                daily_count = len(today_trades)
            except Exception:
                daily_pnl   = 0
                daily_count = 0
        else:
            daily_pnl   = 0
            daily_count = 0

        return {
            "total_trades":    total,
            "win_count":       win_count,
            "loss_count":      loss_count,
            "win_rate":        round(win_rate, 1),
            "net_profit":      round(net_profit, 2),
            "gross_profit":    round(gross_profit, 2),
            "gross_loss":      round(gross_loss, 2),
            "profit_factor":   round(profit_factor, 2),
            "avg_win":         round(avg_win, 2),
            "avg_loss":        round(avg_loss, 2),
            "rr_ratio":        round(rr_ratio, 2),
            "best_trade":      round(best_trade, 2),
            "worst_trade":     round(worst_trade, 2),
            "max_consec_loss": max_consec_loss,
            "daily_pnl":       round(daily_pnl, 2),
            "daily_trades":    daily_count,
        }

    except Exception as e:
        print(f"❌ Performance stats error: {e}")
        return None


def print_performance_report():
    """Print a formatted performance report to console and send to Telegram."""
    stats = get_performance_stats()

    if not stats:
        print("📊 No completed trades yet.")
        return

    wr   = stats["win_rate"]
    pf   = stats["profit_factor"]
    rr   = stats["rr_ratio"]
    net  = stats["net_profit"]
    dpnl = stats["daily_pnl"]

    # Colour-coded verdict
    if wr >= 60 and pf >= 1.5:
        verdict = "✅ PROFITABLE"
    elif wr >= 50 and pf >= 1.2:
        verdict = "⚠️  BORDERLINE"
    else:
        verdict = "❌ NEEDS REVIEW"

    report = (
        f"\n{'='*48}\n"
        f"  📊 BOT PERFORMANCE REPORT  {verdict}\n"
        f"{'='*48}\n"
        f"  Trades      : {stats['total_trades']}  "
        f"(W:{stats['win_count']} / L:{stats['loss_count']})\n"
        f"  Win Rate    : {wr}%\n"
        f"  Net P&L     : ${net:+.2f}\n"
        f"  Today P&L   : ${dpnl:+.2f}  ({stats['daily_trades']} trades)\n"
        f"  Profit Factor: {pf}\n"
        f"  Avg Win     : ${stats['avg_win']:.2f}  |  "
        f"Avg Loss: ${stats['avg_loss']:.2f}\n"
        f"  RR Ratio    : {rr:.2f}:1\n"
        f"  Best Trade  : ${stats['best_trade']:.2f}  |  "
        f"Worst: ${stats['worst_trade']:.2f}\n"
        f"  Max Loss Streak: {stats['max_consec_loss']}\n"
        f"{'='*48}"
    )

    print(report)

    # Send to Telegram
    tg_msg = (
        f"📊 *Performance Report*\n"
        f"{'—'*28}\n"
        f"Trades: {stats['total_trades']}  |  Win Rate: *{wr}%*\n"
        f"Net P\\&L: `${net:+.2f}`\n"
        f"Today: `${dpnl:+.2f}` ({stats['daily_trades']} trades)\n"
        f"Profit Factor: `{pf}`  |  RR: `{rr:.2f}:1`\n"
        f"Max Loss Streak: `{stats['max_consec_loss']}`\n"
        f"Verdict: {verdict}"
    )
    try:
        send_telegram_msg(tg_msg)
    except Exception:
        pass


def update_trade_result_in_csv(ticket, profit, entry_price=None):
    """
    Update the result field for a completed trade in trades.csv.
    Matches by ticket first, falls back to entry_price if no ticket column.
    """
    try:
        if not os.path.exists("trades.csv"):
            return

        df = _read_trades_csv()
        if df is None:
            return

        if "result" not in df.columns:
            df["result"] = None

        updated = False

        # Primary match: by ticket number
        if "ticket" in df.columns and ticket is not None:
            mask = df["ticket"].astype(str) == str(ticket)
            if mask.any():
                df.loc[mask, "result"] = profit
                updated = True

        # Fallback: match by entry price (for old rows without ticket)
        if not updated and entry_price is not None and "entry" in df.columns:
            mask = (
                (pd.to_numeric(df["entry"], errors="coerce").round(2) == round(float(entry_price), 2)) &
                (df["result"].isna() | (df["result"] == "") | (df["result"] == "None"))
            )
            if mask.any():
                # Update only the most recent matching row
                idx = df[mask].index[-1]
                df.loc[idx, "result"] = profit
                updated = True

        if updated:
            df.to_csv("trades.csv", index=False)
        else:
            print(f"⚠️ Could not match trade ticket={ticket} entry={entry_price} in trades.csv")

    except Exception as e:
        print(f"❌ CSV result update error: {e}")


def backfill_results_from_mt5():
    """
    Query MT5 deal history and fill in missing 'result' values in trades.csv.
    Called on startup to recover results from previous sessions.
    Matches by ticket (position_id), not by price — price-based matching was
    unreliable because entry price rarely equals exit price.
    """
    try:
        if not os.path.exists("trades.csv"):
            return

        # Read with on_bad_lines="skip" to handle old 16-field rows mixed with new 17-field rows
        try:
            df = pd.read_csv("trades.csv", on_bad_lines="skip")
        except TypeError:
            # pandas < 1.3 fallback
            df = pd.read_csv("trades.csv", error_bad_lines=False)

        if df.empty:
            return

        # Find rows with empty result
        missing_mask = df["result"].isna() | (df["result"].astype(str) == "") | (df["result"].astype(str) == "None")
        if not missing_mask.any():
            print("✅ trades.csv: all results already filled")
            return

        missing_count = missing_mask.sum()
        print(f"🔍 Backfilling results for {missing_count} trades from MT5 history...")

        # Query last 30 days of deal history
        from_date = datetime.now() - timedelta(days=30)
        deals = mt5.history_deals_get(from_date, datetime.now())

        if deals is None or len(deals) == 0:
            print("⚠️ No MT5 deal history found — trades may still be open")
            return

        # Build map: position_id → cumulative profit from closing deals
        # MT5 closing deals (entry=1) have position_id linking back to opening
        deal_profit_map = {}
        for deal in deals:
            if deal.magic != MAGIC_NUMBER:
                continue
            if deal.entry == 1:  # entry=1 = closing deal
                pos_id = int(deal.position_id)
                # Sum in case multiple partial closes for same position
                deal_profit_map[pos_id] = deal_profit_map.get(pos_id, 0.0) + round(float(deal.profit), 2)

        if not deal_profit_map:
            print("⚠️ No closing deals found in MT5 history for this magic number")
            return

        # Match by ticket column (which stores position_id)
        if "ticket" not in df.columns:
            print("⚠️ No ticket column in trades.csv — cannot match by position_id")
            return

        filled = 0
        for idx in df[missing_mask].index:
            try:
                ticket = int(df.loc[idx, "ticket"])
                if ticket in deal_profit_map:
                    df.loc[idx, "result"] = deal_profit_map[ticket]
                    filled += 1
            except Exception:
                continue

        if filled > 0:
            df.to_csv("trades.csv", index=False)
            print(f"✅ Backfilled {filled}/{missing_count} trade results from MT5 history")
            if filled < missing_count:
                print(f"   {missing_count - filled} trades still open or history unavailable")
        else:
            print("⚠️ No matches found — tickets may be outside 30-day window")

    except Exception as e:
        print(f"❌ Backfill error: {e}")


# ── Telegram credentials ──────────────────────────────────────────────────────
# Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as environment variables or in .env
#   export TELEGRAM_BOT_TOKEN="your_token"
#   export TELEGRAM_CHAT_ID="your_chat_id"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "")
if not TELEGRAM_BOT_TOKEN:
    print("⚠️  TELEGRAM_BOT_TOKEN not set — Telegram alerts disabled. "
          "Set it in your .env file or environment before going live.")
# ─────────────────────────────────────────────────────────────────────────────

def send_telegram_msg(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        with open("sentinel_failsafe.log", "a") as f:
            f.write(f"[{time.ctime()}] {message}\n" + "-"*20 + "\n")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}

    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            return True
        print(f"⚠️ Telegram Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

    # Fallback: Save to file if sending fails
    with open("sentinel_failsafe.log", "a") as f:
        f.write(f"[{time.ctime()}] {message}\n" + "-"*20 + "\n")
    return False

# ================= MAIN LOOP =================

def run():
    """Main trading loop"""

    global last_train, last_entry_time, emergency_stop
    
    last_train = 0 
    last_entry_time = 0
    last_market_save = time.time()
    last_full_save = time.time()
    last_smc_update = 0
    last_scan_print = 0
    last_news_print = 0
    last_news_update = 0
    last_mtf_update = time.time()
    last_debug_print = 0
    last_no_edge_print = 0
    last_ny_block_print = 0          # throttle for NY-end entry block message
    last_candle_veto_print = 0       # throttle for candle-pattern veto message
    last_perf_report = time.time()   # performance report timer
    last_signal = None
    cached_news_status = {"current_event": None}
    cached_avoid_trade = False
    cached_news_reason = None
    avoid_trade = False
    news_reason = "Initializing..."
    news_status = {"current_event": None}

    print("\n" + "="*60)
    print("🚀 SMC-AI-GOLD-SENTINEL V4.0 - M5 GOLD TRADING BOT")
    print("="*60)
    print("Features: SMC Concepts + Multi-Timeframe + AI Decision")
    print("Target: 80-90% Win Rate")
    print("="*60 + "\n")

    # Initialize MT5
    print("🔌 Checking MT5 connection...")

    if not MT5_INITIALIZED or not check_mt5_connection():
        if not initialize_mt5():
            print("❌ MT5 initialization failed!")
            sys.exit(1)

    # ✅ FORCE ENABLE SYMBOL HERE
    info = mt5.symbol_info(SYMBOL)

    if info is None:
        print(f"❌ Symbol {SYMBOL} NOT FOUND in broker")
        sys.exit(1)

    if not info.visible:
        print(f"👁️ Enabling {SYMBOL} in Market Watch...")
        if not mt5.symbol_select(SYMBOL, True):
            print(f"❌ Failed to enable {SYMBOL}")
            sys.exit(1)

    print("✅ MT5 ready, symbol:", SYMBOL)

    # Load persisted data (trade history, model state, heatmap)
    initialize_data_system()

    # Recover any missing trade results from MT5 history
    backfill_results_from_mt5()

    mtf_data = get_multi_timeframe_data()
    last_mtf_update = time.time()

    # Enrich mtf_data with trend keys BEFORE initialize_ai so training has them
    if mtf_data:
        overall_trend, trend_strength = detect_mtf_trend(mtf_data)
        mtf_data["overall_trend"] = overall_trend
        mtf_data["trend_score"]   = trend_strength
        for tf_name, tf_df in list(mtf_data.items()):
            if isinstance(tf_df, pd.DataFrame) and len(tf_df) >= 20:
                if tf_name == "H1":
                    mtf_data[f"{tf_name}_trend"] = detect_market_regime(tf_df, fast_ema=50, slow_ema=200)
                else:
                    mtf_data[f"{tf_name}_trend"] = detect_market_regime(tf_df)

    # Fetch live M5 data and pass to initialize_ai so it can train immediately
    # without waiting for the market history file to accumulate
    live_df_for_ai = get_data(TIMEFRAME, BARS)
    initialize_ai(live_df=live_df_for_ai)

    LOOP_DELAY = 5

    print("\n✅ Bot started - Entering main loop...")
    print("Press Ctrl+C to stop\n")

    while True:
        try:
          
            # =============================
            # CONNECTION CHECK
            # =============================

            if not check_mt5_connection():

                print("⚠️ MT5 connection lost")

                if not reconnect_mt5():

                    print("Waiting before retrying...")
                    time.sleep(30)
                    continue

            # =============================
            # EMERGENCY STOP
            # =============================

            if emergency_stop:
                print("⚠️ EMERGENCY STOP ACTIVATED - Bot disabled")
                break

            # =============================
            # GET MARKET DATA FIRST
            # =============================
            # LOOP_BARS = 200 candles (fast, covers all SMC functions)
            df = get_data(TIMEFRAME, LOOP_BARS)

            if df is None or len(df) < 100:
                time.sleep(LOOP_DELAY)
                continue

            df = df.copy().reset_index(drop=True)

            if 'mtf_data' not in locals() or time.time() - last_mtf_update > 60:
                mtf_data = get_multi_timeframe_data()
                last_mtf_update = time.time()

                # ── Enrich mtf_data with trend keys used by AI & entry logic ──
                if mtf_data:
                    overall_trend, trend_strength = detect_mtf_trend(mtf_data)
                    mtf_data["overall_trend"] = overall_trend
                    mtf_data["trend_score"]   = trend_strength
                    # Snapshot keys first to avoid "dict changed size during iteration"
                    for tf_name, tf_df in list(mtf_data.items()):
                        if isinstance(tf_df, pd.DataFrame) and len(tf_df) >= 20:
                            # H1: EMA50/200 — responds to multi-hour regime shifts
                            # All others: default EMA20/50
                            if tf_name == "H1":
                                mtf_data[f"{tf_name}_trend"] = detect_market_regime(tf_df, fast_ema=50, slow_ema=200)
                            else:
                                mtf_data[f"{tf_name}_trend"] = detect_market_regime(tf_df)
            
            print(f"📊 Data: {len(df)} candles")


            news_active, event_name, _ = get_enhanced_news_signal()
            
            # =============================
            # MANAGE OPEN POSITIONS
            # =============================

            # Fetch all positions then filter to this bot's trades only
            all_positions = mt5.positions_get(symbol=SYMBOL)

            if all_positions is None:
                print("⚠️ Failed to retrieve positions")
                time.sleep(LOOP_DELAY)
                continue

            positions = [p for p in all_positions if p.magic == MAGIC_NUMBER]

            if positions and len(positions) > 0:

                print(f"\n📊 Managing {len(positions)} open position(s)...")

                _, _, ai_prob, *_ = should_enter_trade(df, mtf_data, news_active)

                manage_open_positions(df, mtf_data, ai_prob)

                time.sleep(LOOP_DELAY)
                continue

            # =============================
            # RISK ENGINE (ALWAYS RUN)
            # =============================

            reset_daily_loss()

            update_equity_protection()

            if check_daily_loss():
                time.sleep(60)
                continue

            if check_loss_streak():
                print(f"⛔ Loss streak triggered: {consecutive_losses}/{MAX_CONSECUTIVE_LOSSES}")
                time.sleep(60)
                continue

            if trading_paused:
                print(f"⛔ Trading paused: {PAUSE_REASON}")
                time.sleep(60)
                continue

            # =============================
            # DATA SAVING
            # =============================

            if time.time() - last_market_save > 60:
                save_market_history(df)          # save full fetch, not just 100 candles
                last_market_save = time.time()

            if time.time() - last_full_save > 300:
                save_all_data()
                last_full_save = time.time()

            # Performance report every 2 hours
            if time.time() - last_perf_report > 7200:
                print_performance_report()
                last_perf_report = time.time()

            # =============================
            # ENTRY FILTERS (SESSION & SPREAD)
            # =============================

            # 1. Session Filter (London/NY only)

            # 2. Spread Filter
            if spread_too_high():
                print(f"⚠️ Spread too high: {get_spread()} (max: {MAX_SPREAD})")
                time.sleep(LOOP_DELAY)
                continue

            # 3. Volatility Kill Switch
            if volatility_kill_switch(df):
                time.sleep(1) # Wait for the market to calm down
                continue

            # =============================
            # NEWS STATUS PRINT
            # =============================

            now_ts = time.time()

            # 1. PERIODIC NEWS UPDATE (Every 60 seconds)
            if now_ts - last_news_update > 60:
                try:
                    # We get the full status for the "current_event" name
                    news_status = get_live_news_filter_status()
                    # We get the final signal for the trade decision
                    avoid_trade, news_reason, _ = get_enhanced_news_signal()
            
                    last_news_update = now_ts
                except Exception as e:
                    print(f"⚠️ News Logic Error: {e}")
                    # On error, we keep the previous values for safety

            # 2. PERIODIC STATUS PRINT (Every 60 seconds)
            # This keeps your console clean
            if now_ts - last_news_print > 60:
                event_name = news_status.get('current_event') or "None"
                status_icon = "🔴 BLOCK" if avoid_trade else "🟢 CLEAR"
                print(f"📰 News Status: {status_icon} | Event: {event_name}")
                last_news_print = now_ts

            # 3. TRADE BLOCKING LOGIC
            if avoid_trade:
                # We only print this once a minute to avoid spamming the log
                # but we 'continue' every time to ensure no trades are placed.
                time.sleep(LOOP_DELAY) 
                continue

            # =============================
            # UPDATE SMC STRUCTURE
            # =============================

            if time.time() - last_smc_update > 30:
                update_liquidity_heatmap(df)
                detect_fvg(df)
                detect_order_blocks(df)
                detect_liquidity_pools(df)
                calculate_volume_profile(df)
                last_smc_update = time.time()

            # 🔥 NEW: Market Structure (BOS/CHoCH)
            ms_signal, ms_conf, ms_data = get_mss_signal(df)

            # =============================
            # AI TRAINING — BACKGROUND THREAD
            # =============================
            retrain_due = (time.time() - last_train > RETRAIN_HOURS * 3600)

            if retrain_due and not getattr(check_loss_streak, "_training_active", False):
                last_train = time.time()   # mark now so we don't double-trigger

                def _background_train(df_snap):
                    try:
                        check_loss_streak._training_active = True
                        print(f"🔄 [BG] Retraining AI model with {len(df_snap)} candles...")
                        
                        # NO LOCK HERE! Let the heavy math run freely in the background
                        train_models(df_snap)
                        
                        print("✅ [BG] Retrain complete — new model active")
                    except Exception as e:
                        print(f"❌ [BG] Retrain error: {e}")
                    finally:
                        check_loss_streak._training_active = False

                def _fetch_and_train():
                    train_df = get_data(TIMEFRAME, BARS)
                    if train_df is not None and len(train_df) >= 2000:
                        _background_train(train_df)
                    else:
                        print("⚠️ [BG] Could not fetch enough training data (need 2000+)")
                        global last_train
                        last_train = 0  # CRITICAL: Reset timer so it retries instead of waiting 4 hours
                        check_loss_streak._training_active = False

                threading.Thread(target=_fetch_and_train, daemon=True).start()

            # =============================
            # ENTRY DECISION
            # =============================

            mtf_data = mtf_data or {}

            signal, reason, ai_prob, ms_data, buy_prob, sell_prob = should_enter_trade(df, mtf_data, news_active)
            edge = abs(buy_prob - sell_prob)

            if time.time() - last_scan_print > 10:
                print(f"🧠 AI DEBUG → Buy: {buy_prob:.2f} | Sell: {sell_prob:.2f} | Reason: {reason}")
                last_scan_print = time.time()

            if edge < 0.08:
                if time.time() - last_scan_print > 30:
                    print("⚠️ AI No Edge (market unclear)")
                    last_scan_print = time.time()
                continue

            # =============================
            # PRE-CALCULATIONS (SMC context for logging)
            # =============================

            sweep_signal = detect_liquidity_sweep(df, lookback=20, min_sweep=0.2)

            fvg_data = detect_fvg(df)
            current_price = df["close"].iloc[-1]
            fvg_signal = get_fvg_signal(fvg_data, current_price, df)

            m15_trend = mtf_data.get("M15_trend")
            h1_trend  = mtf_data.get("H1_trend")
            h4_trend  = mtf_data.get("H4_trend")
            m5_trend  = ms_data.get("trend") if ms_data else None

            # should_enter_trade() is the single entry gate — no duplicate filtering here

            # =============================
            # LOGGING
            # =============================
            if sweep_signal:
                print(f"🕵️ SMC ALERT: {sweep_signal.upper()} detected")

            if fvg_signal:
                print(f"⚡ FVG: {fvg_signal}")

            # =============================
            # DECISION OUTPUT
            # =============================
            if signal is None or signal == "WAIT":

                if time.time() - last_scan_print > 30:

                    ms_tf = ms_data.get('tf', 'M5') if ms_data else 'M5'

                    print(
                        f"📊 TF CHECK → "
                        f"H4:{h4_trend} | H1:{h1_trend} | M15:{m15_trend} | M5:{m5_trend} | "
                        f"Sweep:{sweep_signal} | FVG:{fvg_signal} | "
                        f"AI:{ai_prob:.2f} | Reason:{reason}"
                    )

                    last_scan_print = time.time()          
             
            elif signal in ["BUY", "SELL"]:
             
                entry_interval = 60 if ai_prob > 0.85 else 90

                if time.time() - last_entry_time < entry_interval:
                    print("⏳ Waiting entry cooldown...")
                    continue

                if trading_paused:
                    print("⛔ Trading paused — skipping execution")
                    continue

                # Asymmetric threshold — matches AI EDGE FILTER in should_enter_trade
                # BUY needs ≥ 0.55, SELL needs ≥ 0.45
                threshold_needed = 0.55 if signal == "BUY" else 0.45
                if ai_prob < threshold_needed:
                    print(f"⚠️ AI Threshold not met ({ai_prob:.2f} < {threshold_needed} for {signal}) — skipped")
                    continue

                if m5_trend == "range" and ai_prob < 0.7:
                    print("⚠️ Skipping weak trade in range market")
                    continue

                # ── HARD LIMIT: MAX_POSITIONS ──────────────────────────────
                # Prevents runaway position opening. positions already filtered
                # by MAGIC_NUMBER so this only counts THIS bot's positions.
                if len(positions) >= MAX_POSITIONS:
                    print(f"⛔ MAX_POSITIONS reached ({len(positions)}/{MAX_POSITIONS}) — skipping new entry")
                    continue

                # ── NY SESSION END PROTECTION ──────────────────────────────
                # Block new entries in last NY_END_BLOCK_MINUTES of NY session.
                # Open positions are still managed normally (trail/BE/partial TP).
                # Avoids fresh trades sitting through the 22:00-00:00 GMT vacuum.
                if is_near_ny_close():
                    if time.time() - last_ny_block_print > 300:
                        print(f"🌙 Within {NY_END_BLOCK_MINUTES}min of NY close — blocking new entries (positions still managed)")
                        last_ny_block_print = time.time()
                    continue

                # ── CANDLE PATTERN VETO ────────────────────────────────────
                # Reads the last 1-3 candles. Refuses entry when anatomy
                # contradicts trade direction (shooting star at top of rally,
                # hammer at bottom of selloff, engulfing, tweezer, etc.).
                # Pure price-action filter — independent of AI.
                veto, veto_pattern = candle_pattern_veto(df, signal)
                if veto:
                    if time.time() - last_candle_veto_print > 60:
                        print(f"🕯️ CANDLE VETO: {signal} blocked by {veto_pattern}")
                        last_candle_veto_print = time.time()
                    continue

                print(f"\n🎯 ENTRY SIGNAL DETECTED: {signal.upper()}")
                print(f"📜 Strategy: {reason}")
                print(f"🧠 AI Confidence: {ai_prob:.2f}")
                print(f"📊 Spread: {get_spread()} | M5: {m5_trend} | M15: {m15_trend}")
                
                # 🚀 EXECUTE
                trade_result = execute_trade(signal, df, ai_prob, ms_data)

                if trade_result:

                    actual_entry = trade_result["entry"]
                    sl_price     = trade_result["sl"]
                    tp_price     = trade_result["tp"]
                    lot          = trade_result["lot"]
                    ticket       = trade_result.get("ticket", None)

                    log_trade({
                        "time":      datetime.now(),
                        "ticket":    ticket,
                        "symbol":    SYMBOL,
                        "signal":    signal,
                        "entry":     actual_entry,
                        "sl":        sl_price,
                        "tp":        tp_price,
                        "lot":       lot,
                        "ai_prob":   ai_prob,
                        "buy_prob":  buy_prob,
                        "sell_prob": sell_prob,
                        "sweep":     sweep_signal,
                        "fvg":       fvg_signal,
                        "trend":     m5_trend,
                        "spread":    get_spread(),
                        "reason":    reason,
                        "result":    None
                    })

                    # --- LOGIC FOR SIGNAL COLOR ---
                    side_emoji = "🔵 BUY" if "BUY" in signal.upper() else "🔴 SELL" if "SELL" in signal.upper() else "⚠️ ALERT"

                    safe_reason = str(reason).replace('_', ' ')
                    safe_sweep = str(sweep_signal).replace('_', ' ')
                    safe_fvg = str(fvg_signal).replace('_', ' ')

                    msg = (f"{side_emoji} **GOLD ENTRY**\n"
                           f"⏱ TF: M5 | 📦 Lot: {lot}\n"
                           f"----------------------------\n"
                           f"🎯 Entry: `{actual_entry:.2f}`\n"
                           f"🛑 SL:    `{sl_price:.2f}`\n"
                           f"✅ TP:    `{tp_price:.2f}`\n"
                           f"----------------------------\n"
                           f"🧠 AI Prob: `{ai_prob:.2f}`\n"
                           f"📊 Spread:  `{get_spread()}`\n"
                           f"🕵️ Sweep:   {safe_sweep}\n"
                           f"⚡ FVG:     {safe_fvg}\n"
                           f"📜 Reason:  _{safe_reason}_")
                    send_telegram_msg(msg)

                    last_entry_time = time.time()
                    last_signal = signal

            time.sleep(LOOP_DELAY)

        except KeyboardInterrupt:

            print("\n\nShutdown requested by user")
            break

        except Exception as e:

            print(f"Main loop error: {e}")
            # Track consecutive loop errors — trigger emergency stop after 10 in a row
            if not hasattr(run, "_consecutive_errors"):
                run._consecutive_errors = 0
            run._consecutive_errors += 1
            if run._consecutive_errors >= 10:
                print("🚨 10 consecutive loop errors — triggering EMERGENCY STOP")
                emergency_stop = True
                try:
                    send_telegram_msg(f"🚨 EMERGENCY STOP\nBot crashed 10 times: {e}")
                except Exception:
                    pass
            time.sleep(10)
        else:
            # Reset error counter on successful loop iteration
            if hasattr(run, "_consecutive_errors"):
                run._consecutive_errors = 0

    # =============================
    # CLEANUP
    # =============================

    print("\nShutting down - saving all data...")

    save_all_data()

    mt5.shutdown()

    print("Bot stopped.")


if __name__ == "__main__":
    run()

