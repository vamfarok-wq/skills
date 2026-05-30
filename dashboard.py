#!/usr/bin/env python3
"""
Live Monitoring Dashboard — SMC-AI-GOLD-SENTINEL
=================================================
Flask web dashboard running on http://localhost:5000

Shows
-----
  • Live equity curve (updates every 5s)
  • Rolling 20-trade win rate gauge
  • Open positions with floating P&L
  • Recent 20 trades with outcome
  • System health: MT5 connection, model age, last news fetch

Usage
-----
    # Run alongside the bot (separate terminal)
    python dashboard.py

    # Custom port
    python dashboard.py --port 8080

    # Read-only (no MT5 queries, just files)
    python dashboard.py --offline
"""

import argparse, json, os, sys, time
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, render_template_string

# ─── OPTIONAL MT5 ─────────────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False

_BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BOT_DIR)

TRADE_HISTORY_FILE = os.path.join(_BOT_DIR, "trade_history.json")
TRADES_CSV         = os.path.join(_BOT_DIR, "trades.csv")
MODEL_STATE_FILE   = os.path.join(_BOT_DIR, "bot_data_smc", "model_state.pkl")
SYMBOL             = "XAUUSD"

app = Flask(__name__)

# ─── DATA HELPERS ─────────────────────────────────────────────────────────────

def _read_trade_history():
    try:
        if os.path.exists(TRADE_HISTORY_FILE):
            with open(TRADE_HISTORY_FILE) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def _read_trades_csv():
    try:
        import pandas as pd
        if os.path.exists(TRADES_CSV):
            df = pd.read_csv(TRADES_CSV, on_bad_lines="skip")
            return df.to_dict("records")
    except Exception:
        pass
    return []


def _model_age_hours():
    try:
        if os.path.exists(MODEL_STATE_FILE):
            mtime = os.path.getmtime(MODEL_STATE_FILE)
            age   = (time.time() - mtime) / 3600
            return round(age, 1)
    except Exception:
        pass
    return None


def _mt5_account():
    if not _MT5_AVAILABLE:
        return None
    try:
        if not mt5.initialize():
            return None
        acc = mt5.account_info()
        pos = mt5.positions_get(symbol=SYMBOL) or []
        mt5.shutdown()
        if not acc:
            return None
        return {
            "balance":    round(acc.balance, 2),
            "equity":     round(acc.equity, 2),
            "margin_free": round(acc.margin_free, 2),
            "positions":  [
                {
                    "ticket":     p.ticket,
                    "type":       "BUY" if p.type == 0 else "SELL",
                    "volume":     p.volume,
                    "open_price": p.price_open,
                    "current":    p.price_current,
                    "profit":     round(p.profit, 2),
                    "sl":         p.sl,
                    "tp":         p.tp,
                }
                for p in pos
            ],
        }
    except Exception:
        return None


def _build_equity_curve(trades):
    """Build equity curve from trade history using R-multiples."""
    curve = [{"x": 0, "y": 1000.0}]
    equity = 1000.0
    for i, t in enumerate(trades):
        profit = float(t.get("profit", 0) or 0)
        equity += profit
        curve.append({"x": i + 1, "y": round(equity, 2)})
    return curve


def _rolling_win_rate(trades, window=20):
    recent = [t for t in trades[-window:] if t.get("profit") is not None]
    if not recent:
        return None
    wins = sum(1 for t in recent if float(t.get("profit", 0)) > 0)
    return round(wins / len(recent) * 100, 1)


# ─── API ENDPOINTS ────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    trades  = _read_trade_history()
    account = _mt5_account()
    model_age = _model_age_hours()

    recent_20 = trades[-20:]
    wins  = sum(1 for t in recent_20 if float(t.get("profit", 0)) > 0)
    losses = sum(1 for t in recent_20 if float(t.get("profit", 0)) < 0)

    return jsonify({
        "total_trades":    len(trades),
        "recent_trades":   recent_20[::-1],   # newest first
        "win_rate_20":     _rolling_win_rate(trades),
        "equity_curve":    _build_equity_curve(trades),
        "account":         account,
        "model_age_hours": model_age,
        "model_fresh":     (model_age is not None and model_age < 5),
        "wins_last_20":    wins,
        "losses_last_20":  losses,
        "server_time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/trades")
def api_trades():
    return jsonify(_read_trade_history()[-100:])


# ─── HTML DASHBOARD ───────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="10">
<title>SMC-AI Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }
  header { background: #161b22; padding: 16px 24px; border-bottom: 1px solid #30363d;
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.2rem; font-weight: 600; }
  header .badge { background: #238636; color: #fff; font-size: 11px;
                  padding: 2px 8px; border-radius: 12px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 12px; padding: 16px 24px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 16px; }
  .card h3 { font-size: 11px; text-transform: uppercase; color: #8b949e;
             letter-spacing: 1px; margin-bottom: 8px; }
  .card .val { font-size: 2rem; font-weight: 700; }
  .card .sub { font-size: 12px; color: #8b949e; margin-top: 4px; }
  .green  { color: #3fb950; }
  .red    { color: #f85149; }
  .yellow { color: #d29922; }
  .chart-wrap { padding: 0 24px 16px; }
  .chart-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; background: #21262d;
       font-weight: 600; font-size: 11px; text-transform: uppercase;
       color: #8b949e; letter-spacing: 1px; }
  td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
  tr:last-child td { border: none; }
  .pos  { color: #3fb950; }
  .neg  { color: #f85149; }
  .neutral { color: #8b949e; }
  .section-title { padding: 8px 24px 4px; font-size: 13px; font-weight: 600;
                   text-transform: uppercase; color: #8b949e; letter-spacing: 1px; }
</style>
</head>
<body>
<header>
  <h1>🤖 SMC-AI-GOLD-SENTINEL</h1>
  <span class="badge" id="live-badge">LIVE</span>
  <span style="margin-left:auto;font-size:12px;color:#8b949e" id="server-time"></span>
</header>

<div class="grid" id="stats-grid">
  <div class="card">
    <h3>Balance</h3>
    <div class="val" id="balance">—</div>
    <div class="sub">Account balance</div>
  </div>
  <div class="card">
    <h3>Equity</h3>
    <div class="val" id="equity">—</div>
    <div class="sub">Floating P&L included</div>
  </div>
  <div class="card">
    <h3>Win Rate (L20)</h3>
    <div class="val" id="win-rate">—</div>
    <div class="sub">Last 20 closed trades</div>
  </div>
  <div class="card">
    <h3>Total Trades</h3>
    <div class="val" id="total-trades">—</div>
    <div class="sub">All recorded</div>
  </div>
  <div class="card">
    <h3>Open Positions</h3>
    <div class="val" id="open-pos">—</div>
    <div class="sub">Live MT5 positions</div>
  </div>
  <div class="card">
    <h3>Model Age</h3>
    <div class="val" id="model-age">—</div>
    <div class="sub">Hours since last retrain</div>
  </div>
</div>

<div class="chart-wrap">
  <div class="chart-card">
    <h3 style="font-size:11px;text-transform:uppercase;color:#8b949e;
               letter-spacing:1px;margin-bottom:12px;">Equity Curve</h3>
    <canvas id="equity-chart" height="80"></canvas>
  </div>
</div>

<div class="section-title">Open Positions</div>
<div style="padding:0 24px 16px">
  <div class="chart-card">
    <table id="pos-table">
      <thead><tr>
        <th>Ticket</th><th>Dir</th><th>Lots</th>
        <th>Open</th><th>Current</th><th>P&amp;L</th><th>SL</th><th>TP</th>
      </tr></thead>
      <tbody id="pos-body"><tr><td colspan="8" class="neutral">No open positions</td></tr></tbody>
    </table>
  </div>
</div>

<div class="section-title">Recent Trades</div>
<div style="padding:0 24px 24px">
  <div class="chart-card">
    <table id="trades-table">
      <thead><tr>
        <th>Time</th><th>Dir</th><th>Profit</th><th>Reason</th>
      </tr></thead>
      <tbody id="trades-body"><tr><td colspan="4" class="neutral">No trades yet</td></tr></tbody>
    </table>
  </div>
</div>

<script>
let chart = null;

async function refresh() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();

    document.getElementById('server-time').textContent = d.server_time;
    document.getElementById('total-trades').textContent = d.total_trades;

    const wr = d.win_rate_20;
    const wrEl = document.getElementById('win-rate');
    wrEl.textContent = wr !== null ? wr + '%' : '—';
    wrEl.className = 'val ' + (wr >= 55 ? 'green' : wr >= 40 ? 'yellow' : 'red');

    const acc = d.account;
    if (acc) {
      document.getElementById('balance').textContent = '$' + acc.balance.toLocaleString();
      const eqEl = document.getElementById('equity');
      const eqDiff = acc.equity - acc.balance;
      eqEl.textContent = '$' + acc.equity.toLocaleString();
      eqEl.className = 'val ' + (eqDiff >= 0 ? 'green' : 'red');
      document.getElementById('open-pos').textContent = acc.positions.length;

      // Positions table
      const pb = document.getElementById('pos-body');
      if (acc.positions.length === 0) {
        pb.innerHTML = '<tr><td colspan="8" class="neutral">No open positions</td></tr>';
      } else {
        pb.innerHTML = acc.positions.map(p => `
          <tr>
            <td>${p.ticket}</td>
            <td class="${p.type==='BUY'?'green':'red'}">${p.type}</td>
            <td>${p.volume}</td>
            <td>${p.open_price}</td>
            <td>${p.current}</td>
            <td class="${p.profit>=0?'pos':'neg'}">${p.profit>=0?'+':''}${p.profit}</td>
            <td>${p.sl||'—'}</td>
            <td>${p.tp||'—'}</td>
          </tr>`).join('');
      }
    } else {
      document.getElementById('balance').textContent = 'N/A';
      document.getElementById('equity').textContent  = 'N/A';
      document.getElementById('open-pos').textContent = '—';
    }

    const ma = d.model_age_hours;
    const maEl = document.getElementById('model-age');
    maEl.textContent = ma !== null ? ma + 'h' : '—';
    maEl.className   = 'val ' + (ma !== null && ma < 5 ? 'green' : 'yellow');

    // Equity curve
    const labels = d.equity_curve.map(p => p.x);
    const values = d.equity_curve.map(p => p.y);
    const ctx    = document.getElementById('equity-chart').getContext('2d');
    if (!chart) {
      chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [{
            data: values,
            borderColor: '#3fb950',
            backgroundColor: 'rgba(63,185,80,0.08)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: true,
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            x: { display: false },
            y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
          }
        }
      });
    } else {
      chart.data.labels = labels;
      chart.data.datasets[0].data = values;
      chart.update('none');
    }

    // Recent trades
    const tb = document.getElementById('trades-body');
    if (!d.recent_trades || d.recent_trades.length === 0) {
      tb.innerHTML = '<tr><td colspan="4" class="neutral">No trades yet</td></tr>';
    } else {
      tb.innerHTML = d.recent_trades.slice(0, 20).map(t => {
        const p   = parseFloat(t.profit || 0);
        const cls = p > 0 ? 'pos' : p < 0 ? 'neg' : 'neutral';
        const ts  = (t.time || '').replace('T', ' ').slice(0, 16);
        return `<tr>
          <td>${ts}</td>
          <td>${t.direction || '—'}</td>
          <td class="${cls}">${p >= 0 ? '+' : ''}${p.toFixed(2)}</td>
          <td>${t.exit_reason || '—'}</td>
        </tr>`;
      }).join('');
    }
  } catch(e) {
    document.getElementById('live-badge').textContent = 'OFFLINE';
    document.getElementById('live-badge').style.background = '#6e7681';
  }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",    type=int, default=5000)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.offline:
        global _MT5_AVAILABLE
        _MT5_AVAILABLE = False

    print(f"\n  SMC-AI Dashboard → http://localhost:{args.port}")
    print("  Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
