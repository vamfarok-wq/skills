#!/usr/bin/env bash
# ============================================================
#  TikTok Shop Ad Maker - one-click start for macOS / Linux
#  In Terminal:   bash start-mac-linux.sh
# ============================================================
cd "$(dirname "$0")" || exit 1

echo
echo "  Setting up (first time takes a few minutes)..."
echo

# Find Python 3
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "  [!] Python 3 is not installed."
  echo "      macOS:  install from https://www.python.org/downloads/  (or: brew install python)"
  echo "      Linux:  sudo apt install python3 python3-venv python3-pip"
  exit 1
fi

# Create virtual environment on first run
if [ ! -x ".venv/bin/python" ]; then
  echo "  Creating environment..."
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
echo "  Installing components (only the first time)..."
python -m pip install -r requirements.txt

echo
echo "  Starting the app... your browser will open automatically."
echo "  (Keep this Terminal window open while you use the app. Press CTRL+C to stop.)"
echo
python app.py
