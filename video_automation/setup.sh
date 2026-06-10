#!/usr/bin/env bash
set -e

echo "================================================="
echo "  YouTube & TikTok Video Automation — Setup"
echo "================================================="

# ── Detect OS / shell environment ─────────────────────────────────────────────
IS_WINDOWS=false
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    IS_WINDOWS=true
fi

# Use 'python' on Windows (where python3 may not exist), else python3
if command -v python3 &> /dev/null; then
    PY=python3
elif command -v python &> /dev/null; then
    PY=python
else
    echo "[ERROR] Python not found. Install Python 3.10+ from https://python.org"
    exit 1
fi

python_version=$($PY --version 2>&1)
echo "Python: $python_version"

# ── [1/5] Virtual environment ──────────────────────────────────────────────────
echo ""
echo "[1/5] Setting up virtual environment …"

if [ ! -d ".venv" ]; then
    $PY -m venv .venv
    echo "      Created .venv"
else
    echo "      .venv already exists."
fi

# Activate — Windows (Git Bash / MINGW) uses Scripts/, Unix uses bin/
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "[ERROR] Could not find activate script in .venv."
    echo "        Try deleting .venv and running setup.sh again."
    exit 1
fi

echo "      Virtual environment active."

# ── [2/5] Python dependencies ──────────────────────────────────────────────────
echo ""
echo "[2/5] Installing Python dependencies …"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "      Dependencies installed."

# ── [3/5] ffmpeg ──────────────────────────────────────────────────────────────
echo ""
echo "[3/5] Checking ffmpeg …"
if command -v ffmpeg &> /dev/null; then
    echo "      ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "      ffmpeg not found."
    if $IS_WINDOWS; then
        echo ""
        echo "      *** Windows: install ffmpeg manually ***"
        echo "      1. Download from: https://www.gyan.dev/ffmpeg/builds/"
        echo "         (grab ffmpeg-release-essentials.zip)"
        echo "      2. Extract and add the 'bin' folder to your PATH"
        echo "      3. Restart Git Bash and re-run setup.sh"
        echo ""
        echo "      OR install via winget (run in PowerShell as Admin):"
        echo "        winget install Gyan.FFmpeg"
        echo ""
    elif command -v apt-get &> /dev/null; then
        echo "      Installing via apt-get …"
        sudo apt-get install -y ffmpeg -q
    elif command -v brew &> /dev/null; then
        echo "      Installing via Homebrew …"
        brew install ffmpeg
    elif command -v choco &> /dev/null; then
        echo "      Installing via Chocolatey …"
        choco install ffmpeg -y
    else
        echo "      [WARNING] Cannot auto-install ffmpeg."
        echo "      Download from: https://ffmpeg.org/download.html"
    fi
fi

# ── [4/5] Font ────────────────────────────────────────────────────────────────
echo ""
echo "[4/5] Downloading Roboto font …"
FONT_DIR="assets/fonts"
FONT_FILE="$FONT_DIR/Roboto-Bold.ttf"
mkdir -p "$FONT_DIR"
if [ ! -f "$FONT_FILE" ]; then
    FONT_URL="https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-Bold.ttf"
    if curl -fsSL "$FONT_URL" -o "$FONT_FILE" 2>/dev/null; then
        echo "      Font downloaded: $FONT_FILE"
    else
        echo "      [WARNING] Could not download font — videos will use system default."
    fi
else
    echo "      Font already present."
fi

# ── [5/5] Config ──────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Configuration …"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "      Created .env from .env.example"
    echo "      *** IMPORTANT: Edit .env and add your API keys! ***"
else
    echo "      .env already exists."
fi

mkdir -p output credentials

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Edit .env — add your ANTHROPIC_API_KEY (required)"
echo "  2. Add YouTube / TikTok keys when ready (optional)"
echo ""
echo "  ── Activate environment first ─────────────────"
if $IS_WINDOWS; then
echo "  source .venv/Scripts/activate"
else
echo "  source .venv/bin/activate"
fi
echo ""
echo "  ── Launch GUI (recommended) ───────────────────"
echo "  python gui.py"
echo "  Opens browser at: http://localhost:5000"
echo ""
echo "  ── Or use CLI ─────────────────────────────────"
echo "  python main.py niches"
echo "  python main.py generate --niche tech"
echo "  python main.py run"
echo "================================================="
