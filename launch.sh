#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch.sh  —  one-click launcher for Shakespeare GPT browser app
# Just run:  bash launch.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"
PYTHON="$VENV/bin/python"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🎭  Shakespeare GPT — Browser App"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Create venv if missing ─────────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv "$VENV"
fi

# ── 2. Install / upgrade dependencies ────────────────────────────────────────
echo "Checking dependencies..."
"$PYTHON" -m pip install -q --upgrade pip
"$PYTHON" -m pip install -q torch requests tensorboard gradio "setuptools<70" matplotlib

echo "All dependencies ready."
echo ""

# ── 3. Launch app (opens browser automatically) ───────────────────────────────
echo "Starting app → http://127.0.0.1:7860"
echo "(Press Ctrl-C to stop)"
echo ""
cd "$DIR"
"$PYTHON" app.py
