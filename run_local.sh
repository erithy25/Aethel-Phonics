#!/usr/bin/env bash
# ============================================================================
# Aethel V1 — Lokales Setup & Start
# ============================================================================
# Dieses Skript richtet alles ein und startet das vollständige
# Command & Control Dashboard auf deinem localhost.
#
# Nutzung:
#   chmod +x run_local.sh
#   ./run_local.sh
#
# Voraussetzung: Python 3.10+ muss installiert sein.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ⬡ Aethel V1 — Command & Control Dashboard                ║"
echo "║    APLD-MPS Multiphysics Simulation Suite                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# --------------------------------------------------------------------------
# 1. Python-Version prüfen
# --------------------------------------------------------------------------
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "FEHLER: Python 3.10+ wird benötigt. Bitte installiere Python."
    echo "  macOS:   brew install python@3.12"
    echo "  Ubuntu:  sudo apt install python3.12 python3.12-venv"
    echo "  Windows: https://python.org/downloads"
    exit 1
fi

echo "[1/4] Python gefunden: $PYTHON ($version)"

# --------------------------------------------------------------------------
# 2. Virtual Environment erstellen (falls nicht vorhanden)
# --------------------------------------------------------------------------
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/4] Erstelle Virtual Environment..."
    $PYTHON -m venv "$VENV_DIR"
else
    echo "[2/4] Virtual Environment existiert bereits."
fi

# Aktivieren
source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null

# --------------------------------------------------------------------------
# 3. Abhängigkeiten installieren
# --------------------------------------------------------------------------
echo "[3/4] Installiere Abhängigkeiten..."
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev,dashboard]"

# --------------------------------------------------------------------------
# 4. Dashboard starten
# --------------------------------------------------------------------------
echo "[4/4] Starte Dashboard..."
echo ""
echo "  ┌─────────────────────────────────────────────────┐"
echo "  │  Dashboard öffnet sich automatisch im Browser   │"
echo "  │  Falls nicht: http://localhost:8501              │"
echo "  │                                                 │"
echo "  │  Beenden mit: Ctrl+C                            │"
echo "  └─────────────────────────────────────────────────┘"
echo ""

streamlit run src/apld_mps/module_h/streamlit_app.py \
    --server.headless=false \
    --browser.gatherUsageStats=false
