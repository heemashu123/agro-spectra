#!/usr/bin/env bash
# =============================================================================
# Agro-Spectra | One-Click Linux / macOS Launcher
# =============================================================================
set -e

echo ""
echo " ============================================================"
echo "  Agro-Spectra | Precision Agriculture RL Ecosystem"
echo "  Meta PyTorch & OpenEnv Hackathon Submission"
echo " ============================================================"
echo ""

# ---- Find Python -------------------------------------------------------------
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python not found. Please install Python 3.9+."
    echo "        Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "        macOS:         brew install python"
    exit 1
fi

echo "[INFO] Using Python: $($PYTHON_CMD --version)"
echo ""

# ---- Install dependencies ----------------------------------------------------
echo "[STEP 1/3] Installing dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt --quiet
echo "[OK] Dependencies ready."
echo ""

# ---- Generate dataset --------------------------------------------------------
echo "[STEP 2/3] Generating synthetic farm dataset..."
$PYTHON_CMD data_generator.py
echo ""

# ---- Train agent -------------------------------------------------------------
echo "[STEP 3/3] Training PPO agent (50,000 timesteps)..."
echo "           This will take approx. 5-10 minutes on CPU."
echo ""
$PYTHON_CMD train_agent.py

echo ""
echo " ============================================================"
echo "  Training complete! Files generated:"
echo "    agro_ppo_model.zip     - Final trained model"
echo "    best_agro_model/       - Best checkpoint during training"
echo "    mock_farm_data.csv     - Synthetic farm dataset"
echo " ============================================================"
echo ""
