#!/bin/bash
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

set -e

# =============================================================================
# HRC Pipeline - Server Deployment Script
#
# This script sets up the HRC (Human-Robot Collaboration) pipeline on a new
# server. It handles:
#   1. Cloning and installing OmniGibson (the BEHAVIOR-1K simulator)
#   2. Applying the HDF5 file-lock patch to OmniGibson
#   3. Installing Python dependencies (bddl3, etc.)
#   4. Downloading BEHAVIOR-1K datasets (assets, robot models, task instances)
#   5. Configuring paths so HRC scripts can find everything
#
# Prerequisites:
#   - Ubuntu 20.04+ with NVIDIA GPU (RTX 3090+ recommended)
#   - CUDA 12.x installed
#   - conda (Miniconda or Anaconda)
#   - The HDF5 rawdata must be transferred separately (see README.md)
#
# Usage:
#   bash setup_hrc.sh [--skip-omnigibson] [--skip-datasets] [--rawdata-dir DIR]
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_OMNIGIBSON=false
SKIP_DATASETS=false
RAWDATA_DIR=""
CONDA_ENV_NAME="behavior"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-omnigibson) SKIP_OMNIGIBSON=true; shift ;;
        --skip-datasets) SKIP_DATASETS=true; shift ;;
        --rawdata-dir) RAWDATA_DIR="$2"; shift 2 ;;
        --env-name) CONDA_ENV_NAME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  HRC Pipeline Setup"
echo "============================================"
echo "  Project dir: $SCRIPT_DIR"
echo "  Conda env:   $CONDA_ENV_NAME"
echo ""

# ---- Step 1: Clone and install OmniGibson ----
if [ "$SKIP_OMNIGIBSON" = false ]; then
    echo "[1/5] Setting up OmniGibson..."

    if [ -d "$SCRIPT_DIR/OmniGibson/.git" ]; then
        echo "  OmniGibson already cloned, pulling latest..."
        git -C "$SCRIPT_DIR/OmniGibson" pull || true
    elif [ -d "$SCRIPT_DIR/OmniGibson" ] && [ ! -d "$SCRIPT_DIR/OmniGibson/.git" ]; then
        echo "  OmniGibson directory exists but is not a git repo."
        echo "  Backing up and re-cloning..."
        mv "$SCRIPT_DIR/OmniGibson" "$SCRIPT_DIR/OmniGibson.bak.$(date +%s)"
        git clone https://github.com/StanfordVL/OmniGibson.git "$SCRIPT_DIR/OmniGibson"
    else
        echo "  Cloning OmniGibson..."
        git clone https://github.com/StanfordVL/OmniGibson.git "$SCRIPT_DIR/OmniGibson"
    fi

    echo "  Installing OmniGibson into conda env '$CONDA_ENV_NAME'..."
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || {
        echo "  Creating conda env '$CONDA_ENV_NAME' (Python 3.10)..."
        conda create -n "$CONDA_ENV_NAME" python=3.10 -y
        conda activate "$CONDA_ENV_NAME"
    }
    pip install -e "$SCRIPT_DIR/OmniGibson" || {
        echo "  WARN: pip install -e OmniGibson failed. You may need Isaac Sim first."
        echo "  See: https://docs.omniverse.nvidia.com/isaacsim/latest/installation/"
    }
else
    echo "[1/5] Skipping OmniGibson setup (--skip-omnigibson)"
fi

# ---- Step 2: Apply the HDF5 file-lock patch ----
echo ""
echo "[2/5] Applying OmniGibson patch (HDF5 r -> r+)..."
PATCH_FILE="$SCRIPT_DIR/patches/data_wrapper.patch"
TARGET_FILE="$SCRIPT_DIR/OmniGibson/omnigibson/envs/data_wrapper.py"

if [ -f "$PATCH_FILE" ] && [ -f "$TARGET_FILE" ]; then
    if grep -q 'h5py.File(input_path, "r+")' "$TARGET_FILE"; then
        echo "  Patch already applied."
    else
        git -C "$SCRIPT_DIR" apply "$PATCH_FILE" && echo "  Patch applied successfully." || {
            echo "  git apply failed, trying manual patch..."
            sed -i 's/h5py\.File(input_path, "r")/h5py.File(input_path, "r+")/' "$TARGET_FILE"
            echo "  Manual sed patch applied."
        }
    fi
else
    echo "  WARN: patch file or target not found. Skipping."
    echo "  Expected: $PATCH_FILE"
    echo "  Target:   $TARGET_FILE"
fi

# ---- Step 3: Install Python dependencies ----
echo ""
echo "[3/5] Installing Python dependencies..."
pip install h5py imageio imageio-ffmpeg pyyaml numpy torch 2>/dev/null || true

if [ -d "$SCRIPT_DIR/bddl3" ]; then
    echo "  Installing bddl3..."
    pip install -e "$SCRIPT_DIR/bddl3" 2>/dev/null || true
fi

# ---- Step 4: Download BEHAVIOR-1K datasets ----
if [ "$SKIP_DATASETS" = false ]; then
    echo ""
    echo "[4/5] Downloading BEHAVIOR-1K datasets..."
    echo "  This will download: behavior-1k-assets, robot-assets, task-instances"
    echo "  (Total ~5-10 GB, may take a while)"
    echo ""

    export OMNI_KIT_ACCEPT_EULA=YES

    python -c "
import omnigibson
from omnigibson.utils.asset_utils import (
    download_omnigibson_robot_assets,
    download_behavior_1k_assets,
    download_2025_challenge_task_instances,
)
print('Downloading robot assets...')
download_omnigibson_robot_assets()
print('Downloading BEHAVIOR-1K assets...')
download_behavior_1k_assets(accept_license=True)
print('Downloading 2025 challenge task instances...')
download_2025_challenge_task_instances()
print('All datasets downloaded.')
" || {
        echo "  WARN: Dataset download failed. You can retry later with:"
        echo "    python -c 'from omnigibson.utils.asset_utils import ...; ...'"
        echo "  Or use:  bash setup.sh --dataset --accept-dataset-tos"
    }
else
    echo "[4/5] Skipping dataset download (--skip-datasets)"
fi

# ---- Step 5: Configure paths ----
echo ""
echo "[5/5] Configuring paths..."

DATA_PATH=$(python -c "from omnigibson.macros import gm; print(gm.DATA_PATH)" 2>/dev/null || echo "$SCRIPT_DIR/datasets")
RAWDATA_TARGET="$DATA_PATH/2025-challenge-rawdata"

if [ -n "$RAWDATA_DIR" ] && [ -d "$RAWDATA_DIR" ]; then
    if [ -d "$RAWDATA_TARGET" ]; then
        echo "  rawdata already exists at $RAWDATA_TARGET"
    else
        echo "  Symlinking rawdata: $RAWDATA_DIR -> $RAWDATA_TARGET"
        ln -s "$RAWDATA_DIR" "$RAWDATA_TARGET"
    fi
elif [ -d "$RAWDATA_TARGET" ]; then
    echo "  rawdata found at $RAWDATA_TARGET"
else
    echo ""
    echo "  ============================================================"
    echo "  WARNING: HDF5 rawdata not found!"
    echo "  ============================================================"
    echo "  The HRC pipeline needs BEHAVIOR-1K rawdata HDF5 files."
    echo "  These are NOT auto-downloadable. You need to:"
    echo ""
    echo "  Option A - rsync from source machine:"
    echo "    rsync -avP user@source:/path/to/2025-challenge-rawdata/ \\"
    echo "      $RAWDATA_TARGET/"
    echo ""
    echo "  Option B - symlink if data is elsewhere on this machine:"
    echo "    ln -s /path/to/2025-challenge-rawdata $RAWDATA_TARGET"
    echo ""
    echo "  Required task directories:"
    echo "    task-0000/  (1 HDF5, ~4MB)   - turning_on_radio"
    echo "    task-0026/  (200 HDF5, ~62GB) - assembling_gift_baskets"
    echo "  ============================================================"
fi

# Update YAML config to use relative paths
HUMAN_USD="$SCRIPT_DIR/assets/human_b962bca9544d/character.usda"
if [ -f "$HUMAN_USD" ]; then
    echo "  Human USD asset found: $HUMAN_USD"
else
    echo "  WARN: Human USD asset not found at $HUMAN_USD"
    echo "  Run: python scripts/convert_all_habitat_humans.py"
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  To run the HRC demo:"
echo "    conda activate $CONDA_ENV_NAME"
echo "    bash cmd.sh"
echo ""
echo "  To verify the setup:"
echo "    python -c 'import omnigibson; print(omnigibson.__version__)'"
echo ""
echo "  If rawdata is missing, see the WARNING above."
echo "============================================"
