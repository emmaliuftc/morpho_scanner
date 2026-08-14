#!/bin/bash
# tools/nerf/train_lego.sh
set -e

DATA_DIR="data/blender/lego"

echo "====================================="
echo " 🧱 Downloading Blender Lego Dataset "
echo "====================================="
# Download via Hugging Face mirror to avoid Google Drive quota errors
.venv_nerf/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='pablovela5620/nerf-synthetic-mirror', repo_type='dataset', allow_patterns='lego/*', local_dir='data/blender')
"

echo "====================================="
echo " 🚀 Starting/Resuming NeRF Training  "
echo "====================================="

LATEST_MODEL_DIR=$(ls -td outputs/*/nerfacto/*/nerfstudio_models 2>/dev/null | head -n 1 || true)
LOAD_ARG=""
if [ -n "$LATEST_MODEL_DIR" ]; then
    echo "Found previous checkpoint: $LATEST_MODEL_DIR. Resuming!"
    LOAD_ARG="--load-dir $LATEST_MODEL_DIR"
fi

# Limit iterations to 5000 so it trains fast
# For nerfstudio CLI, method arguments come before the dataparser arguments!
.venv_nerf/bin/ns-train nerfacto \
    --machine.num-devices 1 \
    --max-num-iterations 5000 \
    --vis tensorboard \
    $LOAD_ARG \
    blender-data \
    --data ${DATA_DIR}

echo "====================================="
echo " ☁️ Exporting Point Cloud (.ply)      "
echo "====================================="
CONFIG_FILE=$(ls -t outputs/*/nerfacto/*/config.yml | head -n 1)

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Could not find config.yml. Training may have failed."
    exit 1
fi

echo "Found config file: $CONFIG_FILE"

mkdir -p tools/nerf/output_lego
.venv_nerf/bin/ns-export pointcloud \
    --load-config "$CONFIG_FILE" \
    --output-dir tools/nerf/output_lego \
    --num-points 1000000 \
    --remove-outliers True \
    --obb-center 0.0 0.0 0.0 \
    --obb-scale 3.0 3.0 3.0 \
    --obb-rotation 0.0 0.0 0.0

echo "====================================="
echo " ✅ Done! PLY saved in tools/nerf/output_lego/ "
echo "====================================="
