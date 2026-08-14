#!/bin/bash
set -e

echo "========================================="
echo "1. Training NeRF Model on 0726 Dataset"
echo "========================================="
# Disabling auto-scaling to preserve the exact /150 scale coordinates in transforms.json
.venv_nerf/bin/ns-train nerfacto \
    --max-num-iterations 2000 \
    --data captures_0726_nerf_dataset \
    --output-dir outputs_0726 \
    --vis tensorboard \
    nerfstudio-data \
    --downscale-factor 1 \
    --orientation-method none \
    --center-method none \
    --auto-scale-poses False

echo ""
echo "========================================="
echo "2. Exporting Cropped Point Cloud"
echo "========================================="
CONFIG_PATH=$(find outputs_0726/captures_0726_nerf_dataset/nerfacto -name "config.yml" | sort | tail -n 1)

if [ -z "$CONFIG_PATH" ]; then
    echo "Error: Could not find config.yml. Did training fail?"
    exit 1
fi

echo "Using config: $CONFIG_PATH"

# Bounding box scale: The 0726 coordinates were manually divided by 150.0
# So an obb-scale of 1.0 NeRF units equals 150mm physical size.
# Using 1.5 gives us a comfortable 225mm crop box around the 140mm clay.
.venv_nerf/bin/ns-export pointcloud \
    --load-config "$CONFIG_PATH" \
    --output-dir exports/0726_pc_cropped \
    --num-points 1000000 \
    --remove-outliers True \
    --normal-method open3d \
    --obb-center 0.0 0.0 0.0 \
    --obb-scale 1.5 1.5 1.5 \
    --obb-rotation 0.0 0.0 0.0

echo ""
echo "========================================="
echo "3. Filtering Point Cloud (SOR)"
echo "========================================="
.venv_nerf/bin/python tools/nerf/filter_cloud.py \
    --input exports/0726_pc_cropped/point_cloud.ply \
    --output exports/0726_pc_cropped/point_cloud_filtered.ply

echo "========================================="
echo "Pipeline complete! Saved to exports/0726_pc_cropped/point_cloud_filtered.ply"
echo "========================================="
