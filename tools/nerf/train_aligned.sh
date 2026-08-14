#!/bin/bash
set -e
echo "Training Aligned NeRF Model"
.venv_nerf/bin/ns-train nerfacto \
    --max-num-iterations 2000 \
    --data data/three_flat_aligned \
    --output-dir outputs_three_flat_aligned \
    --vis tensorboard \
    nerfstudio-data \
    --downscale-factor 1 \
    --orientation-method none \
    --center-method none \
    --auto-scale-poses False

CONFIG_PATH=$(find outputs_three_flat_aligned/three_flat_aligned/nerfacto -name "config.yml" | sort | tail -n 1)
.venv_nerf/bin/ns-export pointcloud \
    --load-config "$CONFIG_PATH" \
    --output-dir exports/three_flat_aligned_pc \
    --num-points 1000000 \
    --remove-outliers True \
    --normal-method open3d \
    --obb-center 0.0 0.0 0.0 \
    --obb-scale 1.0 1.0 1.0 \
    --obb-rotation 0.0 0.0 0.0

.venv_nerf/bin/python tools/nerf/filter_cloud.py \
    --input exports/three_flat_aligned_pc/point_cloud.ply \
    --output exports/three_flat_aligned_pc/point_cloud_filtered.ply

.venv_nerf/bin/python tools/nerf/create_mesh.py \
    --input exports/three_flat_aligned_pc/point_cloud_filtered.ply \
    --output exports/three_flat_aligned_pc/mesh.ply
