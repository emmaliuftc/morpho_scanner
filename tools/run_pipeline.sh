#!/bin/bash
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <raw_images_dir> <golden_calib_json> [--steps <iterations>]"
    echo "Example: $0 captures_8-13_four_lobes captures_0726_clay_checkboard_64_calibrated/calibration_results.json --steps 2000"
    exit 1
fi

RAW_DIR=$1
GOLDEN_CALIB=$2
STEPS=2000

if [ "$3" == "--steps" ] && [ -n "$4" ]; then
    STEPS=$4
fi

# Clean up raw dir string to use as a name (e.g. captures_8-13_four_lobes -> 8-13_four_lobes)
DATASET_NAME=$(basename "$RAW_DIR" | sed 's/captures_//')
TIMESTAMP=$(date +"%d%m%y_%H%M")
SESSION_DIR="nerf_${TIMESTAMP}_${DATASET_NAME}"
PROGRESS_FILE="${SESSION_DIR}/progress.md"
PYTHON_ENV=".venv_nerf/bin/python"

echo "========================================="
echo "Starting Generalized NeRF Pipeline"
echo "Dataset: $RAW_DIR"
echo "Calibration: $GOLDEN_CALIB"
echo "Session Dir: $SESSION_DIR"
echo "Steps: $STEPS"
echo "========================================="

mkdir -p "$SESSION_DIR"
echo "# Pipeline Progress - $DATASET_NAME ($TIMESTAMP)" > "$PROGRESS_FILE"

# Step 1 & 2: Prepare Images and Masks (Downscale to 4)
echo "[Step 1 & 2] Generating Downscaled Images and Silhouette Masks..."
$PYTHON_ENV tools/prepare_images_and_masks.py \
    --calib "$GOLDEN_CALIB" \
    --img_dir "$RAW_DIR" \
    --out_dir "$SESSION_DIR" \
    --scale 4
echo "- [x] Step 1 & 2: Generated downscaled images and masks" >> "$PROGRESS_FILE"

# Step 3: Apply Golden Calibration (calculate starting angle from QR code and inherit table matrix)
echo "[Step 3] Applying Golden Calibration with QR starting angle..."
$PYTHON_ENV tools/apply_golden_calibration.py \
    --calib "$GOLDEN_CALIB" \
    --frame0 "$RAW_DIR/capture_0.jpg" \
    --output_unaligned "$SESSION_DIR/transforms_unaligned.json" \
    --output_aligned "$SESSION_DIR/transforms_aligned.json" \
    --scale 4.0
echo "- [x] Step 3: Generated transforms.json" >> "$PROGRESS_FILE"

# Step 4: Train NeRF Model
echo "[Step 4] Training NeRF Model ($STEPS steps)..."
cp "$SESSION_DIR/transforms_aligned.json" "$SESSION_DIR/transforms.json"

.venv_nerf/bin/ns-train nerfacto \
    --data "$SESSION_DIR" \
    --output-dir "${SESSION_DIR}/outputs" \
    --vis tensorboard \
    --max-num-iterations "$STEPS" \
    --pipeline.datamanager.train-num-rays-per-batch 8192 \
    --pipeline.model.disable-scene-contraction True \
    --pipeline.model.background-color white \
    --pipeline.model.proposal-initial-sampler uniform \
    --pipeline.model.near-plane 0.1 \
    --pipeline.model.far-plane 2.5 \
    --pipeline.model.camera-optimizer.mode off \
    --pipeline.model.use-average-appearance-embedding False \
    nerfstudio-data \
    --eval-mode all \
    --center-method none \
    --auto-scale-poses False \
    --orientation-method none

echo "- [x] Step 4: Trained NeRF Model" >> "$PROGRESS_FILE"

# Find the config file path dynamically
CONFIG_PATH=$(find "${SESSION_DIR}/outputs" -name "config.yml" | sort -r | head -n 1)

# Step 5: Extract raw point cloud
echo "[Step 5] Extracting Raw Point Cloud..."
.venv_nerf/bin/ns-export pointcloud \
    --load-config "$CONFIG_PATH" \
    --output-dir "${SESSION_DIR}/pointcloud_raw" \
    --num-points 1000000 \
    --remove-outliers True \
    --normal-method open3d \
    --obb-center 0.0 0.0 0.0 \
    --obb-scale 0.8 0.8 0.8 \
    --obb-rotation 0.0 0.0 0.0

mv "${SESSION_DIR}/pointcloud_raw/point_cloud.ply" "${SESSION_DIR}/pointcloud_raw.ply"
echo "- [x] Step 5: Extracted raw point cloud" >> "$PROGRESS_FILE"

# Step 6: Filter and Z-clip the point cloud
echo "[Step 6] Filtering Point Cloud (Z-Clip & SOR)..."
$PYTHON_ENV tools/nerf/filter_cloud.py \
    --input "${SESSION_DIR}/pointcloud_raw.ply" \
    --output "${SESSION_DIR}/pointcloud_filtered.ply"
echo "- [x] Step 6: Filtered point cloud" >> "$PROGRESS_FILE"

# Step 7: Generate Poisson Mesh
echo "[Step 7] Generating Poisson Mesh..."
$PYTHON_ENV tools/nerf/create_mesh.py \
    --input "${SESSION_DIR}/pointcloud_filtered.ply" \
    --output "${SESSION_DIR}/mesh.ply"
echo "- [x] Step 7: Generated Poisson mesh" >> "$PROGRESS_FILE"

echo "Pipeline completed successfully!"
echo "Final mesh saved to: ${SESSION_DIR}/mesh.ply"
echo "- [x] Pipeline fully completed!" >> "$PROGRESS_FILE"
