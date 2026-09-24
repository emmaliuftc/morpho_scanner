#!/bin/bash
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <raw_images_dir> <golden_calib_json> [--steps <iterations>] [--stop-after <stage_number>]"
    echo "Example: $0 captures_8-13_four_lobes captures_0726_clay_checkboard_64_calibrated/calibration_results.json --steps 2000 --stop-after 3"
    exit 1
fi

RAW_DIR=$1
GOLDEN_CALIB=$2
STEPS=2000
START_FROM=1
STOP_AFTER=10

if [ ! -d "$RAW_DIR" ] && [ -d "tests_and_images/$RAW_DIR" ]; then
    RAW_DIR="tests_and_images/$RAW_DIR"
fi
if [ ! -f "$GOLDEN_CALIB" ] && [ -f "tests_and_images/$GOLDEN_CALIB" ]; then
    GOLDEN_CALIB="tests_and_images/$GOLDEN_CALIB"
fi

shift 2
while [[ $# -gt 0 ]]; do
  case $1 in
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --start-from)
      START_FROM="$2"
      shift 2
      ;;
    --stop-after)
      STOP_AFTER="$2"
      shift 2
      ;;
    --session-dir)
      CUSTOM_SESSION_DIR="$2"
      shift 2
      ;;
    --mask-dir)
      MASK_DIR_ARG="--mask_dir $2"
      shift 2
      ;;
    --no-filter-green)
      MASK_OPTIONS="$MASK_OPTIONS --no_filter_green"
      shift 1
      ;;
    --filter-blue)
      MASK_OPTIONS="$MASK_OPTIONS --filter_blue"
      shift 1
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Clean up raw dir string to use as a name (e.g. captures_8-13_four_lobes -> 8-13_four_lobes)
DATASET_NAME=$(basename "$RAW_DIR" | sed 's/captures_//')
TIMESTAMP=$(date +"%d%m%y_%H%M")

if [ -n "$CUSTOM_SESSION_DIR" ]; then
    SESSION_DIR="$CUSTOM_SESSION_DIR"
elif [ -d "tests_and_images" ]; then
    SESSION_DIR="tests_and_images/nerf_${TIMESTAMP}_${DATASET_NAME}"
else
    SESSION_DIR="nerf_${TIMESTAMP}_${DATASET_NAME}"
fi
PROGRESS_FILE="${SESSION_DIR}/progress.md"

if [ -z "$PYTHON_ENV" ]; then
    if [ -f ".venv_nerf/bin/python" ]; then
        PYTHON_ENV=".venv_nerf/bin/python"
    elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
        PYTHON_ENV="$VIRTUAL_ENV/bin/python"
    elif [ -f "/home/coding/github/morpho_scanner/.venv_nerf/bin/python" ]; then
        PYTHON_ENV="/home/coding/github/morpho_scanner/.venv_nerf/bin/python"
    elif [ -f "../morpho_scanner/.venv_nerf/bin/python" ]; then
        PYTHON_ENV="../morpho_scanner/.venv_nerf/bin/python"
    else
        PYTHON_ENV="python3"
    fi
fi

NS_BIN_DIR="$(dirname "$PYTHON_ENV")"
NS_TRAIN="$NS_BIN_DIR/ns-train"
NS_EXPORT="$NS_BIN_DIR/ns-export"
if [ ! -x "$NS_TRAIN" ]; then
    NS_TRAIN="ns-train"
fi
if [ ! -x "$NS_EXPORT" ]; then
    NS_EXPORT="ns-export"
fi

echo "========================================="
echo "Starting Generalized NeRF Pipeline"
echo "Dataset: $RAW_DIR"
echo "Calibration: $GOLDEN_CALIB"
echo "Session Dir: $SESSION_DIR"
echo "Steps: $STEPS"
echo "Stop After Stage: $STOP_AFTER"
echo "========================================="

mkdir -p "$SESSION_DIR"
echo "# Pipeline Progress - $DATASET_NAME ($TIMESTAMP)" > "$PROGRESS_FILE"

if [ "$START_FROM" -le 2 ] && [ "$STOP_AFTER" -ge 2 ]; then
    # Step 1 & 2: Prepare Images and Masks (Downscale to 4)
    echo "[Step 1 & 2] Generating Downscaled Images and Silhouette Masks..."
    $PYTHON_ENV tools/prepare_images_and_masks.py \
        --calib "$GOLDEN_CALIB" \
        --img_dir "$RAW_DIR" \
        --out_dir "$SESSION_DIR" \
        --scale 4 $MASK_DIR_ARG $MASK_OPTIONS

    # If full-resolution images exist in images_1, generate native resolution masks
    if [ -d "${SESSION_DIR}/images_1" ] && [ ! -d "${SESSION_DIR}/masks_1" ]; then
        echo "Running foreground extraction on native resolution images..."
        $PYTHON_ENV tools/generate_masks.py --calib "$GOLDEN_CALIB" --img_dir "${SESSION_DIR}/images_1" --out_dir "${SESSION_DIR}/masks_1" $MASK_OPTIONS
    fi
    echo "- [x] Step 1 & 2: Generated downscaled images and masks" >> "$PROGRESS_FILE"
fi

if [ "$START_FROM" -le 3 ] && [ "$STOP_AFTER" -ge 3 ]; then
    # Step 3: Apply Golden Calibration (calculate starting angle from QR code and inherit table matrix)
    echo "[Step 3] Applying Golden Calibration with QR starting angle..."
    $PYTHON_ENV tools/apply_golden_calibration.py \
        --calib "$GOLDEN_CALIB" \
        --frame0 "$RAW_DIR/capture_0.jpg" \
        --output_unaligned "$SESSION_DIR/transforms_unaligned.json" \
        --output_aligned "$SESSION_DIR/transforms_aligned.json" \
        --scale 4.0
    echo "- [x] Step 3: Generated transforms.json" >> "$PROGRESS_FILE"

    echo "Generating Camera Pose Visualization..."
    $PYTHON_ENV tools/plot_transforms_3d.py \
        --json "$SESSION_DIR/transforms_aligned.json" \
        --out "$SESSION_DIR/camera_poses_3d.png" \
        --title "$DATASET_NAME Camera Poses"

    echo "Annotating preview images with Z-axis and Angles..."
    $PYTHON_ENV tools/annotate_images.py \
        --session_dir "$SESSION_DIR" \
        --calib "$GOLDEN_CALIB" \
        --scale 4
fi

if [ "$START_FROM" -le 4 ] && [ "$STOP_AFTER" -ge 4 ]; then
    # Step 4: Train NeRF Model
    echo "[Step 4] Training NeRF Model ($STEPS steps)..."
    $PYTHON_ENV -c "
import json
with open('$SESSION_DIR/transforms_aligned.json', 'r') as f:
    transforms = json.load(f)
for i, frame in enumerate(transforms['frames']):
    frame['file_path'] = f'masked_images_preview_4/preview_capture_{i}.png'
    if 'mask_path' in frame:
        del frame['mask_path']
with open('$SESSION_DIR/transforms.json', 'w') as f:
    json.dump(transforms, f, indent=4)
"

    "$NS_TRAIN" nerfacto \
        --data "$SESSION_DIR" \
        --output-dir "${SESSION_DIR}/outputs" \
        --vis tensorboard \
        --max-num-iterations "$STEPS" \
        --pipeline.datamanager.train-num-rays-per-batch 8192 \
        --pipeline.model.disable-scene-contraction True \
        --pipeline.model.background-color random \
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
fi

if [ "$START_FROM" -le 5 ] && [ "$STOP_AFTER" -ge 5 ]; then
    # Find the config file path dynamically
    CONFIG_PATH=$(find "${SESSION_DIR}/outputs" -name "config.yml" | sort -r | head -n 1)

    # Step 5: Extract raw point cloud
    echo "[Step 5] Extracting Raw Point Clouds (Full and OBB)..."
    
    echo " -> Extracting Full Scene..."
    "$NS_EXPORT" pointcloud \
        --load-config "$CONFIG_PATH" \
        --output-dir "${SESSION_DIR}/pointcloud_raw_full" \
        --num-points 1000000 \
        --remove-outliers True \
        --normal-method open3d

    mv "${SESSION_DIR}/pointcloud_raw_full/point_cloud.ply" "${SESSION_DIR}/pointcloud_raw_full.ply"

    echo " -> Extracting Cropped (OBB)..."
    "$NS_EXPORT" pointcloud \
        --load-config "$CONFIG_PATH" \
        --output-dir "${SESSION_DIR}/pointcloud_raw_obb" \
        --num-points 1000000 \
        --remove-outliers True \
        --normal-method open3d \
        --obb-center 0.0 0.0 0.0 \
        --obb-scale 0.8 0.8 0.8 \
        --obb-rotation 0.0 0.0 0.0

    mv "${SESSION_DIR}/pointcloud_raw_obb/point_cloud.ply" "${SESSION_DIR}/pointcloud_raw_obb.ply"
    
    echo "- [x] Step 5: Extracted raw point clouds (Full and OBB)" >> "$PROGRESS_FILE"
fi

if [ "$START_FROM" -le 6 ] && [ "$STOP_AFTER" -ge 6 ]; then
    # Step 6: Generate Solid Volumes and Export Bio-Formats
    echo "[Step 6] Generating Solid 2.5D Volumes and Exporting Napari NPY..."
    
    for ply in "${SESSION_DIR}/pointcloud_raw_full.ply" "${SESSION_DIR}/pointcloud_raw_obb.ply"; do
        if [ -f "$ply" ]; then
            solid_ply="${ply%.ply}_solid_table.ply"
            reprojected_ply="${ply%.ply}_solid_table_reprojected.ply"
            echo " -> Extruding solid table volume: $solid_ply"
            $PYTHON_ENV tools/fill_25d_extrusion.py --input "$ply" --output "$solid_ply" --pitch 0.002
            
            echo " -> Creating legacy _reprojected copy..."
            cp "$solid_ply" "$reprojected_ply"
            
            echo " -> Exporting Napari NPY..."
            $PYTHON_ENV tools/export_bio_format.py --ply "$solid_ply"
        fi
    done
    echo "- [x] Step 6: Generated Solid Volumes & Bio-Formats" >> "$PROGRESS_FILE"
fi

if [ "$START_FROM" -le 7 ] && [ "$STOP_AFTER" -ge 7 ]; then
    # Step 7: Generate Orbiting GIFs
    echo "[Step 7] Generating Orbiting GIFs for Point Clouds..."
    
    for ply in "${SESSION_DIR}/pointcloud_raw_full.ply" "${SESSION_DIR}/pointcloud_raw_obb.ply" "${SESSION_DIR}/pointcloud_raw_full_solid_table_reprojected.ply" "${SESSION_DIR}/pointcloud_raw_obb_solid_table_reprojected.ply"; do
        if [ -f "$ply" ]; then
            out_gif="${ply%.ply}.gif"
            echo " -> Rendering Point Cloud Video: $ply"
            $PYTHON_ENV tools/render_pointcloud_video.py --ply "$ply" --out "$out_gif"
        fi
    done
    
    echo "- [x] Step 7: Generated Orbiting GIFs" >> "$PROGRESS_FILE"
fi

if [ "$START_FROM" -le 8 ] && [ "$STOP_AFTER" -ge 8 ]; then
    # Step 8: Generate side-by-side projections
    echo "[Step 8] Generating side-by-side projections for pointcloud_raw_full.ply..."
    
    if [ -f "${SESSION_DIR}/pointcloud_raw_full.ply" ]; then
        $PYTHON_ENV tools/project_all_masks.py --dir "$SESSION_DIR" --ply "pointcloud_raw_full.ply"
    fi
    echo "- [x] Step 8: Generated side-by-side projections" >> "$PROGRESS_FILE"
fi

if [ "$START_FROM" -le 9 ] && [ "$STOP_AFTER" -ge 9 ]; then
    # Step 9: 17-Feature Morphometrics & Haralick GLCM Texture Analysis
    echo "[Step 9] Extracting 17-Feature Morphometric & 3D Haralick Texture Profile..."
    HARALICK_DIR="${SESSION_DIR}/haralick_morphometry"
    mkdir -p "$HARALICK_DIR"
    
    for npy in "${SESSION_DIR}/pointcloud_raw_obb_solid_table_volume.npy" "${SESSION_DIR}/pointcloud_raw_full_solid_table_volume.npy"; do
        if [ -f "$npy" ]; then
            echo " -> Computing Haralick & morphometrics for: $npy"
            $PYTHON_ENV tools/extract_haralick_features.py --npy "$npy" --out_dir "$HARALICK_DIR"
        fi
    done
    echo "- [x] Step 9: Extracted 17-Feature Morphometrics & Haralick Texture" >> "$PROGRESS_FILE"
fi

if [ "$START_FROM" -le 10 ] && [ "$STOP_AFTER" -ge 10 ]; then
    # Step 10: 50-Slice Pathology Z-Stack Cross-Sections & Microtome GIF
    echo "[Step 10] Generating 50-Slice Pathology Z-Stack Cross-Sections & Microtome GIF..."
    
    # Prioritize OBB solid table volume if present, otherwise use full
    TARGET_PLY=""
    if [ -f "${SESSION_DIR}/pointcloud_raw_obb_solid_table.ply" ]; then
        TARGET_PLY="${SESSION_DIR}/pointcloud_raw_obb_solid_table.ply"
    elif [ -f "${SESSION_DIR}/pointcloud_raw_full_solid_table.ply" ]; then
        TARGET_PLY="${SESSION_DIR}/pointcloud_raw_full_solid_table.ply"
    fi
    
    if [ -n "$TARGET_PLY" ]; then
        TARGET_NPY="${TARGET_PLY%.ply}_volume.npy"
        if [ -f "$TARGET_NPY" ]; then
            Z_SLICES_DIR="${SESSION_DIR}/z_slices_50"
            echo " -> Generating 50 Z-slices for: $TARGET_PLY"
            $PYTHON_ENV tools/generate_z_slices.py --npy "$TARGET_NPY" --ply "$TARGET_PLY" --out_dir "$Z_SLICES_DIR" --slices 50
        fi
    fi
    echo "- [x] Step 10: Generated 50-Slice Pathology Z-Stack & Microtome GIF" >> "$PROGRESS_FILE"
fi

echo "Pipeline script finished!"
if [ "$START_FROM" -le 10 ] && [ "$STOP_AFTER" -ge 10 ]; then
    echo "- [x] Pipeline fully completed!" >> "$PROGRESS_FILE"
fi
