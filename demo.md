# Software Pipeline Demonstration & Documentation (`demo.md`)

* **Dataset:** [`tests_and_images/captures_8-16_lesion_3`]
* **Compute Platform:** NVIDIA L4 GPU (24GB VRAM) on Linux through Google Cloud Compute
* **Session Directory:** [`tests_and_images/nerf_240926_0008_8-16_lesion_3/`]
* **Total Runtime:** ~4.5 minutes (including NeRF training, 1M point extraction, 2.5D extrusion, Haralick profiling, and 50 Z-slicing)

## Overview

* Stage 1: Image Preprocessing
* Stage 2: Multi-Factor Segmentation
* Stage 3: Calibration & Pose Alignment
* Stage 4: Neural Radiance Field Optimization (NeRF)
* Stage 5: Point Cloud Raycasting
* Stage 6: 2.5D Solid Extrusion & Voxelization
* Stage 7: Orbiting Visualization
* Stage 8: Mask Reprojection
* Stage 9: Feature Extraction
* Stage 10: Visualization


## Workflow Initiation

The entire workflow was launched via a single root command:

```bash
./tools/run_pipeline.sh tests_and_images/captures_8-16_lesion_3 \
    tests_and_images/captures_8-13_calibration_results/calibration.json \
    --steps 2000
```

### Stage 1 & 2: Image Preprocessing & Multi-Factor Segmentation
* **Script:** [`tools/prepare_images_and_masks.py`]
* **Operations:**
  * Downscaled 64 raw $4608 \times 2592$ captures by factor of 4 ($1152 \times 648$).
  * Deep learning foreground proposal via $U^2$-Net (`rembg`) with binary hole-filling.
  * Deterministic HSV Chroma-Green plate exclusion (`lower=[35, 40, 40]`, `upper=[85, 255, 255]`).
  * Geometric ArUco tag convex hull exclusion dilated with a $41 \times 41$ morphological kernel.
  * $5 \times 5$ morphological opening for zero floating boundary noise.
* **Artifacts Created:**
  * [`images_4/`]: 64 undistorted downscaled captures.
  * [`masks_4/`]: 64 clean binary silhouette masks.
  * [`masked_images_preview_4/`]: Composite foreground previews.

---

### Stage 3: Calibration & Pose Alignment
* **Scripts:** [`tools/apply_golden_calibration.py`], [`tools/plot_transforms_3d.py`]
* **Operations:**
  * Detected ChArUco tag in `capture_0.jpg` to determine turntable starting phase angle ($\theta_0$).
  * Synthesized 64 rigid camera poses along the circular orbit with pitch angle $\alpha \approx 40.8^\circ$ and radius $R = 174.097\text{ mm}$.
  * Aligned turntable table plane with the mathematical horizontal plane ($Z = 0$).
* **Artifacts Created:**
  * [`transforms_aligned.json`]: Canonical NeRF camera trajectory.
  * [`camera_poses_3d.png`]: 3D vector diagram of the 64 camera viewing frustums.

---

### Stage 4: Neural Radiance Field Optimization (NeRF)
* **Engine:** Nerfstudio (`ns-train nerfacto`)
* **Operations:**
  * Trained continuous volumetric radiance fields with multi-resolution hash encoding (`tiny-cuda-nn`).
  * Processed 2,000 iterations @ ~120,000 rays/sec (training time: 2 min 14 sec).
* **Artifacts Created:**
  * [`outputs/.../step-000001999.ckpt`]: Neural network weights checkpoint.
  * [`outputs/.../config.yml`]: Complete hyperparameter and dataparser configuration.

---

### Stage 5: Point Cloud Raycasting
* **Engine:** Nerfstudio (`ns-export pointcloud`)
* **Operations:**
  * Dense volumetric sampling raycasting 1,000,000 discrete points.
  * Extracted both the unconstrained full scene and the cropped Oriented Bounding Box (OBB, scale $0.8 \times 0.8 \times 0.8$).
  * Open3D statistical outlier removal and surface normal estimation.
* **Artifacts Created:**
  * [`pointcloud_raw_full.ply`]: 1,000,000 points full point cloud ($27.0\text{ MB}$).
  * [`pointcloud_raw_obb.ply`]: 999,582 points cropped lesion point cloud ($27.0\text{ MB}$).

---

### Stage 6: 2.5D Solid Extrusion & Voxelization
* **Scripts:** [`tools/fill_25d_extrusion.py`], [`tools/export_bio_format.py`]
* **Operations:**
  * **Table Trim:** Trimmed below-table reconstruction noise ($Z > 0.0\text{ mm}$).
  * **2.5D Solid Infill:** Extruded downward vertical voxel pillars from the specimen surface to $Z = 0$, transforming the hollow NeRF shell into a 100% solid, watertight biopsy volume.
  * **Chirality Preservation:** Applied a $180^\circ$ rotation around the X-axis to orient the specimen right-side up while strictly preserving anatomical left/right chirality.
  * **Voxelization:** Converted into a 3D binary occupancy grid (`.npy`) at pitch $0.002 = 0.3482\text{ mm/voxel}$ ($0.0422\text{ mm}^3$ unit voxel volume).
* **Artifacts Created:**
  * [`pointcloud_raw_obb_solid_table.ply`]: Solid extruded point cloud ($34.2\text{ MB}$).
  * [`pointcloud_raw_obb_solid_table_volume.npy`]: 3D binary numpy volume ($139 \times 110 \times 24$ voxels, $367\text{ KB}$), natively compatible with **Napari**.

---

### Stage 7: Orbiting Visualization
* **Script:** [`tools/render_pointcloud_video.py`]
* **Operations:**
  * Synthesized a complete $360^\circ$ smooth orbiting camera fly-around rendering each point cloud into high-framerate GIFs.
* **Artifacts Created:**
  * [`pointcloud_raw_full.gif`]: Full scene orbit animation ($13.0\text{ MB}$).
  * [`pointcloud_raw_obb.gif`]: Cropped hollow shell orbit animation ($25.7\text{ MB}$).
  * [`pointcloud_raw_obb_solid_table_reprojected.gif`]: Final solid 2.5D extruded volume orbit animation ($20.5\text{ MB}$).

---

### Stage 8: Mask Reprojection
* **Script:** [`tools/project_all_masks.py`]
* **Operations:**
  * Closed-loop verification: reprojected the reconstructed 3D point cloud back onto all 64 camera sensor planes through the calibrated intrinsic matrix $\mathbf{K}$.
  * Generated side-by-side comparative panels displaying the original physical camera mask versus the projected NeRF silhouette.
* **Artifacts Created:**
  * [`projections_side_by_side/`]: 64 side-by-side PNG comparisons (`compare_capture_0.png` through `compare_capture_63.png`).

---

### Stage 9: Feature Extraction
* **Script:** [`tools/extract_haralick_features.py`]
* **Operations:**
  * Executed the clinical 17-feature diagnostic extraction:
    1. Metric volume calculation in voxels, $\text{mm}^3$, and $\text{cm}^3$.
    2. Marching Cubes triangular surface area mesh calculation.
    3. Wadell's sphericity index: $\Psi = \frac{\pi^{1/3}(6V)^{2/3}}{\text{SA}}$.
    4. Recursive morphological erosion with string-prefix distance and hierarchical dendrogram clustering to count topological lobes.
    5. 13 3D Haralick Gray-Level Co-occurrence Matrix (GLCM) texture metrics averaged over all 13 unique 3D spatial directions.
* **Artifacts Created:**
  * [`haralick_morphometry/pointcloud_raw_obb_morphometry_report.md`]: Markdown summary report.
  * [`haralick_morphometry/pointcloud_raw_obb_features.json`]: Machine-readable JSON dictionary.
  * [`haralick_morphometry/pointcloud_raw_obb_features.csv`]: Tabular dataset record.
  * [`haralick_morphometry/pointcloud_raw_obb_morphology_radar.png`]: Normalized polar radar chart.

---

### Stage 10: Visualization
* **Script:** [`tools/generate_z_slices.py`]
* **Operations:**
  * Digitally sliced the solid 3D binary volume into 50 equidistant metric horizontal sections from table datum ($Z = 0.00\text{ mm}$) to tumor apex ($Z = 7.92\text{ mm}$).
  * Computed cross-sectional area ($\text{mm}^2$ and $\text{cm}^2$) for each individual slice.
  * Rendered individual cross-section PNGs with metric axes, assembled an overall $5 \times 10$ montage overview, and rendered an animated ping-pong microtome GIF.
* **Artifacts Created:**
  * [`z_slices_50/`]: 50 individual high-resolution PNG slices (`slice_00.png` to `slice_49.png`).
  * [`z_slices_50/z_slices_50_montage.png`]: Complete $5 \times 10$ montage overview across all 50 elevation planes.
  * [`z_slices_50_timelapse.gif`]: 50-slice microtome serial cross-sectioning timelapse animation.

---

## Session Directory Artifact Tree

All generated files in [`tests_and_images/nerf_240926_0008_8-16_lesion_3/`] after running the pipeline:

```
nerf_240926_0008_8-16_lesion_3/
├── progress.md                                      # 10-stage completion audit checklist
├── transforms.json                                  # NeRF dataparser camera transforms
├── transforms_aligned.json                          # Coordinate-aligned golden transforms
├── transforms_unaligned.json                        # Pre-rotation raw transforms
├── camera_poses_3d.png                              # 3D visual plot of camera viewing orbit
├── images_4/                                        # 64 downscaled captures (1152x648)
├── masks_4/                                         # 64 multi-factor binary silhouette masks
├── masked_images_preview_4/                         # 64 masked foreground input images for NeRF
├── masked_with_annotation_4/                        # 64 annotated verification frames
├── outputs/                                         # NeRF training weights & TensorBoard logs
│   └── nerf_240926_.../nerfacto/.../step-000001999.ckpt
├── pointcloud_raw_full.ply                          # 1,000,000 points full point cloud (27MB)
├── pointcloud_raw_full.gif                          # 360° orbiting animation of full scene (13MB)
├── pointcloud_raw_obb.ply                           # 999,582 points cropped lesion point cloud (27MB)
├── pointcloud_raw_obb.gif                           # 360° orbiting animation of lesion (25.7MB)
├── pointcloud_raw_obb_solid_table.ply               # 2.5D solid extruded watertight mesh (34.2MB)
├── pointcloud_raw_obb_solid_table_reprojected.gif   # 360° orbiting animation of solid biopsy (20.5MB)
├── pointcloud_raw_obb_solid_table_volume.npy        # Napari 3D binary occupancy grid (139x110x24)
├── haralick_morphometry/                            # 17-feature clinical diagnostic outputs
│   ├── pointcloud_raw_obb_morphometry_report.md     # Formatted clinical morphological summary
│   ├── pointcloud_raw_obb_features.json             # Structured JSON feature vector
│   ├── pointcloud_raw_obb_features.csv              # Tabular CSV metric record
│   └── pointcloud_raw_obb_morphology_radar.png      # 6-axis normalized morphological radar chart
├── z_slices_50/                                     # Virtual microtome cross-sectional series
│   ├── slice_00.png ... slice_49.png                # 50 metric cross-section PNGs with mm axes
│   └── z_slices_50_montage.png                      # Complete 5x10 visual montage of all slices
├── z_slices_50_timelapse.gif                        # Microtome animated serial slicing timelapse
└── projections_side_by_side/                        # 64 closed-loop reprojection verification PNGs
    └── compare_capture_0.png ... compare_capture_63.png
```

---

## Reproducibility

To replicate this exact run on any other dataset (e.g. `captures_8-16_lesion_4` or `captures_8-16_hemisphere`):

```bash
# General automated run (auto-timestamped, 2000 steps)
./tools/run_pipeline.sh <raw_captures_dir> tests_and_images/captures_8-13_calibration_results/calibration.json --steps 2000

# High-resolution convergence run (20,000 steps)
./tools/run_pipeline.sh <raw_captures_dir> tests_and_images/captures_8-13_calibration_results/calibration.json --steps 20000
```
