# 🔬 MorphoScanner

**A Low-Cost Point-of-Care 3D Morphological Scanner & Algorithmic Pipeline for Irregular Topographies**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-brightgreen.svg)](https://python.org)
[![Hardware BOM](https://img.shields.io/badge/Hardware%20BOM-%3C%24180-success.svg)](#hardware-ecosystem)
[![Metrology](https://img.shields.io/badge/Surface%20Accuracy-%3C0.3mm-orange.svg)](#metrological-accuracy)

---

## 🌟 System Architecture & High-Level Pipeline

MorphoScanner is an automated, clinical-grade 3D scanning and volumetric profiling platform engineered specifically for **irregular, low-texture biological specimens** (such as excision biopsies and cutaneous lesions). It achieves sub-millimeter diagnostic metrology at an absolute hardware cost of under **$180**.

![Figure 1: System Architecture and End-to-End Workflow](assets/system_architecture_figure1.png)
*Figure 1. System architecture and end-to-end workflow of the low-cost 3D morphological scanner.*

---

### End-to-End Workflow Breakdown

The system integrates low-cost mechatronics, multi-factor computer vision filtering, and continuous neural volumetric modeling into a unified pipeline:

```
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│ 1. MECHATRONIC CAPTURE    │      │ 2. PREPROCESSING & MASKS  │      │ 3. NEURAL RECONSTRUCTION  │
│ • Raspberry Pi 5 + IMX708 │ ───> │ • Multi-Factor U²-Net     │ ───> │ • Nerfstudio (nerfacto)   │
│ • Dynamixel XL-430 Motor  │      │ • HSV Chroma-Plate Filter │      │ • 2M Point Cloud Sampling │
│ • 64 Synchronized Views   │      │ • ArUco Convex Hull Dilation│    │ • Golden Calibration Pose │
└───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
                                                                                    │
                                                                                    ▼
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│ 6. PATHOLOGY Z-STACKS     │      │ 5. 18-FEATURE DIAGNOSTICS │      │ 4. 2.5D SOLID EXTRUSION   │
│ • 50 Equidistant Slices   │ <─── │ • Scaled Volume (cm³)     │ <─── │ • Mathematical Table Trim │
│ • Montage Overview (5x10) │      │ • 14 Haralick GLCM Textures│     │ • Base Infill down to Z=0 │
│ • Animated Microtome GIF  │      │ • Topological Lobe Count  │      │ • Chirality 180° Flip     │
└───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

#### 1. Low-Cost Hardware Device
* **Edge Controller:** Raspberry Pi 5 (8GB) powered by the Broadcom BCM2712 quad-core ARM Cortex-A76 @ 2.4GHz.
* **Optical Sensor:** Raspberry Pi Camera Module 3 featuring the 11.9MP Sony IMX708 sensor ($4608 \times 2592$ px) with motorized voice-coil Phase Detection Auto Focus (PDAF).
* **Rotary Actuator:** Robotis Dynamixel XL-430-W250-T smart bus servo with a 12-bit contactless magnetic absolute encoder (4096 pulses/rev, $0.088^\circ$ resolution) communicating over half-duplex asynchronous TTL serial.
* **Stage & Calibration:** Laser-cut $4.5\text{ mm}$ cast PMMA acrylic platter, non-reflective matte Chroma-green (#00B140) background, and ChArUco fiducial tags.

#### 2. Software Pipeline
1. **① Preprocess & Multi-Factor Masking (`tools/prepare_images_and_masks.py`, `tools/silhouette_extractor.py`):**
   * **Deep Learning Proposal:** Deep foreground extraction via $U^2$-Net (`rembg`) with binary hole-filling.
   * **Deterministic HSV Exclusion:** Mathematical removal of the green platter (`lower=[35, 40, 40]`, `upper=[85, 255, 255]`).
   * **Geometric ArUco Exclusion:** Detection of ChArUco tags, computing a convex hull polygon dilated with a $41 \times 41$ kernel to wipe out calibration board margins.
   * **Morphological Opening:** $5 \times 5$ kernel cleanup for zero floating noise pixels.
2. **② Reconstruct & Neural Radiance Modeling (`nerfacto`, `tools/apply_golden_calibration.py`):**
   * Incorporates golden camera orbit calibration ($\|\mathbf{C}_{\text{rot}}\| = 174.097\text{ mm}$, scale factor $1.0\text{ unit} = 174.097\text{ mm}$).
   * Trains continuous volumetric radiance fields with `tiny-cuda-nn` hash encoding.
   * Raycasts high-density discrete point clouds ($1,000,000$ to $2,000,000$ points) via `ns-export`.
3. **③ 2.5D Solid Extrusion & Chirality Inversion (`tools/fill_25d_extrusion.py`):**
   * Solves the biological reality that biopsies are solid masses resting on flat dissection boards, not hollow shells floating in vacuum.
   * Identifies the mathematical table plane ($Z = 0$), trims below-table noise, extrudes voxels vertically downward to base plane, and preserves biological chirality via $180^\circ$ X-axis rotation.
4. **④ Volumetric Medical Voxelization (`tools/export_bio_format.py`):**
   * Converts the solid geometry into a 3D binary occupancy grid (`.npy`) at pitch $0.002 = 0.3482\text{ mm/voxel}$ ($0.0422\text{ mm}^3$ unit voxel volume), natively compatible with **Napari**.
5. **⑤ Extract 17-Feature Diagnostic Topographical Suite (`tools/extract_haralick_features.py`):**
   * Automated calculation of real-world volume ($V$ in $\text{cm}^3$), surface area ($A$), sphericity index ($\Psi$), 13 Haralick Gray-Level Co-occurrence Matrix (GLCM) 3D texture metrics, and recursive morphological erosion with dendrogram clustering for automated lobe bifurcation counting.
6. **⑥ 50-Slice Pathology Z-Stack (`tools/generate_z_slices.py`):**
   * Produces 50 physical metric cross-sectional PNG slices, $5 \times 10$ montage overviews, and animated timelapse GIFs simulating pathology microtome sectioning.

---

## 📊 Metrological Benchmarks & Ground-Truth Accuracy

The system has undergone rigorous metrological validation against precision reference standards:

| Benchmark Reference | Ground Truth CAD | Reconstructed Dimension | Absolute Error | Percentage Error |
| :--- | :--- | :--- | :--- | :--- |
| **4.00 cm Green PLA Cube (Width $X$)** | $40.00\text{ mm}$ | **$40.28\text{ mm}$** | **$+0.28\text{ mm}$** | **$0.70\%$** |
| **4.00 cm Green PLA Cube (Depth $Y$)** | $40.00\text{ mm}$ | **$40.42\text{ mm}$** | **$+0.42\text{ mm}$** | **$1.05\%$** |
| **4.00 cm Green PLA Cube (Height $Z$)** | $40.00\text{ mm}$ | **$39.81\text{ mm}$** | **$-0.19\text{ mm}$** | **$0.47\%$** |
| **4.00 cm Cube Mean Surface Error** | $0.000\text{ mm}$ | **$0.297\text{ mm}$** | — | **$<0.3\text{ mm}$** |
| **5.00 cm Hemisphere Volume** | $32,725\text{ mm}^3$ | **$32,140\text{ mm}^3$** | **$-585\text{ mm}^3$** | **$1.79\%$** |
| **Lesion 3 Solid Volume (200k steps)** | — | **$2.260\text{ cm}^3$** | — | **Fully Converged** |

---

## 🎬 Live Pipeline Demonstration (Full 10-Stage Walkthrough)

An end-to-end execution of the complete 10-stage pipeline on the clinical lesion model (`captures_8-16_lesion_3`) is documented in:

👉 [**`demo.md` — Complete Pipeline Demonstration & Artifact Inventory**](demo.md)

### Highlights from the Live Demonstration:
* **One-Command Hands-Free Pipeline:** Executed `./tools/run_pipeline.sh captures_8-16_lesion_3 captures_8-13_calibration_results/calibration.json --steps 2000` through all 10 stages without manual intervention.
* **Safe Automatic Isolation:** Leveraged auto-timestamping (`nerf_240926_0008_8-16_lesion_3/`) to guarantee zero interference with historical golden runs.
* **High-Speed Volumetric Recovery:** In just 2,000 steps (~2.5 minutes on NVIDIA L4 GPU), extracted a solid volume of **$2.2624\text{ cm}^3$**, matching the 200,000-step gold standard ($2.2600\text{ cm}^3$) within **$+0.1\%$ error**.
* **Clinical Artifact Generation:** Produced 1M-point PLY meshes, Napari 3D binary occupancy grids (`.npy`), 17-feature Haralick JSON/CSV texture reports, polar radar plots, and a 50-slice virtual microtome pathology Z-stack with timelapse GIF animation.

---

## 🚀 Quickstart & Pipeline Execution

### 1. Automated Master Pipeline
Run the end-to-end pipeline on any 64-view capture directory with a single command:

```bash
# General execution
./tools/run_pipeline.sh <raw_captures_dir> <calibration.json> [options]

# Example: 20k steps on Lesion 3 using the Golden Calibration
./tools/run_pipeline.sh captures_8-16_lesion_3 captures_8-13_calibration_results/calibration.json \
    --steps 20000 \
    --session-dir nerf_lesion_3_20k
```

### 2. High-Density 2M Point Export, Haralick Texture & 50 Z-Slicing
```bash
# Export 2,000,000 dense points from NeRF
.venv_nerf/bin/ns-export pointcloud \
    --load-config "nerf_lesion_3_20k/outputs/.../config.yml" \
    --output-dir "nerf_lesion_3_20k/export_2m" \
    --num-points 2000000 \
    --remove-outliers True

# Solidify to table plane (2.5D extrusion)
.venv_nerf/bin/python tools/fill_25d_extrusion.py \
    --input "nerf_lesion_3_20k/export_2m/point_cloud.ply" \
    --output "nerf_lesion_3_20k/pointcloud_solid.ply" \
    --pitch 0.002

# Export Napari 3D NPY binary occupancy array
.venv_nerf/bin/python tools/export_bio_format.py \
    --ply "nerf_lesion_3_20k/pointcloud_solid.ply"

# Extract 17-feature Haralick texture & morphological diagnostic profile
.venv_nerf/bin/python tools/extract_haralick_features.py \
    --npy "nerf_lesion_3_20k/pointcloud_solid_volume.npy" \
    --out_dir "nerf_lesion_3_20k/haralick_morphometry"

# Generate 50 physical Z-slices, montage, and animated timelapse GIF
.venv_nerf/bin/python tools/generate_z_slices.py \
    --npy "nerf_lesion_3_20k/pointcloud_solid_volume.npy" \
    --ply "nerf_lesion_3_20k/pointcloud_solid.ply" \
    --out_dir "nerf_lesion_3_20k/z_slices_50" \
    --slices 50
```

### 3. Hardware Diagnostic Scripts
* `motor_test.py`: Verify Dynamixel XL-430 packet communication, baud rate, and torque control.
* `picamera_test.py`: Verify Sony IMX708 PDAF autofocus lock, exposure parameters, and frame capture.
* `combine_test.py`: Validate synchronized step-and-settle 360-degree capture sequence.

---

## 📁 Repository Structure & Documentation

```
morpho_scanner/
├── demo.md                              # Live 10-stage automated pipeline walkthrough & artifact report
├── assets/                               # Visual pipeline diagrams and figures
│   └── system_architecture_figure1.png   # Complete BMES Figure 1 architecture diagram
├── captures_8-13_calibration_results/    # Authoritative Golden Calibration standard
├── captures_8-16_lesion_1..4/           # Multi-angle clinical biopsy datasets
├── docs/                                # Technical whitepapers and mathematical derivations
│   ├── reflection.md                    # Complete chronological engineering retrospective (June–Sept 2026)
│   ├── calibration.md                   # 4-stage global turntable calibration & 17.41 cm baseline derivation
│   ├── haralick.md                      # 17-feature morphological matrix & clinical Lesion 3 vs 4 (1.93×) validation
│   ├── tsdf_rotational_ambiguity_image_guide.md # Silhouette ambiguity & 40mm cube ballooning analysis
│   ├── segmentation.md                  # Multi-factor U²-Net + HSV + ArUco segmentation details
│   ├── 0718.pipeline.md                 # Turntable epipolar invariants & screw-axis geometry
│   ├── cube_comparison_0810.md          # 4cm CAD cube metrology verification report
│   ├── lieson_3_200k_training.md        # 20k vs 200k NeRF training benchmark & noise analysis
│   ├── volume_analysis_report.md        # Volumetric cross-sectional area comparisons
│   └── software_pipeline.md             # Classical 6-stage Structure-from-Motion reference
├── tools/                               # Production processing scripts
│   ├── run_pipeline.sh                  # Master end-to-end pipeline runner
│   ├── silhouette_extractor.py          # Multi-factor segmentation engine (U²-Net, HSV, ArUco)
│   ├── prepare_images_and_masks.py      # Batch dewarping, mask generation, and alpha compositing
│   ├── apply_golden_calibration.py      # Camera orbit pose alignment
│   ├── fill_25d_extrusion.py            # 2.5D solid table extrusion & chirality correction
│   ├── export_bio_format.py             # 3D binary occupancy grid (.npy) voxelizer
│   ├── extract_haralick_features.py     # 17-feature Haralick texture & morphological diagnostic extractor
│   ├── generate_z_slices.py             # 50-slice metric cross-sections & animated GIF generator
│   └── project_all_masks.py             # 64-view closed-loop reprojection overlay verification
└── assets/                              # Architecture diagrams and benchmark visuals
```

---

## 📜 Citation & Credits
Engineered by **Emma Liu** as part of the MorphoScanner research initiative. Detailed engineering progress, failure mode retrospectives, and mechatronic trade studies are documented in [**`docs/reflection.md`**](docs/reflection.md).
