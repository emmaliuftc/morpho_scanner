# Morpho Scanner Visualization Tools

This directory contains utility scripts to project, verify, and visualize the derived camera trajectory calibration and 3D reconstruction outputs.

---

## 1. visualize_center_on_masked.py
Projects the derived 3D turntable center onto the **green-filtered (masked)** images where the background is white. 

### How to Run:
```bash
.venv/bin/python tools/visualize_center_on_masked.py
```
*   **Input Poses:** `COLMAP/workspace/sparse_unmasked_triangulated/`
*   **Input Images:** `COLMAP/workspace/images/`
*   **Output Folder:** `COLMAP/workspace/visualizations/plate_center_masked/`

---

## 2. visualize_z_axle.py
Projects the **3D rotation Z-axis (Z-axle)** line and normal vector arrow onto the **unmasked raw** images. This shows the physical spindle and center of rotation of the turntable relative to the Lego block.

### How to Run:
```bash
.venv/bin/python tools/visualize_z_axle.py
```
*   **Input Poses:** `COLMAP/workspace/sparse_unmasked_triangulated/`
*   **Input Images:** `COLMAP/workspace/raw_images/`
*   **Output Folder:** `COLMAP/workspace/visualizations/z_axle_projections/`

---

## 3. render_3d_lego.py
Renders a 3D perspective surface plot of the Poisson reconstruction mesh in shaded red color, applying a cylinder spatial crop based on the derived center to filter out any background/rim outliers.

*(Note: This script requires `open3d` and should be run using the conda environment).*

### How to Run:
```bash
/home/coding/miniconda/envs/napari-env/bin/python tools/render_3d_lego.py
```
*   **Input Mesh:** `COLMAP/workspace/visualizations/poisson_mesh.ply`
*   **Input Point Cloud:** `COLMAP/workspace/lego_block_final.ply`
*   **Output Render Image:** `COLMAP/workspace/visualizations/lego_rendered_3d.png`

---

## 4. test_true_step_size.py
Triangulates the point cloud of the Lego block using the theoretical motor step size of $11.25^\circ$ ($360^\circ / 32$) instead of the physical $10.0^\circ$ gear-ratio corrected step size. This serves as a comparison script to prove the existence of the turntable gear ratio.

### How to Run:
```bash
.venv/bin/python tools/test_true_step_size.py
```
*   **Input Poses:** Analytical circular trajectory with $11.25^\circ$ increments
*   **Input Images:** `COLMAP/workspace/images/`
*   **Output Poses:** `COLMAP/workspace/sparse_ideal_triangulated/`

---

## 5. fit_gear_ratio.py
Reads the actual 3D camera poses reconstructed by COLMAP in `sparse/1`, fits a circle to the camera centers using SVD plane projection, unwraps the camera rotation angles, and runs linear regression to fit the actual step size of the turntable (proving the $9:8$ gear ratio).

### How to Run:
```bash
.venv/bin/python tools/fit_gear_ratio.py
```
*   **Input Reconstruction:** `COLMAP/workspace/sparse/1`
*   **Output:** Prints the fitted rotation slope in radians and degrees, expected step size, and residual error.
