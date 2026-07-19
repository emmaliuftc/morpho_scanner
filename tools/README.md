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
