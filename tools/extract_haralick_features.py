#!/usr/bin/env python3
"""
tools/extract_haralick_features.py
==================================
Morphometric & Haralick Texture Feature Extraction Pipeline for MorphoScanner.

Extracts the clinical 17-feature morphological vector from a 3D binary occupancy grid (.npy):
  [0]  Volume (voxel count + physical volume in mm³ and cm³)
  [1]  Surface Area (Marching cubes mesh in grid units and mm²)
  [2]  Sphericity index: Ψ = π^(1/3) * (6V)^(2/3) / SA
  [3]  Lobe Count (Recursive erosion with adaptive footprint + hierarchical string-prefix clustering)
  [4-16] 13 3D Haralick GLCM texture features (mean across all 13 spatial 3D directions):
       - Haralick 1: Angular Second Moment (ASM / Energy)
       - Haralick 2: Contrast
       - Haralick 3: Correlation
       - Haralick 4: Sum of Squares: Variance
       - Haralick 5: Inverse Difference Moment (IDM / Homogeneity)
       - Haralick 6: Sum Average
       - Haralick 7: Sum Variance
       - Haralick 8: Sum Entropy
       - Haralick 9: Entropy
       - Haralick 10: Difference Variance
       - Haralick 11: Difference Entropy
       - Haralick 12: Information Measure of Correlation I (IMC-I)
       - Haralick 13: Information Measure of Correlation II (IMC-II)

Outputs:
  - <name>_features.json
  - <name>_features.csv
  - <name>_morphometry_report.md
  - <name>_morphology_radar.png (optional plot)
"""

import os
import sys
import json
import csv
import math
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from skimage import measure, morphology
import mahotas
from scipy import spatial, cluster as sp_cluster

# Suppress benign warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

HARALICK_NAMES = [
    "ASM", "Contrast", "Correlation", "Sum_of_Squares_Variance", "IDM",
    "Sum_Average", "Sum_Variance", "Sum_Entropy", "Entropy",
    "Diff_Variance", "Diff_Entropy", "IMC_I", "IMC_II"
]

FEATURE_NAMES = ["Volume_voxels", "Surface_Area", "Sphericity", "Lobe_Count"] + HARALICK_NAMES

MAX_EROSION_DEPTH = 20
DEFAULT_GOLDEN_SCALE_MM = 174.09726104178188  # mm per NeRF unit
DEFAULT_VOXEL_PITCH = 0.002                    # NeRF units


def fill_safe(binary_vol):
    """
    Fill small internal cavities slice-by-slice and then in 3D.
    Guards against over-inflation (>3× volume).
    """
    original_count = np.count_nonzero(binary_vol)
    if original_count == 0:
        return binary_vol.astype(bool)

    newfile = np.zeros_like(binary_vol, dtype=bool)
    for i in range(len(binary_vol)):
        roi = binary_vol[i, :, :] > 0
        try:
            new = morphology.remove_small_holes(roi, area_threshold=4000)
        except Exception:
            new = roi
        newfile[i, :, :] = new

    try:
        newfile = morphology.remove_small_holes(newfile, area_threshold=100000)
    except Exception:
        pass

    filled_count = np.count_nonzero(newfile)
    if filled_count > 3 * original_count:
        print(f"    ⚠️ Warning: Hole-filling inflated volume {original_count} → {filled_count}, reverting to unfilled.")
        return binary_vol.astype(bool)

    return newfile


def compute_volume(binary_vol):
    """Count of nonzero solid voxels."""
    return int(np.count_nonzero(binary_vol))


def compute_surface_area(binary_vol):
    """Surface area via Marching Cubes mesh triangulation."""
    try:
        verts, faces, normals, values = measure.marching_cubes(binary_vol.astype(np.uint8), step_size=1)
        sa = float(measure.mesh_surface_area(verts, faces))
    except Exception as e:
        print(f"    ⚠️ Marching cubes failed ({e}), returning 0.0")
        sa = 0.0
    return sa


def compute_sphericity(volume, surface_area):
    """Wadell's sphericity index: Ψ = π^(1/3) * (6V)^(2/3) / SA."""
    if surface_area <= 0 or volume <= 0:
        return 0.0
    return float(pow(math.pi, 1.0 / 3.0) * pow(6.0 * volume, 2.0 / 3.0) / surface_area)


def compute_haralick(binary_vol):
    """Compute 13 Haralick texture features averaged over 13 3D spatial directions."""
    vol = binary_vol.astype(np.int32)
    try:
        h = mahotas.features.haralick(vol, return_mean=True)
        return h.tolist()
    except Exception as e:
        warnings.warn(f"Haralick computation failed: {e}")
        return [0.0] * 13


def count_comp(binary_vol, depth, id_tuple, ids, footprint, max_depth=MAX_EROSION_DEPTH):
    """
    Recursive morphological erosion component decomposition.
    Assigns hierarchical branching codes (e.g. 'a', 'ab', 'aba').
    """
    if depth >= max_depth:
        ids.append(id_tuple)
        return

    try:
        eroded = morphology.erosion(binary_vol.astype(np.uint8), footprint=footprint).astype(bool)
    except Exception:
        ids.append(id_tuple)
        return

    if np.count_nonzero(eroded) < 20:
        ids.append(id_tuple)
        return

    labels, labelcount = measure.label(eroded, connectivity=1, return_num=True)
    if labelcount > 0:
        found_any = False
        for i in range(1, labelcount + 1):
            component = (labels == i)
            if np.count_nonzero(component) < 50:
                continue
            found_any = True
            new_code = id_tuple[1] + chr(min(i + 96, 122))
            count_comp(component, depth + 1, (depth + 1, new_code), ids, footprint, max_depth)
        if not found_any:
            ids.append(id_tuple)
    else:
        ids.append(id_tuple)


def longest_common_prefix(s1, s2):
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return s1[:i]


def string_prefix_distance(s1, s2, max_len):
    return max_len - len(longest_common_prefix(s1, s2))


def compute_lobe_count(binary_vol):
    """
    Lobe counting via recursive erosion, string-prefix distance, and hierarchical linkage clustering.
    """
    try:
        cleaned = morphology.remove_small_objects(binary_vol.astype(bool), min_size=100, connectivity=1)
    except TypeError:
        cleaned = morphology.remove_small_objects(binary_vol.astype(bool), max_size=100, connectivity=1)

    if np.count_nonzero(cleaned) < 100:
        return 1

    shape = cleaned.shape
    fz = max(1, min(shape[0] // 15, 3))
    fxy = max(3, min(min(shape[1], shape[2]) // 10, 8))
    footprint = np.ones(shape=(fz, fxy, fxy), dtype=np.uint8)

    ids = []
    count_comp(cleaned, 0, (0, "a"), ids, footprint)
    finalcodes = [id_tuple[1] for id_tuple in ids]

    if len(finalcodes) <= 1:
        return 1

    max_string_len = max(len(s) for s in finalcodes)

    def pdist_wrapper(u, v):
        return string_prefix_distance(u[0], v[0], max_string_len)

    string_array = np.array(finalcodes, dtype=object).reshape(-1, 1)

    try:
        dist_mat = spatial.distance.pdist(string_array, metric=pdist_wrapper)
        Z = sp_cluster.hierarchy.linkage(dist_mat, method='average')
        d = sp_cluster.hierarchy.dendrogram(
            Z, labels=finalcodes,
            color_threshold=0.8 * max(Z[:, 2]),
            no_plot=True
        )
        colors = [str(c) for c in d['leaves_color_list']]
        if len(np.unique(colors)) == 1:
            return 1 if len(colors) > 10 else len(colors)
        else:
            count = 0
            if colors.count('C0') > 5:
                count = 1
            else:
                count += colors.count('C0')
            count += len(np.unique([c for c in colors if c != 'C0']))
            return max(1, count)
    except Exception:
        return 1


def generate_radar_plot(metrics, out_path, specimen_name):
    """Generate a clean dark-theme radar chart of normalized diagnostic features."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        categories = [
            'Volume\n(rel.)', 'Surface Area\n(rel.)', 'Sphericity\n(Ψ)',
            'Lobe Count\n(#)', 'Haralick ASM\n(Energy)', 'Haralick Corr.\n(Spatial)'
        ]
        
        # Scale values to [0, 1] relative to typical lesion baselines for visualization
        v_norm = min(1.0, metrics['volume_cm3'] / 5.0)
        sa_norm = min(1.0, metrics['surface_area_cm2'] / 80.0)
        sph_norm = min(1.0, metrics['sphericity'] / 0.5)
        lobe_norm = min(1.0, metrics['lobe_count'] / 5.0)
        asm_norm = min(1.0, metrics['haralick']['ASM'])
        corr_norm = min(1.0, max(0.0, metrics['haralick']['Correlation']))

        values = [v_norm, sa_norm, sph_norm, lobe_norm, asm_norm, corr_norm]
        values += values[:1]  # repeat first value to close polygon

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True), facecolor='#0b1329')
        ax.set_facecolor('#0f172a')

        ax.fill(angles, values, color='#38bdf8', alpha=0.35)
        ax.plot(angles, values, color='#38bdf8', linewidth=2.5, marker='o', markersize=6)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='#e2e8f0', fontsize=10, fontweight='bold')
        ax.tick_params(colors='#94a3b8', pad=12)
        ax.set_yticklabels([])
        ax.grid(color='#334155', linestyle='--', linewidth=0.8)
        ax.spines['polar'].set_color('#475569')

        ax.set_title(f"Morphological Radar Profile: {specimen_name}", 
                     color='#38bdf8', fontsize=13, fontweight='bold', pad=25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#0b1329')
        plt.close()
        return True
    except Exception as e:
        print(f"    ⚠️ Could not generate radar plot: {e}")
        return False


def run_feature_extraction(npy_path, out_dir=None, name=None, calib_scale=DEFAULT_GOLDEN_SCALE_MM, voxel_pitch=DEFAULT_VOXEL_PITCH, make_plot=True):
    npy_path = Path(npy_path).resolve()
    if not npy_path.exists():
        raise FileNotFoundError(f"Binary volume file not found: {npy_path}")

    if name is None:
        name = npy_path.stem.replace("_solid_table_volume", "").replace("_volume", "")

    if out_dir is None:
        out_dir = npy_path.parent
    else:
        out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🔬 MorphoScanner Morphometric & Haralick Texture Analysis")
    print(f"Specimen: {name}")
    print(f"Source: {npy_path}")
    print(f"{'='*70}")

    # Load 3D binary volume
    vol = np.load(npy_path)
    raw_voxels = int(np.count_nonzero(vol))
    print(f"  Input Array Shape: {vol.shape}, Nonzero Voxels: {raw_voxels:,}")

    # Safe hole-filling
    filled = fill_safe(vol)
    filled_voxels = compute_volume(filled)
    print(f"  Hole-filled Volume: {filled_voxels:,} voxels")

    # Physical scaling derivations
    voxel_width_mm = voxel_pitch * calib_scale          # ~0.34819 mm
    voxel_vol_mm3 = voxel_width_mm ** 3                 # ~0.042215 mm³
    voxel_area_mm2 = voxel_width_mm ** 2                # ~0.121239 mm²

    volume_mm3 = filled_voxels * voxel_vol_mm3
    volume_cm3 = volume_mm3 / 1000.0

    # Marching cubes surface area
    sa_grid = compute_surface_area(filled)
    surface_area_mm2 = sa_grid * voxel_area_mm2
    surface_area_cm2 = surface_area_mm2 / 100.0

    # Sphericity
    sphericity = compute_sphericity(filled_voxels, sa_grid)

    # Lobe count
    print("  Calculating topological lobe count via recursive erosion...")
    lobe_count = compute_lobe_count(filled)
    print(f"  Detected Lobe Count: {lobe_count}")

    # 13 Haralick Texture Features
    print("  Computing 13 3D Haralick GLCM texture features across all directions...")
    haralick_vals = compute_haralick(filled)
    haralick_dict = {k: float(v) for k, v in zip(HARALICK_NAMES, haralick_vals)}

    metrics = {
        "specimen": name,
        "timestamp": datetime.now().isoformat(),
        "source_npy": str(npy_path),
        "grid_shape": list(vol.shape),
        "voxel_pitch_nerf": voxel_pitch,
        "calibration_scale_mm": calib_scale,
        "voxel_pitch_mm": voxel_width_mm,
        "unit_voxel_volume_mm3": voxel_vol_mm3,
        "raw_voxel_count": raw_voxels,
        "volume_voxels": filled_voxels,
        "volume_mm3": volume_mm3,
        "volume_cm3": volume_cm3,
        "surface_area_grid": sa_grid,
        "surface_area_mm2": surface_area_mm2,
        "surface_area_cm2": surface_area_cm2,
        "sphericity": sphericity,
        "lobe_count": lobe_count,
        "haralick": haralick_dict
    }

    # Print summary
    print(f"\n{'-'*70}")
    print(f"  📊 METRIC SUMMARY:")
    print(f"    • Solid Volume:    {filled_voxels:,} voxels | {volume_mm3:,.1f} mm³ | {volume_cm3:.4f} cm³")
    print(f"    • Surface Area:    {sa_grid:,.1f} grid units | {surface_area_mm2:,.1f} mm² | {surface_area_cm2:.2f} cm²")
    print(f"    • Sphericity (Ψ):  {sphericity:.4f}")
    print(f"    • Lobe Count:      {lobe_count} lobes")
    print(f"  🔬 KEY HARALICK DESCRIPTORS:")
    print(f"    • ASM (Energy):    {haralick_dict['ASM']:.6f}")
    print(f"    • Contrast:        {haralick_dict['Contrast']:.6f}")
    print(f"    • Correlation:     {haralick_dict['Correlation']:.6f}")
    print(f"    • IDM (Homog.):    {haralick_dict['IDM']:.6f}")
    print(f"{'-'*70}")

    # Save JSON
    json_path = out_dir / f"{name}_features.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"  ✅ Saved JSON metrics: {json_path}")

    # Save CSV
    csv_path = out_dir / f"{name}_features.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Specimen", "Volume_voxels", "Volume_cm3", "Surface_Area_mm2", "Sphericity", "Lobe_Count"] + HARALICK_NAMES
        row = [name, filled_voxels, f"{volume_cm3:.4f}", f"{surface_area_mm2:.2f}", f"{sphericity:.4f}", lobe_count] + [f"{v:.6f}" for v in haralick_vals]
        writer.writerow(header)
        writer.writerow(row)
    print(f"  ✅ Saved CSV metrics: {csv_path}")

    # Save Markdown Report
    md_path = out_dir / f"{name}_morphometry_report.md"
    with open(md_path, 'w') as f:
        f.write(f"# Morphometric & Haralick Texture Report: {name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Source Voxel Grid:** `{npy_path.name}` ({vol.shape[0]}×{vol.shape[1]}×{vol.shape[2]})  \n\n")
        f.write("## 1. Physical Volumetric & Morphological Metrics\n\n")
        f.write("| Parameter | Measured Value | Unit / Scale |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Solid Volume ($V$)** | **{filled_voxels:,}** voxels | **{volume_cm3:.4f} cm³** ({volume_mm3:,.1f} mm³) |\n")
        f.write(f"| **Surface Area ($A$)** | **{sa_grid:,.1f}** mesh units | **{surface_area_cm2:.2f} cm²** ({surface_area_mm2:,.1f} mm²) |\n")
        f.write(f"| **Sphericity ($\Psi$)** | **{sphericity:.4f}** | Dimensionless ($\Psi=1.0$ is perfect sphere) |\n")
        f.write(f"| **Topological Lobes** | **{lobe_count}** | Recursive erosion & dendrogram clustering |\n\n")
        f.write("## 2. 13 3D Haralick GLCM Texture Descriptors\n\n")
        f.write("| Feature # | Descriptor Name | Value | Clinical & Physical Meaning |\n")
        f.write("| :---: | :--- | :--- | :--- |\n")
        f.write(f"| 1 | Angular Second Moment (ASM) | `{haralick_dict['ASM']:.6f}` | Energy / internal solid density homogeneity |\n")
        f.write(f"| 2 | Contrast | `{haralick_dict['Contrast']:.6f}` | Local gradient variance / absence of voids |\n")
        f.write(f"| 3 | Correlation | `{haralick_dict['Correlation']:.6f}` | Spatial voxel continuity across 13 3D directions |\n")
        f.write(f"| 4 | Sum of Squares (Variance) | `{haralick_dict['Sum_of_Squares_Variance']:.6f}` | Internal dispersion of voxel co-occurrences |\n")
        f.write(f"| 5 | Inverse Difference Moment (IDM) | `{haralick_dict['IDM']:.6f}` | Local boundary smoothness approaching 1.0 |\n")
        f.write(f"| 6 | Sum Average | `{haralick_dict['Sum_Average']:.6f}` | Background-to-mass transition average |\n")
        f.write(f"| 7 | Sum Variance | `{haralick_dict['Sum_Variance']:.6f}` | Dispersion of sum distribution |\n")
        f.write(f"| 8 | Sum Entropy | `{haralick_dict['Sum_Entropy']:.6f}` | Order / randomness of sum distribution |\n")
        f.write(f"| 9 | Entropy | `{haralick_dict['Entropy']:.6f}` | Structural disorder of binary occupancy |\n")
        f.write(f"| 10 | Difference Variance | `{haralick_dict['Diff_Variance']:.6f}` | Edge transition dispersion |\n")
        f.write(f"| 11 | Difference Entropy | `{haralick_dict['Diff_Entropy']:.6f}` | Crispness of boundary transitions |\n")
        f.write(f"| 12 | Info Measure of Correlation I | `{haralick_dict['IMC_I']:.6f}` | Mutual information across adjacent voxels |\n")
        f.write(f"| 13 | Info Measure of Correlation II | `{haralick_dict['IMC_II']:.6f}` | Cross-entropy scaling for spatial distribution |\n")
    print(f"  ✅ Saved Markdown report: {md_path}")

    # Optional Radar Plot
    if make_plot:
        plot_path = out_dir / f"{name}_morphology_radar.png"
        if generate_radar_plot(metrics, plot_path, name):
            print(f"  ✅ Saved radar plot: {plot_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 17-feature Haralick and morphological metrics from 3D binary volume")
    parser.add_argument("--npy", required=True, help="Path to 3D binary occupancy grid .npy file")
    parser.add_argument("--out_dir", default=None, help="Directory to save output reports and metrics (default: same as npy)")
    parser.add_argument("--name", default=None, help="Specimen name identifier (default: derived from file stem)")
    parser.add_argument("--calib_scale", type=float, default=DEFAULT_GOLDEN_SCALE_MM, help="Metric scale factor mm/unit")
    parser.add_argument("--voxel_pitch", type=float, default=DEFAULT_VOXEL_PITCH, help="Voxel grid pitch in NeRF units")
    parser.add_argument("--no_plot", action="store_true", help="Disable radar plot generation")

    args = parser.parse_args()
    run_feature_extraction(
        npy_path=args.npy,
        out_dir=args.out_dir,
        name=args.name,
        calib_scale=args.calib_scale,
        voxel_pitch=args.voxel_pitch,
        make_plot=not args.no_plot
    )
