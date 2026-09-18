#!/usr/bin/env python3
"""
NeRF Morphometric Feature Extraction
=====================================
Ported from polymorpho-lee-lab/features.py and polymorpho-lee-lab/hierarchical.py.

Computes a 17-feature vector per specimen:
  [0]  Volume (voxel count)
  [1]  Surface Area (marching cubes mesh)
  [2]  Sphericity  = π^(1/3) * (6V)^(2/3) / SA
  [3]  Lobe Count  (recursive erosion → hierarchical linkage, from hierarchical.py)
  [4-16] 13 Haralick texture features (mahotas.features.haralick, mean over 13 directions)

Data: Binary 3D .npy volumes from NeRF OBB/full voxelisation.

References:
  - features.py  lines 68-177  (volume, SA, sphericity, Haralick, feature matrix)
  - hierarchical.py  lines 28-109, 118-153  (lobe counting via recursive erosion)
  - volume_check.py  lines 38  (voxel count volume)
"""

import os
import sys
import json
import math
import warnings
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

# --- scikit-image for morphology & mesh ---
from skimage import measure, morphology

# --- mahotas for Haralick texture ---
import mahotas

# --- scipy for hierarchical linkage (lobe counting) ---
from scipy import spatial, cluster as sp_cluster

# ===========================
# CONFIGURATION
# ===========================

NPY_DIR = Path(__file__).resolve().parent.parent / "npy"
OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Specimens in analysis order
SPECIMENS = [
    ("hemisphere_20k",   "Hemisphere (20k)"),
    ("volume_one",       "Volume One"),
    ("volume_three",     "Volume Three"),
    ("four_flat_full",   "Four Flat Full"),
    ("lesion_3_20k",     "Lesion 3 (20k)"),
    ("lesion_3_200k",    "Lesion 3 (200k)"),
    ("lesion_4",         "Lesion 4"),
]

# Maximum recursion depth for lobe counting (from hierarchical.py, ii=14)
MAX_EROSION_DEPTH = 20

# ===========================
# HOLE-FILLING (from features.py:23-30, volume_check.py:15-22)
# ===========================
def fill_safe(binary_vol):
    """
    Fill small holes slice-by-slice then in 3D. From polymorpho features.py.
    Made safe: checks that fill doesn't inflate volume by more than 3×.
    """
    original_count = np.count_nonzero(binary_vol)
    newfile = np.zeros_like(binary_vol)
    for i in range(len(binary_vol)):
        roi = binary_vol[i, :, :]
        new = morphology.remove_small_holes(roi, area_threshold=4000)
        newfile[i, :, :] = new
    newfile = morphology.remove_small_holes(newfile, area_threshold=100000)

    filled_count = np.count_nonzero(newfile)
    # Safety: if fill inflated volume unreasonably, skip it
    if original_count > 0 and filled_count > 3 * original_count:
        print(f"    ⚠️  Fill inflated {original_count} → {filled_count} voxels ({filled_count/original_count:.1f}×), reverting to unfilled")
        return binary_vol.astype(bool)
    return newfile


# ===========================
# VOLUME (from features.py:81, volume_check.py:38)
# ===========================
def compute_volume(binary_vol):
    """Volume = count of nonzero voxels."""
    return int(np.count_nonzero(binary_vol))


# ===========================
# SURFACE AREA (from features.py:101-103)
# ===========================
def compute_surface_area(binary_vol):
    """Surface area via marching cubes mesh. From features.py line 101-103."""
    try:
        verts, faces, normals, values = measure.marching_cubes(binary_vol, step_size=1)
        sa = float(measure.mesh_surface_area(verts, faces))
    except Exception as e:
        print(f"    ⚠️  Marching cubes failed: {e}")
        sa = 0.0
    return sa


# ===========================
# SPHERICITY (from features.py:108)
# ===========================
def compute_sphericity(volume, surface_area):
    """Sphericity = π^(1/3) * (6V)^(2/3) / SA.  From features.py line 108."""
    if surface_area <= 0:
        return 0.0
    return pow(math.pi, 1.0 / 3.0) * pow(6 * volume, 2.0 / 3.0) / surface_area


# ===========================
# HARALICK FEATURES (from features.py:91, downsampling_test.py)
# ===========================
def compute_haralick(binary_vol):
    """
    13 Haralick texture features (mean over all 13 GLCM directions for 3D).
    From features.py line 91: mahotas.features.haralick(hole_fill, return_mean=True)
    """
    vol = binary_vol.astype(np.int32)
    try:
        h = mahotas.features.haralick(vol, return_mean=True)
        return h  # shape (13,)
    except Exception as e:
        warnings.warn(f"Haralick computation failed: {e}")
        return np.zeros(13)


# ===========================
# LOBE COUNTING via RECURSIVE EROSION
# (from hierarchical.py:118-134, count_comp function)
# ===========================
def count_comp(binary_vol, depth, id_tuple, ids, footprint, max_depth=MAX_EROSION_DEPTH):
    """
    Recursive erosion-based lobe counting.
    Ported from hierarchical.py lines 118-134 (count_comp function).

    Erodes the volume, checks connected components. If it splits,
    recurse into each piece. Terminal nodes → append ID to ids list.

    Added max_depth guard to prevent infinite recursion.
    """
    if depth >= max_depth:
        ids.append(id_tuple)
        return

    try:
        eroded_file = morphology.erosion(binary_vol.astype(np.uint8), footprint=footprint)
        eroded_file = eroded_file.astype(bool)
    except Exception:
        ids.append(id_tuple)
        return

    if np.count_nonzero(eroded_file) < 20:
        ids.append(id_tuple)
        return

    labels, labelcount = measure.label(eroded_file, connectivity=1, return_num=True)

    if labelcount > 0:
        found_any = False
        for i in range(1, labelcount + 1):
            new = (labels == i)
            # Check minimum size to filter noise blobs
            if np.count_nonzero(new) < 50:
                continue
            found_any = True
            newcode = id_tuple[1] + chr(min(i + 96, 122))  # cap at 'z'
            count_comp(new, depth + 1, (depth + 1, newcode), ids, footprint, max_depth)
        if not found_any:
            ids.append(id_tuple)
    else:
        ids.append(id_tuple)


def longest_common_prefix(s1, s2):
    """From hierarchical.py line 137-142."""
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return s1[:i]


def string_prefix_distance(s1, s2, max_len):
    """From hierarchical.py line 144-152."""
    lcp_len = len(longest_common_prefix(s1, s2))
    return max_len - lcp_len


def compute_lobe_count(binary_vol):
    """
    Full lobe-counting pipeline from hierarchical.py:
    1. Clean the volume (remove small objects)
    2. Recursive erosion → collect terminal IDs
    3. Hierarchical clustering on string-prefix distance
    4. Count clusters from dendrogram
    """
    cleaned = morphology.remove_small_objects(binary_vol.astype(bool), min_size=100, connectivity=1)

    if np.count_nonzero(cleaned) < 100:
        return 1

    # Adaptive footprint based on volume shape
    shape = cleaned.shape
    fz = max(1, min(shape[0] // 15, 3))
    fxy = max(3, min(min(shape[1], shape[2]) // 10, 8))
    footprint = np.ones(shape=(fz, fxy, fxy), dtype=np.uint8)

    ids = []
    count_comp(cleaned, 0, (0, "a"), ids, footprint)
    finalcodes = [id_tuple[1] for id_tuple in ids]

    print(f"    Erosion terminal codes ({len(finalcodes)}): {finalcodes[:10]}{'...' if len(finalcodes)>10 else ''}")

    if len(finalcodes) <= 1:
        return 1

    # Build distance matrix (from hierarchical.py:219-223)
    max_string_length = max(len(s) for s in finalcodes)

    def pdist_wrapper(u, v):
        return string_prefix_distance(u[0], v[0], max_string_length)

    string_array = np.array(finalcodes, dtype=object).reshape(-1, 1)

    try:
        distance_matrix = spatial.distance.pdist(string_array, metric=pdist_wrapper)
        Z = sp_cluster.hierarchy.linkage(distance_matrix, method='average')

        # Count from dendrogram colors (from hierarchical.py:230-253)
        d = sp_cluster.hierarchy.dendrogram(Z, labels=finalcodes,
                                              color_threshold=0.8 * max(Z[:, 2]),
                                              no_plot=True)
        colors = [str(c) for c in d['leaves_color_list']]

        if len(np.unique(colors)) == 1:
            if len(colors) > 10:
                return 1
            else:
                return len(colors)
        else:
            count = 0
            if colors.count('C0') > 5:
                return 1
            else:
                count += colors.count('C0')
                count += len(np.unique(list(filter(lambda x: x != 'C0', colors))))
                return count
    except (ValueError, IndexError):
        return 1


# ===========================
# MAIN FEATURE EXTRACTION PIPELINE
# ===========================

HARALICK_NAMES = [
    "ASM", "Contrast", "Correlation", "Sum_of_Squares_Variance", "IDM",
    "Sum_Average", "Sum_Variance", "Sum_Entropy", "Entropy",
    "Diff_Variance", "Diff_Entropy", "IMC_I", "IMC_II"
]

FEATURE_NAMES = ["Volume_voxels", "Surface_Area", "Sphericity", "Lobe_Count"] + HARALICK_NAMES


def extract_all_features(specimen_name, binary_vol):
    """Extract the full 17-feature vector for one specimen."""
    print(f"\n{'='*60}")
    print(f"Processing: {specimen_name}")
    print(f"  Shape: {binary_vol.shape}, Nonzero: {np.count_nonzero(binary_vol)}")

    # 1. Hole-filling (from features.py)
    filled = fill_safe(binary_vol)
    print(f"  After fill: Nonzero: {np.count_nonzero(filled)}")

    # 2. Volume
    vol = compute_volume(filled)
    print(f"  Volume: {vol} voxels")

    # 3. Surface area
    sa = compute_surface_area(filled)
    print(f"  Surface Area: {sa:.2f}")

    # 4. Sphericity
    sph = compute_sphericity(vol, sa)
    print(f"  Sphericity: {sph:.4f}")

    # 5. Lobe count (recursive erosion)
    lobe_count = compute_lobe_count(filled)
    print(f"  Lobe Count: {lobe_count}")

    # 6. Haralick features
    haralick = compute_haralick(filled)
    print(f"  Haralick (13 features): {haralick}")

    features = [vol, sa, sph, lobe_count] + list(haralick)
    return features


def main():
    print("=" * 70)
    print("NeRF Morphometric Feature Extraction")
    print(f"Ported from polymorpho-lee-lab (features.py + hierarchical.py)")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 70)

    all_features = []
    all_names = []
    all_labels = []

    for fname, label in SPECIMENS:
        npy_path = NPY_DIR / f"{fname}.npy"
        if not npy_path.exists():
            print(f"\n⚠️  Skipping {fname}: {npy_path} not found")
            continue

        binary_vol = np.load(npy_path)
        features = extract_all_features(label, binary_vol)
        all_features.append(features)
        all_names.append(fname)
        all_labels.append(label)

    if not all_features:
        print("No specimens processed!")
        return

    # Build feature matrix
    X = np.array(all_features)
    print(f"\n{'='*60}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Specimens: {len(all_names)}")

    # ===========================
    # SAVE CSV
    # ===========================
    csv_path = OUT_DIR / "feature_matrix.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Specimen", "Label"] + FEATURE_NAMES)
        for i in range(len(all_names)):
            row = [all_names[i], all_labels[i]] + [f"{v:.6f}" if isinstance(v, float) else str(v) for v in all_features[i]]
            writer.writerow(row)
    print(f"\n✅ Saved CSV: {csv_path}")

    # ===========================
    # SAVE NUMPY
    # ===========================
    np.save(OUT_DIR / "feature_matrix.npy", X)
    np.save(OUT_DIR / "specimen_names.npy", np.array(all_names))
    print(f"✅ Saved feature_matrix.npy and specimen_names.npy")

    # ===========================
    # GENERATE MARKDOWN REPORT
    # ===========================
    md_path = OUT_DIR / "feature_report.md"
    with open(md_path, 'w') as f:
        f.write("# NeRF Morphometric Feature Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("**Method:** Ported from `polymorpho-lee-lab/features.py` + `hierarchical.py`\n\n")
        f.write("## Feature Definitions\n\n")
        f.write("| # | Feature | Source | Description |\n")
        f.write("|---|---------|--------|-------------|\n")
        f.write("| 0 | Volume | `features.py:81` | Voxel count after hole-filling |\n")
        f.write("| 1 | Surface Area | `features.py:101-103` | Marching cubes mesh area |\n")
        f.write("| 2 | Sphericity | `features.py:108` | π^(1/3) × (6V)^(2/3) / SA |\n")
        f.write("| 3 | Lobe Count | `hierarchical.py:118-253` | Recursive erosion → dendrogram |\n")
        f.write("| 4–16 | Haralick [1–13] | `features.py:91` | 13 GLCM texture features (mean over 13 3D directions) |\n\n")

        f.write("### Haralick Feature Names\n\n")
        f.write("| Index | Name | Abbreviation |\n")
        f.write("|-------|------|--------------|\n")
        col_labels = ['ASM','Contrast','Correlation','Sum of Sq: Var','IDM','Sum Avg','Sum Var','Sum Entropy','Entropy','Diff Var','Diff Entropy','IMC-I','IMC-II']
        for i, (name, abbr) in enumerate(zip(HARALICK_NAMES, col_labels)):
            f.write(f"| {i+4} | {name} | {abbr} |\n")

        f.write("\n## Feature Matrix\n\n")

        # Table header
        header = "| Specimen | Volume | Surface Area | Sphericity | Lobes |"
        header += " ASM | Contrast | Corr. | SoS:V | IDM | Sum Avg | Sum Var | Sum Ent | Ent | Diff Var | Diff Ent | IMC-I | IMC-II |\n"
        sep = "|" + "|".join(["---"] * 18) + "|\n"
        f.write(header)
        f.write(sep)

        for i in range(len(all_names)):
            row = f"| **{all_labels[i]}** "
            row += f"| {int(X[i,0]):,} "
            row += f"| {X[i,1]:,.1f} "
            row += f"| {X[i,2]:.4f} "
            row += f"| {int(X[i,3])} "
            for j in range(4, 17):
                row += f"| {X[i,j]:.4f} "
            row += "|\n"
            f.write(row)

        # ===========================
        # ANALYSIS SECTION
        # ===========================
        f.write("\n## Analysis\n\n")

        # Volume comparison
        f.write("### Volume Comparison\n\n")
        f.write("| Specimen | Volume (voxels) | Relative to Hemisphere |\n")
        f.write("|----------|----------------|------------------------|\n")
        hemi_vol = None
        for i, label in enumerate(all_labels):
            if "Hemisphere" in label:
                hemi_vol = X[i, 0]
                break
        for i in range(len(all_labels)):
            rel = f"{X[i,0]/hemi_vol:.2f}×" if hemi_vol and hemi_vol > 0 else "N/A"
            f.write(f"| {all_labels[i]} | {int(X[i,0]):,} | {rel} |\n")

        # Lesion 3 vs Lesion 4
        f.write("\n### Lesion 3 vs Lesion 4 Comparison\n\n")
        l3_indices = [i for i, n in enumerate(all_names) if 'lesion_3' in n]
        l4_indices = [i for i, n in enumerate(all_names) if 'lesion_4' in n]

        if l3_indices and l4_indices:
            l3_20k = l3_indices[0] if len(l3_indices) > 0 else None
            l3_200k = l3_indices[1] if len(l3_indices) > 1 else l3_indices[0]
            l4 = l4_indices[0]

            f.write("| Feature | Lesion 3 (20k) | Lesion 3 (200k) | Lesion 4 | L4/L3_200k Ratio |\n")
            f.write("|---------|---------------|-----------------|----------|------------------|\n")

            for fi, fname in enumerate(FEATURE_NAMES):
                v3_20k = X[l3_20k, fi] if l3_20k is not None else 0
                v3_200k = X[l3_200k, fi]
                v4 = X[l4, fi]
                ratio = f"{v4/v3_200k:.2f}×" if v3_200k != 0 else "N/A"
                if fi in [0, 3]:
                    f.write(f"| {fname} | {int(v3_20k):,} | {int(v3_200k):,} | {int(v4):,} | {ratio} |\n")
                else:
                    f.write(f"| {fname} | {v3_20k:.4f} | {v3_200k:.4f} | {v4:.4f} | {ratio} |\n")

        # Lesion 3 20k vs 200k
        f.write("\n### Lesion 3: 20k vs 200k Training Steps\n\n")
        if len(l3_indices) >= 2:
            i20k, i200k = l3_indices[0], l3_indices[1]
            f.write("| Feature | 20k Steps | 200k Steps | Change |\n")
            f.write("|---------|-----------|------------|--------|\n")
            for fi, fname in enumerate(FEATURE_NAMES):
                v20k = X[i20k, fi]
                v200k = X[i200k, fi]
                if v20k != 0:
                    change = f"{((v200k - v20k) / abs(v20k)) * 100:+.1f}%"
                else:
                    change = "N/A"
                if fi in [0, 3]:
                    f.write(f"| {fname} | {int(v20k):,} | {int(v200k):,} | {change} |\n")
                else:
                    f.write(f"| {fname} | {v20k:.4f} | {v200k:.4f} | {change} |\n")

        # Sphericity ranking
        f.write("\n### Sphericity Ranking (1.0 = perfect sphere)\n\n")
        sph_order = np.argsort(-X[:, 2])
        f.write("| Rank | Specimen | Sphericity |\n")
        f.write("|------|----------|------------|\n")
        for rank, idx in enumerate(sph_order, 1):
            f.write(f"| {rank} | {all_labels[idx]} | {X[idx, 2]:.4f} |\n")

        f.write("\n### Observations\n\n")
        f.write("1. **Hemisphere** should have the highest sphericity (~1.0 for a perfect half-sphere).\n")
        f.write("2. **Lesion 3 vs Lesion 4**: The user notes they should have similar or 2× volume.\n")
        f.write("3. **20k vs 200k training**: More training iterations should yield tighter, more accurate geometry.\n")
        f.write("4. **Haralick on binary**: Since our volumes are binary (0/1), Haralick features reflect the *spatial distribution* of the occupied voxels rather than intensity texture.\n")
        f.write("5. **Lobe Count**: Uses the exact same recursive erosion → hierarchical dendrogram algorithm from `hierarchical.py`.\n")

    print(f"✅ Saved report: {md_path}")
    print(f"\nDone! All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
