#!/usr/bin/env python3
"""
Generate high-resolution visual assets for Haralick morphological analysis (docs/haralick.md and Slide 27).
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

os.makedirs("docs/images/haralick", exist_ok=True)

# -------------------------------------------------------------
# 1. Figure 1: Turntable Real Photo Comparison
# -------------------------------------------------------------
print("Generating Figure 1: Turntable Real Photo Comparison...")
img3 = cv2.imread("captures_8-16_lesion_3/capture_0.jpg")
img4 = cv2.imread("captures_8-16_lesion_4/capture_0.jpg")

# Crops around the lesions
# Lesion 3: x=1800, y=844, w=1052, h=692
# Lesion 4: x=1520, y=800, w=1344, h=856
crop3 = img3[700:1700, 1600:3000]
crop4 = img4[700:1700, 1400:3000]

# Resize crops to identical dimensions (1000x1400)
crop3 = cv2.resize(crop3, (1200, 800))
crop4 = cv2.resize(crop4, (1200, 800))

# Convert BGR to RGB
crop3 = cv2.cvtColor(crop3, cv2.COLOR_BGR2RGB)
crop4 = cv2.cvtColor(crop4, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0f172a')

# Panel A: Lesion 3
axes[0].imshow(crop3)
axes[0].set_title("Lesion 3 (Baseline Specimen $V_0$)\nPhysical Turntable Capture (Step 0)", 
                  color='#38bdf8', fontsize=15, fontweight='bold', pad=12)
axes[0].axis('off')
text_box3 = (
    "• Intended Baseline Volume ($V_0$)\n"
    "• Compact, 2-lobed morphology\n"
    "• High Sphericity: 0.240\n"
    "• Surface Area: 26,630 px²"
)
axes[0].text(0.04, 0.06, text_box3, transform=axes[0].transAxes,
             fontsize=12, color='white', family='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#1e293b', edgecolor='#38bdf8', alpha=0.85, lw=1.5))

# Panel B: Lesion 4
axes[1].imshow(crop4)
axes[1].set_title("Lesion 4 (Intentionally Molded with $2.0\\times$ Clay Volume)\nPhysical Turntable Capture (Step 0)", 
                  color='#f43f5e', fontsize=15, fontweight='bold', pad=12)
axes[1].axis('off')
text_box4 = (
    "• Target: Double Clay Volume ($2.0\\times V_0$)\n"
    "• Irregular, 3-lobed spreading morphology\n"
    "• Lower Sphericity: 0.145 (Flattened)\n"
    "• Surface Area: 68,361 px² (2.57× spreading)"
)
axes[1].text(0.04, 0.06, text_box4, transform=axes[1].transAxes,
             fontsize=12, color='white', family='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#1e293b', edgecolor='#f43f5e', alpha=0.85, lw=1.5))

plt.suptitle("Physical Turntable Ground Truth: Lesion 3 vs Lesion 4\nDemonstrating Intentional Mass Doubling Combined with Morphological Deformation",
             color='white', fontsize=17, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig("docs/images/haralick/lesion3_vs_lesion4_turntable.png", dpi=200, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("Saved docs/images/haralick/lesion3_vs_lesion4_turntable.png")


# -------------------------------------------------------------
# 2. Figure 2: 3D Solid Voxel Meshes Comparison
# -------------------------------------------------------------
print("Generating Figure 2: 3D Solid Voxel Meshes Comparison...")
l3 = np.load("haralick/npy/lesion_3_200k.npy")
l4 = np.load("haralick/npy/lesion_4.npy")

# Marching cubes
v3, f3, _, _ = measure.marching_cubes(l3, step_size=2)
v4, f4, _, _ = measure.marching_cubes(l4, step_size=2)

fig = plt.figure(figsize=(16, 8), facecolor='#0f172a')

# Subplot 1: Lesion 3 3D Mesh
ax1 = fig.add_subplot(1, 2, 1, projection='3d', facecolor='#0f172a')
mesh3 = Poly3DCollection(v3[f3], alpha=0.85, edgecolor='#0284c7', linewidth=0.2)
mesh3.set_facecolor('#38bdf8')
ax1.add_collection3d(mesh3)
ax1.set_xlim(0, max(l3.shape))
ax1.set_ylim(0, max(l3.shape))
ax1.set_zlim(0, max(l3.shape))
ax1.view_init(elev=28, azim=-55)
ax1.set_title("Lesion 3 (200k steps)\nSolid Extruded Volume: 48,022 voxels", color='#38bdf8', fontsize=14, fontweight='bold', pad=10)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.tick_params(colors='#94a3b8')

# Subplot 2: Lesion 4 3D Mesh
ax2 = fig.add_subplot(1, 2, 2, projection='3d', facecolor='#0f172a')
mesh4 = Poly3DCollection(v4[f4], alpha=0.85, edgecolor='#be123c', linewidth=0.2)
mesh4.set_facecolor('#fb7185')
ax2.add_collection3d(mesh4)
ax2.set_xlim(0, max(l4.shape))
ax2.set_ylim(0, max(l4.shape))
ax2.set_zlim(0, max(l4.shape))
ax2.view_init(elev=28, azim=-55)
ax2.set_title("Lesion 4\nSolid Extruded Volume: 92,553 voxels (1.93× Doubled)", color='#fb7185', fontsize=14, fontweight='bold', pad=10)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.tick_params(colors='#94a3b8')

plt.suptitle("3D Reconstructed Solid Volumes (NeRF + 2.5D Table Extrusion)\nAccurately Capturing the 1.93× Mass Doubling Despite Asymmetric Morphology",
             color='white', fontsize=17, fontweight='bold', y=0.96)
plt.tight_layout()
fig.savefig("docs/images/haralick/lesion3_vs_lesion4_3d_voxels.png", dpi=200, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("Saved docs/images/haralick/lesion3_vs_lesion4_3d_voxels.png")


# -------------------------------------------------------------
# 3. Figure 3: Morphological Parameter Comparison (Bar / Metric Chart)
# -------------------------------------------------------------
print("Generating Figure 3: Parameter Comparison Chart...")
features = ['Volume\n(voxels)', 'Surface Area\n(px²)', 'Sphericity\n(compactness)', 'Lobe Count\n(# lobes)', 'Haralick ASM\n(Energy)', 'Haralick Corr.\n(Correlation)']
l3_vals = [48022, 26629.8, 0.2399, 2, 0.7149, 0.7834]
l4_vals = [92553, 68361.3, 0.1447, 3, 0.9872, 0.7467]
ratios = [l4/l3 for l3, l4 in zip(l3_vals, l4_vals)]

fig, (ax_bar, ax_ratio) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor='#0f172a')

# Normalized bar chart
x = np.arange(len(features))
width = 0.35

# Normalize each feature to max=1.0 for visual balance
l3_norm = [v / max(v1, v2) for v, v1, v2 in zip(l3_vals, l3_vals, l4_vals)]
l4_norm = [v / max(v1, v2) for v, v1, v2 in zip(l4_vals, l3_vals, l4_vals)]

rects1 = ax_bar.bar(x - width/2, l3_norm, width, label='Lesion 3 (Baseline)', color='#38bdf8', edgecolor='white', lw=1)
rects2 = ax_bar.bar(x + width/2, l4_norm, width, label='Lesion 4 (Double Volume)', color='#f43f5e', edgecolor='white', lw=1)

ax_bar.set_title("Normalized Morphological Metrics (Lesion 3 vs 4)", color='white', fontsize=14, fontweight='bold', pad=10)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(features, color='#e2e8f0', fontsize=11, fontweight='bold')
ax_bar.set_facecolor('#1e293b')
ax_bar.tick_params(colors='#94a3b8')
ax_bar.set_ylim(0, 1.25)
ax_bar.legend(loc='upper right', facecolor='#0f172a', edgecolor='#94a3b8', labelcolor='white')
ax_bar.grid(axis='y', linestyle='--', alpha=0.3, color='#94a3b8')

for i, (val3, val4) in enumerate(zip(l3_vals, l4_vals)):
    ax_bar.text(i - width/2, l3_norm[i] + 0.03, f"{val3:,.0f}" if val3 > 10 else f"{val3:.3f}", 
                ha='center', color='#38bdf8', fontsize=9.5, fontweight='bold')
    ax_bar.text(i + width/2, l4_norm[i] + 0.03, f"{val4:,.0f}" if val4 > 10 else f"{val4:.3f}", 
                ha='center', color='#f43f5e', fontsize=9.5, fontweight='bold')

# Ratio Bar Chart (Highlighting the 1.93x Volume doubling)
colors = ['#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#6366f1', '#14b8a6']
bars = ax_ratio.bar(features, ratios, color=colors, edgecolor='white', width=0.55, lw=1)
ax_ratio.axhline(1.0, color='#94a3b8', linestyle='--', lw=1.5, label='Baseline 1.0×')
ax_ratio.axhline(2.0, color='#22c55e', linestyle=':', lw=2, label='Target Double (2.0×)')

ax_ratio.set_title("Relative Change Ratio (Lesion 4 / Lesion 3)\nProving Physical Volume Doubling (1.93×)", 
                   color='white', fontsize=14, fontweight='bold', pad=10)
ax_ratio.set_xticks(range(len(features)))
ax_ratio.set_xticklabels(features, color='#e2e8f0', fontsize=11, fontweight='bold')
ax_ratio.set_facecolor('#1e293b')
ax_ratio.tick_params(colors='#94a3b8')
ax_ratio.set_ylim(0, 3.0)
ax_ratio.grid(axis='y', linestyle='--', alpha=0.3, color='#94a3b8')
ax_ratio.legend(loc='upper right', facecolor='#0f172a', edgecolor='#94a3b8', labelcolor='white')

for bar, r in zip(bars, ratios):
    yval = bar.get_height()
    ax_ratio.text(bar.get_x() + bar.get_width()/2.0, yval + 0.07, f"{r:.2f}×", 
                  ha='center', va='bottom', color='white', fontsize=12, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f172a', edgecolor=bar.get_facecolor(), lw=1.5))

plt.suptitle("Quantitative Feature Analysis: Decoupling True Volume from Deforming Morphology",
             color='white', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig("docs/images/haralick/morphological_parameter_radar.png", dpi=200, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("Saved docs/images/haralick/morphological_parameter_radar.png")


# -------------------------------------------------------------
# 4. Figure 4: Master Slide-Ready Graphic for Slide 27
# -------------------------------------------------------------
print("Generating Figure 4: Master Slide-Ready Graphic for Slide 27...")
fig = plt.figure(figsize=(18, 10), facecolor='#0b1329')

# Grid layout: Top 4 panels (Capture 3, Capture 4, Mesh 3, Mesh 4), Bottom: Table
gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.2], hspace=0.32, wspace=0.22)

# Top Left: Capture Lesion 3
ax_c3 = fig.add_subplot(gs[0, 0])
ax_c3.imshow(crop3)
ax_c3.set_title("Lesion 3 Real Photo ($V_0$)", color='#38bdf8', fontsize=13, fontweight='bold')
ax_c3.axis('off')

# Top 2nd: Mesh Lesion 3
ax_m3 = fig.add_subplot(gs[0, 1], projection='3d', facecolor='#0b1329')
m3 = Poly3DCollection(v3[f3], alpha=0.85, edgecolor='#0284c7', linewidth=0.2)
m3.set_facecolor('#38bdf8')
ax_m3.add_collection3d(m3)
ax_m3.set_xlim(0, max(l3.shape))
ax_m3.set_ylim(0, max(l3.shape))
ax_m3.set_zlim(0, max(l3.shape))
ax_m3.view_init(elev=28, azim=-55)
ax_m3.set_title("Lesion 3 Solid NeRF\n48,022 voxels", color='#38bdf8', fontsize=13, fontweight='bold')
ax_m3.xaxis.pane.fill = False
ax_m3.yaxis.pane.fill = False
ax_m3.zaxis.pane.fill = False
ax_m3.tick_params(colors='#64748b')

# Top 3rd: Capture Lesion 4
ax_c4 = fig.add_subplot(gs[0, 2])
ax_c4.imshow(crop4)
ax_c4.set_title("Lesion 4 Real Photo ($2.0\\times V_0$)", color='#f43f5e', fontsize=13, fontweight='bold')
ax_c4.axis('off')

# Top Right: Mesh Lesion 4
ax_m4 = fig.add_subplot(gs[0, 3], projection='3d', facecolor='#0b1329')
m4 = Poly3DCollection(v4[f4], alpha=0.85, edgecolor='#be123c', linewidth=0.2)
m4.set_facecolor('#fb7185')
ax_m4.add_collection3d(m4)
ax_m4.set_xlim(0, max(l4.shape))
ax_m4.set_ylim(0, max(l4.shape))
ax_m4.set_zlim(0, max(l4.shape))
ax_m4.view_init(elev=28, azim=-55)
ax_m4.set_title("Lesion 4 Solid NeRF\n92,553 voxels (1.93×)", color='#fb7185', fontsize=13, fontweight='bold')
ax_m4.xaxis.pane.fill = False
ax_m4.yaxis.pane.fill = False
ax_m4.zaxis.pane.fill = False
ax_m4.tick_params(colors='#64748b')

# Bottom: Full Table
ax_table = fig.add_subplot(gs[1, :])
ax_table.axis('off')

table_data = [
    ["Parameter / Morphological Metric", "Lesion 3 (Baseline)", "Lesion 4 (Double Volume)", "Ratio (L4 / L3)", "Clinical / Algorithmic Significance"],
    ["Volume (Solid Infill Voxels)", "48,022 voxels", "92,553 voxels", "1.93× (+92.7%)", "Catches intentional 2.0× mass doubling despite shape distortion"],
    ["Surface Area (Marching Cubes)", "26,629.8 px²", "68,361.3 px²", "2.57× (+156.7%)", "Spreading peripheral margins; non-linear surface expansion"],
    ["Sphericity (Compactness)", "0.2399", "0.1447", "0.60× (-39.7%)", "Severe flattening & irregular elongation (loss of sphericity)"],
    ["Lobe Count (Hierarchical Erosion)", "2 lobes", "3 lobes", "1.50× (+50.0%)", "Multi-lobed invasion pattern detected via dendrogram analysis"],
    ["Haralick 1: Angular Second Moment (ASM)", "0.7149", "0.9872", "1.38× (+38.1%)", "Higher internal textural homogeneity of solid infill"],
    ["Haralick 2: Contrast", "0.0513", "0.0026", "0.05× (-95.0%)", "Minimal boundary noise / smooth interior density"],
    ["Haralick 3: Correlation", "0.7834", "0.7467", "0.95× (-4.7%)", "High spatial continuity across 13 3D GLCM directions"],
    ["Haralick 5: Inverse Diff. Moment (IDM)", "0.9744", "0.9987", "1.02× (+2.5%)", "Tight local voxel coherence and solid core integrity"]
]

col_widths = [0.26, 0.16, 0.18, 0.14, 0.26]
t = ax_table.table(cellText=table_data, colWidths=col_widths, loc='center', cellLoc='center')
t.auto_set_font_size(False)
t.set_fontsize(11)
t.scale(1, 1.65)

# Styling table cells
for (row, col), cell in t.get_celld().items():
    cell.set_edgecolor('#334155')
    if row == 0:
        cell.set_facecolor('#1e293b')
        cell.set_text_props(color='#38bdf8', fontweight='bold')
    elif row == 1: # Volume row highlighted
        cell.set_facecolor('#064e3b')
        cell.set_text_props(color='#34d399', fontweight='bold')
    else:
        cell.set_facecolor('#0f172a' if row % 2 == 0 else '#1e293b')
        cell.set_text_props(color='white')
        if col == 3:
            cell.set_text_props(color='#fbbf24', fontweight='bold')

plt.suptitle("Validation of 3D Morphometric Pipeline on Deforming Clinical Clay Tumors\nLesion 4 Proves Accurate 1.93× Volumetric Tracking Under Severe Morphological Alteration",
             color='white', fontsize=17, fontweight='bold', y=0.97)

fig.savefig("docs/images/haralick/haralick_slide27_table_graphic.png", dpi=200, bbox_inches='tight', facecolor='#0b1329')
plt.close()
print("Saved docs/images/haralick/haralick_slide27_table_graphic.png")

print("All visuals generated successfully!")
