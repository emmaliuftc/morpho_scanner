import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pycolmap
import sys

# Paths
workspace = Path("COLMAP/workspace")
sparse_path = workspace / "sparse_unmasked_triangulated"
image_dir = workspace / "raw_images" # Raw images showing turntable context
output_dir = workspace / "visualizations" / "z_axle_projections"
output_dir.mkdir(exist_ok=True, parents=True)

# Derived center point and axis normal
derived_plate_center_3d = np.array([0.8568, 1.5746, 2.7133])
normal = np.array([0.20758226, 0.71598242, 0.66654241]) # rotation axis vector

# Load reconstruction to get camera parameters and image poses
if not sparse_path.exists():
    print(f"Error: Reconstruction folder not found at {sparse_path}")
    sys.exit(1)
    
recon = pycolmap.Reconstruction(str(sparse_path))
camera = recon.cameras[1]
f, cx, cy, k = camera.params

print(f"Projecting Z-axle line from center: {derived_plate_center_3d} along axis: {normal}")

# Project and draw on each raw image
for img_id, img in sorted(recon.images.items(), key=lambda x: x[1].name):
    name = img.name
    image_path = image_dir / name
    if not image_path.exists():
        continue
        
    img_data = plt.imread(str(image_path))
    
    R = img.cam_from_world().rotation.matrix()
    T = img.cam_from_world().translation
    
    # Generate points along the 3D Z-axle line
    # s ranges from -2.0 (above plate) to 0.5 (below plate)
    s_vals = np.linspace(-2.0, 0.5, 100)
    line_pts_3d = np.array([derived_plate_center_3d + s * normal for s in s_vals])
    
    # Project all 3D line points to 2D image plane
    px_coords = []
    py_coords = []
    
    for pt_3d in line_pts_3d:
        point_cam = R @ pt_3d + T
        x_n = point_cam[0] / point_cam[2]
        y_n = point_cam[1] / point_cam[2]
        r2 = x_n**2 + y_n**2
        distortion = 1.0 + k * r2
        px = f * distortion * x_n + cx
        py = f * distortion * y_n + cy
        px_coords.append(px)
        py_coords.append(py)
        
    # Find projection of the center point itself
    center_cam = R @ derived_plate_center_3d + T
    cx_n = center_cam[0] / center_cam[2]
    cy_n = center_cam[1] / center_cam[2]
    cr2 = cx_n**2 + cy_n**2
    cdist = 1.0 + k * cr2
    cpx = f * cdist * cx_n + cx
    cpy = f * cdist * cy_n + cy
    
    # Create Matplotlib Figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_data)
    
    # Draw Z-axle line (dashed yellow line)
    ax.plot(px_coords, py_coords, color='cyan', linestyle='--', linewidth=3, label='Rotation Z-Axle')
    
    # Draw center dot and target crosshair
    ax.plot(cpx, cpy, 'ro', markersize=5) # Red center dot
    circle = plt.Circle((cpx, cpy), 30, color='lime', fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.plot([cpx - 50, cpx + 50], [cpy, cpy], color='lime', linewidth=2)
    ax.plot([cpx, cpx], [cpy - 50, cpy + 50], color='lime', linewidth=2)
    
    # Labels
    ax.text(cpx + 40, cpy - 40, f"Plate Center ({int(cpx)}, {int(cpy)})", 
            color='lime', fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    
    # Draw a line vector indicator for direction
    top_pt_cam = R @ (derived_plate_center_3d - 1.0 * normal) + T
    tx_n = top_pt_cam[0] / top_pt_cam[2]
    ty_n = top_pt_cam[1] / top_pt_cam[2]
    tr2 = tx_n**2 + ty_n**2
    tdist = 1.0 + k * tr2
    tpx = f * tdist * tx_n + cx
    tpy = f * tdist * ty_n + cy
    
    ax.annotate("", xy=(tpx, tpy), xytext=(cpx, cpy),
                arrowprops=dict(arrowstyle="->", color="cyan", lw=3))
    ax.text(tpx + 20, tpy - 20, "Z-Axle Normal (n)", color='cyan', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    
    ax.axis('off')
    
    # Save output
    out_path = output_dir / f"z_axle_{name}"
    plt.savefig(str(out_path), bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    
    print(f"  Saved Z-axle visualization: {out_path.name}")

print(f"\nAll Z-axle visualizations saved successfully to {output_dir}")
