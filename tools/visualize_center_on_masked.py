import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pycolmap
import sys

# Paths
workspace = Path("COLMAP/workspace")
sparse_path = workspace / "sparse_unmasked_triangulated"
image_dir = workspace / "images" # Masked images
output_dir = workspace / "visualizations" / "plate_center_masked"
output_dir.mkdir(exist_ok=True, parents=True)

# Derived center point
derived_plate_center_3d = np.array([0.8568, 1.5746, 2.7133])

# Load reconstruction to get camera parameters and image poses
if not sparse_path.exists():
    print(f"Error: Reconstruction folder not found at {sparse_path}")
    sys.exit(1)
    
recon = pycolmap.Reconstruction(str(sparse_path))
camera = recon.cameras[1]
f, cx, cy, k = camera.params

print(f"Projecting derived center: {derived_plate_center_3d}")

# Project and draw on each masked image
for img_id, img in sorted(recon.images.items(), key=lambda x: x[1].name):
    name = img.name
    image_path = image_dir / name
    if not image_path.exists():
        continue
        
    img_data = plt.imread(str(image_path))
    
    R = img.cam_from_world().rotation.matrix()
    T = img.cam_from_world().translation
    
    # Project derived center
    point_cam = R @ derived_plate_center_3d + T
    x_n = point_cam[0] / point_cam[2]
    y_n = point_cam[1] / point_cam[2]
    r2 = x_n**2 + y_n**2
    distortion = 1.0 + k * r2
    px = f * distortion * x_n + cx
    py = f * distortion * y_n + cy
    
    # Create Matplotlib Figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_data)
    
    # Draw crosshair and circle
    ax.plot(px, py, 'ro', markersize=4) # Red center dot
    circle = plt.Circle((px, py), 30, color='lime', fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.plot([px - 50, px + 50], [py, py], color='lime', linewidth=2)
    ax.plot([px, px], [py - 50, py + 50], color='lime', linewidth=2)
    
    # Label
    ax.text(px + 40, py - 40, f"Derived Center ({int(px)}, {int(py)})", 
            color='lime', fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    
    ax.axis('off')
    
    # Save output
    out_path = output_dir / f"masked_center_{name}"
    plt.savefig(str(out_path), bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    
    print(f"  Saved masked center visualization: {out_path.name} at pixel ({int(px)}, {int(py)})")

print(f"\nAll masked center visualizations saved successfully to {output_dir}")
