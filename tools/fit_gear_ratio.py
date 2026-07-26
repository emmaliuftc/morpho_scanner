import pycolmap
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
import sys

# Paths
workspace = Path("COLMAP/workspace")
recon_path = workspace / "sparse" / "1" # Reference actual reconstruction

if not recon_path.exists():
    print(f"Error: Reference reconstruction not found at {recon_path}")
    sys.exit(1)

# Load reconstruction
recon = pycolmap.Reconstruction(str(recon_path))

# 1. Fit normal and circle center to camera centers
centers = []
for img_id, img in recon.images.items():
    centers.append(img.projection_center())
centers = np.array(centers)
n_images = len(centers)

center_mean = np.mean(centers, axis=0)
centered_centers = centers - center_mean
U, S, Vt = np.linalg.svd(centered_centers)
normal = Vt[2, :]
if normal[2] < 0:
    normal = -normal

u_axis = Vt[0, :]
v_axis = Vt[1, :]

# Fit circle
A = np.column_stack([2 * (centered_centers @ u_axis), 2 * (centered_centers @ v_axis), np.ones(n_images)])
B = (centered_centers @ u_axis)**2 + (centered_centers @ v_axis)**2
uc, vc, C_val = np.linalg.lstsq(A, B, rcond=None)[0]
circle_center_3d = center_mean + uc * u_axis + vc * v_axis

# Get angles for each registered image
measured_angles = []
image_indices = []

for img_id, img in sorted(recon.images.items(), key=lambda x: x[1].name):
    name = img.name
    idx = int(name.split("_")[1].split(".")[0])
    
    # Compute center relative to circle center
    pos = img.projection_center() - circle_center_3d
    u = np.dot(pos, u_axis)
    v = np.dot(pos, v_axis)
    
    angle = np.arctan2(v, u)
    measured_angles.append(angle)
    image_indices.append(idx)

measured_angles = np.array(measured_angles)
image_indices = np.array(image_indices)

# Sort by index
sort_idx = np.argsort(image_indices)
sorted_indices = image_indices[sort_idx]
sorted_angles = measured_angles[sort_idx]

# Unwrap angles
unwrapped_angles = np.unwrap(sorted_angles)

# Fit line: angle = slope * index + intercept
slope, intercept = np.polyfit(sorted_indices, unwrapped_angles, 1)

print("Fitting turntable rotation angle against image index:")
print(f"  Fitted slope (radians per step): {slope:.6f}")
print(f"  Fitted step size (degrees):     {np.degrees(slope):.4f}°")
print(f"  Expected step size (degrees):   11.2500°")
print(f"  Fitted intercept (radians):     {intercept:.6f}")
print(f"  Fitted intercept (degrees):     {np.degrees(intercept):.4f}°")

# Calculate residuals
residuals = unwrapped_angles - (slope * sorted_indices + intercept)
max_residual = np.max(np.abs(residuals))
print(f"  Max residual error:             {np.degrees(max_residual):.4f}°")
