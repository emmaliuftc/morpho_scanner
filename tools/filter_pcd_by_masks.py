import numpy as np
import open3d as o3d
import cv2
import os
from scipy.spatial.transform import Rotation as Rot

# 1. Load calibration data
K_cal = np.array([
    [3566.2276687812728, 0.0, 2304.0],
    [0.0, 3566.2276687812728, 1296.0],
    [0.0, 0.0, 1.0]
])
C_rot = np.array([-2.9488, -9.3668, 173.3226])
normal = np.array([0.0107, 0.6351, 0.7724])
normal_up = -normal
step_size_deg = 11.5529

# Construct R_cam
ref = np.array([1.0, 0.0, 0.0])
x_col = ref - np.dot(ref, normal_up) * normal_up
x_col = x_col / np.linalg.norm(x_col)
y_col = np.cross(normal_up, x_col)
R_cam = np.column_stack((x_col, y_col, normal_up))

# Load point cloud
pcd_path = "captures_7-25_nerf_dataset/point_cloud.ply"
pcd = o3d.io.read_point_cloud(pcd_path)
pts_opengl = np.asarray(pcd.points)

# Convert from OpenGL (NeRF) back to OpenCV world space
pts = pts_opengl.copy()
pts[:, 1] = -pts[:, 1]
pts[:, 2] = -pts[:, 2]
N = len(pts)

print(f"Loaded {N} points from point cloud.")

# Load original masks
masks = []
num_images = 32
for i in range(num_images):
    mask_path = f"captures_7-25_nerf_dataset/masks/mask_{i:02d}.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    masks.append(mask)

# Vectorized point cloud visual hull check
valid_counts = np.zeros(N, dtype=int)
W, H = 4608, 2592

for i in range(num_images):
    angle_deg = i * step_size_deg
    angle_rad = np.deg2rad(angle_deg)
    R_i = Rot.from_rotvec(angle_rad * normal_up).as_matrix()
    R_eff = np.dot(R_i, R_cam)
    
    # Project all points in parallel
    # P_cam = pts @ R_eff.T + C_rot
    pts_cam = np.dot(pts, R_eff.T) + C_rot.reshape(1, 3)
    zc = pts_cam[:, 2]
    
    # Prevent divide-by-zero
    zc_safe = np.where(zc > 1e-5, zc, 1e-5)
    
    u = np.round(K_cal[0, 0] * pts_cam[:, 0] / zc_safe + K_cal[0, 2]).astype(int)
    v = np.round(K_cal[1, 1] * pts_cam[:, 1] / zc_safe + K_cal[1, 2]).astype(int)
    
    valid_mask = (zc > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    inside_mask = np.zeros(N, dtype=bool)
    if np.any(valid_mask):
        inside_mask[valid_mask] = masks[i][v[valid_mask], u[valid_mask]] == 255
        
    valid_counts += inside_mask.astype(int)

# Keep points that land inside the mask in at least 15 out of 32 views
keep_mask = valid_counts >= 15
keep_indices = np.where(keep_mask)[0]

# Save filtered point cloud
filtered_pcd = pcd.select_by_index(keep_indices)
o3d.io.write_point_cloud("captures_7-25_nerf_dataset/point_cloud_filtered.ply", filtered_pcd)
print(f"Kept {len(keep_indices)} points. Saved to captures_7-25_nerf_dataset/point_cloud_filtered.ply")
