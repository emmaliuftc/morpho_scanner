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
pcd_path = "captures_7-25_nerf_dataset/point_cloud_filtered.ply"
pcd = o3d.io.read_point_cloud(pcd_path)

# Run Poisson reconstruction
print("Running Poisson surface reconstruction...")
# Estimate normals if needed (our PLY already has them, but let's orient them correctly)
pcd.estimate_normals()
pcd.orient_normals_towards_camera_location(camera_location=np.zeros(3))

# Run reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

# Clean low density regions to get a tight mesh
densities = np.asarray(densities)
vertices_to_remove = densities < np.percentile(densities, 5)
mesh.remove_vertices_by_mask(vertices_to_remove)

# Simplify mesh if needed
mesh = mesh.simplify_quadric_decimation(15000)

mesh_path = "captures_7-25_nerf_dataset/mesh.ply"
o3d.io.write_triangle_mesh(mesh_path, mesh)
print(f"Mesh created with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
print(f"Saved to {mesh_path}")

# Load original images and project mesh
W, H = 4608, 2592
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

for idx in [0, 8, 16, 24]:
    img_path = f"captures_7-25_lob_with_checkbox/capture_{idx}.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load original image: {img_path}")
        continue
    
    # Orbit angle for camera
    angle_deg = idx * step_size_deg
    angle_rad = np.deg2rad(angle_deg)
    R_i = Rot.from_rotvec(angle_rad * normal_up).as_matrix()
    R_eff = np.dot(R_i, R_cam)
    
    # Project vertices
    pts_cam = np.dot(vertices, R_eff.T) + C_rot.reshape(1, 3)
    zc = pts_cam[:, 2]
    zc_safe = np.where(zc > 1e-5, zc, 1e-5)
    
    u = np.round(K_cal[0, 0] * pts_cam[:, 0] / zc_safe + K_cal[0, 2]).astype(int)
    v = np.round(K_cal[1, 1] * pts_cam[:, 1] / zc_safe + K_cal[1, 2]).astype(int)
    
    # Project triangles
    # We will draw the projected triangles as a translucent overlay
    overlay = img.copy()
    
    # Extract unique edges to draw
    edges = set()
    for tri in triangles:
        edges.add(tuple(sorted((tri[0], tri[1]))))
        edges.add(tuple(sorted((tri[1], tri[2]))))
        edges.add(tuple(sorted((tri[2], tri[0]))))
        
    for p1, p2 in edges:
        # Check if both vertices project inside screen
        if (zc[p1] > 0 and zc[p2] > 0 and
            0 <= u[p1] < W and 0 <= v[p1] < H and
            0 <= u[p2] < W and 0 <= v[p2] < H):
            cv2.line(overlay, (u[p1], v[p1]), (u[p2], v[p2]), (0, 255, 255), 1) # Cyan wireframe
            
    # Blend overlay with original image
    alpha = 0.5
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Downscale for easier previewing/loading
    blended_small = cv2.resize(blended, (1152, 648))
    
    # Save projected image
    out_path = f"captures_7-25_nerf_dataset/mesh_projection_{idx:02d}.jpg"
    cv2.imwrite(out_path, blended_small)
    print(f"Saved projected mesh overlay to {out_path}")
