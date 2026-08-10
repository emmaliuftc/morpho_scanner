import cv2
import numpy as np
import glob
import os
import sys
import json
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d

CALIB_JSON = "captures_0726_clay_checkboard_64_calibrated/calibration_results.json"
NERF_DIR = "captures_0726_nerf_dataset"
INPUT_FOLDER = "captures_0726_clay_checkboard_64"

with open(CALIB_JSON, "r") as f:
    cal_data = json.load(f)

K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float64)
dist_cal = np.array(cal_data["distortion_coefficients"], dtype=np.float64)
C_rot = np.array(cal_data["plate_center_mm"], dtype=np.float64)
normal = np.array(cal_data["plate_normal"], dtype=np.float64)
normal = normal / np.linalg.norm(normal)
step_size_deg = float(cal_data["step_size_deg"])

# Basis setup
ref = np.array([1.0, 0.0, 0.0])
x_col = ref - np.dot(ref, normal) * normal
x_col = x_col / np.linalg.norm(x_col)
y_col = np.cross(normal, x_col)
R_cam = np.column_stack((x_col, y_col, normal))

def get_camera_pose_for_frame(i):
    angle_deg = i * step_size_deg
    angle_rad = np.deg2rad(angle_deg)
    R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
    R_eff = np.dot(R_i, R_cam)
    t_eff = C_rot
    return R_eff, t_eff

print("--- POST-PROCESSING NERF POINT CLOUD & RECONSTRUCTING SURFACE MESH ---")

# 1. Load exported point cloud from NeRF
raw_ply_path = os.path.join(NERF_DIR, "point_cloud.ply")
pcd = o3d.io.read_point_cloud(raw_ply_path)
pts_opengl = np.asarray(pcd.points)
colors_raw = np.asarray(pcd.colors)
N = len(pts_opengl)
print(f"Loaded {N} points from raw NeRF point cloud.")

# 2. Convert from NeRF space to OpenCV World space (mm)
pts_opencv_mm = pts_opengl * 150.0

# 3. Filter points with Visual Hull Silhouettes across 64 frames
print("Filtering point cloud with visual hull masks...")
image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")),
                     key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

valid_counts = np.zeros(N, dtype=int)
W, H = 4608, 2592

for idx, img_path in enumerate(image_paths):
    mask_path = f"captures_0726_clay_checkboard_64_tsdf/masks/mask_{idx:02d}.png"
    if not os.path.exists(mask_path):
        continue
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    R, t = get_camera_pose_for_frame(idx)
    
    pts_cam = np.dot(pts_opencv_mm, R.T) + t.reshape(1, 3)
    zc = pts_cam[:, 2]
    zc_safe = np.where(zc > 1e-5, zc, 1e-5)
    
    u = np.round(K_cal[0, 0] * pts_cam[:, 0] / zc_safe + K_cal[0, 2]).astype(int)
    v = np.round(K_cal[1, 1] * pts_cam[:, 1] / zc_safe + K_cal[1, 2]).astype(int)
    
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (zc > 0)
    inside_mask = np.zeros(N, dtype=bool)
    inside_mask[valid_uv] = mask[v[valid_uv], u[valid_uv]] > 0
    valid_counts += inside_mask.astype(int)

keep_mask = valid_counts >= 15
keep_indices = np.where(keep_mask)[0]
print(f"Kept {len(keep_indices)} / {N} points after visual hull consistency filter.")

pcd_filtered = pcd.select_by_index(keep_indices)
filtered_ply_path = os.path.join(NERF_DIR, "point_cloud_filtered.ply")
o3d.io.write_point_cloud(filtered_ply_path, pcd_filtered)
print(f"Saved filtered PLY to: {filtered_ply_path}")

# 4. Poisson Surface Reconstruction
print("Running Poisson surface reconstruction...")
pcd_filtered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_filtered, depth=8)

densities = np.asarray(densities)
vertices_to_remove = densities < np.percentile(densities, 5)
mesh.remove_vertices_by_mask(vertices_to_remove)

mesh_ply_path = os.path.join(NERF_DIR, "mesh.ply")
o3d.io.write_triangle_mesh(mesh_ply_path, mesh)

vertices_opengl = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
print(f"Mesh created with {len(vertices_opengl)} vertices and {len(triangles)} triangles. Saved to {mesh_ply_path}")

# 5. Project Mesh Wireframe Overlay onto Original Images
print("Projecting NeRF mesh wireframe overlays onto capture images...")
vertices_opencv_mm = vertices_opengl * 150.0

edges = set()
for tri in triangles:
    edges.add(tuple(sorted((tri[0], tri[1]))))
    edges.add(tuple(sorted((tri[1], tri[2]))))
    edges.add(tuple(sorted((tri[2], tri[0]))))

for idx in [0, 16, 32, 48]:
    if idx < len(image_paths):
        img_path = image_paths[idx]
        raw_img = cv2.imread(img_path)
        img = cv2.undistort(raw_img, K_cal, dist_cal, None, K_cal)
        
        R, t = get_camera_pose_for_frame(idx)
        pts_cam = np.dot(vertices_opencv_mm, R.T) + t.reshape(1, 3)
        zc = pts_cam[:, 2]
        zc_safe = np.where(zc > 1e-5, zc, 1e-5)
        
        u = np.round(K_cal[0, 0] * pts_cam[:, 0] / zc_safe + K_cal[0, 2]).astype(int)
        v = np.round(K_cal[1, 1] * pts_cam[:, 1] / zc_safe + K_cal[1, 2]).astype(int)
        
        overlay = img.copy()
        for p1, p2 in edges:
            if (zc[p1] > 0 and zc[p2] > 0 and
                0 <= u[p1] < W and 0 <= v[p1] < H and
                0 <= u[p2] < W and 0 <= v[p2] < H):
                cv2.line(overlay, (u[p1], v[p1]), (u[p2], v[p2]), (0, 255, 255), 2)
                
        blended = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        blended_small = cv2.resize(blended, (1152, 648))
        
        out_proj_path = os.path.join(NERF_DIR, f"mesh_projection_{idx:02d}.jpg")
        cv2.imwrite(out_proj_path, blended_small)
        print(f"Saved projected mesh overlay to {out_proj_path}")

# 6. Render 3D Perspective Preview
print("Rendering 3D perspective view of NeRF mesh...")
mesh_o3d = o3d.io.read_triangle_mesh(mesh_ply_path)
mesh_o3d.compute_vertex_normals()

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=1280, height=960)
vis.add_geometry(mesh_o3d)

ctr = vis.get_view_control()
ctr.set_zoom(0.7)
ctr.set_front([0.5, -0.5, -0.7])
ctr.set_lookat([0, 0, 0])
ctr.set_up([0, 0, 1])

vis.poll_events()
vis.update_renderer()
render_path = os.path.join(NERF_DIR, "nerf_mesh_3d_preview.png")
vis.capture_screen_image(render_path, do_render=True)
vis.destroy_window()
print(f"Saved 3D perspective render to {render_path}")

print("\n✅ NERF POST-PROCESSING & MESHING FINISHED SUCCESSFULLY!")
