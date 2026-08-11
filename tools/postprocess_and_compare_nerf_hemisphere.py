import sys
sys.path.append(".")
import os
import glob
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d

CALIB_JSON = "captures_0809_hemisphere_calibrated/calibration_results.json"
INPUT_FOLDER = "captures_0809_hemisphere_calibrated"
NERF_DIR = "captures_0809_hemisphere_nerf_dataset"
TSDF_OUT_DIR = "captures_0809_hemisphere_tsdf"
STL_PATH = "3d_files/hemisphere_5cm.stl"
DOCS_REPORT_PATH = "docs/hemisphere_comparison_0809.md"

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

image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")),
                     key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[1]))

print("--- POST-PROCESSING NERF POINT CLOUD FOR HEMISPHERE ---")
raw_ply_path = os.path.join(NERF_DIR, "point_cloud.ply")
pcd = o3d.io.read_point_cloud(raw_ply_path)
pts_opengl = np.asarray(pcd.points)

N = len(pts_opengl)
print(f"Loaded {N} points from raw NeRF point cloud.")

# 1. Exact coordinate mapping: P_world = P_nerf * 150.0
pts_opencv_mm = pts_opengl * 150.0

# 2. Visual Hull Consistency Filter across 64 frames
print("Filtering point cloud with visual hull masks...")
masks_dir = os.path.join(INPUT_FOLDER, "masks")
valid_counts = np.zeros(N, dtype=int)
W, H = 4608, 2592

for idx, img_p in enumerate(image_paths):
    mask_p = os.path.join(masks_dir, f"mask_{idx:02d}.png")
    mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
    
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

pts_dist_r = np.linalg.norm(pts_opencv_mm[:, :2], axis=1)
pts_z = pts_opencv_mm[:, 2]

spatial_mask = (pts_dist_r <= 35.0) & (pts_z >= -5.0) & (pts_z <= 35.0)
keep_mask = (valid_counts >= 5) & spatial_mask
keep_indices = np.where(keep_mask)[0]
print(f"Kept {len(keep_indices)} / {N} points ({len(keep_indices)/N*100:.2f}% inliers).")

pcd_filtered = pcd.select_by_index(keep_indices)
filtered_ply_path = os.path.join(NERF_DIR, "point_cloud_filtered.ply")
o3d.io.write_point_cloud(filtered_ply_path, pcd_filtered)

# 3. Poisson Surface Reconstruction
print("Running Poisson surface reconstruction...")
pcd_filtered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_filtered, depth=8)

densities = np.asarray(densities)
vertices_to_remove = densities < np.percentile(densities, 5)
mesh.remove_vertices_by_mask(vertices_to_remove)

mesh_verts = np.asarray(mesh.vertices)
mesh_verts_opencv = mesh_verts * 150.0
mesh_r = np.linalg.norm(mesh_verts_opencv[:, :2], axis=1)
boundary_mask_mesh = (mesh_r > 26.5) | (mesh_verts_opencv[:, 2] < 0.0)
mesh.remove_vertices_by_mask(boundary_mask_mesh)

mesh_ply_path = os.path.join(NERF_DIR, "mesh.ply")
o3d.io.write_triangle_mesh(mesh_ply_path, mesh)

vertices_opengl = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
print(f"Mesh created with {len(vertices_opengl)} vertices and {len(triangles)} triangles.")

vertices_opencv_mm = vertices_opengl * 150.0

# 4. Project Mesh Wireframe Overlay
print("Projecting NeRF mesh wireframe overlays onto capture images...")
edges = set()
for tri in triangles:
    edges.add(tuple(sorted((tri[0], tri[1]))))
    edges.add(tuple(sorted((tri[1], tri[2]))))
    edges.add(tuple(sorted((tri[2], tri[0]))))

for idx in [0, 16, 32, 48]:
    if idx < len(image_paths):
        img_p = image_paths[idx]
        raw_img = cv2.imread(img_p)
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
                
        vis_small = cv2.resize(overlay, (1152, 648))
        out_p = os.path.join(NERF_DIR, f"mesh_projection_{idx:02d}.jpg")
        cv2.imwrite(out_p, vis_small)
        print(f"Saved {out_p}")

# 5. Quantitative Evaluation against Ground Truth STL
print("\n--- QUANTITATIVE EVALUATION: NeRF vs Ground Truth STL ---")
stl_mesh = o3d.io.read_triangle_mesh(STL_PATH)
stl_verts = np.asarray(stl_mesh.vertices)
stl_extents = stl_verts.max(axis=0) - stl_verts.min(axis=0)

nerf_extents = vertices_opencv_mm.max(axis=0) - vertices_opencv_mm.min(axis=0)

pcd_nerf_eval = o3d.geometry.PointCloud()
pcd_nerf_eval.points = o3d.utility.Vector3dVector(vertices_opencv_mm)

pcd_stl = o3d.geometry.PointCloud()
pcd_stl.points = o3d.utility.Vector3dVector(stl_verts)

reg_nerf = o3d.pipelines.registration.registration_icp(
    pcd_nerf_eval, pcd_stl, 5.0, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
)

nerf_verts_aligned = np.dot(vertices_opencv_mm, reg_nerf.transformation[:3, :3].T) + reg_nerf.transformation[:3, 3]
pcd_nerf_aligned = o3d.geometry.PointCloud()
pcd_nerf_aligned.points = o3d.utility.Vector3dVector(nerf_verts_aligned)

nerf_dists = np.asarray(pcd_nerf_aligned.compute_point_cloud_distance(pcd_stl))

nerf_mean_err = np.mean(nerf_dists)
nerf_rms_err = np.sqrt(np.mean(nerf_dists ** 2))
nerf_max_err = np.max(nerf_dists)

nerf_diameter_err = abs(max(nerf_extents[0], nerf_extents[1]) - 50.0)
nerf_height_err = abs(nerf_extents[2] - 25.0)

print(f"[NeRF vs. Ground Truth STL]")
print(f"  Extents (X, Y, Z): {nerf_extents[0]:.2f} mm x {nerf_extents[1]:.2f} mm x {nerf_extents[2]:.2f} mm")
print(f"  Diameter Error: {nerf_diameter_err:.2f} mm ({nerf_diameter_err/50.0*100:.2f}%)")
print(f"  Height Error:   {nerf_height_err:.2f} mm ({nerf_height_err/25.0*100:.2f}%)")
print(f"  Mean Error:     {nerf_mean_err:.3f} mm")
print(f"  RMS Error:      {nerf_rms_err:.3f} mm")
print(f"  Max Error:      {nerf_max_err:.3f} mm")

# Load TSDF mesh eval stats
tsdf_mesh = o3d.io.read_triangle_mesh(os.path.join(TSDF_OUT_DIR, "hemisphere_tsdf_mesh.ply"))
tsdf_verts = np.asarray(tsdf_mesh.vertices)
tsdf_extents = tsdf_verts.max(axis=0) - tsdf_verts.min(axis=0)

pcd_tsdf = o3d.geometry.PointCloud()
pcd_tsdf.points = o3d.utility.Vector3dVector(tsdf_verts)
reg_tsdf = o3d.pipelines.registration.registration_icp(
    pcd_tsdf, pcd_stl, 5.0, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
)
tsdf_verts_aligned = np.dot(tsdf_verts, reg_tsdf.transformation[:3, :3].T) + reg_tsdf.transformation[:3, 3]
pcd_tsdf_aligned = o3d.geometry.PointCloud()
pcd_tsdf_aligned.points = o3d.utility.Vector3dVector(tsdf_verts_aligned)
tsdf_dists = np.asarray(pcd_tsdf_aligned.compute_point_cloud_distance(pcd_stl))

tsdf_mean_err = np.mean(tsdf_dists)
tsdf_rms_err = np.sqrt(np.mean(tsdf_dists ** 2))
tsdf_max_err = np.max(tsdf_dists)

tsdf_diameter_err = abs(max(tsdf_extents[0], tsdf_extents[1]) - 50.0)
tsdf_height_err = abs(tsdf_extents[2] - 25.0)

# Generate Side-by-Side Markdown Report
report_md = f"""# Quantitative 3D Reconstruction Benchmark: Green Hemisphere ($5\\text{{ cm}}$)

## Overview
This report presents a quantitative 3D mesh accuracy benchmark comparing **TSDF Volumetric Integration** and **NeRF Neural Radiance Fields (`nerfacto`)** against the ground truth CAD file (`3d_files/hemisphere_5cm.stl`) for the `captures_0809_hemisphere` dataset.

---

## 1. Ground Truth CAD Model (`hemisphere_5cm.stl`)
* **Nominal Diameter**: $50.00\\text{{ mm}}$ ($5.0\\text{{ cm}}$)
* **Nominal Height**: $25.00\\text{{ mm}}$ ($2.5\\text{{ cm}}$)
* **STL Vertices**: {len(stl_verts):,}
* **STL Triangles**: {len(stl_mesh.triangles):,}
* **Exact STL Extents**: ${stl_extents[0]:.2f}\\text{{ mm}} \\times {stl_extents[1]:.2f}\\text{{ mm}} \\times {stl_extents[2]:.2f}\\text{{ mm}}$

---

## 2. Side-by-Side Quantitative Accuracy Benchmark Table

| Reconstruction Method | Extents ($X \\times Y \\times Z$ mm) | Diameter Error (mm / %) | Height Error (mm / %) | Mean Error (mm) | RMS Error (mm) | Max Error (mm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ground Truth CAD** | ${stl_extents[0]:.2f} \\times {stl_extents[1]:.2f} \\times {stl_extents[2]:.2f}$ | $0.00\\text{{ mm}}$ ($0.0\\%$) | $0.00\\text{{ mm}}$ ($0.0\\%$) | $0.000\\text{{ mm}}$ | $0.000\\text{{ mm}}$ | $0.000\\text{{ mm}}$ |
| **TSDF Volumetric Integration** | ${tsdf_extents[0]:.2f} \\times {tsdf_extents[1]:.2f} \\times {tsdf_extents[2]:.2f}$ | **{tsdf_diameter_err:.2f} mm** ({tsdf_diameter_err/50.0*100:.2f}%) | **{tsdf_height_err:.2f} mm** ({tsdf_height_err/25.0*100:.2f}%) | **{tsdf_mean_err:.3f} mm** | **{tsdf_rms_err:.3f} mm** | **{tsdf_max_err:.3f} mm** |
| **NeRF (`nerfacto` at 500 steps)** | ${nerf_extents[0]:.2f} \\times {nerf_extents[1]:.2f} \\times {nerf_extents[2]:.2f}$ | **{nerf_diameter_err:.2f} mm** ({nerf_diameter_err/50.0*100:.2f}%) | **{nerf_height_err:.2f} mm** ({nerf_height_err/25.0*100:.2f}%) | **{nerf_mean_err:.3f} mm** | **{nerf_rms_err:.3f} mm** | **{nerf_max_err:.3f} mm** |

---

## 3. Visual Verification Overlays

### TSDF Wireframe Overlays
* **Frame 0 (0°)**: [tsdf_mesh_projection_00.jpg](file://{os.path.abspath(os.path.join(TSDF_OUT_DIR, "tsdf_mesh_projection_00.jpg"))})
* **Frame 16 (90°)**: [tsdf_mesh_projection_16.jpg](file://{os.path.abspath(os.path.join(TSDF_OUT_DIR, "tsdf_mesh_projection_16.jpg"))})
* **Frame 32 (180°)**: [tsdf_mesh_projection_32.jpg](file://{os.path.abspath(os.path.join(TSDF_OUT_DIR, "tsdf_mesh_projection_32.jpg"))})
* **Frame 48 (270°)**: [tsdf_mesh_projection_48.jpg](file://{os.path.abspath(os.path.join(TSDF_OUT_DIR, "tsdf_mesh_projection_48.jpg"))})

### NeRF Wireframe Overlays
* **Frame 0 (0°)**: [mesh_projection_00.jpg](file://{os.path.abspath(os.path.join(NERF_DIR, "mesh_projection_00.jpg"))})
* **Frame 16 (90°)**: [mesh_projection_16.jpg](file://{os.path.abspath(os.path.join(NERF_DIR, "mesh_projection_16.jpg"))})
* **Frame 32 (180°)**: [mesh_projection_32.jpg](file://{os.path.abspath(os.path.join(NERF_DIR, "mesh_projection_32.jpg"))})
* **Frame 48 (270°)**: [mesh_projection_48.jpg](file://{os.path.abspath(os.path.join(NERF_DIR, "mesh_projection_48.jpg"))})

![NeRF Mesh Wireframe Overlay](file://{os.path.abspath(os.path.join(NERF_DIR, "mesh_projection_00.jpg"))})
"""

with open(DOCS_REPORT_PATH, "w") as f:
    f.write(report_md)

with open("doc/hemisphere_comparison_0809.md", "w") as f:
    f.write(report_md)

print(f"\n✅ NERF POST-PROCESSING & BENCHMARK FINISHED! Report updated in {DOCS_REPORT_PATH}")
