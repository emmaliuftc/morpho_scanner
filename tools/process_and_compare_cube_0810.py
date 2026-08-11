import sys
sys.path.append(".")
import os
import glob
import json
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

INPUT_DIR = "captures_0810_cube"
CALIB_REF = "captures_0726_clay_checkboard_64_calibrated/calibration_results.json"
CALIB_OUT_DIR = "captures_0810_cube_calibrated"
TSDF_OUT_DIR = "captures_0810_cube_tsdf"
MASKS_DIR = os.path.join(CALIB_OUT_DIR, "masks")
STL_PATH = "3d_files/cube_4cm.stl"
DOCS_PATH = "docs/cube_comparison_0810.md"

os.makedirs(CALIB_OUT_DIR, exist_ok=True)
os.makedirs(TSDF_OUT_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

with open(CALIB_REF, "r") as f:
    cal_data = json.load(f)

K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float64)
dist_cal = np.array(cal_data["distortion_coefficients"], dtype=np.float64)
C_rot = np.array(cal_data["plate_center_mm"], dtype=np.float64)
normal = np.array(cal_data["plate_normal"], dtype=np.float64)
normal = normal / np.linalg.norm(normal)
step_size_deg = float(cal_data["step_size_deg"])

# Basis setup: normal is Z-axis
ref = np.array([1.0, 0.0, 0.0])
x_col = ref - np.dot(ref, normal) * normal
x_col = x_col / np.linalg.norm(x_col)
y_col = np.cross(normal, x_col)
R_cam = np.column_stack((x_col, y_col, normal))

cx_proj = int(round(K_cal[0, 0] * C_rot[0] / C_rot[2] + K_cal[0, 2]))
cy_proj = int(round(K_cal[1, 1] * C_rot[1] / C_rot[2] + K_cal[1, 2]))

print("=== STEP 1: Building Calibrated Dataset for 4cm Green Cube ===")
image_paths = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.jpg")),
    key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[1])
)

h, w = 2592, 4608

for idx, img_p in enumerate(image_paths):
    img = cv2.imread(img_p)
    undist = cv2.undistort(img, K_cal, dist_cal)
    
    out_undist_p = os.path.join(CALIB_OUT_DIR, f"capture_{idx}.jpg")
    cv2.imwrite(out_undist_p, undist)
    
    # Annotated image
    vis = undist.copy()
    cv2.circle(vis, (cx_proj, cy_proj), 12, (0, 255, 0), -1)
    cv2.circle(vis, (cx_proj, cy_proj), 30, (0, 255, 0), 2)
    
    # Draw rotation Z-vector line
    axis_end_3d = C_rot + normal * 60.0 # 60mm along normal
    u_end = int(round(K_cal[0, 0] * axis_end_3d[0] / axis_end_3d[2] + K_cal[0, 2]))
    v_end = int(round(K_cal[1, 1] * axis_end_3d[1] / axis_end_3d[2] + K_cal[1, 2]))
    cv2.arrowedLine(vis, (cx_proj, cy_proj), (u_end, v_end), (255, 255, 0), 3, cv2.LINE_AA)
    
    cv2.imwrite(os.path.join(CALIB_OUT_DIR, f"annotated_capture_{idx}.jpg"), vis)

with open(os.path.join(CALIB_OUT_DIR, "calibration_results.json"), "w") as f:
    json.dump(cal_data, f, indent=2)

print(f"✅ Created {len(image_paths)} calibrated captures in {CALIB_OUT_DIR}!")

print("\n=== STEP 2: Silhouette Segmentation & Placement Offset Detection ===")
masks = []
for idx in range(len(image_paths)):
    img_p = os.path.join(CALIB_OUT_DIR, f"capture_{idx}.jpg")
    img = cv2.imread(img_p)
    b, g, r = cv2.split(img)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    
    # Green cube segmentation
    m = np.where((h_ch >= 30) & (h_ch <= 95) & (s_ch > 60) & (v_ch < 185), 255, 0).astype(np.uint8)
    mask_roi = np.zeros_like(b)
    cv2.circle(mask_roi, (cx_proj, cy_proj), 1100, 255, -1)
    
    m = cv2.bitwise_and(m, mask_roi)
    
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros_like(m)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(solid_mask, [c], -1, 255, -1)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    solid_mask = cv2.morphologyEx(solid_mask, cv2.MORPH_CLOSE, kernel_close)
    
    cv2.imwrite(os.path.join(MASKS_DIR, f"mask_{idx:02d}.png"), solid_mask)
    masks.append(solid_mask)

# Compute 3D placement offset vector in local turntable frame
cnts0, _ = cv2.findContours(masks[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c0 = max(cnts0, key=cv2.contourArea)
(x_fit0, y_fit0), _ = cv2.minEnclosingCircle(c0)

dx_cam_mm = (x_fit0 - cx_proj) * C_rot[2] / K_cal[0, 0]
dy_cam_mm = (y_fit0 - cy_proj) * C_rot[2] / K_cal[1, 1]
delta_local = R_cam.T @ np.array([dx_cam_mm, dy_cam_mm, 0.0])

print(f"Generated {len(masks)} silhouette masks in {MASKS_DIR}")
print(f"Cube Placement Offset (Turntable Frame): {delta_local} mm")

print("\n=== STEP 3: TSDF Volumetric Reconstruction & Marching Cubes ===")
grid_size_mm = 140.0
grid_res = 256
voxel_size = grid_size_mm / grid_res

vox_x = np.linspace(-grid_size_mm / 2.0, grid_size_mm / 2.0, grid_res)
vox_y = np.linspace(-grid_size_mm / 2.0, grid_size_mm / 2.0, grid_res)
vox_z = np.linspace(-grid_size_mm / 2.0, grid_size_mm / 2.0, grid_res)
X_g, Y_g, Z_g = np.meshgrid(vox_x, vox_y, vox_z, indexing='ij')
voxel_coords = np.vstack([X_g.ravel(), Y_g.ravel(), Z_g.ravel()]) # 3 x N

num_voxels = voxel_coords.shape[1]
carve_counts = np.zeros(num_voxels, dtype=np.int32)
voxel_in_bg = np.zeros(num_voxels, dtype=bool)

dist_origin = np.linalg.norm(voxel_coords[:2, :], axis=0)
voxel_in_bg[dist_origin > 38.0] = True
voxel_in_bg[voxel_coords[2, :] < -2.0] = True
voxel_in_bg[voxel_coords[2, :] > 45.0] = True

coords_in_obj = voxel_coords + delta_local.reshape(3, 1)

for i in range(len(image_paths)):
    if (i % 16) == 0 or i == len(image_paths) - 1:
        print(f"Carving TSDF frame {i+1}/{len(image_paths)}...")
    mask = masks[i]
    
    angle_deg = i * step_size_deg
    angle_rad = np.deg2rad(angle_deg)
    R_i = Rot.from_rotvec(-angle_rad * np.array([0, 0, 1.0])).as_matrix()
    
    pts_rotated = R_i @ coords_in_obj
    cam_pts = (R_cam @ pts_rotated + C_rot.reshape(3, 1))
    zc = cam_pts[2, :]
    valid_z = zc > 1.0
    
    u = np.round((K_cal[0, 0] * cam_pts[0, :] / zc) + K_cal[0, 2]).astype(int)
    v = np.round((K_cal[1, 1] * cam_pts[1, :] / zc) + K_cal[1, 2]).astype(int)
    
    valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    valid = valid_z & valid_uv
    
    pixel_vals = np.zeros(num_voxels, dtype=np.uint8)
    pixel_vals[valid] = mask[v[valid], u[valid]]
    
    hit_bg = pixel_vals == 0
    carve_counts[hit_bg] += 1

voxel_in_bg[carve_counts >= 32] = True

tsdf_grid = np.ones((grid_res, grid_res, grid_res), dtype=np.float32)
tsdf_grid[voxel_in_bg.reshape((grid_res, grid_res, grid_res))] = 1.0
tsdf_grid[(~voxel_in_bg).reshape((grid_res, grid_res, grid_res))] = -1.0

from skimage.measure import marching_cubes
verts_vox, faces, normals, values = marching_cubes(tsdf_grid, level=0.0)

# Convert voxel indices to mm coordinates
verts_x = vox_x[0] + verts_vox[:, 0] * voxel_size
verts_y = vox_y[0] + verts_vox[:, 1] * voxel_size
verts_z = vox_z[0] + verts_vox[:, 2] * voxel_size
verts = np.column_stack([verts_x, verts_y, verts_z])

tsdf_mesh = o3d.geometry.TriangleMesh()
tsdf_mesh.vertices = o3d.utility.Vector3dVector(verts)
tsdf_mesh.triangles = o3d.utility.Vector3iVector(faces)

# Filter boundary wall artifacts
verts_r = np.linalg.norm(verts[:, :2], axis=1)
boundary_mask = (verts_r > 35.0) | (verts[:, 2] < 0.0)
tsdf_mesh.remove_vertices_by_mask(boundary_mask)

tsdf_ply_path = os.path.join(TSDF_OUT_DIR, "cube_tsdf_mesh.ply")
o3d.io.write_triangle_mesh(tsdf_ply_path, tsdf_mesh)
print(f"Saved Cube TSDF Mesh to: {tsdf_ply_path}")

print("\n=== STEP 4: Quantitative ICP Benchmark against 3d_files/cube_4cm.stl ===")
stl_mesh = o3d.io.read_triangle_mesh(STL_PATH)
stl_verts = np.asarray(stl_mesh.vertices)
stl_extents = stl_verts.max(axis=0) - stl_verts.min(axis=0)

clean_verts = np.asarray(tsdf_mesh.vertices)
tsdf_extents = clean_verts.max(axis=0) - clean_verts.min(axis=0)

pcd_stl = stl_mesh.sample_points_uniformly(number_of_points=20000)
pcd_tsdf = tsdf_mesh.sample_points_uniformly(number_of_points=20000)

reg_icp = o3d.pipelines.registration.registration_icp(
    pcd_tsdf, pcd_stl, max_correspondence_distance=5.0,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

dists = np.asarray(pcd_tsdf.compute_point_cloud_distance(pcd_stl))
mean_err = np.mean(dists)
rms_err = np.sqrt(np.mean(dists**2))

w_err = abs(tsdf_extents[0] - stl_extents[0])
d_err = abs(tsdf_extents[1] - stl_extents[1])
h_err = abs(tsdf_extents[2] - stl_extents[2])

print(f"[TSDF Cube vs. Ground Truth STL]")
print(f"  CAD STL Extents (X, Y, Z): {stl_extents[0]:.2f} mm x {stl_extents[1]:.2f} mm x {stl_extents[2]:.2f} mm")
print(f"  TSDF Extents    (X, Y, Z): {tsdf_extents[0]:.2f} mm x {tsdf_extents[1]:.2f} mm x {tsdf_extents[2]:.2f} mm")
print(f"  Width Error (X):  {w_err:.2f} mm ({w_err/stl_extents[0]*100:.2f}%)")
print(f"  Depth Error (Y):  {d_err:.2f} mm ({d_err/stl_extents[1]*100:.2f}%)")
print(f"  Height Error (Z): {h_err:.2f} mm ({h_err/stl_extents[2]*100:.2f}%)")
print(f"  Mean Error:       {mean_err:.3f} mm")
print(f"  RMS Error:        {rms_err:.3f} mm")

print("\n--- Generating Wireframe Projection Overlays ---")
triangles = np.asarray(tsdf_mesh.triangles)

for f_idx in [0, 16, 32, 48]:
    img_p = os.path.join(CALIB_OUT_DIR, f"capture_{f_idx}.jpg")
    img = cv2.imread(img_p)
    
    angle_deg = f_idx * step_size_deg
    angle_rad = np.deg2rad(angle_deg)
    R_i = Rot.from_rotvec(-angle_rad * np.array([0, 0, 1.0])).as_matrix()
    
    pts_in_obj = clean_verts + delta_local.reshape(1, 3)
    pts_rotated = R_i @ pts_in_obj.T
    cam_pts = (R_cam @ pts_rotated + C_rot.reshape(3, 1))
    zc = cam_pts[2, :]
    valid = zc > 1.0
    
    u = np.round((K_cal[0, 0] * cam_pts[0, :] / zc) + K_cal[0, 2]).astype(int)
    v = np.round((K_cal[1, 1] * cam_pts[1, :] / zc) + K_cal[1, 2]).astype(int)
    
    vis = img.copy()
    for tri in triangles[::3]:
        if valid[tri[0]] and valid[tri[1]] and valid[tri[2]]:
            pt1 = (u[tri[0]], v[tri[0]])
            pt2 = (u[tri[1]], v[tri[1]])
            pt3 = (u[tri[2]], v[tri[2]])
            cv2.line(vis, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(vis, pt2, pt3, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(vis, pt3, pt1, (0, 255, 255), 1, cv2.LINE_AA)
            
    out_proj = os.path.join(TSDF_OUT_DIR, f"tsdf_cube_projection_{f_idx:02d}.jpg")
    cv2.imwrite(out_proj, vis)

print("✅ TSDF Cube Pipeline & Benchmark Complete!")

print("\n=== STEP 5: Generating Quantitative Documentation Report ===")
doc_content = f"""# Quantitative 3D Reconstruction Benchmark: 4cm Green Cube

## Overview
This document presents the quantitative 3D reconstruction benchmark for the **$4\\text{{ cm}} \\times 4\\text{{ cm}} \\times 4\\text{{ cm}}$ Green Cube (`captures_0810_cube`)** comparing **TSDF Volumetric Integration** evaluated against the empirical CAD model (`3d_files/cube_4cm.stl`).

---

## 1. Ground Truth CAD Model (`cube_4cm.stl`)
* **Nominal Dimensions**: ${stl_extents[0]:.2f}\\text{{ mm}} \\times {stl_extents[1]:.2f}\\text{{ mm}} \\times {stl_extents[2]:.2f}\\text{{ mm}}$
* **STL Vertices**: {len(stl_verts)}
* **STL Triangles**: {len(stl_mesh.triangles)}

---

## 2. Quantitative Accuracy Benchmark Table

| Dataset & Method | Extents ($X \\times Y \\times Z$ mm) | Width Error ($X$) | Depth Error ($Y$) | Height Error ($Z$) | Mean Surface Error | RMS Surface Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ground Truth CAD (`cube_4cm.stl`)** | ${stl_extents[0]:.2f} \\times {stl_extents[1]:.2f} \\times {stl_extents[2]:.2f}\\text{{ mm}}$ | $0.00\\text{{ mm}}$ ($0.0\\%$) | $0.00\\text{{ mm}}$ ($0.0\\%$) | $0.00\\text{{ mm}}$ ($0.0\\%$) | $0.000\\text{{ mm}}$ | $0.000\\text{{ mm}}$ |
| **TSDF (4cm Green Cube)** | ${tsdf_extents[0]:.2f} \\times {tsdf_extents[1]:.2f} \\times {tsdf_extents[2]:.2f}\\text{{ mm}}$ | **${w_err:.2f}\\text{{ mm}}$** (**${w_err/stl_extents[0]*100:.2f}\\%$**) | **${d_err:.2f}\\text{{ mm}}$** (**${d_err/stl_extents[1]*100:.2f}\\%$**) | **${h_err:.2f}\\text{{ mm}}$** (**${h_err/stl_extents[2]*100:.2f}\\%$**) | **${mean_err:.3f}\\text{{ mm}}$** | **${rms_err:.3f}\\text{{ mm}}$** |

---

## 3. Visual Wireframe Projection Overlay

![TSDF Cube Wireframe](file://{os.path.abspath(os.path.join(TSDF_OUT_DIR, "tsdf_cube_projection_00.jpg"))})
"""

with open(DOCS_PATH, "w") as f:
    f.write(doc_content)

# Copy to doc/ folder as well
os.makedirs("doc", exist_ok=True)
with open("doc/cube_comparison_0810.md", "w") as f:
    f.write(doc_content)

print(f"✅ Documentation written to {DOCS_PATH} and doc/cube_comparison_0810.md!")
