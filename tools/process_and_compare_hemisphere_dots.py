import sys
sys.path.append(".")
import os
import glob
import json
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

CALIB_DIR = "captures_0809_hemisphere_dots_calibrated"
CALIB_JSON = os.path.join(CALIB_DIR, "calibration_results.json")
STL_PATH = "3d_files/hemisphere_5cm.stl"
TSDF_OUT_DIR = "captures_0809_hemisphere_dots_tsdf"
MASKS_DIR = os.path.join(CALIB_DIR, "masks")

os.makedirs(TSDF_OUT_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

with open(CALIB_JSON, "r") as f:
    cal_data = json.load(f)

K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float64)
dist_cal = np.array(cal_data["distortion_coefficients"], dtype=np.float64)
C_rot = np.array(cal_data["plate_center_mm"], dtype=np.float64)
normal = np.array(cal_data["plate_normal"], dtype=np.float64)
normal = normal / np.linalg.norm(normal)
step_size_deg = float(cal_data["step_size_deg"])

# Frame basis: normal is Z-axis
ref = np.array([1.0, 0.0, 0.0])
x_col = ref - np.dot(ref, normal) * normal
x_col = x_col / np.linalg.norm(x_col)
y_col = np.cross(normal, x_col)
R_cam = np.column_stack((x_col, y_col, normal))

cx_proj = int(round(K_cal[0, 0] * C_rot[0] / C_rot[2] + K_cal[0, 2]))
cy_proj = int(round(K_cal[1, 1] * C_rot[1] / C_rot[2] + K_cal[1, 2]))

print("--- STAGE 1: Generating Silhouette Masks & Computing Object Placement Offset ---")
image_paths = sorted(
    glob.glob(os.path.join(CALIB_DIR, "capture_*.jpg")),
    key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[1])
)
image_paths = [p for p in image_paths if "annotated" not in os.path.basename(p)]

masks = []
for idx, img_p in enumerate(image_paths):
    img = cv2.imread(img_p)
    b, g, r = cv2.split(img)
    
    mask_sat = np.where((b < 150) & (r < 170), 255, 0).astype(np.uint8)
    mask_roi = np.zeros_like(mask_sat)
    cv2.circle(mask_roi, (cx_proj, cy_proj), 600, 255, -1)
    
    mask = cv2.bitwise_and(mask_sat, mask_roi)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    cnts, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros_like(mask_closed)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(solid_mask, [c], -1, 255, -1)
    
    cv2.imwrite(os.path.join(MASKS_DIR, f"mask_{idx:02d}.png"), solid_mask)
    masks.append(solid_mask)

# Compute object placement offset in local turntable frame
cnts0, _ = cv2.findContours(masks[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c0 = max(cnts0, key=cv2.contourArea)
(x_fit0, y_fit0), _ = cv2.minEnclosingCircle(c0)

dx_cam_mm = (x_fit0 - cx_proj) * C_rot[2] / K_cal[0, 0]
dy_cam_mm = (y_fit0 - cy_proj) * C_rot[2] / K_cal[1, 1]
delta_local = R_cam.T @ np.array([dx_cam_mm, dy_cam_mm, 0.0])

print(f"Generated {len(masks)} silhouette masks in {MASKS_DIR}")
print(f"Object Placement Offset (Turntable Frame): {delta_local} mm")

print("\n--- STAGE 2: TSDF Volumetric Reconstruction ---")
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
voxel_in_bg[dist_origin > 30.0] = True
voxel_in_bg[voxel_coords[2, :] < -2.0] = True
voxel_in_bg[voxel_coords[2, :] > 30.0] = True

h, w = 2592, 4608
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
boundary_mask = (verts_r > 26.5) | (verts[:, 2] < 0.0)
tsdf_mesh.remove_vertices_by_mask(boundary_mask)

tsdf_ply_path = os.path.join(TSDF_OUT_DIR, "hemisphere_dots_tsdf_mesh.ply")
o3d.io.write_triangle_mesh(tsdf_ply_path, tsdf_mesh)
print(f"Saved TSDF Dots Mesh to: {tsdf_ply_path}")

print("\n--- STAGE 3: Quantitative Comparison against hemisphere_5cm.stl ---")
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

diam_err = abs(tsdf_extents[0] - 50.0)
height_err = abs(tsdf_extents[2] - 25.0)

print(f"[TSDF Dots vs. Ground Truth STL]")
print(f"  Extents (X, Y, Z): {tsdf_extents[0]:.2f} mm x {tsdf_extents[1]:.2f} mm x {tsdf_extents[2]:.2f} mm")
print(f"  Diameter Error: {diam_err:.2f} mm ({diam_err/50.0*100:.2f}%)")
print(f"  Height Error:   {height_err:.2f} mm ({height_err/25.0*100:.2f}%)")
print(f"  Mean Error:     {mean_err:.3f} mm")
print(f"  RMS Error:      {rms_err:.3f} mm")

print("\n--- STAGE 4: Wireframe Projection Overlays ---")
triangles = np.asarray(tsdf_mesh.triangles)

for f_idx in [0, 16, 32, 48]:
    img_p = os.path.join(CALIB_DIR, f"capture_{f_idx}.jpg")
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
            
    out_proj = os.path.join(TSDF_OUT_DIR, f"tsdf_dots_projection_{f_idx:02d}.jpg")
    cv2.imwrite(out_proj, vis)

print("✅ TSDF Dots Pipeline & Benchmark Complete!")
