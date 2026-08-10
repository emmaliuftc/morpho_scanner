import cv2
import numpy as np
import os
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

nerf_mesh_path = os.path.join(NERF_DIR, "mesh.ply")
nerf_mesh = o3d.io.read_triangle_mesh(nerf_mesh_path)
vertices_opengl = np.asarray(nerf_mesh.vertices)

# The WRONG transformation previously used (Y/Z negation + R_z270 rotation)
vertices_opencv = vertices_opengl.copy()
vertices_opencv[:, 1] = -vertices_opencv[:, 1]
vertices_opencv[:, 2] = -vertices_opencv[:, 2]
vertices_opencv_mm = vertices_opencv * 150.0

R_z270 = Rot.from_rotvec(np.deg2rad(270) * np.array([0.0, 0.0, 1.0])).as_matrix()
vertices_opencv_mm = np.dot(vertices_opencv_mm, R_z270.T)

triangles = np.asarray(nerf_mesh.triangles)
edges = set()
for tri in triangles:
    edges.add(tuple(sorted((tri[0], tri[1]))))
    edges.add(tuple(sorted((tri[1], tri[2]))))
    edges.add(tuple(sorted((tri[2], tri[0]))))

W, H = 4608, 2592

for idx in [0, 16, 32, 48]:
    img_path = os.path.join(INPUT_FOLDER, f"capture_{idx}.jpg")
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
            
    vis_small = cv2.resize(overlay, (1152, 648))
    out_path = os.path.join(NERF_DIR, f"wrong_mesh_projection_{idx:02d}.jpg")
    cv2.imwrite(out_path, vis_small)
    print(f"Saved {out_path}")
