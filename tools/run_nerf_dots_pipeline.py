import sys
sys.path.append(".")
import os
import glob
import json
import cv2
import numpy as np
import subprocess
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

CALIB_DIR = "captures_0809_hemisphere_dots_calibrated"
CALIB_JSON = os.path.join(CALIB_DIR, "calibration_results.json")
STL_PATH = "3d_files/hemisphere_5cm.stl"
NERF_DATASET_DIR = "captures_0809_hemisphere_dots_nerf_dataset"
IMAGES_8_DIR = os.path.join(NERF_DATASET_DIR, "images_8")
OUTPUT_DIR = "outputs/hemisphere_dots/nerfacto"

os.makedirs(IMAGES_8_DIR, exist_ok=True)

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

print("--- STAGE 1: Preparing Downscaled NeRF Dataset for Speckle Dots ---")
image_paths = sorted(
    glob.glob(os.path.join(CALIB_DIR, "capture_*.jpg")),
    key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[1])
)
image_paths = [p for p in image_paths if "annotated" not in os.path.basename(p)]

# Downscale factor = 8 -> 576 x 1024
scale_factor = 8.0
w_8, h_8 = int(4608 / scale_factor), int(2592 / scale_factor)
fl_x_8 = K_cal[0, 0] / scale_factor
fl_y_8 = K_cal[1, 1] / scale_factor
cx_8 = K_cal[0, 2] / scale_factor
cy_8 = K_cal[1, 2] / scale_factor

frames = []

for idx, img_p in enumerate(image_paths):
    img = cv2.imread(img_p)
    img_8 = cv2.resize(img, (w_8, h_8), interpolation=cv2.INTER_AREA)
    rel_path = f"images_8/frame_{idx:05d}.jpg"
    out_img_p = os.path.join(NERF_DATASET_DIR, rel_path)
    cv2.imwrite(out_img_p, img_8)
    
    R_eff, t_eff = get_camera_pose_for_frame(idx)
    
    # NeRF c2w matrix (OpenCV to OpenGL camera convention)
    # OpenCV: Z forwards, Y downwards, X rightwards
    # OpenGL: Z backwards, Y upwards, X rightwards
    c2w_cv = np.eye(4)
    c2w_cv[:3, :3] = R_eff
    c2w_cv[:3, 3] = t_eff / 100.0 # scale down to decimeters for NeRF box
    
    c2w_gl = c2w_cv.copy()
    c2w_gl[0:3, 1] *= -1.0
    c2w_gl[0:3, 2] *= -1.0
    
    frames.append({
        "file_path": rel_path,
        "transform_matrix": c2w_gl.tolist()
    })

transforms_8_data = {
    "fl_x": fl_x_8,
    "fl_y": fl_y_8,
    "cx": cx_8,
    "cy": cy_8,
    "w": w_8,
    "h": h_8,
    "camera_model": "OPENCV",
    "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
    "frames": frames
}

with open(os.path.join(NERF_DATASET_DIR, "transforms_8.json"), "w") as f:
    json.dump(transforms_8_data, f, indent=2)

with open(os.path.join(NERF_DATASET_DIR, "transforms.json"), "w") as f:
    json.dump(transforms_8_data, f, indent=2)

print(f"✅ Created NeRF dataset with {len(frames)} frames in {NERF_DATASET_DIR}!")

print("\n--- STAGE 2: Training NeRF (nerfacto, 500 steps) ---")
train_cmd = [
    ".venv_nerf/bin/ns-train", "nerfacto",
    "--data", NERF_DATASET_DIR,
    "--output-dir", "outputs/hemisphere_dots",
    "--experiment-name", "nerfacto",
    "--max-num-iterations", "500",
    "--machine.device-type", "cpu",
    "--pipeline.model.predict-normals", "True",
    "--pipeline.model.background-color", "white",
    "nerfstudio-data",
    "--auto-scale-poses", "False",
    "--center-method", "none",
    "--orientation-method", "none"
]

subprocess.run(train_cmd, check=True)
print("✅ NeRF Training Complete!")

print("\n--- STAGE 3: Extracting Mesh & ICP Evaluation vs. Ground Truth STL ---")
# Find config file
config_paths = glob.glob("outputs/hemisphere_dots/nerfacto/nerfacto/*/config.yml")
if not config_paths:
    print("Error: Could not find config.yml!")
    sys.exit(1)
    
latest_config = sorted(config_paths)[-1]
mesh_out_dir = "captures_0809_hemisphere_dots_nerf"
os.makedirs(mesh_out_dir, exist_ok=True)
raw_mesh_path = os.path.join(mesh_out_dir, "raw_mesh.ply")

export_cmd = [
    ".venv_nerf/bin/ns-export", "pointcloud",
    "--load-config", latest_config,
    "--output-dir", mesh_out_dir,
    "--num-points", "50000",
    "--remove-outliers", "True",
    "--normal-method", "open3d"
]
subprocess.run(export_cmd, check=True)

pcd_files = glob.glob(os.path.join(mesh_out_dir, "*.ply"))
if pcd_files:
    pcd = o3d.io.read_point_cloud(pcd_files[0])
    # Convert points back to mm coordinates
    pts = np.asarray(pcd.points) * 100.0
    
    # Filter boundary wall artifacts
    pts_r = np.linalg.norm(pts[:, :2], axis=1)
    valid_pts = (pts_r <= 26.5) & (pts[:, 2] >= 0.0) & (pts[:, 2] <= 30.0)
    clean_pts = pts[valid_pts]
    
    pcd_clean = o3d.geometry.PointCloud()
    pcd_clean.points = o3d.utility.Vector3dVector(clean_pts)
    
    pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_clean, depth=8)
    
    clean_mesh_path = os.path.join(mesh_out_dir, "hemisphere_dots_nerf_mesh.ply")
    o3d.io.write_triangle_mesh(clean_mesh_path, mesh)
    
    # ICP Evaluation
    stl_mesh = o3d.io.read_triangle_mesh(STL_PATH)
    pcd_stl = stl_mesh.sample_points_uniformly(number_of_points=20000)
    pcd_nerf = mesh.sample_points_uniformly(number_of_points=20000)
    
    reg_icp = o3d.pipelines.registration.registration_icp(
        pcd_nerf, pcd_stl, max_correspondence_distance=5.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    dists = np.asarray(pcd_nerf.compute_point_cloud_distance(pcd_stl))
    mean_err = np.mean(dists)
    rms_err = np.sqrt(np.mean(dists**2))
    
    c_verts = np.asarray(mesh.vertices)
    nerf_extents = c_verts.max(axis=0) - c_verts.min(axis=0)
    diam_err = abs(nerf_extents[0] - 50.0)
    height_err = abs(nerf_extents[2] - 25.0)
    
    print(f"\n[NeRF Dots vs. Ground Truth STL]")
    print(f"  Extents (X, Y, Z): {nerf_extents[0]:.2f} mm x {nerf_extents[1]:.2f} mm x {nerf_extents[2]:.2f} mm")
    print(f"  Diameter Error: {diam_err:.2f} mm ({diam_err/50.0*100:.2f}%)")
    print(f"  Height Error:   {height_err:.2f} mm ({height_err/25.0*100:.2f}%)")
    print(f"  Mean Error:     {mean_err:.3f} mm")
    print(f"  RMS Error:      {rms_err:.3f} mm")
    
    # Render wireframe projection overlays
    triangles = np.asarray(mesh.triangles)
    for f_idx in [0, 16, 32, 48]:
        img_p = os.path.join(CALIB_DIR, f"capture_{f_idx}.jpg")
        img = cv2.imread(img_p)
        
        R_eff, t_eff = get_camera_pose_for_frame(f_idx)
        P_world = c_verts + C_rot
        P_cam = (R_eff.T @ (P_world - t_eff).T).T
        valid = P_cam[:, 2] > 0
        
        u = np.round(K_cal[0,0] * P_cam[:, 0] / P_cam[:, 2] + K_cal[0,2]).astype(int)
        v = np.round(K_cal[1,1] * P_cam[:, 1] / P_cam[:, 2] + K_cal[1,2]).astype(int)
        
        vis = img.copy()
        for tri in triangles[::3]:
            if valid[tri[0]] and valid[tri[1]] and valid[tri[2]]:
                pt1 = (u[tri[0]], v[tri[0]])
                pt2 = (u[tri[1]], v[tri[1]])
                pt3 = (u[tri[2]], v[tri[2]])
                cv2.line(vis, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.line(vis, pt2, pt3, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.line(vis, pt3, pt1, (0, 255, 255), 1, cv2.LINE_AA)
                
        out_proj = os.path.join(mesh_out_dir, f"nerf_dots_projection_{f_idx:02d}.jpg")
        cv2.imwrite(out_proj, vis)

print("✅ NeRF Dots Pipeline & Benchmark Complete!")
