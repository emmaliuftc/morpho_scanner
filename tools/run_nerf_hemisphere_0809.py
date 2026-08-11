import sys
sys.path.append(".")
import os
import glob
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d

# Paths & Setup
CALIB_JSON = "captures_0809_hemisphere_calibrated/calibration_results.json"
INPUT_FOLDER = "captures_0809_hemisphere_calibrated"
NERF_DATASET_DIR = "captures_0809_hemisphere_nerf_dataset"
STL_PATH = "3d_files/hemisphere_5cm.stl"
DOCS_REPORT_PATH = "docs/hemisphere_comparison_0809.md"

os.makedirs(NERF_DATASET_DIR, exist_ok=True)
images_8x_dir = os.path.join(NERF_DATASET_DIR, "images_8")
masks_8x_dir = os.path.join(NERF_DATASET_DIR, "masks_8")
os.makedirs(images_8x_dir, exist_ok=True)
os.makedirs(masks_8x_dir, exist_ok=True)

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

print(f"=== STAGE 1: DOWNSCALING & PREPARING NERF DATASET ({len(image_paths)} FRAMES) ===")
downscale = 8
fx = float(K_cal[0, 0] / downscale)
fy = float(K_cal[1, 1] / downscale)
cx = float(K_cal[0, 2] / downscale)
cy = float(K_cal[1, 2] / downscale)
orig_w, orig_h = 4608, 2592
w = int(orig_w / downscale) # 576
h = int(orig_h / downscale) # 324

flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])

transforms = {
    "fl_x": fx,
    "fl_y": fy,
    "cx": cx,
    "cy": cy,
    "w": w,
    "h": h,
    "camera_model": "OPENCV",
    "k1": float(dist_cal[0]),
    "k2": float(dist_cal[1]),
    "p1": float(dist_cal[2]),
    "p2": float(dist_cal[3]),
    "k3": float(dist_cal[4]),
    "frames": []
}

masks_orig_dir = os.path.join(INPUT_FOLDER, "masks")

for idx, img_p in enumerate(image_paths):
    raw_img = cv2.imread(img_p)
    img_ds = cv2.resize(raw_img, (w, h), interpolation=cv2.INTER_AREA)
    out_img_p = os.path.join(images_8x_dir, f"frame_{idx:05d}.jpg")
    cv2.imwrite(out_img_p, img_ds)
    
    mask_orig_p = os.path.join(masks_orig_dir, f"mask_{idx:02d}.png")
    mask_orig = cv2.imread(mask_orig_p, cv2.IMREAD_GRAYSCALE)
    mask_ds = cv2.resize(mask_orig, (w, h), interpolation=cv2.INTER_NEAREST)
    out_mask_p = os.path.join(masks_8x_dir, f"mask_{idx:05d}.png")
    cv2.imwrite(out_mask_p, mask_ds)
    
    R, t = get_camera_pose_for_frame(idx)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    
    c2w = np.linalg.inv(w2c)
    c2w[:3, 3] /= 150.0 # Scale by 150.0
    c2w_opengl = np.dot(c2w, flip_yz)
    
    frame_entry = {
        "file_path": f"images_8/frame_{idx:05d}.jpg",
        "mask_path": f"masks_8/mask_{idx:05d}.png",
        "transform_matrix": c2w_opengl.tolist()
    }
    transforms["frames"].append(frame_entry)

json_p = os.path.join(NERF_DATASET_DIR, "transforms_8.json")
with open(json_p, "w") as f:
    json.dump(transforms, f, indent=2)

print(f"Saved NeRF transforms JSON to: {json_p}")
