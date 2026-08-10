import cv2
import numpy as np
import glob
import os
import sys
import json
from scipy.spatial.transform import Rotation as Rot

sys.path.append("tools")
from silhouette_extractor import SilhouetteExtractor

INPUT_FOLDER = "captures_0726_clay_checkboard_64"
CALIB_JSON = "captures_0726_clay_checkboard_64_calibrated/calibration_results.json"
OUTPUT_DIR = "captures_0726_nerf_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)
images_out_dir = os.path.join(OUTPUT_DIR, "images")
masks_out_dir = os.path.join(OUTPUT_DIR, "masks")
os.makedirs(images_out_dir, exist_ok=True)
os.makedirs(masks_out_dir, exist_ok=True)

with open(CALIB_JSON, "r") as f:
    cal_data = json.load(f)

K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float64)
dist_cal = np.array(cal_data["distortion_coefficients"], dtype=np.float64)
C_rot = np.array(cal_data["plate_center_mm"], dtype=np.float64)
normal = np.array(cal_data["plate_normal"], dtype=np.float64)
normal = normal / np.linalg.norm(normal)
step_size_deg = float(cal_data["step_size_deg"])

CENTER_2D = (2166, 1145)
PLATE_RADIUS_PIXELS = 1100

# Basis setup
ref = np.array([1.0, 0.0, 0.0])
x_col = ref - np.dot(ref, normal) * normal
x_col = x_col / np.linalg.norm(x_col)
y_col = np.cross(normal, x_col)
R_cam = np.column_stack((x_col, y_col, normal))

image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")),
                     key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

scale_factor = 8
orig_w, orig_h = 4608, 2592
new_w, new_h = orig_w // scale_factor, orig_h // scale_factor

fl_x = K_cal[0, 0] / scale_factor
fl_y = K_cal[1, 1] / scale_factor
cx = K_cal[0, 2] / scale_factor
cy = K_cal[1, 2] / scale_factor

transforms_data = {
    "fl_x": fl_x,
    "fl_y": fl_y,
    "cx": cx,
    "cy": cy,
    "w": new_w,
    "h": new_h,
    "camera_model": "OPENCV",
    "k1": 0.0,
    "k2": 0.0,
    "p1": 0.0,
    "p2": 0.0,
    "frames": []
}

extractor = SilhouetteExtractor()
flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])

print(f"Preparing NeRF dataset from {len(image_paths)} images...")

for idx, img_path in enumerate(image_paths):
    basename = os.path.basename(img_path)
    stem = os.path.splitext(basename)[0]
    out_img_name = f"{stem}.png"
    out_mask_name = f"mask_{stem}.png"
    
    out_img_path = os.path.join(OUTPUT_DIR, out_img_name)
    out_mask_path = os.path.join(OUTPUT_DIR, out_mask_name)
    
    # 1. Load & Undistort Image
    raw_img = cv2.imread(img_path)
    img_undist = cv2.undistort(raw_img, K_cal, dist_cal, None, K_cal)
    
    # 2. Extract Silhouette Mask
    existing_mask_path = f"captures_0726_clay_checkboard_64_tsdf/masks/mask_{idx:02d}.png"
    if os.path.exists(existing_mask_path):
        mask = cv2.imread(existing_mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        mask = extractor.get_silhouette_mask(img_undist, CENTER_2D, PLATE_RADIUS_PIXELS)
        
    # 3. Resize Image & Mask by 8x
    img_small = cv2.resize(img_undist, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    cv2.imwrite(out_img_path, img_small)
    cv2.imwrite(out_mask_path, mask_small)
    
    # 4. Compute Pose Matrix c2w
    angle_deg = idx * step_size_deg
    angle_rad = np.deg2rad(angle_deg)
    R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
    R_eff = np.dot(R_i, R_cam)
    
    w2c = np.eye(4)
    w2c[:3, :3] = R_eff
    w2c[:3, 3] = C_rot
    
    c2w = np.linalg.inv(w2c)
    
    # Scale translation down by 150.0 for NeRF bounding box
    c2w[:3, 3] /= 150.0
    
    # Apply OpenGL Y-Z flip
    c2w_opengl = np.dot(c2w, flip_yz)
    
    frame_entry = {
        "file_path": out_img_name,
        "mask_path": out_mask_name,
        "transform_matrix": c2w_opengl.tolist()
    }
    transforms_data["frames"].append(frame_entry)
    
    if (idx + 1) % 16 == 0 or idx == len(image_paths) - 1:
        print(f"Processed frame {idx+1}/{len(image_paths)}: {out_img_name}")

# Save transforms_8.json
transforms_json_path = os.path.join(OUTPUT_DIR, "transforms_8.json")
with open(transforms_json_path, "w") as f:
    json.dump(transforms_data, f, indent=4)

print(f"\n✅ NeRF dataset preparation complete! Saved to: {transforms_json_path}")
