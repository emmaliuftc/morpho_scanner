import sys
sys.path.append(".")
import os
import glob
import json
import cv2
import numpy as np

SRC_CALIB_JSON = "captures_0726_clay_checkboard_64_calibrated/calibration_results.json"
INPUT_FOLDER = "captures_0809_hemisphere"
OUTPUT_FOLDER = "captures_0809_hemisphere_calibrated"
REPORT_PATH = os.path.join(OUTPUT_FOLDER, "calibration_0809_hemisphere.md")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with open(SRC_CALIB_JSON, "r") as f:
    cal_data = json.load(f)

K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float64)
dist_cal = np.array(cal_data["distortion_coefficients"], dtype=np.float64)
C_rot = np.array(cal_data["plate_center_mm"], dtype=np.float64)
normal = np.array(cal_data["plate_normal"], dtype=np.float64)
step_size_deg = float(cal_data["step_size_deg"])

# Save calibration_results.json
out_calib_json = os.path.join(OUTPUT_FOLDER, "calibration_results.json")
with open(out_calib_json, "w") as f:
    json.dump(cal_data, f, indent=2)

print(f"Saved {out_calib_json}")

# Undistort all 64 capture images
image_paths = sorted(
    glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")),
    key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[1])
)

print(f"Undistorting {len(image_paths)} capture images into {OUTPUT_FOLDER}...")

for idx, img_path in enumerate(image_paths):
    raw_img = cv2.imread(img_path)
    img_undist = cv2.undistort(raw_img, K_cal, dist_cal, None, K_cal)
    out_img_path = os.path.join(OUTPUT_FOLDER, f"capture_{idx}.jpg")
    cv2.imwrite(out_img_path, img_undist)

print(f"Undistorted all {len(image_paths)} images successfully.")

# Create calibration markdown report
report_content = f"""# Turntable Trajectory Calibration: `captures_0809_hemisphere_calibrated`

## 1. Overview
This directory contains the calibrated capture dataset for the green 5cm hemisphere scan (`captures_0809_hemisphere`), processed using the high-precision camera intrinsic and extrinsic turntable trajectory calibration matching `captures_0726_clay_checkboard_64_calibrated`.

---

## 2. Calibrated Trajectory Parameters
* **Turntable Center $C_{{\\text{{rot}}}}$**: `[{C_rot[0]:.2f}, {C_rot[1]:.2f}, {C_rot[2]:.2f}]` mm
* **Rotation Axis Normal $\\hat{{n}}$**: `[{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]`
* **Angular Step Size**: ${step_size_deg:.4f}^\\circ$ per frame
* **Total Frames**: {len(image_paths)}

---

## 3. Camera Matrix $K$ & Lens Distortion
* **Focal Length**: $f_x = {K_cal[0,0]:.2f}\\text{{ px}}, f_y = {K_cal[1,1]:.2f}\\text{{ px}}$
* **Principal Point**: $c_x = {K_cal[0,2]:.1f}\\text{{ px}}, c_y = {K_cal[1,2]:.1f}\\text{{ px}}$
* **Distortion Coefficients**: `{dist_cal.tolist()}`
"""

with open(REPORT_PATH, "w") as f:
    f.write(report_content)

print(f"Saved report to {REPORT_PATH}")
