import sys
sys.path.append(".")
import os
import glob
import json
import cv2
import numpy as np

SRC_DIR = "captures_0809_hemisphere_dots"
CALIB_JSON_SRC = "captures_0726_clay_checkboard_64_calibrated/calibration_results.json"
OUT_DIR = "captures_0809_hemisphere_dots_calibrated"
MASKS_DIR = os.path.join(OUT_DIR, "masks")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

with open(CALIB_JSON_SRC, "r") as f:
    cal_data = json.load(f)

# Save calibration_results.json in OUT_DIR
with open(os.path.join(OUT_DIR, "calibration_results.json"), "w") as f:
    json.dump(cal_data, f, indent=2)

K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float64)
dist_cal = np.array(cal_data["distortion_coefficients"], dtype=np.float64)
C_rot = np.array(cal_data["plate_center_mm"], dtype=np.float64)
normal = np.array(cal_data["plate_normal"], dtype=np.float64)
normal = normal / np.linalg.norm(normal)
step_size_deg = float(cal_data["step_size_deg"])

# Project C_rot (2166, 1145)
cx = int(round(K_cal[0, 0] * C_rot[0] / C_rot[2] + K_cal[0, 2]))
cy = int(round(K_cal[1, 1] * C_rot[1] / C_rot[2] + K_cal[1, 2]))

# Project Z-axis endpoint
z_end_3d = C_rot + 100.0 * normal
ax = int(round(K_cal[0, 0] * z_end_3d[0] / z_end_3d[2] + K_cal[0, 2]))
ay = int(round(K_cal[1, 1] * z_end_3d[1] / z_end_3d[2] + K_cal[1, 2]))

z_line_pts = []
for d in np.linspace(-150.0, 150.0, 100):
    pt_3d = C_rot + d * normal
    px = int(round(K_cal[0, 0] * pt_3d[0] / pt_3d[2] + K_cal[0, 2]))
    py = int(round(K_cal[1, 1] * pt_3d[1] / pt_3d[2] + K_cal[1, 2]))
    z_line_pts.append((px, py))

images = sorted(
    glob.glob(os.path.join(SRC_DIR, "capture_*.jpg")),
    key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[1])
)

print(f"Building calibrated dataset for {len(images)} images in {OUT_DIR}...")

for idx, img_path in enumerate(images):
    raw_img = cv2.imread(img_path)
    # Undistort
    img = cv2.undistort(raw_img, K_cal, dist_cal, None, K_cal)
    
    # Save undistorted clean image for TSDF / NeRF dataset
    undist_clean_path = os.path.join(OUT_DIR, f"capture_{idx}.jpg")
    cv2.imwrite(undist_clean_path, img)
    
    # Draw annotations
    annotated = img.copy()
    for j in range(0, len(z_line_pts) - 2, 4):
        pt1 = z_line_pts[j]
        pt2 = z_line_pts[min(j + 2, len(z_line_pts) - 1)]
        cv2.line(annotated, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)
        
    cv2.arrowedLine(annotated, (cx, cy), (ax, ay), (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.12)
    sz = 50
    cv2.line(annotated, (cx - sz, cy), (cx + sz, cy), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(annotated, (cx, cy - sz), (cx, cy + sz), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(annotated, (cx, cy), 35, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
    
    cv2.putText(annotated, f"Plate Center: ({cx}, {cy})", (50, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(annotated, f"Frame {idx}/64 | step={step_size_deg:.4f} deg", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(annotated, "Z-axis (rotation)", (ax + 25, ay - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2, cv2.LINE_AA)
                
    annotated_path = os.path.join(OUT_DIR, f"annotated_capture_{idx}.jpg")
    cv2.imwrite(annotated_path, annotated)

print(f"✅ Created calibrated dataset in {OUT_DIR}!")
