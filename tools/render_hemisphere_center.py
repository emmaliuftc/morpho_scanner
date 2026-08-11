import sys
sys.path.append(".")
import os
import glob
import json
import cv2
import numpy as np

CALIB_JSON = "captures_0809_hemisphere_calibrated/calibration_results.json"
CALIB_DIR = "captures_0809_hemisphere_calibrated"
OUT_ANNOTATED_DIR = os.path.join(CALIB_DIR, "center_annotations")
os.makedirs(OUT_ANNOTATED_DIR, exist_ok=True)

with open(CALIB_JSON, "r") as f:
    cal_data = json.load(f)

K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float64)
dist_cal = np.array(cal_data["distortion_coefficients"], dtype=np.float64)
C_rot = np.array(cal_data["plate_center_mm"], dtype=np.float64)
normal = np.array(cal_data["plate_normal"], dtype=np.float64)
normal = normal / np.linalg.norm(normal)
step_size_deg = float(cal_data["step_size_deg"])

# Project C_rot into camera image coordinates
# P_cam = C_rot
cx_float = float(K_cal[0, 0] * C_rot[0] / C_rot[2] + K_cal[0, 2])
cy_float = float(K_cal[1, 1] * C_rot[1] / C_rot[2] + K_cal[1, 2])
cx, cy = int(round(cx_float)), int(round(cy_float))

# Project Z-axis endpoint (100mm along normal vector)
z_end_3d = C_rot + 100.0 * normal
ax_float = float(K_cal[0, 0] * z_end_3d[0] / z_end_3d[2] + K_cal[0, 2])
ay_float = float(K_cal[1, 1] * z_end_3d[1] / z_end_3d[2] + K_cal[1, 2])
ax, ay = int(round(ax_float)), int(round(ay_float))

# Dashed Z-axis points
z_line_pts = []
for d in np.linspace(-150.0, 150.0, 100):
    pt_3d = C_rot + d * normal
    px = int(round(K_cal[0, 0] * pt_3d[0] / pt_3d[2] + K_cal[0, 2]))
    py = int(round(K_cal[1, 1] * pt_3d[1] / pt_3d[2] + K_cal[1, 2]))
    z_line_pts.append((px, py))

images = sorted(
    glob.glob(os.path.join(CALIB_DIR, "capture_*.jpg")),
    key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[1])
)

print(f"Projected Plate Center Pixel: ({cx}, {cy})")
print(f"Annotating {len(images)} images in {CALIB_DIR}...")

for idx, img_path in enumerate(images):
    img = cv2.imread(img_path)
    
    # Draw dashed Z-axis
    for j in range(0, len(z_line_pts) - 2, 4):
        pt1 = z_line_pts[j]
        pt2 = z_line_pts[min(j + 2, len(z_line_pts) - 1)]
        cv2.line(img, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)
        
    # Z-axis arrow
    cv2.arrowedLine(img, (cx, cy), (ax, ay), (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.12)
    
    # Center crosshair
    sz = 50
    cv2.line(img, (cx - sz, cy), (cx + sz, cy), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(img, (cx, cy - sz), (cx, cy + sz), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 35, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
    
    # Overlay Text
    cv2.putText(img, f"Plate Center: ({cx}, {cy})", (50, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(img, f"Frame {idx}/64 | step={step_size_deg:.4f} deg", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "Z-axis (rotation)", (ax + 25, ay - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Overwrite in CALIB_DIR directly and also save preview in center_annotations/
    cv2.imwrite(img_path, img)
    cv2.imwrite(os.path.join(OUT_ANNOTATED_DIR, f"annotated_{idx:02d}.jpg"), img)

print(f"✅ Annotated all {len(images)} images in {CALIB_DIR}!")
