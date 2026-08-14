import cv2
import numpy as np
import json
import os
import shutil

# Load calibration
calib_path = "captures_8-13_three_flat_calibrated/calibration_results.json"
with open(calib_path, "r") as f:
    cal = json.load(f)

C_rot = np.array(cal['plate_center_mm'], dtype=np.float64)
normal = np.array(cal['plate_normal'], dtype=np.float64)

# The images are scaled by 4
scale = 4.0
K = np.array(cal['camera_matrix_K'], dtype=np.float64)
K[:2, :] /= scale
dist = np.zeros(5, dtype=np.float64)  # Undistorted!

img_path = "data/three_flat_real_calib/images_4/capture_0.png"
if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
    exit(1)

img = cv2.imread(img_path)

rvec0 = np.zeros((3, 1), dtype=np.float64)
tvec0 = np.zeros((3, 1), dtype=np.float64)

# Project center
center_2d, _ = cv2.projectPoints(
    C_rot.reshape(1, 1, 3), rvec0, tvec0, K, dist
)
cx = int(round(center_2d[0][0][0]))
cy = int(round(center_2d[0][0][1]))

# Z-axis line
s_vals = np.linspace(-40, 120, 300)
z_pts_3d = np.array([C_rot + s * normal for s in s_vals], dtype=np.float64)
z_pts_2d, _ = cv2.projectPoints(z_pts_3d, rvec0, tvec0, K, dist)
z_pts_px = z_pts_2d.reshape(-1, 2).astype(np.int32)

# Arrow tip
arrow_3d = (C_rot + 80 * normal).reshape(1, 1, 3).astype(np.float64)
arrow_2d, _ = cv2.projectPoints(arrow_3d, rvec0, tvec0, K, dist)
ax = int(round(arrow_2d[0][0][0]))
ay = int(round(arrow_2d[0][0][1]))

# Draw
for j in range(0, len(z_pts_px) - 2, 4):
    pt1 = tuple(z_pts_px[j])
    pt2 = tuple(z_pts_px[min(j + 2, len(z_pts_px) - 1)])
    cv2.line(img, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

cv2.arrowedLine(img, (cx, cy), (ax, ay), (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.12)

sz = 20
cv2.line(img, (cx-sz, cy), (cx+sz, cy), (0, 255, 0), 2, cv2.LINE_AA)
cv2.line(img, (cx, cy-sz), (cx, cy+sz), (0, 255, 0), 2, cv2.LINE_AA)
cv2.circle(img, (cx, cy), 15, (0, 255, 0), 2, cv2.LINE_AA)
cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)

out_path = "undistorted_axis_vis.png"
cv2.imwrite(out_path, img)
print("Saved to", out_path)

# Copy to artifacts so user can see it
art_path = "/home/coding/.gemini/antigravity-cli/brain/013018fb-2735-47e1-ad61-549c02a8a1c3/scratch/undistorted_axis_vis.png"
os.makedirs(os.path.dirname(art_path), exist_ok=True)
shutil.copy(out_path, art_path)
print("Copied to artifacts.")
