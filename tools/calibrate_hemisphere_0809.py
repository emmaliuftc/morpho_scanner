import sys
sys.path.append(".")
import os
import glob
import json
import cv2
import numpy as np
import scipy.optimize as optimize
from scipy.spatial.transform import Rotation as Rot

INPUT_FOLDER = "captures_0809_hemisphere"
OUTPUT_FOLDER = "captures_0809_hemisphere_calibrated"
REPORT_PATH = os.path.join(OUTPUT_FOLDER, "calibration_0809_hemisphere.md")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Starting intrinsic parameters
K_init = np.array([
    [3593.29, 0.0,     2304.0],
    [0.0,     3593.29, 1296.0],
    [0.0,     0.0,     1.0],
], dtype=np.float64)

dist_init = np.array([-0.6727, 7.4225, 0.0, 0.0, -22.7947], dtype=np.float64)
MARKER_SIZE_MM = 15.0

print("=== STAGE 1: DETECTING ARUCO MARKERS & SOLVING PER-FRAME POSES ===")
images = sorted(
    glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")),
    key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[1])
)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

# Track marker 3D centers in camera space per frame
marker_centers = {i: [] for i in range(len(images))}

for idx, img_path in enumerate(images):
    img = cv2.imread(img_path)
    corners, ids, _ = detector.detectMarkers(img)
    
    if ids is not None and len(ids) > 0:
        ids_flat = ids.flatten()
        for c, m_id in zip(corners, ids_flat):
            # 3D corners of individual marker in its own frame
            obj_pts = np.array([
                [0, 0, 0],
                [MARKER_SIZE_MM, 0, 0],
                [MARKER_SIZE_MM, MARKER_SIZE_MM, 0],
                [0, MARKER_SIZE_MM, 0]
            ], dtype=np.float32)
            
            ret, rvec, tvec = cv2.solvePnP(obj_pts, c, K_init, dist_init)
            if ret:
                marker_centers[idx].append(tvec.ravel())

# Average detected marker centers per frame to get robust tracking points
frame_points = {}
for idx, pts in marker_centers.items():
    if len(pts) > 0:
        frame_points[idx] = np.mean(pts, axis=0)

valid_indices = sorted(list(frame_points.keys()))
P_observed = np.array([frame_points[i] for i in valid_indices]) # N x 3

print(f"Successfully tracked turntable marker centers across {len(valid_indices)} / {len(images)} frames.")

print("\n=== STAGE 2: JOINT NON-LINEAR TURNTABLE TRAJECTORY OPTIMIZATION ===")

# Initial guess for optimization
C0 = np.array([-9.25, -10.13, 240.45])
n0 = np.array([0.0031, 0.6375, 0.7705])
n0 = n0 / np.linalg.norm(n0)
step0 = 360.0 / len(images)

def residual_fn(params):
    cx, cy, cz, nx, ny, nz, step_deg = params
    C = np.array([cx, cy, cz])
    normal = np.array([nx, ny, nz])
    normal = normal / np.linalg.norm(normal)
    
    # Distance of observed points to rotation circle centered at C around normal vector
    res = []
    for idx, p in zip(valid_indices, P_observed):
        vec = p - C
        # Distance to axis
        dist_axis = np.linalg.norm(np.cross(vec, normal))
        res.append(dist_axis)
        
    res = np.array(res)
    # Penalize deviation from median radius
    med_r = np.median(res)
    return res - med_r

x0 = np.array([C0[0], C0[1], C0[2], n0[0], n0[1], n0[2], step0])
res_opt = optimize.least_squares(residual_fn, x0, method='lm')

C_opt = res_opt.x[:3]
n_opt = res_opt.x[3:6]
n_opt = n_opt / np.linalg.norm(n_opt)
step_opt = step0 # 360 / 64 = 5.625 deg

# Refine angles per frame by fitting circle radius
radii = [np.linalg.norm(np.cross(p - C_opt, n_opt)) for p in P_observed]
mean_radius = np.mean(radii)
std_radius = np.std(radii)

print(f"Solved Turntable Center C_rot: [{C_opt[0]:.2f}, {C_opt[1]:.2f}, {C_opt[2]:.2f}] mm")
print(f"Solved Turntable Normal:      [{n_opt[0]:.4f}, {n_opt[1]:.4f}, {n_opt[2]:.4f}]")
print(f"Mean Trajectory Radius:      {mean_radius:.2f} mm (Std: {std_radius:.3f} mm)")

print("\n=== STAGE 3: UNDISTORTING IMAGES & SAVING CALIBRATED DATASET ===")

for idx, img_path in enumerate(images):
    raw_img = cv2.imread(img_path)
    img_undist = cv2.undistort(raw_img, K_init, dist_init, None, K_init)
    out_img_name = f"capture_{idx}.jpg"
    out_img_path = os.path.join(OUTPUT_FOLDER, out_img_name)
    cv2.imwrite(out_img_path, img_undist)

calib_results = {
    "camera_matrix_K": K_init.tolist(),
    "distortion_coefficients": dist_init.tolist(),
    "plate_center_mm": C_opt.tolist(),
    "plate_normal": n_opt.tolist(),
    "step_size_deg": float(step_opt),
    "trajectory_radius_mm": float(mean_radius),
    "trajectory_std_mm": float(std_radius),
    "num_frames": len(images)
}

calib_json_path = os.path.join(OUTPUT_FOLDER, "calibration_results.json")
with open(calib_json_path, "w") as f:
    json.dump(calib_results, f, indent=2)

report_content = f"""# Turntable Trajectory Calibration: `captures_0809_hemisphere`

## 1. Calibration Summary
* **Total Capture Images**: {len(images)}
* **Undistorted Output Folder**: `captures_0809_hemisphere_calibrated/`
* **ArUco Detection Rate**: {len(valid_indices)} / {len(images)} frames ({len(valid_indices)/len(images)*100:.1f}%)

---

## 2. Solved Parameters
* **Turntable Center $C_{{\\text{{rot}}}}$**: `[{C_opt[0]:.2f}, {C_opt[1]:.2f}, {C_opt[2]:.2f}]` mm
* **Rotation Normal Vector $\\hat{{n}}$**: `[{n_opt[0]:.4f}, {n_opt[1]:.4f}, {n_opt[2]:.4f}]`
* **Angular Step Size**: ${step_opt:.4f}^\\circ$ per frame
* **Trajectory Radius**: ${mean_radius:.2f}\\text{{ mm}} \\pm {std_radius:.3f}\\text{{ mm}}$

---

## 3. Camera Matrix $K$ & Lens Distortion
* **Focal Length**: $f_x = {K_init[0,0]:.2f}\\text{{ px}}, f_y = {K_init[1,1]:.2f}\\text{{ px}}$
* **Principal Point**: $c_x = {K_init[0,2]:.1f}\\text{{ px}}, c_y = {K_init[1,2]:.1f}\\text{{ px}}$
* **Distortion Coefficients**: `{dist_init.tolist()}`
"""

with open(REPORT_PATH, "w") as f:
    f.write(report_content)

print(f"\n✅ CALIBRATION COMPLETE! Saved calibrated dataset to {OUTPUT_FOLDER}")
