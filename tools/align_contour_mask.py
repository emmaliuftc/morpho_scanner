import os
import json
import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

def run_mask_contour_alignment():
    CALIB_DIR = "captures_0810_cube_calibrated"
    CALIB_REF = os.path.join(CALIB_DIR, "calibration_results.json")
    NERF_OUT_DIR = "captures_0810_cube_nerf"
    os.makedirs(NERF_OUT_DIR, exist_ok=True)

    with open(CALIB_REF, "r") as f:
        cal_data = json.load(f)

    K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float64)
    C_rot = np.array(cal_data["plate_center_mm"], dtype=np.float64) # mm
    normal = np.array(cal_data["plate_normal"], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    step_size_deg = float(cal_data["step_size_deg"])

    # Frame basis
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal) * normal
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal, x_col)
    R_cam = np.column_stack((x_col, y_col, normal))

    # 4cm cube CAD surface grid in mm (Z from 0 to 40mm)
    grid_pts = []
    x = np.linspace(-20, 20, 30)
    y = np.linspace(-20, 20, 30)
    z = np.linspace(0, 40, 30)
    X, Y = np.meshgrid(x, y)
    grid_pts.append(np.column_stack([X.ravel(), Y.ravel(), np.full_like(X.ravel(), 40.0)]))
    grid_pts.append(np.column_stack([X.ravel(), Y.ravel(), np.full_like(X.ravel(), 0.0)]))
    X, Z = np.meshgrid(x, z)
    grid_pts.append(np.column_stack([X.ravel(), np.full_like(X.ravel(), -20.0), Z.ravel()]))
    grid_pts.append(np.column_stack([X.ravel(), np.full_like(X.ravel(), 20.0), Z.ravel()]))
    Y, Z = np.meshgrid(y, z)
    grid_pts.append(np.column_stack([np.full_like(Y.ravel(), -20.0), Y.ravel(), Z.ravel()]))
    grid_pts.append(np.column_stack([np.full_like(Y.ravel(), 20.0), Y.ravel(), Z.ravel()]))
    pts_3d_raw = np.vstack(grid_pts)

    frame_indices = [0, 16, 32, 48]

    for idx in frame_indices:
        mask_path = os.path.join(CALIB_DIR, "masks", f"mask_{idx}.png")
        img_path = os.path.join(CALIB_DIR, f"capture_{idx}.jpg")
        img = cv2.imread(img_path)
        h_img, w_img, _ = img.shape

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (30, 60, 0), (95, 255, 185))

        # Extract 2D contour from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        largest_cnt = max(contours, key=cv2.contourArea)

        # Create exact 2D boundary distance transform map
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [largest_cnt], -1, 255, 2)
        dist_map = cv2.distanceTransform(cv2.bitwise_not(contour_mask), cv2.DIST_L2, 5)

        # Fit 3D points inside contour (tx, ty, tz=0, rot_deg=0, scale=1.0)
        def loss_func(params):
            tx, ty, rot_deg, scale = params
            pts = (pts_3d_raw * scale) + np.array([tx, ty, 0.0])
            angle_deg = (idx * step_size_deg) + rot_deg
            angle_rad = np.deg2rad(angle_deg)
            R_i = Rot.from_rotvec(-angle_rad * np.array([0, 0, 1.0])).as_matrix()

            pts_rot = R_i @ pts.T
            cam_pts = (R_cam @ pts_rot) + C_rot.reshape(3, 1)
            zc = cam_pts[2, :]

            u = np.round((K_cal[0, 0] * cam_pts[0, :] / zc) + K_cal[0, 2]).astype(int)
            v = np.round((K_cal[1, 1] * cam_pts[1, :] / zc) + K_cal[1, 2]).astype(int)
            valid = (zc > 10.0) & (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)

            if valid.sum() == 0:
                return 1e9
            return np.mean(dist_map[v[valid], u[valid]]**2)

        bounds = [(-60.0, 0.0), (-40.0, 10.0), (-45.0, 45.0), (0.85, 1.15)]
        init_params = [-35.0, -12.0, 0.0, 1.0]

        res = minimize(loss_func, init_params, bounds=bounds, method='L-BFGS-B')
        opt_tx, opt_ty, opt_rot_deg, opt_scale = res.x

        print(f"✅ Frame {idx:02d} Contour Alignment:")
        print(f"   Position Offset: [{opt_tx:.2f}, {opt_ty:.2f}, 0.00] mm")
        print(f"   Rotation Offset: {opt_rot_deg:.2f}°")
        print(f"   Scale Factor: {opt_scale:.4f}x ({opt_scale * 40.0:.2f} mm)")

        # Render overlay: RED detected contour line + CYAN 3D points
        vis = img.copy()

        # 1. Draw detected 2D mask contour in thick RED line (BGR: 0, 0, 255)
        cv2.drawContours(vis, [largest_cnt], -1, (0, 0, 255), 3, cv2.LINE_AA)

        # 2. Draw re-aligned 3D point cloud cube points in CYAN dots (BGR: 255, 255, 0)
        pts_opt = (pts_3d_raw * opt_scale) + np.array([opt_tx, opt_ty, 0.0])
        angle_deg = (idx * step_size_deg) + opt_rot_deg
        angle_rad = np.deg2rad(angle_deg)
        R_i = Rot.from_rotvec(-angle_rad * np.array([0, 0, 1.0])).as_matrix()

        pts_rot = R_i @ pts_opt.T
        cam_pts = (R_cam @ pts_rot) + C_rot.reshape(3, 1)
        zc = cam_pts[2, :]

        u = np.round((K_cal[0, 0] * cam_pts[0, :] / zc) + K_cal[0, 2]).astype(int)
        v = np.round((K_cal[1, 1] * cam_pts[1, :] / zc) + K_cal[1, 2]).astype(int)
        valid = (zc > 10.0) & (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)

        for pu, pv in zip(u[valid][::2], v[valid][::2]):
            cv2.circle(vis, (pu, pv), 2, (255, 255, 0), -1, cv2.LINE_AA)

        out_proj = os.path.join(NERF_OUT_DIR, f"nerf_cube_contour_aligned_{idx:02d}.jpg")
        cv2.imwrite(out_proj, vis)
        print(f"✅ Saved contour overlay & 3D re-aligned projection to {out_proj}")

if __name__ == '__main__':
    run_mask_contour_alignment()
