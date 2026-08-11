import os
import json
import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

def auto_detect_and_align_view(frame_idx=0):
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

    # Load image and extract 2D green cube mask
    img_path = os.path.join(CALIB_DIR, f"capture_{frame_idx}.jpg")
    img = cv2.imread(img_path)
    h_img, w_img, _ = img.shape

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (30, 60, 0), (95, 255, 185))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Compute distance transform from mask boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"❌ No object mask detected in frame {frame_idx}")
        return

    largest_cnt = max(contours, key=cv2.contourArea)
    mask_cnt = np.zeros_like(mask)
    cv2.drawContours(mask_cnt, [largest_cnt], -1, 255, -1)
    boundary_img = cv2.Canny(mask_cnt, 100, 200)
    dist_map = cv2.distanceTransform(cv2.bitwise_not(boundary_img), cv2.DIST_L2, 5)

    # 4cm cube CAD surface points in mm
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

    # Coarse search over (tx, ty, angle_deg)
    best_loss = 1e9
    best_params = (0, 0, 0, 1.0)
    
    print("Running coarse 2D-3D silhouette registration search...")
    for test_angle in np.arange(0.0, 90.0, 15.0):
        angle_rad = np.deg2rad(test_angle)
        R_i = Rot.from_rotvec(-angle_rad * np.array([0, 0, 1.0])).as_matrix()
        
        for tx in np.linspace(-45, -15, 7):
            for ty in np.linspace(-25, 5, 7):
                pts = pts_3d_raw + np.array([tx, ty, 13.0])
                pts_rot = R_i @ pts.T
                cam_pts = (R_cam @ pts_rot) + C_rot.reshape(3, 1)
                zc = cam_pts[2, :]
                
                u = np.round((K_cal[0, 0] * cam_pts[0, :] / zc) + K_cal[0, 2]).astype(int)
                v = np.round((K_cal[1, 1] * cam_pts[1, :] / zc) + K_cal[1, 2]).astype(int)
                valid = (zc > 10.0) & (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
                
                if valid.sum() > 0:
                    loss = np.mean(dist_map[v[valid], u[valid]]**2)
                    if loss < best_loss:
                        best_loss = loss
                        best_params = (tx, ty, test_angle, 1.0)

    print(f"✅ Coarse Alignment Result: tx={best_params[0]:.1f}, ty={best_params[1]:.1f}, angle={best_params[2]:.1f}°")

    # Fine-tune optimization over [tx, ty, tz, angle_deg, scale]
    def obj_func(params):
        tx, ty, tz, test_angle, scale = params
        pts = (pts_3d_raw * scale) + np.array([tx, ty, tz])
        angle_rad = np.deg2rad(test_angle)
        R_i = Rot.from_rotvec(-angle_rad * np.array([0, 0, 1.0])).as_matrix()
        
        pts_rot = R_i @ pts.T
        cam_pts = (R_cam @ pts_rot) + C_rot.reshape(3, 1)
        zc = cam_pts[2, :]
        valid = zc > 10.0
        
        u = np.round((K_cal[0, 0] * cam_pts[0, :] / zc) + K_cal[0, 2]).astype(int)
        v = np.round((K_cal[1, 1] * cam_pts[1, :] / zc) + K_cal[1, 2]).astype(int)
        
        valid_uv = valid & (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
        if valid_uv.sum() == 0:
            return 1e9
        return np.mean(dist_map[v[valid_uv], u[valid_uv]]**2)

    bounds = [(-60.0, 0.0), (-40.0, 10.0), (0.0, 30.0), (0.0, 90.0), (0.85, 1.15)]
    init_p = [best_params[0], best_params[1], 13.0, best_params[2], 1.0]
    
    print("Fine-tuning 3D scale, position, and orientation angle...")
    res = minimize(obj_func, init_p, bounds=bounds, method='L-BFGS-B')
    opt_tx, opt_ty, opt_tz, opt_angle, opt_scale = res.x

    print(f"✅ Auto-Detected View Orientation & Scale Alignment Complete!")
    print(f"  Detected View Angle: {opt_angle:.2f}°")
    print(f"  Position Offset (tx, ty, tz): [{opt_tx:.2f}, {opt_ty:.2f}, {opt_tz:.2f}] mm")
    print(f"  Fitted Scale Factor: {opt_scale:.4f}x ({opt_scale * 40.0:.2f} mm cube extent)")

    # Render final auto-aligned overlay across frames
    for f_idx in [0, 16, 32, 48]:
        frame_angle_deg = (f_idx * step_size_deg) + opt_angle
        angle_rad = np.deg2rad(frame_angle_deg)
        R_i = Rot.from_rotvec(-angle_rad * np.array([0, 0, 1.0])).as_matrix()
        
        pts_final = (pts_3d_raw * opt_scale) + np.array([opt_tx, opt_ty, opt_tz])
        pts_rot = R_i @ pts_final.T
        cam_pts = (R_cam @ pts_rot) + C_rot.reshape(3, 1)
        zc = cam_pts[2, :]
        
        u = np.round((K_cal[0, 0] * cam_pts[0, :] / zc) + K_cal[0, 2]).astype(int)
        v = np.round((K_cal[1, 1] * cam_pts[1, :] / zc) + K_cal[1, 2]).astype(int)
        valid = (zc > 10.0) & (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
        
        f_img_path = os.path.join(CALIB_DIR, f"capture_{f_idx}.jpg")
        f_img = cv2.imread(f_img_path)
        vis = f_img.copy()
        
        for pu, pv in zip(u[valid][::2], v[valid][::2]):
            cv2.circle(vis, (pu, pv), 2, (0, 255, 255), -1, cv2.LINE_AA) # Yellow dots
            
        out_proj = os.path.join(NERF_OUT_DIR, f"nerf_cube_auto_aligned_frame_{f_idx:02d}.jpg")
        cv2.imwrite(out_proj, vis)
        print(f"✅ Saved auto-aligned projection overlay to {out_proj}")

if __name__ == '__main__':
    auto_detect_and_align_view(0)
