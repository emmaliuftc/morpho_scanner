import os
import json
import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

def run_2d_3d_alignment():
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

    # 40mm cube surface grid centered at origin on turntable plate Z=0 to Z=40mm
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

    # Distance maps for 2D silhouette loss
    frame_indices = [0, 16, 32, 48]
    dist_maps = {}
    
    for idx in frame_indices:
        mask_path = os.path.join(CALIB_DIR, "masks", f"mask_{idx}.png")
        if not os.path.exists(mask_path):
            img_path = os.path.join(CALIB_DIR, f"capture_{idx}.jpg")
            img = cv2.imread(img_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (30, 60, 0), (95, 255, 185))
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
        inv_mask = cv2.bitwise_not(mask)
        dist_map = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
        dist_maps[idx] = dist_map

    # Lock Z=0 to turntable plane: optimize [tx, ty, rot_offset_deg, scale]
    def loss_func(params):
        tx, ty, rot_offset_deg, scale = params
        total_loss = 0.0
        num_eval = 0
        
        pts = (pts_3d_raw * scale) + np.array([tx, ty, 0.0])
        
        for idx in frame_indices:
            angle_deg = (idx * step_size_deg) + rot_offset_deg
            angle_rad = np.deg2rad(angle_deg)
            R_i = Rot.from_rotvec(-angle_rad * np.array([0, 0, 1.0])).as_matrix()
            
            pts_rot = R_i @ pts.T
            cam_pts = (R_cam @ pts_rot) + C_rot.reshape(3, 1)
            
            zc = cam_pts[2, :]
            valid = zc > 10.0
            
            u = np.round((K_cal[0, 0] * cam_pts[0, :] / zc) + K_cal[0, 2]).astype(int)
            v = np.round((K_cal[1, 1] * cam_pts[1, :] / zc) + K_cal[1, 2]).astype(int)
            
            dist_map = dist_maps[idx]
            h, w = dist_map.shape
            
            valid_uv = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u_v = u[valid_uv]
            v_v = v[valid_uv]
            
            if len(u_v) > 0:
                errs = dist_map[v_v, u_v]
                total_loss += np.mean(errs**2)
                num_eval += 1
                
        if num_eval == 0:
            return 1e9
        return total_loss / num_eval

    # Bounded optimization: tx, ty in [-60, 60], rot in [0, 90], scale in [0.9, 1.1]
    bounds = [(-60.0, 60.0), (-60.0, 60.0), (0.0, 90.0), (0.90, 1.10)]
    init_params = [-32.0, -18.0, 0.0, 1.0]
    
    print("Running Turntable-Plane 2D-3D Silhouette Pose Optimization (L-BFGS-B)...")
    res = minimize(loss_func, init_params, bounds=bounds, method='L-BFGS-B')
    
    opt_tx, opt_ty, opt_rot_deg, opt_scale = res.x
    print(f"✅ Turntable-Plane Optimization Converged!")
    print(f"  Optimal Translation (tx, ty): [{opt_tx:.2f}, {opt_ty:.2f}] mm")
    print(f"  Optimal Rotation Angle Offset: {opt_rot_deg:.2f}°")
    print(f"  Optimal Scale Factor: {opt_scale:.4f}x ({opt_scale * 40.0:.2f} mm cube extent)")

    # Render aligned projections with Z=0 plane locked
    pts_opt = (pts_3d_raw * opt_scale) + np.array([opt_tx, opt_ty, 0.0])
    
    for idx in frame_indices:
        angle_deg = (idx * step_size_deg) + opt_rot_deg
        angle_rad = np.deg2rad(angle_deg)
        R_i = Rot.from_rotvec(-angle_rad * np.array([0, 0, 1.0])).as_matrix()
        
        pts_rot = R_i @ pts_opt.T
        cam_pts = (R_cam @ pts_rot) + C_rot.reshape(3, 1)
        
        zc = cam_pts[2, :]
        u = np.round((K_cal[0, 0] * cam_pts[0, :] / zc) + K_cal[0, 2]).astype(int)
        v = np.round((K_cal[1, 1] * cam_pts[1, :] / zc) + K_cal[1, 2]).astype(int)
        
        img_path = os.path.join(CALIB_DIR, f"capture_{idx}.jpg")
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        valid = (zc > 10.0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        vis = img.copy()
        for pu, pv in zip(u[valid][::2], v[valid][::2]):
            cv2.circle(vis, (pu, pv), 2, (0, 255, 255), -1, cv2.LINE_AA) # Yellow dots
            
        out_proj = os.path.join(NERF_OUT_DIR, f"nerf_cube_aligned_projection_{idx:02d}.jpg")
        cv2.imwrite(out_proj, vis)
        print(f"✅ Saved clean aligned projection overlay to {out_proj}")

if __name__ == '__main__':
    run_2d_3d_alignment()
