import os
import json
import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

def render_full_scale_ply_contour_overlay():
    CALIB_DIR = "captures_0810_cube_calibrated"
    CALIB_REF = os.path.join(CALIB_DIR, "calibration_results.json")
    NERF_PLY = "captures_0810_cube_nerf/export/cube_nerf_1000_pcd.ply"
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

    # Load NeRF exported 3D Point Cloud PLY
    pcd = o3d.io.read_point_cloud(NERF_PLY)
    pts_nerf = np.asarray(pcd.points) # Nx3

    pcd_center = pts_nerf.mean(axis=0)
    pts_centered = pts_nerf - pcd_center
    max_extent = np.max(np.abs(pts_centered))
    pts_unit = pts_centered / max_extent # Scale to [-1, 1] box

    frame_indices = [0, 16, 32, 48]

    # Exact physical scale extent for 40mm cube fitting red mask contour
    scale_extent = 20.0 # radius in mm (20mm radius = 40mm total length)
    t_offset = np.array([-35.0, -12.0, 20.0])

    for idx in frame_indices:
        mask_path = os.path.join(CALIB_DIR, "masks", f"mask_{idx}.png")
        img_path = os.path.join(CALIB_DIR, f"capture_{idx}.jpg")
        img = cv2.imread(img_path)
        h_img, w_img, _ = img.shape

        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gt_mask = cv2.inRange(hsv, (30, 60, 0), (95, 255, 185))

        _, gt_mask_bin = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(gt_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        largest_cnt = max(contours, key=cv2.contourArea)

        vis = img.copy()

        # 1. Draw RED detected mask contour outline
        cv2.drawContours(vis, [largest_cnt], -1, (0, 0, 255), 3, cv2.LINE_AA)

        # 2. Project full scale 3D NeRF point cloud in CYAN (BGR: 255, 255, 0)
        pts_opt = (pts_unit * scale_extent) + t_offset
        angle_deg = idx * step_size_deg
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

        out_proj = os.path.join(NERF_OUT_DIR, f"nerf_cube_exact_ply_aligned_{idx:02d}.jpg")
        cv2.imwrite(out_proj, vis)
        print(f"✅ Saved full-scale NeRF PLY overlay to {out_proj}")

if __name__ == '__main__':
    render_full_scale_ply_contour_overlay()
