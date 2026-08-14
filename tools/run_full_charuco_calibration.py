#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os
import json
import scipy.optimize as optimize
from scipy.spatial.transform import Rotation as Rot

# Configuration
IMAGES_FOLDER = "captures_8-13_calibration"
OUTPUT_FOLDER = "captures_8-13_calibration_results"
SQUARE_SIZE_MM = 15.0
MARKER_SIZE_MM = 9.0
DICT_ID = cv2.aruco.DICT_4X4_50
GRID_SIZE = (11, 8)

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    images = sorted(glob.glob(os.path.join(IMAGES_FOLDER, "*.jpg")), 
                    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    print(f"Found {len(images)} images in {IMAGES_FOLDER}.")

    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    board = cv2.aruco.CharucoBoard(GRID_SIZE, SQUARE_SIZE_MM, MARKER_SIZE_MM, dictionary)
    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    
    objpoints = []
    imgpoints = []
    image_shape = None
    
    all_corners = {}
    all_ids = {}
    
    print("Detecting ChArUco corners...")
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]
            
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        if charuco_ids is not None and len(charuco_ids) >= 4:
            board_corners_3d = np.asarray(board.getChessboardCorners())
            obj_pts = board_corners_3d[charuco_ids.flatten()]
            objpoints.append(obj_pts.astype(np.float32))
            imgpoints.append(charuco_corners.astype(np.float32))
            
            all_corners[idx] = charuco_corners
            all_ids[idx] = charuco_ids

    print(f"Detected corners in {len(objpoints)} frames.")
    
    if len(objpoints) < 5:
        raise ValueError("Insufficient frames for calibration.")

    # 1. Calibrate Camera Intrinsics
    print("\n--- STEP 1: CAMERA CALIBRATION ---")
    mtx_guess = np.array([
        [3000.0, 0.0, image_shape[0] / 2.0],
        [0.0, 3000.0, image_shape[1] / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    dist_guess = np.zeros(5, dtype=np.float32)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | 
             cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST)
             
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, mtx_guess, dist_guess, flags=flags
    )
    print(f"Reprojection Error: {ret:.4f} px")
    print(f"K:\n{K}")
    print(f"DIST:\n{dist}")
    
    # 2. Extract per-frame poses (board origin in camera space)
    print("\n--- STEP 2: POSE ESTIMATION ---")
    tvecs_raw = {}
    rvecs_raw = {}
    for idx, charuco_corners in all_corners.items():
        charuco_ids = all_ids[idx]
        obj_pts = board_corners_3d[charuco_ids.flatten()]
        ret_pnp, rvec, tvec = cv2.solvePnP(obj_pts, charuco_corners, K, dist)
        if ret_pnp:
            tvecs_raw[idx] = tvec.ravel()
            rvecs_raw[idx] = rvec.ravel()
            
    # Remove outliers
    tvecs_filtered = {idx: t for idx, t in tvecs_raw.items() if t[2] > 100.0}
    tvecs_arr = np.array(list(tvecs_filtered.values()))
    
    # 3. Trajectory Fitting
    print("\n--- STEP 3: TURNTABLE TRAJECTORY FITTING ---")
    mean_t = np.mean(tvecs_arr, axis=0)
    centered_t = tvecs_arr - mean_t
    U, S, Vt = np.linalg.svd(centered_t)
    normal_init = Vt[2, :]
    if normal_init[2] < 0:
        normal_init = -normal_init
        
    u_axis = Vt[0, :]
    v_axis = Vt[1, :]
    proj_u = centered_t @ u_axis
    proj_v = centered_t @ v_axis
    
    A = np.column_stack([2 * proj_u, 2 * proj_v, np.ones(len(proj_u))])
    B = proj_u**2 + proj_v**2
    uc, vc, _ = np.linalg.lstsq(A, B, rcond=None)[0]
    
    el_init = np.arcsin(normal_init[2])
    az_init = np.arctan2(normal_init[1], normal_init[0])
    p0_init = tvecs_raw[min(tvecs_raw.keys())]
    
    def get_axes(el, az):
        n = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
        n = n / np.linalg.norm(n)
        u = np.cross(n, [1, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        return n, u, v
        
    def loss_fn(params):
        uc, vc, el, az, step_size, p0x, p0y, p0z = params
        n, u, v = get_axes(el, az)
        c_rot = mean_t + uc * u + vc * v
        p0 = np.array([p0x, p0y, p0z])
        errors = []
        for i, obs_pt in tvecs_filtered.items():
            angle_rad = np.deg2rad(i * step_size)
            R_i = Rot.from_rotvec(-angle_rad * n).as_matrix()
            pred_pt = np.dot(R_i, p0 - c_rot) + c_rot
            errors.extend(pred_pt - obs_pt)
        return np.array(errors)
        
    initial_guess = [uc, vc, el_init, az_init, 360.0/len(images), p0_init[0], p0_init[1], p0_init[2]]
    res = optimize.least_squares(loss_fn, initial_guess)
    
    uc, vc, el, az, step_size_deg, p0x, p0y, p0z = res.x
    normal, u, v = get_axes(el, az)
    C_rot = mean_t + uc * u + vc * v
    p0 = np.array([p0x, p0y, p0z])
    offset = np.dot(p0 - C_rot, normal)
    C_rot_planar = C_rot + offset * normal
    
    rmse = np.sqrt(np.mean(res.fun**2))
    print(f"C_rot (mm): {C_rot_planar.round(4)}")
    print(f"Normal: {normal.round(4)}")
    print(f"Step Size: {step_size_deg:.4f} degrees/frame")
    print(f"Trajectory RMSE: {rmse:.4f} mm")
    
    # Output to JSON
    out_json = {
        "camera_matrix_K": K.tolist(),
        "distortion_coefficients": dist.tolist(),
        "reprojection_error_px": float(ret),
        "plate_center_mm": C_rot_planar.tolist(),
        "plate_normal": normal.tolist(),
        "step_size_deg": float(abs(step_size_deg)),
        "trajectory_rmse_mm": float(rmse),
        "camera_to_plate_center_distance_mm": float(np.linalg.norm(C_rot_planar)),
        "frames": []
    }

    # Prepare Renderings
    print("\n--- STEP 4: RENDERING AND PER-FRAME DETAILS ---")
    rvec0, tvec0 = np.zeros((3, 1)), np.zeros((3, 1))
    center_2d, _ = cv2.projectPoints(C_rot_planar.reshape(1, 1, 3).astype(np.float64), rvec0, tvec0, K, dist)
    cx, cy = int(round(center_2d[0][0][0])), int(round(center_2d[0][0][1]))
    
    arrow_3d = (C_rot_planar + 80 * normal).reshape(1, 1, 3).astype(np.float64)
    arrow_2d, _ = cv2.projectPoints(arrow_3d, rvec0, tvec0, K, dist)
    ax, ay = int(round(arrow_2d[0][0][0])), int(round(arrow_2d[0][0][1]))

    for idx, img_path in enumerate(images):
        rotation_degree = float(idx * abs(step_size_deg))
        frame_data = {
            "frame_index": idx,
            "filename": os.path.basename(img_path),
            "rotation_degree": rotation_degree
        }
        
        if idx in tvecs_raw:
            frame_data["board_origin_tvec"] = tvecs_raw[idx].tolist()
            frame_data["board_origin_rvec"] = rvecs_raw[idx].tolist()
            
        out_json["frames"].append(frame_data)
        
        img = cv2.imread(img_path)
        cv2.arrowedLine(img, (cx, cy), (ax, ay), (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.12)
        sz = 50
        cv2.line(img, (cx-sz, cy), (cx+sz, cy), (0, 255, 0), 3, cv2.LINE_AA)
        cv2.line(img, (cx, cy-sz), (cx, cy+sz), (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 35, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
        
        cv2.putText(img, f"Center: ({cx}, {cy}) | Dist: {np.linalg.norm(C_rot_planar):.1f}mm", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"Frame {idx} | Angle: {rotation_degree:.2f} deg", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        
        out_img_path = os.path.join(OUTPUT_FOLDER, os.path.basename(img_path))
        cv2.imwrite(out_img_path, img)

    json_path = os.path.join(OUTPUT_FOLDER, "calibration.json")
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=4)
        
    print(f"Complete! JSON and rendered images saved to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
