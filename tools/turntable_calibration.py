import cv2
import numpy as np
import os
import json
import scipy.optimize as optimize
from scipy.spatial.transform import Rotation as Rot

def calibrate_turntable_trajectory(images, K_cal, DIST_cal, fallback_C_rot=None, fallback_normal=None, fallback_step_size=None):
    """
    On-the-fly calibration using ChArUco board origin tracking.
    Solves for the exact C_rot, normal, and step size using joint least-squares optimization.
    """
    print("\n--- RUNNING DYNAMIC TURNTABLE CALIBRATION ---")
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # The physical board layout coordinates (15mm squares, 9mm markers)
    board = cv2.aruco.CharucoBoard((11, 8), 15.0, 9.0, dictionary)
    detector_params = cv2.aruco.DetectorParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)
    
    tvecs_raw = {}
    board_corners_3d = np.asarray(board.getChessboardCorners())
    
    print("Detecting ChArUco board origin positions...")
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
        if charuco_ids is not None and len(charuco_ids) >= 4:
            obj_pts = board_corners_3d[charuco_ids.flatten()]
            ret, rvec, tvec = cv2.solvePnP(obj_pts, charuco_corners, K_cal, DIST_cal)
            if ret:
                tvecs_raw[idx] = tvec.ravel()
                
    if len(tvecs_raw) < 5:
        print(f"Warning: Only detected board poses in {len(tvecs_raw)} frames. Insufficient data for dynamic calibration.")
        if fallback_C_rot is not None:
            print("Falling back to default calibration parameters.")
            return fallback_C_rot, fallback_normal, fallback_step_size
        else:
            raise ValueError("Insufficient frames for turntable calibration and no fallbacks provided.")
        
    tvecs_filtered = {}
    for idx, t in tvecs_raw.items():
        if t[2] > 100.0:  # Exclude frames with Z < 100 mm (outliers like 16 and 27)
            tvecs_filtered[idx] = t
            
    if len(tvecs_filtered) < 5:
        print("Warning: Insufficient inliers remaining after outlier filtering.")
        if fallback_C_rot is not None:
            print("Falling back to default calibration parameters.")
            return fallback_C_rot, fallback_normal, fallback_step_size
        else:
            raise ValueError("Insufficient inliers for turntable calibration and no fallbacks provided.")
        
    tvecs = np.array(list(tvecs_filtered.values()))
    
    # Initial Guess via SVD plane and 2D circle fitting
    mean_t = np.mean(tvecs, axis=0)
    centered_t = tvecs - mean_t
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
    uc, vc, C_val = np.linalg.lstsq(A, B, rcond=None)[0]
    C_rot_init = mean_t + uc * u_axis + vc * v_axis
    
    el_init = np.arcsin(normal_init[2])
    az_init = np.arctan2(normal_init[1], normal_init[0])
    
    p0_init = tvecs_raw[min(tvecs_raw.keys())]
    initial_guess = [
        uc, vc,
        el_init, az_init,
        11.50,
        p0_init[0], p0_init[1], p0_init[2]
    ]
    
    def get_axes(el, az):
        n = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
        n = n / np.linalg.norm(n)
        if abs(n[0]) < 0.9:
            u = np.cross(n, [1, 0, 0])
        else:
            u = np.cross(n, [0, 1, 0])
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
        
    res = optimize.least_squares(loss_fn, initial_guess)
    
    uc, vc, el, az, step_size_deg, p0x, p0y, p0z = res.x
    normal, u, v = get_axes(el, az)
    C_rot = mean_t + uc * u + vc * v
    
    p0 = np.array([p0x, p0y, p0z])
    offset = np.dot(p0 - C_rot, normal)
    C_rot_planar = C_rot + offset * normal
    
    print("\nTurntable calibration successful!")
    print(f"  -> Solved Center C_rot: {C_rot_planar.round(4)}")
    print(f"  -> Solved Normal:       {normal.round(4)}")
    print(f"  -> Solved Step Size:     {step_size_deg:.4f} degrees")
    print(f"  -> Trajectory RMSE:     {np.sqrt(np.mean(res.fun**2)):.4f} mm")
    
    return C_rot_planar, normal, step_size_deg

def save_turntable_calibration(C_rot, normal, step_size, filepath):
    """Saves turntable trajectory calibration parameters to a JSON file."""
    data = {
        "C_rot": C_rot.tolist(),
        "normal": normal.tolist(),
        "step_size": float(step_size)
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved turntable trajectory calibration to: {filepath}")

def load_turntable_calibration(filepath, fallback_C_rot=None, fallback_normal=None, fallback_step_size=None):
    """Loads turntable trajectory calibration parameters from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Warning: Turntable trajectory file '{filepath}' not found.")
        if fallback_C_rot is not None:
            print("Using fallback turntable trajectory parameters.")
            return fallback_C_rot, fallback_normal, fallback_step_size
        else:
            raise FileNotFoundError(f"Turntable trajectory file '{filepath}' not found and no fallbacks provided.")
            
    with open(filepath, "r") as f:
        data = json.load(f)
    C_rot = np.array(data["C_rot"], dtype=np.float32)
    normal = np.array(data["normal"], dtype=np.float32)
    step_size = float(data["step_size"])
    print(f"Loaded turntable trajectory calibration from: {filepath}")
    return C_rot, normal, step_size
