import cv2
import numpy as np
import glob
import os
import scipy.optimize as optimize
from scipy.spatial.transform import Rotation as Rot

# ==========================================
# Turntable Trajectory Joint Optimizer
# ==========================================

# Folder containing the 32 calibration images
IMAGES_FOLDER = "captures_7-25_lob_with_checkbox"

# Calibrated Camera Parameters (K and distortion)
K = np.array([
    [3565.4767, 0.0, 2304.0],
    [0.0, 3565.4767, 1296.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DIST = np.array([[-0.44031736, 7.83951075, 0.0, 0.0, -43.24147593]], dtype=np.float32)

def fit_trajectory():
    print("=== TURNTABLE TRAJECTORY JOINT OPTIMIZER ===")
    
    # 1. Initialize ChArUco board detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((11, 8), 15.0, 9.0, dictionary)
    detector_params = cv2.aruco.DetectorParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)
    
    images = sorted(glob.glob(os.path.join(IMAGES_FOLDER, "*.jpg")),
                    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    tvecs_raw = {}
    board_corners_3d = np.asarray(board.getChessboardCorners())
    
    print("Detecting ChArUco board poses...")
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
        if charuco_ids is not None and len(charuco_ids) >= 4:
            obj_pts = board_corners_3d[charuco_ids.flatten()]
            ret, rvec, tvec = cv2.solvePnP(obj_pts, charuco_corners, K, DIST)
            if ret:
                tvecs_raw[idx] = tvec.ravel()
                
    print(f"Detected ChArUco board origin positions in {len(tvecs_raw)} / {len(images)} frames.")
    
    # 2. Filter out tracking outliers (such as steep angles resulting in incorrect PnP depth)
    tvecs_filtered = {}
    for idx, t in tvecs_raw.items():
        if t[2] > 100.0:  # Exclude frames with Z < 100 mm (outliers like 16 and 27)
            tvecs_filtered[idx] = t
            
    print(f"Keeping {len(tvecs_filtered)} inliers after outlier filtering.")
    
    indices = list(tvecs_filtered.keys())
    tvecs = np.array(list(tvecs_filtered.values()))
    
    # 3. Initial Guess via plane and circle fitting
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
    
    # Represent normal using spherical coordinates
    el_init = np.arcsin(normal_init[2])
    az_init = np.arctan2(normal_init[1], normal_init[0])
    
    # Initial parameters vector: [uc, vc, el, az, step_size_deg, p0x, p0y, p0z]
    # By parameterizing the center inside the 2D plane (uc, vc), we lock C_rot to the turntable surface
    p0_init = tvecs_raw[0]
    initial_guess = [
        uc, vc,
        el_init, az_init,
        11.50, # Initial guess for step size (degrees)
        p0_init[0], p0_init[1], p0_init[2]
    ]
    
    # 4. Joint Least-Squares Non-linear Optimization
    def get_axes(el, az):
        normal = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
        normal = normal / np.linalg.norm(normal)
        # Create u and v axes orthogonal to normal
        if abs(normal[0]) < 0.9:
            u = np.cross(normal, [1, 0, 0])
        else:
            u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        return normal, u, v
        
    def loss_fn(params):
        uc, vc, el, az, step_size, p0x, p0y, p0z = params
        norm, u, v = get_axes(el, az)
        c_rot = mean_t + uc * u + vc * v
        p0 = np.array([p0x, p0y, p0z])
        
        errors = []
        for i, obs_pt in tvecs_filtered.items():
            angle_rad = np.deg2rad(i * step_size)
            R_i = Rot.from_rotvec(-angle_rad * norm).as_matrix()
            # Predicted 3D position of board origin at frame i
            pred_pt = np.dot(R_i, p0 - c_rot) + c_rot
            errors.extend(pred_pt - obs_pt)
        return np.array(errors)
        
    res = optimize.least_squares(loss_fn, initial_guess)
    
    uc, vc, el, az, step_size_deg, p0x, p0y, p0z = res.x
    normal, u, v = get_axes(el, az)
    C_rot = mean_t + uc * u + vc * v
    
    # Shift P0 along normal to be exactly coplanar with the turntable surface Z=0
    # In the local grid frame: P_local = R_cam.T @ (P0 - C_rot).
    # R_cam columns are [x_col, y_col, normal].
    # So P_local[2] = dot(P0 - C_rot, normal) represents the height offset of P0 relative to the plate.
    # To force the center to represent the turntable surface plane exactly:
    offset = np.dot(p0x - C_rot[0], normal)
    # The true center at the board origin plane height:
    C_rot_planar = C_rot + offset * normal
    
    print("\n" + "="*40)
    print("        OPTIMIZED TRAJECTORY VALUES     ")
    print("="*40)
    print(f"Center C_rot:    {np.array2string(C_rot_planar, precision=8, separator=', ')}")
    print(f"Normal:          {np.array2string(normal, precision=8, separator=', ')}")
    print(f"Step Size:       {step_size_deg:.6f} degrees")
    print(f"Fitting RMSE:    {np.sqrt(np.mean(res.fun**2)):.4f} mm")
    print("="*40)
    
if __name__ == "__main__":
    fit_trajectory()
