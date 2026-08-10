import cv2
import numpy as np
import os
import json

def calibrate_camera_intrinsics(images, square_size_mm=15.0, marker_size_mm=9.0, fallback_K=None, fallback_dist=None):
    """
    On-the-fly camera intrinsic calibration using the ChArUco board corners.
    Returns calibrated camera matrix K and distortion coefficients.
    """
    print("\n--- RUNNING DYNAMIC CAMERA INTRINSIC CALIBRATION ---")
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # The physical board layout coordinates are scaled by square_size_mm
    board = cv2.aruco.CharucoBoard((11, 8), square_size_mm, marker_size_mm, dictionary)
    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    
    objpoints = []  # 3D points in board coordinate system
    imgpoints = []  # 2D points in image plane
    image_shape = None
    
    print("Detecting ChArUco corners on all images...")
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]
            
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
        if charuco_ids is not None and len(charuco_ids) >= 4:
            board_corners_3d = np.asarray(board.getChessboardCorners())
            obj_pts = board_corners_3d[charuco_ids.flatten()]
            objpoints.append(obj_pts.astype(np.float32))
            imgpoints.append(charuco_corners.astype(np.float32))
            
    if len(objpoints) < 5:
        print(f"Warning: Only found corners in {len(objpoints)} images. Cannot run camera calibration.")
        if fallback_K is not None and fallback_dist is not None:
            print("Falling back to default camera intrinsics.")
            return fallback_K, fallback_dist
        else:
            raise ValueError("Insufficient frames for camera calibration and no default fallbacks provided.")
        
    print(f"Calibrating camera matrix using {len(objpoints)} successful frames...")
    
    # Initialize camera matrix guess at the image center
    mtx_guess = np.array([
        [3000.0, 0.0, image_shape[0] / 2.0],
        [0.0, 3000.0, image_shape[1] / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    dist_guess = np.zeros(5, dtype=np.float32)
    
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS |
        cv2.CALIB_FIX_ASPECT_RATIO |
        cv2.CALIB_FIX_PRINCIPAL_POINT |
        cv2.CALIB_ZERO_TANGENT_DIST
    )
    
    ret, K_opt, dist_opt, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, mtx_guess, dist_guess, flags=flags
    )
    
    print("Camera calibration successful!")
    print(f"  -> Calibrated focal length fx, fy: {K_opt[0,0]:.2f}")
    
    return K_opt, dist_opt

def save_camera_calibration(K, dist, filepath):
    """Saves camera calibration parameters to a JSON file."""
    data = {
        "K": K.tolist(),
        "dist": dist.tolist()
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved camera intrinsics to: {filepath}")

def load_camera_calibration(filepath, fallback_K=None, fallback_dist=None):
    """Loads camera calibration parameters from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Warning: Calibration file '{filepath}' not found.")
        if fallback_K is not None and fallback_dist is not None:
            print("Using fallback camera intrinsics.")
            return fallback_K, fallback_dist
        else:
            raise FileNotFoundError(f"Calibration file '{filepath}' not found and no fallbacks provided.")
            
    with open(filepath, "r") as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float32)
    dist = np.array(data["dist"], dtype=np.float32)
    print(f"Loaded camera intrinsics from: {filepath}")
    return K, dist
