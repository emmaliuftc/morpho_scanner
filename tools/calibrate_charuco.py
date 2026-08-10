import numpy as np
from pathlib import Path
import cv2
import glob
import os

def calibrate_charuco_camera(images_folder, square_size_mm=15.0, marker_size_mm=9.0):
    print("=== CHARUCO BOARD CAMERA CALIBRATION ===")
    print(f"Loading images from '{images_folder}'...")
    
    # 1. Initialize ChArUco Board (11 columns, 8 rows of squares)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # The physical board layout coordinates are scaled by square_size_mm
    board = cv2.aruco.CharucoBoard((11, 8), square_size_mm, marker_size_mm, dictionary)
    
    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    
    images = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))
    if not images:
        print("Error: No images found.")
        return
        
    objpoints = []  # 3D points in board coordinate system
    imgpoints = []  # 2D points in image plane
    image_shape = None
    successful_frames = 0
    
    print(f"Processing {len(images)} images...")
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]
            
        # Detect board corners
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        
        # We need at least 4 corners to establish homography / pose
        if charuco_ids is not None and len(charuco_ids) >= 4:
            # Get the exact 3D coordinates for the detected corner IDs
            # board.getChessboardCorners() returns all corners coordinates (shape: (N, 3))
            board_corners_3d = np.asarray(board.getChessboardCorners())
            
            # Map corner IDs to their corresponding 3D coordinates
            obj_pts = board_corners_3d[charuco_ids.flatten()]
            
            objpoints.append(obj_pts.astype(np.float32))
            imgpoints.append(charuco_corners.astype(np.float32))
            successful_frames += 1
            print(f"  {os.path.basename(img_path)}: Detected {len(charuco_ids)} corners (Success)")
        else:
            print(f"  {os.path.basename(img_path)}: Insufficient corners (Skipped)")
            
    if successful_frames < 5:
        print("Error: Insufficient frames with valid ChArUco corners.")
        return
        
    print(f"\nRunning constrained calibration with {successful_frames} frames...")
    # Initialize camera matrix at the exact image center
    mtx = np.array([
        [3000.0, 0.0, image_shape[0] / 2.0],
        [0.0, 3000.0, image_shape[1] / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)
    
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS |
        cv2.CALIB_FIX_ASPECT_RATIO |
        cv2.CALIB_FIX_PRINCIPAL_POINT |
        cv2.CALIB_ZERO_TANGENT_DIST
    )
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, mtx, dist, flags=flags
    )
    
    # Calculate reprojection error
    total_error = 0
    total_points = 0
    for idx in range(len(objpoints)):
        imgpoints_projected, _ = cv2.projectPoints(
            objpoints[idx], rvecs[idx], tvecs[idx], mtx, dist
        )
        diff = imgpoints[idx].reshape(-1, 2) - imgpoints_projected.reshape(-1, 2)
        error = np.sum(np.linalg.norm(diff, axis=1))
        total_error += error
        total_points += len(objpoints[idx])
        
    mean_error = total_error / total_points
    
    print("\n" + "="*40)
    print("           CALIBRATION RESULTS          ")
    print("="*40)
    print(f"Reprojection Error: {mean_error:.4f} pixels (closer to 0 is better)")
    print("\n1. Camera Intrinsic Matrix (K):")
    print(mtx)
    print("\n2. Radial & Tangential Distortion (dist):")
    print(dist)
    print("="*40)
    
    # Save the calibration results
    out_path = Path("calibration_data.txt")
    with open(out_path, "w") as f:
        f.write("=== CAMERA CALIBRATION DATA ===\n")
        f.write(f"Reprojection Error: {mean_error:.6f} pixels\n\n")
        f.write("Camera Matrix (K):\n")
        f.write(np.array2string(mtx) + "\n\n")
        f.write("Distortion Coefficients (dist):\n")
        f.write(np.array2string(dist) + "\n")
    print(f"\nCalibration data saved to {out_path.absolute()}")
    
    return mtx, dist

if __name__ == "__main__":
    calibrate_charuco_camera("captures_7-25_lob_with_checkbox")
