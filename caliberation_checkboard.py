import numpy as np
import cv2
import glob
import os

def calibrate_camera(images_folder, square_size_mm, checkerboard_size):
    """
    Calibrates a camera using a set of checkerboard images.
    
    Parameters:
    - images_folder (str): Path to the directory containing checkerboard images (e.g., '*.jpg').
    - square_size_mm (float): The physical size of a single square on the printed board in millimeters.
    - checkerboard_size (tuple): The number of INNER corners (columns, rows). 
                                 For a 10x7 squares board, this is (9, 6).
    """
    print(f"Starting calibration using images in '{images_folder}'...")
    
    # Termination criteria for sub-pixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    # These represent the true, undistorted 3D locations of the corners in the real world.
    # We multiply by square_size_mm to give the coordinates real-world scale (millimeters).
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size_mm

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Ensure you are using the correct file extension for your camera
    images = glob.glob(os.path.join(images_folder, '*.jpg'))
    
    if not images:
        print("Error: No images found. Check the folder path and file extension.")
        return

    print(f"Found {len(images)} images. Processing...")

    image_shape = None
    successful_images = 0

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Save the image shape for the calibration function later
        if image_shape is None:
            image_shape = gray.shape[::-1] 

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            successful_images += 1
            objpoints.append(objp)
            
            # Refine corner locations to sub-pixel accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            # Optional: Draw and display the corners (Uncomment to debug visually)
            # cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
        else:
            print(f"Warning: Could not find checkerboard in {fname}")

    # cv2.destroyAllWindows()

    if successful_images < 10:
        print(f"\nWarning: Only found corners in {successful_images} images. You should aim for at least 15-20 good captures for an accurate calibration.")

    print(f"\nSuccessfully processed {successful_images} images. Calculating Intrinsic Matrix and Distortion...")
    
    # Perform camera calibration
    # Output: 
    # mtx = Camera Matrix (K) containing focal lengths (fx, fy) and principal point (cx, cy)
    # dist = Distortion Coefficients (k1, k2, p1, p2, k3)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

    # Calculate the mean reprojection error to evaluate calibration quality
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print("\n--- CALIBRATION RESULTS ---")
    print(f"Total Reprojection Error: {mean_error/len(objpoints):.4f} pixels (closer to 0 is better)")
    
    print("\n1. Camera Matrix (K):")
    print(mtx)
    print("-> Use this matrix directly in your Bundle Adjustment script.")
    
    print("\n2. Distortion Coefficients (dist):")
    print(dist)
    print("-> Use these coefficients in cv2.projectPoints within your Bundle Adjustment.")
    
    return mtx, dist

if __name__ == "__main__":
    # --- USER CONFIGURATION REQUIRED ---
    
    # 1. Put all your checkerboard photos in a folder named 'calibration_images' next to this script.
    FOLDER_PATH = "checkboard" 
    
    # 2. Measure the physical width of one square on your printed paper in millimeters.
    # IMPORTANT: Change this to match your physical printout!
    SQUARE_SIZE_MM = 15.0 
    
    # 3. Count the INNER intersecting corners. 
    # If the board has 10 squares wide and 7 squares tall, inner corners are 9x6.
    INNER_CORNERS = (9, 6) 
    
    # Create folder if it doesn't exist (to prevent crash before user reads instructions)
    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH)
        print(f"Created folder '{FOLDER_PATH}'. Please put your checkerboard JPGs in there and run again.")
    else:
        # Run calibration
        calibrate_camera(FOLDER_PATH, SQUARE_SIZE_MM, INNER_CORNERS)