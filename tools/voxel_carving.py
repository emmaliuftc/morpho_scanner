import cv2
import numpy as np
import glob
import os
import rembg
import scipy.optimize as optimize
from PIL import Image
from skimage import measure
from scipy import ndimage
from scipy.spatial.transform import Rotation as Rot

# ==========================================
# 1. USER CONFIGURATION & DEFAULT FALLBACKS
# ==========================================
INPUT_FOLDER = "captures_7-25_lob_with_checkbox"
OUTPUT_FOLDER = "captures_7-25_voxel_carving"
NUM_IMAGES = 32

# Voxel Grid Settings
GRID_RESOLUTION = 256  # 256x256x256 voxels
GRID_SIZE_MM = 120.0   # 12.0 cm physical space enclosing the clay sculpture
VOXEL_SIZE = GRID_SIZE_MM / GRID_RESOLUTION

# Default Fallback Camera Intrinsics (K)
DEFAULT_K = np.array([
    [3565.4767, 0.0,       2304.0],
    [0.0,       3565.4767, 1296.0],
    [0.0,       0.0,       1.0]
], dtype=np.float32)

# Default Fallback Distortion Coefficients (dist)
DEFAULT_DIST = np.array([[-0.44031736, 7.83951075, 0.0, 0.0, -43.24147593]], dtype=np.float32)

# Default Fallback Turntable Parameters
DEFAULT_NORMAL = np.array([0.01059424, 0.63511793, 0.77234253])
DEFAULT_C_ROT = np.array([-2.97537598, -24.52457008, 150.91695089])
DEFAULT_STEP_SIZE = 11.552961

# The 2D pixel radius of your green plate to crop out the table/background
PLATE_RADIUS_PIXELS = 1100 
# The 2D pixel coordinate of the plate center
CENTER_2D = (2345, 1183)

# Initialize rembg session globally to avoid reloading model weights for every frame
print("Initializing rembg session...")
SESSION = rembg.new_session("u2net")

# ==========================================
# 2. DYNAMIC CAMERA INTRINSIC CALIBRATION
# ==========================================

def calibrate_camera_intrinsics(images):
    """
    On-the-fly camera intrinsic calibration using the ChArUco board corners.
    Returns calibrated camera matrix K and distortion coefficients.
    """
    print("\n--- RUNNING DYNAMIC CAMERA INTRINSIC CALIBRATION ---")
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # The physical board layout coordinates are scaled by square_size_mm (15mm squares, 9mm markers)
    board = cv2.aruco.CharucoBoard((11, 8), 15.0, 9.0, dictionary)
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
        print("Falling back to default camera intrinsics.")
        return DEFAULT_K, DEFAULT_DIST
        
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
    
    # Save the calibration results for records
    cal_file = os.path.join(OUTPUT_FOLDER, "calibrated_intrinsics.txt")
    with open(cal_file, "w") as f:
        f.write("=== DYNAMIC CAMERA CALIBRATION RESULTS ===\n")
        f.write("Camera Matrix (K):\n")
        f.write(np.array2string(K_opt) + "\n\n")
        f.write("Distortion Coefficients (dist):\n")
        f.write(np.array2string(dist_opt) + "\n")
        
    print("Camera calibration successful!")
    print(f"  -> Calibrated focal length fx, fy: {K_opt[0,0]:.2f}")
    print(f"  -> Calibration data saved to: {cal_file}")
    
    return K_opt, dist_opt

# ==========================================
# 3. DYNAMIC TURNTABLE CALIBRATION
# ==========================================

def calibrate_turntable_trajectory(images, K_cal, DIST_cal):
    """
    On-the-fly calibration using ChArUco board origin tracking.
    Solves for the exact C_rot, normal, and step size using joint least-squares optimization.
    """
    print("\n--- RUNNING DYNAMIC TURNTABLE CALIBRATION ---")
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
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
        print("Falling back to default calibration parameters.")
        return DEFAULT_C_ROT, DEFAULT_NORMAL, DEFAULT_STEP_SIZE
        
    tvecs_filtered = {}
    for idx, t in tvecs_raw.items():
        if t[2] > 100.0:  # Exclude frames with Z < 100 mm (outliers like 16 and 27)
            tvecs_filtered[idx] = t
            
    if len(tvecs_filtered) < 5:
        print("Warning: Insufficient inliers remaining after outlier filtering.")
        print("Falling back to default calibration parameters.")
        return DEFAULT_C_ROT, DEFAULT_NORMAL, DEFAULT_STEP_SIZE
        
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
    
    # Constrain C_rot to turntable surface plane Z=0
    offset = np.dot(p0x - C_rot[0], normal)
    C_rot_planar = C_rot + offset * normal
    
    print("\nTurntable calibration successful!")
    print(f"  -> Solved Center C_rot: {C_rot_planar.round(4)}")
    print(f"  -> Solved Normal:       {normal.round(4)}")
    print(f"  -> Solved Step Size:     {step_size_deg:.4f} degrees")
    print(f"  -> Trajectory RMSE:     {np.sqrt(np.mean(res.fun**2)):.4f} mm")
    
    return C_rot_planar, normal, step_size_deg

# ==========================================
# 4. VOXEL CARVING & MESHING FUNCTIONS
# ==========================================

def create_voxel_grid(C_rot, R_cam):
    """Generates the 3D coordinates in Camera 0 space for every voxel in the grid."""
    print(f"Initializing {GRID_RESOLUTION}^3 Voxel Grid ({GRID_SIZE_MM}mm across)...")
    
    half_size = GRID_SIZE_MM / 2.0
    x = np.linspace(-half_size, half_size, GRID_RESOLUTION)
    y = np.linspace(-half_size, half_size, GRID_RESOLUTION)
    z = np.linspace(-15.0, GRID_SIZE_MM - 15.0, GRID_RESOLUTION)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points_local = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    grid_points_cam0 = np.dot(grid_points_local, R_cam.T) + C_rot
    voxel_states = np.ones(GRID_RESOLUTION**3, dtype=bool)
    
    return grid_points_cam0, voxel_states, xx.shape

def get_silhouette_mask(img, session):
    """
    Isolates the clay object by masking out the background, green plate, and ChArUco board.
    Returns a binary mask where 255 is the object, 0 is background.
    """
    height, width = img.shape[:2]
    
    plate_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(plate_mask, CENTER_2D, PLATE_RADIUS_PIXELS, 255, -1)
    
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    rembg_out = rembg.remove(img_pil, session=session)
    alpha = np.array(rembg_out.split()[-1]) > 128
    filled = ndimage.binary_fill_holes(ndimage.binary_closing(alpha, iterations=5))
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_green_mask = cv2.bitwise_not(green_mask)
    
    obj_mask = (filled & (non_green_mask > 0)).astype(np.uint8) * 255
    obj_mask = cv2.bitwise_and(obj_mask, plate_mask)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(img)
    
    aruco_mask = np.ones((height, width), dtype=np.uint8) * 255
    if ids is not None and len(ids) > 0:
        all_pts = []
        for corner_set in corners:
            for pt in corner_set[0]:
                all_pts.append(pt)
        all_pts = np.array(all_pts, dtype=np.int32)
        
        hull = cv2.convexHull(all_pts)
        cv2.fillConvexPoly(aruco_mask, hull, 0)
        
        kernel = np.ones((41, 41), np.uint8)
        aruco_mask = cv2.erode(aruco_mask, kernel)
        
    final_mask = cv2.bitwise_and(obj_mask, aruco_mask)
    
    kernel_open = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
    final_mask = ndimage.binary_fill_holes(final_mask > 0).astype(np.uint8) * 255
    
    return final_mask

def carve_voxels(grid_points, voxel_states, image_paths, session, C_rot, normal, step_size, K_cal, DIST_cal):
    """Iterates through images, projects active voxels, and carves them away."""
    
    for i, path in enumerate(image_paths):
        print(f"Carving frame {i+1}/{len(image_paths)}: {os.path.basename(path)}")
        
        raw_img = cv2.imread(path)
        img = cv2.undistort(raw_img, K_cal, DIST_cal, None, K_cal)
        height, width = img.shape[:2]
        
        mask = get_silhouette_mask(img, session)
        
        mask_filename = f"mask_{i:02d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, mask_filename), mask)
        
        angle_deg = i * step_size
        angle_rad = np.deg2rad(angle_deg)
        R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
        
        active_indices = np.where(voxel_states)[0]
        active_points = grid_points[active_indices]
        
        if len(active_points) == 0:
            print("Warning: All voxels carved away!")
            break
            
        cam_points = np.dot(active_points - C_rot, R_i.T) + C_rot
        valid_z = cam_points[:, 2] > 0.1 
        
        u = (K_cal[0,0] * cam_points[:, 0] / cam_points[:, 2]) + K_cal[0,2]
        v = (K_cal[1,1] * cam_points[:, 1] / cam_points[:, 2]) + K_cal[1,2]
        
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        
        valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        valid_mask = valid_z & valid_uv
        
        check_u = u[valid_mask]
        check_v = v[valid_mask]
        pixel_values = mask[check_v, check_u]
        hit_background = pixel_values == 0
        
        points_to_carve_indices = np.where(valid_mask)[0][hit_background]
        global_indices_to_carve = active_indices[points_to_carve_indices]
        
        voxel_states[global_indices_to_carve] = False
        
        print(f"  -> Carved {len(global_indices_to_carve)} voxels. {np.sum(voxel_states)} remaining.")

    return voxel_states

def generate_mesh(voxel_states, grid_shape):
    """Removes noise blobs and runs Marching Cubes to generate a mesh."""
    print("\nFiltering noise...")
    
    voxel_grid = voxel_states.reshape(grid_shape)
    labeled_array, num_features = ndimage.label(voxel_grid)
    
    if num_features == 0:
        print("Error: No object remaining after carving.")
        return None, None
        
    sizes = ndimage.sum(voxel_grid, labeled_array, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    cleaned_grid = (labeled_array == largest_label).astype(float)
    
    print("Applying 3D Gaussian smoothing to the voxel density field...")
    smoothed_grid = ndimage.gaussian_filter(cleaned_grid, sigma=1.5)
    
    print("Generating 3D Mesh via Marching Cubes...")
    verts, faces, _, _ = measure.marching_cubes(
        smoothed_grid, level=0.5, spacing=(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
    )
    
    half_size = GRID_SIZE_MM / 2.0
    verts[:, 0] -= half_size
    verts[:, 1] -= half_size
    verts[:, 2] -= 15.0
    
    return verts, faces

def save_obj(filename, verts, faces):
    """Saves the mesh as a standard .obj file."""
    with open(filename, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Saved 3D Model to: {filename}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: {OUTPUT_FOLDER}")
        
    images = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")),
                    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                    
    if len(images) != NUM_IMAGES:
        print(f"Warning: Found {len(images)} images, expected {NUM_IMAGES}.")
        
    # 1. Dynamic Camera Intrinsic Calibration
    K_cal, dist_cal = calibrate_camera_intrinsics(images)
    
    # 2. Dynamic Turntable Trajectory Calibration (using calibrated K and dist)
    C_rot_opt, normal_opt, step_size_opt = calibrate_turntable_trajectory(images, K_cal, dist_cal)
    
    # 3. Construct local coordinate basis dynamically from the optimized rotation axis
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal_opt) * normal_opt
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal_opt, x_col)
    R_cam_opt = np.column_stack((x_col, y_col, normal_opt))
    
    # 4. Generate Voxel Grid
    grid_points, voxel_states, grid_shape = create_voxel_grid(C_rot_opt, R_cam_opt)
    
    # 5. Run the carving process
    final_voxels = carve_voxels(grid_points, voxel_states, images, SESSION, C_rot_opt, normal_opt, step_size_opt, K_cal, dist_cal)
    
    # 6. Generate and save mesh
    verts, faces = generate_mesh(final_voxels, grid_shape)
    if verts is not None:
        save_path = os.path.join(OUTPUT_FOLDER, "reconstructed_clay.obj")
        save_obj(save_path, verts, faces)
        print("\nVoxel Carving Complete! Open the .obj file in a 3D viewer.")
