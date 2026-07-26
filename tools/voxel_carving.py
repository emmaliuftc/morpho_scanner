import cv2
import numpy as np
import glob
import os
import rembg
from PIL import Image
from skimage import measure
from scipy import ndimage
from scipy.spatial.transform import Rotation as Rot

# ==========================================
# 1. USER CONFIGURATION & CALIBRATION DATA
# ==========================================
INPUT_FOLDER = "captures_7-25_lob_with_checkbox"
OUTPUT_FOLDER = "captures_7-25_voxel_carving"
NUM_IMAGES = 32

# Voxel Grid Settings
GRID_RESOLUTION = 256  # 256x256x256 voxels
GRID_SIZE_MM = 120.0   # 12.0 cm physical space enclosing the clay sculpture
VOXEL_SIZE = GRID_SIZE_MM / GRID_RESOLUTION

# Calibrated Camera Matrix (K)
K = np.array([
    [3565.4767, 0.0,       2304.0],
    [0.0,       3565.4767, 1296.0],
    [0.0,       0.0,       1.0]
], dtype=np.float32)

# Calibrated Distortion Coefficients (dist)
DIST = np.array([[-0.44031736, 7.83951075, 0.0, 0.0, -43.24147593]], dtype=np.float32)

# Turntable Geometry & Jointly Optimized Trajectory Parameters
# Obtained from outlier-filtered non-linear least-squares fitting
normal = np.array([0.01058765, 0.63511738, 0.77234307])
normal = normal / np.linalg.norm(normal)
C_rot = np.array([-3.2555728, -26.45521897, 152.54530728])
step_size_deg = 11.552973  # Optimized physical step size in degrees per picture

# Construct R_cam to define the local grid orientation aligned with the turntable surface plane
ref = np.array([1.0, 0.0, 0.0])
x_col = ref - np.dot(ref, normal) * normal
x_col = x_col / np.linalg.norm(x_col)
y_col = np.cross(normal, x_col)
R_cam = np.column_stack((x_col, y_col, normal))

# The 2D pixel radius of your green plate to crop out the table/background
PLATE_RADIUS_PIXELS = 1100 
# The 2D pixel coordinate of the plate center (from your CBA)
CENTER_2D = (2345, 1183)

# Initialize rembg session globally to avoid reloading model weights for every frame
print("Initializing rembg session...")
SESSION = rembg.new_session("u2net")

def create_voxel_grid():
    """Generates the 3D coordinates in Camera 0 space for every voxel in the grid."""
    print(f"Initializing {GRID_RESOLUTION}^3 Voxel Grid ({GRID_SIZE_MM}mm across)...")
    
    # Create 1D arrays for X, Y, Z coordinates centered on 0 in the turntable surface plane
    half_size = GRID_SIZE_MM / 2.0
    x = np.linspace(-half_size, half_size, GRID_RESOLUTION)
    y = np.linspace(-half_size, half_size, GRID_RESOLUTION)
    # Start Z from -15.0mm to capture the bottom of the sculpture that sits slightly below the turntable plane
    z = np.linspace(-15.0, GRID_SIZE_MM - 15.0, GRID_RESOLUTION)
    
    # Create 3D meshgrid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten into a list of local 3D points (N x 3)
    grid_points_local = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    
    # Convert local grid coordinates to Camera 0 coordinate space using R_cam and C_rot
    grid_points_cam0 = np.dot(grid_points_local, R_cam.T) + C_rot
    
    # Initialize all voxels as 'True' (Solid/Object)
    voxel_states = np.ones(GRID_RESOLUTION**3, dtype=bool)
    
    return grid_points_cam0, voxel_states, xx.shape

def get_silhouette_mask(img, session):
    """
    Isolates the clay object by masking out the background, green plate, and ChArUco board.
    Uses rembg + HSV thresholding + convex hull of detected ArUco markers for perfect masking.
    Returns a binary mask where 255 is the object, 0 is background.
    """
    height, width = img.shape[:2]
    
    # 1. Base Mask: Only look inside the green plate circle
    plate_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(plate_mask, CENTER_2D, PLATE_RADIUS_PIXELS, 255, -1)
    
    # 2. rembg Mask: Isolate foreground elements (clay sculpture and ChArUco board)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    rembg_out = rembg.remove(img_pil, session=session)
    alpha = np.array(rembg_out.split()[-1]) > 128
    filled = ndimage.binary_fill_holes(ndimage.binary_closing(alpha, iterations=5))
    
    # 3. HSV Color Mask: Remove any green paper background highlights inside the plate
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_green_mask = cv2.bitwise_not(green_mask)
    
    # Combine rembg with green color exclusion and plate bounds
    obj_mask = (filled & (non_green_mask > 0)).astype(np.uint8) * 255
    obj_mask = cv2.bitwise_and(obj_mask, plate_mask)
    
    # 4. ChArUco Board Mask: Detect markers and mask out the entire board using convex hull + dilation
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
        
        # Convex hull of all detected marker corners covers the board surface
        hull = cv2.convexHull(all_pts)
        cv2.fillConvexPoly(aruco_mask, hull, 0)
        
        # Dilate the board mask slightly to completely wipe out outer white squares/margins
        kernel = np.ones((41, 41), np.uint8)
        aruco_mask = cv2.erode(aruco_mask, kernel)
        
    # 5. Final Combined Mask
    final_mask = cv2.bitwise_and(obj_mask, aruco_mask)
    
    # Morphological opening to clean up small floating pixels
    kernel_open = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Ensure there are absolutely no hollow spots inside the clay mask
    final_mask = ndimage.binary_fill_holes(final_mask > 0).astype(np.uint8) * 255
    
    return final_mask

def carve_voxels(grid_points, voxel_states, image_paths, session):
    """Iterates through images, projects active voxels, and carves them away."""
    
    for i, path in enumerate(image_paths):
        print(f"Carving frame {i+1}/{len(image_paths)}: {os.path.basename(path)}")
        
        # Load and UNDISTORT the image first.
        # This is much faster than applying distortion to millions of 3D points.
        raw_img = cv2.imread(path)
        img = cv2.undistort(raw_img, K, DIST, None, K)
        height, width = img.shape[:2]
        
        # Get the binary silhouette mask
        mask = get_silhouette_mask(img, session)
        
        # Save a verification mask image inside the output folder to monitor quality
        mask_filename = f"mask_{i:02d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, mask_filename), mask)
        
        # Calculate Turntable Rotation for this specific frame
        angle_deg = i * step_size_deg
        angle_rad = np.deg2rad(angle_deg)
        
        # Generate the camera pose rotation relative to Camera 0
        R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
        
        # Get only the currently SOLID voxels to save processing time
        active_indices = np.where(voxel_states)[0]
        active_points = grid_points[active_indices]
        
        if len(active_points) == 0:
            print("Warning: All voxels carved away!")
            break
            
        # Project 3D camera 0 points into camera coordinates for frame i using R_i around C_rot
        cam_points = np.dot(active_points - C_rot, R_i.T) + C_rot
        
        # Prevent division by zero for points behind the camera
        valid_z = cam_points[:, 2] > 0.1 
        
        # Project to 2D image plane (Linear Pinhole model, since image is undistorted)
        u = (K[0,0] * cam_points[:, 0] / cam_points[:, 2]) + K[0,2]
        v = (K[1,1] * cam_points[:, 1] / cam_points[:, 2]) + K[1,2]
        
        # Round to nearest pixel coordinate
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        
        # Filter points that land OUTSIDE the image boundaries
        valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        valid_mask = valid_z & valid_uv
        
        # The Carving Step: Check the mask
        check_u = u[valid_mask]
        check_v = v[valid_mask]
        
        # Get mask values (255 = object, 0 = background)
        pixel_values = mask[check_v, check_u]
        
        # Find which active points hit the background (value == 0)
        hit_background = pixel_values == 0
        
        # Map back to the original active_indices and set them to False
        points_to_carve_indices = np.where(valid_mask)[0][hit_background]
        global_indices_to_carve = active_indices[points_to_carve_indices]
        
        voxel_states[global_indices_to_carve] = False
        
        print(f"  -> Carved {len(global_indices_to_carve)} voxels. {np.sum(voxel_states)} remaining.")

    return voxel_states

def generate_mesh(voxel_states, grid_shape):
    """Removes noise blobs and runs Marching Cubes to generate a mesh."""
    print("\nFiltering noise...")
    
    # Reshape the flat boolean array back into a 3D grid
    voxel_grid = voxel_states.reshape(grid_shape)
    
    # Find connected components (blobs)
    labeled_array, num_features = ndimage.label(voxel_grid)
    
    if num_features == 0:
        print("Error: No object remaining after carving.")
        return None, None
        
    # Find the largest blob (assuming it's the main object)
    sizes = ndimage.sum(voxel_grid, labeled_array, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    
    # Isolate the largest blob
    cleaned_grid = (labeled_array == largest_label).astype(float)
    
    # Apply 3D Gaussian smoothing to convert binary voxels to a smooth density field
    print("Applying 3D Gaussian smoothing to the voxel density field...")
    smoothed_grid = ndimage.gaussian_filter(cleaned_grid, sigma=1.5)
    
    print("Generating 3D Mesh via Marching Cubes...")
    # Marching cubes creates vertices and faces from the smoothed continuous grid
    verts, faces, _, _ = measure.marching_cubes(
        smoothed_grid, level=0.5, spacing=(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
    )
    
    # Offset vertices so they are centered around 0 in physical space
    half_size = GRID_SIZE_MM / 2.0
    verts[:, 0] -= half_size
    verts[:, 1] -= half_size
    # Adjust for the grid Z offset
    verts[:, 2] -= 15.0
    
    return verts, faces

def save_obj(filename, verts, faces):
    """Saves the mesh as a standard .obj file for Blender/Meshlab."""
    with open(filename, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # OBJ indices are 1-based
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Saved 3D Model to: {filename}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: {OUTPUT_FOLDER}")
        
    # Numerically sort images using lambda key
    images = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")),
                    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                    
    if len(images) != NUM_IMAGES:
        print(f"Warning: Found {len(images)} images, expected {NUM_IMAGES}.")
        
    grid_points, voxel_states, grid_shape = create_voxel_grid()
    
    # Run the carving process
    final_voxels = carve_voxels(grid_points, voxel_states, images, SESSION)
    
    # Generate and save mesh
    verts, faces = generate_mesh(final_voxels, grid_shape)
    if verts is not None:
        save_path = os.path.join(OUTPUT_FOLDER, "reconstructed_clay.obj")
        save_obj(save_path, verts, faces)
        print("\nVoxel Carving Complete! Open the .obj file in a 3D viewer.")
