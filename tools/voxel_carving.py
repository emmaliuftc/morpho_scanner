import cv2
import numpy as np
import glob
import os
from skimage import measure
from scipy import ndimage
from scipy.spatial.transform import Rotation as Rot

# Import modular pipeline components
from camera_calibration import calibrate_camera_intrinsics
from turntable_calibration import calibrate_turntable_trajectory
from silhouette_extractor import SilhouetteExtractor

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

# ==========================================
# 2. VOXEL CARVING & MESHING PIPELINE
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

def carve_voxels(grid_points, voxel_states, image_paths, extractor, C_rot, normal, step_size, K_cal, DIST_cal):
    """Iterates through images, projects active voxels, and carves them away."""
    
    for i, path in enumerate(image_paths):
        print(f"Carving frame {i+1}/{len(image_paths)}: {os.path.basename(path)}")
        
        raw_img = cv2.imread(path)
        img = cv2.undistort(raw_img, K_cal, DIST_cal, None, K_cal)
        height, width = img.shape[:2]
        
        # Get silhouette mask using the modular extractor component
        mask = extractor.get_silhouette_mask(img, CENTER_2D, PLATE_RADIUS_PIXELS)
        
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
        
    # 1. Dynamic Camera Intrinsic Calibration stage
    K_cal, dist_cal = calibrate_camera_intrinsics(images, fallback_K=DEFAULT_K, fallback_dist=DEFAULT_DIST)
    
    # 2. Dynamic Turntable Trajectory Calibration stage
    C_rot_opt, normal_opt, step_size_opt = calibrate_turntable_trajectory(
        images, K_cal, dist_cal, 
        fallback_C_rot=DEFAULT_C_ROT, fallback_normal=DEFAULT_NORMAL, fallback_step_size=DEFAULT_STEP_SIZE
    )
    
    # Initialize the modular silhouette extraction model
    extractor = SilhouetteExtractor()
    
    # 3. Construct local coordinate basis dynamically from the optimized rotation axis
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal_opt) * normal_opt
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal_opt, x_col)
    R_cam_opt = np.column_stack((x_col, y_col, normal_opt))
    
    # 4. Generate Voxel Grid
    grid_points, voxel_states, grid_shape = create_voxel_grid(C_rot_opt, R_cam_opt)
    
    # 5. Run the carving process
    final_voxels = carve_voxels(grid_points, voxel_states, images, extractor, C_rot_opt, normal_opt, step_size_opt, K_cal, dist_cal)
    
    # 6. Generate and save mesh
    verts, faces = generate_mesh(final_voxels, grid_shape)
    if verts is not None:
        save_path = os.path.join(OUTPUT_FOLDER, "reconstructed_clay.obj")
        save_obj(save_path, verts, faces)
        print("\nVoxel Carving Complete! Open the .obj file in a 3D viewer.")
