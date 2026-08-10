import cv2
import numpy as np
import glob
import os
import sys
from skimage import measure
from scipy import ndimage
from scipy.spatial.transform import Rotation as Rot

# Import modular pipeline components from the same directory
from camera_calibration import calibrate_camera_intrinsics
from turntable_calibration import calibrate_turntable_trajectory
from silhouette_extractor import SilhouetteExtractor

# ==========================================
# 1. SYSTEM CONFIGURATION & DATA PATHS
# ==========================================
INPUT_FOLDER = "captures_7-25_lob_with_checkbox"
OUTPUT_FOLDER = "captures_7-25_tsdf"
NUM_IMAGES = 32

# Voxel Grid Settings (256x256x256, 12.0 cm across to match voxel carving space)
GRID_RESOLUTION = 256
GRID_SIZE_MM = 120.0  
VOXEL_SIZE = GRID_SIZE_MM / GRID_RESOLUTION

# TSDF Parameters
TRUNC_MARGIN = VOXEL_SIZE * 8.0 
MAX_WEIGHT = 100.0  # Caps the weight to allow later frames to correct earlier errors

# Default Fallback Camera Intrinsics
DEFAULT_K = np.array([
    [3565.4767, 0.0,       2304.0],
    [0.0,       3565.4767, 1296.0],
    [0.0,       0.0,       1.0]
], dtype=np.float32)

DEFAULT_DIST = np.array([[-0.44031736, 7.83951075, 0.0, 0.0, -43.24147593]], dtype=np.float32)

# Default Fallback Turntable Parameters
DEFAULT_NORMAL = np.array([0.01059424, 0.63511793, 0.77234253])
DEFAULT_C_ROT = np.array([-2.97537598, -24.52457008, 150.91695089])
DEFAULT_STEP_SIZE = 11.552961

# The 2D pixel radius of your green plate to crop out the table/background
PLATE_RADIUS_PIXELS = 1100 
# The 2D pixel coordinate of the plate center
CENTER_2D = (2345, 1183)

class Logger(object):
    """Dual logger to write to both stdout and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def compute_stereo_depth(img1, img2, R1, t1, R2, t2, K_cal, dist_cal):
    """
    Computes a depth map for img1 using stereo matching with img2.
    Uses OpenCV's epipolar rectification and Semi-Global Block Matching (SGBM).
    """
    height, width = img1.shape[:2]
    
    # 1. Calculate relative pose between the two frames
    R_rel = np.dot(R2, R1.T)
    t_rel = (t2 - np.dot(R_rel, t1)).reshape(3, 1)
    
    # 2. Stereo Rectify to align the images horizontally
    R1_rect, R2_rect, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_cal, dist_cal, K_cal, dist_cal, (width, height), R_rel, t_rel, flags=cv2.CALIB_ZERO_DISPARITY
    )
    
    # Compute undistortion and rectification mapping
    map1_x, map1_y = cv2.initUndistortRectifyMap(K_cal, dist_cal, R1_rect, P1, (width, height), cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K_cal, dist_cal, R2_rect, P2, (width, height), cv2.CV_32FC1)
    
    # Apply maps to warp images
    rect_img1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
    rect_img2 = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)
    
    # Convert to grayscale for stereo matching
    gray1 = cv2.cvtColor(rect_img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rect_img2, cv2.COLOR_BGR2GRAY)
    
    # 3. Compute Disparity using SGBM (Tuned for smooth clay features)
    window_size = 5
    min_disp = 384
    num_disp = 256
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
    
    # 4. Project Disparity back to 3D Space
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    depth_rectified = np.abs(points_3d[:, :, 2])
    
    # Filter out invalid depth values
    valid_mask = (disparity > min_disp) & (depth_rectified > 50.0) & (depth_rectified < 1500.0)
    depth_rectified[~valid_mask] = 0.0
    
    # 5. Reverse the Rectification mapping to get the depth map aligned with the ORIGINAL img1
    map1_x_inv, map1_y_inv = cv2.initUndistortRectifyMap(P1[:, :3], None, R1_rect.T, K_cal, (width, height), cv2.CV_32FC1)
    depth_original_view = cv2.remap(depth_rectified, map1_x_inv, map1_y_inv, cv2.INTER_NEAREST)
    
    return depth_original_view

class TSDFVolume:
    def __init__(self, resolution, size_mm, K_cal):
        self.res = resolution
        self.size = size_mm
        self.voxel_size = size_mm / resolution
        self.K = K_cal
        
        print(f"Initializing {resolution}^3 TSDF Grid ({size_mm}mm across, {self.voxel_size:.2f}mm/voxel)...")
        
        # TSDF values initialized to 1.0 (Assume everything is empty space initially)
        self.tsdf = np.ones((resolution, resolution, resolution), dtype=np.float32)
        # Weights initialized to 0
        self.weights = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        # Generate the physical 3D coordinate for every voxel
        half_size = size_mm / 2.0
        x = np.linspace(-half_size, half_size, resolution)
        y = np.linspace(-half_size, half_size, resolution)
        # Start Z from -15.0mm to match voxel carving bounds
        z = np.linspace(-15.0, size_mm - 15.0, resolution)
        
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        
        # Flatten into a 3xN array of 3D points
        self.voxel_coords = np.vstack((xv.ravel(), yv.ravel(), zv.ravel()))
        self.num_voxels = self.voxel_coords.shape[1]

    def integrate(self, depth_map, silhouette_mask, R, t):
        """Projects the TSDF grid into the image and updates distances based on the depth map."""
        height, width = depth_map.shape
        
        # 1. Transform Voxel Coordinates to Camera Space
        # P_cam = R * P_world + t
        cam_pts = np.dot(R, self.voxel_coords) + t.reshape(3, 1)
        voxel_depths = cam_pts[2, :]
        
        # Prevent division by zero for points behind the camera
        valid_z = voxel_depths > 1.0 
        
        # 2. Project 3D points to 2D image pixels
        u = (self.K[0,0] * cam_pts[0, :] / voxel_depths) + self.K[0,2]
        v = (self.K[1,1] * cam_pts[1, :] / voxel_depths) + self.K[1,2]
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        
        # 3. Filter points that project OUTSIDE the image boundaries
        valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        valid_mask = valid_z & valid_uv
        
        valid_u = u[valid_mask]
        valid_v = v[valid_mask]
        valid_voxel_depths = voxel_depths[valid_mask]
        
        # 4. Sample the Depth Map and Silhouette Mask at these pixels
        measured_depths = depth_map[valid_v, valid_u]
        mask_values = silhouette_mask[valid_v, valid_u]
        
        # 5. Compute the Signed Distance Field (SDF)
        sdf = measured_depths - valid_voxel_depths
        
        # Condition A: Valid Stereo Depth Measurement within Truncation Band
        valid_stereo = (measured_depths > 0) & (sdf >= -TRUNC_MARGIN)
        
        # Condition B: Voxel projects to Background (Silhouette Mask == 0)
        is_background = (mask_values == 0)
        
        # Points to update are either valid stereo points OR background points
        update_mask = valid_stereo | is_background
        
        # Apply the update mask to narrow down the data
        final_valid_indices = np.where(valid_mask)[0][update_mask]
        final_sdf = sdf[update_mask]
        final_is_bg = is_background[update_mask]
        
        # Truncate the SDF values between [-1, 1]
        tsdf_update = np.minimum(1.0, np.maximum(-1.0, final_sdf / TRUNC_MARGIN))
        
        # Force background points to +1.0 (Empty) regardless of stereo noise
        tsdf_update[final_is_bg] = 1.0
        
        # 6. Apply Running Average to Voxel Grid
        old_tsdf = self.tsdf.ravel()[final_valid_indices]
        old_weight = self.weights.ravel()[final_valid_indices]
        
        # Update weights (give background carving a higher weight to ensure it deletes noise)
        weight_update = np.ones_like(tsdf_update)
        weight_update[final_is_bg] = 5.0 
        
        new_weight = old_weight + weight_update
        new_weight = np.minimum(new_weight, MAX_WEIGHT)
        
        # TSDF Moving Average Formula
        new_tsdf = (old_tsdf * old_weight + tsdf_update * weight_update) / new_weight
        
        # Store back into grid
        self.tsdf.ravel()[final_valid_indices] = new_tsdf
        self.weights.ravel()[final_valid_indices] = new_weight

    def color_mesh(self, verts, image_paths, get_camera_pose_fn, extractor, dist_cal):
        """Projects mesh vertices back to input views to sample and average BGR/RGB colors."""
        print("\n--- COLORING 3D MESH VERTICES FROM INPUT IMAGES ---")
        num_verts = len(verts)
        vert_colors = np.zeros((num_verts, 3), dtype=np.float32)
        vert_weights = np.zeros(num_verts, dtype=np.float32)
        
        # Vertices are already in the correct local coordinate space
        local_verts = verts
        
        print("Projecting vertices and averaging colors...")
        for i, path in enumerate(image_paths):
            img = cv2.undistort(cv2.imread(path), self.K, dist_cal)
            mask_path = os.path.join(OUTPUT_FOLDER, "masks", f"mask_{i:02d}.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            R, t = get_camera_pose_fn(i)
            
            # Project vertices
            cam_pts = np.dot(R, local_verts.T) + t.reshape(3, 1) # shape (3, N)
            voxel_depths = cam_pts[2, :]
            
            valid_z = voxel_depths > 1.0
            
            u = (self.K[0,0] * cam_pts[0, :] / voxel_depths) + self.K[0,2]
            v = (self.K[1,1] * cam_pts[1, :] / voxel_depths) + self.K[1,2]
            u = np.round(u).astype(int)
            v = np.round(v).astype(int)
            
            height, width = img.shape[:2]
            valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            valid_mask = valid_z & valid_uv
            
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue
                
            v_idx = v[valid_indices]
            u_idx = u[valid_indices]
            inside_silhouette = mask[v_idx, u_idx] > 0
            visible_indices = valid_indices[inside_silhouette]
            
            if len(visible_indices) == 0:
                continue
                
            vis_v = v[visible_indices]
            vis_u = u[visible_indices]
            
            # Sample BGR colors and convert to RGB
            colors = img[vis_v, vis_u, ::-1]
            
            vert_colors[visible_indices] += colors
            vert_weights[visible_indices] += 1.0
            
        # Compute average
        final_colors = np.zeros((num_verts, 3), dtype=np.float32)
        has_weight = vert_weights > 0
        final_colors[has_weight] = vert_colors[has_weight] / vert_weights[has_weight, None]
        final_colors[~has_weight] = [128.0, 128.0, 128.0]
        
        return final_colors.astype(np.uint8)

    def extract_mesh(self, output_path, image_paths, get_camera_pose_fn, extractor, dist_cal):
        """Converts the TSDF volume to a 3D mesh, colors it, and saves as OBJ and PLY."""
        print("\nExtracting 3D Mesh using Marching Cubes...")
        
        try:
            verts, faces, _, _ = measure.marching_cubes(self.tsdf, level=0.0, spacing=(self.voxel_size, self.voxel_size, self.voxel_size))
            
            # Center vertices mathematically
            half_size = self.size / 2.0
            verts[:, 0] -= half_size
            verts[:, 1] -= half_size
            verts[:, 2] -= 15.0
            
            # Run Vertex Color Fusion
            colors = self.color_mesh(verts, image_paths, get_camera_pose_fn, extractor, dist_cal)
            
            # Save as OBJ
            with open(output_path, 'w') as f:
                for v, c in zip(verts, colors):
                    f.write(f"v {v[0]} {v[1]} {v[2]} {c[0]/255.0:.4f} {c[1]/255.0:.4f} {c[2]/255.0:.4f}\n")
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            print(f"Successfully saved colored OBJ mesh to: {output_path}")
            
            # Save as PLY
            ply_path = output_path.replace(".obj", ".ply")
            with open(ply_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(verts)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                for v, c in zip(verts, colors):
                    f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
            print(f"Successfully saved colored PLY mesh to: {ply_path}")
            
        except ValueError as e:
            print(f"Mesh Extraction Failed: {e}")

    def initialize_with_visual_hull(self, image_paths, extractor, get_camera_pose_fn, dist_cal):
        """Initializes the TSDF grid by running a fast voxel carving pass."""
        print("\n--- INITIALIZING TSDF VOLUME WITH VISUAL HULL (VOXEL CARVING) ---")
        
        # Initialize TSDF to -1.0 (Assume solid) and weight to 1.0
        self.tsdf.fill(-1.0)
        self.weights.fill(1.0)
        
        height, width = 2592, 4608
        voxel_in_background = np.zeros(self.num_voxels, dtype=bool)
        
        for i, path in enumerate(image_paths):
            print(f"Carving visual hull frame {i+1}/{len(image_paths)}: {os.path.basename(path)}")
            raw_img = cv2.imread(path)
            img = cv2.undistort(raw_img, self.K, dist_cal, None, self.K)
            
            # Get silhouette mask
            mask = extractor.get_silhouette_mask(img, CENTER_2D, PLATE_RADIUS_PIXELS)
            
            R, t = get_camera_pose_fn(i)
            
            # Project voxels to camera
            cam_pts = np.dot(R, self.voxel_coords) + t.reshape(3, 1)
            voxel_depths = cam_pts[2, :]
            
            valid_z = voxel_depths > 1.0
            
            u = (self.K[0,0] * cam_pts[0, :] / voxel_depths) + self.K[0,2]
            v = (self.K[1,1] * cam_pts[1, :] / voxel_depths) + self.K[1,2]
            u = np.round(u).astype(int)
            v = np.round(v).astype(int)
            
            valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            valid_mask = valid_z & valid_uv
            
            check_u = u[valid_mask]
            check_v = v[valid_mask]
            pixel_values = mask[check_v, check_u]
            
            hit_bg = pixel_values == 0
            
            bg_indices = np.where(valid_mask)[0][hit_bg]
            voxel_in_background[bg_indices] = True
            
            outside_indices = np.where(~valid_mask)[0]
            voxel_in_background[outside_indices] = True
            
        # Set background voxels to empty and lock weight
        self.tsdf.ravel()[voxel_in_background] = 1.0
        self.weights.ravel()[voxel_in_background] = MAX_WEIGHT
        
        print(f"Visual Hull Initialization Complete. Solid voxels remaining: {np.sum(~voxel_in_background)}")

if __name__ == "__main__":
    # Create output directories
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    depth_maps_dir = os.path.join(OUTPUT_FOLDER, "depth_maps")
    masks_dir = os.path.join(OUTPUT_FOLDER, "masks")
    os.makedirs(depth_maps_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Direct logger to log file and console
    log_file = os.path.join(OUTPUT_FOLDER, "tsdf_run.log")
    sys.stdout = Logger(log_file)
    
    print("=== STARTING MODULAR TSDF FUSION PIPELINE ===")
    print(f"Log file created at: {log_file}")
    
    image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")),
                         key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    if len(image_paths) < 2:
        print("Error: Need at least 2 images for Stereo TSDF integration.")
        exit()
        
    # 1. Dynamic Camera Intrinsic Calibration stage
    K_cal, dist_cal = calibrate_camera_intrinsics(image_paths, fallback_K=DEFAULT_K, fallback_dist=DEFAULT_DIST)
    
    # 2. Dynamic Turntable Trajectory Calibration stage
    C_rot, normal, step_size_deg = calibrate_turntable_trajectory(
        image_paths, K_cal, dist_cal, 
        fallback_C_rot=DEFAULT_C_ROT, fallback_normal=DEFAULT_NORMAL, fallback_step_size=DEFAULT_STEP_SIZE
    )
    
    # Initialize the modular silhouette extraction model
    extractor = SilhouetteExtractor()
    
    # 3. Construct basis dynamically
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal) * normal
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal, x_col)
    R_cam = np.column_stack((x_col, y_col, normal))
    
    volume = TSDFVolume(GRID_RESOLUTION, GRID_SIZE_MM, K_cal)
    
    def get_camera_pose_for_frame(frame_index):
        """Calculates the 3D pose of the camera relative to the plate at a specific frame."""
        angle_deg = frame_index * step_size_deg
        angle_rad = np.deg2rad(angle_deg)
        R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
        R_eff = np.dot(R_i, R_cam)
        t_eff = C_rot
        return R_eff, t_eff
        
    # Pre-initialize TSDF volume using the visual hull
    volume.initialize_with_visual_hull(image_paths, extractor, get_camera_pose_for_frame, dist_cal)
        
    for i in range(len(image_paths)):
        img1_path = image_paths[i]
        img2_path = image_paths[(i + 1) % len(image_paths)]
        
        print(f"\nProcessing Stereo Pair {i+1}/{len(image_paths)}: {os.path.basename(img1_path)} & {os.path.basename(img2_path)}")
        
        # Load and UNDISTORT images
        img1 = cv2.undistort(cv2.imread(img1_path), K_cal, dist_cal)
        img2 = cv2.undistort(cv2.imread(img2_path), K_cal, dist_cal)
        
        # Get Positional Data
        R1, t1 = get_camera_pose_for_frame(i)
        R2, t2 = get_camera_pose_for_frame((i + 1) % len(image_paths))
        
        # Generate data
        silhouette = extractor.get_silhouette_mask(img1, CENTER_2D, PLATE_RADIUS_PIXELS)
        depth_map = compute_stereo_depth(img1, img2, R1, t1, R2, t2, K_cal, dist_cal)
        
        # Save intermediate results
        mask_save_path = os.path.join(masks_dir, f"mask_{i:02d}.png")
        cv2.imwrite(mask_save_path, silhouette)
        
        # For depth map visualization, normalize depth to 0-255 and save
        max_depth = np.max(depth_map)
        if max_depth > 0:
            depth_visual = (depth_map / max_depth * 255.0).astype(np.uint8)
        else:
            depth_visual = np.zeros_like(depth_map, dtype=np.uint8)
        depth_save_path = os.path.join(depth_maps_dir, f"depth_{i:02d}.png")
        cv2.imwrite(depth_save_path, depth_visual)
        
        # Integrate into TSDF
        valid_depth_pixels = np.sum(depth_map > 0)
        print(f"  -> Integrating {valid_depth_pixels} valid depth pixels into Volume...")
        volume.integrate(depth_map, silhouette, R1, t1)
        
    # Finalize
    mesh_path = os.path.join(OUTPUT_FOLDER, "clay_tsdf_mesh.obj")
    volume.extract_mesh(mesh_path, image_paths, get_camera_pose_for_frame, extractor, dist_cal)
    
    print("\n=== TSDF PIPELINE COMPLETED SUCCESSFULLY ===")
