import os
import cv2
import json
import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as Rot

# Config
INPUT_DIR = "captures_0726_clay_checkboard_64_calibrated"
MASKS_DIR = "optimization_0813/masks"
OUTPUT_DIR = "optimization_0813"
GRID_RESOLUTION = 256
GRID_SIZE_MM = 120.0

def load_calibration():
    calib_json_path = os.path.join(INPUT_DIR, "calibration_results.json")
    with open(calib_json_path, "r") as f:
        cal_data = json.load(f)
    
    K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float64)
    dist_cal = np.array(cal_data["distortion_coefficients"], dtype=np.float64)
    C_rot = np.array(cal_data["plate_center_mm"], dtype=np.float64)
    normal = np.array(cal_data["plate_normal"], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    step_size_deg = float(cal_data["step_size_deg"])
    
    return K_cal, dist_cal, C_rot, normal, step_size_deg

def carve_voxels(K_cal, C_rot, normal, step_size_deg):
    print(f"Initializing {GRID_RESOLUTION}^3 Voxel Grid...")
    
    # Establish coordinate basis
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal) * normal
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal, x_col)
    R_cam = np.column_stack((x_col, y_col, normal))
    
    half_size = GRID_SIZE_MM / 2.0
    x = np.linspace(-half_size, half_size, GRID_RESOLUTION)
    y = np.linspace(-half_size, half_size, GRID_RESOLUTION)
    z = np.linspace(-15.0, GRID_SIZE_MM - 15.0, GRID_RESOLUTION)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points_local = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    grid_points_cam0 = np.dot(grid_points_local, R_cam.T) + C_rot
    
    voxel_states = np.ones(GRID_RESOLUTION**3, dtype=bool)
    
    # Iterate over 64 masks
    for i in range(64):
        mask_path = os.path.join(MASKS_DIR, f"mask_{i:02d}.png")
        if not os.path.exists(mask_path):
            print(f"Warning: {mask_path} missing!")
            continue
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape
        
        angle_rad = np.deg2rad(i * step_size_deg)
        R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
        
        active_indices = np.where(voxel_states)[0]
        active_points = grid_points_cam0[active_indices]
        
        # Project active voxels to camera
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
        
        # Check against Phase 1 Binary Mask
        pixel_values = mask[check_v, check_u]
        hit_background = pixel_values == 0
        
        points_to_carve_indices = np.where(valid_mask)[0][hit_background]
        global_indices_to_carve = active_indices[points_to_carve_indices]
        
        voxel_states[global_indices_to_carve] = False
        if i % 16 == 0 or i == 63:
            print(f"  Frame {i:02d}: {np.sum(voxel_states)} voxels remain.")
            
    voxel_grid = voxel_states.reshape((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION))
    return voxel_grid, (x, y, z)

def compute_sdt(voxel_grid):
    print("Computing Exact Signed Distance Transform (SDT)...")
    # Inside = negative distance to surface, Outside = positive distance
    
    # distance to background (0)
    dist_inside = ndimage.distance_transform_edt(voxel_grid)
    
    # distance to foreground (1)
    dist_outside = ndimage.distance_transform_edt(~voxel_grid)
    
    # Combine: inside is negative, outside is positive
    sdt = dist_outside - dist_inside
    
    # Convert voxel distance to millimeter distance
    voxel_size_mm = GRID_SIZE_MM / GRID_RESOLUTION
    sdt_mm = sdt * voxel_size_mm
    return sdt_mm

def main():
    K_cal, dist_cal, C_rot, normal, step_size_deg = load_calibration()
    
    # 1. Project 2D masks into 3D binary volume
    voxel_grid, (x, y, z) = carve_voxels(K_cal, C_rot, normal, step_size_deg)
    
    # 2. Calculate continuous SDT field
    sdt_mm = compute_sdt(voxel_grid)
    
    # 3. Create Continuous Mathematical Function (RegularGridInterpolator)
    print("Wrapping SDT in continuous RegularGridInterpolator...")
    sdt_interpolator = RegularGridInterpolator((x, y, z), sdt_mm, bounds_error=False, fill_value=GRID_SIZE_MM)
    
    # Save artifacts for visualization
    np.save(os.path.join(OUTPUT_DIR, "sdt_volume.npy"), sdt_mm)
    print(f"Saved SDT volume to {OUTPUT_DIR}/sdt_volume.npy")

    # Generate a visual slice of the SDT
    mid_z = GRID_RESOLUTION // 2
    slice_2d = sdt_mm[:, :, mid_z]
    
    # Normalize for visualization (blue = inside, red = outside)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    plt.imshow(slice_2d, cmap='coolwarm', origin='lower')
    plt.contour(slice_2d, levels=[0], colors='black', linewidths=2) # Draw the zero-level set (surface)
    plt.title(f"SDT Cross Section (Z = {z[mid_z]:.2f} mm)\nBlack Line = Implicit Surface")
    plt.colorbar(label="Distance to surface (mm)")
    plt.savefig(os.path.join(OUTPUT_DIR, "sdt_slice_preview.png"))
    print(f"Saved SDT slice visualization to {OUTPUT_DIR}/sdt_slice_preview.png")

if __name__ == "__main__":
    main()
