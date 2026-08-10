import cv2
import numpy as np
import glob
import os
import json
import subprocess
import sys
from scipy.spatial.transform import Rotation as Rot

# Import modular pipeline components
from camera_calibration import calibrate_camera_intrinsics
from turntable_calibration import calibrate_turntable_trajectory

# ==========================================
# 1. PIPELINE CONFIGURATION
# ==========================================
INPUT_FOLDER = "captures_7-25_lob_with_checkbox"
OUTPUT_FOLDER = "captures_7-25_nerf_dataset"
IMAGES_OUT_FOLDER = os.path.join(OUTPUT_FOLDER, "images")
NUM_IMAGES = 32
RUN_NERF_TRAINING = False  # Training requires CUDA GPU, set to True to attempt training on CPU

# Fallback values for calibration
DEFAULT_K = np.array([
    [3565.4767, 0.0,       2304.0],
    [0.0,       3565.4767, 1296.0],
    [0.0,       0.0,       1.0]
], dtype=np.float32)

DEFAULT_DIST = np.array([[-0.44031736, 7.83951075, 0.0, 0.0, -43.24147593]], dtype=np.float32)

DEFAULT_NORMAL = np.array([0.01059424, 0.63511793, 0.77234253])
DEFAULT_C_ROT = np.array([-2.97537598, -24.52457008, 150.91695089])
DEFAULT_STEP_SIZE = 11.552961

def get_c2w_matrix(frame_index, R_cam, C_rot, normal, step_size_deg):
    """
    Calculates the Camera-to-World (c2w) 4x4 matrix for NeRF/3DGS.
    World coordinates assume the clay object is stationary at (0,0,0) and the camera orbits it.
    """
    angle_deg = frame_index * step_size_deg
    angle_rad = np.deg2rad(angle_deg)
    
    # Rotation of the turntable at step i (using positive sign to match TSDF orbit direction with normal_up)
    R_i = Rot.from_rotvec(angle_rad * normal).as_matrix()
    
    # Effective camera pose relative to rotating object coordinate system
    R_eff = np.dot(R_i, R_cam)
    t_eff = C_rot
    
    # Construct 4x4 w2c matrix
    w2c = np.eye(4)
    w2c[:3, :3] = R_eff
    w2c[:3, 3] = t_eff
    
    # Camera-to-World (c2w) is the inverse of w2c
    c2w = np.linalg.inv(w2c)
    
    # Scale translation to unit sphere space for NeRF (keeping turntable center at 0,0,0)
    c2w[:3, 3] /= 150.0
    
    # --- CRITICAL STEP: OPENCV to OPENGL CONVERSION ---
    # OpenCV cameras look down +Z, with +Y down.
    # NeRF (OpenGL) expects cameras to look down -Z, with +Y up.
    flip_yz = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    c2w_opengl = np.dot(c2w, flip_yz)
    
    return c2w_opengl

def run_nerf_training(data_dir):
    """
    Executes the Nerfstudio training command end-to-end.
    """
    print("\n" + "="*50)
    print("🚀 STARTING NERFSTUDIO TRAINING")
    print("="*50)
    
    # Using the ns-train installed in the python3.11 virtualenv
    ns_train_bin = os.path.join(".venv_nerf", "bin", "ns-train")
    
    # If no GPU is available, train on CPU
    cmd = [ns_train_bin, "nerfacto", "--data", data_dir, "--device", "cpu"]
    
    print(f"Executing: {' '.join(cmd)}\n")
    
    try:
        process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        process.communicate() # Wait for training to finish
        
        if process.returncode == 0:
            print("\n✅ NeRF training completed successfully!")
        else:
            print(f"\n❌ Training exited with code {process.returncode}.")
            
    except FileNotFoundError:
        print(f"\n❌ Error: '{ns_train_bin}' command not found.")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Create output directories
    os.makedirs(IMAGES_OUT_FOLDER, exist_ok=True)
    
    image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")),
                        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    if len(image_paths) != NUM_IMAGES:
        print(f"Warning: Found {len(image_paths)} images, expected {NUM_IMAGES}.")
        
    print("\n--- RUNNING DYNAMIC CAMERA INTRINSIC CALIBRATION ---")
    K_cal, dist_cal = calibrate_camera_intrinsics(image_paths, fallback_K=DEFAULT_K, fallback_dist=DEFAULT_DIST)
    print(f"  -> Calibrated focal length fx: {K_cal[0, 0]:.2f}")
    
    print("\n--- RUNNING DYNAMIC TURNTABLE CALIBRATION ---")
    C_rot, normal, step_size_deg = calibrate_turntable_trajectory(
        image_paths, K_cal, dist_cal, 
        fallback_C_rot=DEFAULT_C_ROT, fallback_normal=DEFAULT_NORMAL, fallback_step_size=DEFAULT_STEP_SIZE
    )
    print(f"  -> Solved turntable rotation center: {C_rot}")
    print(f"  -> Solved step size: {step_size_deg:.4f} degrees per image")
    
    # Construct basis for camera space (negating normal to ensure +Z is physically pointing up)
    normal_up = -normal
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal_up) * normal_up
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal_up, x_col)
    R_cam = np.column_stack((x_col, y_col, normal_up))
    
    # Constants for green plate circle mask
    CENTER_2D = (2345, 1183)
    PLATE_RADIUS_PIXELS = 1100
    
    # Create masks directory
    MASKS_OUT_FOLDER = os.path.join(OUTPUT_FOLDER, "masks")
    os.makedirs(MASKS_OUT_FOLDER, exist_ok=True)
    
    # Initialize the NeRF transforms dictionary
    transforms_dict = {
        "camera_model": "OPENCV",
        "fl_x": float(K_cal[0, 0]),
        "fl_y": float(K_cal[1, 1]),
        "cx": float(K_cal[0, 2]),
        "cy": float(K_cal[1, 2]),
        "w": 4608,
        "h": 2592,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": []
    }

    print(f"\nExporting NeRF dataset to {OUTPUT_FOLDER}...")

    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        out_filename = filename.replace(".jpg", ".png")
        print(f"Processing frame {i+1}/{len(image_paths)}: {filename} -> {out_filename}")
        
        # 1. Load and Undistort
        raw_img = cv2.imread(path)
        img = cv2.undistort(raw_img, K_cal, dist_cal, None, K_cal)
        
        # 2. Load pre-saved silhouette mask if available to save processing time
        mask_path = os.path.join("captures_7-25_tsdf", "masks", f"mask_{i:02d}.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            from silhouette_extractor import SilhouetteExtractor
            extractor = SilhouetteExtractor()
            mask = extractor.get_silhouette_mask(img, CENTER_2D, PLATE_RADIUS_PIXELS)
        
        # 3. Save mask file to masks output folder
        out_mask_filename = f"mask_{i:02d}.png"
        cv2.imwrite(os.path.join(MASKS_OUT_FOLDER, out_mask_filename), mask)
        
        # 4. Save as transparent PNG (RGBA)
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask
        
        out_img_path = os.path.join(IMAGES_OUT_FOLDER, out_filename)
        cv2.imwrite(out_img_path, rgba)
        
        # 5. Calculate c2w transform (using normal_up)
        c2w = get_c2w_matrix(i, R_cam, C_rot, normal_up, step_size_deg)
        
        # Append to JSON format
        frame_data = {
            "file_path": f"images/{out_filename}",
            "mask_path": f"masks/{out_mask_filename}",
            "transform_matrix": c2w.tolist()
        }
        transforms_dict["frames"].append(frame_data)

    # Write transforms.json
    json_path = os.path.join(OUTPUT_FOLDER, "transforms.json")
    with open(json_path, 'w') as f:
        json.dump(transforms_dict, f, indent=4)
        
    print(f"\n✅ Success! NeRF dataset prepared at: {OUTPUT_FOLDER}")
    
    if RUN_NERF_TRAINING:
        run_nerf_training(OUTPUT_FOLDER)
    else:
        print(f"You can now run this folder through Nerfstudio or Gaussian Splatting.")
        print(f"Example: .venv_nerf/bin/ns-train nerfacto --data {OUTPUT_FOLDER}")
