import json
import numpy as np
import os

def generate_metric_transforms():
    calib_file = "captures_0726_clay_checkboard_64_calibrated/calibration_results.json"
    output_dir = "captures_0726_nerf_metric"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(calib_file, 'r') as f:
        calib = json.load(f)

    # 1. Base math setup (same as new pipeline)
    P_c = np.array(calib['plate_center_mm'])
    N = np.array(calib['plate_normal'])
    N = N / np.linalg.norm(N)
    
    Z_obj = N
    temp = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(Z_obj, temp)) > 0.99:
        temp = np.array([0.0, 1.0, 0.0])
    X_obj = np.cross(temp, Z_obj)
    X_obj = X_obj / np.linalg.norm(X_obj)
    
    Y_obj = np.cross(Z_obj, X_obj)
    Y_obj = Y_obj / np.linalg.norm(Y_obj)
    
    R_base_to_cam = np.column_stack((X_obj, Y_obj, Z_obj))
    T_base_to_cam = np.eye(4)
    T_base_to_cam[:3, :3] = R_base_to_cam
    T_base_to_cam[:3, 3] = P_c

    K = np.array(calib['camera_matrix_K'])
    dist = calib['distortion_coefficients']

    # Downscale resolution by 8 like the original 0726 prep script
    scale_factor = 8.0
    w = 4608 // 8
    h = 2592 // 8

    transforms = {
        "camera_model": "OPENCV",
        "w": w,
        "h": h,
        "fl_x": K[0, 0] / scale_factor,
        "fl_y": K[1, 1] / scale_factor,
        "cx": K[0, 2] / scale_factor,
        "cy": K[1, 2] / scale_factor,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": []
    }

    step_size = calib['step_size_deg']
    
    for idx in range(calib['n_images']):
        theta_deg = idx * step_size
        theta = np.radians(theta_deg)
        
        R_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        T_obj_to_base = np.eye(4)
        T_obj_to_base[:3, :3] = R_z
        
        T_obj_to_cam = T_base_to_cam @ T_obj_to_base
        T_cam_to_obj = np.linalg.inv(T_obj_to_cam)
        
        # OpenGL flip
        T_cam_to_obj[:, 1:3] *= -1.0
        
        # Apply metric scaling (Meters = / 1000.0) instead of the old (/ 150.0)
        T_cam_to_obj[:3, 3] /= 1000.0
        
        transforms['frames'].append({
            "file_path": f"capture_{idx}.png",
            "mask_path": f"mask_capture_{idx}.png",
            "transform_matrix": T_cam_to_obj.tolist()
        })
        
    out_file = os.path.join(output_dir, "transforms.json")
    with open(out_file, 'w') as f:
        json.dump(transforms, f, indent=4)
        
    print(f"Generated {out_file} using absolute metric scaling (Meters)")

if __name__ == "__main__":
    generate_metric_transforms()
