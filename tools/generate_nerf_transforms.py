import json
import numpy as np
import os
import argparse
from scipy.spatial.transform import Rotation as Rot

def get_c2w_matrices(calibration_json_path: str, scale: float = 1.0):
    with open(calibration_json_path, 'r') as f:
        calib = json.load(f)

    # Plate center in camera coordinates (mm)
    C_rot = np.array(calib['plate_center_mm'])
    
    # Plate normal in camera coordinates
    normal = np.array(calib['plate_normal'])
    normal = normal / np.linalg.norm(normal)
    
    # Base setup (EXACTLY mirroring prepare_nerf_0726.py)
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal) * normal
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal, x_col)
    R_cam = np.column_stack((x_col, y_col, normal))

    # Note: K is the 3x3 intrinsics matrix
    K = np.array(calib['camera_matrix_K'])

    # Nerfstudio transforms.json
    transforms = {
        "camera_model": "OPENCV",
        "w": int(4608 // scale),
        "h": int(2592 // scale),
        "fl_x": K[0, 0] / scale,
        "fl_y": K[1, 1] / scale,
        "cx": K[0, 2] / scale,
        "cy": K[1, 2] / scale,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": []
    }

    step_size_deg = calib['step_size_deg']
    # If the calib script detected a negative step, we just use it directly
    if calib.get('step_sign', 'positive') == 'negative':
        pass # Already handled by the original script? Wait, 0726 had step_size_deg = 6.01, and step_sign = negative.
        # But prepare_nerf_0726.py ignored step_sign and just did: idx * step_size_deg.
        # Let's do exactly what prepare_nerf_0726.py did!
    
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    
    for idx in range(calib['n_images']):
        angle_deg = idx * step_size_deg
        angle_rad = np.deg2rad(angle_deg)
        
        # Exact same rotation logic as 0726
        R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
        R_eff = np.dot(R_i, R_cam)
        
        w2c = np.eye(4)
        w2c[:3, :3] = R_eff
        w2c[:3, 3] = C_rot
        
        c2w = np.linalg.inv(w2c)
        c2w[:3, 3] /= 150.0
        c2w_opengl = np.dot(c2w, flip_yz)
        
        transforms['frames'].append({
            "file_path": f"images_{int(scale)}/capture_{idx}.png",
            "mask_path": f"masks_{int(scale)}/mask_capture_{idx}.png",
            "transform_matrix": c2w_opengl.tolist()
        })
        
    return transforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    transforms = get_c2w_matrices(args.calib, args.scale)
    with open(args.output, "w") as f:
        json.dump(transforms, f, indent=4)
    print(f"Successfully wrote {args.output}")
