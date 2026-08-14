import json
import numpy as np
import os
import argparse
import cv2
from scipy.spatial.transform import Rotation as Rot

def get_c2w_matrices(calibration_json_path: str, scale: float = 1.0, align_to_markers: bool = False, frame_0_path: str = ""):
    with open(calibration_json_path, 'r') as f:
        calib = json.load(f)

    C_rot = np.array(calib['plate_center_mm'])
    normal = np.array(calib['plate_normal'])
    normal = normal / np.linalg.norm(normal)
    K = np.array(calib['camera_matrix_K'])
    dist = np.array(calib['distortion_coefficients'])
    
    # We want a stable base frame R_cam.
    # The simplest base frame has Z = normal.
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal) * normal
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal, x_col)
    R_cam = np.column_stack((x_col, y_col, normal))

    starting_angle_rad = 0.0

    if align_to_markers and frame_0_path:
        print("Aligning Frame 0 to markers on the turntable...")
        img = cv2.imread(frame_0_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
        
        s = 15.0
        ox, oy = 20.7254, 20.2022
        m14 = np.array([[0,0,0],[s,0,0],[s,s,0],[0,s,0]], dtype=np.float32)
        m20 = np.array([[ox,oy,0],[ox+s,oy,0],[ox+s,oy+s,0],[ox,oy+s,0]], dtype=np.float32)
        board_pts = {14: m14, 20: m20}
        
        if ids is not None:
            ids = ids.flatten()
            obj_f, img_f = [], []
            for mid in [14, 20]:
                if mid in ids:
                    c = corners[list(ids).index(mid)][0]
                    obj_f.append(board_pts[mid])
                    img_f.append(c.astype(np.float64))
            if len(obj_f) > 0:
                obj_all = np.vstack(obj_f)
                img_all = np.vstack(img_f)
                ok, rvec, tvec = cv2.solvePnP(obj_all, img_all, K, dist)
                if ok:
                    # Marker origin in camera space
                    M_cam = tvec.ravel()
                    # Vector from C_rot to Marker Origin
                    vec = M_cam - C_rot
                    # Project onto the turntable plane
                    vec_planar = vec - np.dot(vec, normal) * normal
                    
                    # We want to measure the angle of vec_planar relative to our x_col, y_col basis
                    v_x = np.dot(vec_planar, x_col)
                    v_y = np.dot(vec_planar, y_col)
                    marker_angle_rad = np.arctan2(v_y, v_x)
                    
                    # We want the starting angle to be exactly -marker_angle_rad so that the marker always lands on the X-axis in world space
                    starting_angle_rad = -marker_angle_rad
                    print(f"Calculated starting phase angle: {np.rad2deg(starting_angle_rad):.2f} degrees")

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

    # Our scaling factor (174.0 would make the camera exact distance ~1.0)
    # The camera distance from center is norm(C_rot).
    cam_dist_mm = np.linalg.norm(C_rot)
    scale_factor = cam_dist_mm  # This forces camera to be exactly distance 1.0!

    step_size_deg = calib['step_size_deg']
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    
    # Normally we do 64 frames. I'll hardcode 64 or check the directory.
    # The Golden calib JSON has n_images, but let's assume 64 frames for three_flat.
    n_frames = 64
    
    for idx in range(n_frames):
        # We add the starting angle here!
        angle_rad = starting_angle_rad + np.deg2rad(idx * step_size_deg)
        
        R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
        R_eff = np.dot(R_i, R_cam)
        
        w2c = np.eye(4)
        w2c[:3, :3] = R_eff
        w2c[:3, 3] = C_rot
        
        c2w = np.linalg.inv(w2c)
        c2w[:3, 3] /= scale_factor  # Normalize camera distance to ~1.0
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
    parser.add_argument("--frame0", required=True)
    parser.add_argument("--output_unaligned", required=True)
    parser.add_argument("--output_aligned", required=True)
    parser.add_argument("--scale", type=float, default=4.0)
    args = parser.parse_args()

    # 1. Unaligned
    trans_unaligned = get_c2w_matrices(args.calib, args.scale, align_to_markers=False)
    with open(args.output_unaligned, "w") as f:
        json.dump(trans_unaligned, f, indent=4)
    print(f"Wrote {args.output_unaligned}")

    # 2. Aligned
    trans_aligned = get_c2w_matrices(args.calib, args.scale, align_to_markers=True, frame_0_path=args.frame0)
    with open(args.output_aligned, "w") as f:
        json.dump(trans_aligned, f, indent=4)
    print(f"Wrote {args.output_aligned}")
