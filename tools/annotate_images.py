import json
import numpy as np
import cv2
import os
import argparse
from scipy.spatial.transform import Rotation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_dir", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    with open(args.calib, "r") as f:
        calib = json.load(f)

    with open(os.path.join(args.session_dir, "transforms_aligned.json"), "r") as f:
        transforms = json.load(f)

    K = np.array(calib['camera_matrix_K'])
    dist = np.array(calib['distortion_coefficients'])
    P_c = np.array(calib['plate_center_mm'])
    normal = np.array(calib['plate_normal'])
    normal = normal / np.linalg.norm(normal)

    # 3D points
    center_3d = P_c
    z_axis_tip_3d = P_c - 100.0 * normal # 10cm Z axis length

    # The images are already undistorted when saved in images_4 or masked_images_preview_4!
    # BUT they are scaled down by args.scale.
    # So we need to scale down the camera matrix K.
    K_scaled = K.copy()
    K_scaled[0, 0] /= args.scale
    K_scaled[1, 1] /= args.scale
    K_scaled[0, 2] /= args.scale
    K_scaled[1, 2] /= args.scale

    # Project 3D points to 2D
    # Since images are ALREADY undistorted, we use zero distortion.
    zero_dist = np.zeros(4)
    # The points P_c and z_axis_tip_3d are in camera space already! 
    # (cv2.solvePnP returned camera relative coords for the marker, and the calibration assumes camera is at origin [0,0,0]).
    # So rvec and tvec are simply zeros for projecting world->camera since world IS camera space in calibration.json!
    rvec = np.zeros(3)
    tvec = np.zeros(3)

    pts_3d = np.array([center_3d, z_axis_tip_3d])
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K_scaled, zero_dist)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)

    c_2d = tuple(pts_2d[0])
    z_2d = tuple(pts_2d[1])

    in_dir = os.path.join(args.session_dir, f"masked_images_preview_{args.scale}")
    out_dir = os.path.join(args.session_dir, f"masked_with_annotation_{args.scale}")
    os.makedirs(out_dir, exist_ok=True)

    # We need to extract the angle from the transforms
    # Frame 0 angle is our baseline if we want relative, but let's just get the absolute Z angle.
    # Or even better, we can just compute the relative angle from Frame 0.
    frame0_mat = np.array(transforms["frames"][0]["transform_matrix"])
    r0 = Rotation.from_matrix(frame0_mat[:3, :3])
    euler0 = r0.as_euler("xyz", degrees=True)
    base_z = euler0[2]

    for i, frame in enumerate(transforms["frames"]):
        img_path = os.path.join(in_dir, f"preview_capture_{i}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(in_dir, f"preview_capture_{i}.jpg")
        if not os.path.exists(img_path):
            img_path = frame["file_path"].replace(f"images_{args.scale}", in_dir).replace("capture", "preview_capture")
        
        if not os.path.exists(img_path):
            print(f"Skipping {img_path}, not found.")
            continue

        img = cv2.imread(img_path)

        # Draw Center
        cv2.circle(img, c_2d, 5, (0, 0, 255), -1)
        # Draw Z Axis
        cv2.line(img, c_2d, z_2d, (0, 255, 0), 3)
        cv2.putText(img, "Z", z_2d, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, "Center", (c_2d[0]+10, c_2d[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Get angle
        mat = np.array(frame["transform_matrix"])
        r = Rotation.from_matrix(mat[:3, :3])
        euler = r.as_euler("xyz", degrees=True)
        angle_z = (euler[2] - base_z) % 360.0

        cv2.putText(img, f"Frame {i}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(img, f"Turntable Angle: {angle_z:.1f} deg", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        out_path = os.path.join(out_dir, f"annotated_capture_{i}.jpg")
        cv2.imwrite(out_path, img)

    print(f"Annotated images saved to {out_dir}")

if __name__ == "__main__":
    main()
