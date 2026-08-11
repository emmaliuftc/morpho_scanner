import cv2
import numpy as np
import glob
import os
import imageio
from scipy.spatial.transform import Rotation as Rot

# Import modular pipeline components
from camera_calibration import calibrate_camera_intrinsics
from turntable_calibration import calibrate_turntable_trajectory

# ==========================================
# Turntable Axis & Normal Pole Visualizer
# ==========================================

INPUT_FOLDER = "captures_7-25_lob_with_checkbox"
OUTPUT_FOLDER = "captures_7-25_trajectory_visualization"
NUM_IMAGES = 32

# Default Fallbacks
DEFAULT_K = np.array([
    [3565.4767, 0.0,       2304.0],
    [0.0,       3565.4767, 1296.0],
    [0.0,       0.0,       1.0]
], dtype=np.float32)

DEFAULT_DIST = np.array([[-0.44031736, 7.83951075, 0.0, 0.0, -43.24147593]], dtype=np.float32)

DEFAULT_NORMAL = np.array([0.01059424, 0.63511793, 0.77234253])
DEFAULT_C_ROT = np.array([-2.97537598, -24.52457008, 150.91695089])
DEFAULT_STEP_SIZE = 11.552961

def draw_trajectory_visualization():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: {OUTPUT_FOLDER}")
        
    images = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")),
                    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    # 1. Dynamic Camera Intrinsic Calibration stage
    K_cal, dist_cal = calibrate_camera_intrinsics(images, fallback_K=DEFAULT_K, fallback_dist=DEFAULT_DIST)
    
    # 2. Dynamic Turntable Trajectory Calibration stage
    C_rot, normal, step_size_deg = calibrate_turntable_trajectory(
        images, K_cal, dist_cal, 
        fallback_C_rot=DEFAULT_C_ROT, fallback_normal=DEFAULT_NORMAL, fallback_step_size=DEFAULT_STEP_SIZE
    )
    
    # Construct base coordinate vectors on the turntable surface
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal) * normal
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal, x_col)
    
    # Create video writer
    video_path = os.path.join(OUTPUT_FOLDER, "turntable_trajectory_visualization.mp4")
    video_writer = imageio.get_writer(video_path, fps=8.0, codec='libx264')
    
    def project_points(pts_3d):
        u = (K_cal[0,0] * pts_3d[:, 0] / pts_3d[:, 2]) + K_cal[0,2]
        v = (K_cal[1,1] * pts_3d[:, 1] / pts_3d[:, 2]) + K_cal[1,2]
        return np.column_stack([u, v])
        
    print("\nRendering visualization frames...")
    for i, path in enumerate(images):
        raw_img = cv2.imread(path)
        # Undistort to match the pinhole projection coordinate frame
        img = cv2.undistort(raw_img, K_cal, dist_cal, None, K_cal)
        
        angle_deg = i * step_size_deg
        angle_rad = np.deg2rad(angle_deg)
        R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
        
        # 1. Render Axis Pole (Z-axis, Blue)
        pole_pts_3d = C_rot + np.outer(np.linspace(0, 80, 50), normal)
        pole_pts_2d = project_points(pole_pts_3d)
        for pt_idx in range(len(pole_pts_2d) - 1):
            pt1 = tuple(pole_pts_2d[pt_idx].astype(int))
            pt2 = tuple(pole_pts_2d[pt_idx+1].astype(int))
            cv2.line(img, pt1, pt2, (255, 0, 0), 4, cv2.LINE_AA) # Blue in BGR
            
        # 2. Render rotating local X-axis (Red)
        x_dir = np.dot(R_i, x_col)
        x_pts_3d = C_rot + np.outer(np.linspace(0, 50, 50), x_dir)
        x_pts_2d = project_points(x_pts_3d)
        for pt_idx in range(len(x_pts_2d) - 1):
            pt1 = tuple(x_pts_2d[pt_idx].astype(int))
            pt2 = tuple(x_pts_2d[pt_idx+1].astype(int))
            cv2.line(img, pt1, pt2, (0, 0, 255), 4, cv2.LINE_AA) # Red in BGR
            
        # 3. Render rotating local Y-axis (Green)
        y_dir = np.dot(R_i, y_col)
        y_pts_3d = C_rot + np.outer(np.linspace(0, 50, 50), y_dir)
        y_pts_2d = project_points(y_pts_3d)
        for pt_idx in range(len(y_pts_2d) - 1):
            pt1 = tuple(y_pts_2d[pt_idx].astype(int))
            pt2 = tuple(y_pts_2d[pt_idx+1].astype(int))
            cv2.line(img, pt1, pt2, (0, 255, 0), 4, cv2.LINE_AA) # Green in BGR
            
        # 4. Draw marker at center of rotation
        center_2d = project_points(C_rot[None, :])[0]
        cv2.drawMarker(img, tuple(center_2d.astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 30, 4, cv2.LINE_AA)
        
        # 5. Text Overlays
        cv2.putText(img, f"Frame: {i+1}/32", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, f"Turntable Angle: {angle_deg:.2f} deg", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, "Z-Axis (Blue Pole) | X-Axis (Red Hand) | Y-Axis (Green Hand)", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Save frame to file
        filename = f"trajectory_frame_{i:02d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), img)
        
        # Write to video
        video_writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print(f"  -> Rendered frame {i+1}/32")
        
    video_writer.close()
    print(f"\nSuccess! Trajectory visualization video saved to: {video_path}")

if __name__ == "__main__":
    draw_trajectory_visualization()
