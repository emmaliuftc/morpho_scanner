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
# Mesh Projection Overlay Visualizer
# ==========================================

INPUT_FOLDER = "captures_7-25_lob_with_checkbox"
OUTPUT_FOLDER = "captures_7-25_tsdf"
NUM_IMAGES = 32

DEFAULT_K = np.array([
    [3565.4767, 0.0,       2304.0],
    [0.0,       3565.4767, 1296.0],
    [0.0,       0.0,       1.0]
], dtype=np.float32)

DEFAULT_DIST = np.array([[-0.44031736, 7.83951075, 0.0, 0.0, -43.24147593]], dtype=np.float32)

DEFAULT_NORMAL = np.array([0.01059424, 0.63511793, 0.77234253])
DEFAULT_C_ROT = np.array([-2.97537598, -24.52457008, 150.91695089])
DEFAULT_STEP_SIZE = 11.552961

def load_mesh(obj_path):
    print(f"Loading mesh from {obj_path}...")
    vertices = []
    colors = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    # OBJ stores RGB as 0.0 - 1.0, convert to 0-255 BGR for OpenCV
                    colors.append([float(parts[6])*255.0, float(parts[5])*255.0, float(parts[4])*255.0])
                else:
                    colors.append([128.0, 128.0, 128.0])
                    
    return np.array(vertices), np.array(colors)

def generate_projections():
    obj_path = os.path.join(OUTPUT_FOLDER, "clay_tsdf_mesh.obj")
    if not os.path.exists(obj_path):
        print(f"Error: Mesh file {obj_path} not found.")
        return
        
    vertices, colors = load_mesh(obj_path)
    print(f"Loaded {len(vertices)} mesh vertices.")
    
    # Vertices are already in the correct local coordinate space
    local_verts = vertices
    
    images = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")),
                    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                    
    # 1. Dynamic Camera Intrinsic Calibration stage
    K_cal, dist_cal = calibrate_camera_intrinsics(images, fallback_K=DEFAULT_K, fallback_dist=DEFAULT_DIST)
    
    # 2. Dynamic Turntable Trajectory Calibration stage
    C_rot, normal, step_size_deg = calibrate_turntable_trajectory(
        images, K_cal, dist_cal, 
        fallback_C_rot=DEFAULT_C_ROT, fallback_normal=DEFAULT_NORMAL, fallback_step_size=DEFAULT_STEP_SIZE
    )
    
    # Construct basis dynamically
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal) * normal
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal, x_col)
    R_cam = np.column_stack((x_col, y_col, normal))
    
    projections_dir = os.path.join(OUTPUT_FOLDER, "mesh_projections")
    os.makedirs(projections_dir, exist_ok=True)
    
    # Create video writer
    video_path = os.path.join(OUTPUT_FOLDER, "mesh_projection_overlay.mp4")
    video_writer = imageio.get_writer(video_path, fps=8.0, codec='libx264')
    
    print("\nProjecting mesh vertices onto all 32 frames...")
    for i, path in enumerate(images):
        raw_img = cv2.imread(path)
        img = cv2.undistort(raw_img, K_cal, dist_cal, None, K_cal)
        height, width = img.shape[:2]
        
        # Camera pose for frame i
        angle_deg = i * step_size_deg
        angle_rad = np.deg2rad(angle_deg)
        R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
        R_eff = np.dot(R_i, R_cam)
        t_eff = C_rot
        
        # Project vertices
        cam_pts = np.dot(R_eff, local_verts.T) + t_eff.reshape(3, 1)
        voxel_depths = cam_pts[2, :]
        
        valid_z = voxel_depths > 1.0
        
        u = (K_cal[0,0] * cam_pts[0, :] / voxel_depths) + K_cal[0,2]
        v = (K_cal[1,1] * cam_pts[1, :] / voxel_depths) + K_cal[1,2]
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        
        valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        valid_mask = valid_z & valid_uv
        
        valid_u = u[valid_mask]
        valid_v = v[valid_mask]
        valid_colors = colors[valid_mask]
        
        # Create a copy to draw overlay
        overlay = img.copy()
        
        # Draw all projected vertices on the image
        # Using vectorized coloring by index assignment is extremely fast
        # To make it a solid overlay, we can draw small circles
        for pt_idx in range(len(valid_u)):
            cv2.circle(overlay, (valid_u[pt_idx], valid_v[pt_idx]), 2, tuple(map(int, valid_colors[pt_idx])), -1, cv2.LINE_AA)
            
        # Semi-transparent blend: 70% original image, 30% mesh point overlay
        blend = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)
        
        # Text overlays
        cv2.putText(blend, f"Frame: {i+1}/32", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(blend, f"Angle: {angle_deg:.2f} deg", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(blend, "Projected 3D TSDF Mesh Overlay", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Save frame to file
        filename = f"projection_{i:02d}.jpg"
        cv2.imwrite(os.path.join(projections_dir, filename), blend)
        
        # Write to video
        video_writer.append_data(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
        print(f"  -> Generated overlay frame {i+1}/32")
        
    video_writer.close()
    print(f"\nSuccess! Mesh projection overlay video saved to: {video_path}")

if __name__ == "__main__":
    generate_projections()
