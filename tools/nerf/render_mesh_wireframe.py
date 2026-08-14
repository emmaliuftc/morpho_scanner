import cv2
import numpy as np
import glob
import os
import json
import argparse
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--transforms", required=True, help="Path to transforms.json used for training")
parser.add_argument("--images_dir", required=True, help="Path to raw full-res images")
parser.add_argument("--calib", required=True, help="Original calibration for K and dist")
args = parser.parse_args()

with open(args.calib, "r") as f:
    cal_data = json.load(f)

K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float64)
dist_cal = np.array(cal_data["distortion_coefficients"], dtype=np.float64)

with open(args.transforms, "r") as f:
    transforms_data = json.load(f)

frames = transforms_data["frames"]
# Create a mapping from image index to c2w_opengl
c2w_map = {}
for frame in frames:
    file_path = frame["file_path"]
    basename = os.path.basename(file_path)
    # capture_0.png or capture_0.jpg
    idx = int(basename.split('_')[-1].split('.')[0])
    c2w_map[idx] = np.array(frame["transform_matrix"], dtype=np.float64)

print(f"Loading mesh {args.mesh}")
mesh = o3d.io.read_triangle_mesh(args.mesh)
vertices_opengl = np.asarray(mesh.vertices) # (N, 3)
triangles = np.asarray(mesh.triangles)

edges = set()
for tri in triangles:
    edges.add(tuple(sorted((tri[0], tri[1]))))
    edges.add(tuple(sorted((tri[1], tri[2]))))
    edges.add(tuple(sorted((tri[2], tri[0]))))

image_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")),
                     key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

W, H = 4608, 2592
os.makedirs(args.out_dir, exist_ok=True)

flip_yz = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

for idx in [0, 16, 32, 48]:
    if idx < len(image_paths) and idx in c2w_map:
        img_path = image_paths[idx]
        raw_img = cv2.imread(img_path)
        img = cv2.undistort(raw_img, K_cal, dist_cal, None, K_cal)

        c2w_opengl = c2w_map[idx] # 4x4 matrix
        
        # Invert to get w2c_opengl
        w2c_opengl = np.linalg.inv(c2w_opengl)
        
        # Add homogeneous coordinate to vertices
        ones = np.ones((vertices_opengl.shape[0], 1))
        v_homog = np.hstack((vertices_opengl, ones)) # (N, 4)
        
        # Transform to camera space
        v_cam_opengl = (w2c_opengl @ v_homog.T).T # (N, 4)
        v_cam_opengl = v_cam_opengl[:, :3] # (N, 3)
        
        # Convert OpenGL camera -> OpenCV camera
        v_cam_opencv = (flip_yz @ v_cam_opengl.T).T
        
        zc = v_cam_opencv[:, 2]
        zc_safe = np.where(zc > 1e-5, zc, 1e-5)

        u = np.round(K_cal[0, 0] * v_cam_opencv[:, 0] / zc_safe + K_cal[0, 2]).astype(int)
        v = np.round(K_cal[1, 1] * v_cam_opencv[:, 1] / zc_safe + K_cal[1, 2]).astype(int)

        overlay = img.copy()
        for p1, p2 in edges:
            if (zc[p1] > 0 and zc[p2] > 0 and
                0 <= u[p1] < W and 0 <= v[p1] < H and
                0 <= u[p2] < W and 0 <= v[p2] < H):
                cv2.line(overlay, (u[p1], v[p1]), (u[p2], v[p2]), (0, 255, 255), 2)

        blended = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        blended_small = cv2.resize(blended, (1152, 648))

        out_proj_path = os.path.join(args.out_dir, f"mesh_projection_{idx:02d}.jpg")
        cv2.imwrite(out_proj_path, blended_small)
        print(f"Saved {out_proj_path}")
