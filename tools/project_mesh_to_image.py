import open3d as o3d
import cv2
import json
import numpy as np
import argparse
import os

def project_mesh(mesh_path, transforms_path, output_dir):
    print(f"Loading mesh from {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print(f"Loading transforms from {transforms_path}...")
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)

    # Intrinsics
    fl_x = transforms['fl_x']
    fl_y = transforms['fl_y']
    cx = transforms['cx']
    cy = transforms['cy']
    w = transforms['w']
    h = transforms['h']
    
    K = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ])
    
    os.makedirs(output_dir, exist_ok=True)

    # Pick a few specific frames to render
    frames_to_render = [0, len(transforms['frames'])//4, len(transforms['frames'])//2]
    
    for frame_idx in frames_to_render:
        frame = transforms['frames'][frame_idx]
        img_path = frame['file_path']
        
        # Determine relative or absolute path
        base_dir = os.path.dirname(transforms_path)
        img_full_path = os.path.join(base_dir, img_path)
        
        if not os.path.exists(img_full_path):
            print(f"Cannot find image: {img_full_path}")
            continue

        print(f"Projecting onto {img_path}...")
        img = cv2.imread(img_full_path)
        if img is None:
            continue
            
        c2w = np.array(frame['transform_matrix'])
        # NeRF to OpenCV camera convention
        # c2w is camera-to-world. w2c is world-to-camera
        w2c = np.linalg.inv(c2w)
        
        # c2w uses OpenGL convention (y up, z back)
        # We need OpenCV convention (y down, z forward)
        R_gl2cv = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        
        w2c[:3, :3] = R_gl2cv @ w2c[:3, :3]
        w2c[:3, 3] = R_gl2cv @ w2c[:3, 3]

        # Transform vertices
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        pts_cam = (R @ vertices.T).T + t
        
        zc = pts_cam[:, 2]
        valid = zc > 1e-5
        
        u = np.round(K[0, 0] * pts_cam[:, 0] / zc + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * pts_cam[:, 1] / zc + K[1, 2]).astype(int)
        
        overlay = img.copy()
        
        edges = set()
        for tri in triangles:
            edges.add(tuple(sorted((tri[0], tri[1]))))
            edges.add(tuple(sorted((tri[1], tri[2]))))
            edges.add(tuple(sorted((tri[2], tri[0]))))
            
        for p1, p2 in edges:
            if valid[p1] and valid[p2]:
                if (0 <= u[p1] < w and 0 <= v[p1] < h and 
                    0 <= u[p2] < w and 0 <= v[p2] < h):
                    cv2.line(overlay, (u[p1], v[p1]), (u[p2], v[p2]), (0, 255, 255), 2)
                    
        alpha = 0.6
        blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # Scale down for output preview
        blended = cv2.resize(blended, (1152, 648))
        
        out_name = f"projection_{os.path.basename(img_path)}"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, blended)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    project_mesh(args.mesh, args.transforms, args.output_dir)
