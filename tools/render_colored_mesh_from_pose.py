import open3d as o3d
import json
import numpy as np
import argparse
import os
import cv2

def render_from_pose(mesh_path, transforms_path, output_dir):
    print(f"Loading mesh from {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
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
    
    os.makedirs(output_dir, exist_ok=True)

    # Pick a few specific frames to render
    frames_to_render = [0, len(transforms['frames'])//4, len(transforms['frames'])//2]
    
    for frame_idx in frames_to_render:
        frame = transforms['frames'][frame_idx]
        img_path = frame['file_path']
        
        print(f"Rendering view matching {img_path}...")
            
        c2w = np.array(frame['transform_matrix'])
        w2c = np.linalg.inv(c2w)
        
        R_gl2cv = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        
        w2c[:3, :3] = R_gl2cv @ w2c[:3, :3]
        w2c[:3, 3] = R_gl2cv @ w2c[:3, 3]

        R = w2c[:3, :3]
        t = w2c[:3, 3]
        
        # Transform all vertices to camera space
        pts_cam = (R @ vertices.T).T + t
        zc = pts_cam[:, 2]
        
        # Project to 2D
        K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
        u = np.round(K[0, 0] * pts_cam[:, 0] / np.maximum(zc, 1e-5) + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * pts_cam[:, 1] / np.maximum(zc, 1e-5) + K[1, 2]).astype(int)
        
        # Create blank image
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Sort triangles by depth (Painter's algorithm)
        # Compute average Z for each triangle
        tri_z = np.mean(zc[triangles], axis=1)
        # Sort descending (furthest first)
        sorted_tri_idx = np.argsort(tri_z)[::-1]
        
        colors = np.asarray(mesh.vertex_colors) * 255.0
        
        for idx in sorted_tri_idx:
            tri = triangles[idx]
            p1, p2, p3 = tri
            
            # Check if triangle is behind camera
            if zc[p1] < 0 or zc[p2] < 0 or zc[p3] < 0:
                continue
                
            pts = np.array([[u[p1], v[p1]], [u[p2], v[p2]], [u[p3], v[p3]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Simple color averaging for the face
            avg_color = np.mean(colors[tri], axis=0)
            # OpenCV expects BGR
            bgr_color = (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))
            
            cv2.fillConvexPoly(img, pts, bgr_color)
            
        # Scale down for output preview
        img = cv2.resize(img, (1152, 648))
        
        out_name = f"render_{os.path.basename(img_path)}"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, img)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    render_from_pose(args.mesh, args.transforms, args.output_dir)
