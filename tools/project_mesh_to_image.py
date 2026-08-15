import open3d as o3d
import numpy as np
import cv2
import json
import argparse
import os

def project_mesh(mesh_path, transforms_path, image_name, bg_image_path, output_path, alpha=0.5):
    print(f"Loading from {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors) * 255.0 if mesh.has_vertex_colors() else None
    
    # If no vertices loaded, it might be a point cloud instead of a mesh
    if len(vertices) == 0:
        print("Falling back to reading as point cloud...")
        pcd = o3d.io.read_point_cloud(mesh_path)
        vertices = np.asarray(pcd.points)
        triangles = np.array([])
        colors = np.asarray(pcd.colors) * 255.0 if pcd.has_colors() else None
    
    print(f"Loading transforms from {transforms_path}...")
    with open(transforms_path, 'r') as f:
        meta = json.load(f)
        
    width = meta['w']
    height = meta['h']
    fl_x = meta['fl_x']
    fl_y = meta['fl_y']
    cx = meta['cx']
    cy = meta['cy']
    
    K = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ])
    
    # Find the specific camera for the image
    c2w = None
    for frame in meta['frames']:
        if image_name in frame['file_path']:
            c2w = np.array(frame['transform_matrix'])
            break
            
    if c2w is None:
        print(f"Error: Could not find camera for {image_name} in {transforms_path}")
        return
        
    print("Found Camera to World matrix:")
    print(c2w)
    
    # NeRF standard is OpenGL (X right, Y up, Z backward)
    # OpenCV is X right, Y down, Z forward
    # C2W takes points in Camera space to World space.
    # W2C takes points from World space to Camera space.
    w2c = np.linalg.inv(c2w)
    
    # Transform vertices to Camera space
    pts_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    pts_cam = (w2c @ pts_homo.T).T
    pts_cam = pts_cam[:, :3]
    
    # Apply OpenGL to OpenCV camera coordinate flip if needed
    # NeRF transforms.json stores OpenGL C2W matrices.
    # When we invert to W2C, pts_cam are in OpenGL camera space (Y up, Z back)
    # We want OpenCV camera space (Y down, Z forward) for projection
    pts_cam[:, 1] *= -1
    pts_cam[:, 2] *= -1
    
    zc = pts_cam[:, 2]
    
    # Project to 2D
    u = np.round(K[0, 0] * pts_cam[:, 0] / np.maximum(zc, 1e-5) + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pts_cam[:, 1] / np.maximum(zc, 1e-5) + K[1, 2]).astype(int)
    
    print(f"Loading background image {bg_image_path}...")
    bg_img = cv2.imread(bg_image_path)
    if bg_img is None:
        print(f"Error: Could not load {bg_image_path}")
        return
        
    # Resize bg_img if it doesn't match transforms
    if bg_img.shape[0] != height or bg_img.shape[1] != width:
        bg_img = cv2.resize(bg_img, (width, height))
        
    overlay = bg_img.copy()
    
    if len(triangles) > 0:
        # Draw wireframe
        print("Drawing projected wireframe onto image...")
        for tri in triangles:
            p1, p2, p3 = tri
            # Only draw if in front of camera
            if zc[p1] < 0 or zc[p2] < 0 or zc[p3] < 0:
                continue
                
            pts = np.array([[u[p1], v[p1]], [u[p2], v[p2]], [u[p3], v[p3]]], np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
    else:
        # Draw point cloud
        print("No triangles found, projecting as a point cloud...")
        for i in range(len(vertices)):
            if zc[i] > 0:
                color = (0, 255, 0)
                if colors is not None:
                    # OpenCV uses BGR natively for drawing
                    color = (int(colors[i][2]), int(colors[i][1]), int(colors[i][0]))
                cv2.circle(overlay, (u[i], v[i]), radius=1, color=color, thickness=-1)
                
    # Blend with original to make it transparent
    result = cv2.addWeighted(bg_img, 1.0 - alpha, overlay, alpha, 0)
    
    cv2.imwrite(output_path, result)
    print(f"Saved projection overlay to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--image_name", required=True, help="Name in JSON (e.g. capture_0)")
    parser.add_argument("--bg_image", required=True, help="Actual image to draw on")
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="Opacity of the overlay (0.0 to 1.0)")
    args = parser.parse_args()
    
    project_mesh(args.mesh, args.transforms, args.image_name, args.bg_image, args.output, args.alpha)
