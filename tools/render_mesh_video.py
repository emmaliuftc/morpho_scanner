import open3d as o3d
import numpy as np
import cv2
import argparse
import os

def render_mesh_orbit(ply_path, output_mp4, width=1280, height=720, frames=120):
    print(f"Loading colored mesh from {ply_path}...")
    mesh = o3d.io.read_triangle_mesh(ply_path)
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors) * 255.0
    
    if len(vertices) == 0:
        print("Error: Mesh is empty!")
        return

    # Center the mesh
    center = np.mean(vertices, axis=0)
    vertices -= center
    
    # Calculate bounding box to frame the camera
    max_radius = np.max(np.linalg.norm(vertices, axis=1))
    
    # Camera intrinsics
    fov = 45.0
    focal = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    K = np.array([
        [focal, 0, width / 2.0],
        [0, focal, height / 2.0],
        [0, 0, 1]
    ])
    
    # Camera distance based on bounding box
    camera_z = max_radius * 2.5
    
    frames_list = []
    
    print(f"Rendering {frames} frames directly to GIF...")
    for i in range(frames):
        angle = (i / frames) * 2 * np.pi
        
        # Orbit around Y axis
        R_y = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        
        # Tilt slightly down
        tilt = np.deg2rad(15)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(tilt), -np.sin(tilt)],
            [0, np.sin(tilt), np.cos(tilt)]
        ])
        
        R = R_x @ R_y
        
        # Transform all vertices to camera space
        pts_cam = (R @ vertices.T).T
        pts_cam[:, 2] += camera_z
        zc = pts_cam[:, 2]
        
        # Project to 2D
        u = np.round(K[0, 0] * pts_cam[:, 0] / np.maximum(zc, 1e-5) + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * pts_cam[:, 1] / np.maximum(zc, 1e-5) + K[1, 2]).astype(int)
        
        # Create blank image (RGB)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sort triangles by depth (Painter's algorithm)
        tri_z = np.mean(zc[triangles], axis=1)
        sorted_tri_idx = np.argsort(tri_z)[::-1]
        
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
            # Use RGB color directly
            rgb_color = (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
            
            cv2.fillConvexPoly(img, pts, rgb_color)
            
        frames_list.append(img)
        
        if (i+1) % 5 == 0:
            print(f"Rendered {i+1}/{frames} frames...")
            
    print("Encoding GIF... this might take a moment.")
    import imageio
    imageio.mimsave(output_mp4, frames_list, fps=15, loop=0)
    print(f"GIF directly saved to {output_mp4}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    render_mesh_orbit(args.ply, args.out)
