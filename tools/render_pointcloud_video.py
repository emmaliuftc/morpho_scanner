import open3d as o3d
import numpy as np
import cv2
import imageio
import argparse
import os

def render_pointcloud_orbit(ply_path, output_mp4, width=1280, height=720, frames=120):
    print(f"Loading point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255.0
    colors = colors.astype(np.uint8)
    
    if len(points) == 0:
        print("Error: Point cloud is empty!")
        return

    # Center the point cloud
    center = np.mean(points, axis=0)
    points -= center
    
    # Calculate bounding box to frame the camera
    max_radius = np.max(np.linalg.norm(points, axis=1))
    
    # Camera intrinsics
    fov = 60.0
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
        
        # Orbit around Y axis (assume Y is up)
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
        
        # Transform points
        # Rotate points
        pts_rot = (R @ points.T).T
        
        # Move back by camera_z
        pts_rot[:, 2] += camera_z
        
        # Project to 2D
        zc = pts_rot[:, 2]
        
        # Filter points behind camera
        valid = zc > 1e-5
        pts_valid = pts_rot[valid]
        c_valid = colors[valid]
        zc_valid = zc[valid]
        
        u = np.round(K[0, 0] * pts_valid[:, 0] / zc_valid + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * pts_valid[:, 1] / zc_valid + K[1, 2]).astype(int)
        
        # Filter points outside screen
        screen_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u_screen = u[screen_mask]
        v_screen = v[screen_mask]
        c_screen = c_valid[screen_mask]
        zc_screen = zc_valid[screen_mask]
        
        # Painter's algorithm: sort by depth (furthest first)
        sort_idx = np.argsort(zc_screen)[::-1]
        u_sorted = u_screen[sort_idx]
        v_sorted = v_screen[sort_idx]
        c_sorted = c_screen[sort_idx]
        
        # Create image (RGB)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Colors are already RGB from Open3D
        c_sorted_rgb = c_sorted
        
        # Fast numpy assignment
        img[v_sorted, u_sorted] = c_sorted_rgb
        
        # Draw 2x2 blocks for a denser point cloud appearance
        img[np.clip(v_sorted+1, 0, height-1), u_sorted] = c_sorted_rgb
        img[v_sorted, np.clip(u_sorted+1, 0, width-1)] = c_sorted_rgb
        img[np.clip(v_sorted+1, 0, height-1), np.clip(u_sorted+1, 0, width-1)] = c_sorted_rgb
        
        frames_list.append(img)
        
        if (i+1) % 10 == 0:
            print(f"Rendered {i+1}/{frames} frames...")
            
    print(f"Encoding GIF... this might take a moment.")
    imageio.mimsave(output_mp4, frames_list, fps=15, loop=0)
    print(f"GIF directly saved to {output_mp4}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    render_pointcloud_orbit(args.ply, args.out)
