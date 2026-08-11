import os
import cv2
import numpy as np
import open3d as o3d

def fix_ply_and_generate_preview():
    PLY_PATH = "captures_0810_cube_nerf/export/cube_nerf_1000_pcd.ply"
    OUT_PLY = "captures_0810_cube_nerf/export/cube_nerf_1000_pcd_colored.ply"
    PREVIEW_IMG = "captures_0810_cube_nerf/nerf_point_cloud_preview.png"

    # Read PLY
    pcd = o3d.io.read_point_cloud(PLY_PATH)
    pts = np.asarray(pcd.points)

    print(f"Original PLY Point Count: {len(pts)}")
    print(f"Original Bounds (m): Min {pts.min(axis=0)}, Max {pts.max(axis=0)}")

    # 1. Scale to physical millimeters ([-20, 20] mm)
    if pts.max() < 1.0:
        pts_mm = pts * 1000.0
    else:
        pts_mm = pts

    # 2. Add bright green RGB colors (R=0.0, G=0.9, B=0.3)
    colors = np.zeros_like(pts_mm)
    colors[:, 0] = 0.0  # Red
    colors[:, 1] = 0.9  # Green
    colors[:, 2] = 0.3  # Blue

    # Update Open3D Point Cloud
    pcd_fixed = o3d.geometry.PointCloud()
    pcd_fixed.points = o3d.utility.Vector3dVector(pts_mm)
    pcd_fixed.colors = o3d.utility.Vector3dVector(colors)

    # Save fixed PLY
    o3d.io.write_point_cloud(OUT_PLY, pcd_fixed)
    o3d.io.write_point_cloud(PLY_PATH, pcd_fixed) # Overwrite original with colored mm version
    print(f"✅ Fixed PLY saved to {OUT_PLY} and overwritten at {PLY_PATH}")
    print(f"Fixed Bounds (mm): Min {pts_mm.min(axis=0)}, Max {pts_mm.max(axis=0)}")

    # 3. Render 3D Point Cloud Screenshot
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=600)
    vis.add_geometry(pcd_fixed)
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.12]) # Sleek dark background
    opt.point_size = 5.0 # Large visible points
    
    vis.poll_events()
    vis.update_renderer()
    
    img_arr = vis.capture_screen_float_buffer(True)
    vis.destroy_window()
    
    img_bgr = (np.asarray(img_arr) * 255).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(PREVIEW_IMG, img_bgr)
    print(f"✅ Rendered 3D point cloud preview image to {PREVIEW_IMG}")

if __name__ == '__main__':
    fix_ply_and_generate_preview()
