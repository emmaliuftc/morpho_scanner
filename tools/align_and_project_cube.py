import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os

def get_calibration_scale(calib_path):
    if not calib_path or not os.path.exists(calib_path):
        return 174.09726
    with open(calib_path, 'r') as f:
        data = json.load(f)
    if 'plate_center_mm' in data:
        return float(np.linalg.norm(data['plate_center_mm']))
    return 174.09726

def align_and_project_cube(input_ply, out_ply, out_png, scale=174.09726):
    print(f"Loading {input_ply}...")
    pcd = o3d.io.read_point_cloud(input_ply)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    # 1. Scale from NeRF space (1 unit = scale mm) to mm
    pts_mm = points * scale
    print(f"Applying true metric calibration scale: 1 NeRF unit = {scale:.3f} mm")
    
    # 2. Flip upright so Z points upward and object sits above the XY plane (Z >= 0)
    # NeRF coordinate system has table at Z=0 and object in negative Z space.
    # Rotating 180 deg around X: X' = X, Y' = -Y, Z' = -Z
    pts_upright = np.zeros_like(pts_mm)
    pts_upright[:, 0] = pts_mm[:, 0]
    pts_upright[:, 1] = -pts_mm[:, 1]
    pts_upright[:, 2] = -pts_mm[:, 2]
    
    # Trim points below table (Z < 0 in upright space)
    valid_mask = pts_upright[:, 2] >= -0.5
    pts_upright = pts_upright[valid_mask]
    if colors is not None:
        colors = colors[valid_mask]
        
    # Center XY around origin
    cx = (np.max(pts_upright[:, 0]) + np.min(pts_upright[:, 0])) / 2.0
    cy = (np.max(pts_upright[:, 1]) + np.min(pts_upright[:, 1])) / 2.0
    pts_upright[:, 0] -= cx
    pts_upright[:, 1] -= cy
    
    # 3. Find optimal rotation around Z to align faces with X and Y axes
    # We sample vertical wall points (between Z=10mm and Z=32mm) where the 4 flat sides are clearest
    wall_mask = (pts_upright[:, 2] > 10.0) & (pts_upright[:, 2] < 32.0)
    wall_pts = pts_upright[wall_mask, :2]
    if len(wall_pts) > 10000:
        sub_idx = np.random.choice(len(wall_pts), 10000, replace=False)
        wall_sample = wall_pts[sub_idx]
    else:
        wall_sample = wall_pts
        
    angles = np.linspace(-45, 45, 901) # 0.1 degree resolution
    rads = np.radians(angles)
    cos_a = np.cos(rads)
    sin_a = np.sin(rads)
    
    min_score = 1e9
    best_angle = 0.0
    
    for ang, ca, sa in zip(angles, cos_a, sin_a):
        rx = wall_sample[:, 0] * ca - wall_sample[:, 1] * sa
        ry = wall_sample[:, 0] * sa + wall_sample[:, 1] * ca
        
        span_x = np.percentile(rx, 97) - np.percentile(rx, 3)
        span_y = np.percentile(ry, 97) - np.percentile(ry, 3)
        area = span_x * span_y
        
        if area < min_score:
            min_score = area
            best_angle = ang
            
    print(f"Optimal in-plane alignment angle: {best_angle:+.2f}° around Z-axis")
    
    # Apply optimal rotation
    best_rad = np.radians(best_angle)
    ca, sa = np.cos(best_rad), np.sin(best_rad)
    rot_x = pts_upright[:, 0] * ca - pts_upright[:, 1] * sa
    rot_y = pts_upright[:, 0] * sa + pts_upright[:, 1] * ca
    rot_z = pts_upright[:, 2] - np.min(pts_upright[:, 2]) # Place table base exactly at Z=0.0 mm
    
    pts_aligned = np.column_stack([rot_x, rot_y, rot_z])
    
    # Center X and Y around (0, 0)
    mid_x = (np.percentile(pts_aligned[:, 0], 99) + np.percentile(pts_aligned[:, 0], 1)) / 2.0
    mid_y = (np.percentile(pts_aligned[:, 1], 99) + np.percentile(pts_aligned[:, 1], 1)) / 2.0
    pts_aligned[:, 0] -= mid_x
    pts_aligned[:, 1] -= mid_y
    
    # Save aligned point cloud
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(pts_aligned)
    if colors is not None:
        aligned_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_ply, aligned_pcd)
    print(f"Saved aligned point cloud to {out_ply}")
    
    # 4. Generate 3-Axis Orthographic Projections (Physical mm)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    
    if len(pts_aligned) > 60000:
        idx = np.random.choice(len(pts_aligned), 60000, replace=False)
        plot_pts = pts_aligned[idx]
        plot_colors = colors[idx] if colors is not None else None
    else:
        plot_pts = pts_aligned
        plot_colors = colors
        
    c_arg = plot_colors if plot_colors is not None else 'tab:blue'
    
    # Top-Down View (XY Plane)
    axs[0].scatter(plot_pts[:, 0], plot_pts[:, 1], s=0.3, alpha=0.6, c=c_arg)
    axs[0].set_title("Top-Down View ($XY$ Plane)\nAligned with X & Y Axes", fontsize=13, fontweight='bold', pad=10)
    axs[0].set_xlabel("X (mm)", fontsize=11, fontweight='bold')
    axs[0].set_ylabel("Y (mm)", fontsize=11, fontweight='bold')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].set_aspect('equal')
    axs[0].set_xlim(-35, 35)
    axs[0].set_ylim(-35, 35)
    
    # Side View (XZ Plane)
    axs[1].scatter(plot_pts[:, 0], plot_pts[:, 2], s=0.3, alpha=0.6, c=c_arg)
    axs[1].set_title("Side View ($XZ$ Plane)\nHeight above $Z=0$ Table", fontsize=13, fontweight='bold', pad=10)
    axs[1].set_xlabel("X (mm)", fontsize=11, fontweight='bold')
    axs[1].set_ylabel("Z Height (mm)", fontsize=11, fontweight='bold')
    axs[1].axhline(0, color='red', linestyle='--', linewidth=1.2, label='Table ($Z=0$)')
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].set_aspect('equal')
    axs[1].set_xlim(-35, 35)
    axs[1].set_ylim(-5, 50)
    axs[1].legend(loc='upper right')
    
    # Front View (YZ Plane)
    axs[2].scatter(plot_pts[:, 1], plot_pts[:, 2], s=0.3, alpha=0.6, c=c_arg)
    axs[2].set_title("Front View ($YZ$ Plane)\nHeight above $Z=0$ Table", fontsize=13, fontweight='bold', pad=10)
    axs[2].set_xlabel("Y (mm)", fontsize=11, fontweight='bold')
    axs[2].set_ylabel("Z Height (mm)", fontsize=11, fontweight='bold')
    axs[2].axhline(0, color='red', linestyle='--', linewidth=1.2, label='Table ($Z=0$)')
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].set_aspect('equal')
    axs[2].set_xlim(-35, 35)
    axs[2].set_ylim(-5, 50)
    axs[2].legend(loc='upper right')
    
    # Calculate dimensional bounds
    x_dim = np.percentile(pts_aligned[:, 0], 98) - np.percentile(pts_aligned[:, 0], 2)
    y_dim = np.percentile(pts_aligned[:, 1], 98) - np.percentile(pts_aligned[:, 1], 2)
    z_dim = np.percentile(pts_aligned[:, 2], 99) - np.min(pts_aligned[:, 2])
    
    fig.suptitle(f"True Physical Scale (mm) Orthographic Projections: 40mm CAD Cube (20k Steps)\n"
                 f"True Aligned Dimensions: {x_dim:.2f} mm (X) × {y_dim:.2f} mm (Y) × {z_dim:.2f} mm (Z)", 
                 fontsize=14, fontweight='bold', y=1.03)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved physical mm orthographic projection plot to {out_png}")
    print(f"True Aligned Dimensions: X={x_dim:.2f} mm, Y={y_dim:.2f} mm, Z={z_dim:.2f} mm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_ply", required=True)
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--calib", default="captures_8-13_calibration_results/calibration.json")
    parser.add_argument("--scale", type=float, default=None)
    args = parser.parse_args()
    
    scale = args.scale if args.scale is not None else get_calibration_scale(args.calib)
    align_and_project_cube(args.input, args.out_ply, args.out_png, scale=scale)
