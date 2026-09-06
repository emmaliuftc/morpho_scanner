import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import imageio
import os
import argparse

def generate_50_z_slices(npy_path, ply_path, out_dir, num_slices=50):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load NPY and PLY
    vol = np.load(npy_path) # Shape: (Nx, Ny, Nz)
    nx, ny, nz = vol.shape
    
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    scale = 174.09726104178188 # mm/unit
    pts_mm = pts * scale
    
    voxel_pitch_nerf = 0.002
    voxel_pitch_mm = voxel_pitch_nerf * scale # ~0.348 mm
    
    x_min, x_max = np.min(pts_mm[:, 0]), np.max(pts_mm[:, 0])
    y_min, y_max = np.min(pts_mm[:, 1]), np.max(pts_mm[:, 1])
    z_min, z_max = np.min(pts_mm[:, 2]), np.max(pts_mm[:, 2])
    
    print("=" * 60)
    print("       NPY 3D OCCUPANCY VOLUME RANGES (Lesion 3)       ")
    print("=" * 60)
    print(f"Array Shape (Nx, Ny, Nz): {nx} × {ny} × {nz}")
    print(f"Voxel Pitch: {voxel_pitch_mm:.4f} mm ({voxel_pitch_nerf:.4f} NeRF units)")
    print(f"X Physical Range: [{x_min:.2f}, {x_max:.2f}] mm (Total Span: {x_max - x_min:.2f} mm)")
    print(f"Y Physical Range: [{y_min:.2f}, {y_max:.2f}] mm (Total Span: {y_max - y_min:.2f} mm)")
    print(f"Z Physical Range: [{z_min:.2f}, {z_max:.2f}] mm (Total Height: {z_max - z_min:.2f} mm)")
    print("=" * 60)
    
    # Coordinates in mm for grid
    x_coords_mm = np.linspace(x_min, x_max, nx)
    y_coords_mm = np.linspace(y_min, y_max, ny)
    
    # 50 target Z levels from z_min to z_max
    z_samples_mm = np.linspace(z_min, z_max, num_slices)
    # Corresponding continuous voxel index in Z
    z_indices_float = (z_samples_mm - z_min) / (z_max - z_min) * (nz - 1) if z_max > z_min else np.zeros(num_slices)
    
    generated_images = []
    
    for i, (z_mm, z_idx) in enumerate(zip(z_samples_mm, z_indices_float)):
        # Interpolate 2D slice from 3D volume
        X_grid, Y_grid = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        Z_grid = np.full_like(X_grid, z_idx, dtype=float)
        
        coords = np.array([X_grid, Y_grid, Z_grid])
        slice_2d = map_coordinates(vol.astype(float), coords, order=1, mode='nearest')
        slice_binary = (slice_2d > 0.4).astype(np.uint8)
        
        # Cross-sectional area
        area_mm2 = np.sum(slice_binary) * (voxel_pitch_mm ** 2)
        area_cm2 = area_mm2 / 100.0
        
        # Plot 2D slice with millimeter axes
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        
        # Extent: [left, right, bottom, top] in mm
        # Note: transpose slice_2d so X is horizontal (columns) and Y is vertical (rows)
        im = ax.imshow(slice_binary.T, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                       cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        
        # Styling
        pct = (i / (num_slices - 1)) * 100.0
        ax.set_title(f"Slice {i+1:02d}/{num_slices} | Z = {z_mm:.2f} mm ({pct:.1f}% Height)\n"
                     f"Cross-Section Area: {area_mm2:.1f} mm² ({area_cm2:.3f} cm²)", 
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("X Position (mm)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Y Position (mm)", fontsize=11, fontweight='bold')
        ax.set_xlim(-28, 28)
        ax.set_ylim(-24, 24)
        ax.grid(True, linestyle='--', alpha=0.5, color='gray')
        ax.axhline(0, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax.axvline(0, color='red', linestyle=':', alpha=0.5, linewidth=1)
        
        # Add scale bar & annotations
        plt.tight_layout()
        
        out_png = os.path.join(out_dir, f"slice_{i:02d}.png")
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Read back for GIF
        generated_images.append(imageio.v2.imread(out_png))
        if (i + 1) % 10 == 0 or i == num_slices - 1:
            print(f"Generated slice {i+1}/{num_slices} (Z = {z_mm:.2f} mm)...")
            
    # Save animated GIF through all 50 slices
    gif_path = os.path.join(os.path.dirname(out_dir), "z_slices_50_timelapse.gif")
    # Loop back and forth (ping-pong) for smooth viewing
    gif_frames = generated_images + generated_images[::-1]
    imageio.mimsave(gif_path, gif_frames, fps=8, loop=0)
    print(f"\nSaved 50-slice animated GIF to {gif_path}")
    
    # Save a 5x10 Montage Overview of all 50 slices
    print("Generating 50-slice montage overview...")
    fig, axs = plt.subplots(5, 10, figsize=(25, 14), facecolor='white')
    axs = axs.ravel()
    
    for i in range(num_slices):
        z_mm = z_samples_mm[i]
        z_idx = z_indices_float[i]
        X_grid, Y_grid = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        coords = np.array([X_grid, Y_grid, np.full_like(X_grid, z_idx, dtype=float)])
        slice_2d = map_coordinates(vol.astype(float), coords, order=1, mode='nearest')
        slice_binary = (slice_2d > 0.4).astype(np.uint8)
        
        axs[i].imshow(slice_binary.T, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                      cmap='Blues', vmin=0, vmax=1)
        axs[i].set_title(f"#{i+1:02d} | Z={z_mm:.1f}mm", fontsize=8, fontweight='bold', pad=3)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlim(-26, 26)
        axs[i].set_ylim(-22, 22)
        
    plt.suptitle(f"Lesion 3 Volumetric Z-Stack: 50 Cross-Sectional Slices from Z={z_min:.2f}mm (Base) to Z={z_max:.2f}mm (Apex)", 
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    montage_path = os.path.join(os.path.dirname(out_dir), "z_slices_50_montage.png")
    plt.savefig(montage_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved 50-slice montage to {montage_path}")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", default="nerf_160826_0824_8-16_lesion_3/pointcloud_raw_obb_solid_table_volume.npy")
    parser.add_argument("--ply", default="nerf_160826_0824_8-16_lesion_3/pointcloud_raw_obb_solid_table.ply")
    parser.add_argument("--out_dir", default="nerf_160826_0824_8-16_lesion_3/z_slices_50")
    parser.add_argument("--slices", type=int, default=50)
    args = parser.parse_args()
    
    generate_50_z_slices(args.npy, args.ply, args.out_dir, args.slices)
