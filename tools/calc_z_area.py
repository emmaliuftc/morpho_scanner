import open3d as o3d
import numpy as np
import argparse

def calc_z_area(ply_path, pitch=0.002):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        print("Empty point cloud!")
        return
    
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    idxs = np.floor((points - min_bound) / pitch).astype(int)
    
    grid_shape = np.ceil((max_bound - min_bound) / pitch).astype(int) + 1
    volume_grid = np.zeros(grid_shape, dtype=np.uint8)
    volume_grid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1
    total_voxels = np.sum(volume_grid)
    
    # Sum area per Z index
    areas = np.sum(volume_grid, axis=(0, 1))
    
    # Bin the Z levels (e.g., bins of 0.01 Z-units, which is 5 slices at pitch 0.002)
    bin_size = 5 
    num_bins = int(np.ceil(grid_shape[2] / bin_size))
    
    print(f"Total Volume: {total_voxels} voxels ({total_voxels * (pitch**3):.6f} units³)")
    print(f"Z-Level Cross-Section Profile (Bin = {bin_size * pitch:.4f} Z-units):")
    
    cumulative_loss = 0
    for b in range(num_bins):
        start_idx = b * bin_size
        end_idx = min((b + 1) * bin_size, grid_shape[2])
        
        bin_voxels = np.sum(areas[start_idx:end_idx])
        
        cumulative_loss += bin_voxels
        loss_pct = (cumulative_loss / total_voxels) * 100
        
        z_start = min_bound[2] + start_idx * pitch
        z_end = min_bound[2] + end_idx * pitch
        
        # Only print the first 15 bins and the last 2 bins to avoid spamming, or just print all if small
        if b < 15 or b >= num_bins - 2:
            print(f"  Z: [{z_start:6.3f} to {z_end:6.3f}) | Volume in bin: {bin_voxels:6d} voxels | Cumulative Loss if cut from bottom: {loss_pct:5.2f}%")
        elif b == 15:
            print("  ... (middle Z layers omitted) ...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    parser.add_argument("--pitch", type=float, default=0.002)
    args = parser.parse_args()
    
    print(f"--- Analyzing Z-Level Area for {args.ply} ---")
    calc_z_area(args.ply, args.pitch)
