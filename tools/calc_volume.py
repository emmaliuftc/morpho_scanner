import open3d as o3d
import numpy as np
import argparse

def calc_volume(ply_path, pitch=0.002):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return 0.0, 0
    
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    grid_shape = np.ceil((max_bound - min_bound) / pitch).astype(int) + 1
    
    volume_grid = np.zeros(grid_shape, dtype=np.uint8)
    idxs = np.floor((points - min_bound) / pitch).astype(int)
    
    volume_grid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1
    
    voxel_count = np.sum(volume_grid)
    volume_units3 = voxel_count * (pitch ** 3)
    
    return volume_units3, voxel_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply1")
    parser.add_argument("ply2")
    parser.add_argument("--pitch", type=float, default=0.002)
    args = parser.parse_args()
    
    print(f"Calculating volume with pitch={args.pitch}...")
    
    vol1, count1 = calc_volume(args.ply1, args.pitch)
    vol2, count2 = calc_volume(args.ply2, args.pitch)
    
    print(f"\nShape 1: {args.ply1}")
    print(f"  Voxels: {count1}")
    print(f"  Volume: {vol1:.8f} units³")
    
    print(f"\nShape 2: {args.ply2}")
    print(f"  Voxels: {count2}")
    print(f"  Volume: {vol2:.8f} units³")
    
    diff = vol2 - vol1
    ratio = vol2 / vol1 if vol1 > 0 else 0
    print(f"\nAnalysis:")
    print(f"  Difference (Shape 2 - Shape 1): {diff:.8f} units³")
    print(f"  Growth Ratio (Shape 2 / Shape 1): {ratio:.4f}x")
