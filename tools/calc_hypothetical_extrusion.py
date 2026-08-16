import open3d as o3d
import numpy as np
import argparse

def get_base_area(ply_path, pitch=0.002):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    idxs = np.floor((points - min_bound) / pitch).astype(int)
    
    # Base is at the lowest Z level. Because of the extrusion, the 2D footprint 
    # of the entire object is exactly the base area.
    xy_pairs = np.unique(idxs[:, :2], axis=0)
    base_area_voxels = len(xy_pairs)
    
    # Correctly compute unique occupied voxels for total volume
    grid_shape = np.ceil((max_bound - min_bound) / pitch).astype(int) + 1
    volume_grid = np.zeros(grid_shape, dtype=np.uint8)
    volume_grid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1
    total_voxels = np.sum(volume_grid)
    
    return total_voxels, base_area_voxels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply1")
    parser.add_argument("ply2")
    parser.add_argument("--pitch", type=float, default=0.002)
    args = parser.parse_args()
    
    v1, a1 = get_base_area(args.ply1, args.pitch)
    v2, a2 = get_base_area(args.ply2, args.pitch)
    
    print(f"Shape 1 Base Area: {a1} voxels/slice")
    print(f"Shape 2 Base Area: {a2} voxels/slice")
    
    print("\n--- Hypothetical Extrusions ---")
    
    for slices_down in [0, 5, 10, 15, 20]:
        dz = slices_down * args.pitch
        new_v1 = v1 + a1 * slices_down
        new_v2 = v2 + a2 * slices_down
        
        ratio = new_v2 / new_v1
        diff_pct = (1.0 - ratio) * 100
        
        print(f"Lowering Z by {dz:.3f} units ({slices_down} slices):")
        print(f"  Shape 1 Vol: {new_v1} voxels")
        print(f"  Shape 2 Vol: {new_v2} voxels")
        print(f"  Shape 2 is now {ratio:.4f}x of Shape 1 ({diff_pct:.2f}% difference)")
