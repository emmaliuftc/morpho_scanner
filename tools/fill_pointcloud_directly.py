import open3d as o3d
import numpy as np
import argparse
from scipy import ndimage
import time

def fill_pointcloud(ply_path, out_ply_path, pitch=0.002, dilation_iters=2):
    print(f"Loading raw point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    if len(points) == 0:
        print("Empty point cloud!")
        return

    # 1. Determine bounding box and create 3D voxel grid
    print(f"Voxelizing with pitch={pitch}...")
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    grid_shape = np.ceil((max_bound - min_bound) / pitch).astype(int) + 1
    print(f"Grid shape: {grid_shape} (Total voxels: {np.prod(grid_shape)})")
    
    # Map points to voxel indices
    idxs = np.floor((points - min_bound) / pitch).astype(int)
    
    # We will also keep track of colors for the surface voxels
    # If multiple points fall in the same voxel, they overwrite (or we could average)
    grid = np.zeros(grid_shape, dtype=bool)
    grid_colors = np.zeros(tuple(grid_shape) + (3,), dtype=np.float32)
    
    grid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = True
    grid_colors[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = colors
    
    # 2. Morphological Closing to seal small holes in the shell
    print(f"Applying binary dilation ({dilation_iters} iterations) to seal the point cloud shell...")
    t0 = time.time()
    sealed_grid = ndimage.binary_dilation(grid, iterations=dilation_iters)
    print(f"Dilation took {time.time()-t0:.2f} seconds.")
    
    # 3. Fill interior holes
    print("Applying 3D binary fill holes to solidify the interior...")
    t0 = time.time()
    filled_grid = ndimage.binary_fill_holes(sealed_grid)
    print(f"Fill holes took {time.time()-t0:.2f} seconds.")
    
    # 4. Erode back to original boundary
    print(f"Applying binary erosion ({dilation_iters} iterations) to restore original bounds...")
    t0 = time.time()
    final_solid_grid = ndimage.binary_erosion(filled_grid, iterations=dilation_iters)
    print(f"Erosion took {time.time()-t0:.2f} seconds.")
    
    # Optional: Combine with original grid to ensure no original points were lost during erosion
    final_solid_grid = final_solid_grid | grid
    
    # 5. Extract solid voxels back into a point cloud
    print("Extracting solid points...")
    solid_idxs = np.argwhere(final_solid_grid)
    solid_points = solid_idxs * pitch + min_bound
    
    print(f"Generated {len(solid_points)} solid points!")
    
    # 6. Color the interior
    # We color the interior black (or we can implement a vertical ray drop)
    # The user asked: "filling all the pixels below the shell same same color as the one above or with black"
    print("Coloring solid interior black, while keeping surface colored...")
    final_colors = np.zeros((len(solid_points), 3), dtype=np.float32)
    
    # Identify which of the final points were in the original surface grid
    surface_mask = grid[solid_idxs[:, 0], solid_idxs[:, 1], solid_idxs[:, 2]]
    
    # Copy original colors for surface points, rest remain black [0,0,0]
    final_colors[surface_mask] = grid_colors[solid_idxs[surface_mask, 0], solid_idxs[surface_mask, 1], solid_idxs[surface_mask, 2]]
    
    solid_pcd = o3d.geometry.PointCloud()
    solid_pcd.points = o3d.utility.Vector3dVector(solid_points)
    solid_pcd.colors = o3d.utility.Vector3dVector(final_colors)
    
    print(f"Saving direct solid blob to {out_ply_path}...")
    o3d.io.write_point_cloud(out_ply_path, solid_pcd)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input raw .ply")
    parser.add_argument("--output", required=True, help="Output solid .ply")
    parser.add_argument("--pitch", type=float, default=0.002, help="Voxel resolution")
    parser.add_argument("--iters", type=int, default=2, help="Dilation iterations to seal holes")
    args = parser.parse_args()
    
    fill_pointcloud(args.input, args.output, args.pitch, args.iters)
