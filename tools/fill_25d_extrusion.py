import open3d as o3d
import numpy as np
import argparse
import time

def fill_25d_extrusion(ply_path, out_ply_path, pitch=0.002):
    print(f"Loading raw point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    if len(points) == 0:
        print("Empty point cloud!")
        return

    print(f"Discretizing with pitch={pitch}...")
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    grid_shape = np.ceil((max_bound - min_bound) / pitch).astype(int) + 1
    print(f"Grid shape: {grid_shape}")
    
    idxs = np.floor((points - min_bound) / pitch).astype(int)
    
    print("Auto-detecting orientation and table floor plane...")
    hist, bins = np.histogram(idxs[:, 2], bins=50)
    max_bin_idx = np.argmax(hist)
    table_z_idx = int(np.round((bins[max_bin_idx] + bins[max_bin_idx+1]) / 2.0))
    
    # Check if the table is at the top or bottom of the bounding box
    min_z = np.min(idxs[:, 2])
    max_z = np.max(idxs[:, 2])
    
    is_upside_down = (table_z_idx - min_z) > (max_z - table_z_idx)
    
    # Track the surface shell for each X,Y column
    # If upside down, the surface is the MINIMUM Z
    # If right-side up, the surface is the MAXIMUM Z
    surface_z_grid = np.full((grid_shape[0], grid_shape[1]), 999999 if is_upside_down else -999999, dtype=int)
    color_grid = np.zeros((grid_shape[0], grid_shape[1], 3), dtype=np.float32)
    
    print(f"Detected Orientation: {'UPSIDE DOWN (Extruding UP)' if is_upside_down else 'RIGHT-SIDE UP (Extruding DOWN)'}")
    t0 = time.time()
    for i in range(len(idxs)):
        ix, iy, iz = idxs[i]
        if is_upside_down:
            if iz < surface_z_grid[ix, iy]:
                surface_z_grid[ix, iy] = iz
                color_grid[ix, iy] = colors[i]
        else:
            if iz > surface_z_grid[ix, iy]:
                surface_z_grid[ix, iy] = iz
                color_grid[ix, iy] = colors[i]
            
    print(f"Heightmap built in {time.time()-t0:.2f} seconds.")
    
    # We use a robust percentile to define the concrete floor just beyond the table peak
    if is_upside_down:
        global_table_z_idx = int(np.percentile(idxs[:, 2], 95))
    else:
        global_table_z_idx = int(np.percentile(idxs[:, 2], 5))
        
    print(f"Detected global flat table floor at Z-index: {global_table_z_idx}")
    
    t0 = time.time()
    solid_points = []
    solid_colors = []
    
    for ix in range(grid_shape[0]):
        for iy in range(grid_shape[1]):
            surf_iz = surface_z_grid[ix, iy]
            if (is_upside_down and surf_iz != 999999) or (not is_upside_down and surf_iz != -999999):
                
                if is_upside_down:
                    # Fill from the surface of the lobe UP to the table floor
                    start_iz = surf_iz
                    end_iz = max(global_table_z_idx, surf_iz)
                else:
                    # Fill from the table floor UP to the surface of the lobe
                    start_iz = min(global_table_z_idx, surf_iz)
                    end_iz = surf_iz
                    
                for iz in range(start_iz, end_iz + 1):
                    solid_points.append([ix, iy, iz])
                    # Keep the exact original color for the surface crust, dye the inside blue
                    if iz == surf_iz:
                        solid_colors.append(color_grid[ix, iy])
                    else:
                        solid_colors.append([0.0, 0.5, 1.0])
                        
    print(f"Extrusion complete in {time.time()-t0:.2f} seconds.")
    print(f"Generated {len(solid_points)} perfectly solid points anchored to the flat table layer!")
    
    solid_points = np.array(solid_points, dtype=np.float32) * pitch + min_bound
    solid_colors = np.array(solid_colors, dtype=np.float32)
    
    solid_pcd = o3d.geometry.PointCloud()
    solid_pcd.points = o3d.utility.Vector3dVector(solid_points)
    solid_pcd.colors = o3d.utility.Vector3dVector(solid_colors)
    
    print(f"Saving concrete 2.5D solid blob to {out_ply_path}...")
    o3d.io.write_point_cloud(out_ply_path, solid_pcd)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input raw .ply")
    parser.add_argument("--output", required=True, help="Output solid .ply")
    parser.add_argument("--pitch", type=float, default=0.002, help="Voxel resolution")
    args = parser.parse_args()
    
    fill_25d_extrusion(args.input, args.output, args.pitch)
