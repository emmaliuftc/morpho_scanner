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
        return

    # Trim points where Z < 0 (which the user says is "less than negative z" ?
    # Wait, if table is at +0.09 and tip is at -0.13, trimming Z < 0 cuts the object in half.
    # The user says "We shall only trim off things less than negative z".
    # This might mean "z < table_z". But wait, the previous code in fill_25d_extrusion.py that the user called "totally wrong"
    # was the one I just wrote, where I added:
    # valid_mask = points[:, 2] >= 0
    # points = points[valid_mask]
    # So my trimming logic was what they called "totally wrong"!
    # "This is also totally wrong. I think you were looking at the wrong side of the z, it is from some positive z as the table downward to 0, or lower, which is actually the top of the shell"
    # They are pointing out that my `points[:, 2] >= 0` trimmed exactly the wrong part! The object IS in the negative Z!
    # So I shouldn't trim Z < 0! I should just NOT trim, or trim Z > table_z.
    
    print(f"Discretizing with pitch={pitch}...")
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    grid_shape = np.ceil((max_bound - min_bound) / pitch).astype(int) + 1
    idxs = np.floor((points - min_bound) / pitch).astype(int)
    
    # Calculate mass to find orientation (table location)
    z_min, z_max = np.min(idxs[:, 2]), np.max(idxs[:, 2])
    z_range = z_max - z_min
    top_mass = np.sum(idxs[:, 2] > z_max - (z_range * 0.2))
    bottom_mass = np.sum(idxs[:, 2] < z_min + (z_range * 0.2))
    
    is_upside_down = top_mass > bottom_mass
    
    if is_upside_down:
        # Table is at max Z. The lobe points DOWNWARDS.
        # So for each (X,Y), the "surface crust" of the lobe is the MINIMUM Z.
        surface_z_grid = np.full((grid_shape[0], grid_shape[1]), 999999, dtype=int)
        for ix, iy, iz in idxs:
            if iz < surface_z_grid[ix, iy]:
                surface_z_grid[ix, iy] = iz
        
        # Table floor is roughly at the 95th percentile
        global_table_z_idx = int(np.percentile(idxs[:, 2], 95))
    else:
        # Table is at min Z. The lobe points UPWARDS.
        # So for each (X,Y), the "surface crust" of the lobe is the MAXIMUM Z.
        surface_z_grid = np.full((grid_shape[0], grid_shape[1]), -999999, dtype=int)
        for ix, iy, iz in idxs:
            if iz > surface_z_grid[ix, iy]:
                surface_z_grid[ix, iy] = iz
                
        # Table floor is roughly at the 5th percentile
        global_table_z_idx = int(np.percentile(idxs[:, 2], 5))

    solid_points = []
    solid_colors = []
    
    # 1. KEEP ALL ORIGINAL POINTS (User requested keeping outside color)
    # Actually, we can optionally trim noise above the table here.
    for i in range(len(points)):
        # To strictly follow "keep all outside color", we keep everything.
        # The user was complaining about MY trimming ("only trim off things less than negative z. I think you were looking at the wrong side of the z")
        # I'll just keep all points.
        solid_points.append(points[i])
        solid_colors.append(colors[i])
        
    filled_count = 0
    # 2. FILL INTERIOR
    for ix in range(grid_shape[0]):
        for iy in range(grid_shape[1]):
            surf_iz = surface_z_grid[ix, iy]
            if is_upside_down:
                if surf_iz != 999999:
                    # Fill from the lobe crust UP to the table
                    # We start at surf_iz + 1 to avoid overlapping the crust
                    for iz in range(surf_iz + 1, global_table_z_idx):
                        pt_x = ix * pitch + min_bound[0]
                        pt_y = iy * pitch + min_bound[1]
                        pt_z = iz * pitch + min_bound[2]
                        solid_points.append([pt_x, pt_y, pt_z])
                        solid_colors.append([0.0, 0.5, 1.0])
                        filled_count += 1
            else:
                if surf_iz != -999999:
                    # Fill from the table UP to the lobe crust
                    for iz in range(global_table_z_idx + 1, surf_iz):
                        pt_x = ix * pitch + min_bound[0]
                        pt_y = iy * pitch + min_bound[1]
                        pt_z = iz * pitch + min_bound[2]
                        solid_points.append([pt_x, pt_y, pt_z])
                        solid_colors.append([0.0, 0.5, 1.0])
                        filled_count += 1
                        
    print(f"Generated {filled_count} inner blue points, plus {len(points)} original points kept.")
    
    solid_points = np.array(solid_points, dtype=np.float32)
    solid_colors = np.array(solid_colors, dtype=np.float32)
    
    solid_pcd = o3d.geometry.PointCloud()
    solid_pcd.points = o3d.utility.Vector3dVector(solid_points)
    solid_pcd.colors = o3d.utility.Vector3dVector(solid_colors)
    
    o3d.io.write_point_cloud(out_ply_path, solid_pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pitch", type=float, default=0.002)
    args = parser.parse_args()
    fill_25d_extrusion(args.input, args.output, args.pitch)
