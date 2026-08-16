import open3d as o3d
import numpy as np
import argparse
import time

def fill_25d_extrusion(ply_path, out_ply_path, pitch=0.002, z_comp_mm=0.0):
    print(f"Loading raw point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    has_normals = pcd.has_normals()
    if has_normals:
        normals = np.asarray(pcd.normals)
    else:
        normals = np.zeros_like(points)
    
    if len(points) == 0:
        return

    # Convert mm compensation to NeRF units (1 unit = 150mm)
    z_comp_units = z_comp_mm / 150.0

    # 1. Mathematical Ground Truth: 
    # The golden calibration defines the table perfectly at Z=0.
    # The cameras are at negative Z, looking at Z=0.
    # Therefore, the object sits in the negative Z space (Z < 0).
    # Anything with Z > z_comp_units is noise underneath the new compensated table.
    print(f"Trimming noise below the mathematical table plane (Z > {z_comp_units:.4f})...")
    valid_mask = points[:, 2] <= z_comp_units
    points = points[valid_mask]
    colors = colors[valid_mask]
    if has_normals:
        normals = normals[valid_mask]
    
    if len(points) == 0:
        print("Empty point cloud after trimming!")
        return

    print(f"Discretizing with pitch={pitch}...")
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    grid_shape = np.ceil((max_bound - min_bound) / pitch).astype(int) + 1
    idxs = np.floor((points - min_bound) / pitch).astype(int)
    
    # 2. Track the surface shell for each X,Y column
    # Since the object is in negative Z, the crust closest to the camera is the MINIMUM Z.
    surface_z_grid = np.full((grid_shape[0], grid_shape[1]), 999999, dtype=int)
    for ix, iy, iz in idxs:
        if iz < surface_z_grid[ix, iy]:
            surface_z_grid[ix, iy] = iz

    # We fill up to the compensated table.
    table_iz = int(np.floor((z_comp_units - min_bound[2]) / pitch))

    solid_points = []
    solid_colors = []
    solid_normals = []
    
    # Keep ALL valid original points and their colors/normals
    for i in range(len(points)):
        solid_points.append(points[i])
        solid_colors.append(colors[i])
        solid_normals.append(normals[i])
        
    filled_count = 0
    # Fill the interior from the shell up to the table
    for ix in range(grid_shape[0]):
        for iy in range(grid_shape[1]):
            surf_iz = surface_z_grid[ix, iy]
            if surf_iz != 999999:
                # Fill from the shell crust UP to the mathematical table
                for iz in range(surf_iz + 1, table_iz):
                    pt_x = ix * pitch + min_bound[0]
                    pt_y = iy * pitch + min_bound[1]
                    pt_z = iz * pitch + min_bound[2]
                    solid_points.append([pt_x, pt_y, pt_z])
                    solid_colors.append([0.0, 0.5, 1.0])
                    solid_normals.append([0.0, 0.0, -1.0]) # Pointing towards the camera origin
                    filled_count += 1
                        
    print(f"Generated {filled_count} inner blue points, plus {len(points)} original points kept.")
    
    solid_points = np.array(solid_points, dtype=np.float32)
    solid_colors = np.array(solid_colors, dtype=np.float32)
    solid_normals = np.array(solid_normals, dtype=np.float32)
    
    # 3. Orient the point cloud to be RIGHT-SIDE UP for biological viewers (Meshlab, Napari)
    # The raw NeRF coordinate system has Z pointing DOWN.
    # To fix this WITHOUT mirroring (which breaks chirality/handedness), we ROTATE 180 degrees around X.
    # X' = X, Y' = -Y, Z' = -Z
    print("Rotating 180 degrees around X to make object upright and preserve biological chirality...")
    solid_points[:, 1] = -solid_points[:, 1]
    solid_points[:, 2] = -solid_points[:, 2]
    
    if has_normals:
        solid_normals[:, 1] = -solid_normals[:, 1]
        solid_normals[:, 2] = -solid_normals[:, 2]
    
    # Center X and Y around the origin (0,0) for perfectly clean viewer framing
    center_x = (np.max(solid_points[:, 0]) + np.min(solid_points[:, 0])) / 2.0
    center_y = (np.max(solid_points[:, 1]) + np.min(solid_points[:, 1])) / 2.0
    solid_points[:, 0] -= center_x
    solid_points[:, 1] -= center_y
    
    solid_pcd = o3d.geometry.PointCloud()
    solid_pcd.points = o3d.utility.Vector3dVector(solid_points)
    solid_pcd.colors = o3d.utility.Vector3dVector(solid_colors)
    if has_normals:
        solid_pcd.normals = o3d.utility.Vector3dVector(solid_normals)
    
    o3d.io.write_point_cloud(out_ply_path, solid_pcd)
    print(f"Saved correctly oriented, solid point cloud to {out_ply_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pitch", type=float, default=0.002)
    parser.add_argument("--z_comp", type=float, default=0.0, help="Z-compensation in mm (e.g. 4.5)")
    args = parser.parse_args()
    fill_25d_extrusion(args.input, args.output, args.pitch, args.z_comp)
