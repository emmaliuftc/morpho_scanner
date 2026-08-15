import open3d as o3d
import numpy as np
import argparse
import os

def process(ply_path):
    print(f"Loading {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        print("Point cloud is empty!")
        return
    
    # 1. Reproject: Invert Z so lobes point UP, and shift so table is at Z=0
    print("Reprojecting coordinate system (lobes UP, table at Z=0)...")
    
    # Auto-detect orientation by analyzing volumetric mass distribution
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    z_range = z_max - z_min
    
    top_threshold = z_max - (z_range * 0.2)
    bottom_threshold = z_min + (z_range * 0.2)
    
    top_mass = np.sum(points[:, 2] > top_threshold)
    bottom_mass = np.sum(points[:, 2] < bottom_threshold)
    
    is_upside_down = top_mass > bottom_mass
    
    if is_upside_down:
        print("Point cloud is UPSIDE DOWN. Inverting Z-axis to point lobes UP...")
        # The perfectly flat table plane we generated is currently at the maximum Z coordinate
        table_z_old = np.max(points[:, 2]) 
        # Invert the Z axis so the lobes point UP instead of hanging down
        points[:, 2] = -points[:, 2]
        # Shift the entire point cloud up so the table lies exactly on the Z=0 plane
        points[:, 2] += table_z_old
    else:
        print("Point cloud is RIGHT-SIDE UP. Shifting table perfectly to Z=0...")
        # The perfectly flat table plane we generated is currently at the minimum Z coordinate
        table_z_old = np.min(points[:, 2])
        # Shift the entire point cloud down so the table lies exactly on the Z=0 plane
        points[:, 2] -= table_z_old
    
    # Center X and Y around the origin (0,0) for perfectly clean viewer framing
    center_x = (np.max(points[:, 0]) + np.min(points[:, 0])) / 2.0
    center_y = (np.max(points[:, 1]) + np.min(points[:, 1])) / 2.0
    points[:, 0] -= center_x
    points[:, 1] -= center_y
    
    # Update point cloud and save
    pcd.points = o3d.utility.Vector3dVector(points)
    out_ply = ply_path.replace(".ply", "_reprojected.ply")
    print(f"Saving reprojected point cloud to {out_ply}")
    o3d.io.write_point_cloud(out_ply, pcd)
    
    # 2. Convert to 3D Numpy Array (Napari/Bio-format volumetric blob)
    # The generation script used a discrete pitch of 0.002
    pitch = 0.002
    print(f"Voxelizing into 3D Numpy occupancy grid with pitch={pitch}...")
    
    # Shift coordinates to strictly positive indices (0 to N) for the numpy array
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    grid_shape = np.ceil((max_bound - min_bound) / pitch).astype(int) + 1
    print(f"Generated 3D Numpy Array Shape: {grid_shape}")
    
    # We use uint8 (1 for object, 0 for background) which Napari and Scikit-Image natively love
    volume = np.zeros(grid_shape, dtype=np.uint8)
    
    idxs = np.floor((points - min_bound) / pitch).astype(int)
    
    # Inject all the solid points into the 3D numpy array
    volume[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1
    
    out_npy = ply_path.replace(".ply", "_volume.npy")
    print(f"Saving dense 3D binary numpy array to {out_npy}")
    np.save(out_npy, volume)
    
    print("Done! Reprojected PLY and Napari NPY generated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="Input solid .ply file to reproject and export")
    args = parser.parse_args()
    
    process(args.ply)
