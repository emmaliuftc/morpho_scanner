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
        
    # The PLY is ALREADY right-side up and centered by fill_25d_extrusion.py!
    # Convert directly to 3D Numpy Array (Napari/Bio-format volumetric blob)
    pitch = 0.002
    print(f"Voxelizing into 3D Numpy occupancy grid with pitch={pitch}...")
    
    # Shift coordinates to strictly positive indices (0 to N) for the numpy array
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    grid_shape = np.ceil((max_bound - min_bound) / pitch).astype(int) + 1
    print(f"Generated 3D Numpy Array Shape: {grid_shape}")
    
    volume = np.zeros(grid_shape, dtype=np.uint8)
    
    idxs = np.floor((points - min_bound) / pitch).astype(int)
    volume[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1
    
    out_npy = ply_path.replace(".ply", "_volume.npy")
    print(f"Saving dense 3D binary numpy array to {out_npy}")
    np.save(out_npy, volume)
    
    print("Done! Napari NPY generated successfully.")
    return out_npy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--extract_features", action="store_true", help="Extract 17-feature Haralick and morphometric vector")
    args = parser.parse_args()
    
    out_npy = process(args.ply)
    if args.extract_features and out_npy:
        try:
            from tools.extract_haralick_features import run_feature_extraction
        except ImportError:
            from extract_haralick_features import run_feature_extraction
        run_feature_extraction(npy_path=out_npy)
