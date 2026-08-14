import argparse
import open3d as o3d
import os

def main():
    parser = argparse.ArgumentParser(description="Filter point cloud outliers using Statistical Outlier Removal (SOR)")
    parser.add_argument("--input", type=str, required=True, help="Input .ply file path")
    parser.add_argument("--output", type=str, required=True, help="Output .ply file path")
    parser.add_argument("--neighbors", type=int, default=50, help="Number of neighbors for SOR (default 50)")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="Standard deviation ratio for SOR (default 2.0)")
    args = parser.parse_args()

    print(f"Loading point cloud from {args.input}...")
    if not os.path.exists(args.input):
        print(f"Error: {args.input} does not exist.")
        return
        
    pcd = o3d.io.read_point_cloud(args.input)
    print(f"Original points: {len(pcd.points)}")

    print(f"Applying Z-axis floor clip (removing points below the table)...")
    import numpy as np
    pts = np.asarray(pcd.points)
    # The normal (Z-axis) points DOWN into the table. 
    # Therefore, the object is at Z < 0, and the noise under the table is at Z > 0.
    # We keep everything <= -0.015 (slicing off the bottom ~2.6mm of the model)
    # This guarantees the QR code is severed from the main clay blob.
    mask = pts[:, 2] <= -0.015
    pcd = pcd.select_by_index(np.where(mask)[0])
    print(f"Points after Z-clip: {len(pcd.points)}")

    print(f"Applying Statistical Outlier Removal (neighbors={args.neighbors}, std_ratio={args.std_ratio})...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=args.neighbors, std_ratio=args.std_ratio)
    clean_pcd = pcd.select_by_index(ind)

    print(f"Points after SOR: {len(clean_pcd.points)} (Removed {len(pcd.points) - len(clean_pcd.points)} points)")

    print(f"Saving to {args.output}...")
    o3d.io.write_point_cloud(args.output, clean_pcd)
    print("Done!")

if __name__ == "__main__":
    main()
