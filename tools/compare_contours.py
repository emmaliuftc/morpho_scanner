import open3d as o3d
import numpy as np
import argparse
import matplotlib.pyplot as plt

def compare_meshes(mesh1_path, mesh2_path, out_error_ply):
    print(f"Loading generated contour from {mesh1_path}...")
    mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
    
    print(f"Loading golden contour from {mesh2_path}...")
    mesh2 = o3d.io.read_triangle_mesh(mesh2_path)
    
    print("Sampling 1,000,000 points from both surfaces for accurate distance computation...")
    pcd1 = mesh1.sample_points_uniformly(number_of_points=1000000)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=1000000)
    
    print("Computing point-to-point distances from Generated -> Golden...")
    # For each point in pcd1, finds distance to nearest in pcd2
    dists1_to_2 = pcd1.compute_point_cloud_distance(pcd2)
    dists1_to_2 = np.asarray(dists1_to_2)
    
    print("Computing point-to-point distances from Golden -> Generated...")
    dists2_to_1 = pcd2.compute_point_cloud_distance(pcd1)
    dists2_to_1 = np.asarray(dists2_to_1)
    
    # Metrics
    mean_dist1 = np.mean(dists1_to_2)
    rmse1 = np.sqrt(np.mean(dists1_to_2**2))
    max_dist1 = np.max(dists1_to_2)
    
    mean_dist2 = np.mean(dists2_to_1)
    rmse2 = np.sqrt(np.mean(dists2_to_1**2))
    max_dist2 = np.max(dists2_to_1)
    
    # Symmetrical distance (Chamfer-like)
    chamfer_l1 = (mean_dist1 + mean_dist2) / 2.0
    chamfer_l2 = (rmse1 + rmse2) / 2.0
    hausdorff = max(max_dist1, max_dist2)
    
    print("\n" + "="*50)
    print("                METRICS (mm)                ")
    print("="*50)
    print(f"Generated -> Golden (How much generated deviates):")
    print(f"  Mean Error: {mean_dist1:.4f} mm")
    print(f"  RMSE:       {rmse1:.4f} mm")
    print(f"  Max Error:  {max_dist1:.4f} mm")
    print()
    print(f"Golden -> Generated (How much golden is missing):")
    print(f"  Mean Error: {mean_dist2:.4f} mm")
    print(f"  RMSE:       {rmse2:.4f} mm")
    print(f"  Max Error:  {max_dist2:.4f} mm")
    print()
    print(f"Symmetric Metrics:")
    print(f"  Chamfer L1: {chamfer_l1:.4f} mm")
    print(f"  Chamfer L2: {chamfer_l2:.4f} mm")
    print(f"  Hausdorff:  {hausdorff:.4f} mm")
    print("="*50 + "\n")
    
    print(f"Coloring the generated point cloud by error heatmap and saving to {out_error_ply}...")
    # Color mapping: 0mm -> Blue, max_dist -> Red
    # We will cap the color mapping at 95th percentile to prevent outliers from washing out the colormap
    vmax = np.percentile(dists1_to_2, 95)
    if vmax == 0: vmax = 0.001
    normalized_dists = np.clip(dists1_to_2 / vmax, 0, 1)
    
    cmap = plt.get_cmap('jet')
    colors = cmap(normalized_dists)[:, :3] # Get RGB from RGBA
    
    pcd1.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_error_ply, pcd1)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--out_error", required=True)
    args = parser.parse_args()
    
    compare_meshes(args.generated, args.golden, args.out_error)
