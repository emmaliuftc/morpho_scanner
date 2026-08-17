import open3d as o3d
import numpy as np
import argparse
import matplotlib.pyplot as plt

def load_geometry_as_pcd(path, scale=150.0, num_points=1000000):
    # Try reading as mesh first
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) > 0:
        if path.endswith('.stl'):
            # STL is already in mm
            pass
        else:
            # NeRF mesh scaled by 150
            mesh.scale(scale, center=(0, 0, 0))
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        extents = np.asarray(pcd.points).max(axis=0) - np.asarray(pcd.points).min(axis=0)
        return pcd, extents
    
    # Otherwise read as point cloud
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    # Check if points are already in mm (max magnitude > 5.0)
    if np.max(np.abs(points)) < 5.0:
        points = points * scale
    pcd.points = o3d.utility.Vector3dVector(points)
    extents = points.max(axis=0) - points.min(axis=0)
    
    # If point cloud has too few/many points, resample
    if len(points) > num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
        pcd = pcd.select_by_index(idx)
    return pcd, extents

def evaluate_geometry_against_stl(eval_path, stl_path, out_error_ply):
    print(f"Loading generated geometry from {eval_path}...")
    pcd_eval, eval_extents = load_geometry_as_pcd(eval_path, scale=150.0, num_points=1000000)
    
    print(f"Loading ground truth STL from {stl_path}...")
    pcd_stl, stl_extents = load_geometry_as_pcd(stl_path, scale=1.0, num_points=1000000)
    
    print(f"Ground Truth STL Extents (X, Y, Z): {stl_extents[0]:.2f} mm x {stl_extents[1]:.2f} mm x {stl_extents[2]:.2f} mm")
    print(f"Generated Geometry Extents (X, Y, Z): {eval_extents[0]:.2f} mm x {eval_extents[1]:.2f} mm x {eval_extents[2]:.2f} mm")
    
    print("Downsampling to 50,000 points for ICP alignment...")
    idx_eval = np.random.choice(len(pcd_eval.points), 50000, replace=False)
    pcd_eval_icp = pcd_eval.select_by_index(idx_eval)
    
    idx_stl = np.random.choice(len(pcd_stl.points), 50000, replace=False)
    pcd_stl_icp = pcd_stl.select_by_index(idx_stl)
    
    print("Performing ICP Alignment (Point-to-Point)...")
    reg = o3d.pipelines.registration.registration_icp(
        pcd_eval_icp, pcd_stl_icp, 10.0, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    
    print("ICP Transformation Matrix:")
    print(reg.transformation)
    pcd_eval.transform(reg.transformation)
    
    print("Computing point-to-point distances from Generated -> Golden...")
    dists1_to_2 = pcd_eval.compute_point_cloud_distance(pcd_stl)
    dists1_to_2 = np.asarray(dists1_to_2)
    
    print("Computing point-to-point distances from Golden -> Generated...")
    dists2_to_1 = pcd_stl.compute_point_cloud_distance(pcd_eval)
    dists2_to_1 = np.asarray(dists2_to_1)
    
    # Metrics
    mean_dist1 = np.mean(dists1_to_2)
    rmse1 = np.sqrt(np.mean(dists1_to_2**2))
    max_dist1 = np.max(dists1_to_2)
    
    mean_dist2 = np.mean(dists2_to_1)
    rmse2 = np.sqrt(np.mean(dists2_to_1**2))
    max_dist2 = np.max(dists2_to_1)
    
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
    
    print(f"Coloring the aligned generated point cloud by error heatmap and saving to {out_error_ply}...")
    vmax = np.percentile(dists1_to_2, 95)
    if vmax == 0: vmax = 0.001
    normalized_dists = np.clip(dists1_to_2 / vmax, 0, 1)
    
    cmap = plt.get_cmap('jet')
    colors = cmap(normalized_dists)[:, :3]
    pcd_eval.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_error_ply, pcd_eval)
    print(f"Saved error heatmap point cloud to {out_error_ply}")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True, help="Path to generated mesh or point cloud PLY")
    parser.add_argument("--golden", required=True, help="Path to golden STL")
    parser.add_argument("--out_error", required=True, help="Path to output error pointcloud PLY")
    args = parser.parse_args()
    
    evaluate_geometry_against_stl(args.generated, args.golden, args.out_error)
