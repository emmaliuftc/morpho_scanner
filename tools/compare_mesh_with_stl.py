import open3d as o3d
import numpy as np
import argparse
import matplotlib.pyplot as plt

def evaluate_mesh_against_stl(eval_mesh_path, stl_path, out_error_ply):
    print(f"Loading generated mesh from {eval_mesh_path}...")
    eval_mesh = o3d.io.read_triangle_mesh(eval_mesh_path)
    # Scale from NeRF units (if 1 unit = 150mm) to mm
    eval_mesh.scale(150.0, center=(0, 0, 0))
    eval_verts = np.asarray(eval_mesh.vertices)
    eval_extents = eval_verts.max(axis=0) - eval_verts.min(axis=0)
    
    print(f"Loading ground truth STL from {stl_path}...")
    stl_mesh = o3d.io.read_triangle_mesh(stl_path)
    stl_verts = np.asarray(stl_mesh.vertices)
    stl_extents = stl_verts.max(axis=0) - stl_verts.min(axis=0)
    
    print(f"Ground Truth STL Extents (X, Y, Z): {stl_extents[0]:.2f} mm x {stl_extents[1]:.2f} mm x {stl_extents[2]:.2f} mm")
    print(f"Generated Mesh Extents (X, Y, Z): {eval_extents[0]:.2f} mm x {eval_extents[1]:.2f} mm x {eval_extents[2]:.2f} mm")
    
    print("Sampling 1,000,000 points from both surfaces for accurate distance computation...")
    pcd_eval = eval_mesh.sample_points_uniformly(number_of_points=1000000)
    pcd_stl = stl_mesh.sample_points_uniformly(number_of_points=1000000)
    
    print("Downsampling to 50,000 points for ICP alignment...")
    pcd_eval_icp = eval_mesh.sample_points_uniformly(number_of_points=50000)
    pcd_stl_icp = stl_mesh.sample_points_uniformly(number_of_points=50000)
    
    print("Performing ICP Alignment (Point-to-Point)...")
    reg = o3d.pipelines.registration.registration_icp(
        pcd_eval_icp, pcd_stl_icp, 5.0, np.eye(4),
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
    
    # Save the aligned mesh as well for visualization
    eval_mesh.transform(reg.transformation)
    aligned_mesh_path = out_error_ply.replace(".ply", "_aligned_mesh.ply")
    o3d.io.write_triangle_mesh(aligned_mesh_path, eval_mesh)
    print(f"Saved aligned generated mesh to {aligned_mesh_path}")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True, help="Path to generated mesh PLY")
    parser.add_argument("--golden", required=True, help="Path to golden STL")
    parser.add_argument("--out_error", required=True, help="Path to output error pointcloud PLY")
    args = parser.parse_args()
    
    evaluate_mesh_against_stl(args.generated, args.golden, args.out_error)
