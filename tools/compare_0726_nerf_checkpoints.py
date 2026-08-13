import os
import sys
import glob
import json
import torch
import numpy as np
import open3d as o3d
import yaml

def compare_checkpoints():
    CKPT1_PATH = "/home/coding/github/morpho_scanner/outputs/captures_0726_nerf_dataset/nerfacto/2026-07-27_045135/nerfstudio_models/step-000001999.ckpt"
    CKPT2_PATH = "/home/coding/github/morpho_scanner/0813_0726_cerf_rerun/captures_0726_nerf_dataset/nerfacto/2026-08-13_183217/nerfstudio_models/step-000001999.ckpt"

    CFG1_PATH = "/home/coding/github/morpho_scanner/outputs/captures_0726_nerf_dataset/nerfacto/2026-07-27_045135/config.yml"
    CFG2_PATH = "/home/coding/github/morpho_scanner/0813_0726_cerf_rerun/captures_0726_nerf_dataset/nerfacto/2026-08-13_183217/config.yml"

    MESH1_PATH = "/home/coding/github/morpho_scanner/captures_0726_nerf_dataset/mesh.ply"
    MESH2_PATH = "/home/coding/github/morpho_scanner/0813_0726_cerf_rerun/mesh.ply"

    PCD1_PATH = "/home/coding/github/morpho_scanner/captures_0726_nerf_dataset/point_cloud_filtered.ply"
    PCD2_PATH = "/home/coding/github/morpho_scanner/0813_0726_cerf_rerun/point_cloud_filtered.ply"

    print("==========================================================================")
    print("      COMPARING 0726 ORIGINAL VS 0813 RERUN CHECKPOINTS & OUTPUTS         ")
    print("==========================================================================")

    # 1. Compare Checkpoint Files and Model Weight Differences
    print("\n--- 1. CHECKPOINT WEIGHTS COMPARISON ---")
    if os.path.exists(CKPT1_PATH) and os.path.exists(CKPT2_PATH):
        ckpt1 = torch.load(CKPT1_PATH, map_location="cpu")
        ckpt2 = torch.load(CKPT2_PATH, map_location="cpu")

        state1 = ckpt1.get("pipeline", ckpt1.get("state_dict", ckpt1))
        state2 = ckpt2.get("pipeline", ckpt2.get("state_dict", ckpt2))

        common_keys = set(state1.keys()).intersection(set(state2.keys()))
        print(f"  Checkpoint 1 (2026-07-27): {len(state1)} keys")
        print(f"  Checkpoint 2 (2026-08-13): {len(state2)} keys")
        print(f"  Common keys: {len(common_keys)}")

        total_param_diff = 0.0
        max_param_diff = 0.0
        matched_tensors = 0

        for key in common_keys:
            t1 = state1[key]
            t2 = state2[key]
            if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                if t1.shape == t2.shape and t1.numel() > 0:
                    diff = (t1.float() - t2.float()).abs()
                    mean_diff = diff.mean().item()
                    m_diff = diff.max().item()
                    total_param_diff += mean_diff
                    if m_diff > max_param_diff:
                        max_param_diff = m_diff
                    matched_tensors += 1

        avg_param_diff = total_param_diff / max(matched_tensors, 1)
        print(f"  Average Absolute Weight Difference across {matched_tensors} tensors: {avg_param_diff:.6f}")
        print(f"  Maximum Weight Difference across all tensors: {max_param_diff:.6f}")
    else:
        print("❌ One of the checkpoint files was not found.")

    # 2. Compare Configurations
    print("\n--- 2. CONFIGURATION COMPARISON ---")
    if os.path.exists(CFG1_PATH) and os.path.exists(CFG2_PATH):
        with open(CFG1_PATH, "r") as f:
            cfg1 = yaml.unsafe_load(f)
        with open(CFG2_PATH, "r") as f:
            cfg2 = yaml.unsafe_load(f)

        print(f"  Config 1 Device: {getattr(cfg1.machine, 'device_type', 'cpu')}, Seed: {getattr(cfg1, 'seed', 'N/A')}")
        print(f"  Config 2 Device: {getattr(cfg2.machine, 'device_type', 'cuda')}, Seed: {getattr(cfg2, 'seed', 'N/A')}")

    # 3. Compare 3D Meshes & Point Clouds
    print("\n--- 3. 3D GEOMETRY (POINT CLOUD & MESH) COMPARISON ---")
    if os.path.exists(MESH1_PATH) and os.path.exists(MESH2_PATH):
        mesh1 = o3d.io.read_triangle_mesh(MESH1_PATH)
        mesh2 = o3d.io.read_triangle_mesh(MESH2_PATH)

        v1 = np.asarray(mesh1.vertices)
        v2 = np.asarray(mesh2.vertices)
        f1 = np.asarray(mesh1.triangles)
        f2 = np.asarray(mesh2.triangles)

        print(f"  Original 0726 Mesh: {len(v1)} vertices, {len(f1)} triangles")
        print(f"  Rerun 0813 Mesh:    {len(v2)} vertices, {len(f2)} triangles")

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(v1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(v2)

        dists1_to_2 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
        dists2_to_1 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))

        mean_dist_12 = np.mean(dists1_to_2)
        mean_dist_21 = np.mean(dists2_to_1)
        hausdorff = max(np.max(dists1_to_2), np.max(dists2_to_1))
        chamfer = (mean_dist_12 + mean_dist_21) / 2.0

        print(f"  Mesh Point-to-Mesh Distance (0726 -> 0813): Mean = {mean_dist_12:.4f} mm, Std = {np.std(dists1_to_2):.4f} mm")
        print(f"  Mesh Point-to-Mesh Distance (0813 -> 0726): Mean = {mean_dist_21:.4f} mm, Std = {np.std(dists2_to_1):.4f} mm")
        print(f"  Symmetric Chamfer Distance: {chamfer:.4f} mm")
        print(f"  Hausdorff Distance (Max deviation): {hausdorff:.4f} mm")
    else:
        print("  Mesh files comparison pending.")

    # 4. Compare Point Clouds
    if os.path.exists(PCD1_PATH) and os.path.exists(PCD2_PATH):
        pcd1 = o3d.io.read_point_cloud(PCD1_PATH)
        pcd2 = o3d.io.read_point_cloud(PCD2_PATH)

        pts1 = np.asarray(pcd1.points)
        pts2 = np.asarray(pcd2.points)

        print(f"\n  Original 0726 Point Cloud: {len(pts1)} points")
        print(f"  Rerun 0813 Point Cloud:    {len(pts2)} points")

        d12 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
        d21 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
        ch_pcd = (np.mean(d12) + np.mean(d21)) / 2.0

        print(f"  Point Cloud Chamfer Distance: {ch_pcd:.4f} mm")

    print("\n==========================================================================")

if __name__ == "__main__":
    compare_checkpoints()
