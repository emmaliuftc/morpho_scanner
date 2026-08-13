import os
import sys
import torch
import numpy as np
import open3d as o3d
import trimesh
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.exporter.exporter_utils import generate_point_cloud

def train_and_extract_200_step_meshes():
    DATASET_DIR = "captures_0810_cube_calibrated"
    OUTPUT_BASE = "captures_0810_cube_nerf/step_eval"
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("🚀 Initializing Nerfstudio Python Trainer API...")
    config: TrainerConfig = method_configs["nerfacto"]
    config.data = DATASET_DIR
    config.output_dir = OUTPUT_BASE
    config.max_num_iterations = 1000
    config.save_only_latest_checkpoint = False
    config.vis = "disabled"

    trainer = config.setup()

    target_steps = [200, 400, 600, 800, 1000]

    print("🚀 Starting Step-by-Step NeRF Training (200, 400, 600, 800, 1000)...")

    for step in range(1, 1001):
        loss_dict, metrics_dict, num_rays = trainer.train_iteration(step)

        if step in target_steps:
            print(f"\n📍 Reached Step {step}! Extracting 3D Point Cloud & Fitting Watertight Mesh...")
            step_out_dir = os.path.join(OUTPUT_BASE, f"step_{step:04d}")
            os.makedirs(step_out_dir, exist_ok=True)

            # Generate Point Cloud directly from trained pipeline
            pipeline = trainer.pipeline
            pipeline.eval()
            with torch.no_grad():
                pcd = generate_point_cloud(
                    pipeline,
                    num_points=7500,
                    remove_outliers=True,
                    estimate_normals=False
                )
            pipeline.train()

            pts = np.asarray(pcd.points)
            if len(pts) > 0 and np.max(np.abs(pts)) < 1.0:
                pts = pts * 1000.0 # scale to mm

            # Save 3D Point Cloud PLY
            ply_path = os.path.join(step_out_dir, f"point_cloud_step_{step:04d}.ply")
            pcd_mm = o3d.geometry.PointCloud()
            pcd_mm.points = o3d.utility.Vector3dVector(pts)
            o3d.io.write_point_cloud(ply_path, pcd_mm)

            # Fit Watertight 3D Surface Mesh
            tm_mesh = trimesh.Trimesh(vertices=pts).convex_hull
            trimesh.repair.fill_holes(tm_mesh)
            tm_mesh.fix_normals()

            vertices = np.asarray(tm_mesh.vertices)
            faces = np.asarray(tm_mesh.faces)

            stl_mesh_path = os.path.join(step_out_dir, f"cube_mesh_step_{step:04d}.stl")
            obj_mesh_path = os.path.join(step_out_dir, f"cube_mesh_step_{step:04d}.obj")

            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d_mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(stl_mesh_path, o3d_mesh)
            o3d.io.write_triangle_mesh(obj_mesh_path, o3d_mesh)

            # Render 3D Point Cloud & Solid Mesh Side-by-Side Visualization
            fig = plt.figure(figsize=(12, 5), facecolor='#111116')
            
            # Subplot 1: 3D Point Cloud
            ax1 = fig.add_subplot(121, projection='3d', facecolor='#111116')
            sub_pts = pts[np.random.choice(len(pts), min(2500, len(pts)), replace=False)]
            ax1.scatter(sub_pts[:, 0], sub_pts[:, 1], sub_pts[:, 2], c='#00E664', s=3, alpha=0.8)
            ax1.set_title(f"Step {step}: NeRF 3D Point Cloud", color='white', fontsize=12)
            ax1.set_xlabel("X (mm)", color='white')
            ax1.set_ylabel("Y (mm)", color='white')
            ax1.set_zlabel("Z (mm)", color='white')
            ax1.tick_params(colors='white')
            ax1.xaxis.pane.fill = False
            ax1.yaxis.pane.fill = False
            ax1.zaxis.pane.fill = False

            # Subplot 2: Fitted Solid Watertight Surface Mesh
            ax2 = fig.add_subplot(122, projection='3d', facecolor='#111116')
            verts = vertices[faces]
            poly = Poly3DCollection(verts, facecolors='#00E664', edgecolors='#005724', linewidths=0.4, alpha=0.9)
            ax2.add_collection3d(poly)
            ax2.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
            ax2.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
            ax2.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
            ax2.set_title(f"Step {step}: Fitted Solid Surface Mesh", color='white', fontsize=12)
            ax2.set_xlabel("X (mm)", color='white')
            ax2.set_ylabel("Y (mm)", color='white')
            ax2.set_zlabel("Z (mm)", color='white')
            ax2.tick_params(colors='white')
            ax2.xaxis.pane.fill = False
            ax2.yaxis.pane.fill = False
            ax2.zaxis.pane.fill = False

            plt.tight_layout()
            preview_img = os.path.join(OUTPUT_BASE, f"step_{step:04d}_vis.png")
            plt.savefig(preview_img, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()

            print(f"✅ Saved Step {step:04d} Point Cloud PLY, STL Mesh, and Preview Image to {preview_img}")

    print("\n🎉 Completed all 200-step NeRF point cloud extractions and 3D mesh fittings!")

if __name__ == '__main__':
    train_and_extract_200_step_meshes()
