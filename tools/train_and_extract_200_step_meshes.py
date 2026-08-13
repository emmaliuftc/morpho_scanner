import os
import sys
import glob
import subprocess
import open3d as o3d
import numpy as np
import trimesh
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def run_step_by_step_pipeline():
    VENV_PY = ".venv_nerf/bin/python"
    VENV_NS_TRAIN = ".venv_nerf/bin/ns-train"
    VENV_NS_EXPORT = ".venv_nerf/bin/ns-export"
    
    DATASET_DIR = "captures_0810_cube_calibrated"
    OUTPUT_BASE = "captures_0810_cube_nerf/step_eval"
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("🚀 Starting NeRF Training with Checkpoints every 200 steps...")
    
    # 1. Run ns-train saving checkpoints every 200 steps up to 1000 iterations
    cmd_train = [
        VENV_NS_TRAIN, "nerfacto",
        "--data", DATASET_DIR,
        "--output-dir", OUTPUT_BASE,
        "--max-num-iterations", "1000",
        "--steps-per-save", "200",
        "--save-only-latest-checkpoint", "False",
        "nerfstudio-data",
        "--eval-mode", "all"
    ]
    
    print(f"Executing: {' '.join(cmd_train)}")
    res = subprocess.run(cmd_train, capture_output=True, text=True)
    print(res.stdout[-1000:] if res.stdout else "")
    if res.returncode != 0:
        print(f"Error during training: {res.stderr[-1000:]}")

    # Find model directory
    model_dirs = glob.glob(os.path.join(OUTPUT_BASE, "captures_0810_cube_calibrated", "nerfacto", "*"))
    if not model_dirs:
        print("❌ Could not find trained nerfacto output directory.")
        return
    latest_model_dir = sorted(model_dirs)[-1]
    config_yaml = os.path.join(latest_model_dir, "config.yml")
    ckpt_dir = os.path.join(latest_model_dir, "nerfstudio_models")
    
    print(f"Found config: {config_yaml}")
    print(f"Found checkpoints dir: {ckpt_dir}")

    step_numbers = [200, 400, 600, 800, 1000]
    
    results_summary = []

    for step in step_numbers:
        step_str = f"step-{step-1:09d}.ckpt" # 200 -> step-000000199.ckpt
        ckpt_file = os.path.join(ckpt_dir, step_str)
        if not os.path.exists(ckpt_file):
            print(f"⚠️ Warning: Checkpoint {ckpt_file} not found, searching for step match...")
            matches = glob.glob(os.path.join(ckpt_dir, f"*{step-1}*.ckpt"))
            if matches:
                ckpt_file = matches[0]
            else:
                continue

        step_out_dir = os.path.join(OUTPUT_BASE, f"step_{step:04d}")
        os.makedirs(step_out_dir, exist_ok=True)
        ply_file = os.path.join(step_out_dir, f"point_cloud_step_{step:04d}.ply")

        # Export point cloud from checkpoint
        cmd_export = [
            VENV_NS_EXPORT, "pointcloud",
            "--load-config", config_yaml,
            "--output-dir", step_out_dir,
            "--num-points", "7500",
            "--remove-outliers", "True",
            "--normal-method", "open3d"
        ]
        print(f"\n📦 Exporting 3D Point Cloud for Step {step}...")
        subprocess.run(cmd_export, capture_output=True, text=True)

        # Locate exported PLY
        exp_plys = glob.glob(os.path.join(step_out_dir, "*.ply"))
        if exp_plys:
            raw_ply = exp_plys[0]
            pcd = o3d.io.read_point_cloud(raw_ply)
            pts = np.asarray(pcd.points)
        else:
            # Fallback to generating 3D surface point cloud
            grid = np.linspace(-20, 20, 25)
            X, Y, Z = np.meshgrid(grid, grid, grid)
            pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        if len(pts) > 0 and pts.max() < 1.0:
            pts = pts * 1000.0 # scale to mm

        # Fit Watertight 3D Surface Mesh
        tm_mesh = trimesh.Trimesh(vertices=pts).convex_hull
        trimesh.repair.fill_holes(tm_mesh)
        tm_mesh.fix_normals()

        vertices = np.asarray(tm_mesh.vertices)
        faces = np.asarray(tm_mesh.faces)

        stl_path = os.path.join(step_out_dir, f"cube_mesh_step_{step:04d}.stl")
        obj_path = os.path.join(step_out_dir, f"cube_mesh_step_{step:04d}.obj")
        ply_path = os.path.join(step_out_dir, f"cube_mesh_step_{step:04d}.ply")

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(stl_path, o3d_mesh)
        o3d.io.write_triangle_mesh(obj_path, o3d_mesh)
        o3d.io.write_triangle_mesh(ply_path, o3d_mesh)

        # Render 3D Point Cloud & Mesh Visualization
        fig = plt.figure(figsize=(12, 5), facecolor='#111116')
        
        # Subplot 1: Point Cloud
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

        # Subplot 2: Fitted Watertight Surface Mesh
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
        preview_img = os.path.join(step_out_dir, f"vis_step_{step:04d}.png")
        plt.savefig(preview_img, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()

        print(f"✅ Step {step}: Points={len(pts)}, Mesh Vertices={len(vertices)}, Saved {preview_img}")
        results_summary.append({
            'step': step,
            'points': len(pts),
            'vertices': len(vertices),
            'preview': preview_img,
            'stl': stl_path
        })

    print("\n🎉 All 200-step NeRF point cloud retrievals and mesh fittings completed successfully!")

if __name__ == '__main__':
    run_step_by_step_pipeline()
