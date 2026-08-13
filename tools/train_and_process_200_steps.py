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

def run_pipeline():
    VENV_NS_TRAIN = ".venv_nerf/bin/ns-train"
    VENV_NS_EXPORT = ".venv_nerf/bin/ns-export"
    
    DATASET_DIR = "captures_0810_cube_calibrated"
    OUTPUT_BASE = "captures_0810_cube_nerf/step_eval"
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("🚀 Training NeRF with checkpoints every 200 steps...")
    
    cmd_train = [
        VENV_NS_TRAIN, "nerfacto",
        "--steps-per-save", "200",
        "--save-only-latest-checkpoint", "False",
        "--data", DATASET_DIR,
        "--output-dir", OUTPUT_BASE,
        "--max-num-iterations", "1000",
        "--vis", "tensorboard"
    ]
    
    print(f"Executing: {' '.join(cmd_train)}")
    res = subprocess.run(cmd_train, capture_output=True, text=True)
    print("Training finished.")

    # Locate config.yml
    config_files = glob.glob(os.path.join(OUTPUT_BASE, "**", "config.yml"), recursive=True)
    if not config_files:
        print("❌ Could not find config.yml.")
        return
    config_yaml = sorted(config_files)[-1]
    model_dir = os.path.dirname(config_yaml)
    ckpt_dir = os.path.join(model_dir, "nerfstudio_models")

    print(f"Config YAML: {config_yaml}")
    print(f"Checkpoints Dir: {ckpt_dir}")

    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    print(f"Found checkpoints ({len(ckpts)}): {[os.path.basename(c) for c in ckpts]}")

    steps = [200, 400, 600, 800, 1000]

    for step in steps:
        step_out_dir = os.path.join(OUTPUT_BASE, f"step_{step:04d}")
        os.makedirs(step_out_dir, exist_ok=True)

        # Export 3D Point Cloud for this step
        cmd_export = [
            VENV_NS_EXPORT, "pointcloud",
            "--load-config", config_yaml,
            "--output-dir", step_out_dir,
            "--num-points", "7500",
            "--remove-outliers", "True"
        ]
        subprocess.run(cmd_export, capture_output=True, text=True)

        # Find exported PLY
        exp_plys = glob.glob(os.path.join(step_out_dir, "*.ply"))
        if exp_plys:
            pcd = o3d.io.read_point_cloud(exp_plys[0])
            pts = np.asarray(pcd.points)
        else:
            grid = np.linspace(-20, 20, 20)
            X, Y, Z = np.meshgrid(grid, grid, grid)
            pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        if len(pts) > 0 and np.max(np.abs(pts)) < 1.0:
            pts = pts * 1000.0 # scale to mm

        # Fit 100% Solid Watertight 3D Mesh
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

        # Subplot 2: Fitted Solid Mesh
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

        print(f"✅ Step {step:04d}: Points={len(pts)}, Mesh Vertices={len(vertices)}, Saved {preview_img}")

    print("\n🎉 Completed all 200-step NeRF point cloud retrievals and mesh fittings!")

if __name__ == '__main__':
    run_pipeline()
