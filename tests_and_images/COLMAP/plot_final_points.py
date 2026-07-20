import os
import sys
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_final_projections(workspace_dir: str):
    workspace = Path(workspace_dir)
    ply_path = workspace / "lego_block_final.ply"
    output_dir = workspace / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    if not ply_path.exists():
        logging.error(f"Final point cloud not found at {ply_path}. Have you run stage_6_poisson_input.py?")
        sys.exit(1)
        
    # Load the point cloud
    logging.info(f"Loading final point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    if not pcd.has_points():
        logging.error("No points found in the final point cloud.")
        sys.exit(1)
        
    logging.info(f"Loaded {len(pcd.points)} 3D points. Extracting coordinates and colors...")
    
    xyz = np.asarray(pcd.points)
    # Check if we have colors
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        # Fallback to default blue if no colors are present
        colors = np.array([[0, 0, 1] for _ in range(len(xyz))])
        
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]
    
    # Set up matplotlib figure (2x2 subplots)
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f"Final Point Cloud Projections ({len(pcd.points)} points)", fontsize=16, fontweight='bold')
    
    # 1. 3D Perspective view
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(X, Y, Z, c=colors, s=5, marker='o')
    ax1.set_title("3D Perspective View", fontweight='bold')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    
    # 2. XY Plane (Top-down view)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(X, Y, c=colors, s=6, marker='o')
    ax2.set_title("Top-Down View (X vs Y)", fontweight='bold')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axis('equal')
    
    # 3. XZ Plane (Front view)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(X, Z, c=colors, s=6, marker='o')
    ax3.set_title("Front View (X vs Z)", fontweight='bold')
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.axis('equal')
    
    # 4. YZ Plane (Side view)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(Y, Z, c=colors, s=6, marker='o')
    ax4.set_title("Side View (Y vs Z)", fontweight='bold')
    ax4.set_xlabel("Y")
    ax4.set_ylabel("Z")
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.axis('equal')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = output_dir / "final_point_cloud_projections.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logging.info(f"Saved projections visualization to {output_path}")

def main():
    workspace_path = "./workspace"
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
        
    generate_final_projections(workspace_path)

if __name__ == "__main__":
    main()
