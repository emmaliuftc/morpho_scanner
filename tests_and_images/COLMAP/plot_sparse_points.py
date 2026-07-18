import os
import sys
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pycolmap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_projections(workspace_dir: str):
    workspace = Path(workspace_dir)
    sparse_model_dir = workspace / "sparse" / "0"
    output_dir = workspace / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    if not sparse_model_dir.exists():
        logging.error(f"Sparse model directory not found at {sparse_model_dir}. Have you run Stage 4?")
        sys.exit(1)
        
    # Load the reconstruction
    logging.info(f"Loading sparse reconstruction from {sparse_model_dir}...")
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read(str(sparse_model_dir))
    
    points_3d = reconstruction.points3D
    if not points_3d:
        logging.error("No 3D points found in the reconstruction.")
        sys.exit(1)
        
    logging.info(f"Loaded {len(points_3d)} 3D points. Extracting coordinates and colors...")
    
    # Extract XYZ and RGB colors
    xyz = []
    colors = []
    for point in points_3d.values():
        xyz.append(point.xyz)
        # Normalize RGB to [0, 1] for matplotlib
        colors.append(point.color / 255.0)
        
    xyz = np.array(xyz)
    colors = np.array(colors)
    
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]
    
    # Set up matplotlib figure (2x2 subplots)
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f"Sparse Point Cloud Projections ({len(points_3d)} points)", fontsize=16, fontweight='bold')
    
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
    
    output_path = output_dir / "point_cloud_projections.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logging.info(f"Saved projections visualization to {output_path}")

def main():
    workspace_path = "./workspace"
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
        
    generate_projections(workspace_path)

if __name__ == "__main__":
    main()
