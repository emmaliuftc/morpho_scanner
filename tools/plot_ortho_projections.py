import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_ortho(ply_path, out_png):
    print(f"Loading {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        print("Empty point cloud!")
        return
        
    # Scale from NeRF units to millimeters
    points = points * 150.0

    # Randomly subsample for faster plotting if huge
    if len(points) > 50000:
        idx = np.random.choice(len(points), 50000, replace=False)
        points = points[idx]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # XY Plane (Top-Down)
    axs[0].scatter(points[:, 0], points[:, 1], s=0.1, alpha=0.5, c='g')
    axs[0].set_title("Top-Down View (XY Plane)")
    axs[0].set_xlabel("X (mm)")
    axs[0].set_ylabel("Y (mm)")
    axs[0].set_aspect('equal')
    
    # XZ Plane (Side View)
    axs[1].scatter(points[:, 0], points[:, 2], s=0.1, alpha=0.5, c='b')
    axs[1].set_title("Side View (XZ Plane)")
    axs[1].set_xlabel("X (mm)")
    axs[1].set_ylabel("Z (mm)")
    axs[1].set_aspect('equal')
    
    # YZ Plane (Front View)
    axs[2].scatter(points[:, 1], points[:, 2], s=0.1, alpha=0.5, c='r')
    axs[2].set_title("Front View (YZ Plane)")
    axs[2].set_xlabel("Y (mm)")
    axs[2].set_ylabel("Z (mm)")
    axs[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved projections to {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    parser.add_argument("out")
    args = parser.parse_args()
    plot_ortho(args.ply, args.out)
