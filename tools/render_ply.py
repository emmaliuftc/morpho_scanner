import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def render_point_cloud(ply_path, out_png, max_points=100000):
    print(f"Loading point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    if len(points) == 0:
        print("Error: Point cloud is empty!")
        return
        
    print(f"Loaded {len(points)} points.")
    
    # Subsample if too large to avoid matplotlib freezing/OOM
    if len(points) > max_points:
        print(f"Subsampling to {max_points} points for rendering...")
        # Random choice
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]
            
    # Create 3 subplots for Top (XY), Front (XZ), Side (YZ)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
    
    # We want a dark theme for better visibility of points
    for ax in axs:
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
            
    # Normalize colors for matplotlib (must be [0, 1])
    if colors is not None:
        if colors.max() > 1.0:
            colors = colors / 255.0
    else:
        colors = 'white'
        
    s = 0.5  # marker size
    
    # Top View (X-Y)
    print("Rendering Top View...")
    axs[0].scatter(points[:, 0], points[:, 1], c=colors, s=s, marker='.', alpha=0.8)
    axs[0].set_title("Top View (XY)", color='white')
    axs[0].set_aspect('equal', 'datalim')
    
    # Front View (X-Z)
    print("Rendering Front View...")
    axs[1].scatter(points[:, 0], points[:, 2], c=colors, s=s, marker='.', alpha=0.8)
    axs[1].set_title("Front View (XZ)", color='white')
    axs[1].set_aspect('equal', 'datalim')
    
    # Side View (Y-Z)
    print("Rendering Side View...")
    axs[2].scatter(points[:, 1], points[:, 2], c=colors, s=s, marker='.', alpha=0.8)
    axs[2].set_title("Side View (YZ)", color='white')
    axs[2].set_aspect('equal', 'datalim')
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved render to {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", help="Path to input .ply file")
    parser.add_argument("png", help="Path to output .png file")
    args = parser.parse_args()
    
    render_point_cloud(args.ply, args.png)
