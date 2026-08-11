import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import numpy as np

def render_3d_point_cloud():
    PLY_PATH = "captures_0810_cube_nerf/export/cube_nerf_1000_pcd.ply"
    PREVIEW_IMG = "captures_0810_cube_nerf/nerf_point_cloud_preview.png"

    pcd = o3d.io.read_point_cloud(PLY_PATH)
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(colors) == 0:
        colors = np.zeros_like(pts)
        colors[:, 1] = 0.9 # Green

    fig = plt.figure(figsize=(10, 8), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d', facecolor='#111116')

    # Subsample points for clean rendering
    indices = np.random.choice(len(pts), min(3000, len(pts)), replace=False)
    sub_pts = pts[indices]
    sub_cols = colors[indices]

    ax.scatter(sub_pts[:, 0], sub_pts[:, 1], sub_pts[:, 2], c=sub_cols, s=4, alpha=0.8)

    ax.set_title("NeRF Reconstructed 3D Point Cloud (7,350 Points)", color='white', fontsize=14, pad=15)
    ax.set_xlabel("X (mm)", color='white')
    ax.set_ylabel("Y (mm)", color='white')
    ax.set_zlabel("Z (mm)", color='white')
    ax.tick_params(colors='white')
    
    # Hide grid panes for sleek dark look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig(PREVIEW_IMG, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"✅ Rendered 3D Point Cloud visualization screenshot to {PREVIEW_IMG}")

if __name__ == '__main__':
    render_3d_point_cloud()
