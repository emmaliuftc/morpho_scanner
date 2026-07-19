import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d
from pathlib import Path
import sys

# Paths
mesh_path = Path("COLMAP/workspace/visualizations/poisson_mesh.ply")
pcd_path = Path("COLMAP/workspace/lego_block_final.ply")
out_path = Path("COLMAP/workspace/visualizations/lego_rendered_3d.png")

# Load final filtered point cloud
if not pcd_path.exists():
    print(f"Error: Point cloud not found at {pcd_path}")
    sys.exit(1)
    
pcd = o3d.io.read_point_cloud(str(pcd_path))
points = np.asarray(pcd.points)

print(f"Loaded {len(points)} points from final point cloud.")

# Load Poisson mesh to render surface triangles
if not mesh_path.exists():
    print(f"Error: Poisson mesh not found at {mesh_path}")
    sys.exit(1)
    
mesh = o3d.io.read_triangle_mesh(str(mesh_path))
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

print(f"Loaded {len(vertices)} vertices and {len(triangles)} triangles from Poisson mesh.")

# Filter mesh vertices to match the cylinder crop of stage 6
# Cylinder filter parameters
plate_center = np.array([0.8568, 1.5746, 2.7133])
normal = np.array([0.20758226, 0.71598242, 0.66654241])

V = vertices - plate_center
h = V @ normal
U = V - np.outer(h, normal)
r = np.linalg.norm(U, axis=1)

# Keep mask
keep_mask = (r < 0.8) & (h < 0.2)
kept_indices = np.where(keep_mask)[0]

# Remap vertices and filter triangles
vertex_map = {old: new for new, old in enumerate(kept_indices)}
filtered_triangles = []
for tri in triangles:
    if tri[0] in vertex_map and tri[1] in vertex_map and tri[2] in vertex_map:
        filtered_triangles.append([vertex_map[tri[0]], vertex_map[tri[1]], vertex_map[tri[2]]])

filtered_vertices = vertices[keep_mask]
filtered_triangles = np.array(filtered_triangles)

print(f"After cropping: {len(filtered_vertices)} vertices and {len(filtered_triangles)} triangles.")

# Set up matplotlib figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Render the 3D surface triangles with shading
if len(filtered_triangles) > 0:
    ax.plot_trisurf(filtered_vertices[:, 0], filtered_vertices[:, 1], filtered_vertices[:, 2], 
                    triangles=filtered_triangles, 
                    color='crimson', edgecolor='none', alpha=0.9, shade=True)
else:
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='crimson', s=40, depthshade=True)

# Add some nice lighting and clean bounds
ax.set_title("3D Reconstructed Lego Block Mesh (Poisson Surface)", fontsize=16, fontweight='bold')
ax.set_xlabel("X (reconstruction scale)")
ax.set_ylabel("Y (reconstruction scale)")
ax.set_zlabel("Z (reconstruction scale)")

# Set camera angle for a beautiful perspective
ax.view_init(elev=30, azim=45)

# Keep aspect ratio equal
max_range = np.array([filtered_vertices[:, 0].max() - filtered_vertices[:, 0].min(), 
                      filtered_vertices[:, 1].max() - filtered_vertices[:, 1].min(), 
                      filtered_vertices[:, 2].max() - filtered_vertices[:, 2].min()]).max() / 2.0

mid_x = (filtered_vertices[:, 0].max() + filtered_vertices[:, 0].min()) * 0.5
mid_y = (filtered_vertices[:, 1].max() + filtered_vertices[:, 1].min()) * 0.5
mid_z = (filtered_vertices[:, 2].max() + filtered_vertices[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.savefig(str(out_path), bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved 3D render successfully to {out_path}")
