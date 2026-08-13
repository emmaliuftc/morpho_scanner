import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

mesh_path = sys.argv[1]
out_path = sys.argv[2]
title = sys.argv[3]

mesh = o3d.io.read_triangle_mesh(mesh_path)
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

fig = plt.figure(figsize=(10, 8), facecolor='#111116')
ax = fig.add_subplot(111, projection='3d', facecolor='#111116')
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
verts = vertices[triangles]
poly = Poly3DCollection(verts, facecolors='#00E664', edgecolors='#005724', linewidths=0.2, alpha=0.85)
ax.add_collection3d(poly)

# Set equal aspect ratio bounds
max_range = np.array([vertices[:,0].max()-vertices[:,0].min(), 
                      vertices[:,1].max()-vertices[:,1].min(), 
                      vertices[:,2].max()-vertices[:,2].min()]).max() / 2.0
mid_x = (vertices[:,0].max()+vertices[:,0].min()) * 0.5
mid_y = (vertices[:,1].max()+vertices[:,1].min()) * 0.5
mid_z = (vertices[:,2].max()+vertices[:,2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_title(title, color='white', fontsize=14, pad=15)
ax.set_xlabel("X", color='white')
ax.set_ylabel("Y", color='white')
ax.set_zlabel("Z", color='white')
ax.tick_params(colors='white')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.tight_layout()
plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
print(f"Saved {out_path}")
