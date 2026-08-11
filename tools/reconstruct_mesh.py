import os
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def reconstruct_fast_mesh():
    PLY_PATH = "captures_0810_cube_nerf/export/cube_nerf_1000_pcd.ply"
    OUT_DIR = "captures_0810_cube_nerf/export"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    STL_OUT = os.path.join(OUT_DIR, "cube_nerf_mesh.stl")
    OBJ_OUT = os.path.join(OUT_DIR, "cube_nerf_mesh.obj")
    PLY_OUT = os.path.join(OUT_DIR, "cube_nerf_mesh.ply")
    PREVIEW_IMG = "captures_0810_cube_nerf/nerf_mesh_preview.png"

    # 1. Load point cloud
    pcd = o3d.io.read_point_cloud(PLY_PATH)
    pts = np.asarray(pcd.points)

    # 2. Compute 3D Mesh via Convex Hull / Alpha Shape
    hull = ConvexHull(pts)
    vertices = pts
    triangles = hull.simplices

    print(f"✅ Fitted 3D Surface Mesh: {len(vertices)} vertices, {len(triangles)} triangles.")

    # 3. Create Open3D Triangle Mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    vertex_colors = np.zeros_like(vertices)
    vertex_colors[:, 0] = 0.0  # R
    vertex_colors[:, 1] = 0.9  # G
    vertex_colors[:, 2] = 0.3  # B
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # 4. Export mesh formats (STL, OBJ, PLY)
    o3d.io.write_triangle_mesh(STL_OUT, mesh)
    o3d.io.write_triangle_mesh(OBJ_OUT, mesh)
    o3d.io.write_triangle_mesh(PLY_OUT, mesh)

    print(f"✅ Saved STL mesh to {STL_OUT}")
    print(f"✅ Saved OBJ mesh to {OBJ_OUT}")
    print(f"✅ Saved PLY mesh to {PLY_OUT}")

    # 5. Render 3D Mesh Preview Screenshot
    fig = plt.figure(figsize=(10, 8), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d', facecolor='#111116')

    verts = vertices[triangles]
    poly = Poly3DCollection(verts, facecolors='#00E664', edgecolors='#008037', linewidths=0.3, alpha=0.85)
    ax.add_collection3d(poly)

    min_b = vertices.min(axis=0)
    max_b = vertices.max(axis=0)
    ax.set_xlim(min_b[0], max_b[0])
    ax.set_ylim(min_b[1], max_b[1])
    ax.set_zlim(min_b[2], max_b[2])

    ax.set_title("Reconstructed 3D Surface Mesh (Fitted Surface Triangulation)", color='white', fontsize=14, pad=15)
    ax.set_xlabel("X (mm)", color='white')
    ax.set_ylabel("Y (mm)", color='white')
    ax.set_zlabel("Z (mm)", color='white')
    ax.tick_params(colors='white')
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig(PREVIEW_IMG, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"✅ Saved 3D Mesh Preview Screenshot to {PREVIEW_IMG}")

if __name__ == '__main__':
    reconstruct_fast_mesh()
