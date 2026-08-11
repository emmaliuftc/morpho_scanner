import os
import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def fix_watertight_mesh():
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

    # 2. Fit 3D Mesh using Trimesh Convex Hull + Hole Filling to guarantee 100% Solid Watertight Cube
    tm_mesh = trimesh.Trimesh(vertices=pts).convex_hull
    trimesh.repair.fill_holes(tm_mesh)
    tm_mesh.fix_normals()

    vertices = np.asarray(tm_mesh.vertices)
    faces = np.asarray(tm_mesh.faces)
    
    print(f"✅ Solid Watertight Mesh: {len(vertices)} vertices, {len(faces)} faces.")
    print(f"Is Mesh Watertight? {tm_mesh.is_watertight}")

    # 3. Export to Open3D & Save Mesh Formats
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()

    vertex_colors = np.zeros_like(vertices)
    vertex_colors[:, 0] = 0.0  # R
    vertex_colors[:, 1] = 0.9  # G
    vertex_colors[:, 2] = 0.3  # B
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    o3d.io.write_triangle_mesh(STL_OUT, o3d_mesh)
    o3d.io.write_triangle_mesh(OBJ_OUT, o3d_mesh)
    o3d.io.write_triangle_mesh(PLY_OUT, o3d_mesh)

    print(f"✅ Saved Solid STL mesh: {STL_OUT}")
    print(f"✅ Saved Solid OBJ mesh: {OBJ_OUT}")
    print(f"✅ Saved Solid PLY mesh: {PLY_OUT}")

    # 4. Render COMPLETE Solid 3D Mesh Preview (Rendering 100% of triangles, NO subsampling!)
    fig = plt.figure(figsize=(10, 8), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d', facecolor='#111116')

    verts = vertices[faces]
    poly = Poly3DCollection(
        verts,
        facecolors='#00E664',
        edgecolors='#005724',
        linewidths=0.4,
        alpha=0.95
    )
    ax.add_collection3d(poly)

    min_b = vertices.min(axis=0)
    max_b = vertices.max(axis=0)
    ax.set_xlim(min_b[0], max_b[0])
    ax.set_ylim(min_b[1], max_b[1])
    ax.set_zlim(min_b[2], max_b[2])

    ax.set_title("Solid Watertight 3D Mesh Reconstruction (100% Closed Cube)", color='white', fontsize=14, pad=15)
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
    print(f"✅ Saved Solid 3D Mesh Preview Screenshot to {PREVIEW_IMG}")

if __name__ == '__main__':
    fix_watertight_mesh()
