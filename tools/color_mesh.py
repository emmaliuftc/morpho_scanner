import open3d as o3d
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", type=str, required=True, help="Original point cloud with colors")
    parser.add_argument("--mesh", type=str, required=True, help="Generated mesh without colors")
    parser.add_argument("--output", type=str, required=True, help="Output colored mesh")
    args = parser.parse_args()

    print(f"Loading point cloud from {args.pcd}...")
    pcd = o3d.io.read_point_cloud(args.pcd)
    if not pcd.has_colors():
        print("Point cloud does not have colors!")
        return

    print(f"Loading mesh from {args.mesh}...")
    mesh = o3d.io.read_triangle_mesh(args.mesh)

    print("Building KDTree and mapping colors to mesh vertices...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_colors = np.zeros_like(mesh_vertices)
    pcd_colors = np.asarray(pcd.colors)

    for i in range(len(mesh_vertices)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(mesh_vertices[i], 1)
        mesh_colors[i] = pcd_colors[idx[0]]

    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    
    print(f"Saving colored mesh to {args.output}...")
    o3d.io.write_triangle_mesh(args.output, mesh)
    print("Done!")

if __name__ == "__main__":
    main()
