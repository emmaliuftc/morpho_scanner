import argparse
import open3d as o3d
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Create a Poisson mesh from a filtered point cloud")
    parser.add_argument("--input", type=str, required=True, help="Input .ply point cloud")
    parser.add_argument("--output", type=str, required=True, help="Output .ply mesh")
    parser.add_argument("--depth", type=int, default=8, help="Poisson reconstruction depth (lower = smoother contour)")
    args = parser.parse_args()

    print(f"Loading filtered point cloud from {args.input}...")
    pcd = o3d.io.read_point_cloud(args.input)
    
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    print(f"Running Poisson Surface Reconstruction (depth={args.depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.depth)
    
    densities = np.asarray(densities)
    if len(densities) > 0:
        print("Cleaning up low-density vertices (5th percentile)...")
        vertices_to_remove = densities < np.percentile(densities, 5)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    
    mesh.compute_vertex_normals()
    
    print("Applying Taubin 3D Smoothing (smoothing bumps without shrinking volume)...")
    # 15 iterations of Taubin smoothing acts like a robust 3D Gaussian blur for surface geometry
    mesh = mesh.filter_smooth_taubin(number_of_iterations=15)
    mesh.compute_vertex_normals()
    
    print("Filtering out small blobs (keeping only the largest connected component)...")
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    if len(cluster_n_triangles) > 0:
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
    
    print(f"Mesh created with {len(np.asarray(mesh.vertices))} vertices and {len(np.asarray(mesh.triangles))} triangles.")
    print(f"Saving to {args.output}...")
    o3d.io.write_triangle_mesh(args.output, mesh)
    
    print("Running topological hole filling (PyMeshLab)...")
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(args.output)
        
        # Repair non-manifold geometry before trying to close holes
        try:
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_vertices()
        except Exception as e:
            print(f"Warning: Non-manifold repair failed: {e}")
            
        # Close holes up to a reasonably large size (e.g. 1000 edges)
        try:
            ms.meshing_close_holes(maxholesize=1000)
        except Exception as e:
            print(f"Warning: Hole closing failed: {e}")
            
        ms.save_current_mesh(args.output)
        print("Hole filling complete!")
    except ImportError:
        print("pymeshlab not found, skipping hole filling step.")
    except Exception as e:
        print(f"An unexpected error occurred during hole filling: {e}")
        
    print("Done!")

if __name__ == "__main__":
    main()
