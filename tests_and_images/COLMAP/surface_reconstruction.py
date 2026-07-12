import os
import sys
import logging
from pathlib import Path
import numpy as np
import open3d as o3d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def reconstruct_surface(workspace_dir: str):
    workspace = Path(workspace_dir)
    ply_path = workspace / "visualizations" / "sparse_model.ply"
    output_dir = workspace / "visualizations"
    
    if not ply_path.exists():
        logging.error(f"Sparse point cloud not found at {ply_path}. Have you run Stage 4 and visualize_stage_4.py?")
        sys.exit(1)
        
    logging.info(f"Loading point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    if not pcd.has_points():
        logging.error("Loaded point cloud has 0 points.")
        sys.exit(1)
        
    logging.info(f"Point cloud loaded with {len(pcd.points)} points.")
    
    # 1. Normal Estimation (Crucial for Poisson and Ball Pivoting)
    logging.info("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    
    # 2. Method A: Poisson Surface Reconstruction
    logging.info("Running Poisson Surface Reconstruction...")
    # depth=8 or 9 controls the reconstruction resolution. For a sparse cloud, depth=7 or 8 is appropriate.
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    
    # Crop the mesh to the bounding box of the point cloud to avoid bloated hulls
    bbox = pcd.get_oriented_bounding_box(robust=True)
    poisson_mesh_cropped = poisson_mesh.crop(bbox)
    
    poisson_output_path = output_dir / "poisson_mesh.ply"
    o3d.io.write_triangle_mesh(str(poisson_output_path), poisson_mesh_cropped)
    logging.info(f"Saved Poisson mesh to {poisson_output_path}")
    
    # 3. Method B: Ball Pivoting Reconstruction
    logging.info("Running Ball Pivoting Reconstruction...")
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    # Radii of the balls used for pivoting (tuned based on average distance between points)
    radii = [avg_dist, avg_dist * 2, avg_dist * 4]
    
    bp_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    
    bp_output_path = output_dir / "ball_pivoting_mesh.ply"
    o3d.io.write_triangle_mesh(str(bp_output_path), bp_mesh)
    logging.info(f"Saved Ball Pivoting mesh to {bp_output_path}")


def main():
    workspace_path = "./workspace"
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
        
    reconstruct_surface(workspace_path)

if __name__ == "__main__":
    main()
