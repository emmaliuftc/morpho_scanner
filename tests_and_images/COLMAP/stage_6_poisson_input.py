import os
import sys
import logging
from pathlib import Path
import open3d as o3d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the current script's directory is in python path to import stage_1
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stage_1 import LegoReconstructionPipeline

def stage_6_poisson_input(pipeline: LegoReconstructionPipeline):
    logging.info("Initiating Stage 6 with Poisson Mesh Input")
    
    mesh_path = str(pipeline.workspace / "visualizations" / "poisson_mesh.ply")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Poisson mesh absent at expected path: {mesh_path}")
        
    logging.info(f"Loading mesh from {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Convert TriangleMesh to PointCloud by extracting vertices, colors, and normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    if mesh.has_vertex_colors():
        pcd.colors = mesh.vertex_colors
    if mesh.has_vertex_normals():
        pcd.normals = mesh.vertex_normals
        
    initial_points = len(pcd.points)
    logging.info(f"Point cloud initialized from mesh vertices: {initial_points} points.")
    
    # Cylinder filter based on derived turntable center to purge rim outliers
    logging.info("Applying cylinder filter based on derived turntable center...")
    import numpy as np
    plate_center = np.array([0.8568, 1.5746, 2.7133])
    normal = np.array([0.20758226, 0.71598242, 0.66654241]) # trajectory normal
    
    points = np.asarray(pcd.points)
    V = points - plate_center
    h = V @ normal
    U = V - np.outer(h, normal)
    r = np.linalg.norm(U, axis=1)
    
    # Keep only points within 0.8 units of center axis and above plate base
    keep_mask = (r < 0.8) & (h < 0.2)
    
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[keep_mask])
    if mesh.has_vertex_colors():
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[keep_mask])
    if mesh.has_vertex_normals():
        filtered_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[keep_mask])
        
    logging.info(f"Points post-cylinder filter: {len(filtered_pcd.points)} (Purged {initial_points - len(filtered_pcd.points)} points)")
    
    # 1. Statistical Outlier Removal (SOR)
    logging.info("Executing Statistical Outlier Removal (SOR)...")
    cleaned_pcd, ind = filtered_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    filtered_points = len(cleaned_pcd.points)
    logging.info(f"Points post-SOR: {filtered_points} (Purged {initial_points - filtered_points} spatial anomalies)")
    
    # 2. Oriented Bounding Box Cropping (concentric crop)
    logging.info("Executing Oriented Bounding Box cropping...")
    bbox = cleaned_pcd.get_oriented_bounding_box(robust=True)
    bbox.scale(0.95, bbox.get_center())
    final_pcd = cleaned_pcd.crop(bbox)
    logging.info(f"Points post-cropping: {len(final_pcd.points)}")
    
    # 3. Normal vector estimation
    logging.info("Estimating normals...")
    final_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )
    
    # Save output
    final_path = str(pipeline.workspace / "lego_block_final.ply")
    o3d.io.write_point_cloud(final_path, final_pcd)
    logging.info(f"Stage 6 Complete: Final point cloud saved to {final_path}")

def main():
    workspace_path = "./workspace"
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
        
    print(f"Loading COLMAP workspace at: {os.path.abspath(workspace_path)}")
    pipeline = LegoReconstructionPipeline(workspace_path, init_workspace=False)
    
    try:
        stage_6_poisson_input(pipeline)
        print("Success! Stage 6 completed using Poisson mesh input.")
    except Exception as e:
        logging.error(f"Stage 6 failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
