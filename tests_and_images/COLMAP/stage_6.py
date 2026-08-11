import os
import sys
import logging
from pathlib import Path
import open3d as o3d

# Establish standard logging protocol for the pipeline execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the current script's directory is in python path to import stage_1
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stage_1 import LegoReconstructionPipeline

def stage_6_spatial_post_processing(pipeline: LegoReconstructionPipeline):
    """
    Sanitizes the dense point cloud utilizing KD-Tree Statistical Outlier 
    Removal and Oriented Bounding Box coordinate cropping.
    """
    logging.info("Initiating Stage 6: Open3D Point Cloud Post-Processing")
    
    raw_cloud_path = str(pipeline.dense_dir / "fused_dense.ply")
    
    if not os.path.exists(raw_cloud_path):
        raise FileNotFoundError(f"Dense point cloud absent at expected path: {raw_cloud_path}")
        
    # Ingest the dense point cloud tensor into system memory
    logging.info("Loading volumetric data...")
    pcd = o3d.io.read_point_cloud(raw_cloud_path)
    
    initial_points = len(pcd.points)
    logging.info(f"Pre-filter spatial tensor magnitude: {initial_points} points")
    
    # 1. Statistical Outlier Removal Execution
    # nb_neighbors: Number of adjacent points to analyze via KD-Tree
    # std_ratio: Standard deviation threshold; lower values yield aggressive culling
    logging.info("Executing Statistical Outlier Removal (SOR)...")
    cleaned_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    
    filtered_points = len(cleaned_pcd.points)
    logging.info(f"Tensor magnitude post-SOR: {filtered_points} "
                 f"(Purged {initial_points - filtered_points} spatial anomalies)")
                 
    # 2. Automated Concentric Cropping (Isolating the Lego Geometry)
    # Formulate an Oriented Bounding Box based on PCA analysis of the cloud
    bbox = cleaned_pcd.get_oriented_bounding_box(robust=True)
    
    # Scale the bounding box to concentrically constrain the geometric edges.
    # A multiplier of 0.95 reduces the box volume by 5%, effectively stripping 
    # away potential tuning plate artifacts on the extreme periphery.
    bbox.scale(0.95, bbox.get_center())
    
    # Execute crop using the scaled geometric primitive
    final_pcd = cleaned_pcd.crop(bbox)
    
    logging.info(f"Final production point count post-cropping: {len(final_pcd.points)}")
    
    # Recompute surface normal vectors utilizing covariance analysis 
    # to facilitate accurate shading and rendering in downstream applications.
    final_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Export the finalized, pristine point cloud asset
    final_path = str(pipeline.final_cloud_path)
    o3d.io.write_point_cloud(final_path, final_pcd)
    
    logging.info(f"Stage 6 Complete: Production-ready asset secured at {final_path}")


def main():
    workspace_path = "./workspace"
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
        
    print(f"Loading COLMAP workspace at: {os.path.abspath(workspace_path)}")
    # We set init_workspace=False to preserve existing folders and outputs
    pipeline = LegoReconstructionPipeline(workspace_path, init_workspace=False)
    
    try:
        stage_6_spatial_post_processing(pipeline)
        print("Success! Stage 6 Point Cloud Refinement completed.")
    except Exception as e:
        logging.error(f"Stage 6 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
