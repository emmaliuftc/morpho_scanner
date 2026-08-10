import os
import sys
import logging
from pathlib import Path
import pycolmap

# Establish standard logging protocol for the pipeline execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the current script's directory is in python path to import stage_1
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stage_1 import LegoReconstructionPipeline

def stage_4_sparse_sfm_reconstruction(pipeline: LegoReconstructionPipeline):
    """
    Executes the incremental mapping pipeline to resolve 6-DOF camera poses,
    triangulate 3D coordinates, and perform Ceres bundle adjustment.
    """
    logging.info("Initiating Stage 4: Sparse Reconstruction (SfM)")
    
    # Configure Incremental Mapping options for pycolmap 4.x
    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.min_model_size = 10  # Ensure the solver doesn't output degenerate graphs
    
    # Execute the incremental mapping solver. 
    # Yields a dictionary of pycolmap.Reconstruction objects.
    maps = pycolmap.incremental_mapping(
        database_path=pipeline.db_path,
        image_path=pipeline.image_dir,
        output_path=pipeline.sparse_dir,
        options=mapper_options
    )
    
    if not maps:
        logging.error("SfM Solver Failure: No viable geometry was reconstructed.")
        raise RuntimeError("Reconstruction algorithm failed to register camera models.")
        
    # Depending on graph connectivity, COLMAP may output multiple disjoint sub-models.
    # We select the model with the largest number of registered images as our best model.
    best_model_id = max(maps.keys(), key=lambda k: maps[k].num_reg_images())
    best_model = maps[best_model_id]
    logging.info(f"SfM Graph convergence successful (Model ID: {best_model_id}): {best_model.summary()}")
    
    # Export the sparse geometry to disk (generates cameras.bin, images.bin, points3D.bin)
    model_output_dir = pipeline.sparse_dir / "0"
    model_output_dir.mkdir(exist_ok=True)
    best_model.write(str(model_output_dir))
    
    logging.info(f"Stage 4 Complete: Sparse 3D framework and camera graph exported to {model_output_dir}")


def main():
    workspace_path = "./workspace"
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
        
    print(f"Loading COLMAP workspace at: {os.path.abspath(workspace_path)}")
    # We set init_workspace=False to preserve the database and matches computed in Stage 3
    pipeline = LegoReconstructionPipeline(workspace_path, init_workspace=False)
    
    # Clean up the output directory if it exists to avoid conflicts
    sparse_model_dir = pipeline.sparse_dir / "0"
    if sparse_model_dir.exists():
        logging.info(f"Cleaning existing sparse model output directory at {sparse_model_dir}")
        shutil = __import__('shutil')
        shutil.rmtree(sparse_model_dir)
        
    try:
        stage_4_sparse_sfm_reconstruction(pipeline)
        print("Success! Stage 4 Sparse SfM Reconstruction completed.")
    except Exception as e:
        logging.error(f"Stage 4 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
