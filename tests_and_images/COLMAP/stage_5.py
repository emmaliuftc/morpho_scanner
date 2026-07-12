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

def stage_5_dense_mvs_reconstruction(pipeline: LegoReconstructionPipeline):
    """
    Executes the Multi-View Stereo (MVS) pipeline.
    This stage requires a CUDA-enabled GPU for PatchMatch stereo depth estimation.
    """
    logging.info("Initiating Stage 5: Dense MVS Volumetric Reconstruction")
    
    sparse_model_dir = pipeline.sparse_dir / "0"
    if not sparse_model_dir.exists():
        logging.error(f"Sparse model directory not found at {sparse_model_dir}. Have you run Stage 4?")
        sys.exit(1)
        
    # 1. Undistortion Phase: Warp images to the ideal pinhole projection
    logging.info("Undistorting images...")
    pycolmap.undistort_images(
        output_path=pipeline.mvs_dir,
        input_path=sparse_model_dir,
        image_path=pipeline.image_dir
    )
    logging.info("Epipolar undistortion complete.")
    
    # CUDA requirement verification
    # If CUDA is not available, we cannot run PatchMatch stereo.
    if not getattr(pycolmap, 'has_cuda', False):
        logging.critical("Dense stereo reconstruction requires CUDA. Halting execution before PatchMatch stereo.")
        raise EnvironmentError(
            "PyCOLMAP was not compiled with CUDA acceleration, or no CUDA-compatible GPU is available in this environment. "
            "PatchMatch stereo depth estimation requires a GPU. To compute dense depth maps, please run this on a GPU-enabled machine."
        )

    # 2. Depth Map Phase: PatchMatch photometric consistency estimation
    logging.info("Running PatchMatch Stereo depth estimation...")
    patch_match_options = pycolmap.PatchMatchOptions()
    pycolmap.patch_match_stereo(
        workspace_path=pipeline.mvs_dir,
        options=patch_match_options
    )
    logging.info("PatchMatch Stereo depth matrix formulation complete.")

    # 3. Fusion Phase: Consolidate depth maps into a singular point cloud
    logging.info("Running Stereo Depth Fusion...")
    fusion_output_path = pipeline.dense_dir / "fused_dense.ply"
    stereo_fusion_options = pycolmap.StereoFusionOptions()
    
    # Enforce strict multi-view consensus: a point must be seen by at least 3 cameras 
    # to be included in the final geometry, drastically reducing specular noise.
    stereo_fusion_options.min_num_pixels = 3 
    
    pycolmap.stereo_fusion(
        output_path=fusion_output_path,
        workspace_path=pipeline.mvs_dir,
        options=stereo_fusion_options
    )
    
    logging.info(f"Stage 5 Complete: Dense geometry synthesized at {fusion_output_path}")


def main():
    workspace_path = "./workspace"
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
        
    print(f"Loading COLMAP workspace at: {os.path.abspath(workspace_path)}")
    # We set init_workspace=False to preserve the database and reconstruction computed in earlier stages
    pipeline = LegoReconstructionPipeline(workspace_path, init_workspace=False)
    
    try:
        stage_5_dense_mvs_reconstruction(pipeline)
        print("Success! Stage 5 Dense MVS completed.")
    except Exception as e:
        logging.error(f"Stage 5 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
