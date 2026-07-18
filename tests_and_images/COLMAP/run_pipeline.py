import os
import sys
import logging
import shutil
from pathlib import Path

# Ensure the current script's directory is in python path to import the stage scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stage_1 import LegoReconstructionPipeline, stage_1_dynamic_masking
from stage_2 import stage_2_sift_feature_extraction
from stage_3 import stage_3_exhaustive_matching
from stage_4 import stage_4_sparse_sfm_reconstruction
from stage_5 import stage_5_dense_mvs_reconstruction
from stage_6 import stage_6_spatial_post_processing

# Establish standard logging protocol for the pipeline execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(workspace_path: str, source_captures_path: str):
    logging.info(f"Initializing End-to-End Reconstruction Pipeline at: {os.path.abspath(workspace_path)}")
    
    # Initialize workspace (wipes previous database/directory configs for a fresh run)
    pipeline = LegoReconstructionPipeline(workspace_path, init_workspace=True)
    
    # Copy images to raw_images if not already present
    existing_raw = list(pipeline.raw_dir.glob("*.jpg")) + list(pipeline.raw_dir.glob("*.jpeg"))
    if not existing_raw:
        if os.path.exists(source_captures_path):
            logging.info(f"Copying source images from {os.path.abspath(source_captures_path)} to {pipeline.raw_dir}")
            src_path = Path(source_captures_path)
            copied_count = 0
            for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
                for img_file in src_path.glob(ext):
                    shutil.copy2(img_file, pipeline.raw_dir / img_file.name)
                    copied_count += 1
            logging.info(f"Copied {copied_count} files.")
        else:
            logging.warning(f"Source captures directory '{source_captures_path}' not found.")
            logging.warning("Please deposit raw images to workspace/raw_images manually.")
            
    try:
        # Stage 1: AI-Driven Background Removal
        stage_1_dynamic_masking(pipeline)
        
        # Stage 2: SIFT Feature Extraction
        stage_2_sift_feature_extraction(pipeline)
        
        # Stage 3: Feature Matching
        stage_3_exhaustive_matching(pipeline)
        
        # Stage 4: Sparse SfM Reconstruction
        stage_4_sparse_sfm_reconstruction(pipeline)
        
        # Stage 5: Dense MVS Reconstruction
        stage_5_dense_mvs_reconstruction(pipeline)
        
        # Stage 6: Point Cloud Post-processing
        stage_6_spatial_post_processing(pipeline)
        
        logging.info(f"SYSTEM SUCCESS: E2E Photogrammetry Pipeline resolved smoothly. "
                     f"Final volumetric asset archived at: {pipeline.final_cloud_path}")
                     
    except Exception as e:
        logging.error(f"PIPELINE FATAL EXCEPTION: {str(e)}")
        sys.exit(1)

def main():
    workspace_path = "./workspace"
    source_captures_path = "../captures"
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
    if len(sys.argv) > 2:
        source_captures_path = sys.argv[2]
        
    run_pipeline(workspace_path, source_captures_path)

if __name__ == "__main__":
    main()
