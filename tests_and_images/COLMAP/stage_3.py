import os
import sys
import logging
import pycolmap

# Establish standard logging protocol for the pipeline execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the current script's directory is in python path to import stage_1
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stage_1 import LegoReconstructionPipeline

def stage_3_exhaustive_matching(pipeline: LegoReconstructionPipeline):
    """
    Performs exhaustive O(N^2) feature matching and RANSAC-driven 
    geometric verification to establish epipolar constraints.
    """
    logging.info("Initiating Stage 3: Exhaustive Feature Matching")
    
    # Configure SIFT matching options
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.sift.max_ratio = 0.8  # Lowe's ratio test to reject ambiguous stud matches
    matching_options.sift.max_distance = 0.7  # Absolute descriptor distance threshold
    
    # Execute exhaustive matching
    pycolmap.match_exhaustive(
        database_path=pipeline.db_path,
        matching_options=matching_options
    )
    
    # Instantiate database connection to verify verification yields
    db = pycolmap.Database.open(str(pipeline.db_path))
    logging.info("Stage 3 Complete: Epipolar geometries verified and committed to database.")
    logging.info(f"Total matched image pairs: {db.num_matched_image_pairs()}")
    logging.info(f"Total verified image pairs: {db.num_verified_image_pairs()}")
    logging.info(f"Total raw matches: {db.num_matches()}")
    logging.info(f"Total verified inlier matches: {db.num_inlier_matches()}")


def main():
    workspace_path = "./workspace"
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
        
    print(f"Loading COLMAP workspace at: {os.path.abspath(workspace_path)}")
    # We set init_workspace=False to preserve SIFT features already extracted in Stage 2
    pipeline = LegoReconstructionPipeline(workspace_path, init_workspace=False)
    
    try:
        stage_3_exhaustive_matching(pipeline)
        print("Success! Stage 3 Exhaustive Feature Matching completed.")
    except Exception as e:
        logging.error(f"Stage 3 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
