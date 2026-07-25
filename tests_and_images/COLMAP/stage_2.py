import os
import sys
import logging
import pycolmap

# Establish standard logging protocol for the pipeline execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the current script's directory is in python path to import stage_1
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stage_1 import LegoReconstructionPipeline

def stage_2_sift_feature_extraction(pipeline: LegoReconstructionPipeline):
    """
    Executes SIFT feature extraction on the masked images using PyCOLMAP.
    Forces the mathematical solver to share a single intrinsic camera model
    across all frames.
    """
    logging.info("Initiating Stage 2: SIFT Feature Extraction")
    
    # Configure SIFT mathematical parameters
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift.max_num_features = 8192  # Robust limit for rigid object photogrammetry
    
    # Configure Image Reader Options to enforce camera models and link the binary masks
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "SIMPLE_RADIAL"
    reader_options.mask_path = str(pipeline.mask_dir)
    
    # Execute feature extraction
    # CameraMode.SINGLE mandates that the bundle adjuster optimizes one unified intrinsic array
    pycolmap.extract_features(
        database_path=pipeline.db_path,
        image_path=pipeline.image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=extraction_options,
        reader_options=reader_options
    )
    
    # Instantiate database connection to verify extraction yields
    db = pycolmap.Database.open(str(pipeline.db_path))
    logging.info(f"Stage 2 Complete: Features securely written to database for {db.num_images()} images.")
    
    # Generate SIFT keypoint visualizations on the cleanly whitened images for all images
    logging.info("Generating SIFT keypoint visualizations for all images...")
    from PIL import Image, ImageDraw
    from pathlib import Path
    
    vis_dir = pipeline.workspace / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    images = db.read_all_images()
    for img_entry in images:
        img_id = img_entry.image_id
        img_name = img_entry.name
        
        # Read keypoints from database
        kpts = db.read_keypoints(img_id)
        
        # Load the whitened image
        img_path = pipeline.image_dir / img_name
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        for kp in kpts:
            x, y = kp[0], kp[1]
            r = 12
            draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=3)
            
        # Naming format: sift_keypoints_{id}_clean_white_bg.jpg
        base_name = Path(img_name).stem
        vis_name = f"sift_keypoints_{base_name}_clean_white_bg.jpg"
        img.save(vis_dir / vis_name)
        logging.info(f"Saved visualization: {vis_name} ({len(kpts)} keypoints)")


def main():
    workspace_path = "./workspace"
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
        
    print(f"Loading COLMAP workspace at: {os.path.abspath(workspace_path)}")
    # We set init_workspace=False to preserve the images and masks created in Stage 1
    pipeline = LegoReconstructionPipeline(workspace_path, init_workspace=False)
    
    # Make sure database.db is removed before extraction starts, to ensure a clean start
    if pipeline.db_path.exists():
        pipeline.db_path.unlink()
        logging.info("Cleaned existing COLMAP SQLite database for a fresh SIFT extraction.")
        
    try:
        stage_2_sift_feature_extraction(pipeline)
        print("Success! Stage 2 SIFT Feature Extraction completed.")
    except Exception as e:
        logging.error(f"Stage 2 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
