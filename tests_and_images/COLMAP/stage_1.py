import os
import shutil
import logging
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import rembg

# Establish standard logging protocol for the pipeline execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LegoReconstructionPipeline:
    """
    End-to-End photogrammetry orchestration pipeline managing the data flow 
    from raw tuning-plate images to a dense, filtered 3D point cloud.
    """
    def __init__(self, workspace_dir: str, init_workspace: bool = True):
        self.workspace = Path(workspace_dir)
        
        # Define strict structural paths mandated by the COLMAP architecture
        self.raw_dir = self.workspace / "raw_images"
        self.image_dir = self.workspace / "images"
        self.mask_dir = self.workspace / "masks"
        self.db_path = self.workspace / "database.db"
        self.sparse_dir = self.workspace / "sparse"
        self.dense_dir = self.workspace / "dense"
        self.mvs_dir = self.workspace / "mvs"
        self.final_cloud_path = self.workspace / "lego_block_final.ply"
        
        if init_workspace:
            self._initialize_workspace()

    def _initialize_workspace(self):
        """
        Creates the required directory tree for COLMAP execution and purges 
        legacy SQLite databases to ensure a clean geometric reconstruction graph.
        """
        directories = [self.raw_dir, self.image_dir, self.mask_dir, 
                       self.sparse_dir, self.dense_dir, self.mvs_dir]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Purge existing database to prevent foreign feature contamination
        if self.db_path.exists():
            self.db_path.unlink()
            logging.info("Purged existing COLMAP SQLite database.")


def stage_1_dynamic_masking(pipeline: LegoReconstructionPipeline):
    """
    Applies neural-network-driven background removal to generate binary masks.
    Enforces COLMAP's strict filename.ext.png naming convention to guarantee
    feature extraction ignores static room geometry.
    """
    logging.info("Initiating Stage 1: Neural Background Masking")
    
    # Initialize the rembg session (loads the U2-Net weights into system memory)
    session = rembg.new_session("u2net")
    
    raw_images = list(pipeline.raw_dir.glob("*.jpg")) + list(pipeline.raw_dir.glob("*.jpeg"))
    
    if len(raw_images) != 36:
        logging.warning(f"Target dataset size is 36 images. Detected {len(raw_images)}. Proceeding.")

    for img_path in raw_images:
        # Replicate the raw image into the active COLMAP target directory
        target_img_path = pipeline.image_dir / img_path.name
        shutil.copy2(img_path, target_img_path)
        
        # Ingest image into the rembg inference engine
        input_image = Image.open(img_path)
        
        # The engine outputs an RGBA image; transparent regions signify the background
        output_image = rembg.remove(input_image, session=session)
        
        # Extract the alpha channel array to serve as the foundational binary mask
        alpha_channel = output_image.split()[-1]
        mask_array = np.array(alpha_channel)
        
        # Binarization: Foreground pixels designated as 255 (white), Background as 0 (black)
        binary_mask = np.where(mask_array > 10, 255, 0).astype(np.uint8)
        mask_image = Image.fromarray(binary_mask)
        
        # Save mask enforcing the precise COLMAP naming convention: filename.ext.png
        mask_filename = f"{img_path.name}.png"
        mask_path = pipeline.mask_dir / mask_filename
        mask_image.save(mask_path)
        logging.info(f"Generated mask for {img_path.name} -> {mask_filename}")
        
    logging.info(f"Stage 1 Complete: {len(raw_images)} Binary masks generated successfully.")


def main():
    # Workspace directory: default is ./workspace (relative to execution path)
    # Source images directory: default is the project's captures/ folder
    workspace_path = "./workspace"
    source_captures_path = "../captures"
    
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
    if len(sys.argv) > 2:
        source_captures_path = sys.argv[2]
        
    print(f"Setting up COLMAP workspace at: {os.path.abspath(workspace_path)}")
    pipeline = LegoReconstructionPipeline(workspace_path)
    
    # Check if raw_images directory is empty
    existing_raw = list(pipeline.raw_dir.glob("*.jpg")) + list(pipeline.raw_dir.glob("*.jpeg"))
    if not existing_raw:
        if os.path.exists(source_captures_path):
            print(f"Copying raw images from {os.path.abspath(source_captures_path)} to {pipeline.raw_dir}")
            src_path = Path(source_captures_path)
            copied_count = 0
            for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
                for img_file in src_path.glob(ext):
                    shutil.copy2(img_file, pipeline.raw_dir / img_file.name)
                    copied_count += 1
            print(f"Copied {copied_count} files.")
        else:
            print(f"Warning: Source captures directory '{source_captures_path}' not found.")
            print(f"Please place your raw images in '{pipeline.raw_dir}' before running Stage 1.")
            
    # Execute Stage 1 masking
    try:
        stage_1_dynamic_masking(pipeline)
        print("Success! Setup and Stage 1 completed.")
    except Exception as e:
        logging.error(f"Stage 1 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
