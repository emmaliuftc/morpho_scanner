import os
import sys
import logging
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pycolmap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_keypoints(workspace_dir: str, num_images_to_visualize: int = 3):
    workspace = Path(workspace_dir)
    db_path = workspace / "database.db"
    image_dir = workspace / "images"
    output_dir = workspace / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    if not db_path.exists():
        logging.error(f"Database not found at {db_path}. Have you run Stage 2?")
        sys.exit(1)
        
    # Open the COLMAP database
    db = pycolmap.Database.open(str(db_path))
    images = db.read_all_images()
    
    if not images:
        logging.error("No images found in the database. SIFT features may not be extracted yet.")
        sys.exit(1)
        
    logging.info(f"Found {len(images)} images in the database. Visualizing first {min(num_images_to_visualize, len(images))} images...")
    
    for i, img_entry in enumerate(images[:num_images_to_visualize]):
        img_id = img_entry.image_id
        img_name = img_entry.name
        img_path = image_dir / img_name
        
        if not img_path.exists():
            logging.warning(f"Image file not found at {img_path}, skipping.")
            continue
            
        # Read keypoints from the database
        kpts = db.read_keypoints(img_id)
        num_kpts = len(kpts)
        logging.info(f"Image '{img_name}' (ID: {img_id}): found {num_kpts} SIFT keypoints.")
        
        # Load the image using PIL
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Draw each keypoint
        # kpts shape is (N, 6). Typically: columns 0 and 1 are X and Y.
        for kp in kpts:
            x, y = kp[0], kp[1]
            r = 12  # Radius of the circle to make it visible on high-res images
            draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=3)
            
        # Save the annotated image
        output_path = output_dir / f"sift_keypoints_{img_name}"
        img.save(output_path)
        logging.info(f"Saved visualization to {output_path}")

def main():
    workspace_path = "./workspace"
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
        
    visualize_keypoints(workspace_path)

if __name__ == "__main__":
    main()
