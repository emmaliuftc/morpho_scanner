import os
import sys
import logging
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pycolmap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_matches(workspace_dir: str):
    workspace = Path(workspace_dir)
    db_path = workspace / "database.db"
    image_dir = workspace / "images"
    output_dir = workspace / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    if not db_path.exists():
        logging.error(f"Database not found at {db_path}. Have you run Stage 3?")
        sys.exit(1)
        
    db = pycolmap.Database.open(str(db_path))
    images = db.read_all_images()
    
    if len(images) < 2:
        logging.error("At least 2 images are required in the database to visualize matches.")
        sys.exit(1)
        
    # Find a pair of images that has a good number of verified inlier matches
    best_pair = None
    max_inliers = 0
    
    # We will search the first few image combinations to find a good pair
    search_limit = min(len(images), 10)
    for i in range(search_limit):
        for j in range(i + 1, search_limit):
            id1, id2 = images[i].image_id, images[j].image_id
            if db.exists_two_view_geometry(id1, id2):
                geom = db.read_two_view_geometry(id1, id2)
                num_inliers = geom.inlier_matches.shape[0]
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_pair = (images[i], images[j], geom.inlier_matches)
                    
    if not best_pair or max_inliers == 0:
        logging.error("No image pairs with verified inlier matches were found. Have you run Stage 3 successfully?")
        sys.exit(1)
        
    img_a, img_b, inliers = best_pair
    id_a, id_b = img_a.image_id, img_b.image_id
    name_a, name_b = img_a.name, img_b.name
    
    logging.info(f"Visualizing matches for pair: {name_a} (ID: {id_a}) <-> {name_b} (ID: {id_b})")
    logging.info(f"Found {max_inliers} verified inlier matches.")
    
    path_a = image_dir / name_a
    path_b = image_dir / name_b
    
    if not path_a.exists() or not path_b.exists():
        logging.error(f"One or both of the image files ({name_a}, {name_b}) is missing from {image_dir}.")
        sys.exit(1)
        
    # Load images
    pil_a = Image.open(path_a).convert("RGB")
    pil_b = Image.open(path_b).convert("RGB")
    
    w_a, h_a = pil_a.size
    w_b, h_b = pil_b.size
    
    # Create side-by-side canvas
    canvas_w = w_a + w_b
    canvas_h = max(h_a, h_b)
    canvas = Image.new("RGB", (canvas_w, canvas_h))
    canvas.paste(pil_a, (0, 0))
    canvas.paste(pil_b, (w_a, 0))
    
    draw = ImageDraw.Draw(canvas)
    
    # Read keypoints
    kpts_a = db.read_keypoints(id_a)
    kpts_b = db.read_keypoints(id_b)
    
    # Set a seed to make visualization reproducible, but choose colors nicely
    random.seed(42)
    
    # Draw matched lines and keypoint circles
    # To avoid cluttering, we can draw a subset if there are too many, but let's draw them all with transparency or thin lines
    for idx_a, idx_b in inliers:
        pt_a = (kpts_a[idx_a, 0], kpts_a[idx_a, 1])
        pt_b = (kpts_b[idx_b, 0] + w_a, kpts_b[idx_b, 1])
        
        # Pick a random vibrant color
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        
        # Draw line
        draw.line([pt_a, pt_b], fill=color, width=4)
        
        # Draw keypoints on both images
        r = 10
        draw.ellipse((pt_a[0] - r, pt_a[1] - r, pt_a[0] + r, pt_a[1] + r), outline=color, width=3)
        draw.ellipse((pt_b[0] - r, pt_b[1] - r, pt_b[0] + r, pt_b[1] + r), outline=color, width=3)
        
    # Save the output
    output_filename = f"matches_{Path(name_a).stem}_vs_{Path(name_b).stem}.jpg"
    output_path = output_dir / output_filename
    canvas.save(output_path)
    
    logging.info(f"Saved matches visualization to {output_path}")

def main():
    workspace_path = "./workspace"
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
        
    visualize_matches(workspace_path)

if __name__ == "__main__":
    main()
