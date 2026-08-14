import os
import glob
import cv2
import numpy as np

def convert_to_mask(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
        
    # Check if the image is 4-channel (RGBA) or 1-channel (grayscale/alpha mask)
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
    elif len(img.shape) == 2:
        alpha = img
    else:
        # For RGB images with black background, convert to grayscale and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
    # Binarize the alpha channel
    _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological Operations from Phase 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = np.zeros_like(opened)
        final_mask[labels == largest_label] = 255
    else:
        final_mask = opened
        
    cv2.imwrite(img_path, final_mask)
    return True

def main():
    mask_dir = "/home/coding/github/morpho_scanner/optimization_0813_three_lobs/masks"
    
    image_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    # Exclude preview masks if they accidentally got here
    image_paths = [p for p in image_paths if "capture_" in os.path.basename(p)]
    print(f"Found {len(image_paths)} rembg outputs to convert.")
    
    for i, img_path in enumerate(image_paths):
        success = convert_to_mask(img_path)
        if success:
            if (i+1) % 10 == 0 or (i+1) == len(image_paths):
                print(f"Processed {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        else:
            print(f"Failed to load/process {os.path.basename(img_path)}")

if __name__ == "__main__":
    main()
