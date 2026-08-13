import os
import glob
import cv2
import sys
import json
import numpy as np

# Import the existing robust SilhouetteExtractor
sys.path.append("tools")
from silhouette_extractor import SilhouetteExtractor

def main():
    input_dir = "captures_0726_clay_checkboard_64_calibrated"
    output_dir = "optimization_0813/masks"
    calib_json_path = os.path.join(input_dir, "calibration_results.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(calib_json_path):
        print(f"Error: {calib_json_path} not found.")
        return
        
    with open(calib_json_path, "r") as f:
        cal_data = json.load(f)
        
    K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float64)
    dist_cal = np.array(cal_data["distortion_coefficients"], dtype=np.float64)
    
    # 0726 specific parameters used in prepare_nerf_0726.py
    CENTER_2D = (2166, 1145)
    PLATE_RADIUS_PIXELS = 1100
    
    extractor = SilhouetteExtractor()
    
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")),
                         key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    print(f"Extracting robust Step 1 masks for {len(image_paths)} images...")
    
    for idx, img_path in enumerate(image_paths):
        out_mask_name = f"mask_{idx:02d}.png"
        out_mask_path = os.path.join(output_dir, out_mask_name)
        
        # In the 0726 run, we cached masks in the TSDF directory. We can reuse them to save 10+ minutes!
        cached_mask_path = os.path.join("captures_0726_clay_checkboard_64_tsdf/masks", out_mask_name)
        
        if os.path.exists(cached_mask_path):
            # Just copy the perfectly cached mask (it is already undistorted and isolated)
            mask = cv2.imread(cached_mask_path, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(out_mask_path, mask)
        else:
            print(f"Cache miss for {out_mask_name}, computing from scratch (this takes a while)...")
            raw_img = cv2.imread(img_path)
            img_undist = cv2.undistort(raw_img, K_cal, dist_cal, None, K_cal)
            mask = extractor.get_silhouette_mask(img_undist, CENTER_2D, PLATE_RADIUS_PIXELS)
            cv2.imwrite(out_mask_path, mask)
            
        if (idx + 1) % 10 == 0 or idx == len(image_paths) - 1:
            print(f"Processed {idx+1}/{len(image_paths)} masks")
            
    print("Step 1 Segmentation complete. All masks saved to optimization_0813/masks/")

if __name__ == "__main__":
    main()
