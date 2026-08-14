import os
import glob
import cv2
import json
import numpy as np
import argparse
from silhouette_extractor import SilhouetteExtractor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    with open(args.calib, "r") as f:
        calib = json.load(f)

    # Compute 2D center by projecting 3D plate center
    K = np.array(calib['camera_matrix_K'])
    P_c = np.array(calib['plate_center_mm'])
    dist = np.array(calib['distortion_coefficients'])
    
    uv_homog = K @ P_c
    u = uv_homog[0] / uv_homog[2]
    v = uv_homog[1] / uv_homog[2]
    center_2d = (int(round(u)), int(round(v)))
    
    # Radius in pixels
    plate_radius_mm = 74.0
    plate_radius_pixels = int(round((plate_radius_mm / P_c[2]) * K[0,0]))
    
    print(f"Computed plate center: {center_2d}, radius: {plate_radius_pixels} px")
    
    images_out = os.path.join(args.out_dir, f"images_{args.scale}")
    masks_out = os.path.join(args.out_dir, f"masks_{args.scale}")
    masked_images_out = os.path.join(args.out_dir, f"masked_images_preview_{args.scale}")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)
    os.makedirs(masked_images_out, exist_ok=True)
    
    extractor = SilhouetteExtractor()
    
    images = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")), 
                   key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                   
    orig_w, orig_h = 4608, 2592
    new_w, new_h = orig_w // args.scale, orig_h // args.scale
                   
    for i, img_path in enumerate(images):
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]
        
        out_img = os.path.join(images_out, f"{stem}.png")
        out_mask = os.path.join(masks_out, f"mask_{stem}.png")
        out_masked_preview = os.path.join(masked_images_out, f"preview_{stem}.jpg")
        
        if os.path.exists(out_img) and os.path.exists(out_mask) and os.path.exists(out_masked_preview):
            continue
            
        print(f"Processing {i+1}/{len(images)}: {basename}")
        
        # Load and undistort full res image using all OpenCV coefficients (crucial for k3!)
        raw_img = cv2.imread(img_path)
        img_undist = cv2.undistort(raw_img, K, dist, None, K)
        
        # Extract silhouette mask on the undistorted image
        mask = extractor.get_silhouette_mask(img_undist, center_2d, plate_radius_pixels)
        
        # Downscale by the scale factor
        img_small = cv2.resize(img_undist, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Apply mask for visual preview
        masked_img_small = cv2.bitwise_and(img_small, img_small, mask=mask_small)
        
        cv2.imwrite(out_img, img_small)
        cv2.imwrite(out_mask, mask_small)
        cv2.imwrite(out_masked_preview, masked_img_small)

if __name__ == "__main__":
    main()
