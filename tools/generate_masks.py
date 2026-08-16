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
    parser.add_argument("--no_filter_green", action="store_true", help="Disable green background filtering")
    parser.add_argument("--filter_blue", action="store_true", help="Enable blueish background filtering")
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
    
    # Radius in pixels: approx 74mm physical radius of the green plate
    # projected onto the image plane using the focal length and Z distance
    plate_radius_mm = 74.0
    plate_radius_pixels = int(round((plate_radius_mm / P_c[2]) * K[0,0]))
    
    print(f"Computed plate center: {center_2d}, radius: {plate_radius_pixels} px")
    
    os.makedirs(args.out_dir, exist_ok=True)
    extractor = SilhouetteExtractor(
        filter_green=not args.no_filter_green,
        filter_blue=args.filter_blue
    )
    
    images = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")), 
                   key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                   
    for i, img_path in enumerate(images):
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]
        out_path = os.path.join(args.out_dir, f"mask_{stem}.png")
        
        if os.path.exists(out_path):
            continue
            
        print(f"Processing {i+1}/{len(images)}: {basename}")
        
        # Load and undistort full res image
        raw_img = cv2.imread(img_path)
        img_undist = cv2.undistort(raw_img, K, dist, None, K)
        
        # Extract silhouette mask
        mask = extractor.get_silhouette_mask(img_undist, center_2d, plate_radius_pixels)
        
        cv2.imwrite(out_path, mask)

if __name__ == "__main__":
    main()
