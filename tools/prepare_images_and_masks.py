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
    parser.add_argument("--mask_dir", type=str, default=None, help="Optional directory containing pre-computed masks. If not provided, U2Net will be used.")
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
    
    if args.mask_dir is None:
        extractor = SilhouetteExtractor(
            filter_green=not args.no_filter_green,
            filter_blue=args.filter_blue
        )
    
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
        
        if args.mask_dir is not None:
            # Read the user's perfect precomputed mask (already downscaled and undistorted)
            pre_mask_path = os.path.join(args.mask_dir, f"mask_{stem}.png")
            if not os.path.exists(pre_mask_path):
                pre_mask_path = os.path.join(args.mask_dir, f"{stem}.png")
            if not os.path.exists(pre_mask_path):
                pre_mask_path = os.path.join(args.mask_dir, f"{stem}.jpg")
                
            mask_small = cv2.imread(pre_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_small is None:
                raise ValueError(f"Could not load pre-computed mask from: {pre_mask_path}")
        else:
            # Extract silhouette mask on the undistorted image using U2Net
            mask = extractor.get_silhouette_mask(img_undist, center_2d, plate_radius_pixels)
            # Downscale by the scale factor
            mask_small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
        # Downscale the undistorted raw image by the scale factor
        img_small = cv2.resize(img_undist, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Apply mask for visual preview (we also save an RGBA version for proper NeRF background randomization!)
        masked_img_small = cv2.bitwise_and(img_small, img_small, mask=mask_small)
        
        # Create an RGBA version where the mask is the alpha channel
        b, g, r = cv2.split(masked_img_small)
        rgba_img = cv2.merge((b, g, r, mask_small))
        
        # We must save as .png to preserve the alpha channel!
        out_masked_preview_png = out_masked_preview.replace('.jpg', '.png')
        
        cv2.imwrite(out_img, img_small)
        cv2.imwrite(out_mask, mask_small)
        cv2.imwrite(out_masked_preview_png, rgba_img)

if __name__ == "__main__":
    main()
