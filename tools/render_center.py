import cv2
import numpy as np
import glob
import os
import imageio

# ==========================================
# USER CONFIGURATION
# ==========================================

# The folder where your 32 raw turntable images are stored
INPUT_FOLDER = "captures_7-25_lob_with_checkbox" 
# The folder where the marked images will be saved
OUTPUT_FOLDER = "captures_7-25_marked_center" 

# Choose how you want to define your center point. 
# Set to True if your CBA output gave you a 3D coordinate (X, Y, Z) in camera space.
# Set to False if your CBA just output the final 2D pixel coordinate (u, v).
USE_3D_PROJECTION = True 

# --- Option A: 2D Pixel Coordinate ---
# (e.g., The "Winning Center" from your previous screenshot)
CENTER_2D = (2210.0, 590.0) 

# --- Option B: 3D Camera Coordinate ---
# (If your CBA outputs the physical [X, Y, Z] of the center in the camera's reference frame)
CENTER_3D = np.array([2.10257324, -5.71812931, 181.21088812]) 
# Your calibrated Camera Matrix (K)
K = np.array([
    [3565.4767, 0.0,       2304.0],
    [0.0,       3565.4767, 1296.0],
    [0.0,       0.0,       1.0]
], dtype=np.float32)
# Your calibrated Distortion Coefficients (dist)
DIST = np.array([[-0.44031736, 7.83951075, 0.0, 0.0, -43.24147593]], dtype=np.float32)

def render_center_point():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: {OUTPUT_FOLDER}")

    # Grab all JPGs from the input folder
    image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")))
    
    if not image_paths:
        print(f"Error: No .jpg images found in '{INPUT_FOLDER}'.")
        return

    print(f"Found {len(image_paths)} images. Processing...")

    # Calculate the exact 2D pixel coordinate
    if USE_3D_PROJECTION:
        # We project the 3D point using K and dist. 
        # rvec and tvec are 0 because the 3D point is already relative to the camera.
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))
        projected_points, _ = cv2.projectPoints(CENTER_3D, rvec, tvec, K, DIST)
        
        # Extract the (u, v) coordinates
        target_pixel = (int(projected_points[0][0][0]), int(projected_points[0][0][1]))
        print(f"Projected 3D point {CENTER_3D} to 2D pixel: {target_pixel}")
    else:
        # Use the hardcoded 2D coordinate
        target_pixel = (int(CENTER_2D[0]), int(CENTER_2D[1]))
        print(f"Using provided 2D pixel: {target_pixel}")

    # Prepare the Video Writer using imageio (with libx264 for Chrome compatibility)
    video_path = os.path.join(OUTPUT_FOLDER, "rotation_center_check.mp4")
    video_writer = imageio.get_writer(video_path, fps=10.0, codec='libx264')

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        
        # 1. Draw a prominent red cross/star marker at the center point
        cv2.drawMarker(
            img, 
            target_pixel, 
            color=(0, 0, 255), # Red in BGR
            markerType=cv2.MARKER_STAR, 
            markerSize=40, 
            thickness=3, 
            line_type=cv2.LINE_AA
        )
        
        # 2. Add an inner dot for sub-pixel precision viewing
        cv2.circle(img, target_pixel, radius=4, color=(0, 255, 255), thickness=-1)

        # 3. Add text overlay with coordinates
        text = f"Calculated Center: ({target_pixel[0]}, {target_pixel[1]})"
        cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(img, f"Frame: {i+1}/32", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        # Save the marked image
        filename = os.path.basename(path)
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(out_path, img)
        
        # Write frame to the video (converting BGR to RGB for imageio)
        video_writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        print(f"Processed frame {i+1}: {filename}")

    video_writer.close()
    print(f"\nSuccess! Marked images saved to '{OUTPUT_FOLDER}'.")
    print(f"A validation video was also saved as '{video_path}'. Play this video to visually verify the center point!")

if __name__ == "__main__":
    render_center_point()
