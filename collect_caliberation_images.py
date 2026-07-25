import cv2
import time
import os
from picamera2 import Picamera2
from libcamera import controls

# Define the folder name expected by the calibration script
OUTPUT_FOLDER = "calibration_images"

# Ensure the output directory exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created directory: {OUTPUT_FOLDER}")

print("Initializing Picamera2...")
picam = Picamera2()

# Configure main stream to use high resolution for precise sharpness analysis
picam.preview_configuration.main.size = (4608, 2592)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.align()

picam.configure("preview")
picam.start()

# Locking the focus is CRITICAL for camera calibration.
# Do not change this LensPosition after calibration!
picam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 6.0})
            
# Wait for the lens to mechanically settle
time.sleep(0.4)

print(f"Camera started. Auto-saving images to '{OUTPUT_FOLDER}'...")
print("Move the checkerboard around to different angles and corners.")
print("Press 'q' (if window is open) or Ctrl+C in terminal to exit.")

image_counter = 0

try:
    while True:
        # Capture the image frame
        image = picam.capture_array()
        
        # Generate an incremental filename with zero-padding (e.g., calib_img_001.jpg)
        filename = os.path.join(OUTPUT_FOLDER, f"calib_img_{image_counter:03d}.jpg")
        
        # Save the image
        cv2.imwrite(filename, image)
        print(f"[{image_counter}] Saved: {filename}")
        
        image_counter += 1
        
        # Give yourself time to move the checkerboard to a new position
        time.sleep(1.0) 
        
        # Note: cv2.waitKey() only catches keys if a cv2 window is currently active. 
        # If running headlessly via VSCode, you can stop it using Ctrl+C.
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    # Gracefully handle Ctrl+C stopping
    print("\nCapture manually stopped via terminal (Ctrl+C).")

finally:
    picam.stop()
    print(f"Camera shut down. Successfully captured {image_counter} calibration images.")
    print(f"You can now run 'calibrate_camera.py' to process the images in '{OUTPUT_FOLDER}'.")