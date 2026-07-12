import cv2
import time
import sys
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

def calculate_sharpness(image):
    """
    Computes the focus measure of an image using the Variance of Laplacian method.
    Higher values represent more high-frequency edges, meaning a sharper image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def run_calibration(start_pos=0.0, end_pos=20.0, step=0.5):
    print("Initializing Picamera2...")
    picam = Picamera2()
    
    # Configure main stream to use high resolution for precise sharpness analysis
    picam.preview_configuration.main.size = (4608, 2592)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    
    picam.configure("preview")
    picam.start()
    
    print("Camera preview started. Beginning focus sweep...")
    print(f"Sweeping lens positions from {start_pos} to {end_pos} (step: {step})")
    print("-" * 60)
    print(f"{'Lens Position':<15} | {'Sharpness Score':<20} | {'Status':<15}")
    print("-" * 60)
    
    best_score = -1.0
    best_position = start_pos
    best_image = None
    worst_image = None
    worst_score = float('inf')
    worst_position = start_pos
    
    sweep_range = np.arange(start_pos, end_pos + 0.001, step)
    
    try:
        for pos in sweep_range:
            pos_val = float(pos)
            # Set manual focus and specify the lens position
            picam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": pos_val})
            
            # Wait for the lens to mechanically settle
            time.sleep(0.4)
            
            # Capture the image array
            image = picam.capture_array()
            
            # Calculate sharpness
            score = calculate_sharpness(image)
            
            status = ""
            if score > best_score:
                best_score = score
                best_position = pos_val
                best_image = image.copy()
                status = "* NEW BEST *"
            if score < worst_score:
                worst_score = score
                worst_position = pos_val
                worst_image = image.copy()
                
            print(f"{pos_val:<15.2f} | {score:<20.4f} | {status:<15}")
            
        print("-" * 60)
        print("Focus sweep completed.")
        print(f"Optimal Lens Position: {best_position:.2f} (Sharpness Score: {best_score:.4f})")
        print(f"Worst Lens Position:   {worst_position:.2f} (Sharpness Score: {worst_score:.4f})")
        
        # Save verification images
        if best_image is not None:
            cv2.imwrite("calibration_best.jpg", best_image)
            print("Saved optimal focus test image to 'calibration_best.jpg'")
        if worst_image is not None:
            cv2.imwrite("calibration_worst.jpg", worst_image)
            print("Saved worst focus test image to 'calibration_worst.jpg'")
            
        print("\nTo lock this focus point in your scanner script, use:")
        print(f'picam.set_controls({{"AfMode": controls.AfModeEnum.Manual, "LensPosition": {best_position:.2f}}})')
        
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user.")
    finally:
        picam.stop()
        print("Camera preview stopped.")

if __name__ == "__main__":
    start = 0.0
    end = 15.0
    step = 0.5
    
    if len(sys.argv) > 1:
        start = float(sys.argv[1])
    if len(sys.argv) > 2:
        end = float(sys.argv[2])
    if len(sys.argv) > 3:
        step = float(sys.argv[3])
        
    run_calibration(start, end, step)
