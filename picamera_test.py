import cv2
import time
from picamera2 import Picamera2
from libcamera import controls

print("Initializing Picamera2...")
picam = Picamera2()

# Configure main stream to use high resolution for precise sharpness analysis
picam.preview_configuration.main.size = (4608, 2592)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.align()

picam.configure("preview")
picam.start()

picam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 6.0})
            
# Wait for the lens to mechanically settle
time.sleep(0.4)

print("Camera preview started. Press 'q' to exit.")

try:
    while True:
        
        image = picam.capture_array()
        cv2.imwrite("scanned_output.jpg", image)
        print("image saved!")
        time.sleep(0.5)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    picam.stop()