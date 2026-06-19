import cv2
import time
from picamera2 import Picamera2
from libcamera import controls

picam = Picamera2()

picam.preview_configuration.main.size = (1280, 720)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.align()
picam.set_controls({"AfMode": controls.AfModeEnum.Continuous})

picam.configure("preview")
picam.start()

print("Camera preview started. Press 'q' to exit.")

try:
    while True:
        
        image = picam.capture_array()
        cv2.imwrite("scanned_output.jpg", image)
        print("image saved!")
        time.sleep(0.25)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    picam.stop()