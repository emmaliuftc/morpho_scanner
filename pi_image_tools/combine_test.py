import cv2
import time
from picamera2 import Picamera2
from libcamera import controls
import motor_test
from dynamixel_sdk import *
import select


# Make these match the actual ID numbers.  
MOTOR = 1

# If you have more/fewer than 4 motors make sure to adjust this list
motors = [MOTOR]

# This identifies the USB port where the motor controller is attached
port = PortHandler('/dev/ttyUSB0')
# This object contains the methods for reading/writing
packet_handler = PacketHandler(2.0)

motor_test.setup()
motor_test.set_op_mode()


picam = Picamera2()

picam.preview_configuration.main.size = (4608, 2592)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.align()
picam.set_controls({"AfMode": controls.AfModeEnum.Auto})
# picam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 10.0})
# picam.set_controls({"AfMode": controls.AfModeEnum.Continuous})

picam.configure("preview")
picam.start()

print("Camera preview started. Press 'q' to exit.")

try:
    for i in range(32):
        motor_test.drop()
        time.sleep(2)
        image = picam.capture_array()
        cv2.imwrite(f"capture_{i}.jpg", image)
        print(f"{i} image added")
finally:
    picam.stop()