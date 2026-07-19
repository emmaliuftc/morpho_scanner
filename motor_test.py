import time
import math
from dynamixel_sdk import *
import os

# Make these match the actual ID numbers.  
MOTOR = 1

# If you have more/fewer than 4 motors make sure to adjust this list
motors = [MOTOR]

# This identifies the USB port where the motor controller is attached
port = PortHandler('/dev/ttyUSB0')
# This object contains the methods for reading/writing
packet_handler = PacketHandler(2.0)

def setup():
    # Start up both handlers
    print("Opening USB port and establishing connection...\n")
    port.openPort()
    port.setBaudRate(57600)

    # Read the ID numbers from the motor memory to test connection.
    # Count how many successes to make sure that all are successes.
    print("Test reading from each motor:")
    read_success = 0
    while read_success < len(motors):
        for motor in motors:
            # ID is 1 byte, stored at memory address 7.
            motor_id, result, error = packet_handler.read1ByteTxRx(
                    port, motor, 7)
            if result != COMM_SUCCESS:
                print("Read result was not a success.  The SDK says:")
                print(f"{packet_handler.getTxRxResult(result)}") 
            elif error != 0:
                print("Error found in reading.  The SDK says:")
                print(f"{packet_handler.getRxPacketError(error)}")
            else:
                print(f"Initial connection to motor {motor_id} successful.")
                read_success += 1
        if read_success < len(motors):
            print("Not all motors succeeded.  Retrying in 1 second.\n\n")
            time.sleep(1)

# Set operating mode to extended position
def set_op_mode():
    packet_handler.write1ByteTxRx(port, MOTOR, 64, 0) # turn off torque
    packet_handler.write1ByteTxRx(port, MOTOR, 11, 4) # enable extended position control
    packet_handler.write1ByteTxRx(port, MOTOR, 64, 1) # turn on torque
    time.sleep(0.1)
   
def go_to_position(pos):
    packet_handler.write4ByteTxRx(port, MOTOR, 116, pos)

def drop(n: int):
    position = get_position()
    new_position = (position + n)
    go_to_position(new_position)

def get_position():
    position,_,_ = packet_handler.read4ByteTxRx(port, MOTOR, 132)
    return position

# setup()
# set_op_mode()
# drop()