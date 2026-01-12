#!/usr/bin/env python3
"""
Press Enter to print current neck motor positions.
Press 'q' + Enter to quit.
"""

import os
import sys

# Dynamixel settings
DEVICENAME = os.environ.get('DYNAMIXEL_PORT', '/dev/ttyUSB0')
BAUDRATE = 57600
ID_YAW = 0
ID_PITCH = 1
ADDR_PRESENT_POSITION = 132
ADDR_HARDWARE_ERROR = 70

try:
    from dynamixel_sdk import PortHandler, PacketHandler
except ImportError:
    print("dynamixel_sdk not found. Install with: pip3 install dynamixel-sdk")
    sys.exit(1)

def main():
    port_handler = PortHandler(DEVICENAME)
    packet_handler = PacketHandler(2.0)
    
    if not port_handler.openPort():
        print(f"Failed to open port {DEVICENAME}")
        sys.exit(1)
    
    if not port_handler.setBaudRate(BAUDRATE):
        print(f"Failed to set baudrate {BAUDRATE}")
        sys.exit(1)
    
    print(f"Connected to {DEVICENAME}")
    print("Press Enter to read positions, 'q' + Enter to quit\n")
    
    while True:
        user_input = input()
        if user_input.lower() == 'q':
            break
        
        # Read positions
        yaw_pos, yaw_result, yaw_err = packet_handler.read4ByteTxRx(port_handler, ID_YAW, ADDR_PRESENT_POSITION)
        pitch_pos, pitch_result, pitch_err = packet_handler.read4ByteTxRx(port_handler, ID_PITCH, ADDR_PRESENT_POSITION)
        
        # Read hardware error status
        yaw_hw_err, _, _ = packet_handler.read1ByteTxRx(port_handler, ID_YAW, ADDR_HARDWARE_ERROR)
        pitch_hw_err, _, _ = packet_handler.read1ByteTxRx(port_handler, ID_PITCH, ADDR_HARDWARE_ERROR)
        
        print(f"YAW   (ID {ID_YAW}): pos={yaw_pos:5d}  result={yaw_result}  hw_err=0x{yaw_hw_err:02X}")
        print(f"PITCH (ID {ID_PITCH}): pos={pitch_pos:5d}  result={pitch_result}  hw_err=0x{pitch_hw_err:02X}")
        
        if yaw_hw_err != 0:
            print(f"  -> YAW has hardware error!")
        if pitch_hw_err != 0:
            print(f"  -> PITCH has hardware error!")
        print()
    
    port_handler.closePort()
    print("Done.")

if __name__ == "__main__":
    main()

