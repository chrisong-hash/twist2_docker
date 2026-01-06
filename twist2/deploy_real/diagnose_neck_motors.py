#!/usr/bin/env python3
"""
Diagnose Dynamixel Neck Motors
Checks connection, torque, operating mode, and tests movement
"""

import sys
import time

try:
    from dynamixel_sdk import *
except ImportError:
    print("ERROR: pip install dynamixel-sdk")
    sys.exit(1)

# Configuration
PORT = '/dev/ttyUSB0'
BAUDRATE = 57600  # Default Dynamixel baud rate
PROTOCOL = 2.0

# Motor IDs
ID_YAW = 0
ID_PITCH = 1

# Control table addresses for X-series (XM430, XC330, XL330, etc.)
# These are the same for most Protocol 2.0 servos
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_LED = 65
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_MOVING = 122
ADDR_HARDWARE_ERROR = 70

# Operating modes
MODE_POSITION = 3
MODE_EXTENDED_POSITION = 4
MODE_VELOCITY = 1

def main():
    print("=" * 60)
    print("  Dynamixel Neck Motor Diagnostic")
    print("=" * 60)
    print(f"  Port: {PORT}")
    print(f"  Baudrate: {BAUDRATE}")
    print(f"  Motors: Yaw(ID={ID_YAW}), Pitch(ID={ID_PITCH})")
    print("=" * 60)
    print()
    
    # Initialize
    port_handler = PortHandler(PORT)
    packet_handler = PacketHandler(PROTOCOL)
    
    # Open port
    if not port_handler.openPort():
        print(f"❌ Failed to open port {PORT}")
        print("   Check: ls /dev/ttyUSB*")
        print("   Fix:   sudo chmod 777 /dev/ttyUSB0")
        return
    print(f"✓ Port opened: {PORT}")
    
    # Set baudrate
    if not port_handler.setBaudRate(BAUDRATE):
        print(f"❌ Failed to set baudrate {BAUDRATE}")
        return
    print(f"✓ Baudrate set: {BAUDRATE}")
    print()
    
    # Check each motor
    for motor_id, name in [(ID_YAW, "YAW"), (ID_PITCH, "PITCH")]:
        print(f"--- Motor {name} (ID={motor_id}) ---")
        
        # Ping motor
        model_number, result, error = packet_handler.ping(port_handler, motor_id)
        if result != COMM_SUCCESS:
            print(f"  ❌ Ping failed: {packet_handler.getTxRxResult(result)}")
            print(f"     Check motor ID in Dynamixel Wizard")
            continue
        print(f"  ✓ Ping OK - Model: {model_number}")
        
        # Check hardware error
        hw_error, result, error = packet_handler.read1ByteTxRx(port_handler, motor_id, ADDR_HARDWARE_ERROR)
        if result == COMM_SUCCESS:
            if hw_error != 0:
                print(f"  ⚠ Hardware Error: {hw_error}")
            else:
                print(f"  ✓ No hardware errors")
        
        # Check operating mode
        op_mode, result, error = packet_handler.read1ByteTxRx(port_handler, motor_id, ADDR_OPERATING_MODE)
        if result == COMM_SUCCESS:
            mode_names = {1: "Velocity", 3: "Position", 4: "Extended Position", 5: "Current-based Position"}
            mode_name = mode_names.get(op_mode, f"Unknown({op_mode})")
            print(f"  Operating Mode: {mode_name}")
            if op_mode not in [3, 4, 5]:
                print(f"  ⚠ Motor not in position mode! Set to mode 3 in Dynamixel Wizard")
        
        # Check torque enable
        torque, result, error = packet_handler.read1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE)
        if result == COMM_SUCCESS:
            print(f"  Torque Enable: {'ON' if torque else 'OFF'}")
        
        # Read current position
        pos, result, error = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
        if result == COMM_SUCCESS:
            print(f"  Current Position: {pos} (center=2048)")
        
        print()
    
    # Test movement
    print("--- Movement Test ---")
    print("This will try to move each motor slightly.")
    input("Press Enter to continue (or Ctrl+C to cancel)...")
    print()
    
    for motor_id, name in [(ID_YAW, "YAW"), (ID_PITCH, "PITCH")]:
        print(f"Testing {name} motor (ID={motor_id})...")
        
        # Disable torque first (required to change operating mode)
        packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 0)
        time.sleep(0.1)
        
        # Set position mode
        result, error = packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_OPERATING_MODE, MODE_POSITION)
        if result != COMM_SUCCESS:
            print(f"  ⚠ Failed to set position mode: {packet_handler.getTxRxResult(result)}")
        
        # Enable torque
        result, error = packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
        if result != COMM_SUCCESS:
            print(f"  ❌ Failed to enable torque: {packet_handler.getTxRxResult(result)}")
            continue
        print(f"  ✓ Torque enabled")
        
        # Blink LED to confirm communication
        packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_LED, 1)
        time.sleep(0.2)
        packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_LED, 0)
        print(f"  ✓ LED blinked (confirms communication)")
        
        # Read current position
        current_pos, _, _ = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
        print(f"  Current position: {current_pos}")
        
        # Move to center
        target = 2048
        print(f"  Moving to center ({target})...")
        result, error = packet_handler.write4ByteTxRx(port_handler, motor_id, ADDR_GOAL_POSITION, target)
        if result != COMM_SUCCESS:
            print(f"  ❌ Write failed: {packet_handler.getTxRxResult(result)}")
        else:
            print(f"  ✓ Command sent")
        
        time.sleep(1)
        
        # Check if it moved
        new_pos, _, _ = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
        print(f"  New position: {new_pos}")
        
        if abs(new_pos - current_pos) > 10:
            print(f"  ✓ Motor moved!")
        else:
            print(f"  ⚠ Motor did NOT move. Check:")
            print(f"     - Is motor physically able to move? (not blocked)")
            print(f"     - Is motor powered? (12V supply connected)")
            print(f"     - Check wiring and connections")
        
        print()
    
    # Cleanup
    packet_handler.write1ByteTxRx(port_handler, ID_YAW, ADDR_TORQUE_ENABLE, 0)
    packet_handler.write1ByteTxRx(port_handler, ID_PITCH, ADDR_TORQUE_ENABLE, 0)
    port_handler.closePort()
    
    print("Diagnostic complete!")
    print()
    print("If motors still don't move:")
    print("  1. Check 12V power supply to motors")
    print("  2. In Dynamixel Wizard: set Operating Mode = 3 (Position)")
    print("  3. In Dynamixel Wizard: manually test motor movement")


if __name__ == "__main__":
    main()

