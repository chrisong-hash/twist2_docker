#!/usr/bin/env python3
"""
Neck Motor Test Script
======================
Tests Dynamixel neck motors (yaw + pitch) to verify they can be commanded.

Run on robot:
  python3 test_neck.py

Options:
  --device /dev/ttyUSB0   Serial device (default: /dev/ttyUSB0)
  --yaw_center 1338       Yaw motor center position
  --pitch_center 695      Pitch motor center position
  --no_move               Only read positions, don't move motors
"""

import argparse
import time
import math
import sys

# Dynamixel configuration (same as robot_peripherals.py)
BAUDRATE = 57600
ID_YAW = 0
ID_PITCH = 1
PROTOCOL_VERSION = 2.0

# Register addresses
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112
ADDR_HARDWARE_ERROR = 70

# Default center positions (calibrated 2026-01-14)
DEFAULT_YAW_CENTER = 1122
DEFAULT_PITCH_CENTER = 1685

# Movement range from center (in position units, ~0.088 deg each)
YAW_RANGE = 792    # ±792 from center (~70 degrees each way, symmetric)
PITCH_RANGE = 500  # ±500 from center (~44 degrees each way)

# Absolute physical limits (safety clamps)
YAW_ABS_MIN = 330
YAW_ABS_MAX = 1970
PITCH_ABS_MIN = 1630
PITCH_ABS_MAX = 2667


def check_motor_error(packet_handler, port_handler, motor_id, name):
    """Check and print hardware error status"""
    hw_err, result, _ = packet_handler.read1ByteTxRx(port_handler, motor_id, ADDR_HARDWARE_ERROR)
    if result == 0:
        if hw_err != 0:
            errors = []
            if hw_err & 0x01: errors.append("Input Voltage")
            if hw_err & 0x04: errors.append("Overheating")
            if hw_err & 0x08: errors.append("Motor Encoder")
            if hw_err & 0x10: errors.append("Electrical Shock")
            if hw_err & 0x20: errors.append("Overload")
            print(f"  ⚠️  {name} HARDWARE ERROR: 0x{hw_err:02X} ({', '.join(errors)})")
            return False
        return True
    else:
        print(f"  ❌ {name}: Cannot read error status")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test Dynamixel neck motors')
    parser.add_argument('--device', default='/dev/ttyUSB0', help='Serial device')
    parser.add_argument('--yaw_center', type=int, default=DEFAULT_YAW_CENTER, help='Yaw center position')
    parser.add_argument('--pitch_center', type=int, default=DEFAULT_PITCH_CENTER, help='Pitch center position')
    parser.add_argument('--no_move', action='store_true', help='Only read, do not move motors')
    args = parser.parse_args()

    print("=" * 50)
    print("🔧 Neck Motor Test")
    print("=" * 50)
    print(f"  Device:       {args.device}")
    print(f"  Yaw center:   {args.yaw_center}")
    print(f"  Pitch center: {args.pitch_center}")
    print(f"  Mode:         {'READ ONLY' if args.no_move else 'READ + MOVE'}")
    print("=" * 50)
    print()

    # Import Dynamixel SDK
    try:
        from dynamixel_sdk import PortHandler, PacketHandler
        print("✓ Dynamixel SDK imported")
    except ImportError:
        print("❌ dynamixel_sdk not installed!")
        print("   Install with: pip3 install dynamixel-sdk")
        sys.exit(1)

    # Open port
    port_handler = PortHandler(args.device)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        print(f"❌ Failed to open {args.device}")
        print("   Check: ls -la /dev/ttyUSB*")
        sys.exit(1)
    print(f"✓ Opened {args.device}")

    if not port_handler.setBaudRate(BAUDRATE):
        print(f"❌ Failed to set baudrate {BAUDRATE}")
        sys.exit(1)
    print(f"✓ Baudrate set to {BAUDRATE}")

    # Calculate dynamic limits from center
    yaw_min = args.yaw_center - YAW_RANGE
    yaw_max = args.yaw_center + YAW_RANGE
    pitch_min = args.pitch_center - PITCH_RANGE
    pitch_max = args.pitch_center + PITCH_RANGE

    # Test each motor
    motors = [
        (ID_YAW, "YAW", args.yaw_center, yaw_min, yaw_max),
        (ID_PITCH, "PITCH", args.pitch_center, pitch_min, pitch_max),
    ]

    print()
    print("-" * 50)
    print("Testing motors...")
    print("-" * 50)

    all_ok = True
    for motor_id, name, center, pos_min, pos_max in motors:
        print(f"\n[{name}] Motor ID {motor_id}")
        
        # Try to reboot motor to clear errors
        print(f"  Rebooting motor...")
        packet_handler.reboot(port_handler, motor_id)
        time.sleep(0.5)
        
        # Check for hardware errors
        if not check_motor_error(packet_handler, port_handler, motor_id, name):
            all_ok = False
        
        # Read current position
        pos, result, err = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
        if result == 0:
            print(f"  ✓ Current position: {pos}")
            print(f"    Range: [{pos_min}, {pos_max}], Center: {center}")
            if pos < pos_min or pos > pos_max:
                print(f"  ⚠️  Position out of expected range!")
        else:
            print(f"  ❌ Failed to read position (error={result})")
            all_ok = False
            continue

        if args.no_move:
            continue

        # Configure motor
        print(f"  Configuring motor...")
        
        # Set operating mode to position control
        result, err = packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_OPERATING_MODE, 3)
        if result != 0:
            print(f"  ❌ Failed to set operating mode")
            all_ok = False
            continue
        
        # Set profile velocity (speed)
        packet_handler.write4ByteTxRx(port_handler, motor_id, ADDR_PROFILE_VELOCITY, 50)
        
        # Enable torque
        result, err = packet_handler.write1ByteTxRx(port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
        if result != 0:
            print(f"  ❌ Failed to enable torque")
            all_ok = False
            continue
        print(f"  ✓ Torque enabled")

    if args.no_move:
        print()
        print("=" * 50)
        print("Read-only test complete!")
        print("=" * 50)
        port_handler.closePort()
        sys.exit(0 if all_ok else 1)

    # Movement test
    print()
    print("-" * 50)
    print("Movement test...")
    print("-" * 50)

    def move_to(yaw_pos, pitch_pos, description):
        print(f"\n  → {description}")
        print(f"    YAW: {yaw_pos}, PITCH: {pitch_pos}")
        
        # Clamp positions to dynamic limits
        yaw_pos = max(yaw_min, min(yaw_max, yaw_pos))
        pitch_pos = max(pitch_min, min(pitch_max, pitch_pos))
        
        # Apply absolute physical limits (safety)
        yaw_pos = max(YAW_ABS_MIN, min(YAW_ABS_MAX, yaw_pos))
        pitch_pos = max(PITCH_ABS_MIN, min(PITCH_ABS_MAX, pitch_pos))
        
        packet_handler.write4ByteTxRx(port_handler, ID_YAW, ADDR_GOAL_POSITION, yaw_pos)
        packet_handler.write4ByteTxRx(port_handler, ID_PITCH, ADDR_GOAL_POSITION, pitch_pos)
        time.sleep(1.5)  # Increased wait time for slower motors
        
        # Read actual positions
        yaw_actual, _, _ = packet_handler.read4ByteTxRx(port_handler, ID_YAW, ADDR_PRESENT_POSITION)
        pitch_actual, _, _ = packet_handler.read4ByteTxRx(port_handler, ID_PITCH, ADDR_PRESENT_POSITION)
        
        yaw_err = abs(yaw_actual - yaw_pos)
        pitch_err = abs(pitch_actual - pitch_pos)
        
        if yaw_err < 100 and pitch_err < 100:
            print(f"    ✓ Reached target (error: yaw={yaw_err}, pitch={pitch_err})")
            return True
        else:
            print(f"    ⚠️  Position error large (yaw={yaw_err}, pitch={pitch_err})")
            return False

    # Movement sequence
    test_offset = 200  # Position units to move (about 17 degrees)
    
    movements = [
        (args.yaw_center, args.pitch_center, "CENTER"),
        (args.yaw_center + test_offset, args.pitch_center, "YAW RIGHT"),
        (args.yaw_center - test_offset, args.pitch_center, "YAW LEFT"),
        (args.yaw_center, args.pitch_center, "CENTER"),
        (args.yaw_center, args.pitch_center + test_offset, "PITCH DOWN"),
        (args.yaw_center, args.pitch_center - test_offset, "PITCH UP"),
        (args.yaw_center, args.pitch_center, "CENTER (final)"),
    ]

    move_ok = True
    for yaw, pitch, desc in movements:
        if not move_to(yaw, pitch, desc):
            move_ok = False

    # Disable torque
    print()
    print("Disabling torque...")
    packet_handler.write1ByteTxRx(port_handler, ID_YAW, ADDR_TORQUE_ENABLE, 0)
    packet_handler.write1ByteTxRx(port_handler, ID_PITCH, ADDR_TORQUE_ENABLE, 0)

    port_handler.closePort()

    # Summary
    print()
    print("=" * 50)
    if all_ok and move_ok:
        print("✅ All tests PASSED!")
        print("   Neck motors are working correctly.")
    else:
        print("⚠️  Some tests had issues.")
        print("   Check the output above for details.")
    print("=" * 50)

    sys.exit(0 if (all_ok and move_ok) else 1)


if __name__ == "__main__":
    main()

