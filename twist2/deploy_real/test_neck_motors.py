#!/usr/bin/env python3
"""
Simple Neck Motor Test Script
Tests Dynamixel motors for TWIST2 neck (yaw ID=0, pitch ID=1)

Usage:
    python test_neck_motors.py              # Basic test (move to positions)
    python test_neck_motors.py --redis      # Read from Redis (for full pipeline test)
    python test_neck_motors.py --interactive  # Interactive control with keyboard
"""

import argparse
import time
import math

try:
    from dynamixel_sdk import *
except ImportError:
    print("ERROR: dynamixel-sdk not installed!")
    print("Install with: pip install dynamixel-sdk")
    exit(1)

# =============================================
# CONFIGURATION - Adjust these for your setup
# =============================================
DEVICENAME = '/dev/ttyUSB0'  # USB port
BAUDRATE = 57600             # Default Dynamixel baud rate
PROTOCOL_VERSION = 2.0       # Protocol 2.0 for X-series

# Motor IDs
ID_YAW = 0    # Neck yaw motor
ID_PITCH = 1  # Neck pitch motor

# Control table addresses (for X-series like XM430, XL330, etc.)
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_OPERATING_MODE = 11

# Position limits (adjust based on your mechanical limits)
# Default: 0-4095 for full 360° (2048 = center)
DXL_CENTER = 2048
DXL_MIN = 1024   # -90° from center
DXL_MAX = 3072   # +90° from center


class NeckController:
    def __init__(self, port=DEVICENAME, baudrate=BAUDRATE):
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        
        # Open port
        if not self.port_handler.openPort():
            raise Exception(f"Failed to open port {port}")
        print(f"✓ Opened port: {port}")
        
        # Set baudrate
        if not self.port_handler.setBaudRate(baudrate):
            raise Exception(f"Failed to set baudrate {baudrate}")
        print(f"✓ Set baudrate: {baudrate}")
        
        # Enable torque on both motors
        self._enable_torque(ID_YAW)
        self._enable_torque(ID_PITCH)
        print("✓ Torque enabled on both motors")
    
    def _enable_torque(self, motor_id):
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
        if result != COMM_SUCCESS:
            print(f"⚠ Motor {motor_id}: {self.packet_handler.getTxRxResult(result)}")
        elif error != 0:
            print(f"⚠ Motor {motor_id}: {self.packet_handler.getRxPacketError(error)}")
    
    def _disable_torque(self, motor_id):
        self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 0)
    
    def set_position(self, motor_id, position):
        """Set motor position (0-4095, 2048=center)"""
        position = int(max(DXL_MIN, min(DXL_MAX, position)))
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, motor_id, ADDR_GOAL_POSITION, position)
        return result == COMM_SUCCESS
    
    def get_position(self, motor_id):
        """Get current motor position"""
        position, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, motor_id, ADDR_PRESENT_POSITION)
        if result == COMM_SUCCESS:
            return position
        return None
    
    def set_neck_angles(self, yaw_rad, pitch_rad):
        """
        Set neck angles in radians
        yaw: positive = turn left, negative = turn right
        pitch: positive = look up, negative = look down
        """
        # Convert radians to Dynamixel position (roughly 0.088° per unit)
        yaw_pos = DXL_CENTER + int(yaw_rad * 180 / math.pi / 0.088)
        pitch_pos = DXL_CENTER + int(pitch_rad * 180 / math.pi / 0.088)
        
        self.set_position(ID_YAW, yaw_pos)
        self.set_position(ID_PITCH, pitch_pos)
    
    def center(self):
        """Move both motors to center position"""
        self.set_position(ID_YAW, DXL_CENTER)
        self.set_position(ID_PITCH, DXL_CENTER)
    
    def close(self):
        """Disable torque and close port"""
        self._disable_torque(ID_YAW)
        self._disable_torque(ID_PITCH)
        self.port_handler.closePort()
        print("✓ Motors disabled, port closed")


def test_basic_motion(neck):
    """Basic test: move to various positions"""
    print("\n=== Basic Motion Test ===")
    
    print("Moving to center...")
    neck.center()
    time.sleep(1)
    
    print("Testing YAW (left/right)...")
    neck.set_position(ID_YAW, DXL_CENTER + 500)  # Turn left
    time.sleep(0.5)
    neck.set_position(ID_YAW, DXL_CENTER - 500)  # Turn right
    time.sleep(0.5)
    neck.set_position(ID_YAW, DXL_CENTER)        # Center
    time.sleep(0.5)
    
    print("Testing PITCH (up/down)...")
    neck.set_position(ID_PITCH, DXL_CENTER + 300)  # Look up
    time.sleep(0.5)
    neck.set_position(ID_PITCH, DXL_CENTER - 300)  # Look down
    time.sleep(0.5)
    neck.set_position(ID_PITCH, DXL_CENTER)        # Center
    time.sleep(0.5)
    
    print("Testing combined motion...")
    neck.set_neck_angles(0.3, 0.2)   # Look up-left
    time.sleep(0.5)
    neck.set_neck_angles(-0.3, -0.2) # Look down-right
    time.sleep(0.5)
    neck.center()
    
    print("✓ Basic motion test complete!")


def test_redis_control(neck):
    """Read neck commands from Redis and execute them"""
    print("\n=== Redis Control Test ===")
    print("Reading neck commands from Redis key: action_neck_unitree_g1_with_hands")
    print("Run your teleop script in another terminal to send commands")
    print("Press Ctrl+C to stop\n")
    
    try:
        import redis
        import json
    except ImportError:
        print("ERROR: redis not installed! pip install redis")
        return
    
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    try:
        r.ping()
        print("✓ Connected to Redis")
    except:
        print("ERROR: Cannot connect to Redis. Is redis-server running?")
        return
    
    try:
        while True:
            neck_data_str = r.get("action_neck_unitree_g1_with_hands")
            if neck_data_str:
                try:
                    neck_data = json.loads(neck_data_str)
                    yaw, pitch = neck_data[0], neck_data[1]
                    neck.set_neck_angles(yaw, pitch)
                    print(f"\rYaw: {math.degrees(yaw):+6.1f}°  Pitch: {math.degrees(pitch):+6.1f}°", end="")
                except Exception as e:
                    print(f"\rError parsing: {e}", end="")
            time.sleep(0.02)  # 50Hz
    except KeyboardInterrupt:
        print("\n\nStopping Redis control...")
        neck.center()


def test_interactive(neck):
    """Interactive keyboard control"""
    print("\n=== Interactive Control ===")
    print("Controls:")
    print("  w/s : Pitch up/down")
    print("  a/d : Yaw left/right")
    print("  c   : Center")
    print("  q   : Quit")
    print()
    
    import sys
    import tty
    import termios
    
    yaw_pos = DXL_CENTER
    pitch_pos = DXL_CENTER
    step = 100
    
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            ch = sys.stdin.read(1)
            if ch == 'q':
                break
            elif ch == 'w':
                pitch_pos = min(DXL_MAX, pitch_pos + step)
            elif ch == 's':
                pitch_pos = max(DXL_MIN, pitch_pos - step)
            elif ch == 'a':
                yaw_pos = min(DXL_MAX, yaw_pos + step)
            elif ch == 'd':
                yaw_pos = max(DXL_MIN, yaw_pos - step)
            elif ch == 'c':
                yaw_pos = DXL_CENTER
                pitch_pos = DXL_CENTER
            
            neck.set_position(ID_YAW, yaw_pos)
            neck.set_position(ID_PITCH, pitch_pos)
            
            yaw_deg = (yaw_pos - DXL_CENTER) * 0.088
            pitch_deg = (pitch_pos - DXL_CENTER) * 0.088
            sys.stdout.write(f"\rYaw: {yaw_deg:+6.1f}°  Pitch: {pitch_deg:+6.1f}°  ")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\n")


def main():
    parser = argparse.ArgumentParser(description='Test TWIST2 neck motors')
    parser.add_argument('--port', default=DEVICENAME, help='Serial port')
    parser.add_argument('--baudrate', type=int, default=BAUDRATE, help='Baudrate')
    parser.add_argument('--redis', action='store_true', help='Read from Redis')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive keyboard control')
    args = parser.parse_args()
    
    print("=" * 50)
    print("  TWIST2 Neck Motor Test")
    print("=" * 50)
    print(f"Port: {args.port}")
    print(f"Baudrate: {args.baudrate}")
    print(f"Yaw Motor ID: {ID_YAW}")
    print(f"Pitch Motor ID: {ID_PITCH}")
    print()
    
    try:
        neck = NeckController(args.port, args.baudrate)
        
        if args.redis:
            test_redis_control(neck)
        elif args.interactive:
            test_interactive(neck)
        else:
            test_basic_motion(neck)
        
        neck.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Check USB connection: ls {args.port}")
        print(f"  2. Grant permission: sudo chmod 777 {args.port}")
        print(f"  3. Verify motor IDs in Dynamixel Wizard (Yaw={ID_YAW}, Pitch={ID_PITCH})")
        print(f"  4. Verify baudrate matches Dynamixel Wizard setting ({args.baudrate})")


if __name__ == "__main__":
    main()

