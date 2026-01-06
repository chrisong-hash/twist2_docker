#!/usr/bin/env python3
"""
Local Neck Controller for TWIST2
================================
Reads neck commands from Redis and sends to Dynamixel motors.
This is the local version of what runs on G1 via docker_neck.sh.

Usage:
    python neck_controller_local.py
    python neck_controller_local.py --port /dev/ttyUSB1

Works with:
    - teleop.sh
    - hybrid_teleop.sh  
    - Any script that sends to Redis key: action_neck_unitree_g1_with_hands
"""

import argparse
import time
import math
import json
import signal
import sys

try:
    from dynamixel_sdk import *
except ImportError:
    print("ERROR: dynamixel-sdk not installed!")
    print("Install with: pip install dynamixel-sdk")
    sys.exit(1)

try:
    import redis
except ImportError:
    print("ERROR: redis not installed!")
    print("Install with: pip install redis")
    sys.exit(1)

# =============================================
# CONFIGURATION
# =============================================
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 57600             # Default Dynamixel baud rate
PROTOCOL_VERSION = 2.0

# Motor IDs (set in Dynamixel Wizard)
ID_YAW = 0
ID_PITCH = 1

# Control table addresses (X-series)
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

# Position conversion
# XC330: 4096 steps per 360° → 0.088° per step
DXL_STEPS_PER_DEG = 4096 / 360  # ~11.38 steps per degree
POSITION_SCALE = 180 / math.pi * DXL_STEPS_PER_DEG  # radians to Dynamixel units

# =============================================
# CALIBRATION - Adjust these for your neck!
# =============================================
# These are the Dynamixel positions when the neck is at its physical center
# Use Dynamixel Wizard to find these values:
#   1. Manually move neck to center position
#   2. Read the "Present Position" value
#   3. Enter those values here
YAW_CENTER = 2048      # Change this to your yaw center position
PITCH_CENTER = 2048    # Change this to your pitch center position

# Invert direction if needed (1.0 or -1.0)
YAW_DIRECTION = 1.0
PITCH_DIRECTION = 1.0

# Position limits (in Dynamixel units from center)
MAX_OFFSET = 1000  # Limit movement to ±1000 units from center (~±88°)


class NeckController:
    def __init__(self, port=DEVICENAME, baudrate=BAUDRATE):
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.running = True
        
        # Open port
        if not self.port_handler.openPort():
            raise Exception(f"Failed to open port {port}")
        
        if not self.port_handler.setBaudRate(baudrate):
            raise Exception(f"Failed to set baudrate {baudrate}")
        
        print(f"[Neck] Connected to {port} @ {baudrate} baud")
        
        # Enable torque with verification
        for motor_id, name in [(ID_YAW, "Yaw"), (ID_PITCH, "Pitch")]:
            # First ping to verify motor is responding
            model, result, error = self.packet_handler.ping(self.port_handler, motor_id)
            if result != COMM_SUCCESS:
                print(f"[Neck] ❌ Cannot ping {name} motor (ID={motor_id})")
                print(f"       Error: {self.packet_handler.getTxRxResult(result)}")
                continue
            print(f"[Neck] ✓ {name} motor found (ID={motor_id}, Model={model})")
            
            # Enable torque
            result, error = self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
            if result != COMM_SUCCESS:
                print(f"[Neck] ❌ Failed to enable torque on {name}: {self.packet_handler.getTxRxResult(result)}")
                continue
            if error != 0:
                print(f"[Neck] ❌ Error enabling torque on {name}: {self.packet_handler.getRxPacketError(error)}")
                continue
            
            # Verify torque was enabled
            time.sleep(0.05)
            torque_status, result, error = self.packet_handler.read1ByteTxRx(
                self.port_handler, motor_id, ADDR_TORQUE_ENABLE)
            if result == COMM_SUCCESS and torque_status == 1:
                print(f"[Neck] ✓ {name} torque ENABLED")
            else:
                print(f"[Neck] ⚠ {name} torque status: {torque_status} (expected 1)")
    
    def _write_byte(self, motor_id, addr, value):
        self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, addr, value)
    
    def _write_dword(self, motor_id, addr, value):
        self.packet_handler.write4ByteTxRx(self.port_handler, motor_id, addr, int(value))
    
    def set_angles(self, yaw_rad, pitch_rad):
        """Set neck angles in radians, using calibrated center positions"""
        # Convert radians to offset from center (with direction)
        yaw_offset = int(yaw_rad * POSITION_SCALE * YAW_DIRECTION)
        pitch_offset = int(pitch_rad * POSITION_SCALE * PITCH_DIRECTION)
        
        # Clamp to safe range
        yaw_offset = max(-MAX_OFFSET, min(MAX_OFFSET, yaw_offset))
        pitch_offset = max(-MAX_OFFSET, min(MAX_OFFSET, pitch_offset))
        
        # Apply calibrated center positions
        yaw_pos = YAW_CENTER + yaw_offset
        pitch_pos = PITCH_CENTER + pitch_offset
        
        self._write_dword(ID_YAW, ADDR_GOAL_POSITION, yaw_pos)
        self._write_dword(ID_PITCH, ADDR_GOAL_POSITION, pitch_pos)
    
    def center(self):
        """Move to calibrated center position"""
        self._write_dword(ID_YAW, ADDR_GOAL_POSITION, YAW_CENTER)
        self._write_dword(ID_PITCH, ADDR_GOAL_POSITION, PITCH_CENTER)
    
    def close(self):
        """Disable motors and close"""
        self._write_byte(ID_YAW, ADDR_TORQUE_ENABLE, 0)
        self._write_byte(ID_PITCH, ADDR_TORQUE_ENABLE, 0)
        self.port_handler.closePort()
        print("[Neck] Motors disabled, port closed")


def main():
    global YAW_CENTER, PITCH_CENTER, YAW_DIRECTION, PITCH_DIRECTION
    
    parser = argparse.ArgumentParser(description='Local Neck Controller for TWIST2')
    parser.add_argument('--port', default=DEVICENAME, help='Serial port (default: /dev/ttyUSB0)')
    parser.add_argument('--baudrate', type=int, default=BAUDRATE, help='Baudrate (default: 57600)')
    parser.add_argument('--redis_host', default='localhost', help='Redis host')
    parser.add_argument('--redis_port', type=int, default=6379, help='Redis port')
    parser.add_argument('--yaw_center', type=int, default=YAW_CENTER, help='Yaw motor center position (default: 2048)')
    parser.add_argument('--pitch_center', type=int, default=PITCH_CENTER, help='Pitch motor center position (default: 2048)')
    parser.add_argument('--invert_yaw', action='store_true', help='Invert yaw direction')
    parser.add_argument('--invert_pitch', action='store_true', help='Invert pitch direction')
    args = parser.parse_args()
    
    # Apply calibration from args
    YAW_CENTER = args.yaw_center
    PITCH_CENTER = args.pitch_center
    if args.invert_yaw:
        YAW_DIRECTION = -1.0
    if args.invert_pitch:
        PITCH_DIRECTION = -1.0
    
    print("=" * 60)
    print("  TWIST2 Local Neck Controller")
    print("=" * 60)
    print(f"  Serial Port: {args.port}")
    print(f"  Redis: {args.redis_host}:{args.redis_port}")
    print(f"  Redis Key: action_neck_unitree_g1_with_hands")
    print("=" * 60)
    print()
    
    # Connect to Redis
    try:
        r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
        r.ping()
        print(f"[Redis] Connected to {args.redis_host}:{args.redis_port}")
    except Exception as e:
        print(f"[Redis] ERROR: Cannot connect - {e}")
        print("[Redis] Make sure redis-server is running: redis-server --daemonize yes")
        sys.exit(1)
    
    # Connect to neck motors
    try:
        neck = NeckController(args.port, args.baudrate)
    except Exception as e:
        print(f"[Neck] ERROR: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check connection: ls {args.port}")
        print(f"  2. Grant permission: sudo chmod 777 {args.port}")
        print(f"  3. Verify baudrate matches Dynamixel Wizard ({args.baudrate})")
        sys.exit(1)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n[Neck] Shutting down...")
        neck.center()
        time.sleep(0.5)
        neck.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Main loop - read from Redis and control motors
    print("\n[Neck] Running... Press Ctrl+C to stop")
    print(f"[Neck] Calibration: YAW_CENTER={YAW_CENTER}, PITCH_CENTER={PITCH_CENTER}")
    print("[Neck] Waiting for commands from teleop...\n")
    
    last_yaw, last_pitch = None, None
    loop_count = 0
    no_data_count = 0
    
    while True:
        try:
            # Read neck command from Redis
            neck_data_str = r.get("action_neck_unitree_g1_with_hands")
            
            if neck_data_str:
                neck_data = json.loads(neck_data_str)
                yaw = float(neck_data[0])
                pitch = float(neck_data[1])
                
                # Send to motors
                neck.set_angles(yaw, pitch)
                
                # Print status - show when values change
                loop_count += 1
                yaw_deg = math.degrees(yaw)
                pitch_deg = math.degrees(pitch)
                
                # Check if values changed
                if last_yaw is None or abs(yaw - last_yaw) > 0.01 or abs(pitch - last_pitch) > 0.01:
                    print(f"[Neck] Yaw: {yaw_deg:+6.1f}°  Pitch: {pitch_deg:+6.1f}° (updated)    ")
                elif loop_count % 50 == 0:
                    print(f"[Neck] Yaw: {yaw_deg:+6.1f}°  Pitch: {pitch_deg:+6.1f}° (no change)", end="\r")
                
                last_yaw, last_pitch = yaw, pitch
                no_data_count = 0
            else:
                no_data_count += 1
                if no_data_count == 50:  # After ~1 second of no data
                    print("[Neck] ⚠ No data in Redis - is teleop running?")
                elif no_data_count % 250 == 0:  # Every ~5 seconds
                    print("[Neck] ⚠ Still no data from Redis...")
            
            time.sleep(0.02)  # 50Hz
            
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"\n[Neck] Error: {e}")
            time.sleep(0.1)


if __name__ == "__main__":
    main()

