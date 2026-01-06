#!/usr/bin/env python3
"""
Headless Neck Teleop
====================
Gets head pose from PICO VR and sends neck commands to Redis.
No MuJoCo display needed - just tracks your head movement!

Usage:
    python neck_teleop_headless.py
    python neck_teleop_headless.py --invert_pitch  # If pitch is reversed
    python neck_teleop_headless.py --invert_yaw    # If yaw is reversed

Requirements:
    - XRobotToolkit PC Service running
    - PICO VR connected and streaming
"""

import argparse
import json
import time
import sys
import signal

try:
    import redis
except ImportError:
    print("ERROR: pip install redis")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: pip install numpy")
    sys.exit(1)

try:
    from general_motion_retargeting import XRobotStreamer
    from general_motion_retargeting import human_head_to_robot_neck
except ImportError:
    print("ERROR: GMR not installed!")
    print("Make sure you're in the gmr conda environment")
    sys.exit(1)


class HeadlessNeckTeleop:
    def __init__(self, redis_host='localhost', redis_port=6379, 
                 neck_scale=1.5, invert_yaw=False, invert_pitch=False):
        self.neck_scale = neck_scale
        self.invert_yaw = invert_yaw
        self.invert_pitch = invert_pitch
        self.running = True
        
        # Connect to Redis
        print(f"[Redis] Connecting to {redis_host}:{redis_port}...")
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        try:
            self.redis_client.ping()
            print("[Redis] ✓ Connected")
        except Exception as e:
            print(f"[Redis] ❌ Failed: {e}")
            sys.exit(1)
        
        # Initialize XRobot streamer for PICO VR
        print("[XRobot] Initializing PICO VR connection...")
        self.streamer = XRobotStreamer()
        print("[XRobot] ✓ Initialized")
        
        # Handle Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        print("\n[Teleop] Shutting down...")
        self.running = False
        self.redis_client.set("action_neck_unitree_g1_with_hands", json.dumps([0.0, 0.0]))
        sys.exit(0)
    
    def run(self):
        print("\n" + "=" * 50)
        print("  Headless Neck Teleop")
        print("=" * 50)
        print("  Move your head to control the neck!")
        print("  Press Ctrl+C to stop")
        print("=" * 50 + "\n")
        
        loop_count = 0
        last_yaw, last_pitch = 0.0, 0.0
        no_data_count = 0
        
        while self.running:
            try:
                # Get data from PICO VR
                smplx_data, _, _, _, _ = self.streamer.get_current_frame()
                
                neck_yaw, neck_pitch = 0.0, 0.0
                got_data = False
                
                # Get neck angles from body tracking
                if smplx_data is not None and isinstance(smplx_data, dict):
                    if 'Head' in smplx_data and 'Spine3' in smplx_data:
                        try:
                            neck_yaw, neck_pitch = human_head_to_robot_neck(smplx_data)
                            got_data = True
                        except:
                            pass
                
                if got_data:
                    # Apply inversions
                    if self.invert_yaw:
                        neck_yaw = -neck_yaw
                    if self.invert_pitch:
                        neck_pitch = -neck_pitch
                    
                    # Apply scale
                    neck_yaw *= self.neck_scale
                    neck_pitch *= self.neck_scale
                    
                    # Send to Redis
                    self.redis_client.set(
                        "action_neck_unitree_g1_with_hands", 
                        json.dumps([float(neck_yaw), float(neck_pitch)])
                    )
                    
                    # Print status when values change
                    loop_count += 1
                    if abs(neck_yaw - last_yaw) > 0.02 or abs(neck_pitch - last_pitch) > 0.02:
                        yaw_deg = np.degrees(neck_yaw)
                        pitch_deg = np.degrees(neck_pitch)
                        print(f"[Neck] Yaw: {yaw_deg:+6.1f}°  Pitch: {pitch_deg:+6.1f}°")
                    
                    last_yaw, last_pitch = neck_yaw, neck_pitch
                    no_data_count = 0
                else:
                    no_data_count += 1
                    if no_data_count == 100:
                        print("[XRobot] Waiting for tracking data...")
                
                time.sleep(0.02)  # 50Hz
                
            except Exception as e:
                print(f"[Error] {e}")
                time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description='Headless Neck Teleop')
    parser.add_argument('--redis_host', default='localhost', help='Redis host')
    parser.add_argument('--redis_port', type=int, default=6379, help='Redis port')
    parser.add_argument('--neck_scale', type=float, default=1.5, help='Neck movement scale')
    parser.add_argument('--invert_yaw', action='store_true', help='Invert yaw direction')
    parser.add_argument('--invert_pitch', action='store_true', help='Invert pitch direction')
    args = parser.parse_args()
    
    teleop = HeadlessNeckTeleop(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        neck_scale=args.neck_scale,
        invert_yaw=args.invert_yaw,
        invert_pitch=args.invert_pitch
    )
    teleop.run()


if __name__ == "__main__":
    main()
