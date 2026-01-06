#!/usr/bin/env python3
"""
Simple Neck Test - Sends test commands to Redis
No display/MuJoCo needed - just tests the Redis → Neck pipeline

Usage:
    python test_neck_redis_sender.py              # Sine wave test
    python test_neck_redis_sender.py --manual     # Manual input
"""

import argparse
import time
import math
import json

try:
    import redis
except ImportError:
    print("ERROR: pip install redis")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description='Send neck commands to Redis')
    parser.add_argument('--redis_host', default='localhost')
    parser.add_argument('--redis_port', type=int, default=6379)
    parser.add_argument('--manual', action='store_true', help='Manual angle input')
    args = parser.parse_args()
    
    print("=" * 50)
    print("  Neck Redis Sender Test")
    print("=" * 50)
    print(f"Sending to: action_neck_unitree_g1_with_hands")
    print()
    
    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    
    try:
        r.ping()
        print("✓ Connected to Redis\n")
    except:
        print("ERROR: Cannot connect to Redis")
        print("Start it with: redis-server --daemonize yes")
        return
    
    if args.manual:
        print("Enter angles in degrees (yaw pitch), or 'q' to quit:")
        print("Example: 30 15  (yaw=30°, pitch=15°)\n")
        
        while True:
            try:
                inp = input("yaw pitch> ").strip()
                if inp.lower() == 'q':
                    break
                
                parts = inp.split()
                if len(parts) >= 2:
                    yaw_deg = float(parts[0])
                    pitch_deg = float(parts[1])
                    yaw_rad = math.radians(yaw_deg)
                    pitch_rad = math.radians(pitch_deg)
                    
                    r.set("action_neck_unitree_g1_with_hands", json.dumps([yaw_rad, pitch_rad]))
                    print(f"  Sent: yaw={yaw_deg:.1f}°, pitch={pitch_deg:.1f}°")
                else:
                    print("  Enter two numbers: yaw pitch")
            except ValueError:
                print("  Invalid input")
            except KeyboardInterrupt:
                break
    else:
        # Automatic sine wave test
        print("Running sine wave test... Press Ctrl+C to stop\n")
        t = 0
        try:
            while True:
                # Sine wave: ±30° yaw, ±20° pitch
                yaw_deg = 30 * math.sin(t * 0.5)
                pitch_deg = 20 * math.sin(t * 0.7)
                
                yaw_rad = math.radians(yaw_deg)
                pitch_rad = math.radians(pitch_deg)
                
                r.set("action_neck_unitree_g1_with_hands", json.dumps([yaw_rad, pitch_rad]))
                
                print(f"\rYaw: {yaw_deg:+6.1f}°  Pitch: {pitch_deg:+6.1f}°", end="")
                
                t += 0.02
                time.sleep(0.02)  # 50Hz
                
        except KeyboardInterrupt:
            print("\n\nStopping...")
            # Center the neck
            r.set("action_neck_unitree_g1_with_hands", json.dumps([0.0, 0.0]))
            print("Sent center position [0, 0]")


if __name__ == "__main__":
    main()

