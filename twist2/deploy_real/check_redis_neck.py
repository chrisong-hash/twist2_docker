#!/usr/bin/env python3
"""Quick check of Redis neck data - run on HOST to debug Redis connection"""

import redis
import json
import time
import sys

print("Checking Redis for neck data...")
print("=" * 50)

try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print("✓ Connected to Redis at localhost:6379")
except Exception as e:
    print(f"❌ Cannot connect to Redis: {e}")
    print("\nMake sure Docker container is running with Redis")
    sys.exit(1)

print("\nMonitoring Redis key: action_neck_unitree_g1_with_hands")
print("Run teleop in Docker to see values change...")
print("Press Ctrl+C to stop\n")

last_value = None
try:
    while True:
        value = r.get("action_neck_unitree_g1_with_hands")
        
        if value != last_value:
            if value:
                data = json.loads(value)
                import math
                yaw_deg = math.degrees(data[0])
                pitch_deg = math.degrees(data[1])
                print(f"[{time.strftime('%H:%M:%S')}] Yaw: {yaw_deg:+6.1f}°  Pitch: {pitch_deg:+6.1f}°")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] No data (key doesn't exist)")
            last_value = value
        
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\nStopped.")

