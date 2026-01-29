#!/usr/bin/env python3
"""
Motion Server V6.2 Buffered - Provides "future" observations through time-delay buffering

For real-time teleop deployment:
- We can't see the future, but we can DELAY our output
- Buffer incoming motion for 0.5s (25 frames at 50Hz)
- "Current" = 0.5s ago, "Future" = progressively more recent inputs

This gives the V6.2 student policy the same "future sight" structure it was trained with,
at the cost of 0.5s response latency.

Buffer structure (25 frames, indices 0-24):
- Frame 0 (oldest): Used as "current" observation
- Frame 5: Used as "future +0.1s"
- Frame 15: Used as "future +0.3s"
- Frame 24 (newest): Used as "future +0.5s" (actual real-time input)
"""

import argparse
import json
import time
import numpy as np
import redis
from collections import deque
from rich import print


class BufferedMotionServer:
    """Motion server with time-delay buffering for V6.2 future observations."""
    
    def __init__(self, redis_ip='localhost', robot_type='unitree_g1_with_hands'):
        self.redis_client = redis.Redis(host=redis_ip, port=6379, db=0)
        self.robot_type = robot_type
        
        # V6.2 config: future steps [5, 15, 25] at 50Hz = [0.1s, 0.3s, 0.5s]
        self.buffer_size = 25  # 0.5s buffer at 50Hz
        self.future_indices = [5, 15, 24]  # Relative to "current" (index 0)
        
        # Observation dimensions
        self.obs_dim = 35  # 6 + 29 (root info + dof_pos)
        
        # Initialize buffer with zeros (will fill with first real observation)
        self.motion_buffer = deque(maxlen=self.buffer_size)
        self._buffer_initialized = False
        
        # Control rate
        self.dt = 0.02  # 50Hz
        
        print(f"[BufferedMotion] Initialized with {self.buffer_size} frame buffer (0.5s latency)")
        print(f"[BufferedMotion] Future indices: {self.future_indices}")
        print(f"[BufferedMotion] Redis: {redis_ip}, Robot: {robot_type}")
    
    def _init_buffer(self, first_obs):
        """Initialize buffer with copies of first observation."""
        for _ in range(self.buffer_size):
            self.motion_buffer.append(first_obs.copy())
        self._buffer_initialized = True
        print(f"[BufferedMotion] Buffer initialized with first observation")
    
    def run(self):
        """Main loop: read input, buffer, output delayed observations."""
        print(f"[BufferedMotion] Starting buffered motion server...")
        print(f"[BufferedMotion] Listening for: motion_raw_{self.robot_type}")
        print(f"[BufferedMotion] Publishing to:")
        print(f"  - action_body_{self.robot_type} (delayed 0.5s, 35 dims)")
        print(f"  - action_mimic_future_{self.robot_type} (future, 105 dims)")
        
        step_count = 0
        
        try:
            while True:
                t0 = time.time()
                
                # Read input motion observation from the RAW motion source
                # The original motion server should publish to: motion_raw_*
                # We process it and publish delayed version to: action_body_*
                input_key = f"motion_raw_{self.robot_type}"
                input_data = self.redis_client.get(input_key)
                
                if input_data is None:
                    # No raw motion data yet
                    if step_count == 0:
                        print(f"[BufferedMotion] Waiting for motion_raw_{self.robot_type}...")
                    time.sleep(0.01)
                    continue
                
                # Parse input
                current_input = np.array(json.loads(input_data), dtype=np.float32)
                
                if len(current_input) < self.obs_dim:
                    print(f"[BufferedMotion] WARNING: Input has {len(current_input)} dims, expected {self.obs_dim}")
                    time.sleep(0.01)
                    continue
                
                current_input = current_input[:self.obs_dim]
                
                # Initialize buffer if needed
                if not self._buffer_initialized:
                    self._init_buffer(current_input)
                
                # Add current input to buffer (newest)
                self.motion_buffer.append(current_input.copy())
                
                # Extract observations from buffer
                # Index 0 = oldest = "current" for policy (delayed by 0.5s)
                # Higher indices = more recent = "future" from policy's perspective
                buffer_list = list(self.motion_buffer)
                
                current_obs = buffer_list[0]  # 0.5s ago = policy's "now"
                
                # Future observations (0.1s, 0.3s, 0.5s ahead from policy's perspective)
                future_obs_list = []
                for idx in self.future_indices:
                    if idx < len(buffer_list):
                        future_obs_list.append(buffer_list[idx])
                    else:
                        future_obs_list.append(buffer_list[-1])  # Use latest if buffer too small
                
                future_obs = np.concatenate(future_obs_list)  # 3 * 35 = 105 dims
                
                # Publish to Redis
                pipeline = self.redis_client.pipeline()
                pipeline.set(f"action_body_{self.robot_type}", json.dumps(current_obs.tolist()))
                pipeline.set(f"action_mimic_future_{self.robot_type}", json.dumps(future_obs.tolist()))
                pipeline.execute()
                
                step_count += 1
                
                # Debug output every 100 steps
                if step_count % 100 == 0:
                    print(f"[BufferedMotion] Step {step_count} | "
                          f"input[0]={current_input[0]:.2f} -> current[0]={current_obs[0]:.2f} | "
                          f"future_sum={np.abs(future_obs).sum():.1f}")
                
                # Maintain rate
                elapsed = time.time() - t0
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)
        
        except KeyboardInterrupt:
            print(f"\n[BufferedMotion] Interrupted after {step_count} steps")


def main():
    parser = argparse.ArgumentParser(description="Buffered Motion Server for V6.2 with future observations")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis IP address")
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands", help="Robot type identifier")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  V6.2 Buffered Motion Server")
    print("  Provides 'future sight' through 0.5s time-delay buffering")
    print("=" * 60)
    
    server = BufferedMotionServer(
        redis_ip=args.redis_ip,
        robot_type=args.robot
    )
    server.run()


if __name__ == "__main__":
    main()

