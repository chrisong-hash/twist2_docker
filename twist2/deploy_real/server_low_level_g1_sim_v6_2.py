"""
Server Low Level G1 Sim V6.2 - Uses real future observations from Motion Server V6.2

This is a modified version of server_low_level_g1_sim.py that:
1. Reads future observations from Redis key `action_mimic_future_{robot}` (105 dims)
2. Uses history buffer (10 frames)
3. Expects total observation size of 1502 dims (127*11 + 105)

For use with g1_stu_future_v6_2 student policy.
"""

import argparse
import json
import time
import numpy as np
import redis
import mujoco
import torch
import yaml
from rich import print
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
import os
from data_utils.rot_utils import quatToEuler

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class OnnxPolicyWrapper:
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature."""

    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))


def get_gravity_orientation_from_quat(quaternion):
    """Get gravity orientation from IMU quaternion (w, x, y, z order)"""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


class SimControllerV6_2:
    """MuJoCo-based simulation controller for V6.2 student with real future observations."""
    
    def __init__(self, xml_path, policy_path, device='cuda', record_video=False, record_proprio=False):
        self.device = device
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Load policy (ONNX)
        if policy_path.endswith('.onnx'):
            if ort is None:
                raise ImportError("onnxruntime not installed")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
            session = ort.InferenceSession(policy_path, providers=providers)
            input_name = session.get_inputs()[0].name
            self.policy = OnnxPolicyWrapper(session, input_name)
            print(f"[V6.2] Loaded ONNX policy from {policy_path}")
        else:
            self.policy = torch.jit.load(policy_path, map_location=device)
            print(f"[V6.2] Loaded JIT policy from {policy_path}")
        
        # Robot configuration
        self.num_actions = 29
        self.default_dof_pos = np.array([
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # Left leg
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # Right leg
            0.0, 0.0, 0.0,                     # Waist
            0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # Left arm
            0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0 # Right arm
        ], dtype=np.float32)
        
        self.ankle_idx = [4, 5, 10, 11]
        
        # V6.2 observation structure (matches training config)
        self.n_mimic_obs = 35           # Current mimic obs
        self.n_proprio = 92             # Proprioceptive obs
        self.n_obs_single = 127         # n_mimic_obs + n_proprio
        self.history_len = 10           # Must be 1,10,20,50 (HistoryEncoder constraint)
        self.n_future_obs = 105         # 3 future frames × 35 dims
        
        # Total: 127 × 11 + 105 = 1502
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_future_obs
        
        print(f"[V6.2] Observation Configuration:")
        print(f"  n_mimic_obs: {self.n_mimic_obs}")
        print(f"  n_proprio: {self.n_proprio}")
        print(f"  n_obs_single: {self.n_obs_single}")
        print(f"  history_len: {self.history_len}")
        print(f"  n_future_obs: {self.n_future_obs} (3 frames × 35)")
        print(f"  total_obs_size: {self.total_obs_size}")
        
        # Initialize history buffer
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))
        
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        
        # Control parameters
        self.control_dt = 0.02  # 50Hz
        self.action_scale = 0.25
        
        # PD gains
        self.kps = np.array([100]*6 + [100]*6 + [200]*3 + [40]*7 + [40]*7, dtype=np.float32)
        self.kds = np.array([2]*6 + [2]*6 + [5]*3 + [2]*7 + [2]*7, dtype=np.float32)
        
        self.record_video = record_video
        self.record_proprio = record_proprio
        self.proprio_recordings = [] if record_proprio else None
        
        # Redis client placeholder
        self.redis_client = None
    
    def reset_sim(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
    
    def run(self, redis_ip='localhost', robot_type='unitree_g1_with_hands'):
        """Main control loop with real future observations."""
        
        # Connect to Redis
        self.redis_client = redis.Redis(host=redis_ip, port=6379, db=0)
        self.redis_client.ping()
        print(f"[V6.2] Connected to Redis at {redis_ip}")
        
        # Reset simulation
        self.reset_sim()
        self.data.qpos[2] = 0.78  # Initial height
        mujoco.mj_forward(self.model, self.data)
        
        # Launch viewer
        viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        
        print(f"[V6.2] Starting control loop...")
        print(f"[V6.2] Waiting for motion server to publish:")
        print(f"       - action_body_{robot_type} (35 dims)")
        print(f"       - action_mimic_future_{robot_type} (105 dims)")
        
        step_count = 0
        policy_step_count = 0
        
        try:
            while viewer.is_running():
                t0 = time.time()
                
                # Get robot state
                qpos = self.data.qpos.copy()
                qvel = self.data.qvel.copy()
                
                dof_pos = qpos[7:7+self.num_actions]
                dof_vel = qvel[6:6+self.num_actions]
                quat = qpos[3:7]  # w, x, y, z
                ang_vel = qvel[3:6]
                rpy = quatToEuler(quat)
                
                # Build proprioceptive observation
                obs_body_dof_vel = dof_vel.copy()
                obs_body_dof_vel[self.ankle_idx] = 0.0  # Zero ankle velocity
                
                obs_proprio = np.concatenate([
                    ang_vel * 0.25,                      # 3 dims
                    rpy[:2],                              # 2 dims (roll, pitch)
                    (dof_pos - self.default_dof_pos),    # 29 dims
                    obs_body_dof_vel * 0.05,             # 29 dims
                    self.last_action                      # 29 dims
                ])  # Total: 92 dims
                
                # Fetch from Redis (with pipeline for efficiency)
                pipeline = self.redis_client.pipeline()
                pipeline.get(f"action_body_{robot_type}")
                pipeline.get(f"action_mimic_future_{robot_type}")  # V6.2: Real future!
                redis_results = pipeline.execute()
                
                if redis_results[0] is None:
                    # No data yet, wait
                    time.sleep(0.01)
                    continue
                
                # Parse current frame
                action_mimic = np.array(json.loads(redis_results[0]), dtype=np.float32)
                
                # Parse future frames (V6.2: Real future observations!)
                if redis_results[1] is not None:
                    future_obs = np.array(json.loads(redis_results[1]), dtype=np.float32)
                    if len(future_obs) != self.n_future_obs:
                        print(f"[V6.2] WARNING: Expected {self.n_future_obs} future dims, got {len(future_obs)}")
                        future_obs = np.zeros(self.n_future_obs, dtype=np.float32)
                else:
                    # Fallback: use current frame repeated (like old behavior)
                    future_obs = np.tile(action_mimic, 3)[:self.n_future_obs]
                
                # Build full observation
                obs_full = np.concatenate([action_mimic, obs_proprio])  # 127 dims
                
                # Update history
                obs_hist = np.array(self.proprio_history_buf).flatten()  # 15 × 127 = 1905 dims
                self.proprio_history_buf.append(obs_full)
                
                # Combine: current (127) + history (1905) + future (105) = 2137
                obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
                
                # Verify size
                if obs_buf.shape[0] != self.total_obs_size:
                    print(f"[V6.2] ERROR: Expected {self.total_obs_size} obs, got {obs_buf.shape[0]}")
                    continue
                
                # Run policy
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0)
                if 'cuda' in self.device:
                    obs_tensor = obs_tensor.to(self.device)
                
                with torch.no_grad():
                    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                
                # Convert to target positions
                target_dof_pos = raw_action * self.action_scale + self.default_dof_pos
                self.last_action = raw_action.copy()
                
                # PD control
                torque = self.kps * (target_dof_pos - dof_pos) - self.kds * dof_vel
                self.data.ctrl[:self.num_actions] = torque
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Update viewer
                viewer.sync()
                
                step_count += 1
                policy_step_count += 1
                
                # Print progress
                if policy_step_count % 100 == 0:
                    print(f"[V6.2] Step {policy_step_count} | future_obs: [{future_obs[0]:.2f}, ..., {future_obs[-1]:.2f}]")
                
                # Maintain real-time pace
                elapsed = time.time() - t0
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)
        
        except KeyboardInterrupt:
            print("\n[V6.2] Interrupted by user")
        finally:
            viewer.close()
            print(f"[V6.2] Simulation ended after {step_count} steps")


def main():
    parser = argparse.ArgumentParser(description="Sim2Sim Controller V6.2 with real future observations")
    parser.add_argument("--xml", type=str, required=True, help="Path to MuJoCo XML")
    parser.add_argument("--policy", type=str, required=True, help="Path to ONNX/JIT policy")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis IP")
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands", help="Robot type")
    args = parser.parse_args()
    
    controller = SimControllerV6_2(
        xml_path=args.xml,
        policy_path=args.policy,
        device=args.device
    )
    
    controller.run(redis_ip=args.redis_ip, robot_type=args.robot)


if __name__ == "__main__":
    main()

