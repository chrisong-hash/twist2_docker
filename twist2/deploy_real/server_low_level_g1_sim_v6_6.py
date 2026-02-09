"""
V6.6 Sim2Sim Controller - SIMPLIFIED

Key difference from original:
- FREEZES physics when no motion data (robot stands perfectly still)
- Otherwise identical to original server_low_level_g1_sim.py

Usage:
- Without motion server: robot stays frozen in initial pose
- With motion server: robot tracks motion like original
"""

import argparse
import json
import time
import numpy as np
import mujoco
import mujoco.viewer
import redis
from collections import deque
from data_utils.rot_utils import quatToEuler
import onnxruntime as ort


class SimControllerV6_6:
    def __init__(self, xml_path, policy_path, device='cuda'):
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Load ONNX policy
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(policy_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.obs_size = self.session.get_inputs()[0].shape[1]
        print(f"[V6.6] Loaded policy: {policy_path}")
        print(f"[V6.6] Observation size: {self.obs_size}")
        
        # Robot parameters - EXACTLY matching original
        self.num_actions = 29
        self.default_dof_pos = np.array([
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # Left leg
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # Right leg
            0.0, 0.0, 0.0,                     # Waist
            0.0, 0.2, 0.0, 1.2, 0.0, 0.0, 0.0,   # Left arm
            0.0, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0   # Right arm
        ], dtype=np.float32)
        
        self.mujoco_init_qpos = np.concatenate([
            np.array([0, 0, 0.793]),
            np.array([1, 0, 0, 0]),
            self.default_dof_pos
        ])
        
        self.ankle_idx = [4, 5, 10, 11]
        
        # PD gains - EXACTLY matching original server
        self.kps = np.array([100, 100, 100, 150, 40, 40,
                            100, 100, 100, 150, 40, 40,
                            150, 150, 150,
                            40, 40, 40, 40, 4.0, 4.0, 4.0,
                            40, 40, 40, 40, 4.0, 4.0, 4.0], dtype=np.float32)
        self.kds = np.array([15, 15, 15, 25, 15, 15,
                            15, 15, 15, 25, 15, 15,
                            25, 25, 25,
                            20, 20, 20, 20, 2.0, 2.0, 2.0,
                            20, 20, 20, 20, 2.0, 2.0, 2.0], dtype=np.float32)
        
        # History buffer
        self.history_len = 10
        self.proprio_history_buf = deque(maxlen=self.history_len)
        
        # Timing - 100Hz control (matching original server)
        self.sim_dt = float(self.model.opt.timestep)
        self.sim_decimation = 10  # 10 physics steps per control step = 100Hz
        
        print(f"[V6.6] Physics dt: {self.sim_dt}, decimation: {self.sim_decimation}")
        print(f"[V6.6] FREEZE MODE: Robot will freeze when no motion data")
    
    def run(self, redis_ip='localhost', robot_type='unitree_g1_with_hands'):
        redis_client = redis.Redis(host=redis_ip, port=6379)
        print(f"[V6.6] Connected to Redis at {redis_ip}")
        
        # Clear any stale motion data from previous runs
        redis_client.delete(f"action_body_{robot_type}")
        redis_client.delete(f"action_mimic_future_{robot_type}")
        print(f"[V6.6] Cleared stale Redis keys")
        
        # Initialize simulation
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.mujoco_init_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        
        # Reset state
        last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.proprio_history_buf.clear()
        history_initialized = False
        
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        print("[V6.6] Viewer started. Waiting for motion data...")
        
        step_count = 0
        frozen_steps = 0
        
        try:
            while viewer.is_running():
                # Get motion data from Redis
                body_data = redis_client.get(f"action_body_{robot_type}")
                
                # === V6.6 KEY FEATURE: FREEZE when no motion data ===
                if body_data is None:
                    # No motion data - freeze physics completely
                    frozen_steps += 1
                    viewer.sync()
                    time.sleep(0.01)  # 100Hz timing
                    if frozen_steps % 100 == 1:
                        waist_z = self.data.qpos[2]
                        print(f"[V6.6] FROZEN (no motion data) | Steps: {frozen_steps} | Waist Z: {waist_z:.3f}")
                    continue
                
                # Motion data available - run policy
                if frozen_steps > 0:
                    print(f"[V6.6] Motion data received - resuming simulation")
                    frozen_steps = 0
                
                step_count += 1
                
                # Get current state
                qpos = self.data.qpos.copy()
                qvel = self.data.qvel.copy()
                dof_pos = qpos[7:7+self.num_actions]
                dof_vel = qvel[6:6+self.num_actions]
                ang_vel = qvel[3:6]
                rpy = quatToEuler(qpos[3:7])
                
                # Zero ankle velocity (matches original)
                obs_body_dof_vel = dof_vel.copy()
                obs_body_dof_vel[self.ankle_idx] = 0.0
                
                # Build proprioceptive observation (matches original)
                obs_proprio = np.concatenate([
                    ang_vel * 0.25,
                    rpy[:2],
                    dof_pos - self.default_dof_pos,
                    obs_body_dof_vel * 0.05,
                    last_action
                ])
                
                # Parse motion data
                action_mimic = np.array(json.loads(body_data), dtype=np.float32)
                
                # Build full observation
                obs_full = np.concatenate([action_mimic, obs_proprio])
                
                # Initialize history buffer
                if not history_initialized:
                    for _ in range(self.history_len):
                        self.proprio_history_buf.append(obs_full.copy())
                    history_initialized = True
                
                # Get history
                obs_hist = np.array(self.proprio_history_buf).flatten()
                self.proprio_history_buf.append(obs_full)
                
                # Build observation buffer (matches original: current + history + future)
                future_obs = action_mimic.copy()
                obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
                
                # Pad/truncate to match policy input size
                if len(obs_buf) < self.obs_size:
                    obs_buf = np.concatenate([obs_buf, np.zeros(self.obs_size - len(obs_buf))])
                obs_buf = obs_buf[:self.obs_size]
                
                # Run policy
                obs_tensor = obs_buf.astype(np.float32).reshape(1, -1)
                raw_action = self.session.run(None, {self.input_name: obs_tensor})[0].squeeze()
                raw_action = np.clip(raw_action, -10, 10)  # Original clipping
                
                # Compute target position
                target_dof_pos = raw_action * 0.5 + self.default_dof_pos
                last_action = raw_action.copy()
                
                # Run physics with PD control
                for _ in range(self.sim_decimation):
                    cur_pos = self.data.qpos[7:7+self.num_actions].copy()
                    cur_vel = self.data.qvel[6:6+self.num_actions].copy()
                    torque = self.kps * (target_dof_pos - cur_pos) - self.kds * cur_vel
                    self.data.ctrl[:self.num_actions] = torque
                    mujoco.mj_step(self.model, self.data)
                
                viewer.sync()
                
                if step_count % 100 == 0:
                    waist_z = self.data.qpos[2]
                    print(f"[V6.6] Step {step_count} | Waist Z: {waist_z:.3f}")
                    
        finally:
            viewer.close()
            print(f"[V6.6] Ended. Active steps: {step_count}, Frozen steps: {frozen_steps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--redis_ip', type=str, default='localhost')
    parser.add_argument('--robot', type=str, default='unitree_g1_with_hands')
    args = parser.parse_args()
    
    controller = SimControllerV6_6(args.xml, args.policy, args.device)
    controller.run(args.redis_ip, args.robot)
