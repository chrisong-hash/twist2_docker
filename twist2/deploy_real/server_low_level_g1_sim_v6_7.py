"""
V6.7 Sim2Sim Controller - HYBRID

Switches between two policies based on commanded motion:
- When mimic velocity is near zero (standing) → Use DEFAULT policy (stable standing)
- When mimic velocity is significant (moving) → Use V6.3 policy (better tracking)

This combines the best of both:
- Default policy's stability when standing
- V6.3's superior motion tracking
"""

import argparse
import json
import time
import numpy as np
import mujoco
import mujoco.viewer
import redis
import torch
from collections import deque
from data_utils.rot_utils import quatToEuler
import onnxruntime as ort


class HybridController:
    def __init__(self, xml_path, default_policy_path, tracking_policy_path, device='cuda'):
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Load BOTH policies
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        
        # Default policy (V6 student) - good at standing
        self.default_session = ort.InferenceSession(default_policy_path, providers=providers)
        self.default_input_name = self.default_session.get_inputs()[0].name
        self.default_obs_size = self.default_session.get_inputs()[0].shape[1]
        print(f"[V6.7] Loaded DEFAULT policy: {default_policy_path}")
        print(f"[V6.7]   Obs size: {self.default_obs_size}")
        
        # Tracking policy (V6.3) - good at motion tracking
        self.tracking_session = ort.InferenceSession(tracking_policy_path, providers=providers)
        self.tracking_input_name = self.tracking_session.get_inputs()[0].name
        self.tracking_obs_size = self.tracking_session.get_inputs()[0].shape[1]
        print(f"[V6.7] Loaded TRACKING policy: {tracking_policy_path}")
        print(f"[V6.7]   Obs size: {self.tracking_obs_size}")
        
        # Robot parameters
        self.num_actions = 29
        self.default_dof_pos = np.array([
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.2, 0.0, 1.2, 0.0, 0.0, 0.0,
            0.0, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        self.mujoco_init_qpos = np.concatenate([
            np.array([0, 0, 0.793]),
            np.array([1, 0, 0, 0]),
            self.default_dof_pos
        ])
        
        self.ankle_idx = [4, 5, 10, 11]
        
        # PD gains (from original server)
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
        
        # History buffers (separate for each policy type)
        self.history_len = 10
        self.proprio_history_buf = deque(maxlen=self.history_len)
        
        # Timing - 100Hz
        self.sim_dt = float(self.model.opt.timestep)
        self.sim_decimation = 10
        
        # Velocity threshold for switching policies
        # mimic_obs[0:2] = root_vel_xy, mimic_obs[5] = yaw_ang_vel
        self.VELOCITY_THRESHOLD = 0.05  # Switch to tracking when velocity > this
        
        # Track last mimic for velocity estimation
        self.last_mimic = None
        
        print(f"[V6.7] HYBRID MODE:")
        print(f"  - Velocity threshold: {self.VELOCITY_THRESHOLD}")
        print(f"  - Standing: DEFAULT policy")
        print(f"  - Moving: TRACKING policy (V6.3)")
    
    def _is_standing(self, action_mimic):
        """Check if the commanded motion is near-zero (standing still)."""
        # Check root velocities in mimic obs
        # mimic_obs = [root_vel_x, root_vel_y, root_pos_z, roll, pitch, yaw_ang_vel, dof_pos(29)]
        root_vel_xy = action_mimic[0:2]
        yaw_ang_vel = action_mimic[5]
        
        # Also check if dof_pos is changing (compare with last)
        dof_change = 0.0
        if self.last_mimic is not None:
            dof_change = np.abs(action_mimic[6:] - self.last_mimic[6:]).mean()
        
        vel_magnitude = np.sqrt(root_vel_xy[0]**2 + root_vel_xy[1]**2 + yaw_ang_vel**2)
        
        is_standing = vel_magnitude < self.VELOCITY_THRESHOLD and dof_change < 0.02
        
        return is_standing, vel_magnitude, dof_change
    
    def run(self, redis_ip='localhost', robot_type='unitree_g1_with_hands'):
        redis_client = redis.Redis(host=redis_ip, port=6379)
        print(f"[V6.7] Connected to Redis at {redis_ip}")
        
        # Initialize simulation
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.mujoco_init_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        
        # Reset state
        last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.proprio_history_buf.clear()
        history_initialized = False
        self.last_mimic = None
        
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        print("[V6.7] Viewer started. Waiting for motion data...")
        
        step_count = 0
        default_steps = 0
        tracking_steps = 0
        
        try:
            while viewer.is_running():
                # Get motion data from Redis
                body_data = redis_client.get(f"action_body_{robot_type}")
                future_data = redis_client.get(f"action_mimic_future_{robot_type}")
                
                if body_data is None:
                    # No data - just wait (like original server)
                    viewer.sync()
                    time.sleep(0.01)
                    continue
                
                step_count += 1
                
                # Parse motion data
                action_mimic = np.array(json.loads(body_data), dtype=np.float32)
                future_obs = np.array(json.loads(future_data), dtype=np.float32) if future_data else action_mimic.copy()
                
                # Determine which policy to use
                is_standing, vel_mag, dof_change = self._is_standing(action_mimic)
                self.last_mimic = action_mimic.copy()
                
                # Get current state
                qpos = self.data.qpos.copy()
                qvel = self.data.qvel.copy()
                dof_pos = qpos[7:7+self.num_actions]
                dof_vel = qvel[6:6+self.num_actions]
                ang_vel = qvel[3:6]
                rpy = quatToEuler(qpos[3:7])
                
                obs_body_dof_vel = dof_vel.copy()
                obs_body_dof_vel[self.ankle_idx] = 0.0
                
                # Build proprioceptive observation
                obs_proprio = np.concatenate([
                    ang_vel * 0.25,
                    rpy[:2],
                    dof_pos - self.default_dof_pos,
                    obs_body_dof_vel * 0.05,
                    last_action
                ])
                
                # Build full observation
                obs_full = np.concatenate([action_mimic, obs_proprio])
                
                # Initialize history buffer
                if not history_initialized:
                    for _ in range(self.history_len):
                        self.proprio_history_buf.append(obs_full.copy())
                    history_initialized = True
                
                obs_hist = np.array(self.proprio_history_buf).flatten()
                self.proprio_history_buf.append(obs_full)
                
                # Choose policy and build appropriate observation
                if is_standing:
                    # Use DEFAULT policy (1432 dims: current + history + future=current)
                    default_steps += 1
                    future_for_default = action_mimic.copy()
                    obs_buf = np.concatenate([obs_full, obs_hist, future_for_default])
                    
                    if len(obs_buf) < self.default_obs_size:
                        obs_buf = np.concatenate([obs_buf, np.zeros(self.default_obs_size - len(obs_buf))])
                    obs_buf = obs_buf[:self.default_obs_size]
                    
                    obs_tensor = obs_buf.astype(np.float32).reshape(1, -1)
                    raw_action = self.default_session.run(None, {self.default_input_name: obs_tensor})[0].squeeze()
                    policy_name = "DEFAULT"
                else:
                    # Use TRACKING policy (V6.3 with actual future obs)
                    tracking_steps += 1
                    
                    # V6.3 obs: current(127) + history(1270) + future(105) = 1502
                    if len(future_obs) < 105:
                        future_obs = np.tile(action_mimic, 3)[:105]
                    
                    obs_buf = np.concatenate([obs_full, obs_hist, future_obs[:105]])
                    
                    if len(obs_buf) < self.tracking_obs_size:
                        obs_buf = np.concatenate([obs_buf, np.zeros(self.tracking_obs_size - len(obs_buf))])
                    obs_buf = obs_buf[:self.tracking_obs_size]
                    
                    obs_tensor = obs_buf.astype(np.float32).reshape(1, -1)
                    raw_action = self.tracking_session.run(None, {self.tracking_input_name: obs_tensor})[0].squeeze()
                    policy_name = "TRACKING"
                
                raw_action = np.clip(raw_action, -10, 10)
                
                # Compute target position
                target_dof_pos = raw_action * 0.5 + self.default_dof_pos
                last_action = raw_action.copy()
                
                # Run physics
                for _ in range(self.sim_decimation):
                    cur_pos = self.data.qpos[7:7+self.num_actions].copy()
                    cur_vel = self.data.qvel[6:6+self.num_actions].copy()
                    torque = self.kps * (target_dof_pos - cur_pos) - self.kds * cur_vel
                    self.data.ctrl[:self.num_actions] = torque
                    mujoco.mj_step(self.model, self.data)
                
                viewer.sync()
                
                if step_count % 100 == 0:
                    waist_z = self.data.qpos[2]
                    pct_default = 100 * default_steps / step_count if step_count > 0 else 0
                    print(f"[V6.7] Step {step_count} | {policy_name} | Waist Z: {waist_z:.3f} | "
                          f"vel={vel_mag:.3f} | Default: {pct_default:.0f}%")
                    
        finally:
            viewer.close()
            print(f"[V6.7] Ended. Total: {step_count}, Default: {default_steps}, Tracking: {tracking_steps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', type=str, required=True)
    parser.add_argument('--default_policy', type=str, required=True, help='Policy for standing (V6 student)')
    parser.add_argument('--tracking_policy', type=str, required=True, help='Policy for tracking (V6.3)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--redis_ip', type=str, default='localhost')
    parser.add_argument('--robot', type=str, default='unitree_g1_with_hands')
    args = parser.parse_args()
    
    controller = HybridController(args.xml, args.default_policy, args.tracking_policy, args.device)
    controller.run(args.redis_ip, args.robot)


