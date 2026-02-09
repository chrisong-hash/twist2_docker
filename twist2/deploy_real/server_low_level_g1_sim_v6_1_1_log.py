"""
Server Low Level G1 Sim V6.1.1 with LOGGING

Same as V6.2 controller but with extensive logging for debugging:
- Logs all inputs (mimic_obs, proprio, history, future)
- Logs all outputs (raw_action, target_pos)
- Logs robot state (actual positions, velocities)
- Saves to JSON for post-analysis
"""

import argparse
import json
import time
import numpy as np
import redis
import mujoco
import torch
from collections import deque
import mujoco.viewer as mjv
import os
from datetime import datetime
from data_utils.rot_utils import quatToEuler

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class OnnxPolicyWrapper:
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
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


class SimControllerV6_1_1_Log:
    """V6.1.1 controller with extensive logging."""
    
    def __init__(self, xml_path, policy_path, device='cuda', log_dir='./logs_v6_1_1'):
        self.device = device
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Load policy
        if policy_path.endswith('.onnx'):
            if ort is None:
                raise ImportError("onnxruntime not installed")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
            session = ort.InferenceSession(policy_path, providers=providers)
            input_name = session.get_inputs()[0].name
            self.policy = OnnxPolicyWrapper(session, input_name)
            print(f"[V6.1.1-LOG] Loaded ONNX policy from {policy_path}")
        else:
            self.policy = torch.jit.load(policy_path, map_location=device)
        
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
        
        # Joint names for logging
        self.joint_names = [
            'L_hip_pitch', 'L_hip_roll', 'L_hip_yaw', 'L_knee', 'L_ankle_pitch', 'L_ankle_roll',
            'R_hip_pitch', 'R_hip_roll', 'R_hip_yaw', 'R_knee', 'R_ankle_pitch', 'R_ankle_roll',
            'waist_yaw', 'waist_roll', 'waist_pitch',
            'L_shoulder_pitch', 'L_shoulder_roll', 'L_shoulder_yaw', 'L_elbow',
            'L_wrist_roll', 'L_wrist_pitch', 'L_wrist_yaw',
            'R_shoulder_pitch', 'R_shoulder_roll', 'R_shoulder_yaw', 'R_elbow',
            'R_wrist_roll', 'R_wrist_pitch', 'R_wrist_yaw'
        ]
        
        # V6.1.1 observation structure (same as V6.2)
        self.n_mimic_obs = 35
        self.n_proprio = 92
        self.n_obs_single = 127
        self.history_len = 10
        self.n_future_obs = 105
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_future_obs  # 1502
        
        # History buffer
        self.proprio_history_buf = deque(maxlen=self.history_len)
        self._history_initialized = False
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        
        # Control parameters
        self.sim_dt = 0.001
        self.train_sim_dt = 0.002
        self.train_decimation = 10
        self.control_dt = self.train_decimation * self.train_sim_dt
        self.sim_decimation = int(self.control_dt / self.sim_dt)
        self.action_scale = 0.5
        
        # PD gains
        self.kps = np.array([100, 100, 100, 150, 40, 40,
                            100, 100, 100, 150, 40, 40,
                            150, 150, 150,
                            40, 40, 40, 40, 4.0, 4.0, 4.0,
                            40, 40, 40, 40, 4.0, 4.0, 4.0], dtype=np.float32)
        self.kds = np.array([2, 2, 2, 4, 2, 2,
                            2, 2, 2, 4, 2, 2,
                            4, 4, 4,
                            5, 5, 5, 5, 2.0, 2.0, 2.0,
                            5, 5, 5, 5, 2.0, 2.0, 2.0], dtype=np.float32)
        
        # Logging data
        self.log_data = {
            'metadata': {
                'policy_path': policy_path,
                'timestamp': datetime.now().isoformat(),
                'obs_structure': {
                    'n_mimic_obs': self.n_mimic_obs,
                    'n_proprio': self.n_proprio,
                    'n_obs_single': self.n_obs_single,
                    'history_len': self.history_len,
                    'n_future_obs': self.n_future_obs,
                    'total_obs_size': self.total_obs_size
                }
            },
            'frames': []
        }
        
        self.redis_client = None
    
    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
    
    def run(self, redis_ip='localhost', robot_type='unitree_g1_with_hands', max_steps=500):
        self.redis_client = redis.Redis(host=redis_ip, port=6379, db=0)
        self.redis_client.ping()
        print(f"[V6.1.1-LOG] Connected to Redis at {redis_ip}")
        
        self.reset_sim()
        self.data.qpos[2] = 0.78
        mujoco.mj_forward(self.model, self.data)
        
        viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        
        print(f"[V6.1.1-LOG] Starting control loop (max {max_steps} steps)...")
        print(f"[V6.1.1-LOG] Logging to {self.log_dir}")
        
        step_count = 0
        
        try:
            while viewer.is_running() and step_count < max_steps:
                t0 = time.time()
                
                # Get robot state
                qpos = self.data.qpos.copy()
                qvel = self.data.qvel.copy()
                
                dof_pos = qpos[7:7+self.num_actions]
                dof_vel = qvel[6:6+self.num_actions]
                quat = qpos[3:7]
                ang_vel = qvel[3:6]
                rpy = quatToEuler(quat)
                root_pos = qpos[:3]
                
                # Build proprio
                obs_body_dof_vel = dof_vel.copy()
                obs_body_dof_vel[self.ankle_idx] = 0.0
                
                obs_proprio = np.concatenate([
                    ang_vel * 0.25,
                    rpy[:2],
                    (dof_pos - self.default_dof_pos),
                    obs_body_dof_vel * 0.05,
                    self.last_action
                ])
                
                # Fetch from Redis
                pipeline = self.redis_client.pipeline()
                pipeline.get(f"action_body_{robot_type}")
                pipeline.get(f"action_mimic_future_{robot_type}")
                redis_results = pipeline.execute()
                
                if redis_results[0] is None:
                    time.sleep(0.01)
                    continue
                
                action_mimic = np.array(json.loads(redis_results[0]), dtype=np.float32)
                
                if redis_results[1] is not None:
                    future_obs = np.array(json.loads(redis_results[1]), dtype=np.float32)
                    if len(future_obs) != self.n_future_obs:
                        future_obs = np.zeros(self.n_future_obs, dtype=np.float32)
                else:
                    future_obs = np.tile(action_mimic, 3)[:self.n_future_obs]
                
                # Build full observation
                obs_full = np.concatenate([action_mimic, obs_proprio])
                
                if not self._history_initialized:
                    for _ in range(self.history_len):
                        self.proprio_history_buf.append(obs_full.copy())
                    self._history_initialized = True
                
                obs_hist = np.array(self.proprio_history_buf).flatten()
                self.proprio_history_buf.append(obs_full)
                
                obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
                
                # Run policy
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).clip(-100, 100)
                if 'cuda' in self.device:
                    obs_tensor = obs_tensor.to(self.device)
                
                with torch.no_grad():
                    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                raw_action = np.clip(raw_action, -100, 100)
                
                target_dof_pos = raw_action * self.action_scale + self.default_dof_pos
                
                # === LOG THIS FRAME ===
                frame_data = {
                    'step': step_count,
                    'time': step_count * self.control_dt,
                    
                    # Inputs
                    'input': {
                        'action_mimic': action_mimic.tolist(),
                        'future_obs': future_obs.tolist(),
                        'obs_proprio': obs_proprio.tolist(),
                        'obs_full': obs_full.tolist(),
                        # Key components
                        'ang_vel': (ang_vel * 0.25).tolist(),
                        'roll_pitch': rpy[:2].tolist(),
                        'dof_pos_error': (dof_pos - self.default_dof_pos).tolist(),
                        'dof_vel_scaled': (obs_body_dof_vel * 0.05).tolist(),
                        'last_action': self.last_action.tolist(),
                    },
                    
                    # Outputs
                    'output': {
                        'raw_action': raw_action.tolist(),
                        'target_dof_pos': target_dof_pos.tolist(),
                    },
                    
                    # Actual robot state
                    'robot_state': {
                        'root_pos': root_pos.tolist(),
                        'quat': quat.tolist(),
                        'rpy': rpy.tolist(),
                        'dof_pos': dof_pos.tolist(),
                        'dof_vel': dof_vel.tolist(),
                    },
                    
                    # Statistics
                    'stats': {
                        'raw_action_min': float(raw_action.min()),
                        'raw_action_max': float(raw_action.max()),
                        'raw_action_std': float(raw_action.std()),
                        'raw_action_abs_mean': float(np.abs(raw_action).mean()),
                        'dof_pos_error_abs_max': float(np.abs(dof_pos - self.default_dof_pos).max()),
                        # Wrist actions (indices 18-20, 25-27)
                        'wrist_actions': raw_action[[18,19,20,25,26,27]].tolist(),
                    }
                }
                self.log_data['frames'].append(frame_data)
                
                # Store for next step
                self.last_action = raw_action.copy()
                
                # PD control
                for sim_step in range(self.sim_decimation):
                    current_dof_pos = self.data.qpos[7:7+self.num_actions].copy()
                    current_dof_vel = self.data.qvel[6:6+self.num_actions].copy()
                    torque = self.kps * (target_dof_pos - current_dof_pos) - self.kds * current_dof_vel
                    self.data.ctrl[:self.num_actions] = torque
                    mujoco.mj_step(self.model, self.data)
                
                viewer.sync()
                step_count += 1
                
                # Print progress
                if step_count % 50 == 0:
                    print(f"[V6.1.1-LOG] Step {step_count}/{max_steps}")
                    print(f"  raw_action: min={raw_action.min():.4f}, max={raw_action.max():.4f}, std={raw_action.std():.4f}")
                    print(f"  wrist_L: {raw_action[18:21]}")
                    print(f"  wrist_R: {raw_action[25:28]}")
                
                # Real-time pacing
                elapsed = time.time() - t0
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)
        
        except KeyboardInterrupt:
            print("\n[V6.1.1-LOG] Interrupted")
        finally:
            viewer.close()
            
            # Save logs
            log_file = os.path.join(self.log_dir, f'v6_1_1_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(log_file, 'w') as f:
                json.dump(self.log_data, f, indent=2)
            print(f"[V6.1.1-LOG] Saved {len(self.log_data['frames'])} frames to {log_file}")
            
            # Print summary
            self._print_summary()
    
    def _print_summary(self):
        """Print summary statistics from logged data."""
        if not self.log_data['frames']:
            return
        
        print("\n" + "="*60)
        print("  V6.1.1 LOG SUMMARY")
        print("="*60)
        
        # Collect statistics
        raw_actions = np.array([f['output']['raw_action'] for f in self.log_data['frames']])
        target_positions = np.array([f['output']['target_dof_pos'] for f in self.log_data['frames']])
        actual_positions = np.array([f['robot_state']['dof_pos'] for f in self.log_data['frames']])
        
        print(f"\nTotal frames: {len(self.log_data['frames'])}")
        print(f"\n--- Raw Action Statistics ---")
        print(f"  Overall: mean={raw_actions.mean():.4f}, std={raw_actions.std():.4f}")
        print(f"  Range: [{raw_actions.min():.4f}, {raw_actions.max():.4f}]")
        
        # Per-joint statistics
        print(f"\n--- Per-Joint Action Statistics (mean ± std) ---")
        for i, name in enumerate(self.joint_names):
            mean = raw_actions[:, i].mean()
            std = raw_actions[:, i].std()
            if std > 0.1:  # Highlight high-variance joints
                print(f"  {name:20s}: {mean:+.4f} ± {std:.4f}  *** HIGH VARIANCE")
            else:
                print(f"  {name:20s}: {mean:+.4f} ± {std:.4f}")
        
        # Tracking error
        tracking_error = np.abs(target_positions - actual_positions)
        print(f"\n--- Tracking Error (target - actual) ---")
        print(f"  Mean: {tracking_error.mean():.4f}")
        print(f"  Max: {tracking_error.max():.4f}")
        
        # Action jerk (change in action)
        if len(raw_actions) > 1:
            action_changes = np.diff(raw_actions, axis=0)
            print(f"\n--- Action Jerk (frame-to-frame change) ---")
            print(f"  Mean abs change: {np.abs(action_changes).mean():.4f}")
            print(f"  Max abs change: {np.abs(action_changes).max():.4f}")
            
            # Which joints have highest jerk?
            jerk_per_joint = np.abs(action_changes).mean(axis=0)
            top_jerk_idx = np.argsort(jerk_per_joint)[-5:][::-1]
            print(f"\n  Top 5 jittery joints:")
            for idx in top_jerk_idx:
                print(f"    {self.joint_names[idx]:20s}: avg_jerk={jerk_per_joint[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="V6.1.1 Sim2Sim with Logging")
    parser.add_argument("--xml", type=str, required=True)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--redis_ip", type=str, default="localhost")
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--log_dir", type=str, default="./logs_v6_1_1")
    args = parser.parse_args()
    
    controller = SimControllerV6_1_1_Log(
        xml_path=args.xml,
        policy_path=args.policy,
        device=args.device,
        log_dir=args.log_dir
    )
    
    controller.run(
        redis_ip=args.redis_ip,
        robot_type=args.robot,
        max_steps=args.max_steps
    )


if __name__ == "__main__":
    main()


