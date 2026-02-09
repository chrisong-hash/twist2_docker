"""
Server Low Level G1 Sim V6.3 - V6.3 Student Controller with Wrist Dampening

VALIDATED CONFIG:
- Baseline PD gains (stable)
- Wrist-only dampening when static (reduces visual jitter without affecting balance)
- Static detection: threshold=0.01, blend=0.9, scale=0.1

Test result: Waist Z min=0.780 over 200 steps (stable with wrist dampening)
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


class SimControllerV6_3:
    """V6.3 controller with wrist fixes and stabilization."""
    
    # Wrist joint indices
    WRIST_INDICES = [18, 19, 20, 25, 26, 27]  # L_wrist_roll/pitch/yaw, R_wrist_roll/pitch/yaw
    
    # V6.3 TUNING PARAMETERS (VALIDATED: wrist-only damping when static)
    WRIST_ACTION_LIMIT = None  
    ENABLE_STATIC_DETECTION = True   # Enable static detection
    STATIC_THRESHOLD = 0.01          # How similar consecutive mimic obs must be
    STATIC_WRIST_BLEND = 0.9         # Smooth wrist actions when static
    STATIC_WRIST_SCALE = 0.1         # Scale wrist actions by this when static
    
    def __init__(self, xml_path, policy_path, device='cuda'):
        self.device = device
        
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
            print(f"[V6.3] Loaded ONNX policy from {policy_path}")
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
        
        # Observation structure (same as V6.2)
        self.n_mimic_obs = 35
        self.n_proprio = 92
        self.n_obs_single = 127
        self.history_len = 10
        self.n_future_obs = 105
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_future_obs
        
        print(f"[V6.3] Observation size: {self.total_obs_size}")
        
        # History buffer
        self.proprio_history_buf = deque(maxlen=self.history_len)
        self._history_initialized = False
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_mimic = None  # For static detection
        self.static_counter = 0  # Count consecutive static frames
        self.smoothed_wrist_action = np.zeros(6, dtype=np.float32)  # For wrist smoothing
        
        # Control parameters
        self.sim_dt = 0.001
        self.train_sim_dt = 0.002
        self.train_decimation = 10
        self.control_dt = self.train_decimation * self.train_sim_dt
        self.sim_decimation = int(self.control_dt / self.sim_dt)
        self.action_scale = 0.5
        
        # PD gains - V6.3: Baseline + wrist dampening when static
        self.kps = np.array([100, 100, 100, 150, 40, 40,    # Left leg
                            100, 100, 100, 150, 40, 40,     # Right leg  
                            150, 150, 150,                   # Waist
                            40, 40, 40, 40, 4.0, 4.0, 4.0,  # Left arm
                            40, 40, 40, 40, 4.0, 4.0, 4.0], dtype=np.float32)  # Right arm
        self.kds = np.array([2, 2, 2, 4, 2, 2,              # Left leg
                            2, 2, 2, 4, 2, 2,               # Right leg
                            4, 4, 4,                         # Waist
                            5, 5, 5, 5, 2.0, 2.0, 2.0,      # Left arm
                            5, 5, 5, 5, 2.0, 2.0, 2.0], dtype=np.float32)  # Right arm
        
        print(f"[V6.3] CONFIG: Wrist dampening when static")
        print(f"  - Static threshold: {self.STATIC_THRESHOLD}")
        print(f"  - Wrist blend: {self.STATIC_WRIST_BLEND}")
        print(f"  - Wrist scale: {self.STATIC_WRIST_SCALE}")
        
        self.redis_client = None
    
    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
    
    def _detect_static(self, action_mimic, future_obs):
        """Detect if we're in static pose (motion server stopped or standing still)."""
        if not self.ENABLE_STATIC_DETECTION:
            return False
            
        # Check if future is just tiled current
        if len(future_obs) >= 35:
            future_frame_1 = future_obs[:35]
            diff = np.abs(action_mimic - future_frame_1).mean()
            if diff < self.STATIC_THRESHOLD:
                return True
        
        # Check if mimic hasn't changed
        if self.last_mimic is not None:
            diff = np.abs(action_mimic - self.last_mimic).mean()
            if diff < self.STATIC_THRESHOLD:
                return True
        
        return False
    
    def run(self, redis_ip='localhost', robot_type='unitree_g1_with_hands'):
        self.redis_client = redis.Redis(host=redis_ip, port=6379, db=0)
        self.redis_client.ping()
        print(f"[V6.3] Connected to Redis at {redis_ip}")
        
        self.reset_sim()
        self.data.qpos[2] = 0.78
        mujoco.mj_forward(self.model, self.data)
        
        viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        
        print(f"[V6.3] Starting control loop...")
        
        step_count = 0
        static_count = 0
        
        try:
            while viewer.is_running():
                t0 = time.time()
                
                # Get robot state
                qpos = self.data.qpos.copy()
                qvel = self.data.qvel.copy()
                
                dof_pos = qpos[7:7+self.num_actions]
                dof_vel = qvel[6:6+self.num_actions]
                quat = qpos[3:7]
                ang_vel = qvel[3:6]
                rpy = quatToEuler(quat)
                
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
                
                # Detect static pose
                is_static = self._detect_static(action_mimic, future_obs)
                self.last_mimic = action_mimic.copy()
                
                # Build observation
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
                
                # === V6.3 WRIST DAMPENING WHEN STATIC ===
                # Key: Keep body actions for balance, only dampen wrists
                if is_static:
                    self.static_counter += 1
                    static_count += 1
                else:
                    self.static_counter = 0
                
                if self.static_counter > 5:  # After 5 static frames
                    # Smooth and scale wrist actions only
                    wrist_actions = raw_action[self.WRIST_INDICES].copy()
                    scaled_wrist = wrist_actions * self.STATIC_WRIST_SCALE
                    self.smoothed_wrist_action = (self.STATIC_WRIST_BLEND * self.smoothed_wrist_action + 
                                                   (1 - self.STATIC_WRIST_BLEND) * scaled_wrist)
                    raw_action[self.WRIST_INDICES] = self.smoothed_wrist_action
                else:
                    self.smoothed_wrist_action = raw_action[self.WRIST_INDICES].copy()
                
                target_dof_pos = raw_action * self.action_scale + self.default_dof_pos
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
                
                # Progress
                if step_count % 100 == 0:
                    static_pct = 100 * static_count / step_count
                    print(f"[V6.3] Step {step_count} | Static: {static_pct:.1f}%")
                    print(f"  wrist_L: {raw_action[18:21]} | wrist_R: {raw_action[25:28]}")
                
                # Real-time pacing
                elapsed = time.time() - t0
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)
        
        except KeyboardInterrupt:
            print("\n[V6.3] Interrupted")
        finally:
            viewer.close()
            print(f"[V6.3] Ended after {step_count} steps")


def main():
    parser = argparse.ArgumentParser(description="V6.3 Sim2Sim with Wrist Fixes")
    parser.add_argument("--xml", type=str, required=True)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--redis_ip", type=str, default="localhost")
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands")
    args = parser.parse_args()
    
    controller = SimControllerV6_3(
        xml_path=args.xml,
        policy_path=args.policy,
        device=args.device
    )
    
    controller.run(redis_ip=args.redis_ip, robot_type=args.robot)


if __name__ == "__main__":
    main()

