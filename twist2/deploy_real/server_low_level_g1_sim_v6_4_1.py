"""
Server Low Level G1 Sim V6.4.1 - Privileged Predictor Student

Input: 1397 dims (127 current + 127*10 history)
NO future observations - predictor estimates privileged info from history
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

    def __call__(self, obs_tensor):
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        return torch.from_numpy(np.asarray(result, dtype=np.float32))


class SimControllerV6_4_1:
    """V6.4.1 controller - Privileged Predictor (NO future obs needed)."""
    
    WRIST_INDICES = [18, 19, 20, 25, 26, 27]
    
    # Static detection parameters (from V6.3)
    ENABLE_STATIC_DETECTION = True
    STATIC_THRESHOLD = 0.01
    STATIC_WRIST_BLEND = 0.9
    STATIC_WRIST_SCALE = 0.1
    
    def __init__(self, xml_path, policy_path, device='cuda'):
        self.device = device
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        if policy_path.endswith('.onnx'):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
            session = ort.InferenceSession(policy_path, providers=providers)
            input_name = session.get_inputs()[0].name
            self.policy = OnnxPolicyWrapper(session, input_name)
            print(f"[V6.4.1] Loaded ONNX policy from {policy_path}")
        else:
            self.policy = torch.jit.load(policy_path, map_location=device)
        
        self.num_actions = 29
        self.default_dof_pos = np.array([
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
            0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        self.ankle_idx = [4, 5, 10, 11]
        
        # V6.4.1 observation structure (NO future obs!)
        self.n_mimic_obs = 35
        self.n_proprio = 92
        self.n_obs_single = 127
        self.history_len = 10
        self.total_obs_size = self.n_obs_single * (self.history_len + 1)  # 1397
        
        print(f"[V6.4.1] Observation size: {self.total_obs_size} (NO future obs)")
        
        self.proprio_history_buf = deque(maxlen=self.history_len)
        self._history_initialized = False
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_mimic = None
        self.static_counter = 0
        self.smoothed_wrist_action = np.zeros(6, dtype=np.float32)
        
        self.sim_dt = 0.001
        self.control_dt = 0.02
        self.sim_decimation = 20
        self.action_scale = 0.5
        
        # PD gains (baseline)
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
        
        print(f"[V6.4.1] Wrist dampening when static: blend={self.STATIC_WRIST_BLEND}, scale={self.STATIC_WRIST_SCALE}")
        
        self.redis_client = None
    
    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
    
    def _is_static(self, action_mimic):
        if not self.ENABLE_STATIC_DETECTION:
            return False
        if self.last_mimic is not None:
            diff = np.abs(action_mimic - self.last_mimic).mean()
            return diff < self.STATIC_THRESHOLD
        return False
    
    def run(self, redis_ip='localhost', robot_type='unitree_g1_with_hands'):
        self.redis_client = redis.Redis(host=redis_ip, port=6379, db=0)
        self.redis_client.ping()
        print(f"[V6.4.1] Connected to Redis")
        
        self.reset_sim()
        self.data.qpos[2] = 0.78
        mujoco.mj_forward(self.model, self.data)
        
        viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        
        print(f"[V6.4.1] Starting control loop (NO motion server needed for standing)")
        
        step_count = 0
        static_count = 0
        
        try:
            while viewer.is_running():
                t0 = time.time()
                
                qpos = self.data.qpos.copy()
                qvel = self.data.qvel.copy()
                
                dof_pos = qpos[7:7+self.num_actions]
                dof_vel = qvel[6:6+self.num_actions]
                quat = qpos[3:7]
                ang_vel = qvel[3:6]
                rpy = quatToEuler(quat)
                
                obs_body_dof_vel = dof_vel.copy()
                obs_body_dof_vel[self.ankle_idx] = 0.0
                
                obs_proprio = np.concatenate([
                    ang_vel * 0.25,
                    rpy[:2],
                    (dof_pos - self.default_dof_pos),
                    obs_body_dof_vel * 0.05,
                    self.last_action
                ])
                
                # Fetch mimic obs from Redis (just for tracking target)
                body_data = self.redis_client.get(f"action_body_{robot_type}")
                if body_data is not None:
                    action_mimic = np.array(json.loads(body_data), dtype=np.float32)
                else:
                    action_mimic = np.zeros(self.n_mimic_obs, dtype=np.float32)
                
                # Static detection
                is_static = self._is_static(action_mimic)
                self.last_mimic = action_mimic.copy()
                
                if is_static:
                    self.static_counter += 1
                    static_count += 1
                else:
                    self.static_counter = 0
                
                # Build observation (NO future obs for V6.4.1!)
                obs_full = np.concatenate([action_mimic, obs_proprio])  # 127
                
                if not self._history_initialized:
                    for _ in range(self.history_len):
                        self.proprio_history_buf.append(obs_full.copy())
                    self._history_initialized = True
                
                obs_hist = np.array(self.proprio_history_buf).flatten()  # 1270
                self.proprio_history_buf.append(obs_full)
                
                # V6.4.1: current + history only (NO future)
                obs_buf = np.concatenate([obs_full, obs_hist])  # 1397
                
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).clip(-100, 100)
                if 'cuda' in self.device:
                    obs_tensor = obs_tensor.to(self.device)
                
                with torch.no_grad():
                    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                raw_action = np.clip(raw_action, -100, 100)
                
                # Wrist dampening when static
                if self.static_counter > 5:
                    wrist_actions = raw_action[self.WRIST_INDICES].copy() * self.STATIC_WRIST_SCALE
                    self.smoothed_wrist_action = (self.STATIC_WRIST_BLEND * self.smoothed_wrist_action + 
                                                   (1 - self.STATIC_WRIST_BLEND) * wrist_actions)
                    raw_action[self.WRIST_INDICES] = self.smoothed_wrist_action
                else:
                    self.smoothed_wrist_action = raw_action[self.WRIST_INDICES].copy()
                
                target_dof_pos = raw_action * self.action_scale + self.default_dof_pos
                self.last_action = raw_action.copy()
                
                for _ in range(self.sim_decimation):
                    current_dof_pos = self.data.qpos[7:7+self.num_actions].copy()
                    current_dof_vel = self.data.qvel[6:6+self.num_actions].copy()
                    torque = self.kps * (target_dof_pos - current_dof_pos) - self.kds * current_dof_vel
                    self.data.ctrl[:self.num_actions] = torque
                    mujoco.mj_step(self.model, self.data)
                
                viewer.sync()
                step_count += 1
                
                if step_count % 100 == 0:
                    waist_z = self.data.qpos[2]
                    static_pct = 100 * static_count / step_count
                    print(f"[V6.4.1] Step {step_count} | waist_z={waist_z:.3f} | static={static_pct:.1f}%")
                
                elapsed = time.time() - t0
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)
        
        except KeyboardInterrupt:
            print("\n[V6.4.1] Interrupted")
        finally:
            viewer.close()
            print(f"[V6.4.1] Ended after {step_count} steps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--redis_ip", type=str, default="localhost")
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands")
    args = parser.parse_args()
    
    controller = SimControllerV6_4_1(
        xml_path=args.xml,
        policy_path=args.policy,
        device=args.device
    )
    
    controller.run(redis_ip=args.redis_ip, robot_type=args.robot)


if __name__ == "__main__":
    main()


