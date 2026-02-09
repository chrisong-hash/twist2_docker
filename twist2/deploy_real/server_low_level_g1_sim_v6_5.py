"""
V6.5 Sim2Sim Controller

V6.5 = Default student architecture with jerk penalty during training
Observation: 1397 dims = 127 × 11 (current + 10 history, NO separate future)

This is different from:
- Default sim2sim (1432 dims): has extra 35 future dims
- V6.3 (1502 dims): has 105 future dims (3 timesteps)
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


class SimControllerV6_5:
    def __init__(self, xml_path, policy_path, device='cuda'):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(policy_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.obs_size = self.session.get_inputs()[0].shape[1]
        print(f"[V6.5] Loaded policy: {policy_path}")
        print(f"[V6.5] Observation size: {self.obs_size}")
        
        # Verify expected size
        expected_size = 127 * 11  # 1397
        if self.obs_size != expected_size:
            print(f"[V6.5] WARNING: Expected {expected_size}, got {self.obs_size}")
        
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
        
        # History buffer
        self.history_len = 10
        self.proprio_history_buf = deque(maxlen=self.history_len)
        
        # Timing - 100Hz
        self.sim_dt = float(self.model.opt.timestep)
        self.sim_decimation = 10
        
        print(f"[V6.5] Physics dt: {self.sim_dt}, decimation: {self.sim_decimation}")
        print(f"[V6.5] Architecture: 127×11 = 1397 dims (NO separate future)")
    
    def run(self, redis_ip='localhost', robot_type='unitree_g1_with_hands'):
        redis_client = redis.Redis(host=redis_ip, port=6379)
        print(f"[V6.5] Connected to Redis at {redis_ip}")
        
        # Initialize simulation
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.mujoco_init_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        
        last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.proprio_history_buf.clear()
        history_initialized = False
        
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        print("[V6.5] Viewer started. Waiting for motion data...")
        
        step_count = 0
        
        try:
            while viewer.is_running():
                body_data = redis_client.get(f"action_body_{robot_type}")
                
                if body_data is None:
                    viewer.sync()
                    time.sleep(0.01)
                    continue
                
                step_count += 1
                
                # Get current state
                qpos = self.data.qpos.copy()
                qvel = self.data.qvel.copy()
                dof_pos = qpos[7:7+self.num_actions]
                dof_vel = qvel[6:6+self.num_actions]
                ang_vel = qvel[3:6]
                rpy = quatToEuler(qpos[3:7])
                
                obs_body_dof_vel = dof_vel.copy()
                obs_body_dof_vel[self.ankle_idx] = 0.0
                
                # Build proprioceptive observation (92 dims)
                obs_proprio = np.concatenate([
                    ang_vel * 0.25,           # 3
                    rpy[:2],                  # 2
                    dof_pos - self.default_dof_pos,  # 29
                    obs_body_dof_vel * 0.05,  # 29
                    last_action               # 29
                ])  # Total: 92
                
                # Get mimic observation (35 dims)
                action_mimic = np.array(json.loads(body_data), dtype=np.float32)
                
                # Build current obs (127 = 35 + 92)
                obs_full = np.concatenate([action_mimic, obs_proprio])
                
                # Initialize history
                if not history_initialized:
                    for _ in range(self.history_len):
                        self.proprio_history_buf.append(obs_full.copy())
                    history_initialized = True
                
                # Get history BEFORE appending current
                obs_hist = np.array(self.proprio_history_buf).flatten()  # 10 × 127 = 1270
                self.proprio_history_buf.append(obs_full)
                
                # V6.5: NO separate future obs - just current + history
                # 127 (current) + 1270 (history) = 1397
                obs_buf = np.concatenate([obs_full, obs_hist])
                
                # Ensure correct size
                if len(obs_buf) < self.obs_size:
                    obs_buf = np.concatenate([obs_buf, np.zeros(self.obs_size - len(obs_buf))])
                obs_buf = obs_buf[:self.obs_size]
                
                # Run policy
                obs_tensor = obs_buf.astype(np.float32).reshape(1, -1)
                raw_action = self.session.run(None, {self.input_name: obs_tensor})[0].squeeze()
                raw_action = np.clip(raw_action, -10, 10)
                
                # Compute target position
                target_dof_pos = raw_action * 0.5 + self.default_dof_pos
                last_action = raw_action.copy()
                
                # Run physics at 1000Hz
                for _ in range(self.sim_decimation):
                    cur_pos = self.data.qpos[7:7+self.num_actions].copy()
                    cur_vel = self.data.qvel[6:6+self.num_actions].copy()
                    torque = self.kps * (target_dof_pos - cur_pos) - self.kds * cur_vel
                    self.data.ctrl[:self.num_actions] = torque
                    mujoco.mj_step(self.model, self.data)
                
                viewer.sync()
                
                if step_count % 100 == 0:
                    waist_z = self.data.qpos[2]
                    print(f"[V6.5] Step {step_count} | Waist Z: {waist_z:.3f}")
                    
        finally:
            viewer.close()
            print(f"[V6.5] Ended after {step_count} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--redis_ip', type=str, default='localhost')
    parser.add_argument('--robot', type=str, default='unitree_g1_with_hands')
    args = parser.parse_args()
    
    controller = SimControllerV6_5(args.xml, args.policy, args.device)
    controller.run(args.redis_ip, args.robot)


