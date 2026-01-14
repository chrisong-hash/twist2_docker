"""
GearWbcPolicy - GROOT Whole Body Control Policy Wrapper

This policy provides stable locomotion even with arbitrary upper body poses.
It uses two ONNX models:
- Balance policy: for standing/small velocity commands
- Walk policy: for walking/movement

Key feature: Takes torso orientation RPY as input to compensate for upper body movements.
"""

import os
import collections
from typing import Optional

import numpy as np
import yaml

try:
    import onnxruntime as ort
except ImportError:
    print("[GearWBC] Warning: onnxruntime not installed, policy will not work")
    ort = None

try:
    import torch
except ImportError:
    print("[GearWBC] Warning: torch not installed, policy will not work")
    torch = None


def get_gravity_orientation(quat):
    """Get gravity vector in body frame from quaternion [w, x, y, z]"""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    # Conjugate quaternion
    q_conj = np.array([w, -x, -y, -z])
    
    # Gravity in world frame
    gravity_vec = np.array([0.0, 0.0, -1.0])
    
    # Rotate gravity by inverse of quaternion
    w, x, y, z = q_conj
    return np.array([
        gravity_vec[0] * (w**2 + x**2 - y**2 - z**2)
        + gravity_vec[1] * 2 * (x * y - w * z)
        + gravity_vec[2] * 2 * (x * z + w * y),
        gravity_vec[0] * 2 * (x * y + w * z)
        + gravity_vec[1] * (w**2 - x**2 + y**2 - z**2)
        + gravity_vec[2] * 2 * (y * z - w * x),
        gravity_vec[0] * 2 * (x * z - w * y)
        + gravity_vec[1] * 2 * (y * z + w * x)
        + gravity_vec[2] * (w**2 - x**2 - y**2 + z**2),
    ])


class GearWbcPolicy:
    """
    GROOT Whole Body Control Policy for G1 robot locomotion.
    
    This policy can stabilize the robot regardless of upper body pose because
    it takes torso orientation as input and compensates for it.
    
    Joint ordering (15 DOF):
        0-5: Left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll)
        6-11: Right leg (same order)
        12-14: Waist (yaw, roll, pitch) - Note: GROOT uses this ordering
    """
    
    def __init__(self, policy_dir: str = None):
        """Initialize the GearWBC policy.
        
        Args:
            policy_dir: Path to directory containing ONNX models and config.
                       Defaults to twist2/assets/ckpts/gear_wbc/
        """
        if policy_dir is None:
            # Default path relative to this file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            policy_dir = os.path.join(script_dir, "..", "assets", "ckpts", "gear_wbc")
        
        self.policy_dir = policy_dir
        
        # Load config
        config_path = os.path.join(policy_dir, "g1_gear_wbc.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Convert config arrays to numpy
        array_keys = ["kps", "kds", "default_angles", "cmd_scale", "cmd_init"]
        for key in array_keys:
            if key in self.config:
                self.config[key] = np.array(self.config[key], dtype=np.float32)
        
        # Load ONNX models
        balance_path = os.path.join(policy_dir, "GR00T-WholeBodyControl-Balance.onnx")
        walk_path = os.path.join(policy_dir, "GR00T-WholeBodyControl-Walk.onnx")
        
        self.balance_policy = self._load_onnx_policy(balance_path)
        self.walk_policy = self._load_onnx_policy(walk_path)
        
        # Initialize state
        self.num_actions = self.config["num_actions"]  # 15
        self.num_obs = self.config["num_obs"]  # 516
        self.obs_history_len = self.config["obs_history_len"]  # 6
        
        self.obs_history = collections.deque(maxlen=self.obs_history_len)
        self.obs_buffer = np.zeros(self.num_obs, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        
        # Command state
        self.cmd = self.config["cmd_init"].copy()  # [vx, vy, vyaw]
        self.height_cmd = self.config["height_cmd"]  # 0.74
        self.roll_cmd = self.config["rpy_cmd"][0]
        self.pitch_cmd = self.config["rpy_cmd"][1]
        self.yaw_cmd = self.config["rpy_cmd"][2]
        self.freq_cmd = self.config["freq_cmd"]
        
        # Gait phase tracking
        self.gait_indices = np.zeros(1, dtype=np.float32)
        
        # Default angles (legs + waist, 15 DOF)
        self.default_angles = self.config["default_angles"].copy()
        
        # PD gains
        self.kps = self.config["kps"].copy()
        self.kds = self.config["kds"].copy()
        
        # Reorder for our joint convention (if needed)
        # GROOT order: same as ours for legs+waist
        self.default_angles_reorder = np.zeros(29, dtype=np.float32)
        self.default_angles_reorder[:15] = self.default_angles
        
        # Full 29-DOF PD gains (use config for legs+waist, default for arms)
        self.full_kps = np.zeros(29, dtype=np.float32)
        self.full_kds = np.zeros(29, dtype=np.float32)
        self.full_kps[:15] = self.kps
        self.full_kds[:15] = self.kds
        # Default arm gains (can be overridden by caller)
        self.full_kps[15:] = 100.0
        self.full_kds[15:] = 2.0
        
        print(f"[GearWBC] Policy loaded from {policy_dir}")
        print(f"[GearWBC] Balance policy: standing/small commands")
        print(f"[GearWBC] Walk policy: movement commands")
        print(f"[GearWBC] Num actions: {self.num_actions}, Obs dim: {self.num_obs}")
    
    def _load_onnx_policy(self, model_path: str):
        """Load ONNX policy and return inference function."""
        print(f"[GearWBC] Loading ONNX policy from {model_path}")
        session = ort.InferenceSession(model_path)
        
        def run_inference(input_tensor):
            if isinstance(input_tensor, torch.Tensor):
                input_np = input_tensor.cpu().numpy()
            else:
                input_np = input_tensor
            ort_inputs = {session.get_inputs()[0].name: input_np}
            ort_outs = session.run(None, ort_inputs)
            return ort_outs[0]
        
        return run_inference
    
    def reset(self):
        """Reset policy state - call when switching to GearWBC mode."""
        self.obs_history.clear()
        self.obs_buffer = np.zeros(self.num_obs, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.gait_indices = np.zeros(1, dtype=np.float32)
        self.cmd = self.config["cmd_init"].copy()
        print("[GearWBC] Policy reset for clean mode switch")
    
    def _compute_observation(self, qj, dqj, quat, ang_vel, torso_rpy=None):
        """Compute single observation vector (86 dim).
        
        Args:
            qj: Joint positions (15 DOF - legs + waist)
            dqj: Joint velocities (15 DOF)
            quat: Base orientation quaternion [w, x, y, z]
            ang_vel: Base angular velocity [wx, wy, wz]
            torso_rpy: Torso orientation relative to waist [roll, pitch, yaw]
        """
        n_joints = len(qj)
        
        # Update gait phase
        self.gait_indices = np.fmod(self.gait_indices + 0.02 * self.freq_cmd, 1.0)
        
        # Compute clock inputs (not used in current observation but kept for compatibility)
        durations = 0.5
        phases = 0.5
        foot_indices = [
            np.fmod(self.gait_indices + phases, 1.0),  # FL
            self.gait_indices.copy(),  # FR
        ]
        
        # Scale values
        qj_scaled = (qj - self.default_angles[:n_joints]) * self.config["dof_pos_scale"]
        dqj_scaled = dqj * self.config["dof_vel_scale"]
        gravity_ori = get_gravity_orientation(quat)
        omega_scaled = ang_vel * self.config["ang_vel_scale"]
        
        # Use torso orientation if provided, otherwise use defaults
        if torso_rpy is not None:
            roll_cmd = torso_rpy[0]
            pitch_cmd = torso_rpy[1]
            yaw_cmd = torso_rpy[2]
        else:
            roll_cmd = self.roll_cmd
            pitch_cmd = self.pitch_cmd
            yaw_cmd = self.yaw_cmd
        
        # Create observation (86 dim)
        # Format: cmd(3) + height(1) + rpy(3) + omega(3) + gravity(3) + qj(29) + dqj(29) + action(15)
        single_obs = np.zeros(86, dtype=np.float32)
        single_obs[0:3] = self.cmd[:3] * self.config["cmd_scale"]
        single_obs[3:4] = self.height_cmd
        single_obs[4:7] = np.array([roll_cmd, pitch_cmd, yaw_cmd])
        single_obs[7:10] = omega_scaled
        single_obs[10:13] = gravity_ori
        
        # Pad joint positions/velocities to 29 DOF (legs+waist=15, arms=14 zeros)
        qj_padded = np.zeros(29, dtype=np.float32)
        dqj_padded = np.zeros(29, dtype=np.float32)
        qj_padded[:n_joints] = qj_scaled
        dqj_padded[:n_joints] = dqj_scaled
        
        single_obs[13:42] = qj_padded
        single_obs[42:71] = dqj_padded
        single_obs[71:86] = self.action
        
        return single_obs
    
    def compute(self, qj, dqj, ang_vel, quat, vel_cmd, torso_rpy=None):
        """Compute lower body joint positions from velocity command and robot state.
        
        Args:
            qj: Current joint positions (at least first 15 for legs+waist)
            dqj: Current joint velocities
            ang_vel: Base angular velocity [wx, wy, wz]
            quat: Base orientation quaternion [w, x, y, z]
            vel_cmd: Velocity command [vx, vy, vyaw]
            torso_rpy: Optional torso orientation for compensation [roll, pitch, yaw]
        
        Returns:
            action: Target joint positions for legs+waist (15 DOF)
            kps: Position gains (15 DOF)
            kds: Velocity gains (15 DOF)
        """
        # Update velocity command
        self.cmd = np.array(vel_cmd, dtype=np.float32)
        
        # Extract legs+waist joints
        qj_lower = qj[:15].astype(np.float32) if len(qj) >= 15 else np.zeros(15, dtype=np.float32)
        dqj_lower = dqj[:15].astype(np.float32) if len(dqj) >= 15 else np.zeros(15, dtype=np.float32)
        
        # Compute observation
        single_obs = self._compute_observation(qj_lower, dqj_lower, quat, ang_vel, torso_rpy)
        
        # Update history
        self.obs_history.append(single_obs)
        while len(self.obs_history) < self.obs_history_len:
            self.obs_history.appendleft(np.zeros_like(single_obs))
        
        # Build full observation with history
        for i, hist_obs in enumerate(self.obs_history):
            start_idx = i * len(single_obs)
            end_idx = start_idx + len(single_obs)
            self.obs_buffer[start_idx:end_idx] = hist_obs
        
        # Select policy based on velocity command magnitude
        cmd_magnitude = np.linalg.norm(self.cmd)
        if cmd_magnitude < 0.05:
            policy = self.balance_policy
        else:
            policy = self.walk_policy
        
        # Run inference
        obs_tensor = self.obs_buffer.reshape(1, -1).astype(np.float32)
        self.action = policy(obs_tensor).squeeze()
        
        # Convert action to target joint positions
        target_dof_pos = self.action * self.config["action_scale"] + self.default_angles
        
        return target_dof_pos, self.kps, self.kds


if __name__ == "__main__":
    # Quick test
    print("Testing GearWbcPolicy...")
    policy = GearWbcPolicy()
    
    # Dummy inputs
    qj = np.zeros(29, dtype=np.float32)
    dqj = np.zeros(29, dtype=np.float32)
    ang_vel = np.zeros(3, dtype=np.float32)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity
    vel_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    action, kps, kds = policy.compute(qj, dqj, ang_vel, quat, vel_cmd)
    print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
    print("Test passed!")


