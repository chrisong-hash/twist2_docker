#!/usr/bin/env python3
"""
Merged Teleop with dex-retargeting - Proper IK-based finger retargeting
========================================================================
Same as teleop_merged.py but uses dex-retargeting library for proper
inverse kinematics hand retargeting instead of distance-based calculations.

Benefits over distance-based approach:
- Proper thumb bend vs rotation separation
- Full range finger closure
- No manual calibration needed
- IK-optimized joint angles

UNITREE CONTROLLER MAPPING (read from Redis):
=============================================
  A       : Cycle states (idle → preview → teleop → pause)
  B       : Toggle WALK ↔ BALANCE mode
  X       : Toggle UPPER BODY (arms+hands) freeze
  Select  : EMERGENCY SHUTDOWN

PICO VR:
========
  Body tracking : Full body motion capture
  Hand tracking : Finger positions → Inspire hands via dex-retargeting
"""

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional

# RoboMimic path (mounted in Docker container)
ROBOMIMIC_PATH = "/workspace/RoboMimic_Deploy"


def check_setup(verbose=False):
    """
    Verify all dependencies and files are available before starting.
    Returns True if all checks pass, False otherwise.
    """
    errors = []
    warnings = []
    checks = []
    
    def status(msg):
        if verbose:
            print(msg)
        else:
            sys.stdout.write(f"\r[Setup] {msg}...".ljust(60))
            sys.stdout.flush()
    
    # 1. RoboMimic_Deploy
    status("Checking RoboMimic_Deploy")
    if os.path.isdir(ROBOMIMIC_PATH):
        checks.append("✓")
    else:
        checks.append("✗")
        errors.append(f"RoboMimic_Deploy not found: {ROBOMIMIC_PATH}\n   Fix: Mount in docker-compose.yml")
    
    # 2. LocoMode policy
    status("Checking LocoMode policy")
    policy_dir = os.path.join(ROBOMIMIC_PATH, "policy", "loco_mode")
    config_path = os.path.join(policy_dir, "config", "LocoMode.yaml")
    if os.path.isfile(config_path):
        try:
            import yaml
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            model_path = os.path.join(policy_dir, "model", config.get("policy_path", ""))
            if os.path.isfile(model_path):
                checks.append("✓")
            else:
                checks.append("✗")
                errors.append(f"LocoMode model missing: {model_path}")
        except Exception as e:
            checks.append("✗")
            errors.append(f"LocoMode config error: {e}")
    else:
        checks.append("✗")
        errors.append(f"LocoMode config missing: {config_path}")
    
    # 3. PyTorch
    status("Checking PyTorch")
    try:
        import torch
        checks.append("✓")
        if not torch.cuda.is_available():
            warnings.append("CUDA not available - CPU only")
    except ImportError:
        checks.append("✗")
        errors.append("PyTorch not installed")
    
    # 4. MuJoCo
    status("Checking MuJoCo")
    try:
        import mujoco
        checks.append("✓")
    except ImportError:
        checks.append("✗")
        errors.append("MuJoCo not installed")
    
    # 5. GMR
    status("Checking GMR")
    try:
        from general_motion_retargeting import GeneralMotionRetargeting, XRobotStreamer, ROBOT_XML_DICT
        if "unitree_g1" in ROBOT_XML_DICT and os.path.isfile(ROBOT_XML_DICT["unitree_g1"]):
            checks.append("✓")
        else:
            checks.append("✗")
            errors.append("G1 robot XML not found in GMR")
    except ImportError as e:
        checks.append("✗")
        errors.append(f"GMR import failed: {e}")
    
    # 6. Redis
    status("Checking Redis")
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        checks.append("✓")
    except ImportError:
        checks.append("✗")
        errors.append("redis-py not installed")
    except:
        checks.append("⚠")
        warnings.append("Redis not running (OK for preview mode)")
    
    # 7. dex-retargeting
    status("Checking dex-retargeting")
    try:
        from dex_retargeting.constants import HandType, RetargetingType, RobotName
        from dex_retargeting.retargeting_config import RetargetingConfig
        checks.append("✓")
    except ImportError as e:
        checks.append("✗")
        errors.append(f"dex-retargeting not installed: {e}")
    
    # 8. Inspire hand URDF
    status("Checking Inspire hand URDF")
    urdf_path = Path(__file__).parent / "inspire_hand" / "inspire_hand_right.urdf"
    if urdf_path.exists():
        checks.append("✓")
    else:
        checks.append("✗")
        errors.append(f"Inspire hand URDF missing: {urdf_path}")
    
    # 9. Other deps
    status("Checking dependencies")
    try:
        import yaml, numpy
        from loop_rate_limiters import RateLimiter
        checks.append("✓")
    except ImportError as e:
        checks.append("✗")
        errors.append(f"Missing dependency: {e}")
    
    sys.stdout.write("\r" + " "*60 + "\r")
    sys.stdout.flush()
    
    check_str = "".join(checks)
    if errors:
        print(f"[Setup] {check_str} FAILED ({len(errors)} errors)\n")
        for err in errors:
            print(f"  ✗ {err}")
        print()
        return False
    elif warnings:
        print(f"[Setup] {check_str} OK ({len(warnings)} warnings)")
        return True
    else:
        print(f"[Setup] {check_str} OK")
        return True


# Run setup check before importing heavy modules
if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    if not check_setup(verbose=verbose):
        print("Please fix the errors above before running.\n")
        sys.exit(1)

# Now import heavy dependencies
import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
import torch
import yaml
import redis
from loop_rate_limiters import RateLimiter
from rich import print

# TWIST2 GMR imports
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import XRobotStreamer
from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting import human_head_to_robot_neck

from data_utils.params import DEFAULT_MIMIC_OBS
from data_utils.rot_utils import euler_from_quaternion_np, quat_diff_np, quat_rotate_inverse_np
from robot_control.inspire_hand_wrapper import DualHandController

# dex-retargeting imports
from dex_retargeting.constants import (
    OPERATOR2MANO,
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig

# XRoboToolkit SDK for RAW hand data (bypasses GMR coordinate transformations)
import xrobotoolkit_sdk as xrt


# ============== Pico to MediaPipe conversion (from XRoboToolkit) ==============
PICO_TO_MEDIAPIPE = {
    1: 0,   # Wrist
    2: 1,   # Thumb_metacarpal -> THUMB_CMC
    3: 2,   # Thumb_proximal   -> THUMB_MCP
    4: 3,   # Thumb_distal     -> THUMB_IP
    5: 4,   # Thumb_tip        -> THUMB_TIP
    7: 5,   # Index_proximal   -> INDEX_FINGER_MCP
    8: 6,   # Index_intermediate -> INDEX_FINGER_PIP
    9: 7,   # Index_distal     -> INDEX_FINGER_DIP
    10: 8,  # Index_tip        -> INDEX_FINGER_TIP
    12: 9,  # Middle_proximal  -> MIDDLE_FINGER_MCP
    13: 10, # Middle_intermediate -> MIDDLE_FINGER_PIP
    14: 11, # Middle_distal    -> MIDDLE_FINGER_DIP
    15: 12, # Middle_tip       -> MIDDLE_FINGER_TIP
    17: 13, # Ring_proximal    -> RING_FINGER_MCP
    18: 14, # Ring_intermediate -> RING_FINGER_PIP
    19: 15, # Ring_distal      -> RING_FINGER_DIP
    20: 16, # Ring_tip         -> RING_FINGER_TIP
    22: 17, # Little_proximal  -> PINKY_MCP
    23: 18, # Little_intermediate -> PINKY_PIP
    24: 19, # Little_distal    -> PINKY_DIP
    25: 20, # Little_tip       -> PINKY_TIP
}

# Pico joint names in order (26 joints)
PICO_JOINT_NAMES = [
    "Wrist", "Palm",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip", 
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip"
]


def pico_raw_to_mediapipe(hand_state: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert RAW Pico SDK hand array to MediaPipe 21-joint format.
    This is the EXACT conversion from XRoboToolkit dex_hand_utils.py.
    
    Args:
        hand_state: Raw SDK array of shape (26, 7) or (27, 7) where each row is
                    [x, y, z, qx, qy, qz, qw]
    
    Returns:
        MediaPipe format: (21, 3) array centered at wrist, or None if invalid
    """
    if hand_state is None:
        return None
    
    hand_state = np.array(hand_state)
    
    if len(hand_state.shape) != 2 or hand_state.shape[0] < 26 or hand_state.shape[1] < 3:
        return None
    
    if np.all(hand_state == 0):
        return None
    
    # Convert to MediaPipe format (21, 3)
    mediapipe_state = np.zeros((21, 3), dtype=float)
    for pico_idx, mediapipe_idx in PICO_TO_MEDIAPIPE.items():
        if pico_idx < hand_state.shape[0]:
            mediapipe_state[mediapipe_idx] = hand_state[pico_idx, :3]
    
    # Center at wrist
    mediapipe_state = mediapipe_state - mediapipe_state[0:1, :]
    
    return mediapipe_state


def pico_hand_dict_to_mediapipe(hand_data_dict: dict, side: str) -> Optional[np.ndarray]:
    """
    Convert Pico hand dict format to MediaPipe 21-joint format.
    NOTE: This uses the dictionary format from GMR XRobotStreamer which has
          coordinate transformations applied. For better accuracy, use
          pico_raw_to_mediapipe with raw SDK data instead.
    
    Args:
        hand_data_dict: Dict with keys like "LeftHandWrist", "LeftHandThumbTip", etc.
                        Each value is [pos, rot] where pos = [x, y, z]
        side: "left" or "right"
    
    Returns:
        MediaPipe format: (21, 3) array centered at wrist, or None if invalid
    """
    if not hand_data_dict:
        return None
    
    prefix = f"{side.capitalize()}Hand"
    
    # Build Pico state array (26, 3) - positions only
    pico_positions = np.zeros((26, 3), dtype=float)
    
    for i, joint_name in enumerate(PICO_JOINT_NAMES):
        key = prefix + joint_name
        if key in hand_data_dict:
            data = hand_data_dict[key]
            if isinstance(data, list) and len(data) >= 1:
                pos = data[0]  # [x, y, z]
                if isinstance(pos, (list, np.ndarray)) and len(pos) >= 3:
                    pico_positions[i] = [pos[0], pos[1], pos[2]]
    
    # Check if we have valid data (at least wrist)
    if np.all(pico_positions[0] == 0):
        return None
    
    # Convert to MediaPipe format (21, 3)
    mediapipe_state = np.zeros((21, 3), dtype=float)
    for pico_idx, mediapipe_idx in PICO_TO_MEDIAPIPE.items():
        if pico_idx < len(pico_positions):
            mediapipe_state[mediapipe_idx] = pico_positions[pico_idx]
    
    # Center at wrist
    mediapipe_state = mediapipe_state - mediapipe_state[0:1, :]
    
    return mediapipe_state


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Compute the 3D coordinate frame (orientation only) from detected 3d key points.
    From XRoboToolkit dex_hand_utils.py
    
    Args:
        keypoint_3d_array: MediaPipe format (21, 3) array
    
    Returns:
        3x3 rotation matrix for wrist in MANO convention
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]  # wrist, index, middle

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram–Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / (np.linalg.norm(x) + 1e-6)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame


class DexHandTracker:
    """
    Uses dex-retargeting library for proper IK-based hand retargeting.
    Directly outputs Inspire hand joint angles.
    """
    
    def __init__(self, hand_type: str, urdf_dir: str):
        """
        Args:
            hand_type: "left" or "right"
            urdf_dir: Directory containing inspire_hand_left.urdf and inspire_hand_right.urdf
        """
        self.hand_type_str = hand_type
        self.hand_type = HandType.left if hand_type == "left" else HandType.right
        self.urdf_path = os.path.join(urdf_dir, f"inspire_hand_{hand_type}.urdf")
        
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        
        # Get config path from dex-retargeting
        self.config_path = get_default_config_path(RobotName.inspire, RetargetingType.vector, self.hand_type)
        self.OPERATOR2MANO = OPERATOR2MANO[self.hand_type]
        
        # Set URDF directory
        robot_dir = Path(self.urdf_path).parent.parent
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        
        # Build retargeting module
        self.retargeting = RetargetingConfig.load_from_file(self.config_path).build()
        
        # CRITICAL: Disable internal low-pass filter (was causing stuck values!)
        # Default alpha is 0.2 which causes heavy smoothing. Set to 1.0 for instant response.
        self.retargeting.filter.alpha = 1.0
        
        # Smoothing (optional, can apply externally)
        self.prev_qpos = None
        self.smoothing_alpha = 0.3
        
    def retarget(self, hand_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Retarget MediaPipe hand positions to Inspire hand joint angles.
        
        Args:
            hand_pos: MediaPipe format (21, 3) array centered at wrist
            
        Returns:
            6 joint angles for Inspire hand: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
            Returns None if retargeting fails.
        """
        if hand_pos is None or hand_pos.shape != (21, 3):
            return self.prev_qpos
        
        try:
            # Estimate wrist orientation
            wrist_rot = estimate_frame_from_hand_points(hand_pos)
            
            # Transform to MANO frame
            transformed_pos = hand_pos @ wrist_rot @ self.OPERATOR2MANO
            
            # Prepare reference values
            indices = self.retargeting.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = transformed_pos[task_indices, :] - transformed_pos[origin_indices, :]
            
            # Retarget
            qpos = self.retargeting.retarget(ref_value)
            
            # Smooth
            if self.prev_qpos is not None:
                qpos = self.smoothing_alpha * qpos + (1 - self.smoothing_alpha) * self.prev_qpos
            self.prev_qpos = qpos
            
            return qpos
            
        except (RuntimeWarning, RuntimeError) as e:
            if self.prev_qpos is not None:
                return self.prev_qpos
            return None


# ============== Locomotion Policy ==============
class LocoModePolicy:
    """Velocity-conditioned locomotion policy from RoboMimic"""
    
    def __init__(self):
        policy_dir = os.path.join(ROBOMIMIC_PATH, "policy", "loco_mode")
        config_path = os.path.join(policy_dir, "config", "LocoMode.yaml")
        
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        self.policy_path = os.path.join(policy_dir, "model", config["policy_path"])
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        self.joint2motor_idx = np.array(config["joint2motor_idx"], dtype=np.int32)
        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]
        self.ang_vel_scale = config["ang_vel_scale"]
        self.dof_pos_scale = config["dof_pos_scale"]
        self.dof_vel_scale = config["dof_vel_scale"]
        self.action_scale = config["action_scale"]
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        
        cmd_range = config["cmd_range"]
        self.range_velx = np.array([cmd_range["lin_vel_x"][0], cmd_range["lin_vel_x"][1]], dtype=np.float32)
        self.range_vely = np.array([cmd_range["lin_vel_y"][0], cmd_range["lin_vel_y"][1]], dtype=np.float32)
        self.range_velz = np.array([cmd_range["ang_vel_z"][0], cmd_range["ang_vel_z"][1]], dtype=np.float32)
        
        self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self._needs_reset = True
        
        self.policy = torch.jit.load(self.policy_path)
        
        for _ in range(50):
            with torch.inference_mode():
                obs_tensor = self.obs.reshape(1, -1).astype(np.float32)
                self.policy(torch.from_numpy(obs_tensor))
                
        print("[green][LocoMode] Locomotion policy loaded[/green]")
        
        self.default_angles_reorder = np.zeros(29, dtype=np.float32)
        for i in range(len(self.joint2motor_idx)):
            motor_idx = self.joint2motor_idx[i]
            self.default_angles_reorder[motor_idx] = self.default_angles[i]
    
    def compute(self, qj, dqj, ang_vel, gravity_ori, vel_cmd):
        """Compute leg joint positions from velocity command"""
        vx = np.clip(vel_cmd[0], -1, 1) * (self.range_velx[1] if vel_cmd[0] > 0 else -self.range_velx[0])
        vy = np.clip(vel_cmd[1], -1, 1) * (self.range_vely[1] if vel_cmd[1] > 0 else -self.range_vely[0])
        vyaw = np.clip(vel_cmd[2], -1, 1) * (self.range_velz[1] if vel_cmd[2] > 0 else -self.range_velz[0])
        cmd = np.array([vx, vy, vyaw], dtype=np.float32) * self.cmd_scale
        
        for i in range(len(self.joint2motor_idx)):
            self.qj_obs[i] = qj[self.joint2motor_idx[i]]
            self.dqj_obs[i] = dqj[self.joint2motor_idx[i]]
        
        qj_scaled = (self.qj_obs - self.default_angles) * self.dof_pos_scale
        dqj_scaled = self.dqj_obs * self.dof_vel_scale
        ang_vel_scaled = ang_vel * self.ang_vel_scale
        
        self.obs[:3] = ang_vel_scaled
        self.obs[3:6] = gravity_ori
        self.obs[6:9] = cmd
        self.obs[9:9 + self.num_actions] = qj_scaled
        self.obs[9 + self.num_actions:9 + self.num_actions * 2] = dqj_scaled
        self.obs[9 + self.num_actions * 2:9 + self.num_actions * 3] = self.action
        
        with torch.inference_mode():
            obs_tensor = self.obs.reshape(1, -1).astype(np.float32)
            self.action = self.policy(torch.from_numpy(obs_tensor).clip(-100, 100)).clip(-100, 100).detach().numpy().squeeze()
        
        loco_action = self.action * self.action_scale + self.default_angles
        
        action_reorder = np.zeros(29, dtype=np.float32)
        for i in range(len(self.joint2motor_idx)):
            motor_idx = self.joint2motor_idx[i]
            action_reorder[motor_idx] = loco_action[i]
                
        return action_reorder
    
    def reset(self):
        """Reset action buffer"""
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self._needs_reset = False
        print("[LocoMode] Action buffer reset")


def extract_mimic_obs(qpos, last_qpos, dt=1/30):
    """Extract mimic observations from robot joint positions (35 dims)"""
    root_pos, last_root_pos = qpos[0:3], last_qpos[0:3]
    root_quat, last_root_quat = qpos[3:7], last_qpos[3:7]
    robot_joints = qpos[7:].copy()
    base_vel = (root_pos - last_root_pos) / dt
    base_ang_vel = quat_diff_np(last_root_quat, root_quat, scalar_first=True) / dt
    roll, pitch, yaw = euler_from_quaternion_np(root_quat.reshape(1, -1), scalar_first=True)
    base_vel_local = quat_rotate_inverse_np(root_quat, base_vel, scalar_first=True)
    base_ang_vel_local = quat_rotate_inverse_np(root_quat, base_ang_vel, scalar_first=True)
    
    mimic_obs = np.concatenate([
        base_vel_local[:2],
        root_pos[2:3],
        roll, pitch,
        base_ang_vel_local[2:3],
        robot_joints
    ])
    return mimic_obs


# ============== Unitree Controller Button Mapping ==============
UNITREE_BUTTONS = {
    "A": 0x0100,
    "B": 0x0200,
    "X": 0x0400,
    "Y": 0x0800,
    "R1": 0x0001,
    "L1": 0x0002,
    "start": 0x0004,
    "select": 0x0008,
    "R2": 0x0010,
    "L2": 0x0020,
    "F1": 0x0040,
    "F2": 0x0080,
    "up": 0x1000,
    "right": 0x2000,
    "down": 0x4000,
    "left": 0x8000
}


class MergedTeleopDex:
    """
    Merged teleop with dex-retargeting for proper IK-based finger tracking.
    - Body tracking from Pico VR (GMR)
    - Finger tracking from Pico VR (dex-retargeting)
    - State control from Unitree controller (via Redis)
    """
    
    INTERP_DURATION = 0.25
    
    # Auto-balance on backward stop: when backward walking stops, auto-switch to balance mode
    BACKWARD_STOP_THRESHOLD = 0.05  # velocity below this = stopped
    
    DEFAULT_STANDING_LEGS = np.array([
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
    ], dtype=np.float32)
    
    DEFAULT_STANDING_JOINTS = np.array([
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
        0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
    ], dtype=np.float32)
    
    def __init__(self, args):
        self.args = args
        self.robot_name = "unitree_g1"
        
        # State machine
        self.state = "idle"
        self.previous_teleop_state = "teleop_full"
        self.paused_qpos = None
        self.paused_mimic_obs = None
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        
        # Interpolation state
        self.is_interpolating = False
        self.interp_start_time = 0.0
        self.interp_start_qpos = None
        self.interp_target_qpos = None
        self.interp_from_state = None
        self.interp_to_state = None
        
        # Unitree controller state (read from Redis)
        self._prev_unitree_keys = 0
        
        # Upper body freeze state
        self.hands_paused = False
        self.frozen_arm_obs = None
        self._capture_frozen_arms = False  # Flag to capture arm positions on next frame
        
        # Auto-balance on backward stop
        self._was_walking_backward = False  # Track if we were walking backward
        
        # Inspire hands
        self.use_inspire_hands = getattr(args, 'use_inspire_hands', False)
        self.inspire_hand_controller = None
        
        # dex-retargeting hand trackers
        self.dex_left_tracker = None
        self.dex_right_tracker = None
        
        # Height and retargeting
        self.estimated_height = args.actual_human_height
        self.retarget = None
        
        # Smooth filtering
        self.enable_smooth = args.smooth
        self.smooth_window_size = args.smooth_window_size
        self.smooth_history = []
        
        # Initialize systems
        print("\n[cyan]Initializing Merged Teleop (dex-retargeting)...[/cyan]")
        self._setup_locomotion_policy()
        self._setup_teleop_streamer()
        self._setup_retargeting()
        self._setup_mujoco()
        self._setup_redis()
        self._setup_inspire_hands()
        self._setup_dex_hand_trackers()
        
        print("\n[green]Systems initialized![/green]")
        self._print_controls()
    
    def _setup_locomotion_policy(self):
        print("\n[1/7] Loading locomotion policy...")
        self.loco_policy = LocoModePolicy()
    
    def _setup_teleop_streamer(self):
        print("\n[2/7] Connecting to Pico VR...")
        try:
            self.teleop_streamer = XRobotStreamer()
            print("[green]XRobotStreamer initialized[/green]")
        except Exception as e:
            print(f"[red]ERROR: Failed to initialize XRobotStreamer: {e}[/red]")
            self.teleop_streamer = None
    
    def _setup_retargeting(self, height=None):
        print("\n[3/7] Setting up GMR retargeting...")
        if height is None:
            height = self.estimated_height
        
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self.retarget = GMR(
                src_human="xrobot",
                tgt_robot="unitree_g1",
                actual_human_height=height,
            )
        finally:
            sys.stdout = old_stdout
        print(f"[green]GMR initialized (height: {height:.2f}m)[/green]")
    
    def _setup_mujoco(self):
        print("\n[4/7] Setting up MuJoCo preview...")
        xml_path = str(ROBOT_XML_DICT["unitree_g1"])
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.last_qpos = None
        self._last_valid_qpos = None
        self._using_fallback = False
        self.robot_base_id = self.model.body("pelvis").id
        print("[green]MuJoCo ready[/green]")
    
    def _setup_redis(self):
        print("\n[5/7] Connecting to Redis...")
        self.redis_client = redis.Redis(host=self.args.redis_ip, port=6379, db=0)
        self.redis_pipeline = self.redis_client.pipeline()
        self.redis_client.ping()
        
        # Clear shutdown signal from previous runs
        self.redis_client.delete("robot_shutdown")
        
        # Initialize _prev_unitree_keys with CURRENT controller state
        try:
            ctrl_data = self.redis_client.get("unitree_controller")
            if ctrl_data:
                data = json.loads(ctrl_data)
                self._prev_unitree_keys = data.get("keys", 0)
                print(f"[cyan]Synced with Unitree controller (keys={self._prev_unitree_keys})[/cyan]")
            else:
                self._prev_unitree_keys = 0
        except:
            self._prev_unitree_keys = 0
        
        print("[green]Redis connected[/green]")
    
    def _setup_inspire_hands(self):
        if not self.use_inspire_hands:
            print("\n[6/7] Inspire hands: DISABLED")
            return
        
        print("\n[6/7] Setting up Inspire hands...")
        try:
            left_ip = getattr(self.args, 'inspire_left_ip', '192.168.123.210')
            right_ip = getattr(self.args, 'inspire_right_ip', '192.168.123.211')
            
            self.inspire_hand_controller = DualHandController(
                left_ip=left_ip,
                right_ip=right_ip,
                timeout=3.0,
                async_mode=True
            )
            
            # Set force limit to prevent fingers getting stuck
            # Range: 0-3000 (per Inspire manual), lower = softer grip
            # Manual example uses 300 for soft, recommended: 500-800
            force_limit = getattr(self.args, 'hand_force_limit', 500)
            print(f"  → Setting force limit to {force_limit}...")
            self.inspire_hand_controller.set_force_limit(force_limit)
            time.sleep(0.3)
            
            # Test sequence - open/close/open to verify connectivity
            print("  Testing hands...")
            print("  → Opening hands...")
            self.inspire_hand_controller.open_both()
            time.sleep(1.0)
            
            print("  → Closing hands...")
            self.inspire_hand_controller.close_both()
            time.sleep(1.0)
            
            print("  → Opening hands again...")
            self.inspire_hand_controller.open_both()
            time.sleep(0.5)
            
            print("[green]Inspire hands connected and tested[/green]")
            
        except Exception as e:
            print(f"[red]Failed to initialize Inspire hands: {e}[/red]")
            self.inspire_hand_controller = None
            self.use_inspire_hands = False
    
    def _setup_dex_hand_trackers(self):
        print("\n[7/7] Setting up dex-retargeting hand trackers...")
        urdf_dir = str(Path(__file__).parent / "inspire_hand")
        
        try:
            self.dex_left_tracker = DexHandTracker("left", urdf_dir)
            print("[green]Left hand tracker ready[/green]")
        except Exception as e:
            print(f"[red]Failed to init left hand tracker: {e}[/red]")
            self.dex_left_tracker = None
        
        try:
            self.dex_right_tracker = DexHandTracker("right", urdf_dir)
            print("[green]Right hand tracker ready[/green]")
        except Exception as e:
            print(f"[red]Failed to init right hand tracker: {e}[/red]")
            self.dex_right_tracker = None
    
    def _print_controls(self):
        print("\n" + "="*60)
        print("  MERGED TELEOP (dex-retargeting) - IK-based finger tracking")
        print("="*60)
        print("\n[yellow]UNITREE CONTROLLER (via Redis):[/yellow]")
        print("  A       : Cycle states (idle → preview → teleop → pause)")
        print("  B       : Toggle WALK ↔ BALANCE mode")
        print("  X       : Toggle UPPER BODY (arms+hands) freeze")
        print("  [red]Select  : EMERGENCY SHUTDOWN[/red]")
        print("\n[yellow]PICO VR:[/yellow]")
        print("  Body    : Full body motion tracking")
        print("  Hands   : Finger tracking → Inspire hands (dex-retargeting)")
        print("\n[yellow]UNITREE JOYSTICK (in walk mode):[/yellow]")
        print("  Left stick  : Forward/back + strafe")
        print("  Right stick : Rotate")
        print("\n[cyan]States:[/cyan]")
        print("  idle/preview: MuJoCo tracks (robot not moving yet)")
        print("  teleop_full : Balance mode (full body tracking)")
        print("  teleop_loco : Walk mode (legs walk, arms track)")
        print("  paused      : MuJoCo tracks, robot FROZEN")
        print("\n[green]NO CALIBRATION NEEDED - dex-retargeting uses IK![/green]")
        print("="*60 + "\n")
    
    def get_teleop_data(self):
        """Get data from Pico VR"""
        if self.teleop_streamer is not None:
            try:
                return self.teleop_streamer.get_current_frame()
            except:
                return None, None, None, None, None
        return None, None, None, None, None
    
    def _read_unitree_controller(self):
        """Read Unitree controller state from Redis"""
        try:
            ctrl_data = self.redis_client.get("unitree_controller")
            if ctrl_data:
                data = json.loads(ctrl_data)
                return data.get("keys", 0), data.get("lx", 0), data.get("ly", 0), data.get("rx", 0), data.get("ry", 0)
        except:
            pass
        return 0, 0, 0, 0, 0
    
    def _check_button(self, keys, button_name):
        """Check if a button is pressed"""
        return (keys & UNITREE_BUTTONS[button_name]) != 0
    
    def update_state_from_unitree(self, current_qpos=None):
        """Update state machine based on Unitree controller input"""
        keys, lx, ly, rx, ry = self._read_unitree_controller()
        
        # Detect button presses (rising edge)
        a_pressed = self._check_button(keys, "A") and not self._check_button(self._prev_unitree_keys, "A")
        b_pressed = self._check_button(keys, "B") and not self._check_button(self._prev_unitree_keys, "B")
        x_pressed = self._check_button(keys, "X") and not self._check_button(self._prev_unitree_keys, "X")
        select_pressed = self._check_button(keys, "select") and not self._check_button(self._prev_unitree_keys, "select")
        
        self._prev_unitree_keys = keys
        
        # Emergency stop
        if select_pressed:
            print("[red]EMERGENCY STOP![/red]")
            self.redis_client.set("robot_shutdown", "1")
            self.state = "exit"
            return
        
        # A: Cycle states
        if a_pressed and not self.is_interpolating:
            if self.state == "idle":
                self.state = "preview"
                print("[cyan]→ PREVIEW mode[/cyan]")
            elif self.state == "preview":
                # Instant transition - MuJoCo was already tracking
                self.state = "teleop_full"
                print("[green]→ TELEOP_FULL active[/green]")
            elif self.state in ["teleop_full", "teleop_loco"]:
                self.previous_teleop_state = self.state
                self.paused_qpos = current_qpos.copy() if current_qpos is not None else None
                self.paused_mimic_obs = None
                self.state = "paused"
                print("[yellow]→ PAUSED[/yellow]")
            elif self.state == "paused":
                # Unpausing - return to previous teleop state
                self.state = self.previous_teleop_state
                self.paused_mimic_obs = None
                print(f"[green]→ {self.previous_teleop_state.upper()} (unpaused)[/green]")
        
        # B: Toggle walk/balance (with interpolation)
        if b_pressed and not self.is_interpolating and self.state in ["teleop_full", "teleop_loco"]:
            if self.state == "teleop_full":
                print("[cyan]→ Switching to WALK mode[/cyan]")
                self._start_interpolation("teleop_full", "teleop_loco", current_qpos)
            else:
                print("[cyan]→ Switching to BALANCE mode[/cyan]")
                self._start_interpolation("teleop_loco", "teleop_full", current_qpos)
        
        # X: Toggle upper body freeze
        if x_pressed:
            self.hands_paused = not self.hands_paused
            if self.hands_paused:
                self._capture_frozen_arms = True  # Capture on next frame
                print("[yellow]Upper body FROZEN[/yellow]")
            else:
                self.frozen_arm_obs = None
                self._hands_paused_warned = False  # Reset warning
                print("[green]Upper body TRACKING[/green]")
        
        # Joystick for locomotion (only when locomotion is active)
        if self._is_locomotion_active():
            # Unitree joystick: ly=forward/back, lx=strafe, rx=rotate
            self.vel_cmd[0] = ly   # Forward/back
            self.vel_cmd[1] = -lx  # Strafe (inverted)
            self.vel_cmd[2] = -rx  # Rotate (inverted: right stick right = rotate right)
        else:
            self.vel_cmd[:] = 0
    
    def _start_interpolation(self, from_state, to_state, current_qpos):
        """Start smooth interpolation between states"""
        self.is_interpolating = True
        self.interp_start_time = time.time()
        self.interp_from_state = from_state
        self.interp_to_state = to_state
        if current_qpos is not None:
            self.interp_start_qpos = current_qpos.copy()
            self.interp_target_qpos = current_qpos.copy()  # Initialize target
        
        # Reset smooth history on state transition
        self.reset_smooth_history()
        
        # Reset LocoMode policy when switching TO teleop_loco
        if to_state == "teleop_loco":
            self.loco_policy.reset()
    
    def _is_locomotion_active(self):
        """Check if locomotion mode is active (for velocity commands)"""
        if self.state == "teleop_loco":
            return True
        # Also active during interpolation TO teleop_loco
        if self.is_interpolating and self.interp_to_state == "teleop_loco":
            elapsed = time.time() - self.interp_start_time
            alpha = min(1.0, elapsed / self.INTERP_DURATION)
            return alpha > 0.5  # Enable after halfway through interpolation
        return False
    
    def _check_auto_balance_on_backward_stop(self, current_qpos):
        """
        Auto-transition to balance mode when backward walking stops.
        Similar to pressing A+Y - gives robot time to stabilize.
        """
        if not self._is_locomotion_active() or self.is_interpolating:
            self._was_walking_backward = False
            return
        
        # Check backward velocity (negative vel_cmd[0] = walking backward)
        is_walking_backward = self.vel_cmd[0] < -self.BACKWARD_STOP_THRESHOLD
        
        # Detect transition: was walking backward, now stopped
        if self._was_walking_backward and not is_walking_backward:
            print("\n[yellow]→ Backward walk stopped, auto-switching to BALANCE mode[/yellow]")
            self._start_interpolation("teleop_loco", "teleop_full", current_qpos)
        
        self._was_walking_backward = is_walking_backward
    
    def _send_velocity_command_only(self):
        """Send only velocity command to Redis (called every loop in teleop mode)"""
        if self._is_locomotion_active():
            self.redis_client.set("loco_vel_cmd", json.dumps(self.vel_cmd.tolist()))
        else:
            self.redis_client.set("loco_vel_cmd", json.dumps([0.0, 0.0, 0.0]))
    
    def apply_smooth(self, mimic_obs):
        """Apply sliding window smoothing to mimic observations"""
        if not self.enable_smooth or mimic_obs is None:
            return mimic_obs
        
        self.smooth_history.append(mimic_obs.copy())
        if len(self.smooth_history) > self.smooth_window_size:
            self.smooth_history.pop(0)
        
        return np.mean(self.smooth_history, axis=0)
    
    def reset_smooth_history(self):
        """Reset smooth filter history (on state transitions)"""
        self.smooth_history = []
    
    def _is_teleop_state(self):
        """Check if we're in an active teleop state"""
        teleop_states = ["teleop_full", "teleop_loco", "paused"]
        if self.state in teleop_states:
            return True
        if self.is_interpolating and self.interp_to_state in teleop_states:
            return True
        return False
    
    def _control_inspire_hands_dex(self, left_hand_data, right_hand_data):
        """Control Inspire hands using dex-retargeting IK with RAW SDK data.
        
        IMPORTANT: Uses raw SDK arrays directly (like XRoboToolkit) instead of
        GMR's dictionary format, which applies coordinate transformations that
        break dex-retargeting.
        """
        if not self.use_inspire_hands or self.inspire_hand_controller is None:
            return
        
        if self.hands_paused:
            # Warn once that hands are paused
            if not hasattr(self, '_hands_paused_warned') or not self._hands_paused_warned:
                print("[yellow]Hands paused (press X to unfreeze)[/yellow]")
                self._hands_paused_warned = True
            return
        
        try:
            # Get RAW hand data directly from SDK (bypasses GMR coordinate transforms)
            # This is exactly how XRoboToolkit does it in teleop_inspire_hand_placo.py
            left_qpos = None
            right_qpos = None
            
            # LEFT HAND - get raw SDK array
            if self.dex_left_tracker:
                left_active = xrt.get_left_hand_is_active()
                if left_active:
                    left_raw = np.array(xrt.get_left_hand_tracking_state())
                    if left_raw is not None and left_raw.shape[0] >= 26 and not np.all(left_raw == 0):
                        mp_left = pico_raw_to_mediapipe(left_raw)
                        if mp_left is not None:
                            left_qpos = self.dex_left_tracker.retarget(mp_left)
                            # Debug
                            if not hasattr(self, '_left_hand_debug_count'):
                                self._left_hand_debug_count = 0
                            self._left_hand_debug_count += 1
                            if self._left_hand_debug_count % 100 == 1:
                                print(f"\n[HAND] Left tracking: qpos[:3]={left_qpos[:3] if left_qpos is not None else None}")
            
            # RIGHT HAND - get raw SDK array
            if self.dex_right_tracker:
                if xrt.get_right_hand_is_active():
                    right_raw = np.array(xrt.get_right_hand_tracking_state())
                    if right_raw is not None and right_raw.shape[0] >= 26 and not np.all(right_raw == 0):
                        mp_right = pico_raw_to_mediapipe(right_raw)
                        if mp_right is not None:
                            right_qpos = self.dex_right_tracker.retarget(mp_right)
            
            # Debug output (show only the 6 DOFs we use)
            if self.args.verbose:
                DEX_TO_INSPIRE = [4, 6, 2, 0, 9, 8]
                if left_qpos is not None and len(left_qpos) >= 12:
                    l_inspire = left_qpos[DEX_TO_INSPIRE]
                    l_str = f"[{','.join(f'{x:.2f}' for x in l_inspire)}]"
                else:
                    l_str = "None"
                if right_qpos is not None and len(right_qpos) >= 12:
                    r_inspire = right_qpos[DEX_TO_INSPIRE]
                    r_str = f"[{','.join(f'{x:.2f}' for x in r_inspire)}]"
                else:
                    r_str = "None"
                print(f"\r[dex] L={l_str} R={r_str}", end="                              ")
            
            # Send to Inspire hands
            # HYBRID APPROACH:
            # - Fingers: use dex-retargeting (works well)
            # - Thumb: use simple distance-based from raw Pico data (dex-retargeting doesn't work for thumb)
            #
            # dex output order: [index_prox, index_inter, middle_prox, middle_inter, 
            #                    pinky_prox, pinky_inter, ring_prox, ring_inter,
            #                    thumb_yaw, thumb_pitch, thumb_inter, thumb_distal]
            # Inspire order: [Little, Ring, Middle, Index, Thumb_Bend, Thumb_Rotate]
            
            # Pico joint indices for thumb distance calculation
            THUMB_TIP = 5
            PALM = 0
            THUMB_PROX = 3
            LITTLE_PROX = 22
            
            # Calibration values from user testing (2024-01-20):
            # Thumb bend: ThumbTip-Palm distance
            #   Open: 9.11cm, Fist: 7.30cm (change: -1.81cm)
            # Thumb rotation: ThumbProx-Pinky distance  
            #   Fist: 8.18cm (min), Open/Rotated: 9.22cm (max)
            THUMB_BEND_OPEN = 0.091   # 9.1cm (open palm)
            THUMB_BEND_CLOSED = 0.073  # 7.3cm (fist)
            THUMB_ROT_INWARD = 0.082   # 8.2cm (thumb tucked in)
            THUMB_ROT_OUTWARD = 0.092  # 9.2cm (thumb spread out)
            
            def get_thumb_from_pico(side):
                """Get thumb bend and rotation from raw Pico data"""
                try:
                    if side == "left":
                        if not xrt.get_left_hand_is_active():
                            return None, None
                        hs = np.array(xrt.get_left_hand_tracking_state())
                    else:
                        if not xrt.get_right_hand_is_active():
                            return None, None
                        hs = np.array(xrt.get_right_hand_tracking_state())
                    
                    if hs.shape[0] < 26:
                        return None, None
                    
                    thumb_tip = hs[THUMB_TIP, :3]
                    palm = hs[PALM, :3]
                    thumb_prox = hs[THUMB_PROX, :3]
                    little_prox = hs[LITTLE_PROX, :3]
                    
                    # Thumb bend: distance from thumb tip to palm
                    # Smaller distance = more bent = lower Inspire value (0=closed)
                    bend_dist = np.linalg.norm(thumb_tip - palm)
                    bend_normalized = (bend_dist - THUMB_BEND_CLOSED) / (THUMB_BEND_OPEN - THUMB_BEND_CLOSED)
                    bend_normalized = np.clip(bend_normalized, 0.0, 1.0)
                    thumb_bend = int(bend_normalized * 2000)  # 0=closed, 2000=open
                    
                    # Thumb rotation: distance from thumb proximal to pinky proximal
                    # Larger distance = more outward = higher Inspire value
                    rot_dist = np.linalg.norm(thumb_prox - little_prox)
                    rot_normalized = (rot_dist - THUMB_ROT_INWARD) / (THUMB_ROT_OUTWARD - THUMB_ROT_INWARD)
                    rot_normalized = np.clip(rot_normalized, 0.0, 1.0)
                    thumb_rot = int(rot_normalized * 2000)  # 0=inward, 2000=outward
                    
                    return thumb_bend, thumb_rot
                except:
                    return None, None
            
            def dex_to_inspire_angles(qpos, thumb_bend_override=None, thumb_rot_override=None):
                """Convert 12-DOF dex output to 6-DOF Inspire angles"""
                angles = np.zeros(6, dtype=np.int16)
                
                # Finger joints: proximal only, range 0-1.57 rad
                # Map: 0 rad (open) → 2000, 1.57 rad (closed) → 0
                finger_indices = [4, 6, 2, 0]  # pinky, ring, middle, index
                for i, idx in enumerate(finger_indices):
                    val = np.clip((1.57 - qpos[idx]) / 1.57 * 2000, 0, 2000)
                    angles[i] = int(val)
                
                # Thumb: use override from distance-based calculation if available
                if thumb_bend_override is not None:
                    angles[4] = thumb_bend_override
                else:
                    # Fallback to dex-retargeting (less accurate)
                    thumb_bend_total = qpos[9] + qpos[10] + qpos[11]
                    angles[4] = int(np.clip((1.57 - thumb_bend_total) / 1.57 * 2000, 0, 2000))
                
                if thumb_rot_override is not None:
                    angles[5] = thumb_rot_override
                else:
                    angles[5] = int(np.clip(qpos[8] / 1.308 * 2000, 0, 2000))
                
                return angles
            
            # Process left hand
            if left_qpos is not None and len(left_qpos) >= 12:
                left_thumb_bend, left_thumb_rot = get_thumb_from_pico("left")
                left_angles = dex_to_inspire_angles(left_qpos, left_thumb_bend, left_thumb_rot)
                self.inspire_hand_controller.left_hand.set_angles(left_angles)
            
            # Process right hand
            if right_qpos is not None and len(right_qpos) >= 12:
                right_thumb_bend, right_thumb_rot = get_thumb_from_pico("right")
                right_angles = dex_to_inspire_angles(right_qpos, right_thumb_bend, right_thumb_rot)
                self.inspire_hand_controller.right_hand.set_angles(right_angles)
                
        except Exception as e:
            if self.args.verbose:
                print(f"[red]dex-retargeting error: {e}[/red]")
    
    def send_to_redis(self, mimic_obs, neck_data):
        """Send mimic_obs, neck data, velocity cmd, and state info to Redis"""
        try:
            if mimic_obs is not None:
                # Use correct key name and JSON format (matching teleop_hybrid.py)
                self.redis_pipeline.set(
                    "action_body_unitree_g1_with_hands",
                    json.dumps(mimic_obs.tolist())
                )
            if neck_data is not None:
                self.redis_pipeline.set(
                    "action_neck_unitree_g1_with_hands",
                    json.dumps(neck_data)
                )
            
            # Send velocity command based on locomotion state
            if self._is_locomotion_active():
                self.redis_pipeline.set("loco_vel_cmd", json.dumps(self.vel_cmd.tolist()))
            else:
                self.redis_pipeline.set("loco_vel_cmd", json.dumps([0.0, 0.0, 0.0]))
            
            # Send state info for sim2real
            effective_state = self.interp_to_state if self.is_interpolating else self.state
            state_info = {
                "state": effective_state,
                "actual_state": self.state,
                "is_interpolating": self.is_interpolating,
                "interp_to_state": self.interp_to_state if self.is_interpolating else None,
            }
            self.redis_pipeline.set("teleop_state_info", json.dumps(state_info))
            
            # Send timestamp (matching teleop_hybrid.py)
            t_action = int(time.time() * 1000)
            self.redis_pipeline.set("t_action", t_action)
            
            self.redis_pipeline.execute()
        except Exception as e:
            if self.args.verbose:
                print(f"[red]Redis error: {e}[/red]")
    
    def _get_interpolated_qpos(self, current_target_qpos):
        """Get interpolated qpos during transitions"""
        if not self.is_interpolating:
            return current_target_qpos
        
        elapsed = time.time() - self.interp_start_time
        alpha = min(elapsed / self.INTERP_DURATION, 1.0)
        
        # Update target position with current data (if available)
        # EXCEPT for paused state - keep the frozen target
        if current_target_qpos is not None and self.interp_to_state != "paused":
            self.interp_target_qpos = current_target_qpos.copy()
            # Apply state-specific poses
            if self.interp_to_state == "teleop_loco":
                # Locomotion mode: legs + waist at LocoMode default
                self.interp_target_qpos[7:7+12] = self.loco_policy.default_angles_reorder[:12]
                self.interp_target_qpos[7+12:7+15] = self.loco_policy.default_angles_reorder[12:15]
        
        # Check if interpolation complete
        if alpha >= 1.0:
            self.is_interpolating = False
            self.state = self.interp_to_state
            print(f"\n[green]→ {self.state.upper()} mode active[/green]")
        
        if self.interp_start_qpos is None or self.interp_target_qpos is None:
            return current_target_qpos
        
        return (1 - alpha) * self.interp_start_qpos + alpha * self.interp_target_qpos
    
    def run(self):
        """Main teleop loop"""
        # Suppress loop_rate_limiters warnings
        import logging
        logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)
        
        rate = RateLimiter(frequency=self.args.target_fps)
        
        print(f"\nStarting in state: {self.state}")
        print("Waiting for Pico VR data...")
        
        # Neck limits
        NECK_YAW_LIMIT = 1.22   # ~70 degrees in radians
        NECK_PITCH_LIMIT = 0.77  # ~44 degrees in radians
        
        # No data warning tracking
        no_data_warnings = 0
        last_warning_time = 0
        
        # Try to create MuJoCo viewer, fall back to headless if it fails
        viewer = None
        viewer_ctx = None
        try:
            viewer_ctx = mjv.launch_passive(
                model=self.model, 
                data=self.data, 
                show_left_ui=False, 
                show_right_ui=False
            )
            viewer = viewer_ctx.__enter__()
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1
            print("[green]MuJoCo viewer started[/green]")
        except Exception as e:
            print(f"[yellow]MuJoCo viewer failed: {e}[/yellow]")
            print("[cyan]Running in HEADLESS mode (no visualization)[/cyan]")
            viewer = None
            viewer_ctx = None
        
        try:
            while (viewer is None or viewer.is_running()) and self.state != "exit":
                # Get Pico data
                body_pose, left_hand, right_hand, controller, headset = self.get_teleop_data()
                
                # Warn if no data received
                if self.teleop_streamer is not None:
                    if body_pose is None:
                        no_data_warnings += 1
                        current_time = time.time()
                        if current_time - last_warning_time > 5.0:
                            print(f"\n[yellow]Warning: No Pico data received ({no_data_warnings} attempts)[/yellow]")
                            last_warning_time = current_time
                    else:
                        no_data_warnings = 0
                        # Print status in headless mode
                        if viewer is None:
                            loco = "WALK" if self._is_locomotion_active() else "BAL "
                            freeze = "FRZ" if self._upper_body_frozen else "   "
                            sys.stdout.write(f"\r[{self.state:10s}] {loco} {freeze} vel:({self.vel_cmd[0]:+.2f},{self.vel_cmd[1]:+.2f},{self.vel_cmd[2]:+.2f})")
                            sys.stdout.flush()
                
                # Process retargeting first to get current qpos for state machine
                qpos = None
                if body_pose is not None and self.retarget is not None:
                    try:
                        qpos = self.retarget.retarget(body_pose, offset_to_ground=True)
                        if qpos is not None:
                            self._last_valid_qpos = qpos.copy()
                            # Debug: occasional status
                            if not hasattr(self, '_retarget_count'):
                                self._retarget_count = 0
                            self._retarget_count += 1
                            if self._retarget_count % 100 == 1:
                                print(f"\n[GMR] Retarget OK: len={len(qpos)}, qpos[7:10]={qpos[7:10]}")
                    except Exception as e:
                        if not hasattr(self, '_gmr_error_printed'):
                            print(f"[red]Retarget error: {e}[/red]")
                            self._gmr_error_printed = True
                elif body_pose is None:
                    if not hasattr(self, '_no_body_warned'):
                        print("[yellow]No body pose data from Pico[/yellow]")
                        self._no_body_warned = True
                elif self.retarget is None:
                    if not hasattr(self, '_no_retarget_warned'):
                        print("[red]GMR retarget is None![/red]")
                        self._no_retarget_warned = True
                
                # Update state from Unitree controller
                self.update_state_from_unitree(qpos)
                
                # Control Inspire hands (only in teleop states)
                if self._is_teleop_state():
                    self._control_inspire_hands_dex(left_hand, right_hand)
                
                # Auto-transition from idle to preview when data arrives
                if self.state == "idle" and body_pose is not None:
                    self.state = "preview"
                    print("\n→ PREVIEW mode: Pico data received!")
                
                # Send velocity commands if in teleop states
                if self._is_teleop_state():
                    self._send_velocity_command_only()
                
                # Process visualization and Redis sending
                if qpos is not None:
                    # Apply interpolation if active
                    if self.is_interpolating:
                        qpos = self._get_interpolated_qpos(qpos)
                    elif self.state == "teleop_loco":
                        # In locomotion mode: legs + waist at LocoMode default
                        qpos[7:7+12] = self.loco_policy.default_angles_reorder[:12]  # Legs
                        qpos[7+12:7+15] = self.loco_policy.default_angles_reorder[12:15]  # Waist
                        
                        # Auto-balance: switch to balance mode when backward walking stops
                        self._check_auto_balance_on_backward_stop(qpos)
                    
                    # Update MuJoCo visualization
                    self.data.qpos[:] = qpos
                    mj.mj_forward(self.model, self.data)
                    
                    # Camera follows the robot (if viewer exists)
                    if viewer is not None:
                        robot_pos = self.data.xpos[self.robot_base_id]
                        viewer.cam.lookat[:] = robot_pos
                        viewer.cam.distance = 3.0
                    
                    # Extract mimic observations
                    if self.last_qpos is None:
                        self.last_qpos = qpos.copy()
                    mimic_obs = extract_mimic_obs(qpos, self.last_qpos, dt=1/self.args.target_fps)
                    self.last_qpos = qpos.copy()
                    
                    # Apply smooth filtering
                    mimic_obs = self.apply_smooth(mimic_obs)
                    
                    if mimic_obs is not None:
                        # TWIST2 behavior: Let body rotation pass through from retargeting
                        # (No waist_yaw lock - robot follows human torso rotation)
                        
                        # Upper body freeze: capture arm positions on request
                        if self._capture_frozen_arms:
                            self.frozen_arm_obs = mimic_obs[21:35].copy()
                            self._capture_frozen_arms = False
                            print(f"[cyan]Arm positions captured for freeze[/cyan]")
                        
                        # Upper body freeze: use frozen arm positions
                        if self.hands_paused and self.frozen_arm_obs is not None:
                            mimic_obs[21:35] = self.frozen_arm_obs
                        
                        # === YAW CONTROL (TWIST2 behavior) ===
                        # teleop_full: Let yaw velocity pass through from body tracking
                        # teleop_loco: Use joystick for yaw (walking mode)
                        # paused/other: Lock to 0
                        effective_state = self.interp_to_state if self.is_interpolating else self.state
                        if effective_state == "teleop_full":
                            # TWIST2: Natural yaw from body tracking (no modification)
                            pass
                        elif effective_state == "teleop_loco":
                            # Walking: joystick controls yaw
                            mimic_obs[5] = self.vel_cmd[2]
                        else:
                            # Paused/idle: no rotation
                            mimic_obs[5] = 0.0
                    
                    # Get neck data from body tracking (uses Spine3 and Head joints)
                    neck_data = None
                    if body_pose is not None:
                        try:
                            neck_yaw, neck_pitch = human_head_to_robot_neck(body_pose)
                            neck_yaw = float(np.clip(neck_yaw, -NECK_YAW_LIMIT, NECK_YAW_LIMIT))
                            neck_pitch = float(np.clip(neck_pitch, -NECK_PITCH_LIMIT, NECK_PITCH_LIMIT))
                            neck_data = [neck_yaw, neck_pitch]
                        except Exception as e:
                            if not hasattr(self, '_neck_error_logged'):
                                print(f"[yellow]Neck tracking error: {e}[/yellow]")
                                self._neck_error_logged = True
                    
                    # Send to Redis if in teleop states
                    if self._is_teleop_state():
                        if self.state == "paused":
                            if self.paused_mimic_obs is None and mimic_obs is not None:
                                self.paused_mimic_obs = mimic_obs.copy()
                                print("[cyan]Paused: Robot frozen at current pose[/cyan]")
                            self.send_to_redis(self.paused_mimic_obs, neck_data)
                        else:
                            self.paused_mimic_obs = None
                            self.send_to_redis(mimic_obs, neck_data)
                            
                            # Debug: print mimic_obs status occasionally
                            if not hasattr(self, '_send_debug_count'):
                                self._send_debug_count = 0
                            self._send_debug_count += 1
                            if self._send_debug_count % 100 == 1:  # Every ~2s at 50fps
                                obs_sum = np.sum(np.abs(mimic_obs)) if mimic_obs is not None else 0
                                print(f"\n[DEBUG] Sending mimic_obs: sum={obs_sum:.2f}, state={self.state}")
                else:
                    # Even if no body data, send state info if in teleop mode
                    if self._is_teleop_state():
                        self.send_to_redis(None, None)
                
                if viewer is not None:
                    viewer.sync()
                rate.sleep()
        finally:
            # Cleanup viewer if it was created
            if viewer_ctx is not None:
                try:
                    viewer_ctx.__exit__(None, None, None)
                except:
                    pass
        
        print("\n\nExiting...")


def parse_args():
    parser = argparse.ArgumentParser(description="Merged Teleop with dex-retargeting")
    parser.add_argument("--robot", type=str, default="unitree_g1", help="Robot name")
    parser.add_argument("--actual_human_height", type=float, default=1.8, help="Human height in meters")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis server IP")
    parser.add_argument("--target_fps", type=int, default=50, help="Target FPS")
    parser.add_argument("--smooth", action="store_true", help="Enable smooth filtering")
    parser.add_argument("--smooth_window_size", type=int, default=4, help="Smooth filter window size")
    parser.add_argument("--use_inspire_hands", action="store_true", help="Enable Inspire hands")
    parser.add_argument("--inspire_left_ip", type=str, default="192.168.123.210", help="Left Inspire hand IP")
    parser.add_argument("--inspire_right_ip", type=str, default="192.168.123.211", help="Right Inspire hand IP")
    parser.add_argument("--hand_force_limit", type=int, default=500, 
                        help="Inspire hand force limit (0-3000). Lower=softer grip, prevents stuck fingers. 300=soft, 500-800=normal, 3000=max. Default: 500")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    teleop = MergedTeleopDex(args)
    try:
        teleop.run()
    except KeyboardInterrupt:
        print("\n[yellow]Interrupted[/yellow]")
