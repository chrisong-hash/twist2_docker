#!/usr/bin/env python3
"""
Merged Teleop - Pico Finger Tracking + Unitree Controller State Control
========================================================================
Complete teleop with:
- Full body tracking via Pico VR (GMR)
- Pico finger tracking → Inspire hands
- Unitree controller for state control (via Redis)
- Hybrid locomotion (balance/walk modes)
- Neck tracking

UNITREE CONTROLLER MAPPING (read from Redis):
=============================================
  A       : Cycle states (idle → preview → teleop → pause)
  B       : Toggle WALK ↔ BALANCE mode
  X       : Toggle UPPER BODY (arms+hands) freeze
  Select  : EMERGENCY SHUTDOWN

PICO VR:
========
  Body tracking : Full body motion capture
  Hand tracking : Finger positions → Inspire hands

UNITREE JOYSTICK (for locomotion in walk mode):
  Left stick    : Forward/back + strafe
  Right stick   : Rotate

States:
  idle/preview : MuJoCo tracks your motion (robot not moving yet)
  teleop_full  : Balance mode (full body tracking)
  teleop_loco  : Walk mode (legs walk, arms track)
  paused       : MuJoCo tracks, but robot FREEZES (reposition freely)

KEY BEHAVIOR:
  - Direct tracking: mimic_obs passed through directly
  - Robot does NOT rotate when you rotate (waist_yaw at index 20 locked at 0)
  - Yaw rotation ONLY from joystick in walk mode
  - Fingers tracked from Pico VR hand tracking
"""

import argparse
import json
import os
import sys
import time
import subprocess

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
    
    # 7. Other deps
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
from pathlib import Path
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


# ============== Hand Gesture Processor (from xrobot_teleop_hand_tracking.py) ==============
class HandGestureProcessor:
    """Process Pico hand tracking data into Inspire hand values
    
    Supports two-stage thumb calibration:
      Stage 1: FIST with thumb bent max and to side → calibrates thumb BEND
      Stage 2: EXTENDED fingers with thumb rotated inward → calibrates thumb ROTATION
    
    Calibration is saved to JSON for persistence between runs.
    """
    
    FINGER_TIPS = {
        'thumb': 'ThumbTip', 'index': 'IndexTip', 'middle': 'MiddleTip',
        'ring': 'RingTip', 'little': 'LittleTip'
    }
    INSPIRE_ORDER = ['little', 'ring', 'middle', 'index', 'thumb']
    INSPIRE_MAX = 1000
    CALIB_FILE = "/workspace/twist2/deploy_real/thumb_calibration.json"
    
    def __init__(self):
        # Default calibration values for fingers (thumb to palm distance)
        self.max_extension = {
            'thumb': 0.08, 'index': 0.18, 'middle': 0.19, 'ring': 0.17, 'little': 0.15
        }
        self.min_extension = {
            'thumb': 0.03, 'index': 0.05, 'middle': 0.05, 'ring': 0.05, 'little': 0.05
        }
        
        # Thumb calibration (per side)
        # Thumb bend: ThumbTip to Palm distance (independent of rotation)
        self.thumb_bend_max = {'left': 0.12, 'right': 0.12}  # Extended (thumb far from palm)
        self.thumb_bend_min = {'left': 0.04, 'right': 0.04}  # Bent (thumb close to palm)
        
        # Thumb rotation: ThumbTip to IndexKnuckle distance  
        self.thumb_rot_max = {'left': 0.12, 'right': 0.12}   # Rotated outward
        self.thumb_rot_min = {'left': 0.04, 'right': 0.04}   # Rotated inward
        
        self.smoothing_alpha = 0.3
        self.prev_values = {'left': {}, 'right': {}}
        self.calibrated = {'left': False, 'right': False}
        
        # Load saved calibration if exists
        self._load_calibration()
    
    def _load_calibration(self):
        """Load calibration from JSON file if it exists"""
        try:
            if os.path.exists(self.CALIB_FILE):
                with open(self.CALIB_FILE, 'r') as f:
                    data = json.load(f)
                
                # Load thumb bend calibration
                if 'thumb_bend_min' in data:
                    self.thumb_bend_min = data['thumb_bend_min']
                if 'thumb_bend_max' in data:
                    self.thumb_bend_max = data['thumb_bend_max']
                
                # Load thumb rotation calibration
                if 'thumb_rot_min' in data:
                    self.thumb_rot_min = data['thumb_rot_min']
                if 'thumb_rot_max' in data:
                    self.thumb_rot_max = data['thumb_rot_max']
                
                # Load finger extension calibration
                if 'finger_min_ext' in data:
                    self.min_extension.update(data['finger_min_ext'])
                if 'finger_max_ext' in data:
                    self.max_extension.update(data['finger_max_ext'])
                
                # Mark as calibrated
                if 'calibrated' in data:
                    self.calibrated = data['calibrated']
                
                print(f"[green]Loaded calibration from {self.CALIB_FILE}[/green]")
                if self.calibrated.get('left'):
                    print(f"  Left: bend=[{self.thumb_bend_min['left']:.3f}, {self.thumb_bend_max['left']:.3f}], rot=[{self.thumb_rot_min['left']:.3f}, {self.thumb_rot_max['left']:.3f}]")
                if self.calibrated.get('right'):
                    print(f"  Right: bend=[{self.thumb_bend_min['right']:.3f}, {self.thumb_bend_max['right']:.3f}], rot=[{self.thumb_rot_min['right']:.3f}, {self.thumb_rot_max['right']:.3f}]")
        except Exception as e:
            print(f"[yellow]No saved calibration found (will use defaults): {e}[/yellow]")
    
    def _save_calibration(self):
        """Save calibration to JSON file"""
        try:
            data = {
                'thumb_bend_min': self.thumb_bend_min,
                'thumb_bend_max': self.thumb_bend_max,
                'thumb_rot_min': self.thumb_rot_min,
                'thumb_rot_max': self.thumb_rot_max,
                'finger_min_ext': self.min_extension,
                'finger_max_ext': self.max_extension,
                'calibrated': self.calibrated
            }
            with open(self.CALIB_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[red]Failed to save calibration: {e}[/red]")
    
    def get_joint_position(self, hand_data, joint_name):
        """Get 3D position of a joint from hand data"""
        if not hand_data:
            return None
        
        possible_keys = [
            joint_name,
            f"LeftHand{joint_name}",
            f"RightHand{joint_name}",
        ]
        
        for key in possible_keys:
            if key in hand_data:
                data = hand_data[key]
                if isinstance(data, list) and len(data) >= 1:
                    pos = data[0]
                    if isinstance(pos, (list, np.ndarray)) and len(pos) >= 3:
                        return np.array([pos[0], pos[1], pos[2]])
                elif isinstance(data, dict) and 'position' in data:
                    pos = data['position']
                    return np.array([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)])
        
        if 'joints' in hand_data:
            for joint in hand_data['joints']:
                if joint.get('name') == joint_name:
                    pos = joint.get('position', {})
                    return np.array([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)])
        
        return None
    
    def calibrate_thumb_bend(self, hand_data, side):
        """Stage 1: Calibrate thumb BEND - call with FIST, thumb bent max
        Uses ThumbTip to Palm distance (independent of rotation)
        """
        thumb_tip = self.get_joint_position(hand_data, 'ThumbTip')
        palm = self.get_joint_position(hand_data, 'Palm')
        
        if thumb_tip is None or palm is None:
            print(f"[red]Could not find ThumbTip or Palm joints[/red]")
            return False
        
        # This is the minimum distance (thumb fully bent toward palm)
        dist = np.linalg.norm(thumb_tip - palm)
        self.thumb_bend_min[side] = dist
        print(f"[Calib] {side} thumb BEND min (ThumbTip→Palm bent): {dist:.4f}m")
        self._save_calibration()  # Save after each step
        return True
    
    def calibrate_thumb_rotation_min(self, hand_data, side):
        """Calibrate thumb ROTATION MIN - call with fingers extended, thumb rotated inward
        Uses ThumbProximal to LittleProximal (pinky base) distance
        When thumb is inward, distance to pinky base is SMALLER
        """
        thumb_prox = self.get_joint_position(hand_data, 'ThumbProximal')
        little_prox = self.get_joint_position(hand_data, 'LittleProximal')
        
        if thumb_prox is None or little_prox is None:
            print(f"[red]Could not find ThumbProximal or LittleProximal joints[/red]")
            return False
        
        # This is the minimum distance (thumb rotated inward)
        dist = np.linalg.norm(thumb_prox - little_prox)
        self.thumb_rot_min[side] = dist
        print(f"[Calib] {side} thumb ROTATION min (inward): {dist:.4f}m")
        self._save_calibration()
        return True
    
    def calibrate_thumb_rotation_max(self, hand_data, side):
        """Calibrate thumb ROTATION MAX - call with fully open palm, thumb spread outward
        Also captures thumb BEND MAX (extended thumb position)
        Uses ThumbProximal to LittleProximal (pinky base) distance for rotation
        Uses ThumbTip to Palm distance for bend
        """
        thumb_prox = self.get_joint_position(hand_data, 'ThumbProximal')
        little_prox = self.get_joint_position(hand_data, 'LittleProximal')
        thumb_tip = self.get_joint_position(hand_data, 'ThumbTip')
        palm = self.get_joint_position(hand_data, 'Palm')
        
        if thumb_prox is None or little_prox is None:
            print(f"[red]Could not find ThumbProximal or LittleProximal joints[/red]")
            return False
        
        # Rotation max: distance to pinky when thumb is outward
        rot_dist = np.linalg.norm(thumb_prox - little_prox)
        self.thumb_rot_max[side] = rot_dist
        print(f"[Calib] {side} thumb ROTATION max (outward): {rot_dist:.4f}m")
        
        # Also capture thumb bend max (extended thumb = larger distance to palm)
        if thumb_tip is not None and palm is not None:
            bend_dist = np.linalg.norm(thumb_tip - palm)
            self.thumb_bend_max[side] = bend_dist
            print(f"[Calib] {side} thumb BEND max (ThumbTip→Palm extended): {bend_dist:.4f}m")
        
        self.calibrated[side] = True
        print(f"[Calib] {side} hand calibration COMPLETE!")
        self._save_calibration()
        return True
    
    def calculate_inspire_value(self, hand_data, finger, side):
        """Calculate Inspire value (0-1000) for a finger (not thumb)"""
        tip_name = self.FINGER_TIPS[finger]
        tip = self.get_joint_position(hand_data, tip_name)
        ref = self.get_joint_position(hand_data, 'Palm')
        
        if tip is None or ref is None:
            return self.prev_values[side].get(finger, 0)
        
        distance = np.linalg.norm(tip - ref)
        min_ext = self.min_extension[finger]
        max_ext = self.max_extension[finger]
        
        # Auto-expand range if distance is outside calibrated range
        if distance < min_ext:
            self.min_extension[finger] = distance
            min_ext = distance
        if distance > max_ext:
            self.max_extension[finger] = distance
            max_ext = distance
        
        # Normalize: large distance (open) → 0, small distance (closed) → 1000
        if max_ext <= min_ext:
            max_ext = min_ext + 0.01  # Prevent division by zero
        normalized = (max_ext - distance) / (max_ext - min_ext)
        normalized = np.clip(normalized, 0.0, 1.0)
        value = int(normalized * self.INSPIRE_MAX)
        
        # Smooth
        if finger in self.prev_values[side]:
            value = int(self.smoothing_alpha * value + 
                       (1 - self.smoothing_alpha) * self.prev_values[side][finger])
        self.prev_values[side][finger] = value
        return value
    
    def calculate_thumb_bend(self, hand_data, side):
        """Calculate thumb BEND value (0-1000) - DOF 5
        Uses ThumbTip to Palm distance - independent of thumb rotation
        """
        thumb_tip = self.get_joint_position(hand_data, 'ThumbTip')
        palm = self.get_joint_position(hand_data, 'Palm')
        
        if thumb_tip is None or palm is None:
            return self.prev_values[side].get('thumb_bend', 0)
        
        dist = np.linalg.norm(thumb_tip - palm)
        min_d = self.thumb_bend_min[side]
        max_d = self.thumb_bend_max[side]
        
        # Normalize: large distance (extended) → 0, small distance (bent) → 1000
        if max_d <= min_d:
            max_d = min_d + 0.04  # Fallback
        
        normalized = (max_d - dist) / (max_d - min_d)
        normalized = np.clip(normalized, 0.0, 1.0)
        value = int(normalized * self.INSPIRE_MAX)
        
        # Smooth
        if 'thumb_bend' in self.prev_values[side]:
            value = int(self.smoothing_alpha * value +
                       (1 - self.smoothing_alpha) * self.prev_values[side]['thumb_bend'])
        self.prev_values[side]['thumb_bend'] = value
        return value
    
    def calculate_thumb_rotation(self, hand_data, side):
        """Calculate thumb ROTATION value (0-1000) - DOF 6
        Uses ThumbProximal to LittleProximal (pinky base) distance
        - Thumb inward (toward index): smaller distance → high value (1000)
        - Thumb outward (spread): larger distance → low value (0)
        """
        thumb_prox = self.get_joint_position(hand_data, 'ThumbProximal')
        little_prox = self.get_joint_position(hand_data, 'LittleProximal')
        
        if thumb_prox is None or little_prox is None:
            return self.prev_values[side].get('thumb_rot', 500)
        
        dist = np.linalg.norm(thumb_prox - little_prox)
        min_d = self.thumb_rot_min[side]
        max_d = self.thumb_rot_max[side]
        
        # Ensure min < max
        if max_d <= min_d:
            max_d = min_d + 0.05  # Fallback
        
        # Normalize: large distance (outward) → 0, small distance (inward) → 1000
        normalized = (max_d - dist) / (max_d - min_d)
        normalized = np.clip(normalized, 0.0, 1.0)
        value = int(normalized * self.INSPIRE_MAX)
        
        # Smooth
        if 'thumb_rot' in self.prev_values[side]:
            value = int(self.smoothing_alpha * value +
                       (1 - self.smoothing_alpha) * self.prev_values[side]['thumb_rot'])
        self.prev_values[side]['thumb_rot'] = value
        return value
    
    def process_hand(self, hand_data, side):
        """Process hand data into 6 DOF Inspire values
        Returns: [Little, Ring, Middle, Index, ThumbBend, ThumbRot]
        """
        if not hand_data:
            return None
        
        values = []
        # First 4 fingers (not thumb)
        for finger in ['little', 'ring', 'middle', 'index']:
            val = self.calculate_inspire_value(hand_data, finger, side)
            values.append(val)
        
        # Thumb bend (DOF 5)
        thumb_bend = self.calculate_thumb_bend(hand_data, side)
        values.append(thumb_bend)
        
        # Thumb rotation (DOF 6)
        thumb_rot = self.calculate_thumb_rotation(hand_data, side)
        values.append(thumb_rot)
        
        return values


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


class MergedTeleop:
    """
    Merged teleop with:
    - Body tracking from Pico VR
    - Finger tracking from Pico VR
    - State control from Unitree controller (via Redis)
    """
    
    INTERP_DURATION = 0.25
    
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
        self._unitree_a_pressed = False
        self._unitree_b_pressed = False
        self._unitree_x_pressed = False
        self._unitree_select_pressed = False
        
        # Upper body freeze state
        self.hands_paused = False
        self.frozen_arm_obs = None
        self._capture_frozen_arms = False
        
        # Hand gesture processor for finger tracking
        self.gesture_processor = HandGestureProcessor()
        
        # Inspire hands
        self.use_inspire_hands = getattr(args, 'use_inspire_hands', False)
        self.inspire_hand_controller = None
        
        # Height and retargeting
        self.estimated_height = args.actual_human_height
        self.retarget = None
        
        # Smooth filtering
        self.enable_smooth = args.smooth
        self.smooth_window_size = args.smooth_window_size
        self.smooth_history = []
        
        # Initialize systems
        print("\n[cyan]Initializing Merged Teleop...[/cyan]")
        self._setup_locomotion_policy()
        self._setup_teleop_streamer()
        self._setup_retargeting()
        self._setup_mujoco()
        self._setup_redis()
        self._setup_inspire_hands()
        
        print("\n[green]Systems initialized![/green]")
        self._print_controls()
    
    def _setup_locomotion_policy(self):
        print("\n[1/6] Loading locomotion policy...")
        self.loco_policy = LocoModePolicy()
    
    def _setup_teleop_streamer(self):
        print("\n[2/6] Connecting to Pico VR...")
        try:
            self.teleop_streamer = XRobotStreamer()
            print("[green]XRobotStreamer initialized[/green]")
        except Exception as e:
            print(f"[red]ERROR: Failed to initialize XRobotStreamer: {e}[/red]")
            self.teleop_streamer = None
    
    def _setup_retargeting(self, height=None):
        print("\n[3/6] Setting up GMR retargeting...")
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
        print("\n[4/6] Setting up MuJoCo preview...")
        xml_path = str(ROBOT_XML_DICT["unitree_g1"])
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.last_qpos = None
        self._last_valid_qpos = None
        self._using_fallback = False
        self.robot_base_id = self.model.body("pelvis").id
        print("[green]MuJoCo ready[/green]")
    
    def _setup_redis(self):
        print("\n[5/6] Connecting to Redis...")
        self.redis_client = redis.Redis(host=self.args.redis_ip, port=6379, db=0)
        self.redis_pipeline = self.redis_client.pipeline()
        self.redis_client.ping()
        
        # Clear shutdown signal from previous runs
        self.redis_client.delete("robot_shutdown")
        
        # Initialize _prev_unitree_keys with CURRENT controller state
        # This prevents stale button presses from triggering on startup
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
            print("\n[6/6] Inspire hands: DISABLED")
            return
        
        print("\n[6/6] Setting up Inspire hands...")
        try:
            left_ip = getattr(self.args, 'inspire_left_ip', '192.168.123.210')
            right_ip = getattr(self.args, 'inspire_right_ip', '192.168.123.211')
            
            self.inspire_hand_controller = DualHandController(
                left_ip=left_ip,
                right_ip=right_ip,
                timeout=3.0
            )
            
            time.sleep(0.5)
            self.inspire_hand_controller.open_both()
            print("[green]Inspire hands connected[/green]")
            
        except Exception as e:
            print(f"[red]Failed to initialize Inspire hands: {e}[/red]")
            self.inspire_hand_controller = None
            self.use_inspire_hands = False
    
    def _print_controls(self):
        print("\n" + "="*60)
        print("  MERGED TELEOP - Unitree Controller + Pico Finger Tracking")
        print("="*60)
        print("\n[yellow]UNITREE CONTROLLER (via Redis):[/yellow]")
        print("  A       : Cycle states (idle → preview → teleop → pause)")
        print("  B       : Toggle WALK ↔ BALANCE mode")
        print("  X       : Toggle UPPER BODY (arms+hands) freeze")
        print("  [red]Select  : EMERGENCY SHUTDOWN[/red]")
        print("\n[yellow]PICO VR:[/yellow]")
        print("  Body    : Full body motion tracking")
        print("  Hands   : Finger tracking → Inspire hands")
        print("\n[yellow]UNITREE JOYSTICK (in walk mode):[/yellow]")
        print("  Left stick  : Forward/back + strafe")
        print("  Right stick : Rotate")
        print("\n[cyan]States:[/cyan]")
        print("  idle/preview: MuJoCo tracks (robot not moving yet)")
        print("  teleop_full : Balance mode (full body tracking)")
        print("  teleop_loco : Walk mode (legs walk, arms track)")
        print("  paused      : MuJoCo tracks, robot FROZEN")
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
        
        # Emergency stop (Select)
        if select_pressed:
            self.state = "exit"
            print("\n[red]→ EMERGENCY SHUTDOWN (Select pressed)[/red]")
            self._send_shutdown_signal()
            self._prev_unitree_keys = keys
            return
        
        # X = Toggle upper body freeze
        if x_pressed and not self.is_interpolating and self.state != "exit":
            self.hands_paused = not self.hands_paused
            if self.hands_paused:
                self._capture_frozen_arms = True
                print(f"\n[yellow][UPPER BODY] FROZEN[/yellow]")
            else:
                self.frozen_arm_obs = None
                print(f"\n[yellow][UPPER BODY] TRACKING[/yellow]")
        
        # B = Toggle walk/balance mode
        if b_pressed and not self.is_interpolating and self.state != "exit":
            if self.state == "teleop_full":
                print("\n[cyan]→ Switching to WALK mode[/cyan]")
                self._start_interpolation("teleop_full", "teleop_loco", current_qpos)
            elif self.state == "teleop_loco":
                print("\n[cyan]→ Switching to BALANCE mode[/cyan]")
                self._start_interpolation("teleop_loco", "teleop_full", current_qpos)
        
        
        # A = Cycle states
        if a_pressed and not self.is_interpolating and self.state != "exit":
            if self.state == "idle":
                self.state = "preview"
                print("\n[cyan]→ PREVIEW mode[/cyan]")
            elif self.state == "preview":
                self.state = "teleop_full"
                print("\n[green]→ TELEOP_FULL active[/green]")
            elif self.state in ["teleop_full", "teleop_loco"]:
                self.previous_teleop_state = self.state
                self.paused_qpos = current_qpos.copy() if current_qpos is not None else None
                self.state = "paused"
                self.paused_mimic_obs = None
                print(f"\n[green]→ PAUSED[/green]")
            elif self.state == "paused":
                self.state = self.previous_teleop_state
                self.paused_mimic_obs = None
                print(f"\n[green]→ {self.previous_teleop_state.upper()} (unpaused)[/green]")
        
        # Joystick for locomotion is read in main loop from Unitree controller
        
        self._prev_unitree_keys = keys
    
    def _start_interpolation(self, from_state, to_state, current_qpos):
        """Start interpolation between states"""
        self.is_interpolating = True
        self.interp_start_time = time.time()
        self.interp_from_state = from_state
        self.interp_to_state = to_state
        
        if current_qpos is not None:
            self.interp_start_qpos = current_qpos.copy()
            self.interp_target_qpos = current_qpos.copy()
            
            if to_state == "teleop_loco":
                self.interp_target_qpos[7:7+12] = self.loco_policy.default_angles_reorder[:12]
                self.interp_target_qpos[7+12:7+15] = self.loco_policy.default_angles_reorder[12:15]
    
    def _get_interpolated_qpos(self, current_target_qpos):
        """Get interpolated qpos during state transition"""
        if not self.is_interpolating:
            return current_target_qpos
        
        elapsed = time.time() - self.interp_start_time
        progress = min(elapsed / self.INTERP_DURATION, 1.0)
        
        if progress >= 1.0:
            self.is_interpolating = False
            self.state = self.interp_to_state
            return current_target_qpos
        
        if self.interp_start_qpos is None:
            return current_target_qpos
        
        interpolated = self.interp_start_qpos + progress * (current_target_qpos - self.interp_start_qpos)
        return interpolated
    
    def _is_teleop_state(self):
        """Check if in active teleop state"""
        return self.state in ["teleop_full", "teleop_loco", "paused"]
    
    def _is_locomotion_active(self):
        """Check if locomotion mode is active"""
        return self.state == "teleop_loco" or (self.is_interpolating and self.interp_to_state == "teleop_loco")
    
    def _send_shutdown_signal(self):
        """Send shutdown signal to robot"""
        try:
            self.redis_client.set("robot_shutdown", "1")
            print("[red]  Shutdown signal sent![/red]")
        except Exception as e:
            print(f"[red]  Warning: Could not send shutdown signal: {e}[/red]")
    
    def _validate_quaternions(self, smplx_data):
        """Check if smplx_data contains valid quaternions (non-zero norm)"""
        if smplx_data is None:
            return False
        
        try:
            for joint_name, value in smplx_data.items():
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    pos, quat = value
                    if isinstance(quat, (list, np.ndarray)) and len(quat) == 4:
                        norm = np.linalg.norm(quat)
                        if norm < 1e-6:
                            return False
            return True
        except:
            return False
    
    def _is_qpos_valid(self, qpos):
        """Check if qpos contains valid values (no NaN/inf, reasonable bounds)"""
        if qpos is None:
            return False
        if np.any(np.isnan(qpos)) or np.any(np.isinf(qpos)):
            return False
        if np.any(np.abs(qpos[:3]) > 100):  # root position
            return False
        if np.any(np.abs(qpos[7:]) > 10):  # joint angles
            return False
        return True
    
    def process_retargeting(self, smplx_data):
        """Run GMR retargeting on SMPLX data"""
        if not self._validate_quaternions(smplx_data):
            self._using_fallback = True
            return self._last_valid_qpos
        
        try:
            # offset_to_ground=True shifts robot so feet are at ground level
            qpos = self.retarget.retarget(smplx_data, offset_to_ground=True)
        except ValueError as e:
            if "zero norm" in str(e):
                self._using_fallback = True
                return self._last_valid_qpos
            raise
        except Exception:
            self._using_fallback = True
            return self._last_valid_qpos
        
        if not self._is_qpos_valid(qpos):
            self._using_fallback = True
            return self._last_valid_qpos
        
        self._last_valid_qpos = qpos.copy()
        self._using_fallback = False
        return qpos
    
    def apply_smooth(self, obs):
        """Apply sliding window smoothing"""
        if not self.enable_smooth:
            return obs
        
        self.smooth_history.append(obs)
        if len(self.smooth_history) > self.smooth_window_size:
            self.smooth_history.pop(0)
        
        return np.mean(self.smooth_history, axis=0)
    
    def _control_inspire_hands_from_tracking(self, left_hand_data, right_hand_data):
        """Control Inspire hands using Pico finger tracking"""
        if not self.use_inspire_hands or self.inspire_hand_controller is None:
            return
        
        if self.hands_paused:
            return
        
        try:
            # Extract hand data from tuple format
            left_hand_dict = None
            right_hand_dict = None
            left_active = False
            right_active = False
            
            if isinstance(left_hand_data, tuple) and len(left_hand_data) >= 2:
                left_active, left_hand_dict = left_hand_data
                if not left_active:
                    left_hand_dict = None
            elif isinstance(left_hand_data, dict):
                left_hand_dict = left_hand_data
                left_active = True
            
            if isinstance(right_hand_data, tuple) and len(right_hand_data) >= 2:
                right_active, right_hand_dict = right_hand_data
                if not right_active:
                    right_hand_dict = None
            elif isinstance(right_hand_data, dict):
                right_hand_dict = right_hand_data
                right_active = True
            
            # Debug verbose: show what data we're getting
            if self.args.verbose:
                if left_hand_data is not None or right_hand_data is not None:
                    print(f"\r[HandRaw] L_type={type(left_hand_data).__name__} R_type={type(right_hand_data).__name__} L_active={left_active} R_active={right_active}", end="    ")
            
            # Process hand data - returns [Little, Ring, Middle, Index, ThumbBend, ThumbRot] in 0-1000 range
            left_values = self.gesture_processor.process_hand(left_hand_dict, 'left') if left_hand_dict else None
            right_values = self.gesture_processor.process_hand(right_hand_dict, 'right') if right_hand_dict else None
            
            # Debug output (verbose mode)
            if self.args.verbose and (left_values or right_values):
                # Show DOF values: [Little, Ring, Middle, Index, ThumbBend, ThumbRot]
                l_rot = left_values[5] if left_values else "?"
                r_rot = right_values[5] if right_values else "?"
                # Also show calibration ranges
                l_range = f"[{self.gesture_processor.thumb_rot_min['left']:.3f}-{self.gesture_processor.thumb_rot_max['left']:.3f}]"
                r_range = f"[{self.gesture_processor.thumb_rot_min['right']:.3f}-{self.gesture_processor.thumb_rot_max['right']:.3f}]"
                print(f"\r[Rot] L={l_rot}{l_range} R={r_rot}{r_range}", end="                              ")
            
            # Send to Inspire hands using set_angles (expects 0-2000 range)
            # HandGestureProcessor: 0 = open, 1000 = closed
            # Inspire hands: 0 = closed, 2000 = open (INVERTED)
            # So we need: inspire_value = 2000 - (processor_value * 2)
            if left_values:
                # Invert: 0→2000 (open), 1000→0 (closed)
                left_angles = 2000 - np.array(left_values, dtype=np.int16) * 2
                self.inspire_hand_controller.left_hand.set_angles(left_angles)
            
            if right_values:
                right_angles = 2000 - np.array(right_values, dtype=np.int16) * 2
                self.inspire_hand_controller.right_hand.set_angles(right_angles)
                
        except Exception as e:
            print(f"[red]Inspire hand error: {e}[/red]")
    
    def send_to_redis(self, mimic_obs, neck_data=None):
        """Send observations to Redis"""
        if mimic_obs is not None:
            self.redis_pipeline.set(
                "action_body_unitree_g1_with_hands",
                json.dumps(mimic_obs.tolist())
            )
        
        if neck_data is not None:
            self.redis_pipeline.set(
                "action_neck_unitree_g1_with_hands",
                json.dumps(neck_data)
            )
        
        # Send state info
        state_info = {
            "state": self.state,
            "locomotion_active": self._is_locomotion_active(),
            "hands_paused": self.hands_paused,
        }
        self.redis_pipeline.set("teleop_state_info", json.dumps(state_info))
        
        self.redis_pipeline.execute()
    
    def _run_calibration_prompt(self):
        """Prompt user for thumb calibration at startup"""
        # Check if already calibrated
        if self.gesture_processor.calibrated.get('left') and self.gesture_processor.calibrated.get('right'):
            print("\n[green]✓ Thumb calibration loaded from file[/green]")
            response = input("Recalibrate thumb? (y/N): ").strip().lower()
            if response != 'y':
                return
        else:
            print("\n[yellow]No thumb calibration found.[/yellow]")
            response = input("Calibrate thumb now? (Y/n): ").strip().lower()
            if response == 'n':
                print("[yellow]Skipping calibration - using defaults[/yellow]")
                return
        
        print("\n" + "="*60)
        print("[cyan]THUMB CALIBRATION[/cyan]")
        print("="*60)
        print("Make sure your hands are visible to Pico VR.")
        print("You'll do 3 poses per hand (left first, then right).")
        print("="*60)
        
        # Wait for hand tracking
        print("\nWaiting for Pico hand tracking...")
        for _ in range(100):  # 10 second timeout
            _, left_hand_data, right_hand_data, _, _ = self.get_teleop_data()
            left_ok = isinstance(left_hand_data, tuple) and len(left_hand_data) >= 2 and left_hand_data[0]
            right_ok = isinstance(right_hand_data, tuple) and len(right_hand_data) >= 2 and right_hand_data[0]
            if left_ok or right_ok:
                print("[green]✓ Hand tracking detected![/green]")
                break
            time.sleep(0.1)
        else:
            print("[red]Timeout waiting for hand tracking. Skipping calibration.[/red]")
            return
        
        # Stage 1: Left thumb bend (fist)
        print("\n" + "-"*40)
        print("[cyan]Stage 1/6: LEFT THUMB BEND[/cyan]")
        print("-"*40)
        print("Make a FIST with your LEFT hand")
        print("Thumb should be BENT and tucked against fingers")
        input("Press ENTER when ready...")
        
        for _ in range(30):  # 3 second retry
            _, left_hand_data, _, _, _ = self.get_teleop_data()
            if isinstance(left_hand_data, tuple) and len(left_hand_data) >= 2:
                left_active, left_dict = left_hand_data
                if left_active and left_dict:
                    if self.gesture_processor.calibrate_thumb_bend(left_dict, 'left'):
                        print("[green]✓ Captured![/green]")
                        break
            time.sleep(0.1)
        else:
            print("[yellow]Could not capture - using default[/yellow]")
        
        # Stage 2: Left thumb rotation inward
        print("\n" + "-"*40)
        print("[cyan]Stage 2/6: LEFT THUMB ROTATION (INWARD)[/cyan]")
        print("-"*40)
        print("EXTEND all fingers on LEFT hand")
        print("Rotate thumb INWARD (toward index finger)")
        input("Press ENTER when ready...")
        
        for _ in range(30):
            _, left_hand_data, _, _, _ = self.get_teleop_data()
            if isinstance(left_hand_data, tuple) and len(left_hand_data) >= 2:
                left_active, left_dict = left_hand_data
                if left_active and left_dict:
                    if self.gesture_processor.calibrate_thumb_rotation_min(left_dict, 'left'):
                        print("[green]✓ Captured![/green]")
                        break
            time.sleep(0.1)
        else:
            print("[yellow]Could not capture - using default[/yellow]")
        
        # Stage 3: Left thumb rotation outward (open palm)
        print("\n" + "-"*40)
        print("[cyan]Stage 3/6: LEFT THUMB ROTATION (OUTWARD)[/cyan]")
        print("-"*40)
        print("FULLY OPEN your LEFT hand (flat palm)")
        print("Thumb spread OUTWARD as far as possible")
        input("Press ENTER when ready...")
        
        for _ in range(30):
            _, left_hand_data, _, _, _ = self.get_teleop_data()
            if isinstance(left_hand_data, tuple) and len(left_hand_data) >= 2:
                left_active, left_dict = left_hand_data
                if left_active and left_dict:
                    if self.gesture_processor.calibrate_thumb_rotation_max(left_dict, 'left'):
                        print("[green]✓ Captured![/green]")
                        break
            time.sleep(0.1)
        else:
            print("[yellow]Could not capture - using default[/yellow]")
        
        # Stage 4: Right thumb bend (fist)
        print("\n" + "-"*40)
        print("[cyan]Stage 4/6: RIGHT THUMB BEND[/cyan]")
        print("-"*40)
        print("Make a FIST with your RIGHT hand")
        print("Thumb should be BENT and tucked against fingers")
        input("Press ENTER when ready...")
        
        for _ in range(30):
            _, _, right_hand_data, _, _ = self.get_teleop_data()
            if isinstance(right_hand_data, tuple) and len(right_hand_data) >= 2:
                right_active, right_dict = right_hand_data
                if right_active and right_dict:
                    if self.gesture_processor.calibrate_thumb_bend(right_dict, 'right'):
                        print("[green]✓ Captured![/green]")
                        break
            time.sleep(0.1)
        else:
            print("[yellow]Could not capture - using default[/yellow]")
        
        # Stage 5: Right thumb rotation inward
        print("\n" + "-"*40)
        print("[cyan]Stage 5/6: RIGHT THUMB ROTATION (INWARD)[/cyan]")
        print("-"*40)
        print("EXTEND all fingers on RIGHT hand")
        print("Rotate thumb INWARD (toward index finger)")
        input("Press ENTER when ready...")
        
        for _ in range(30):
            _, _, right_hand_data, _, _ = self.get_teleop_data()
            if isinstance(right_hand_data, tuple) and len(right_hand_data) >= 2:
                right_active, right_dict = right_hand_data
                if right_active and right_dict:
                    if self.gesture_processor.calibrate_thumb_rotation_min(right_dict, 'right'):
                        print("[green]✓ Captured![/green]")
                        break
            time.sleep(0.1)
        else:
            print("[yellow]Could not capture - using default[/yellow]")
        
        # Stage 6: Right thumb rotation outward (open palm)
        print("\n" + "-"*40)
        print("[cyan]Stage 6/6: RIGHT THUMB ROTATION (OUTWARD)[/cyan]")
        print("-"*40)
        print("FULLY OPEN your RIGHT hand (flat palm)")
        print("Thumb spread OUTWARD as far as possible")
        input("Press ENTER when ready...")
        
        for _ in range(30):
            _, _, right_hand_data, _, _ = self.get_teleop_data()
            if isinstance(right_hand_data, tuple) and len(right_hand_data) >= 2:
                right_active, right_dict = right_hand_data
                if right_active and right_dict:
                    if self.gesture_processor.calibrate_thumb_rotation_max(right_dict, 'right'):
                        print("[green]✓ Captured![/green]")
                        break
            time.sleep(0.1)
        else:
            print("[yellow]Could not capture - using default[/yellow]")
        
        print("\n" + "="*60)
        print("[green]CALIBRATION COMPLETE![/green]")
        print("="*60)
        print("Calibration saved. Won't need to recalibrate next time.")
        print("="*60 + "\n")
    
    def run(self):
        """Main loop"""
        import logging
        logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)
        
        # Calibration prompt at startup
        self._run_calibration_prompt()
        
        rate = RateLimiter(frequency=self.args.target_fps)
        
        print(f"\nStarting in state: {self.state}")
        print("Waiting for Pico VR data...")
        
        with mjv.launch_passive(
            model=self.model, 
            data=self.data, 
            show_left_ui=False, 
            show_right_ui=False
        ) as viewer:
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1
            
            while viewer.is_running() and self.state != "exit":
                # Get Pico VR data
                smplx_data, left_hand_data, right_hand_data, controller_data, headset_data = self.get_teleop_data()
                
                # Update state from Unitree controller
                current_qpos = self.data.qpos.copy() if self._last_valid_qpos is not None else None
                self.update_state_from_unitree(current_qpos)
                
                # Update joystick velocity from Unitree controller (for locomotion)
                keys, lx, ly, rx, ry = self._read_unitree_controller()
                if self._is_locomotion_active():
                    # Unitree joystick: ly=forward/back, lx=strafe, rx=rotate
                    self.vel_cmd[0] = ly   # Forward/back
                    self.vel_cmd[1] = -lx  # Strafe (inverted)
                    self.vel_cmd[2] = -rx  # Rotate (inverted: right stick right = rotate right)
                else:
                    self.vel_cmd[:] = 0
                
                # Send velocity command
                if self._is_locomotion_active():
                    self.redis_client.set("loco_vel_cmd", json.dumps(self.vel_cmd.tolist()))
                else:
                    self.redis_client.set("loco_vel_cmd", json.dumps([0.0, 0.0, 0.0]))
                
                # Control Inspire hands with Pico tracking
                self._control_inspire_hands_from_tracking(left_hand_data, right_hand_data)
                
                # Auto-transition from idle to preview when data arrives
                if self.state == "idle" and smplx_data is not None:
                    self.state = "preview"
                    print("\n→ PREVIEW mode: Pico data received!")
                
                # Process retargeting
                qpos = None
                if smplx_data is not None and self.retarget is not None:
                    qpos = self.process_retargeting(smplx_data)
                
                # Visualization and Redis
                if qpos is not None:
                    if self.is_interpolating:
                        qpos = self._get_interpolated_qpos(qpos)
                    elif self.state == "teleop_loco":
                        # In locomotion mode: legs + waist at LocoMode default, ARMS TRACK from GMR
                        qpos[7:7+12] = self.loco_policy.default_angles_reorder[:12]  # Legs
                        qpos[7+12:7+15] = self.loco_policy.default_angles_reorder[12:15]  # Waist
                        # Arms (7+15:7+29) keep GMR tracking from qpos
                    
                    self.data.qpos[:] = qpos
                    mj.mj_forward(self.model, self.data)
                    
                    # Camera follows the robot
                    robot_pos = self.data.xpos[self.robot_base_id]
                    viewer.cam.lookat[:] = robot_pos
                    viewer.cam.distance = 3.0
                    
                    # Extract mimic obs
                    if self.last_qpos is None:
                        self.last_qpos = qpos.copy()
                    mimic_obs = extract_mimic_obs(qpos, self.last_qpos, dt=1/self.args.target_fps)
                    self.last_qpos = qpos.copy()
                    
                    mimic_obs = self.apply_smooth(mimic_obs)
                    
                    if mimic_obs is not None:
                        # Lock waist_yaw to prevent rotation
                        mimic_obs[20] = 0.0
                    
                    # Capture frozen arms
                    if self._capture_frozen_arms and mimic_obs is not None:
                        self.frozen_arm_obs = mimic_obs[21:35].copy()
                        self._capture_frozen_arms = False
                    
                    # Apply frozen arms
                    if self.hands_paused and self.frozen_arm_obs is not None and mimic_obs is not None:
                        mimic_obs[21:35] = self.frozen_arm_obs
                    
                    # Yaw control
                    if mimic_obs is not None:
                        effective_state = self.interp_to_state if self.is_interpolating else self.state
                        if effective_state == "teleop_full":
                            mimic_obs[5] = 0.0
                        elif effective_state == "teleop_loco":
                            mimic_obs[5] = self.vel_cmd[2]
                        else:
                            mimic_obs[5] = 0.0
                    
                    # Neck tracking
                    neck_data = None
                    try:
                        neck_yaw, neck_pitch = human_head_to_robot_neck(smplx_data)
                        NECK_YAW_LIMIT = 1.22
                        NECK_PITCH_LIMIT = 0.77
                        neck_yaw = float(np.clip(neck_yaw, -NECK_YAW_LIMIT, NECK_YAW_LIMIT))
                        neck_pitch = float(np.clip(neck_pitch, -NECK_PITCH_LIMIT, NECK_PITCH_LIMIT))
                        neck_data = [neck_yaw, neck_pitch]
                    except:
                        pass
                    
                    # Send to Redis
                    if self._is_teleop_state():
                        if self.state == "paused":
                            if self.paused_mimic_obs is None and mimic_obs is not None:
                                self.paused_mimic_obs = mimic_obs.copy()
                            self.send_to_redis(self.paused_mimic_obs, neck_data)
                        else:
                            self.paused_mimic_obs = None
                            self.send_to_redis(mimic_obs, neck_data)
                
                viewer.sync()
                rate.sleep()
        
        print("\n\nExiting...")


def parse_args():
    parser = argparse.ArgumentParser(description="Merged Teleop - Pico + Unitree Controller")
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--actual_human_height", type=float, default=1.7)
    parser.add_argument("--target_fps", type=float, default=50.0)
    parser.add_argument("--redis_ip", type=str, default="localhost")
    parser.add_argument("--use_inspire_hands", action="store_true")
    parser.add_argument("--inspire_left_ip", type=str, default="192.168.123.210")
    parser.add_argument("--inspire_right_ip", type=str, default="192.168.123.211")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--smooth_window_size", type=int, default=3)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("  MERGED TELEOP")
    print("  Pico Finger Tracking + Unitree Controller")
    print("="*60)
    
    teleop = MergedTeleop(args)
    
    try:
        teleop.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

