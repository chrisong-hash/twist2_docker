#!/usr/bin/env python3
"""
Hybrid Locomotion + Teleoperation System
=========================================
This integrates with the existing TWIST2 workflow:

1. Start XRobotToolkit on Pico
2. Connect using the app
3. Run this script
4. MuJoCo preview shows motion - calibrate until tracking works
5. Press button to enable robot locomotion

Lower body (joints 0-11): RoboMimic LocoMode policy (joystick → walking)
Upper body (joints 12-28): TWIST2 GMR tracking (Pico VR → arm/waist motion)
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
    
    Compact output: single line progress, details only on failure.
    """
    errors = []
    warnings = []
    checks = []
    
    # Helper to update progress
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
        errors.append("PyTorch not installed\n   Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    
    # 4. MuJoCo
    status("Checking MuJoCo")
    try:
        import mujoco
        checks.append("✓")
    except ImportError:
        checks.append("✗")
        errors.append("MuJoCo not installed\n   Fix: pip install mujoco")
    
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
        errors.append(f"GMR import failed: {e}\n   Fix: pip install -e /workspace/GMR")
    
    # 6. Redis
    status("Checking Redis")
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        checks.append("✓")
    except ImportError:
        checks.append("✗")
        errors.append("redis-py not installed\n   Fix: pip install redis")
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
    
    # Clear progress line
    sys.stdout.write("\r" + " "*60 + "\r")
    sys.stdout.flush()
    
    # Summary - single line for success, details for failures
    check_str = "".join(checks)
    if errors:
        print(f"[Setup] {check_str} FAILED ({len(errors)} errors)\n")
        for err in errors:
            print(f"  ✗ {err}")
        print()
        return False
    elif warnings:
        print(f"[Setup] {check_str} OK ({len(warnings)} warnings)")
        if verbose:
            for w in warnings:
                print(f"  ⚠ {w}")
        return True
    else:
        print(f"[Setup] {check_str} OK")
        return True


# Run setup check before importing heavy modules
if __name__ == "__main__":
    # Check for --verbose flag early
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    if not check_setup(verbose=verbose):
        print("Please fix the errors above before running.\n")
        sys.exit(1)

# Now import heavy dependencies (after setup check passes)
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
        
        # Velocity ranges
        cmd_range = config["cmd_range"]
        self.range_velx = np.array([cmd_range["lin_vel_x"][0], cmd_range["lin_vel_x"][1]], dtype=np.float32)
        self.range_vely = np.array([cmd_range["lin_vel_y"][0], cmd_range["lin_vel_y"][1]], dtype=np.float32)
        self.range_velz = np.array([cmd_range["ang_vel_z"][0], cmd_range["ang_vel_z"][1]], dtype=np.float32)
        
        # State variables
        self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        
        # Load policy
        self.policy = torch.jit.load(self.policy_path)
        
        # Warm up
        for _ in range(50):
            with torch.inference_mode():
                obs_tensor = self.obs.reshape(1, -1).astype(np.float32)
                self.policy(torch.from_numpy(obs_tensor))
                
        print("[green][LocoMode] Locomotion policy loaded[/green]")
        print(f"  Velocity range: vx=[{self.range_velx[0]:.2f}, {self.range_velx[1]:.2f}], "
              f"vy=[{self.range_vely[0]:.2f}, {self.range_vely[1]:.2f}]")
        
        # Reorder for motor indices
        self.default_angles_reorder = np.zeros(29, dtype=np.float32)
        for i in range(len(self.joint2motor_idx)):
            motor_idx = self.joint2motor_idx[i]
            self.default_angles_reorder[motor_idx] = self.default_angles[i]
    
    def compute(self, qj, dqj, ang_vel, gravity_ori, vel_cmd):
        """Compute leg joint positions from velocity command"""
        # Scale velocity command
        vx = np.clip(vel_cmd[0], -1, 1) * (self.range_velx[1] if vel_cmd[0] > 0 else -self.range_velx[0])
        vy = np.clip(vel_cmd[1], -1, 1) * (self.range_vely[1] if vel_cmd[1] > 0 else -self.range_vely[0])
        vyaw = np.clip(vel_cmd[2], -1, 1) * (self.range_velz[1] if vel_cmd[2] > 0 else -self.range_velz[0])
        cmd = np.array([vx, vy, vyaw], dtype=np.float32) * self.cmd_scale
        
        # Reorder joints for policy
        for i in range(len(self.joint2motor_idx)):
            self.qj_obs[i] = qj[self.joint2motor_idx[i]]
            self.dqj_obs[i] = dqj[self.joint2motor_idx[i]]
        
        # Scale observations
        qj_scaled = (self.qj_obs - self.default_angles) * self.dof_pos_scale
        dqj_scaled = self.dqj_obs * self.dof_vel_scale
        ang_vel_scaled = ang_vel * self.ang_vel_scale
        
        # Build observation vector
        self.obs[:3] = ang_vel_scaled
        self.obs[3:6] = gravity_ori
        self.obs[6:9] = cmd
        self.obs[9:9 + self.num_actions] = qj_scaled
        self.obs[9 + self.num_actions:9 + self.num_actions * 2] = dqj_scaled
        self.obs[9 + self.num_actions * 2:9 + self.num_actions * 3] = self.action
        
        # Run policy
        with torch.inference_mode():
            obs_tensor = self.obs.reshape(1, -1).astype(np.float32)
            self.action = self.policy(torch.from_numpy(obs_tensor).clip(-100, 100)).clip(-100, 100).detach().numpy().squeeze()
        
        # Convert to joint positions
        loco_action = self.action * self.action_scale + self.default_angles
        
        # Reorder for motor indices - return full 29 joints
        action_reorder = np.zeros(29, dtype=np.float32)
        for i in range(len(self.joint2motor_idx)):
            motor_idx = self.joint2motor_idx[i]
            action_reorder[motor_idx] = loco_action[i]
                
        return action_reorder


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
        base_vel_local[:2],  # xy velocity (2 dims)
        root_pos[2:3],       # z position (1 dim)
        roll, pitch,         # roll, pitch (2 dims)
        base_ang_vel_local[2:3],  # yaw angular velocity (1 dim)
        robot_joints         # joint positions (29 dims)
    ])
    return mimic_obs


class HybridLocoTeleop:
    """
    Hybrid system: LocoMode legs + TWIST2 GMR upper body
    
    Follows same workflow as teleop_inspire.sh:
    1. Connect to Pico via XRobotStreamer
    2. Preview in MuJoCo
    3. Enable robot via button
    """
    
    def __init__(self, args):
        self.args = args
        self.robot_name = "unitree_g1"
        
        # State machine
        self.state = "idle"  # idle -> preview -> teleop -> exit
        self.locomotion_enabled = False
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        
        # Height estimation
        self.estimated_height = args.actual_human_height  # Start with default
        self.height_samples = []  # Collect samples for averaging
        self.height_estimation_done = False
        self.height_estimation_frames = 0
        self.retarget = None  # Will be initialized after height estimation
        
        # Initialize systems
        print("\n[cyan]Initializing Hybrid Locomotion + Teleoperation...[/cyan]")
        self._setup_locomotion_policy()
        self._setup_teleop_streamer()
        # Don't setup retargeting yet - wait for height estimation
        self._setup_mujoco()
        self._setup_redis()
        
        print("\n[green]Systems initialized![/green]")
        print(f"[yellow]Height estimation: Waiting for Pico data... (default: {self.estimated_height:.2f}m)[/yellow]")
        self._print_controls()
    
    def _setup_locomotion_policy(self):
        """Load RoboMimic locomotion policy"""
        print("\n[1/5] Loading locomotion policy...")
        self.loco_policy = LocoModePolicy()
    
    def _setup_teleop_streamer(self):
        """Initialize Pico VR connection via XRobotStreamer"""
        print("\n[2/5] Connecting to Pico VR...")
        self.teleop_streamer = XRobotStreamer()
        print("[green]XRobotStreamer initialized - waiting for Pico data[/green]")
    
    def _setup_retargeting(self, height=None):
        """Initialize GMR for upper body retargeting with given height"""
        if height is None:
            height = self.estimated_height
        
        # Suppress GMR's verbose output (DoF names, body names, etc.)
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
    
    def _setup_mujoco(self):
        """Setup MuJoCo simulation for preview"""
        print("\n[4/5] Setting up MuJoCo preview...")
        xml_path = str(ROBOT_XML_DICT["unitree_g1"])
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        # Don't set initial pose - let MuJoCo use model's default
        # This prevents spasming legs before Pico data arrives
        self.last_qpos = None  # Will be set on first valid frame
        self._last_valid_qpos = None  # Will be set on first valid frame
        self._using_fallback = False
        # Get robot base body ID for camera tracking
        self.robot_base_id = self.model.body("pelvis").id
        print("[green]MuJoCo ready[/green]")
    
    def _setup_redis(self):
        """Setup Redis connection for robot communication"""
        print("\n[5/5] Connecting to Redis...")
        self.redis_client = redis.Redis(host=self.args.redis_ip, port=6379, db=0)
        self.redis_pipeline = self.redis_client.pipeline()
        self.redis_client.ping()
        print("[green]Redis connected[/green]")
    
    def _print_controls(self):
        print("\n" + "="*60)
        print("  HYBRID LOCOMOTION + TELEOPERATION")
        print("="*60)
        print("\n[yellow]Controls:[/yellow]")
        print("  Right A button (key_one): Toggle preview/teleop mode")
        print("  B button (key_two)      : Exit program")
        print("  Left A button (key_one) : Exit program (alternate)")
        print("  Left joystick           : Walk forward/back/strafe")
        print("  Right joystick          : Rotate left/right")
        print("  Right trigger           : HOLD to enable leg locomotion")
        print("\n[yellow]States:[/yellow]")
        print("  idle    : Waiting for Pico data")
        print("  preview : MuJoCo shows your motion (calibrate here)")
        print("  teleop  : Sending to robot via Redis")
        print("\n[cyan]Workflow:[/cyan]")
        print("  1. Ensure Pico is connected via XRobotToolkit app")
        print("  2. Move around until MuJoCo reflects your motion")
        print("  3. Press Right A to enter teleop mode")
        print("  4. HOLD Right Trigger + use joystick to walk")
        print("  5. Release trigger to stop walking (upper body still tracks)")
        print("="*60 + "\n")
    
    def get_teleop_data(self):
        """Get data from Pico VR"""
        if self.teleop_streamer is not None:
            return self.teleop_streamer.get_current_frame()
        return None, None, None, None, None
    
    def update_state(self, controller_data):
        """Update state machine based on controller input"""
        if controller_data is None:
            return
        
        # Controller data has nested structure: {'RightController': {...}, 'LeftController': {...}}
        right_ctrl = controller_data.get("RightController", {})
        left_ctrl = controller_data.get("LeftController", {})
        
        # Right A button - toggle state
        right_key_one = right_ctrl.get("key_one", False)
        if right_key_one and not self._right_key_was_pressed:
            if self.state == "idle":
                self.state = "preview"
                print("[cyan]→ PREVIEW mode: MuJoCo shows your motion[/cyan]")
            elif self.state == "preview":
                self.state = "teleop"
                print("[green]→ TELEOP mode: Sending to robot![/green]")
            elif self.state == "teleop":
                self.state = "preview"
                print("[cyan]→ PREVIEW mode: Robot paused[/cyan]")
        self._right_key_was_pressed = right_key_one
        
        # Left A button - exit
        left_key_one = left_ctrl.get("key_one", False)
        if left_key_one and not self._left_key_was_pressed:
            self.state = "exit"
            print("\n→ EXIT requested (Left A)")
        self._left_key_was_pressed = left_key_one
        
        # B button (key_two) on either controller - exit
        right_key_two = right_ctrl.get("key_two", False)
        left_key_two = left_ctrl.get("key_two", False)
        if (right_key_two or left_key_two) and not self._b_key_was_pressed:
            self.state = "exit"
            print("\n→ EXIT requested (B button)")
        self._b_key_was_pressed = right_key_two or left_key_two
        
        # Joystick for locomotion
        left_axis = left_ctrl.get("axis", [0, 0])
        right_axis = right_ctrl.get("axis", [0, 0])
        
        # Left joystick for movement
        self.vel_cmd[0] = left_axis[1]   # forward/backward
        self.vel_cmd[1] = -left_axis[0]  # strafe
        # Right joystick for rotation
        self.vel_cmd[2] = -right_axis[0]  # yaw
        
        # Apply deadzone
        for i in range(3):
            if abs(self.vel_cmd[i]) < 0.1:
                self.vel_cmd[i] = 0.0
        
        # Right trigger enables locomotion
        trigger_right = right_ctrl.get("index_trig", 0.0)
        if isinstance(trigger_right, bool):
            trigger_right = 1.0 if trigger_right else 0.0
        self.locomotion_enabled = trigger_right > 0.5
    
    def _validate_quaternions(self, smplx_data):
        """Check if smplx_data contains valid quaternions (non-zero norm)"""
        if smplx_data is None:
            return False
        
        # smplx_data is {joint_name: [[x,y,z], [w,x,y,z]], ...}
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
        # Check reasonable bounds (positions < 100m, angles < 2*pi)
        if np.any(np.abs(qpos[:3]) > 100):  # root position
            return False
        if np.any(np.abs(qpos[7:]) > 10):  # joint angles
            return False
        return True
    
    def process_retargeting(self, smplx_data):
        """Run GMR retargeting on SMPLX data"""
        # Validate quaternions before processing
        if not self._validate_quaternions(smplx_data):
            # Return last known good qpos if available
            self._using_fallback = True
            return self._last_valid_qpos  # May be None if no valid frame yet
        
        try:
            # offset_to_ground=True shifts robot so feet are at ground level
            # Without this, robot position = human's world position (could be anywhere)
            qpos = self.retarget.retarget(smplx_data, offset_to_ground=True)
        except ValueError as e:
            # Catch quaternion errors - use fallback
            if "zero norm" in str(e):
                self._using_fallback = True
                return self._last_valid_qpos
            raise
        except Exception:
            # Any other error - use fallback
            self._using_fallback = True
            return self._last_valid_qpos
        
        # Validate qpos values
        if not self._is_qpos_valid(qpos):
            self._using_fallback = True
            return self._last_valid_qpos
        
        # Save as last valid qpos
        self._last_valid_qpos = qpos.copy()
        self._using_fallback = False
        
        # For MuJoCo visualization only: use standing pose for legs
        # The actual robot leg control is handled by sim2real with LocoMode + real robot state
        standing_legs = np.array([
            -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,  # left leg
            -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,  # right leg
        ], dtype=np.float32)
        qpos[7:7+12] = standing_legs
        
        # Upper body (12-28) comes from GMR retargeting - this goes to sim2real via mimic_obs
        # Legs (0-11) will be computed by LocoMode in sim2real with real robot state feedback
        
        return qpos
    
    def send_to_redis(self, mimic_obs, neck_data=None):
        """Send observations to Redis for sim2real"""
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
        
        # Send velocity command for LocoMode (only when trigger held)
        if self.locomotion_enabled:
            self.redis_pipeline.set(
                "loco_vel_cmd",
                json.dumps(self.vel_cmd.tolist())
            )
        else:
            # Zero velocity when not walking
            self.redis_pipeline.set(
                "loco_vel_cmd",
                json.dumps([0.0, 0.0, 0.0])
            )
        
        t_action = int(time.time() * 1000)
        self.redis_pipeline.set("t_action", t_action)
        self.redis_pipeline.execute()
    
    def _print_status(self, controller, smplx_data):
        """Print live status on a single replacing line (max 79 chars for terminal)"""
        # Build compact status string
        p = "P" if smplx_data is not None else "-"
        c = "C" if controller is not None else "-"
        
        # Get joystick values
        lx, ly, rx = 0.0, 0.0, 0.0
        trig = 0.0
        if controller:
            left_ctrl = controller.get("LeftController", {})
            right_ctrl = controller.get("RightController", {})
            left_axis = left_ctrl.get("axis", [0, 0])
            right_axis = right_ctrl.get("axis", [0, 0])
            lx, ly = left_axis[0] if len(left_axis) > 0 else 0, left_axis[1] if len(left_axis) > 1 else 0
            rx = right_axis[0] if len(right_axis) > 0 else 0
            trig = right_ctrl.get("index_trig", 0)
            if isinstance(trig, bool):
                trig = 1.0 if trig else 0.0
        
        loco = "W" if self.locomotion_enabled else "-"
        fb = "F" if getattr(self, '_using_fallback', False) else "-"
        # Compact: [state] P:- C:- L:(+0.0,+0.0) R:+0.0 T:0 W:- F:- v:(+0.0,+0.0,+0.0)
        status = (f"[{self.state:7s}] {p}{c}{fb} "
                  f"L:({lx:+.1f},{ly:+.1f}) R:{rx:+.1f} T:{trig:.0f} {loco} "
                  f"v:({self.vel_cmd[0]:+.1f},{self.vel_cmd[1]:+.1f},{self.vel_cmd[2]:+.1f})")
        
        # Print with carriage return - keep under 80 chars
        sys.stdout.write(f"\r{status:<79}")
        sys.stdout.flush()
    
    def run(self):
        """Main loop"""
        # Suppress loop_rate_limiters warnings
        import logging
        logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)
        
        rate = RateLimiter(frequency=self.args.target_fps)
        self._right_key_was_pressed = False
        self._left_key_was_pressed = False
        self._b_key_was_pressed = False
        
        print(f"\nStarting in state: {self.state}")
        print("Waiting for Pico VR data...")
        
        with mjv.launch_passive(
            model=self.model, 
            data=self.data, 
            show_left_ui=False, 
            show_right_ui=False
        ) as viewer:
            # Match original teleop settings
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1
            
            while viewer.is_running() and self.state != "exit":
                # Get Pico data
                smplx_data, left_hand, right_hand, controller, headset = self.get_teleop_data()
                
                # Update state machine
                self.update_state(controller)
                
                # Print live status (replacing line)
                self._print_status(controller, smplx_data)
                
                # Auto-transition from idle to preview when data arrives
                if self.state == "idle" and smplx_data is not None:
                    self.state = "preview"
                    print("\n→ PREVIEW mode: Pico data received!")
                
                # Process retargeting if we have data
                if smplx_data is not None:
                    qpos = self.process_retargeting(smplx_data)
                    
                    # Skip if no valid qpos yet
                    if qpos is None:
                        continue
                    
                    # Update MuJoCo visualization
                    self.data.qpos[:] = qpos
                    mj.mj_forward(self.model, self.data)
                    
                    # Camera follows the robot
                    robot_pos = self.data.xpos[self.robot_base_id]
                    viewer.cam.lookat[:] = robot_pos
                    viewer.cam.distance = 3.0
                    
                    # Extract mimic observations (use qpos as last_qpos if first frame)
                    if self.last_qpos is None:
                        self.last_qpos = qpos.copy()
                    mimic_obs = extract_mimic_obs(qpos, self.last_qpos, dt=1/self.args.target_fps)
                    self.last_qpos = qpos.copy()
                    
                    # Get neck data
                    neck_data = None
                    try:
                        neck_yaw, neck_pitch = human_head_to_robot_neck(smplx_data)
                        neck_data = [neck_yaw * 0.5, neck_pitch * 0.5]
                    except:
                        pass
                    
                    # Send to Redis if in teleop mode
                    if self.state == "teleop":
                        self.send_to_redis(mimic_obs, neck_data)
                
                viewer.sync()
                rate.sleep()
        
        print("\n\nExiting...")


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Locomotion + Teleoperation")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--actual_human_height", type=float, default=1.65)
    parser.add_argument("--redis_ip", type=str, default="localhost")
    parser.add_argument("--target_fps", type=int, default=50)
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose setup output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hybrid = HybridLocoTeleop(args)
    hybrid.run()
