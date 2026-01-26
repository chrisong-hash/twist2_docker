#!/usr/bin/env python3
"""
Hybrid Locomotion + Teleoperation System v3 (GROOT-style controls)
===================================================================
Separate upper/lower body control. Uses GROOT GearWBC policy for
stable locomotion with arbitrary arm poses.

States:
  idle    : Waiting for Pico VR data
  preview : MuJoCo shows your motion (calibrate here)
  teleop  : Active teleoperation
  exit    : Shutdown

Controls:
  Left X (tap)       : Toggle preview ↔ teleop
  Right A (teleop)   : Toggle UPPER body (tracking ↔ frozen)
  Right B (teleop)   : Toggle LOWER body (standing ↔ walking)
  Right A+B (hold 1s): EMERGENCY SHUTDOWN
  Left joystick      : Walk direction (when walking mode)
  Right joystick     : Rotation (when walking mode)
  Left trigger       : Open left hand
  Right trigger      : Open right hand
  Left grip          : Close left hand
  Right grip         : Close right hand

Upper body:
  TRACKING: Arms follow your motion (GMR)
  FROZEN:   Arms hold current position

Lower body:
  STANDING: Full body teleop (TWIST2 policy)
  WALKING:  Joystick locomotion (GearWBC policy)
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
from robot_control.inspire_hand_wrapper import DualHandController


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
        
        # Track if policy needs reset on mode switch
        self._needs_reset = True
        
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
    
    def reset(self):
        """Reset action buffer - call when switching to LocoMode to avoid stale history"""
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self._needs_reset = False
        print("[LocoMode] Action buffer reset for clean mode switch")


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
    Hybrid system with improved state machine and smooth transitions.
    
    States:
      idle        : Waiting for Pico VR data
      preview     : MuJoCo shows your motion (calibrate here)
      teleop_full : Full body teleop (TWIST2 controls all joints)
      teleop_loco : Locomotion mode (LocoMode legs + GMR upper body)
      paused      : Robot holds default standing pose
    
    All mode transitions use 1-second interpolation for smooth motion.
    """
    
    # Interpolation duration in seconds
    INTERP_DURATION = 1.0
    
    # Default standing pose for legs (used in pause and teleop_full)
    DEFAULT_STANDING_LEGS = np.array([
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,  # left leg
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,  # right leg
    ], dtype=np.float32)
    
    def __init__(self, args):
        self.args = args
        self.robot_name = "unitree_g1"
        
        # State machine - valid states: idle, preview, teleop, exit
        self.state = "idle"
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        
        # Upper body: tracking or frozen
        self.upper_body_paused = False
        self._frozen_upper_qpos = None  # Arms pose when frozen
        
        # Lower body: standing (TWIST2) or walking (GearWBC)
        self.locomotion_active = False  # True = walking, False = standing
        
        # Frozen pose when returning to preview (robot holds position)
        self._frozen_preview_qpos = None  # Full body pose held when leaving teleop
        
        # Interpolation state
        self.is_interpolating = False
        self.interp_start_time = 0.0
        self.interp_start_qpos = None  # Starting joint positions
        self.interp_target_qpos = None  # Target joint positions
        self.interp_from_state = None   # State we're transitioning from
        self.interp_to_state = None     # State we're transitioning to
        
        # Button state tracking (for edge detection and hold detection)
        self._right_a_was_pressed = False
        self._left_x_was_pressed = False
        self._left_y_was_pressed = False
        self._ab_hold_start_time = None  # For A+B hold detection
        
        # Height setting
        self.estimated_height = args.actual_human_height
        self.retarget = None  # Will be initialized in _setup_retargeting
        
        # Smooth filtering settings
        self.enable_smooth = args.smooth
        self.smooth_window_size = args.smooth_window_size
        self.smooth_history = []  # Store recent observations for sliding window
        
        # Velocity scaling (to match GROOT's intended walking speed range)
        # GROOT uses raw commands ~0.2-0.4, which after cmd_scale [2.0, 2.0, 0.5] gives reasonable speeds
        # Without scaling, joystick at 1.0 would give 2.0 m/s - way too fast!
        self.vel_scale_forward = args.vel_scale_forward   # Max ~0.6 m/s forward
        self.vel_scale_backward = args.vel_scale_backward # Max ~0.4 m/s backward (more conservative)
        self.vel_scale_strafe = args.vel_scale_strafe     # Max ~0.5 m/s strafe
        self.vel_scale_yaw = args.vel_scale_yaw           # Max ~0.2 rad/s yaw
        
        # Inspire hands (toggle: 0=open, 1=closed)
        self.hand_controller = None
        self._left_hand_closed = False   # Toggle state
        self._right_hand_closed = False
        self._left_trigger_was_pressed = False   # For edge detection
        self._right_trigger_was_pressed = False
        
        # Initialize systems
        print("\n[cyan]Initializing Hybrid Locomotion + Teleoperation v2...[/cyan]")
        self._setup_locomotion_policy()
        self._setup_teleop_streamer()
        self._setup_retargeting()  # Initialize GMR with default height
        self._setup_mujoco()
        self._setup_redis()
        self._setup_inspire_hands()
        
        print("\n[green]Systems initialized![/green]")
        if self.enable_smooth:
            print(f"[cyan]Smooth filtering: ENABLED (window size: {self.smooth_window_size} frames)[/cyan]")
        else:
            print("[yellow]Smooth filtering: DISABLED[/yellow]")
        
        # Print velocity scaling info
        print(f"\n[cyan]Velocity scaling (for GROOT GearWBC compatibility):[/cyan]")
        print(f"  Forward:  {self.vel_scale_forward:.2f} → max {self.vel_scale_forward * 2.0:.1f} m/s")
        print(f"  Backward: {self.vel_scale_backward:.2f} → max {self.vel_scale_backward * 2.0:.1f} m/s")
        print(f"  Strafe:   {self.vel_scale_strafe:.2f} → max {self.vel_scale_strafe * 2.0:.1f} m/s")
        print(f"  Yaw:      {self.vel_scale_yaw:.2f} → max {self.vel_scale_yaw * 0.5:.2f} rad/s")
        print(f"  [yellow]Tune with --vel_scale_forward, --vel_scale_backward, etc.[/yellow]")
        
        self._print_controls()
    
    def _setup_locomotion_policy(self):
        """Load RoboMimic locomotion policy"""
        print("\n[1/5] Loading locomotion policy...")
        self.loco_policy = LocoModePolicy()
    
    def _setup_teleop_streamer(self):
        """Initialize Pico VR connection via XRobotStreamer"""
        print("\n[2/5] Connecting to Pico VR...")
        try:
            self.teleop_streamer = XRobotStreamer()
            print("[green]XRobotStreamer initialized - waiting for Pico data[/green]")
        except Exception as e:
            print(f"[red]ERROR: Failed to initialize XRobotStreamer: {e}[/red]")
            print("[yellow]Make sure:")
            print("  1. XRobotToolkit app is running on Pico")
            print("  2. Pico is connected to this PC via the app")
            print("  3. xrobotoolkit_sdk is installed: pip install xrobotoolkit-sdk[/yellow]")
            import traceback
            traceback.print_exc()
            self.teleop_streamer = None
    
    def _setup_retargeting(self, height=None):
        """Initialize GMR for upper body retargeting with given height"""
        print("\n[3/5] Setting up GMR retargeting...")
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
        print(f"[green]GMR initialized (height: {height:.2f}m)[/green]")
    
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
        # Clear any previous shutdown signal
        self.redis_client.delete("robot_shutdown")
        print("[green]Redis connected[/green]")
    
    def _setup_inspire_hands(self):
        """Setup Inspire hand controllers (enabled by default)"""
        if getattr(self.args, 'no_inspire_hands', False):
            print("\n[6/6] Inspire hands: [yellow]DISABLED (--no_inspire_hands)[/yellow]")
            return
        
        print("\n[6/6] Connecting to Inspire hands...")
        try:
            self.hand_controller = DualHandController(
                left_ip=self.args.inspire_left_ip,
                right_ip=self.args.inspire_right_ip,
                async_mode=True
            )
            print(f"[green]Inspire hands connected[/green]")
            print(f"  Left: {self.args.inspire_left_ip}")
            print(f"  Right: {self.args.inspire_right_ip}")
        except Exception as e:
            print(f"[red]WARNING: Failed to connect to Inspire hands: {e}[/red]")
            self.hand_controller = None
    
    def _control_inspire_hands(self, controller_data):
        """Control Inspire hands - trigger=open, grip=close"""
        if self.hand_controller is None:
            return
        
        # Only control hands in teleop state (not idle/preview)
        if self.state not in ["teleop", "teleop_full", "teleop_loco", "paused"]:
            return
        
        left_ctrl = controller_data.get('LeftController', {})
        right_ctrl = controller_data.get('RightController', {})
        
        # Debug: print available keys once
        if not hasattr(self, '_controller_keys_printed'):
            print(f"[HAND DEBUG] Left controller keys: {list(left_ctrl.keys())}")
            print(f"[HAND DEBUG] Right controller keys: {list(right_ctrl.keys())}")
            self._controller_keys_printed = True
        
        # Get trigger and grip values (Pico uses 'index_trig' and 'grip')
        left_trigger = left_ctrl.get('index_trig', 0)
        right_trigger = right_ctrl.get('index_trig', 0)
        left_grip = left_ctrl.get('grip', 0)
        right_grip = right_ctrl.get('grip', 0)
        
        # Convert bool to float if needed
        if isinstance(left_trigger, bool):
            left_trigger = 1.0 if left_trigger else 0.0
        if isinstance(right_trigger, bool):
            right_trigger = 1.0 if right_trigger else 0.0
        if isinstance(left_grip, bool):
            left_grip = 1.0 if left_grip else 0.0
        if isinstance(right_grip, bool):
            right_grip = 1.0 if right_grip else 0.0
        
        # Track previous state for edge detection (print messages)
        prev_left = self._left_hand_closed
        prev_right = self._right_hand_closed
        
        # Trigger = open (0.0), Grip = close (1.0)
        # Grip takes priority if both pressed
        if left_grip > 0.5:
            self._left_hand_closed = True
        elif left_trigger > 0.5:
            self._left_hand_closed = False
        
        if right_grip > 0.5:
            self._right_hand_closed = True
        elif right_trigger > 0.5:
            self._right_hand_closed = False
        
        # Print state changes
        if self._left_hand_closed != prev_left:
            state = "CLOSED" if self._left_hand_closed else "OPEN"
            print(f"[HAND] Left: {state}")
        if self._right_hand_closed != prev_right:
            state = "CLOSED" if self._right_hand_closed else "OPEN"
            print(f"[HAND] Right: {state}")
        
        # Send to hands (0=open, 1=closed)
        left_pos = 1.0 if self._left_hand_closed else 0.0
        right_pos = 1.0 if self._right_hand_closed else 0.0
        self.hand_controller.ctrl_dual_hand(left_pos, right_pos)
    
    def _send_shutdown_signal(self):
        """Send shutdown signal to robot server via Redis"""
        try:
            # Send shutdown signal that server will check
            self.redis_client.set("robot_shutdown", "1")
            print("[red]  Shutdown signal sent![/red]")
        except Exception as e:
            print(f"[red]  Warning: Could not send shutdown signal: {e}[/red]")
    
    def _print_controls(self):
        print("\n" + "="*60)
        print("  HYBRID LOCOMOTION + TELEOPERATION v3")
        print("="*60)
        print("\n[yellow]Controls:[/yellow]")
        print("  Right A            : Toggle preview ↔ teleop")
        print("  Left X (teleop)    : Toggle hands (open ↔ closed)")
        print("  Left Y (teleop)    : Toggle walking (standing ↔ walking)")
        print("  [red]A+B (hold 1s)    : EMERGENCY SHUTDOWN[/red]")
        print("  Left joystick      : Walk direction (in walking mode)")
        print("  Right joystick     : Rotation (in walking mode)")
        print("  Triggers           : Open hands")
        print("  Grips              : Close hands")
        print("\n[yellow]States:[/yellow]")
        print("  idle    → Pico connects → preview")
        print("  preview ↔ Right A      ↔ teleop")
        print("\n[cyan]Workflow:[/cyan]")
        print("  1. Connect Pico via XRobotToolkit → auto enters preview")
        print("  2. Calibrate until MuJoCo reflects your motion")
        print("  3. Press Right A → enters teleop_full")
        print("  4. Press Left X → switches to teleop_loco (walking mode)")
        print("  5. Use joysticks to walk in teleop_loco")
        print("  6. Press Left X again → back to teleop_full")
        print("  7. Press Right A → pause, Right A again → unpause")
        print("\n[cyan]Note:[/cyan] All transitions have 1-second smooth interpolation")
        print("="*60 + "\n")
    
    def get_teleop_data(self):
        """Get data from Pico VR"""
        if self.teleop_streamer is not None:
            try:
                return self.teleop_streamer.get_current_frame()
            except Exception as e:
                # Silently handle errors - Pico might disconnect temporarily
                return None, None, None, None, None
        return None, None, None, None, None
    
    def _start_interpolation(self, from_state, to_state, current_qpos):
        """Start interpolation from current pose to target state's pose"""
        self.is_interpolating = True
        self.interp_start_time = time.time()
        self.interp_from_state = from_state
        self.interp_to_state = to_state
        
        # Reset smooth history on state transition to avoid carrying old data
        self.reset_smooth_history()
        
        # Reset LocoMode policy when switching TO teleop_loco to avoid stale action history
        if to_state == "teleop_loco":
            self.loco_policy.reset()
        
        # Save starting position
        if current_qpos is not None:
            self.interp_start_qpos = current_qpos.copy()
        else:
            # Use MuJoCo's current qpos as fallback
            self.interp_start_qpos = self.data.qpos.copy()
        
        # Determine target position based on target state
        self.interp_target_qpos = self.interp_start_qpos.copy()
        
        if to_state == "paused":
            # Target: default standing pose for all joints
            self.interp_target_qpos[7:7+12] = self.DEFAULT_STANDING_LEGS
            # Keep upper body at current position (will be overwritten if we have retarget data)
        elif to_state == "teleop_full":
            # Target: current retargeted pose (full body from GMR)
            # In teleop_full, entire body tracks - no override needed
            # interp_target_qpos is already set from current_qpos (GMR tracking)
            pass
        elif to_state == "teleop_loco":
            # Target: locomotion mode - legs + waist from LocoMode, ARMS TRACK from GMR
            self.interp_target_qpos[7:7+12] = self.loco_policy.default_angles_reorder[:12]  # Legs
            self.interp_target_qpos[7+12:7+15] = self.loco_policy.default_angles_reorder[12:15]  # Waist
            # Arms (7+15:7+29) keep GMR tracking from interp_start_qpos
        
        print(f"\n[cyan]→ Interpolating: {from_state} → {to_state} (1.0s)[/cyan]")
    
    def _get_interpolated_qpos(self, current_target_qpos):
        """Get interpolated qpos between start and target"""
        if not self.is_interpolating:
            return current_target_qpos
        
        elapsed = time.time() - self.interp_start_time
        alpha = min(1.0, elapsed / self.INTERP_DURATION)
        
        # Apply smootherstep for acceleration-capped transition
        # This transforms linear alpha into smooth S-curve with zero velocity/acceleration at boundaries
        # Formula: 6t^5 - 15t^4 + 10t^3 (quintic smoothstep)
        alpha = alpha * alpha * alpha * (alpha * (6 * alpha - 15) + 10)
        
        # Update target position with current data (if available)
        if current_target_qpos is not None:
            self.interp_target_qpos = current_target_qpos.copy()
            # Apply state-specific poses
            if self.interp_to_state == "paused":
                self.interp_target_qpos[7:7+12] = self.DEFAULT_STANDING_LEGS
            elif self.interp_to_state == "teleop_full":
                # In teleop_full, entire body tracks GMR - no override needed
                pass
            elif self.interp_to_state == "teleop_loco":
                # Locomotion mode: legs + waist at LocoMode default, arms track GMR
                self.interp_target_qpos[7:7+12] = self.loco_policy.default_angles_reorder[:12]  # Legs
                self.interp_target_qpos[7+12:7+15] = self.loco_policy.default_angles_reorder[12:15]  # Waist
                # Arms (7+15:7+29) keep GMR tracking
        
        # Linear interpolation
        interpolated = (1.0 - alpha) * self.interp_start_qpos + alpha * self.interp_target_qpos
        
        # Check if interpolation is complete
        if alpha >= 1.0:
            self.is_interpolating = False
            self.state = self.interp_to_state
            print(f"\n[green]→ {self.state.upper()} mode active[/green]")
        
        return interpolated
    
    def update_state(self, controller_data, current_qpos=None):
        """Update state machine based on controller input
        
        NEW BUTTON MAPPING (GROOT-style):
        - Right A (tap): Toggle UPPER body pause
        - Right B (tap): Toggle LOWER body pause  
        - Right A+B (hold 1s): EMERGENCY QUIT
        - Left X (tap): Toggle locomotion mode (standing ↔ walking)
        
        Args:
            controller_data: Dict with RightController and LeftController data
            current_qpos: Current robot qpos for interpolation (optional)
        """
        if controller_data is None:
            return
        
        right_ctrl = controller_data.get("RightController", {})
        left_ctrl = controller_data.get("LeftController", {})
        
        # Get current button states
        right_a = right_ctrl.get("key_one", False)  # Right A
        right_b = right_ctrl.get("key_two", False)  # Right B
        left_x = left_ctrl.get("key_one", False)    # Left X
        left_y = left_ctrl.get("key_two", False)    # Left Y
        
        # A+B held for 1 second = EMERGENCY QUIT
        if right_a and right_b:
            if self._ab_hold_start_time is None:
                self._ab_hold_start_time = time.time()
            elif time.time() - self._ab_hold_start_time >= 1.0:
                self.state = "exit"
                print("\n[red]→ EMERGENCY SHUTDOWN (A+B held 1s)[/red]")
                print("[red]  Sending shutdown signal to robot server...[/red]")
                self._send_shutdown_signal()
                return
        else:
            self._ab_hold_start_time = None
        
        # Detect button presses (rising edge)
        right_a_pressed = right_a and not self._right_a_was_pressed and not right_b
        left_x_pressed = left_x and not self._left_x_was_pressed
        left_y_pressed = left_y and not self._left_y_was_pressed
        
        # State transitions
        if self.state == "exit":
            pass  # No button processing in exit state
        elif self.state == "idle":
            # A button starts preview
            if right_a_pressed:
                self.state = "preview"
                print("\n[cyan]→ PREVIEW mode: Calibrate your pose[/cyan]")
        elif self.state == "preview":
            # A enters teleop
            if right_a_pressed:
                self.state = "teleop"
                self._frozen_preview_qpos = None  # Clear frozen pose when entering teleop
                print("\n[green]→ TELEOP mode: Robot follows you![/green]")
                self._print_pause_status()
        elif self.state == "teleop":
            # A: Return to preview (robot holds current position)
            if right_a_pressed:
                if current_qpos is not None:
                    self._frozen_preview_qpos = current_qpos.copy()
                self.state = "preview"
                print("\n[cyan]→ PREVIEW mode: Robot holding position[/cyan]")
                print("  Press A to resume teleop")
            
            # X toggles hands (all open ↔ all closed)
            if left_x_pressed:
                # Toggle both hands together
                new_state = not self._left_hand_closed
                self._left_hand_closed = new_state
                self._right_hand_closed = new_state
                state_str = "CLOSED" if new_state else "OPEN"
                print(f"\n[cyan]→ HANDS: {state_str}[/cyan]")
                if self.hand_controller:
                    pos = 1.0 if new_state else 0.0
                    self.hand_controller.ctrl_dual_hand(pos, pos)
            
            # Y toggles lower body (standing ↔ walking)
            if left_y_pressed:
                self.locomotion_active = not self.locomotion_active
                if self.locomotion_active:
                    print(f"\n[cyan]→ LOWER: WALKING (joystick locomotion)[/cyan]")
                else:
                    print(f"\n[green]→ LOWER: STANDING (full body teleop)[/green]")
                self._print_pause_status()
        
        # Update button state tracking
        self._right_a_was_pressed = right_a
        self._left_x_was_pressed = left_x
        self._left_y_was_pressed = left_y
        
        # Joystick for locomotion - only active when locomotion_active
        left_axis = left_ctrl.get("axis", [0, 0])
        right_axis = right_ctrl.get("axis", [0, 0])
        
        if self.state == "teleop" and self.locomotion_active:
            # Left joystick for movement (with velocity scaling for GROOT compatibility)
            # GROOT policy expects raw commands in ~0.2-0.4 range, not 0-1
            raw_forward = left_axis[1] if len(left_axis) > 1 else 0.0
            raw_strafe = -left_axis[0] if len(left_axis) > 0 else 0.0
            raw_yaw = -right_axis[0] if len(right_axis) > 0 else 0.0
            
            # Apply asymmetric scaling for forward vs backward (backward is less stable)
            if raw_forward >= 0:
                self.vel_cmd[0] = raw_forward * self.vel_scale_forward
            else:
                self.vel_cmd[0] = raw_forward * self.vel_scale_backward
            
            self.vel_cmd[1] = raw_strafe * self.vel_scale_strafe
            self.vel_cmd[2] = raw_yaw * self.vel_scale_yaw
        else:
            # Zero velocity when not in locomotion mode
            self.vel_cmd[:] = 0.0
    
    def _print_pause_status(self):
        """Print current status"""
        upper_status = "[yellow]FROZEN[/yellow]" if self.upper_body_paused else "[green]TRACKING[/green]"
        lower_status = "[cyan]WALKING[/cyan]" if self.locomotion_active else "[green]STANDING[/green]"
        print(f"  Upper: {upper_status} | Lower: {lower_status}")
    
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
        
        # Return full qpos - leg pose will be set based on state in run() loop:
        # - teleop_full/paused: DEFAULT_STANDING_LEGS
        # - teleop_loco: GMR legs (sim2real uses LocoMode with real robot state)
        # - interpolating: blend between states
        
        return qpos
    
    def _is_locomotion_active(self):
        """Check if locomotion mode is active (for velocity commands)"""
        # New state model: check locomotion_active flag
        if self.state == "teleop" and self.locomotion_active:
            return True
        # Legacy state: teleop_loco
        if self.state == "teleop_loco":
            return True
        # Also active during interpolation TO teleop_loco (legacy)
        if self.is_interpolating and self.interp_to_state == "teleop_loco":
            elapsed = time.time() - self.interp_start_time
            alpha = min(1.0, elapsed / self.INTERP_DURATION)
            return alpha > 0.5  # Enable after halfway through interpolation
        return False
    
    def apply_smooth(self, mimic_obs):
        """Apply sliding window smoothing to mimic observations to reduce jitter"""
        if not self.enable_smooth or mimic_obs is None:
            return mimic_obs
            
        # Convert to numpy array if needed
        obs_array = np.array(mimic_obs) if not isinstance(mimic_obs, np.ndarray) else mimic_obs.copy()
        
        # Add current observation to history
        self.smooth_history.append(obs_array)
        
        # Keep only the recent window_size observations
        if len(self.smooth_history) > self.smooth_window_size:
            self.smooth_history.pop(0)
            
        # Apply sliding window average
        if len(self.smooth_history) >= 2:  # Need at least 2 observations for smoothing
            # Stack all observations in history
            history_stack = np.stack(self.smooth_history, axis=0)  # Shape: (history_len, obs_dim)
            # Compute mean across the time dimension
            smoothed_obs = np.mean(history_stack, axis=0)
            return smoothed_obs
        else:
            # Not enough history, return original observation
            return obs_array
    
    def reset_smooth_history(self):
        """Reset smooth history (call when transitioning states)"""
        self.smooth_history = []
    
    def _send_velocity_command_only(self):
        """Send only velocity command to Redis (called every loop iteration in teleop mode)"""
        if self._is_locomotion_active():
            # Send current velocity command when locomotion is enabled
            vel_cmd_to_send = self.vel_cmd.copy()
            self.redis_client.set(
                "loco_vel_cmd",
                json.dumps(vel_cmd_to_send.tolist())
            )
        else:
            # Zero velocity when not in locomotion mode
            self.redis_client.set(
                "loco_vel_cmd",
                json.dumps([0.0, 0.0, 0.0])
            )
    
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
        
        # Send velocity command based on locomotion state
        if self._is_locomotion_active():
            self.redis_pipeline.set(
                "loco_vel_cmd",
                json.dumps(self.vel_cmd.tolist())
            )
        else:
            self.redis_pipeline.set(
                "loco_vel_cmd",
                json.dumps([0.0, 0.0, 0.0])
            )
        
        # Send current state info (for sim2real to know the mode)
        # Legacy state for backwards compatibility
        if self.state == "teleop":
            legacy_state = "teleop_loco" if self.locomotion_active else "teleop_full"
        else:
            legacy_state = self.state
        
        state_info = {
            "state": legacy_state,  # Legacy state for server
            "upper_body_paused": self.upper_body_paused,
            "locomotion_active": self.locomotion_active,
        }
        self.redis_pipeline.set("teleop_state_info", json.dumps(state_info))
        
        # Debug: print state being sent (occasionally)
        if not hasattr(self, '_last_state_debug') or (time.time() - self._last_state_debug) > 2.0:
            u = "F" if self.upper_body_paused else "T"
            l = "W" if self.locomotion_active else "S"
            print(f"\n[DEBUG] U:{u} L:{l} → {legacy_state}")
            self._last_state_debug = time.time()
        
        t_action = int(time.time() * 1000)
        self.redis_pipeline.set("t_action", t_action)
        self.redis_pipeline.execute()
    
    def _print_status(self, controller, smplx_data):
        """Print live status on a single replacing line (max 79 chars for terminal)"""
        # Build compact status string
        p = "P" if smplx_data is not None else "-"
        c = "C" if controller is not None else "-"
        
        # Get joystick and grip values
        lx, ly, rx = 0.0, 0.0, 0.0
        grip = 0
        if controller:
            left_ctrl = controller.get("LeftController", {})
            right_ctrl = controller.get("RightController", {})
            left_axis = left_ctrl.get("axis", [0, 0])
            right_axis = right_ctrl.get("axis", [0, 0])
            lx, ly = left_axis[0] if len(left_axis) > 0 else 0, left_axis[1] if len(left_axis) > 1 else 0
            rx = right_axis[0] if len(right_axis) > 0 else 0
            grip_val = right_ctrl.get("grip", 0)
            grip = 1 if (isinstance(grip_val, bool) and grip_val) or (isinstance(grip_val, (int, float)) and grip_val > 0.5) else 0
        
        # Interpolation progress
        if self.is_interpolating:
            elapsed = time.time() - self.interp_start_time
            progress = min(100, int(100 * elapsed / self.INTERP_DURATION))
            interp_str = f"→{self.interp_to_state[:4]}:{progress:2d}%"
        else:
            interp_str = "------"
        
        loco = "L" if self._is_locomotion_active() else "-"
        fb = "F" if getattr(self, '_using_fallback', False) else "-"
        
        # Show state (max 10 chars)
        state_display = self.state[:10]
        
        # Compact status line
        status = (f"[{state_display:10s}] {p}{c}{fb} G:{grip} {loco} "
                  f"J:({lx:+.1f},{ly:+.1f},{rx:+.1f}) "
                  f"v:({self.vel_cmd[0]:+.1f},{self.vel_cmd[1]:+.1f},{self.vel_cmd[2]:+.1f}) {interp_str}")
        
        # Print with carriage return - keep under 80 chars
        sys.stdout.write(f"\r{status:<79}")
        sys.stdout.flush()
    
    def _is_teleop_state(self):
        """Check if current state is a teleop state (should send data to robot)"""
        # New state machine uses "teleop" with separate upper/lower pause
        # Legacy states kept for compatibility
        teleop_states = ["teleop", "teleop_full", "teleop_loco", "paused"]
        if self.state in teleop_states:
            return True
        if self.is_interpolating and self.interp_to_state in teleop_states:
            return True
        # Also send data in preview if we have a frozen pose (robot should hold position)
        if self.state == "preview" and self._frozen_preview_qpos is not None:
            return True
        return False
    
    def run(self):
        """Main loop"""
        # Suppress loop_rate_limiters warnings
        import logging
        logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)
        
        rate = RateLimiter(frequency=self.args.target_fps)
        
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
            
            # Track connection status
            no_data_warnings = 0
            last_warning_time = 0
            
            while viewer.is_running() and self.state != "exit":
                # Get Pico data
                smplx_data, left_hand, right_hand, controller, headset = self.get_teleop_data()
                
                # Warn if no data received and teleop_streamer exists
                if self.teleop_streamer is not None:
                    if smplx_data is None and controller is None:
                        no_data_warnings += 1
                        current_time = time.time()
                        # Warn every 5 seconds
                        if current_time - last_warning_time > 5.0:
                            print(f"\n[yellow]Warning: No Pico data received ({no_data_warnings} attempts)[/yellow]")
                            print("[yellow]Make sure:")
                            print("  1. XRobotToolkit app is running on Pico")
                            print("  2. Pico is connected to this PC")
                            print("  3. Check app connection status[/yellow]")
                            last_warning_time = current_time
                    else:
                        no_data_warnings = 0  # Reset counter when data arrives
                
                # Process retargeting first to get current qpos for state machine
                qpos = None
                if smplx_data is not None:
                    qpos = self.process_retargeting(smplx_data)
                
                # Update state machine with current qpos for interpolation
                self.update_state(controller, qpos)
                
                # Control Inspire hands (trigger=close, grip=open)
                if controller:
                    self._control_inspire_hands(controller)
                
                # Print live status (replacing line)
                self._print_status(controller, smplx_data)
                
                # Auto-transition from idle to preview when data arrives
                if self.state == "idle" and smplx_data is not None:
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
                    elif self.state == "preview" and self._frozen_preview_qpos is not None:
                        # Preview with frozen pose: robot holds position, MuJoCo shows frozen pose
                        qpos = self._frozen_preview_qpos.copy()
                    elif self.state == "teleop":
                        # Upper body: tracking or frozen
                        if self.upper_body_paused and self._frozen_upper_qpos is not None:
                            qpos[7+15:7+29] = self._frozen_upper_qpos  # Arms frozen
                        # else: qpos already has GMR tracking for arms
                        
                        # Lower body: standing (TWIST2) or walking (GearWBC)
                        if self.locomotion_active:
                            # Walking mode: legs at loco default (server uses GearWBC)
                            qpos[7:7+12] = self.loco_policy.default_angles_reorder[:12]
                            qpos[7+12:7+15] = self.loco_policy.default_angles_reorder[12:15]
                        # else: standing mode - qpos already has GMR tracking (uses TWIST2)
                    
                    # LEGACY STATES (for backwards compatibility)
                    elif self.state == "paused":
                        qpos[7:7+12] = self.DEFAULT_STANDING_LEGS
                    elif self.state == "teleop_full":
                        pass  # Full body GMR tracking
                    elif self.state == "teleop_loco":
                        qpos[7:7+12] = self.loco_policy.default_angles_reorder[:12]
                        qpos[7+12:7+15] = self.loco_policy.default_angles_reorder[12:15]
                    
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
                    
                    # Apply smooth filtering to reduce jitter
                    mimic_obs = self.apply_smooth(mimic_obs)
                    
                    # Get neck data
                    neck_data = None
                    try:
                        neck_yaw, neck_pitch = human_head_to_robot_neck(smplx_data)
                        neck_data = [neck_yaw * 0.5, neck_pitch * 0.5]
                    except:
                        pass
                    
                    # Send to Redis if in teleop states
                    if self._is_teleop_state():
                        self.send_to_redis(mimic_obs, neck_data)
                else:
                    # Even if no smplx_data, send state info if in teleop mode
                    if self._is_teleop_state():
                        self.send_to_redis(None, None)
                
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
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Enable smooth filtering for mimic observations to reduce jitter.",
    )
    parser.add_argument(
        "--smooth_window_size",
        type=int,
        default=5,
        help="Window size for sliding window smoothing (default: 5 frames).",
    )
    # Inspire hand arguments (enabled by default)
    parser.add_argument("--use_inspire_hands", action="store_true", default=True, help="Enable Inspire hand control (default: enabled)")
    parser.add_argument("--no_inspire_hands", action="store_true", help="Disable Inspire hand control")
    parser.add_argument("--inspire_left_ip", type=str, default="192.168.123.210", help="Left Inspire hand IP")
    parser.add_argument("--inspire_right_ip", type=str, default="192.168.123.211", help="Right Inspire hand IP")
    # Velocity scaling arguments (to match GROOT's intended speed range)
    parser.add_argument("--vel_scale_forward", type=float, default=0.3, 
                        help="Max forward velocity scale (default: 0.3, gives ~0.6 m/s after cmd_scale)")
    parser.add_argument("--vel_scale_backward", type=float, default=0.2,
                        help="Max backward velocity scale (default: 0.2, more conservative for stability)")
    parser.add_argument("--vel_scale_strafe", type=float, default=0.25,
                        help="Max strafe velocity scale (default: 0.25)")
    parser.add_argument("--vel_scale_yaw", type=float, default=0.4,
                        help="Max yaw rotation scale (default: 0.4, gives ~0.2 rad/s after cmd_scale)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hybrid = HybridLocoTeleop(args)
    hybrid.run()
