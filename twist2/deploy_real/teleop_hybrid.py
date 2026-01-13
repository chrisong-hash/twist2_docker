#!/usr/bin/env python3
"""
Hybrid Teleop - Full Robot Control
==================================
Complete teleop system with hybrid locomotion, neck tracking, and Inspire hands.

States:
  idle        : Waiting for Pico VR data
  preview     : MuJoCo shows your motion (calibrate here)
  teleop_full : Full body teleop (TWIST2 controls all joints)
  teleop_loco : Locomotion mode (LocoMode legs + GMR upper body)
  paused      : Robot FREEZES (ignores motion until unpaused)

Controls:
  Right A (key_one)  : Cycle: preview → teleop → pause → teleop → pause...
  Left X (key_one)   : Toggle teleop_full ↔ teleop_loco (only in teleop mode)
  Right A+B together : EMERGENCY SHUTDOWN (stops teleop + robot server)
  Left joystick      : Walk (in teleop_loco mode)
  Right joystick     : Rotate (in teleop_loco mode)
  Left Trigger       : Close left Inspire hand
  Right Trigger      : Close right Inspire hand
  Left Grip          : Open left Inspire hand
  Right Grip         : Open right Inspire hand

All mode transitions have smooth 1-second interpolation.
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
      paused      : Robot FREEZES at current pose (ignores your motion)
    
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
        
        # State machine - valid states: idle, preview, teleop_full, teleop_loco, paused, exit
        self.state = "idle"
        self.previous_teleop_state = "teleop_full"  # Remember which teleop state to return to from pause
        self.paused_qpos = None  # Frozen pose for paused state
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        
        # Interpolation state
        self.is_interpolating = False
        self.interp_start_time = 0.0
        self.interp_start_qpos = None  # Starting joint positions
        self.interp_target_qpos = None  # Target joint positions
        self.interp_from_state = None   # State we're transitioning from
        self.interp_to_state = None     # State we're transitioning to
        
        # Button state tracking (for edge detection)
        self._right_key_was_pressed = False
        self._left_key_was_pressed = False
        self._emergency_was_pressed = False
        
        # Inspire hand state
        self.use_inspire_hands = getattr(args, 'use_inspire_hands', False)
        self.inspire_hand_controller = None
        self.hand_left_position = 0.0   # 0.0 = open, 1.0 = closed
        self.hand_right_position = 0.0
        self.hand_movement_step = 0.05  # 5% per frame when held
        
        # Height setting
        self.estimated_height = args.actual_human_height
        self.retarget = None  # Will be initialized in _setup_retargeting
        
        # Smooth filtering settings
        self.enable_smooth = args.smooth
        self.smooth_window_size = args.smooth_window_size
        self.smooth_history = []  # Store recent observations for sliding window
        
        # Initialize systems
        print("\n[cyan]Initializing Hybrid Teleop...[/cyan]")
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
        self._print_controls()
    
    def _setup_locomotion_policy(self):
        """Load RoboMimic locomotion policy"""
        print("\n[1/6] Loading locomotion policy...")
        self.loco_policy = LocoModePolicy()
    
    def _setup_teleop_streamer(self):
        """Initialize Pico VR connection via XRobotStreamer"""
        print("\n[2/6] Connecting to Pico VR...")
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
        print("\n[3/6] Setting up GMR retargeting...")
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
        print("\n[4/6] Setting up MuJoCo preview...")
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
        print("\n[5/6] Connecting to Redis...")
        self.redis_client = redis.Redis(host=self.args.redis_ip, port=6379, db=0)
        self.redis_pipeline = self.redis_client.pipeline()
        self.redis_client.ping()
        # Clear any previous shutdown signal
        self.redis_client.delete("robot_shutdown")
        print("[green]Redis connected[/green]")
    
    def _setup_inspire_hands(self):
        """Setup Inspire hands if enabled"""
        if not self.use_inspire_hands:
            print("\n[6/6] Inspire hands: DISABLED")
            return
        
        print("\n[6/6] Setting up Inspire hands...")
        try:
            left_ip = getattr(self.args, 'inspire_left_ip', '192.168.123.210')
            right_ip = getattr(self.args, 'inspire_right_ip', '192.168.123.211')
            
            print(f"    Connecting: Left={left_ip}, Right={right_ip}")
            
            self.inspire_hand_controller = DualHandController(
                left_ip=left_ip,
                right_ip=right_ip,
                timeout=3.0
            )
            
            # Open both hands initially
            time.sleep(0.5)
            self.inspire_hand_controller.open_both()
            print("[green]Inspire hands connected and initialized[/green]")
            
        except Exception as e:
            print(f"[red]Failed to initialize Inspire hands: {e}[/red]")
            self.inspire_hand_controller = None
            self.use_inspire_hands = False
    
    def _control_inspire_hands(self, controller_data):
        """Control Inspire hands based on trigger/grip state"""
        if not self.use_inspire_hands or self.inspire_hand_controller is None:
            return
        
        if controller_data is None:
            return
        
        right_ctrl = controller_data.get("RightController", {})
        left_ctrl = controller_data.get("LeftController", {})
        
        # Get trigger and grip values
        right_trig = right_ctrl.get("index_trig", 0)
        left_trig = left_ctrl.get("index_trig", 0)
        right_grip = right_ctrl.get("grip", 0)
        left_grip = left_ctrl.get("grip", 0)
        
        # Convert bool to float if needed
        if isinstance(right_trig, bool):
            right_trig = 1.0 if right_trig else 0.0
        if isinstance(left_trig, bool):
            left_trig = 1.0 if left_trig else 0.0
        if isinstance(right_grip, bool):
            right_grip = 1.0 if right_grip else 0.0
        if isinstance(left_grip, bool):
            left_grip = 1.0 if left_grip else 0.0
        
        # Right hand control: trigger=close, grip=open
        if right_trig > 0.5:
            self.hand_right_position = min(1.0, self.hand_right_position + self.hand_movement_step)
        elif right_grip > 0.5:
            self.hand_right_position = max(0.0, self.hand_right_position - self.hand_movement_step)
        
        # Left hand control: trigger=close, grip=open
        if left_trig > 0.5:
            self.hand_left_position = min(1.0, self.hand_left_position + self.hand_movement_step)
        elif left_grip > 0.5:
            self.hand_left_position = max(0.0, self.hand_left_position - self.hand_movement_step)
        
        try:
            # Inspire hands: 0.0=open, 1.0=closed (invert our convention)
            left_inspire_pos = 1.0 - self.hand_left_position
            right_inspire_pos = 1.0 - self.hand_right_position
            
            self.inspire_hand_controller.ctrl_dual_hand(left_inspire_pos, right_inspire_pos)
        except Exception as e:
            print(f"[red]Inspire hand error: {e}[/red]")
    
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
        print("  HYBRID TELEOP - Full Robot Control")
        print("="*60)
        print("\n[yellow]Controls:[/yellow]")
        print("  Right A            : Cycle preview → teleop → pause → teleop...")
        print("  Left X             : Toggle teleop_full ↔ teleop_loco")
        print("  [red]Right A+B       : EMERGENCY SHUTDOWN[/red]")
        print("  Left joystick      : Walk (only in teleop_loco)")
        print("  Right joystick     : Rotate (only in teleop_loco)")
        if self.use_inspire_hands:
            print("  [cyan]Triggers         : Close hands[/cyan]")
            print("  [cyan]Grips            : Open hands[/cyan]")
        print("\n[yellow]States:[/yellow]")
        print("  idle        : Waiting for Pico data")
        print("  preview     : MuJoCo preview (calibrate here)")
        print("  teleop_full : Full body teleop (default)")
        print("  teleop_loco : Locomotion mode (legs walk, upper tracks)")
        print("  paused      : Robot FREEZES (ignores motion until unpaused)")
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
            # Target: FREEZE at the current pose (captured in self.paused_qpos)
            # interp_target_qpos = interp_start_qpos (no change - just hold position)
            pass
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
        # EXCEPT for paused state - keep the frozen target
        if current_target_qpos is not None and self.interp_to_state != "paused":
            self.interp_target_qpos = current_target_qpos.copy()
            # Apply state-specific poses
            if self.interp_to_state == "teleop_full":
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
        
        Args:
            controller_data: Dict with RightController and LeftController data
            current_qpos: Current robot qpos for interpolation (optional)
        """
        if controller_data is None:
            return
        
        # Don't process button inputs during interpolation (except exit)
        # Controller data has nested structure: {'RightController': {...}, 'LeftController': {...}}
        right_ctrl = controller_data.get("RightController", {})
        left_ctrl = controller_data.get("LeftController", {})
        
        # Get current button states
        right_key_one = right_ctrl.get("key_one", False)  # Right A
        left_key_one = left_ctrl.get("key_one", False)    # Left X button
        right_key_two = right_ctrl.get("key_two", False)  # Right B
        left_key_two = left_ctrl.get("key_two", False)    # Left Y button
        
        # DEBUG: Show raw button states every 2 seconds
        if not hasattr(self, '_btn_debug_time'):
            self._btn_debug_time = 0
        if time.time() - self._btn_debug_time > 2.0:
            print(f"\n[DEBUG BTN] RA:{right_key_one} LX:{left_key_one} RB:{right_key_two} LY:{left_key_two}")
            self._btn_debug_time = time.time()
        
        # Detect button presses (rising edge)
        right_a_pressed = right_key_one and not self._right_key_was_pressed
        left_x_pressed = left_key_one and not self._left_key_was_pressed
        
        # Emergency stop: A+B on right controller (both pressed together)
        a_plus_b_pressed = right_key_one and right_key_two
        if a_plus_b_pressed and not self._emergency_was_pressed:
            self.state = "exit"
            print("\n[red]→ EMERGENCY SHUTDOWN requested (A+B)[/red]")
            print("[red]  Sending shutdown signal to robot server...[/red]")
            self._send_shutdown_signal()
        self._emergency_was_pressed = a_plus_b_pressed
        
        # Process other buttons only if not interpolating
        if not self.is_interpolating and self.state != "exit":
            
            # Right A - cycle: preview → teleop_full → pause → teleop_full → pause...
            if right_a_pressed:
                print(f"\n[yellow]Right A pressed in state: {self.state}[/yellow]")
                if self.state == "idle":
                    self.state = "preview"
                    print("[cyan]→ PREVIEW mode: MuJoCo shows your motion[/cyan]")
                elif self.state == "preview":
                    # Enter teleop_full by default (with interpolation)
                    print("[cyan]→ Starting TELEOP_FULL[/cyan]")
                    self._start_interpolation("preview", "teleop_full", current_qpos)
                elif self.state in ["teleop_full", "teleop_loco"]:
                    # Go to pause (remember current teleop mode and freeze current pose)
                    self.previous_teleop_state = self.state
                    self.paused_qpos = current_qpos.copy()  # Freeze the current pose
                    print(f"[cyan]→ PAUSING (robot freezes at current pose)[/cyan]")
                    self._start_interpolation(self.state, "paused", current_qpos)
                elif self.state == "paused":
                    # Return to previous teleop state (with interpolation)
                    print(f"[cyan]→ UNPAUSING to {self.previous_teleop_state}[/cyan]")
                    self._start_interpolation("paused", self.previous_teleop_state, current_qpos)
            
            # Left X - toggle teleop_full ↔ teleop_loco (only when in teleop mode)
            if left_x_pressed:
                print(f"\n[yellow]Left X pressed in state: {self.state}[/yellow]")
                if self.state == "teleop_full":
                    print("[cyan]→ Switching to TELEOP_LOCO (walking mode)[/cyan]")
                    self._start_interpolation("teleop_full", "teleop_loco", current_qpos)
                elif self.state == "teleop_loco":
                    print("[cyan]→ Switching to TELEOP_FULL[/cyan]")
                    self._start_interpolation("teleop_loco", "teleop_full", current_qpos)
        
        # Update button state tracking
        self._right_key_was_pressed = right_key_one
        self._left_key_was_pressed = left_key_one
        
        # Joystick for locomotion - only active in teleop_loco
        left_axis = left_ctrl.get("axis", [0, 0])
        right_axis = right_ctrl.get("axis", [0, 0])
        
        if self.state == "teleop_loco" or (self.is_interpolating and self.interp_to_state == "teleop_loco"):
            # Left joystick for movement
            self.vel_cmd[0] = left_axis[1] if len(left_axis) > 1 else 0.0   # forward/backward
            self.vel_cmd[1] = -left_axis[0] if len(left_axis) > 0 else 0.0  # strafe
            # Right joystick for rotation
            self.vel_cmd[2] = -right_axis[0] if len(right_axis) > 0 else 0.0  # yaw
        else:
            # Zero velocity when not in locomotion mode
            self.vel_cmd[:] = 0.0
    
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
        if self.state == "teleop_loco":
            return True
        # Also active during interpolation TO teleop_loco
        if self.is_interpolating and self.interp_to_state == "teleop_loco":
            # Gradually enable - use interpolation progress
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
            # Debug: show neck data occasionally
            if not hasattr(self, '_neck_debug_count'):
                self._neck_debug_count = 0
            self._neck_debug_count += 1
            if self._neck_debug_count % 100 == 1:  # Every ~2 seconds at 50fps
                print(f"\n[cyan]Neck: yaw={neck_data[0]:.3f}, pitch={neck_data[1]:.3f}[/cyan]")
        
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
        # Determine effective state (use target state during interpolation)
        if self.is_interpolating and self.interp_to_state is not None:
            effective_state = self.interp_to_state
        else:
            effective_state = self.state
        
        state_info = {
            "state": effective_state,  # Send effective state (target during interpolation)
            "actual_state": self.state,  # For debugging
            "is_interpolating": self.is_interpolating,
            "interp_to_state": self.interp_to_state if self.is_interpolating else None,
        }
        self.redis_pipeline.set("teleop_state_info", json.dumps(state_info))
        
        # Debug: print state being sent (occasionally)
        if not hasattr(self, '_last_state_debug') or (time.time() - self._last_state_debug) > 2.0:
            print(f"\n[DEBUG] Sending state: {effective_state} (actual: {self.state}, interp: {self.is_interpolating})")
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
        """Check if current state or interpolation target is a teleop state"""
        teleop_states = ["teleop_full", "teleop_loco", "paused"]
        if self.state in teleop_states:
            return True
        if self.is_interpolating and self.interp_to_state in teleop_states:
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
                
                # Control Inspire hands (only in teleop states, not preview)
                if self._is_teleop_state():
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
                    elif self.state == "paused":
                        # In paused, robot FREEZES at the pose captured when entering pause
                        if self.paused_qpos is not None:
                            qpos = self.paused_qpos.copy()
                        else:
                            # Fallback: standing pose
                            qpos[7:7+12] = self.DEFAULT_STANDING_LEGS
                    elif self.state == "teleop_full":
                        # In teleop_full, use GMR tracking for ENTIRE body (legs + upper)
                        # qpos is already set from GMR, no override needed
                        pass
                    elif self.state == "teleop_loco":
                        # In locomotion mode: legs + waist at LocoMode default, ARMS TRACK from GMR
                        qpos[7:7+12] = self.loco_policy.default_angles_reorder[:12]  # Legs
                        qpos[7+12:7+15] = self.loco_policy.default_angles_reorder[12:15]  # Waist
                        # Arms (7+15:7+29) keep GMR tracking from qpos
                    
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
                    
                    # Get neck data from head tracking
                    neck_data = None
                    try:
                        neck_yaw, neck_pitch = human_head_to_robot_neck(smplx_data)
                        neck_data = [float(neck_yaw * 0.5), float(neck_pitch * 0.5)]
                    except Exception as e:
                        # Only log error once
                        if not hasattr(self, '_neck_error_logged'):
                            print(f"[yellow]Neck tracking error: {e}[/yellow]")
                            self._neck_error_logged = True
                    
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
    parser = argparse.ArgumentParser(description="Hybrid Teleop - Full Robot Control")
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
    # Inspire hand arguments
    parser.add_argument(
        "--use_inspire_hands",
        action="store_true",
        help="Enable Inspire hand control via triggers/grips.",
    )
    parser.add_argument(
        "--inspire_left_ip",
        type=str,
        default="192.168.123.210",
        help="IP address for left Inspire hand.",
    )
    parser.add_argument(
        "--inspire_right_ip",
        type=str,
        default="192.168.123.211",
        help="IP address for right Inspire hand.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hybrid = HybridLocoTeleop(args)
    hybrid.run()
