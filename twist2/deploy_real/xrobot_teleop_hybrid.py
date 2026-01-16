"""
TWIST2 Hybrid Teleoperation - PICO + Manus Glove Modes

Combines PICO controller trigger-based control with Manus glove fine control.
User can switch between modes using controller buttons.

Boot Sequence:
  1. Press Unitree START → Ready state
  2. Press Unitree A → Preop mode

Teleop Modes:
  - PICO Right A: Enter PICO finger mode (trigger-based)
  - Unitree A: Enter Manus glove mode (fine control)
  - PICO Right A: Return to preop (from either mode)
"""

import argparse
import json
import os
import sys
import time
import threading

import mujoco as mj
import numpy as np
import redis
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import XRobotStreamer
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT
from rich import print

from data_utils.params import DEFAULT_MIMIC_OBS
from data_utils.rot_utils import euler_from_quaternion_np, quat_diff_np, quat_rotate_inverse_np
from data_utils.fps_monitor import FPSMonitor
from robot_control.inspire_hand_wrapper import DualHandController
from robot_control.common.remote_controller import RemoteController

# Import hybrid state machine
from hybrid_state_machine import HybridStateMachine

# Import Xsens/Manus components
from xsens_manus_integration import XsensStreamer, get_finger_data_by_name, TeleVisionStyleMapper


def extract_mimic_obs_whole_body(qpos, last_qpos, dt=1/30):
    """Extract whole body mimic observations from robot joint positions (35 dims)"""
    root_pos, last_root_pos = qpos[0:3], last_qpos[0:3]
    root_quat, last_root_quat = qpos[3:7], last_qpos[3:7]
    robot_joints = qpos[7:].copy()
    base_vel = (root_pos - last_root_pos) / dt
    base_ang_vel = quat_diff_np(last_root_quat, root_quat, scalar_first=True) / dt
    roll, pitch, yaw = euler_from_quaternion_np(root_quat.reshape(1, -1), scalar_first=True)
    base_vel_local = quat_rotate_inverse_np(root_quat, base_vel, scalar_first=True)
    base_ang_vel_local = quat_rotate_inverse_np(root_quat, base_ang_vel, scalar_first=True)
    
    height = root_pos[2:3]
    mimic_obs = np.concatenate([
        base_vel_local[:2],
        root_pos[2:3],
        roll, pitch,
        base_ang_vel_local[2:3],
        robot_joints
    ])
    
    return mimic_obs


class ManusFingerThread:
    """
    Separate thread for Manus glove finger tracking
    Non-blocking finger data processing
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        
        # Shared data (protected by lock)
        self._lock = threading.Lock()
        self._latest_finger_data = None
        self._running = False
        
        # Mapper for Manus → Inspire
        self.mapper = TeleVisionStyleMapper()
        
        # Latest angles for status display
        self.latest_left_angles = None
        self.latest_right_angles = None
        
        # Thread
        self._thread = None
        
        # Rate limiting
        self.target_fps = 100
    
    def set_finger_data(self, raw_decoded):
        """Called from main thread to provide new finger data"""
        if raw_decoded and raw_decoded.get('finger_segments'):
            with self._lock:
                self._latest_finger_data = raw_decoded
    
    def get_latest_angles(self):
        """Get latest computed angles for display"""
        with self._lock:
            return self.latest_left_angles, self.latest_right_angles
    
    def start(self):
        """Start the finger tracking thread"""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[ManusThread] Started")
    
    def stop(self):
        """Stop the finger tracking thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[ManusThread] Stopped")
    
    def _run(self):
        """Main finger tracking loop (runs in separate thread)"""
        rate = RateLimiter(frequency=self.target_fps, warn=False)
        
        while self._running:
            # Get latest finger data
            with self._lock:
                raw_decoded = self._latest_finger_data
                self._latest_finger_data = None  # Consume
            
            if raw_decoded:
                try:
                    finger_data = get_finger_data_by_name(raw_decoded)
                    
                    # Map to Inspire angles
                    left_angles = self.mapper.map_to_inspire(finger_data, 'left')
                    right_angles = self.mapper.map_to_inspire(finger_data, 'right')
                    
                    # Store for retrieval
                    with self._lock:
                        self.latest_left_angles = left_angles.copy()
                        self.latest_right_angles = right_angles.copy()
                
                except Exception as e:
                    pass  # Silently ignore errors
            
            rate.sleep()


class HybridTeleopSystem:
    """
    Unified teleoperation system with robot boot sequence
    Supports PICO trigger and Manus glove finger control modes
    """
    
    def __init__(self, args):
        self.args = args
        self.robot_name = args.robot
        self.xml_file = ROBOT_XML_DICT[args.robot]
        self.robot_base = ROBOT_BASE_DICT[args.robot]
        
        # Initialize state tracking
        self.last_qpos = None
        self.last_time = time.time()
        self.target_fps = args.target_fps
        self.measured_dt = 1 / self.target_fps
        
        # Components
        self.pico_streamer = None
        self.xsens_streamer = None
        self.manus_thread = None
        self.redis_client = None
        self.retarget = None
        self.model = None
        self.data = None
        self.inspire_hands = None
        self.unitree_remote = RemoteController()
        
        # Hybrid state machine
        self.state_machine = HybridStateMachine()
        
        # Rate limiter
        self.rate = None
        
        # FPS monitoring
        self.fps_monitor = FPSMonitor(
            enable_detailed_stats=args.measure_fps,
            quick_print_interval=100,
            detailed_print_interval=1000,
            expected_fps=self.target_fps,
            name="Hybrid Teleop Loop",
            enable_quick_print=False
        )
    
    def setup(self):
        """Initialize all systems"""
        print("=" * 60)
        print("TWIST2 HYBRID TELEOPERATION SYSTEM")
        print("=" * 60)
        
        # Setup PICO streamer
        print("\n[SETUP] Initializing PICO streamer...")
        self.pico_streamer = XRobotStreamer()
        print("[✓] PICO streamer ready")
        
        # Setup Xsens streamer (always running in background)
        if self.args.enable_manus:
            print("\n[SETUP] Initializing Xsens/Manus streamer...")
            offsets_path = os.path.join(
                os.path.dirname(__file__),
                "xsens_manus_integration",
                "offsets.json"
            )
            self.xsens_streamer = XsensStreamer(
                ip="0.0.0.0",
                port=self.args.xsens_port,
                offsets_path=offsets_path
            )
            self.xsens_streamer.start()
            print(f"[✓] Xsens streamer listening on port {self.args.xsens_port}")
            
            # Start Manus finger thread
            self.manus_thread = ManusFingerThread(redis_client=None)
            self.manus_thread.start()
            print("[✓] Manus finger thread started")
        else:
            print("\n[SETUP] Manus glove mode DISABLED")
        
        # Setup Redis
        print("\n[SETUP] Connecting to Redis...")
        self.redis_client = redis.Redis(host=self.args.redis_ip, port=6379, db=0)
        self.redis_client.ping()
        print(f"[✓] Redis connected: {self.args.redis_ip}:6379")
        
        # Setup GMR retargeting
        print("\n[SETUP] Initializing GMR retargeting...")
        self.retarget = GMR(
            src_human="xrobot",
            tgt_robot=self.robot_name,
            actual_human_height=self.args.actual_human_height,
        )
        print("[✓] GMR retargeting ready")
        
        # Setup MuJoCo
        print("\n[SETUP] Loading MuJoCo model...")
        self.model = mj.MjModel.from_xml_path(str(self.xml_file))
        self.data = mj.MjData(self.model)
        print("[✓] MuJoCo model loaded")
        
        # Setup Inspire hands
        if self.args.use_inspire_hands:
            print("\n[SETUP] Initializing Inspire hands...")
            try:
                self.inspire_hands = DualHandController(
                    left_ip=self.args.inspire_left_ip,
                    right_ip=self.args.inspire_right_ip,
                    async_mode=True
                )
                print(f"[✓] Inspire hands connected")
                print(f"    Left: {self.args.inspire_left_ip}")
                print(f"    Right: {self.args.inspire_right_ip}")
            except Exception as e:
                print(f"[!] Inspire hands failed to initialize: {e}")
                self.inspire_hands = None
        
        # Setup rate limiter
        self.rate = RateLimiter(frequency=self.target_fps, warn=False)
        
        print("\n" + "=" * 60)
        print("BOOT SEQUENCE")
        print("=" * 60)
        print("\n1. Press Unitree START button → Ready state")
        print("2. Press Unitree A button → Preop mode")
        print("\nTeleop Modes:")
        print("  - PICO Right A: Enter PICO finger mode")
        print("  - Unitree A: Enter Manus glove mode")
        print("  - PICO Right A: Return to preop (from either mode)")
        print("\n" + "=" * 60)
        print("\n[BOOT] Waiting for Unitree START button...")
    
    def run(self):
        """Main control loop"""
        self.setup()
        
        try:
            while True:
                # Get all inputs
                pico_controller = self.pico_streamer.get_controller_data()
                pico_body = self.pico_streamer.get_processed_body_data()
                
                # Get Xsens/Manus data (if enabled)
                if self.args.enable_manus and self.xsens_streamer:
                    xsens_raw = self.xsens_streamer.get_raw_decoded()
                    if xsens_raw and self.manus_thread:
                        self.manus_thread.set_finger_data(xsens_raw)
                
                # Update unitree remote state from Redis
                self._update_unitree_remote()
                
                # Update state machine
                prev_state = self.state_machine.state
                current_state = self.state_machine.update(
                    pico_controller,
                    self.unitree_remote
                )
                
                # Handle state transitions
                if self.state_machine.has_state_changed():
                    self._on_state_changed(prev_state, current_state)
                
                # Handle current state
                if current_state == "boot":
                    self._handle_boot_state()
                    
                elif current_state == "ready":
                    self._handle_ready_state()
                    
                elif current_state == "preop":
                    self._handle_preop_state()
                    
                elif current_state == "teleop_pico":
                    self._handle_pico_mode(pico_body, pico_controller)
                    
                elif current_state == "teleop_manus":
                    self._handle_manus_mode(pico_body)
                
                # FPS monitoring
                self.fps_monitor.tick()
                
                # Rate limiting
                self.rate.sleep()
                
        except KeyboardInterrupt:
            print("\n\n[EXIT] Ctrl+C detected, shutting down...")
        finally:
            self.cleanup()
    
    def _on_state_changed(self, prev_state, new_state):
        """Handle state transition actions"""
        print(f"\n{'='*60}")
        print(f"STATE TRANSITION: {prev_state} → {new_state}")
        print(f"{'='*60}")
        
        if new_state == "ready":
            print("[READY] Robot interface initialized")
            print("\nPress Unitree A button to enter PREOP mode...")
            
        elif new_state == "preop":
            print("[PREOP] Moving to preview pose...")
            self._move_to_default_pose()
            print("\nSelect teleop mode:")
            print("  - PICO Right A: PICO finger control")
            print("  - Unitree A: Manus glove control")
            
        elif new_state == "teleop_pico":
            print("[ACTIVE] PICO finger mode")
            print("  - Triggers + X/Y: Control fingers")
            print("  - Grips + X/Y: Control thumbs")
            print("  - Press PICO Right A to return to preop")
            
        elif new_state == "teleop_manus":
            if self.args.enable_manus:
                print("[ACTIVE] Manus glove mode")
                print("  - Gloves control all fingers")
                print("  - Press PICO Right A to return to preop")
            else:
                print("[WARNING] Manus mode requested but not enabled!")
                print("  Returning to preop...")
                self.state_machine.state = "preop"
    
    def _handle_boot_state(self):
        """Boot state - waiting for Unitree START"""
        pass
    
    def _handle_ready_state(self):
        """Ready state - robot initialized, waiting for preop"""
        pass
    
    def _handle_preop_state(self):
        """Preop/preview mode - default pose"""
        default_obs = DEFAULT_MIMIC_OBS[self.robot_name]
        self.redis_client.set(
            f"action_body_{self.robot_name}_with_hands",
            json.dumps(default_obs.tolist())
        )
        
        # Open hands to neutral
        if self.inspire_hands:
            self.inspire_hands.open_both()
    
    def _handle_pico_mode(self, pico_body, pico_controller):
        """PICO trigger-based control"""
        # Retarget body from PICO
        if pico_body is not None:
            try:
                qpos = self.retarget.retarget(pico_body)
                self._publish_body_to_redis(qpos)
            except Exception as e:
                pass  # Silently ignore retargeting errors
        
        # Get PICO hand state
        hand_state = self.state_machine.get_pico_hand_state()
        left_pos = hand_state['left_position']
        right_pos = hand_state['right_position']
        left_thumb = hand_state['left_thumb']
        right_thumb = hand_state['right_thumb']
        
        # Convert to Inspire angles
        left_angles = self._normalized_to_inspire(left_pos, left_thumb)
        right_angles = self._normalized_to_inspire(right_pos, right_thumb)
        
        # Send to Inspire hands
        if self.inspire_hands:
            self.inspire_hands.left_hand.set_angles(left_angles)
            self.inspire_hands.right_hand.set_angles(right_angles)
        
        # Publish to Redis
        self._publish_hands_to_redis(left_angles, right_angles)
    
    def _handle_manus_mode(self, pico_body):
        """Manus glove fine control"""
        # Retarget body from PICO (still using PICO for body)
        if pico_body is not None:
            try:
                qpos = self.retarget.retarget(pico_body)
                self._publish_body_to_redis(qpos)
            except Exception as e:
                pass
        
        # Get Manus finger angles from thread
        if self.manus_thread:
            left_angles, right_angles = self.manus_thread.get_latest_angles()
            
            if left_angles is not None and right_angles is not None:
                # Send to Inspire hands
                if self.inspire_hands:
                    self.inspire_hands.left_hand.set_angles(left_angles)
                    self.inspire_hands.right_hand.set_angles(right_angles)
                
                # Publish to Redis
                self._publish_hands_to_redis(left_angles, right_angles)
    
    def _move_to_default_pose(self):
        """Move robot to default preview pose"""
        default_obs = DEFAULT_MIMIC_OBS[self.robot_name]
        self.redis_client.set(
            f"action_body_{self.robot_name}_with_hands",
            json.dumps(default_obs.tolist())
        )
        
        if self.inspire_hands:
            self.inspire_hands.open_both()
        
        time.sleep(1.0)
    
    def _normalized_to_inspire(self, finger_pos, thumb_rot):
        """
        Convert normalized positions to Inspire angles
        
        Args:
            finger_pos: 0.0 (open) to 1.0 (closed)
            thumb_rot: 0.0 (outward) to 1.0 (inward)
        
        Returns:
            6-DOF angles [Little, Ring, Middle, Index, Thumb_Bend, Thumb_Rotate]
        """
        # Inspire: 0=closed, 2000=open (inverted)
        finger_angle = 2000 - int(finger_pos * 2000)
        angles = np.full(6, finger_angle, dtype=np.int16)
        
        # Thumb rotation
        thumb_angle = 2000 - int(thumb_rot * 2000)
        angles[5] = thumb_angle
        
        return angles
    
    def _update_unitree_remote(self):
        """Update unitree remote controller state from Redis"""
        try:
            remote_data = self.redis_client.get("unitree_remote_state")
            if remote_data:
                data = json.loads(remote_data)
                self.unitree_remote.button = data.get('button', [0]*16)
                self.unitree_remote.lx = data.get('lx', 0.0)
                self.unitree_remote.ly = data.get('ly', 0.0)
                self.unitree_remote.rx = data.get('rx', 0.0)
                self.unitree_remote.ry = data.get('ry', 0.0)
        except Exception as e:
            pass
    
    def _publish_body_to_redis(self, qpos):
        """Publish body state to Redis"""
        # Compute mimic obs
        if self.last_qpos is None:
            self.last_qpos = qpos.copy()
        
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0:
            self.measured_dt = dt
        
        mimic_obs = extract_mimic_obs_whole_body(qpos, self.last_qpos, self.measured_dt)
        
        self.redis_client.set(
            f"action_body_{self.robot_name}_with_hands",
            json.dumps(mimic_obs.tolist())
        )
        
        # Update for next iteration
        self.last_qpos = qpos.copy()
        self.last_time = current_time
    
    def _publish_hands_to_redis(self, left_angles, right_angles):
        """Publish hand state to Redis"""
        # Normalize to 0-1
        left_norm = (left_angles / 2000.0).tolist()
        right_norm = (right_angles / 2000.0).tolist()
        
        # Pad to 7-DOF
        while len(left_norm) < 7:
            left_norm.append(0.0)
        while len(right_norm) < 7:
            right_norm.append(0.0)
        
        self.redis_client.set(
            f"action_hand_left_{self.robot_name}_with_hands",
            json.dumps(left_norm)
        )
        self.redis_client.set(
            f"action_hand_right_{self.robot_name}_with_hands",
            json.dumps(right_norm)
        )
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n[CLEANUP] Shutting down...")
        
        if self.manus_thread:
            self.manus_thread.stop()
        
        if self.xsens_streamer:
            self.xsens_streamer.stop()
        
        if self.inspire_hands:
            self.inspire_hands.open_both()
            time.sleep(0.3)
            self.inspire_hands.stop()
        
        print("[CLEANUP] Complete")


def main():
    parser = argparse.ArgumentParser(description="TWIST2 Hybrid Teleoperation")
    
    # Basic options
    parser.add_argument("--robot", choices=["unitree_g1", "unitree_h1_2"], default="unitree_g1")
    parser.add_argument("--actual_human_height", type=float, default=1.8, help="Human height in meters")
    parser.add_argument("--target_fps", type=int, default=100, help="Target FPS")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis host")
    parser.add_argument("--measure_fps", type=int, default=0, help="Enable FPS measurement")
    
    # Inspire hands
    parser.add_argument("--use_inspire_hands", action="store_true", help="Enable Inspire hands")
    parser.add_argument("--inspire_left_ip", type=str, default="192.168.123.210", help="Left Inspire hand IP")
    parser.add_argument("--inspire_right_ip", type=str, default="192.168.123.211", help="Right Inspire hand IP")
    
    # Manus/Xsens options
    parser.add_argument("--enable_manus", action="store_true", help="Enable Manus glove mode")
    parser.add_argument("--xsens_port", type=int, default=9763, help="Xsens UDP port")
    
    args = parser.parse_args()
    
    system = HybridTeleopSystem(args)
    system.run()


if __name__ == "__main__":
    main()
