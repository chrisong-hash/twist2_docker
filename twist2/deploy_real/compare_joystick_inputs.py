#!/usr/bin/env python3
"""
Compare Joystick Inputs: Pico VR vs Unitree Remote Controller
=============================================================
This script compares the raw joystick values from:
1. Pico VR controllers (via XRobotStreamer)
2. Unitree remote controller (via LowState)

Both are processed identically (no deadzone) to match RoboMimic's behavior.

It shows:
- Raw joystick values (format: [forward, strafe, yaw])
- Scaled velocity commands (using RoboMimic's scale_values)
- Final command after cmd_scale multiplication
- Differences between the two sources

IMPORTANT: This script must be run in a conda environment with Python 3.8:
    - Locally: conda activate robomimic  (Python 3.8)
    - Docker:  conda activate gmr        (Python 3.10, but may need 3.8 for unitree_sdk2py)
    
    export LD_LIBRARY_PATH=/home/robo/CodeSpace/twist2_docker/unitree_sdk2/thirdparty/lib/x86_64:$LD_LIBRARY_PATH
    unset CYCLONEDDS_HOME
    python compare_joystick_inputs.py --net enp4s0
"""

import sys
import os
import numpy as np
import time
import struct
import yaml
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.absolute()))

# Try to import Pico VR
try:
    from general_motion_retargeting import XRobotStreamer
    HAS_PICO = True
except ImportError:
    HAS_PICO = False
    print("[WARNING] XRobotStreamer not available - Pico input will be unavailable")

# Try to import Unitree SDK - use unitree_interface (same as working code)
HAS_UNITREE = False
HAS_UNITREE_INTERFACE = False

# Add paths for unitree_interface (same as example scripts)
# Check both Docker paths (/workspace) and local paths
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(script_dir, '../..'))

possible_paths = [
    '/workspace/unitree_sdk2/python_binding/build/lib',  # Docker path
    '/workspace/unitree_sdk2/python_binding',  # Docker path
    os.path.join(workspace_root, 'unitree_sdk2/python_binding/build/lib'),  # Local path
    os.path.join(workspace_root, 'unitree_sdk2/python_binding'),  # Local path
    os.path.join(script_dir, '../../unitree_sdk2/python_binding/build/lib'),  # Relative
    os.path.join(script_dir, '../../unitree_sdk2/python_binding'),  # Relative
]

for path in possible_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        sys.path.insert(0, abs_path)
        # Also check for .so files in that directory
        if os.path.isdir(abs_path):
            so_files = [f for f in os.listdir(abs_path) if f.startswith('unitree_interface') and f.endswith('.so')]
            if so_files:
                print(f"[Unitree] Found unitree_interface at: {abs_path}")
                break

try:
    # First try unitree_interface (same as g1_wrapper.py uses)
    import unitree_interface
    HAS_UNITREE_INTERFACE = True
    HAS_UNITREE = True
    print("[Unitree] Using unitree_interface (same as working teleop code)")
except ImportError as e:
    print(f"[Unitree] unitree_interface not found, trying raw DDS SDK...")
    # Fallback to raw DDS SDK
    try:
        # Try RoboMimic's unitree_sdk2_python first (same as their deploy_real.py)
        # Check both Docker and local paths
        possible_robomimic_paths = [
            "/workspace/RoboMimic_Deploy",  # Docker
            "/home/robo/CodeSpace/RoboMimic_Deploy",  # Local
        ]
        
        robomimic_sdk_path = None
        robomimic_base = None
        for base_path in possible_robomimic_paths:
            sdk_path = os.path.join(base_path, "unitree_sdk2_python")
            if os.path.exists(sdk_path):
                robomimic_sdk_path = sdk_path
                robomimic_base = base_path
                break
        
        if robomimic_sdk_path and os.path.exists(robomimic_sdk_path):
            # Add the parent directory (RoboMimic_Deploy) to path so imports work
            sys.path.insert(0, robomimic_base)
            sys.path.insert(0, robomimic_sdk_path)
            
            # Try to install it if not already available
            try:
                import unitree_sdk2py
            except ImportError:
                # Try installing from RoboMimic's directory
                import subprocess
                print(f"[INFO] Attempting to install unitree_sdk2py from {robomimic_sdk_path}...")
                result = subprocess.run(
                    ["pip", "install", "-q", robomimic_sdk_path],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"[WARNING] Failed to install: {result.stderr}")
        
        # Also try standard locations
        sys.path.insert(0, '/workspace/unitree_sdk2/python_binding/build/lib')
        sys.path.insert(0, '/workspace/unitree_sdk2/python_binding')
        
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
        HAS_UNITREE = True
        print("[Unitree] Using raw DDS SDK (fallback)")
    except ImportError as e:
        HAS_UNITREE = False
        print(f"[WARNING] Unitree SDK not available - Remote controller input will be unavailable")
        print(f"[WARNING] Import error: {e}")
        print(f"[WARNING] Tried: unitree_interface, RoboMimic_Deploy/unitree_sdk2_python, unitree_sdk2/python_binding")

# RoboMimic's scale_values function
def scale_values(values, target_ranges):
    """RoboMimic's scale_values: maps [-1, 1] to [min, max]"""
    scaled = []
    for val, (new_min, new_max) in zip(values, target_ranges):
        scaled_val = (val + 1) * (new_max - new_min) / 2 + new_min
        scaled.append(scaled_val)
    return np.array(scaled)

# Load LocoMode config to get velocity ranges
def load_locomode_config():
    """Load LocoMode config to get velocity ranges"""
    # Check both Docker and local paths
    possible_config_paths = [
        "/workspace/RoboMimic_Deploy/policy/loco_mode/config/LocoMode.yaml",  # Docker
        "/home/robo/CodeSpace/RoboMimic_Deploy/policy/loco_mode/config/LocoMode.yaml",  # Local
    ]
    
    config_path = None
    for path in possible_config_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if config_path is None:
        print(f"[ERROR] LocoMode config not found. Tried:")
        for path in possible_config_paths:
            print(f"  - {path}")
        return None, None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        cmd_range = config["cmd_range"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        print(f"[INFO] Loaded LocoMode config from: {config_path}")
        return cmd_range, cmd_scale
    except Exception as e:
        print(f"[ERROR] Failed to load LocoMode config from {config_path}: {e}")
        return None, None

class UnitreeRemoteController:
    """Wrapper to read Unitree remote controller values"""
    def __init__(self, net="enp4s0"):
        self.net = net
        self.low_state = None
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.message_count = 0
        self.last_update_time = 0
        self.first_message_received = False
        self.robot = None
        self.use_interface = False
        
        if not HAS_UNITREE:
            print("[ERROR] Unitree SDK not available")
            return
        
        # Try unitree_interface first (same as working code)
        if HAS_UNITREE_INTERFACE:
            try:
                print(f"[Unitree] Using unitree_interface (same as g1_wrapper.py)")
                self.robot = unitree_interface.create_robot(net, unitree_interface.RobotType.G1, unitree_interface.MessageType.HG)
                self.use_interface = True
                print("[Unitree] ✓ Robot interface created")
                # Test connection by reading state
                try:
                    test_state = self.robot.read_low_state()
                    print(f"[Unitree] ✓ Successfully read robot state (tick={test_state.tick if hasattr(test_state, 'tick') else 'N/A'})")
                    self.first_message_received = True
                    self.last_update_time = time.time()
                except Exception as e:
                    print(f"[Unitree] ⚠ Could not read initial state: {e}")
                return
            except Exception as e:
                print(f"[Unitree] Failed to use unitree_interface: {e}")
                print("[Unitree] Falling back to raw DDS SDK...")
                self.use_interface = False
        
        # Fallback to raw DDS SDK
        if not self.use_interface:
            try:
                print(f"[Unitree] Using raw DDS SDK on interface: {net}")
                ChannelFactoryInitialize(0, net)
                print("[Unitree] ChannelFactory initialized")
                
                self.lowstate_sub = ChannelSubscriber("rt/lowstate", LowStateHG)
                print("[Unitree] Subscriber created")
                
                print(f"[Unitree] Initializing callback with queue length 10...")
                self.lowstate_sub.Init(self._low_state_handler, 10)
                print(f"[Unitree] Subscribed to rt/lowstate on {net}")
                print("[Unitree] Waiting for messages...")
                print("[Unitree] Make sure robot is in debugging mode (hold L2+R2 on remote)")
                
                # Give DDS time to establish connection and receive first message
                # Check multiple times to see if messages start coming
                for i in range(5):
                    time.sleep(1.0)
                    if self.message_count > 0:
                        print(f"[Unitree] ✓ Received {self.message_count} message(s)!")
                        break
                    else:
                        print(f"[Unitree] Still waiting... ({i+1}/5)")
                
                if self.message_count == 0:
                    print("[Unitree] ⚠ No messages received after 5 seconds")
                    print("[Unitree] Troubleshooting:")
                    print("  1. Check robot is powered on")
                    print("  2. Hold L2+R2 on remote to enter debugging mode")
                    print("  3. Check network: ping 192.168.123.164")
                    print("  4. Verify network interface: ifconfig | grep enp4s0")
                    print("  5. Try restarting the script")
                else:
                    print(f"[Unitree] Connection established! Message count: {self.message_count}")
            except Exception as e:
                print(f"[ERROR] Failed to initialize Unitree SDK: {e}")
                import traceback
                traceback.print_exc()
                self.lowstate_sub = None
    
    def _low_state_handler(self, msg):
        """Callback for LowState messages"""
        self.low_state = msg
        self.message_count += 1
        self.last_update_time = time.time()
        self.first_message_received = True
        
        # Print first message to confirm callback is working
        if self.message_count == 1:
            print(f"[Unitree] ✓ FIRST MESSAGE RECEIVED! (tick={msg.tick if hasattr(msg, 'tick') else 'N/A'})")
        
        # Extract joystick values from wireless_remote (same as RoboMimic)
        try:
            data = msg.wireless_remote
            if len(data) >= 24:
                self.lx = struct.unpack("f", bytes(data[4:8]))[0]
                self.rx = struct.unpack("f", bytes(data[8:12]))[0]
                self.ry = struct.unpack("f", bytes(data[12:16]))[0]
                self.ly = struct.unpack("f", bytes(data[20:24]))[0]
                
                # Debug: Print first few messages to see if data is non-zero
                if self.message_count <= 5:
                    print(f"[Unitree] Message {self.message_count}: lx={self.lx:.4f}, ly={self.ly:.4f}, rx={self.rx:.4f}, ry={self.ry:.4f}")
                    print(f"[Unitree] Raw bytes (first 24): {data[:24].hex()}")
            else:
                print(f"[WARNING] wireless_remote data too short: {len(data)} bytes (got {len(data)})")
        except Exception as e:
            print(f"[ERROR] Failed to parse wireless_remote: {e}")
            import traceback
            traceback.print_exc()
    
    def get_vel_cmd(self):
        """Get velocity command in RoboMimic format: [ly, -lx, -rx]"""
        if self.use_interface:
            # Use unitree_interface (same as working code)
            try:
                controller = self.robot.read_wireless_controller()
                if controller is not None:
                    self.ly = controller.ly
                    self.lx = controller.lx
                    self.rx = controller.rx
                    self.ry = controller.ry
                    self.last_update_time = time.time()
                    self.message_count += 1
                    return np.array([self.ly, -self.lx, -self.rx], dtype=np.float32)
            except Exception as e:
                # Silently fail - controller might not be available yet
                return None
        
        # Use raw DDS data
        if self.low_state is None:
            return None
        return np.array([self.ly, -self.lx, -self.rx], dtype=np.float32)
    
    def is_connected(self):
        """Check if connected to robot"""
        if self.use_interface:
            # For unitree_interface, try to read controller to verify connection
            try:
                controller = self.robot.read_wireless_controller()
                if controller is not None:
                    return True
            except:
                pass
            return False
        
        # For raw DDS
        # Must have received at least one message
        if not self.first_message_received or self.low_state is None:
            return False
        # Check if we're receiving messages (within last 2 seconds)
        # This is the most important check - if messages are coming, we're connected
        if self.last_update_time == 0:
            return False  # No messages received yet
        if time.time() - self.last_update_time > 2.0:
            return False
        # If we're receiving messages, we're connected (regardless of tick value)
        return True
    
    def get_debug_info(self):
        """Get detailed debug information"""
        info = {
            'has_sdk': HAS_UNITREE,
            'low_state_received': self.low_state is not None,
            'message_count': self.message_count,
            'last_update_age': time.time() - self.last_update_time if self.last_update_time > 0 else None,
        }
        if self.low_state is not None:
            info['tick'] = self.low_state.tick
            info['wireless_remote_length'] = len(self.low_state.wireless_remote) if hasattr(self.low_state, 'wireless_remote') else 0
        return info
    
    def get_status(self):
        """Get connection status info"""
        if self.low_state is None:
            return "No messages received"
        age = time.time() - self.last_update_time
        if age > 2.0:
            return f"Stale data (last update {age:.1f}s ago, messages: {self.message_count})"
        tick_info = f"tick: {self.low_state.tick}" if self.low_state.tick > 0 else "tick: 0 (may be normal)"
        return f"Connected ({tick_info}, messages: {self.message_count}, age: {age:.2f}s)"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare joystick inputs from Pico VR vs Unitree remote")
    parser.add_argument("--net", type=str, default="enp4s0", 
                       help="Network interface connected to robot (default: enp4s0)")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor joystick values continuously (don't run test sequence)")
    args = parser.parse_args()
    
    print("="*80)
    print("Joystick Input Comparison: Pico VR vs Unitree Remote Controller")
    print("="*80)
    print()
    print("This script will show:")
    print("  1. Raw joystick values from each source")
    print("  2. Scaled velocity commands (using RoboMimic's scale_values)")
    print("  3. Final command after cmd_scale multiplication")
    print()
    print("NOTE: Both controllers are processed identically (NO deadzone)")
    print("      Format: [forward, strafe, yaw] = [ly, -lx, -rx]")
    print()
    print("Press Ctrl+C to exit")
    print("="*80)
    print()
    
    # Load LocoMode config
    cmd_range, cmd_scale = load_locomode_config()
    if cmd_range is None:
        print("[ERROR] Cannot proceed without LocoMode config")
        return
    
    range_velx = [cmd_range["lin_vel_x"][0], cmd_range["lin_vel_x"][1]]
    range_vely = [cmd_range["lin_vel_y"][0], cmd_range["lin_vel_y"][1]]
    range_velz = [cmd_range["ang_vel_z"][0], cmd_range["ang_vel_z"][1]]
    
    print(f"Velocity ranges from LocoMode config:")
    print(f"  vx: {range_velx}")
    print(f"  vy: {range_vely}")
    print(f"  vyaw: {range_velz}")
    print(f"  cmd_scale: {cmd_scale}")
    print()
    
    # Initialize Pico VR
    pico_streamer = None
    if HAS_PICO:
        try:
            pico_streamer = XRobotStreamer()
            print("[Pico] XRobotStreamer initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Pico: {e}")
    
    # Initialize Unitree remote
    unitree_remote = None
    if HAS_UNITREE:
        try:
            unitree_remote = UnitreeRemoteController(net=args.net)
            print(f"[Unitree] Remote controller initialized on {args.net}")
            print("[Unitree] Waiting for robot connection...")
            print("[Unitree] Make sure:")
            print("  1. Robot is powered on")
            print("  2. Robot is in debugging mode (L2+R2 on remote)")
            print("  3. Ethernet cable is connected")
            print("  4. Network interface is correct")
            print()
            
            # Wait for connection with status updates
            # If we already got messages during init, skip the wait
            if unitree_remote.message_count == 0:
                print("[Unitree] No messages received during initialization, waiting longer...")
                timeout = 10
                start_time = time.time()
                last_status_time = 0
                while not unitree_remote.is_connected() and (time.time() - start_time) < timeout:
                    time.sleep(0.5)
                    # Print status every 2 seconds
                    if time.time() - last_status_time > 2.0:
                        status = unitree_remote.get_status()
                        debug_info = unitree_remote.get_debug_info()
                        print(f"[Unitree] Status: {status}")
                        print(f"[Unitree] Debug: {debug_info}")
                        if unitree_remote.message_count > 0:
                            # Show current joystick values even if not "connected"
                            vel_cmd = unitree_remote.get_vel_cmd()
                            if vel_cmd is not None:
                                print(f"[Unitree] Current joystick: [{vel_cmd[0]:.4f}, {vel_cmd[1]:.4f}, {vel_cmd[2]:.4f}]")
                        last_status_time = time.time()
            
            if unitree_remote.is_connected():
                print("[Unitree] ✓ Connected to robot!")
                print(f"[Unitree] Received {unitree_remote.message_count} messages")
                # Show current joystick values
                vel_cmd = unitree_remote.get_vel_cmd()
                if vel_cmd is not None:
                    print(f"[Unitree] Current joystick: [{vel_cmd[0]:.4f}, {vel_cmd[1]:.4f}, {vel_cmd[2]:.4f}]")
                    print(f"[Unitree] Raw values: lx={unitree_remote.lx:.4f}, ly={unitree_remote.ly:.4f}, rx={unitree_remote.rx:.4f}, ry={unitree_remote.ry:.4f}")
                    
                    # Check if joystick data is all zeros (remote might not be paired/active)
                    if (abs(unitree_remote.lx) < 0.001 and abs(unitree_remote.ly) < 0.001 and 
                        abs(unitree_remote.rx) < 0.001 and abs(unitree_remote.ry) < 0.001):
                        print("[WARNING] All joystick values are zero!")
                        print("[WARNING] Possible causes:")
                        print("  1. Remote controller not paired with robot")
                        print("  2. Remote controller is off or out of battery")
                        print("  3. Robot not receiving remote signals")
                        print("  4. Try pressing buttons on remote to wake it up")
                        print("  5. Check remote LED - should be on when paired")
            else:
                print("[Unitree] ✗ Timeout waiting for robot connection")
                print(f"[Unitree] Final status: {unitree_remote.get_status()}")
                
                # Print debug info
                debug_info = unitree_remote.get_debug_info()
                print("\n[Unitree] Debug Information:")
                for key, value in debug_info.items():
                    print(f"  {key}: {value}")
                
                print()
                print("Troubleshooting:")
                print("  1. Make sure you're in 'twist2' conda environment (not 'gmr')")
                print("  2. Check network interface: ifconfig | grep -A 5 enp")
                print("  3. Try pinging robot: ping 192.168.123.164")
                print("  4. Check robot is in debugging mode (L2+R2 on remote)")
                print("  5. Try different network interface name (use --net <interface>)")
                print("  6. Make sure robot is powered on and remote is paired")
                print("  7. Check if unitree_sdk2py is installed:")
                print("     python -c 'import unitree_sdk2py; print(unitree_sdk2py.__file__)'")
                print("  8. If using RoboMimic SDK, check it's mounted:")
                print("     ls -la /workspace/RoboMimic_Deploy/unitree_sdk2_python")
                print()
                response = input("Continue anyway (Pico only)? (y/n): ")
                if response.lower() != 'y':
                    print("Exiting...")
                    return
        except Exception as e:
            print(f"[ERROR] Failed to initialize Unitree remote: {e}")
            import traceback
            traceback.print_exc()
    
    if pico_streamer is None and unitree_remote is None:
        print("[ERROR] No input sources available!")
        return
    
    # Monitor mode: just show joystick values continuously
    if args.monitor:
        print()
        print("="*80)
        print("MONITOR MODE: Showing joystick values continuously")
        print("="*80)
        print("Press Ctrl+C to exit")
        print()
        
        try:
            while True:
                # Get Pico input
                pico_vel_cmd = None
                pico_raw_str = "N/A"
                if pico_streamer is not None:
                    try:
                        smplx_data, left_hand, right_hand, controller_data, headset = pico_streamer.get_current_frame()
                        if controller_data is not None:
                            left_ctrl = controller_data.get("LeftController", {})
                            right_ctrl = controller_data.get("RightController", {})
                            left_axis = left_ctrl.get("axis", [0, 0])
                            right_axis = right_ctrl.get("axis", [0, 0])
                            
                            # Same format as Unitree remote: [ly, -lx, -rx]
                            pico_vel_cmd = np.array([
                                left_axis[1],      # forward/backward (ly)
                                -left_axis[0],     # strafe (-lx)
                                -right_axis[0]     # yaw (-rx)
                            ], dtype=np.float32)
                            pico_raw_str = f"ly={left_axis[1]:7.4f} lx={left_axis[0]:7.4f} rx={right_axis[0]:7.4f} ry={right_axis[1]:7.4f}"
                    except Exception as e:
                        pico_raw_str = f"Error: {str(e)[:30]}"
                
                # Get Unitree input
                unitree_vel_cmd = None
                unitree_raw_str = "N/A"
                if unitree_remote is not None and unitree_remote.is_connected():
                    unitree_vel_cmd = unitree_remote.get_vel_cmd()
                    if unitree_vel_cmd is not None:
                        unitree_raw_str = f"lx={unitree_remote.lx:7.4f} ly={unitree_remote.ly:7.4f} rx={unitree_remote.rx:7.4f} ry={unitree_remote.ry:7.4f}"
                
                # Display both side by side
                line = ""
                if pico_vel_cmd is not None:
                    line += f"[Pico]   [{pico_vel_cmd[0]:7.4f}, {pico_vel_cmd[1]:7.4f}, {pico_vel_cmd[2]:7.4f}] | Raw: {pico_raw_str} | "
                else:
                    line += f"[Pico]   Not available | "
                
                if unitree_vel_cmd is not None:
                    line += f"[Unitree] [{unitree_vel_cmd[0]:7.4f}, {unitree_vel_cmd[1]:7.4f}, {unitree_vel_cmd[2]:7.4f}] | Raw: {unitree_raw_str}"
                else:
                    line += f"[Unitree] {unitree_remote.get_status() if unitree_remote else 'Not initialized'}"
                
                # Calculate difference if both available
                if pico_vel_cmd is not None and unitree_vel_cmd is not None:
                    diff = pico_vel_cmd - unitree_vel_cmd
                    diff_mag = np.linalg.norm(diff)
                    line += f" | Diff: [{diff[0]:7.4f}, {diff[1]:7.4f}, {diff[2]:7.4f}] (mag: {diff_mag:.4f})"
                
                print(f"\r{line}", end="", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nExiting monitor mode...")
            return
    
    print()
    print("Starting automated comparison test...")
    print()
    
    # Test sequence
    test_sequence = [
        ("Push LEFT joystick FORWARD on both controllers", [0, 1.0, 0]),  # forward
        ("Push LEFT joystick BACKWARD on both controllers", [0, -1.0, 0]),  # backward
        ("Push LEFT joystick RIGHT on both controllers", [0, 0, 1.0]),  # right (strafe)
        ("Push LEFT joystick LEFT on both controllers", [0, 0, -1.0]),  # left (strafe)
        ("Push RIGHT joystick RIGHT on both controllers", [1.0, 0, 0]),  # yaw right
        ("Push RIGHT joystick LEFT on both controllers", [-1.0, 0, 0]),  # yaw left
        ("CENTER both joysticks (let go)", [0, 0, 0]),  # centered
    ]
    
    try:
        for test_name, expected_direction in test_sequence:
            print("\n" + "="*80)
            print(f"TEST: {test_name}")
            print("="*80)
            print("Get ready...")
            for i in range(3, 0, -1):
                print(f"  Starting in {i}...")
                time.sleep(1)
            
            print("\n>>> CAPTURING DATA (3 seconds) <<<")
            
            # Collect samples over 3 seconds
            samples = []
            start_time = time.time()
            while time.time() - start_time < 3.0:
                # Get Pico input
                pico_vel_cmd = None
                pico_raw = None
                if pico_streamer is not None:
                    try:
                        smplx_data, left_hand, right_hand, controller_data, headset = pico_streamer.get_current_frame()
                        if controller_data is not None:
                            left_ctrl = controller_data.get("LeftController", {})
                            right_ctrl = controller_data.get("RightController", {})
                            left_axis = left_ctrl.get("axis", [0, 0])
                            right_axis = right_ctrl.get("axis", [0, 0])
                            
                            # Same format as Unitree remote: [ly, -lx, -rx]
                            pico_raw = np.array([
                                left_axis[1],      # forward/backward (ly)
                                -left_axis[0],     # strafe (-lx)
                                -right_axis[0]     # yaw (-rx)
                            ], dtype=np.float32)
                            
                            # NO deadzone applied (matching Unitree remote exactly)
                            pico_vel_cmd = pico_raw.copy()
                    except Exception as e:
                        pass
                
                # Get Unitree remote input
                unitree_vel_cmd = None
                unitree_raw = None
                if unitree_remote is not None and unitree_remote.is_connected():
                    unitree_raw = unitree_remote.get_vel_cmd()
                    if unitree_raw is not None:
                        unitree_vel_cmd = unitree_raw.copy()
                        # No deadzone applied (matching RoboMimic exactly)
                
                # Store sample
                if pico_raw is not None or unitree_raw is not None:
                    samples.append({
                        'pico_raw': pico_raw.copy() if pico_raw is not None else None,
                        'pico_vel_cmd': pico_vel_cmd.copy() if pico_vel_cmd is not None else None,
                        'unitree_raw': unitree_raw.copy() if unitree_raw is not None else None,
                        'unitree_vel_cmd': unitree_vel_cmd.copy() if unitree_vel_cmd is not None else None,
                    })
                
                time.sleep(0.05)  # 20 Hz sampling
            
            # Calculate averages
            if samples:
                pico_raw_avg = np.mean([s['pico_raw'] for s in samples if s['pico_raw'] is not None], axis=0) if any(s['pico_raw'] is not None for s in samples) else None
                pico_vel_avg = np.mean([s['pico_vel_cmd'] for s in samples if s['pico_vel_cmd'] is not None], axis=0) if any(s['pico_vel_cmd'] is not None for s in samples) else None
                unitree_raw_avg = np.mean([s['unitree_raw'] for s in samples if s['unitree_raw'] is not None], axis=0) if any(s['unitree_raw'] is not None for s in samples) else None
                unitree_vel_avg = np.mean([s['unitree_vel_cmd'] for s in samples if s['unitree_vel_cmd'] is not None], axis=0) if any(s['unitree_vel_cmd'] is not None for s in samples) else None
            else:
                pico_raw_avg = pico_vel_avg = unitree_raw_avg = unitree_vel_avg = None
            
            # Print results
            print("\n" + "="*80)
            print("RESULTS:")
            print("="*80)
            print()
            
            # Pico VR
            print("PICO VR CONTROLLER (averaged over 3 seconds):")
            if pico_raw_avg is not None:
                print(f"  Raw values:        [{pico_raw_avg[0]:7.4f}, {pico_raw_avg[1]:7.4f}, {pico_raw_avg[2]:7.4f}]")
                print(f"  (No deadzone)      [{pico_vel_avg[0]:7.4f}, {pico_vel_avg[1]:7.4f}, {pico_vel_avg[2]:7.4f}]")
                
                # Scale using RoboMimic's formula
                pico_scaled = scale_values(pico_vel_avg, [range_velx, range_vely, range_velz])
                pico_final = pico_scaled * cmd_scale
                
                print(f"  After scale_values: [{pico_scaled[0]:7.4f}, {pico_scaled[1]:7.4f}, {pico_scaled[2]:7.4f}]")
                print(f"  After cmd_scale:    [{pico_final[0]:7.4f}, {pico_final[1]:7.4f}, {pico_final[2]:7.4f}]")
                print(f"  Magnitude:          {np.linalg.norm(pico_vel_avg):.4f}")
            else:
                print("  [No data]")
            print()
            
            # Unitree Remote
            print("UNITREE REMOTE CONTROLLER (averaged over 3 seconds):")
            if unitree_raw_avg is not None:
                print(f"  Raw values:        [{unitree_raw_avg[0]:7.4f}, {unitree_raw_avg[1]:7.4f}, {unitree_raw_avg[2]:7.4f}]")
                print(f"  (No deadzone)      [{unitree_vel_avg[0]:7.4f}, {unitree_vel_avg[1]:7.4f}, {unitree_vel_avg[2]:7.4f}]")
                
                # Scale using RoboMimic's formula
                unitree_scaled = scale_values(unitree_vel_avg, [range_velx, range_vely, range_velz])
                unitree_final = unitree_scaled * cmd_scale
                
                print(f"  After scale_values: [{unitree_scaled[0]:7.4f}, {unitree_scaled[1]:7.4f}, {unitree_scaled[2]:7.4f}]")
                print(f"  After cmd_scale:    [{unitree_final[0]:7.4f}, {unitree_final[1]:7.4f}, {unitree_final[2]:7.4f}]")
                print(f"  Magnitude:          {np.linalg.norm(unitree_vel_avg):.4f}")
            else:
                print("  [No data - robot not connected]")
            print()
            
            # Comparison
            if pico_vel_avg is not None and unitree_vel_avg is not None:
                print("COMPARISON:")
                diff = pico_vel_avg - unitree_vel_avg
                print(f"  Difference (raw):   [{diff[0]:7.4f}, {diff[1]:7.4f}, {diff[2]:7.4f}]")
                
                diff_scaled = pico_scaled - unitree_scaled
                print(f"  Difference (scaled): [{diff_scaled[0]:7.4f}, {diff_scaled[1]:7.4f}, {diff_scaled[2]:7.4f}]")
                print()
            
            print("="*80)
            if expected_direction[0] == 0 and expected_direction[1] == 0 and expected_direction[2] == 0:
                print("NOTE: When joystick is CENTERED, scale_values maps 0 → middle of range!")
                print("      This means zero input becomes non-zero velocity command!")
            print("="*80)
            
            # Wait for Enter to continue
            if test_name != test_sequence[-1][0]:  # Not the last test
                input("\nPress ENTER to continue to next test...")
            else:
                print("\nAll tests complete!")
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
        # Clean up XRobotStreamer gracefully
        if pico_streamer is not None:
            try:
                # XRobotStreamer cleanup if needed
                pass
            except:
                pass

if __name__ == "__main__":
    main()

