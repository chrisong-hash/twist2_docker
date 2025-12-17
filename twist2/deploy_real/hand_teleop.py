#!/usr/bin/env python3
"""
Hand-Only Teleop with Safety Features

Controls Inspire hands from Pico hand tracking WITHOUT leg/body control.
Includes multiple safety triggers:
- Clap detection: PAUSE
- Controller A button: Tap=PAUSE, Hold 500ms=EXIT
- Voice commands: "pause"/"stop"=PAUSE, "resume"/"yes"=RESUME (if available)

States:
- idle: Waiting for hands to be detected
- active: Controlling Inspire hands
- paused: Frozen at current position
- exit: Program terminates

The hands can be tested without the robot being activated.
"""

import argparse
import sys
import time
import os
import numpy as np
import threading
import select

# Try to import XRoboToolkit
try:
    from general_motion_retargeting import XRobotStreamer
except ImportError:
    print("ERROR: Could not import XRobotStreamer")
    print("Make sure XRoboToolkit is running")
    sys.exit(1)

# Try to import Inspire hand controller
try:
    from robot_control.inspire_hand_wrapper import DualHandController, InspireHandController
except ImportError:
    print("WARNING: Could not import InspireHandController - hands will not be controlled")
    DualHandController = None

# Try to import Redis for voice commands (from external service)
try:
    import redis
    import json as json_module
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("INFO: Redis not available for voice commands (pip install redis)")


class SafetyMonitor:
    """Monitors multiple safety triggers"""
    
    def __init__(self, streamer, enable_voice=False, voice_model_path=None):
        self.streamer = streamer
        
        # Clap detection with latch
        self.CLAP_DISTANCE_THRESHOLD = 0.10  # 10cm
        self.CLAP_RELEASE_THRESHOLD = 0.20  # 20cm - must separate this far before next clap
        self.OPEN_THRESHOLD = 300  # Below this = finger is OPEN
        self.clap_latched = False  # True after clap detected, until hands separate
        
        # Controller button detection
        self.a_button_pressed_time = None
        self.a_button_was_pressed = False  # Track previous state
        self.A_HOLD_DURATION = 0.5  # 500ms for exit
        
        # Voice detection via Redis (from external voice_service.py on host)
        self.enable_voice = enable_voice and REDIS_AVAILABLE
        self.voice_command = None
        self.voice_running = False
        self.voice_status = "disabled"
        self.last_voice_text = ""
        self.last_voice_time = 0
        self.redis_client = None
        self.last_voice_timestamp = 0
        
        if self.enable_voice:
            self._start_redis_voice()
    
    def _start_redis_voice(self):
        """Connect to Redis for voice commands from external service"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            self.voice_running = True
            self.voice_status = "redis:waiting"
            print("[VOICE] ✓ Connected to Redis - waiting for voice_service.py on host")
            print("[VOICE]   Run on HOST: python scripts/voice_service.py")
        except Exception as e:
            print(f"[VOICE] ✗ Redis connection failed: {e}")
            self.voice_status = f"redis error: {e}"
            self.enable_voice = False
    
    def _check_redis_voice(self):
        """Check Redis for voice commands from external service"""
        if not self.redis_client:
            return
        
        try:
            # Check for voice command in Redis
            cmd_json = self.redis_client.get('voice_command')
            if cmd_json:
                cmd = json_module.loads(cmd_json)
                # Only process if timestamp is newer than last processed
                if cmd.get('timestamp', 0) > self.last_voice_timestamp:
                    self.last_voice_timestamp = cmd['timestamp']
                    command = cmd.get('command')
                    self.last_voice_text = cmd.get('text', '')
                    self.last_voice_time = time.time()
                    self.voice_status = "redis:active"
                    
                    # 'stop' is emergency exit, 'pause' is just pause
                    if command == 'stop':
                        self.voice_command = 'stop'  # Will trigger exit
                    else:
                        self.voice_command = command
        except Exception as e:
            pass  # Redis errors are non-fatal
    
    def check_clap(self, left_hand_data, right_hand_data, left_inspire, right_inspire):
        """
        Check for clap gesture with latch (both hands open + close together)
        Returns True only on the rising edge (when clap first detected)
        Must separate hands before next clap can be detected
        """
        if not left_hand_data or not right_hand_data:
            return False
        
        # Get palm positions
        left_palm_key = "LeftHandPalm"
        right_palm_key = "RightHandPalm"
        
        if left_palm_key not in left_hand_data or right_palm_key not in right_hand_data:
            return False
        
        left_palm = np.array(left_hand_data[left_palm_key][0])
        right_palm = np.array(right_hand_data[right_palm_key][0])
        
        palm_distance = np.linalg.norm(left_palm - right_palm)
        
        # Check if both hands are open
        fingers = ['index', 'middle', 'ring', 'little']
        left_open = all(left_inspire.get(f, 500) < self.OPEN_THRESHOLD for f in fingers)
        right_open = all(right_inspire.get(f, 500) < self.OPEN_THRESHOLD for f in fingers)
        
        hands_together = left_open and right_open and palm_distance < self.CLAP_DISTANCE_THRESHOLD
        hands_separated = palm_distance > self.CLAP_RELEASE_THRESHOLD
        
        # Latch logic: only trigger once per clap
        if self.clap_latched:
            # Waiting for hands to separate
            if hands_separated:
                self.clap_latched = False  # Reset latch
            return False
        else:
            # Looking for clap
            if hands_together:
                self.clap_latched = True  # Set latch
                return True  # Trigger!
            return False
    
    def check_controller_buttons(self):
        """Check controller buttons for pause/exit"""
        try:
            controller_data = self.streamer.get_controller_data()
            
            # A button (right controller key_one)
            a_pressed = controller_data['RightController']['key_one']
            
            # Debug: print when button state changes
            if a_pressed != self.a_button_was_pressed:
                print(f"[DEBUG] A button: {a_pressed}")
            
            if a_pressed:
                if self.a_button_pressed_time is None:
                    self.a_button_pressed_time = time.time()
                
                hold_duration = time.time() - self.a_button_pressed_time
                
                if hold_duration >= self.A_HOLD_DURATION:
                    self.a_button_was_pressed = a_pressed
                    return 'exit'  # Long press = exit
            else:
                if self.a_button_pressed_time is not None:
                    # Button was released
                    hold_duration = time.time() - self.a_button_pressed_time
                    self.a_button_pressed_time = None
                    self.a_button_was_pressed = a_pressed
                    
                    if hold_duration < self.A_HOLD_DURATION:
                        return 'pause'  # Short tap = pause toggle
            
            self.a_button_was_pressed = a_pressed
            return None
            
        except Exception as e:
            print(f"[DEBUG] Controller error: {e}")
            return None
    
    def check_voice(self):
        """Check for voice commands from Redis. Returns ('pause'/'resume', text) or (None, None)"""
        if not self.enable_voice:
            return None, None
        
        # Poll Redis for new voice commands
        self._check_redis_voice()
        
        cmd = self.voice_command
        text = self.last_voice_text if cmd else None
        self.voice_command = None
        return cmd, text
    
    def stop(self):
        """Stop all monitoring"""
        self.voice_running = False
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass


class HandGestureProcessor:
    """Process hand tracking data into Inspire hand values"""
    
    FINGER_TIPS = {
        'thumb': 'ThumbTip',
        'index': 'IndexTip',
        'middle': 'MiddleTip',
        'ring': 'RingTip',
        'little': 'LittleTip'
    }
    
    INSPIRE_ORDER = ['little', 'ring', 'middle', 'index', 'thumb']
    INSPIRE_MAX = 1000
    
    def __init__(self, streamer=None):
        self.streamer = streamer
        
        # Default calibration (can be updated via calibrate())
        self.max_extension = {
            'thumb': 0.05, 'index': 0.18, 'middle': 0.19, 'ring': 0.17, 'little': 0.15
        }
        self.min_extension = {
            'thumb': 0.02, 'index': 0.05, 'middle': 0.05, 'ring': 0.05, 'little': 0.05
        }
        # Based on observed data:
        # Full rotation: ~5.45cm, Full open: ~6.2cm, 50%: ~6.4cm
        self.thumb_rot_max_dist = 0.065  # 6.5cm - spread out → 0
        self.thumb_rot_min_dist = 0.054  # 5.4cm - rotated in → 1000
        
        self.smoothing_alpha = 0.4
        self.prev_values = {'left': {}, 'right': {}}
    
    def get_all_finger_distances(self, hand_data, side):
        """Get all finger distances for calibration"""
        distances = {}
        palm = self.get_joint_position(hand_data, 'Palm', side)
        if palm is None:
            return distances
        
        for finger in self.INSPIRE_ORDER:
            tip = self.get_joint_position(hand_data, self.FINGER_TIPS[finger], side)
            if finger == 'thumb':
                ref = self.get_joint_position(hand_data, 'ThumbProximal', side)
            else:
                ref = palm
            
            if tip is not None and ref is not None:
                distances[finger] = np.linalg.norm(tip - ref)
        
        # Thumb rotation
        thumb_prox = self.get_joint_position(hand_data, 'ThumbProximal', side)
        if thumb_prox is not None and palm is not None:
            distances['thumb_rot'] = np.linalg.norm(thumb_prox - palm)
        
        return distances
    
    def calibrate(self, streamer):
        """Interactive calibration for hand size"""
        print("\n" + "="*70)
        print("🎯 CALIBRATION MODE")
        print("="*70)
        print("This will calibrate finger tracking for your hand size.")
        print("For each pose, hold steady and press ENTER.\n")
        
        calib_min = {'left': {}, 'right': {}}
        calib_max = {'left': {}, 'right': {}}
        
        for side in ['left', 'right']:
            side_upper = side.upper()
            
            # Step 1: Closed fist
            print(f"\n{'='*70}")
            print(f"📍 {side_upper} HAND - Step 1/2: CLOSED FIST")
            print(f"{'='*70}")
            print(f"Make a CLOSED FIST with your {side} hand (thumb tucked into palm)")
            print("Hold steady and press ENTER...")
            
            samples = []
            while True:
                if side == 'left':
                    is_active, hand_data = streamer.get_left_hand_data()
                else:
                    is_active, hand_data = streamer.get_right_hand_data()
                
                if is_active and hand_data:
                    distances = self.get_all_finger_distances(hand_data, side)
                    if len(distances) >= 5:
                        samples.append(distances)
                        print(f"\r  Live: Thumb={distances.get('thumb',0)*100:.1f}cm, "
                              f"Index={distances.get('index',0)*100:.1f}cm, "
                              f"ThumbRot={distances.get('thumb_rot',0)*100:.1f}cm  ", end="", flush=True)
                
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    sys.stdin.readline()
                    break
                time.sleep(0.05)
            
            if samples:
                avg = {}
                for key in samples[-1].keys():
                    values = [s.get(key, 0) for s in samples[-10:] if key in s]
                    avg[key] = np.mean(values) if values else 0
                calib_min[side] = avg
                print(f"\n✓ Captured CLOSED FIST")
            
            # Step 2: Open palm
            print(f"\n{'='*70}")
            print(f"📍 {side_upper} HAND - Step 2/2: OPEN PALM")
            print(f"{'='*70}")
            print(f"Now OPEN your {side} hand fully (fingers spread)")
            print("Hold steady and press ENTER...")
            
            samples = []
            while True:
                if side == 'left':
                    is_active, hand_data = streamer.get_left_hand_data()
                else:
                    is_active, hand_data = streamer.get_right_hand_data()
                
                if is_active and hand_data:
                    distances = self.get_all_finger_distances(hand_data, side)
                    if len(distances) >= 5:
                        samples.append(distances)
                        print(f"\r  Live: Thumb={distances.get('thumb',0)*100:.1f}cm, "
                              f"Index={distances.get('index',0)*100:.1f}cm, "
                              f"ThumbRot={distances.get('thumb_rot',0)*100:.1f}cm  ", end="", flush=True)
                
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    sys.stdin.readline()
                    break
                time.sleep(0.05)
            
            if samples:
                avg = {}
                for key in samples[-1].keys():
                    values = [s.get(key, 0) for s in samples[-10:] if key in s]
                    avg[key] = np.mean(values) if values else 0
                calib_max[side] = avg
                print(f"\n✓ Captured OPEN PALM")
        
        # Apply calibration (average both hands)
        for finger in self.INSPIRE_ORDER:
            min_vals = [calib_min[s].get(finger, 0) for s in ['left', 'right']]
            max_vals = [calib_max[s].get(finger, 0) for s in ['left', 'right']]
            
            if any(v > 0 for v in min_vals):
                self.min_extension[finger] = np.mean([v for v in min_vals if v > 0])
            if any(v > 0 for v in max_vals):
                self.max_extension[finger] = np.mean([v for v in max_vals if v > 0])
        
        # Thumb rotation
        rot_min = [calib_min[s].get('thumb_rot', 0) for s in ['left', 'right']]
        rot_max = [calib_max[s].get('thumb_rot', 0) for s in ['left', 'right']]
        if any(v > 0 for v in rot_min):
            self.thumb_rot_min_dist = np.mean([v for v in rot_min if v > 0])
        if any(v > 0 for v in rot_max):
            self.thumb_rot_max_dist = np.mean([v for v in rot_max if v > 0])
        
        print("\n" + "="*70)
        print("✅ CALIBRATION COMPLETE!")
        print("="*70)
        print("\nCalibration Results (in cm):")
        print(f"{'Finger':<12} {'Closed':<12} {'Open':<12}")
        print("-" * 36)
        for finger in self.INSPIRE_ORDER:
            print(f"{finger:<12} {self.min_extension[finger]*100:>8.2f}cm  {self.max_extension[finger]*100:>8.2f}cm")
        print(f"{'thumb_rot':<12} {self.thumb_rot_min_dist*100:>8.2f}cm  {self.thumb_rot_max_dist*100:>8.2f}cm")
        
        print("\nPress ENTER to start hand teleop...")
        input()
    
    def get_joint_position(self, hand_data, joint_name, side):
        """Get 3D position of a joint"""
        key = f"{side.capitalize()}Hand{joint_name}"
        if key in hand_data:
            return np.array(hand_data[key][0])
        return None
    
    def get_thumb_debug(self, hand_data, side):
        """Get debug info for thumb rotation calculation"""
        thumb_tip = self.get_joint_position(hand_data, 'ThumbTip', side)
        thumb_prox = self.get_joint_position(hand_data, 'ThumbProximal', side)
        index_meta = self.get_joint_position(hand_data, 'IndexMetacarpal', side)
        palm = self.get_joint_position(hand_data, 'Palm', side)
        
        if any(x is None for x in [thumb_tip, thumb_prox, index_meta, palm]):
            return None
        
        return {
            'thumb_tip': thumb_tip,
            'thumb_prox': thumb_prox,
            'index_meta': index_meta,
            'palm': palm,
            'tip_to_index': np.linalg.norm(thumb_tip - index_meta),
            'prox_to_palm': np.linalg.norm(thumb_prox - palm),
            'tip_to_palm': np.linalg.norm(thumb_tip - palm),
        }
    
    def process_hand(self, hand_data, side):
        """
        Process hand data into Inspire values
        Returns: array of 6 values [Little, Ring, Middle, Index, ThumbBend, ThumbRot]
        """
        output = []
        inspire_dict = {}
        
        palm = self.get_joint_position(hand_data, 'Palm', side)
        if palm is None:
            return None, {}
        
        for finger in self.INSPIRE_ORDER:
            tip = self.get_joint_position(hand_data, self.FINGER_TIPS[finger], side)
            
            if finger == 'thumb':
                ref = self.get_joint_position(hand_data, 'ThumbProximal', side)
            else:
                ref = palm
            
            if tip is None or ref is None:
                inspire_dict[finger] = 500
                output.append(500)
                continue
            
            distance = np.linalg.norm(tip - ref)
            min_ext = self.min_extension[finger]
            max_ext = self.max_extension[finger]
            
            normalized = (max_ext - distance) / (max_ext - min_ext)
            normalized = np.clip(normalized, 0.0, 1.0)
            
            value = int(normalized * self.INSPIRE_MAX)
            
            # Smoothing
            if finger in self.prev_values[side]:
                value = int(self.smoothing_alpha * value + 
                           (1 - self.smoothing_alpha) * self.prev_values[side][finger])
            self.prev_values[side][finger] = value
            
            value = int(np.clip(value, 0, self.INSPIRE_MAX))
            inspire_dict[finger] = value
            output.append(value)
        
        # Thumb rotation - simple ThumbProximal to Palm distance
        # Smaller distance = more rotation = higher value
        thumb_prox = self.get_joint_position(hand_data, 'ThumbProximal', side)
        
        if thumb_prox is not None and palm is not None:
            dist = np.linalg.norm(thumb_prox - palm)
            
            # Map distance to 0-1000
            # Small distance (rotated in) → high value
            # Large distance (spread out) → low value
            normalized = (self.thumb_rot_max_dist - dist) / (self.thumb_rot_max_dist - self.thumb_rot_min_dist)
            normalized = np.clip(normalized, 0.0, 1.0)
            thumb_rot = int(normalized * self.INSPIRE_MAX)
            
            if 'thumb_rot' in self.prev_values[side]:
                thumb_rot = int(self.smoothing_alpha * thumb_rot +
                               (1 - self.smoothing_alpha) * self.prev_values[side]['thumb_rot'])
            self.prev_values[side]['thumb_rot'] = thumb_rot
        else:
            thumb_rot = 0
        
        inspire_dict['thumb_rot'] = thumb_rot
        output.append(thumb_rot)
        
        return np.array(output), inspire_dict


class HandTeleop:
    """Main hand teleop controller with safety features"""
    
    # States
    STATE_IDLE = 'idle'
    STATE_ACTIVE = 'active'
    STATE_PAUSED = 'paused'
    STATE_EXIT = 'exit'
    
    def __init__(self, args):
        self.args = args
        self.state = self.STATE_IDLE
        
        # Frozen values when paused
        self.frozen_left = None
        self.frozen_right = None
        
        # Initialize components
        print("\n" + "="*70)
        print("🤖 HAND TELEOP - Initializing...")
        print("="*70)
        
        # XRoboToolkit streamer
        self.streamer = XRobotStreamer()
        print("✓ XRobotStreamer connected")
        
        # Hand gesture processor
        self.gesture_processor = HandGestureProcessor(self.streamer)
        print("✓ Gesture processor ready")
        
        # Calibration
        if not (hasattr(args, 'skip_calibration') and args.skip_calibration):
            print("\nWould you like to calibrate for your hand size? (y/n): ", end="", flush=True)
            response = input().strip().lower()
            if response in ['y', 'yes', '']:
                self.gesture_processor.calibrate(self.streamer)
            else:
                print("Skipping calibration, using default values.")
        
        # Safety monitor
        voice_model = args.voice_model if hasattr(args, 'voice_model') else None
        self.safety = SafetyMonitor(
            self.streamer, 
            enable_voice=args.enable_voice if hasattr(args, 'enable_voice') else False,
            voice_model_path=voice_model
        )
        print("✓ Safety monitor ready")
        
        # Inspire hands
        self.hand_controller = None
        if DualHandController and args.enable_hands:
            try:
                self.hand_controller = DualHandController(
                    left_ip=args.left_hand_ip,
                    right_ip=args.right_hand_ip,
                    timeout=3.0,
                    async_mode=True
                )
                self.hand_controller.open_both()
                print(f"✓ Inspire hands connected: L={args.left_hand_ip}, R={args.right_hand_ip}")
            except Exception as e:
                print(f"✗ Inspire hands failed: {e}")
                self.hand_controller = None
        else:
            print("ℹ Inspire hands disabled (simulation mode)")
        
        print("\n" + "="*70)
        print("🎮 CONTROLS")
        print("="*70)
        print("  👏 CLAP (both open hands together) → PAUSE/RESUME")
        print("  🎮 A Button TAP → PAUSE/RESUME")
        print("  🎮 A Button HOLD (500ms) → EXIT")
        if self.safety.enable_voice:
            print("  🎤 Say 'pause'/'stop' → PAUSE")
            print("  🎤 Say 'resume'/'yes'/'go' → RESUME")
        print("  ⌨️  Ctrl+C → EXIT")
        print("="*70 + "\n")
    
    def inspire_to_raw(self, inspire_values):
        """Convert 0-1000 Inspire values to 0-2000 raw angle values"""
        # Our values: 0=open, 1000=closed
        # Inspire raw: 0=closed, 2000=open (inverted!)
        # So we need to invert: 0→2000, 1000→0
        inverted = 1000 - np.array(inspire_values)
        return (inverted * 2).astype(int)
    
    def send_to_hands(self, left_values, right_values):
        """Send values to Inspire hands"""
        if self.hand_controller is None:
            return
        
        try:
            # Convert to raw angles
            left_raw = self.inspire_to_raw(left_values)
            right_raw = self.inspire_to_raw(right_values)
            
            # Send to hands
            self.hand_controller.left_hand.set_angles(left_raw)
            self.hand_controller.right_hand.set_angles(right_raw)
        except Exception as e:
            print(f"[ERROR] Hand control: {e}")
    
    def display_status(self, left_values, right_values, left_dict, right_dict, 
                        left_hand_data=None, right_hand_data=None):
        """Display current status with debug info"""
        # Clear screen
        print("\033[H\033[J", end="")
        
        state_emoji = {
            self.STATE_IDLE: "⏳",
            self.STATE_ACTIVE: "🟢",
            self.STATE_PAUSED: "⏸️",
            self.STATE_EXIT: "🛑"
        }
        
        print(f"{'='*70}")
        print(f"{state_emoji.get(self.state, '?')} STATE: {self.state.upper()}")
        print(f"{'='*70}")
        
        # Voice status (via Redis from external service)
        voice_status = self.safety.voice_status if hasattr(self.safety, 'voice_status') else "N/A"
        if voice_status.startswith("redis:"):
            voice_icon = "🎤"
            # Show last detected text if recent (within 5 seconds)
            if hasattr(self.safety, 'last_voice_time') and time.time() - self.safety.last_voice_time < 5:
                last_text = self.safety.last_voice_text
                print(f"{voice_icon} Voice (Redis): Last: \"{last_text}\"")
            else:
                print(f"{voice_icon} Voice (Redis): Listening via host service")
        elif voice_status == "disabled":
            if REDIS_AVAILABLE:
                print(f"🔇 Voice: DISABLED (use --enable_voice + run voice_service.py on host)")
            else:
                print(f"🔇 Voice: Redis not available (pip install redis)")
        else:
            print(f"⚠️ Voice: {voice_status}")
        
        hand_data_list = [('LEFT', left_values, left_dict, left_hand_data), 
                          ('RIGHT', right_values, right_dict, right_hand_data)]
        
        for side, values, d, hand_data in hand_data_list:
            if values is not None:
                print(f"\n{side} HAND:")
                dof_names = ['Little', 'Ring', 'Middle', 'Index', 'ThumbBend', 'ThumbRot']
                for i, name in enumerate(dof_names):
                    val = values[i]
                    bar_len = int(val / 50)
                    bar = '█' * bar_len + '░' * (20 - bar_len)
                    print(f"  {name:10} {val:4} [{bar}]")
                
                # Debug info for thumb
                if hand_data:
                    side_lower = side.lower()
                    debug = self.gesture_processor.get_thumb_debug(hand_data, side_lower)
                    if debug:
                        print(f"\n  --- THUMB DEBUG ---")
                        print(f"  ThumbTip pos:    [{debug['thumb_tip'][0]*100:.1f}, {debug['thumb_tip'][1]*100:.1f}, {debug['thumb_tip'][2]*100:.1f}] cm")
                        print(f"  ThumbProx pos:   [{debug['thumb_prox'][0]*100:.1f}, {debug['thumb_prox'][1]*100:.1f}, {debug['thumb_prox'][2]*100:.1f}] cm")
                        print(f"  IndexMeta pos:   [{debug['index_meta'][0]*100:.1f}, {debug['index_meta'][1]*100:.1f}, {debug['index_meta'][2]*100:.1f}] cm")
                        print(f"  Palm pos:        [{debug['palm'][0]*100:.1f}, {debug['palm'][1]*100:.1f}, {debug['palm'][2]*100:.1f}] cm")
                        print(f"  ---")
                        print(f"  ThumbTip→IndexMeta dist: {debug['tip_to_index']*100:.2f} cm")
                        print(f"  ThumbProx→Palm dist:     {debug['prox_to_palm']*100:.2f} cm")
                        print(f"  ThumbTip→Palm dist:      {debug['tip_to_palm']*100:.2f} cm")
                        print(f"  ---")
                        print(f"  Calib max_dist: {self.gesture_processor.thumb_rot_max_dist*100:.2f} cm")
                        print(f"  Calib min_dist: {self.gesture_processor.thumb_rot_min_dist*100:.2f} cm")
            else:
                print(f"\n{side} HAND: Not detected")
        
        if self.state == self.STATE_PAUSED:
            print(f"\n⏸️  PAUSED - Hands frozen at current position")
            print(f"   👏 Clap or press A to resume")
    
    def run(self):
        """Main loop"""
        print("Starting hand teleop... Press Ctrl+C to exit")
        
        try:
            while self.state != self.STATE_EXIT:
                # Check safety triggers
                
                # 1. Controller buttons
                button_action = self.safety.check_controller_buttons()
                if button_action == 'exit':
                    print("\n🛑 A BUTTON HELD - EXITING...")
                    self.state = self.STATE_EXIT
                    break
                elif button_action == 'pause':
                    if self.state == self.STATE_PAUSED:
                        self.state = self.STATE_ACTIVE
                        print("\n▶️  RESUMED via A button")
                    elif self.state == self.STATE_ACTIVE:
                        self.state = self.STATE_PAUSED
                        print("\n⏸️  PAUSED via A button")
                
                # 2. Voice commands
                voice_cmd, voice_text = self.safety.check_voice()
                if voice_cmd == 'stop':
                    # Emergency stop - exit program
                    print(f"\n🛑 EMERGENCY STOP via voice: \"{voice_text}\"")
                    self.state = self.STATE_EXIT
                    break
                elif voice_cmd == 'pause' and self.state == self.STATE_ACTIVE:
                    self.state = self.STATE_PAUSED
                    print(f"\n⏸️  PAUSED via voice: \"{voice_text}\"")
                elif voice_cmd == 'resume' and self.state == self.STATE_PAUSED:
                    self.state = self.STATE_ACTIVE
                    print(f"\n▶️  RESUMED via voice: \"{voice_text}\"")
                
                # Get hand data
                left_active, left_hand_data = self.streamer.get_left_hand_data()
                right_active, right_hand_data = self.streamer.get_right_hand_data()
                
                # Process hands
                left_values, left_dict = None, {}
                right_values, right_dict = None, {}
                
                if left_active and left_hand_data:
                    left_values, left_dict = self.gesture_processor.process_hand(left_hand_data, 'left')
                
                if right_active and right_hand_data:
                    right_values, right_dict = self.gesture_processor.process_hand(right_hand_data, 'right')
                
                # 3. Clap detection
                if left_hand_data and right_hand_data:
                    is_clap = self.safety.check_clap(left_hand_data, right_hand_data, left_dict, right_dict)
                    if is_clap:
                        if self.state == self.STATE_PAUSED:
                            self.state = self.STATE_ACTIVE
                            print("\n▶️  RESUMED via clap")
                        elif self.state == self.STATE_ACTIVE:
                            self.state = self.STATE_PAUSED
                            # Freeze current values
                            self.frozen_left = left_values.copy() if left_values is not None else None
                            self.frozen_right = right_values.copy() if right_values is not None else None
                            print("\n⏸️  PAUSED via clap")
                        time.sleep(0.5)  # Debounce
                
                # State machine
                if self.state == self.STATE_IDLE:
                    # Transition to active when hands detected
                    if left_active or right_active:
                        self.state = self.STATE_ACTIVE
                
                elif self.state == self.STATE_ACTIVE:
                    # Send hand values to Inspire hands
                    if left_values is not None or right_values is not None:
                        l = left_values if left_values is not None else np.full(6, 0)
                        r = right_values if right_values is not None else np.full(6, 0)
                        self.send_to_hands(l, r)
                
                elif self.state == self.STATE_PAUSED:
                    # Keep sending frozen values
                    if self.frozen_left is not None or self.frozen_right is not None:
                        l = self.frozen_left if self.frozen_left is not None else np.full(6, 0)
                        r = self.frozen_right if self.frozen_right is not None else np.full(6, 0)
                        self.send_to_hands(l, r)
                
                # Display with debug info
                if self.state == self.STATE_PAUSED:
                    self.display_status(self.frozen_left, self.frozen_right, {}, {}, 
                                       left_hand_data, right_hand_data)
                else:
                    self.display_status(left_values, right_values, left_dict, right_dict,
                                       left_hand_data, right_hand_data)
                
                time.sleep(0.02)  # ~50Hz
                
        except KeyboardInterrupt:
            print("\n\n🛑 Ctrl+C - EXITING...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        self.safety.stop()
        
        if self.hand_controller:
            print("Opening hands...")
            self.hand_controller.open_both()
            time.sleep(0.5)
            self.hand_controller.stop()
        
        print("✓ Hand teleop terminated safely")


def parse_args():
    parser = argparse.ArgumentParser(description="Hand-only teleop with safety features")
    
    parser.add_argument("--enable_hands", action="store_true",
                        help="Enable Inspire hand control (otherwise simulation mode)")
    parser.add_argument("--left_hand_ip", type=str, default="192.168.123.210",
                        help="IP address for left Inspire hand")
    parser.add_argument("--right_hand_ip", type=str, default="192.168.123.211",
                        help="IP address for right Inspire hand")
    parser.add_argument("--enable_voice", action="store_true",
                        help="Enable voice commands (requires vosk)")
    parser.add_argument("--voice_model", type=str, default=None,
                        help="Path to vosk model directory")
    parser.add_argument("--skip_calibration", "-s", action="store_true",
                        help="Skip calibration and use default values")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    teleop = HandTeleop(args)
    teleop.run()

