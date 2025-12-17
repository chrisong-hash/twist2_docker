#!/usr/bin/env python3
"""
Gesture Detection Test Script
Tests hand tracking from Pico and maps to Inspire hand output

Focus:
    - Thumb rotation calculation
    - Display observed min/max ranges for each finger
"""

import numpy as np
import time
import sys
import os
import select

# Try to import XRoboToolkit
try:
    from general_motion_retargeting import XRobotStreamer
except ImportError:
    print("ERROR: Could not import XRobotStreamer")
    print("Make sure you're in the gmr conda environment and XRoboToolkit is running")
    sys.exit(1)


class HandGestureDetector:
    """Detect gestures from XRoboToolkit hand tracking data"""
    
    FINGER_TIPS = {
        'thumb': 'ThumbTip',
        'index': 'IndexTip',
        'middle': 'MiddleTip',
        'ring': 'RingTip',
        'little': 'LittleTip'
    }
    
    # Inspire hand DOF order: Little, Ring, Middle, Index, Thumb_Bend, Thumb_Rotate
    INSPIRE_ORDER = ['little', 'ring', 'middle', 'index', 'thumb']
    INSPIRE_DOF_NAMES = ['Little', 'Ring', 'Middle', 'Index', 'ThumbBend', 'ThumbRot']
    
    # Inspire hand range
    INSPIRE_MIN = 0     # Open
    INSPIRE_MAX = 1000  # Closed
    
    # Clap detection
    CLAP_DISTANCE_THRESHOLD = 0.10  # 10 cm
    OPEN_THRESHOLD = 300  # Below this = finger is OPEN
    
    def __init__(self):
        self.streamer = XRobotStreamer()
        
        # Calibration values for tip-to-palm distance (in meters)
        # These will be used to map distance to Inspire values
        self.max_extension = {  # Distance when finger is fully extended (open)
            'thumb': 0.05,   # ThumbTip to ThumbProximal when extended
            'index': 0.18,
            'middle': 0.19,
            'ring': 0.17,
            'little': 0.15
        }
        self.min_extension = {  # Distance when finger is fully bent (closed)
            'thumb': 0.02,   # ThumbTip to ThumbProximal when bent
            'index': 0.05,
            'middle': 0.05,
            'ring': 0.05,
            'little': 0.05
        }
        
        # Thumb rotation calibration (ThumbProximal to Palm)
        # Based on observed data: full rot ~5.45cm, spread out ~6.2-6.4cm
        self.thumb_rot_max_dist = 0.065  # 6.5cm - spread out → 0
        self.thumb_rot_min_dist = 0.054  # 5.4cm - rotated in → 1000
        
        # Track observed min/max for calibration display
        self.observed_min = {'left': {f: float('inf') for f in self.INSPIRE_ORDER + ['thumb_rot']},
                            'right': {f: float('inf') for f in self.INSPIRE_ORDER + ['thumb_rot']}}
        self.observed_max = {'left': {f: float('-inf') for f in self.INSPIRE_ORDER + ['thumb_rot']},
                            'right': {f: float('-inf') for f in self.INSPIRE_ORDER + ['thumb_rot']}}
        
        # Track RAW distances for calibration
        # thumb_bend: ThumbTip to ThumbProximal
        # thumb_rot: ThumbProximal to Palm
        self.raw_dist_min = {'left': {'thumb_bend': float('inf'), 'thumb_rot': float('inf')},
                            'right': {'thumb_bend': float('inf'), 'thumb_rot': float('inf')}}
        self.raw_dist_max = {'left': {'thumb_bend': float('-inf'), 'thumb_rot': float('-inf')},
                            'right': {'thumb_bend': float('-inf'), 'thumb_rot': float('-inf')}}
        
        # Smoothing
        self.smoothing_alpha = 0.4
        self.prev_inspire = {'left': {}, 'right': {}}
        
        # Clap state
        self.clap_cooldown = 0
        
        # Per-hand calibration (will be set during calibrate())
        self.calibrated = False
        self.calib_min = {'left': {}, 'right': {}}  # Closed fist distances
        self.calib_max = {'left': {}, 'right': {}}  # Open palm distances
    
    def get_all_finger_distances(self, hand_data, side):
        """Get all finger tip-to-reference distances for calibration"""
        distances = {}
        palm = self.get_joint_position(hand_data, 'Palm', side)
        
        for finger in self.INSPIRE_ORDER:
            tip = self.get_joint_position(hand_data, self.FINGER_TIPS[finger], side)
            if finger == 'thumb':
                ref = self.get_joint_position(hand_data, 'ThumbProximal', side)
            else:
                ref = palm
            
            if tip is not None and ref is not None:
                distances[finger] = np.linalg.norm(tip - ref)
        
        # Thumb rotation: ThumbProximal to Palm
        thumb_prox = self.get_joint_position(hand_data, 'ThumbProximal', side)
        if thumb_prox is not None and palm is not None:
            distances['thumb_rot'] = np.linalg.norm(thumb_prox - palm)
        
        return distances
    
    def calibrate(self):
        """Interactive calibration for left and right hands"""
        print("\n" + "="*70)
        print("🎯 CALIBRATION MODE")
        print("="*70)
        print("This will calibrate the finger tracking for your hand size.")
        print("For each pose, hold steady and press ENTER to capture.\n")
        
        for side in ['left', 'right']:
            side_upper = side.upper()
            
            # Step 1: Closed fist
            print(f"\n{'='*70}")
            print(f"📍 {side_upper} HAND - Step 1/2: CLOSED FIST")
            print(f"{'='*70}")
            print(f"Make a CLOSED FIST with your {side} hand (thumb tucked into palm)")
            print("Hold steady...")
            
            # Wait for hand to be active and stable
            samples = []
            while True:
                if side == 'left':
                    is_active, hand_data = self.streamer.get_left_hand_data()
                else:
                    is_active, hand_data = self.streamer.get_right_hand_data()
                
                if is_active and hand_data:
                    distances = self.get_all_finger_distances(hand_data, side)
                    if len(distances) >= 5:  # All fingers detected
                        samples.append(distances)
                        # Show live preview
                        print(f"\r  Live: Thumb={distances.get('thumb',0)*100:.1f}cm, "
                              f"Index={distances.get('index',0)*100:.1f}cm, "
                              f"ThumbRot={distances.get('thumb_rot',0)*100:.1f}cm  "
                              f"[Press ENTER to capture]", end="", flush=True)
                
                # Check for Enter key (non-blocking)
                import select
                import sys
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    sys.stdin.readline()
                    break
                
                time.sleep(0.05)
            
            # Average the last few samples
            if samples:
                avg_closed = {}
                for key in samples[-1].keys():
                    values = [s.get(key, 0) for s in samples[-10:] if key in s]
                    avg_closed[key] = np.mean(values) if values else 0
                self.calib_min[side] = avg_closed
                print(f"\n✓ Captured CLOSED FIST for {side_upper} hand")
            
            # Step 2: Open palm
            print(f"\n{'='*70}")
            print(f"📍 {side_upper} HAND - Step 2/2: OPEN PALM")
            print(f"{'='*70}")
            print(f"Now OPEN your {side} hand fully (fingers spread)")
            print("Hold steady...")
            
            samples = []
            while True:
                if side == 'left':
                    is_active, hand_data = self.streamer.get_left_hand_data()
                else:
                    is_active, hand_data = self.streamer.get_right_hand_data()
                
                if is_active and hand_data:
                    distances = self.get_all_finger_distances(hand_data, side)
                    if len(distances) >= 5:
                        samples.append(distances)
                        print(f"\r  Live: Thumb={distances.get('thumb',0)*100:.1f}cm, "
                              f"Index={distances.get('index',0)*100:.1f}cm, "
                              f"ThumbRot={distances.get('thumb_rot',0)*100:.1f}cm  "
                              f"[Press ENTER to capture]", end="", flush=True)
                
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    sys.stdin.readline()
                    break
                
                time.sleep(0.05)
            
            if samples:
                avg_open = {}
                for key in samples[-1].keys():
                    values = [s.get(key, 0) for s in samples[-10:] if key in s]
                    avg_open[key] = np.mean(values) if values else 0
                self.calib_max[side] = avg_open
                print(f"\n✓ Captured OPEN PALM for {side_upper} hand")
        
        # Apply calibration
        self._apply_calibration()
        self.calibrated = True
        
        print("\n" + "="*70)
        print("✅ CALIBRATION COMPLETE!")
        print("="*70)
        self._print_calibration_summary()
        print("\nPress ENTER to start gesture detection...")
        input()
    
    def _apply_calibration(self):
        """Apply calibrated values to the extension dictionaries"""
        # Average left and right hand calibrations
        for finger in self.INSPIRE_ORDER:
            min_vals = [self.calib_min[s].get(finger, 0) for s in ['left', 'right']]
            max_vals = [self.calib_max[s].get(finger, 0) for s in ['left', 'right']]
            
            self.min_extension[finger] = np.mean([v for v in min_vals if v > 0]) if any(v > 0 for v in min_vals) else self.min_extension[finger]
            self.max_extension[finger] = np.mean([v for v in max_vals if v > 0]) if any(v > 0 for v in max_vals) else self.max_extension[finger]
        
        # Thumb rotation
        rot_min = [self.calib_min[s].get('thumb_rot', 0) for s in ['left', 'right']]
        rot_max = [self.calib_max[s].get('thumb_rot', 0) for s in ['left', 'right']]
        
        if any(v > 0 for v in rot_min):
            self.thumb_rot_min_dist = np.mean([v for v in rot_min if v > 0])
        if any(v > 0 for v in rot_max):
            self.thumb_rot_max_dist = np.mean([v for v in rot_max if v > 0])
    
    def _print_calibration_summary(self):
        """Print a summary of the calibration"""
        print("\nCalibration Results (in cm):")
        print(f"{'Finger':<12} {'Closed (min)':<15} {'Open (max)':<15}")
        print("-" * 42)
        for finger in self.INSPIRE_ORDER:
            print(f"{finger:<12} {self.min_extension[finger]*100:>10.2f}cm    {self.max_extension[finger]*100:>10.2f}cm")
        print(f"{'thumb_rot':<12} {self.thumb_rot_min_dist*100:>10.2f}cm    {self.thumb_rot_max_dist*100:>10.2f}cm")
        
    def get_joint_position(self, hand_data, joint_name, side='left'):
        """Get 3D position of a joint"""
        key = f"{side.capitalize()}Hand{joint_name}"
        if key in hand_data:
            return np.array(hand_data[key][0])  # [pos, rot] -> pos
        return None
    
    def calculate_inspire_value(self, hand_data, finger, side='left'):
        """
        Calculate Inspire hand value for a finger (0=open, 1000=closed)
        For thumb: distance from ThumbTip to ThumbProximal (measures curl)
        For other fingers: distance from fingertip to palm
        """
        tip_pos = self.get_joint_position(hand_data, self.FINGER_TIPS[finger], side)
        
        if finger == 'thumb':
            # Thumb: measure tip to proximal distance (direct curl measurement)
            ref_pos = self.get_joint_position(hand_data, 'ThumbProximal', side)
        else:
            # Other fingers: measure tip to palm distance
            ref_pos = self.get_joint_position(hand_data, 'Palm', side)
        
        if tip_pos is None or ref_pos is None:
            return None
        
        distance = np.linalg.norm(tip_pos - ref_pos)
        
        # Map distance to Inspire value
        min_ext = self.min_extension[finger]
        max_ext = self.max_extension[finger]
        
        # Normalize: 0 (at max_ext/open) to 1 (at min_ext/closed)
        normalized = (max_ext - distance) / (max_ext - min_ext)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Convert to Inspire value
        inspire_val = int(normalized * self.INSPIRE_MAX)
        
        # Apply smoothing
        if finger in self.prev_inspire[side]:
            inspire_val = int(self.smoothing_alpha * inspire_val + 
                             (1 - self.smoothing_alpha) * self.prev_inspire[side][finger])
        self.prev_inspire[side][finger] = inspire_val
        
        # Clamp to valid range
        inspire_val = int(np.clip(inspire_val, self.INSPIRE_MIN, self.INSPIRE_MAX))
        
        # Track observed range
        if inspire_val < self.observed_min[side][finger]:
            self.observed_min[side][finger] = inspire_val
        if inspire_val > self.observed_max[side][finger]:
            self.observed_max[side][finger] = inspire_val
        
        return inspire_val
    
    def calculate_thumb_rotation(self, hand_data, side='left'):
        """
        Calculate thumb rotation (0=rotated out, 1000=rotated in)
        Based on ThumbProximal to Palm distance
        - When thumb rotates in toward palm: Proximal gets closer to Palm → high value
        - When thumb spreads out: Proximal moves away from Palm → low value
        Note: Cannot reliably distinguish forward from backward flex with distance alone
        """
        thumb_proximal = self.get_joint_position(hand_data, 'ThumbProximal', side)
        palm = self.get_joint_position(hand_data, 'Palm', side)
        
        if thumb_proximal is None or palm is None:
            return 500  # Default middle
        
        # Simple distance from ThumbProximal to Palm
        proximal_to_palm_dist = np.linalg.norm(thumb_proximal - palm)
        
        # Map to 0-1000:
        # - Large distance (spread out) → 0
        # - Small distance (rotated in) → 1000
        max_dist = self.thumb_rot_max_dist
        min_dist = self.thumb_rot_min_dist
        
        normalized = (max_dist - proximal_to_palm_dist) / (max_dist - min_dist)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        thumb_rot = int(normalized * self.INSPIRE_MAX)
        
        # Apply smoothing
        key = 'thumb_rot'
        if key in self.prev_inspire[side]:
            thumb_rot = int(self.smoothing_alpha * thumb_rot + 
                           (1 - self.smoothing_alpha) * self.prev_inspire[side][key])
        self.prev_inspire[side][key] = thumb_rot
        
        # Clamp
        thumb_rot = int(np.clip(thumb_rot, self.INSPIRE_MIN, self.INSPIRE_MAX))
        
        # Track observed range
        if thumb_rot < self.observed_min[side]['thumb_rot']:
            self.observed_min[side]['thumb_rot'] = thumb_rot
        if thumb_rot > self.observed_max[side]['thumb_rot']:
            self.observed_max[side]['thumb_rot'] = thumb_rot
        
        return thumb_rot
    
    def get_inspire_output(self, hand_data, side='left'):
        """
        Get Inspire hand output array
        Returns array of 6 values: [Little, Ring, Middle, Index, Thumb_Bend, Thumb_Rotate]
        """
        output = []
        inspire_dict = {}
        
        for finger in self.INSPIRE_ORDER:
            val = self.calculate_inspire_value(hand_data, finger, side)
            if val is not None:
                inspire_dict[finger] = val
                output.append(val)
            else:
                inspire_dict[finger] = 500
                output.append(500)
        
        # Calculate thumb rotation
        thumb_rot = self.calculate_thumb_rotation(hand_data, side)
        inspire_dict['thumb_rot'] = thumb_rot
        output.append(thumb_rot)
        
        return np.array(output), inspire_dict
    
    def detect_gesture(self, inspire_dict, hand_data=None, side='left'):
        """Simple gesture detection based on observed values"""
        index = inspire_dict.get('index', 500)
        middle = inspire_dict.get('middle', 500)
        ring = inspire_dict.get('ring', 500)
        little = inspire_dict.get('little', 500)
        thumb = inspire_dict.get('thumb', 500)
        thumb_rot = inspire_dict.get('thumb_rot', 500)
        
        is_open = lambda v: v < 300
        is_closed = lambda v: v > 600  # Lowered from 700 to catch index finger
        fingers_closed = all(is_closed(v) for v in [index, middle, ring, little])
        fingers_open = all(is_open(v) for v in [index, middle, ring, little])
        
        # Check if thumb is pointing UP by comparing Y positions
        thumb_pointing_up = False
        if hand_data:
            thumb_tip = self.get_joint_position(hand_data, 'ThumbTip', side)
            thumb_base = self.get_joint_position(hand_data, 'ThumbMetacarpal', side)
            if thumb_tip is not None and thumb_base is not None:
                # Y axis is typically up in VR tracking
                # Thumb pointing up if tip is significantly higher than base
                y_diff = thumb_tip[1] - thumb_base[1]
                thumb_pointing_up = y_diff > 0.03  # 3cm higher = pointing up
        
        # Check gestures (order matters - most specific first)
        if fingers_closed and thumb_pointing_up:
            return "👍 THUMBS UP"
        if fingers_closed and not thumb_pointing_up:
            return "✊ FIST"
        if is_open(index) and all(is_closed(v) for v in [middle, ring, little]):
            return "👆 POINTING"
        if is_open(index) and is_open(middle) and ring > 600 and little > 600:
            return "✌️ PEACE"
        if is_open(index) and is_open(little) and middle > 700 and ring > 700:
            return "🤘 ROCK"
        if fingers_open:
            return "🖐️ OPEN HAND"
        
        return "--- (no gesture)"
    
    def check_clap(self, left_hand_data, right_hand_data, left_inspire, right_inspire):
        """
        Check for clap: both hands open + close together
        """
        left_palm = self.get_joint_position(left_hand_data, 'Palm', 'left')
        right_palm = self.get_joint_position(right_hand_data, 'Palm', 'right')
        
        if left_palm is None or right_palm is None:
            return False, float('inf')
        
        palm_distance = np.linalg.norm(left_palm - right_palm)
        
        # Check if both hands are open
        left_open = all(left_inspire.get(f, 500) < self.OPEN_THRESHOLD 
                       for f in ['index', 'middle', 'ring', 'little'])
        right_open = all(right_inspire.get(f, 500) < self.OPEN_THRESHOLD 
                        for f in ['index', 'middle', 'ring', 'little'])
        
        is_clap = (left_open and right_open and 
                   palm_distance < self.CLAP_DISTANCE_THRESHOLD)
        
        return is_clap, palm_distance
    
    def display_hand_status(self, inspire_output, inspire_dict, side, palm_distance=None, hand_data=None):
        """Display hand status with visual bars and observed ranges"""
        print(f"\n{'='*70}")
        print(f"{side.upper()} HAND - Inspire Values (0=open, 1000=closed)")
        print(f"{'='*70}")
        
        # Show raw thumb distances for calibration
        if hand_data:
            thumb_tip = self.get_joint_position(hand_data, 'ThumbTip', side)
            thumb_proximal = self.get_joint_position(hand_data, 'ThumbProximal', side)
            palm = self.get_joint_position(hand_data, 'Palm', side)
            
            thumb_bend_dist = None  # ThumbTip to ThumbProximal
            thumb_rot_dist = None   # ThumbProximal to Palm
            
            if thumb_tip is not None and thumb_proximal is not None:
                thumb_bend_dist = np.linalg.norm(thumb_tip - thumb_proximal)
                # Track min/max
                if thumb_bend_dist < self.raw_dist_min[side]['thumb_bend']:
                    self.raw_dist_min[side]['thumb_bend'] = thumb_bend_dist
                if thumb_bend_dist > self.raw_dist_max[side]['thumb_bend']:
                    self.raw_dist_max[side]['thumb_bend'] = thumb_bend_dist
                    
            if thumb_proximal is not None and palm is not None:
                thumb_rot_dist = np.linalg.norm(thumb_proximal - palm)
                # Track min/max
                if thumb_rot_dist < self.raw_dist_min[side]['thumb_rot']:
                    self.raw_dist_min[side]['thumb_rot'] = thumb_rot_dist
                if thumb_rot_dist > self.raw_dist_max[side]['thumb_rot']:
                    self.raw_dist_max[side]['thumb_rot'] = thumb_rot_dist
            
            # Display current and observed ranges
            if thumb_bend_dist is not None:
                tb_min = self.raw_dist_min[side]['thumb_bend']
                tb_max = self.raw_dist_max[side]['thumb_bend']
                tb_range = f"[{tb_min*100:.1f}-{tb_max*100:.1f}]" if tb_min != float('inf') else "[-.---.--]"
                print(f"📐 ThumbBend (Tip→Prox): {thumb_bend_dist*100:.1f}cm {tb_range}")
            if thumb_rot_dist is not None:
                tr_min = self.raw_dist_min[side]['thumb_rot']
                tr_max = self.raw_dist_max[side]['thumb_rot']
                tr_range = f"[{tr_min*100:.1f}-{tr_max*100:.1f}]" if tr_min != float('inf') else "[-.---.--]"
                print(f"📐 ThumbRot (Prox→Palm): {thumb_rot_dist*100:.1f}cm {tr_range}")
        
        print(f"{'DOF':10} {'Value':>6}  {'Bar':22} {'Observed Range':>20}")
        print(f"{'-'*70}")
        
        for i, dof_name in enumerate(self.INSPIRE_DOF_NAMES):
            inspire_val = inspire_output[i]
            
            # Get observed range
            if dof_name == 'ThumbRot':
                key = 'thumb_rot'
            else:
                key = self.INSPIRE_ORDER[i] if i < 5 else 'thumb_rot'
            
            obs_min = self.observed_min[side].get(key, 0)
            obs_max = self.observed_max[side].get(key, 0)
            
            if obs_min == float('inf'):
                obs_min = '-'
            if obs_max == float('-inf'):
                obs_max = '-'
            
            # Bar: full when closed (1000), empty when open (0)
            bar_len = int(inspire_val / 50)  # 0-20 range for 0-1000
            bar_len = min(20, max(0, bar_len))
            bar = '█' * bar_len + '░' * (20 - bar_len)
            
            range_str = f"[{obs_min:>4} - {obs_max:>4}]" if obs_min != '-' else "[  -  -   -  ]"
            print(f"{dof_name:10} {inspire_val:>6}  [{bar}] {range_str:>20}")
        
        if palm_distance is not None:
            print(f"\n📏 Palm distance: {palm_distance*100:.1f} cm")
    
    def run(self, skip_calibration=False):
        """Main loop - display hand tracking data"""
        print("\n" + "="*70)
        print("GESTURE DETECTION TEST - Focus: Thumb Rotation & Finger Ranges")
        print("="*70)
        print("Make sure XRoboToolkit is running and connected to Pico")
        
        if not skip_calibration:
            print("\nWould you like to calibrate for your hand size? (y/n): ", end="", flush=True)
            response = input().strip().lower()
            if response in ['y', 'yes']:
                self.calibrate()
            else:
                print("Skipping calibration, using default values.")
        
        print("\nMove your fingers to observe the min/max ranges")
        print("Press Ctrl+C to exit")
        print("="*70)
        
        try:
            while True:
                # Get hand data
                left_active, left_hand_data = self.streamer.get_left_hand_data()
                right_active, right_hand_data = self.streamer.get_right_hand_data()
                
                # Clear screen
                print("\033[H\033[J", end="")
                
                print("="*70)
                print("GESTURE TEST - Move fingers to see ranges | Ctrl+C to exit")
                print("="*70)
                
                # Process hands
                left_inspire_output = None
                left_inspire_dict = {}
                right_inspire_output = None
                right_inspire_dict = {}
                
                if left_active and left_hand_data:
                    left_inspire_output, left_inspire_dict = self.get_inspire_output(left_hand_data, 'left')
                
                if right_active and right_hand_data:
                    right_inspire_output, right_inspire_dict = self.get_inspire_output(right_hand_data, 'right')
                
                # Check for clap (no cooldown - continuous detection)
                palm_dist = None
                clap_detected = False
                if left_hand_data and right_hand_data:
                    clap_detected, palm_dist = self.check_clap(
                        left_hand_data, right_hand_data,
                        left_inspire_dict, right_inspire_dict
                    )
                
                # Display left hand
                if left_active and left_hand_data and left_inspire_output is not None:
                    self.display_hand_status(left_inspire_output, left_inspire_dict, 'left', palm_dist, left_hand_data)
                    if clap_detected:
                        print(f"\n🎯 Gesture: 👏 CLAP!")
                    else:
                        gesture = self.detect_gesture(left_inspire_dict, left_hand_data, 'left')
                        print(f"\n🎯 Gesture: {gesture}")
                else:
                    print("\n[LEFT HAND] Not active")
                
                # Display right hand
                if right_active and right_hand_data and right_inspire_output is not None:
                    self.display_hand_status(right_inspire_output, right_inspire_dict, 'right', None, right_hand_data)
                    if clap_detected:
                        print(f"\n🎯 Gesture: 👏 CLAP!")
                    else:
                        gesture = self.detect_gesture(right_inspire_dict, right_hand_data, 'right')
                        print(f"\n🎯 Gesture: {gesture}")
                else:
                    print("\n[RIGHT HAND] Not active")
                
                # Rate limit
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("FINAL OBSERVED RANGES")
            print("="*70)
            for side in ['left', 'right']:
                print(f"\n{side.upper()} HAND:")
                for dof in self.INSPIRE_ORDER + ['thumb_rot']:
                    obs_min = self.observed_min[side].get(dof, '-')
                    obs_max = self.observed_max[side].get(dof, '-')
                    if obs_min != float('inf') and obs_max != float('-inf'):
                        print(f"  {dof:12}: {obs_min:4} - {obs_max:4}")
            print("\n👋 Test ended by user")


def main():
    skip_calib = '--skip-calibration' in sys.argv or '-s' in sys.argv
    detector = HandGestureDetector()
    detector.run(skip_calibration=skip_calib)


if __name__ == "__main__":
    main()
