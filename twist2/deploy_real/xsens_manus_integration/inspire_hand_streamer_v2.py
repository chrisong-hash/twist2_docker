#!/usr/bin/env python3
"""
Inspire Hand Streamer V2 - Uses calibration-optimized Metacarpal→Distal distance.

Best metric from systematic calibration analysis:
- Metric: dist_3d (3D Euclidean distance)
- Segment pair: Metacarpal → Distal
- Separation score: 4000-5000 (excellent)
- Overlap between open/closed: 0%

Key differences from v1:
- Uses Metacarpal→Distal 3D distance instead of wrist-relative Y position
- Each finger has calibrated open/closed distance thresholds
- More robust and accurate than previous approaches
"""

import socket
import struct
import numpy as np
import threading
import queue
import time
from typing import Dict, Optional, Tuple


class InspireHandController:
    """Controller for a single Inspire Hand via Modbus TCP
    
    Same as v1 - handles low-level communication with Inspire hand.
    """
    
    # Register addresses
    SET_ANGLE_REG = 1486
    GET_ANGLE_REG = 1546
    
    # Angle range: 0=closed, 1000=open
    ANGLE_MIN = 0
    ANGLE_MAX = 1000
    
    # Modbus constants
    PORT = 6000
    UNIT_ID = 1
    TIMEOUT = 2.0
    
    # DOF order: [Little, Ring, Middle, Index, Thumb_Bend, Thumb_Rotate]
    DOF_NAMES = ['Little', 'Ring', 'Middle', 'Index', 'Thumb_Bend', 'Thumb_Rotate']
    NUM_DOFS = 6
    
    MAX_QUEUE_SIZE = 10
    
    def __init__(self, ip, port=PORT, timeout=TIMEOUT, async_mode=True):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.async_mode = async_mode
        self.sock = None
        self.transaction_id = 0
        
        self.command_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.worker_thread = None
        self.running = False
        
        if self.async_mode:
            self._start_worker_thread()
    
    def _start_worker_thread(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_thread, daemon=True)
        self.worker_thread.start()
    
    def _worker_thread(self):
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
                if command is None:
                    break
                func_code, reg_addr, data_to_send = command
                if self._connect():
                    self._send_modbus_command_raw(func_code, reg_addr, data_to_send)
                self.command_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[INSPIRE ASYNC ERROR] Hand {self.ip}: {e}")
                self.disconnect()
    
    def _connect(self):
        try:
            if self.sock:
                self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.ip, self.port))
            return True
        except socket.error as e:
            self.sock = None
            return False
    
    def disconnect(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
    
    def _send_modbus_command(self, function_code, start_address, data):
        if self.async_mode:
            try:
                if self.command_queue.qsize() >= self.MAX_QUEUE_SIZE:
                    try:
                        self.command_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.command_queue.put((function_code, start_address, data))
                return True
            except Exception as e:
                return False
        else:
            if not self.sock or self.sock.fileno() == -1:
                if not self._connect():
                    return None
            return self._send_modbus_command_raw(function_code, start_address, data)
    
    def _send_modbus_command_raw(self, function_code, start_address, data):
        try:
            self.transaction_id += 1
            if self.transaction_id > 65535:
                self.transaction_id = 1
            
            if function_code == 0x03:
                count = data
                pdu = struct.pack(">BHH", function_code, start_address, count)
            elif function_code == 0x06:
                value = data
                pdu = struct.pack(">BHH", function_code, start_address, value)
            elif function_code == 0x10:
                values = data if isinstance(data, (list, np.ndarray)) else [data]
                count = len(values)
                byte_count = count * 2
                pdu = struct.pack(">BHHB", function_code, start_address, count, byte_count)
                for val in values:
                    pdu += struct.pack(">H", int(val))
            else:
                return None
            
            mbap = struct.pack(">HHHB", self.transaction_id, 0, len(pdu) + 1, self.UNIT_ID)
            message = mbap + pdu
            
            self.sock.sendall(message)
            response = self.sock.recv(1024)
            
            if self.sock:
                self.sock.close()
                self.sock = None
            
            if len(response) < 8:
                return None
            
            resp_func_code = response[7]
            if resp_func_code & 0x80:
                return None
            
            return response[7:]
            
        except (socket.timeout, socket.error, Exception):
            self.disconnect()
            return None
    
    def set_angles(self, angles):
        if len(angles) != self.NUM_DOFS:
            return False
        angles_clamped = np.clip(angles, self.ANGLE_MIN, self.ANGLE_MAX).astype(np.int16)
        return self._send_modbus_command(0x10, self.SET_ANGLE_REG, angles_clamped.tolist())
    
    def open_hand(self):
        angles = np.full(self.NUM_DOFS, self.ANGLE_MAX, dtype=np.int16)
        return self.set_angles(angles)
    
    def close_hand(self):
        angles = np.zeros(self.NUM_DOFS, dtype=np.int16)
        angles[5] = 500  # Thumb rotate neutral
        return self.set_angles(angles)
    
    def stop(self):
        if self.async_mode and self.running:
            self.running = False
            self.command_queue.put(None)
            if self.worker_thread:
                self.worker_thread.join(timeout=2.0)
        self.disconnect()


class TeleVisionStyleMapper:
    """Maps Manus glove finger data to Inspire hand angles.
    
    Uses the BEST metric from calibration analysis:
    - Metric: dist_3d (3D Euclidean distance)
    - Segment pair: Metacarpal → Distal (palm bone to fingertip)
    
    This provides 0% overlap between open and closed states with
    separation scores of 4000-5000 (excellent discrimination).
    """
    
    # Finger segment naming in Manus data
    FINGER_SEGMENTS = {
        'thumb': ['FirstMetacarpal', 'FirstProximalPhalange', 'FirstDistalPhalange'],
        'index': ['SecondMetacarpal', 'SecondProximalPhalange', 'SecondMiddlePhalange', 'SecondDistalPhalange'],
        'middle': ['ThirdMetacarpal', 'ThirdProximalPhalange', 'ThirdMiddlePhalange', 'ThirdDistalPhalange'],
        'ring': ['FourthMetacarpal', 'FourthProximalPhalange', 'FourthMiddlePhalange', 'FourthDistalPhalange'],
        'pinky': ['FifthMetacarpal', 'FifthProximalPhalange', 'FifthMiddlePhalange', 'FifthDistalPhalange'],
    }
    
    def __init__(self):
        # Finger to Inspire DOF mapping
        self.finger_to_dof = {
            'pinky': 0,   # Little
            'ring': 1,    # Ring  
            'middle': 2,  # Middle
            'index': 3,   # Index
            # thumb_bend is 4, thumb_rotate is 5
        }
        
        # Calibration: Metacarpal→Distal dist_3d ranges from actual recorded data
        # From calibration_open_palm.json and calibration_closed_fist.json:
        # OPEN:   thumb=0.071, index=0.129, middle=0.137, ring=0.130, pinky=0.112-0.114
        # CLOSED: thumb=0.065-0.070, index=0.074-0.077, middle=0.067-0.073, ring=0.064-0.073, pinky=0.060-0.064
        self.dist_calibration = {
            'thumb':  {'open': 0.071, 'closed': 0.065},  # Very small range!
            'index':  {'open': 0.129, 'closed': 0.075},
            'middle': {'open': 0.137, 'closed': 0.070},
            'ring':   {'open': 0.131, 'closed': 0.068},
            'pinky':  {'open': 0.113, 'closed': 0.062},
        }
        
        # Smoothing filter
        self.alpha = 0.4  # Higher = more responsive, lower = smoother
        self.prev_angles = {'left': None, 'right': None}
    
    def calibrate_open(self, finger_data):
        """Calibrate open hand distances from current Manus data."""
        print("[Mapper] Calibrating OPEN hand (Metacarpal→Distal dist_3d)...")
        
        for hand in ['left', 'right']:
            hand_data = finger_data.get(hand, {})
            print(f"  {hand.upper()} hand:")
            
            for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                dist = self._get_metacarpal_to_distal_dist(hand_data, finger, hand)
                if dist is not None:
                    self.dist_calibration[finger]['open'] = dist
                    print(f"    {finger}: {dist:.4f}m")
    
    def calibrate_closed(self, finger_data):
        """Calibrate closed hand distances from current Manus data."""
        print("[Mapper] Calibrating CLOSED hand (Metacarpal→Distal dist_3d)...")
        
        for hand in ['left', 'right']:
            hand_data = finger_data.get(hand, {})
            print(f"  {hand.upper()} hand:")
            
            for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                dist = self._get_metacarpal_to_distal_dist(hand_data, finger, hand)
                if dist is not None:
                    self.dist_calibration[finger]['closed'] = dist
                    print(f"    {finger}: {dist:.4f}m")
    
    def _get_segment_position(self, hand_data: Dict, finger: str, seg_type: str, hand: str) -> Optional[np.ndarray]:
        """Get position of a specific segment.
        
        Args:
            hand_data: Dict with finger segment lists
            finger: 'thumb', 'index', etc.
            seg_type: 'Metacarpal', 'Proximal', 'Middle', 'Distal'
            hand: 'left' or 'right'
        
        Returns:
            Position array or None
        """
        segs = hand_data.get(finger, [])
        if not segs:
            return None
        
        # Map segment type to index in finger segments
        finger_segs = self.FINGER_SEGMENTS[finger]
        
        if seg_type == 'Metacarpal':
            target_suffix = finger_segs[0]  # e.g., 'SecondMetacarpal'
        elif seg_type == 'Proximal':
            target_suffix = finger_segs[1]  # e.g., 'SecondProximalPhalange'
        elif seg_type == 'Middle':
            if len(finger_segs) >= 4:
                target_suffix = finger_segs[2]
            else:
                return None  # Thumb doesn't have middle
        elif seg_type == 'Distal':
            target_suffix = finger_segs[-1]  # Last segment
        else:
            return None
        
        # Find segment by name
        prefix = 'Left' if hand == 'left' else 'Right'
        target_name = f"{prefix}{target_suffix}"
        
        for seg in segs:
            if seg.get('name') == target_name:
                return np.array(seg['position'])
        
        # Fallback: for Distal, use segment with max Z (farthest from palm)
        if seg_type == 'Distal':
            best_seg = None
            max_z = float('-inf')
            for seg in segs:
                if 'position' in seg:
                    z = seg['position'][2]
                    if z > max_z:
                        max_z = z
                        best_seg = seg
            if best_seg:
                return np.array(best_seg['position'])
        
        return None
    
    def _get_metacarpal_to_distal_dist(self, hand_data: Dict, finger: str, hand: str) -> Optional[float]:
        """Get 3D distance from Metacarpal to Distal segment.
        
        This is the BEST metric from calibration analysis:
        - Separation score: 4000-5000
        - Overlap: 0%
        """
        meta_pos = self._get_segment_position(hand_data, finger, 'Metacarpal', hand)
        distal_pos = self._get_segment_position(hand_data, finger, 'Distal', hand)
        
        if meta_pos is None or distal_pos is None:
            return None
        
        return float(np.linalg.norm(distal_pos - meta_pos))
    
    def _compute_finger_curl(self, hand_data: Dict, finger: str, hand: str) -> float:
        """Compute finger curl using Metacarpal→Distal dist_3d.
        
        This metric was found to have the BEST separation between open and closed:
        - Right ring: 5010 separation score
        - Right middle: 4299 separation score
        - Right index: 4289 separation score
        - 0% overlap between open and closed states
        
        Args:
            hand_data: Dict with finger segment lists
            finger: Finger name
            hand: 'left' or 'right'
            
        Returns:
            Curl value 0.0 (open) to 1.0 (closed)
        """
        dist = self._get_metacarpal_to_distal_dist(hand_data, finger, hand)
        
        if dist is None:
            return 0.5  # Default to mid-position
        
        # Get calibrated ranges
        calib = self.dist_calibration.get(finger, {'open': 0.13, 'closed': 0.07})
        dist_open = calib['open']
        dist_closed = calib['closed']
        
        # Map: larger distance = open (0 curl), smaller = closed (1 curl)
        if dist_open == dist_closed:
            return 0.5
        
        curl = (dist_open - dist) / (dist_open - dist_closed)
        
        return np.clip(curl, 0.0, 1.0)
    
    def _compute_thumb_bend(self, hand_data: Dict, hand: str) -> float:
        """Compute thumb bend using Carpus→Distal dist_3d.
        
        Thumb uses a different segment pair than other fingers because
        Metacarpal→Distal barely changes for thumb (only 1mm).
        Carpus→Distal has much better separation (819 vs nearly 0).
        """
        # Get Carpus (wrist) position - use the wrist data
        wrist = hand_data.get('wrist')
        if not wrist or 'position' not in wrist:
            return 0.5
        carpus_pos = np.array(wrist['position'])
        
        # Get thumb distal position
        distal_pos = self._get_segment_position(hand_data, 'thumb', 'Distal', hand)
        if distal_pos is None:
            return 0.5
        
        dist = float(np.linalg.norm(distal_pos - carpus_pos))
        
        # Calibrated from analysis:
        # Open: 0.111m, Closed: 0.091-0.094m
        dist_open = 0.111
        dist_closed = 0.092
        
        if dist_open == dist_closed:
            return 0.5
        
        curl = (dist_open - dist) / (dist_open - dist_closed)
        return np.clip(curl, 0.0, 1.0)
    
    def _compute_thumb_rotation(self, hand_data: Dict, hand: str) -> float:
        """Compute thumb rotation (abduction/adduction).
        
        Based on distance between thumb distal and ring distal.
        - Close together = opposition (rotated into palm) = 0
        - Far apart = abducted (rotated away) = 1000
        """
        thumb_pos = self._get_segment_position(hand_data, 'thumb', 'Distal', hand)
        ring_pos = self._get_segment_position(hand_data, 'ring', 'Distal', hand)
        
        if thumb_pos is None or ring_pos is None:
            return 0.5  # Default
        
        # 3D distance between thumb and ring tips
        dist = np.linalg.norm(thumb_pos - ring_pos)
        
        # Calibrated ranges
        # Opposition (thumb touching ring): ~0.03-0.05m
        # Abducted (thumb away): ~0.10-0.15m
        dist_opposition = 0.04
        dist_abducted = 0.12
        
        rotation = (dist - dist_opposition) / (dist_abducted - dist_opposition)
        
        return np.clip(rotation, 0.0, 1.0)
    
    def map_to_inspire(self, finger_data: Dict, hand: str) -> np.ndarray:
        """Map Manus finger data to Inspire hand angles.
        
        Uses Metacarpal→Distal dist_3d metric (best from calibration analysis):
        - 0% overlap between open/closed states
        - Separation scores of 4000-5000
        
        Args:
            finger_data: Output from get_finger_data_by_name()
            hand: 'left' or 'right'
            
        Returns:
            Array of 6 angles [Little, Ring, Middle, Index, Thumb_Bend, Thumb_Rotate]
            Range: 0 (closed) to 1000 (open)
        """
        angles = np.full(6, 500, dtype=np.int16)  # Default to mid-position
        
        hand_data = finger_data.get(hand, {})
        if not hand_data:
            return angles
        
        # Compute finger curls using Metacarpal→Distal dist_3d
        for finger in ['index', 'middle', 'ring', 'pinky']:
            curl = self._compute_finger_curl(hand_data, finger, hand)
            dof_idx = self.finger_to_dof[finger]
            # Inspire: 0=closed, 1000=open → invert curl
            angles[dof_idx] = int((1.0 - curl) * 1000)
        
        # Compute thumb bend
        thumb_bend = self._compute_thumb_bend(hand_data, hand)
        angles[4] = int((1.0 - thumb_bend) * 1000)
        
        # Compute thumb rotation
        thumb_rot = self._compute_thumb_rotation(hand_data, hand)
        angles[5] = int(thumb_rot * 1000)
        
        # Apply smoothing
        if self.prev_angles[hand] is not None:
            angles = (self.alpha * angles + 
                     (1 - self.alpha) * self.prev_angles[hand]).astype(np.int16)
        self.prev_angles[hand] = angles.copy()
        
        return angles
    
    def get_finger_distances(self, finger_data: Dict, hand: str) -> Dict[str, float]:
        """Get Metacarpal→Distal distances for debugging.
        
        Returns dict with distances for each finger.
        """
        hand_data = finger_data.get(hand, {})
        
        distances = {}
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            dist = self._get_metacarpal_to_distal_dist(hand_data, finger, hand)
            if dist is not None:
                distances[finger] = dist
        
        return distances


class InspireHandStreamerV2:
    """Streams Manus glove data to Inspire hands using TeleVision-style mapping.
    
    Uses fingertip positions instead of quaternion-based angle extraction.
    """
    
    def __init__(self, 
                 left_hand_ip="192.168.123.210",
                 right_hand_ip="192.168.123.211",
                 enable_left=True,
                 enable_right=True,
                 async_mode=True):
        """Initialize the streamer."""
        self.enable_left = enable_left
        self.enable_right = enable_right
        
        # TeleVision-style mapper
        self.mapper = TeleVisionStyleMapper()
        
        # Initialize Inspire hand controllers
        self.left_hand = None
        self.right_hand = None
        
        if enable_left:
            try:
                self.left_hand = InspireHandController(left_hand_ip, async_mode=async_mode)
                print(f"[InspireStreamerV2] Left hand connected: {left_hand_ip}")
            except Exception as e:
                print(f"[InspireStreamerV2] Left hand failed: {e}")
        
        if enable_right:
            try:
                self.right_hand = InspireHandController(right_hand_ip, async_mode=async_mode)
                print(f"[InspireStreamerV2] Right hand connected: {right_hand_ip}")
            except Exception as e:
                print(f"[InspireStreamerV2] Right hand failed: {e}")
        
        # State
        self._lock = threading.Lock()
        self._latest_left_angles = None
        self._latest_right_angles = None
    
    def calibrate_open(self, finger_data):
        """Calibrate with open hand pose."""
        self.mapper.calibrate_open(finger_data)
    
    def calibrate_closed(self, finger_data):
        """Calibrate with closed hand pose."""
        self.mapper.calibrate_closed(finger_data)
    
    def process_finger_data(self, finger_data) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process finger data and send to Inspire hands.
        
        Args:
            finger_data: Output from get_finger_data_by_name()
            
        Returns:
            Tuple of (left_angles, right_angles)
        """
        left_angles = None
        right_angles = None
        
        if finger_data is None:
            return None, None
        
        # Map left hand
        if self.enable_left and self.left_hand:
            left_angles = self.mapper.map_to_inspire(finger_data, 'left')
            self.left_hand.set_angles(left_angles)
            with self._lock:
                self._latest_left_angles = left_angles.copy()
        
        # Map right hand
        if self.enable_right and self.right_hand:
            right_angles = self.mapper.map_to_inspire(finger_data, 'right')
            self.right_hand.set_angles(right_angles)
            with self._lock:
                self._latest_right_angles = right_angles.copy()
        
        return left_angles, right_angles
    
    def get_latest_angles(self):
        """Get the latest angles sent to each hand."""
        with self._lock:
            return (
                self._latest_left_angles.copy() if self._latest_left_angles is not None else None,
                self._latest_right_angles.copy() if self._latest_right_angles is not None else None
            )
    
    def get_finger_distances(self, finger_data, hand):
        """Get Metacarpal→Distal distances for debugging."""
        return self.mapper.get_finger_distances(finger_data, hand)
    
    def open_hands(self):
        """Open both hands."""
        if self.left_hand:
            self.left_hand.open_hand()
        if self.right_hand:
            self.right_hand.open_hand()
    
    def close_hands(self):
        """Close both hands."""
        if self.left_hand:
            self.left_hand.close_hand()
        if self.right_hand:
            self.right_hand.close_hand()
    
    def stop(self):
        """Stop the streamer and clean up."""
        if self.left_hand:
            self.left_hand.stop()
        if self.right_hand:
            self.right_hand.stop()
        print("[InspireStreamerV2] Stopped")


def main():
    """Test the TeleVision-style Inspire hand streamer."""
    import argparse
    from udp_listener import decode_xsens_packet, get_finger_data_by_name
    
    parser = argparse.ArgumentParser(description="Inspire Hand Streamer V2 - TeleVision-style")
    parser.add_argument("--left_ip", type=str, default="192.168.123.210")
    parser.add_argument("--right_ip", type=str, default="192.168.123.211")
    parser.add_argument("--port", type=int, default=9763)
    parser.add_argument("--no_left", action="store_true")
    parser.add_argument("--no_right", action="store_true")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration")
    parser.add_argument("--dry_run", action="store_true", help="Don't send to hands")
    parser.add_argument("--debug", action="store_true", help="Print fingertip positions")
    args = parser.parse_args()
    
    print("=" * 60)
    print("INSPIRE HAND STREAMER V2 - TeleVision-style Mapping")
    print("=" * 60)
    print("Uses fingertip positions in wrist-relative coordinates")
    print("Similar to TeleVision/Apple Vision Pro approach")
    print()
    
    # Initialize streamer
    if args.dry_run:
        print("[DRY RUN] Not connecting to Inspire hands")
        streamer = InspireHandStreamerV2(
            left_hand_ip=args.left_ip,
            right_hand_ip=args.right_ip,
            enable_left=False,
            enable_right=False
        )
    else:
        streamer = InspireHandStreamerV2(
            left_hand_ip=args.left_ip,
            right_hand_ip=args.right_ip,
            enable_left=not args.no_left,
            enable_right=not args.no_right
        )
    
    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', args.port))
    sock.settimeout(1.0)
    
    print(f"Listening for Xsens/Manus data on port {args.port}...")
    print("Press Ctrl+C to stop")
    print()
    
    # Calibration
    if args.calibrate:
        print("CALIBRATION MODE")
        print("-" * 40)
        
        print("Step 1: Hold hands FULLY OPEN and press Enter...")
        input()
        try:
            data, _ = sock.recvfrom(4096)
            decoded = decode_xsens_packet(data)
            if decoded:
                finger_data = get_finger_data_by_name(decoded)
                streamer.calibrate_open(finger_data)
        except socket.timeout:
            print("No data received")
        
        print()
        print("Step 2: Make a FIST (curl all fingers) and press Enter...")
        input()
        try:
            data, _ = sock.recvfrom(4096)
            decoded = decode_xsens_packet(data)
            if decoded:
                finger_data = get_finger_data_by_name(decoded)
                streamer.calibrate_closed(finger_data)
        except socket.timeout:
            print("No data received")
        
        print()
        print("Calibration complete!")
        print("-" * 40)
        print()
    
    # Main loop
    frame_count = 0
    try:
        while True:
            try:
                data, _ = sock.recvfrom(4096)
                decoded = decode_xsens_packet(data)
                
                if decoded:
                    finger_data = get_finger_data_by_name(decoded)
                    
                    # Debug: print Metacarpal→Distal distances
                    if args.debug and frame_count % 60 == 0:
                        print("\n--- Metacarpal→Distal distances (meters) ---")
                        for hand in ['left', 'right']:
                            dists = streamer.get_finger_distances(finger_data, hand)
                            print(f"{hand.upper()}:")
                            for finger, d in dists.items():
                                calib = streamer.mapper.dist_calibration.get(finger, {})
                                print(f"  {finger}: {d:.4f}m (open={calib.get('open', 0):.3f}, closed={calib.get('closed', 0):.3f})")
                    
                    # Process and send
                    left_angles, right_angles = streamer.process_finger_data(finger_data)
                    
                    # Print status
                    if frame_count % 30 == 0:
                        print(f"\rFrame {frame_count}: ", end="")
                        if left_angles is not None:
                            print(f"L=[{','.join(f'{a:4d}' for a in left_angles)}] ", end="")
                        if right_angles is not None:
                            print(f"R=[{','.join(f'{a:4d}' for a in right_angles)}]", end="")
                        print("   ", end="", flush=True)
                    
                    frame_count += 1
                    
            except socket.timeout:
                continue
                
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        streamer.open_hands()
        time.sleep(0.3)
        streamer.stop()
        sock.close()
        print("Done.")


if __name__ == "__main__":
    main()

