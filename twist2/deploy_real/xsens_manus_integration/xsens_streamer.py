#!/usr/bin/env python3
"""
XsensStreamer - Streams Xsens UDP data in BVH-compatible format for GMR.

Matches the data format produced by GMR_xsens BVH parser so we can use
the same IK config (xsens_bvh_to_g1.json) and offsets.json calibration.
"""

import socket
import threading
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

from udp_listener import decode_xsens_packet


class XsensStreamer:
    """
    Streams Xsens MVN UDP data in BVH-compatible format for GMR retargeting.
    
    Key insight: The BVH parser applies axis reordering (zxy) and Euler offsets.
    We need to match that output format for the IK config to work correctly.
    """
    
    # Xsens MVN UDP segment names → BVH joint names
    # This allows us to use xsens_bvh_to_g1.json IK config
    # Xsens MVN UDP segment names → BVH joint names
    # This allows us to use xsens_bvh_to_g1.json IK config
    XSENS_UDP_TO_BVH = {
        "Pelvis": "Hips",
        "L5": "Chest",       # Spine segments
        "L3": "Chest2",
        "T12": "Chest3",
        "T8": "Chest4",
        "Neck": "Neck",
        "Head": "Head",
        "RightShoulder": "RightCollar",
        "RightUpperArm": "RightShoulder",
        "RightForeArm": "RightElbow",
        "RightHand": "RightWrist",
        "LeftShoulder": "LeftCollar",
        "LeftUpperArm": "LeftShoulder",
        "LeftForeArm": "LeftElbow",
        "LeftHand": "LeftWrist",
        "RightUpperLeg": "RightHip",
        "RightLowerLeg": "RightKnee",
        "RightFoot": "RightAnkle",
        "RightToe": "RightToe",
        "LeftUpperLeg": "LeftHip",
        "LeftLowerLeg": "LeftKnee",
        "LeftFoot": "LeftAnkle",
        "LeftToe": "LeftToe",
    }
    
    # Bone-specific rotation corrections
    # Xsens arm bones have local +X pointing UP, but GMR expects +X along the bone
    # This requires a -90° rotation around Y to fix
    ARM_BONE_CORRECTION_JOINTS = {
        "LeftElbow", "LeftWrist", "RightElbow", "RightWrist",
        "LeftShoulder", "RightShoulder",  # Upper arm joints
    }
    
    def __init__(self, ip="0.0.0.0", port=9763, offsets_path=None):
        """
        Args:
            ip: IP address to bind UDP socket
            port: UDP port for Xsens data (default 9763)
            offsets_path: Path to offsets.json calibration file
        """
        self._ip = ip
        self._port = port
        self._sock = None
        self._running = False
        self._latest_data = None
        self._latest_raw = None  # Raw decoded data for finger access
        self._lock = threading.Lock()
        self._thread = None
        
        # Load offsets calibration if provided
        self._offsets = {}
        if offsets_path and os.path.exists(offsets_path):
            self._load_offsets(offsets_path)
            print(f"[XsensStreamer] Loaded offsets from {offsets_path}")
        else:
            print("[XsensStreamer] No offsets.json - using identity offsets")
            
        # Coordinate transform: Xsens (X-right, Y-forward, Z-up) → GMR (X-forward, Y-left, Z-up)
        # Position: (x, y, z) → (y, -x, z)
        # This is a -90° rotation around Z-axis
        self._frame_change = R.from_euler('z', -90, degrees=True)
        
        # Arm bone correction: DISABLED for now to see baseline behavior
        # None = no correction applied
        self._arm_bone_correction = None
        
    def _load_offsets(self, path):
        """Load Euler angle offsets from offsets.json."""
        try:
            with open(path, 'r') as f:
                raw_offsets = json.load(f)
            # Convert to numpy arrays [X, Y, Z] in degrees
            for joint_name, offset_dict in raw_offsets.items():
                self._offsets[joint_name] = np.array([
                    offset_dict.get('X', 0.0),
                    offset_dict.get('Y', 0.0),
                    offset_dict.get('Z', 0.0)
                ])
        except Exception as e:
            print(f"[XsensStreamer] Error loading offsets: {e}")
            
    def _udp_listener_thread(self):
        """Background thread to receive UDP packets."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self._ip, self._port))
        self._sock.settimeout(1.0)
        
        print(f"[XsensStreamer] Listening on {self._ip}:{self._port}")
        
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
                decoded = decode_xsens_packet(data)
                if decoded:
                    with self._lock:
                        self._latest_data = decoded
                        self._latest_raw = decoded  # Keep raw data for finger access
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[XsensStreamer] Error: {e}")
                
    def start(self):
        """Start the UDP listener thread."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._udp_listener_thread, daemon=True)
            self._thread.start()
            
    def stop(self):
        """Stop the UDP listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            self._sock.close()
        print("[XsensStreamer] Stopped")
        
    def _transform_orientation(self, quat_wxyz, pelvis_rot=None, is_pelvis=False, bvh_joint_name=None):
        """Transform quaternion from Xsens global to GMR frame.
        
        Key insight: BVH loader outputs GLOBAL orientations (via quat_fk).
        We must also output GLOBAL orientations, not body-relative.
        
        All joints: Apply room-to-world coordinate transform (same for all joints)
        """
        w, x, y, z = quat_wxyz
        rot_xsens = R.from_quat([x, y, z, w])  # scipy uses xyzw
        
        # All joints: Transform from Xsens room frame to GMR world frame
        # Pre-multiply by frame change (not similarity transform)
        rot_gmr = self._frame_change * rot_xsens
        
        quat_xyzw = rot_gmr.as_quat()
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # wxyz
    
    def _apply_offset(self, quat_wxyz, bvh_joint_name):
        """Apply calibration offset to orientation."""
        if bvh_joint_name in self._offsets:
            offset_xyz = self._offsets[bvh_joint_name]
            if np.any(offset_xyz != 0):
                w, x, y, z = quat_wxyz
                rot = R.from_quat([x, y, z, w])
                offset_rot = R.from_euler('xyz', offset_xyz, degrees=True)
                rot = rot * offset_rot  # Post-multiply offset
                quat_xyzw = rot.as_quat()
                return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return quat_wxyz
    
    def _transform_position(self, pos_xyz, pelvis_pos=None, pelvis_rot=None, is_pelvis=False):
        """Transform position from Xsens global to GMR frame.
        
        Key insight: BVH loader outputs GLOBAL positions (via quat_fk).
        We must also output GLOBAL positions, not body-relative.
        
        All joints: Transform room coordinates to GMR world (same transform for all)
        """
        pos = np.array(pos_xyz)
        
        # All joints: Transform from Xsens room frame to GMR world frame
        # Xsens room: X-right, Y-forward, Z-up
        # GMR world: X-forward, Y-left, Z-up
        # This matches BVH loader which outputs GLOBAL positions
        x, y, z = pos
        return np.array([y, -x, z])
        
    def get_processed_body_data(self):
        """
        Get body data in BVH-compatible format for GMR.
        
        Matches the output of load_xsens_file() so we can use xsens_bvh_to_g1.json.
        
        Key: Xsens UDP gives GLOBAL positions. We make them pelvis-relative
        so left/right are correct regardless of which way the person faces.
        
        Returns:
            dict: {bvh_joint_name: (position_array, quaternion_wxyz_array)} or None
        """
        with self._lock:
            if self._latest_data is None:
                return None
            xsens_data = self._latest_data
            self._latest_data = None  # Consume data
            
        if not xsens_data.get('segments'):
            return None
            
        # Build segment lookup by UDP name
        segments_dict = {seg['name']: seg for seg in xsens_data['segments']}
        
        # First, get pelvis pose for body-relative transform
        pelvis_pos = None
        pelvis_rot = None
        if 'Pelvis' in segments_dict:
            pelvis_seg = segments_dict['Pelvis']
            pelvis_pos = np.array(pelvis_seg['position'])
            quat_raw = np.array(pelvis_seg['quaternion'])
            quat_norm = np.linalg.norm(quat_raw)
            if quat_norm > 0.001:
                quat_raw = quat_raw / quat_norm
                # Convert to scipy Rotation (xyzw format)
                pelvis_rot = R.from_quat([quat_raw[1], quat_raw[2], quat_raw[3], quat_raw[0]])
        
        # Convert to BVH-compatible format
        body_pose_dict = {}
        
        for udp_name, bvh_name in self.XSENS_UDP_TO_BVH.items():
            seg = segments_dict.get(udp_name)
            if seg:
                # Get raw position and quaternion from UDP
                pos_raw = np.array(seg['position'])
                quat_raw = np.array(seg['quaternion'])  # wxyz
                
                # Normalize quaternion
                quat_norm = np.linalg.norm(quat_raw)
                if quat_norm > 0.001:
                    quat_raw = quat_raw / quat_norm
                else:
                    quat_raw = np.array([1.0, 0.0, 0.0, 0.0])
                
                # Check if this is the pelvis joint
                is_pelvis = (udp_name == "Pelvis")
                
                # Transform position: Global Xsens → GMR (body-relative for non-pelvis)
                position = self._transform_position(pos_raw, pelvis_pos, pelvis_rot, is_pelvis)
                
                # Transform orientation: Global Xsens → GMR (body-relative for non-pelvis)
                orientation = self._transform_orientation(quat_raw, pelvis_rot, is_pelvis, bvh_name)
                
                # Apply calibration offset
                orientation = self._apply_offset(orientation, bvh_name)

                # Store as tuple (position, orientation) matching load_xsens_file output
                body_pose_dict[bvh_name] = (position, orientation)
        
        # Create FootMod entries (matching load_xsens_file)
        if "LeftAnkle" in body_pose_dict and "LeftToe" in body_pose_dict:
            body_pose_dict["LeftFootMod"] = (
                body_pose_dict["LeftAnkle"][0],  # Position from ankle
                body_pose_dict["LeftAnkle"][1]   # Orientation from ankle
            )
        if "RightAnkle" in body_pose_dict and "RightToe" in body_pose_dict:
            body_pose_dict["RightFootMod"] = (
                body_pose_dict["RightAnkle"][0],
                body_pose_dict["RightAnkle"][1]
            )
                
        return body_pose_dict
    
    def get_current_frame(self):
        """
        Compatibility method matching XRobotStreamer interface.
        
        Returns:
            tuple: (body_data, left_hand, right_hand, controller, headset)
        """
        body_data = self.get_processed_body_data()
        return body_data, None, None, None, None
    
    def get_raw_decoded(self):
        """
        Get the raw decoded Xsens packet data (for finger tracking).
        
        This does NOT consume the data, allowing multiple reads of the same frame.
        Use get_finger_data_by_name() from udp_listener to process finger segments.
        
        Returns:
            dict: Raw decoded packet with 'segments', 'finger_segments', etc. or None
        """
        with self._lock:
            return self._latest_raw


if __name__ == "__main__":
    # Test the streamer
    import time
    
    print("Testing XsensStreamer with BVH-compatible output...")
    
    # Try to load offsets from GMR_xsens
    offsets_path = "/home/robo/CodeSpace/xsens_teleop/GMR_xsens/offsets.json"
    
    streamer = XsensStreamer(port=9763, offsets_path=offsets_path)
    streamer.start()
    
    try:
        for i in range(50):
            data = streamer.get_processed_body_data()
            if data:
                print(f"\nFrame {i}:")
                for name in ["Hips", "RightHip", "LeftHip", "RightShoulder", "LeftShoulder"]:
                    if name in data:
                        pos, quat = data[name]
                        print(f"  {name}: pos={np.round(pos, 2)}, quat={np.round(quat, 3)}")
            else:
                print(".", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
