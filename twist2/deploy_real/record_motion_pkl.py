#!/usr/bin/env python3
"""
Motion Recording Script for Lower Level Policy Training

Records retargeted robot motion from PICO/XRobo teleop and saves to .pkl format.

Usage:
    conda activate gmr
    python record_motion_pkl.py --output_dir ../assets/TWIST2_full/eastworlds --prefix sk_walk

Controls:
    - Right controller A (key_one): Start/pause teleop
    - Left controller Y (key_two): Start/stop recording & save
    - Left controller X (key_one): Exit program
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT
from general_motion_retargeting import XRobotStreamer
from rich import print


class MotionRecorder:
    """Records motion frames and saves to pkl format for training."""
    
    def __init__(self, model, data, output_dir, prefix="sk_walk", fps=30):
        self.model = model
        self.data = data
        self.output_dir = output_dir
        self.prefix = prefix
        self.fps = fps
        self.recording = False
        self.frames = None
        
        # Get link names from mujoco model (excluding world)
        self.link_body_list = [model.body(i).name for i in range(model.nbody) if model.body(i).name != "world"]
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Find next available index
        self.current_index = self._find_next_index()
        print(f"[MotionRecorder] Output dir: {output_dir}")
        print(f"[MotionRecorder] Next file index: {self.current_index:03d}")
        print(f"[MotionRecorder] Link bodies ({len(self.link_body_list)}): {self.link_body_list[:5]}...")
    
    def _find_next_index(self):
        """Find the next available index for the output file."""
        existing_files = [f for f in os.listdir(self.output_dir) 
                        if f.startswith(self.prefix) and f.endswith('.pkl')]
        if not existing_files:
            return 1
        
        indices = []
        for f in existing_files:
            try:
                # Extract index from filename like sk_walk_001.pkl
                idx_str = f.replace(self.prefix + "_", "").replace(".pkl", "")
                indices.append(int(idx_str))
            except ValueError:
                continue
        
        return max(indices) + 1 if indices else 1
    
    def start_recording(self):
        """Start a new recording session."""
        if self.recording:
            print("[MotionRecorder] Already recording!")
            return False
        
        self.frames = {
            'root_pos': [],
            'root_rot': [],
            'dof_pos': [],
            'local_body_pos': [],
        }
        self.recording = True
        self.start_time = time.time()
        print(f"[MotionRecorder] Recording started... (will save as {self.prefix}_{self.current_index:03d}.pkl)")
        return True
    
    def add_frame(self, qpos):
        """Add a frame from retargeted qpos."""
        if not self.recording:
            return
        
        # Extract components from qpos
        root_pos = qpos[0:3].copy()
        # GMR/MuJoCo uses wxyz (scalar-first), but loader expects xyzw, so convert here
        root_rot_wxyz = qpos[3:7].copy()
        root_rot = root_rot_wxyz[[1, 2, 3, 0]]  # wxyz -> xyzw for storage
        dof_pos = qpos[7:].copy()    # 29 DOFs for G1
        
        # Update mujoco state and get body positions
        self.data.qpos[:] = qpos.copy()
        mj.mj_forward(self.model, self.data)
        
        # Get local body positions (relative to root)
        # xpos gives global positions, we need local
        root_pos_global = self.data.xpos[1].copy()  # pelvis position
        local_body_pos = []
        
        for i in range(1, self.model.nbody):  # Skip world (index 0)
            body_pos_global = self.data.xpos[i].copy()
            # Local position relative to root
            local_pos = body_pos_global - root_pos_global
            local_body_pos.append(local_pos)
        
        local_body_pos = np.array(local_body_pos)
        
        self.frames['root_pos'].append(root_pos)
        self.frames['root_rot'].append(root_rot)
        self.frames['dof_pos'].append(dof_pos)
        self.frames['local_body_pos'].append(local_body_pos)
    
    def stop_and_save(self):
        """Stop recording and save to pkl file."""
        if not self.recording:
            print("[MotionRecorder] Not recording!")
            return None
        
        self.recording = False
        duration = time.time() - self.start_time
        num_frames = len(self.frames['root_pos'])
        
        if num_frames < 10:
            print(f"[MotionRecorder] Too few frames ({num_frames}), discarding...")
            self.frames = None
            return None
        
        # Calculate actual FPS
        actual_fps = num_frames / duration if duration > 0 else self.fps
        
        # Build motion data dict
        motion_data = {
            'fps': actual_fps,
            'root_pos': np.array(self.frames['root_pos']),
            'root_rot': np.array(self.frames['root_rot']),
            'dof_pos': np.array(self.frames['dof_pos']),
            'local_body_pos': np.array(self.frames['local_body_pos']),
            'link_body_list': self.link_body_list,
        }
        
        # Save to file
        filename = f"{self.prefix}_{self.current_index:03d}.pkl"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(motion_data, f)
        
        print(f"[MotionRecorder] Saved {filename}")
        print(f"  - Frames: {num_frames}")
        print(f"  - Duration: {duration:.2f}s")
        print(f"  - FPS: {actual_fps:.2f}")
        print(f"  - root_pos shape: {motion_data['root_pos'].shape}")
        print(f"  - root_rot shape: {motion_data['root_rot'].shape}")
        print(f"  - dof_pos shape: {motion_data['dof_pos'].shape}")
        print(f"  - local_body_pos shape: {motion_data['local_body_pos'].shape}")
        
        # Increment index for next recording
        self.current_index += 1
        self.frames = None
        
        return filepath
    
    def get_status(self):
        """Get current recording status."""
        if self.recording:
            num_frames = len(self.frames['root_pos'])
            duration = time.time() - self.start_time
            return f"Recording: {num_frames} frames, {duration:.1f}s"
        return "Idle"


def main():
    parser = argparse.ArgumentParser(description='Record motion to pkl for training')
    parser.add_argument('--robot', default='unitree_g1', choices=['unitree_g1'])
    parser.add_argument('--output_dir', default='../assets/TWIST2_full/eastworlds',
                       help='Output directory for pkl files')
    parser.add_argument('--prefix', default='sk_walk',
                       help='Prefix for output files (e.g., sk_walk -> sk_walk_001.pkl)')
    parser.add_argument('--actual_human_height', type=float, default=1.6,
                       help='Actual human height for retargeting')
    parser.add_argument('--target_fps', type=int, default=30,
                       help='Target recording FPS')
    args = parser.parse_args()
    
    # Setup paths
    xml_file = ROBOT_XML_DICT[args.robot]
    robot_base = ROBOT_BASE_DICT[args.robot]
    
    # Initialize MuJoCo
    model = mj.MjModel.from_xml_path(str(xml_file))
    data = mj.MjData(model)
    
    # Initialize motion recorder
    recorder = MotionRecorder(
        model=model,
        data=data,
        output_dir=args.output_dir,
        prefix=args.prefix,
        fps=args.target_fps
    )
    
    # Initialize XRobot streamer
    streamer = XRobotStreamer()
    print("[Main] XRobot streamer initialized")
    
    # Initialize retargeting
    retarget = GMR(
        src_human="xrobot",
        tgt_robot="unitree_g1",
        actual_human_height=args.actual_human_height,
    )
    print("[Main] GMR retargeting initialized")
    
    # State tracking
    last_qpos = None
    teleop_active = False
    right_key_was_pressed = False
    left_key_two_was_pressed = False
    left_key_one_was_pressed = False
    
    rate = RateLimiter(frequency=args.target_fps, warn=False)
    
    print("\n" + "="*60)
    print("Motion Recording Controls:")
    print("  Right A (key_one): Start/pause teleop")
    print("  Left Y (key_two):  Start/stop recording & save pkl")
    print("  Left X (key_one):  Exit program")
    print("="*60 + "\n")
    
    with mjv.launch_passive(model=model, data=data, 
                           show_left_ui=False, show_right_ui=False) as viewer:
        
        while viewer.is_running():
            # Get teleop data
            smplx_data, left_hand, right_hand, controller_data, headset = streamer.get_current_frame()
            
            # Handle controller input
            if controller_data is not None:
                right_key_current = controller_data.get('RightController', {}).get('key_one', False)
                left_key_two_current = controller_data.get('LeftController', {}).get('key_two', False)
                left_key_one_current = controller_data.get('LeftController', {}).get('key_one', False)
                
                # Right A - toggle teleop
                if right_key_current and not right_key_was_pressed:
                    teleop_active = not teleop_active
                    print(f"[Teleop] {'ACTIVE' if teleop_active else 'PAUSED'}")
                
                # Left Y - toggle recording
                if left_key_two_current and not left_key_two_was_pressed:
                    if recorder.recording:
                        recorder.stop_and_save()
                    else:
                        recorder.start_recording()
                
                # Left X - exit
                if left_key_one_current and not left_key_one_was_pressed:
                    print("[Main] Exit requested")
                    if recorder.recording:
                        recorder.stop_and_save()
                    break
                
                right_key_was_pressed = right_key_current
                left_key_two_was_pressed = left_key_two_current
                left_key_one_was_pressed = left_key_one_current
            
            # Process retargeting if teleop active and data available
            qpos = None
            if teleop_active and smplx_data is not None:
                qpos = retarget.retarget(smplx_data, offset_to_ground=True)
                last_qpos = qpos.copy()
                
                # Record frame
                recorder.add_frame(qpos)
                
                # Update visualization
                data.qpos[:] = qpos.copy()
                mj.mj_forward(model, data)
            
            # Update camera
            if last_qpos is not None:
                robot_base_pos = data.xpos[model.body(robot_base).id]
                viewer.cam.lookat = robot_base_pos
                viewer.cam.distance = 3.0
            
            # Print status
            status = recorder.get_status()
            teleop_status = "TELEOP" if teleop_active else "PAUSED"
            print(f"\r[{teleop_status}] {status}    ", end="", flush=True)
            
            viewer.sync()
            rate.sleep()
    
    print("\n[Main] Finished")


if __name__ == "__main__":
    main()
