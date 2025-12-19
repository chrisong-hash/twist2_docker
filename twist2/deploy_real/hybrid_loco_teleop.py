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

# RoboMimic path (mounted in Docker container)
ROBOMIMIC_PATH = "/workspace/RoboMimic_Deploy"


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
        
        # Initialize systems
        print("\n[cyan]Initializing Hybrid Locomotion + Teleoperation...[/cyan]")
        self._setup_locomotion_policy()
        self._setup_teleop_streamer()
        self._setup_retargeting()
        self._setup_mujoco()
        self._setup_redis()
        
        print("\n[green]Systems initialized![/green]")
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
    
    def _setup_retargeting(self):
        """Initialize GMR for upper body retargeting"""
        print("\n[3/5] Setting up motion retargeting...")
        self.retarget = GMR(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            height=self.args.actual_human_height,
            fix_root=True
        )
        print("[green]GMR retargeting ready[/green]")
    
    def _setup_mujoco(self):
        """Setup MuJoCo simulation for preview"""
        print("\n[4/5] Setting up MuJoCo preview...")
        xml_path = ROBOT_XML_DICT["unitree_g1"]
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.data.qpos[7:] = DEFAULT_MIMIC_OBS[self.robot_name][6:]
        self.last_qpos = self.data.qpos.copy()
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
        print("  Left A button (key_one) : Exit program")
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
        smplx_data = self.teleop_streamer.get_smplx_data()
        left_hand = self.teleop_streamer.get_left_hand_data()
        right_hand = self.teleop_streamer.get_right_hand_data()
        controller = self.teleop_streamer.get_controller_data()
        headset = self.teleop_streamer.get_headset_data()
        return smplx_data, left_hand, right_hand, controller, headset
    
    def update_state(self, controller_data):
        """Update state machine based on controller input"""
        if controller_data is None:
            return
        
        # Right A button - toggle state
        right_key_one = controller_data.get("key_one", False)
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
        left_data = self.teleop_streamer.get_left_controller_data()
        if left_data:
            left_key_one = left_data.get("key_one", False)
            if left_key_one:
                self.state = "exit"
                print("[red]→ EXIT requested[/red]")
        
        # Joystick for locomotion
        axis = controller_data.get("axis", [0, 0])
        left_axis = left_data.get("axis", [0, 0]) if left_data else [0, 0]
        
        # Left joystick for movement (from left controller)
        self.vel_cmd[0] = left_axis[1]   # forward/backward
        self.vel_cmd[1] = -left_axis[0]  # strafe
        # Right joystick for rotation
        self.vel_cmd[2] = -axis[0]       # yaw
        
        # Apply deadzone
        for i in range(3):
            if abs(self.vel_cmd[i]) < 0.1:
                self.vel_cmd[i] = 0.0
        
        # Right trigger enables locomotion
        trigger_right = controller_data.get("trigger", 0.0)
        self.locomotion_enabled = trigger_right > 0.5
    
    def process_retargeting(self, smplx_data):
        """Run GMR retargeting on SMPLX data"""
        qpos = self.retarget.retarget(smplx_data)
        
        # If locomotion enabled, replace legs with LocoMode output
        if self.locomotion_enabled:
            # Get current state for locomotion policy
            robot_qj = qpos[7:].copy()
            robot_dqj = np.zeros(29)  # Approximate
            ang_vel = np.zeros(3)
            gravity_ori = np.array([0, 0, -1])
            
            # Compute leg positions from locomotion policy
            loco_action = self.loco_policy.compute(
                robot_qj, robot_dqj, ang_vel, gravity_ori, self.vel_cmd
            )
            
            # Replace leg joints (0-11) with locomotion output
            qpos[7:7+12] = loco_action[:12]
        
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
        
        t_action = int(time.time() * 1000)
        self.redis_pipeline.set("t_action", t_action)
        self.redis_pipeline.execute()
    
    def run(self):
        """Main loop"""
        rate = RateLimiter(frequency=self.args.target_fps)
        self._right_key_was_pressed = False
        
        print(f"\n[yellow]Starting in state: {self.state}[/yellow]")
        print("[cyan]Waiting for Pico VR data...[/cyan]\n")
        
        with mjv.launch_passive(
            model=self.model, 
            data=self.data, 
            show_left_ui=False, 
            show_right_ui=False
        ) as viewer:
            
            while viewer.is_running() and self.state != "exit":
                # Get Pico data
                smplx_data, left_hand, right_hand, controller, headset = self.get_teleop_data()
                
                # Update state machine
                self.update_state(controller)
                
                # Auto-transition from idle to preview when data arrives
                if self.state == "idle" and smplx_data is not None:
                    self.state = "preview"
                    print("[cyan]→ PREVIEW mode: Pico data received![/cyan]")
                
                # Process retargeting if we have data
                if smplx_data is not None:
                    qpos = self.process_retargeting(smplx_data)
                    
                    # Update MuJoCo visualization
                    self.data.qpos[:] = qpos
                    mj.mj_forward(self.model, self.data)
                    
                    # Extract mimic observations
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
                
                # Show locomotion status in title
                if self.locomotion_enabled:
                    vel_str = f"vx={self.vel_cmd[0]:.2f} vy={self.vel_cmd[1]:.2f} vyaw={self.vel_cmd[2]:.2f}"
                    # Note: Can't set title in passive viewer, but we print status
                
                viewer.sync()
                rate.sleep()
        
        print("\n[yellow]Exiting...[/yellow]")


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Locomotion + Teleoperation")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--actual_human_height", type=float, default=1.65)
    parser.add_argument("--redis_ip", type=str, default="localhost")
    parser.add_argument("--target_fps", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hybrid = HybridLocoTeleop(args)
    hybrid.run()
