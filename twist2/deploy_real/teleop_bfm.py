#!/usr/bin/env python3
"""
BFM-Zero Teleop Server
======================
Reads VR tracking from PICO via XRobotStreamer, uses GMR for retargeting,
and publishes full pose data in BFM-Zero format to Redis.

Usage:
    python teleop_bfm.py [--vis] [--redis_ip localhost]
    
This script publishes to:
    - bfm_motion_{robot}  : Full BFM pose (root_pos, root_quat, dof_pos)
    - action_body_{robot} : Original mimic_obs format (for compatibility)
"""

import argparse
import json
import time
import numpy as np
import redis
import mujoco
from mujoco.viewer import launch_passive
from rich import print
import sys
import os

# Add GMR path
GMR_PATH = "/workspace/GMR/general_motion_retargeting"
if GMR_PATH not in sys.path:
    sys.path.insert(0, "/workspace/GMR")

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import XRobotStreamer, ROBOT_XML_DICT
from data_utils.params import DEFAULT_MIMIC_OBS
from data_utils.rot_utils import euler_from_quaternion_np, quat_rotate_inverse_np
from bfm_logger import BFMLogger


class BFMTeleopServer:
    def __init__(
        self,
        robot: str = "unitree_g1_with_hands",
        redis_ip: str = "localhost",
        vis: bool = True,
        control_dt: float = 0.02,
    ):
        self.robot = robot
        self.control_dt = control_dt
        self.vis = vis
        
        print(f"[bold cyan]BFM-Zero Teleop Server[/bold cyan]")
        print(f"  Robot: {robot}")
        print(f"  Redis: {redis_ip}")
        print(f"  Visualization: {vis}")
        print()
        
        # Connect to Redis
        self.redis_client = redis.Redis(host=redis_ip, port=6379, db=0)
        self.redis_client.ping()
        print("[green]✓ Redis connected[/green]")
        
        # Initialize logger
        self.logger = BFMLogger("teleop", "real", log_dir="/workspace/twist2/logs")
        self.logger.log_settings({
            "robot": robot,
            "redis_ip": redis_ip,
            "control_dt": control_dt,
            "vis": vis,
        })
        
        # Initialize XRobotStreamer (PICO connection)
        print("[yellow]Initializing XRobotStreamer...[/yellow]")
        try:
            self.streamer = XRobotStreamer()
            print("[green]✓ XRobotStreamer initialized - waiting for PICO data[/green]")
        except Exception as e:
            print(f"[red]ERROR: Failed to initialize XRobotStreamer: {e}[/red]")
            print("[yellow]Make sure PICO is connected and streaming[/yellow]")
            raise
        
        # Initialize GMR (motion retargeting)
        print("[yellow]Initializing GMR...[/yellow]")
        import io
        import sys
        # Suppress GMR verbose output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self.gmr = GMR(
                src_human="xrobot",
                tgt_robot="unitree_g1",
                actual_human_height=1.75,  # Default human height, can be calibrated
            )
        finally:
            sys.stdout = old_stdout
        print("[green]✓ GMR initialized[/green]")
        
        # Setup visualization
        self.viewer = None
        if vis:
            xml_path = ROBOT_XML_DICT.get("unitree_g1")
            if xml_path and os.path.exists(str(xml_path)):
                self.mjm = mujoco.MjModel.from_xml_path(str(xml_path))
                self.mjd = mujoco.MjData(self.mjm)
                self.viewer = launch_passive(self.mjm, self.mjd, show_left_ui=False, show_right_ui=False)
                print("[green]✓ MuJoCo viewer launched[/green]")
            else:
                print(f"[yellow]Warning: XML not found at {xml_path}, visualization disabled[/yellow]")
                self.vis = False
        
        # State
        self.frame_count = 0
        self.last_root_pos = np.zeros(3)
        self.last_root_quat = np.array([1, 0, 0, 0])  # w, x, y, z
        self.last_dof_pos = np.zeros(29)
        
        print()
        print("[bold green]Ready! Waiting for PICO tracking data...[/bold green]")
        print("[dim]Press Ctrl+C to stop[/dim]")
        print()
    
    def get_teleop_data(self):
        """Get current VR tracking data from PICO using XRobotStreamer."""
        try:
            # get_current_frame returns: (body_pose_dict, left_hand, right_hand, controller, headset)
            return self.streamer.get_current_frame()
        except Exception as e:
            return None, None, None, None, None
    
    def retarget_to_robot(self, body_pose_dict):
        """Use GMR to convert PICO body tracking to full robot pose."""
        if body_pose_dict is None:
            return None, None, None
        
        try:
            # Run GMR retargeting - returns qpos array: [root_pos(3), root_quat(4), dof_pos(N)]
            qpos = self.gmr.retarget(body_pose_dict, offset_to_ground=True)
            
            if qpos is not None and len(qpos) >= 7:
                root_pos = qpos[:3]
                root_quat = qpos[3:7]  # MuJoCo uses [w,x,y,z]
                dof_pos = qpos[7:7+29] if len(qpos) >= 36 else qpos[7:]
                return root_pos, root_quat, dof_pos
            
        except Exception as e:
            if not hasattr(self, '_retarget_error_printed'):
                print(f"[yellow]Retarget warning: {e}[/yellow]")
                self._retarget_error_printed = True
        
        return None, None, None
    
    def build_mimic_obs(self, root_pos, root_quat, dof_pos):
        """Build mimic_obs format for TWIST2 compatibility."""
        from scipy.spatial.transform import Rotation as R
        
        # Compute velocities (simple finite difference)
        root_vel = (root_pos - self.last_root_pos) / self.control_dt
        
        # Root frame velocities - need to reshape for batch function
        root_quat_batch = root_quat.reshape(1, -1)
        root_vel_batch = root_vel.reshape(1, -1)
        root_vel_local = quat_rotate_inverse_np(root_quat_batch, root_vel_batch).flatten()
        
        # Euler angles using scipy (handles single quaternion)
        # root_quat is [w, x, y, z] for MuJoCo, scipy uses [x, y, z, w]
        quat_xyzw = np.array([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        euler = R.from_quat(quat_xyzw).as_euler('xyz')
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        # Angular velocity (simplified - would need proper computation)
        yaw_vel = 0.0
        
        # mimic_obs: [vel_xy(2), height(1), roll(1), pitch(1), yaw_vel(1), dof_pos(29)]
        mimic_obs = np.concatenate([
            root_vel_local[:2],  # vel_xy
            [root_pos[2]],       # height
            [roll, pitch],       # roll, pitch
            [yaw_vel],           # yaw_vel
            dof_pos,             # dof_pos (29)
        ])
        
        return mimic_obs
    
    def publish_to_redis(self, root_pos, root_quat, dof_pos, root_vel, root_ang_vel):
        """Publish pose data to Redis in both BFM and mimic_obs formats."""
        # BFM format (full pose)
        bfm_motion = {
            "root_pos": root_pos.tolist(),
            "root_quat": root_quat.tolist(),  # [w, x, y, z]
            "dof_pos": dof_pos.tolist(),
            "root_vel": root_vel.tolist(),
            "root_ang_vel": root_ang_vel.tolist(),
            "fps": 1.0 / self.control_dt,
            "frame": self.frame_count,
        }
        self.redis_client.set(f"bfm_motion_{self.robot}", json.dumps(bfm_motion))
        
        # LOG: Output data
        self.logger.log_motion_in(
            frame=self.frame_count,
            root_pos=root_pos,
            root_quat=root_quat,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
        )
        self.logger.log("REDIS_OUT", key=f"bfm_motion_{self.robot}")
        self.logger.next_iteration()
        
        # mimic_obs format (for TWIST2 compatibility)
        mimic_obs = self.build_mimic_obs(root_pos, root_quat, dof_pos)
        self.redis_client.set(f"action_body_{self.robot}", json.dumps(mimic_obs.tolist()))
        
        # Hand data (zeros for now)
        self.redis_client.set(f"action_hand_left_{self.robot}", json.dumps(np.zeros(7).tolist()))
        self.redis_client.set(f"action_hand_right_{self.robot}", json.dumps(np.zeros(7).tolist()))
    
    def publish_controller_to_redis(self, controller_data):
        """Publish controller data to Redis for state machine."""
        if controller_data is not None:
            self.redis_client.set(f"controller_{self.robot}", json.dumps(controller_data))
    
    def update_visualization(self, root_pos, root_quat, dof_pos):
        """Update MuJoCo visualization."""
        if not self.vis or self.viewer is None:
            return
        
        # Set pose
        self.mjd.qpos[:3] = root_pos
        # Convert quat from [w,x,y,z] to MuJoCo's [w,x,y,z] (same)
        self.mjd.qpos[3:7] = root_quat
        self.mjd.qpos[7:7+len(dof_pos)] = dof_pos
        
        mujoco.mj_forward(self.mjm, self.mjd)
        
        # Camera follow
        pelvis_pos = self.mjd.xpos[self.mjm.body("pelvis").id]
        self.viewer.cam.lookat = pelvis_pos
        self.viewer.cam.distance = 2.0
        self.viewer.sync()
    
    def run(self):
        """Main teleop loop."""
        last_print_time = time.time()
        no_data_warnings = 0
        
        try:
            while True:
                t_start = time.time()
                
                # 1. Get VR tracking data
                body_pose_dict, left_hand, right_hand, controller, headset = self.get_teleop_data()
                
                if body_pose_dict is None:
                    # No data - publish default pose
                    no_data_warnings += 1
                    if no_data_warnings % 100 == 1:
                        print(f"[yellow]Waiting for PICO data... ({no_data_warnings})[/yellow]")
                    self.redis_client.set(f"action_body_{self.robot}", 
                                         json.dumps(DEFAULT_MIMIC_OBS[self.robot].tolist()))
                    time.sleep(self.control_dt)
                    continue
                else:
                    no_data_warnings = 0
                
                # 2. Retarget to robot
                root_pos, root_quat, dof_pos = self.retarget_to_robot(body_pose_dict)
                
                if root_pos is None:
                    # Retargeting failed - use last known pose
                    root_pos = self.last_root_pos
                    root_quat = self.last_root_quat
                    dof_pos = self.last_dof_pos
                
                # 3. Compute velocities (skip first frame to avoid spikes)
                if self.frame_count == 0:
                    # First frame: initialize last values, use zero velocity
                    root_vel = np.zeros(3)
                    root_ang_vel = np.zeros(3)
                    self.last_root_pos = root_pos.copy()
                    self.last_root_quat = root_quat.copy()
                else:
                    root_vel = (root_pos - self.last_root_pos) / self.control_dt
                    # Compute angular velocity from quaternion change
                    from scipy.spatial.transform import Rotation as R
                    try:
                        r_old = R.from_quat([self.last_root_quat[1], self.last_root_quat[2], self.last_root_quat[3], self.last_root_quat[0]])  # xyzw
                        r_new = R.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])  # xyzw
                        r_delta = r_new * r_old.inv()
                        axis_angle = r_delta.as_rotvec()  # axis * angle
                        root_ang_vel = axis_angle / self.control_dt
                        # Clamp extreme values (sanity check)
                        root_ang_vel = np.clip(root_ang_vel, -10.0, 10.0)  # Max 10 rad/s
                    except:
                        root_ang_vel = np.zeros(3)
                
                # 4. Publish to Redis
                self.publish_to_redis(root_pos, root_quat, dof_pos, root_vel, root_ang_vel)
                
                # 4.5 Publish controller data (for state machine in unified script)
                self.publish_controller_to_redis(controller)
                
                # 5. Update visualization
                self.update_visualization(root_pos, root_quat, dof_pos)
                
                # 6. Update state
                self.last_root_pos = root_pos.copy()
                self.last_root_quat = root_quat.copy()
                self.last_dof_pos = dof_pos.copy()
                self.frame_count += 1
                
                # Print status
                if time.time() - last_print_time > 2.0:
                    print(f"[Frame {self.frame_count}] root=[{root_pos[0]:.2f},{root_pos[1]:.2f},{root_pos[2]:.2f}]")
                    last_print_time = time.time()
                
                # Rate control
                elapsed = time.time() - t_start
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)
        
        except KeyboardInterrupt:
            print("\n[yellow]Stopping teleop server...[/yellow]")
        finally:
            # Publish default pose on exit
            self.redis_client.set(f"action_body_{self.robot}", 
                                 json.dumps(DEFAULT_MIMIC_OBS[self.robot].tolist()))
            if self.viewer:
                self.viewer.close()


def main():
    parser = argparse.ArgumentParser(description="BFM-Zero Teleop Server")
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands",
                        choices=["unitree_g1", "unitree_g1_with_hands"])
    parser.add_argument("--redis_ip", type=str, default="localhost")
    parser.add_argument("--vis", action="store_true", help="Show MuJoCo visualization")
    parser.add_argument("--freq", type=int, default=50, help="Control frequency (Hz)")
    args = parser.parse_args()
    
    server = BFMTeleopServer(
        robot=args.robot,
        redis_ip=args.redis_ip,
        vis=args.vis,
        control_dt=1.0 / args.freq,
    )
    server.run()


if __name__ == "__main__":
    main()

