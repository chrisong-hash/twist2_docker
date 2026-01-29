#!/usr/bin/env python
"""
Motion Server V6.2 - Publishes current frame + 3 future frames (0.1s, 0.3s, 0.5s ahead)

For use with g1_stu_future_v6_2 student policy that expects real future observations.

Changes from original:
- Publishes `action_mimic_future_{robot}` with 3 future frames (105 dims = 3 * 35)
- Future steps: [5, 15, 25] at 50Hz = [0.1s, 0.3s, 0.5s]
"""

import argparse
import time
import redis
import json
import numpy as np
import isaacgym
import torch
from rich import print
import os
import mujoco
from mujoco.viewer import launch_passive
from pose.utils.motion_lib_pkl import MotionLib
from data_utils.rot_utils import euler_from_quaternion_torch, quat_rotate_inverse_torch

from data_utils.params import DEFAULT_MIMIC_OBS


# V6.2 future steps: 0.1s, 0.3s, 0.5s ahead at 50Hz
FUTURE_STEPS_V6_2 = [5, 15, 25]


def build_mimic_obs_single_frame(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    robot_type: str = "g1"
):
    """
    Build mimic_obs for a single frame (35 dims).
    Returns: mimic_obs (35,), root_pos, root_rot, dof_pos, root_vel, root_ang_vel
    """
    device = torch.device("cuda")
    
    # Single step tensor
    tar_motion_steps_tensor = torch.tensor([0], device=device, dtype=torch.int)
    
    motion_times = torch.tensor([t_step * control_dt], device=device).unsqueeze(-1)
    obs_motion_times = tar_motion_steps_tensor * control_dt + motion_times
    obs_motion_times = obs_motion_times.flatten()
    
    motion_ids = torch.zeros(1, dtype=torch.int, device=device)
    
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, root_pos_delta_local, root_rot_delta_local = motion_lib.calc_motion_frame(motion_ids, obs_motion_times)

    roll, pitch, yaw = euler_from_quaternion_torch(root_rot, scalar_first=False)
    roll = roll.reshape(1, 1, 1)
    pitch = pitch.reshape(1, 1, 1)

    root_vel_local = quat_rotate_inverse_torch(root_rot, root_vel, scalar_first=False).reshape(1, 1, 3)
    root_ang_vel_local = quat_rotate_inverse_torch(root_rot, root_ang_vel, scalar_first=False).reshape(1, 1, 3)
    root_vel = root_vel.reshape(1, 1, 3)
    root_ang_vel = root_ang_vel.reshape(1, 1, 3)
    root_pos = root_pos.reshape(1, 1, 3)
    dof_pos = dof_pos.reshape(1, 1, dof_pos.shape[-1])
    
    # mimic_obs: root_vel_xy(2) + root_pos_z(1) + roll_pitch(2) + yaw_ang_vel(1) + dof_pos(29) = 35 dims
    mimic_obs_buf = torch.cat((
        root_vel_local[..., :2],      # 2 dims (xy velocity)
        root_pos[..., 2:3],           # 1 dim (z position)
        roll, pitch,                   # 2 dims (roll/pitch orientation)
        root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
        dof_pos,                       # 29 dims (joint positions)
    ), dim=-1)
    
    mimic_obs_buf = mimic_obs_buf.reshape(-1)
    
    return (
        mimic_obs_buf.detach().cpu().numpy(),
        root_pos.detach().cpu().numpy().squeeze(),
        root_rot.detach().cpu().numpy().squeeze(),
        dof_pos.detach().cpu().numpy().squeeze(),
        root_vel.detach().cpu().numpy().squeeze(),
        root_ang_vel.detach().cpu().numpy().squeeze()
    )


def build_future_obs(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    future_steps: list,
    num_total_steps: int,
    robot_type: str = "g1"
):
    """
    Build future mimic_obs for multiple future frames.
    
    Args:
        motion_lib: Motion library
        t_step: Current time step
        control_dt: Control timestep (0.02s at 50Hz)
        future_steps: List of future step offsets [5, 15, 25]
        num_total_steps: Total steps in motion (for clamping)
    
    Returns:
        future_obs: (num_future_frames * 35,) flattened future observations
    """
    device = torch.device("cuda")
    
    future_obs_list = []
    
    for step_offset in future_steps:
        # Calculate future time step, clamped to motion length
        future_t_step = min(t_step + step_offset, num_total_steps - 1)
        
        # Build mimic_obs for this future frame
        mimic_obs, _, _, _, _, _ = build_mimic_obs_single_frame(
            motion_lib=motion_lib,
            t_step=future_t_step,
            control_dt=control_dt,
            robot_type=robot_type
        )
        future_obs_list.append(mimic_obs)
    
    # Concatenate all future frames: [frame1(35), frame2(35), frame3(35)] = 105 dims
    future_obs = np.concatenate(future_obs_list)
    
    return future_obs


def main(args, xml_file, robot_base):
    motion_started = False if args.use_remote_control else True
    
    if args.use_remote_control:
        print("[Motion Server V6.2] Remote control enabled. Waiting for start signal...")

    if args.vis:
        sim_model = mujoco.MjModel.from_xml_path(xml_file)
        sim_data = mujoco.MjData(sim_model)
        viewer = launch_passive(model=sim_model, data=sim_data, show_left_ui=False, show_right_ui=False)
            
    # Connect to Redis
    redis_client = redis.Redis(host=args.redis_ip, port=6379, db=0)
    redis_client.ping()

    # Load motion library
    device = "cuda" if torch.cuda.is_available() else "cpu"
    motion_lib = MotionLib(args.motion_file, device=device)
    
    # Control parameters
    control_dt = 0.02  # 50Hz
    
    # Compute number of steps
    motion_id = torch.tensor([0], device=device, dtype=torch.long)
    motion_length = motion_lib.get_motion_length(motion_id)
    num_steps = int(motion_length / control_dt)
    
    # Future steps for V6.2: [5, 15, 25] = [0.1s, 0.3s, 0.5s]
    future_steps = FUTURE_STEPS_V6_2
    
    print(f"[Motion Server V6.2] Configuration:")
    print(f"  Motion length: {motion_length.item():.2f}s ({num_steps} steps)")
    print(f"  Control dt: {control_dt:.3f}s (50Hz)")
    print(f"  Future steps: {future_steps} = [{', '.join([f'{s*control_dt:.2f}s' for s in future_steps])}]")
    print(f"  Future obs dims: {len(future_steps) * 35} = {len(future_steps)} frames × 35 dims")

    # Get start frame for interpolation back to default
    start_frame_mimic_obs, _, _, _, _, _ = build_mimic_obs_single_frame(
        motion_lib=motion_lib,
        t_step=0,
        control_dt=control_dt,
        robot_type=args.robot
    )
    
    last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
    
    def check_remote_control_signals():
        if not args.use_remote_control:
            return True, False
        try:
            start_signal = redis_client.get("motion_start_signal")
            start_pressed = start_signal == b"1" if start_signal else False
            exit_signal = redis_client.get("motion_exit_signal") 
            exit_pressed = exit_signal == b"1" if exit_signal else False
            return start_pressed, exit_pressed
        except Exception:
            return False, False
    
    if args.use_remote_control:
        redis_client.set("motion_start_signal", "0")
        redis_client.set("motion_exit_signal", "0")
    
    try:
        t_step = 0
        while True:
            t0 = time.time()
            
            # Handle remote control
            if args.use_remote_control:
                start_pressed, exit_pressed = check_remote_control_signals()
                if exit_pressed:
                    print("[Motion Server V6.2] Exit signal received, stopping...")
                    break
                if not motion_started and start_pressed:
                    print("[Motion Server V6.2] Start signal received, beginning motion...")
                    motion_started = True
                elif not motion_started:
                    # Send default pose while waiting
                    idle_obs = start_frame_mimic_obs if args.send_start_frame_as_end_frame else DEFAULT_MIMIC_OBS[args.robot]
                    redis_client.set(f"motion_raw_{args.robot}", json.dumps(idle_obs.tolist()))
                    redis_client.set(f"action_hand_left_{args.robot}", json.dumps(np.zeros(7).tolist()))
                    redis_client.set(f"action_hand_right_{args.robot}", json.dumps(np.zeros(7).tolist()))
                    # Also send empty future obs
                    empty_future = np.zeros(len(future_steps) * 35).tolist()
                    redis_client.set(f"action_mimic_future_{args.robot}", json.dumps(empty_future))
                    
                    elapsed = time.time() - t0
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)
                    continue

            # === BUILD CURRENT FRAME ===
            mimic_obs, root_pos, root_rot, dof_pos, root_vel, root_ang_vel = build_mimic_obs_single_frame(
                motion_lib=motion_lib,
                t_step=t_step,
                control_dt=control_dt,
                robot_type=args.robot
            )
            
            # === BUILD FUTURE FRAMES (V6.2 addition) ===
            future_obs = build_future_obs(
                motion_lib=motion_lib,
                t_step=t_step,
                control_dt=control_dt,
                future_steps=future_steps,
                num_total_steps=num_steps,
                robot_type=args.robot
            )
            
            # === PUBLISH TO REDIS ===
            mimic_obs_list = mimic_obs.tolist()
            future_obs_list = future_obs.tolist()
            
            # Publish RAW motion data (for buffered server to process)
            redis_client.set(f"motion_raw_{args.robot}", json.dumps(mimic_obs_list))
            redis_client.set(f"action_hand_left_{args.robot}", json.dumps(np.zeros(7).tolist()))
            redis_client.set(f"action_hand_right_{args.robot}", json.dumps(np.zeros(7).tolist()))
            redis_client.set(f"action_neck_{args.robot}", json.dumps(np.zeros(2).tolist()))
            
            last_mimic_obs = mimic_obs
            
            # Print progress
            future_times = [f"{(t_step + s) * control_dt:.2f}s" for s in future_steps]
            print(f"Step {t_step:4d} | t={t_step*control_dt:.2f}s | future: [{', '.join(future_times)}]", end="\r")

            # Visualization
            if args.vis:
                sim_data.qpos[:3] = root_pos
                root_rot_vis = root_rot[[3, 0, 1, 2]]  # Reorder quaternion
                sim_data.qpos[3:7] = root_rot_vis
                sim_data.qpos[7:] = dof_pos
                mujoco.mj_forward(sim_model, sim_data)
                robot_base_pos = sim_data.xpos[sim_model.body(robot_base).id]
                viewer.cam.lookat = robot_base_pos
                viewer.cam.distance = 2.0
                viewer.sync()
            
            t_step += 1
            if t_step >= num_steps:
                break
                
            # Maintain real-time pace
            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
    
    except KeyboardInterrupt:
        print("\n[Motion Server V6.2] Keyboard interrupt. Interpolating to default pose...")
    except Exception as e:
        print(f"\n[Motion Server V6.2] Error: {e}")
    finally:
        # Interpolate back to default pose
        print("\n[Motion Server V6.2] Interpolating to default pose...")
        time_back_to_default = 2.0
        target_obs = start_frame_mimic_obs if args.send_start_frame_as_end_frame else DEFAULT_MIMIC_OBS[args.robot]
        
        for i in range(int(time_back_to_default / control_dt)):
            alpha = i / (time_back_to_default / control_dt)
            interp_obs = last_mimic_obs + (target_obs - last_mimic_obs) * alpha
            redis_client.set(f"motion_raw_{args.robot}", json.dumps(interp_obs.tolist()))
            # Also interpolate future obs to target (all same)
            future_target = np.tile(target_obs, len(future_steps))
            redis_client.set(f"action_mimic_future_{args.robot}", json.dumps(future_target.tolist()))
            time.sleep(control_dt)
        
        redis_client.set(f"motion_raw_{args.robot}", json.dumps(target_obs.tolist()))
        if args.vis:
            viewer.close()
        print("[Motion Server V6.2] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion Server V6.2 with 0.5s future sight")
    parser.add_argument("--motion_file", 
                        default="../assets/example_motions/0807_yanjie_walk_001.pkl",
                        help="Path to motion .pkl file")
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands", 
                        choices=["unitree_g1", "unitree_g1_with_hands"])
    parser.add_argument("--vis", action="store_true", help="Visualize the motion")
    parser.add_argument("--use_remote_control", action="store_true", 
                        help="Use remote control signals from robot controller")
    parser.add_argument("--send_start_frame_as_end_frame", action="store_true", 
                        help="Use motion's first frame as end frame instead of default pose")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis IP")
    args = parser.parse_args()

    args.vis = True
    
    print(f"[Motion Server V6.2]")
    print(f"  Robot: {args.robot}")
    print(f"  Motion file: {args.motion_file}")
    print(f"  Future frames: {FUTURE_STEPS_V6_2} (0.1s, 0.3s, 0.5s ahead)")
    
    HERE = os.path.dirname(os.path.abspath(__file__))
    
    if args.robot in ["unitree_g1", "unitree_g1_with_hands"]:
        xml_file = f"{HERE}/../assets/g1/g1_mocap_29dof.xml"
        robot_base = "pelvis"
    else:
        raise ValueError(f"Robot type {args.robot} not supported")
    
    main(args, xml_file, robot_base)


