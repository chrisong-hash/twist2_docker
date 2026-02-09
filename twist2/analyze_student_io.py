"""
Analyze Student Policy Input/Output

Records observations (input) and actions (output) for different student models
using the same motion file, for comparison analysis.

Usage:
    python analyze_student_io.py --motion_file <path> --output_dir <path>
"""

import argparse
import numpy as np
import pickle
import os
import time
from collections import defaultdict

import onnxruntime as ort


def load_motion_data(motion_file):
    """Load motion data from pkl file."""
    with open(motion_file, 'rb') as f:
        data = pickle.load(f)
    return data


def build_observation(motion_data, frame_idx, history_buf, future_frames=None):
    """
    Build observation from motion data at given frame.
    
    This mimics what the deployment server does.
    """
    # Get current frame data
    if frame_idx >= len(motion_data['dof_pos']):
        frame_idx = len(motion_data['dof_pos']) - 1
    
    dof_pos = motion_data['dof_pos'][frame_idx]
    
    # Mock proprio observation (in real deployment this comes from robot state)
    # For analysis, we use the reference motion as "perfect tracking"
    ang_vel = np.zeros(3)  # base angular velocity
    imu = np.zeros(2)  # roll, pitch
    
    # Default joint positions (G1 robot)
    default_dof_pos = np.array([
        0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # left leg
        0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # right leg
        0.0, 0.0, 0.0,  # waist
        0.0, 0.3, 0.0, 0.6, 0.0, 0.0, 0.0,  # left arm
        0.0, -0.3, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm
    ])
    
    dof_vel = np.zeros(29)  # joint velocities
    last_action = np.zeros(29)  # last action
    
    # Scales (from V6 config)
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    ang_vel_scale = 0.25
    
    # Build mimic observation
    # For student: root_vel_xy(2) + root_pos_z(1) + roll_pitch(2) + yaw_ang_vel(1) + dof_pos(29) = 35
    root_vel_xy = np.zeros(2)
    root_pos_z = np.array([0.75])  # default height
    roll_pitch = np.zeros(2)
    yaw_ang_vel = np.zeros(1)
    
    mimic_obs = np.concatenate([
        root_vel_xy,
        root_pos_z,
        roll_pitch,
        yaw_ang_vel,
        dof_pos
    ])  # 35 dims
    
    # Build proprio observation
    proprio_obs = np.concatenate([
        ang_vel * ang_vel_scale,  # 3
        imu,  # 2
        (dof_pos - default_dof_pos) * dof_pos_scale,  # 29
        dof_vel * dof_vel_scale,  # 29
        last_action,  # 29
    ])  # 92 dims
    
    # Current observation
    obs_current = np.concatenate([mimic_obs, proprio_obs])  # 127 dims
    
    # Build future observations if needed
    future_obs = None
    if future_frames is not None:
        future_obs_list = []
        for future_step in future_frames:
            future_idx = min(frame_idx + future_step, len(motion_data['dof_pos']) - 1)
            future_dof = motion_data['dof_pos'][future_idx]
            
            # Same structure as mimic_obs
            future_single = np.concatenate([
                root_vel_xy,
                root_pos_z,
                roll_pitch,
                yaw_ang_vel,
                future_dof
            ])  # 35 dims
            future_obs_list.append(future_single)
        
        future_obs = np.concatenate(future_obs_list)  # 35 * num_future_frames
    
    return obs_current, future_obs


def run_inference(model_path, motion_data, num_frames=500, has_future=False, future_frames=None):
    """
    Run inference on a student model and record I/O.
    
    Returns:
        dict with 'observations', 'actions', 'timestamps'
    """
    print(f"Loading model: {model_path}")
    
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return None
    
    # Get input/output info
    input_info = session.get_inputs()
    output_info = session.get_outputs()
    
    print(f"  Inputs: {[(i.name, i.shape) for i in input_info]}")
    print(f"  Outputs: {[(o.name, o.shape) for o in output_info]}")
    
    # Determine history length from input shape
    input_shape = input_info[0].shape
    if len(input_shape) == 2:
        total_obs_dim = input_shape[1]
    else:
        total_obs_dim = input_shape[-1]
    
    # Initialize history buffer
    history_len = 10
    obs_single_dim = 127
    
    if has_future:
        # V6.2 format: obs_current(127) + history(127*10) + future(35*3)
        expected_dim = obs_single_dim * (history_len + 1) + 35 * len(future_frames)
    else:
        # V6 format: obs_current(127) + history(127*10)
        expected_dim = obs_single_dim * (history_len + 1)
    
    print(f"  Expected obs dim: {expected_dim}, Actual: {total_obs_dim}")
    
    history_buf = np.zeros((history_len, obs_single_dim))
    
    results = {
        'observations': [],
        'actions': [],
        'timestamps': [],
        'model_path': model_path,
    }
    
    # Run inference
    for frame_idx in range(min(num_frames, len(motion_data['dof_pos']))):
        obs_current, future_obs = build_observation(
            motion_data, frame_idx, history_buf, 
            future_frames if has_future else None
        )
        
        # Build full observation
        history_flat = history_buf.flatten()
        
        if has_future and future_obs is not None:
            obs_full = np.concatenate([obs_current, history_flat, future_obs])
        else:
            obs_full = np.concatenate([obs_current, history_flat])
        
        # Pad or truncate to match model input
        if len(obs_full) < total_obs_dim:
            obs_full = np.pad(obs_full, (0, total_obs_dim - len(obs_full)))
        elif len(obs_full) > total_obs_dim:
            obs_full = obs_full[:total_obs_dim]
        
        # Run inference
        obs_tensor = obs_full.astype(np.float32).reshape(1, -1)
        
        try:
            outputs = session.run(None, {input_info[0].name: obs_tensor})
            action = outputs[0][0]
        except Exception as e:
            print(f"  Inference error at frame {frame_idx}: {e}")
            action = np.zeros(29)
        
        # Record
        results['observations'].append(obs_full.copy())
        results['actions'].append(action.copy())
        results['timestamps'].append(frame_idx * 0.02)  # 50Hz
        
        # Update history
        history_buf = np.roll(history_buf, -1, axis=0)
        history_buf[-1] = obs_current
    
    # Convert to numpy arrays
    results['observations'] = np.array(results['observations'])
    results['actions'] = np.array(results['actions'])
    results['timestamps'] = np.array(results['timestamps'])
    
    print(f"  Recorded {len(results['timestamps'])} frames")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_file', type=str, required=True, 
                        help='Path to motion pkl file')
    parser.add_argument('--output_dir', type=str, default='./analysis_output',
                        help='Directory to save results')
    parser.add_argument('--num_frames', type=int, default=500,
                        help='Number of frames to analyze')
    
    # Model paths (adjust as needed)
    parser.add_argument('--v6_model', type=str, 
                        default='legged_gym/logs/h1/student_v6/twist2_v6_student.onnx',
                        help='V6 student ONNX path')
    parser.add_argument('--v6_2_model', type=str,
                        default='legged_gym/logs/h1/student_v6_2/model_6000.onnx', 
                        help='V6.2 student ONNX path')
    parser.add_argument('--default_model', type=str,
                        default='legged_gym/logs/h1/student/default_student.onnx',
                        help='Default student ONNX path')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load motion data
    print(f"Loading motion: {args.motion_file}")
    motion_data = load_motion_data(args.motion_file)
    print(f"  Motion length: {len(motion_data['dof_pos'])} frames")
    
    # V6.2 future frames
    future_frames_v6_2 = [5, 15, 25]  # 0.1s, 0.3s, 0.5s
    
    all_results = {}
    
    # Run each model
    models = [
        ('v6', args.v6_model, False, None),
        ('v6_2', args.v6_2_model, True, future_frames_v6_2),
        ('default', args.default_model, False, None),
    ]
    
    for name, path, has_future, future_frames in models:
        if os.path.exists(path):
            print(f"\n{'='*50}")
            print(f"Running {name} model")
            print(f"{'='*50}")
            results = run_inference(path, motion_data, args.num_frames, has_future, future_frames)
            if results:
                all_results[name] = results
        else:
            print(f"\nSkipping {name}: {path} not found")
    
    # Save results
    output_path = os.path.join(args.output_dir, 'student_io_comparison.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump({
            'motion_file': args.motion_file,
            'motion_data': motion_data,
            'results': all_results,
        }, f)
    
    print(f"\n{'='*50}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*50}")
    
    # Print summary statistics
    print("\nAction Statistics:")
    for name, results in all_results.items():
        actions = results['actions']
        print(f"\n{name}:")
        print(f"  Action mean: {np.mean(actions, axis=0)[:6]}...")  # First 6 joints
        print(f"  Action std:  {np.std(actions, axis=0)[:6]}...")
        print(f"  Action range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
        
        # Compute jerk (for spasm analysis)
        if len(actions) > 2:
            jerk = actions[2:] - 2*actions[1:-1] + actions[:-2]
            jerk_magnitude = np.sqrt(np.sum(jerk**2, axis=1))
            print(f"  Jerk mean: {np.mean(jerk_magnitude):.4f}")
            print(f"  Jerk max:  {np.max(jerk_magnitude):.4f}")


if __name__ == '__main__':
    main()



