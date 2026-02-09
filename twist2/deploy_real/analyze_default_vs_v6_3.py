"""
Deep Analysis: Default vs V6.3 Standing Still Behavior

Goal: Understand why V6.3's left wrist moves more despite outputting calmer actions.

Hypotheses:
1. Feedback loop amplification - small actions cause position changes that feed back
2. Different observation structure affects policy behavior  
3. Future observations cause anticipatory movements
4. Different action-to-position mapping due to PD controller dynamics
"""

import json
import numpy as np
import redis
import mujoco
from collections import deque
from data_utils.rot_utils import quatToEuler
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def detailed_policy_analysis(policy_path, name, obs_size, use_future=False, num_steps=500):
    """Run detailed analysis of policy behavior during standing still."""
    
    model = mujoco.MjModel.from_xml_path("../assets/g1/g1_sim2sim_29dof.xml")
    data = mujoco.MjData(model)
    
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
    num_actions = 29
    default_dof_pos = np.array([
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
        0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0
    ], dtype=np.float32)
    ankle_idx = [4, 5, 10, 11]
    WRIST_INDICES = [18, 19, 20, 25, 26, 27]
    WRIST_NAMES = ["L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw", 
                   "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw"]
    
    # PD gains (same for both)
    kps = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 150, 150, 150,
                    40, 40, 40, 40, 4.0, 4.0, 4.0, 40, 40, 40, 40, 4.0, 4.0, 4.0], dtype=np.float32)
    kds = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 4, 4, 4,
                    5, 5, 5, 5, 2.0, 2.0, 2.0, 5, 5, 5, 5, 2.0, 2.0, 2.0], dtype=np.float32)
    
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.78
    mujoco.mj_forward(model, data)
    
    history_len = 10
    history_buf = deque(maxlen=history_len)
    last_action = np.zeros(num_actions, dtype=np.float32)
    
    r = redis.Redis(host="localhost", port=6379)
    
    # Data collection
    records = {
        'step': [],
        'waist_z': [],
        'wrist_actions': [],      # Policy output for wrists
        'wrist_positions': [],    # Actual joint positions
        'wrist_velocities': [],   # Joint velocities
        'wrist_torques': [],      # Applied torques
        'wrist_target_pos': [],   # Target positions (action * 0.5 + default)
        'wrist_pos_error': [],    # target - current
        'action_scale': 0.5,
    }
    
    for step in range(num_steps):
        waist_z = data.qpos[2]
        if waist_z < 0.3:
            print(f"  FELL at step {step}")
            break
        
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        dof_pos = qpos[7:7+num_actions]
        dof_vel = qvel[6:6+num_actions]
        ang_vel = qvel[3:6]
        rpy = quatToEuler(qpos[3:7])
        
        obs_body_dof_vel = dof_vel.copy()
        obs_body_dof_vel[ankle_idx] = 0.0
        
        obs_proprio = np.concatenate([ang_vel * 0.25, rpy[:2], dof_pos - default_dof_pos, 
                                      obs_body_dof_vel * 0.05, last_action])
        
        # Use zero mimic (standing pose)
        action_mimic = np.zeros(35, dtype=np.float32)
        
        obs_full = np.concatenate([action_mimic, obs_proprio])
        
        if use_future:
            future_obs = np.zeros(105, dtype=np.float32)  # Standing still = zeros
        
        if len(history_buf) < history_len:
            for _ in range(history_len):
                history_buf.append(obs_full.copy())
        
        obs_hist = np.array(history_buf).flatten()
        history_buf.append(obs_full)
        
        if use_future:
            obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
        else:
            obs_buf = np.concatenate([obs_full, obs_hist])
        
        if len(obs_buf) < obs_size:
            obs_buf = np.concatenate([obs_buf, np.zeros(obs_size - len(obs_buf))])
        obs_buf = obs_buf[:obs_size]
        
        obs_tensor = np.clip(obs_buf, -100, 100).astype(np.float32).reshape(1, -1)
        raw_action = session.run(None, {input_name: obs_tensor})[0].squeeze()
        raw_action = np.clip(raw_action, -100, 100)
        
        # Compute target position and torque
        target_dof_pos = raw_action * 0.5 + default_dof_pos
        
        # Record data BEFORE physics step
        records['step'].append(step)
        records['waist_z'].append(waist_z)
        records['wrist_actions'].append(raw_action[WRIST_INDICES].copy())
        records['wrist_positions'].append(dof_pos[WRIST_INDICES].copy())
        records['wrist_velocities'].append(dof_vel[WRIST_INDICES].copy())
        records['wrist_target_pos'].append(target_dof_pos[WRIST_INDICES].copy())
        records['wrist_pos_error'].append((target_dof_pos - dof_pos)[WRIST_INDICES].copy())
        
        # Compute torque for wrists
        wrist_torque = kps[WRIST_INDICES] * (target_dof_pos[WRIST_INDICES] - dof_pos[WRIST_INDICES]) \
                     - kds[WRIST_INDICES] * dof_vel[WRIST_INDICES]
        records['wrist_torques'].append(wrist_torque)
        
        last_action = raw_action.copy()
        
        # Run physics
        for _ in range(20):
            cur_pos = data.qpos[7:7+num_actions].copy()
            cur_vel = data.qvel[6:6+num_actions].copy()
            torque = kps * (target_dof_pos - cur_pos) - kds * cur_vel
            data.ctrl[:num_actions] = torque
            mujoco.mj_step(model, data)
    
    # Convert to numpy arrays
    for key in ['wrist_actions', 'wrist_positions', 'wrist_velocities', 
                'wrist_torques', 'wrist_target_pos', 'wrist_pos_error']:
        records[key] = np.array(records[key])
    
    return records, WRIST_NAMES


def analyze_and_compare():
    """Run analysis on both policies and compare."""
    
    print("=" * 70)
    print("DEEP ANALYSIS: Default vs V6.3 Standing Still")
    print("=" * 70)
    
    # Run analysis
    print("\n[1/2] Analyzing DEFAULT policy...")
    default_records, wrist_names = detailed_policy_analysis(
        "/workspace/twist2/assets/ckpts/twist2_1017_25k.onnx",
        "DEFAULT",
        obs_size=1432,
        use_future=False,
        num_steps=300
    )
    
    print("[2/2] Analyzing V6.3 policy...")
    v63_records, _ = detailed_policy_analysis(
        "/workspace/twist2/legged_gym/logs/g1_priv_mimic/v6_3_fixed/model_27500.onnx",
        "V6.3",
        obs_size=1502,
        use_future=True,
        num_steps=300
    )
    
    # === ANALYSIS ===
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    
    # 1. Action statistics
    print("\n[1] WRIST ACTION STATISTICS (Policy Output)")
    print("-" * 50)
    print(f"{'Joint':<20} {'Default Mean':>12} {'Default Std':>12} {'V6.3 Mean':>12} {'V6.3 Std':>12}")
    for i, name in enumerate(wrist_names):
        d_mean = default_records['wrist_actions'][:, i].mean()
        d_std = default_records['wrist_actions'][:, i].std()
        v_mean = v63_records['wrist_actions'][:, i].mean()
        v_std = v63_records['wrist_actions'][:, i].std()
        print(f"{name:<20} {d_mean:>12.4f} {d_std:>12.4f} {v_mean:>12.4f} {v_std:>12.4f}")
    
    # 2. Position statistics
    print("\n[2] WRIST POSITION STATISTICS (Actual Joint Angles)")
    print("-" * 50)
    print(f"{'Joint':<20} {'Default Mean':>12} {'Default Std':>12} {'V6.3 Mean':>12} {'V6.3 Std':>12}")
    for i, name in enumerate(wrist_names):
        d_mean = default_records['wrist_positions'][:, i].mean()
        d_std = default_records['wrist_positions'][:, i].std()
        v_mean = v63_records['wrist_positions'][:, i].mean()
        v_std = v63_records['wrist_positions'][:, i].std()
        diff = v_std - d_std
        marker = "⬆️" if diff > 0.1 else ("⬇️" if diff < -0.1 else "")
        print(f"{name:<20} {d_mean:>12.4f} {d_std:>12.4f} {v_mean:>12.4f} {v_std:>12.4f} {marker}")
    
    # 3. Position error (target - current)
    print("\n[3] WRIST POSITION ERROR (Target - Current)")
    print("-" * 50)
    print(f"{'Joint':<20} {'Default Mean':>12} {'Default Std':>12} {'V6.3 Mean':>12} {'V6.3 Std':>12}")
    for i, name in enumerate(wrist_names):
        d_mean = default_records['wrist_pos_error'][:, i].mean()
        d_std = default_records['wrist_pos_error'][:, i].std()
        v_mean = v63_records['wrist_pos_error'][:, i].mean()
        v_std = v63_records['wrist_pos_error'][:, i].std()
        print(f"{name:<20} {d_mean:>12.4f} {d_std:>12.4f} {v_mean:>12.4f} {v_std:>12.4f}")
    
    # 4. Torque statistics
    print("\n[4] WRIST TORQUE STATISTICS")
    print("-" * 50)
    print(f"{'Joint':<20} {'Default Mean':>12} {'Default Std':>12} {'V6.3 Mean':>12} {'V6.3 Std':>12}")
    for i, name in enumerate(wrist_names):
        d_mean = default_records['wrist_torques'][:, i].mean()
        d_std = default_records['wrist_torques'][:, i].std()
        v_mean = v63_records['wrist_torques'][:, i].mean()
        v_std = v63_records['wrist_torques'][:, i].std()
        print(f"{name:<20} {d_mean:>12.4f} {d_std:>12.4f} {v_mean:>12.4f} {v_std:>12.4f}")
    
    # 5. Feedback loop analysis
    print("\n[5] FEEDBACK LOOP ANALYSIS")
    print("-" * 50)
    print("Checking correlation between position error and next action...")
    
    for i, name in enumerate(wrist_names):
        # Compute correlation between position error at t and action at t+1
        d_pos_err = default_records['wrist_pos_error'][:-1, i]
        d_next_act = default_records['wrist_actions'][1:, i]
        d_corr = np.corrcoef(d_pos_err, d_next_act)[0, 1] if len(d_pos_err) > 10 else 0
        
        v_pos_err = v63_records['wrist_pos_error'][:-1, i]
        v_next_act = v63_records['wrist_actions'][1:, i]
        v_corr = np.corrcoef(v_pos_err, v_next_act)[0, 1] if len(v_pos_err) > 10 else 0
        
        print(f"{name:<20} Default: {d_corr:>7.3f}  V6.3: {v_corr:>7.3f}")
    
    # 6. Oscillation detection
    print("\n[6] OSCILLATION DETECTION (Sign Changes in Action)")
    print("-" * 50)
    for i, name in enumerate(wrist_names):
        d_sign_changes = np.sum(np.abs(np.diff(np.sign(default_records['wrist_actions'][:, i]))) > 0)
        v_sign_changes = np.sum(np.abs(np.diff(np.sign(v63_records['wrist_actions'][:, i]))) > 0)
        print(f"{name:<20} Default: {d_sign_changes:>5}  V6.3: {v_sign_changes:>5}")
    
    # 7. Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    # Check which wrist has higher position variance in V6.3
    d_l_pitch_std = default_records['wrist_positions'][:, 1].std()
    v_l_pitch_std = v63_records['wrist_positions'][:, 1].std()
    
    d_l_pitch_act_std = default_records['wrist_actions'][:, 1].std()
    v_l_pitch_act_std = v63_records['wrist_actions'][:, 1].std()
    
    print(f"\nL_wrist_pitch:")
    print(f"  Action std:   Default={d_l_pitch_act_std:.4f}  V6.3={v_l_pitch_act_std:.4f}")
    print(f"  Position std: Default={d_l_pitch_std:.4f}  V6.3={v_l_pitch_std:.4f}")
    
    if v_l_pitch_act_std < d_l_pitch_act_std and v_l_pitch_std > d_l_pitch_std:
        print("\n  ⚠️ PARADOX DETECTED:")
        print("     V6.3 outputs SMALLER actions but has LARGER position variance!")
        print("\n  POSSIBLE CAUSES:")
        print("     1. V6.3's actions are more correlated with current error (positive feedback)")
        print("     2. V6.3's history observations encode position errors that get amplified")
        print("     3. The low PD gains (KP=4.0) can't reject disturbances effectively")
        print("\n  RECOMMENDATIONS FOR V6.5 DEPLOYMENT:")
        print("     1. Add action smoothing (EMA) to break feedback loop")
        print("     2. Increase wrist KP from 4.0 to ~10-20 for better disturbance rejection")
        print("     3. Add dead-zone for small actions (ignore actions < threshold)")
    
    # Save plot
    print("\n[7] Generating comparison plot...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i, (name, ax) in enumerate(zip(wrist_names[:3], axes[0])):
        ax.plot(default_records['wrist_positions'][:, i], label='Default', alpha=0.7)
        ax.plot(v63_records['wrist_positions'][:, i], label='V6.3', alpha=0.7)
        ax.set_title(f'{name} Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for i, (name, ax) in enumerate(zip(wrist_names[:3], axes[1])):
        ax.plot(default_records['wrist_actions'][:, i], label='Default', alpha=0.7)
        ax.plot(v63_records['wrist_actions'][:, i], label='V6.3', alpha=0.7)
        ax.set_title(f'{name} Action')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/twist2/analysis_default_vs_v63.png', dpi=150)
    print("  Saved to: /workspace/twist2/analysis_default_vs_v63.png")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    analyze_and_compare()


