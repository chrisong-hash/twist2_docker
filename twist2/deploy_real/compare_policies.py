"""Compare Default vs V6.3 standing still behavior."""

import json
import numpy as np
import redis
import mujoco
from collections import deque
from data_utils.rot_utils import quatToEuler
import onnxruntime as ort


def test_policy(policy_path, name, obs_size, use_future=False):
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
    
    waist_z_list = []
    action_mag_list = []
    wrist_mag_list = []
    wrist_actions_all = []
    
    for step in range(200):
        waist_z = data.qpos[2]
        waist_z_list.append(waist_z)
        
        if waist_z < 0.3:
            break
        
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        dof_pos = qpos[7:7+num_actions]
        dof_vel = qvel[6:6+num_actions]
        ang_vel = qvel[3:6]
        rpy = quatToEuler(qpos[3:7])
        
        obs_body_dof_vel = dof_vel.copy()
        obs_body_dof_vel[ankle_idx] = 0.0
        
        obs_proprio = np.concatenate([ang_vel * 0.25, rpy[:2], dof_pos - default_dof_pos, obs_body_dof_vel * 0.05, last_action])
        
        body_data = r.get("action_body_unitree_g1_with_hands")
        action_mimic = np.array(json.loads(body_data), dtype=np.float32) if body_data else np.zeros(35, dtype=np.float32)
        
        obs_full = np.concatenate([action_mimic, obs_proprio])  # 127
        
        if use_future:
            future_data = r.get("action_mimic_future_unitree_g1_with_hands")
            future_obs = np.array(json.loads(future_data), dtype=np.float32) if future_data else np.tile(action_mimic, 3)[:105]
        
        if len(history_buf) < history_len:
            for _ in range(history_len):
                history_buf.append(obs_full.copy())
        
        obs_hist = np.array(history_buf).flatten()
        history_buf.append(obs_full)
        
        if use_future:
            obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
        else:
            obs_buf = np.concatenate([obs_full, obs_hist])
        
        # Pad if needed
        if len(obs_buf) < obs_size:
            obs_buf = np.concatenate([obs_buf, np.zeros(obs_size - len(obs_buf))])
        obs_buf = obs_buf[:obs_size]
        
        obs_tensor = np.clip(obs_buf, -100, 100).astype(np.float32).reshape(1, -1)
        raw_action = session.run(None, {input_name: obs_tensor})[0].squeeze()
        raw_action = np.clip(raw_action, -100, 100)
        
        action_mag_list.append(np.abs(raw_action).mean())
        wrist_mag_list.append(np.abs(raw_action[WRIST_INDICES]).mean())
        wrist_actions_all.append(raw_action[WRIST_INDICES].copy())
        
        target_dof_pos = raw_action * 0.5 + default_dof_pos
        last_action = raw_action.copy()
        
        for _ in range(20):
            cur_pos = data.qpos[7:7+num_actions].copy()
            cur_vel = data.qvel[6:6+num_actions].copy()
            torque = kps * (target_dof_pos - cur_pos) - kds * cur_vel
            data.ctrl[:num_actions] = torque
            mujoco.mj_step(model, data)
    
    wrist_actions_all = np.array(wrist_actions_all)
    
    print(f"\n{name}:")
    print(f"  Steps survived: {len(waist_z_list)}")
    print(f"  Waist Z: min={min(waist_z_list):.3f}, avg={np.mean(waist_z_list):.3f}")
    print(f"  Action mag: mean={np.mean(action_mag_list):.4f}, std={np.std(action_mag_list):.4f}")
    print(f"  WRIST mag: mean={np.mean(wrist_mag_list):.4f}, std={np.std(wrist_mag_list):.4f}")
    print(f"  WRIST per-joint std: {wrist_actions_all.std(axis=0)}")
    
    return {
        "steps": len(waist_z_list),
        "waist_z_min": min(waist_z_list),
        "waist_z_avg": np.mean(waist_z_list),
        "action_mean": np.mean(action_mag_list),
        "action_std": np.std(action_mag_list),
        "wrist_mean": np.mean(wrist_mag_list),
        "wrist_std": np.std(wrist_mag_list),
        "wrist_per_joint_std": wrist_actions_all.std(axis=0).tolist()
    }


if __name__ == "__main__":
    print("=" * 60)
    print("COMPARISON: Default vs V6.3 (Standing Still)")
    print("=" * 60)
    
    # Test Default
    default_result = test_policy(
        "/workspace/twist2/assets/ckpts/twist2_v6_student.onnx",
        "DEFAULT (twist2_v6_student, 1432 dims)",
        obs_size=1432,
        use_future=False
    )
    
    # Test V6.3
    v6_3_result = test_policy(
        "/workspace/twist2/legged_gym/logs/g1_priv_mimic/v6_3_fixed/model_27500.onnx",
        "V6.3 (with future obs, 1502 dims)",
        obs_size=1502,
        use_future=True
    )
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"                    DEFAULT     V6.3")
    print(f"  Steps:            {default_result['steps']:>7}    {v6_3_result['steps']:>7}")
    print(f"  Waist Z min:      {default_result['waist_z_min']:>7.3f}    {v6_3_result['waist_z_min']:>7.3f}")
    print(f"  Action mean:      {default_result['action_mean']:>7.4f}    {v6_3_result['action_mean']:>7.4f}")
    print(f"  Action std:       {default_result['action_std']:>7.4f}    {v6_3_result['action_std']:>7.4f}")
    print(f"  WRIST mean:       {default_result['wrist_mean']:>7.4f}    {v6_3_result['wrist_mean']:>7.4f}")
    print(f"  WRIST std:        {default_result['wrist_std']:>7.4f}    {v6_3_result['wrist_std']:>7.4f}")
    
    if v6_3_result['wrist_std'] > 0 and default_result['wrist_std'] > 0:
        ratio = v6_3_result['wrist_std'] / default_result['wrist_std']
        print(f"\n  V6.3 wrist variability is {ratio:.1f}x default")
    print("=" * 60)


