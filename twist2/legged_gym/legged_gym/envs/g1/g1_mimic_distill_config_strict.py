"""
G1 Mimic Distill Config - STRICT TERMINATION VARIANT
=====================================================
This config tests the hypothesis that lenient termination (229°) allows
the policy to "survive" while drifting globally, rather than learning
proper motion tracking.

Key changes from default:
- termination_roll: 4.0 → 1.0 (229° → 57°)
- termination_pitch: 4.0 → 1.0 (229° → 57°)
- entropy_coef: 0.005 → 0.01 (more exploration)

Run with:
  python train.py --task g1_priv_mimic_strict --exptid strict_termination_v1

To use this config, register it in:
  legged_gym/legged_gym/envs/__init__.py
"""

from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class G1MimicPrivStrictCfg(HumanoidMimicCfg):
    """Strict termination variant - forces policy to maintain proper orientation."""
    
    class env(HumanoidMimicCfg.env):
        tar_motion_steps_priv = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        tar_motion_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096
        num_actions = 29
        obs_type = 'priv'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0
        
        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_motion_steps_priv) * (21 + num_actions + 3*9)
        n_mimic_obs_single = 6 + 29
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_priv_info = 3 + 3 + 4 + 3*9 + 2 + 4 + 1 + 2*num_actions
        history_len = 10
        
        n_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        
        num_observations = n_priv_obs_single
        num_privileged_obs = n_priv_obs_single

        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 10
        
        randomize_start_pos = True
        randomize_start_yaw = False
        
        history_encoding = True
        contact_buf_len = 10
        
        normalize_obs = True
        
        enable_early_termination = True
        pose_termination = True
        pose_termination_dist = 0.7
        rand_reset = True
        
        track_root = False
        root_tracking_termination_dist = 2.0
     
        dof_err_w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Left Leg
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Right Leg
                     1.0, 1.0, 1.0,                  # Waist
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Left Arm
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Right Arm

    class init_state(HumanoidMimicCfg.init_state):
        pos = [0.0, 0.0, 0.78]
        default_joint_angles = {
            'left_hip_pitch_joint': -0.1,
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.1,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            'waist_yaw_joint': 0.0,
            'waist_pitch_joint': 0.0,
            'waist_roll_joint': 0.0,
            'left_shoulder_pitch_joint': 0.3,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.5,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'right_shoulder_pitch_joint': 0.3,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.5,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        }

    class control(HumanoidMimicCfg.control):
        control_type = 'P'
        stiffness = {
            'hip_pitch': 100, 'hip_roll': 100, 'hip_yaw': 100,
            'knee': 150, 'ankle_pitch': 40, 'ankle_roll': 40,
            'waist_yaw': 200, 'waist_pitch': 200, 'waist_roll': 200,
            'shoulder_pitch': 40, 'shoulder_roll': 40, 'shoulder_yaw': 40,
            'elbow': 40, 'wrist_roll': 40, 'wrist_pitch': 40, 'wrist_yaw': 40,
        }
        damping = {
            'hip_pitch': 2, 'hip_roll': 2, 'hip_yaw': 2,
            'knee': 4, 'ankle_pitch': 2, 'ankle_roll': 2,
            'waist_yaw': 4, 'waist_pitch': 4, 'waist_roll': 4,
            'shoulder_pitch': 2, 'shoulder_roll': 2, 'shoulder_yaw': 2,
            'elbow': 2, 'wrist_roll': 2, 'wrist_pitch': 2, 'wrist_yaw': 2,
        }
        action_scale = 0.25
        decimation = 4

    class asset(HumanoidMimicCfg.asset):
        file = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29_with_hand_frame.xml"
        terminate_after_contacts_on = ['pelvis']
        self_collisions = 0
        whole_body_indices = [
            'pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
            'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
            'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
            'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
            'waist_yaw_link', 'waist_pitch_link', 'waist_roll_link',
            'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
            'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
            'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
            'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link',
        ]
        fix_base_link = False

    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = []
        regularization_scale = 1.0
        regularization_scale_range = [0.8, 2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        
        class scales:
            # ORIGINAL reward scales (not the v2 modified ones)
            # This isolates the termination hypothesis
            tracking_joint_dof = 2.0
            tracking_joint_vel = 0.2
            tracking_root_translation_z = 1.0
            tracking_root_rotation = 1.0
            tracking_root_linear_vel = 1.0
            tracking_root_angular_vel = 1.0
            tracking_keybody_pos = 2.0
            tracking_keybody_pos_global = 2.0
            alive = 0.5
            feet_slip = -0.1
            feet_contact_forces = -5e-4
            feet_stumble = -1.25
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01
            feet_air_time = 5.0
            ang_vel_xy = -0.01
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 500
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        
        # =====================================================
        # STRICT TERMINATION (key change for this experiment)
        # =====================================================
        termination_roll = 1.0    # was 4.0 (229° → 57°)
        termination_pitch = 1.0   # was 4.0 (229° → 57°)
        root_height_diff_threshold = 0.3

    class evaluations:
        tracking_joint_dof = True
        tracking_joint_vel = True
        tracking_root_translation = True
        tracking_root_rotation = True
        tracking_root_vel = True
        tracking_root_ang_vel = True
        tracking_keybody_pos = True
        tracking_root_pose_delta_local = True
        tracking_root_rotation_delta_local = True

    class domain_rand:
        domain_rand_general = True
        
        randomize_gravity = True
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        
        randomize_friction = True
        friction_range = [0.1, 2.]
        
        randomize_base_mass = True
        added_mass_range = [-3., 3]
        
        randomize_base_com = True
        added_com_range = [-0.05, 0.05]
        
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 1.0
        
        push_end_effector = False
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 10.0

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]

        action_delay = True
        action_buf_len = 8
    
    class noise(HumanoidMimicCfg.noise):
        add_noise = True
        noise_increasing_steps = 50_000
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.1
        
    class motion(HumanoidMimicCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        reset_consec_frames = 30
        key_bodies = ["left_rubber_hand", "right_rubber_hand", "left_ankle_roll_link", 
                      "right_ankle_roll_link", "left_knee_link", "right_knee_link", 
                      "left_elbow_link", "right_elbow_link", "head_mocap"]
        upper_key_bodies = ["left_rubber_hand", "right_rubber_hand", "left_elbow_link", 
                           "right_elbow_link", "head_mocap"]
        sample_ratio = 1.0
        motion_smooth = True
        motion_decompose = False

        # Use the same walking-focused dataset
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/walking_focused.yaml"


class G1MimicPrivStrictCfgPPO(HumanoidMimicCfgPPO):
    """PPO config with higher entropy for more exploration."""
    
    seed = 1
    
    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunnerMimic'
        max_iterations = 1_000_002

        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
    
    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        
        # =====================================================
        # HIGHER ENTROPY (key change for this experiment)
        # =====================================================
        entropy_coef = 0.01  # was 0.005 (2x more exploration)
    
    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128


