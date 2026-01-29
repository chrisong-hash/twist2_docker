"""
G1 Mimic Distill Config V7 - RANDOM MOTION FREEZE + ANTI-SPASM

Key features:
1. Random Motion Freeze - teaches stability at any pose
2. Anti-spasm (jerk penalty) - prevents oscillation without killing responsiveness
3. Standing still rewards - encourages holding position when target is stationary
4. NO future observations for student - deployment-friendly approach

During training:
- ~1 freeze per episode, lasting 0.5-1.5 seconds
- Robot is rewarded for maintaining the "shape" (±10% joint tolerance)
- Robot gets bonus for being stable during freeze (low velocity)
- Jerk penalty catches back-and-forth oscillation, not smooth fast movements

Key insight: The freeze mechanism teaches "when to stop".
Jerk penalty teaches "how to move smoothly".
Together they create responsive but stable motion.

Benefits:
- Learns to be stable at ANY pose, not just during transitions
- Removes reliance on momentum-based motion
- Prevents spasming/oscillation (from V6.2 learnings)
- Responsive tracking during motion + stable during freeze
- Student uses history only (no future obs) for reliable deployment
"""

from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class G1MimicPrivCfgV7(HumanoidMimicCfg):
    
    # ================================================================
    # V7 SPECIAL: Freeze configuration
    # ================================================================
    class freeze:
        # Probability of triggering freeze per step
        # 0.002 per step * 500 steps/episode = ~63% chance of at least one freeze per episode
        # Average ~1 freeze per episode
        probability = 0.002
        
        # Freeze duration range (seconds)
        duration_min = 0.5
        duration_max = 1.5
        
        # Shape tolerance: 10% of joint range
        # If joint is within ±10% of target, it's considered "matching"
        shape_tolerance = 0.1
        
        # Stability bonus during freeze (multiplier for freeze_stability reward)
        # Set equal to tracking reward scale (~1.0)
        stability_bonus = 1.0
    
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
     
        dof_err_w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    ]
        
        global_obs = False
    
    class terrain(HumanoidMimicCfg.terrain):
        mesh_type = 'plane'
        height = [0, 0.00]
        horizontal_scale = 0.1
    
    class init_state(HumanoidMimicCfg.init_state):
        pos = [0, 0, 1.0]
        default_joint_angles = {
            'left_hip_pitch_joint': -0.2,
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.4,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            
            'right_hip_pitch_joint': -0.2,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.4,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,
            
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.4,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.2,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': -0.4,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 1.2,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        }
    
    class control(HumanoidMimicCfg.control):
        stiffness = {'hip_yaw': 100,
                    'hip_roll': 100,
                    'hip_pitch': 100,
                    'knee': 150,
                    'ankle': 40,
                    'waist': 150,
                    'shoulder': 40,
                    'elbow': 40,
                    'wrist': 40,
                    }
        damping = {  'hip_yaw': 2,
                    'hip_roll': 2,
                    'hip_pitch': 2,
                    'knee': 4,
                    'ankle': 2,
                    'waist': 4,
                    'shoulder': 5,
                    'elbow': 5,
                    'wrist': 5,
                    }

        action_scale = 0.5
        decimation = 10
    
    class sim(HumanoidMimicCfg.sim):
        dt = 0.002
        
    class normalization(HumanoidMimicCfg.normalization):
        clip_actions = 5.0
    
    class asset(HumanoidMimicCfg.asset):
        file = f'{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_custom_collision_29dof.urdf'
        
        torso_name: str = 'pelvis'
        chest_name: str = 'imu_in_torso'
        thigh_name: str = 'hip'
        shank_name: str = 'knee'
        foot_name: str = 'ankle_roll_link'
        waist_name: list = ['torso_link', 'waist_roll_link', 'waist_yaw_link']
        upper_arm_name: str = 'shoulder_roll_link'
        lower_arm_name: str = 'elbow_link'
        hand_name: list = ['right_rubber_hand', 'left_rubber_hand']

        feet_bodies = ['left_ankle_roll_link', 'right_ankle_roll_link']
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["shoulder", "elbow", "hip", "knee"]
        terminate_after_contacts_on = []
        
        dof_armature = [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2 + [0.0103] * 3 + \
            [0.003597, 0.003597, 0.003597, 0.003597, 0.003597, 0.00425, 0.00425] * 2
        
        collapse_fixed_joints = False
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = []
        regularization_scale = 1.0
        regularization_scale_range = [0.8,2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        
        class scales:
            # ================================================================
            # V7: Rewards for motion freeze training
            # ================================================================
            # Use V4-style balanced rewards as base
            # The freeze mechanism adds implicit stability training
            # ================================================================
            
            # === TRACKING REWARDS ===
            tracking_joint_dof = 1.5
            tracking_joint_vel = 0.2
            tracking_root_translation_z = 1.0
            tracking_root_rotation = 2.0
            tracking_root_linear_vel = 2.5
            tracking_root_angular_vel = 2.0
            tracking_keybody_pos = 2.5
            tracking_keybody_pos_global = 3.0
            
            # === SURVIVAL ===
            alive = 0.3
            
            # === GAIT ===
            feet_air_time = 5.0
            
            # === PENALTIES (reduced action_rate for responsiveness) ===
            # The freeze mechanism teaches "when to stop" - we don't need
            # action_rate to force smoothness. Let the robot move freely.
            orientation = -2.0
            dof_pos_limits = -2.5
            action_rate = -0.01  # MINIMAL - just prevent extreme snapping
            ang_vel_xy = -0.05   # REDUCED - allow natural walking wobble
            feet_stumble = -1.25
            dof_torque_limits = -1.0
            feet_slip = -0.1
            feet_contact_forces = -5e-4
            dof_vel = -5e-5      # REDUCED - allow faster joint movement
            dof_acc = -2e-8      # REDUCED - allow faster acceleration
            ankle_dof_acc = -4e-8
            ankle_dof_vel = -1e-4
            
            # Note: freeze_stability reward is added separately in the environment
            # with scale = cfg.freeze.stability_bonus (1.0)
            
            # === STANDING STILL / DEFAULT POSE ===
            # When reference motion has low velocity, reward staying still
            standing_still = 2.0  # Reward low actual velocity when target is stationary
            default_pose_tracking = 1.5  # Reward being at default joint angles when idle
            
            # === ANTI-SPASM (Jerk penalty) ===
            # Penalizes rapid DIRECTION changes (oscillation) without penalizing fast smooth movements
            # This is different from action_rate which penalizes ALL fast changes
            # Jerk = change in acceleration = (a_t - 2*a_{t-1} + a_{t-2})
            action_jerk = -0.5  # Penalize rapid direction changes (oscillation/spasming)


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
        
        # Keep termination lenient for teleop
        termination_roll = 4.0
        termination_pitch = 4.0
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
        
        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        
        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]
        
        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-3., 3]
        
        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.05, 0.05]
        
        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 1.0
        
        push_end_effector = (False and domain_rand_general)
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 10.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]

        action_delay = (True and domain_rand_general)
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
        key_bodies = ["left_rubber_hand", "right_rubber_hand", "left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link", "left_elbow_link", "right_elbow_link", "head_mocap"]
        upper_key_bodies = ["left_rubber_hand", "right_rubber_hand", "left_elbow_link", "right_elbow_link", "head_mocap"]
        sample_ratio = 1.0
        motion_smooth = True
        motion_decompose = False

        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/walking_focused.yaml"



class G1MimicPrivCfgPPOV7(HumanoidMimicCfgPPO):
    seed = 1
    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunnerMimic'
        max_iterations = 20_000  # Capped at 20k like V6

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
        entropy_coef = 0.005
    
    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128

