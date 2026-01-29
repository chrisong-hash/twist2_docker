"""
V6.2 Student Config - Optimized for 0.5s max deployment lag

Key changes from V6:
1. Future sight: 3 frames up to 0.5s [5, 15, 25] = [0.1s, 0.3s, 0.5s]
2. Higher dagger_coef: 0.5 → 0.3 (more teacher imitation)
3. Lower action std: More stable, less exploration

Observation size: 1502 dims (127*11 + 105 future)
"""

from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO
from legged_gym.envs.g1.g1_mimic_distill_config_v6 import G1MimicPrivCfgV6, G1MimicPrivCfgPPOV6
from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


# Future frames within 0.5s at 50Hz (step * 0.02s = time)
# 3 frames spread across 0.5s for lower compute overhead
# Step 5 = 0.1s, Step 15 = 0.3s, Step 25 = 0.5s
TAR_MOTION_STEPS_FUTURE_V6_2 = [5, 15, 25]


class G1MimicStuFutureCfgV6_2(G1MimicPrivCfgV6):
    """Student config with 0.5s future sight - inherits V6 rewards"""
    
    class env(G1MimicPrivCfgV6.env):
        obs_type = 'student_future'
        
        # Current frame only for mimic_obs (future handled separately)
        tar_motion_steps = [0]
        
        # Future motion frames (up to 0.5s = 25 steps at 50Hz)
        tar_motion_steps_future = TAR_MOTION_STEPS_FUTURE_V6_2
        
        # History length (must be 1, 10, 20, or 50 - HistoryEncoder constraint)
        history_len = 10
        
        # Observation dimensions
        n_mimic_obs_single = 6 + 29  # root_vel_xy(2) + root_pos_z(1) + roll_pitch(2) + yaw_ang_vel(1) + dof_pos(29)
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single  # 35
        n_proprio = G1MimicPrivCfg.env.n_proprio  # 92
        
        # Future observation dimensions
        n_future_obs_single = 6 + 29  # Same structure as mimic_obs
        n_future_obs = len(tar_motion_steps_future) * n_future_obs_single  # 3 * 35 = 105
        
        # Total observation size
        n_obs_single = n_mimic_obs + n_proprio  # 127
        num_observations = n_obs_single * (history_len + 1) + n_future_obs  # 127*11 + 105 = 1502


class G1MimicStuFutureCfgPPOV6_2(HumanoidMimicCfgPPO):
    """PPO config for V6.2 student with DAgger"""
    seed = 1
    
    class teachercfg(G1MimicPrivCfgPPOV6):
        """Use V6 teacher"""
        pass
    
    class runner:
        # Policy/algorithm selection for DAgger+PPO
        policy_class_name = 'ActorCriticFuture'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        
        # Training iterations
        num_steps_per_env = 24
        max_iterations = 30_000
        warm_iters = 200
        
        # Logging
        save_interval = 500
        experiment_name = 'student_v6_2'
        run_name = ''
        
        # Load/resume
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        # Teacher reference (V6) - MUST match actual path: logs/h1/responsive_v6/
        teacher_experiment_name = 'responsive_v6'
        teacher_proj_name = 'h1'
        teacher_checkpoint = 20000
        eval_student = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        # Standard PPO params
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        entropy_coef = 0.003  # Slightly lower for stability
        
        # Higher dagger coefficient for more teacher imitation
        dagger_coef = 0.5        # was 0.2 - start with strong teacher guidance
        dagger_coef_min = 0.3    # was 0.1 - maintain teacher influence
        dagger_coef_anneal_steps = 20000  # Slower annealing
        
        # Action std schedule - start stable, allow some exploration later
        std_schedule = [0.8, 0.3, 5000, 2000]  # was [1.0, 0.4, ...]

    class policy(HumanoidMimicCfgPPO.policy):
        # Lower action std for more stable outputs
        action_std = [0.5] * 12 + [0.3] * 3 + [0.4] * 14  # was [0.7, 0.4, 0.5]
        init_noise_std = 0.8  # was 1.0
        
        # Context length matches history
        obs_context_len = 11  # history_len + 1
        
        # Network architecture
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        
        # Motion/future/history encoding
        motion_latent_dim = 128
        future_latent_dim = 128
        history_latent_dim = 128
        
        # Future encoder specific
        num_future_steps = len(TAR_MOTION_STEPS_FUTURE_V6_2)  # 3
        num_future_observations = 3 * 35  # 105

