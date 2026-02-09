"""
V6.4 Student Config - State Estimation Approach

Key concept:
- Instead of providing future observations, the student learns to ESTIMATE
  the privileged information from history using a state estimator network.

Architecture:
  [proprio + history] → [State Estimator] → estimated_latent
  [proprio + history + estimated_latent] → [Actor] → action

Observation size: 1397 dims (127*11) - NO future obs
This is the same as default student but with state estimation architecture.

Comparison:
- V6.2: 1502 dims (127*11 + 105 future) - uses actual future frames
- V6.4: 1397 dims (127*11) - estimates latent from history only
"""

from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO
from legged_gym.envs.g1.g1_mimic_distill_config_v6 import G1MimicPrivCfgV6, G1MimicPrivCfgPPOV6
from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class G1MimicStuStateEstCfgV6_4(G1MimicPrivCfgV6):
    """
    Student config with State Estimation - NO future observations.
    The state estimator learns to predict privileged info from history.
    """
    
    class env(G1MimicPrivCfgV6.env):
        obs_type = 'student'  # Standard student (no future), but with state est architecture
        
        # Current frame only for mimic_obs
        tar_motion_steps = [0]
        
        # NO future observations - state estimator predicts latent instead
        tar_motion_steps_future = []  # Empty!
        
        # History length (must be 10 for HistoryEncoder)
        history_len = 10
        
        # Observation dimensions
        n_mimic_obs_single = 6 + 29  # root info + dof_pos = 35
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single  # 35
        n_proprio = G1MimicPrivCfg.env.n_proprio  # 92
        
        # NO future observations
        n_future_obs_single = 0
        n_future_obs = 0
        
        # Total observation size (same as default student)
        n_obs_single = n_mimic_obs + n_proprio  # 127
        num_observations = n_obs_single * (history_len + 1)  # 127*11 = 1397


class G1MimicStuStateEstCfgPPOV6_4(HumanoidMimicCfgPPO):
    """PPO config for V6.4 student with State Estimation"""
    seed = 1
    
    class teachercfg(G1MimicPrivCfgPPOV6):
        """Use V6 teacher"""
        pass
    
    class runner:
        # KEY CHANGE: Use state estimation architecture
        policy_class_name = 'ActorCriticStateEst'
        algorithm_class_name = 'DAggerStateEst'
        runner_class_name = 'OnPolicyDaggerRunner'
        
        # Training iterations
        num_steps_per_env = 24
        max_iterations = 30_000
        warm_iters = 200
        
        # Logging
        save_interval = 500
        experiment_name = 'student_v6_4'
        run_name = ''
        
        # Load/resume
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        # Teacher reference (V6)
        teacher_experiment_name = 'responsive_v6'
        teacher_proj_name = 'h1'
        teacher_checkpoint = 20000
        eval_student = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        # PPO params
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        entropy_coef = 0.003
        
        # DAgger params - same as V6.1.1 (L2 loss)
        dagger_coef = 1.0        # Strong action imitation
        dagger_coef_min = 0.3    # Maintain teacher influence
        dagger_coef_anneal_steps = 20000
        
        # Optional: auxiliary latent loss (0 = disabled, try 0.1-0.5 if enabled)
        latent_loss_coef = 0.0
        
        # Action std schedule
        std_schedule = [0.8, 0.3, 5000, 2000]

    class policy:
        # State estimator config
        latent_size = 64  # Size of estimated privileged latent
        estimator_hidden_dims = [256, 256, 128]  # State estimator network
        
        # History encoder
        history_latent_dim = 128
        
        # Actor network
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        
        # Activation and normalization
        activation = 'silu'
        layer_norm = True
        
        # Action noise
        fix_action_std = True
        action_std = [0.5] * 12 + [0.3] * 3 + [0.4] * 14  # 29 actions
        init_noise_std = 0.8


