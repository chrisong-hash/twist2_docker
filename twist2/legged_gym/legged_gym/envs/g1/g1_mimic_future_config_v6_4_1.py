"""
V6.4.1 Student Config - Privileged Information Predictor

Key Concept:
  - Student ACTOR has the SAME input structure as Teacher!
  - A PREDICTOR network estimates the missing privileged info
  - Student actor can be initialized from teacher weights

Architecture:
  Student Obs (proprio + history) → [Predictor] → predicted_priv
  [predicted_priv + proprio] = full_obs (same as teacher!)
  full_obs → [Actor (teacher architecture)] → action

Teacher obs structure (1734 dims):
  - priv_mimic_obs: 1540 dims (20 future frames × 77 features)
  - proprio: 92 dims
  - priv_info: 102 dims (friction, mass, contacts, etc.)

Student obs (1397 dims):
  - Current: mimic(35) + proprio(92) = 127
  - History: 10 × 127 = 1270
  - Total: 1397

Predictor predicts:
  - priv_mimic_obs (1540) + priv_info (102) = 1642 dims
"""

from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO
from legged_gym.envs.g1.g1_mimic_distill_config_v6 import G1MimicPrivCfgV6, G1MimicPrivCfgPPOV6
from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class G1MimicStuPrivPredCfgV6_4_1(G1MimicPrivCfgV6):
    """
    Student config where student actor matches teacher's input structure.
    A predictor fills in the missing privileged information.
    """
    
    class env(G1MimicPrivCfgV6.env):
        obs_type = 'student'  # Student observation type
        
        # Current frame for student's mimic obs
        tar_motion_steps = [0]
        
        # History length
        history_len = 10
        
        # === Student's observable dimensions ===
        n_mimic_obs_single = 6 + 29  # root info + dof_pos = 35
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single  # 35
        n_proprio = G1MimicPrivCfg.env.n_proprio  # 92
        
        # Student single obs = mimic + proprio
        n_obs_single = n_mimic_obs + n_proprio  # 127
        
        # Total student observations (for predictor input)
        num_observations = n_obs_single * (history_len + 1)  # 1397
        
        # === Teacher's privileged dimensions (what predictor targets) ===
        # These are inherited from parent but listed for clarity
        # n_priv_mimic_obs = 20 * 77 = 1540
        # n_priv_info = 102
        # num_privileged_obs = 1540 + 92 + 102 = 1734


class G1MimicStuPrivPredCfgPPOV6_4_1(HumanoidMimicCfgPPO):
    """PPO config for V6.4.1 - Student with Privileged Predictor"""
    seed = 1
    
    class teachercfg(G1MimicPrivCfgPPOV6):
        """Use V6 teacher"""
        pass
    
    class runner:
        # KEY: Use privileged predictor architecture
        policy_class_name = 'ActorCriticPrivPredictor'
        algorithm_class_name = 'DAggerPrivPredictor'
        runner_class_name = 'OnPolicyDaggerRunner'
        
        # Training iterations
        num_steps_per_env = 24
        max_iterations = 30_000
        warm_iters = 200
        
        # Logging
        save_interval = 500
        experiment_name = 'student_v6_4_1'
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
        
        # Loss weights
        dagger_coef = 1.0        # Action imitation weight
        priv_pred_coef = 1.0     # Privileged prediction weight (supervised!)
        dagger_coef_min = 0.3
        dagger_coef_anneal_steps = 20000
        
        # Std schedule
        std_schedule = [0.8, 0.3, 5000, 2000]

    class policy:
        # Predictor network (student obs → privileged info)
        predictor_hidden_dims = [512, 512, 256]  # Larger for 1642-dim output
        history_latent_dim = 256  # Encode history well
        
        # Actor/Critic (SAME as teacher for actor!)
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        
        # Activation and std
        activation = 'silu'
        fix_action_std = True
        action_std = [0.5] * 12 + [0.3] * 3 + [0.4] * 14  # 29 actions
        init_noise_std = 0.8


