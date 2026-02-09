"""
V6.5 Student Config - Default Architecture + Jerk Penalty

Key concept:
- Uses the ORIGINAL/DEFAULT student architecture (1432 dims, no future obs)
- Adds jerk penalty like V6.3 to reduce oscillation
- Best of both: Original's stability + V6.3's smoothness

Observation structure (1432 dims):
- Current mimic obs: 35 dims (tar_motion_steps = [1])
- Current proprio: 92 dims
- History: 10 × (35 + 92) = 1270 dims
- NO future observations

This is the same as twist2_1017_25k but with jerk penalty during training.
"""

from legged_gym.envs.g1.g1_mimic_distill_config import (
    G1MimicStuRLCfg,
    G1MimicStuRLCfgDAgger,
    G1MimicPrivCfg,
)
from legged_gym.envs.g1.g1_mimic_distill_config_v6 import G1MimicPrivCfgV6


class G1MimicStuCfgV6_5(G1MimicStuRLCfg):
    """V6.5: Default student with jerk penalty"""
    
    class env(G1MimicStuRLCfg.env):
        # Same as default student - no future obs
        pass
    
    class rewards(G1MimicPrivCfgV6.rewards):
        """Override rewards to add jerk penalty like V6.3"""
        
        class scales(G1MimicPrivCfgV6.rewards.scales):
            # ================================================================
            # V6.5: ADD JERK PENALTY FOR SMOOTHNESS
            # ================================================================
            # The original twist2 student sometimes has wrist jitter.
            # Adding jerk penalty encourages smoother actions.
            # This is the same penalty used in V6.3.
            # ================================================================
            
            action_jerk = -0.1  # Penalize oscillation (negative = penalty)


class G1MimicStuCfgPPOV6_5(G1MimicStuRLCfgDAgger):
    """PPO/DAgger config for V6.5 student"""
    seed = 1
    
    class runner(G1MimicStuRLCfgDAgger.runner):
        # Use same policy/algorithm as default student
        policy_class_name = 'ActorCriticTeleop'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        
        # Training iterations
        num_steps_per_env = 24
        max_iterations = 30_000
        warm_iters = 200
        
        # Logging - V6.5 specific
        save_interval = 500
        experiment_name = 'student_v6_5'
        run_name = ''
        
        # Load/resume
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        # Teacher reference (V6) - same as V6.3
        teacher_experiment_name = 'responsive_v6'
        teacher_proj_name = 'h1'
        teacher_checkpoint = 20000
        eval_student = False
        
    class algorithm(G1MimicStuRLCfgDAgger.algorithm):
        # Same DAgger settings as default
        pass
    
    class policy(G1MimicStuRLCfgDAgger.policy):
        # Same policy architecture as default
        pass


