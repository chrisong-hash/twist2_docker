"""
V6.3 Student Config - Anti-Spasm Jerk Penalty

Key changes from V6.2:
1. Added action_jerk penalty (-0.1) to reduce oscillation/spasming
2. Teacher is smooth because it learned action_rate during RL training
3. Student via DAgger only learns output distribution, not temporal smoothness
4. Jerk penalty explicitly teaches the student to be smooth

Philosophy:
- KL divergence matches teacher OUTPUT at each timestep
- But doesn't capture TEMPORAL correlation (smoothness between timesteps)
- Jerk penalty = (a_t - 2*a_{t-1} + a_{t-2})^2 catches oscillation
- Unlike action_rate, jerk allows fast smooth movements but penalizes back-and-forth

Observation size: 1502 dims (127*11 + 105 future) - unchanged from V6.2
"""

from legged_gym.envs.g1.g1_mimic_future_config_v6_2 import (
    G1MimicStuFutureCfgV6_2, 
    G1MimicStuFutureCfgPPOV6_2,
    TAR_MOTION_STEPS_FUTURE_V6_2
)
from legged_gym.envs.g1.g1_mimic_distill_config_v6 import G1MimicPrivCfgV6


class G1MimicStuFutureCfgV6_3(G1MimicStuFutureCfgV6_2):
    """V6.3: Student config with jerk penalty for anti-spasm behavior"""
    
    class env(G1MimicStuFutureCfgV6_2.env):
        # Inherit all V6.2 env settings
        pass
    
    class rewards(G1MimicPrivCfgV6.rewards):
        """Override rewards to add jerk penalty"""
        
        class scales(G1MimicPrivCfgV6.rewards.scales):
            # ================================================================
            # V6.3: ADD JERK PENALTY FOR STUDENT SMOOTHNESS
            # ================================================================
            # The teacher is smooth because it was trained with action_rate
            # penalty during RL. The student via DAgger only learns to match
            # the teacher's output distribution, not temporal smoothness.
            #
            # Jerk = second derivative of action = oscillation detection
            # This specifically catches the "spasming" behavior seen in V6.2
            # ================================================================
            
            action_jerk = -0.1  # Penalize oscillation (negative = penalty)


class G1MimicStuFutureCfgPPOV6_3(G1MimicStuFutureCfgPPOV6_2):
    """PPO config for V6.3 student with DAgger + jerk penalty"""
    seed = 1
    
    class runner(G1MimicStuFutureCfgPPOV6_2.runner):
        # Same as V6.2 but different experiment name
        experiment_name = 'student_v6_3'
        
        # Training iterations - same as V6.2
        max_iterations = 30_000
        
    class algorithm(G1MimicStuFutureCfgPPOV6_2.algorithm):
        # Inherit V6.2 algorithm settings
        pass
    
    class policy(G1MimicStuFutureCfgPPOV6_2.policy):
        # Inherit V6.2 policy settings
        pass

