"""
V6.1.1 Student Config - Pure L2 Behavior Cloning (No KL Divergence)

Key difference from V6.2:
- Uses DAggerL2PPO algorithm (L2 loss on actions) instead of DaggerPPO (KL divergence)
- L2 loss: (student_action - teacher_action)^2
- KL divergence matches distributions, L2 matches raw actions

Rationale:
- KL divergence might not work well when student has information asymmetry
- L2 loss directly minimizes action error regardless of distribution shape
- Simpler loss function might lead to better imitation

Observation size: 1502 dims (127*11 + 105 future) - same as V6.2
"""

from legged_gym.envs.g1.g1_mimic_future_config_v6_2 import (
    G1MimicStuFutureCfgV6_2, 
    G1MimicStuFutureCfgPPOV6_2,
    TAR_MOTION_STEPS_FUTURE_V6_2
)
from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO


class G1MimicStuFutureCfgV6_1_1(G1MimicStuFutureCfgV6_2):
    """V6.1.1: Same observation structure as V6.2, but uses pure L2 loss"""
    pass  # Inherits everything from V6.2


class G1MimicStuFutureCfgPPOV6_1_1(G1MimicStuFutureCfgPPOV6_2):
    """PPO config for V6.1.1 student with PURE L2 LOSS (not KL divergence)"""
    
    class runner(G1MimicStuFutureCfgPPOV6_2.runner):
        # KEY CHANGE: Use DAggerL2PPO instead of DaggerPPO
        algorithm_class_name = 'DAggerL2PPO'  # Pure L2 loss, not KL divergence
        
        # Training iterations
        max_iterations = 30_000
        
        # Experiment name
        experiment_name = 'student_v6_1_1'
        
        # Teacher reference (V6) - same as V6.2
        teacher_experiment_name = 'responsive_v6'
        teacher_proj_name = 'h1'
        teacher_checkpoint = 20000

    class algorithm(G1MimicStuFutureCfgPPOV6_2.algorithm):
        # Same hyperparameters as V6.2, just different loss function
        # The algorithm class itself handles the L2 vs KL difference
        pass
