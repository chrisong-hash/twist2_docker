"""
G1 Mimic Distill with Random Motion Freeze

This environment extends G1MimicDistill to add random motion freezing:
- At random times, the reference motion freezes for 0.5-1.5 seconds
- During freeze, robot is rewarded for maintaining the "shape" (±10% joint tolerance)
- Teaches the robot that it can stop at any point and remain stable

This addresses the stuttering problem by teaching:
- "You can stop anytime and that's okay"
- "Holding a pose is as important as transitioning"
- "Don't rely on momentum - be stable at every frame"
"""

import torch
import numpy as np
from .g1_mimic_distill import G1MimicDistill


class G1MimicDistillFreeze(G1MimicDistill):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Freeze parameters from config
        self._freeze_probability = cfg.freeze.probability  # e.g., 0.002 per step = ~10% chance per 5s episode
        self._freeze_duration_min = cfg.freeze.duration_min  # 0.5s
        self._freeze_duration_max = cfg.freeze.duration_max  # 1.5s
        self._freeze_shape_tolerance = cfg.freeze.shape_tolerance  # 0.1 = 10% of joint range
        self._freeze_stability_bonus = cfg.freeze.stability_bonus  # bonus for low velocity during freeze
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_freeze_buffers()
        
    def _init_freeze_buffers(self):
        """Initialize buffers for motion freeze tracking."""
        # Whether each env is currently frozen
        self._freeze_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Remaining freeze steps for each env
        self._freeze_remaining_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        
        # The motion time at which we froze (to keep reference constant)
        self._freeze_motion_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
        # The reference dof_pos at freeze time (for shape matching)
        self._freeze_ref_dof_pos = torch.zeros_like(self.dof_pos)
        
        # Statistics
        self._total_freeze_count = 0
        self._freeze_success_count = 0  # Survived the freeze
        
    def _get_motion_times(self, env_ids=None):
        """Override to return frozen time for frozen envs."""
        if env_ids is None:
            # Normal time calculation
            motion_times = self.episode_length_buf * self.dt + self._motion_time_offsets
            # Override with frozen time for frozen envs
            motion_times = torch.where(self._freeze_active, self._freeze_motion_time, motion_times)
        else:
            motion_times = self.episode_length_buf[env_ids] * self.dt + self._motion_time_offsets[env_ids]
            # Override with frozen time for frozen envs
            motion_times = torch.where(
                self._freeze_active[env_ids], 
                self._freeze_motion_time[env_ids], 
                motion_times
            )
        return motion_times
    
    def _maybe_trigger_freeze(self):
        """Randomly trigger freeze for non-frozen envs."""
        # Only trigger for envs that are not already frozen
        can_freeze = ~self._freeze_active
        
        # Random trigger based on probability
        trigger = torch.rand(self.num_envs, device=self.device) < self._freeze_probability
        new_freeze = can_freeze & trigger
        
        if new_freeze.any():
            new_freeze_ids = new_freeze.nonzero(as_tuple=False).flatten()
            
            # Set freeze active
            self._freeze_active[new_freeze_ids] = True
            
            # Random duration between min and max
            duration_range = self._freeze_duration_max - self._freeze_duration_min
            random_duration = self._freeze_duration_min + torch.rand(len(new_freeze_ids), device=self.device) * duration_range
            freeze_steps = (random_duration / self.dt).int()
            self._freeze_remaining_steps[new_freeze_ids] = freeze_steps
            
            # Store the current motion time as frozen time
            current_motion_times = self.episode_length_buf[new_freeze_ids] * self.dt + self._motion_time_offsets[new_freeze_ids]
            self._freeze_motion_time[new_freeze_ids] = current_motion_times
            
            # Store the current reference dof_pos for shape matching
            self._freeze_ref_dof_pos[new_freeze_ids] = self._ref_dof_pos[new_freeze_ids].clone()
            
            self._total_freeze_count += len(new_freeze_ids)
    
    def _update_freeze_state(self):
        """Decrement freeze counters and end freeze when done."""
        # Decrement remaining steps for frozen envs
        self._freeze_remaining_steps[self._freeze_active] -= 1
        
        # End freeze when counter reaches 0
        freeze_ended = self._freeze_active & (self._freeze_remaining_steps <= 0)
        if freeze_ended.any():
            ended_ids = freeze_ended.nonzero(as_tuple=False).flatten()
            self._freeze_active[ended_ids] = False
            self._freeze_remaining_steps[ended_ids] = 0
            
            # Count as success if robot didn't fall during freeze
            # (reset_buf would be set if robot fell)
            survived = ~self.reset_buf[ended_ids]
            self._freeze_success_count += survived.sum().item()
    
    def _reset_freeze_state(self, env_ids):
        """Reset freeze state for reset envs."""
        self._freeze_active[env_ids] = False
        self._freeze_remaining_steps[env_ids] = 0
        self._freeze_motion_time[env_ids] = 0
        self._freeze_ref_dof_pos[env_ids] = 0
    
    def reset_idx(self, env_ids):
        """Override to also reset freeze state."""
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            self._reset_freeze_state(env_ids)
    
    def post_physics_step(self):
        """Override to add freeze logic."""
        # Update freeze state (decrement counters, end freezes)
        self._update_freeze_state()
        
        # Call parent post_physics_step
        super().post_physics_step()
        
        # Maybe trigger new freezes (after parent to avoid interfering with reset)
        self._maybe_trigger_freeze()
    
    def _reward_tracking_joint_dof(self):
        """Override to use shape matching during freeze."""
        if not self._freeze_active.any():
            # No freeze active, use normal tracking
            return super()._reward_tracking_joint_dof()
        
        # Calculate normal tracking reward
        normal_reward = super()._reward_tracking_joint_dof()
        
        # Calculate shape reward for frozen envs (looser tolerance)
        # Shape matching: within ±10% of joint range
        dof_range = self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        tolerance = self._freeze_shape_tolerance * dof_range  # e.g., 10% of range
        
        dof_diff = torch.abs(self._freeze_ref_dof_pos - self.dof_pos)
        within_tolerance = (dof_diff <= tolerance).float()
        shape_reward = torch.mean(within_tolerance, dim=-1)  # 0-1 based on how many joints are within tolerance
        
        # Use shape reward for frozen envs, normal reward for others
        reward = torch.where(self._freeze_active, shape_reward, normal_reward)
        return reward
    
    def _reward_freeze_stability(self):
        """Bonus reward for being stable during freeze (low velocity)."""
        if not self._freeze_active.any():
            return torch.zeros(self.num_envs, device=self.device)
        
        # Reward for low joint velocity during freeze
        joint_vel_magnitude = torch.norm(self.dof_vel, dim=-1)
        # Exponential reward: high when velocity is low
        stability_reward = torch.exp(-0.1 * joint_vel_magnitude)
        
        # Also reward low base velocity
        base_vel_magnitude = torch.norm(self.base_lin_vel, dim=-1)
        base_stability = torch.exp(-2.0 * base_vel_magnitude)
        
        combined_stability = 0.5 * stability_reward + 0.5 * base_stability
        
        # Only apply to frozen envs
        reward = torch.where(self._freeze_active, combined_stability, torch.zeros_like(combined_stability))
        return reward
    
    def compute_reward(self):
        """Override to add freeze stability reward."""
        # Call parent reward computation
        super().compute_reward()
        
        # Add freeze stability bonus if configured
        if self._freeze_stability_bonus > 0:
            freeze_stability_rew = self._reward_freeze_stability() * self._freeze_stability_bonus
            self.rew_buf += freeze_stability_rew
            
            # Log it
            if hasattr(self, 'episode_sums') and 'freeze_stability' not in self.episode_sums:
                self.episode_sums['freeze_stability'] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            if hasattr(self, 'episode_sums'):
                self.episode_sums['freeze_stability'] += freeze_stability_rew
    
    def get_freeze_stats(self):
        """Return freeze statistics for logging."""
        return {
            'freeze_total': self._total_freeze_count,
            'freeze_success': self._freeze_success_count,
            'freeze_success_rate': self._freeze_success_count / max(1, self._total_freeze_count),
            'currently_frozen': self._freeze_active.sum().item(),
        }

