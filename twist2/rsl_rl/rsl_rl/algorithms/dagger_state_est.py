"""
V6.4: DAgger with State Estimation

Training approach:
1. Primary loss: L2(student_action, teacher_action)
   - Gradient flows: action_loss → actor → estimated_latent → state_estimator
   - State estimator learns implicitly what information is needed

2. Optional auxiliary loss: L2(estimated_latent, teacher_priv_latent)
   - Explicit supervision if teacher's privileged latent is available
   - Helps state estimator learn faster

The state estimator learns to predict privileged information from history,
enabling sim-to-real transfer where privileged info isn't available.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from rsl_rl.storage import RolloutStorage
from termcolor import cprint


def cosine_decay_weight(init_weight, step, total_steps):
    return init_weight * (0.5 * (1 + math.cos(math.pi * step / total_steps)))


class DAggerStateEst:
    """
    DAgger algorithm with State Estimation for Sim-to-Real.
    
    Uses L2 loss on actions + optional auxiliary loss on latent prediction.
    """
    
    def __init__(self,
                 env,
                 actor_critic,           # ActorCriticStateEst
                 teacher_actor_critic,   # Teacher policy (privileged)
                 teacher_loaded=False,
                 dagger_coef=1.0,        # Weight for action imitation loss
                 dagger_coef_anneal_steps=30000,
                 dagger_coef_min=0.1,
                 latent_loss_coef=0.0,   # Weight for auxiliary latent loss (0 = disabled)
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 dagger_update_freq=20,
                 priv_reg_coef_schedual=[0, 0, 0],
                 grad_penalty_coef_schedule=[0.0, 0.0, 10, 10],
                 std_schedule=[1.0, 1.0, 10, 10],
                 num_hist=10,
                 **kwargs):
        
        self.env = env
        self.device = device
        self.num_hist = num_hist
        
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        
        # Networks
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.teacher_actor_critic = teacher_actor_critic
        self.teacher_actor_critic.to(self.device)
        self.teacher_loaded = teacher_loaded
        
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        
        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # DAgger parameters
        self.dagger_coef = dagger_coef
        self.dagger_coef_init = dagger_coef
        self.dagger_coef_anneal_steps = dagger_coef_anneal_steps
        self.dagger_coef_min = dagger_coef_min
        
        # Auxiliary latent loss
        self.latent_loss_coef = latent_loss_coef
        
        cprint(f"[DAggerStateEst] State Estimation + L2 Action Loss", "cyan")
        cprint(f"[DAggerStateEst] dagger_coef: {dagger_coef}, latent_loss_coef: {latent_loss_coef}", "green")
        
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        self.gradient_penalty_coef_schedule = grad_penalty_coef_schedule
        self.counter = 0
        
        self.fix_std = self.actor_critic.if_fix_std()
        self.std_schedule = std_schedule
    
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, 
            actor_obs_shape, critic_obs_shape, action_shape, 
            self.device
        )
    
    def test_mode(self):
        self.actor_critic.eval()
    
    def train_mode(self):
        self.actor_critic.train()
    
    def act(self, obs, critic_obs, info, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # Get action from student (with state estimation)
        self.transition.actions = self.actor_critic.act(obs).detach()
        
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
            )
        
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
        
        return rewards
    
    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_l2_loss = 0.0
        mean_latent_loss = 0.0
        
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for sample in generator:
            obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
            
            # Forward pass through student
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0] if hid_states_batch else None)
            
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1] if hid_states_batch else None)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            
            # Adaptive learning rate based on KL
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + 
                        (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5,
                        axis=-1
                    )
                    kl_mean = torch.mean(kl)
                    
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate
            
            # PPO Surrogate Loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            
            # Value Loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            
            # Base PPO loss
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            
            ############################################################
            # L2 Action Imitation Loss (with State Estimation)
            ############################################################
            if self.dagger_coef > 0 and self.teacher_loaded:
                # Student action mean (from state estimator path)
                mu_batch_student = mu_batch
                
                # Teacher action mean (uses privileged observations)
                with torch.no_grad():
                    self.teacher_actor_critic.act(critic_obs_batch)
                    mu_batch_teacher = self.teacher_actor_critic.action_mean
                
                # L2 loss on actions
                l2_loss = (mu_batch_student - mu_batch_teacher).pow(2).mean()
                l2_loss_weighted = l2_loss * self.dagger_coef
                mean_l2_loss += l2_loss.item()
                
                loss += l2_loss_weighted
            
            ############################################################
            # Optional: Auxiliary Latent Prediction Loss
            ############################################################
            # This would require extracting teacher's internal latent representation
            # For now, we train the state estimator implicitly through action loss
            # If you have ground truth privileged latent, you can add:
            # latent_loss = (estimated_latent - teacher_latent).pow(2).mean()
            # loss += latent_loss * self.latent_loss_coef
            
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            
            # Update action std if scheduled
            if self.fix_std:
                std_stage = min(max((self.counter - self.std_schedule[2]), 0) / self.std_schedule[3], 1)
                std_coef = std_stage * (self.std_schedule[1] - self.std_schedule[0]) + self.std_schedule[0]
                self.actor_critic.update_std(std_coef)
        
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_l2_loss /= num_updates
        
        self.counter += 1
        self.storage.clear()
        
        # Anneal dagger coefficient
        if self.counter < self.dagger_coef_anneal_steps:
            self.dagger_coef = cosine_decay_weight(self.dagger_coef_init, self.counter, self.dagger_coef_anneal_steps)
            self.dagger_coef = max(self.dagger_coef, self.dagger_coef_min)
        else:
            self.dagger_coef = self.dagger_coef_min
        
        # Return losses (7th value is imitation loss)
        return mean_value_loss, mean_surrogate_loss, 0, 0, 0, 0, mean_l2_loss
    
    def update_counter(self):
        self.counter += 1


