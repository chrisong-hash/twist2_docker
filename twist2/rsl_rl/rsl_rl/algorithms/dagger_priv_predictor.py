"""
V6.4.1: DAgger with Privileged Information Predictor

Training has TWO explicit losses:
1. Prediction Loss: L2(predicted_priv, actual_priv) - supervised signal
2. Action Loss: L2(student_action, teacher_action) - imitation signal

The predictor learns to estimate privileged info, and the student actor
(which has the SAME structure as teacher) uses these predictions.

Key Benefit: Student actor can be initialized from teacher weights!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

from rsl_rl.storage import RolloutStorage
from termcolor import cprint


def cosine_decay_weight(init_weight, step, total_steps):
    return init_weight * (0.5 * (1 + math.cos(math.pi * step / total_steps)))


class DAggerPrivPredictor:
    """
    DAgger with explicit privileged information prediction.
    
    Losses:
    1. priv_loss: L2(predicted_priv, actual_priv)
    2. action_loss: L2(student_action, teacher_action)
    """
    
    def __init__(self,
                 env,
                 actor_critic,           # ActorCriticPrivPredictor
                 teacher_actor_critic,   # Teacher policy
                 teacher_loaded=False,
                 
                 # Loss coefficients
                 dagger_coef=1.0,        # Action imitation weight
                 priv_pred_coef=1.0,     # Privileged prediction weight
                 dagger_coef_anneal_steps=30000,
                 dagger_coef_min=0.3,
                 
                 # PPO params
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
                 
                 # Other
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
        
        # PPO params
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Loss weights
        self.dagger_coef = dagger_coef
        self.dagger_coef_init = dagger_coef
        self.dagger_coef_anneal_steps = dagger_coef_anneal_steps
        self.dagger_coef_min = dagger_coef_min
        self.priv_pred_coef = priv_pred_coef
        
        cprint(f"[DAggerPrivPredictor] Student actor = Teacher structure!", "cyan")
        cprint(f"[DAggerPrivPredictor] action_coef: {dagger_coef}, priv_pred_coef: {priv_pred_coef}", "green")
        
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        self.gradient_penalty_coef_schedule = grad_penalty_coef_schedule
        self.counter = 0
        
        self.fix_std = self.actor_critic.if_fix_std()
        self.std_schedule = std_schedule
        
        # Store privileged dimensions for extracting ground truth
        self.n_priv_mimic_obs = actor_critic.n_priv_mimic_obs
        self.n_proprio = actor_critic.n_proprio
        self.n_priv_info = actor_critic.n_priv_info
    
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
        
        # Get action from student (with priv prediction)
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
    
    def _extract_ground_truth_priv(self, critic_obs):
        """
        Extract ground truth privileged info from critic_obs (full privileged obs).
        
        Critic obs structure: [priv_mimic_obs | proprio | priv_info]
        We want: [priv_mimic_obs | priv_info] (excluding proprio)
        """
        priv_mimic = critic_obs[:, :self.n_priv_mimic_obs]
        priv_info = critic_obs[:, self.n_priv_mimic_obs + self.n_proprio:]
        return torch.cat([priv_mimic, priv_info], dim=1)
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_action_loss = 0.0
        mean_priv_loss = 0.0
        
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for sample in generator:
            obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
            
            # Forward pass through student (with prediction)
            hid_states = hid_states_batch[0] if hid_states_batch else None
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states)
            
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, 
                                                      hidden_states=hid_states_batch[1] if hid_states_batch else None)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            
            # Adaptive learning rate
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
            
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            
            ############################################################
            # LOSS 1: Privileged Prediction Loss (supervised)
            ############################################################
            if self.priv_pred_coef > 0:
                predicted_priv = self.actor_critic.get_predicted_priv()
                gt_priv = self._extract_ground_truth_priv(critic_obs_batch)
                
                priv_loss = (predicted_priv - gt_priv).pow(2).mean()
                priv_loss_weighted = priv_loss * self.priv_pred_coef
                mean_priv_loss += priv_loss.item()
                
                loss += priv_loss_weighted
            
            ############################################################
            # LOSS 2: Action Imitation Loss (L2)
            ############################################################
            if self.dagger_coef > 0 and self.teacher_loaded:
                mu_batch_student = mu_batch
                
                with torch.no_grad():
                    self.teacher_actor_critic.act(critic_obs_batch)
                    mu_batch_teacher = self.teacher_actor_critic.action_mean
                
                action_loss = (mu_batch_student - mu_batch_teacher).pow(2).mean()
                action_loss_weighted = action_loss * self.dagger_coef
                mean_action_loss += action_loss.item()
                
                loss += action_loss_weighted
            
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            
            # Update action std
            if self.fix_std:
                std_stage = min(max((self.counter - self.std_schedule[2]), 0) / self.std_schedule[3], 1)
                std_coef = std_stage * (self.std_schedule[1] - self.std_schedule[0]) + self.std_schedule[0]
                self.actor_critic.update_std(std_coef)
        
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_action_loss /= num_updates
        mean_priv_loss /= num_updates
        
        self.counter += 1
        self.storage.clear()
        
        # Anneal dagger coefficient
        if self.counter < self.dagger_coef_anneal_steps:
            self.dagger_coef = cosine_decay_weight(self.dagger_coef_init, self.counter, self.dagger_coef_anneal_steps)
            self.dagger_coef = max(self.dagger_coef, self.dagger_coef_min)
        else:
            self.dagger_coef = self.dagger_coef_min
        
        # Return: (value, surrogate, priv_reg, ?, ?, ?, action_loss)
        # Note: We repurpose the 3rd return for priv_loss logging
        return mean_value_loss, mean_surrogate_loss, mean_priv_loss, 0, 0, 0, mean_action_loss
    
    def update_counter(self):
        self.counter += 1


