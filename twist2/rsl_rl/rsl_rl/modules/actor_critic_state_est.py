"""
V6.4: Actor-Critic with State Estimator

Architecture:
  Student Obs (proprio + history + mimic) 
       ↓
  [State Estimator] → Estimated Latent (predicts privileged info)
       ↓
  [Actor] ← (obs + estimated latent) → Actions

The state estimator learns to predict privileged/hidden information from
observable data, bridging the sim-to-real gap.

Training:
  1. Action Loss: L2(student_action, teacher_action) - primary objective
  2. Latent Loss (optional): L2(estimated_latent, teacher_latent) - auxiliary supervision
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "silu":
        return nn.SiLU()
    else:
        print(f"Invalid activation function: {act_name}")
        return nn.ELU()


class HistoryEncoder(nn.Module):
    """Encodes observation history using 1D convolutions."""
    
    def __init__(self, input_size, num_steps, output_size, activation):
        super().__init__()
        self.num_steps = num_steps
        
        channel_size = 32
        
        # Project each timestep
        self.encoder = nn.Sequential(
            nn.Linear(input_size, channel_size * 2),
            activation,
        )
        
        # Temporal convolutions
        if num_steps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(channel_size * 2, channel_size, kernel_size=4, stride=2),
                activation,
                nn.Conv1d(channel_size, channel_size, kernel_size=2, stride=1),
                activation,
                nn.Flatten()
            )
            conv_output_size = channel_size * 3
        elif num_steps == 11:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(channel_size * 2, channel_size, kernel_size=4, stride=2),
                activation,
                nn.Conv1d(channel_size, channel_size, kernel_size=2, stride=1),
                activation,
                nn.Flatten()
            )
            conv_output_size = channel_size * 3
        else:
            # Fallback for other history lengths
            self.conv_layers = nn.Sequential(
                nn.Conv1d(channel_size * 2, channel_size, kernel_size=3, stride=2),
                activation,
                nn.Flatten()
            )
            conv_output_size = channel_size * ((num_steps - 3) // 2 + 1)
        
        self.output_layer = nn.Sequential(
            nn.Linear(conv_output_size, output_size),
            activation
        )
    
    def forward(self, obs_history):
        """
        Args:
            obs_history: (batch, num_steps * obs_dim) flattened history
        Returns:
            latent: (batch, output_size)
        """
        batch_size = obs_history.shape[0]
        obs_dim = obs_history.shape[1] // self.num_steps
        
        # Reshape to (batch * num_steps, obs_dim)
        obs_reshaped = obs_history.reshape(batch_size * self.num_steps, obs_dim)
        
        # Encode each timestep
        encoded = self.encoder(obs_reshaped)  # (batch * num_steps, channel * 2)
        
        # Reshape for conv: (batch, channels, num_steps)
        encoded = encoded.reshape(batch_size, self.num_steps, -1).permute(0, 2, 1)
        
        # Temporal convolutions
        conv_out = self.conv_layers(encoded)
        
        # Output projection
        return self.output_layer(conv_out)


class StateEstimator(nn.Module):
    """
    Estimates privileged/latent state from observable information.
    
    Takes: current obs + history encoding
    Outputs: estimated latent (representing privileged info)
    """
    
    def __init__(self, input_size, latent_size, hidden_dims, activation):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_dims[0]))
        layers.append(activation)
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(activation)
        
        layers.append(nn.Linear(hidden_dims[-1], latent_size))
        # No activation on output - let it be unbounded
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ActorCriticStateEst(nn.Module):
    """
    Actor-Critic with State Estimator for Sim-to-Real transfer.
    
    Architecture:
        1. History Encoder: encodes observation history
        2. State Estimator: predicts privileged latent from (current + history encoding)
        3. Actor: takes (current + history + estimated latent) → actions
        4. Critic: takes privileged obs → value (for training only)
    """
    is_recurrent = False
    
    def __init__(self,
                 num_observations,      # Total student observations
                 num_single_obs,        # Single timestep obs (mimic + proprio)
                 num_history_steps,     # Number of history steps (e.g., 10)
                 num_actions,
                 num_critic_obs,        # Privileged observations for critic
                 latent_size=64,        # Size of estimated latent
                 actor_hidden_dims=[512, 512, 256, 128],
                 critic_hidden_dims=[512, 512, 256, 128],
                 estimator_hidden_dims=[256, 256, 128],
                 history_latent_dim=128,
                 activation='silu',
                 init_noise_std=0.8,
                 fix_action_std=True,
                 action_std=None,
                 layer_norm=False,
                 **kwargs):
        
        if kwargs:
            print(f"ActorCriticStateEst: ignoring kwargs: {list(kwargs.keys())}")
        
        super().__init__()
        
        self.num_observations = num_observations
        self.num_single_obs = num_single_obs
        self.num_history_steps = num_history_steps
        self.num_actions = num_actions
        self.latent_size = latent_size
        self.fix_action_std = fix_action_std
        
        activation_fn = get_activation(activation)
        
        # === History Encoder ===
        # Encodes the history portion of observations
        self.history_encoder = HistoryEncoder(
            input_size=num_single_obs,
            num_steps=num_history_steps,
            output_size=history_latent_dim,
            activation=activation_fn
        )
        
        # === State Estimator ===
        # Input: current obs + history latent
        estimator_input_size = num_single_obs + history_latent_dim
        self.state_estimator = StateEstimator(
            input_size=estimator_input_size,
            latent_size=latent_size,
            hidden_dims=estimator_hidden_dims,
            activation=activation_fn
        )
        
        # === Actor ===
        # Input: current obs + history latent + estimated latent
        actor_input_size = num_single_obs + history_latent_dim + latent_size
        
        actor_layers = []
        if layer_norm:
            actor_layers.append(nn.LayerNorm(actor_input_size))
        actor_layers.append(nn.Linear(actor_input_size, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        
        for i in range(len(actor_hidden_dims) - 1):
            if layer_norm:
                actor_layers.append(nn.LayerNorm(actor_hidden_dims[i]))
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            actor_layers.append(activation_fn)
        
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)
        
        # === Critic ===
        # Uses privileged observations (for training only)
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            critic_layers.append(activation_fn)
        
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # === Action Noise ===
        if fix_action_std and action_std is not None:
            self.std = nn.Parameter(torch.tensor(action_std), requires_grad=False)
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=not fix_action_std)
        
        self.distribution = None
        Normal.set_default_validate_args = False
        
        # Store estimated latent for auxiliary loss
        self.estimated_latent = None
    
    def reset(self, dones=None):
        pass
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def _encode_and_estimate(self, obs):
        """
        Encode history and estimate latent state.
        
        Args:
            obs: (batch, num_observations) - full student observation
                 Structure: [current_obs (num_single_obs)] + [history (num_history_steps * num_single_obs)]
        
        Returns:
            current_obs, history_latent, estimated_latent
        """
        # Split observations
        current_obs = obs[:, :self.num_single_obs]
        history_obs = obs[:, self.num_single_obs:]
        
        # Encode history
        history_latent = self.history_encoder(history_obs)
        
        # Estimate privileged latent
        estimator_input = torch.cat([current_obs, history_latent], dim=1)
        estimated_latent = self.state_estimator(estimator_input)
        
        # Store for auxiliary loss
        self.estimated_latent = estimated_latent
        
        return current_obs, history_latent, estimated_latent
    
    def update_distribution(self, obs):
        current_obs, history_latent, estimated_latent = self._encode_and_estimate(obs)
        
        # Combine all features for actor
        actor_input = torch.cat([current_obs, history_latent, estimated_latent], dim=1)
        mean = self.actor(actor_input)
        
        self.distribution = Normal(mean, mean * 0. + self.std)
    
    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, obs, **kwargs):
        """Deterministic action for inference/deployment."""
        current_obs, history_latent, estimated_latent = self._encode_and_estimate(obs)
        actor_input = torch.cat([current_obs, history_latent, estimated_latent], dim=1)
        return self.actor(actor_input)
    
    def evaluate(self, critic_obs, **kwargs):
        """Evaluate value using privileged observations."""
        return self.critic(critic_obs)
    
    def get_estimated_latent(self):
        """Return the estimated latent for auxiliary loss computation."""
        return self.estimated_latent
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data
    
    def if_fix_std(self):
        return self.fix_action_std
    
    def update_std(self, std_coef):
        """Update std for scheduled annealing."""
        if self.fix_action_std:
            pass  # Fixed std doesn't change
        else:
            self.std.data = self.std.data * std_coef
    
    def test(self):
        self.eval()
    
    def train_mode(self):
        self.train()


