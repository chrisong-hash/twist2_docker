"""
V6.4.1: Actor-Critic with Privileged Information Predictor

Key Concept:
  - Student ACTOR has the SAME input structure as Teacher
  - A separate PREDICTOR network fills in the missing privileged info
  - Student can be initialized from teacher weights!

Architecture:
  Student Obs (limited: proprio + history + current mimic)
       ↓
  [Privileged Predictor] → predicted_priv_info (fills gaps)
       ↓
  [Combine: student_obs + predicted_priv] = full_obs (same as teacher!)
       ↓
  [Actor (same as teacher)] → action

Training:
  1. Predictor Loss: L2(predicted_priv, actual_priv) - supervised
  2. Action Loss: L2(student_action, teacher_action) - imitation
  
Benefits:
  - Student actor can be initialized from teacher (same architecture)
  - Predictor target is well-defined (actual privileged values)
  - Easy to debug (check prediction accuracy)
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
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, channel_size * 2),
            activation,
        )
        
        if num_steps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(channel_size * 2, channel_size, kernel_size=4, stride=2),
                activation,
                nn.Conv1d(channel_size, channel_size, kernel_size=2, stride=1),
                activation,
                nn.Flatten()
            )
            conv_output_size = channel_size * 3
        else:
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
        batch_size = obs_history.shape[0]
        obs_dim = obs_history.shape[1] // self.num_steps
        
        obs_reshaped = obs_history.reshape(batch_size * self.num_steps, obs_dim)
        encoded = self.encoder(obs_reshaped)
        encoded = encoded.reshape(batch_size, self.num_steps, -1).permute(0, 2, 1)
        conv_out = self.conv_layers(encoded)
        return self.output_layer(conv_out)


class PrivilegedPredictor(nn.Module):
    """
    Predicts the privileged information that student doesn't observe.
    
    Takes: student's observable info (proprio + history encoding)
    Outputs: predicted privileged info (same dims as actual priv info)
    """
    
    def __init__(self, input_size, priv_size, hidden_dims, activation):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_dims[0]))
        layers.append(activation)
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(activation)
        
        # Output: predicted privileged info
        layers.append(nn.Linear(hidden_dims[-1], priv_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ActorCriticPrivPredictor(nn.Module):
    """
    Actor-Critic where student has SAME input structure as teacher.
    A predictor network fills in the missing privileged information.
    
    Observation structure (teacher's privileged obs):
      [priv_mimic_obs | proprio | priv_info]
      
    Student observes: proprio (+ history for prediction)
    Predictor outputs: priv_mimic_obs, priv_info
    
    This allows:
      1. Student actor to be initialized from teacher weights
      2. Well-defined prediction targets (actual privileged values)
    """
    is_recurrent = False
    
    def __init__(self,
                 # Teacher's full observation structure
                 num_priv_obs,           # Total privileged obs (teacher input)
                 n_priv_mimic_obs,       # Privileged mimic obs dims
                 n_proprio,              # Proprioception dims
                 n_priv_info,            # Privileged info dims
                 
                 # Student's limited observations
                 num_student_obs,        # Student's observable dims
                 n_mimic_obs,            # Current mimic obs (student sees this)
                 num_history_steps,      # History length
                 
                 # Actions
                 num_actions,
                 
                 # Network config
                 predictor_hidden_dims=[512, 256, 256],
                 history_latent_dim=128,
                 actor_hidden_dims=[512, 512, 256, 128],
                 critic_hidden_dims=[512, 512, 256, 128],
                 activation='silu',
                 init_noise_std=0.8,
                 fix_action_std=True,
                 action_std=None,
                 **kwargs):
        
        if kwargs:
            print(f"ActorCriticPrivPredictor: ignoring kwargs: {list(kwargs.keys())}")
        
        super().__init__()
        
        # Store dimensions
        self.num_priv_obs = num_priv_obs
        self.n_priv_mimic_obs = n_priv_mimic_obs
        self.n_proprio = n_proprio
        self.n_priv_info = n_priv_info
        self.num_student_obs = num_student_obs
        self.n_mimic_obs = n_mimic_obs
        self.num_history_steps = num_history_steps
        self.num_actions = num_actions
        self.fix_action_std = fix_action_std
        
        # What we need to predict
        self.n_priv_to_predict = n_priv_mimic_obs + n_priv_info
        
        activation_fn = get_activation(activation)
        
        # Student's single obs = mimic + proprio
        n_student_single = n_mimic_obs + n_proprio
        
        # === History Encoder ===
        self.history_encoder = HistoryEncoder(
            input_size=n_student_single,
            num_steps=num_history_steps,
            output_size=history_latent_dim,
            activation=activation_fn
        )
        
        # === Privileged Predictor ===
        # Input: current obs (mimic + proprio) + history latent
        predictor_input_size = n_student_single + history_latent_dim
        self.priv_predictor = PrivilegedPredictor(
            input_size=predictor_input_size,
            priv_size=self.n_priv_to_predict,
            hidden_dims=predictor_hidden_dims,
            activation=activation_fn
        )
        
        # === Actor (SAME as teacher!) ===
        # Input: full privileged obs = priv_mimic + proprio + priv_info
        actor_layers = []
        actor_layers.append(nn.Linear(num_priv_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            actor_layers.append(activation_fn)
        
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)
        
        # === Critic (uses true privileged obs during training) ===
        critic_layers = []
        critic_layers.append(nn.Linear(num_priv_obs, critic_hidden_dims[0]))
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
        
        # Store predicted privileged info for loss computation
        self.predicted_priv = None
    
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
    
    def _predict_privileged(self, student_obs):
        """
        Predict the privileged information from student's limited observations.
        
        Args:
            student_obs: (batch, num_student_obs) - [current | history]
                current = mimic_obs + proprio
                history = num_history_steps * (mimic_obs + proprio)
        
        Returns:
            predicted_priv: (batch, n_priv_to_predict) - [priv_mimic | priv_info]
        """
        n_student_single = self.n_mimic_obs + self.n_proprio
        
        # Split current and history
        current_obs = student_obs[:, :n_student_single]
        history_obs = student_obs[:, n_student_single:]
        
        # Encode history
        history_latent = self.history_encoder(history_obs)
        
        # Predict privileged info
        predictor_input = torch.cat([current_obs, history_latent], dim=1)
        predicted_priv = self.priv_predictor(predictor_input)
        
        # Store for loss computation
        self.predicted_priv = predicted_priv
        
        return predicted_priv
    
    def _construct_full_obs(self, student_obs, predicted_priv):
        """
        Construct full observation (same structure as teacher) using predictions.
        
        Teacher obs structure: [priv_mimic_obs | proprio | priv_info]
        
        Args:
            student_obs: (batch, num_student_obs)
            predicted_priv: (batch, n_priv_to_predict) = [pred_priv_mimic | pred_priv_info]
        
        Returns:
            full_obs: (batch, num_priv_obs) - same structure as teacher
        """
        n_student_single = self.n_mimic_obs + self.n_proprio
        
        # Extract proprio from current student obs
        current_obs = student_obs[:, :n_student_single]
        proprio = current_obs[:, self.n_mimic_obs:]  # (batch, n_proprio)
        
        # Split predicted priv into mimic and info
        pred_priv_mimic = predicted_priv[:, :self.n_priv_mimic_obs]
        pred_priv_info = predicted_priv[:, self.n_priv_mimic_obs:]
        
        # Construct full obs: [priv_mimic | proprio | priv_info]
        full_obs = torch.cat([pred_priv_mimic, proprio, pred_priv_info], dim=1)
        
        return full_obs
    
    def update_distribution(self, student_obs):
        # Predict privileged info
        predicted_priv = self._predict_privileged(student_obs)
        
        # Construct full observation (same as teacher)
        full_obs = self._construct_full_obs(student_obs, predicted_priv)
        
        # Actor forward (using full obs like teacher)
        mean = self.actor(full_obs)
        self.distribution = Normal(mean, mean * 0. + self.std)
    
    def act(self, student_obs, **kwargs):
        self.update_distribution(student_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, student_obs, **kwargs):
        """Deterministic action for inference/deployment."""
        predicted_priv = self._predict_privileged(student_obs)
        full_obs = self._construct_full_obs(student_obs, predicted_priv)
        return self.actor(full_obs)
    
    def evaluate(self, priv_obs, **kwargs):
        """Evaluate value using TRUE privileged observations."""
        return self.critic(priv_obs)
    
    def get_predicted_priv(self):
        """Return predicted privileged info for auxiliary loss."""
        return self.predicted_priv
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data
    
    def if_fix_std(self):
        return self.fix_action_std
    
    def update_std(self, std_coef):
        if not self.fix_action_std:
            self.std.data = self.std.data * std_coef
    
    def test(self):
        self.eval()
    
    def train_mode(self):
        self.train()
    
    def load_teacher_actor(self, teacher_state_dict):
        """
        Initialize actor from teacher weights.
        Only loads actor weights, not predictor (which is student-specific).
        """
        # Filter for actor weights only
        actor_weights = {k.replace('actor.', ''): v 
                        for k, v in teacher_state_dict.items() 
                        if k.startswith('actor.')}
        
        if actor_weights:
            self.actor.load_state_dict(actor_weights, strict=False)
            print(f"Loaded {len(actor_weights)} actor weights from teacher")


