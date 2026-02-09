#!/usr/bin/env python3
"""
ONNX conversion for V6.4.1 - Privileged Predictor Student

Input: 1397 dims (127*11 = current + history)
Architecture: ActorCriticPrivPredictor
"""

import os, sys
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
import argparse
from termcolor import cprint


def get_activation(act_name):
    if act_name == "silu":
        return nn.SiLU()
    elif act_name == "elu":
        return nn.ELU()
    elif act_name == "relu":
        return nn.ReLU()
    else:
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
            activation,
        )
    
    def forward(self, obs_history):
        # obs_history: (batch, num_steps * input_size)
        batch_size = obs_history.shape[0]
        obs_history = obs_history.view(batch_size, self.num_steps, -1)  # (batch, steps, features)
        
        encoded = self.encoder(obs_history)  # (batch, steps, channel*2)
        encoded = encoded.permute(0, 2, 1)  # (batch, channel*2, steps)
        conv_out = self.conv_layers(encoded)  # (batch, conv_output)
        return self.output_layer(conv_out)


class HardwareV6_4_1_NN(nn.Module):
    """
    Hardware deployment wrapper for V6.4.1 (Privileged Predictor).
    
    Actual architecture from checkpoint:
    - history_encoder: takes history (127*10), outputs 256
    - priv_predictor: takes history_latent (256) + current_obs (127) = 383, outputs 1642
    - actor: takes predicted_priv (1642) + proprio (92) = 1734, outputs 29
    
    Input: 1397 dims (current_obs + history)
    Output: 29 actions
    """
    
    def __init__(self,
                 num_observations=1397,
                 num_actions=29,
                 n_obs_single=127,
                 history_len=10,
                 history_latent_dim=256,
                 predicted_priv_dim=1642,
                 activation='silu'):
        super().__init__()
        
        self.num_observations = num_observations
        self.n_obs_single = n_obs_single
        self.history_len = history_len
        self.n_proprio = 92
        self.n_mimic = 35
        
        activation_fn = get_activation(activation)
        
        # History Encoder (matches checkpoint exactly)
        self.history_encoder = HistoryEncoder(
            input_size=n_obs_single,  # 127
            num_steps=history_len,     # 10
            output_size=history_latent_dim,  # 256
            activation=activation_fn
        )
        
        # Privileged Predictor: (history_latent + current_obs) → predicted_priv
        # Input: 256 + 127 = 383 (from checkpoint)
        # Note: checkpoint has priv_predictor.network.X, we use priv_predictor.network.X
        predictor_input_dim = history_latent_dim + n_obs_single  # 383
        self.priv_predictor = nn.ModuleDict({
            'network': nn.Sequential(
                nn.Linear(predictor_input_dim, 512), activation_fn,
                nn.Linear(512, 512), activation_fn,
                nn.Linear(512, 256), activation_fn,
                nn.Linear(256, predicted_priv_dim)  # 1642
            )
        })
        
        # Actor: (predicted_priv + proprio) → action
        # Input: 1642 + 92 = 1734 (from checkpoint)
        actor_input_dim = predicted_priv_dim + self.n_proprio  # 1734
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, 512), activation_fn,
            nn.Linear(512, 512), activation_fn,
            nn.Linear(512, 256), activation_fn,
            nn.Linear(256, 128), activation_fn,
            nn.Linear(128, num_actions)  # 29
        )
        
        self.normalizer = None
    
    def load_normalizer(self, normalizer):
        self.normalizer = normalizer
    
    def forward(self, obs):
        # obs: (batch, 1397) = current_obs (127) + history (127*10)
        
        # Normalize
        if self.normalizer is not None:
            obs = self.normalizer.normalize(obs)
        
        # Split current and history
        current_obs = obs[:, :self.n_obs_single]  # (batch, 127)
        history_obs = obs[:, self.n_obs_single:]  # (batch, 1270)
        
        # Extract current proprio (for actor input)
        current_proprio = current_obs[:, self.n_mimic:]  # (batch, 92)
        
        # Encode history
        history_latent = self.history_encoder(history_obs)  # (batch, 256)
        
        # Predictor input = history_latent + current_obs
        predictor_input = torch.cat([history_latent, current_obs], dim=-1)  # (batch, 383)
        
        # Predict privileged info
        predicted_priv = self.priv_predictor['network'](predictor_input)  # (batch, 1642)
        
        # Actor input = predicted_priv + current_proprio
        actor_input = torch.cat([predicted_priv, current_proprio], dim=-1)  # (batch, 1734)
        
        # Get action
        action = self.actor(actor_input)  # (batch, 29)
        
        return action


def convert_to_onnx(args):
    ckpt_path = args.ckpt_path
    
    if not os.path.exists(ckpt_path):
        cprint(f"Error: Checkpoint not found: {ckpt_path}", "red")
        return
    
    # V6.4.1 configuration
    num_observations = 1397  # 127 * 11
    num_actions = 29
    n_obs_single = 127
    history_len = 10
    history_latent_dim = 256
    predicted_priv_dim = 1642  # 1540 (priv_mimic) + 102 (priv_info)
    
    print(f"V6.4.1 Privileged Predictor Configuration:")
    print(f"  Observations: {num_observations} (127 × 11)")
    print(f"  Actions: {num_actions}")
    print(f"  History latent: {history_latent_dim}")
    print(f"  Predicted priv dim: {predicted_priv_dim}")
    print("")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    policy = HardwareV6_4_1_NN(
        num_observations=num_observations,
        num_actions=num_actions,
        n_obs_single=n_obs_single,
        history_len=history_len,
        history_latent_dim=history_latent_dim,
        predicted_priv_dim=predicted_priv_dim,
        activation='silu'
    ).to(device)
    
    cprint(f"Loading model from: {ckpt_path}", "green")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if 'normalizer' in checkpoint:
        policy.load_normalizer(checkpoint['normalizer'])
    
    policy.eval()
    
    with torch.no_grad():
        dummy_input = torch.ones(1, num_observations, device=device)
        cprint(f"Input shape: {dummy_input.shape}", "cyan")
        
        onnx_path = ckpt_path.replace('.pt', '.onnx')
        
        torch.onnx.export(
            policy,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        cprint(f"ONNX saved to: {onnx_path}", "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()
    convert_to_onnx(args)

