#!/usr/bin/env python3
"""
ONNX conversion for default student (no future observations)

Input: 1397 dims (127 current + 127*10 history)
Architecture: HardwareStudentNN style (history encoder + actor)
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
    else:
        return nn.ELU()


class HistoryEncoder(nn.Module):
    """History encoder matching checkpoint structure."""
    
    def __init__(self, input_size=127, num_steps=10, output_size=128, activation=None):
        super().__init__()
        if activation is None:
            activation = nn.SiLU()
        
        # Encoder: input_size -> 30
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 30),
            activation,
        )
        
        # Conv layers for 10 steps (matching checkpoint exactly)
        # Input: (batch, 30, 10) -> Conv(30->20, k=4, s=2) -> (batch, 20, 4)
        # -> Conv(20->10, k=2, s=1) -> (batch, 10, 3) -> flatten -> (batch, 30)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(30, 20, kernel_size=4, stride=2),
            activation,
            nn.Conv1d(20, 10, kernel_size=2, stride=1),
            activation,
            nn.Flatten()  # Output: 10 * 3 = 30
        )
        
        # Linear output: 30 -> output_size
        self.linear_output = nn.Sequential(
            nn.Linear(30, output_size),
            activation,
        )
        
        self.num_steps = num_steps
    
    def forward(self, obs_history):
        # obs_history: (batch, num_steps * input_size) = (batch, 1270)
        batch_size = obs_history.shape[0]
        obs_history = obs_history.view(batch_size, self.num_steps, -1)  # (batch, 10, 127)
        
        encoded = self.encoder(obs_history)  # (batch, 10, 30)
        encoded = encoded.permute(0, 2, 1)  # (batch, 30, 10)
        conv_out = self.conv_layers(encoded)  # (batch, 30)
        return self.linear_output(conv_out)  # (batch, 128)


class DefaultStudentNN(nn.Module):
    """
    Default student policy (no future observations).
    
    Input: 1397 dims (current 127 + history 1270)
    Output: 29 actions
    """
    
    def __init__(self,
                 num_observations=1397,
                 num_actions=29,
                 n_obs_single=127,
                 history_len=10,
                 history_latent_dim=128,
                 actor_hidden_dims=[1024, 1024, 512, 256],
                 activation='silu'):
        super().__init__()
        
        self.num_observations = num_observations
        self.n_obs_single = n_obs_single
        self.history_len = history_len
        
        activation_fn = get_activation(activation)
        
        # History Encoder
        self.history_encoder = HistoryEncoder(
            input_size=n_obs_single,
            num_steps=history_len,
            output_size=history_latent_dim,
            activation=activation_fn
        )
        
        # Actor backbone: history_latent + current_obs -> action
        actor_input_dim = history_latent_dim + n_obs_single  # 128 + 127 = 255
        
        actor_layers = []
        prev_dim = actor_input_dim
        for dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, dim))
            actor_layers.append(activation_fn)
            prev_dim = dim
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor_backbone = nn.Sequential(*actor_layers)
        
        self.normalizer = None
    
    def load_normalizer(self, normalizer):
        self.normalizer = normalizer
    
    def forward(self, obs):
        # obs: (batch, 1397) = current (127) + history (1270)
        
        if self.normalizer is not None:
            obs = self.normalizer.normalize(obs)
        
        current_obs = obs[:, :self.n_obs_single]  # (batch, 127)
        history_obs = obs[:, self.n_obs_single:]  # (batch, 1270)
        
        history_latent = self.history_encoder(history_obs)  # (batch, 128)
        
        actor_input = torch.cat([history_latent, current_obs], dim=-1)  # (batch, 255)
        
        action = self.actor_backbone(actor_input)  # (batch, 29)
        
        return action


def convert_to_onnx(args):
    ckpt_path = args.ckpt_path
    
    if not os.path.exists(ckpt_path):
        cprint(f"Error: Checkpoint not found: {ckpt_path}", "red")
        return
    
    # Configuration
    num_observations = 1397
    num_actions = 29
    n_obs_single = 127
    history_len = 10
    history_latent_dim = 128
    actor_hidden_dims = [1024, 1024, 512, 256]
    
    print(f"Default Student Configuration:")
    print(f"  Observations: {num_observations} (127 × 11)")
    print(f"  Actions: {num_actions}")
    print(f"  History latent: {history_latent_dim}")
    print(f"  Actor dims: {actor_hidden_dims}")
    print("")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    policy = DefaultStudentNN(
        num_observations=num_observations,
        num_actions=num_actions,
        n_obs_single=n_obs_single,
        history_len=history_len,
        history_latent_dim=history_latent_dim,
        actor_hidden_dims=actor_hidden_dims,
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

