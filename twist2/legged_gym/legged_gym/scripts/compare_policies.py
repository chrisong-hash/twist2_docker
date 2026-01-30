#!/usr/bin/env python3
"""
Policy Comparison Visualizer
============================
Compare two trained policies side-by-side in Isaac Gym simulation.

Usage:
    # Compare default (20k) vs V2 (new training)
    python compare_policies.py --task g1_priv_mimic \
        --policy1_path ../../logs/g1_priv_mimic/my_experiment/model_20000.pt \
        --policy1_name "Default 20k" \
        --policy2_path ../../logs/h1/backward_walking_v2/model_1500.pt \
        --policy2_name "V2 Reduce Joint"
    
    # Record video comparison
    python compare_policies.py --task g1_priv_mimic \
        --policy1_path ... --policy2_path ... \
        --record_video
"""

import os
import sys
import torch
import numpy as np
from termcolor import cprint
import argparse
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'rsl_rl'))

from legged_gym.envs import *
from legged_gym.gym_utils import get_args, task_registry


def load_policy(ckpt_path, env, device):
    """Load a policy checkpoint and return inference function."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Get policy architecture from environment config
    from rsl_rl.modules.actor_critic_mimic import ActorCriticMimic
    
    # Create policy matching the checkpoint
    policy = ActorCriticMimic(
        num_obs=env.num_obs,
        num_privileged_obs=env.num_privileged_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        activation='silu',
        init_noise_std=1.0,
    ).to(device)
    
    # Load weights
    policy.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    # Load normalizer if available
    normalizer = None
    if 'normalizer' in ckpt:
        normalizer = ckpt['normalizer']
    
    policy.eval()
    return policy, normalizer


def set_comparison_cfg(env_cfg):
    """Configure environment for policy comparison."""
    env_cfg.env.num_envs = 2  # One env per policy
    env_cfg.env.debug_viz = True
    env_cfg.env.episode_length_s = 30
    
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    
    # Disable all randomization for fair comparison
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False
    
    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False


def main():
    parser = argparse.ArgumentParser(description='Compare two policies side-by-side')
    parser.add_argument('--task', type=str, default='g1_priv_mimic', help='Task name')
    parser.add_argument('--policy1_path', type=str, required=True, help='Path to first policy checkpoint')
    parser.add_argument('--policy1_name', type=str, default='Policy 1', help='Display name for policy 1')
    parser.add_argument('--policy2_path', type=str, required=True, help='Path to second policy checkpoint')
    parser.add_argument('--policy2_name', type=str, default='Policy 2', help='Display name for policy 2')
    parser.add_argument('--record_video', action='store_true', help='Record comparison video')
    parser.add_argument('--video_path', type=str, default='comparison.mp4', help='Output video path')
    parser.add_argument('--num_steps', type=int, default=2000, help='Number of simulation steps')
    
    args, unknown = parser.parse_known_args()
    
    # Get environment config
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    set_comparison_cfg(env_cfg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create environment
    cprint(f"Creating environment for task: {args.task}", "cyan")
    env, _ = task_registry.make_env(name=args.task, args=argparse.Namespace(
        task=args.task,
        headless=args.record_video,
        num_envs=2,
        use_jit=False,
        proj_name=args.task,
        exptid='comparison'
    ), env_cfg=env_cfg)
    
    # Load both policies
    cprint(f"\nLoading {args.policy1_name}: {args.policy1_path}", "green")
    policy1, norm1 = load_policy(args.policy1_path, env, device)
    
    cprint(f"Loading {args.policy2_name}: {args.policy2_path}", "green")
    policy2, norm2 = load_policy(args.policy2_path, env, device)
    
    # Setup video recording
    if args.record_video:
        import imageio
        writer = imageio.get_writer(args.video_path, fps=int(1/env.dt))
        cprint(f"\nRecording video to: {args.video_path}", "yellow")
    
    # Run comparison
    cprint(f"\n{'='*60}", "cyan")
    cprint(f"Running comparison: {args.policy1_name} vs {args.policy2_name}", "cyan")
    cprint(f"{'='*60}\n", "cyan")
    
    obs = env.get_observations()
    
    # Track metrics
    rewards1, rewards2 = [], []
    episode_lengths1, episode_lengths2 = [], []
    
    for step in tqdm(range(args.num_steps), desc="Simulating"):
        # Get observations for each environment
        obs1 = obs[0:1]  # First env
        obs2 = obs[1:2]  # Second env
        
        # Get actions from each policy
        with torch.no_grad():
            if norm1 is not None:
                obs1_norm = norm1.normalize(obs1)
            else:
                obs1_norm = obs1
            action1 = policy1.act_inference(obs1_norm)
            
            if norm2 is not None:
                obs2_norm = norm2.normalize(obs2)
            else:
                obs2_norm = obs2
            action2 = policy2.act_inference(obs2_norm)
        
        # Combine actions
        actions = torch.cat([action1, action2], dim=0)
        
        # Step environment
        obs, _, rewards, dones, infos = env.step(actions)
        
        # Track rewards
        rewards1.append(rewards[0].item())
        rewards2.append(rewards[1].item())
        
        # Record video frame
        if args.record_video:
            imgs = env.render_record(mode='rgb_array')
            if imgs is not None:
                # Combine both views horizontally
                combined = np.concatenate([imgs[0], imgs[1]], axis=1)
                writer.append_data(combined)
    
    # Close video
    if args.record_video:
        writer.close()
        cprint(f"\nVideo saved to: {args.video_path}", "green")
    
    # Print summary
    cprint(f"\n{'='*60}", "cyan")
    cprint("Comparison Summary", "cyan")
    cprint(f"{'='*60}", "cyan")
    
    print(f"\n{args.policy1_name}:")
    print(f"  Mean reward: {np.mean(rewards1):.4f}")
    print(f"  Total reward: {np.sum(rewards1):.2f}")
    
    print(f"\n{args.policy2_name}:")
    print(f"  Mean reward: {np.mean(rewards2):.4f}")
    print(f"  Total reward: {np.sum(rewards2):.2f}")
    
    diff = np.mean(rewards2) - np.mean(rewards1)
    print(f"\nDifference: {'+' if diff > 0 else ''}{diff:.4f} ({'+' if diff > 0 else ''}{diff/np.mean(rewards1)*100:.1f}%)")


if __name__ == '__main__':
    main()



