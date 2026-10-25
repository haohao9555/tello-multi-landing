"""Common."""
import argparse
import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

RUNS_DIR = Path(__file__).resolve().parents[3] / 'runs'
HIDDEN = 128


def make_env(env_id, wind_level=None, constant_wind=False):
    """Make env."""
    if env_id == 'fast':
        from tello_rl.fast_env import FastTelloLandEnv
        return FastTelloLandEnv(wind_level=wind_level, constant_wind=constant_wind)
    if env_id == 'multi':
        from tello_rl.multi_env import FastMultiTelloEnv
        return FastMultiTelloEnv(wind_level=wind_level, constant_wind=constant_wind)
    if env_id == 'gazebo':
        raise NotImplementedError('Gazebo environment is available through the simulation module')
    return gym.make(env_id)


def mlp(in_dim, out_dim, hidden=HIDDEN):
    """Mlp."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


def orthogonal_init(net, last_gain):
    """Orthogonal init."""
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    for m in linears:
        nn.init.orthogonal_(m.weight, np.sqrt(2))
        nn.init.zeros_(m.bias)
    nn.init.orthogonal_(linears[-1].weight, last_gain)


def parse_args(algo, default_steps):
    parser = argparse.ArgumentParser(description=f'{algo.upper()} training')
    parser.add_argument('--env', default='fast', help='fast, gazebo, or any Gymnasium environment ID')
    parser.add_argument('--total-steps', type=int, default=default_steps)
    parser.add_argument('--seed', type=int, default=25965245)
    parser.add_argument('--device', default='cpu', help='CPU is recommended for a small MLP and a single environment')
    parser.add_argument('--tag', default='gust', help='Experiment name used to distinguish configurations')
    parser.add_argument('--wind', type=float, default=None, help='Fixed mean wind in m/s; otherwise sample between 0 and 3 each episode')
    parser.add_argument('--wind-type', choices=['gusty', 'constant'], default='gusty',
                        help='gusty: turbulence, gusts and direction drift; constant: fixed wind')
    args = parser.parse_args()
    args.constant_wind = args.wind_type == 'constant'
    wind = 'wrand' if args.wind is None else f'w{args.wind:g}'
    wind += '_const' if args.constant_wind else ''
    args.save_dir = RUNS_DIR / f'{algo}_{args.env}_{args.tag}_{wind}_s{args.seed}'
    os.makedirs(args.save_dir, exist_ok=True)
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class EpisodeLogger:
    """Episodelogger."""

    def __init__(self):
        self.episode_log = []
        self.ep_return, self.ep_len = 0.0, 0

    def step(self, reward):
        self.ep_return += reward
        self.ep_len += 1

    def end_episode(self, step, info):
        if 'all_success' in info:
            success = float(info['all_success'])
        else:
            success = float(info.get('outcome') == 'success')
        error = info.get('landing_error', np.nan)
        self.episode_log.append((step, self.ep_return, self.ep_len, success, error))
        self.ep_return, self.ep_len = 0.0, 0

    def summary(self, last=50):
        recent = np.array(self.episode_log[-last:]) if self.episode_log else np.zeros((1, 5))
        return (f'episodes {len(self.episode_log):>5} | avg return {recent[:, 1].mean():7.2f} | '
                f'success {recent[:, 3].mean() * 100:5.1f}%')
