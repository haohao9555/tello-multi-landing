"""三个算法共用的工具：建环境、MLP、命令行参数、保存目录、训练日志。"""
import argparse
import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

RUNS_DIR = Path(__file__).resolve().parents[3] / 'runs'   # /workspace/tello_ws/runs
HIDDEN = 128


def make_env(env_id, wind_level=None):
    """fast：单机快速环境；multi：四机快速环境；gazebo：Gazebo 环境；其他：任意 Gymnasium 环境 id。"""
    if env_id == 'fast':
        from tello_rl.fast_env import FastTelloLandEnv
        return FastTelloLandEnv(wind_level=wind_level)
    if env_id == 'multi':
        from tello_rl.multi_env import FastMultiTelloEnv
        return FastMultiTelloEnv(wind_level=wind_level)
    if env_id == 'gazebo':
        raise NotImplementedError('Gazebo 环境在 Step 3 接入')
    return gym.make(env_id)


def mlp(in_dim, out_dim, hidden=HIDDEN):
    """和 CARTPOLE 里一样的结构：两层 128 + SiLU。"""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


def orthogonal_init(net, last_gain):
    """和 Atari 版一样的正交初始化；最后一层用单独的 gain。"""
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    for m in linears:
        nn.init.orthogonal_(m.weight, np.sqrt(2))
        nn.init.zeros_(m.bias)
    nn.init.orthogonal_(linears[-1].weight, last_gain)


def parse_args(algo, default_steps):
    parser = argparse.ArgumentParser(description=f'{algo.upper()} 训练')
    parser.add_argument('--env', default='fast', help='fast / gazebo / 任意 Gymnasium 环境 id')
    parser.add_argument('--total-steps', type=int, default=default_steps)
    parser.add_argument('--seed', type=int, default=25965245)
    parser.add_argument('--device', default='cpu', help='小 MLP + 单环境用 CPU 更快')
    parser.add_argument('--tag', default='gust', help='实验名，区分不同设置')
    args = parser.parse_args()
    args.save_dir = RUNS_DIR / f'{algo}_{args.env}_{args.tag}_s{args.seed}'
    os.makedirs(args.save_dir, exist_ok=True)
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)   # 多个训练进程并行时互不抢 CPU


class EpisodeLogger:
    """记录每个训练回合：(step, return, length, success, landing_error)，并定期打印。"""

    def __init__(self):
        self.episode_log = []
        self.ep_return, self.ep_len = 0.0, 0

    def step(self, reward):
        self.ep_return += reward
        self.ep_len += 1

    def end_episode(self, step, info):
        if 'all_success' in info:   # 四机：4 架都成功才算
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
