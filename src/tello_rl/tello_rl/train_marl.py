"""用 vendor/mamujoco_v2 的 MAPPO / HAPPO / MAT 训练四机降落。

  python -m tello_rl.train_marl --algo mappo --seed 1
  python -m tello_rl.train_marl --algo happo --seed 1
  python -m tello_rl.train_marl --algo mat --seed 1
  python -m tello_rl.train_marl --algo mappo --set training.actor_lr=0.0003
  python -m tello_rl.train_marl --algo mappo --zone-curriculum 1500000 3500000   # 先学降落，再逐步加入降落区规则

超参数沿用那边 config.yaml 的 common + 算法段（HARL 调好的值），这里只改环境相关的几项；
观测已经转到各自坐标系（4 架飞机等价），所以不拼 agent 编号。
结果在 runs/marl/<algo>/tello_4drone/seed_<seed>/<时间戳>/（metrics.jsonl、checkpoint）。
"""
import os

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '1')

import argparse
import copy
import importlib
import random
import sys
from pathlib import Path

import torch
import yaml

# MAPPO / HAPPO / MAT / runner 的代码位置：默认用仓库里的 vendor/mamujoco_v2，可用 MARL_CORE_DIR 环境变量指定别的副本
_VENDORED = Path(__file__).resolve().parents[3] / 'vendor' / 'mamujoco_v2'
MARL_CORE_DIR = Path(os.environ.get('MARL_CORE_DIR') or _VENDORED)
sys.path.insert(0, str(MARL_CORE_DIR))

from runner import Runner  # noqa: E402

from tello_rl import multi_task as mt  # noqa: E402
from tello_rl.common import RUNS_DIR  # noqa: E402
from tello_rl.marl_env import VectorTelloMARL  # noqa: E402

ALGORITHMS = {'mappo': ('mappo', 'MAPPO'), 'happo': ('happo', 'HAPPO'), 'mat': ('mat', 'MAT')}


def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_override(config, expression):
    name, value = expression.split('=', 1)
    keys, target = name.split('.'), config
    for key in keys[:-1]:
        target = target[key]
    if keys[-1] not in target:
        raise ValueError(f'未知配置项：{name}')
    target[keys[-1]] = yaml.safe_load(value)


def build_config(algo, seed, device, total_steps, num_envs, overrides):
    document = yaml.safe_load((MARL_CORE_DIR / 'configs' / 'config.yaml').read_text(encoding='utf-8'))
    config = merge(document['common'], document[algo])
    config['algorithm'] = algo
    config['seed'] = seed
    config['device'] = device
    config['env'] = {'scenario': 'TelloLanding', 'agent_conf': '4x3',
                     'episode_limit': mt.MAX_STEPS, 'num_envs': num_envs}
    config['network']['use_agent_id'] = False
    config['training']['total_env_steps'] = total_steps
    config['evaluation']['interval'] = 100_000
    config['output'].update(directory=str(RUNS_DIR / 'marl'), experiment='tello_4drone',
                            log_interval=20_000, checkpoint_interval=500_000)
    for expression in overrides:
        apply_override(config, expression)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--algo', choices=tuple(ALGORITHMS), default='mappo')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--total-steps', type=int, default=5_000_000)
    parser.add_argument('--num-envs', type=int, default=16, help='一次更新 = num_envs 条完整回合')
    parser.add_argument('--zone-curriculum', type=int, nargs=2, metavar=('START', 'END'),
                        help='降落区违规扣分的课程：累计步数到 START 前不扣分，到 END 时满额')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()

    config = build_config(args.algo, args.seed, args.device, args.total_steps, args.num_envs, args.set)
    config['env']['zone_curriculum'] = args.zone_curriculum
    torch.set_num_threads(config['num_threads'])
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])

    environment = VectorTelloMARL(config['env']['num_envs'], config['seed'],
                                  zone_curriculum=config['env']['zone_curriculum'])
    spec = environment.spec
    env_info = {'n_agents': spec.n_agents, 'obs_dim': spec.obs_dim, 'state_dim': spec.state_dim,
                'n_actions': spec.n_actions, 'observation_mask': spec.observation_mask,
                'action_mask': spec.action_mask}
    module, name = ALGORITHMS[args.algo]
    algorithm = getattr(importlib.import_module(f'algorithms.{module}'), name)(env_info, config)
    Runner(config, algorithm, environment).run()


if __name__ == '__main__':
    main()
