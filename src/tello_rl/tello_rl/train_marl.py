"""Train MAPPO, IPPO, HAPPO or MAT on the four-aircraft landing task."""
import os

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '1')

import argparse
import copy
import dataclasses
import importlib
import random
import sys
from pathlib import Path

import torch
import yaml


_VENDORED = Path(__file__).resolve().parents[3] / 'vendor' / 'mamujoco_v2'
MARL_CORE_DIR = Path(os.environ.get('MARL_CORE_DIR') or _VENDORED)
sys.path.insert(0, str(MARL_CORE_DIR))

from runner import Runner  # noqa: E402

from tello_rl import multi_task as mt  # noqa: E402
from tello_rl.common import RUNS_DIR  # noqa: E402
from tello_rl.marl_env import VectorTelloMARL  # noqa: E402

ALGORITHMS = {'mappo': ('mappo', 'MAPPO'), 'ippo': ('ippo', 'IPPO'), 'happo': ('happo', 'HAPPO'), 'mat': ('mat', 'MAT')}


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
        raise ValueError(f'Unknown configuration key: {name}')
    target[keys[-1]] = yaml.safe_load(value)


def build_config(algo, seed, device, total_steps, num_envs, overrides):
    document = yaml.safe_load((MARL_CORE_DIR / 'configs' / 'config.yaml').read_text(encoding='utf-8'))
    config = merge(document['common'], document['mappo' if algo == 'ippo' else algo])
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
    parser.add_argument('--num-envs', type=int, default=16, help='Collect num_envs complete episodes per update')
    parser.add_argument('--reward', choices=['team', 'individual'], default='team',
                        help='team: shared fleet reward; individual: own reward, summing to fleet reward')
    parser.add_argument('--profile', choices=tuple(mt.PROFILES), default='tello',
                        help='Aircraft profile: tello or m30 with quadratic drag; see multi_task.py')
    parser.add_argument('--v-max', type=float, default=None,
                        help='Horizontal command limit in m/s; defaults to the aircraft profile')
    parser.add_argument('--v-max-z', type=float, default=None, help='Vertical command limit in m/s; defaults to the aircraft profile')
    parser.add_argument('--episode-limit', type=int, default=None, help='Episode control-step limit; each step is 0.1 seconds')
    parser.add_argument('--init-checkpoint', type=Path, help='Initialise from a compatible policy; reset stage counters and optimisers')
    parser.add_argument('--reference-config', type=Path, help='Copy reference hyperparameters before applying algorithm, environment, budget and --set overrides')
    parser.add_argument('--wind', type=float, default=None, help='Fixed mean gusty wind in m/s; otherwise sample from 0 to 3 per episode')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()
    if args.v_max_z is not None and not 0 < args.v_max_z < float('inf'):
        parser.error('--v-max-z must be finite and positive')
    if args.episode_limit is not None and args.episode_limit <= 0:
        parser.error('--episode-limit must be a positive integer')

    if args.reference_config is None:
        config = build_config(args.algo, args.seed, args.device, args.total_steps, args.num_envs, args.set)
    else:
        config = yaml.safe_load(args.reference_config.read_text())
        reference_algorithm = config['algorithm']
        config.update(algorithm=args.algo, seed=args.seed, device=args.device)
        config['training']['total_env_steps'] = args.total_steps
        config['env']['num_envs'] = args.num_envs
        config['output']['directory'] = str(RUNS_DIR / 'marl')
        config.pop('initialization', None)
        config['reference_experiment'] = {'config': str(args.reference_config.resolve()),
                                          'algorithm': reference_algorithm}
        for expression in args.set:
            apply_override(config, expression)
    config['env']['wind_level'] = args.wind
    config['env']['reward'] = args.reward
    config['env']['v_max_xy'] = args.v_max
    config['env']['profile'] = args.profile
    config['env']['episode_limit'] = mt.PROFILES[args.profile].max_steps
    config['env']['action_history'] = mt.PROFILES[args.profile].action_history
    profile_overrides = {}
    if args.v_max_z is not None:
        profile_overrides['v_max_z'] = args.v_max_z
    if args.episode_limit is not None:
        profile_overrides['max_steps'] = args.episode_limit
    config['env']['profile_overrides'] = profile_overrides
    profile = dataclasses.replace(mt.PROFILES[args.profile], **profile_overrides)
    config['env']['episode_limit'] = profile.max_steps
    if args.algo == 'ippo':
        config['critic_information'] = 'agent_observation_only; no privileged global state or true wind'
    torch.set_num_threads(config['num_threads'])
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])

    environment = VectorTelloMARL(config['env']['num_envs'], config['seed'], wind_level=config['env']['wind_level'],
                                  individual_reward=config['env']['reward'] == 'individual',
                                  v_max_xy=config['env']['v_max_xy'], profile=profile)
    spec = environment.spec
    env_info = {'n_agents': spec.n_agents, 'obs_dim': spec.obs_dim, 'state_dim': spec.state_dim,
                'n_actions': spec.n_actions, 'observation_mask': spec.observation_mask,
                'action_mask': spec.action_mask}
    module, name = ALGORITHMS[args.algo]
    algorithm = getattr(importlib.import_module(f'algorithms.{module}'), name)(env_info, config)
    if args.init_checkpoint is not None:
        payload = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        source = payload['config']
        if source['algorithm'] != config['algorithm']:
            raise ValueError('Initial checkpoint algorithm does not match')
        source_env = source['env']
        if source_env.get('profile', 'tello') != args.profile:
            raise ValueError('Initial checkpoint aircraft profile does not match')
        if source_env.get('action_history', 1) != profile.action_history:
            raise ValueError('Initial checkpoint action-history length does not match')
        algorithm.load_checkpoint_state(payload['algorithm_state'], resume=False)
        algorithm.total_steps, algorithm.updates = 0, 0
        config['initialization'] = {'checkpoint': str(args.init_checkpoint.resolve()),
                                    'source_env_steps': int(payload['algorithm_state']['total_steps'])}
        print(f'Initial policy: {args.init_checkpoint}；Resetting stage step counter', flush=True)
    Runner(config, algorithm, environment).run()


if __name__ == '__main__':
    main()
