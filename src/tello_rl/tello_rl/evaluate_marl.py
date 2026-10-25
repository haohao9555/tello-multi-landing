"""Evaluate a saved MARL checkpoint on fixed episode seeds."""
import os

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '1')

import argparse
import dataclasses
import importlib
import json
from pathlib import Path

import torch

from tello_rl import multi_task as mt
from tello_rl.evaluate_multi import evaluate
from tello_rl.marl_env import N, N_ACTIONS, VectorTelloMARL, agent_observations, to_world_actions
from tello_rl.train_marl import ALGORITHMS


def load_policy(checkpoint, device='cpu'):
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = payload['config']
    config['device'] = device
    if config['network']['use_gru']:
        raise ValueError('Only MLP actors are supported (use_gru=false)')
    profile = dataclasses.replace(mt.get_profile(config['env'].get('profile')),
                                  **{**config['env'].get('profile_overrides', {}),
                                     'action_history': config['env'].get('action_history', 1)})
    spec = VectorTelloMARL(1, 0, profile=profile).spec
    env_info = {'n_agents': spec.n_agents, 'obs_dim': spec.obs_dim, 'state_dim': spec.state_dim,
                'n_actions': spec.n_actions, 'observation_mask': spec.observation_mask,
                'action_mask': spec.action_mask}
    module, name = ALGORITHMS[config['algorithm']]
    v_max_xy = config['env'].get('v_max_xy')
    algorithm = getattr(importlib.import_module(f'algorithms.{module}'), name)(env_info, config)
    algorithm.load_checkpoint_state(payload['algorithm_state'], resume=False)
    algorithm.train(False)
    latent = torch.zeros(1, N, N_ACTIONS, device=device)

    def policy(flat_obs):
        """Policy."""
        obs = torch.as_tensor(agent_observations(flat_obs), device=device).unsqueeze(0)
        with torch.no_grad():
            sample, _ = algorithm.act(algorithm.actor_inputs(obs), algorithm.init_hidden(1), latent, False)
        return to_world_actions(sample['actions'][0].cpu().numpy())
    policy.v_max_xy = v_max_xy
    policy.profile = profile
    return policy, config['algorithm']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--winds', type=float, nargs='+', default=None, help='Defaults to aircraft-specific wind levels (Tello: 0 1 2 3)')
    parser.add_argument('--no-preflight', action='store_true', help='Disable measured-wind preflight gating for stress testing')
    args = parser.parse_args()

    policy, algo = load_policy(args.checkpoint)
    profile = mt.get_profile(policy.profile)
    fmt = lambda v, f='.1%': format(v, f) if v is not None else '-'
    print(f'{algo.upper()}  {profile.name}  {args.checkpoint}')
    print(f'{"Wind":>6} {"No-go":>6} {"Fleet success":>7} {"Drone success":>7} {"Collision rate":>6} {"Collided drones":>7} '
          f'{"Min separation":>7} {"Touchdown spread":>8} {"Total duration":>6} {"Landing error":>7}   Outcome counts')
    results = []
    for level in args.winds or profile.wind_levels:
        r = evaluate(policy, level, args.episodes, preflight=level > profile.max_wind and not args.no_preflight,
                     v_max_xy=policy.v_max_xy, profile=profile)
        results.append(r)
        print(f'{level:>4.0f} m/s {fmt(r["no_go_rate"]):>6} {fmt(r["all_success"]):>8} {fmt(r["drone_success"]):>8} '
              f'{fmt(r["collision_rate"]):>7} {fmt(r["collided_drones"], ".2f"):>8} {fmt(r["min_sep"], ".2f"):>7} m '
              f'{fmt(r["landing_spread"], ".1f"):>7} s {fmt(r["time"], ".1f"):>5} s {fmt(r["landing_error"], ".3f"):>6} m   '
              f'{r["outcomes"]}')
    out = args.checkpoint.with_name(f'eval_{args.checkpoint.stem}.json')
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
