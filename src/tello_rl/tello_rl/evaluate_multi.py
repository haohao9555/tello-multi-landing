"""Evaluate multi."""
import argparse
import importlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tello_rl import multi_task as mt
from tello_rl import task
from tello_rl.multi_env import FastMultiTelloEnv

EVAL_SEED = 10_000


def evaluate(policy, wind_level, n_episodes, preflight, v_max_xy=None, profile='tello'):
    env = FastMultiTelloEnv(wind_level=wind_level, preflight=preflight, v_max_xy=v_max_xy, profile=profile)
    infos = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=EVAL_SEED + i)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        infos.append(info)
    flown = [r for r in infos if r['go']]
    outcomes = Counter(o for r in flown for o in r['outcomes'])
    spreads = [max(t) - min(t) for r in flown for t in [[x for x in r['touchdown_times'] if x is not None]] if len(t) > 1]
    return {
        'wind': wind_level,
        'no_go_rate': 1 - len(flown) / len(infos),
        'all_success': float(np.mean([r['all_success'] for r in flown])) if flown else None,
        'drone_success': float(np.mean([r['n_success'] / mt.N_DRONES for r in flown])) if flown else None,
        'collision_rate': float(np.mean([r['n_collisions'] > 0 for r in flown])) if flown else None,
        'collided_drones': float(np.mean([r['n_collisions'] for r in flown])) if flown else None,
        'min_sep': float(np.mean([r['min_sep'] for r in flown])) if flown else None,
        'landing_spread': float(np.mean(spreads)) if spreads else None,
        'landing_error': float(np.mean([r['landing_error'] for r in flown if r['landing_error'] is not None]))
        if flown and any(r['landing_error'] is not None for r in flown) else None,
        'time': float(np.mean([r['time'] for r in flown if r['all_success']]))
        if flown and any(r['all_success'] for r in flown) else None,
        'outcomes': dict(outcomes),
    }


def main():
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', required=True, choices=['td3', 'ppo', 'fpo'])
    parser.add_argument('--run', required=True)
    parser.add_argument('--checkpoint', default=None, help='Defaults to <algo>_final.pt')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--winds', type=float, nargs='+', default=list(task.WIND_LEVELS))
    args = parser.parse_args()

    run = Path(args.run)
    ckpt = run / (args.checkpoint or f'{args.algo}_final.pt')
    policy = importlib.import_module(f'tello_rl.{args.algo}').make_policy(ckpt, mt.OBS_DIM, mt.ACT_DIM)

    fmt = lambda v, f='.1%': format(v, f) if v is not None else '-'
    print(f'{args.algo.upper()}  {ckpt}')
    print(f'{"Wind":>6} {"No-go":>6} {"Fleet success":>7} {"Drone success":>7} {"Collision rate":>6} {"Collided drones":>7} '
          f'{"Min separation":>7} {"Touchdown spread":>8} {"Total duration":>6} {"Landing error":>7}   Outcome counts')
    results = []
    for level in args.winds:
        r = evaluate(policy, level, args.episodes, preflight=level > task.MAX_WIND)
        results.append(r)
        print(f'{level:>4.0f} m/s {fmt(r["no_go_rate"]):>6} {fmt(r["all_success"]):>8} {fmt(r["drone_success"]):>8} '
              f'{fmt(r["collision_rate"]):>7} {fmt(r["collided_drones"], ".2f"):>8} {fmt(r["min_sep"], ".2f"):>7} m '
              f'{fmt(r["landing_spread"], ".1f"):>7} s {fmt(r["time"], ".1f"):>5} s {fmt(r["landing_error"], ".3f"):>6} m   '
              f'{r["outcomes"]}')
    out = run / f'eval_multi_{ckpt.stem}.json'
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
