"""四机评估：每个风级跑 N 个回合（固定种子），统计全部成功率、单机成功率、降落区违规、各种失败、最小间距。

  python -m tello_rl.evaluate_multi --algo ppo --run runs/ppo_multi_v1_s1 --checkpoint ppo_step600000.pt
加 --winds 4 5 时会启用起飞前测风，统计"大风不起飞"判断。
"""
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


def evaluate(policy, wind_level, n_episodes, preflight):
    env = FastMultiTelloEnv(wind_level=wind_level, preflight=preflight)
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
    return {
        'wind': wind_level,
        'no_go_rate': 1 - len(flown) / len(infos),
        'all_success': float(np.mean([r['all_success'] for r in flown])) if flown else None,
        'all_landed': float(np.mean([r['all_landed'] for r in flown])) if flown else None,
        'violation_rate': float(np.mean([r['zone_violation_steps'] > 0 for r in flown])) if flown else None,
        'min_landing_gap': float(np.mean([r['min_landing_gap'] for r in flown if r['min_landing_gap'] is not None]))
        if flown and any(r['min_landing_gap'] is not None for r in flown) else None,
        'drone_success': float(np.mean([r['n_success'] / mt.N_DRONES for r in flown])) if flown else None,
        'collision_rate': float(np.mean(['collision' in r['outcomes'] for r in flown])) if flown else None,
        'min_sep': float(np.mean([r['min_sep'] for r in flown])) if flown else None,
        'landing_error': float(np.mean([r['landing_error'] for r in flown if r['landing_error'] is not None]))
        if flown else None,
        'time': float(np.mean([r['time'] for r in flown if r['all_success']]))
        if flown and any(r['all_success'] for r in flown) else None,
        'outcomes': dict(outcomes),
    }


def main():
    import torch
    torch.set_num_threads(1)   # 容器只有约 7.7 核，还有别的实验在跑
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', required=True, choices=['td3', 'ppo', 'fpo'])
    parser.add_argument('--run', required=True)
    parser.add_argument('--checkpoint', default=None, help='默认用 <algo>_final.pt')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--winds', type=float, nargs='+', default=list(task.WIND_LEVELS))
    args = parser.parse_args()

    run = Path(args.run)
    ckpt = run / (args.checkpoint or f'{args.algo}_final.pt')
    policy = importlib.import_module(f'tello_rl.{args.algo}').make_policy(ckpt, mt.OBS_DIM, mt.ACT_DIM)

    fmt = lambda v, f='.1%': format(v, f) if v is not None else '-'
    print(f'{args.algo.upper()}  {ckpt}')
    print(f'{"风速":>6} {"不起飞":>6} {"全部成功":>7} {"全落地":>6} {"有违规":>6} {"单机成功":>7} {"有相撞":>6} '
          f'{"落地间隔":>7} {"总用时":>6} {"着陆误差":>7}   结果分布')
    results = []
    for level in args.winds:
        r = evaluate(policy, level, args.episodes, preflight=level > task.MAX_WIND)
        results.append(r)
        print(f'{level:>4.0f} m/s {fmt(r["no_go_rate"]):>6} {fmt(r["all_success"]):>8} {fmt(r["all_landed"]):>7} '
              f'{fmt(r["violation_rate"]):>7} {fmt(r["drone_success"]):>8} {fmt(r["collision_rate"]):>7} '
              f'{fmt(r["min_landing_gap"], ".1f"):>6} s {fmt(r["time"], ".1f"):>5} s {fmt(r["landing_error"], ".3f"):>6} m   '
              f'{r["outcomes"]}')
    out = run / f'eval_multi_{ckpt.stem}.json'
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
