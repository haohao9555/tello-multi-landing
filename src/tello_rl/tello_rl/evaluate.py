"""统一评估：每个风速等级跑 N 个回合（固定种子，所有算法面对完全相同的场景）。

用法：
  python -m tello_rl.evaluate --algo ppo --run runs/ppo_fast_s1 --episodes 100
输出：每个风级的成功率、着陆误差、着陆速度、用时，并保存 eval_<env>.json
"""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from tello_rl import task
from tello_rl.common import make_env

EVAL_SEED = 10_000   # 评估种子从这里开始，和训练种子错开


def evaluate(policy, env_id, wind_level, n_episodes):
    env = make_env(env_id, wind_level=wind_level)
    records = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=EVAL_SEED + i)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        records.append(info)
    env.close()

    success = np.array([r['outcome'] == 'success' for r in records])
    touched = [r for r in records if 'landing_error' in r]
    outcomes = {k: sum(r['outcome'] == k for r in records)
                for k in ('success', 'hard_landing', 'missed', 'out_of_bounds', 'timeout')}
    return {
        'wind': wind_level,
        'success_rate': float(success.mean()),
        'landing_error': float(np.mean([r['landing_error'] for r in touched])) if touched else None,
        'touchdown_speed': float(np.mean([r['touchdown_speed'] for r in touched])) if touched else None,
        'time': float(np.mean([r['time'] for r in records if r['outcome'] == 'success']))
                if success.any() else None,
        'outcomes': outcomes,
    }


def main():
    import torch
    torch.set_num_threads(1)   # 容器只有约 7.7 核，还有别的实验在跑
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', required=True, choices=['td3', 'ppo', 'fpo'])
    parser.add_argument('--run', required=True, help='训练输出目录，如 runs/ppo_fast_s1')
    parser.add_argument('--env', default='fast')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--checkpoint', default=None, help='默认用 <algo>_final.pt')
    parser.add_argument('--winds', type=float, nargs='+', default=list(task.WIND_LEVELS),
                        help='平均风速等级；可加训练时没见过的 4 测泛化')
    args = parser.parse_args()

    run = Path(args.run)
    ckpt = run / (args.checkpoint or f'{args.algo}_final.pt')
    module = importlib.import_module(f'tello_rl.{args.algo}')
    policy = module.make_policy(ckpt, task.OBS_DIM, task.ACT_DIM)

    results = []
    print(f'{args.algo.upper()}  {ckpt}')
    print(f'{"风速":>6} {"成功率":>8} {"着陆误差":>10} {"着陆速度":>10} {"用时":>8}   结果分布')
    for level in args.winds:
        r = evaluate(policy, args.env, level, args.episodes)
        results.append(r)
        fmt = lambda v, f: format(v, f) if v is not None else '-'
        print(f'{level:>4.0f} m/s {r["success_rate"] * 100:>7.1f}% {fmt(r["landing_error"], ".3f"):>9} m '
              f'{fmt(r["touchdown_speed"], ".2f"):>7} m/s {fmt(r["time"], ".1f"):>6} s   {r["outcomes"]}')

    out = run / f'eval_{args.env}.json'
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
