"""评估 train_marl.py 训练的 MAPPO / HAPPO checkpoint，统计口径与 evaluate_multi.py 完全相同。

  python -m tello_rl.evaluate_marl runs/marl/mappo/tello_4drone/seed_1/<时间戳>/latest.pt
  python -m tello_rl.evaluate_marl <checkpoint> --winds 0 1 2 3 4 5
"""
import os

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '1')

import argparse
import importlib
import json
from pathlib import Path

import torch

from tello_rl import task
from tello_rl.evaluate_multi import evaluate
from tello_rl.marl_env import N, N_ACTIONS, VectorTelloMARL, agent_observations, to_world_actions
from tello_rl.train_marl import ALGORITHMS


def load_policy(checkpoint, device='cpu'):
    torch.set_num_threads(1)   # 容器只有约 7.7 核，还有别的实验在跑
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = payload['config']
    config['device'] = device
    if config['network']['use_gru']:
        raise ValueError('只支持 MLP actor（use_gru=false）')
    spec = VectorTelloMARL(1, 0).spec
    env_info = {'n_agents': spec.n_agents, 'obs_dim': spec.obs_dim, 'state_dim': spec.state_dim,
                'n_actions': spec.n_actions, 'observation_mask': spec.observation_mask,
                'action_mask': spec.action_mask}
    module, name = ALGORITHMS[config['algorithm']]
    algorithm = getattr(importlib.import_module(f'algorithms.{module}'), name)(env_info, config)
    algorithm.load_checkpoint_state(payload['algorithm_state'], resume=False)
    algorithm.train(False)
    latent = torch.zeros(1, N, N_ACTIONS, device=device)

    def policy(flat_obs):
        """确定性动作（不加噪声）：环境 48 维观测 -> 12 维世界系动作。"""
        obs = torch.as_tensor(agent_observations(flat_obs), device=device).unsqueeze(0)
        with torch.no_grad():
            sample, _ = algorithm.act(algorithm.actor_inputs(obs), algorithm.init_hidden(1), latent, False)
        return to_world_actions(sample['actions'][0].cpu().numpy())
    return policy, config['algorithm']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--winds', type=float, nargs='+', default=list(task.WIND_LEVELS))
    args = parser.parse_args()

    policy, algo = load_policy(args.checkpoint)
    fmt = lambda v, f='.1%': format(v, f) if v is not None else '-'
    print(f'{algo.upper()}  {args.checkpoint}')
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
    out = args.checkpoint.with_name(f'eval_{args.checkpoint.stem}.json')
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
