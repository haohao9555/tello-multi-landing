"""Read project checkpoints and run isolated GUI evaluation episodes."""
from pathlib import Path
from collections import deque
import sys
import time
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/tello_rl'))
from tello_rl import multi_task as mt, task
from tello_rl.playback import load_policy, GazeboPlayer
from tello_rl.multi_env import FastMultiTelloEnv


def checkpoints(profile):
    found = []
    for path in (ROOT / 'runs/marl').rglob('config.yaml'):
        config = yaml.safe_load(path.read_text())
        if config.get('env', {}).get('profile', 'tello') != profile:
            continue
        candidates = [p for p in path.parent.glob('*.pt') if p.is_file()]
        if not candidates:
            continue
        # Prefer final weights and the latest checkpoint; retain one per experiment.
        selected = max(candidates, key=lambda p: (p.name == 'latest.pt', p.stat().st_mtime))
        found.append((config.get('output', {}).get('experiment', path.parent.name), selected))
    return sorted(found, key=lambda item: item[1].stat().st_mtime, reverse=True)


def model_info(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    c, state = payload['config'], payload['algorithm_state']
    actor = state['actor']
    layers = [f'{k}: {tuple(v.shape)}' for k, v in actor.items() if k.endswith('weight')]
    count = lambda s: sum(v.numel() for k, v in s.items()
                              if k.endswith(('weight', 'bias', 'raw_sigma')))
    return c, dict(actor_params=count(actor), critic_params=count(state['critic']),
                   layers=layers, steps=state.get('total_steps', 'Unknown'))


def configure_delay(effects, milliseconds):
    steps = int(round(milliseconds / (task.CONTROL_DT * 1000)))
    for effect in effects:
        effect.delay_steps = steps
        effect.queue = deque(np.zeros(3) for _ in range(steps))
    return steps


def episode(path, wind, speed, delay_ms, mode, seed, stopped, camera_target=None):
    if wind > 15:
        raise ValueError('Takeoff blocked: wind speed exceeds 15 m/s.')
    policy, _ = load_policy(path)
    policy.v_max_xy = speed
    if mode == 'Gazebo':
        if policy.profile.name != 'tello':
            raise ValueError('The project Gazebo model currently supports Tello only. Select Fast simulation for M30.')
        with GazeboPlayer(width=1100, height=650) as player:
            for frame in player.play(policy, wind, seed=seed, delay_ms=delay_ms, camera_target=camera_target):
                if stopped():
                    break
                yield dict(t=frame.t, positions=frame.positions, velocities=frame.velocities,
                           status=frame.status, wind=frame.wind, errors=frame.landing_errors,
                           image=frame.image, targets=frame.targets, done=frame.done,
                           bounds=frame.profile.bounds_xy)
        return
    env = FastMultiTelloEnv(wind_level=wind, preflight=True, v_max_xy=speed, profile=policy.profile)
    try:
        obs, info = env.reset(seed=seed)
        configure_delay(env.effects, delay_ms)
        if not info['go']:
            yield dict(t=0, positions=env.pos, velocities=env.vel, status=['no_go']*4,
                       wind=env.wind.current, errors=[None]*4, image=None, targets=env.targets,
                       done=True, bounds=env.profile.bounds_xy)
            return
        obs = env._observe()
        while not stopped():
            start = time.monotonic()
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            done = terminated or truncated
            yield dict(t=env.steps * task.CONTROL_DT, positions=env.pos.copy(), velocities=env.vel.copy(),
                       status=[o or ('timeout' if done else 'flying') for o in env.outcomes],
                       wind=env.wind.current.copy(), errors=list(env.landing_errors), image=None,
                       targets=env.targets, done=done, bounds=env.profile.bounds_xy)
            if done:
                break
            time.sleep(max(0, task.CONTROL_DT - (time.monotonic()-start)))
    finally:
        env.close()
