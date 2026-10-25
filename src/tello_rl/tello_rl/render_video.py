"""Render video."""
import argparse
import importlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from tello_rl import task
from tello_rl.common import RUNS_DIR, make_env

OUTCOME_TEXT = {'success': ('SUCCESS', 'green'), 'hard_landing': ('HARD LANDING', 'red'),
                'missed': ('MISSED PAD', 'red'), 'out_of_bounds': ('OUT OF BOUNDS', 'red'),
                'timeout': ('TIMEOUT', 'orange')}


def rollout(policy, env_id, wind_level, seed):
    env = make_env(env_id, wind_level=wind_level)
    obs, _ = env.reset(seed=seed)
    frames = [(env.pos.copy(), env.wind.current.copy())]
    done, info = False, {}
    while not done:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        frames.append((env.pos.copy(), env.wind.current.copy()))
        done = terminated or truncated
    return frames, info


def draw_pad(ax):
    s = 0.2
    ax.plot([-s, s, s, -s, -s], [-s, -s, s, s, -s], 0, color='darkorange', lw=2)
    t = np.linspace(0, 2 * np.pi, 60)
    ax.plot(task.PAD_RADIUS * np.cos(t), task.PAD_RADIUS * np.sin(t), 0, color='green', lw=1.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True, help='algo:run_dir, for example ppo:runs/ppo_fast_s1')
    parser.add_argument('--env', default='fast')
    parser.add_argument('--winds', type=float, nargs='+', default=list(task.WIND_LEVELS))
    parser.add_argument('--seed', type=int, default=10_007)
    parser.add_argument('--out', default=str(RUNS_DIR / 'landing_compare.mp4'))
    args = parser.parse_args()

    algos = []
    for spec in args.runs:
        name, run = spec.split(':')
        module = importlib.import_module(f'tello_rl.{name}')
        algos.append((name.upper(), module.make_policy(Path(run) / f'{name}_final.pt',
                                                       task.OBS_DIM, task.ACT_DIM)))

    fig = plt.figure(figsize=(5.2 * len(algos), 5.6))
    axes = [fig.add_subplot(1, len(algos), i + 1, projection='3d') for i in range(len(algos))]
    writer = FFMpegWriter(fps=int(round(1 / task.CONTROL_DT)))

    with writer.saving(fig, args.out, dpi=90):
        for wind in args.winds:
            results = [rollout(policy, args.env, wind, args.seed) for _, policy in algos]
            n_frames = max(len(frames) for frames, _ in results) + 15
            for k in range(n_frames):
                for ax, (name, _), (frames, info) in zip(axes, algos, results):
                    ax.cla()
                    i = min(k, len(frames) - 1)
                    traj = np.array([p for p, _ in frames[:i + 1]])
                    pos, w = frames[i]
                    draw_pad(ax)
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='tab:blue', lw=1.5)
                    ax.plot(traj[:, 0], traj[:, 1], 0, color='gray', lw=0.8, alpha=0.5)
                    ax.scatter(*pos, color='black', s=40)

                    ax.quiver(-0.9, -0.9, 1.4, w[0] / 3, w[1] / 3, 0, color='tab:cyan', lw=2)
                    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_zlim(0, 1.6)
                    ax.set_xlabel('x (m)'), ax.set_ylabel('y (m)'), ax.set_zlabel('z (m)')
                    ax.view_init(elev=22, azim=-60)
                    status = f't = {i * task.CONTROL_DT:4.1f} s'
                    color = 'black'
                    if k >= len(frames) - 1:
                        text, color = OUTCOME_TEXT[info['outcome']]
                        err = info.get('landing_error')
                        status = f'{text}' + (f'  error {err * 100:.0f} cm' if err is not None else '')
                    ax.set_title(f'{name}\n{status}', color=color, fontsize=12)
                fig.suptitle(f'Wind {wind:.0f} m/s (cyan arrow = wind direction)', fontsize=14)
                writer.grab_frame()
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
