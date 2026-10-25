"""Record gazebo."""
import argparse
import importlib
import subprocess
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from tello_rl import task
from tello_rl.common import RUNS_DIR
from tello_rl.fast_env import FastTelloLandEnv
from tello_rl.gazebo import GazeboTello

WIDTH, HEIGHT = 1280, 720
HOLD_FRAMES = 15
TOUCH_Z = 0.005
OUTCOME_TEXT = {'success': ('SUCCESS', (40, 170, 60)), 'hard_landing': ('HARD LANDING', (210, 40, 40)),
                'missed': ('MISSED PAD', (210, 40, 40)), 'out_of_bounds': ('OUT OF BOUNDS', (210, 40, 40)),
                'timeout': ('TIMEOUT', (230, 140, 0))}
V_MAX = np.array([task.V_MAX_XY, task.V_MAX_XY, task.V_MAX_Z])

FONT = font_manager.findfont('DejaVu Sans')
BIG, SMALL = ImageFont.truetype(FONT, 34), ImageFont.truetype(FONT, 22)


OVERLAY = True


def draw_overlay(frame, label, wind, t, trail, outcome=None, mean_speed=0.0):
    if not OVERLAY:
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, 'RGBA')

    d.rectangle([0, 0, 520, 130], fill=(255, 255, 255, 190))
    d.text((16, 10), label, font=BIG, fill=(20, 20, 20))
    d.text((16, 56), f'wind {np.linalg.norm(wind):.1f} m/s (mean {mean_speed:.0f})   t = {t:4.1f} s', font=SMALL, fill=(20, 20, 20))
    if outcome:
        text, color = OUTCOME_TEXT[outcome['outcome']]
        err = outcome.get('landing_error')
        d.text((16, 88), text + (f'   error {err * 100:.0f} cm' if err is not None else ''),
               font=SMALL, fill=color)


    size, x0, y0 = 240, WIDTH - 256, 16
    to_px = lambda p: (x0 + size / 2 + p[0] / 2 * size, y0 + size / 2 - p[1] / 2 * size)
    d.rectangle([x0, y0, x0 + size, y0 + size], fill=(255, 255, 255, 200), outline=(80, 80, 80))
    a, b = to_px((-0.2, 0.2)), to_px((0.2, -0.2))
    d.rectangle([a, b], fill=(255, 140, 0, 220))
    a, b = to_px((-task.PAD_RADIUS, task.PAD_RADIUS)), to_px((task.PAD_RADIUS, -task.PAD_RADIUS))
    d.ellipse([a, b], outline=(255, 255, 255), width=2)
    if len(trail) > 1:
        d.line([to_px(p) for p in trail], fill=(30, 90, 200), width=2)
    cx, cy = to_px(trail[-1])
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(0, 0, 0))

    if np.linalg.norm(wind) > 0.05:
        c = np.array(to_px((0, 0)))
        tip = c + np.array([wind[0], -wind[1]]) / 3.0 * size * 0.4
        d.line([tuple(c), tuple(tip)], fill=(0, 170, 200), width=4)
        u = (tip - c) / np.linalg.norm(tip - c)
        n = np.array([-u[1], u[0]])
        d.polygon([tuple(tip + 10 * u), tuple(tip + 7 * n), tuple(tip - 7 * n)], fill=(0, 170, 200))
    d.text((x0 + 6, y0 + size - 26), 'top view  (cyan = wind)', font=ImageFont.truetype(FONT, 14),
           fill=(60, 60, 60))
    return np.asarray(img)


class FollowCamera:
    """Followcamera."""
    AZIMUTH = np.deg2rad(45)
    ELEVATION = np.deg2rad(25)

    def __init__(self):
        self.target, self.dist = None, None

    def update(self, sim, drone_rel, smooth=0.25):
        target = 0.55 * drone_rel + np.array([0.0, 0.0, 0.05])
        dist = max(0.6, 1.2 * np.linalg.norm(drone_rel) + 0.35)
        if self.target is None:
            self.target, self.dist = target, dist
        self.target += smooth * (target - self.target)
        self.dist += smooth * (dist - self.dist)
        back = np.array([-np.cos(self.AZIMUTH) * np.cos(self.ELEVATION),
                         -np.sin(self.AZIMUTH) * np.cos(self.ELEVATION),
                         np.sin(self.ELEVATION)])
        sim.set_camera(self.target + self.dist * back, self.AZIMUTH, self.ELEVATION)


def run_episode(sim, policy, label, wind_level, seed, write):

    scenario = FastTelloLandEnv(wind_level=wind_level)
    scenario.reset(seed=seed)
    start, wind_model = scenario.pos.copy(), scenario.wind
    wind = wind_model.current

    sim.enable(False)
    sim.set_wind(wind)
    sim.set_pose(start)
    sim.enable(True)

    rel, vel, prev_action = start.copy(), np.zeros(3), np.zeros(3)
    trail, steps, info = [start[:2]], 0, {}
    camera = FollowCamera()
    effects = task.RealismEffects(np.random.default_rng(seed))
    obs = effects.observe(rel, vel, prev_action)
    while True:
        action = policy(obs)
        camera.update(sim, rel)
        wind = wind_model.step(task.CONTROL_DT)
        sim.set_wind(wind)
        sim.step(effects.delay(action) * V_MAX)
        new_rel, new_vel, frame = sim.state()
        steps += 1

        touch_vel = new_vel.copy()
        if new_rel[2] <= TOUCH_Z:
            touch_vel[2] = vel[2]
        check_rel = new_rel - np.array([0.0, 0.0, TOUCH_Z])
        _, terminated, truncated, info = task.step_outcome(
            rel - np.array([0.0, 0.0, TOUCH_Z]), check_rel, touch_vel, action, prev_action, steps)
        rel, vel, prev_action = new_rel, new_vel, action
        obs = effects.observe(rel, vel, prev_action)
        trail.append(rel[:2])
        write(draw_overlay(frame, label, wind, steps * task.CONTROL_DT, trail, mean_speed=wind_model.speed))
        if terminated or truncated:
            break


    sim.enable(False)
    for _ in range(HOLD_FRAMES):
        camera.update(sim, rel)
        sim.step(np.zeros(3))
        _, _, frame = sim.state()
        write(draw_overlay(frame, label, wind, steps * task.CONTROL_DT, trail, info, wind_model.speed))
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True, help='algo:run_dir, for example ppo:runs/ppo_fast_s1')
    parser.add_argument('--winds', type=float, nargs='+', default=list(task.WIND_LEVELS))
    parser.add_argument('--seed', type=int, default=10_007)
    parser.add_argument('--out', default=str(RUNS_DIR / 'gazebo_landing.mp4'))
    parser.add_argument('--no-overlay', action='store_true', help='Record raw Gazebo frames without text or minimap overlays')
    args = parser.parse_args()
    global OVERLAY
    OVERLAY = not args.no_overlay

    algos = []
    for spec in args.runs:
        name, run = spec.split(':')
        module = importlib.import_module(f'tello_rl.{name}')
        algos.append((name.upper(), module.make_policy(Path(run) / f'{name}_final.pt',
                                                       task.OBS_DIM, task.ACT_DIM)))

    ffmpeg = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{WIDTH}x{HEIGHT}', '-r', str(int(round(1 / task.CONTROL_DT))), '-i', '-',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', args.out],
        stdin=subprocess.PIPE)
    write = lambda frame: ffmpeg.stdin.write(frame.tobytes())

    sim = GazeboTello(camera=True, width=WIDTH, height=HEIGHT)
    try:
        for i, wind in enumerate(args.winds):
            for name, policy in algos:
                info = run_episode(sim, policy, f'{name}  (trained in fast sim)', wind,
                                   args.seed + i, write)
                print(f'wind {wind:.0f} m/s  {name:>4}: {info.get("outcome")}'
                      + (f'  error {info["landing_error"] * 100:.0f} cm  '
                         f'touchdown {info["touchdown_speed"]:.2f} m/s' if 'landing_error' in info else ''),
                      flush=True)
    finally:
        sim.close()
        ffmpeg.stdin.close()
        ffmpeg.wait()
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
