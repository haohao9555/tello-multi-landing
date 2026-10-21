"""在 Gazebo 四机场景里回放 MAPPO / HAPPO（逐架降落），录成功和失败的片段。

每个片段由 "风速:类型" 指定，如 1:success、3:failure。先用快速环境按种子筛出大概率是这种结果的场景，
再到 Gazebo 里跑；片段上标的是 Gazebo 里实际发生的结果（和训练环境同一套判定规则）。
相撞的飞机会关电机掉下来。

  python -m tello_rl.record_gazebo_multi runs/marl/mappo/.../latest.pt --cases 1:success 3:success 3:failure
输出 runs/gazebo_multi_cases.mp4 和同名 .gif
"""
import os

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '1')

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from tello_rl import multi_task as mt
from tello_rl import task
from tello_rl.common import RUNS_DIR
from tello_rl.evaluate_marl import load_policy
from tello_rl.gazebo_multi import DRONE_COLORS
from tello_rl.multi_env import FastMultiTelloEnv
from tello_rl.playback import GazeboPlayer

N = mt.N_DRONES
WIDTH, HEIGHT = 1920, 1080
HOLD_FRAMES = 20
NAMES = ['N', 'E', 'S', 'W']
COLORS = [tuple(int(255 * c) for c in rgb) for rgb in DRONE_COLORS]
OUTCOME_TEXT = {'success': 'LANDED', 'missed': 'MISSED PAD', 'hard_landing': 'HARD LANDING',
                'occupied': 'PAD TAKEN', 'collision': 'COLLISION', 'out_of_bounds': 'OUT OF BOUNDS',
                'timeout': 'TIMEOUT', 'no_go': 'NO TAKEOFF (wind)'}
FONT = font_manager.findfont('DejaVu Sans')
BIG, MID, SMALL = (ImageFont.truetype(FONT, s) for s in (44, 30, 20))


def draw_overlay(frame, label, wind, mean_speed, t, outcomes, errors, trails, final=None, violations=0):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, 'RGBA')
    d.rectangle([0, 0, 640, 330], fill=(255, 255, 255, 200))
    d.text((20, 12), label, font=BIG, fill=(20, 20, 20))
    d.text((20, 70), f'wind {np.linalg.norm(wind):.1f} m/s (mean {mean_speed:.0f})   t = {t:4.1f} s',
           font=MID, fill=(20, 20, 20))
    d.text((360, 12), f'zone violations: {violations}', font=SMALL, fill=(200, 40, 40) if violations else (60, 60, 60))
    for i in range(N):
        y = 120 + 48 * i
        d.rectangle([22, y + 4, 52, y + 34], fill=COLORS[i])
        if outcomes[i] is None:
            text, color = f'{NAMES[i]}: flying', (60, 60, 60)
        else:
            text = f'{NAMES[i]}: {OUTCOME_TEXT[outcomes[i]]}'
            if errors[i] is not None:
                text += f'  ({errors[i] * 100:.0f} cm)'
            color = (30, 150, 50) if outcomes[i] == 'success' else (200, 40, 40)
        d.text((66, y), text, font=MID, fill=color)
    if final is not None:
        ok = all(o == 'success' for o in outcomes) and violations == 0
        banner = 'ALL 4 LANDED, ONE BY ONE' if ok else 'FAILED'
        d.rectangle([WIDTH // 2 - 300, HEIGHT - 130, WIDTH // 2 + 300, HEIGHT - 40],
                    fill=(40, 160, 60, 220) if ok else (200, 40, 40, 220))
        d.text((WIDTH // 2, HEIGHT - 85), banner, font=BIG, fill=(255, 255, 255), anchor='mm')

    # 右上角俯视小地图：±2.8 m
    size, x0, y0, span = 360, WIDTH - 380, 20, 2.8
    to_px = lambda p: (x0 + size / 2 + p[0] / span * size / 2, y0 + size / 2 - p[1] / span * size / 2)
    d.rectangle([x0, y0, x0 + size, y0 + size], fill=(255, 255, 255, 210), outline=(80, 80, 80))
    za, zb = to_px((-mt.ZONE_RADIUS, mt.ZONE_RADIUS)), to_px((mt.ZONE_RADIUS, -mt.ZONE_RADIUS))
    d.ellipse([za, zb], outline=(200, 40, 40), width=2)   # 降落区
    for pad in mt.PADS:
        a, b = to_px(pad + [-0.2, 0.2]), to_px(pad + [0.2, -0.2])
        d.rectangle([a, b], fill=(255, 140, 0, 230))
    for i in range(N):
        if len(trails[i]) > 1:
            d.line([to_px(p) for p in trails[i]], fill=COLORS[i], width=3)
        cx, cy = to_px(trails[i][-1])
        d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=COLORS[i], outline=(0, 0, 0))
    if np.linalg.norm(wind) > 0.05:
        c = np.array(to_px((0, 0)))
        tip = c + np.array([wind[0], -wind[1]]) / 4.0 * size * 0.35
        d.line([tuple(c), tuple(tip)], fill=(0, 170, 200), width=5)
        u = (tip - c) / np.linalg.norm(tip - c)
        n = np.array([-u[1], u[0]])
        d.polygon([tuple(tip + 14 * u), tuple(tip + 9 * n), tuple(tip - 9 * n)], fill=(0, 170, 200))
    d.text((x0 + 8, y0 + size - 28), 'top view  (cyan = wind)', font=SMALL, fill=(60, 60, 60))
    return np.asarray(img)


def run_episode(player, policy, label, wind_level, seed, write):
    """用 playback.GazeboPlayer 跑一回合，叠加文字后写视频。返回 (每架结果, 违规步数)。"""
    trails = [[] for _ in range(N)]
    frame = None
    for frame in player.play(policy, wind_level, seed, hold_frames=HOLD_FRAMES):
        for i in range(N):
            trails[i].append(frame.positions[i, :2].copy())
        outcomes = [None if s == 'flying' else s for s in frame.status]
        write(draw_overlay(frame.image, label, frame.wind, frame.wind_mean, frame.t, outcomes,
                           frame.landing_errors, trails, final=frame.done or None,
                           violations=frame.zone_violations))
    return frame.status, frame.zone_violations


def candidate_seeds(policy, wind_level, want_success, start=20_000, limit=300):
    """先在快速环境里找"大概率是这种结果"的种子。"""
    env = FastMultiTelloEnv(wind_level=wind_level)
    for seed in range(start, start + limit):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        if info['all_success'] == want_success:
            yield seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--cases', nargs='+', default=['1:success', '3:success', '3:failure'],
                        help='风速:success、风速:failure，或 风速:类型:种子（直接录这个种子）')
    parser.add_argument('--tries', type=int, default=6, help='每个片段在 Gazebo 里最多试几个场景')
    parser.add_argument('--out', default=str(RUNS_DIR / 'gazebo_multi_cases.mp4'))
    args = parser.parse_args()

    policy, algo = load_policy(args.checkpoint)
    workdir = Path(tempfile.mkdtemp(prefix='gz_cases_', dir=RUNS_DIR))
    player = GazeboPlayer(width=WIDTH, height=HEIGHT)
    clips = []
    try:
        for n_case, case in enumerate(args.cases):
            wind_text, kind, *fixed = case.split(':')
            wind_level, want_success = float(wind_text), kind == 'success'
            seeds = [int(fixed[0])] if fixed else candidate_seeds(policy, wind_level, want_success)
            for n_try, seed in enumerate(seeds):
                if n_try >= args.tries:
                    print(f'{case}: {args.tries} 次都没在 Gazebo 里得到想要的结果，跳过', flush=True)
                    break
                clip = workdir / f'case{n_case}_try{n_try}.mp4'
                ffmpeg = subprocess.Popen(
                    ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                     '-s', f'{WIDTH}x{HEIGHT}', '-r', str(int(round(1 / task.CONTROL_DT))), '-i', '-',
                     '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22', str(clip)],
                    stdin=subprocess.PIPE)
                outcomes, violations = run_episode(player, policy, f'{algo.upper()}   case {n_case + 1}', wind_level,
                                                   seed, lambda f: ffmpeg.stdin.write(f.tobytes()))
                ffmpeg.stdin.close()
                ffmpeg.wait()
                got_success = all(o == 'success' for o in outcomes) and violations == 0
                print(f'{case} seed {seed}: Gazebo -> {outcomes}, 违规 {violations} 步', flush=True)
                if fixed or got_success == want_success:
                    clips.append(clip)
                    break
    finally:
        player.close()

    listing = workdir / 'clips.txt'
    listing.write_text(''.join(f"file '{c}'\n" for c in clips))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing),
                    '-c', 'copy', args.out], check=True)
    gif = Path(args.out).with_suffix('.gif')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', args.out, '-vf',
                    'fps=10,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',
                    str(gif)], check=True)
    print(f'-> {args.out}\n-> {gif}\n（单个片段在 {workdir}）')


if __name__ == '__main__':
    main()
