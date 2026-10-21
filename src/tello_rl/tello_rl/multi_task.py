"""四机降落任务：4 架 Tello 一起起飞，从北、东、南、西飞来，逐架降落到 4 个分开的平台上（动态分配）。

坐标原点是降落站中心。平台在 (±PAD_OFFSET, ±PAD_OFFSET)，彼此之间留有空隙。
飞机落哪个平台、按什么顺序降落都不预先指定：成功的条件是落在一个空平台上；落到已被占用的平台算失败。
不能一起降落：降落区（站中心 ZONE_RADIUS 以内、ZONE_HEIGHT 以下）同一时刻只允许一架在飞的飞机，
其余在区外等待；多出来的每架每步扣 ZONE_PENALTY，评估时出现过违规的回合不算成功。
引导奖励用"当前最优分配"下的总距离（对所有分配方式取最小），不偏向任何一种分配；
高度上只引导获准降落的那架往下降，其余在平台上方 HOLD_Z 等待（和"一次只进一架"的规则一致）。
风、指令延迟、传感器噪声、定位丢帧沿用单机任务（task.py）。
"""
from itertools import permutations

import numpy as np

from tello_rl import task

N_DRONES = 4
PAD_OFFSET = 0.5            # 平台中心 (±0.5, ±0.5)：中心间距 1 m，0.4 m 的平台之间留 0.6 m 空隙
PADS = np.array([[PAD_OFFSET, PAD_OFFSET], [-PAD_OFFSET, PAD_OFFSET],
                 [-PAD_OFFSET, -PAD_OFFSET], [PAD_OFFSET, -PAD_OFFSET]])
DIRECTIONS = np.deg2rad([90.0, 0.0, 270.0, 180.0])   # 北、东、南、西

START_RADIUS = (2.0, 2.5)   # 起点离站中心的距离 m
START_LATERAL = 0.3         # 起点横向偏移 ±m
START_Z = (1.0, 1.5)        # 起飞后悬停高度 m
BOUNDS_XY = 3.5
BOUNDS_Z = 3.0
MAX_STEPS = 500             # 50 s（逐架降落需要排队）

ZONE_RADIUS = 1.2           # 降落区：站中心 1.2 m 以内（4 块平台最远角约 0.99 m）
ZONE_HEIGHT = 0.8           # 且高度 0.8 m 以下
ZONE_PENALTY = 1.0          # 降落区里每多一架在飞的飞机，每步扣分
HOLD_Z = 1.0                # 没获准降落的飞机在自己平台上方这个高度等待（高于降落区上限）

COLLISION_DIST = 0.25       # 两机中心距离小于它算相撞（Tello 加保护罩约 0.2 m）
SAFE_DIST = 0.5             # 小于安全距离开始按距离扣分
PROXIMITY_PENALTY = 0.1     # 每对飞机每步最多扣这么多

# 大风不起飞：起飞前测 10 s 平均风速，超过阈值整组不起飞
GO_THRESHOLD = 3.5          # m/s（训练风速 0~3 m/s，留一点余量）
PREFLIGHT_SECONDS = 10.0
ANEMOMETER_NOISE = 0.2      # 风速计测量误差 σ (m/s)

PER_DRONE_OBS = 12          # [x, y, z, vx, vy, vz, 位置是否有效, 是否在飞, 是否在降落区, 上一步指令 ×3]
OBS_DIM = N_DRONES * PER_DRONE_OBS + N_DRONES   # + 4 个平台是否已占用
ACT_DIM = N_DRONES * task.ACT_DIM


def sample_starts(rng):
    """每架飞机从自己的方向飞来：离站中心 2~2.5 m，横向偏移 ±0.3 m，高 1~1.5 m。"""
    starts = np.zeros((N_DRONES, 3))
    for i, theta in enumerate(DIRECTIONS):
        radial = np.array([np.cos(theta), np.sin(theta)])
        lateral = np.array([-radial[1], radial[0]])
        starts[i, :2] = (rng.uniform(*START_RADIUS) * radial
                         + rng.uniform(-START_LATERAL, START_LATERAL) * lateral)
        starts[i, 2] = rng.uniform(*START_Z)
    return starts


def preflight_go(wind_model, rng):
    """起飞前测风：用一份独立的风模型副本模拟 10 s 测量，平均风速 + 测量噪声与阈值比较。"""
    probe = task.WindModel(np.random.default_rng(rng.integers(2 ** 31)), wind_model.speed)
    probe.direction = wind_model.direction
    n = int(PREFLIGHT_SECONDS / task.CONTROL_DT)
    measured = np.mean([np.linalg.norm(probe.step(task.CONTROL_DT)) for _ in range(n)])
    measured += rng.normal(0.0, ANEMOMETER_NOISE)
    return measured <= GO_THRESHOLD, measured


def in_zone(pos):
    """每架飞机是否在降落区里（只看位置，是否在飞由调用方判断）。"""
    pos = np.atleast_2d(pos)
    return (np.linalg.norm(pos[:, :2], axis=1) <= ZONE_RADIUS) & (pos[:, 2] <= ZONE_HEIGHT)


def update_clearance(cleared, pos, active):
    """获准降落的飞机：第一个进入降落区的那架；它落地、出事或离开降落区后，下一架进入的才获准。"""
    inside = active & in_zone(pos)
    if cleared is not None and inside[cleared]:
        return cleared
    candidates = np.flatnonzero(inside)
    return int(candidates[np.argmin(pos[candidates, 2])]) if len(candidates) else None


def height_cost(z, is_cleared):
    """获准的飞机被引导降到平台面；其余在 HOLD_Z 等待。两者在降落区上限处相等，进区时引导项不跳变。"""
    z = max(z, 0.0)
    if is_cleared:
        return 0.5 * z - 0.5 * ZONE_HEIGHT + 0.5 * abs(ZONE_HEIGHT - HOLD_Z)
    return 0.5 * abs(z - HOLD_Z)


def assignment_cost(pos, active, occupied, cleared=None):
    """在飞的飞机分配到空平台的最小总代价：Σ 水平距离 + Σ 高度项（见 height_cost）。"""
    drones = np.flatnonzero(active)
    free = np.flatnonzero(~occupied)
    if len(drones) == 0:
        return 0.0
    dist = np.linalg.norm(pos[drones, None, :2] - PADS[None, free], axis=-1)
    height = np.array([height_cost(pos[i, 2], i == cleared) for i in drones])
    best = min(sum(dist[i, j] for i, j in enumerate(perm))
               for perm in permutations(range(len(free)), len(drones)))
    return best + height.sum()


def nearest_pad(xy):
    d = np.linalg.norm(PADS - xy, axis=1)
    k = int(np.argmin(d))
    return k, float(d[k])


def check_touchdown(xy, vz, occupied):
    """一架飞机触地时的结果。返回 (outcome, 平台编号或 None, 奖励, 着陆误差)。"""
    k, d = nearest_pad(xy)
    speed = abs(float(vz))
    if d <= task.PAD_RADIUS:
        if occupied[k]:
            return 'occupied', None, -task.FAIL_PENALTY, d
        if speed <= task.MAX_TOUCHDOWN_SPEED:
            return 'success', k, task.SUCCESS_BONUS + task.PRECISION_BONUS * (1 - d / task.PAD_RADIUS), d
        return 'hard_landing', None, -task.HARD_LANDING_PENALTY, d
    lo, hi = task.MISS_PENALTY
    return 'missed', None, -(lo + (hi - lo) * min((d - task.PAD_RADIUS) / 0.5, 1.0)), d
