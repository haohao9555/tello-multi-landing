"""降落任务的定义：观测、奖励、终止条件、风速等级。

快速替身环境（fast_env.py）和 Gazebo 环境共用这一份，保证两边训练的是同一个任务。
坐标约定：rel_pos = 无人机底部相对降落板中心的位置（m），z = 离板面高度。
"""
from collections import deque

import numpy as np

CONTROL_DT = 0.1            # 控制周期 10 Hz
MAX_STEPS = 300             # 30 s 超时
V_MAX_XY = 1.0              # 动作 [-1, 1] 对应的最大速度 m/s（与 tello_params.yaml 一致）
V_MAX_Z = 1.0

PAD_RADIUS = 0.15           # 成功着陆的水平误差上限 m
MAX_TOUCHDOWN_SPEED = 0.3   # 成功着陆的竖直速度上限 m/s
BOUNDS_XY = 2.5             # 水平出界距离 m
BOUNDS_Z = 4.0              # 高度上限 m

START_Z = (1.5, 3.0)        # 初始高度范围 m（从正常飞行高度降落，在风里待得更久）
START_XY = 1.0              # 初始水平偏移 ±m（先飞到板子上方再降）

MAX_WIND = 3.0              # 训练时平均风速在 [0, MAX_WIND] 内随机
WIND_LEVELS = (0.0, 1.0, 2.0, 3.0)   # 评估用的平均风速等级 m/s

# 真机因素（两种环境共用；置 0 即关闭，可逐项做消融）
ACTION_DELAY = (0, 2)       # 指令延迟范围（步），每回合随机；1 步 = 0.1 s，对应 WiFi 延迟抖动
POS_NOISE = 0.02            # 位置观测噪声 σ (m)
VEL_NOISE = 0.05            # 速度观测噪声 σ (m/s)
DROPOUT_PROB = 0.1          # 每步拿不到新位置的概率（视觉丢失降落板），此时沿用上一次的位置

# 动力学随机化（只在快速环境里；Gazebo 用标称值，正好检验泛化）
RANDOMIZE_DYNAMICS = True
KV_RANGE = (1.8, 3.2)       # 速度环增益 1/s（标称 2.5；响应时间常数 0.3~0.55 s）
DRAG_RANGE = (0.3, 0.6)     # 风阻 1/s（标称 0.41）
CMD_GAIN_RANGE = (0.8, 1.1)  # 指令 -> 实际速度的比例误差（标称 1.0）

OBS_DIM = 10                # [dx, dy, z, vx, vy, vz, 位置是否有效, 上一步动作 ×3]
ACT_DIM = 3                 # [vx, vy, vz] 指令，归一化到 [-1, 1]

# 奖励
SHAPING = 5.0               # 势函数塑形系数
TIME_PENALTY = 0.02         # 每步（悬停到超时共 -6，比尝试降落更亏）
SMOOTH_PENALTY = 0.05       # 动作变化惩罚（真机更平稳）
SUCCESS_BONUS = 10.0
PRECISION_BONUS = 5.0       # 落得越准越多
HARD_LANDING_PENALTY = 5.0  # 落在板上但太猛
MISS_PENALTY = (5.0, 10.0)  # 没落在板上：离板 0 -> 0.5 m 以上，扣分从 5 线性增加到 10
FAIL_PENALTY = 10.0         # 出界


def sample_start(rng):
    """初始位置：降落板上方 START_Z，水平偏移 ±START_XY，悬停。"""
    x, y = rng.uniform(-START_XY, START_XY, size=2)
    return np.array([x, y, rng.uniform(*START_Z)])


class WindModel:
    """回合内不断变化的风（水平，世界系，m/s）：平均风 + OU 湍流 + 随机阵风 + 风向缓慢漂移。

    湍流和阵风强度都按平均风速缩放，所以 0 级是真正的无风；3 m/s 时阵风约 1~2 m/s。
    快速环境和 Gazebo 环境每个控制步都调用 step()，两边的风完全一致。
    """
    TURB_STD = 0.25             # 湍流标准差 = 0.25 × 平均风速
    TURB_TAU = 1.0              # 湍流相关时间 s
    GUST_RATE = 0.4             # 平均每秒出现阵风的次数
    GUST_SCALE = (0.35, 0.65)   # 阵风强度 = 平均风速 × U(0.35, 0.65)
    GUST_DURATION = (0.5, 1.5)  # s
    DRIFT_MAX = np.deg2rad(30)  # 风向漂移幅度上限
    DRIFT_PERIOD = (10.0, 30.0)  # s

    def __init__(self, rng, level=None):
        self.rng = rng
        self.speed = rng.uniform(0.0, MAX_WIND) if level is None else level
        self.direction = rng.uniform(-np.pi, np.pi)
        self.drift_amp = rng.uniform(0.0, self.DRIFT_MAX)
        self.drift_period = rng.uniform(*self.DRIFT_PERIOD)
        self.drift_phase = rng.uniform(0.0, 2 * np.pi)
        self.turb = rng.normal(0.0, self.TURB_STD * self.speed, size=2)   # 从平稳分布开始
        self.gust = None   # (开始时间, 持续时间, 峰值向量)
        self.t = 0.0
        self.current = self._value()

    @property
    def mean(self):
        return self.speed * np.array([np.cos(self.direction), np.sin(self.direction)])

    def _value(self):
        theta = self.direction + self.drift_amp * np.sin(
            2 * np.pi * self.t / self.drift_period + self.drift_phase)
        wind = self.speed * np.array([np.cos(theta), np.sin(theta)]) + self.turb
        if self.gust is not None:
            start, duration, peak = self.gust
            phase = (self.t - start) / duration
            wind = wind + peak * 0.5 * (1 - np.cos(2 * np.pi * phase))   # 平滑起落
        return wind

    def step(self, dt):
        """推进 dt 秒，返回当前风速向量。"""
        self.t += dt
        decay = np.exp(-dt / self.TURB_TAU)
        self.turb = (decay * self.turb + self.TURB_STD * self.speed
                     * np.sqrt(1 - decay ** 2) * self.rng.normal(size=2))
        if self.gust is not None and self.t - self.gust[0] >= self.gust[1]:
            self.gust = None
        if self.gust is None and self.speed > 0 and self.rng.random() < self.GUST_RATE * dt:
            angle = self.rng.uniform(-np.pi, np.pi)
            peak = self.speed * self.rng.uniform(*self.GUST_SCALE) * np.array([np.cos(angle), np.sin(angle)])
            self.gust = (self.t, self.rng.uniform(*self.GUST_DURATION), peak)
        self.current = self._value()
        return self.current


def make_obs(rel_pos, vel, prev_action, pos_valid=1.0):
    return np.concatenate([rel_pos, vel, [pos_valid], prev_action]).astype(np.float32)


class RealismEffects:
    """指令延迟 + 传感器噪声 + 定位丢帧。环境每步：applied = delay(action)；obs = observe(真实状态)。"""

    def __init__(self, rng):
        self.rng = rng
        self.delay_steps = int(rng.integers(ACTION_DELAY[0], ACTION_DELAY[1] + 1))
        self.queue = deque([np.zeros(ACT_DIM)] * self.delay_steps)
        self.last_pos = None

    def delay(self, action):
        """返回 delay_steps 步之前发出的指令（这一步真正执行的）。"""
        self.queue.append(np.asarray(action, dtype=np.float64))
        return self.queue.popleft()

    def observe(self, rel_pos, vel, prev_action):
        """位置可能丢帧（沿用上次的值，有效位 = 0）；速度来自机载光流，总是有效。"""
        if self.last_pos is not None and self.rng.random() < DROPOUT_PROB:
            pos, valid = self.last_pos, 0.0
        else:
            pos, valid = rel_pos + self.rng.normal(0.0, POS_NOISE, 3), 1.0
            self.last_pos = pos
        return make_obs(pos, vel + self.rng.normal(0.0, VEL_NOISE, 3), prev_action, valid)


def potential(rel_pos):
    """离目标越近越高：水平距离 + 0.5 × 高度。"""
    return -(np.hypot(rel_pos[0], rel_pos[1]) + 0.5 * max(rel_pos[2], 0.0))


def step_outcome(prev_rel_pos, rel_pos, vel, action, prev_action, steps):
    """计算一步的奖励和是否结束。返回 (reward, terminated, truncated, info)。"""
    reward = (SHAPING * (potential(rel_pos) - potential(prev_rel_pos))
              - TIME_PENALTY
              - SMOOTH_PENALTY * float(np.sum((action - prev_action) ** 2)))
    d_xy = float(np.hypot(rel_pos[0], rel_pos[1]))

    if rel_pos[2] <= 0.0:   # 触地
        speed = abs(float(vel[2]))
        if d_xy <= PAD_RADIUS and speed <= MAX_TOUCHDOWN_SPEED:
            outcome = 'success'
            reward += SUCCESS_BONUS + PRECISION_BONUS * (1.0 - d_xy / PAD_RADIUS)
        elif d_xy <= PAD_RADIUS:
            outcome = 'hard_landing'
            reward -= HARD_LANDING_PENALTY
        else:
            outcome = 'missed'
            lo, hi = MISS_PENALTY
            reward -= lo + (hi - lo) * min((d_xy - PAD_RADIUS) / 0.5, 1.0)
        info = {'outcome': outcome, 'landing_error': d_xy, 'touchdown_speed': speed,
                'time': steps * CONTROL_DT}
        return reward, True, False, info

    if d_xy > BOUNDS_XY or rel_pos[2] > BOUNDS_Z:
        reward -= FAIL_PENALTY
        return reward, True, False, {'outcome': 'out_of_bounds', 'time': steps * CONTROL_DT}

    if steps >= MAX_STEPS:
        return reward, False, True, {'outcome': 'timeout', 'time': steps * CONTROL_DT}

    return reward, False, False, {}
