"""Landing task definitions, rewards, termination rules and stochastic wind."""
from collections import deque

import numpy as np

CONTROL_DT = 0.1
MAX_STEPS = 300
V_MAX_XY = 1.0
V_MAX_Z = 1.0

PAD_RADIUS = 0.15
MAX_TOUCHDOWN_SPEED = 0.3
BOUNDS_XY = 2.5
BOUNDS_Z = 4.0

START_Z = (1.5, 3.0)
START_XY = 1.0

MAX_WIND = 3.0
WIND_LEVELS = (0.0, 1.0, 2.0, 3.0)


ACTION_DELAY = (0, 2)
POS_NOISE = 0.02
VEL_NOISE = 0.05
DROPOUT_PROB = 0.1


RANDOMIZE_DYNAMICS = True
KV_RANGE = (1.8, 3.2)
DRAG_RANGE = (0.3, 0.6)
CMD_GAIN_RANGE = (0.8, 1.1)

OBS_DIM = 10
ACT_DIM = 3


SHAPING = 5.0
TIME_PENALTY = 0.02
SMOOTH_PENALTY = 0.05
SUCCESS_BONUS = 10.0
PRECISION_BONUS = 5.0
HARD_LANDING_PENALTY = 5.0
MISS_PENALTY = (5.0, 10.0)
FAIL_PENALTY = 10.0


def sample_start(rng):
    """Sample start."""
    x, y = rng.uniform(-START_XY, START_XY, size=2)
    return np.array([x, y, rng.uniform(*START_Z)])


class WindModel:
    """Windmodel."""
    TURB_STD = 0.25
    TURB_TAU = 1.0
    GUST_RATE = 0.4
    GUST_SCALE = (0.35, 0.65)
    GUST_DURATION = (0.5, 1.5)  # s
    DRIFT_MAX = np.deg2rad(30)
    DRIFT_PERIOD = (10.0, 30.0)  # s

    def __init__(self, rng, level=None, constant=False):
        """Init."""
        self.constant = constant
        self.rng = rng
        self.speed = rng.uniform(0.0, MAX_WIND) if level is None else level
        self.direction = rng.uniform(-np.pi, np.pi)
        self.drift_amp = rng.uniform(0.0, self.DRIFT_MAX)
        self.drift_period = rng.uniform(*self.DRIFT_PERIOD)
        self.drift_phase = rng.uniform(0.0, 2 * np.pi)
        self.turb = rng.normal(0.0, self.TURB_STD * self.speed, size=2)
        self.gust = None
        self.t = 0.0
        self.current = self._value()

    @property
    def mean(self):
        return self.speed * np.array([np.cos(self.direction), np.sin(self.direction)])

    def _value(self):
        if self.constant:
            return self.mean
        theta = self.direction + self.drift_amp * np.sin(
            2 * np.pi * self.t / self.drift_period + self.drift_phase)
        wind = self.speed * np.array([np.cos(theta), np.sin(theta)]) + self.turb
        if self.gust is not None:
            start, duration, peak = self.gust
            phase = (self.t - start) / duration
            wind = wind + peak * 0.5 * (1 - np.cos(2 * np.pi * phase))
        return wind

    def step(self, dt):
        """Step."""
        self.t += dt
        if self.constant:
            self.current = self.mean
            return self.current
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
    """Realismeffects."""

    def __init__(self, rng):
        self.rng = rng
        self.delay_steps = int(rng.integers(ACTION_DELAY[0], ACTION_DELAY[1] + 1))
        self.queue = deque([np.zeros(ACT_DIM)] * self.delay_steps)
        self.last_pos = None

    def delay(self, action):
        """Delay."""
        self.queue.append(np.asarray(action, dtype=np.float64))
        return self.queue.popleft()

    def observe(self, rel_pos, vel, prev_action):
        """Observe."""
        if self.last_pos is not None and self.rng.random() < DROPOUT_PROB:
            pos, valid = self.last_pos, 0.0
        else:
            pos, valid = rel_pos + self.rng.normal(0.0, POS_NOISE, 3), 1.0
            self.last_pos = pos
        return make_obs(pos, vel + self.rng.normal(0.0, VEL_NOISE, 3), prev_action, valid)


def potential(rel_pos):
    """Potential."""
    return -(np.hypot(rel_pos[0], rel_pos[1]) + 0.5 * max(rel_pos[2], 0.0))


def step_outcome(prev_rel_pos, rel_pos, vel, action, prev_action, steps):
    """Step outcome."""
    reward = (SHAPING * (potential(rel_pos) - potential(prev_rel_pos))
              - TIME_PENALTY
              - SMOOTH_PENALTY * float(np.sum((action - prev_action) ** 2)))
    d_xy = float(np.hypot(rel_pos[0], rel_pos[1]))

    if rel_pos[2] <= 0.0:
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
