"""Fast env."""
import gymnasium as gym
import numpy as np

from tello_rl import task

KV = 2.5
DRAG = 0.41
A_MAX = 3.0     # maximumLinearAcceleration m/s^2


class FastTelloLandEnv(gym.Env):
    def __init__(self, wind_level=None, constant_wind=False, substeps=10):
        self.wind_level = wind_level
        self.constant_wind = constant_wind
        self.substeps = substeps
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (task.OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (task.ACT_DIM,), np.float32)
        self.v_max = np.array([task.V_MAX_XY, task.V_MAX_XY, task.V_MAX_Z])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.pos = task.sample_start(rng)
        self.vel = np.zeros(3)
        self.prev_action = np.zeros(3)
        self.wind = task.WindModel(rng, self.wind_level, self.constant_wind)
        self.effects = task.RealismEffects(rng)
        if task.RANDOMIZE_DYNAMICS:
            self.kv = rng.uniform(*task.KV_RANGE)
            self.drag = rng.uniform(*task.DRAG_RANGE)
            self.cmd_gain = rng.uniform(*task.CMD_GAIN_RANGE)
        else:
            self.kv, self.drag, self.cmd_gain = KV, DRAG, 1.0
        self.steps = 0
        return self.effects.observe(self.pos, self.vel, self.prev_action), {'wind': self.wind.mean}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v_cmd = self.effects.delay(action) * self.v_max * self.cmd_gain
        prev_pos = self.pos.copy()
        wind = self.wind.step(task.CONTROL_DT)
        h = task.CONTROL_DT / self.substeps
        for _ in range(self.substeps):
            acc = np.clip(self.kv * (v_cmd - self.vel), -A_MAX, A_MAX)
            acc[:2] -= self.drag * (self.vel[:2] - wind)
            self.vel += acc * h
            self.pos += self.vel * h
            if self.pos[2] <= 0.0:
                break
        self.steps += 1

        reward, terminated, truncated, info = task.step_outcome(
            prev_pos, self.pos, self.vel, action, self.prev_action, self.steps)
        self.prev_action = action
        if terminated:
            self.pos[2] = max(self.pos[2], 0.0)
        obs = self.effects.observe(self.pos, self.vel, self.prev_action)
        return obs, reward, terminated, truncated, info
