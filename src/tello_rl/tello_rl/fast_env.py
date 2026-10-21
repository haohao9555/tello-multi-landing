"""快速替身环境：用 Step 1 在 Gazebo 里实测的闭环响应近似 Tello，numpy 实现，每秒几千步。

用途：调试算法和奖励；Gazebo 训练太慢时的备用方案（训练在这里，评估回 Gazebo）。

动力学（每个控制周期内用 10 个子步积分）：
  水平：v' = clip(Kv·(v_cmd − v), ±A_MAX) − D·(v − w)
  竖直：v' = clip(Kv·(v_cmd − v), ±A_MAX)
  Kv = 2.5、D = 0.41 来自 Step 1 实测：指令 0.5 m/s 实际 0.43；3 m/s 侧风漂移 0.45 m/s
风由 task.WindModel 生成（平均风 + 湍流 + 阵风 + 风向漂移），每个控制步更新一次
"""
import gymnasium as gym
import numpy as np

from tello_rl import task

KV = 2.5        # 标称速度环增益 1/s（velocityGain / 质量）；随机化范围见 task.py
DRAG = 0.41     # 旋翼阻力 + 机身风阻，1/s
A_MAX = 3.0     # maximumLinearAcceleration m/s^2


class FastTelloLandEnv(gym.Env):
    def __init__(self, wind_level=None, substeps=10):
        self.wind_level = wind_level     # None：每回合随机风速；否则固定风速（评估用）
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
        self.wind = task.WindModel(rng, self.wind_level)
        self.effects = task.RealismEffects(rng)
        if task.RANDOMIZE_DYNAMICS:   # 每回合换一架"参数不太一样的 Tello"
            self.kv = rng.uniform(*task.KV_RANGE)
            self.drag = rng.uniform(*task.DRAG_RANGE)
            self.cmd_gain = rng.uniform(*task.CMD_GAIN_RANGE)
        else:
            self.kv, self.drag, self.cmd_gain = KV, DRAG, 1.0
        self.steps = 0
        return self.effects.observe(self.pos, self.vel, self.prev_action), {'wind': self.wind.mean}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v_cmd = self.effects.delay(action) * self.v_max * self.cmd_gain   # 延迟后的指令
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
