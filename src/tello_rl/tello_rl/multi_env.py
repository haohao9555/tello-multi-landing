"""四机降落的快速替身环境（4 架一起起飞、逐架降落，规则见 multi_task.py）。

动力学与单机快速环境相同（fast_env.py），4 架飞机吹同一阵风，各自有独立的
指令延迟、传感器噪声、定位丢帧和动力学参数。
观测 52 维：每架 12 维 × 4 + 4 个平台是否已占用；动作 12 维：每架 [vx, vy, vz] ∈ [-1, 1]。
preflight=True 时（评估用）起飞前先测风，超过阈值整组不起飞，回合直接以 no_go 结束。
"""
import gymnasium as gym
import numpy as np

from tello_rl import multi_task as mt
from tello_rl import task
from tello_rl.fast_env import A_MAX, DRAG, KV

N = mt.N_DRONES


class FastMultiTelloEnv(gym.Env):
    def __init__(self, wind_level=None, preflight=False, substeps=10):
        self.wind_level = wind_level
        self.preflight = preflight
        self.substeps = substeps
        self.zone_weight = 1.0   # 降落区违规扣分的权重（课程学习时由外部从 0 逐步调到 1）
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (mt.OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (mt.ACT_DIM,), np.float32)
        self.v_max = np.array([task.V_MAX_XY, task.V_MAX_XY, task.V_MAX_Z])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.pos = mt.sample_starts(rng)
        self.vel = np.zeros((N, 3))
        self.prev_action = np.zeros((N, 3))
        self.wind = task.WindModel(rng, self.wind_level)
        self.go, self.measured_wind = mt.preflight_go(self.wind, rng) if self.preflight else (True, None)
        self.effects = [task.RealismEffects(rng) for _ in range(N)]
        if task.RANDOMIZE_DYNAMICS:
            self.kv = rng.uniform(*task.KV_RANGE, size=N)
            self.drag = rng.uniform(*task.DRAG_RANGE, size=N)
            self.cmd_gain = rng.uniform(*task.CMD_GAIN_RANGE, size=N)
        else:
            self.kv, self.drag, self.cmd_gain = np.full(N, KV), np.full(N, DRAG), np.ones(N)
        self.active = np.ones(N, dtype=bool)
        self.occupied = np.zeros(N, dtype=bool)
        self.outcomes = [None] * N
        self.landing_errors = [None] * N
        self.touchdown_speeds = [None] * N
        self.min_sep = np.inf
        self.touchdown_times = [None] * N
        self.zone_violation_steps = 0
        self.cleared = None   # 获准降落的飞机编号
        self.steps = 0
        info = {'wind': self.wind.mean, 'go': self.go, 'measured_wind': self.measured_wind}
        return self._observe(), info

    def _observe(self):
        parts = []
        for i in range(N):
            if self.active[i]:
                o = self.effects[i].observe(self.pos[i], self.vel[i], self.prev_action[i])
                o = np.insert(o, 7, [1.0, float(mt.in_zone(self.pos[i])[0])])   # [pos 3, vel 3, valid, active, in_zone, prev 3]
            else:
                o = np.concatenate([self.pos[i], np.zeros(3), [1.0, 0.0, 0.0], np.zeros(3)])
            parts.append(o)
        parts.append(self.occupied.astype(np.float64))
        return np.concatenate(parts).astype(np.float32)

    def _finish(self, reason):
        for i in range(N):
            if self.outcomes[i] is None:
                self.outcomes[i] = reason
        n_success = sum(o == 'success' for o in self.outcomes)
        errors = [e for e in self.landing_errors if e is not None]
        times = sorted(t for t in self.touchdown_times if t is not None)
        return {'outcomes': list(self.outcomes), 'n_success': n_success,
                'all_landed': n_success == N,
                'all_success': n_success == N and self.zone_violation_steps == 0,
                'zone_violation_steps': self.zone_violation_steps,
                'touchdown_times': list(self.touchdown_times),
                'min_landing_gap': float(np.min(np.diff(times))) if len(times) > 1 else None,
                'min_sep': float(self.min_sep),
                'time': self.steps * task.CONTROL_DT, 'go': self.go,
                'landing_error': float(np.mean(errors)) if errors else None,
                'touchdown_speed': float(np.mean([s for s in self.touchdown_speeds if s is not None]))
                if errors else None}

    def step(self, action):
        if not self.go:   # 大风不起飞：回合直接结束
            return self._observe(), 0.0, True, False, self._finish('no_go')

        action = np.clip(np.asarray(action, dtype=np.float64).reshape(N, 3), -1.0, 1.0)
        was_active = self.active.copy()
        occupied_before = self.occupied.copy()
        cost_before = mt.assignment_cost(self.pos, was_active, occupied_before, self.cleared)

        v_cmd = np.zeros((N, 3))
        for i in np.flatnonzero(was_active):
            v_cmd[i] = self.effects[i].delay(action[i]) * self.v_max * self.cmd_gain[i]
        wind = self.wind.step(task.CONTROL_DT)
        h = task.CONTROL_DT / self.substeps
        flying = was_active.copy()
        for _ in range(self.substeps):
            acc = np.clip(self.kv[:, None] * (v_cmd - self.vel), -A_MAX, A_MAX)
            acc[:, :2] -= self.drag[:, None] * (self.vel[:, :2] - wind)
            self.vel[flying] += acc[flying] * h
            self.pos[flying] += self.vel[flying] * h
            flying &= self.pos[:, 2] > 0.0
        self.steps += 1

        reward = -task.TIME_PENALTY * was_active.sum()
        reward -= task.SMOOTH_PENALTY * float(np.sum((action - self.prev_action)[was_active] ** 2))

        # 触地
        for i in np.flatnonzero(was_active & (self.pos[:, 2] <= 0.0)):
            outcome, pad, r, err = mt.check_touchdown(self.pos[i, :2], self.vel[i, 2], self.occupied)
            reward += r
            self.outcomes[i], self.landing_errors[i] = outcome, err
            self.touchdown_times[i] = self.steps * task.CONTROL_DT
            self.touchdown_speeds[i] = abs(float(self.vel[i, 2]))
            if pad is not None:
                self.occupied[pad] = True
            self.active[i] = False
            self.pos[i, 2], self.vel[i] = 0.0, 0.0

        # 出界
        for i in np.flatnonzero(self.active):
            if np.abs(self.pos[i, :2]).max() > mt.BOUNDS_XY or self.pos[i, 2] > mt.BOUNDS_Z:
                reward -= task.FAIL_PENALTY
                self.outcomes[i], self.active[i] = 'out_of_bounds', False

        # 相撞（在飞的飞机之间，以及在飞的飞机撞到已落地的飞机）与安全距离
        for i in range(N):
            for j in range(i + 1, N):
                if not (was_active[i] or was_active[j]):
                    continue
                d = float(np.linalg.norm(self.pos[i] - self.pos[j]))
                if was_active[i] and was_active[j]:
                    self.min_sep = min(self.min_sep, d)
                    if d < mt.SAFE_DIST:
                        reward -= mt.PROXIMITY_PENALTY * min(
                            (mt.SAFE_DIST - d) / (mt.SAFE_DIST - mt.COLLISION_DIST), 1.0)
                if d < mt.COLLISION_DIST:
                    for k in (i, j):
                        if self.active[k]:
                            reward -= task.FAIL_PENALTY
                            self.outcomes[k], self.active[k] = 'collision', False

        # 降落区一次只允许一架：这一步在飞（含刚落地）的飞机里，区内多出来的每架扣分
        in_zone_count = int(np.sum(was_active & mt.in_zone(self.pos)))
        if in_zone_count > 1:
            reward -= mt.ZONE_PENALTY * self.zone_weight * (in_zone_count - 1)
            self.zone_violation_steps += 1

        # 引导：上一步在飞的飞机，按最优分配算总距离的减少量（坠毁的飞机不会因此"得分"）
        self.cleared = mt.update_clearance(self.cleared, self.pos, was_active)
        cost_after = mt.assignment_cost(self.pos, was_active, occupied_before, self.cleared)
        reward += task.SHAPING * (cost_before - cost_after)

        self.prev_action[was_active] = action[was_active]
        terminated = not self.active.any()
        truncated = not terminated and self.steps >= mt.MAX_STEPS
        info = self._finish('timeout') if (terminated or truncated) else {}
        return self._observe(), reward, terminated, truncated, info
