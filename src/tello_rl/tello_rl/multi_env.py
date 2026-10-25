"""Fast four-aircraft landing environment with shared wind and independent observation effects."""
import gymnasium as gym
import numpy as np

from tello_rl import multi_task as mt
from tello_rl import task

N = mt.N_DRONES


def drone_observation(i, effects, pos, vel, prev_action, active, targets=mt.TARGETS):
    """Drone observation."""
    history = np.atleast_2d(prev_action)
    if active:
        o = effects.observe(pos, vel, history[0])
        return np.concatenate([np.insert(o, 7, [1.0, *(targets[i] - o[0:2])]), history[1:].ravel()])
    return np.concatenate([pos, np.zeros(3), [1.0, 0.0], targets[i] - pos[:2], np.zeros(history.size)])


class FastMultiTelloEnv(gym.Env):
    def __init__(self, wind_level=None, preflight=False, substeps=10, constant_wind=False, v_max_xy=None,
                 profile='tello'):
        """Init."""
        self.profile = mt.get_profile(profile)
        self.targets = self.profile.targets
        self.wind_level = wind_level
        self.constant_wind = constant_wind
        self.preflight = preflight
        self.substeps = substeps
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N * self.profile.per_drone_obs,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (mt.ACT_DIM,), np.float32)
        self.v_max_xy = self.profile.v_max_xy if v_max_xy is None else v_max_xy
        self.v_max = np.array([self.v_max_xy, self.v_max_xy, self.profile.v_max_z])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng, p = self.np_random, self.profile
        self.pos = mt.sample_starts(rng, p)
        self.vel = np.zeros((N, 3))
        self.history = np.zeros((N, p.action_history, 3))
        self.wind = task.WindModel(rng, self.wind_level, self.constant_wind)
        self.go, self.measured_wind = mt.preflight_go(self.wind, rng, p) if self.preflight else (True, None)
        self.effects = [task.RealismEffects(rng) for _ in range(N)]
        if task.RANDOMIZE_DYNAMICS:
            self.kv = rng.uniform(*p.kv_range, size=N)
            self.drag = rng.uniform(*p.drag_range, size=N)
            self.cmd_gain = rng.uniform(*task.CMD_GAIN_RANGE, size=N)
            self.drag2 = rng.uniform(*p.drag2_range, size=N) if p.drag2_range[1] > 0 else np.zeros(N)
        else:
            self.kv, self.drag, self.cmd_gain = np.full(N, p.kv), np.full(N, p.drag), np.ones(N)
            self.drag2 = np.full(N, p.drag2)
        self.active = np.ones(N, dtype=bool)
        self.outcomes = [None] * N
        self.landing_errors = [None] * N
        self.touchdown_speeds = [None] * N
        self.touchdown_times = [None] * N
        self.min_sep = np.inf
        self.drone_rewards = np.zeros(N)
        self.steps = 0
        info = {'wind': self.wind.mean, 'go': self.go, 'measured_wind': self.measured_wind}
        return self._observe(), info

    def _observe(self):
        return np.concatenate([
            drone_observation(i, self.effects[i], self.pos[i], self.vel[i], self.history[i], self.active[i],
                              self.targets)
            for i in range(N)]).astype(np.float32)

    def _finish(self, reason):
        for i in range(N):
            if self.outcomes[i] is None:
                self.outcomes[i] = reason
        n_success = sum(o == 'success' for o in self.outcomes)
        errors = [e for e in self.landing_errors if e is not None]
        return {'outcomes': list(self.outcomes), 'n_success': n_success,
                'all_success': n_success == N,
                'n_collisions': sum(o == 'collision' for o in self.outcomes),
                'touchdown_times': list(self.touchdown_times),
                'min_sep': float(self.min_sep),
                'time': self.steps * task.CONTROL_DT, 'go': self.go,
                'landing_error': float(np.mean(errors)) if errors else None,
                'touchdown_speed': float(np.mean([s for s in self.touchdown_speeds if s is not None]))
                if errors else None}

    def step(self, action):
        if not self.go:
            self.drone_rewards = np.zeros(N)
            return self._observe(), 0.0, True, False, self._finish('no_go')

        p = self.profile
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(N, 3), -1.0, 1.0)
        was_active = self.active.copy()
        cost_before = mt.target_costs(self.pos, p)

        v_cmd = np.zeros((N, 3))
        for i in np.flatnonzero(was_active):
            v_cmd[i] = self.effects[i].delay(action[i]) * self.v_max * self.cmd_gain[i]
        wind = self.wind.step(task.CONTROL_DT)
        h = task.CONTROL_DT / self.substeps
        flying = was_active.copy()
        for _ in range(self.substeps):
            acc = np.clip(self.kv[:, None] * (v_cmd - self.vel), -p.a_max, p.a_max)
            v_rel = self.vel[:, :2] - wind
            acc[:, :2] -= (self.drag + self.drag2 * np.linalg.norm(v_rel, axis=1))[:, None] * v_rel
            self.vel[flying] += acc[flying] * h
            self.pos[flying] += self.vel[flying] * h
            flying &= self.pos[:, 2] > 0.0
        self.steps += 1

        rewards = np.zeros(N)
        rewards[was_active] -= task.TIME_PENALTY
        rewards[was_active] -= task.SMOOTH_PENALTY * np.sum((action - self.history[:, 0])[was_active] ** 2, axis=1)


        for i in np.flatnonzero(was_active & (self.pos[:, 2] <= 0.0)):
            outcome, r, err = mt.check_touchdown(i, self.pos[i, :2], self.vel[i, 2], p)
            rewards[i] += r
            self.outcomes[i], self.landing_errors[i] = outcome, err
            self.touchdown_times[i] = self.steps * task.CONTROL_DT
            self.touchdown_speeds[i] = abs(float(self.vel[i, 2]))
            self.active[i] = False
            self.pos[i, 2], self.vel[i] = 0.0, 0.0


        for i in np.flatnonzero(self.active):
            if np.abs(self.pos[i, :2]).max() > p.bounds_xy or self.pos[i, 2] > p.bounds_z:
                rewards[i] -= task.FAIL_PENALTY
                self.outcomes[i], self.active[i] = 'out_of_bounds', False


        for i in range(N):
            for j in range(i + 1, N):
                if not (was_active[i] or was_active[j]):
                    continue
                d = float(np.linalg.norm(self.pos[i] - self.pos[j]))
                if was_active[i] and was_active[j]:
                    self.min_sep = min(self.min_sep, d)
                    if d < p.safe_dist:
                        penalty = mt.PROXIMITY_PENALTY * min(
                            (p.safe_dist - d) / (p.safe_dist - p.collision_dist), 1.0)
                        rewards[[i, j]] -= penalty / 2
                if d < p.collision_dist:
                    for k in (i, j):
                        if self.active[k]:
                            rewards[k] -= task.FAIL_PENALTY
                            self.outcomes[k], self.active[k] = 'collision', False


        cost_after = mt.target_costs(self.pos, p)
        rewards[was_active] += task.SHAPING * (cost_before - cost_after)[was_active]
        self.drone_rewards = rewards

        self.history[was_active] = np.concatenate(
            [action[was_active, None], self.history[was_active, :-1]], axis=1)
        terminated = not self.active.any()
        truncated = not terminated and self.steps >= p.max_steps
        info = self._finish('timeout') if (terminated or truncated) else {}
        return self._observe(), float(rewards.sum()), terminated, truncated, info
