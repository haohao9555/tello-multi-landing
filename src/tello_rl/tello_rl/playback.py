"""Playback."""
from dataclasses import dataclass, field

import numpy as np

from tello_rl import multi_task as mt
from tello_rl import task
from tello_rl.evaluate_marl import load_policy
from tello_rl.multi_env import FastMultiTelloEnv, drone_observation

N = mt.N_DRONES
NAMES = ['N', 'E', 'S', 'W']
TOUCH_Z = 0.005


@dataclass
class Frame:
    t: float
    image: object
    positions: np.ndarray
    velocities: np.ndarray
    wind: np.ndarray
    wind_mean: float
    status: list
                                  #       / 'collision' / 'out_of_bounds' / 'timeout' / 'no_go'
    landing_errors: list
    targets: np.ndarray
    done: bool = False
    success: object = None
    extra: dict = field(default_factory=dict)
    profile: mt.Profile = mt.TELLO


class _Episode:
    """Episode."""

    def __init__(self, profile=mt.TELLO):
        self.profile = profile
        self.active = np.ones(N, dtype=bool)
        self.outcomes, self.errors = [None] * N, [None] * N
        self.prev_action = np.zeros((N, 3))

    def observe(self, rel, vel, effects):
        """Observe."""
        return np.concatenate([drone_observation(i, effects[i], rel[i], vel[i], self.prev_action[i], self.active[i],
                                                 self.profile.targets)
                               for i in range(N)]).astype(np.float32)

    def update(self, rel, vel, prev_vz, was_active, action, disable):
        p = self.profile
        for i in np.flatnonzero(self.active & (rel[:, 2] <= TOUCH_Z)):
            vz = prev_vz[i] if abs(prev_vz[i]) > abs(vel[i, 2]) else vel[i, 2]
            outcome, _, err = mt.check_touchdown(i, rel[i, :2], vz, p)
            self.outcomes[i], self.errors[i], self.active[i] = outcome, err, False
            disable(i)
        for i in np.flatnonzero(self.active):
            if np.abs(rel[i, :2]).max() > p.bounds_xy or rel[i, 2] > p.bounds_z:
                self.outcomes[i], self.active[i] = 'out_of_bounds', False
                disable(i)
        for i in range(N):
            for j in range(i + 1, N):
                if (was_active[i] or was_active[j]) and np.linalg.norm(rel[i] - rel[j]) < p.collision_dist:
                    for k in (i, j):
                        if self.active[k]:
                            self.outcomes[k], self.active[k] = 'collision', False
                            disable(k)
        self.prev_action[was_active] = action[was_active]

    def frame(self, t, image, rel, vel, wind, wind_mean, done=False):
        status = [o or ('flying' if not done else 'timeout') for o in self.outcomes]
        success = all(o == 'success' for o in status) if done else None
        return Frame(t=t, image=image, positions=rel.copy(), velocities=vel.copy(), wind=np.array(wind),
                     wind_mean=wind_mean, status=status, landing_errors=list(self.errors),
                     targets=self.profile.targets, done=done, success=success, profile=self.profile)


def _scenario(wind_level, seed):
    """Scenario."""
    scenario = FastMultiTelloEnv(wind_level=wind_level, preflight=True)
    _, info = scenario.reset(seed=seed)
    return scenario, info['go']


class FollowCamera:
    """Followcamera."""
    ELEVATION = np.deg2rad(45)

    def __init__(self):
        self.target, self.dist = None, None

    def update(self, sim, rel, active, smooth=0.2, focus=None):
        if focus is not None:
            # A true-scale close-up, with the front camera (+X) visible.
            target = rel[int(focus)].copy() + np.array([0., 0., 0.008])
            if self.target is None:
                self.target = target
            self.target += 0.55 * (target - self.target)
            eye = self.target + np.array([0.18, -0.21, 0.145])
            delta = self.target - eye
            sim.set_camera(eye, np.arctan2(delta[1], delta[0]),
                           np.arctan2(-delta[2], np.linalg.norm(delta[:2])))
            return
        points = np.vstack([rel[active], np.c_[mt.PADS, np.zeros(N)]])
        target = (points.min(0) + points.max(0)) / 2
        spread = np.max(np.linalg.norm((points - target)[:, :2], axis=1))
        dist = max(1.8, 2.0 * spread + 0.8)
        if self.target is None or self.dist is None:
            self.target, self.dist = target, dist
        self.target = self.target + smooth * (target - self.target)
        self.dist += smooth * (dist - self.dist)
        back = np.array([0.0, -np.cos(self.ELEVATION), np.sin(self.ELEVATION)])
        sim.set_camera(self.target + self.dist * back, np.pi / 2, self.ELEVATION)


class GazeboPlayer:
    """Gazeboplayer."""

    def __init__(self, width=1280, height=720, follow_camera=True):
        from tello_rl.gazebo_multi import GazeboMultiTello
        self.sim = GazeboMultiTello(camera=True, width=width, height=height)
        self.follow_camera = follow_camera

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.sim.close()

    def play(self, policy, wind_level, seed=0, hold_frames=20, delay_ms=None, camera_target=None):
        if mt.get_profile(getattr(policy, 'profile', None)).name != 'tello':
            raise ValueError('Gazebo currently supports Tello only; use FastPlayer for other profiles')
        sim = self.sim
        v_max_xy = getattr(policy, 'v_max_xy', None) or task.V_MAX_XY
        v_max = np.array([v_max_xy, v_max_xy, task.V_MAX_Z])
        scenario, go = _scenario(wind_level, seed)
        starts, wind_model = scenario.pos.copy(), scenario.wind
        effects = [task.RealismEffects(np.random.default_rng(seed * 10 + i)) for i in range(N)]
        if delay_ms is not None:
            from collections import deque
            delay_steps = max(0, int(round(delay_ms / (task.CONTROL_DT * 1000))))
            for effect in effects:
                effect.delay_steps = delay_steps
                effect.queue = deque(np.zeros(3) for _ in range(delay_steps))
        episode = _Episode()
        camera = FollowCamera() if self.follow_camera else None

        for i in range(N):
            sim.enable(i, False)
        sim.set_wind(wind_model.current)
        if not go:
            for i in range(N):
                sim.set_pose(i, np.append(starts[i, :2], 0.0))
            for _ in range(hold_frames):
                sim.step(np.zeros((N, 3)))
                rel, vel, image = sim.state()
                f = episode.frame(0.0, image, rel, vel, wind_model.current, wind_model.speed, done=True)
                f.status, f.success = ['no_go'] * N, None
                yield f
            return
        for i in range(N):
            sim.set_pose(i, starts[i])
            sim.enable(i, True)
        sim.step(np.zeros((N, 3)))
        rel, vel, _ = sim.state()

        step = 0
        for step in range(1, mt.MAX_STEPS + 1):
            action = policy(episode.observe(rel, vel, effects)).reshape(N, 3)
            if camera:
                camera.update(sim, rel, episode.active, focus=camera_target() if camera_target else None)
            v_cmd = np.zeros((N, 3))
            for i in np.flatnonzero(episode.active):
                v_cmd[i] = effects[i].delay(action[i]) * v_max
            wind = wind_model.step(task.CONTROL_DT)
            sim.set_wind(wind)
            prev_vz, was_active = vel[:, 2].copy(), episode.active.copy()
            sim.step(v_cmd)
            rel, vel, image = sim.state()
            episode.update(rel, vel, prev_vz, was_active, action, lambda i: sim.enable(i, False))
            yield episode.frame(step * task.CONTROL_DT, image, rel, vel, wind, wind_model.speed)
            if not episode.active.any():
                break
        for i in range(N):
            sim.enable(i, False)
        for _ in range(hold_frames):
            if camera:
                camera.update(sim, rel, episode.active, focus=camera_target() if camera_target else None)
            sim.step(np.zeros((N, 3)))
            rel, vel, image = sim.state()
            yield episode.frame(step * task.CONTROL_DT, image, rel, vel, wind_model.current,
                                wind_model.speed, done=True)


class FastPlayer:
    """Fastplayer."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def close(self):
        pass

    def play(self, policy, wind_level, seed=0, hold_frames=20):
        env = FastMultiTelloEnv(wind_level=wind_level, preflight=True, v_max_xy=getattr(policy, 'v_max_xy', None),
                                profile=getattr(policy, 'profile', 'tello'))
        obs, info = env.reset(seed=seed)
        episode = _Episode(env.profile)
        if not info['go']:
            for _ in range(hold_frames):
                rel = env.pos.copy()
                rel[:, 2] = 0.0
                f = episode.frame(0.0, None, rel, np.zeros((N, 3)), env.wind.current, env.wind.speed, done=True)
                f.status, f.success = ['no_go'] * N, None
                yield f
            return
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
            episode.outcomes = list(env.outcomes)
            episode.errors, episode.active = list(env.landing_errors), env.active
            yield episode.frame(env.steps * task.CONTROL_DT, None, env.pos, env.vel, env.wind.current,
                                env.wind.speed, done=done)
        for _ in range(hold_frames - 1):
            yield episode.frame(env.steps * task.CONTROL_DT, None, env.pos, env.vel, env.wind.current,
                                env.wind.speed, done=True)
