"""Agent-coordinate observation wrapper and world-coordinate action conversion."""
from dataclasses import dataclass

import numpy as np

from tello_rl import multi_task as mt
from tello_rl.multi_env import FastMultiTelloEnv

N = mt.N_DRONES
OWN_DIM = mt.PER_DRONE_OBS          # [x, y, z, vx, vy, vz, valid, active, dx, dy, ax, ay, az]
OTHER_DIM = 7                       # [x, y, z, vx, vy, vz, active]
OBS_DIM = OWN_DIM + OTHER_DIM * (N - 1)
STATE_DIM = N * 7 + 2
N_ACTIONS = 3

TURNS = np.round((np.pi / 2 - mt.DIRECTIONS) / (np.pi / 2)).astype(int) % 4


def rotation(turns):
    c, s = np.cos(turns * np.pi / 2), np.sin(turns * np.pi / 2)
    return np.array([[c, -s], [s, c]])


ROTATIONS = [rotation(t) for t in TURNS]


def agent_observations(flat_obs):
    """Agent observations."""
    own_dim = flat_obs.size // N
    drones = flat_obs.reshape(N, own_dim)
    out = np.zeros((N, own_dim + OTHER_DIM * (N - 1)), dtype=np.float32)
    for i in range(N):
        R = ROTATIONS[i]
        own = drones[i].copy()
        own[0:2], own[3:5], own[8:10] = R @ own[0:2], R @ own[3:5], R @ own[8:10]
        for k in range(10, own_dim, 3):
            own[k:k + 2] = R @ own[k:k + 2]
        parts = [own]
        for k in range(1, N):
            other = drones[(i + k) % N]
            parts.append(np.concatenate([R @ other[0:2], other[2:3], R @ other[3:5], other[5:6], other[7:8]]))
        out[i] = np.concatenate(parts)
    return out


def global_state(env):
    per_drone = np.concatenate([env.pos, env.vel, env.active[:, None].astype(np.float64)], axis=1)
    return np.concatenate([per_drone.ravel(), env.wind.current]).astype(np.float32)


def to_world_actions(actions):
    """To world actions."""
    a = 2.0 * np.asarray(actions, dtype=np.float64) - 1.0
    world = a.copy()
    for i in range(N):
        world[i, :2] = ROTATIONS[i].T @ a[i, :2]
    return world.ravel()


@dataclass(frozen=True)
class EnvSpec:
    n_agents: int
    obs_dim: int
    state_dim: int
    n_actions: int
    episode_limit: int
    observation_mask: np.ndarray
    action_mask: np.ndarray


class VectorTelloMARL:
    """Vectortellomarl."""

    def __init__(self, num_envs, seed, wind_level=None, preflight=False, individual_reward=False, v_max_xy=None,
                 profile='tello'):
        """Init."""
        self.individual_reward = individual_reward
        self.batch_size = int(num_envs)
        self.base_seed = int(seed)
        self.envs = [FastMultiTelloEnv(wind_level=wind_level, preflight=preflight, v_max_xy=v_max_xy, profile=profile)
                     for _ in range(num_envs)]
        p = mt.get_profile(profile)
        obs_dim = p.per_drone_obs + OTHER_DIM * (N - 1)
        self.spec = EnvSpec(N, obs_dim, STATE_DIM, N_ACTIONS, p.max_steps,
                            np.ones((N, obs_dim), dtype=np.float32), np.ones((N, N_ACTIONS), dtype=np.float32))
        self.done = [False] * self.batch_size
        self.last = [None] * self.batch_size
        self.infos = [None] * self.batch_size

    def reset(self, seed=None):
        seed = self.base_seed if seed is None and not hasattr(self, '_has_reset') else seed
        self._has_reset = True
        observations, states = [], []
        for k, env in enumerate(self.envs):
            obs, _ = env.reset(seed=None if seed is None else int(seed) + k)
            self.last[k] = (agent_observations(obs), global_state(env))
            self.done[k], self.infos[k] = False, None
            observations.append(self.last[k][0])
            states.append(self.last[k][1])
        return np.stack(observations), np.stack(states)

    def step(self, actions):
        observations, states, rewards, terminated, truncated = [], [], [], [], []
        for k, env in enumerate(self.envs):
            if self.done[k]:
                obs_k, state_k = self.last[k]
                r, te, tr = (np.zeros(N) if self.individual_reward else 0.0), False, True
            else:
                obs, r, te, tr, info = env.step(to_world_actions(actions[k]))
                if self.individual_reward:
                    r = env.drone_rewards.copy()
                obs_k, state_k = agent_observations(obs), global_state(env)
                self.last[k] = (obs_k, state_k)
                if te or tr:
                    self.done[k], self.infos[k] = True, info
            observations.append(obs_k)
            states.append(state_k)
            rewards.append(r)
            terminated.append(te)
            truncated.append(tr)
        return (np.stack(observations), np.stack(states),
                np.asarray(rewards, dtype=np.float32).reshape(self.batch_size, -1),
                np.asarray(terminated, dtype=np.bool_).reshape(-1, 1),
                np.asarray(truncated, dtype=np.bool_).reshape(-1, 1))

    def close(self, force=False):
        pass
