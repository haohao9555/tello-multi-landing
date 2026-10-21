"""把四机降落环境包装成 vendor/mamujoco_v2 的多智能体接口，直接复用那里的 MAPPO / HAPPO / MAT。

每架飞机的观测在"自己的坐标系"里：整个场景旋转 90° 的整数倍，让这架飞机总是从北边飞来。
降落站是 4 重旋转对称的，4 架飞机面对的是同一个问题，MAPPO 共享参数时一个控制器 4 架共用。
  观测 40 维：自己 12 维 + 另外 3 架各 8 维（按顺时针相对次序）+ 4 个平台是否占用（同样旋转）
  全局状态 38 维：真实位置、速度、是否在飞、是否在降落区、平台占用、当前风速（只给 critic，集中训练的特权信息）
  动作：actor 输出 [0, 1] -> [-1, 1]，再从自己的坐标系转回世界坐标系
"""
from dataclasses import dataclass

import numpy as np

from tello_rl import multi_task as mt
from tello_rl.multi_env import FastMultiTelloEnv

N = mt.N_DRONES
OWN_DIM = mt.PER_DRONE_OBS          # [x, y, z, vx, vy, vz, valid, active, in_zone, ax, ay, az]
OTHER_DIM = 8                       # [x, y, z, vx, vy, vz, active, in_zone]
OBS_DIM = OWN_DIM + OTHER_DIM * (N - 1) + N
STATE_DIM = N * 8 + N + 2
N_ACTIONS = 3
# 每架飞机逆时针转几个 90° 后变成"从北边飞来"：北 0、东 1、南 2、西 3
TURNS = np.round((np.pi / 2 - mt.DIRECTIONS) / (np.pi / 2)).astype(int) % 4


def rotation(turns):
    c, s = np.cos(turns * np.pi / 2), np.sin(turns * np.pi / 2)
    return np.array([[c, -s], [s, c]])


ROTATIONS = [rotation(t) for t in TURNS]


def agent_observations(flat_obs):
    """环境的 48 维世界系观测 -> 每架飞机自己坐标系下的 36 维观测，形状 (N, OBS_DIM)。"""
    drones = flat_obs[:N * OWN_DIM].reshape(N, OWN_DIM)
    occupied = flat_obs[N * OWN_DIM:]
    out = np.zeros((N, OBS_DIM), dtype=np.float32)
    for i in range(N):
        R = ROTATIONS[i]
        own = drones[i].copy()
        own[0:2], own[3:5], own[9:11] = R @ own[0:2], R @ own[3:5], R @ own[9:11]
        parts = [own]
        for k in range(1, N):
            other = drones[(i + k) % N]
            parts.append(np.concatenate([R @ other[0:2], other[2:3], R @ other[3:5], other[5:6], other[7:9]]))
        parts.append(np.roll(occupied, TURNS[i]))   # 平台 k 转 t 次后变成平台 (k + t) mod 4
        out[i] = np.concatenate(parts)
    return out


def global_state(env):
    zone = (env.active & mt.in_zone(env.pos))[:, None].astype(np.float64)
    per_drone = np.concatenate([env.pos, env.vel, env.active[:, None].astype(np.float64), zone], axis=1)
    return np.concatenate([per_drone.ravel(), env.occupied.astype(np.float64), env.wind.current]).astype(np.float32)


def to_world_actions(actions):
    """(N, 3) 的 [0, 1] 动作（各自坐标系）-> 12 维世界系动作 [-1, 1]。"""
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
    """和 mamujoco_v2 的 VectorMaMuJoCo 同样的接口；环境很轻，直接在本进程里串行跑。"""

    def __init__(self, num_envs, seed, wind_level=None, preflight=False, zone_curriculum=None):
        """zone_curriculum=(start, end)：累计环境步数到 start 之前不扣降落区违规分，之后线性加到 end 时满额。"""
        self.zone_curriculum = zone_curriculum
        self.env_steps = 0
        self.batch_size = int(num_envs)
        self.base_seed = int(seed)
        self.envs = [FastMultiTelloEnv(wind_level=wind_level, preflight=preflight) for _ in range(num_envs)]
        self.spec = EnvSpec(N, OBS_DIM, STATE_DIM, N_ACTIONS, mt.MAX_STEPS,
                            np.ones((N, OBS_DIM), dtype=np.float32), np.ones((N, N_ACTIONS), dtype=np.float32))
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

    def zone_weight(self):
        if self.zone_curriculum is None:
            return 1.0
        start, end = self.zone_curriculum
        return float(np.clip((self.env_steps - start) / max(end - start, 1), 0.0, 1.0))

    def step(self, actions):
        weight = self.zone_weight()
        observations, states, rewards, terminated, truncated = [], [], [], [], []
        for k, env in enumerate(self.envs):
            if self.done[k]:   # 已结束的环境：和 mamujoco_v2 一样返回 0 奖励 + truncated
                obs_k, state_k = self.last[k]
                r, te, tr = 0.0, False, True
            else:
                env.zone_weight = weight
                self.env_steps += 1
                obs, r, te, tr, info = env.step(to_world_actions(actions[k]))
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
                np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
                np.asarray(terminated, dtype=np.bool_).reshape(-1, 1),
                np.asarray(truncated, dtype=np.bool_).reshape(-1, 1))

    def close(self, force=False):
        pass
