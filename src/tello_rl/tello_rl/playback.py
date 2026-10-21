"""回放接口（给 GUI 用）：加载训练好的策略，逐步返回原始画面和结构化状态。

  from tello_rl.playback import load_policy, GazeboPlayer, FastPlayer

  policy, algo = load_policy('models/mappo_seq.pt')
  with GazeboPlayer(width=1280, height=720) as player:     # 需要 Gazebo Harmonic + GPU（无界面渲染）
      for frame in player.play(policy, wind_level=2.0, seed=0):
          show(frame.image)                                # HxWx3 uint8
          print(frame.t, frame.status, frame.zone_violations)
          if frame.done: ...                               # 回合结束后还会再给 hold_frames 帧，画面停住展示结果

  FastPlayer().play(...) 接口完全相同，但不需要 Gazebo，frame.image 为 None（适合没有 GPU 时先开发界面）。

play() 是阻塞的生成器：GazeboPlayer 每帧要等仿真走完 0.1 s 并渲染完，GUI 里请放在后台线程里迭代。
规则与训练环境（multi_env.py）一致：降落区一次只允许一架在飞的飞机；相撞的飞机关电机掉下来；
平均风速超过 multi_task.GO_THRESHOLD（起飞前测风）时整组不起飞，status 全为 'no_go'。
"""
from dataclasses import dataclass, field

import numpy as np

from tello_rl import multi_task as mt
from tello_rl import task
from tello_rl.evaluate_marl import load_policy  # noqa: F401  （GUI 从这里统一导入）
from tello_rl.multi_env import FastMultiTelloEnv

N = mt.N_DRONES
NAMES = ['N', 'E', 'S', 'W']                       # 飞机编号 0~3 分别从北、东、南、西飞来
TOUCH_Z = 0.005
V_MAX = np.array([task.V_MAX_XY, task.V_MAX_XY, task.V_MAX_Z])


@dataclass
class Frame:
    t: float                      # 仿真时间 s（每帧 0.1 s）
    image: object                 # Gazebo 画面 HxWx3 uint8；FastPlayer 为 None
    positions: np.ndarray         # (4, 3) 相对降落站中心 m，z = 机身底部离平台面高度
    velocities: np.ndarray        # (4, 3) 世界系 m/s
    wind: np.ndarray              # (2,) 当前风速（世界系）m/s
    wind_mean: float              # 本回合平均风速 m/s
    status: list                  # 每架：'flying' / 'success' / 'missed' / 'hard_landing' / 'occupied'
                                  #       / 'collision' / 'out_of_bounds' / 'timeout' / 'no_go'
    landing_errors: list          # 每架落点离平台中心的距离 m（没落地为 None）
    pads_occupied: list           # 4 块平台是否已被占用（平台坐标见 multi_task.PADS）
    in_zone: list                 # 每架是否正在降落区内（只算在飞的）
    zone_violations: int          # 到目前为止降落区违规的步数
    done: bool = False            # 回合已结束（之后的帧是停住展示结果）
    success: object = None        # 结束后：4 架全部成功且零违规为 True；不起飞为 None
    extra: dict = field(default_factory=dict)


class _Episode:
    """一回合的判定逻辑（触地、出界、相撞、降落区），Gazebo 和快速环境共用。"""

    def __init__(self):
        self.active = np.ones(N, dtype=bool)
        self.occupied = np.zeros(N, dtype=bool)
        self.outcomes, self.errors = [None] * N, [None] * N
        self.prev_action = np.zeros((N, 3))
        self.violations = 0

    def observe(self, rel, vel, effects):
        """和 FastMultiTelloEnv._observe 相同的观测。"""
        parts = []
        for i in range(N):
            if self.active[i]:
                o = np.insert(effects[i].observe(rel[i], vel[i], self.prev_action[i]), 7,
                              [1.0, float(mt.in_zone(rel[i])[0])])
            else:
                o = np.concatenate([rel[i], np.zeros(3), [1.0, 0.0, 0.0], np.zeros(3)])
            parts.append(o)
        parts.append(self.occupied.astype(np.float64))
        return np.concatenate(parts).astype(np.float32)

    def update(self, rel, vel, prev_vz, was_active, action, disable):
        for i in np.flatnonzero(self.active & (rel[:, 2] <= TOUCH_Z)):   # 触地（接触后竖直速度用上一步的）
            vz = prev_vz[i] if abs(prev_vz[i]) > abs(vel[i, 2]) else vel[i, 2]
            outcome, pad, _, err = mt.check_touchdown(rel[i, :2], vz, self.occupied)
            self.outcomes[i], self.errors[i], self.active[i] = outcome, err, False
            if pad is not None:
                self.occupied[pad] = True
            disable(i)
        for i in np.flatnonzero(self.active):
            if np.abs(rel[i, :2]).max() > mt.BOUNDS_XY or rel[i, 2] > mt.BOUNDS_Z:
                self.outcomes[i], self.active[i] = 'out_of_bounds', False
                disable(i)
        for i in range(N):
            for j in range(i + 1, N):
                if (was_active[i] or was_active[j]) and np.linalg.norm(rel[i] - rel[j]) < mt.COLLISION_DIST:
                    for k in (i, j):
                        if self.active[k]:
                            self.outcomes[k], self.active[k] = 'collision', False
                            disable(k)
        if np.sum(was_active & mt.in_zone(rel)) > 1:   # 降落区一次只允许一架
            self.violations += 1
        self.prev_action[was_active] = action[was_active]

    def frame(self, t, image, rel, vel, wind, wind_mean, done=False):
        status = [o or ('flying' if not done else 'timeout') for o in self.outcomes]
        success = (all(o == 'success' for o in status) and self.violations == 0) if done else None
        return Frame(t=t, image=image, positions=rel.copy(), velocities=vel.copy(), wind=np.array(wind),
                     wind_mean=wind_mean, status=status, landing_errors=list(self.errors),
                     pads_occupied=self.occupied.tolist(), in_zone=list(self.active & mt.in_zone(rel)),
                     zone_violations=self.violations, done=done, success=success)


def _scenario(wind_level, seed):
    """和评估 / 训练环境同一个种子 -> 同样的起点、风（湍流、阵风）和起飞前测风结果。"""
    scenario = FastMultiTelloEnv(wind_level=wind_level, preflight=True)
    _, info = scenario.reset(seed=seed)
    return scenario, info['go']


class FollowCamera:
    """自动取景：从南边 45° 俯视，对准还在飞的飞机和平台；飞机越往中间收，镜头越近。"""
    ELEVATION = np.deg2rad(45)

    def __init__(self):
        self.target, self.dist = None, None

    def update(self, sim, rel, active, smooth=0.2):
        points = np.vstack([rel[active], np.c_[mt.PADS, np.zeros(N)]])
        target = (points.min(0) + points.max(0)) / 2
        spread = np.max(np.linalg.norm((points - target)[:, :2], axis=1))
        dist = max(1.8, 2.0 * spread + 0.8)
        if self.target is None:
            self.target, self.dist = target, dist
        self.target = self.target + smooth * (target - self.target)
        self.dist += smooth * (dist - self.dist)
        back = np.array([0.0, -np.cos(self.ELEVATION), np.sin(self.ELEVATION)])
        sim.set_camera(self.target + self.dist * back, np.pi / 2, self.ELEVATION)


class GazeboPlayer:
    """在 Gazebo 四机场景里回放。一个实例对应一个 Gazebo 服务器，可连续播放多个回合。"""

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

    def play(self, policy, wind_level, seed=0, hold_frames=20):
        sim = self.sim
        scenario, go = _scenario(wind_level, seed)
        starts, wind_model = scenario.pos.copy(), scenario.wind
        effects = [task.RealismEffects(np.random.default_rng(seed * 10 + i)) for i in range(N)]
        episode = _Episode()
        camera = FollowCamera() if self.follow_camera else None

        for i in range(N):
            sim.enable(i, False)
        sim.set_wind(wind_model.current)
        if not go:   # 大风不起飞：飞机停在起点地面上
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
                camera.update(sim, rel, episode.active)
            v_cmd = np.zeros((N, 3))
            for i in np.flatnonzero(episode.active):
                v_cmd[i] = effects[i].delay(action[i]) * V_MAX
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
                camera.update(sim, rel, episode.active)
            sim.step(np.zeros((N, 3)))
            rel, vel, image = sim.state()
            yield episode.frame(step * task.CONTROL_DT, image, rel, vel, wind_model.current,
                                wind_model.speed, done=True)


class FastPlayer:
    """不需要 Gazebo 的回放（快速环境），接口与 GazeboPlayer 相同，frame.image 为 None。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def close(self):
        pass

    def play(self, policy, wind_level, seed=0, hold_frames=20):
        env = FastMultiTelloEnv(wind_level=wind_level, preflight=True)
        obs, info = env.reset(seed=seed)
        episode = _Episode()
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
            episode.errors, episode.occupied, episode.active = list(env.landing_errors), env.occupied, env.active
            episode.violations = env.zone_violation_steps
            yield episode.frame(env.steps * task.CONTROL_DT, None, env.pos, env.vel, env.wind.current,
                                env.wind.speed, done=done)
        for _ in range(hold_frames - 1):
            yield episode.frame(env.steps * task.CONTROL_DT, None, env.pos, env.vel, env.wind.current,
                                env.wind.speed, done=True)
