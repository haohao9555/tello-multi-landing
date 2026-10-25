"""Four-aircraft profiles, fixed pad assignments and landing outcome checks."""
from dataclasses import dataclass

import numpy as np

from tello_rl import task

N_DRONES = 4
PAD_OFFSET = 0.5
PADS = np.array([[PAD_OFFSET, PAD_OFFSET], [-PAD_OFFSET, PAD_OFFSET],
                 [-PAD_OFFSET, -PAD_OFFSET], [PAD_OFFSET, -PAD_OFFSET]])
DIRECTIONS = np.deg2rad([90.0, 0.0, 270.0, 180.0])
ASSIGNED_PAD = np.array([0, 3, 2, 1])
TARGETS = PADS[ASSIGNED_PAD]

START_RADIUS = (2.0, 2.5)
START_LATERAL = 0.3
START_Z = (1.0, 1.5)
BOUNDS_XY = 3.5
BOUNDS_Z = 3.0
MAX_STEPS = 400             # 40 s

COLLISION_DIST = 0.25
SAFE_DIST = 0.5
PROXIMITY_PENALTY = 0.1


GO_THRESHOLD = 3.5
PREFLIGHT_SECONDS = 10.0
ANEMOMETER_NOISE = 0.2

PER_DRONE_OBS = 13
OBS_DIM = N_DRONES * PER_DRONE_OBS
ACT_DIM = N_DRONES * task.ACT_DIM


@dataclass(frozen=True)
class Profile:
    """Profile."""
    name: str
    pad_offset: float
    pad_radius: float
    max_touchdown_speed: float
    start_radius: tuple
    start_lateral: float
    start_z: tuple
    bounds_xy: float
    bounds_z: float
    max_steps: int
    collision_dist: float
    safe_dist: float
    go_threshold: float
    anemometer_noise: float
    max_wind: float
    wind_levels: tuple
    v_max_xy: float
    v_max_z: float
    a_max: float
    kv: float
    kv_range: tuple
    drag: float
    drag_range: tuple
    drag2: float
    drag2_range: tuple
    length_scale: float
    action_history: int = 1

    @property
    def per_drone_obs(self):
        return 10 + 3 * self.action_history

    @property
    def pads(self):
        o = self.pad_offset
        return np.array([[o, o], [-o, o], [-o, -o], [o, -o]])

    @property
    def targets(self):
        return self.pads[ASSIGNED_PAD]


TELLO = Profile(
    name='tello', pad_offset=PAD_OFFSET, pad_radius=task.PAD_RADIUS, max_touchdown_speed=task.MAX_TOUCHDOWN_SPEED,
    start_radius=START_RADIUS, start_lateral=START_LATERAL, start_z=START_Z, bounds_xy=BOUNDS_XY,
    bounds_z=BOUNDS_Z, max_steps=MAX_STEPS, collision_dist=COLLISION_DIST, safe_dist=SAFE_DIST,
    go_threshold=GO_THRESHOLD, anemometer_noise=ANEMOMETER_NOISE, max_wind=task.MAX_WIND,
    wind_levels=task.WIND_LEVELS,
    v_max_xy=task.V_MAX_XY, v_max_z=task.V_MAX_Z, a_max=3.0, kv=2.5, kv_range=task.KV_RANGE,
    drag=0.41, drag_range=task.DRAG_RANGE, drag2=0.0, drag2_range=(0.0, 0.0), length_scale=1.0)



M30 = Profile(
    name='m30', pad_offset=2.0, pad_radius=0.3, max_touchdown_speed=0.5,
    start_radius=(20.0, 25.0), start_lateral=3.0, start_z=(8.0, 12.0), bounds_xy=35.0,
    bounds_z=25.0, max_steps=600, collision_dist=0.9, safe_dist=1.8,
    go_threshold=15.0, anemometer_noise=0.5, max_wind=15.0, wind_levels=(0.0, 5.0, 10.0, 15.0),
    v_max_xy=23.0, v_max_z=5.0, a_max=7.0, kv=1.5, kv_range=(1.1, 1.9),
    drag=0.12, drag_range=(0.10, 0.14), drag2=0.017, drag2_range=(0.014, 0.020), length_scale=10.0,
    action_history=2)

PROFILES = {'tello': TELLO, 'm30': M30}


def get_profile(profile):
    return profile if isinstance(profile, Profile) else PROFILES[profile or 'tello']


def sample_starts(rng, profile=TELLO):
    """Sample starts."""
    p = profile
    starts = np.zeros((N_DRONES, 3))
    for i, theta in enumerate(DIRECTIONS):
        radial = np.array([np.cos(theta), np.sin(theta)])
        lateral = np.array([-radial[1], radial[0]])
        starts[i, :2] = (rng.uniform(*p.start_radius) * radial
                         + rng.uniform(-p.start_lateral, p.start_lateral) * lateral)
        starts[i, 2] = rng.uniform(*p.start_z)
    return starts


def preflight_go(wind_model, rng, profile=TELLO):
    """Preflight go."""
    probe = task.WindModel(np.random.default_rng(rng.integers(2 ** 31)), wind_model.speed)
    probe.direction = wind_model.direction
    n = int(PREFLIGHT_SECONDS / task.CONTROL_DT)
    measured = np.mean([np.linalg.norm(probe.step(task.CONTROL_DT)) for _ in range(n)])
    measured += rng.normal(0.0, profile.anemometer_noise)
    return measured <= profile.go_threshold, measured


def target_costs(pos, profile=TELLO):
    """Target costs."""
    d = np.linalg.norm(pos[:, :2] - profile.targets, axis=1) + 0.5 * np.maximum(pos[:, 2], 0.0)
    return d / profile.length_scale


def check_touchdown(i, xy, vz, profile=TELLO):
    """Check touchdown."""
    p = profile
    d = float(np.linalg.norm(xy - p.targets[i]))
    speed = abs(float(vz))
    if d <= p.pad_radius:
        if speed <= p.max_touchdown_speed:
            return 'success', task.SUCCESS_BONUS + task.PRECISION_BONUS * (1 - d / p.pad_radius), d
        return 'hard_landing', -task.HARD_LANDING_PENALTY, d
    lo, hi = task.MISS_PENALTY
    return 'missed', -(lo + (hi - lo) * min((d - p.pad_radius) / (0.5 * p.length_scale), 1.0)), d
