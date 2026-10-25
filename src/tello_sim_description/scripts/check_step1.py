#!/usr/bin/env python3
"""Check step1."""
import argparse
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]
WORLD_NAME = 'tello_land'
WORLD_FILE = PKG_DIR / 'worlds' / f'{WORLD_NAME}.sdf'


def quat_rotate(q, v):
    """Quat rotate."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(q_vec, v)
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def tilt_deg(q):
    """Tilt deg."""
    w, x, y, z = q
    cos_tilt = 1 - 2 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_tilt))))


class TelloSim:
    def __init__(self):

        from gz.transport13 import Node
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.odometry_pb2 import Odometry
        from gz.msgs10.twist_pb2 import Twist
        from gz.msgs10.wind_pb2 import Wind
        self.Boolean, self.Twist, self.Wind = Boolean, Twist, Wind

        self.node = Node()
        self.twist_pub = self.node.advertise('/tello/gazebo/command/twist', Twist)
        self.enable_pub = self.node.advertise('/tello/enable', Boolean)
        self.wind_pub = self.node.advertise(f'/world/{WORLD_NAME}/wind', Wind)

        self.lock = threading.Lock()
        self.samples = []  # (t, pos, vel_world, tilt)
        self.node.subscribe(Odometry, '/model/tello/odometry', self._on_odom)

    def _on_odom(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        p = msg.pose.position
        o = msg.pose.orientation
        q = (o.w, o.x, o.y, o.z)
        v_body = (msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z)
        with self.lock:
            self.samples.append((t, (p.x, p.y, p.z), quat_rotate(q, v_body), tilt_deg(q)))

    def latest(self):
        with self.lock:
            return self.samples[-1] if self.samples else None

    def wait_ready(self, timeout=30.0):
        t0 = time.time()
        while self.latest() is None:
            if time.time() - t0 > timeout:
                raise RuntimeError('No /model/tello/odometry received; is the simulation running?')
            time.sleep(0.1)

    def enable(self, on):
        msg = self.Boolean()
        msg.data = on
        for _ in range(3):
            self.enable_pub.publish(msg)
            time.sleep(0.05)

    def set_wind(self, vx, vy):
        msg = self.Wind()
        msg.linear_velocity.x = vx
        msg.linear_velocity.y = vy
        msg.enable_wind = True
        self.wind_pub.publish(msg)

    def fly(self, vx=0.0, vy=0.0, vz=0.0, duration=1.0, stop_if=None):
        """Fly."""
        msg = self.Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = vx, vy, vz
        t_start = self.latest()[0]
        while True:
            self.twist_pub.publish(msg)
            time.sleep(0.02)
            s = self.latest()
            if s[0] - t_start >= duration or (stop_if and stop_if(s)):
                break
        with self.lock:
            return [s for s in self.samples if s[0] >= t_start]


def window(samples, last_sec):
    t_end = samples[-1][0]
    return [s for s in samples if s[0] >= t_end - last_sec]


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def run_checks(sim, wind_speed):
    results = []

    def check(name, ok, detail):
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    sim.wait_ready()
    z_ground = sim.latest()[1][2]
    print(f'Initial height z = {z_ground:.3f} m')

    print('1) Ascend: vz = 0.5 m/s for 2 s, then hold for 3 s')
    sim.enable(True)
    sim.fly(vz=0.5, duration=2.0)
    hover = window(sim.fly(duration=3.0), 1.5)
    z = [s[1][2] for s in hover]
    vxy = [math.hypot(s[2][0], s[2][1]) for s in hover]
    vz = [abs(s[2][2]) for s in hover]
    tilt = [s[3] for s in hover]
    check('Hover height', 0.8 < mean(z) < 1.2, f'z = {mean(z):.3f} m(expected approximately 1.0)')
    check('Stable hover', std(z) < 0.02 and max(vz) < 0.05 and max(vxy) < 0.05 and max(tilt) < 3,
          f'z standard deviation {std(z):.4f} m ; maximum |vz| {max(vz):.3f}, maximum horizontal speed {max(vxy):.3f} m/s, '
          f'maximum tilt {max(tilt):.2f}°')

    print('2) Translate: vx = 0.5 m/s for 3 s')
    track = window(sim.fly(vx=0.5, duration=3.0), 1.0)
    vx = mean([s[2][0] for s in track])
    vy = mean([s[2][1] for s in track])
    check('Velocity tracking', abs(vx - 0.5) < 0.1 and abs(vy) < 0.05,
          f'Measured vx = {vx:.3f} m/s, vy = {vy:.3f} m/s')
    sim.fly(duration=2.0)

    print(f'3) crosswind in +Y direction at {wind_speed} m/s, hold command for 6 s')
    sim.set_wind(0.0, wind_speed)
    windy = sim.fly(duration=6.0)
    late = window(windy, 2.0)
    drift_vy = mean([s[2][1] for s in late])
    drift_y = windy[-1][1][1] - windy[0][1][1]
    dz = max(abs(s[1][2] - windy[0][1][2]) for s in windy)
    max_tilt = max(s[3] for s in windy)
    check('Wind displacement', drift_vy > 0.05 and drift_y > 0.1,
          f'Steady drift velocity vy = {drift_vy:.3f} m/s, Y drift over 6 s {drift_y:.2f} m')
    check('Stable under wind', dz < 0.3 and max_tilt < 25,
          f'Maximum height change {dz:.3f} m, maximum tilt {max_tilt:.1f}°')
    sim.set_wind(0.0, 0.0)
    sim.fly(duration=3.0)

    print('4) Descend: vz = -0.4 m/s until near the surface, then stop motors')
    sim.fly(vz=-0.4, duration=6.0, stop_if=lambda s: s[1][2] < z_ground + 0.03)
    sim.enable(False)
    rest = window(sim.fly(duration=2.0), 0.5)
    z_rest = mean([s[1][2] for s in rest])
    tilt_rest = max(s[3] for s in rest)
    check('Gentle touchdown', abs(z_rest - z_ground) < 0.01 and tilt_rest < 5,
          f'Touchdown z = {z_rest:.3f} m, tilt {tilt_rest:.1f}°')

    return all(ok for _, ok, _ in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--attach', action='store_true', help='Connect to an existing simulation instead of launching one')
    parser.add_argument('--wind', type=float, default=3.0, help='Crosswind speed in m/s')
    args = parser.parse_args()

    server = None
    if not args.attach:

        os.environ['GZ_PARTITION'] = f'tello_check_{os.getpid()}'
        env = dict(os.environ)
        env['GZ_SIM_RESOURCE_PATH'] = f"{PKG_DIR / 'models'}:{env.get('GZ_SIM_RESOURCE_PATH', '')}"
        print('Starting headless simulation ...')
        server = subprocess.Popen(['gz', 'sim', '-s', '-r', str(WORLD_FILE)], env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  start_new_session=True)
    try:
        ok = run_checks(TelloSim(), args.wind)
    finally:
        if server is not None:
            os.killpg(server.pid, signal.SIGINT)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)

    print('\nStep 1 validation passed ✅' if ok else '\nStep 1 validation failed ❌')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
