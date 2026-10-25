"""Gazebo multi."""
import os
import re
import signal
import subprocess
import tempfile
import threading
import time

import numpy as np

from tello_rl import multi_task as mt
from tello_rl import task
from tello_rl.gazebo import (BODY_OFFSET_Z, DESC_DIR, PHYSICS_DT, STEPS_PER_CONTROL, WORLD_NAME,
                             quat_rotate)

N = mt.N_DRONES
DRONE_COLORS = [(0.90, 0.25, 0.25), (0.25, 0.70, 0.30), (0.25, 0.45, 0.90), (0.95, 0.80, 0.15)]

CAMERA_SDF = """
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <model name="recorder_camera">
      <static>true</static>
      <pose>0 -4.6 4.2 0 0.68 1.5708</pose>
      <link name="link">
        <sensor name="camera" type="camera">
          <always_on>1</always_on>
          <update_rate>{fps}</update_rate>
          <topic>/recorder/image</topic>
          <camera>
            <horizontal_fov>1.15</horizontal_fov>
            <image><width>{width}</width><height>{height}</height><format>R8G8B8</format></image>
            <clip><near>0.05</near><far>50</far></clip>
          </camera>
        </sensor>
      </link>
    </model>
"""


def drone_sdf(i):
    """Drone sdf."""
    text = (DESC_DIR / 'models' / 'tello' / 'model.sdf').read_text()
    model = text[text.index('<model name="tello">'):text.rindex('</model>') + len('</model>')]
    name = f'tello_{i + 1}'
    model = model.replace('<model name="tello">', f'<model name="{name}">')
    model = model.replace('<robotNamespace>tello</robotNamespace>', f'<robotNamespace>{name}</robotNamespace>')
    model = model.replace('/model/tello/odometry', f'/model/{name}/odometry')
    r, g, b = DRONE_COLORS[i]
    # Colour only the fleet identification panel; keep the detailed white shell.
    model = re.sub(r'(<visual name="fleet_accent">.*?<ambient>).*?(</ambient>.*?<diffuse>).*?(</diffuse>)',
                   lambda match: f'{match[1]}{r} {g} {b} 1{match[2]}{r} {g} {b} 1{match[3]}',
                   model, flags=re.S)
    x, y = 2.0 * np.cos(mt.DIRECTIONS[i]), 2.0 * np.sin(mt.DIRECTIONS[i])
    return model.replace(f'<model name="{name}">',
                         f'<model name="{name}">\n    <pose>{x:.3f} {y:.3f} {BODY_OFFSET_Z - 0.01:.4f} 0 0 0</pose>', 1)


def multi_world_sdf(camera, width, height):
    world = (DESC_DIR / 'worlds' / f'{WORLD_NAME}.sdf').read_text()
    world = re.sub(r'\s*<include>.*?</include>', '', world, flags=re.S)
    pads = ''.join(f'\n    <include><uri>model://landing_pad</uri><name>pad_{k + 1}</name>'
                   f'<pose>{x} {y} 0 0 0 0</pose></include>' for k, (x, y) in enumerate(mt.PADS))
    drones = ''.join('\n' + drone_sdf(i) for i in range(N))
    extra = CAMERA_SDF.format(fps=int(round(1 / task.CONTROL_DT)), width=width, height=height) if camera else ''
    return world.replace('</world>', pads + drones + extra + '\n  </world>')


class GazeboMultiTello:
    def __init__(self, camera=True, width=1920, height=1080, partition=None):
        os.environ['GZ_PARTITION'] = partition or f'tello_multi_{os.getpid()}'
        from gz.transport13 import Node
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.image_pb2 import Image
        from gz.msgs10.odometry_pb2 import Odometry
        from gz.msgs10.pose_pb2 import Pose
        from gz.msgs10.twist_pb2 import Twist
        from gz.msgs10.wind_pb2 import Wind
        from gz.msgs10.world_control_pb2 import WorldControl
        self.msgs = dict(Boolean=Boolean, Pose=Pose, Twist=Twist, Wind=Wind, WorldControl=WorldControl)

        self._tmp = tempfile.NamedTemporaryFile('w', suffix='.sdf', delete=False)
        self._tmp.write(multi_world_sdf(camera, width, height))
        self._tmp.close()
        cmd = ['gz', 'sim', '-s'] + (['--headless-rendering'] if camera else []) + [self._tmp.name]
        env = dict(os.environ, GZ_SIM_RESOURCE_PATH=str(DESC_DIR / 'models'))
        self.server = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, start_new_session=True)

        self.node = Node()
        self.lock = threading.Lock()
        self.odom = [None] * N
        self.image = None
        self.sim_time = 0.0
        self.camera = camera
        self.twist_pubs = [self.node.advertise(f'/tello_{i + 1}/gazebo/command/twist', Twist) for i in range(N)]
        self.enable_pubs = [self.node.advertise(f'/tello_{i + 1}/enable', Boolean) for i in range(N)]
        self.wind_pub = self.node.advertise(f'/world/{WORLD_NAME}/wind', Wind)
        for i in range(N):
            self.node.subscribe(Odometry, f'/model/tello_{i + 1}/odometry', self._odom_callback(i))
        if camera:
            self.node.subscribe(Image, '/recorder/image', self._on_image)

        deadline = time.time() + 90
        while not any('/control' in s for s in self.node.service_list()):
            if time.time() > deadline:
                self.close()
                raise RuntimeError('Gazebo server startup timed out')
            time.sleep(0.2)
        self.step(np.zeros((N, 3)), timeout=90)

    def _odom_callback(self, i):
        def callback(msg):
            t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
            p, o, v = msg.pose.position, msg.pose.orientation, msg.twist.linear
            with self.lock:
                self.odom[i] = (t, np.array([p.x, p.y, p.z]), np.array([o.w, o.x, o.y, o.z]),
                                np.array([v.x, v.y, v.z]))
        return callback

    def _on_image(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        with self.lock:
            self.image = (t, img.copy())

    def _request(self, service, msg, timeout_ms=5000):
        ok, rep = self.node.request(service, msg, type(msg), self.msgs['Boolean'], timeout_ms)
        return ok and rep.data

    def step(self, v_cmd_world, n=STEPS_PER_CONTROL, timeout=20):
        """Step."""
        for i in range(N):
            v = np.asarray(v_cmd_world[i], dtype=np.float64)
            if self.odom[i] is not None:
                w, x, y, z = self.odom[i][2]
                yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                c, s = np.cos(yaw), np.sin(yaw)
                v = np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])
            twist = self.msgs['Twist']()
            twist.linear.x, twist.linear.y, twist.linear.z = (float(u) for u in v)
            self.twist_pubs[i].publish(twist)
        req = self.msgs['WorldControl']()
        req.pause = True
        req.multi_step = n
        if not self._request(f'/world/{WORLD_NAME}/control', req):
            raise RuntimeError('World control request failed')
        self.sim_time += n * PHYSICS_DT
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                odom_ok = all(o is not None and o[0] >= self.sim_time - 1e-4 for o in self.odom)
                img_ok = (not self.camera or (self.image is not None
                          and self.image[0] >= self.sim_time - task.CONTROL_DT / 2))
            if odom_ok and img_ok:
                return
            time.sleep(0.0005)
        raise TimeoutError(f'Timed out waiting for state at t = {self.sim_time:.2f} s')

    def enable(self, i, on):
        msg = self.msgs['Boolean']()
        msg.data = on
        self.enable_pubs[i].publish(msg)

    def set_wind(self, wind_xy):
        msg = self.msgs['Wind']()
        msg.linear_velocity.x, msg.linear_velocity.y = float(wind_xy[0]), float(wind_xy[1])
        msg.enable_wind = True
        self.wind_pub.publish(msg)

    def set_pose(self, i, rel_pos):
        msg = self.msgs['Pose']()
        msg.name = f'tello_{i + 1}'
        msg.position.x, msg.position.y = float(rel_pos[0]), float(rel_pos[1])
        msg.position.z = float(rel_pos[2]) + BODY_OFFSET_Z
        msg.orientation.w = 1.0
        if not self._request(f'/world/{WORLD_NAME}/set_pose', msg):
            raise RuntimeError(f'set_pose(tello_{i + 1}) request failed')

    def set_camera(self, position, yaw, pitch):
        """Set camera."""
        cy, sy, cp, sp = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(pitch / 2), np.sin(pitch / 2)
        msg = self.msgs['Pose']()
        msg.name = 'recorder_camera'
        msg.position.x, msg.position.y, msg.position.z = (float(v) for v in position)
        msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = cp * cy, -sp * sy, sp * cy, cp * sy
        self._request(f'/world/{WORLD_NAME}/set_pose', msg)

    def state(self):
        """State."""
        with self.lock:
            odom = list(self.odom)
            image = self.image[1] if self.image is not None else None
        rel = np.array([o[1] for o in odom]) - np.array([0.0, 0.0, BODY_OFFSET_Z])
        vel = np.array([quat_rotate(o[2], o[3]) for o in odom])
        return rel, vel, image

    def close(self):
        try:
            os.killpg(self.server.pid, signal.SIGINT)
            self.server.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        os.unlink(self._tmp.name)
