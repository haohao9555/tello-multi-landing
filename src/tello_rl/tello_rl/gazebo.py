"""Gazebo."""
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.append('/usr/lib/python3/dist-packages')

from tello_rl import task

DESC_DIR = Path(__file__).resolve().parents[2] / 'tello_sim_description'
WORLD_NAME = 'tello_land'
PHYSICS_DT = 0.001
STEPS_PER_CONTROL = int(round(task.CONTROL_DT / PHYSICS_DT))

BODY_OFFSET_Z = 0.01 + 0.0205

CAMERA_SDF = """
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <model name="recorder_camera">
      <static>true</static>
      <pose>-1.6 -1.6 1.3 0 0.30 0.785</pose>
      <link name="link">
        <sensor name="camera" type="camera">
          <always_on>1</always_on>
          <update_rate>{fps}</update_rate>
          <topic>/recorder/image</topic>
          <camera>
            <horizontal_fov>1.1</horizontal_fov>
            <image><width>{width}</width><height>{height}</height><format>R8G8B8</format></image>
            <clip><near>0.05</near><far>50</far></clip>
          </camera>
        </sensor>
      </link>
    </model>
"""


def quat_rotate(q, v):
    """Quat rotate."""
    w, x, y, z = q
    t = 2 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)


class GazeboTello:
    def __init__(self, camera=False, width=1280, height=720, partition=None):

        os.environ['GZ_PARTITION'] = partition or f'tello_{os.getpid()}_{id(self)}'
        from gz.transport13 import Node
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.image_pb2 import Image
        from gz.msgs10.odometry_pb2 import Odometry
        from gz.msgs10.pose_pb2 import Pose
        from gz.msgs10.twist_pb2 import Twist
        from gz.msgs10.wind_pb2 import Wind
        from gz.msgs10.world_control_pb2 import WorldControl
        self.msgs = dict(Boolean=Boolean, Pose=Pose, Twist=Twist, Wind=Wind, WorldControl=WorldControl)

        world = DESC_DIR / 'worlds' / f'{WORLD_NAME}.sdf'
        cmd = ['gz', 'sim', '-s', str(world)]
        if camera:
            sdf = world.read_text().replace(
                '</world>', CAMERA_SDF.format(fps=int(round(1 / task.CONTROL_DT)),
                                              width=width, height=height) + '</world>')
            self._tmp = tempfile.NamedTemporaryFile('w', suffix='.sdf', delete=False)
            self._tmp.write(sdf)
            self._tmp.close()
            cmd = ['gz', 'sim', '-s', '--headless-rendering', self._tmp.name]
        env = dict(os.environ, GZ_SIM_RESOURCE_PATH=str(DESC_DIR / 'models'))
        self.server = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, start_new_session=True)

        self.node = Node()
        self.lock = threading.Lock()
        self.odom = None     # (t, pos, quat, vel_body)
        self.image = None    # (t, HxWx3 uint8)
        self.sim_time = 0.0
        self.twist_pub = self.node.advertise('/tello/gazebo/command/twist', Twist)
        self.enable_pub = self.node.advertise('/tello/enable', Boolean)
        self.wind_pub = self.node.advertise(f'/world/{WORLD_NAME}/wind', Wind)
        self.node.subscribe(Odometry, '/model/tello/odometry', self._on_odom)
        if camera:
            self.node.subscribe(Image, '/recorder/image', self._on_image)
        self.camera = camera


        deadline = time.time() + 60
        while not any('/control' in s for s in self.node.service_list()):
            if time.time() > deadline:
                self.close()
                raise RuntimeError('Gazebo server startup timed out')
            time.sleep(0.2)
        self.step(np.zeros(3), timeout=60)



    def _on_odom(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        p, o, v = msg.pose.position, msg.pose.orientation, msg.twist.linear
        with self.lock:
            self.odom = (t, np.array([p.x, p.y, p.z]), np.array([o.w, o.x, o.y, o.z]),
                         np.array([v.x, v.y, v.z]))

    def _on_image(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        with self.lock:
            self.image = (t, img.copy())



    def _request(self, service, msg, timeout_ms=5000):
        ok, rep = self.node.request(service, msg, type(msg), self.msgs['Boolean'], timeout_ms)
        return ok and rep.data

    def step(self, v_cmd, n=STEPS_PER_CONTROL, timeout=10):
        """Step."""
        twist = self.msgs['Twist']()
        twist.linear.x, twist.linear.y, twist.linear.z = (float(v) for v in v_cmd)
        self.twist_pub.publish(twist)
        req = self.msgs['WorldControl']()
        req.pause = True
        req.multi_step = n
        if not self._request(f'/world/{WORLD_NAME}/control', req):
            raise RuntimeError('World control request failed')
        self.sim_time += n * PHYSICS_DT
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                odom_ok = self.odom is not None and self.odom[0] >= self.sim_time - 1e-4
                img_ok = (not self.camera or (self.image is not None
                          and self.image[0] >= self.sim_time - task.CONTROL_DT / 2))
            if odom_ok and img_ok:
                return
            time.sleep(0.0005)
        raise TimeoutError(f'Timed out waiting for state at t = {self.sim_time:.2f} s')

    def enable(self, on):
        msg = self.msgs['Boolean']()
        msg.data = on
        self.enable_pub.publish(msg)

    def set_wind(self, wind_xy):
        msg = self.msgs['Wind']()
        msg.linear_velocity.x, msg.linear_velocity.y = float(wind_xy[0]), float(wind_xy[1])
        msg.enable_wind = True
        self.wind_pub.publish(msg)

    def _set_model_pose(self, name, position, quat=(1.0, 0.0, 0.0, 0.0)):
        msg = self.msgs['Pose']()
        msg.name = name
        msg.position.x, msg.position.y, msg.position.z = (float(v) for v in position)
        msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = quat
        if not self._request(f'/world/{WORLD_NAME}/set_pose', msg):
            raise RuntimeError(f'set_pose({name}) request failed')

    def set_pose(self, rel_pos):
        """Set pose."""
        self._set_model_pose('tello', np.asarray(rel_pos) + np.array([0.0, 0.0, BODY_OFFSET_Z]))

    def set_camera(self, position, yaw, pitch):
        """Set camera."""
        cy, sy, cp, sp = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(pitch / 2), np.sin(pitch / 2)
        self._set_model_pose('recorder_camera', position, (cp * cy, -sp * sy, sp * cy, cp * sy))



    def state(self):
        """State."""
        with self.lock:
            _, pos, quat, vel_body = self.odom
            image = self.image[1] if self.image is not None else None
        rel = pos - np.array([0.0, 0.0, BODY_OFFSET_Z])
        return rel, quat_rotate(quat, vel_body), image

    def close(self):
        try:
            os.killpg(self.server.pid, signal.SIGINT)
            self.server.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            os.killpg(self.server.pid, signal.SIGKILL)
        if hasattr(self, '_tmp'):
            os.unlink(self._tmp.name)
